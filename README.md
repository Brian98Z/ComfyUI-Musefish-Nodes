# ComfyUI-Musefish-Nodes

本节点包提供三条独立处理链路：视频超分与后处理、社交媒体视频下载、音频超分与母带处理。请按素材类型阅读对应章节；各流程的模型、参数和显存参考不可混用。

- [视频超分处理](#视频超分处理)
- [视频下载节点](#视频下载节点)
- [音频超分处理](#音频超分处理)

**通用安装：** 将本目录放入 `ComfyUI/custom_nodes/ComfyUI-Musefish-Nodes`，安装当前环境所需依赖后重启 ComfyUI。各章节列出对应的节点名称、模型与模板依赖。`__init__.py` 为共用扩展入口。

## 视频超分处理

### 功能与快速上手

- **`Musefish PiD Batch Video Upscale`**：PiD 视频 4x 超分（固定 1024 → 4096 路径），批次间复用同一噪声模板保证帧间稳定
- **`AutoBatch Antiflicker`**：对称时间双边滤波去频闪，运动边缘不拖影；`frames_per_batch=0` 自动分批 + `device=auto` CPU 卸载
- **`AutoBatch Image Sharpen FS`**：频率分离锐化（hard/linear light），针对 4K 超分软边；同样自动分批 + CPU 卸载

**快速上手**：PiD 超分输出 → `AutoBatch Antiflicker` → `AutoBatch Image Sharpen FS` → `VHS_VideoCombine`。人像推荐使用模板的低毛刺参数，避免强锐化把发际线、眼睑和脸颊轮廓变成颗粒碎边。

示例模板工作流：`workflows/Musefish_PiD_Batch_Video_Upscale.json`（UUID：`d7de7df1-0bb0-4cf8-bb1e-6f7ee7c5d1d2`）。
该模板在 PiD 输出后接入 `AutoBatchAntiflicker`，再进行自适应锐化与视频合并。

### AutoBatch Antiflicker

节点 ID：`AutoBatchAntiflicker`

功能：对 `IMAGE` 帧批次执行前后帧对称、亮度引导的时间双边滤波，抑制局部频闪，同时拒绝运动边缘，避免单向时间递归造成拖影。实现位于 `musefish_nodes.py`，不依赖或修改 `VideoHelperSuite`。

#### 自动分批与设备卸载

- 滤波张量运算在所选设备（GPU 或 CPU）上执行，输入帧按块搬运、处理完搬回输入所在设备，下游行为不变。
- **自动分批**：按设备当前空闲内存实时计算每批帧数（GPU 按显存、CPU 按系统内存），整段视频不会连同邻居副本/权重张量一起常驻显存或系统内存；`frames_per_batch` 可手动固定每批帧数。
- **设备卸载**：`device=auto` 时 GPU 优先，若显存连 1 帧都放不下自动降级 CPU；也可强制 `gpu` / `cpu`。
- 每块带前后各 1 帧上下文重叠，块内帧始终能看到真实时间邻居，块边界不产生接缝。

推荐连接：

```text
VHS_LoadVideo IMAGE ──→ AutoBatchAntiflicker ──→ VHS_VideoCombine IMAGE
VHS_LoadVideo AUDIO ─────────────────────────→ VHS_VideoCombine audio
```

参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `luma_tmp` | `15` | 亮度时间相似度宽度；提高可减轻亮度频闪，但过高会降低运动纹理稳定性 |
| `chroma_tmp` | `20` | 色度时间相似度宽度；用于背景颜色跳变，通常不需要超过 `20` |
| `frames_per_batch` | `0` | 每批处理帧数；`0` = 按设备空闲内存自动计算 |
| `device` | `auto` | 计算设备：`auto`（GPU 优先，显存不足自动降级 CPU）/ `gpu` / `cpu` |

推荐起点为 `15/20`、`device=auto`。16GB 参考模板后处理 `frames_per_batch=1` 以限制显存峰值；设为 0 可按空闲内存自动估算。算法使用相邻源帧，不递归传播历史，但快速运动仍需检查拖影。模板保留兼容性较好的 H.264 `yuv420p`；需要10-bit输出时另选支持的编码器，不要只修改像素格式。

如果主体出现拖影，优先降低 `luma_tmp`；如果只有背景颜色跳变，保持亮度参数不变、单独提高 `chroma_tmp`。不要把两个参数同时大幅提高。

### AutoBatch Image Sharpen FS

节点 ID：`AutoBatchImageSharpenFS`

功能：基于频率分离的浮点锐化。使用 float32 分批运算、软阈值与亮度梯度边缘保护，减少低幅噪声和轮廓高频被过度增强；不再与旧 RES4LYF 输出逐像素等价。

处理流程：

```text
low_pass  = 浮点 median/gaussian 模糊(images, intensity)  # CPU
detail    = hard/linear light 混合结果 - images            # float32
output    = clamp(images + amount × 软阈值(detail) × 边缘保护, 0, 1)
```

#### 自动分批与设备卸载

- float32 运算按空闲内存的保守预算分批；去频闪预算同时计入前后邻帧。输出预分配，避免 list + cat 造成额外整段副本。
- 遇到内存不足时缩小当前批次并重试，不跳帧；`auto` 可在单帧 GPU OOM 后回退 CPU，显式 `gpu` 不静默换设备。非内存异常与用户中断继续抛出。
- Gaussian 使用 OpenCV 浮点低通；median 小核使用 OpenCV，大核使用 SciPy 浮点中值滤波，避免 uint8 往返量化。大核 median 可能明显更慢，人像默认建议 Gaussian。

参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `method` | `hard` | 混合方式：`hard`（hard light）/ `linear`（linear light） |
| `blur_type` | `median` | 低通方式：`median`（保边缘）/ `gaussian` |
| `intensity` | `6` | 决定低通核尺寸：至少3，基于 intensity−1 取奇数；不是锐化混合强度 |
| `frames_per_batch` | `0` | 每批处理帧数；`0` = 按设备空闲内存自动计算 |
| `device` | `auto` | 计算设备：`auto` / `gpu` / `cpu` |
| `amount` | `1.0` | 锐化残差混合强度，0–2；人像模板为 **0.45**，0为不增强 |
| `noise_threshold` | `0.0` | 0–1范围的残差软阈值；人像模板为 **0.01**，弱于阈值的纹理不额外增强 |

人像参考：`hard / gaussian / 6 / 1 / auto`，`amount=0.45`、`noise_threshold=0.01`。先观察发际线、眼睑、脸颊边缘，再少量提高 amount；不要以强 median/12 锐化制造“清晰感”。此处理抑制锐化引入的毛刺，不保证修复模型生成的所有伪影，也不会恢复原片没有的真实细节。

### 效果案例

以下为旧版效果案例：原视频 480×832（33 帧，约 2 秒）经 PiD 4x 超分至 **2304×4096**，再经去频闪与 hard/median/12 锐化。历史素材保留用于对照，不代表当前人像推荐配置。

| 案例 | 文件 |
| --- | --- |
| 原视频 | [案例-原视频.mp4](assets/案例-原视频.mp4) |
| 4 倍超分 + 后处理 | [案例-4倍超分.mp4](assets/案例-4倍超分.mp4) |

![效果对比图](assets/效果对比图.png)

> 说明：超分视频为 4K 竖屏（2304×4096），文件较大，下载后建议本地播放器或剪辑软件查看；对比细节可重点看发丝、衣物纹理与主体边缘线条的锐度。

### 模板工作流结构

当前模板 `Musefish_PiD_Batch_Video_Upscale.json` 的处理顺序：

```text
LoadVideo
  ├── IMAGE → Musefish PiD Batch Video Upscale
  ├── AUDIO ───────────────────────────────┐
  └── FPS ─────────────────────────────────┤
                                           ▼
Musefish PiD Batch Video Upscale → AutoBatch Antiflicker(15/20/1/auto)
                                  → AutoBatch Image Sharpen FS(hard/gaussian/6/1/auto, amount=0.45, noise_threshold=0.01)
                                  → VHS_VideoCombine(yuv420p)
```

PiD 的原始 `VIDEO` 输出不经过后续图像节点；最终交付应使用经过 `IMAGE` 链路处理后的 `VHS_VideoCombine` 输出。

### 节点

#### Musefish PiD Batch Video Upscale

节点 ID：`MusefishPiDBatchVideoUpscale`

功能：将视频加载节点输出的 `IMAGE` 帧批量送入 PiD 模型，按原始顺序输出超分后的 `VIDEO` 与 `IMAGE`。可选音频和输入帧率会写入 `VIDEO` 输出。

节点会在一次执行中完成：

1. 按内置的 `1024` 长边统一输入帧尺寸；
2. 使用输入 VAE 编码低分辨率帧；
3. 按 `batch_size` 分批执行 PiD 采样；
4. 将 PiD 像素空间采样结果直接以 CPU float32 从 [-1,1] 映射到 [0,1]，无需解码 VAE；
5. 合并所有帧并保留音频、FPS。

VAE 预编码与 PiD 采样分开执行，避免每批反复换入换出模型。低分辨率 latent 暂存 CPU、用后释放；4K 输出直接写入预分配 CPU 张量。采样使用 ComfyUI 标准显存管理，不强制全量模型驻留；VAE/采样 OOM 时当前批次减半，保留同种子噪声与帧序，单帧仍失败则明确报错。PiD 保留全幅推理，不以未经验证的空间切块引入接缝。

**关于 `pixel_chunk_size`：** 它限制的是像素 Transformer 中独立 patch 的 MLP 支路，attention 仍处理全幅序列；较小分块会增加内核调用开销，不是越小越好，默认 `1024`（`2048` 无有意义的端到端收益）。

**attention_backend（默认 `Kitchen`）：** 只影响本节点本次采样，不改动 ComfyUI 全局 attention。`Kitchen`（Comfy Kitchen int8 attention）延迟较低，但显存取决于形状与批次，不能概括为更省；量化与舍入会让输出与 PyTorch/cuDNN 路径存在数值差异，不保证逐像素一致。当前构建没有 `Kitchen` 时会记录日志并自动回退到 `cuDNN`。

**同条件短片测量（用于定位取舍，不是所有素材的速度承诺）：**

| 场景 | 设置 | 观测 |
| --- | --- | --- |
| PiD 节点，Kitchen | 2 帧、2304×4096、同一模型、4 步、seed=0、batch=2、`pixel_chunk_size=1024` | 13.7346 秒，显存分配 9.72 GiB |
| PiD 节点，cuDNN | 同上 | 21.1101 秒，显存分配 8.57 GiB |
| Kitchen 对 cuDNN | 上述同条件 | 延迟降低约 35%，约 1.54×；这是该测量条件的结果，不是通用加速保证 |
| Kitchen 旧 2 帧视觉对照 | 短片对照 | PSNR 51.6–51.8 dB；不是逐像素相同，长运动质量尚未建立结论 |
| batch=3 复测 | 6 帧、同一模型与采样设置 | 79.62 秒，分配 12.81 GiB、保留 15.64 GiB；输出与 batch=2 不同，16GB 显卡不推荐 |
| `pixel_chunk_size` 逆序复测 | 6 帧，2048 对 1024 | 57.82 对 58.09 秒（约 0.46% 差异），6 帧结果相同；因此保留 1024 |

以上数据来自短片与固定参数，只用于定位取舍：分辨率、批次、设备、模型加载状态或素材变化都会改变耗时与显存，不构成速度承诺；`Kitchen` 与 `cuDNN` 的输出也不保证逐像素一致，长视频运动质量需单独检查。

#### 日志与结果边界（FAQ）

- **为什么每个批次都出现 `Model Initialization complete`？** 这是 DynamicVRAM tqdm 首次更新时附带的通用后缀。每批确实会执行准备步骤，但该文字不是磁盘权重每批重新加载的证据。
- **如何选择 backend？** 默认使用 `Kitchen`；当前构建没有 Kitchen 时本节点记录日志并回退 `cuDNN`。也可以显式选择 `cuDNN`，该覆盖只作用于本节点本次执行，不改全局 attention。

### 推荐连接

```text
VHS_LoadVideo
  ├── IMAGE ───────────────┐
  ├── AUDIO ───────────────┤
  └── VHS_VIDEOINFO.FPS ───┤
                            ▼
Musefish PiD Batch Video Upscale
  ├── MODEL      ← UNETLoader
  ├── CLIP       ← CLIPLoader(type=pixeldit)
  ├── encode_vae ← VAELoader(Flux\\UltraFlux-v1-vae.safetensors)
  └── 像素解码   ← 节点内 float32 映射，无需 VAE
                            │
                            ├── VIDEO → SaveVideo
                            └── IMAGE → 预览或视频合并节点
```

#### 颜色校正与频闪抑制

示例工作流在 PiD 输出后增加 `ColorMatchToReference`，并用 `ImageFromBatch(batch_index=0, length=1)` 固定取输入视频首帧作为参考：

```text
VHS_LoadVideo ──→ ImageFromBatch(首帧) ──→ ColorMatchToReference.reference_image
Musefish PiD ───────────────────────────→ ColorMatchToReference.images
ColorMatchToReference ──────────────────→ VHS_VideoCombine
```

默认 `match_strength=0.85`、`batch_size=4`。固定首帧参考会把每帧超分结果的 LAB 均值/标准差拉回同一颜色基准，针对 PiD 帧间色偏造成的频闪；它不能修复输入视频本身的亮度或内容闪烁。需要关闭校正时，断开颜色匹配节点并将 PiD 输出直接接入视频合并节点。

颜色匹配后的结果应从 PiD 的 `IMAGE` 输出进入 `VHS_VideoCombine`；PiD 的 `VIDEO` 输出仍是未经过外部颜色节点的原始视频对象。

`encode_vae` 是唯一需要连接的 VAE 输入。PiD 直接预测像素空间数据；输出使用 float32 映射，避免旧 `pixel_space` VAE 调度和中间低精度舍入。

| 参数 | 推荐值 |
| --- | ---: |
| `batch_size` | `1` 起步；显存足够时提高到 `2` 或 `4` |
| `pixel_chunk_size` | **1024**；0关闭内部优化，较小值降低MLP激活峰值但可能变慢 |
| `attention_backend` | **`Kitchen`**；无 Kitchen 时自动记录日志回退 `cuDNN` |
| `upscale_factor` | `4` |
| `latent_format` | `flux` |
| `degrade_sigma` | `0.0` |
| `cfg` | `1.0` |
| `sampler_name` | `lcm` |
| `scheduler` | `simple` |
| `steps` | `4` |
| `positive_prompt` | `high quality, ultra detailed, sharp details` |

模型内部始终先将输入帧长边缩放到 `1024`，执行固定的 `1024 → 4096` 超分。`upscale_factor` 只控制最终交付尺寸：设置 `2` 时先得到 4096，再缩小到 2048；设置 `3` 时缩小到 3072；设置 `4` 时直接输出 4096。

输入帧放大到模型尺寸时使用浮点 `bicubic`，避免 PIL Lanczos 的 uint8 往返量化；缩小到模型尺寸及2x/3x交付缩放仍使用 `area`。编码前限制到 [0,1]，避免插值过冲。此变更不增加采样步数、不降低1024→4096模型路径分辨率。

模型输入尺寸是内置约束，用户无需设置。

推荐模型：

```text
UNET:
PiD\\pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors

CLIP:
PixelDiT\\gemma_2_2b_it_elm_fp8_scaled.safetensors

encode_vae:
Flux\\UltraFlux-v1-vae.safetensors

decode VAE:
不需要：节点直接执行像素空间 float32 映射
```

模型下载：

- **UNET 与 CLIP（PixelDiT/PiD 系列）**：<https://www.modelscope.cn/models/Comfy-Org/PixelDiT/files>
- **VAE（encode_vae，z-image/flux1 通用）**：<https://www.modelscope.cn/models/Comfy-Org/z_image_turbo/tree/master/split_files/vae>

### 长视频处理建议

- 先将 `VHS_LoadVideo.frame_load_cap` 设为少量帧验证，例如 `2` 或 `4`。
- 确认输出尺寸和模型参数正确后，再增加帧数。
- PiD 超分显存不足时优先降低 `batch_size`，不要改变帧顺序。
- 后处理支持自动分批与 OOM 缩批重试，但仍需整段 CPU 输入/输出存储，不是无限长视频流式处理；内存不足应在加载端缩短片段。显存紧张先用 `frames_per_batch=1`，或显式 `device=cpu`。
- 固定模型输入为长边 `1024`，`upscale_factor=2/3/4` 分别交付约 2048/3072/4096 长边结果；模型计算量按 4 倍路径固定。
- 通过 `VIDEO` 输出连接 `SaveVideo`，由 ComfyUI 统一编码和保存音频。

### 视频稳定性

同一次节点执行会生成一个固定随机噪声模板，并在所有帧批次间复用；`batch_size` 改变不会改变帧对应的随机噪声序列，避免批次边界出现明显闪烁。

如果仍有局部细节闪动：

- 保持 `seed` 固定；
- 使用 `batch_size=1` 先确认模型与 VAE 配置；
- 确认 `encode_vae` 使用 `Flux\\UltraFlux-v1-vae.safetensors`；
- 确认输入帧没有被 `force_rate` 或 `select_every_nth` 大幅抽帧；
- 先用 2–4 帧短片测试，再增加视频长度。

### 视频相关文件

- `musefish_nodes.py`：PiD 超分、自动分批去频闪、频率分离锐化节点及扩展注册。
- `pid_runtime.py`：仅对兼容 PiD 像素块启用独立 MLP 分块；全幅 attention 不切图，模型 clone 的临时对象补丁在成功、异常和中断后恢复。
- [Musefish_PiD_Batch_Video_Upscale.json](workflows/Musefish_PiD_Batch_Video_Upscale.json)：视频超分与后处理模板。
- 模板 UUID：`d7de7df1-0bb0-4cf8-bb1e-6f7ee7c5d1d2`。

## 视频下载节点

### 功能与快速上手

- **`Musefish Video Download`**（节点 ID：`MusefishVideoDownload`）：通用平台视频下载。YouTube / B站 / X/Twitter 走进程内 yt-dlp；抖音走内置 iesdouyin 分享页解析（yt-dlp 的 Douyin 提取器被 a_bogus 签名墙挡死，不可用）；小红书走 yt-dlp 尝试、失败即报错并建议改走浏览器提取。每个文件在节点返回前都经过 MP4 `ftyp` 头 + ffprobe 双重验证。
- **`Musefish WC Video Channels`**（节点 ID：`MusefishWeChatChannels`）：微信视频号一键下载解密。单一直线流程，全自动、无需手动填 key：
  **自动解析粘贴内容 → 扫描客户端内存获取 URL/decode_key → 下载密文 → ftyp 头校验 → 解密 → ffprobe 验证**。
  校验不通过立即中止：key 错误时不会下载完整文件，也不会输出未验证的文件。

### Musefish Video Download 输入

| 输入 | 类型 | 说明 |
| --- | --- | --- |
| `url` | STRING（多行） | 视频 CDN 直链 / 分享链接 / 分享文本，支持"标题+链接"混合文本（自动提取真实 URL） |
| `platform` | COMBO | `Auto`（默认，从 URL 自动检测）/ YouTube / Bilibili / Douyin / Xiaohongshu / X-Twitter；选定平台后 URL 不符直接报错 |
| `quality` | COMBO | best（默认）/ 1080p / 720p / 480p；直链下载时忽略 |
| `chunks` | COMBO | 分块并行下载：1 / 2 / 4 / 8 / 16，默认 4。抖音直链切成 N 段 Range 并发；YouTube/B站走 yt-dlp `concurrent_fragment_downloads`。**前端自动隐藏/吸附**：URL 为空或平台不支持分块时控件隐藏；检测到抖音/YouTube/B站链接时出现并自动吸附最优值 8（可手动改回）；未知平台执行时强制 1 |

### 输出与下游连接

两个下载节点输出一致：

| 输出 | 类型 | 说明 |
| --- | --- | --- |
| `video` | VIDEO | 直接接 `SaveVideo` 等所有 VIDEO 消费节点 |
| `fps` | FLOAT | 下载视频的帧率 |

> 故意**不提供** `images`/`audio` 全量输出：把 194s/1080p 视频解码成帧张量会产生 112GB+ 的 float32 张量，直接撑爆内存。需要帧的消费方请用专用采样节点。

### 保存位置

固定 `ComfyUI/output/musefish/`，文件名 `<平台>_<标题/视频ID>_<时间戳>.mp4`，重名自动加序号。

### 使用注意

- **X/Twitter 与 YouTube 必须走本地代理**（默认 `http://127.0.0.1:7897`，Clash Verge 混合端口；探测不可达时 X 回退直连、YouTube 直接报错）。YouTube 另需 cookie：节点从包目录 `_yt_cookies.txt`（Netscape 格式）读取，无 cookie 会被 bot 墙拒绝（"Sign in to confirm you're not a bot"）。
- **抖音 cookie**：包目录 `_dy_cookies.txt`（从登录浏览器 CDP 抓取，含 ttwid/s_v_web_id）；无 cookie 时 iesdouyin 移动接口返回 `video_layout: null`（bot 检测）。
- **yt-dlp 自更新**：包导入时后台线程检查更新（24h 一次，清华 pip 镜像，不阻塞启动）——YouTube/抖音提取器失效的第一原因就是 yt-dlp 过期。
- **`Musefish WC Video Channels`** 的 `link_or_url` 支持粘贴：媒体 CDN 链接、分享链接、分享文本、Channels API 的 JSON 响应，也可以完全留空（= 取最新播放记录）。缺失的信息（URL/decode_key）自动从本机运行中的客户端内存获取——**使用前先在客户端里播放目标视频几秒，并保持客户端运行**。
- **防自动连播错位（双链接法）**：客户端播完会自动播放下一个视频，最新内存记录未必是你链接的视频。可靠顺序：① 复制分享链接 A → ② 打开 A 播放几秒 → ③ 复制任一其他链接 B → ④ 粘贴 A 运行。最后复制 B 会把目标视频钉在"最近打开"位置，不受连播影响。
- **内存守护**：下载分块间检查整机内存占用，>95% 时暂停等待，避免把机器推进 swap。
- **URL 缓存**：恢复出的带 token URL 缓存在包目录 `_wc_url_cache.json`（微信客户端几分钟内会压实内存中的 URL 行，缓存保证之后仍可重下）；该 URL 是约 48 小时的临时访问凭证，**不要外发含它的工作流 JSON 或截图**。

### 下载相关文件

- `musefish_video_nodes.py`：两个下载节点的 schema、粘贴内容自动解析与执行编排。
- `musefish_video_download.py`：yt-dlp 下载（含分块并发）、加密通道下载/解密、客户端内存扫描、ftyp/ffprobe 验证、URL 缓存。
- `memory_guard.py`：整机内存压力守护（>95% 暂停下载）。
- `_startup_update.py`：yt-dlp 后台自更新（24h 一次）。
- `wxdec_toolchain/`：常驻密钥流守护进程（内含 wasm + Node 守护脚本，无需全局装依赖）。wasm 取自 [Evil0ctal/WeChat-Channels-Video-File-Decryption](https://github.com/Evil0ctal/WeChat-Channels-Video-File-Decryption)（MIT），本仓库的 daemon 为原创重写。
- `web/musefish_video_download_segments.js`：`chunks` 控件的平台感知显隐与自动吸附前端逻辑。
- `tests/test_video_download_nodes.py`：解密管线（以上游样本密文+key 为真值，`tests/fixtures/`）、key 校验门、Range 回退、粘贴解析、平台检测、分块并发参数、节点行为测试。

## DLSS5 超分节点

> **仅支持 NVIDIA RTX 50 系（Blackwell）显卡**：DLSS 5 超分与配套
> RTX 视频超分运行库为 Blackwell 独占，40 系及更早显卡无法创建该 Feature。

### 功能概览

- **图像与短视频**：`MusefishDLSS5NeuralRender` 输入标准 IMAGE 批次、
  固定单 worker；输出接 SaveImage 或 VHS_VideoCombine。提供与流式节点相同的
  「放大参数」和第三位 `vsr_quality`，但整个 float32 输出批次超过 1 GiB 时拒绝。
- `musefish_dlss5.py` + `dlss5_backend/`：把 DLSS5Tool 的 DLSS 5 超分
  （NGX Feature 18）与 RTX 视频超分封装为 `MusefishDLSS5NeuralRender`
  节点（分类 `Musefish/Video`），在隔离子进程里驱动 NVIDIA DLL，崩溃
  不影响 ComfyUI 主进程；详见 [DLSS5_README.md](DLSS5_README.md)。
- 案例效果：[assets/DLSS5超分案例.mp4](assets/DLSS5超分案例.mp4)
  （1472×1280@24fps，10 秒，2× RTX VSR + Feature 18 实测输出）。
- **高清长视频**：`LoadVideo → MusefishDLSS5VideoStream → SaveVideo` 均使用
  VIDEO 类型；内部连续解码、增强、编码和音频封装，SaveVideo 输出成片。
  不积累整批 IMAGE，不需手动拼接。旧分段模板仅供历史兼容。
  内层编码默认 `libx264`；可选 `h264_nvenc` 降低 CPU 编码开销（固定 CQ 27，
  与 `crf` 不是同一画质刻度）。中间文件名自动生成，最终名称在 SaveVideo 设置。
  若接 `Video Slice`，节点会按裁剪的起点与时长处理视频和音频，
  不会再把原片全长送入 DLSS5。
  两节点的「放大参数」均提供 1× (Native)、1.5× (Quality)、2× (Balance)、
  3× (Performance)、4× (Ultra) 与 1K–8K 目标档；第三个控件 `vsr_quality`
  才控制实际 VSR 质量。1.5×/3× 为 VSR 合成缩放，不是原生倍率。
  横竖屏均按短边选 K 档：720p→4K 采用 5120×2880 中间帧后输出 3840×2160；
  1080p→8K 输出 7680×4320（竖屏为 4320×7680），需 GPU 编码并自动使用
  HEVC NVENC。720p 无法用至多 4× VSR 达到 8K。
  四个外部 DLL 放在 `ComfyUI/models/dlss5/`；用户提供的下载分享为
  <https://pan.quark.cn/s/d4c04dc33d25>（非 NVIDIA 官方发行）。文件清单与
  其他依赖见 [DLSS5_README.md](DLSS5_README.md)。

## 音频超分处理

### 功能概览

- **`Musefish UniverSR Model`**：准备 general / speech 模型缓存，可按需下载；输出缓存目录，不处理音频。

- **`Musefish UniverSR General Audio`**（节点 ID：`MusefishUniverSRGeneralAudio`）：固定使用 general 模型，支持 `sr`（仅超分）、`master`（V8 母带）、`sr_master`（超分+母带）和 `stem_mix`（Demucs 分轨+混音）。
- **`Musefish UniverSR Speech Audio`**（节点 ID：`MusefishUniverSRSpeechAudio`）：固定使用 speech 模型，仅支持 `sr` 超分；不暴露模型选择，适合人声/语音增强。
- 两个节点都处理标准 `AUDIO`（`waveform` 为 `B,C,T`），逐批支持单声道/立体声，并通过 `model_cache` 输入连接 `Musefish UniverSR Model` 的缓存输出。
- `input_sr=auto` 的档位判据是 `effective = max(99% rolloff, content cutoff)`。`content cutoff` 指 rolloff 之上频谱首次跌落到 rolloff 电平 20 dB 以下处的频率——MP3 的 rolloff 可低至约 9.6 kHz 而内容一直延伸到约 15 kHz，只看 rolloff 会误选 24k 档，让模型凭空生成 8–12 kHz。档位表：`effective ≤ 5.4 kHz → 8k`（5.0–5.4 kHz 边界带先用 8k 试探）、`≤ 7.2 kHz → 16k`、`≤ 12 kHz → 16k`（内容到 12k 时 16k 条件专注清理与增强，不凭空生成）、其余 `→ 24k`；不再直接输出 12k 档。
- **8k 硬落地护栏**：`effective ≤ 5000 Hz` 时直接采用 8k 档，不做向上校正。校正判据窗口（4150–5850 Hz）会把滤波过渡带残能当成「持续高频证据」，实测把 effective 3400–5000 Hz 的素材全部升到 12k/16k。5.0–5.4 kHz 边界带仍走校正，因此 `effective = 5062 Hz` 的低带宽素材可被升到 12k 档。
- 仅 general 模型做向上校正，条件是候选频带满足：占全谱能量 ≥ 1e-5（-50 dB）、连续带宽 ≥ 约 300 Hz、同时高于噪声底 6 dB 且距候选带峰值不超过 20 dB、至少 25% 的分析帧有能量、并有连续 3 个分析帧的证据（不足 12 帧时要求 2 帧）；带边留 150 Hz 余量，只向上、不向下。复用同一次 FFT，不因容器标称 48 kHz 一律升档。日志形如 `auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`，被护栏拦下时形如 `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`。
- 静音、低底噪、宽带 hiss、孤立尖峰或瞬态通常不能满足连续频带与时域门限；speech 模型用同一 `effective` 判据但不做向上校正；手动 `input_sr` 不检测、不改写。
- 更新后需重启 ComfyUI 以加载主进程中的选档逻辑。24 kHz 是模型支持的最高输入档位，不能保证保留原曲 12 kHz 以上的细节；该校正不改变母带链，也不保证所有音乐都能改善听感。

### 运行环境与模型路径

worker 使用启动 ComfyUI 的 `sys.executable`，支持 Python 3.10–3.13 的 ComfyUI 环境；具体 Torch/音频依赖由当前环境提供。`Musefish UniverSR Model` 节点的缓存目标是 `ComfyUI/models/UniverSR/models/huggingface/<general|speech>`，每个目录需要同时存在 `config.yaml` 与 `pytorch_model.bin`；只有 `download=true` 时才访问网络，按 `HF_ENDPOINT`（默认 `https://hf-mirror.com`）逐文件下载后原子改名，导入插件时不联网，也不用 `snapshot_download`。

处理节点的 `model_cache` 输入（或 `MUSEFISH_UNIVERSR_MODEL_CACHE`）决定 worker 到哪里找权重，解析顺序：该目录本身 → `<cache>/universr-audio|universr-speech` → `<cache>/general|speech` → `<cache>/models--woongzip1--<name>` → `<cache>/huggingface/...`，以及上述目录下的 HuggingFace `snapshots/` 快照。单模型模式（`sr` / `sr_master`）直接连接模型节点输出即可；**`stem_mix` 需要一次解析出两个模型**，因此 `model_cache` 应指向同时含 `general` 与 `speech` 的父目录（即 `ComfyUI/models/UniverSR/models/huggingface`），只连某一个模型节点的输出会找不到另一个模型。`stem_mix` 使用同一 Python 环境中的 Demucs 模块。

### 音频示例工作流

示例文件：[Musefish_UniverSR_Audio.json](workflows/Musefish_UniverSR_Audio.json)，工作流 ID：`6e417a01-39fa-461d-9f27-484c97aeef2e`。这是 **16GB 显存参考模板**，包含音乐与语音两条独立分支，不是把同一段音频先后送入两个模型。

```text
音乐分支（默认启用）
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRGeneralAudio
MusefishUniverSRModel（general）─ model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced（FLAC）
                                             └─ log   → ShowText|pysssss

语音分支（默认旁路）
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRSpeechAudio
MusefishUniverSRModel（speech）── model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced（FLAC）
                                             └─ log   → ShowText|pysssss
```

**使用步骤：**

1. 将 JSON 拖入 ComfyUI，或从工作流模板入口加载。
2. 模板额外使用 `rgthree-comfy` 的 `Fast Groups Bypasser (rgthree)` 和 `ComfyUI-Custom-Scripts` 的 `ShowText|pysssss`。缺少时安装对应节点包；也可删除分组开关并手动旁路分支，日志输出不连接不影响音频处理。
3. 在对应 `LoadAudio` 中上传并选择自己的素材。`测试歌曲.mp3`、`测试语音.wav` 是模板示例文件名，**不代表节点包附带这些音频**。
4. 音乐使用 general 缓存节点，语音使用 speech 缓存节点。模板两者 `download=true`；准备好本地模型后可关闭下载。`stem_mix` 还需要 speech 模型缓存和 Demucs 环境。
5. 用左侧分组开关启用需要的分支。建议一次只运行一条分支，避免误处理另一份测试输入。
6. 运行后在保存节点试听结果、查看日志；输出前缀为 `audio/Musefish_UniverSR`，位于 ComfyUI 配置的输出目录内，模板格式为 FLAC。

### 16GB显卡 参考模板预设

| 项目 | 音乐分支 | 语音分支 |
| --- | --- | --- |
| 模型 | general | speech |
| 处理方式 | `sr_master`（超分+母带） | `sr`（仅超分，固定） |
| `input_sr` | `auto`，启用 general 高频校正 | `auto`，保留原带宽检测 |
| `channel_mode` | `auto` | `auto` |
| ODE | `midpoint`，4 步 | `midpoint`，4 步 |
| `guidance` | 1.5 | 1.5 |
| `chunk_sec` | **30 秒** | **20 秒** |
| 种子策略 | `randomize` | `randomize` |
| 初始状态 | 启用 | 旁路 |

这组预设来自参考工作流，不是所有 16GB 显卡与素材的显存保证。显存不足优先减小 `chunk_sec`，例如从 30 降到 15，再降到 10；分块只控制单段推理规模，不会让整曲输入/输出完全不占内存。不要把视频节点的自动分批规则套到音频节点上。对比参数效果时固定种子，并只改一个参数。

> 预设里写死的 `处理方式`（`sr_master` / `sr`）是显式取值，会被当作手动指定保留；`ODE 4 步` 与 `guidance 1.5` 仍是控件默认值，`auto_params` 开启时按素材实测改写。要完全按上表固定执行，请关闭 `auto_params`；`chunk_sec`、种子策略、`input_sr` 的 `auto` 语义不受该开关影响。

### 处理模式与适用场景

| `mode` | 处理链路 | 适用与注意事项 |
| --- | --- | --- |
| `sr` | UniverSR 超分 → M/S 合成 → 带宽清理 → 分带去齿音（general 默认） → 真峰收口 | 真正低带宽的音乐/音效；speech 节点只提供此模式，且不做分带去齿音 |
| `master` | 跳过模型超分 → V8 母带链（EQ → 多段压 → 干 ER → 高频整形 → TP 反馈限幅 → 19.2k 切除） | 不需要补带宽、只希望调整频谱与动态的素材；不执行超分，ODE、guidance 和超分分块参数不参与此路径，响度保持源状态 |
| `sr_master` | Mid 超分 → Mid 母带（预留 4 dB 真峰余量 + 链内去齿音） → Side 带限 + 14k+ 去相关 → M/S 合成 → 带宽清理 → 真峰收口 | 同时需要带宽重建与母带处理；Side 在 14k 以上用按素材塑形的噪声补宽度，避免高频单声道 |
| `stem_mix` | Demucs 分人声/伴奏 → speech 处理人声、general 处理伴奏 → 人声 8-12k/12-16k 回补 → 伴奏母带 + 12-16k 回补 → 按源电平混音 → 锚定源响度 → TP 收口 | 需要分轨增强；依赖更多模型，资源与耗时通常更高，16GB 双分支模板不代表此模式已做显存保证 |

**自动校正可以用于 `sr_master`，无需新增节点。** General 的自动选档发生在模式分流之前，超分部分直接使用校正结果；`stem_mix` 当前将选定档位传给分轨处理，不是分别对每条分轨重新自动识别。`master` 不做超分，日志中的选档信息不表示它会执行模型带宽重建。

### 处理节点参数

以下是**新建节点的默认值**，与上面的模板预设分开看。

| 参数 | 默认值 / 范围 | 作用与调节建议 |
| --- | --- | --- |
| `audio` | 必接，标准 `AUDIO` | 输入音频；支持批次及单/双声道，输出为 48 kHz |
| `auto_params` | `true`（首个控件） | 按素材实测自动决定 `input_sr` / `ode_steps` / `guidance`，**开启时这三个控件在界面上隐藏**；关闭后它们显示出来、完全按你选的值执行。手动改动过的参数始终优先（见下节） |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000` | 表示内容有效带宽对应的输入采样率，不是期望输出采样率；16k 档对应约 8kHz 带宽。优先自动，有可靠带宽依据时手动覆盖。`auto` 的判据、档位表与 8k 护栏见“功能概览”；`auto_params` 关闭但此项仍为 `auto` 时，选档依然是自动的 |
| `ode_steps` | 4；1–25 | 超分求解步数；提高会增加计算量，不会恢复被错误输入档位预先滤除的原始信息 |
| `guidance` | 1.5；0–5，步长 0.1 | 条件引导强度；提高可能强化生成纹理，也可能增加不自然感。先保留 1.5，再做同种子短片段对比 |
| `mode` | `auto`；`auto`/`sr`/`master`/`sr_master`/`stem_mix` | 仅 general 节点显示；speech 固定仅超分。`auto` 表示交给素材实测选择，选定具体模式即视为手动指定并优先 |
| `channel_mode` | `auto` / `mono` / `stereo` | `auto` 按输入声道数选择，不是按 general/speech 强制选择；`mono` 会将双声道平均，`stereo` 会将单声道复制为双声道，但不能凭空恢复真实声场 |
| `ode_method` | `midpoint`；`euler` / `midpoint` / `rk4` | 超分求解方法；先使用 midpoint，不以更复杂的求解方法保证音质改善 |
| `chunk_sec` | 15；1–120 秒 | 单次超分分块时长；内部分块带约 50ms 重叠拼接。显存不足时减小，不建议长音频直接使用最大值 |
| `seed` | 0；非负整数 | 随机种子；前端生成后策略选 fixed 便于 A/B，模板使用 randomize |
| `deess` | `true` | 分带去齿音（只压 5.5–8.5k 带内分量，默认上限 8 dB，阈值取带内包络中位 +6 dB，与音量无关）。仅 `sr` 模式 + general 模型生效；speech 节点的自然人声 6–8k 本就不冲，开启会把真实齿音压掉，因此该节点自动跳过；`sr_master`/`stem_mix` 用链内去齿音，不重复作用。留默认值时按素材实测（见下节），关掉即视为手动指定 |
| `accel` | `cuDNN TF32`；`fp32`/`cuDNN TF32`/`bf16` | 模型推理的 GPU 加速档（见「加速选项」）。**默认 `cuDNN TF32`**：快 1.18×、与 fp32 差异约 −48 dBFS；`fp32` 是逐样本一致的基准档（最慢）；`bf16` 快 1.35× 但波形差异可闻。`master` 模式不跑模型，此项无作用；无 CUDA 时自动退回 `fp32` |
| `model_cache` | 可选 STRING 连线 | 连接模型节点输出；未连接时使用配置的默认模型目录。切换模式不会把 general 处理节点变成 speech 节点 |
| 输出 `audio` / `log` | AUDIO / STRING | 分别连接音频保存节点和文本展示节点；日志包含实际选档及分块进度 |

### 模型准备节点参数

| 参数 | 新建节点默认值 | 说明 |
| --- | --- | --- |
| `model` | `general` | `general` 下载音乐/音效模型，`speech` 下载语音模型；与对应处理分支匹配 |
| `download` | `false` | `true` 时请求下载模型到缓存；模板预设为 true。离线使用前先确认所需权重完整 |
| 输出 `model_cache` | STRING | 缓存目录引用，不是 AUDIO，不应串在音频信号连线上 |

### 素材驱动参数匹配（`auto_params`，默认开启，首位控件）

与自动选档共用同一套测量（`content_cutoff` + 分带落差），把「修复需求」接到参数上，规则来自 4 素材 × 4 配置的同种子实测：

- `need = clip((-23 - d12) / 20, 0, 1)`，`d12 = B(12-16k) - B(4-6k)`；`need = 0` 表示源自带高频（全带宽成品），超分只会加嘶声 → general 取 `master`；`need > 0` → general 取 `sr_master`。
- speech 模型只提供超分，因此无论 `need` 多大都保持 `sr`（`need = 0` 时日志会说明该素材不需要增强）。
- `guidance`：speech 按 `need < 0.5 → 1.0`，否则 `1.5`；general 固定 `2.0`（音乐域由听感定型，本次实测未覆盖）。
- `ode_steps`：speech 按 `need < 0.5 → 8`（实测与 16 步差 ≤0.25 dB＝噪声内，速度优先），`need ≥ 0.5 → 16`（真修复素材 8 步会差 0.62 dB）；general 固定 `16`（音乐实测决定性：12-16k 落差 −14.1 → −4.6 dB）。
- `deess`：general 恒开；speech 仅在 6-8k 反超 4-6k ≥ 2 dB（实测真实自然齿音为 −1.7…−2.5 dB，开了会误伤）时才开启。
- 素材不可读（静音/过短）时回退到分域默认值并在日志说明；`mode` 若不在该节点允许集合内则保留节点原值（不会让 speech 节点发出 `master` 请求）。

**手动改动优先。** 判断依据是「该值是否仍等于控件默认值」：`input_sr` / `ode_steps` / `guidance` 三项在界面开关开启时隐藏，`mode`（默认 `auto`）与 `deess`（默认 `true`）保持可见。任何被改到非默认值的项都视为手动指定、不会被自动匹配覆盖；留默认值的项则按素材实测取值。日志会打印实际采用值与保留的节点值（`kept node values: ...`），逐条实测依据以 `auto params → ...` 打印。

**界面与顺序。** `auto_params` 位于节点首位，开启时其后的 `input_sr` / `ode_steps` / `guidance` 三项在界面上隐藏，关闭即恢复手动选择。从旧版本更新本插件后请硬刷新页面（`Ctrl+Shift+R`），否则控件仍按旧顺序渲染，保存值可能对不上号并导致排队校验失败；刷新后旧工作流的 `mode` / `chunk_sec` / `seed` 等设置值都能正确读回。

### 加速选项（`accel`）

各档位在 RTX 5070 Ti 上的实测差异：

| 档位 | 加速 | 与 `fp32` 的最大样本差 | 结论 |
| --- | --- | --- | --- |
| `fp32` | — | — | 与既有听感锁定版本**逐样本一致**，需要复现基准时选它 |
| `cuDNN TF32` | **1.18×** | 3.9e-3（≈ −48 dBFS） | **默认档** |
| `bf16` | 1.35× | 0.26（≈ −11.7 dB，rms −38.6 dB） | 已提供，但有可闻精度损失 |

无 CUDA 的机器上任何非 `fp32` 档都会自动退回 `fp32`；该设置只影响本节点的模型推理，不改动 ComfyUI 的全局 GPU 配置。


- **容器采样率不等于有效带宽。** 48 kHz 文件也可能只包含低带宽内容；反过来，99% 能量集中在低频也不代表其余高频可以丢弃。档位用 `effective = max(rolloff, content cutoff)`，并按 `effective ≤ 5000 Hz` 的护栏跳过向上校正，判据与档位表见“功能概览”。
- 示例日志：`auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`。`base` 是原检测档位，`corrected` 是实际采用档位；被 8k 护栏拦下时是 `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`。两者都不表示输出从 16kHz 改为 24kHz；音频输出仍为 **48kHz**。
- **超分后更闷**：先查实际 `input_sr`；再用固定种子、同一短片段对比 `sr` 和 `sr_master`。母带链仍会塑形高频，自动升档不能保证消除它对音色的影响。分带去齿音只在 `deess=true` 且 `mode=sr` + general 时作用，关闭它对比可以确认 5.5–8.5k 的变化。
- **已有完整高频的成品曲**：不要仅因是 MP3 就判为低清。最高 24k 输入档仍可能滤除约 12kHz 以上的原始内容；优先保留原曲，确有母带需求再对比 `master`，而非盲目超分。
- **高频噪感或声场变化**：超分会生成纹理，立体声路径还包含 M/S 处理；先比较仅超分与原曲，不要只靠提高 guidance、步数或音量判断品质。`sr_master`/`sr` 的立体声路径在 14k 以上用去相关噪声补宽度，如听感偏高噪可试 `mode=sr` 直出对比。
- **更新节点代码后需重启 ComfyUI**，并避免复用旧缓存结果；固定种子用于公平比较，修改代码后的首次验证可换一个种子触发重新执行。
- **耗时由 `input_sr` 决定**：8k 档最快、24k 档最贵（60s 素材 `sr_master` + 自动档 24k 约 7–8 分钟，200s 整曲 `master` 因不跑模型通常一分钟内）；长跑期间 GPU 接近满载属正常，不是卡死。
- **缺模型/缺节点**：检查模型缓存、对应 general/speech 权重，以及模板的 rgthree、文本展示和 `SaveAudioAdvanced` 节点；保存节点不可用时升级 ComfyUI 或用当前版本提供的音频保存节点替换。`stem_mix` 报找不到模型时确认 `model_cache` 指向同时含 `general` 与 `speech` 的父目录。

### 音频相关文件

- `musefish_audio.py`：模型准备节点、General/Speech 处理节点、AUDIO 张量适配与隔离 worker 调度。
- `audio_backend/processing.py`：自动带宽检测与 general 校正、分块超分、四种模式的链路与收口。
- `audio_backend/dsp.py`：母带及音频后处理（EQ / 多段压 / 去齿音 / 高频整形 / 限幅 / 带宽清理 / 侧声道去相关）。
- `audio_backend/worker.py`：独立处理进程入口。
- [Musefish_UniverSR_Audio.json](workflows/Musefish_UniverSR_Audio.json)：音乐/语音双分支参考模板，分别连接模型缓存、音频保存和日志展示节点。

## 更新日志

### v1.4.0（2026-09-25）

- 新增 `MusefishDLSS5VideoStream`：文件支持的 VIDEO 逐帧解码、连续 NGX 会话增强、编码并保留音轨；遵循 `Video Slice` 起点和时长，通过 ComfyUI 原生进度事件显示处理进度。
- IMAGE/VIDEO 两节点统一「放大参数」档位（1×、1.5×、2×、3×、4×及 1K–8K）和前端旧工作流控件迁移；1.5×/3× 为 VSR 合成倍率。720p→4K 与 1080p→8K（横竖屏）完成实际出片验证；IMAGE 批次超过 1 GiB 则提示使用流式节点。
- VIDEO 支持可选 NVENC；8K 自动采用 HEVC NVENC，低分辨率继续使用 H.264。NVIDIA 私有运行库统一从 ComfyUI 的 `models/dlss5/` 读取，不随插件分发；安装清单见 [DLSS5_README.md](DLSS5_README.md)。

### v1.3.0（2026-09-22）

**新增：社交媒体视频下载节点**

- `MusefishVideoDownload`：YouTube / B站 / X/Twitter / 抖音 / 小红书平台下载，进程内 yt-dlp + 抖音 iesdouyin 分享页专用解析（a_bogus 签名墙兜底），MP4 `ftyp` + ffprobe 双重验证后才返回。
- `MusefishWeChatChannels`：微信视频号加密视频全自动下载解密（内存扫描取 URL/decode_key、Isaac64 密钥流 XOR 解密、双链接法防自动连播错位、URL 缓存 `_wc_url_cache.json`）。
- `chunks` 分块并行下载（1/2/4/8/16）：抖音直链 Range 分段并发、YouTube/B站 yt-dlp `concurrent_fragment_downloads`；前端按 URL 平台感知显隐并自动吸附最优值 8。
- 健壮性：yt-dlp 启动后台自更新（24h/次）、整机内存压力守护（>95% 暂停）、X/YouTube 自动代理路由（127.0.0.1:7897）+ cookie 文件支持。
- 前端 `web/musefish_video_download_segments.js`；测试 `tests/test_video_download_nodes.py`（36 项，含上游真值解密管线）。

### v1.2.0（2026-09-21）

- 新增 `MusefishDLSS5NeuralRender`：DLSS 5 神经渲染（仅 RTX 50 系 Blackwell），含 RTX VSR 前置超分级联、会话缓存、分辨率/style/mask 全契约缓存键。
