# PiD 视频超分与后处理

## 功能与快速上手

- **`Musefish PiD Batch Video Upscale`**：PiD 视频 4x 超分（固定 1024 → 4096 路径），批次间复用同一噪声模板保证帧间稳定
- **`AutoBatch Antiflicker`**：对称时间双边滤波去频闪，运动边缘不拖影；`frames_per_batch=0` 自动分批 + `device=auto` CPU 卸载
- **`AutoBatch Image Sharpen FS`**：频率分离锐化（hard/linear light），针对 4K 超分软边；同样自动分批 + CPU 卸载

**快速上手**：PiD 超分输出 → `AutoBatch Antiflicker` → `AutoBatch Image Sharpen FS` → `VHS_VideoCombine`。人像推荐使用模板的低毛刺参数，避免强锐化把发际线、眼睑和脸颊轮廓变成颗粒碎边。

示例模板工作流：`workflows/Musefish_PiD_Batch_Video_Upscale.json`（UUID：`d7de7df1-0bb0-4cf8-bb1e-6f7ee7c5d1d2`）。
该模板在 PiD 输出后接入 `AutoBatchAntiflicker`，再进行自适应锐化与视频合并。

## AutoBatch Antiflicker

节点 ID：`AutoBatchAntiflicker`

功能：对 `IMAGE` 帧批次执行前后帧对称、亮度引导的时间双边滤波，抑制局部频闪，同时拒绝运动边缘，避免单向时间递归造成拖影。实现位于 `musefish_nodes.py`，不依赖或修改 `VideoHelperSuite`。

### 自动分批与设备卸载

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

## AutoBatch Image Sharpen FS

节点 ID：`AutoBatchImageSharpenFS`

功能：基于频率分离的浮点锐化。使用 float32 分批运算、软阈值与亮度梯度边缘保护，减少低幅噪声和轮廓高频被过度增强；不再与旧 RES4LYF 输出逐像素等价。

处理流程：

```text
low_pass  = 浮点 median/gaussian 模糊(images, intensity)  # CPU
detail    = hard/linear light 混合结果 - images            # float32
output    = clamp(images + amount × 软阈值(detail) × 边缘保护, 0, 1)
```

### 自动分批与设备卸载

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

## 效果案例

以下为旧版效果案例：原视频 480×832（33 帧，约 2 秒）经 PiD 4x 超分至 **2304×4096**，再经去频闪与 hard/median/12 锐化。历史素材保留用于对照，不代表当前人像推荐配置。

| 案例 | 文件 |
| --- | --- |
| 原视频 | [案例-原视频.mp4](../assets/案例-原视频.mp4) |
| 4 倍超分 + 后处理 | [案例-4倍超分.mp4](../assets/案例-4倍超分.mp4) |

![效果对比图](../assets/效果对比图.png)

> 说明：超分视频为 4K 竖屏（2304×4096），文件较大，下载后建议本地播放器或剪辑软件查看；对比细节可重点看发丝、衣物纹理与主体边缘线条的锐度。

## 模板工作流结构

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

## 节点

### Musefish PiD Batch Video Upscale

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

### 日志与结果边界（FAQ）

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

## 长视频处理建议

- 先将 `VHS_LoadVideo.frame_load_cap` 设为少量帧验证，例如 `2` 或 `4`。
- 确认输出尺寸和模型参数正确后，再增加帧数。
- PiD 超分显存不足时优先降低 `batch_size`，不要改变帧顺序。
- 后处理支持自动分批与 OOM 缩批重试，但仍需整段 CPU 输入/输出存储，不是无限长视频流式处理；内存不足应在加载端缩短片段。显存紧张先用 `frames_per_batch=1`，或显式 `device=cpu`。
- 固定模型输入为长边 `1024`，`upscale_factor=2/3/4` 分别交付约 2048/3072/4096 长边结果；模型计算量按 4 倍路径固定。
- 通过 `VIDEO` 输出连接 `SaveVideo`，由 ComfyUI 统一编码和保存音频。

## 视频稳定性

同一次节点执行会生成一个固定随机噪声模板，并在所有帧批次间复用；`batch_size` 改变不会改变帧对应的随机噪声序列，避免批次边界出现明显闪烁。

如果仍有局部细节闪动：

- 保持 `seed` 固定；
- 使用 `batch_size=1` 先确认模型与 VAE 配置；
- 确认 `encode_vae` 使用 `Flux\\UltraFlux-v1-vae.safetensors`；
- 确认输入帧没有被 `force_rate` 或 `select_every_nth` 大幅抽帧；
- 先用 2–4 帧短片测试，再增加视频长度。

## 视频相关文件

- `musefish_nodes.py`：PiD 超分、自动分批去频闪、频率分离锐化节点及扩展注册。
- `pid_runtime.py`：仅对兼容 PiD 像素块启用独立 MLP 分块；全幅 attention 不切图，模型 clone 的临时对象补丁在成功、异常和中断后恢复。
- [Musefish_PiD_Batch_Video_Upscale.json](../workflows/Musefish_PiD_Batch_Video_Upscale.json)：视频超分与后处理模板。
- 模板 UUID：`d7de7df1-0bb0-4cf8-bb1e-6f7ee7c5d1d2`。
