# ComfyUI-Musefish-Nodes

本节点包提供两条独立处理链路：视频超分与后处理、音频超分与母带处理。请按素材类型阅读对应章节；两类流程的模型、参数和显存参考不可混用。

- [视频超分处理](#视频超分处理)
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

**PiD 内部分块不是空间切图：** `pixel_chunk_size` 只限制像素 Transformer 中独立 patch 的完整 MLP 支路，attention 仍处理全幅 patch 序列。提前释放 attention 临时张量，避免它们与 MLP 峰值重叠。较小分块可能增加内核调用开销，不应越小越好；本机复测保留 `1024` 作为默认值（`2048` 未见有意义的端到端收益，详见下表）。

**attention_backend（默认 `Kitchen`）：** 通过当前 PiD 的 `ModelPatcher` clone 写入 `transformer_options["optimized_attention_override"]`，只影响本节点这次采样，不修改 ComfyUI 全局 attention。`Kitchen` 使用 Comfy Kitchen int8 attention；本机短片测量延迟较低，但显存占用取决于形状和批次，不能概括为更省。量化与累积舍入会使输出与原始 PyTorch/cuDNN 路径有数值差异，不保证逐像素一致。`cuDNN` 使用 Comfy 已有 `attention_pytorch`，在 CUDA 上以局部 SDPA 上下文优先允许 cuDNN，并保留 MATH 作为不支持形状的兼容回退。当前构建没有 Kitchen 时，选择 `Kitchen` 会记录明确日志并自动切换 `cuDNN`；不会吞掉其他运行时异常。

**同条件短片测量（用于定位取舍，不是所有素材的速度承诺）：**

| 场景 | 设置 | 观测 |
| --- | --- | --- |
| PiD 节点，Kitchen | 2 帧、2304×4096、同一模型、4 步、seed=0、batch=2、`pixel_chunk_size=1024` | 13.7346 秒，显存分配 9.72 GiB |
| PiD 节点，cuDNN | 同上 | 21.1101 秒，显存分配 8.57 GiB |
| Kitchen 对 cuDNN | 上述同条件 | 延迟降低约 35%，约 1.54×；这是该测量条件的结果，不是通用加速保证 |
| Kitchen 旧 2 帧视觉对照 | 短片对照 | PSNR 51.6–51.8 dB；不是逐像素相同，长运动质量尚未建立结论 |
| batch=3 复测 | 6 帧、同一模型与采样设置 | 79.62 秒，分配 12.81 GiB、保留 15.64 GiB；输出与 batch=2 不同，16GB 显卡不推荐 |
| `pixel_chunk_size` 逆序复测 | 6 帧，2048 对 1024 | 57.82 对 58.09 秒（约 0.46% 差异），6 帧结果相同；因此保留 1024 |

合成 attention 基准曾测得 2.4–2.69×，只反映 attention/内核片段，**不是端到端 PiD 节点提速**。以上结果来自短片和固定参数；分辨率、批次、设备、模型加载状态或素材变化都可能改变耗时和显存，不能据此承诺任意速度提升。Kitchen 与 cuDNN 的输出也不保证逐像素一致；长视频运动质量需要单独检查。

#### 日志与结果边界（FAQ）

- **为什么每个批次都出现 `Model Initialization complete`？** 这是 DynamicVRAM tqdm 首次更新时附带的通用后缀。每批确实会执行准备步骤，但该文字不是磁盘权重每批重新加载的证据。
- **一次 live cuDNN 记录为什么总耗时较长？** 该次记录总计 249.20 秒，其中 PiD 约 230 秒，后续批次约 17 秒，GPU 利用率 99–100%。它与之前不同长度的任务不能直接比较。
- **之前的整段任务为什么停滞？** 原因未确定；不要仅凭停滞现象归因于 attention backend。
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

## 音频超分处理

### 功能概览

- **`Musefish UniverSR Model`**：准备 general / speech 模型缓存，可按需下载；输出缓存目录，不处理音频。

- **`Musefish UniverSR General Audio`**（节点 ID：`MusefishUniverSRGeneralAudio`）：固定使用 general 模型，支持 `sr`（仅超分）、`master`（V8 母带）、`sr_master`（超分+母带）和 `stem_mix`（Demucs 分轨+混音）。
- **`Musefish UniverSR Speech Audio`**（节点 ID：`MusefishUniverSRSpeechAudio`）：固定使用 speech 模型，仅支持 `sr` 超分；不暴露模型选择，适合人声/语音增强。
- 两个节点都处理标准 `AUDIO`（`waveform` 为 `B,C,T`），逐批支持单声道/立体声，并通过 `model_cache` 输入连接 `Musefish UniverSR Model` 的缓存输出。
- `input_sr=auto` 保留原 99% rolloff 的 8/12/16/24 kHz 基准档位；仅 general 模型追加高频保留复核。候选频带能量须占全谱至少 1e-5（-50 dB），具有至少约 300 Hz 的连续频带，且该频带同时高于噪声底 6 dB、距候选带峰值不超过 20 dB；至少 25% 分析帧有能量，并有连续 3 个分析帧的证据（不足 12 帧时要求 2 帧）。满足条件只向上校正：4–6 kHz 内容对应 12 kHz 档，6–8 kHz 对应 16 kHz 档，8–12 kHz 对应 24 kHz 档，带边留 150 Hz 余量。复用同一次 FFT，不因容器标称 48 kHz 一律升档。日志记录 base/corrected 档位、连续频带宽度和帧占比。
- 该复核用于避免低能量但真实的持续高频被 99% rolloff 忽略，并不能保证 general 模型完美区分音乐、噪声与残响；静音、低底噪、宽带 hiss、孤立尖峰或瞬态通常不会满足连续频带和时域门限。speech 模型保持原 rolloff 选择；手动 `input_sr` 不检测、不改写。
- 更新后需重启 ComfyUI 以加载主进程中的选档逻辑。24 kHz 是模型支持的最高输入档位，不能保证保留原曲 12 kHz 以上的细节；该校正不改变母带链，也不保证所有音乐都能改善听感。

### 运行环境与模型路径

worker 使用启动 ComfyUI 的 `sys.executable`，支持 Python 3.10–3.13 的 ComfyUI 环境；具体 Torch/音频依赖由当前环境提供。`Musefish UniverSR Model` 节点默认将模型缓存到 `ComfyUI/models/UniverSR/models/huggingface` 下的 `models--woongzip1--universr-audio` 或 `models--woongzip1--universr-speech`，仅在 `download=true` 时调用 `huggingface_hub.snapshot_download`，不会在导入插件时联网。也可通过处理节点的 `model_cache` 输入或 `MUSEFISH_UNIVERSR_MODEL_CACHE` 覆盖缓存目录；`stem_mix` 使用同一 Python 环境中的 Demucs 模块。

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

### 处理模式与适用场景

| `mode` | 处理链路 | 适用与注意事项 |
| --- | --- | --- |
| `sr` | UniverSR 超分 → 输出清理 | 真正低带宽的音乐/音效；speech 节点只提供此模式 |
| `master` | 跳过模型超分 → V8 母带链 | 不需要补带宽、只希望调整频谱与动态的素材；不执行超分，ODE、guidance 和超分分块参数不参与此路径 |
| `sr_master` | 超分 → 母带 → 输出清理 | 同时需要带宽重建与母带处理；立体声路径对增强后的 Mid 做母带，再合回 Side |
| `stem_mix` | Demucs 分人声/伴奏 → speech 处理人声、general 处理伴奏 → 伴奏母带 → 混音 | 需要分轨增强；依赖更多模型，资源与耗时通常更高，16GB 双分支模板不代表此模式已做显存保证 |

**自动校正可以用于 `sr_master`，无需新增节点。** General 的自动选档发生在模式分流之前，超分部分直接使用校正结果；`stem_mix` 当前将选定档位传给分轨处理，不是分别对每条分轨重新自动识别。`master` 不做超分，日志中的选档信息不表示它会执行模型带宽重建。

### 处理节点参数

以下是**新建节点的默认值**，与上面的模板预设分开看。

| 参数 | 默认值 / 范围 | 作用与调节建议 |
| --- | --- | --- |
| `audio` | 必接，标准 `AUDIO` | 输入音频；支持批次及单/双声道，输出为 48 kHz |
| `mode` | `sr`；四种模式 | 仅 general 节点显示；speech 固定仅超分 |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000` | 表示内容有效带宽对应的输入采样率，不是期望输出采样率；16k 档对应约 8kHz 带宽。优先自动，有可靠带宽依据时手动覆盖 |
| `channel_mode` | `auto` / `mono` / `stereo` | `auto` 按输入声道数选择，不是按 general/speech 强制选择；`mono` 会将双声道平均，`stereo` 会将单声道复制为双声道，但不能凭空恢复真实声场 |
| `ode_method` | `midpoint`；`euler` / `midpoint` / `rk4` | 超分求解方法；先使用 midpoint，不以更复杂的求解方法保证音质改善 |
| `ode_steps` | 4；1–25 | 超分求解步数；提高会增加计算量，不会恢复被错误输入档位预先滤除的原始信息 |
| `guidance` | 1.5；0–5，步长 0.1 | 条件引导强度；提高可能强化生成纹理，也可能增加不自然感。先保留 1.5，再做同种子短片段对比 |
| `chunk_sec` | 15；1–120 秒 | 单次超分分块时长；内部分块带约 50ms 重叠拼接。显存不足时减小，不建议长音频直接使用最大值 |
| `seed` | 0；非负整数 | 随机种子；前端生成后策略选 fixed 便于 A/B，模板使用 randomize |
| `model_cache` | 可选 STRING 连线 | 连接模型节点输出；未连接时使用配置的默认模型目录。切换模式不会把 general 处理节点变成 speech 节点 |
| 输出 `audio` / `log` | AUDIO / STRING | 分别连接音频保存节点和文本展示节点；日志包含实际选档及分块进度 |

### 模型准备节点参数

| 参数 | 新建节点默认值 | 说明 |
| --- | --- | --- |
| `model` | `general` | `general` 下载音乐/音效模型，`speech` 下载语音模型；与对应处理分支匹配 |
| `download` | `false` | `true` 时请求下载模型到缓存；模板预设为 true。离线使用前先确认所需权重完整 |
| 输出 `model_cache` | STRING | 缓存目录引用，不是 AUDIO，不应串在音频信号连线上 |

### 自动带宽校正与听感排查

- **容器采样率不等于有效带宽。** 48 kHz 文件也可能只包含低带宽内容；反过来，99% 能量集中在低频也不代表其余高频可以丢弃。General 的二次复核用于减少这种低估，具体门槛见本节开头。
- 示例日志：`auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: ...)`。`base` 是原检测档位，`corrected` 是实际采用档位，不表示输出从 16kHz 改为 24kHz；音频输出仍为 **48kHz**。
- **超分后更闷**：先查实际 `input_sr`；再用固定种子、同一短片段对比 `sr` 和 `sr_master`。母带链仍会塑形高频，自动升档不能保证消除它对音色的影响。
- **已有完整高频的成品曲**：不要仅因是 MP3 就判为低清。最高 24k 输入档仍可能滤除约 12kHz 以上的原始内容；优先保留原曲，确有母带需求再对比 `master`，而非盲目超分。
- **高频噪感或声场变化**：超分会生成纹理，立体声路径还包含 M/S 处理；先比较仅超分与原曲，不要只靠提高 guidance、步数或音量判断品质。
- **旧自动行为仍存在**：更新节点代码后需重启 ComfyUI。下次确认日志出现 base/corrected 判定，并避免复用旧缓存结果；固定种子用于公平比较，修改代码后的首次验证可换一个种子触发重新执行。
- **缺模型/缺节点**：检查模型缓存、对应 general/speech 权重，以及模板的 rgthree、文本展示和 `SaveAudioAdvanced` 节点；保存节点不可用时升级 ComfyUI 或用当前版本提供的音频保存节点替换。

### 音频相关文件

- `musefish_audio.py`：模型准备节点、General/Speech 处理节点、AUDIO 张量适配与隔离 worker 调度。
- `audio_backend/processing.py`：自动带宽检测与 general 校正、分块超分和处理模式分流。
- `audio_backend/dsp.py`：母带及音频后处理。
- `audio_backend/worker.py`：独立处理进程入口。
- [Musefish_UniverSR_Audio.json](workflows/Musefish_UniverSR_Audio.json)：音乐/语音双分支参考模板，分别连接模型缓存、音频保存和日志展示节点。
