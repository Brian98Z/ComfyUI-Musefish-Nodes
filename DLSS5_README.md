# Musefish DLSS5 超分节点 — 使用说明

> **硬件要求：NVIDIA RTX 50 系（Blackwell）独占。** DLSS 5 超分
> （NGX Feature 18）与配套的 RTX 视频超分运行库只随 Blackwell 驱动
> 提供；40 系及更早显卡无法创建该 Feature，节点会直接报错。
> 已验证环境：RTX 5070 Ti + 616.56 驱动。

## 这是什么

把 DLSS5Tool 的 NGX Feature 18 封装为两种 ComfyUI 节点：
`MusefishDLSS5NeuralRender` 处理单图和短视频 IMAGE 帧批次（固定单 worker）；
`MusefishDLSS5VideoStream` 接收 VIDEO、逐帧处理并返回 VIDEO，
用于高清和长视频。两者都属于 `Musefish/Video`。

## 文件

- `musefish_dlss5.py` — 节点本体（schema + execute）
- `musefish_dlss5_stream.py` — 长视频单会话解码、增强、编码与音频封装
- `dlss5_backend/session.py` — 会话管理（子进程生命周期、共享内存帧协议）
- `dlss5_backend/worker.py` — 子进程入口（ctypes 驱动 DLL + 共享内存 pinned 注册）
- `models/dlss5/` — ComfyUI 的模型目录（可以由 `--models-directory` 指定），存放四个外部 DLL；不随插件分发
- `workflows/Musefish_DLSS5_Video_Segments.json` — 长视频分段增强案例工作流

## 运行依赖与手动部署（Windows）

### 必需的 NVIDIA/DLSS5Tool 运行库

IMAGE 与 VIDEO 两种节点调用同一个隔离 worker。1× 模式需
`dlssnr_host.dll` 和 `nvngx_dlssnr.dll`；高于 1× 的 VSR 模式还需
`vsr_host.dll` 和 `nvngx_vsr.dll`。把文件放进 ComfyUI 当前模型目录的
`dlss5/` 子文件夹：默认 `ComfyUI/models/dlss5/`；使用 `--models-directory`
时则为该参数指定目录下的 `dlss5/`，而非插件的 `dlss5_backend/dlls/`。

| 文件名 | 用途 |
|---|---|
| `dlssnr_host.dll` | Feature 18 神经渲染宿主；当前代码使用 legacy 宿主 |
| `nvngx_dlssnr.dll` | Feature 18 NGX runtime，由宿主初始化时加载 |
| `vsr_host.dll` | RTX Video Super Resolution 宿主；仅放大倍率高于 1× 时使用 |
| `nvngx_vsr.dll` | VSR NGX runtime |

用户提供的四文件分享地址：<https://pan.quark.cn/s/d4c04dc33d25>。
下载后检查文件名与下表一致，再放入上述 `models/dlss5/` 目录并重启
ComfyUI。此链接是用户提供的云盘分享，**不是 NVIDIA 或 DLSS5Tool 官方
发行地址**；请自行确认获取与使用许可，不要从不明 DLL 镜像站下载。
如已有合法取得的 DLSS5Tool 安装包，也可从其 `_internal/` 手动提取相同
文件。没有核实到这四个私有运行库的官方独立下载 URL。NVIDIA 公共
[DLSS SDK](https://github.com/NVIDIA/DLSS) 不能代替这里的宿主与运行时。
此节点不读取 `models/dlss5/nvngx_dlss.dll` 或 `models/dlss5/caller/`：
它们可能由另一款 RH-DLSS5 插件使用，不属于 Musefish 的四文件清单。

NVIDIA 显卡驱动必须安装并能提供 NGX/D3D12 支持；本节点按项目当前实现要求
ComfyUI 所用 Python 能看到 CUDA 设备。驱动应从 [NVIDIA 官方驱动下载页](https://www.nvidia.com/Download/index.aspx)
按显卡型号/操作系统选择。节点不要求
另装独立 CUDA Toolkit 来运行 NGX；worker 会尝试从当前 ComfyUI PyTorch 安装中定位
`cudart64_*.dll`，仅用于共享内存 pinned 注册加速。找不到 CUDA runtime 或注册失败时，
仍可处理，只是使用普通 pageable 内存路径。勿替换 ComfyUI 自带的 CUDA 版 PyTorch。

### Python 与视频工具

- IMAGE 节点使用当前 ComfyUI 环境提供的 PyTorch/CUDA，并依赖 `numpy`、`cv2`；
  VIDEO 节点复用同一图像处理依赖。`requirements.txt` 包含 NumPy，但不包含
  OpenCV；缺少 `cv2` 时在 ComfyUI 使用的 Python 环境安装 `opencv-python`，
  避免同时安装 GUI 与 headless 两种 OpenCV 包。PyTorch/CUDA 沿用 ComfyUI
  自身安装，不要按独立教程覆盖。
- VIDEO 流式节点另外要求 `ffmpeg` 与 `ffprobe` 两个可执行文件都能通过 `PATH` 查到；
  两者用于读视频/探测媒体，以及生成带音频的中间 MP4。IMAGE 节点不调用外部 FFmpeg。
  可从 [FFmpeg 官方下载页](https://ffmpeg.org/download.html) 进入 Windows 构建来源，
  下载后解压并把含 `ffmpeg.exe`、`ffprobe.exe` 的 `bin` 目录加入启动 ComfyUI 的
  进程 `PATH`，然后重启 ComfyUI。FFmpeg 官网链接到外部 Windows 构建提供方；请遵循
  所选构建的许可证与分发条款。

依赖职责总结：两种节点均需 NVIDIA RTX/NGX 驱动能力、Feature 18 宿主与
运行时 DLL、ComfyUI CUDA/PyTorch、NumPy 与 OpenCV；>1× 模式另需 VSR 两个
DLL；仅 Video Stream 额外需 PATH 中的 FFmpeg/ffprobe。CUDA runtime 的 pinned
内存加速是可选项，不是额外安装门槛。

## 架构

NGX 宿主在**隔离子进程**里运行，ComfyUI 主进程永不加载 NVIDIA DLL——
驱动崩溃不会带垮服务。帧数据走共享内存环，父子各一次拷贝、worker 侧零拷贝。

```
ComfyUI 主进程                           worker 子进程（每个 1 个）
  OpenCV 逐帧转 RGBA8 → slot 0/1 →     dlssnr_process / vsr_process
  RGBA8 紧凑化后转 float32 ← slot 0/1 ← （NGX 引擎在 GPU 上）
```

- **2 深度双缓冲流水线**：父进程填 slot n+1、取 slot n-1 的结果时，
  worker 正在算第 n 帧，父进程的像素转换完全落在引擎的窗口里，
  GPU 不再在帧间空转。协议保持「一请求一有序回执」，时序历史与
  输出字节与串行会话一致（逐比特一致，已回归验证）。
- **短视频 IMAGE 节点固定单 worker**，保留一个会话内的连续时序历史。
  既有 480P 视频并行实测：1 路 31s/17% 显存、2 路 23s/21%、
  4 路 39s/31%、8 路 40s/49%、16 路 41s/85%、32 路 176s/99%。
  这组数据说明显存占用率增加不等于吞吐改善；720P 长视频各档耗时
  159–168s（1–8 路），16 路 233s/98% 显存、32 路 OOM。
  长视频改用新的连续流式节点，不把所有帧作为 IMAGE 批次放进内存。
- **`reset_every_n_frames`**：0 = 只在每个块首帧重置（单会话连续）；
  1 = 每帧独立（等同旧版硬编码行为）；N>0 时按帧号在引擎内就地重置，
  不再重建会话。

一次执行 = 一个会话池 = 若干段完整时序历史（对应 DLSS5Tool
「严格时序（单会话）」导出模式）。

参数与 DLSS5Tool 一一对应：

| 节点参数 | DLSS5Tool 对应 |
|---|---|
| style: default/natural/cinema | 默认/自然/电影 |
| intensity (0–1) | 强度 |
| local_tone (0–1) | 本地色调 |
| local_struct (0–1) | 本地结构 |
| skin_struct (0–1) | 皮肤蒙版强度 |
| use_auto_mask | 皮肤蒙版开关 |
| reset_every_n_frames (0=never) | 时序历史重置（0 为单会话连续；1 为逐帧独立，等同旧版行为） |
| `放大参数`（IMAGE 和 VIDEO；输入 ID `super_resolution`） | 首个控件：`off`、`1× (Native)`、`1.5× (Quality)`、`2× (Balance)`、`3× (Performance)`、`4× (Ultra)`、`1K`、`2K`、`4K`、`8K`。1× 不放大，2×/4× 是 VSR 原生整数倍率；1.5×/3× 由 2×/4× VSR 后 `INTER_AREA` 缩小合成。标签中的 Quality/Balance/Performance/Ultra 是输出档名，不会覆盖第三个控件 `vsr_quality`。K 档按短边目标 1080/1440/2160/4320 挑选可达的最小倍率，横屏与竖屏均支持：1K 可输出 1920×1080 或 1080×1920（取决于输入方向与比例）。输出和 VSR 中间图均不超过 7680×4320 的像素预算；720p→4K 采用 4× 5120×2880 中间图后缩至 3840×2160；1080p→8K 采用 4×，720p→8K 因需超过 4× 会明确拒绝。IMAGE 输出是整个 float32 批次，超过 1 GiB 时拒绝并建议使用流式 VIDEO。 |
| vsr_quality: performance/balanced/quality/ultra | VSR 质量档（NGX PerfQuality 1–4；ultra 为 DLSS5Tool 默认） |
| keep_session: auto/off | auto 在同一任务连续分批期间复用热 worker，任务结束或中断后约 2–3 秒关闭并释放其显存；off 每次节点执行后立即关闭 |
| parallel_workers | IMAGE 节点已移除控件并固定 1 路，长视频使用流式节点 |

超出 [0,1] 的「5× 实验范围」参数已被夹紧（对应 GUI 默认关闭实验范围）。

## 性能（2026-09-23 基线，i5-14600K + RTX 5070 Ti）

真机负载：480×848 801 帧视频 → 2× VSR + Feature 18 ultra（工作流
`LoadVideo → DLSS5 → VideoCombine`）。

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 节点耗时 | 29.76s | **16.75s**（1.78×） |
| 节点期间 GPU 利用率 | 22–26% | **25–49%** |
| 节点期间主进程 CPU | ~12.5–13.4 核 | **~1.5–2.2 核** |
| 逐帧转换 CPU | 330–380 CPU-ms/帧 | **~30 CPU-ms/帧** |
| 整链（含 VHS 解码 + 编码） | 70.81s | **53.31s** |

三处改动：

1. **逐帧像素转换改走原版 numpy C 循环**（原先用 torch 逐算子转换：
   每帧 8 个小 kernel 走 intra-op 线程池，1–2MP 的负载被并行区
   开销淹没，实测 cpu/wall ≈ 10–13，即十几个核在搬 20MB）。
2. **2 深度双缓冲流水线**：父进程转换与引擎 GPU 计算重叠，GPU 不再
   在帧与帧之间空转（引擎单帧 ~20–30ms 的提交/同步延迟原先全是空洞）。
3. **连续分块并行（`parallel_workers`）**：当时低分辨率基准用 2 路，
   吞吐再 ×1.2–1.7（GPU 24% → 40–50%）；高分辨率负载默认改为 1 路。

2026-09-24 增量优化：输入 RGB float32→RGBA8 采用 OpenCV 饱和转换和通道封装；
输出 RGBA8 先压实 RGB8 再转 float32，避开 NumPy 对跨步 RGBA 视图的慢速
逐像素除法。480×848→960×1696 的 120 帧、2 worker 热会话交错 A/B：
低占用时旧路径热跑 61.5–65.9 fps，新路径 75.2–94.1 fps；另一次 GPU 已
饱和（约 99%）时旧/新均在 43–46 fps 左右，瓶颈不在转换。整批输出逐比特
一致；单独微基准输出转换 16.7→2.8 CPU-ms/帧。GPU 占用受其他任务
与时钟影响，不把利用率数字视为画质或硬件算力增加。
同时修复 `super_resolution=off` 的倍率解析（原先尝试 `float("off")`
导致 ValueError）；真机 off、1.5×、2× 均完成输出尺寸与范围冒烟。

> ⚠️ **行为变更（重要）**：旧版 `session.process()` 把 `reset: True` 写死，
> 每帧都重置 NGX 时序历史（节点里算好的 `reset` 从未被使用），
> 等于逐帧独立处理。现在 `reset_every_n_frames=0`（默认）会**保留**时序
> 历史（块内连续），输出与旧版有差异——实测同一素材约 7% 像素变化
> >8/255、mean abs diff ≈3.4/255（时序累积带来的降噪/细节差异）。
> 需要与旧版逐比特一致：把 `reset_every_n_frames` 设为 **1**（已回归验证，
> workers=1/2 均与旧生产路径逐比特一致）。

## 已知限制

1. **必须 v2 宿主的现象**：本机（RTX 5070 Ti + 616.56 驱动）实测
   `dlssnr_host_v2.dll` 首次 `dlssnr_process` 内部等待不返回
   （疑似与 NVIDIA App overlay 的 NGX 组件冲突），**legacy 宿主
   `dlssnr_host.dll` 完全正常**（256px 约 4ms/帧），故 worker 固定用
   legacy。若日后升级驱动想试 v2，改回 `worker.py` 里的 DLL 名即可。
2. 仅支持 SDR RGBA8 路径；HDR 高精度（RGBA16F/PQ/HLG）路径未封装。
3. 流式 VIDEO 的单帧输出及 VSR 中间图上限为 7680×4320 像素；
   IMAGE 节点同用该尺寸边界，但限制整个 float32 输出批次为 1 GiB。
4. `keep_session=auto` 只缓存完全同契约的会话池；参数变化后旧池关闭。
   任务队列不再包含 DLSS5 工作时，后台检查空闲满 2 秒自动关闭 worker；
   VHS Meta Batch 续批期间保持热池，不增加每批冷启动开销。
5. 帧传输走共享内存，worker 侧把两个环缓冲 `cudaHostRegister` 为页锁定
   （pinned）内存，NGX 宿主的 D3D12 每帧 DMA 上传/下载跳过分页暂存拷贝
   （1080p off 档实测稳态 23.0ms → 21.1ms/帧，约 +8%；找不到 CUDA runtime
   时自动退回普通模式，功能无影响）。GPU 独占时实测：off 1080p ≈21ms/帧、
   2× 超分到 4K ≈70ms/帧。GPU 被其他任务占用时按占比变慢；
   attention/PyTorch 加速对该引擎不适用（NGX 为 D3D12 闭源推理）。
6. 节点内部按帧流式处理（OpenCV/NumPy 逐帧转换 RGBA、直接写入输出张量，
   无整批中间副本），自身峰值内存 ≈ 上游输入批次 + 输出批次。
   上游 VHS_LoadVideo 不设 frame_load_cap 时会把整段视频装进内存，
   长视频请配合 frame_load_cap/skip_first_frames 分段。
7. IMAGE 节点固定单 worker；长视频请用 `MusefishDLSS5VideoStream`。
   改动前的并行性能数据仅作历史对照，旧工作流重新保存时移除旧控件。

## 高清长视频：单次运行的流式节点

按画布连接：`LoadVideo (VIDEO) → MusefishDLSS5VideoStream (VIDEO) → SaveVideo`。
中间节点接受文件支持的 VIDEO（LoadVideo 返回磁盘文件引用），逐帧处理并
写入 ComfyUI temp 下一个 MP4，再将文件支持的 VIDEO 交给 SaveVideo 保存到
output。一次运行只保持一个 NGX 会话、两个 RGBA8 共享内存槽和单帧编解码
缓冲；无 900 帧 IMAGE 批次，也不用手动拼接。时序历史从首帧到末帧连续。
中间临时名固定为 `Musefish/DLSS5_stream_<随机值>.mp4`，不暴露前缀控件；
最终成片名称只在 SaveVideo 的 `filename_prefix` 设置。
原音频转 AAC；低于 8K 使用 H.264，`libx264` 的 CRF 默认 19（可调）。
失败或中断时清理未完成 `.part.mp4` 并关闭子进程及 worker。
`encoder` 默认 `libx264`，`crf` 仅控制此模式。可选 `h264_nvenc` 用 GPU 的
NVENC 编码器，固定 CQ 27（并非与 CRF 19 等画质）；在 1280×720@30fps
输入、2× 超分、连续 60 秒（1800 帧）的本机对照中，整链路耗时
74.1s → 65.1s，内层 MP4 大小 95.5MB → 99.1MB。
在本机 GPU 上 H.264 NVENC 宽度超过 4096 会失败；选择 GPU 编码时，
8K 横屏或竖屏自动切换 HEVC NVENC（MP4 `hvc1` 标记，CQ 27）。
`libx264` 不支持本节点的 8K 输出，选择 8K 时需选择 GPU 编码。
已验证 720p→4K 为 3840×2160、1080p→8K 横屏为 7680×4320、
1080p→8K 竖屏为 4320×7680，均含 AAC 音轨。
其他片源的画质和速度需自行比较；NGX 是单会话时序处理，不能仅凭显存空闲
增加并发而保持同等时序结果。
本机测试视频的名义 30fps 与平均帧率有细微差异，流式处理沿用名义帧率，
并验证输出尺寸、帧率、帧数后才返回 VIDEO。
可接 `Video Slice`：读取 VIDEO 的裁剪窗口，对解码与原音频采用相同起点
和时长，并限制解码帧数，防止边界多出一帧。旧版只读取底层文件路径，
忽略 `Video Slice` 的时间窗口；60 秒切片实际会处理原片全部 566.5 秒。
处理进度通过 ComfyUI 原生进度事件发送，按已写入编码器的帧数计算；
封装和验证完成前最多显示 99%，成功落盘后才显示 100%。
前端显示的进度是该节点的帧处理进度，不包含下游 SaveVideo。
`workflows/Musefish_DLSS5_Video_Segments.json` 是历史手动分段示例，
不再是长视频推荐路径。

## 适用范围与案例

IMAGE 节点可接 LoadImage 与短视频帧批次；长视频接上面的三个 VIDEO 节点。
案例效果见 [`assets/DLSS5超分案例.mp4`](assets/DLSS5超分案例.mp4)
（1472×1280@24fps，10 秒；2× RTX VSR + Feature 18）。

## 验证记录（2026-09-23 优化回归）

- **协议等价**：同语义（`reset_every_n_frames=1`）下，2 深度流水线、
  连续分块 2/3 workers 的输出与原串行路径**逐比特一致**（72/20 帧真机素材）。
- **退路等价**：`reset_every_n_frames=1` 与旧生产路径（`session.process()`
  硬编码 reset=1）逐比特一致，workers=1/2 均通过。
- **逐帧转换等价**：numpy 路径与原 torch 路径的 RGBA8 输入字节 60/60 帧
  完全一致；uint8 边界用例（0.5/255、2.5/255、负数、>1）字节一致；
  float32 输出逐比特一致。
- **端到端**：8988 实例 `LoadVideo → DLSS5(workers=2) → VideoCombine`
  全链 success（`output/Musefish/DLSS5_video_00015.mp4`）：节点 16.75s
  （旧 29.76s）、整链 53.31s（旧 70.81s），节点期间 GPU 25–49%、
  主进程 CPU ~1.5–2.2 核（旧 ~13 核）。
- **缓存修复**：`keep_session=auto` 原先只在「已有缓存」时才写回缓存，
  冷启动路径从不入库（等于永远不缓存）；现在同契约第二次执行
  5.52s → 2.95s（2 workers），输出一致。
- **并行扫描**：400 帧真机素材，K=1/2/3 吞吐 18.2/22.4/23.8 fps，
  GPU 20%/36%/42%；K=4 因显存/调度反噬降到 12.0 fps，故上限设为 3。

## 验证记录（2026-09-20）

- 8988 实例 API 提交 LoadImage → MusefishDLSS5NeuralRender → SaveImage
  全链路 success，输出 `output/dlss5_test_00001_.png`。
- 与原图逐像素对比：54.4% 像素变化 >8/255，mean abs diff 8.18，
  视觉上细节增强符合 Feature 18 预期。
- 2× RTX VSR + Feature 18 全链路 success，输出 1728×2304
  （`output/dlss5_sr2x_00001_.png`），四个 VSR 质量档均实测通过。
- 共享内存 IPC 替换 PNG 后复测：8988 全链 off/2x 均 success
  （`shm_off_00001_.png` / `shm_2x_00001_.png`），尺寸正确
  （864×1152 / 1728×2304）；GPU 独占时 off 1080p ≈18.5ms/帧、
  2×到 4K ≈70ms/帧。
- 1.5x 合成档 + 会话缓存：864×1152 输入 1.5× 输出 1296×1728 success；
  连续两次执行 2.3s → 1.0s（缓存命中省 NGX 冷启动）；离线基准
  热会话首帧 34ms vs 冷启动 1271ms（37×）。
- pinned 共享内存上线后回归：39 passed / 4 subtests（pytest，嵌入式
  Python，`--import-mode=importlib`）；开/关会话 3 轮循环无泄漏；
  真实用户负载（1080p→2x，keep_session=auto）在新代码下连续成功出片。
