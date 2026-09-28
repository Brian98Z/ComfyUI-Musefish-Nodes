# ComfyUI-Musefish-Nodes

[English](README.md) | **简体中文**

按用途选择节点：PiD 视频超分与后处理、DLSS5 图像/流式视频增强、社交平台视频下载、UniverSR 音频超分与母带处理。各链路的模型、参数和显存参考不可混用。安装到 `ComfyUI/custom_nodes/ComfyUI-Musefish-Nodes`，在运行 ComfyUI 的 Python 环境安装所需依赖并重启；依赖细节见各分项说明。

| 用途与节点 | 功能亮点 | 关键参数与注意事项 | 完整说明 |
| --- | --- | --- | --- |
| PiD：`MusefishPiDBatchVideoUpscale`、`AutoBatchAntiflicker`、`AutoBatchImageSharpenFS` | 固定 1024→4096 的 PiD 路径；逐批复用噪声模板；去频闪带相邻帧上下文；频率分离锐化可分批和 CPU 卸载 | PiD 推荐 `steps=4`、`sampler=lcm`、`latent_format=flux`；去频闪从 `luma_tmp=15/chroma_tmp=20` 起；人像锐化用 `hard/gaussian/6`、`amount=0.45`、`noise_threshold=0.01`。PiD 后处理后的 IMAGE 接视频合并，不能用未经后处理的 VIDEO 输出代替 | [PiD 与后处理](doc/PiD_README.md) |
| DLSS5：`MusefishDLSS5NeuralRender`、`MusefishDLSS5VideoStream` | IMAGE 批次或文件支持 VIDEO 连续增强；已知源分辨率时隐藏不可用倍率；8K 先在 ≤4K 画面增强再插值放大 | **仅 RTX 50 系**；运行库需用户自行提供。`Upscale` 选输出倍率，`vsr_quality` 独立控制 VSR 质量。IMAGE 整批 float32 输出上限 1 GiB；流式 VIDEO 的 CPU/GPU 编码在宽或高超过 4096 时分别切 HEVC `libx265`/`hevc_nvenc`，CPU `crf=19` 默认、GPU CQ 固定 27。8K 文件不等于 8K 原生神经渲染细节 | [DLSS5 安装与参数](doc/DLSS5_README.md) |
| 视频下载：`MusefishVideoDownload`、`MusefishWeChatChannels` | YouTube/B站/X/抖音/小红书下载及微信视频号本机客户端解密；返回文件支持 VIDEO 和 FPS | `url`、`platform=Auto`、`quality=best`、`chunks`（支持的平台可自动吸附 8）；YouTube/X 需本地代理，YouTube/抖音可能需要本地 cookies；视频号需客户端正在播放目标素材。带 token 的 URL 缓存及 cookie 文件均是凭据，不得上传 | [平台、凭据与解密](doc/VIDEO_DOWNLOAD_README.md) |
| 音频：`MusefishUniverSRModel`、`MusefishUniverSRGeneralAudio`、`MusefishUniverSRSpeechAudio` | general 音乐/音效超分与母带、speech 语音增强；素材驱动选档和参数；支持分块处理 | 模型目录 `ComfyUI/models/UniverSR/models/huggingface/<general\|speech>`；`download=true` 才下载。新建节点 `auto_params=true`、`input_sr=auto`、`ode_steps=4`、`guidance=1.5`、`chunk_sec=15`、`accel=cuDNN TF32`；显存不足先减小 `chunk_sec`。`stem_mix` 需要两个模型与 Demucs | [UniverSR 与母带](doc/UNIVERSR_README.md) |

## 安装与模型

- 共用 Python 依赖见 [`requirements.txt`](requirements.txt)；**DLSS5** 还需 ComfyUI 已有 CUDA/PyTorch、OpenCV、RTX 50 系驱动。高于 1× 的模式需额外 VSR 运行库；四个专有 DLL 放在 `ComfyUI/models/dlss5/`，不随仓库分发。[用户提供的分享地址](https://pan.quark.cn/s/d4c04dc33d25) **不是 NVIDIA 官方发行渠道**，下载前核对许可及文件名；详见 [DLSS5 安装说明](doc/DLSS5_README.md#运行依赖与手动部署windows)。VIDEO 流式节点还要求 `ffmpeg`、`ffprobe` 都在 `PATH`；[FFmpeg 下载入口](https://ffmpeg.org/download.html)。
- **PiD UNET/CLIP**：[PixelDiT 模型](https://www.modelscope.cn/models/Comfy-Org/PixelDiT/files)；**VAE**：[z-image/flux1 VAE](https://www.modelscope.cn/models/Comfy-Org/z_image_turbo/tree/master/split_files/vae)。具体文件名及加载器设置见 [PiD 指南](doc/PiD_README.md#节点)。
- **UniverSR** 的 general/speech 模型各需 `config.yaml` 与 `pytorch_model.bin`；模型节点仅在 `download=true` 时按 `HF_ENDPOINT`（默认 `https://hf-mirror.com`）下载。[缓存与模式说明](doc/UNIVERSR_README.md)。
- 下载平台依赖 yt-dlp；微信视频号另需本机客户端及插件自带的 wasm/Node 解密工具。代理、cookie、安全边界见 [下载说明](doc/VIDEO_DOWNLOAD_README.md)。

## 示例工作流

| 工作流 | 入口与使用前检查 |
| --- | --- |
| [PiD 超分与后处理](workflows/Musefish_PiD_Batch_Video_Upscale.json) | 提供 UNET/CLIP/VAE 和输入视频；16GB 显存从小批次开始 |
| [DLSS5 流式视频](workflows/Musefish_DLSS5_Video_Stream.json) | `LoadVideo → MusefishDLSS5VideoStream → SaveVideo`；先选自己的输入文件；默认 2×、CPU 编码 |
| [UniverSR 音频/语音](workflows/Musefish_UniverSR_Audio.json) | 16GB 参考，两条分支，使用前选择自己的音频和启用对应分支；额外依赖 rgthree、ComfyUI-Custom-Scripts |
| [视频下载](workflows/Musefish_Video_Downloader.json) | 配置分享链接/平台；视频号需先打开客户端并播放目标视频 |

## 注意与版本

- PiD/UniverSR 模板中的 16GB 设置是参考值，不保证所有素材不 OOM；PiD 长视频仍占 CPU 内存。DLSS5 长视频应优先用流式 VIDEO 而非整段 IMAGE 批次。
- 同一段 1080P、22 秒视频放大到 8K：**用户实测 GPU HEVC 79 秒、CPU HEVC 300 秒**；仅作为该素材和设备的对照，非通用速度承诺。[详细质量与编码说明](doc/DLSS5_README.md)。
- **v1.5.0（2026-09-28）**：DLSS5 英文 Upscale、源分辨率选项过滤、8K 稳定增强、CPU HEVC 与新流式模板。历次版本说明及验证数据见各分项文档；[GitHub v1.5.0 Release](https://github.com/Brian98Z/ComfyUI-Musefish-Nodes/releases/tag/v1.5.0)。
