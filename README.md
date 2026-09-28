# ComfyUI-Musefish-Nodes

**English** | [简体中文](README.zh-CN.md)

Choose a workflow by purpose: PiD video upscaling and postprocessing, DLSS5 image/streaming video enhancement, social video downloading, or UniverSR audio upscaling and mastering. Their models, tuning parameters, and VRAM guidance are not interchangeable. Install this directory under `ComfyUI/custom_nodes/ComfyUI-Musefish-Nodes`, install the dependencies required by your workflow in ComfyUI's Python environment, and restart ComfyUI.

| Purpose and nodes | Highlights | Key parameters and constraints | Full guide |
| --- | --- | --- | --- |
| PiD: `MusefishPiDBatchVideoUpscale`, `AutoBatchAntiflicker`, `AutoBatchImageSharpenFS` | Fixed 1024→4096 PiD model path; stable noise reused across batches; temporal filtering with adjacent-frame context; batched frequency-separation sharpening with CPU fallback | Start with PiD `steps=4`, `sampler=lcm`, `latent_format=flux`; antiflicker `luma_tmp=15`, `chroma_tmp=20`. Portrait sharpening: `hard/gaussian/6`, `amount=0.45`, `noise_threshold=0.01`. Encode the **postprocessed IMAGE output**, not PiD's raw VIDEO output | [PiD and postprocessing](PiD_README.en.md) |
| DLSS5: `MusefishDLSS5NeuralRender`, `MusefishDLSS5VideoStream` | IMAGE batches or continuous file-backed VIDEO; incompatible scales hidden when source dimensions are known; enhancement at ≤4K before final 8K resizing | **RTX 50-series only**; proprietary runtimes must be provided separately. `Upscale` selects output size; `vsr_quality` separately selects VSR quality. IMAGE float32 batches are capped at 1 GiB. Streaming VIDEO switches CPU/GPU encoders to HEVC `libx265`/`hevc_nvenc` above 4096 px in either dimension; CPU default `crf=19`, GPU fixed CQ 27. An 8K output is not native 8K neural-render detail | [DLSS5 setup and controls](DLSS5_README.en.md) |
| Download: `MusefishVideoDownload`, `MusefishWeChatChannels` | YouTube, Bilibili, X, Douyin, Xiaohongshu, and local WeChat Channels retrieval/decryption; returns file-backed VIDEO and FPS | `url`, `platform=Auto`, `quality=best`, `chunks` (supported platforms can auto-select 8); YouTube/X require a local proxy and YouTube/Douyin may need local cookies. Keep the WeChat client open playing the target. Cookie files and cached token-bearing URLs are credentials; never publish them | [Platforms, credentials and decryption](VIDEO_DOWNLOAD_README.en.md) |
| Audio: `MusefishUniverSRModel`, `MusefishUniverSRGeneralAudio`, `MusefishUniverSRSpeechAudio` | General music/SFX restoration and mastering, speech enhancement, content-aware settings and chunked inference | Models live under `ComfyUI/models/UniverSR/models/huggingface/<general\|speech>`; download occurs only with `download=true`. New-node defaults: `auto_params=true`, `input_sr=auto`, `ode_steps=4`, `guidance=1.5`, `chunk_sec=15`, `accel=cuDNN TF32`. Reduce `chunk_sec` if VRAM is low. `stem_mix` requires both models and Demucs | [UniverSR audio and mastering](UNIVERSR_README.en.md) |

## Dependencies and models

- Shared Python dependencies are listed in [`requirements.txt`](requirements.txt). **DLSS5** additionally uses ComfyUI's CUDA/PyTorch, OpenCV, and an RTX 50-series driver. Scaling above 1× requires the two VSR runtimes as well. Place the four proprietary DLLs under `ComfyUI/models/dlss5/`; they are **not shipped** in this repository. The [user-provided file share](https://pan.quark.cn/s/d4c04dc33d25) is **not an official NVIDIA distribution**; verify the license and filenames before use. See [DLSS5 deployment](DLSS5_README.en.md). Streaming VIDEO also requires both `ffmpeg` and `ffprobe` on `PATH`; [FFmpeg downloads](https://ffmpeg.org/download.html).
- **PiD UNET and CLIP:** [PixelDiT files](https://www.modelscope.cn/models/Comfy-Org/PixelDiT/files); **encode VAE:** [z-image/flux1 VAE files](https://www.modelscope.cn/models/Comfy-Org/z_image_turbo/tree/master/split_files/vae). See [exact model filenames and loader options](PiD_README.en.md).
- **UniverSR** needs both `config.yaml` and `pytorch_model.bin` for each general/speech model. The model node downloads only if `download=true`, using `HF_ENDPOINT` (default `https://hf-mirror.com`). See [cache layout and processing modes](UNIVERSR_README.en.md).
- Downloader workflows use yt-dlp. WeChat Channels also requires the local client and the bundled wasm/Node decryption tools. See [proxy, cookie and security requirements](VIDEO_DOWNLOAD_README.en.md).

## Example workflows

| Workflow | Before running |
| --- | --- |
| [PiD upscale and postprocess](workflows/Musefish_PiD_Batch_Video_Upscale.json) | Supply the UNET/CLIP/VAE and source video; start with small batches on a 16 GB GPU |
| [DLSS5 streaming video](workflows/Musefish_DLSS5_Video_Stream.json) | `LoadVideo → MusefishDLSS5VideoStream → SaveVideo`; select your source file first; defaults to 2× and CPU encoding |
| [UniverSR music and speech](workflows/Musefish_UniverSR_Audio.json) | A 16 GB reference with two separate branches; replace test audio and enable the intended branch; additionally requires rgthree and ComfyUI-Custom-Scripts |
| [Video downloader](workflows/Musefish_Video_Downloader.json) | Supply the share link/platform; for WeChat Channels, play the desired video in the client first |

## Limits and releases

- PiD/UniverSR 16 GB presets are references, not OOM guarantees. Long PiD batches still consume host RAM. For long DLSS5 video, use the file-backed streaming node rather than one large IMAGE batch.
- For the **same 1080p, 22-second video enlarged to 8K**, the user measured **79 seconds with GPU HEVC** and **300 seconds with CPU HEVC**. This is a single-source/hardware comparison, not a general throughput guarantee. [Quality and encoder details](DLSS5_README.en.md).
- **v1.5.0 (2026-09-28):** English `Upscale` labels, source-resolution option filtering, stable 8K enhancement, CPU HEVC support, and a streaming workflow template. The detailed guides retain implementation history and measurements. [GitHub v1.5.0 release](https://github.com/Brian98Z/ComfyUI-Musefish-Nodes/releases/tag/v1.5.0).
