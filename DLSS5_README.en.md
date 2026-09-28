# Musefish DLSS5 Upscaling Nodes — User Guide

> **Hardware requirement: NVIDIA RTX 50 Series (Blackwell) only.** DLSS 5 upscaling (NGX Feature 18) and the accompanying RTX Video Super Resolution runtime are provided only with Blackwell drivers. RTX 40 Series and earlier GPUs cannot create this feature, and the node will fail directly.
>
> Verified environment: RTX 5070 Ti + driver 616.56.

## What this is

This wraps DLSS5Tool's NGX Feature 18 as two ComfyUI nodes: `MusefishDLSS5NeuralRender` processes single images and short-video IMAGE frame batches (fixed to one worker), while `MusefishDLSS5VideoStream` accepts VIDEO, processes it frame by frame, and returns VIDEO for high-resolution and long-form video. Both are in `Musefish/Video`.

## Files

- `musefish_dlss5.py` — node implementation (schema + execute)
- `musefish_dlss5_stream.py` — single-session decoding, enhancement, encoding, and audio muxing for long videos
- `dlss5_backend/session.py` — session management (child-process lifecycle and shared-memory frame protocol)
- `dlss5_backend/worker.py` — child-process entry point (ctypes DLL driver + shared-memory pinned registration)
- `models/dlss5/` — ComfyUI model directory (can be set with `--models-directory`) containing four external DLLs; they are not distributed with the plugin
- `workflows/Musefish_DLSS5_Video_Stream.json` — example LoadVideo → streaming DLSS5 → SaveVideo workflow

## Runtime dependencies and manual setup (Windows)

### Required NVIDIA/DLSS5Tool runtime libraries

The IMAGE and VIDEO nodes call the same isolated worker. 1× mode requires `dlssnr_host.dll` and `nvngx_dlssnr.dll`; VSR modes above 1× also require `vsr_host.dll` and `nvngx_vsr.dll`. Put the files in the `dlss5/` subdirectory of ComfyUI's active models directory: by default, `ComfyUI/models/dlss5/`; when using `--models-directory`, use `dlss5/` under the directory specified by that option, not the plugin's `dlss5_backend/dlls/`.

| Filename | Purpose |
|---|---|
| `dlssnr_host.dll` | Feature 18 neural-rendering host; the current code uses the legacy host |
| `nvngx_dlssnr.dll` | Feature 18 NGX runtime, loaded by the host during initialization |
| `vsr_host.dll` | RTX Video Super Resolution host; used only for upscaling above 1× |
| `nvngx_vsr.dll` | VSR NGX runtime |

User-provided four-file share: <https://pan.quark.cn/s/d4c04dc33d25>. After downloading, check that the filenames match the table, place them in the `models/dlss5/` directory above, and restart ComfyUI. This link is a user-provided cloud-drive share, **not an official NVIDIA or DLSS5Tool distribution URL**; verify that you have permission to obtain and use the files. Do not download DLLs from unknown mirror sites. If you already have a legitimately obtained DLSS5Tool package, you can also manually extract the same files from its `_internal/` directory. No official standalone download URL for these four private runtime libraries has been verified. The public NVIDIA [DLSS SDK](https://github.com/NVIDIA/DLSS) is not a substitute for the host and runtimes used here. This node does not read `models/dlss5/nvngx_dlss.dll` or `models/dlss5/caller/`: those may be used by another RH-DLSS5 plugin and are not part of Musefish's four-file list.

An NVIDIA graphics driver must be installed and provide NGX/D3D12 support. As required by the project's current implementation, CUDA devices must be visible to the Python environment used by ComfyUI. Select a driver for your GPU and operating system from the [official NVIDIA driver download page](https://www.nvidia.com/Download/index.aspx). The node does not require a separate CUDA Toolkit installation to run NGX. The worker attempts to locate `cudart64_*.dll` from the current ComfyUI PyTorch installation; it is used only to accelerate pinned registration of shared memory. If the CUDA runtime cannot be found or registration fails, processing still works using ordinary pageable memory. Do not replace ComfyUI's bundled CUDA-enabled PyTorch.

### Python and video tools

- The IMAGE node uses the PyTorch/CUDA supplied by the active ComfyUI environment and depends on `numpy` and `cv2`; the VIDEO node reuses the same image-processing dependencies. `requirements.txt` includes NumPy but not OpenCV. If `cv2` is missing, install `opencv-python` in the Python environment used by ComfyUI. Avoid installing both GUI and headless OpenCV packages at the same time. Use the PyTorch/CUDA installation already provided by ComfyUI; do not overwrite it using instructions for a standalone installation.
- The VIDEO streaming node additionally requires both `ffmpeg` and `ffprobe` executables to be discoverable through `PATH`. They are used to read video and inspect media, and to produce an intermediate MP4 with audio. The IMAGE node does not invoke external FFmpeg. Start from the [FFmpeg download page](https://ffmpeg.org/download.html) to find Windows builds. After downloading and extracting a build, add the `bin` directory containing `ffmpeg.exe` and `ffprobe.exe` to the `PATH` of the process that launches ComfyUI, then restart ComfyUI. The FFmpeg site links to third-party Windows build providers; comply with the license and distribution terms of the build you choose.

Dependency summary: both nodes require NVIDIA RTX/NGX driver capability, the Feature 18 host and runtime DLLs, ComfyUI CUDA/PyTorch, NumPy, and OpenCV. Modes above 1× additionally require the two VSR DLLs. Only Video Stream additionally requires FFmpeg/ffprobe on `PATH`. CUDA-runtime-based pinned-memory acceleration is optional, not an additional installation prerequisite.

## Architecture

The NGX host runs in an **isolated child process**. The ComfyUI main process never loads NVIDIA DLLs, so a driver crash will not bring down the service. Frames are transferred through a shared-memory ring: one copy in each direction between parent and worker, with zero-copy processing on the worker side.

```
ComfyUI main process                       worker child process (one per worker)
  OpenCV converts each frame to RGBA8 → slot 0/1 → dlssnr_process / vsr_process
  compact RGBA8, convert to float32 ← slot 0/1 ← (NGX engine runs on the GPU)
```

- **Two-deep double-buffered pipeline:** while the parent fills slot n+1 and retrieves the result from slot n-1, the worker computes frame n. The parent's pixel conversions fit entirely into the engine's processing window, so the GPU no longer idles between frames. The protocol preserves “one request, one ordered reply”; temporal history and output bytes match a serial session (bit-for-bit, regression-tested).
- **The short-video IMAGE node is fixed to one worker**, preserving continuous temporal history within a session. Existing parallel tests on 480p video: 1 worker, 31s/17% VRAM; 2, 23s/21%; 4, 39s/31%; 8, 40s/49%; 16, 41s/85%; 32, 176s/99%. These measurements show that increased VRAM use does not necessarily improve throughput. At 720p, long-video runs took 159–168s with 1–8 workers; 16 workers took 233s/98% VRAM, and 32 workers ran out of memory (OOM). Use the new continuous-stream node for long videos rather than loading every frame into an IMAGE batch.
- **`reset_every_n_frames`:** 0 = reset only on the first frame of each chunk (continuous single session); 1 = independent processing of every frame (same as the old hard-coded behavior); for N>0, reset in place inside the engine according to the frame index, without recreating the session.

One execution = one session pool = a number of complete temporal histories (corresponding to DLSS5Tool's “strict temporal (single session)” export mode).

Parameters correspond one-to-one with DLSS5Tool:

| Node parameter | DLSS5Tool equivalent |
|---|---|
| `style: default/natural/cinema` | Default/Natural/Cinema |
| `intensity` (0–1) | Intensity |
| `local_tone` (0–1) | Local tone |
| `local_struct` (0–1) | Local structure |
| `skin_struct` (0–1) | Skin-mask strength |
| `use_auto_mask` | Skin-mask toggle |
| `reset_every_n_frames` (0=never) | Temporal-history reset (0 = continuous single session; 1 = independent per frame, same as the old behavior) |
| `Upscale` (IMAGE/VIDEO; input ID `super_resolution`) | First control: `off`, `1× (Native)`, `1.5× (Quality)`, `2× (Balance)`, `3× (Performance)`, `4× (Ultra)`, `1K`, `2K`, `4K`, `8K`. For VIDEO connected to Load Video (optionally through Video Slice), the source resolution is read with ffprobe. For IMAGE connected directly to LoadImage, the image resolution is read. When the source size is known, both hide multipliers that exceed output/intermediate-image limits. For other IMAGE-generation or processing sources, the size cannot be known before execution, so the full list is shown and validated at runtime. 1× does not upscale; 2×/4× use native VSR; 1.5×/3× are synthesized by downscaling 2×/4× VSR. K modes select the smallest achievable multiplier for a target short side of 1080/1440/2160/4320; output and VSR intermediate images are both limited to a 7680×4320 pixel budget. 720p→4K uses a 5120×2880 VSR intermediate image, then scales it down to 3840×2160. For 1080p→8K, both VIDEO and IMAGE first perform neural enhancement on an image no larger than 4K, then use interpolation to scale to 8K, avoiding color shifts/noise from direct 8K Feature 18 processing. IMAGE produces a float32 output batch; batches exceeding 1 GiB are rejected with a recommendation to use VIDEO. |
| `vsr_quality: performance/balanced/quality/ultra` | VSR quality preset (NGX PerfQuality 1–4; ultra is DLSS5Tool's default) |
| `keep_session: auto/off` | `auto` reuses a warm worker across batches of the same task; it closes and releases VRAM about 2–3 seconds after the task ends or is interrupted. `off` closes it immediately after each node execution. |
| `parallel_workers` | Removed from IMAGE and fixed to one worker; use the streaming node for long videos |

Parameters in the experimental “5× range” outside [0,1] are clamped (the GUI's experimental range is off by default).

## Performance (2026-09-23 baseline, i5-14600K + RTX 5070 Ti)

Real-machine workload: an 801-frame 480×848 video → 2× VSR + Feature 18 ultra (workflow `LoadVideo → DLSS5 → VideoCombine`).

| Metric | Before optimization | After optimization |
|---|---:|---:|
| Node time | 29.76s | **16.75s** (1.78×) |
| GPU utilization during node | 22–26% | **25–49%** |
| Main-process CPU during node | ~12.5–13.4 cores | **~1.5–2.2 cores** |
| Per-frame conversion CPU | 330–380 CPU-ms/frame | **~30 CPU-ms/frame** |
| Full pipeline (including VHS decode + encode) | 70.81s | **53.31s** |

Three changes:

1. **Per-frame pixel conversion now uses the original NumPy C loops** (it previously used torch operations: eight small kernels per frame entered the intra-op thread pool, and parallelization overhead overwhelmed a 1–2 MP workload. Measured CPU/wall was approximately 10–13, i.e. more than ten cores spent moving 20 MB).
2. **Two-deep double-buffered pipeline:** parent-side conversion overlaps engine GPU computation; the GPU no longer idles between frames (the engine's approximately 20–30 ms per-frame submission/synchronization delay used to be entirely idle time).
3. **Parallel processing in continuous chunks (`parallel_workers`):** the low-resolution benchmark at the time used two workers, improving throughput by a further 1.2–1.7× (GPU utilization 24% → 40–50%); the default for high-resolution workloads was changed to one worker.

Incremental optimization on 2026-09-24: RGB float32→RGBA8 input conversion now uses OpenCV saturation conversion and channel packing; RGBA8 output is first compacted to RGB8 and then converted to float32, avoiding slow per-pixel division over a strided RGBA view in NumPy. In an interleaved A/B test of 120 frames at 480×848→960×1696 with two workers and warm sessions, the old path ran at 61.5–65.9 fps at low utilization, while the new path ran at 75.2–94.1 fps. In another run with the GPU already saturated (about 99%), both old and new paths were around 43–46 fps; conversion was not the bottleneck. Batch output was bit-for-bit identical; a standalone microbenchmark reduced output conversion from 16.7 to 2.8 CPU-ms/frame. GPU utilization depends on other workloads and clock speeds; utilization figures do not indicate improved image quality or increased hardware compute capability. The `super_resolution=off` multiplier parsing bug was also fixed (it previously attempted `float("off")` and raised `ValueError`); real-machine smoke tests completed for off, 1.5×, and 2× with correct output sizes and ranges.

> ⚠️ **Important behavior change:** the old `session.process()` hard-coded `reset: True` on every frame. The `reset` calculated by the node was never used, so processing was independent per frame. The default `reset_every_n_frames=0` now **preserves** temporal history within a chunk, so output differs from the old version. On the same source, approximately 7% of pixels changed by >8/255; mean absolute difference was about 3.4/255 (denoising/detail differences caused by temporal accumulation). For bit-for-bit consistency with the old version, set `reset_every_n_frames` to **1** (regression-tested; workers=1/2 both match the old production path bit-for-bit).

## Known limitations

1. **Observed behavior with the v2 host:** on this machine (RTX 5070 Ti + driver 616.56), the first call to `dlssnr_process` in `dlssnr_host_v2.dll` waits internally and never returns (suspected conflict with the NGX component of the NVIDIA App overlay). The **legacy host, `dlssnr_host.dll`, works normally** (about 4 ms/frame at 256 px), so the worker is fixed to legacy. To try v2 after a future driver upgrade, change the DLL name in `worker.py`.
2. Only the SDR RGBA8 path is supported; HDR high-precision paths (RGBA16F/PQ/HLG) are not wrapped.
3. The streaming VIDEO node limits individual output frames and VSR intermediate images to 7680×4320 pixels. The IMAGE node uses the same dimensions limit but also limits the entire float32 output batch to 1 GiB.
4. `keep_session=auto` caches only session pools with an exactly matching contract; the old pool is closed when parameters change. When the task queue no longer contains DLSS5 work, a background check closes the worker after it has been idle for 2 seconds. The warm pool stays alive across VHS Meta Batch continuation, avoiding a cold start for every batch.
5. Frames are transferred through shared memory. On the worker side, the two ring buffers are registered as pinned memory with `cudaHostRegister`, allowing the NGX host's D3D12 per-frame DMA uploads/downloads to skip pageable staging copies (1080p off-mode steady state measured 23.0ms → 21.1ms/frame, about +8%; if the CUDA runtime cannot be found, it automatically falls back to ordinary mode with no functional impact). When the GPU is dedicated to this work, measured performance is about 21ms/frame at 1080p off and about 70ms/frame for 2× upscaling to 4K. Performance slows in proportion to GPU contention from other tasks. Attention/PyTorch acceleration does not apply to this engine (NGX is closed-source D3D12 inference).
6. The node processes frames as a stream internally (per-frame RGBA conversion with OpenCV/NumPy, written directly to the output tensor, with no full-batch intermediate copy); its own peak memory use is approximately the upstream input batch plus the output batch. Upstream `VHS_LoadVideo` loads the entire video into memory if `frame_load_cap` is not set. For long videos, use `frame_load_cap`/`skip_first_frames` to divide the input into chunks.
7. The IMAGE node is fixed to one worker; use `MusefishDLSS5VideoStream` for long videos. Parallel-performance figures from before the change are historical comparisons only; remove the old control when resaving old workflows.

## High-resolution long videos: single-run streaming node

Connect on the canvas: `LoadVideo (VIDEO) → MusefishDLSS5VideoStream (VIDEO) → SaveVideo`. The middle node accepts file-backed VIDEO (LoadVideo returns a reference to a file), processes frames one at a time, writes an intermediate MP4 under ComfyUI's temp directory, and passes the file-backed VIDEO to SaveVideo for output. A run keeps only one NGX session, two RGBA8 shared-memory slots, and a single-frame decode/encode buffer; it does not create a 900-frame IMAGE batch or require manual stitching. Temporal history remains continuous from the first to the last frame. The intermediate filename is fixed as `Musefish/DLSS5_stream_<random-value>.mp4`; there is no prefix control. Set the final video name only in SaveVideo's `filename_prefix`.

The original audio is converted to AAC. If both width and height are at most 4096, H.264 is used; CPU mode defaults to CRF 19 (adjustable). On failure or interruption, an incomplete `.part.mp4` is removed and the child process and worker are closed.

`encoder` defaults to `libx264 (CPU encoding)`. If output width or height exceeds 4096, it automatically switches to CPU `libx265` (HEVC); `crf` controls both CPU encoders. The optional `h264_nvenc (GPU encoding acceleration)` uses fixed CQ 27 (this is not the same image quality as CRF 19). In a local comparison on 1280×720@30fps input, 2× upscaling, continuous 60 seconds (1800 frames), total pipeline time was 74.1s → 65.1s and intermediate MP4 size was 95.5MB → 99.1MB.

On this machine, H.264 NVENC fails when the width exceeds 4096. In GPU mode, output width/height above 4096 automatically switches to `hevc_nvenc` (HEVC, MP4 `hvc1`, CQ 27). CPU mode can also produce 8K, automatically switching to HEVC `libx265` (MP4 `hvc1`, quality controlled by `crf`), but speed depends on the processor. For the same 1080p, 22-second video upscaled to 8K, the user measured 79 seconds with GPU HEVC and 300 seconds with CPU HEVC (about 3.8× as long). This is a user-provided end-to-end test result, not the 720p/2× benchmark above; speed cannot be inferred for other footage or hardware.

Verified outputs include 720p→4K at 3840×2160, landscape 1080p→8K at 7680×4320, and portrait 1080p→8K at 4320×7680; all include an AAC audio track. Direct 8K neural rendering produced color shifts and background noise in real footage. The VIDEO streaming node now performs neural enhancement on an intermediate image no larger than 4K, then uses interpolation to scale to 8K. An 8K file resolution does not mean native 8K neural-rendered detail. The IMAGE node's 8K mode likewise uses a neural-enhancement intermediate image no larger than 4K, then interpolates to the final size; it has no video encoder and is unaffected by CPU/GPU encoding choices. 8K tensors consume substantial system memory, and IMAGE batches whose output exceeds 1 GiB are still rejected.

Image quality and speed on other sources should be compared for each case. NGX uses single-session temporal processing; increasing concurrency merely because VRAM is available cannot be assumed to preserve the same temporal result.

The nominal 30fps of the local test video differs slightly from its average frame rate; streaming uses the nominal frame rate and returns VIDEO only after verifying output dimensions, frame rate, and frame count.

A `Video Slice` can be connected: its crop window is read from VIDEO, the same start time and duration are applied to decoding and original audio, and the decoded frame count is limited to prevent an extra boundary frame. The old version read only the underlying file path and ignored the `Video Slice` time window; a 60-second slice would actually process all 566.5 seconds of the original clip.

Progress is sent through ComfyUI's native progress events and calculated from the number of frames written to the encoder. It displays at most 99% until muxing and validation finish, and reaches 100% only after successful output is written. The frontend progress is this node's frame-processing progress; it does not include downstream SaveVideo.

`workflows/Musefish_DLSS5_Video_Stream.json` is the recommended current template and defaults to 2×/CPU encoding. Replace the example file in LoadVideo with your own input. For 8K, choose CPU or GPU encoding (HEVC is selected automatically).

## Intended use and example

The IMAGE node can be connected to LoadImage or short-video frame batches; connect long videos to the three VIDEO nodes described above. See the example result at [`assets/DLSS5超分案例.mp4`](assets/DLSS5超分案例.mp4) (1472×1280@24fps, 10 seconds; 2× RTX VSR + Feature 18).

## Verification record (2026-09-23 optimization regression)

- **Protocol equivalence:** with the same semantics (`reset_every_n_frames=1`), output from the two-deep pipeline and continuous chunk processing with 2/3 workers was **bit-for-bit identical** to the original serial path (real-machine clips of 72/20 frames).
- **Fallback equivalence:** `reset_every_n_frames=1` matched the old production path (`session.process()` hard-coded `reset=1`) bit-for-bit; workers=1/2 both passed.
- **Per-frame conversion equivalence:** NumPy and the original torch path produced identical RGBA8 input bytes for all 60/60 frames; uint8 boundary cases (0.5/255, 2.5/255, negative values, >1) matched byte-for-byte; float32 output was bit-for-bit identical.
- **End to end:** the 8988 instance completed `LoadVideo → DLSS5(workers=2) → VideoCombine` successfully (`output/Musefish/DLSS5_video_00015.mp4`): node 16.75s (previously 29.76s), full pipeline 53.31s (previously 70.81s), GPU at 25–49% during the node, and main-process CPU at ~1.5–2.2 cores (previously ~13 cores).
- **Cache fix:** previously, `keep_session=auto` wrote to the cache only when a cache already existed; a cold start was never cached (effectively, it never cached). Now a second execution with the same contract takes 5.52s → 2.95s (2 workers), with identical output.
- **Parallel scan:** on 400 frames of real footage, K=1/2/3 throughput was 18.2/22.4/23.8 fps, with GPU utilization of 20%/36%/42%; K=4 fell to 12.0 fps due to VRAM/scheduling overhead, so the limit is set to 3.

## Verification record (2026-09-20)

- An API run on the 8988 instance completed the full LoadImage → MusefishDLSS5NeuralRender → SaveImage path successfully, producing `output/dlss5_test_00001_.png`.
- Pixel-by-pixel comparison with the original: 54.4% of pixels changed by >8/255; mean absolute difference 8.18. The visible detail enhancement matched expectations for Feature 18.
- The complete 2× RTX VSR + Feature 18 path succeeded, producing 1728×2304 (`output/dlss5_sr2x_00001_.png`); all four VSR quality presets passed real-machine tests.
- Retest after replacing PNG with shared-memory IPC: off/2x both succeeded on the 8988 instance (`shm_off_00001_.png` / `shm_2x_00001_.png`) with correct dimensions (864×1152 / 1728×2304). With the GPU dedicated, off at 1080p was about 18.5ms/frame; 2× to 4K was about 70ms/frame.
- 1.5x synthesized preset + session cache: 864×1152 input successfully produced 1296×1728 at 1.5×. Two consecutive executions took 2.3s → 1.0s (cache hit avoids NGX cold start); offline benchmark warm-session first frame 34ms vs cold start 1271ms (37×).
- Regression after enabling pinned shared memory: 39 passed / 4 subtests (pytest, embedded Python, `--import-mode=importlib`); three open/close-session cycles showed no leaks; real user workload (1080p→2x, `keep_session=auto`) repeatedly produced successful outputs with the new code.
