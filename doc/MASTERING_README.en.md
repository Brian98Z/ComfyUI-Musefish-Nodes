# Musefish Music Mastering (display name `Musefish Music Mastering`)

`musefish_music_mastering` — masters an AUDIO batch with the local PhaseLimiter engine
(loudness unification + true-peak limiting + multi-band dynamics). The engine
driver is the **same module** the Musefish Studio workbench uses
(`server/mastering.py`, imported at runtime from the Studio server tree — the
plugin ships no second copy), so node and workbench results are identical.

## Node

| Item | Value |
|---|---|
| ID | `musefish_music_mastering` |
| Category | Musefish/Audio |
| Inputs | `AUDIO` (batch), `mode`, `level`, `matching`, `loudness_target`, `ceiling`, `format`, `bit_depth`, `filename_prefix` |
| Outputs | `audio` (AUDIO, 44.1 kHz stereo), `path` (STRING, saved file), `info` (STRING, measured loudness) |
| Output node | Yes — saves under ComfyUI `output/` and runs standalone |

Each clip is mastered separately, then the batch is stacked (shorter clips are
zero-padded to the longest).

## Parameters

| Parameter | Values | Notes |
|---|---|---|
| `mode` | `mastering5` / `mastering3` / `mastering2` / `classic` | Algorithm. 5 = best quality, slowest; classic = fastest |
| `level` | 0 – 1 (step 0.25) | Loudness boost strength, **defined only for `mastering5`**; other modes ignore it |
| `matching` | 0 – 1 (step 0.25) | Loudness matching. **Measured: only `classic` changes the output** (−8.12 → −9.55 LUFS) |
| `loudness_target` | −9 / −14 / −16 / −23 LUFS, `off` | Delivery loudness; measured within 0.1 LU |
| `ceiling` | 0 / −1 / −2 dBTP | True-peak ceiling; the limiter sacrifices loudness rather than exceed it |
| `format` | `wav` / `mp3` / `aac` | Saved format; the AUDIO output is always 44.1 kHz stereo float |
| `bit_depth` | `24` / `16` | WAV only (MP3 320 kb/s, AAC 256 kb/s are constant bitrate) |
| `filename_prefix` | string | Output folder prefix; subdirectories and `%year%-%month%-%day%` variables supported |

## Progress callback

`mastering.run(..., on_progress=cb)` reports `cb(done, total, label)` per stage:

- `total` = engine stages (9) + the transcode step for non-WAV + loudness
  measurement; `progress_total(params)` returns the same denominator so callers
  can convert to a percentage or a progress-bar scale.
- Monotonic: the engine re-runs `Normalize` / `CutLowAndHighFreq`, and the
  callback only fires when the step actually advances.
- First event is always `(0, total, "启动引擎")`, the last is
  `(total, total, "测量响度")`; a failure sends no final event.

Consumers:

| Caller | Usage |
|---|---|
| Studio job | `job.progress = {pct, label, stage, elapsed}` → card reads「母带处理 50% · 母带优化 · 12s」 |
| ComfyUI node | progress scale = batch × `progress_total`, advanced per stage; `send_progress_text` prints「[1/2] 母带优化」on the node, throttled to one push per 0.5 s at the same percentage |

## Performance (measured on 14 physical / 20 logical cores, 44.1 kHz stereo)

| Stage | Before | After | Notes |
|---|---|---|---|
| Loudness measurement (150 s file) | 2.27 s | 0.38 s | `loudnorm=print_format=json` → `ebur128=peak=true`; same BS.1770-4 result, ≤0.05 LU/dB apart across WAV/MP3 |
| Full pipeline, one 150 s file (mastering5→mp3) | 11.41 s | 8.66 s | engine's 8.4 s unchanged; the saving is measurement + copying |
| Node, one 60 s clip | 5.29 s | 3.78 s | same two fixes |
| Node, 4×60 s batch (mp3) | 19.15 s | 10.66 s | batch parallelism on top |

- **Batch parallelism**: `_MAX_WORKERS = 4`. A single engine process bursts to
  ~16 cores with a 304 MB RSS peak; 4 workers measured 1.44×, 2 workers 1.32× —
  beyond that CPU contention wins. Lower it on memory-tight machines.
- **Progress stays monotonic**: with parallel clips the reported value is the sum
  of advanced stage cells under a lock, so the bar never jumps backwards.
- **WAV deliveries skip the intermediate copy**: the saved file is read directly
  into the output tensor (saves a 60–100 MB copy); lossy deliveries keep the WAV
  because the tensor must come from a lossless waveform.

## Engine behaviour (re-measure before changing)

- **Fixed 44.1 kHz stereo**: other input rates are resampled by the node;
  channels beyond the first two are dropped.
- **MP3/AAC are transcoded from the engine WAV**: the engine's own mp3/aac
  output bypasses the loudness reference (measured −6.3 LUFS instead of the
  requested −14), so loudness unification always happens on the WAV pipeline.
- **The optimizer is stochastic**: `mastering5/3/2` produce different bytes on
  repeated identical runs while macro metrics (LUFS / LRA / crest) stay stable —
  compare parameter effects across repeated runs, never a single A/B.
- **The engine resolves `phaselimiter/resource` relative to cwd**: the driver
  pins cwd to the install directory.
- **Cancel kills the process tree**: the engine spawns its own ffmpeg child.

## Engine discovery

First existing hit wins (must contain `phaselimiter/bin/phase_limiter.exe`):

1. `server/config.json` → `mastering_dir`
2. env `MUSEFISH_MASTERING_DIR`
3. `<MusefishStudio>/engines/mastering`
4. `D:/AI Mastering` (legacy install)

An explicitly configured but invalid path fails loudly instead of silently
falling back to a stale install.

## Dependencies

`torch`, `torchaudio`, `soundfile`, `numpy` (already in ComfyUI's environment);
ffmpeg for transcoding and measurement (`config.ffmpeg_dir` →
`<root>/ffmpeg/bin` → `PATH`). The engine itself needs no Python packages.

## Studio parity

The Studio workbench (Audio → AI Mastering) calls the same `mastering.py`:
identical parameter semantics, log stages and output naming. The workbench adds
a job queue, progress, cancel and one-click output cleanup; the node mirrors the
workbench's parameter visibility rules (`classic` = matching, `mastering5` =
strength, WAV = bit depth).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `未找到 AI Mastering（PhaseLimiter）` | Engine directory incomplete — set `mastering_dir` per the list above |
| `AI Mastering 资源缺失` | `phaselimiter/resource/` is missing `sound_quality2_cache` or `mastering_reference.json` |
| `无法解码输入音频` | Input is not decodable audio; convert to WAV/MP3 first |
| Delivered loudness misses the target | `loudness_target=off`, or an over-tight ceiling (e.g. 0 dBTP) limits first |
