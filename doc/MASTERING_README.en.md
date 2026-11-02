# Musefish Music Mastering (display name `Musefish Music Mastering`)

`musefish_music_mastering` masters AUDIO with the local PhaseLimiter engine. It shares `server/mastering.py` with Musefish Studio (imported at runtime from the Studio server tree; no duplicate implementation in the plugin).

## Node and parameters

| Item | Value |
|---|---|
| ID / category | `musefish_music_mastering` / Musefish/Audio |
| Inputs | `AUDIO` (batch), `mode`, `level`, `matching`, `loudness_target`, `ceiling`, `format`, `bit_depth`, `filename_prefix` |
| Outputs | `audio` (AUDIO, fixed 44.1 kHz stereo float), `path` (STRING, saved file path), `info` (STRING, measured loudness summary); output node saves under ComfyUI `output/` |
| Batch | Clips are processed separately then stacked; shorter clips are zero-padded to the longest. Other input rates are resampled; channels beyond the first two are dropped. |

| Parameter | Values | Meaning |
|---|---|---|
| `mode` | `mastering5` / `mastering3` / `mastering2` / `classic` | Four mastering algorithms; 5 has the best quality but is slowest, classic is fastest. |
| `level` | 0–1, step 0.25 | Loudness boost strength; only effective for `mastering5`. |
| `matching` | 0–1, step 0.25 | Loudness matching; measured output changes only under `classic` (−8.12 → −9.55 LUFS). |
| `loudness_target` | −9 / −14 / −16 / −23 LUFS, `off` | Target loudness; measured error <0.1 LU. |
| `ceiling` | 0 / −1 / −2 dBTP | True-peak ceiling, default −1 dBTP; when both cannot be met, true-peak ceiling takes priority. |
| `format` | `wav` / `mp3` / `aac` | Saved format; MP3 320 kb/s and AAC 256 kb/s are constant bitrate. MP3/AAC are transcoded from the engine WAV: the engine's built-in mp3/aac output bypasses the loudness reference (measured −6.3 LUFS vs target −14), so loudness is unified in the WAV pipeline. |
| `bit_depth` | `24` / `16` | WAV only. |
| `filename_prefix` | string | Saves under output; supports subdirectories and `%year%-%month%-%day%`. |

## Progress and implementation

`mastering.run(..., on_progress=cb)` reports stages through `cb(done, total, label)`. `total` = engine stages (9) + the non-WAV transcode step + loudness measurement; `progress_total(params)` returns the same denominator. Events are monotonic; first `(0, total, "启动引擎")`, last `(total, total, "测量响度")`; failures send no final event. Node progress sums batch × `progress_total`; Studio uses `job.progress = {pct, label, stage, elapsed}`.

The `mastering5/3/2` engine optimizer is stochastic: identical parameters can produce different output bytes, while LUFS / LRA / crest remain stable. Assess parameter effects with repeated samples, not a single A/B. The engine resolves `phaselimiter/resource` relative to cwd; runtime cwd is pinned to the install directory.

## Measured performance

Measured on 14 physical / 20 logical cores, 44.1 kHz stereo:

| Stage | Before → after | Condition / notes |
|---|---|---|
| Loudness measurement | 2.27 s → 0.38 s | 150 s file; `loudnorm=print_format=json` → ffmpeg `ebur128=peak=true`; both BS.1770-4, difference from old values ≤0.05 LU/dB across WAV/MP3. |
| Full pipeline | 11.41 s → 8.66 s | One 150 s file, mastering5→mp3; engine remains 8.4 s, savings are measurement and copying. |
| Single clip | 5.29 s → 3.78 s | One 60 s node clip. |
| Batch | 19.15 s → 10.66 s | Node batch of 4 × 60 s, mp3. Up to 4 clips concurrent (`_MAX_WORKERS = 4`); one engine process briefly uses ~16 cores, peak RSS 304 MB; measured 1.44× at 4 workers and 1.32× at 2, with more slowed by CPU contention. |

Concurrent progress sums completed stage cells under a lock and remains monotonic. WAV deliveries read the saved file directly into the tensor, saving a 60–100 MB copy; lossy formats retain the intermediate WAV so the tensor comes from a lossless waveform.

## Engine and dependencies

Engine directory resolution uses the first location containing `phaselimiter/bin/phase_limiter.exe`: `mastering_dir` in `server/config.json` → `MUSEFISH_MASTERING_DIR` → `<MusefishStudio>/engines/mastering` → `D:/AI Mastering` (legacy). An invalid explicit path fails without fallback. Missing `phaselimiter/resource/sound_quality2_cache` or `mastering_reference.json` reports missing resources.

Dependencies: `torch`, `torchaudio`, `soundfile`, `numpy` (ComfyUI environment); ffmpeg for transcoding/measurement, searched in `config.ffmpeg_dir` → `<root>/ffmpeg/bin` → `PATH`. The engine has no Python dependencies.

## Studio parity and troubleshooting

Studio (Audio → AI Mastering) and the node share parameter semantics, log stages and output naming. The workbench additionally provides a queue, progress, cancellation and cleanup. Parameter visibility: `matching` only in `classic`, `level` only in `mastering5`, and bit depth only for WAV.

| Symptom | Fix |
|---|---|
| `未找到 AI Mastering（PhaseLimiter）` | Engine directory incomplete; check `mastering_dir`. |
| `AI Mastering 资源缺失` | Check `sound_quality2_cache` and `mastering_reference.json` under `phaselimiter/resource/`. |
| `无法解码输入音频` | Convert input to WAV/MP3 first. |
| Loudness misses target | Check `loudness_target=off` or a ceiling (e.g. 0 dBTP) that prioritizes limiting. |
