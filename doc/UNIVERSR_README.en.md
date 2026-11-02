# UniverSR Audio Super-Resolution and Mastering

### Nodes and Features

- **Musefish UniverSR Model** (`musefish_universr_model`): prepares a general or speech model cache and can download models; its output is a directory reference, not processed audio.
- **Musefish UniverSR General Audio** (`musefish_universr_general_audio`): uses the general model; supports `sr`, `master`, `sr_master`, and `stem_mix`.
- **Musefish UniverSR Speech Audio** (`musefish_universr_speech_audio`): uses the speech model; supports only `sr` for voice/speech.
- Processing nodes accept standard `AUDIO` (`waveform` shaped `B,C,T`), batches, mono/stereo, and output 48 kHz. Connect model cache via `model_cache`.

### Environment, Models, and Workflow

The worker uses ComfyUI's `sys.executable` (Python 3.10–3.13; Torch/audio dependencies come from that environment). Model directories are `ComfyUI/models/UniverSR/models/huggingface/<general|speech>`; each needs both `config.yaml` and `pytorch_model.bin`. Network access occurs only with `download=true`; files are downloaded individually through `HF_ENDPOINT` (default `https://hf-mirror.com`) and atomically renamed. Plugin import does not access the network; `snapshot_download` is not used.

- general: [woongzip1/universr-audio](https://huggingface.co/woongzip1/universr-audio/tree/main); speech: [woongzip1/universr-speech](https://huggingface.co/woongzip1/universr-speech/tree/main).
- `model_cache` (or `MUSEFISH_UNIVERSR_MODEL_CACHE`) can point to a cache directory. The worker also recognizes `<cache>/universr-audio|universr-speech`, `<cache>/general|speech`, `<cache>/models--woongzip1--<name>`, `<cache>/huggingface/...`, and Hugging Face `snapshots/` within them.
- `stem_mix` requires both general and speech models; point `model_cache` to their parent, `ComfyUI/models/UniverSR/models/huggingface`. It also requires Demucs in the same Python environment.
- Workflow: [Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json), ID `6e417a01-39fa-461d-9f27-484c97aeef2e`, a 16 GB VRAM reference with independent music/speech branches. Dependencies: `Fast Groups Bypasser (rgthree)` from `rgthree-comfy` and `ShowText|pysssss` from `ComfyUI-Custom-Scripts`; switches can be removed and log left unconnected. Example names `测试歌曲.mp3` and `测试语音.wav` are not included. Output prefix: `audio/Musefish_UniverSR` in ComfyUI's output directory; template saves FLAC. Both model nodes default to `download=true` in the template.
Template preset: music/general uses `sr_master`, `input_sr=auto`, `channel_mode=auto`, `ode_method=midpoint`/4 steps, `guidance=1.5`, `chunk_sec=30` seconds, seed `randomize`, enabled by default. Speech uses the fixed `sr` mode with otherwise corresponding settings, `chunk_sec=20` seconds, and is bypassed by default. With `auto_params` enabled, measured source properties can override ODE steps and `guidance`; disable it to keep table values. 16 GB is a reference only; reduce `chunk_sec` first if VRAM is insufficient, e.g. 30→15→10.

### Modes and Default Parameters

| `mode` | Chain / use |
| --- | --- |
| `sr` | UniverSR super-resolution, M/S synthesis, bandwidth cleanup, general-model de-essing by default, true-peak finalization; speech offers only this mode and skips de-essing. |
| `master` | No model inference; V8 mastering: EQ, multiband compression, dry ER, high-frequency shaping, TP-feedback limiting, 19.2k cutoff; source loudness retained. |
| `sr_master` | Mid super-resolution/mastering (4 dB true-peak headroom, in-chain de-essing), Side band-limit and decorrelation above 14k, M/S synthesis, bandwidth cleanup, true-peak finalization. |
| `stem_mix` | Demucs vocal/accompaniment separation; speech/general process vocals/accompaniment respectively; restore vocal 8-12k/12-16k and accompaniment 12-16k, mix at source level, anchor source loudness, TP finalization. Higher resource/time use; 16 GB VRAM is not guaranteed. |

| Processing parameter | New-node default / range |
| --- | --- |
| `audio` | Required `AUDIO`; batches and mono/stereo; 48 kHz output |
| `auto_params` | `true`; first control. Automatically selects `input_sr`/`ode_steps`/`guidance`; these controls are hidden when enabled. |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000`; effective-content-bandwidth tier, not output sample rate; 16k corresponds to about 8 kHz bandwidth. |
| `ode_steps` | 4; 1–25 |
| `guidance` | 1.5; 0–5, step 0.1 |
| `mode` | `auto`; `auto`/`sr`/`master`/`sr_master`/`stem_mix`; general only, speech fixed to `sr`. |
| `channel_mode` | `auto` / `mono` / `stereo` |
| `ode_method` | `midpoint`; `euler` / `midpoint` / `rk4` |
| `chunk_sec` | 15; 1–120 seconds, about 50 ms overlap; lower it if VRAM is insufficient. |
| `seed` | 0; non-negative integer; template strategy `randomize`. |
| `deess` | `true`; only `sr` + general; reduces 5.5–8.5k in-band component, default max 8 dB, threshold = in-band envelope median +6 dB; `sr_master`/`stem_mix` use in-chain de-essing. |
| `accel` | `cuDNN TF32`; `fp32` / `cuDNN TF32` / `bf16` |
| `model_cache` | Optional STRING connection; cache directory, not an audio signal. |
| Outputs `audio` / `log` | `AUDIO` / `STRING`; log includes selected tier and chunk progress. |

Model preparation node: `model=general` (or `speech`); `download=false` (`true` downloads); output `model_cache` is STRING.

### Automatic Tier and Parameter Selection

`input_sr=auto` uses `effective = max(99% rolloff, content cutoff)`, where `content cutoff` is the frequency above rolloff where the spectrum first falls 20 dB below the rolloff level. Tiers: `effective ≤ 5.4 kHz → 8k` (probe 8k first at 5.0–5.4 kHz); `≤ 7.2 kHz → 16k`; `≤ 12 kHz → 16k`; otherwise 24k. The 12k tier is not selected. At `effective ≤ 5000 Hz`, the hard guard uses 8k without upward correction. The 4150–5850 Hz correction window once promoted 3400–5000 Hz material incorrectly to 12k/16k; 5062 Hz may still promote to 12k.

Only general applies upward correction: candidate energy ≥1e-5 of total spectrum (-50 dB), contiguous bandwidth about ≥300 Hz, at least 6 dB above noise floor and no more than 20 dB below band peak, energy in ≥25% of analysis frames, and evidence over 3 consecutive frames (2 for fewer than 12 frames); leave 150 Hz at band edges. Correction only goes upward and reuses the FFT. Silence, low noise floor, broadband hiss, isolated spikes/transients usually fail the thresholds. Speech does not correct upward; manual `input_sr` is unchanged. The maximum 24 kHz input tier does not guarantee original detail above 12 kHz.

`auto_params` is enabled by default; rules are from same-seed measurements on 4 sources × 4 configurations: `need = clip((-23 - d12) / 20, 0, 1)`, `d12 = B(12-16k) - B(4-6k)`. General selects `master` for `need=0`, otherwise `sr_master`; speech always uses `sr`. Speech `guidance`: `need < 0.5 → 1.0`, otherwise `1.5`; general fixed at `2.0` (music setting based on listening, not covered by this measurement). Speech `ode_steps`: `need < 0.5 → 8` (difference from 16 steps ≤0.25 dB, within noise), otherwise 16 (8 steps differed by 0.62 dB on restoration-needed material); general fixed at 16 (12-16k difference −14.1 → −4.6 dB). General `deess` is always on; speech enables it only when 6-8k exceeds 4-6k by ≥2 dB (natural sibilants measured −1.7…−2.5 dB). Silence/too-short inputs fall back to domain defaults. Any `input_sr`/`ode_steps`/`guidance`/`mode`/`deess` value changed from its control default is treated as manual and not overridden. Logs include `kept node values: ...` and `auto params → ...`. After updating, hard-refresh with `Ctrl+Shift+R` to prevent old control order from misaligning saved values and failing queue validation.

### Acceleration and Measurements

Measured on RTX 5070 Ti (one 9.5 s chunk, 16 ODE steps):

| Mode | Speed | Maximum sample difference vs. `fp32` |
| --- | --- | --- |
| `fp32` | Baseline, sample-by-sample identical | — |
| `cuDNN TF32` (default) | 1.18× | 3.9e-3 (about −48 dBFS) |
| `bf16` | 1.35× | 0.26 (about −11.7 dB, RMS −38.6 dB) |

`fp32` is slowest; without CUDA, non-`fp32` falls back automatically. This affects only this node, not ComfyUI's global GPU settings. Runtime examples: 60 s `sr_master` at automatic 24k takes about 7–8 minutes; 200 s `master` usually takes under a minute. If VRAM is insufficient, reduce `chunk_sec` first (e.g. 30→15→10); chunking does not eliminate full input/output memory use. Restart ComfyUI after code updates, avoid old cached results, and use a fixed seed for comparisons.

### Notes and Files

- `master` does not run super-resolution; tier-selection logs do not indicate reconstruction. `stem_mix` passes the selected tier to separated tracks and does not redetect each track. Automatic correction does not guarantee improvement; MP3 does not imply low bandwidth, and 24k input may still filter content above about 12 kHz.
- For dull output, check actual `input_sr` and compare `sr`/`sr_master` on a fixed-seed excerpt; de-essing applies only to `sr` + general. For noise/soundstage changes, compare with the original; stereo `sr`/`sr_master` uses decorrelated noise above 14k for width.
- For missing models/nodes, check both weights, workflow dependencies, and `SaveAudioAdvanced`; `stem_mix` cache must contain both models. Example logs: `auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`; guard: `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`. Output remains 48kHz.
- `musefish_audio.py`: nodes, AUDIO adaptation, worker scheduling; `audio_backend/processing.py`: tiering, chunking, mode chains; `audio_backend/dsp.py`: mastering/audio post-processing; `audio_backend/worker.py`: process entry point; [Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json): two-branch template.
