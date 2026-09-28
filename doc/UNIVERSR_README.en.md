# UniverSR Audio Super-Resolution and Mastering

### Overview

- **`Musefish UniverSR Model`** prepares the general or speech model cache and can download models on demand. Its output is a cache-directory reference; it does not process audio.
- **`Musefish UniverSR General Audio`** (node ID: `MusefishUniverSRGeneralAudio`) always uses the general model and supports `sr` (super-resolution only), `master` (V8 mastering), `sr_master` (super-resolution + mastering), and `stem_mix` (Demucs stem separation + mixing).
- **`Musefish UniverSR Speech Audio`** (node ID: `MusefishUniverSRSpeechAudio`) always uses the speech model and supports only `sr` super-resolution. It does not expose model selection and is intended for voice/speech enhancement.
- Both processing nodes accept standard `AUDIO` (`waveform` shaped `B,C,T`), support mono and stereo batches, and take the cache output of `Musefish UniverSR Model` through `model_cache`.
- The `input_sr=auto` tier criterion is `effective = max(99% rolloff, content cutoff)`. `content cutoff` is the frequency above the rolloff where the spectrum first falls 20 dB below the rolloff level. MP3 rolloff can be as low as about 9.6 kHz while content extends to about 15 kHz; using rolloff alone can select the 24k tier incorrectly and make the model generate 8–12 kHz from scratch. Tier table: `effective ≤ 5.4 kHz → 8k` (the 5.0–5.4 kHz boundary band probes 8k first), `≤ 7.2 kHz → 16k`, `≤ 12 kHz → 16k` (when content reaches 12 kHz, the 16k condition focuses on cleanup/enhancement rather than generating content from nothing), and otherwise `→ 24k`. The 12k tier is no longer selected as an output tier.
- **Hard 8k guard:** for `effective ≤ 5000 Hz`, use the 8k tier directly with no upward correction. The correction window (4150–5850 Hz) can mistake residual energy in the filter transition band for persistent high-frequency evidence; tests incorrectly promoted every 3400–5000 Hz effective-bandwidth sample to 12k/16k. The 5.0–5.4 kHz boundary band still goes through correction, so a low-bandwidth sample with `effective = 5062 Hz` can be promoted to 12k.
- Only the general model applies upward correction. A candidate band must meet all of these conditions: energy is at least `1e-5` of total-spectrum energy (-50 dB); contiguous bandwidth is about 300 Hz or wider; it is at least 6 dB above the noise floor and no more than 20 dB below the candidate-band peak; energy is present in at least 25% of analysis frames; and there is evidence in at least 3 consecutive analysis frames (2 frames for inputs with fewer than 12 frames). A 150 Hz margin is left at band edges. Correction only moves upward, never downward. It reuses the same FFT and does not promote every file merely because its container is nominally 48 kHz. Example log: `auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`. When the guard blocks correction, the log looks like `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`.
- Silence, a low noise floor, broadband hiss, isolated spikes, or transients usually cannot satisfy the contiguous-band and time-domain thresholds. The speech model uses the same `effective` criterion but does not apply upward correction. Manually specified `input_sr` is neither detected nor rewritten.
- Restart ComfyUI after updating so the main process loads the tier-selection logic. 24 kHz is the highest input tier supported by the model and does not guarantee preservation of original detail above 12 kHz. This correction does not change the mastering chain and does not guarantee better sound for every music file.

### Runtime Environment and Model Paths

The worker uses the `sys.executable` of the ComfyUI process and supports ComfyUI environments using Python 3.10–3.13; Torch and audio dependencies come from that environment. The `Musefish UniverSR Model` node targets this cache location:

`ComfyUI/models/UniverSR/models/huggingface/<general|speech>`

Each model directory must contain both `config.yaml` and `pytorch_model.bin`. The node accesses the network only when `download=true`; it downloads files individually through `HF_ENDPOINT` (default `https://hf-mirror.com`) and atomically renames them after completion. Importing the plugin does not access the network, and `snapshot_download` is not used.

The processing node's `model_cache` input (or `MUSEFISH_UNIVERSR_MODEL_CACHE`) determines where the worker searches for weights, in this order: the directory itself → `<cache>/universr-audio|universr-speech` → `<cache>/general|speech` → `<cache>/models--woongzip1--<name>` → `<cache>/huggingface/...`, including Hugging Face `snapshots/` under those directories. In single-model modes (`sr` / `sr_master`), connect the model node output directly. **`stem_mix` must resolve both models**, so `model_cache` must point to a parent directory containing both `general` and `speech` (that is, `ComfyUI/models/UniverSR/models/huggingface`). Connecting only one model node's output will leave the other model unavailable. `stem_mix` uses the Demucs module in the same Python environment.

### Audio Example Workflow

Example file: [Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json), workflow ID: `6e417a01-39fa-461d-9f27-484c97aeef2e`. This is a **16 GB VRAM reference template** with two independent branches for music and speech; it does not send the same audio through both models in sequence.

```text
Music branch (enabled by default)
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRGeneralAudio
MusefishUniverSRModel (general) ─ model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced (FLAC)
                                             └─ log   → ShowText|pysssss

Speech branch (bypassed by default)
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRSpeechAudio
MusefishUniverSRModel (speech) ─ model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced (FLAC)
                                             └─ log   → ShowText|pysssss
```

**How to use:**

1. Drag the JSON into ComfyUI or load it from the workflow-template entry point.
2. The template additionally uses `Fast Groups Bypasser (rgthree)` from `rgthree-comfy` and `ShowText|pysssss` from `ComfyUI-Custom-Scripts`. Install the corresponding node packs if these nodes are missing. Alternatively, remove the group switches and bypass branches manually; leaving the log output unconnected does not affect audio processing.
3. Upload and select your own media in the appropriate `LoadAudio` node. `测试歌曲.mp3` and `测试语音.wav` are example filenames in the template; **those audio files are not included with the node package**.
4. Use the general cache node for music and the speech cache node for speech. Both template model nodes have `download=true`; turn downloads off once the models are available locally. `stem_mix` also requires the speech model cache and a Demucs environment.
5. Enable the branch you need with the group switch on the left. Running only one branch at a time is recommended to avoid accidentally processing the other test input.
6. After the run, audition the result at the save node and inspect the log. The output prefix is `audio/Musefish_UniverSR`, under ComfyUI's configured output directory; the template saves FLAC.

### 16 GB GPU Reference Template Preset

| Item | Music branch | Speech branch |
| --- | --- | --- |
| Model | general | speech |
| Processing mode | `sr_master` (super-resolution + mastering) | `sr` (super-resolution only; fixed) |
| `input_sr` | `auto`, with general high-frequency correction | `auto`, retaining bandwidth detection |
| `channel_mode` | `auto` | `auto` |
| ODE | `midpoint`, 4 steps | `midpoint`, 4 steps |
| `guidance` | 1.5 | 1.5 |
| `chunk_sec` | **30 seconds** | **20 seconds** |
| Seed strategy | `randomize` | `randomize` |
| Initial state | Enabled | Bypassed |

These values come from the reference workflow; they are not a VRAM guarantee for every 16 GB GPU and every source. If memory is insufficient, reduce `chunk_sec` first, for example from 30 to 15 and then to 10. Chunking controls the size of each inference segment; it does not eliminate memory use for the complete input/output audio. Do not apply video nodes' automatic-batching rules to audio nodes. For parameter comparisons, keep the seed fixed and change only one parameter at a time.

> The template's explicit `mode` values (`sr_master` / `sr`) are treated as manual selections and retained. `ODE 4 steps` and `guidance 1.5` are still the controls' default values and are overwritten from measured source properties when `auto_params` is enabled. To run exactly with every value in the table, turn `auto_params` off. This switch does not change the semantics of `chunk_sec`, the seed strategy, or `input_sr=auto`.

### Processing Modes and Use Cases

| `mode` | Processing chain | Use and notes |
| --- | --- | --- |
| `sr` | UniverSR super-resolution → M/S synthesis → bandwidth cleanup → band-limited de-essing (enabled by default for general) → true-peak finalization | For genuinely bandwidth-limited music/effects. The speech node offers only this mode and does not apply band-limited de-essing. |
| `master` | Skip model super-resolution → V8 mastering chain (EQ → multiband compression → dry ER → high-frequency shaping → TP-feedback limiting → 19.2k cutoff) | For material that needs no bandwidth reconstruction but does need spectral/dynamic adjustment. No super-resolution runs; ODE, guidance, and super-resolution chunk parameters do not apply. Loudness remains at the source level. |
| `sr_master` | Mid super-resolution → Mid mastering (reserve 4 dB of true-peak headroom + in-chain de-essing) → Side band-limit + decorrelation above 14k → M/S synthesis → bandwidth cleanup → true-peak finalization | For sources needing both bandwidth reconstruction and mastering. Above 14k, Side uses source-shaped noise to add width and avoid mono high frequencies. |
| `stem_mix` | Demucs separates vocals/accompaniment → speech processes vocals and general processes accompaniment → restore vocal 8–12k/12–16k → master accompaniment + restore 12–16k → mix at source levels → anchor source loudness → true-peak finalization | For stem enhancement. It needs more models and generally more resources/time. The 16 GB two-branch template does not establish a VRAM guarantee for this mode. |

Automatic correction can be used with `sr_master`; no extra node is needed. General-model tier selection happens before mode dispatch, and the super-resolution portion uses the corrected tier. `stem_mix` passes the selected tier to the separated tracks; it does not independently redetect each track. `master` does not run super-resolution, so tier-selection information in its log does not mean model bandwidth reconstruction will take place.

### Processing Node Parameters

The following are the **defaults for a newly created node**, separate from the template preset above.

| Parameter | Default / range | Function and tuning notes |
| --- | --- | --- |
| `audio` | Required; standard `AUDIO` | Audio input; supports batches and mono/stereo. Output is 48 kHz. |
| `auto_params` | `true` (first control) | Automatically selects `input_sr` / `ode_steps` / `guidance` based on measured source properties. **When enabled, these three controls are hidden in the UI.** When disabled, they are shown and the selected values are used. Any parameter manually changed from its default takes precedence (see below). |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000` | Input sample-rate tier corresponding to the effective bandwidth of the content, not the desired output sample rate; the 16k tier corresponds to about 8 kHz bandwidth. Prefer automatic selection and override manually when bandwidth evidence is reliable. See “Overview” for `auto` criteria, tiers, and the 8k guard. If `auto_params` is off while this remains `auto`, tier selection is still automatic. |
| `ode_steps` | 4; 1–25 | Number of super-resolution solver steps. More steps require more computation and cannot recover original information already removed by an incorrectly selected input tier. |
| `guidance` | 1.5; 0–5, step 0.1 | Conditional guidance strength. Higher values may strengthen generated texture and may also sound less natural. Keep 1.5 initially, then compare short excerpts using the same seed. |
| `mode` | `auto`; `auto` / `sr` / `master` / `sr_master` / `stem_mix` | Shown only on the general node; speech is fixed to super-resolution only. `auto` delegates selection to source measurements. Selecting a specific mode is a manual override and takes precedence. |
| `channel_mode` | `auto` / `mono` / `stereo` | `auto` follows the input channel count; it does not force a choice based on general vs. speech. `mono` averages stereo channels; `stereo` duplicates mono to two channels. Neither can recreate a genuine soundstage from nothing. |
| `ode_method` | `midpoint`; `euler` / `midpoint` / `rk4` | Super-resolution solver method. Start with midpoint; a more complex solver is not a guarantee of better sound. |
| `chunk_sec` | 15; 1–120 seconds | Duration of each super-resolution chunk; chunks are joined with about 50 ms overlap. Reduce it if VRAM is insufficient; using the maximum for long audio is not recommended. |
| `seed` | 0; non-negative integer | Random seed. The frontend's `randomize` strategy is suitable for generation; choose `fixed` for A/B comparisons. The template uses `randomize`. |
| `deess` | `true` | Band-limited de-essing (only reduces the in-band component from 5.5–8.5k, default maximum 8 dB; threshold is the in-band envelope median +6 dB and is independent of overall loudness). Applies only to `sr` + general; natural speech sibilants at 6–8k are not usually excessive, so the speech node skips it to avoid damaging them. `sr_master` / `stem_mix` use in-chain de-essing and do not apply it again. With the default left unchanged, source measurements determine its state (see below); turning it off counts as a manual selection. |
| `accel` | `cuDNN TF32`; `fp32` / `cuDNN TF32` / `bf16` | GPU acceleration mode for model inference (see “Acceleration Options”). Default `cuDNN TF32`: 1.18× faster, with about -48 dBFS difference from fp32. `fp32` is the sample-by-sample-consistent reference and the slowest. `bf16` is 1.35× faster but can produce audible waveform differences. No effect in `master`, which does not run the model. Without CUDA, automatically falls back to `fp32`. |
| `model_cache` | Optional STRING connection | Connect the model node output. If unconnected, the configured default model directory is used. Changing mode does not turn a general processing node into a speech node. |
| Output `audio` / `log` | AUDIO / STRING | Connect respectively to an audio save node and a text display node. The log includes the actual selected tier and chunk progress. |

### Model Preparation Node Parameters

| Parameter | New-node default | Description |
| --- | --- | --- |
| `model` | `general` | `general` downloads the music/effects model; `speech` downloads the speech model. Match it to the corresponding processing branch. |
| `download` | `false` | When `true`, requests a model download to the cache; the template sets this to true. Before offline use, verify that all required weights are present. |
| Output `model_cache` | STRING | Cache-directory reference, not AUDIO; do not connect it to an audio-signal input. |

### Source-Driven Parameter Matching (`auto_params`, enabled by default; first control)

This uses the same measurements as automatic tier selection (`content_cutoff` and band-level differences) to map restoration need to processing parameters. Rules come from same-seed measurements across 4 sources × 4 configurations:

- `need = clip((-23 - d12) / 20, 0, 1)`, where `d12 = B(12-16k) - B(4-6k)`. `need = 0` means the source already contains high frequencies (a full-band finished mix); super-resolution would only add hiss, so general selects `master`. `need > 0` selects `sr_master` for general.
- The speech model only provides super-resolution, so it remains in `sr` regardless of `need` (when `need = 0`, the log explains that the source does not need enhancement).
- `guidance`: speech uses `1.0` when `need < 0.5`, otherwise `1.5`; general always uses `2.0` (the music-domain setting is based on listening and was not covered by this measurement run).
- `ode_steps`: speech uses `8` when `need < 0.5` (difference from 16 steps was ≤0.25 dB, within noise, so speed wins) and `16` when `need ≥ 0.5` (on a genuinely restoration-needy source, 8 steps differed by 0.62 dB). General always uses `16` (decisive in music measurements: 12–16k band difference improved from -14.1 to -4.6 dB).
- `deess`: always on for general; for speech, on only when 6–8k exceeds 4–6k by at least 2 dB. Natural sibilants measured -1.7…-2.5 dB, so enabling it for those would cause damage.
- If the source cannot be read (silence/too short), fall back to domain defaults and explain this in the log. If `mode` is outside the current node's allowed set, preserve the node's existing value; this prevents the speech node from requesting `master`.

**Manual changes take precedence.** Whether a value is still equal to its control default determines its status: `input_sr` / `ode_steps` / `guidance` are hidden while the UI switch is on; `mode` (default `auto`) and `deess` (default `true`) remain visible. Any item changed from its default is treated as manually set and is not overwritten by automatic matching; values left at their defaults are selected from source measurements. The log prints the values actually used and retained node values (`kept node values: ...`), as well as the evidence for each decision (`auto params → ...`).

**UI and ordering.** `auto_params` is the first node control. When enabled, the following `input_sr` / `ode_steps` / `guidance` controls are hidden; disabling it restores manual selection. After updating the plugin from an older version, hard-refresh the page (`Ctrl+Shift+R`); otherwise controls may still render in the old order, saved values can map to the wrong controls, and queue validation may fail. After refresh, settings such as `mode` / `chunk_sec` / `seed` in existing workflows are read back correctly.

### Acceleration Options (`accel`)

Measured differences on an RTX 5070 Ti:

| Mode | Speed-up | Maximum sample difference vs. `fp32` | Conclusion |
| --- | --- | --- | --- |
| `fp32` | — | — | Sample-by-sample identical to the previously listening-validated version; use it when reproducing the reference. |
| `cuDNN TF32` | **1.18×** | 3.9e-3 (≈ -48 dBFS) | **Default mode.** |
| `bf16` | 1.35× | 0.26 (≈ -11.7 dB, RMS -38.6 dB) | Available, but has audible precision loss. |

On machines without CUDA, any non-`fp32` selection automatically falls back to `fp32`. This setting affects model inference in this node only; it does not change ComfyUI's global GPU configuration.

### Troubleshooting and Operational Notes

- **Container sample rate is not effective bandwidth.** A 48 kHz file may contain only low-bandwidth content; conversely, having 99% of energy in low frequencies does not mean the remaining high frequencies can be discarded. Tiers use `effective = max(rolloff, content cutoff)` and skip upward correction under the `effective ≤ 5000 Hz` guard. See “Overview” for the criteria and tier table.
- Example log: `auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`. `base` is the initial detected tier and `corrected` is the tier actually used. When blocked by the 8k guard: `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`. Neither means the output changes from 16 kHz to 24 kHz; audio output remains **48 kHz**.
- **Output sounds duller after super-resolution:** first check the actual `input_sr`; then compare `sr` and `sr_master` on the same short excerpt with a fixed seed. The mastering chain still shapes high frequencies, so automatic tier promotion cannot guarantee an unchanged timbre. Band-limited de-essing acts only when `deess=true`, `mode=sr`, and model=general; disabling it for comparison can identify changes in 5.5–8.5k.
- **Finished music already has full high frequencies:** do not classify it as low-resolution just because it is MP3. Even the highest 24k input tier may filter original content above about 12 kHz. Prefer preserving the original; if mastering is needed, compare `master` rather than blindly applying super-resolution.
- **High-frequency noise or soundstage changes:** super-resolution generates texture, and stereo processing also includes M/S processing. Compare super-resolution-only output with the original first; do not judge quality only by increasing guidance, steps, or volume. The stereo paths for `sr_master` / `sr` use decorrelated noise above 14k to add width; if it sounds too noisy, compare direct `mode=sr` output.
- **Restart ComfyUI after node-code updates** and avoid reusing cached results from older code. Use a fixed seed for fair comparisons; on the first validation after code changes, changing the seed can force a fresh execution.
- **Runtime depends on `input_sr`:** 8k is fastest and 24k is most expensive. A 60-second `sr_master` source with automatic selection of 24k takes about 7–8 minutes; a 200-second full-track `master` run generally takes less than a minute because it does not run the model. Near-full GPU utilization during long runs is normal, not a hang.
- **Missing model or node:** check the model cache and the appropriate general/speech weights, plus the template's rgthree, text display, and `SaveAudioAdvanced` nodes. If the save node is unavailable, upgrade ComfyUI or replace it with an audio save node provided by your version. If `stem_mix` cannot find a model, confirm `model_cache` points to the parent directory containing both `general` and `speech`.

### Audio-Related Files

- `musefish_audio.py`: model preparation node, General/Speech processing nodes, AUDIO tensor adaptation, and isolated worker scheduling.
- `audio_backend/processing.py`: automatic bandwidth detection and general-model correction, chunked super-resolution, and the four processing-mode chains/finalization.
- `audio_backend/dsp.py`: mastering and audio post-processing (EQ / multiband compression / de-essing / high-frequency shaping / limiting / bandwidth cleanup / Side-channel decorrelation).
- `audio_backend/worker.py`: isolated processing-process entry point.
- [Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json): music/speech two-branch reference template, with separate model-cache, audio-save, and log-display connections.
