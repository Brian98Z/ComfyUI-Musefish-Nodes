# Musefish ASR Node

`musefish_asr` transcribes ComfyUI AUDIO batches using the same `server/asr.py` and model library as Musefish Studio.

## Node and parameters

| Item | Value |
|---|---|
| ID / Display name / Category | `musefish_asr` / `Musefish ASR` / Musefish/Audio |
| Input | AUDIO batch; `engine`, `language`, `precision` |
| `engine` (default) | `large-v3-turbo（多语种·推荐）` (default); `large-v3（多语种·最强）`; `small（多语种·最快）`; `funasr（中文·带标点）` |
| `language` (default) | `auto（自动）` (default), `zh（中文）`, `yue（粤语）`, `en（英语）`, `ja（日语）`, `ko（韩语）` |
| `precision` (default) | `float16` (default) or `int8`; `int8` halves VRAM/RAM at similar speed |
| Outputs | `text` (STRING, newline-separated clips); `list_path` (STRING, absolute `.list` path) |
| Execution | Output node; runs independently and displays recognized text in its panel |

## Engines, models, and paths

| Engine | Use | First download | Model directory |
|---|---|---|---|
| `large-v3-turbo` / `large-v3` / `small` | Multilingual faster-whisper | ~1.6 GB / ~2.9 GB / ~0.5 GB | `faster-whisper-large-v3-turbo/`, `faster-whisper-large-v3/`, `faster-whisper-small/` |
| `funasr` | Paraformer-large + VAD + punctuation; Chinese segmentation | ~1.1 GB | `speech_paraformer-*` etc. |

When no model root is explicitly set, resolution priority is `MUSEFISH_ASR_MODELS` > `folder_paths` `models/audio_encoders/faster-whisper` > fallback (`studio/data/models/asr`). Model library path: `<ComfyUI>/models/audio_encoders/faster-whisper`. Missing models download automatically on first execution (HF; default `HF_ENDPOINT=https://hf-mirror.com`); the Studio workbench uses a 409 → confirmation → progress flow. Models work offline after download.

`language` applies to faster-whisper only (`auto`/`zh`/`yue`/`en`/`ja`/`ko`); funasr supports `zh`/`yue`. faster-whisper detections of `zh`/`yue` are re-decoded by FunASR for punctuation; funasr provides Chinese segmentation.

## Performance and output format

- With CUDA (`BatchedInferencePipeline`, `batch_size=8`), large-v3-turbo processed 60s of audio in ~7s in measured conditions. CPU falls back to sequential decoding.
- Each `.list` line is `absolute wav path|speaker|language|text`, compatible with GPT-SoVITS annotation lists. One line per clip; `text` joins the clip texts.
