# Musefish ASR Node

`musefish_asr` — transcribes ComfyUI AUDIO batches to text. The engine code is
**shared** with Musefish Studio's `server/asr.py` (the node injects the Studio
server directory at import time; no second copy lives in this repo), and the
model library is `<ComfyUI>/models/audio_encoders/faster-whisper` (or set `MUSEFISH_ASR_MODELS`), so results match the Studio
workbench exactly.

## Node

| Item | Value |
|---|---|
| ID | `musefish_asr` |
| Category | Musefish/Audio |
| Inputs | `AUDIO` (batch), `engine`, `language`, `precision` |
| Outputs | `text` (STRING, one line per clip), `list_path` (STRING, absolute path to the .list file) |
| Output node | Yes (can run with no downstream; recognized text streams to the node panel) |

## Engines and models

| engine | Notes | First download | Location |
|---|---|---|---|
| `large-v3-turbo` | faster-whisper, best speed/quality balance | ~1.6 GB | `<model root>/faster-whisper-large-v3-turbo/` |
| `large-v3` | faster-whisper, strongest | ~2.9 GB | `.../faster-whisper-large-v3/` |
| `small` | faster-whisper, fastest | ~0.5 GB | `.../faster-whisper-small/` |
| `funasr` | Paraformer-large + VAD + punctuation; Chinese output arrives punctuated | ~1.1 GB | `.../speech_paraformer-*` etc. |

- Missing models **download automatically on first execution** (HuggingFace,
  default `HF_ENDPOINT=https://hf-mirror.com`); the Studio workbench instead
  shows a 409 → confirm → progress-bar flow.
- Once downloaded everything runs offline. `language` applies to whisper only
  (auto/zh/yue/en/ja/ko); funasr is zh/yue. When whisper detects zh/yue the
  clip is re-decoded through FunASR for punctuation.

## Performance

- CUDA runs through `BatchedInferencePipeline` (`batch_size=8`) chunked
  parallel encode: 60 s of audio with large-v3 fp16 measured **7.1 s**
  (RTF≈0.12), turbo warm **0.8 s**; CPU automatically falls back to
  sequential decoding.
- `precision=int8` halves VRAM/RAM; `beam_size=1` (greedy) is tuned for
  transcription.
- Process-level model cache (LRU=1): repeat calls skip loading; switching
  engines frees the previous model.
- The ComfyUI node resamples to 16 kHz in memory and feeds arrays straight to
  the engines — no temporary-WAV decode round trip.

## Data format

Each `.list` line: `absolute wav path|speaker|language|text` (compatible with
GPT-SoVITS annotation lists). Batches produce one row per clip; the `text`
output joins every row's text with newlines.

## Dependencies

`funasr`, `faster-whisper`, `librosa`, `soundfile`, `torch` (already present
in ComfyUI). Imports are lazy: only the engine you call is loaded.

## Studio integration

The Musefish Studio workbench (Audio → ASR) and this node call the same
`asr.py`: identical models, parameter semantics, and `.list` outputs. The
workbench adds a job queue, 409 → model-download progress flow, and inline
text display with download.
