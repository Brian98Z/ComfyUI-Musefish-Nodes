# Musefish ASR 节点（`Musefish ASR`）

`musefish_asr`：将 ComfyUI AUDIO 批次转写为文本；与 Musefish Studio 共用 `server/asr.py` 和模型库。

## 节点与参数

| 项 | 值 |
|---|---|
| ID / 分类 | `musefish_asr` / Musefish/Audio |
| 输入 | AUDIO 批次；`engine`、`language`、`precision` |
| `engine`（默认） | `large-v3-turbo（多语种·推荐）`（默认）；`large-v3（多语种·最强）`；`small（多语种·最快）`；`funasr（中文·带标点）` |
| `language`（默认） | `auto（自动）`（默认）、`zh（中文）`、`yue（粤语）`、`en（英语）`、`ja（日语）`、`ko（韩语）` |
| `precision`（默认） | `float16`（默认）或 `int8`；`int8` 显存/内存减半，速度相近 |
| 输出 | `text`（STRING，多 clip 按行拼接）；`list_path`（STRING，`.list` 绝对路径） |
| 执行 | 输出节点，可独立运行；面板显示识别文本 |

## 引擎、模型与路径

| 引擎 | 用途 | 首次下载 | 模型目录 |
|---|---|---|---|
| `large-v3-turbo` / `large-v3` / `small` | faster-whisper，多语种 | ~1.6 GB / ~2.9 GB / ~0.5 GB | `faster-whisper-large-v3-turbo/`、`faster-whisper-large-v3/`、`faster-whisper-small/` |
| `funasr` | Paraformer-large + VAD + 标点；中文断句/标点 | ~1.1 GB | `speech_paraformer-*` 等 |

未显式指定模型库时，解析优先级：`MUSEFISH_ASR_MODELS` > `folder_paths` 的 `models/audio_encoders/faster-whisper` > fallback（`studio/data/models/asr`）。模型库路径：`<ComfyUI>/models/audio_encoders/faster-whisper`。缺失模型首次执行自动下载（HF，默认 `HF_ENDPOINT=https://hf-mirror.com`）；Studio 工作台走 409 → 确认 → 进度流程。下载后可离线运行。

`language` 仅用于 faster-whisper（auto/zh/yue/en/ja/ko）；funasr 支持 zh/yue。faster-whisper 识别到 zh/yue 时会交由 FunASR 二次解码以加标点；funasr 为中文断句引擎。

## 性能与输出格式

- CUDA（`BatchedInferencePipeline`，`batch_size=8`）下，large-v3-turbo 处理 60s 音频实测约 7s。CPU 自动回退单条解码。
- `.list` 每行 `wav绝对路径|说话人|语言|文本`，兼容 GPT-SoVITS 标注；每个 clip 一行，`text` 拼接各行文本。
