# Musefish ASR 节点（节点显示名 `Musefish ASR`）

`musefish_asr` — 把 ComfyUI 的 AUDIO 批次转写成文本。引擎代码与
[Musefish Studio](../../../MusefishStudio) 的 `server/asr.py` **同源**（节点运行时
注入 Studio server 目录后 import，插件内不维护第二份实现），模型库共用
`<ComfyUI>/models/audio_encoders/faster-whisper`（也可用环境变量 `MUSEFISH_ASR_MODELS` 指定），与 Studio 工作台的识别结果完全一致。

## 节点

| 项 | 值 |
|---|---|
| ID | `musefish_asr` |
| 分类 | Musefish/Audio |
| 输入 | `AUDIO`（批次）、`engine`、`language`、`precision` |
| 输出 | `text`（STRING，多 clip 按行拼接）、`list_path`（STRING，.list 明细绝对路径） |
| 输出节点 | 是（可不接下游直接执行；面板实时显示识别文本） |

## 引擎与模型

| engine | 说明 | 首次下载 | 位置 |
|---|---|---|---|
| `large-v3-turbo（多语种·推荐）` | faster-whisper，速度/质量平衡 | ~1.6 GB | `<模型库>/faster-whisper-large-v3-turbo/` |
| `large-v3（多语种·最强）` | faster-whisper | ~2.9 GB | `.../faster-whisper-large-v3/` |
| `small（多语种·最快）` | faster-whisper | ~0.5 GB | `.../faster-whisper-small/` |
| `funasr（中文·带标点）` | Paraformer-large + VAD + 标点，中文输出自动带标点 | ~1.1 GB | `.../speech_paraformer-*` 等 |

- 模型缺失时**首次执行自动下载**（HF，默认 `HF_ENDPOINT=https://hf-mirror.com`）；
  Studio 工作台则走 409 → 确认 → 进度条流程。
- 下载后全部离线。`language` 只对 whisper 生效（auto/zh/yue/en/ja/ko）；
  funasr 仅 zh/yue。whisper 检出 zh/yue 时会用 FunASR 二次解码以获得标点。

## 性能

- CUDA 下走 `BatchedInferencePipeline`（`batch_size=8`）分块并行，
  60 s 音频 large-v3 fp16 实测 **7.1 s**（RTF≈0.12）、turbo warm **0.8 s**；
  CPU 自动回退单条解码。
- `precision=int8` 显存/内存减半；beam_size=1（贪心）为转写场景调优。
- 模型进程级缓存（LRU=1）：同引擎重复调用免加载；切换引擎自动释放旧模型。
- ComfyUI 节点路径：张量重采样 16k 后**内存直喂**引擎，无临时 WAV 解码往返。

## 数据格式

`.list` 每行：`wav绝对路径|说话人|语言|文本`（与 GPT-SoVITS 标注格式兼容）。
多 clip 批次每个 clip 一行；`text` 输出为各行文本的拼接。

## 依赖

`funasr`、`faster-whisper`、`librosa`、`soundfile`、`torch`（ComfyUI 环境已有）。
仅使用 `funasr` 引擎时可不装 `faster-whisper`，反之亦然（延迟导入，用到才加载）。

## Studio 联动

Musefish Studio 工作台（音频处理 → 语音识别）与 ComfyUI 节点调用同一
`asr.py`：模型、参数语义、`.list` 产物互通。工作台额外提供任务队列、
409→模型下载进度条、识别文本直显与下载。
