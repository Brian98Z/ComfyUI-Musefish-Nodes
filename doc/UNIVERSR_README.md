# UniverSR 音频超分与母带处理

### 节点与功能

- **Musefish UniverSR Model**（`musefish_universr_model`）：准备 general/speech 模型缓存，可按需下载；输出目录引用，不处理音频。
- **Musefish UniverSR General Audio**（`musefish_universr_general_audio`）：general 模型，支持 `sr`、`master`、`sr_master`、`stem_mix`。
- **Musefish UniverSR Speech Audio**（`musefish_universr_speech_audio`）：speech 模型，仅 `sr`，用于语音/人声。
- 处理节点接收标准 `AUDIO`（`waveform` 为 `B,C,T`），支持批次、单/双声道，输出 48 kHz；通过 `model_cache` 接模型缓存。

### 环境、模型与工作流

ComfyUI worker 使用其 `sys.executable`（Python 3.10–3.13；Torch/音频依赖来自该环境）。模型目录为 `ComfyUI/models/UniverSR/models/huggingface/<general|speech>`；每个模型须同时有 `config.yaml` 和 `pytorch_model.bin`。仅 `download=true` 时联网，按 `HF_ENDPOINT`（默认 `https://hf-mirror.com`）逐文件下载并原子改名；导入插件不联网，不使用 `snapshot_download`。

- general：[woongzip1/universr-audio](https://huggingface.co/woongzip1/universr-audio/tree/main)；speech：[woongzip1/universr-speech](https://huggingface.co/woongzip1/universr-speech/tree/main)。
- `model_cache`（或 `MUSEFISH_UNIVERSR_MODEL_CACHE`）可指向缓存目录；worker 也识别 `<cache>/universr-audio|universr-speech`、`<cache>/general|speech`、`<cache>/models--woongzip1--<name>`、`<cache>/huggingface/...` 及其中的 Hugging Face `snapshots/`。
- `stem_mix` 同时需要 general 与 speech 两套模型，`model_cache` 指向其父目录 `ComfyUI/models/UniverSR/models/huggingface`；还需同一 Python 环境中的 Demucs 模块。
- 工作流：[Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json)，ID `6e417a01-39fa-461d-9f27-484c97aeef2e`，16GB 显存参考模板，音乐/语音为独立分支。依赖 `rgthree-comfy` 的 `Fast Groups Bypasser (rgthree)` 与 `ComfyUI-Custom-Scripts` 的 `ShowText|pysssss`；可删开关或不接日志。示例名 `测试歌曲.mp3`、`测试语音.wav` 不随插件附带。输出前缀 `audio/Musefish_UniverSR`（ComfyUI 输出目录），模板保存 FLAC；两模型节点预设 `download=true`。
模板预设：音乐 general，`sr_master`、`input_sr=auto`、`channel_mode=auto`、`ode_method=midpoint`/4 步、`guidance=1.5`、`chunk_sec=30` 秒、种子 `randomize`，默认启用；语音 speech 固定 `sr`，其余对应设置相同但 `chunk_sec=20` 秒，默认旁路。`auto_params` 开启时 ODE 步数和 `guidance` 会按素材实测改写；要固定表值请关闭。16GB 仅参考，显存不足先减 `chunk_sec`，如 30→15→10。

### 处理模式与默认参数

| `mode` | 链路/用途 |
| --- | --- |
| `sr` | UniverSR 超分、M/S 合成、带宽清理、general 默认分带去齿音、真峰收口；speech 仅此模式且不去齿音。 |
| `master` | 不跑模型；V8 母带：EQ、多段压、干 ER、高频整形、TP 反馈限幅、19.2k 切除；响度保持源状态。 |
| `sr_master` | Mid 超分与母带（预留 4 dB 真峰余量、链内去齿音），Side 带限及 14k+ 去相关、M/S 合成、带宽清理、真峰收口。 |
| `stem_mix` | Demucs 分人声/伴奏，speech/general 分别处理，回补人声 8-12k/12-16k 与伴奏 12-16k，按源电平混音、锚定源响度、TP 收口；耗时/资源较高，不保证 16GB 显存可用。 |

| 处理节点参数 | 新建默认值 / 范围 |
| --- | --- |
| `audio` | 必接 `AUDIO`；批次及单/双声道，输出 48 kHz |
| `auto_params` | `true`；首个控件，自动选 `input_sr`/`ode_steps`/`guidance`，开启时三项隐藏 |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000`；内容带宽档，不是输出采样率；16k 档约 8kHz 带宽 |
| `ode_steps` | 4；1–25 |
| `guidance` | 1.5；0–5，步长 0.1 |
| `mode` | `auto`；`auto`/`sr`/`master`/`sr_master`/`stem_mix`；仅 general 显示，speech 固定 `sr` |
| `channel_mode` | `auto` / `mono` / `stereo` |
| `ode_method` | `midpoint`；`euler` / `midpoint` / `rk4` |
| `chunk_sec` | 15；1–120 秒，约 50ms 重叠拼接；显存不足时降低此值 |
| `seed` | 0；非负整数；模板策略 `randomize` |
| `deess` | `true`；仅 `sr` + general，压 5.5–8.5k 带内分量，默认上限 8 dB、阈值为带内包络中位 +6 dB；`sr_master`/`stem_mix` 用链内去齿音 |
| `accel` | `cuDNN TF32`；`fp32` / `cuDNN TF32` / `bf16` |
| `model_cache` | 可选 STRING；缓存目录连接，不接音频信号 |
| 输出 `audio` / `log` | `AUDIO` / `STRING`；日志含实际选档与分块进度 |

模型准备节点参数：`model=general`（或 `speech`）；`download=false`（`true` 下载）；输出 `model_cache` 为 STRING。

### 自动选档与自动参数

`input_sr=auto` 使用 `effective = max(99% rolloff, content cutoff)`；`content cutoff` 是 rolloff 以上频谱首次低于 rolloff 电平 20 dB 的频率。档位：`effective ≤ 5.4 kHz → 8k`（5.0–5.4 kHz 边界先试 8k）；`≤ 7.2 kHz → 16k`；`≤ 12 kHz → 16k`；其余 24k，不输出 12k 档。`effective ≤ 5000 Hz` 时强制 8k、不向上校正；校正窗口 4150–5850 Hz 曾将 3400–5000 Hz 素材误升至 12k/16k，5062 Hz 仍可能升至 12k。

仅 general 向上校正：候选频带占全谱能量 ≥1e-5（-50 dB）、连续带宽约 ≥300 Hz、高于噪声底 6 dB 且距带峰 ≤20 dB、至少 25% 分析帧有能量，连续 3 帧（少于 12 帧时 2 帧）；带边留 150 Hz，只升不降，复用 FFT。静音、低底噪、宽带 hiss、孤立尖峰/瞬态通常不满足条件；speech 不向上校正，手动 `input_sr` 不变。最高输入档 24 kHz 不保证保留 12 kHz 以上原始细节。

`auto_params` 默认开启，规则来自 4 素材 × 4 配置同种子实测：`need = clip((-23 - d12) / 20, 0, 1)`，`d12 = B(12-16k) - B(4-6k)`；general 在 `need=0` 用 `master`，否则 `sr_master`；speech 始终 `sr`。speech `guidance`：`need < 0.5 → 1.0`，否则 `1.5`；general 固定 `2.0`（音乐域由听感定型，未实测覆盖）。speech `ode_steps`：`need < 0.5 → 8`（与 16 步差 ≤0.25 dB，噪声内），否则 16（修复素材 8 步差 0.62 dB）；general 固定 16（12-16k 落差 −14.1 → −4.6 dB）。general `deess` 恒开；speech 仅在 6-8k 比 4-6k 高 ≥2 dB 时开（自然齿音实测 −1.7…−2.5 dB）。素材静音/过短时回退分域默认值。改离控件默认值的 `input_sr`/`ode_steps`/`guidance`/`mode`/`deess` 视为手动，不覆盖；日志含 `kept node values: ...` 与 `auto params → ...`。更新插件后硬刷新 `Ctrl+Shift+R`，避免旧控件顺序导致保存值错位、队列校验失败。

### 加速与实测

RTX 5070 Ti 测试（1 个 9.5 s 分块，16 ODE steps）：

| 档位 | 速度 | 相对 `fp32` 最大样本差 |
| --- | --- | --- |
| `fp32` | 基准，逐样本一致 | — |
| `cuDNN TF32`（默认） | 1.18× | 3.9e-3（约 −48 dBFS） |
| `bf16` | 1.35× | 0.26（约 −11.7 dB，RMS −38.6 dB） |

`fp32` 最慢；无 CUDA 时非 `fp32` 自动回退。设置仅影响本节点推理，不改 ComfyUI 全局 GPU 配置。耗时示例：60s 素材 `sr_master`、自动 24k 约 7–8 分钟；200s `master` 通常一分钟内。显存不足先降低 `chunk_sec`（如 30→15→10）；分块不消除整段输入/输出内存。更新节点代码需重启 ComfyUI、避免旧缓存；固定种子用于比较。

### 注意事项与文件

- `master` 不执行超分；选档日志不表示模型重建。`stem_mix` 的选档传给分轨，不分别重测。自动校正不保证改善所有素材；MP3 不等于低带宽，24k 输入仍可能滤除约 12kHz 以上内容。
- 听感变闷时检查实际 `input_sr`，固定种子对比 `sr`/`sr_master`；`deess` 仅在 `sr` + general 生效。噪感/声场变化可与原曲对比；立体声 `sr`/`sr_master` 在 14k 以上用去相关噪声补宽度。
- 缺模型/节点时检查两套权重、模板依赖及 `SaveAudioAdvanced`；`stem_mix` 的缓存必须含两模型。日志示例：`auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`；护栏示例：`auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`。输出仍为 48kHz。
- `musefish_audio.py`：节点、AUDIO 适配与 worker 调度；`audio_backend/processing.py`：选档、分块与模式链；`audio_backend/dsp.py`：母带和音频后处理；`audio_backend/worker.py`：处理进程入口；[Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json)：双分支模板。
