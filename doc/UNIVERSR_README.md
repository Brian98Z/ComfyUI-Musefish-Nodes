# UniverSR 音频超分与母带处理

### 功能概览

- **`Musefish UniverSR Model`**：准备 general / speech 模型缓存，可按需下载；输出缓存目录，不处理音频。

- **`Musefish UniverSR General Audio`**（节点 ID：`MusefishUniverSRGeneralAudio`）：固定使用 general 模型，支持 `sr`（仅超分）、`master`（V8 母带）、`sr_master`（超分+母带）和 `stem_mix`（Demucs 分轨+混音）。
- **`Musefish UniverSR Speech Audio`**（节点 ID：`MusefishUniverSRSpeechAudio`）：固定使用 speech 模型，仅支持 `sr` 超分；不暴露模型选择，适合人声/语音增强。
- 两个节点都处理标准 `AUDIO`（`waveform` 为 `B,C,T`），逐批支持单声道/立体声，并通过 `model_cache` 输入连接 `Musefish UniverSR Model` 的缓存输出。
- `input_sr=auto` 的档位判据是 `effective = max(99% rolloff, content cutoff)`。`content cutoff` 指 rolloff 之上频谱首次跌落到 rolloff 电平 20 dB 以下处的频率——MP3 的 rolloff 可低至约 9.6 kHz 而内容一直延伸到约 15 kHz，只看 rolloff 会误选 24k 档，让模型凭空生成 8–12 kHz。档位表：`effective ≤ 5.4 kHz → 8k`（5.0–5.4 kHz 边界带先用 8k 试探）、`≤ 7.2 kHz → 16k`、`≤ 12 kHz → 16k`（内容到 12k 时 16k 条件专注清理与增强，不凭空生成）、其余 `→ 24k`；不再直接输出 12k 档。
- **8k 硬落地护栏**：`effective ≤ 5000 Hz` 时直接采用 8k 档，不做向上校正。校正判据窗口（4150–5850 Hz）会把滤波过渡带残能当成「持续高频证据」，实测把 effective 3400–5000 Hz 的素材全部升到 12k/16k。5.0–5.4 kHz 边界带仍走校正，因此 `effective = 5062 Hz` 的低带宽素材可被升到 12k 档。
- 仅 general 模型做向上校正，条件是候选频带满足：占全谱能量 ≥ 1e-5（-50 dB）、连续带宽 ≥ 约 300 Hz、同时高于噪声底 6 dB 且距候选带峰值不超过 20 dB、至少 25% 的分析帧有能量、并有连续 3 个分析帧的证据（不足 12 帧时要求 2 帧）；带边留 150 Hz 余量，只向上、不向下。复用同一次 FFT，不因容器标称 48 kHz 一律升档。日志形如 `auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`，被护栏拦下时形如 `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`。
- 静音、低底噪、宽带 hiss、孤立尖峰或瞬态通常不能满足连续频带与时域门限；speech 模型用同一 `effective` 判据但不做向上校正；手动 `input_sr` 不检测、不改写。
- 更新后需重启 ComfyUI 以加载主进程中的选档逻辑。24 kHz 是模型支持的最高输入档位，不能保证保留原曲 12 kHz 以上的细节；该校正不改变母带链，也不保证所有音乐都能改善听感。

### 运行环境与模型路径

worker 使用启动 ComfyUI 的 `sys.executable`，支持 Python 3.10–3.13 的 ComfyUI 环境；具体 Torch/音频依赖由当前环境提供。`Musefish UniverSR Model` 节点的缓存目标是 `ComfyUI/models/UniverSR/models/huggingface/<general|speech>`，每个目录需要同时存在 `config.yaml` 与 `pytorch_model.bin`；只有 `download=true` 时才访问网络，按 `HF_ENDPOINT`（默认 `https://hf-mirror.com`）逐文件下载后原子改名，导入插件时不联网，也不用 `snapshot_download`。

处理节点的 `model_cache` 输入（或 `MUSEFISH_UNIVERSR_MODEL_CACHE`）决定 worker 到哪里找权重，解析顺序：该目录本身 → `<cache>/universr-audio|universr-speech` → `<cache>/general|speech` → `<cache>/models--woongzip1--<name>` → `<cache>/huggingface/...`，以及上述目录下的 HuggingFace `snapshots/` 快照。单模型模式（`sr` / `sr_master`）直接连接模型节点输出即可；**`stem_mix` 需要一次解析出两个模型**，因此 `model_cache` 应指向同时含 `general` 与 `speech` 的父目录（即 `ComfyUI/models/UniverSR/models/huggingface`），只连某一个模型节点的输出会找不到另一个模型。`stem_mix` 使用同一 Python 环境中的 Demucs 模块。

### 音频示例工作流

示例文件：[Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json)，工作流 ID：`6e417a01-39fa-461d-9f27-484c97aeef2e`。这是 **16GB 显存参考模板**，包含音乐与语音两条独立分支，不是把同一段音频先后送入两个模型。

```text
音乐分支（默认启用）
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRGeneralAudio
MusefishUniverSRModel（general）─ model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced（FLAC）
                                             └─ log   → ShowText|pysssss

语音分支（默认旁路）
LoadAudio ── AUDIO ──────────────────────→ MusefishUniverSRSpeechAudio
MusefishUniverSRModel（speech）── model_cache ─→ ↑
                                             ├─ audio → SaveAudioAdvanced（FLAC）
                                             └─ log   → ShowText|pysssss
```

**使用步骤：**

1. 将 JSON 拖入 ComfyUI，或从工作流模板入口加载。
2. 模板额外使用 `rgthree-comfy` 的 `Fast Groups Bypasser (rgthree)` 和 `ComfyUI-Custom-Scripts` 的 `ShowText|pysssss`。缺少时安装对应节点包；也可删除分组开关并手动旁路分支，日志输出不连接不影响音频处理。
3. 在对应 `LoadAudio` 中上传并选择自己的素材。`测试歌曲.mp3`、`测试语音.wav` 是模板示例文件名，**不代表节点包附带这些音频**。
4. 音乐使用 general 缓存节点，语音使用 speech 缓存节点。模板两者 `download=true`；准备好本地模型后可关闭下载。`stem_mix` 还需要 speech 模型缓存和 Demucs 环境。
5. 用左侧分组开关启用需要的分支。建议一次只运行一条分支，避免误处理另一份测试输入。
6. 运行后在保存节点试听结果、查看日志；输出前缀为 `audio/Musefish_UniverSR`，位于 ComfyUI 配置的输出目录内，模板格式为 FLAC。

### 16GB显卡 参考模板预设

| 项目 | 音乐分支 | 语音分支 |
| --- | --- | --- |
| 模型 | general | speech |
| 处理方式 | `sr_master`（超分+母带） | `sr`（仅超分，固定） |
| `input_sr` | `auto`，启用 general 高频校正 | `auto`，保留原带宽检测 |
| `channel_mode` | `auto` | `auto` |
| ODE | `midpoint`，4 步 | `midpoint`，4 步 |
| `guidance` | 1.5 | 1.5 |
| `chunk_sec` | **30 秒** | **20 秒** |
| 种子策略 | `randomize` | `randomize` |
| 初始状态 | 启用 | 旁路 |

这组预设来自参考工作流，不是所有 16GB 显卡与素材的显存保证。显存不足优先减小 `chunk_sec`，例如从 30 降到 15，再降到 10；分块只控制单段推理规模，不会让整曲输入/输出完全不占内存。不要把视频节点的自动分批规则套到音频节点上。对比参数效果时固定种子，并只改一个参数。

> 预设里写死的 `处理方式`（`sr_master` / `sr`）是显式取值，会被当作手动指定保留；`ODE 4 步` 与 `guidance 1.5` 仍是控件默认值，`auto_params` 开启时按素材实测改写。要完全按上表固定执行，请关闭 `auto_params`；`chunk_sec`、种子策略、`input_sr` 的 `auto` 语义不受该开关影响。

### 处理模式与适用场景

| `mode` | 处理链路 | 适用与注意事项 |
| --- | --- | --- |
| `sr` | UniverSR 超分 → M/S 合成 → 带宽清理 → 分带去齿音（general 默认） → 真峰收口 | 真正低带宽的音乐/音效；speech 节点只提供此模式，且不做分带去齿音 |
| `master` | 跳过模型超分 → V8 母带链（EQ → 多段压 → 干 ER → 高频整形 → TP 反馈限幅 → 19.2k 切除） | 不需要补带宽、只希望调整频谱与动态的素材；不执行超分，ODE、guidance 和超分分块参数不参与此路径，响度保持源状态 |
| `sr_master` | Mid 超分 → Mid 母带（预留 4 dB 真峰余量 + 链内去齿音） → Side 带限 + 14k+ 去相关 → M/S 合成 → 带宽清理 → 真峰收口 | 同时需要带宽重建与母带处理；Side 在 14k 以上用按素材塑形的噪声补宽度，避免高频单声道 |
| `stem_mix` | Demucs 分人声/伴奏 → speech 处理人声、general 处理伴奏 → 人声 8-12k/12-16k 回补 → 伴奏母带 + 12-16k 回补 → 按源电平混音 → 锚定源响度 → TP 收口 | 需要分轨增强；依赖更多模型，资源与耗时通常更高，16GB 双分支模板不代表此模式已做显存保证 |

**自动校正可以用于 `sr_master`，无需新增节点。** General 的自动选档发生在模式分流之前，超分部分直接使用校正结果；`stem_mix` 当前将选定档位传给分轨处理，不是分别对每条分轨重新自动识别。`master` 不做超分，日志中的选档信息不表示它会执行模型带宽重建。

### 处理节点参数

以下是**新建节点的默认值**，与上面的模板预设分开看。

| 参数 | 默认值 / 范围 | 作用与调节建议 |
| --- | --- | --- |
| `audio` | 必接，标准 `AUDIO` | 输入音频；支持批次及单/双声道，输出为 48 kHz |
| `auto_params` | `true`（首个控件） | 按素材实测自动决定 `input_sr` / `ode_steps` / `guidance`，**开启时这三个控件在界面上隐藏**；关闭后它们显示出来、完全按你选的值执行。手动改动过的参数始终优先（见下节） |
| `input_sr` | `auto` / `8000` / `12000` / `16000` / `24000` | 表示内容有效带宽对应的输入采样率，不是期望输出采样率；16k 档对应约 8kHz 带宽。优先自动，有可靠带宽依据时手动覆盖。`auto` 的判据、档位表与 8k 护栏见“功能概览”；`auto_params` 关闭但此项仍为 `auto` 时，选档依然是自动的 |
| `ode_steps` | 4；1–25 | 超分求解步数；提高会增加计算量，不会恢复被错误输入档位预先滤除的原始信息 |
| `guidance` | 1.5；0–5，步长 0.1 | 条件引导强度；提高可能强化生成纹理，也可能增加不自然感。先保留 1.5，再做同种子短片段对比 |
| `mode` | `auto`；`auto`/`sr`/`master`/`sr_master`/`stem_mix` | 仅 general 节点显示；speech 固定仅超分。`auto` 表示交给素材实测选择，选定具体模式即视为手动指定并优先 |
| `channel_mode` | `auto` / `mono` / `stereo` | `auto` 按输入声道数选择，不是按 general/speech 强制选择；`mono` 会将双声道平均，`stereo` 会将单声道复制为双声道，但不能凭空恢复真实声场 |
| `ode_method` | `midpoint`；`euler` / `midpoint` / `rk4` | 超分求解方法；先使用 midpoint，不以更复杂的求解方法保证音质改善 |
| `chunk_sec` | 15；1–120 秒 | 单次超分分块时长；内部分块带约 50ms 重叠拼接。显存不足时减小，不建议长音频直接使用最大值 |
| `seed` | 0；非负整数 | 随机种子；前端生成后策略选 fixed 便于 A/B，模板使用 randomize |
| `deess` | `true` | 分带去齿音（只压 5.5–8.5k 带内分量，默认上限 8 dB，阈值取带内包络中位 +6 dB，与音量无关）。仅 `sr` 模式 + general 模型生效；speech 节点的自然人声 6–8k 本就不冲，开启会把真实齿音压掉，因此该节点自动跳过；`sr_master`/`stem_mix` 用链内去齿音，不重复作用。留默认值时按素材实测（见下节），关掉即视为手动指定 |
| `accel` | `cuDNN TF32`；`fp32`/`cuDNN TF32`/`bf16` | 模型推理的 GPU 加速档（见「加速选项」）。**默认 `cuDNN TF32`**：快 1.18×、与 fp32 差异约 −48 dBFS；`fp32` 是逐样本一致的基准档（最慢）；`bf16` 快 1.35× 但波形差异可闻。`master` 模式不跑模型，此项无作用；无 CUDA 时自动退回 `fp32` |
| `model_cache` | 可选 STRING 连线 | 连接模型节点输出；未连接时使用配置的默认模型目录。切换模式不会把 general 处理节点变成 speech 节点 |
| 输出 `audio` / `log` | AUDIO / STRING | 分别连接音频保存节点和文本展示节点；日志包含实际选档及分块进度 |

### 模型准备节点参数

| 参数 | 新建节点默认值 | 说明 |
| --- | --- | --- |
| `model` | `general` | `general` 下载音乐/音效模型，`speech` 下载语音模型；与对应处理分支匹配 |
| `download` | `false` | `true` 时请求下载模型到缓存；模板预设为 true。离线使用前先确认所需权重完整 |
| 输出 `model_cache` | STRING | 缓存目录引用，不是 AUDIO，不应串在音频信号连线上 |

### 素材驱动参数匹配（`auto_params`，默认开启，首位控件）

与自动选档共用同一套测量（`content_cutoff` + 分带落差），把「修复需求」接到参数上，规则来自 4 素材 × 4 配置的同种子实测：

- `need = clip((-23 - d12) / 20, 0, 1)`，`d12 = B(12-16k) - B(4-6k)`；`need = 0` 表示源自带高频（全带宽成品），超分只会加嘶声 → general 取 `master`；`need > 0` → general 取 `sr_master`。
- speech 模型只提供超分，因此无论 `need` 多大都保持 `sr`（`need = 0` 时日志会说明该素材不需要增强）。
- `guidance`：speech 按 `need < 0.5 → 1.0`，否则 `1.5`；general 固定 `2.0`（音乐域由听感定型，本次实测未覆盖）。
- `ode_steps`：speech 按 `need < 0.5 → 8`（实测与 16 步差 ≤0.25 dB＝噪声内，速度优先），`need ≥ 0.5 → 16`（真修复素材 8 步会差 0.62 dB）；general 固定 `16`（音乐实测决定性：12-16k 落差 −14.1 → −4.6 dB）。
- `deess`：general 恒开；speech 仅在 6-8k 反超 4-6k ≥ 2 dB（实测真实自然齿音为 −1.7…−2.5 dB，开了会误伤）时才开启。
- 素材不可读（静音/过短）时回退到分域默认值并在日志说明；`mode` 若不在该节点允许集合内则保留节点原值（不会让 speech 节点发出 `master` 请求）。

**手动改动优先。** 判断依据是「该值是否仍等于控件默认值」：`input_sr` / `ode_steps` / `guidance` 三项在界面开关开启时隐藏，`mode`（默认 `auto`）与 `deess`（默认 `true`）保持可见。任何被改到非默认值的项都视为手动指定、不会被自动匹配覆盖；留默认值的项则按素材实测取值。日志会打印实际采用值与保留的节点值（`kept node values: ...`），逐条实测依据以 `auto params → ...` 打印。

**界面与顺序。** `auto_params` 位于节点首位，开启时其后的 `input_sr` / `ode_steps` / `guidance` 三项在界面上隐藏，关闭即恢复手动选择。从旧版本更新本插件后请硬刷新页面（`Ctrl+Shift+R`），否则控件仍按旧顺序渲染，保存值可能对不上号并导致排队校验失败；刷新后旧工作流的 `mode` / `chunk_sec` / `seed` 等设置值都能正确读回。

### 加速选项（`accel`）

各档位在 RTX 5070 Ti 上的实测差异：

| 档位 | 加速 | 与 `fp32` 的最大样本差 | 结论 |
| --- | --- | --- | --- |
| `fp32` | — | — | 与既有听感锁定版本**逐样本一致**，需要复现基准时选它 |
| `cuDNN TF32` | **1.18×** | 3.9e-3（≈ −48 dBFS） | **默认档** |
| `bf16` | 1.35× | 0.26（≈ −11.7 dB，rms −38.6 dB） | 已提供，但有可闻精度损失 |

无 CUDA 的机器上任何非 `fp32` 档都会自动退回 `fp32`；该设置只影响本节点的模型推理，不改动 ComfyUI 的全局 GPU 配置。


- **容器采样率不等于有效带宽。** 48 kHz 文件也可能只包含低带宽内容；反过来，99% 能量集中在低频也不代表其余高频可以丢弃。档位用 `effective = max(rolloff, content cutoff)`，并按 `effective ≤ 5000 Hz` 的护栏跳过向上校正，判据与档位表见“功能概览”。
- 示例日志：`auto input_sr=24000 (base 16000; corrected 24000; persistent high-frequency evidence: 1137 Hz contiguous band, 100% frames, run 96; ...)`。`base` 是原检测档位，`corrected` 是实际采用档位；被 8k 护栏拦下时是 `auto input_sr=8000 (base 8000; upward correction skipped for effective 3152 Hz <= 5000 Hz; ...)`。两者都不表示输出从 16kHz 改为 24kHz；音频输出仍为 **48kHz**。
- **超分后更闷**：先查实际 `input_sr`；再用固定种子、同一短片段对比 `sr` 和 `sr_master`。母带链仍会塑形高频，自动升档不能保证消除它对音色的影响。分带去齿音只在 `deess=true` 且 `mode=sr` + general 时作用，关闭它对比可以确认 5.5–8.5k 的变化。
- **已有完整高频的成品曲**：不要仅因是 MP3 就判为低清。最高 24k 输入档仍可能滤除约 12kHz 以上的原始内容；优先保留原曲，确有母带需求再对比 `master`，而非盲目超分。
- **高频噪感或声场变化**：超分会生成纹理，立体声路径还包含 M/S 处理；先比较仅超分与原曲，不要只靠提高 guidance、步数或音量判断品质。`sr_master`/`sr` 的立体声路径在 14k 以上用去相关噪声补宽度，如听感偏高噪可试 `mode=sr` 直出对比。
- **更新节点代码后需重启 ComfyUI**，并避免复用旧缓存结果；固定种子用于公平比较，修改代码后的首次验证可换一个种子触发重新执行。
- **耗时由 `input_sr` 决定**：8k 档最快、24k 档最贵（60s 素材 `sr_master` + 自动档 24k 约 7–8 分钟，200s 整曲 `master` 因不跑模型通常一分钟内）；长跑期间 GPU 接近满载属正常，不是卡死。
- **缺模型/缺节点**：检查模型缓存、对应 general/speech 权重，以及模板的 rgthree、文本展示和 `SaveAudioAdvanced` 节点；保存节点不可用时升级 ComfyUI 或用当前版本提供的音频保存节点替换。`stem_mix` 报找不到模型时确认 `model_cache` 指向同时含 `general` 与 `speech` 的父目录。

### 音频相关文件

- `musefish_audio.py`：模型准备节点、General/Speech 处理节点、AUDIO 张量适配与隔离 worker 调度。
- `audio_backend/processing.py`：自动带宽检测与 general 校正、分块超分、四种模式的链路与收口。
- `audio_backend/dsp.py`：母带及音频后处理（EQ / 多段压 / 去齿音 / 高频整形 / 限幅 / 带宽清理 / 侧声道去相关）。
- `audio_backend/worker.py`：独立处理进程入口。
- [Musefish_UniverSR_Audio.json](../workflows/Musefish_UniverSR_Audio.json)：音乐/语音双分支参考模板，分别连接模型缓存、音频保存和日志展示节点。
