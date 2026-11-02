# Musefish Music Mastering 节点（节点显示名 `Musefish Music Mastering`）

`musefish_music_mastering` — 用本机 PhaseLimiter 引擎给 AUDIO 批次做母带（响度统一 +
真峰限幅 + 多段动态处理）。引擎驱动与 [Musefish Studio](../../../MusefishStudio)
的 `server/mastering.py` **同源**（节点运行时注入 Studio server 目录后 import，
插件内不维护第二份实现），与工作台「AI Mastering 母带」结果完全一致。

## 节点

| 项 | 值 |
|---|---|
| ID | `musefish_music_mastering` |
| 分类 | Musefish/Audio |
| 输入 | `AUDIO`（批次）、`mode`、`level`、`matching`、`loudness_target`、`ceiling`、`format`、`bit_depth`、`filename_prefix` |
| 输出 | `audio`（AUDIO，44.1 kHz 立体声）、`path`（STRING，落盘文件路径）、`info`（STRING，实测响度摘要） |
| 输出节点 | 是（保存到 ComfyUI `output/` 并可直接执行） |

单个 clip 一进一出；批次逐条处理再堆叠，长度不一致时按最长补齐零。

## 参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `mode` | `mastering5` / `mastering3` / `mastering2` / `classic` | 母带算法。5 最佳音质最慢，classic 最快 |
| `level` | 0 – 1（步进 0.25） | 响度提升强度，**仅 `mastering5` 定义**，其余模式引擎忽略 |
| `matching` | 0 – 1（步进 0.25） | 响度匹配度，**实测只有 `classic` 改变输出**（−8.12 → −9.55 LUFS） |
| `loudness_target` | −9 / −14 / −16 / −23 LUFS、`off` | 交付响度目标；实测误差 <0.1 LU |
| `ceiling` | 0 / −1 / −2 dBTP | 真峰上限；限幅到不了目标时牺牲响度而不超真峰 |
| `format` | `wav` / `mp3` / `aac` | 落盘格式；AUDIO 输出恒为 44.1 kHz 立体声 float |
| `bit_depth` | `24` / `16` | 仅 WAV 生效（MP3 320 kb/s、AAC 256 kb/s 恒定码率） |
| `filename_prefix` | 字符串 | 保存到 ComfyUI output 目录，支持子目录与 `%year%-%month%-%day%` 变量 |

## 进度回调

`mastering.run(..., on_progress=cb)` 逐阶段上报 `cb(done, total, label)`：

- `total` = 引擎阶段数（9）+ 非 WAV 的转码步骤 + 响度测量；`progress_total(params)`
  可直接拿到同一分母，调用方据此换算百分比或进度条刻度。
- 单调不回退：引擎会把 `Normalize`、`CutLowAndHighFreq` 跑多轮，回调只在前进一步时触发。
- 首帧固定 `(0, total, "启动引擎")`，末帧 `(total, total, "测量响度")`；失败时不发末帧。

两个消费方：

| 调用方 | 用法 |
|---|---|
| Studio 任务 | `job.progress = {pct, label, stage, elapsed}`，任务卡片显示「母带处理 50% · 母带优化 · 12s」 |
| ComfyUI 节点 | 进度条刻度 = 批次 × `progress_total`，逐阶段推进；同时用 `send_progress_text` 在节点上打出「[1/2] 母带优化」，同一百分比 0.5 s 内不重复刷 |

## 性能（本机实测：14 物理核 / 20 逻辑核，44.1 kHz 立体声）

| 环节 | 优化前 | 优化后 | 说明 |
|---|---|---|---|
| 响度测量（150 s 文件） | 2.27 s | 0.38 s | `loudnorm=print_format=json` → `ebur128=peak=true`，同为 BS.1770-4，跨 WAV/MP3 与旧值差 ≤0.05 LU/dB |
| 单文件全链路（150 s，mastering5→mp3） | 11.41 s | 8.66 s | 引擎 8.4 s 不变，省下的是测量与复制 |
| 节点单段（60 s） | 5.29 s | 3.78 s | 同上 |
| 节点 4 段批次（60 s，mp3） | 19.15 s | 10.66 s | 批次并发 + 上述两项 |

- **批次并发**：`_MAX_WORKERS = 4`。引擎单进程瞬时占用约 16 核、峰值 RSS 304 MB，
  实测 4 并发 1.44×、2 并发 1.32×；再往上 CPU 争抢反而变慢。内存敏感可下调。
- **进度仍单调**：并发下进度 = 各 clip 已推进格数之和，加锁更新，进度条不回跳。
- **WAV 交付不再复制中间 WAV**：直接读落盘文件回填张量（省一次 60–100 MB 拷贝）；
  有损交付才保留 WAV，因为张量必须来自无损波形。

## 引擎行为（改前先复测）

- **固定 44.1 kHz 立体声**：输入其他采样率在节点侧重采样；多声道取前两轨。
- **MP3/AAC 由引擎出 WAV 后再转码**：引擎自带的 mp3/aac 输出会绕过响度参考
  （实测 −6.3 LUFS 而非目标 −14），所以响度统一恒在 WAV 管线完成。
- **优化器随机**：`mastering5/3/2` 同一命令重复执行输出字节不同，但宏观指标
  （LUFS / LRA / 峰均比）稳定；比较参数效果必须重复取样，不能单次 A/B。
- **引擎相对 cwd 读 `phaselimiter/resource`**：运行时 cwd 固定为安装目录。
- **取消按进程树杀**：引擎会自起 ffmpeg 子进程，单 terminate 会留孤儿。

## 引擎安装位置

解析顺序（首个含 `phaselimiter/bin/phase_limiter.exe` 的生效）：

1. `server/config.json` 的 `mastering_dir`
2. 环境变量 `MUSEFISH_MASTERING_DIR`
3. `<MusefishStudio>/engines/mastering`
4. `D:/AI Mastering`（旧安装目录）

配了但无效会直接报错，不会悄悄回落到旧目录。缺引擎时节点抛错并给出上述提示。

## 依赖

`torch`、`torchaudio`、`soundfile`、`numpy`（ComfyUI 环境已有）；转码与测量用
ffmpeg（`config.ffmpeg_dir` → `<root>/ffmpeg/bin` → `PATH`）。引擎自身不需要
Python 依赖。

## 与 Studio 联动

Studio 工作台（音频处理 → AI Mastering 母带）与节点调用同一 `mastering.py`：
参数语义、日志阶段、产物命名一致。工作台额外提供任务队列、进度条、取消与
一键清理输出；节点的显隐规则（`classic` 才有匹配度、`mastering5` 才有强度、
非 WAV 无位深）与工作台界面一致。

## 排错

| 现象 | 原因 / 处理 |
|---|---|
| `未找到 AI Mastering（PhaseLimiter）` | 引擎目录不完整；按上面顺序配置 `mastering_dir` |
| `AI Mastering 资源缺失` | `phaselimiter/resource/` 下缺 `sound_quality2_cache` 或 `mastering_reference.json` |
| `无法解码输入音频` | 输入不是可解码音频；先转 WAV/MP3 再进节点 |
| 输出响度不是目标值 | `loudness_target=off`，或真峰上限过严（如 0 dBTP）导致限幅优先 |
