# Musefish Music Mastering（显示名 `Musefish Music Mastering`）

`musefish_music_mastering` 使用本机 PhaseLimiter 引擎处理 AUDIO；与 Musefish Studio 共用 `server/mastering.py`（运行时从 Studio server 导入，不在插件内复制）。

## 节点与参数

| 项 | 值 |
|---|---|
| ID / 分类 | `musefish_music_mastering` / Musefish/Audio |
| 输入 | `AUDIO`（批次）、`mode`、`level`、`matching`、`loudness_target`、`ceiling`、`format`、`bit_depth`、`filename_prefix` |
| 输出 | `audio`（AUDIO，固定 44.1 kHz 立体声 float）、`path`（STRING，保存文件路径）、`info`（STRING，实测响度摘要）；输出节点保存到 ComfyUI `output/` |
| 批次 | 每段独立处理后堆叠；较短段按最长段补零。输入其他采样率会重采样，多声道只取前两轨。 |

| 参数 | 取值 | 语义 |
|---|---|---|
| `mode` | `mastering5` / `mastering3` / `mastering2` / `classic` | 四种母带算法；5 音质最佳但最慢，classic 最快。 |
| `level` | 0–1，步进 0.25 | 响度提升强度；只对 `mastering5` 生效。 |
| `matching` | 0–1，步进 0.25 | 响度匹配；实测仅 `classic` 改变输出（−8.12 → −9.55 LUFS）。 |
| `loudness_target` | −9 / −14 / −16 / −23 LUFS、`off` | 目标响度；实测误差 <0.1 LU。 |
| `ceiling` | 0 / −1 / −2 dBTP | 真峰上限，默认 −1 dBTP；无法兼顾时优先不超真峰。 |
| `format` | `wav` / `mp3` / `aac` | 落盘格式；MP3 320 kb/s、AAC 256 kb/s 恒定码率。MP3/AAC 由引擎 WAV 转码：引擎自带 mp3/aac 输出绕过响度参考，实测 −6.3 LUFS（目标 −14），故响度统一在 WAV 管线完成。 |
| `bit_depth` | `24` / `16` | 仅 WAV 生效。 |
| `filename_prefix` | 字符串 | 保存到 output，支持子目录及 `%year%-%month%-%day%`。 |

## 进度与实现

`mastering.run(..., on_progress=cb)` 以 `cb(done, total, label)` 回报阶段；`total` = 引擎阶段（9）+ 非 WAV 转码步骤 + 响度测量，`progress_total(params)` 返回同一分母。回调单调；首帧 `(0, total, "启动引擎")`，末帧 `(total, total, "测量响度")`，失败无末帧。节点进度以批次 × `progress_total` 累加；Studio 使用 `job.progress = {pct, label, stage, elapsed}`。

引擎 `mastering5/3/2` 优化器有随机性：相同参数重复输出字节不同，但 LUFS / LRA / 峰均比稳定；A/B 参数判断须重复取样，不能依据单次比较。引擎按 cwd 找 `phaselimiter/resource`，运行时 cwd 固定为安装目录。

## 性能实测

本机 14 物理核 / 20 逻辑核、44.1 kHz 立体声：

| 环节 | 优化前 → 优化后 | 条件/说明 |
|---|---|---|
| 响度测量 | 2.27 s → 0.38 s | 150 s 文件；`loudnorm=print_format=json` → ffmpeg `ebur128=peak=true`，均为 BS.1770-4，WAV/MP3 与旧值差 ≤0.05 LU/dB。 |
| 全链路 | 11.41 s → 8.66 s | 单文件 150 s，mastering5→mp3；引擎 8.4 s 不变，节省测量与复制。 |
| 单段 | 5.29 s → 3.78 s | 节点单个 60 s clip。 |
| 批次 | 19.15 s → 10.66 s | 节点 4 段 × 60 s，mp3。最多 4 段并发（`_MAX_WORKERS = 4`）；单引擎进程瞬时约 16 核、峰值 RSS 304 MB；4 并发 1.44×、2 并发 1.32×，更多因 CPU 争用变慢。 |

并发进度加锁累加已完成阶段格，保持单调。WAV 直接读落盘文件回填张量，省一次 60–100 MB 拷贝；有损格式保留 WAV 中间文件，以无损波形生成张量。

## 安装与依赖

引擎目录按首个含 `phaselimiter/bin/phase_limiter.exe` 的位置解析：`server/config.json` 的 `mastering_dir` → `MUSEFISH_MASTERING_DIR` → `<MusefishStudio>/engines/mastering` → `D:/AI Mastering`（旧目录）。显式配置无效即报错，不回落。缺少 `phaselimiter/resource/sound_quality2_cache` 或 `mastering_reference.json` 会报资源缺失。

依赖：`torch`、`torchaudio`、`soundfile`、`numpy`（ComfyUI 环境）；ffmpeg 转码/测量路径依次为 `config.ffmpeg_dir` → `<root>/ffmpeg/bin` → `PATH`。引擎无 Python 依赖。

## Studio 与排错

Studio（音频处理 → AI Mastering 母带）与节点参数语义、日志阶段、产物命名一致；工作台另有队列、进度、取消、清理。可见参数规则：仅 `classic` 显示 `matching`，仅 `mastering5` 显示 `level`，非 WAV 不显示位深。

| 现象 | 处理 |
|---|---|
| `未找到 AI Mastering（PhaseLimiter）` | 引擎目录不完整，检查 `mastering_dir`。 |
| `AI Mastering 资源缺失` | 检查 `phaselimiter/resource/` 下 `sound_quality2_cache`、`mastering_reference.json`。 |
| `无法解码输入音频` | 转为 WAV/MP3 后输入。 |
| 响度非目标值 | 检查 `loudness_target=off` 或真峰上限（例如 0 dBTP）导致限幅优先。 |
