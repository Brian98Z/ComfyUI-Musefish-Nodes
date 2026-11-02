"""Musefish Music Mastering node (id `musefish_music_mastering`): PhaseLimiter 母带处理 inside ComfyUI.

Reuses the exact engine driver written for Musefish Studio (server/mastering.py)
so the ComfyUI node and the Studio workbench share one implementation, one
engine install (`engines/mastering`) and one behaviour — same 44.1 kHz stereo
chain, same loudness reference, same WAV-then-transcode delivery. The plugin
tree does NOT vendor a second copy: the module is loaded from the Studio
checkout via the MUSEFISH_STUDIO_SERVER env var, falling back to a junction
relative ../../server (the studio/ bundle inside this repo).

Inputs are Comfy AUDIO batches; every clip is written to a temp WAV (44.1 kHz
stereo float), mastered, saved to the output folder, and returned as a new
AUDIO batch at 44.1 kHz stereo plus the saved path and measured loudness.

实测参数语义（与 Studio 工作台一致，改前先复测）：
- `mastering5/3/2` 的优化器随机：同参重复执行字节不同，宏观指标稳定。
- `level` 仅 `mastering5` 定义；`matching` 实测只在 `classic` 改变输出响度。
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import comfy.utils
from comfy_api.latest import io


def _import_studio_mastering():
    """Locate the Studio server tree and import its mastering module (single
    source of truth for engine location, CLI, loudness measurement, transcode).

    Order matters: mastering.install() derives the engine root from the module's
    own directory, so a bare bundle copy would look for
    ``<plugin>/studio/engines/mastering`` first. The dev checkout comes before the
    bundle so a developer machine keeps using the install the Studio workbench
    itself uses; a distributed install has no checkout and falls through.
    """
    candidates = []
    env = os.environ.get("MUSEFISH_STUDIO_SERVER")
    if env:
        candidates.append(Path(env))
    # dev checkout next to the plugin's sibling tree
    candidates.append(Path(r"D:\MusefishStudio\server"))
    # studio/ bundle inside this plugin: <plugin>/studio/server
    candidates.append(Path(__file__).resolve().parent / "studio" / "server")
    for base in candidates:
        if (base / "mastering.py").is_file():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            import mastering  # noqa: PLC0415
            return mastering
    raise RuntimeError(
        "Musefish Mastering: 未找到 Studio server 目录（mastering.py）。"
        "设置环境变量 MUSEFISH_STUDIO_SERVER 指向 MusefishStudio/server 后重启 ComfyUI。")


_mastering = _import_studio_mastering()

_ENGINE_RATE = 44100
# 并发母带的进程数上限。引擎内部已能吃满十几个核（实测瞬时 16 核 / 峰值 RSS
# 304 MB），所以只做小并发：4 段批次实测 1.46× 加速，再往上 CPU 争抢反而变慢。
_MAX_WORKERS = 4
_MODES = [
    "mastering5（母带5 · 最佳音质·最慢）",
    "mastering3（母带3 · 均衡）",
    "mastering2（母带2 · 快）",
    "classic（经典）",
]
_LOUDNESS = [
    "-14 LUFS（流媒体）",
    "-16 LUFS（播客）",
    "-23 LUFS（广播）",
    "-9 LUFS（高响度）",
    "off（不统一）",
]
_CEILING = ["-1 dBTP（推荐）", "-2 dBTP（保守）", "0 dBTP（不设限）"]
_FORMATS = ["wav", "mp3", "aac"]
_BIT_DEPTHS = ["24", "16"]


def _mode_value(label: str) -> str:
    return str(label).split("（", 1)[0].strip() or "mastering5"


def _output_dir() -> Path:
    """ComfyUI output folder (fallback: temp dir when imported outside ComfyUI)."""
    try:
        import folder_paths  # noqa: PLC0415
        return Path(folder_paths.get_output_directory())
    except Exception:  # noqa: BLE001
        return Path(tempfile.gettempdir()) / "musefish_music_mastering_output"


def _save_base(filename_prefix: str, out_dir: Path):
    """Resolve (folder, subfolder, stem, counter) the way ComfyUI's save nodes do."""
    try:
        import folder_paths  # noqa: PLC0415
        full, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(out_dir))
        return Path(full), subfolder, stem, counter
    except Exception:  # noqa: BLE001 - bare import outside ComfyUI
        subfolder = os.path.dirname(filename_prefix)
        stem = os.path.basename(filename_prefix)
        folder = out_dir / subfolder if subfolder else out_dir
        return folder, subfolder, stem, 0


def _write_engine_input(waveform: torch.Tensor, sample_rate: int, path: Path) -> None:
    """[C, T] float tensor → 44.1 kHz stereo float WAV (lossless handoff)."""
    import numpy as np
    import soundfile as sf

    data = waveform.detach().to(device="cpu", dtype=torch.float32)  # [C, T]
    if data.shape[0] == 1:
        data = data.repeat(2, 1)
    elif data.shape[0] > 2:
        data = data[:2]
    if sample_rate != _ENGINE_RATE:
        import torchaudio.functional as AF  # noqa: PLC0415
        data = AF.resample(data, sample_rate, _ENGINE_RATE)
    sf.write(str(path), data.numpy().T, _ENGINE_RATE, subtype="FLOAT")


def _read_wav(path: Path) -> torch.Tensor:
    """WAV file → [C, T] float32 tensor (engine output is always 44.1 kHz)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    if rate != _ENGINE_RATE:
        import torchaudio.functional as AF  # noqa: PLC0415
        data = AF.resample(torch.from_numpy(data.T), rate, _ENGINE_RATE).numpy().T
    return torch.from_numpy(data.T.copy())


class _ProgressText:
    """节点面板上的实时状态行（ComfyUI `send_progress_text`）。

    数字进度走 ProgressBar；这个通道只发文本（第几段 + 引擎阶段）。两者并存：
    进度条给百分比，文本给当前在做什么。通道缺失（老前端/无头执行）时静默，
    绝不因为显示失败而中断母带。
    """

    _MIN_INTERVAL = 0.5  # 秒：同一百分比重复时不刷屏

    def __init__(self, node_cls, batch: int = 1) -> None:
        self._node_id = getattr(getattr(node_cls, "hidden", None), "unique_id", None)
        self._batch = max(1, int(batch))
        self._last_at = 0.0
        self._last_pct = -1

    def push(self, done: int, total: int, text: str) -> None:
        if not self._node_id:
            return
        pct = int(done * 100 / max(total, 1))
        now = time.monotonic()
        if pct == self._last_pct and now - self._last_at < self._MIN_INTERVAL:
            return
        self._last_at, self._last_pct = now, pct
        try:
            from server import PromptServer  # noqa: PLC0415 - 仅 ComfyUI 进程内可用
            PromptServer.instance.send_progress_text(f"{pct}% {text}", self._node_id)
        except Exception:  # noqa: BLE001
            pass


class MusefishMusicMastering(io.ComfyNode):
    """母带处理：AUDIO → 母带后的 AUDIO + 落盘文件（WAV/MP3/AAC）。

    与 Musefish Studio「AI Mastering」共用同一引擎驱动（server/mastering.py）：
    固定 44.1 kHz 立体声；响度统一走引擎参考电平（实测误差 <0.1 LU），真峰上限
    单独给；MP3/AAC 由引擎出 WAV 后再转码（引擎自带的 mp3/aac 会绕过响度参考）。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="musefish_music_mastering",
            display_name="Musefish Music Mastering",
            search_aliases=["music mastering", "mastering", "母带", "音乐母带",
                            "混音", "loudness", "响度", "phaselimiter", "限幅",
                            "limiter", "响度统一"],
            category="Musefish/Audio",
            is_output_node=True,
            description=(
                "Master AUDIO batches with the local PhaseLimiter engine. "
                "Fixed 44.1 kHz stereo chain; loudness unification uses the "
                "engine reference (measured within 0.1 LU) plus a separate "
                "true-peak ceiling. MP3/AAC are transcoded from the engine WAV. "
                "Shares engine code with Musefish Studio (server/mastering.py)."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="任意长度的 AUDIO 批次；每条 clip 单独母带（固定 44.1 kHz 立体声）。",
                ),
                io.Combo.Input(
                    "mode",
                    options=_MODES,
                    default=_MODES[0],
                    tooltip=(
                        "mastering5=最佳音质最慢（优化器随机，同参重跑字节不同但指标稳定）；"
                        "classic=经典，响度匹配度在此模式下才真正生效。"
                    ),
                ),
                io.Float.Input(
                    "level",
                    default=0.5, min=0.0, max=1.0, step=0.25,
                    tooltip="响度提升强度：仅 mastering5 生效（其余模式引擎忽略）。",
                ),
                io.Float.Input(
                    "matching",
                    default=1.0, min=0.0, max=1.0, step=0.25,
                    tooltip=(
                        "响度匹配度：实测只在 classic 改变输出（−8.12 → −9.55 LUFS），"
                        "mastering5/3/2 下输出无差别。"
                    ),
                ),
                io.Combo.Input(
                    "loudness_target",
                    options=_LOUDNESS,
                    default=_LOUDNESS[0],
                    tooltip="交付响度目标；off=不统一（保留引擎默认电平）。",
                ),
                io.Combo.Input(
                    "ceiling",
                    options=_CEILING,
                    default=_CEILING[0],
                    tooltip="真峰上限（true peak）；限幅到不了目标时牺牲响度而不超真峰。",
                ),
                io.Combo.Input(
                    "format",
                    options=_FORMATS,
                    default="wav",
                    tooltip="落盘格式；AUDIO 输出始终是 44.1 kHz 立体声 float。",
                ),
                io.Combo.Input(
                    "bit_depth",
                    options=_BIT_DEPTHS,
                    default="24",
                    tooltip="仅 WAV 生效（MP3/AAC 为恒定码率）。",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="audio/musefish_mastered",
                    tooltip="保存到 ComfyUI output 目录；可用子目录前缀，支持 %year%-%month%-%day% 等变量。",
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Audio.Output("audio", tooltip="母带后的批次（44.1 kHz 立体声）。"),
                io.String.Output("path", tooltip="落盘文件绝对路径（多 clip 时取最后一个）。"),
                io.String.Output("info", tooltip="实测响度与引擎参数摘要。"),
            ],
        )

    @classmethod
    def execute(cls, audio: dict, mode: str = _MODES[0], level: float = 0.5,
                matching: float = 1.0, loudness_target: str = _LOUDNESS[0],
                ceiling: str = _CEILING[0], format: str = "wav",
                bit_depth: str = "24",
                filename_prefix: str = "audio/musefish_mastered") -> io.NodeOutput:
        inst = _mastering.install()
        if not inst["ok"]:
            raise RuntimeError("Musefish Music Mastering: " + inst["hint"])
        if audio is None:
            raise ValueError("Musefish Music Mastering: input audio is None")

        waveform = audio["waveform"]      # [B, C, T] float32
        sample_rate = int(audio["sample_rate"])
        batch = int(waveform.shape[0])
        if batch < 1:
            raise ValueError("AUDIO 批次为空")

        fmt = format if format in _FORMATS else "wav"
        params = {
            "mastering_mode": _mode_value(mode),
            "mastering5_mastering_level": level,
            "mastering_matching_level": matching,
            "loudness_target": loudness_target,
            "ceiling": ceiling,
            "output_format": fmt,
            "bit_depth": int(bit_depth) if str(bit_depth).isdigit() else 24,
        }

        workdir = Path(tempfile.mkdtemp(prefix="musefish_music_mastering_"))
        # 进度刻度 = 批次 × 单文件阶段数（含转码步骤），阶段推进时进度条连续走
        stage_total = max(1, int(_mastering.progress_total(params)))
        bar = comfy.utils.ProgressBar(batch * stage_total)
        reporter = _ProgressText(cls, batch)
        out_dir = _output_dir()
        folder, subfolder, stem, counter = _save_base(filename_prefix, out_dir)
        folder.mkdir(parents=True, exist_ok=True)

        def should_stop() -> bool:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
            except Exception:  # noqa: BLE001
                return True
            return False

        print(f"[MusefishMusicMastering] {params['mastering_mode']} | {fmt} | {batch} clip(s)",
              flush=True)
        workers = min(batch, _MAX_WORKERS)
        # 进度：每个 clip 占 stage_total 格；全局值 = 各 clip 已推进格数之和（线程间
        # 单调，进度条不会因为并发回跳）。引擎单进程峰值 RSS 304 MB，故并发上限保守。
        steps = [0] * batch
        clock = threading.Lock()

        def advance(index: int, step: int, label: str) -> None:
            with clock:
                step = max(0, min(int(step), stage_total))
                if step <= steps[index]:
                    return
                steps[index] = step
                done = sum(steps)
                bar.update_absolute(done)
                reporter.push(done, batch * stage_total, f"[{index + 1}/{batch}] {label}")

        def master_clip(i: int):
            src_wav = workdir / f"in_{i:04d}.wav"
            _write_engine_input(waveform[i], sample_rate, src_wav)
            name = f"{stem}_{counter + i:05}.{fmt}"
            dest = folder / name
            # WAV 交付直接读落盘文件；有损交付才额外留中间 WAV 给张量回填
            keep_wav = (workdir / f"mastered_{i:04d}.wav") if fmt != "wav" else None

            def on_progress(done: int, total: int, label: str) -> None:
                advance(i, done, label)

            info = _mastering.run(
                src_wav, dest, params, workdir=workdir / f"job_{i:04d}",
                log=lambda line: print(f"[MusefishMusicMastering] {line}", flush=True),
                on_progress=on_progress,
                is_cancelled=should_stop,
                keep_wav=keep_wav,
            )
            if info.get("cancelled"):
                raise KeyboardInterrupt("取消：母带任务已中断")
            tensor_src = keep_wav if (keep_wav and keep_wav.is_file()) else dest
            if tensor_src.suffix.lower() != ".wav":
                # 回填 AUDIO 必须有无损波形：mp3/aac 交付时保留的中间 WAV 才是源
                raise RuntimeError(
                    f"Musefish Music Mastering: 缺少母带中间 WAV（{tensor_src}），无法回填 AUDIO")
            tensor = _read_wav(tensor_src)
            advance(i, stage_total, f"完成 {name}")
            return tensor, dest, info

        results: list = [None] * batch
        if workers == 1:
            for i in range(batch):
                if should_stop():
                    raise KeyboardInterrupt
                results[i] = master_clip(i)
        else:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="musefish-mastering") as pool:
                futures = {pool.submit(master_clip, i): i for i in range(batch)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    if should_stop():
                        for other in futures:
                            other.cancel()
                        raise KeyboardInterrupt
                    results[idx] = fut.result()

        clips = [item[0] for item in results]
        saved = [item[1] for item in results]
        infos = [item[2] for item in results]

        lengths = {int(c.shape[-1]) for c in clips}
        if len(lengths) > 1:
            target = max(lengths)
            clips = [torch.nn.functional.pad(c, (0, target - c.shape[-1])) for c in clips]
        out_waveform = torch.stack(clips, dim=0)  # [B, C, T]

        first = infos[0]
        summary = (f"{first['mode']} · {first['format']}/{first['bit_depth']}bit · "
                   f"目标 {first['loudness_target']} LUFS · 真峰上限 {first['ceiling']} dBTP"
                   + _mastering.loudness_summary(first))
        print(f"[MusefishMusicMastering] done: {saved[-1]} | {summary}", flush=True)
        ui_audio = [{"filename": p.name, "subfolder": subfolder, "type": "output"}
                    for p in saved]
        return io.NodeOutput(
            {"waveform": out_waveform, "sample_rate": _ENGINE_RATE},
            str(saved[-1]), summary,
            ui={"audio": ui_audio, "text": [summary]},
        )
