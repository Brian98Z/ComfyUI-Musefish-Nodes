"""AI Mastering（PhaseLimiter）引擎驱动。

唯一实现，两个调用方共用：

- Studio 工作台任务（server/app.py::_run_mastering）
- ComfyUI 插件节点（custom_nodes/ComfyUI-Musefish-Nodes/musefish_music_mastering.py）

引擎行为（实测结论，改动前先复测）：

- 固定 44.1 kHz 立体声；--bit_depth 只决定输出 WAV 位深。
- 引擎自带的 mp3/aac 输出会绕过 --reference 响度统一（实测 -6.3 LUFS 而非目标
  -14），所以引擎永远出 WAV，mp3/aac 由 ffmpeg 从这份 WAV 转码。
- mastering5/3/2 的优化器是随机的：同一命令连续执行输出字节不同，但宏观指标
  （LUFS / LRA / 峰均比）稳定；比较参数效果必须重复取样，不能单次 A/B。
- 实测只有 classic 的 --mastering_matching_level 改变输出响度（-8.12 → -9.55
  LUFS），mastering5/3/2 下该参数无差别；--mastering5_mastering_level 仅
  mastering5 定义。
- 引擎相对 cwd 读取 phaselimiter/resource 与 tmp，运行时 cwd 固定为安装目录。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODES = ("mastering5", "mastering3", "mastering2", "classic")
FORMATS = ("wav", "mp3", "aac")

_MASTERING_ENV = "MUSEFISH_MASTERING_DIR"

# 引擎输出混着 ffmpeg 横幅；只把阶段行搬进日志，并把阶段映射成一句话进度。
# 顺序即引擎实际执行顺序；进度取"已到达的最大序号"，重复出现的阶段（Normalize/
# CutLowAndHighFreq 会跑多轮）不会让进度回退。
_INTERESTING = re.compile(r"(lap:|Mastering |mastering \d|gain:|threshold:|clip |"
                          r"exception|error|Error|quick exit)", re.IGNORECASE)
_STAGES = (
    ("create normalized", "转换采样格式"),
    ("load wave", "读取音频"),
    ("CutLowAndHighFreq", "频段处理"),
    ("Normalize", "响度归一"),
    ("AutoMastering", "母带优化"),
    ("pre-compression", "前置压缩"),
    ("pre-CutLow", "前置频段"),
    ("limiting", "限幅"),
    ("save", "写出文件"),
)
_STAGE_INDEX = {key.lower(): i for i, (key, _label) in enumerate(_STAGES)}
# 非 WAV 交付多一步 ffmpeg 转码，进度分母随之 +1
_TRANSCODE_LABEL = "转码"

# ebur128 Summary 段解析（逐帧表格在上面，不能一起匹配）
_LOUDNESS_I_RE = re.compile(r"I:\s*(-?[\d.]+)\s*LUFS")
_TRUE_PEAK_RE = re.compile(r"True peak:\s*Peak:\s*(-?[\d.]+)")
_PEAK_RE = re.compile(r"Peak:\s*(-?[\d.]+)\s*dBFS")


def _total_steps(p: dict) -> int:
    """单文件进度总量：引擎阶段 + （非 WAV 的）转码 + 响度测量。"""
    return len(_STAGES) + (1 if p["format"] != "wav" else 0) + 1


def progress_total(params: dict) -> int:
    """单文件进度总量，供调用方换算百分比/进度条刻度。"""
    return _total_steps(normalize_params(params))


def _config() -> dict:
    """每次调用重读 config.json：改配置不必重启服务。"""
    try:
        return json.loads((HERE / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def ffmpeg_bin() -> Path | None:
    """config.ffmpeg_dir → <root>/ffmpeg/bin → PATH 兜底。"""
    cfg = _config()
    candidates = [Path(cfg["ffmpeg_dir"]) if cfg.get("ffmpeg_dir") else None,
                  ROOT / "ffmpeg" / "bin"]
    for base in candidates:
        if base and (base / "ffmpeg.exe").is_file():
            return base / "ffmpeg.exe"
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def install() -> dict:
    """Locate the PhaseLimiter install and its required resource files.

    显式配置（config.mastering_dir 或环境变量 MUSEFISH_MASTERING_DIR）具有排他性：
    配了但无效就报错，不再悄悄回落到自动探测 —— 否则用户改了安装位置仍会命中旧目录。
    只有完全未配置时才探测 <root>/engines/mastering 与旧安装目录 D:/AI Mastering。
    """
    cfg = _config()
    configured = str(cfg.get("mastering_dir") or os.environ.get(_MASTERING_ENV) or "").strip()
    candidates = ([Path(configured)] if configured else
                  [ROOT / "engines" / "mastering", Path("D:/AI Mastering")])
    for base in candidates:
        exe = base / "phaselimiter" / "bin" / "phase_limiter.exe"
        if not exe.is_file():
            continue
        resource = base / "phaselimiter" / "resource"
        gui = base / "phaselimiter-gui.exe"
        return {
            "ok": True,
            "dir": str(base),
            "exe": str(exe),
            "cache": str(resource / "sound_quality2_cache"),
            "reference": str(resource / "mastering_reference.json"),
            # 精简形态只带引擎，GUI 前端不一定在
            "gui": str(gui) if gui.is_file() else "",
        }
    hint = (f"mastering_dir 已指向 {configured}，但其中没有 phaselimiter/bin/phase_limiter.exe"
            f"（引擎文件缺失或被移走）"
            if configured else
            f"engines/mastering 缺失：请确认该目录随仓库完整拷贝，或在 server/config.json 的 "
            f"mastering_dir / 环境变量 {_MASTERING_ENV} 指定引擎目录")
    return {"ok": False, "dir": "", "exe": "", "cache": "", "reference": "", "gui": "",
            "searched": [str(c) for c in candidates], "hint": hint}


def db_param(value, default: float | None, lo: float, hi: float) -> float | None:
    """Parse a dB parameter that UIs label with its unit ("-14 LUFS", "-1 dBTP").

    "off"/"关闭"/"不统一"/"不设限"/empty → None (feature disabled).
    Anything unparsable → default.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(lo, min(hi, float(value)))
    text = str(value).strip()
    lowered = text.lower()
    if (not text or lowered.startswith(("off", "none"))
            or text.startswith(("关闭", "不统一", "不设限"))):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        return max(lo, min(hi, float(match.group())))
    except ValueError:
        return default


def _unit(value, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_params(params: dict) -> dict:
    """把 UI/节点参数收敛成引擎参数（越界与非法值回落到默认）。"""
    params = params or {}
    mode = str(params.get("mastering_mode") or "mastering5").strip().lower()
    if mode not in MODES:
        mode = "mastering5"
    fmt = str(params.get("output_format") or "wav").strip().lower()
    if fmt not in FORMATS:
        fmt = "wav"
    bit_depth = 24 if int(_unit(params.get("bit_depth", 16), 16, 16, 24)) == 24 else 16
    return {
        "mode": mode,
        "format": fmt,
        "bit_depth": bit_depth,
        "level": _unit(params.get("mastering5_mastering_level"), 0.5, 0.0, 1.0),
        "matching": _unit(params.get("mastering_matching_level"), 1.0, 0.0, 1.0),
        "loudness_target": db_param(params.get("loudness_target"), -14.0, -40.0, -5.0),
        "ceiling": db_param(params.get("ceiling"), -1.0, -6.0, 0.0) or 0.0,
    }


def build_command(inst: dict, src: Path, dest_wav: Path, p: dict, engine_tmp: Path,
                  ffmpeg: Path | None) -> list[str]:
    """引擎命令行。响度统一走 --reference loudness，真峰上限单独给
    （limiter 到不了目标时牺牲响度而不是超真峰）。"""
    cmd = [
        inst["exe"],
        "--input", str(src),
        "--output", str(dest_wav),
        "--mastering",
        "--mastering_mode", p["mode"],
        "--mastering5_mastering_level", f"{p['level']:g}",
        "--mastering_matching_level", f"{p['matching']:g}",
        "--output_format", "wav",
        "--bit_depth", str(p["bit_depth"] if p["format"] == "wav" else 24),
        "--sound_quality2_cache", inst["cache"],
        "--mastering_reference_file", inst["reference"],
        "--ceiling", f"{p['ceiling']:g}",
        "--ceiling_mode", "true_peak",
    ]
    if p["loudness_target"] is not None:
        cmd += ["--reference", f"{p['loudness_target']:g}", "--reference_mode", "loudness"]
    if ffmpeg is not None:
        # 引擎自己调 ffmpeg 解码/封装；显式给绝对路径，免依赖 PATH
        cmd += ["--ffmpeg", str(ffmpeg)]
    # 引擎会先写未压缩中间波形再改名到 --output；临时目录按任务隔离，
    # 转码完成后整体删除——否则引擎目录里会留下一份与输入等大的 wav。
    cmd += ["--tmp", str(engine_tmp), "--quick_exit"]
    return cmd


def describe(p: dict) -> str:
    """一行参数摘要（两份日志共用同一措辞）。"""
    target = ("引擎默认" if p["loudness_target"] is None
              else f"{p['loudness_target']:g} LUFS")
    return (f"{p['mode']} · {p['format']}/{p['bit_depth']}bit · "
            f"level={p['level']:g} matching={p['matching']:g} · "
            f"目标 {target} · 真峰上限 {p['ceiling']:g} dBTP")


def measure_loudness(path: Path, ffmpeg: Path | None = None) -> tuple[float | None, float | None]:
    """Integrated loudness (LUFS) and true peak (dBTP) of a rendered file.

    Uses ffmpeg's `ebur128` filter instead of `loudnorm=print_format=json`:
    same ITU-R BS.1770-4 measurement, but 6.6× faster on the same file
    (150 s stereo: 0.34 s vs 2.27 s — loudnorm's gating pass dominates the
    profile). Cross-measured on WAV + MP3, differences stay within ±0.05 LU/dB
    of loudnorm, which is below the 0.1 unit the UI prints.

    One decode pass; the numbers are reported next to the result so the user
    sees the loudness actually delivered instead of trusting flags.
    """
    ffmpeg = ffmpeg or ffmpeg_bin()
    if ffmpeg is None or not Path(ffmpeg).is_file():
        return None, None
    try:
        out = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-nostats", "-i", str(path),
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    blob = out.stderr or ""
    # 只认最后一段 Summary：上面是逐帧滚动表，Summary 才是整段结果
    summary = blob[blob.rfind("Summary:"):] if "Summary:" in blob else blob
    try:
        lufs = float(_LOUDNESS_I_RE.search(summary).group(1))
        peak_match = _TRUE_PEAK_RE.search(summary) or _PEAK_RE.search(summary)
        peak = float(peak_match.group(1)) if peak_match else -70.0
    except (AttributeError, ValueError, TypeError):
        return None, None
    return (lufs if lufs > -70 else None), (peak if peak > -70 else None)


def encode_audio(ffmpeg: Path, src: Path, dst: Path, fmt: str) -> None:
    """Transcode the mastered WAV into mp3/aac.

    The engine's own --output_format mp3/aac skips the loudness reference
    (measured: -6.3 LUFS instead of the requested -14), so loudness unification
    always happens on the WAV pipeline and the delivery encode comes after it.
    """
    args = ["-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn"]
    if fmt == "mp3":
        args += ["-c:a", "libmp3lame", "-b:a", "320k"]
    else:  # aac
        args += ["-c:a", "aac", "-b:a", "256k"]
    try:
        proc = subprocess.run([str(ffmpeg), *args, str(dst)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"转码 {fmt} 失败：{exc}") from exc
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        detail = ((proc.stderr or "").strip().splitlines() or ["无输出"])[-1][:300]
        raise RuntimeError(f"转码 {fmt} 失败：{detail}")


def _kill_tree(proc: subprocess.Popen, log=None) -> None:
    """引擎自己会起 ffmpeg 子进程：单个 terminate 只会杀掉引擎、留下 ffmpeg，
    所以按进程树杀。"""
    try:
        import proc_tree  # 与 app.py 同目录
        proc_tree.kill_tree(proc, timeout=10, log=log)
    except Exception:  # noqa: BLE001 - proc_tree 缺失时退回单进程终止
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _join_job(proc: subprocess.Popen, log=None) -> None:
    """挂到 kill-on-close job：父进程意外退出时引擎子进程也不会遗留。"""
    try:
        import proc_tree
        proc_tree.join(proc)
    except Exception:  # noqa: BLE001
        pass


def run(src: Path, dest: Path, params: dict, *, workdir: Path,
        log=None, on_progress=None, is_cancelled=None, keep_wav: Path | None = None) -> dict:
    """Run the mastering engine on one file.

    src       输入音频（引擎自己用 ffmpeg 解码，任意格式）
    dest      交付文件（按 params.output_format 决定扩展名/封装）
    workdir   本次任务的临时目录（内部 td/tmp 会在结束时删除）
    keep_wav  可选：mp3/aac 交付时额外保留一份引擎原始 WAV（节点据它回填
              AUDIO 张量，避免再解码一遍有损文件）

    log(line)                      日志回调
    on_progress(done, total, label) 进度回调：引擎阶段推进 + 转码步骤，单调不回退，
                                    total 由 progress_total(params) 给出；开始时先发
                                    (0, total, "启动引擎")，结束时发 (total, total, "完成")
    is_cancelled() → bool          取消检查

    返回 dict：{cancelled, output, mode, format, bit_depth, level, matching,
    loudness_target, ceiling, lufs, true_peak, seconds}；失败抛 RuntimeError。
    """
    inst = install()
    if not inst["ok"]:
        raise RuntimeError("未找到 AI Mastering（PhaseLimiter）：" + inst["hint"])
    missing = [p for p in (inst["cache"], inst["reference"]) if not Path(p).is_file()]
    if missing:
        raise RuntimeError("AI Mastering 资源缺失：" + "；".join(missing) + "（引擎目录不完整）")

    src = Path(src)
    if not src.is_file():
        raise RuntimeError(f"输入文件不存在：{src}")
    p = normalize_params(params)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    engine_tmp = workdir / "tmp"
    engine_tmp.mkdir(parents=True, exist_ok=True)

    ffmpeg = ffmpeg_bin()  # 引擎解码输入与转码都靠它，缺失时引擎会退回 PATH
    wav_out = dest if p["format"] == "wav" else engine_tmp / f"{src.stem}_mastered.wav"
    cmd = build_command(inst, src, wav_out, p, engine_tmp, ffmpeg)

    started = time.time()
    total_steps = _total_steps(p)
    transcode_step = len(_STAGES) + 1 if p["format"] != "wav" else None
    reached = {"step": 0, "label": ""}

    def emit(step: int, label: str) -> None:
        """阶段推进只前进不后退（引擎会把 Normalize 等阶段跑多轮）。"""
        if step < reached["step"]:
            return
        reached["step"], reached["label"] = step, label
        if on_progress:
            on_progress(step, total_steps, label)

    emit(0, "启动引擎")
    if log:
        log(f"AI Mastering 开始：{src.name} · {describe(p)}")
    proc = subprocess.Popen(
        cmd, cwd=inst["dir"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _join_job(proc, log)

    tail: list[str] = []  # 最后几行原文：失败时直接进错误信息，不必翻控制台

    def pump() -> None:
        for raw in proc.stdout or []:
            line = (raw or "").strip()
            if not line:
                continue
            if _INTERESTING.search(line):
                if log:
                    log(f"mastering: {line[:200]}")
                tail.append(line[:200])
                del tail[:-6]
                lowered = line.lower()
                for key, label in _STAGES:
                    if key.lower() in lowered:
                        emit(_STAGE_INDEX[key.lower()] + 1, label)
                        break

    def cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    reader = threading.Thread(target=pump, daemon=True,
                              name=f"mastering-{os.getpid()}")
    reader.start()
    code = None
    while True:
        try:
            code = proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if cancelled():
                _kill_tree(proc, log)
                reader.join(timeout=2)
                shutil.rmtree(workdir, ignore_errors=True)
                return {"cancelled": True, "output": ""}
    reader.join(timeout=5)
    if code != 0:
        shutil.rmtree(engine_tmp, ignore_errors=True)
        detail = " | ".join(tail[-4:]) or "无输出"
        if re.search(r"(Error opening input|Invalid data found|unsuccessful)", detail, re.I):
            raise RuntimeError(
                f"无法解码输入音频（请确认是有效音频，或先转成 WAV/MP3）：{detail}")
        raise RuntimeError(f"PhaseLimiter 退出码 {code}：{detail}")
    if not wav_out.is_file() or wav_out.stat().st_size == 0:
        shutil.rmtree(engine_tmp, ignore_errors=True)
        raise RuntimeError("PhaseLimiter 未生成输出文件（输入可能不是可解码的音频）")

    try:
        if keep_wav is not None and Path(keep_wav) != wav_out:
            Path(keep_wav).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav_out, keep_wav)
        if p["format"] != "wav":
            if ffmpeg is None or not Path(ffmpeg).is_file():
                raise RuntimeError("导出 mp3/aac 需要 ffmpeg：请检查 config.ffmpeg_dir 或 PATH")
            emit(transcode_step, f"{_TRANSCODE_LABEL} {p['format']}")
            encode_audio(ffmpeg, wav_out, dest, p["format"])
    finally:
        shutil.rmtree(engine_tmp, ignore_errors=True)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"未生成 {p['format']} 输出文件")
    if cancelled():
        shutil.rmtree(workdir, ignore_errors=True)
        return {"cancelled": True, "output": ""}

    emit(total_steps, "测量响度")
    lufs, true_peak = measure_loudness(dest, ffmpeg)
    return {
        "cancelled": False,
        "output": str(dest),
        "bytes": dest.stat().st_size,
        "seconds": round(time.time() - started, 1),
        "mode": p["mode"],
        "format": p["format"],
        "bit_depth": p["bit_depth"],
        "level": p["level"],
        "matching": p["matching"],
        "loudness_target": p["loudness_target"],
        "ceiling": p["ceiling"],
        "lufs": round(lufs, 1) if lufs is not None else None,
        "true_peak": round(true_peak, 1) if true_peak is not None else None,
    }


def loudness_summary(info: dict) -> str:
    """' · -14.1 LUFS / 真峰 -9.5 dBTP'（无测量结果时为空串）。"""
    if info.get("lufs") is None:
        return ""
    text = f" · {info['lufs']:.1f} LUFS"
    if info.get("true_peak") is not None:
        text += f" / 真峰 {info['true_peak']:.1f} dBTP"
    return text
