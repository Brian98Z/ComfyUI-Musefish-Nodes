#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Musefish Studio 首启环境准备（仅标准库）。

三步全自动，任一步已就绪则跳过：
1. 找基础解释器（内嵌 runtime/python → 系统 python，需 3.11+ 64 位）
2. 建 runtime/venv 并安装依赖：ComfyUI 内核 requirements + 插件 requirements + yt-dlp/psutil
   （PyPI 清华镜像起步，失败自动换源；torch 走 pytorch.org cu130 轮子）
3. 克隆/更新 ComfyUI 内核到 data/comfyui/（git 可用则 clone，否则 pip 装 comfyui-cli 替代）

用法：
  python tools/env_setup.py            # 全流程
  python tools/env_setup.py --check    # 只报告，不执行
  python tools/env_setup.py --mirror pypi
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# 清华主源(大包稳);阿里对部分大 wheel 限速,只作第二回退
PYPI_DEFAULT = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_FALLBACKS = (
    "https://mirrors.aliyun.com/pypi/simple",
    "https://pypi.org/simple",
)
COMFYUI_REPO = "https://github.com/comfyorg/ComfyUI.git"
# 上游不打版本 tag；Pin 到与开发机验证版本 v0.39.0 完全一致的 commit
#（插件 pid_runtime 依赖 comfy.ldm.pixeldit，主线后续版本已移除该模块）。
COMFYUI_TAG = "b0b743566f65daafc423b4fea8a2fbda94b3384a"
PIP_TIMEOUT = "60"       # per-request timeout so a stalled mirror fails fast

for _p in ("MUSEFISH_REQUIREMENTS_COMFY", "MUSEFISH_REQUIREMENTS_PLUGIN"):
    pass  # documentation hook: paths resolved below against ROOT


def root() -> Path:
    override = os.environ.get("MUSEFISH_STUDIO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def venv_python(rt: Path) -> Path:
    return rt / "runtime" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")


def comfy_dir(rt: Path) -> Path:
    return rt / "data" / "comfyui"


def plugin_reqs(rt: Path) -> Path:
    # 源码仓形态: studio/tools 的上一级找 data/custom_nodes; 包内形态: tools 同级 data
    for cand in (
        rt / "data" / "custom_nodes" / "ComfyUI-Musefish-Nodes" / "requirements.txt",
        rt / "requirements-plugin.txt",
    ):
        if cand.is_file():
            return cand
    # 源码仓里插件目录可能尚未复制: 从 comfy junction/内嵌取,最后退回内置清单
    return rt / "tools" / "requirements-plugin.txt"


def find_base_interpreter(rt: Path) -> str | None:
    cands = []
    if os.environ.get("MUSEFISH_PYTHON"):
        cands.append(os.environ["MUSEFISH_PYTHON"])
    cands.append(str(rt / "runtime" / "python" / "python.exe"))
    import shutil

    for name in ("python.exe", "python3.exe", "python", "python3"):
        cands.append(shutil.which(name) or "")
    for c in cands:
        if c and Path(c).is_file() and _is_ok(c):
            # normalize to native backslash form: venv/pip bootstrap goes
            # through cmd.exe, which mis-parses forward-slash absolutes
            return str(Path(c).resolve())
    return None


def _is_ok(path: str) -> bool:
    try:
        r = subprocess.run(
            [path, "-c", "import struct,sys;print(f'{sys.version_info.major}{sys.version_info.minor} {struct.calcsize(\"P\")*8}')"],
            capture_output=True, text=True, timeout=12)
        out = r.stdout.strip()
        major = int(out.split()[0])
        if not (r.returncode == 0 and major >= 311 and out.endswith("64")):
            return False
        # must be able to create venvs (embeddable distributions ship without it)
        r2 = subprocess.run([path, "-c", "import venv"], capture_output=True, timeout=12)
        return r2.returncode == 0
    except Exception:
        return False


def venv_ready(rt: Path) -> bool:
    vp = venv_python(rt)
    if not vp.is_file():
        return False
    try:
        r = subprocess.run([str(vp), "-c", "import torch;import yaml;import safetensors"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def ensure_venv(rt: Path, base: str, check: bool, mirror: str) -> dict:
    vp = venv_python(rt)
    info = {"step": "venv", "python": str(vp)}
    if venv_ready(rt):
        info["state"] = "ready"
        return info
    if check:
        info["state"] = "missing"
        return info
    venv_dir = vp.parent.parent
    info["state"] = "creating"
    if not vp.is_file():
        r_venv = subprocess.run([base, "-m", "venv", str(venv_dir)],
                                capture_output=True, text=True)
        if r_venv.returncode != 0:
            msg = (r_venv.stdout + r_venv.stderr)
            if "No module named venv" in msg:
                raise RuntimeError(
                    "当前 Python 缺少 venv 模块（内嵌解释器需安装；"
                    "系统 Python 请勾选 tcl/venv 组件或改用完整安装包）: " + msg[-300:])
            raise RuntimeError("venv 创建失败: " + msg[-300:])
    mirrors = [PYPI_DEFAULT] if mirror == "auto" else [mirror]
    if mirror == "auto":
        mirrors += [m for m in PYPI_FALLBACKS if m != PYPI_DEFAULT]
    # two-phase install: torch first (2GB wheel, most likely to stall) then the
    # rest — a failure in phase 2 no longer discards the torch download
    phases = [
        ("torch", "torch", [TORCH_INDEX]),
        ("deps", str(rt / "tools" / "requirements-bundle.txt"), mirrors),
        ("plugin", str(plugin_reqs(rt)), mirrors),
    ]
    for phase_name, target, idxs in phases:
        for idx in ([mirror] if mirror != "auto" else idxs):
            cmd = [
                str(vp), "-m", "pip", "install",
                "--index-url", idx,
                "--timeout", PIP_TIMEOUT,
                "--retries", "5",
                target,
            ]
            if phase_name == "deps":
                cmd.insert(3, "--extra-index-url")
                cmd.insert(4, TORCH_INDEX)  # resolve torch pin, never re-download
            info["state"] = f"pip:{phase_name} ({idx.split('/')[2]})"
            print(f"[env_setup] pip install {phase_name} from {idx} …", flush=True)
            r = subprocess.run(cmd)  # stream pip output straight to stdout/stderr
            if r.returncode == 0:
                break
        else:
            raise RuntimeError(f"pip 阶段 {phase_name} 在全部镜像上失败（输出见上方日志）")
    info["state"] = "ready"
    return info


def ensure_comfyui(rt: Path, check: bool) -> dict:
    info = {"step": "comfyui", "dir": str(comfy_dir(rt))}
    target = comfy_dir(rt)
    if (target / "main.py").is_file():
        info["state"] = "ready"
        return info
    if check:
        info["state"] = "missing"
        return info
    target.parent.mkdir(parents=True, exist_ok=True)
    info["state"] = "cloning"
    git = _which_git()
    if git:
        cmd = ["git", "clone", "--depth", "1"]
        if COMFYUI_TAG:
            cmd += ["--branch", COMFYUI_TAG]
        cmd += [COMFYUI_REPO, str(target)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("git clone ComfyUI 失败: " + (r.stdout + r.stderr)[-400:])
    else:
        raise RuntimeError("未找到 git，且当前版本不支持无 git 安装 ComfyUI 内核（请安装 Git 后重试）")
    # cloned core carries its own pinned requirements (frontend version etc.) —
    # install them into the venv so engine and deps are authoritatively aligned
    core_req = target / "requirements.txt"
    if core_req.is_file():
        info["state"] = "pip: core requirements"
        print("[env_setup] pip install ComfyUI core requirements …", flush=True)
        r = subprocess.run([str(venv_python(rt)), "-m", "pip", "install",
                            "--index-url", PYPI_DEFAULT, "--timeout", PIP_TIMEOUT,
                            "-r", str(core_req)])
        if r.returncode != 0:
            raise RuntimeError("ComfyUI core requirements 安装失败（输出见上方日志）")
    info["state"] = "ready"
    return info


def _which_git() -> str | None:
    import shutil
    return shutil.which("git")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    check = "--check" in args
    mirror = "auto"
    if "--mirror" in args:
        mirror = args[args.index("--mirror") + 1]
    rt = root()
    report: dict = {"root": str(rt), "platform": platform.platform()}

    base = find_base_interpreter(rt)
    report["base_interpreter"] = base or "未找到（需 Python 3.11+ x64）"
    if not base:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 3
    if "--no-venv" not in args:
        report["venv"] = ensure_venv(rt, base, check, mirror)
    if "--no-comfyui" not in args:
        report["comfyui"] = ensure_comfyui(rt, check)

    print(json.dumps(report, ensure_ascii=False, indent=1))
    states = [v.get("state") for v in report.values() if isinstance(v, dict)]
    return 0 if all(s in ("ready", "missing") for s in states) and not check else (0 if check else 0)


if __name__ == "__main__":
    sys.exit(main())
