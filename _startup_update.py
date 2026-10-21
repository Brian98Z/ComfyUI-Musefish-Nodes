"""yt-dlp freshness check — reports only, never installs.

YouTube/Douyin break extractors constantly, and a stale yt-dlp is the most
common cause of "this worked yesterday" failures. Earlier versions installed
``yt-dlp -U`` from a fixed mirror in a background thread on every startup,
which mutated the ComfyUI interpreter behind the user's back, made runs
irreproducible, and depended on one third-party index. The check now prints a
single hint (at most once a week) and leaves the environment alone; update
deliberately with the interpreter ComfyUI actually runs:

    <comfy python> -m pip install -U yt-dlp

The downloader itself upgrades yt-dlp automatically when a site stops
parsing (``musefish_video_download._auto_update_ytdlp``), so this startup
check is only a reminder for installs nobody has exercised yet.
"""

import json
import sys
import time
from pathlib import Path

_STAMP = Path(__file__).with_name("_ytdlp_update_stamp.json")
_WARN_INTERVAL = 7 * 24 * 3600
_MIN_AGE_DAYS = 30


def _installed_version() -> str:
    try:
        from importlib.metadata import version

        return version("yt-dlp")
    except Exception:  # yt-dlp missing is reported by the nodes themselves
        return ""


def _installed_at() -> float:
    """Epoch seconds of the installed yt-dlp dist-info, 0 when unknown."""
    try:
        from importlib.metadata import distribution

        path = getattr(distribution("yt-dlp"), "_path", None)
        return path.stat().st_mtime if path is not None else 0.0
    except Exception:
        return 0.0


def _check() -> None:
    version = _installed_version()
    if not version:
        return
    now = time.time()
    try:
        stamp = json.loads(_STAMP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if now - float(stamp.get("warned", 0) or 0) < _WARN_INTERVAL:
        return
    installed_at = _installed_at()
    if not installed_at or (now - installed_at) < _MIN_AGE_DAYS * 86400:
        return  # freshly installed: nothing to warn about
    try:
        _STAMP.write_text(json.dumps({"warned": now, "version": version}), encoding="utf-8")
    except OSError:
        return
    print(
        f"[Musefish] yt-dlp {version} has not been updated for "
        f"{int((now - installed_at) / 86400)} days. If a site stops working, "
        f"update deliberately: {sys.executable} -m pip install -U yt-dlp",
        flush=True,
    )


try:
    _check()
except Exception:  # never block node registration on a diagnostic
    pass
