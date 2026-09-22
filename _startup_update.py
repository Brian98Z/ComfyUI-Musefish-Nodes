"""Background yt-dlp self-update.

YouTube/Douyin break extractors constantly; a stale yt-dlp is the #1 cause
of "this worked yesterday" failures. ComfyUI imports this package at
startup, so the update runs off the import path in a daemon thread (never
blocks node registration) at most once every 24 h.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_STAMP = Path(__file__).with_name("_ytdlp_update_stamp.json")
_INTERVAL = 24 * 3600  # check once a day
_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def _should_update() -> bool:
    try:
        last = json.loads(_STAMP.read_text(encoding="utf-8")).get("ts", 0)
    except (OSError, ValueError):
        last = 0
    return (time.time() - last) > _INTERVAL


def _mark_updated() -> None:
    try:
        _STAMP.write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
    except OSError:
        pass


def _update() -> None:
    if not _should_update():
        return
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "-i", _MIRROR],
            capture_output=True, text=True, timeout=600,
        )
        _mark_updated()
        if result.returncode == 0 and "Successfully installed" in (result.stdout or ""):
            print(
                "[Musefish] yt-dlp updated at startup; restart ComfyUI (or "
                "ignore — the new version loads next boot).",
                flush=True,
            )
    except Exception as exc:  # never block startup on update problems
        print(f"[Musefish] yt-dlp startup update skipped: {exc}", flush=True)


threading.Thread(target=_update, name="musefish-ytdlp-update", daemon=True).start()
