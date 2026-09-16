"""JSON-lines CLI worker for the portable UniverSR backend.

Usage::

    python audio_backend/worker.py --request request.json

Only protocol objects are written to stdout. Diagnostic/model output is sent to
stderr, so the Comfy node can poll stdout without a PIPE deadlock.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import traceback
from pathlib import Path
from threading import Event

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio_backend.processing import process_request
else:
    from .processing import process_request


_CANCEL = Event()
_PROTOCOL_OUT = sys.stdout


def _handle_signal(signum, _frame) -> None:
    _CANCEL.set()
    print(f"worker received signal {signum}; cancellation requested", file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    json.dump(payload, _PROTOCOL_OUT, ensure_ascii=False, separators=(",", ":"))
    _PROTOCOL_OUT.write("\n")
    _PROTOCOL_OUT.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniverSR audio processing in an isolated worker")
    parser.add_argument("--request", required=True, help="JSON request file")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)
    args = _parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request JSON must be an object")
    except Exception as exc:
        _emit({"type": "error", "message": f"invalid request: {exc}"})
        return 2

    def progress(percent: int, message: str) -> None:
        if _CANCEL.is_set():
            raise KeyboardInterrupt("cancelled")
        _emit({"type": "progress", "percent": max(0, min(100, int(percent))), "message": str(message)})

    try:
        # UniverSR itself may print import/model diagnostics. Keep stdout a
        # strict JSON-lines channel while retaining those diagnostics on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            output_path = process_request(request, progress=progress, cancel=_CANCEL.is_set)
        if _CANCEL.is_set():
            raise KeyboardInterrupt("cancelled")
        _emit({"type": "result", "output_path": str(output_path), "sample_rate": 48000})
        return 0
    except KeyboardInterrupt as exc:
        print(f"worker cancelled: {exc}", file=sys.stderr, flush=True)
        _emit({"type": "error", "message": "cancelled"})
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
