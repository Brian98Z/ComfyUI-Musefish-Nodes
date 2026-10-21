"""JSON-lines CLI worker for the portable UniverSR backend.

Usage::

    python audio_backend/worker.py --request request.json
    python audio_backend/worker.py --requests first.json second.json ...

A batch keeps one interpreter (and one loaded checkpoint) alive for every
request, so a multi-clip batch pays the model load once.

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
    parser.add_argument("--request", help="JSON request file")
    parser.add_argument("--requests", nargs="+", help="ordered JSON request files (batch)")
    args = parser.parse_args()
    if bool(args.request) == bool(args.requests):
        parser.error("pass exactly one of --request or --requests")
    args.paths = args.requests or [args.request]
    return args


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)
    args = _parse_args()
    requests = []
    for index, path in enumerate(args.paths):
        try:
            request = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request JSON must be an object")
        except Exception as exc:
            _emit({"type": "error", "index": index, "message": f"invalid request: {exc}"})
            return 2
        requests.append(request)

    def progress_for(index: int):
        def progress(percent: int, message: str) -> None:
            if _CANCEL.is_set():
                raise KeyboardInterrupt("cancelled")
            _emit({"type": "progress", "index": index, "percent": max(0, min(100, int(percent))),
                   "message": str(message)})
        return progress

    exit_code = 0
    for index, request in enumerate(requests):
        _emit({"type": "start", "index": index})
        try:
            # UniverSR itself may print import/model diagnostics. Keep stdout a
            # strict JSON-lines channel while retaining those diagnostics on stderr.
            with contextlib.redirect_stdout(sys.stderr):
                output_path = process_request(request, progress=progress_for(index), cancel=_CANCEL.is_set)
            if _CANCEL.is_set():
                raise KeyboardInterrupt("cancelled")
            _emit({"type": "result", "index": index, "output_path": str(output_path), "sample_rate": 48000})
        except KeyboardInterrupt as exc:
            print(f"worker cancelled: {exc}", file=sys.stderr, flush=True)
            _emit({"type": "error", "index": index, "message": "cancelled"})
            return 130
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "index": index, "message": str(exc)})
            exit_code = 1
            break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
