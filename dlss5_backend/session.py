"""dlss5_backend — crash-isolated DLSS5 neural-render backend for ComfyUI.

Drives DLSS5Tool's host DLLs (`dlssnr_host.dll` + `nvngx_dlssnr.dll`) in a
disposable worker process so NGX faults can never take down the ComfyUI server.

Protocol (worker.py, JSON lines on stdout):
  parent -> worker: {"op": "init", "width", "height", "style", "intensity",
                     "local_tone", "local_struct", "skin_struct", "use_auto_mask",
                     "super_resolution_scale", "vsr_quality",
                     "input_shm": [name, ...], "output_shm": [name, ...],
                     "dll_dir", "log_path"}
  worker -> parent: {"ok": true, "event": "ready"}
  parent -> worker: {"op": "frame", "slot": 0|1, "reset": bool}
  worker -> parent: {"ok": true, "event": "frame"}
  parent -> worker: {"op": "close"}
  worker -> parent: {"ok": true, "event": "closed"} then exits
  any failure:      {"ok": false, "error": "...", "log_tail": "..."}

Frames travel through two shared-memory ring slots (input at source
resolution, output at super_resolution target) — zero-copy on the worker
side, one array copy each way on the parent. NGX requires contiguous RGBA8
host buffers; shared memory avoids the 85ms/frame PNG encode+decode tax at
2x target sizes (measured 27x faster than the previous PNG file protocol).

The two slots make the session a 2-deep pipeline: while the worker is inside
its NGX call for frame n, the parent fills slot n+1 and converts the result
of slot n-1. The parent's per-frame pixel conversion therefore overlaps the
engine instead of serialising with it — measured at 480x848 -> 2x ultra, the
engine call is ~17ms/frame, so the parent's conversion runs entirely inside
that window and the GPU stops idling between frames. Ordering is preserved
(one request in, one ordered ack out, same reset flags), so the temporal
history and the output bytes are identical to a strictly sequential session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import folder_paths

_BACKEND_DIR = Path(__file__).resolve().parent
_DLL_DIR = Path(folder_paths.models_dir) / "dlss5"
# Legacy host is the verified working backend on this machine (RTX 5070 Ti,
# driver 616.56): full frame loop completes in ~4ms/frame at 256px. The v2
# host (dlssnr_host_v2.dll) hangs inside its first dlssnr_process call on
# this driver stack, so it is intentionally not used.
_HOST_DLL = _DLL_DIR / "dlssnr_host.dll"
_RUNTIME_DLL = _DLL_DIR / "nvngx_dlssnr.dll"
_VSR_HOST_DLL = _DLL_DIR / "vsr_host.dll"
_VSR_RUNTIME_DLL = _DLL_DIR / "nvngx_vsr.dll"
_WORKER = _BACKEND_DIR / "worker.py"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Ring depth: 2 keeps one slot being written by the parent while the worker
# owns the other, which is all the overlap the engine's latency needs.
_RING_DEPTH = 2


class DLSS5Error(RuntimeError):
    """The isolated DLSS5 worker failed, exited, or stopped responding."""


def backend_available(super_resolution_scale: int = 1) -> tuple[bool, str]:
    """Check only the runtime files required by the chosen VSR scale."""
    required = (_HOST_DLL, _RUNTIME_DLL)
    if super_resolution_scale > 1:
        required += (_VSR_HOST_DLL, _VSR_RUNTIME_DLL)
    for path in required:
        if not path.is_file():
            return False, f"missing {path} (install the DLSS5 runtime in models/dlss5)"
    return True, "ok"


class DLSS5Session:
    """One worker process bound to one (width, height, style) contract.

    Frame flow — strictly ordered, up to `_RING_DEPTH` frames in flight::

        view = session.next_input()   # shared buffer for the next frame
        view[:, :, :3] = pixels       # caller fills it (no copy on this side)
        session.push(reset=bool)      # hand the frame to the worker
        ...
        rgba = session.pull()         # ordered ack, returns that frame's slot

    A slot returned by `pull()` stays valid until the next `push()` of the
    same slot (i.e. `_RING_DEPTH` frames later); callers copy out immediately.
    """

    def __init__(self, width: int, height: int, style: int, intensity: float,
                 local_tone: float, local_struct: float, skin_struct: float,
                 use_auto_mask: bool, log_path: Path | None = None,
                 super_resolution_scale: int = 1, vsr_quality: int = 4):
        ok, detail = backend_available(super_resolution_scale)
        if not ok:
            raise DLSS5Error(detail)
        if int(super_resolution_scale) not in (1, 2, 4):
            raise DLSS5Error(f"super_resolution_scale must be 1/2/4, got {super_resolution_scale}")
        self.width = int(width)
        self.height = int(height)
        self.super_resolution_scale = int(super_resolution_scale)
        self._proc: subprocess.Popen | None = None
        self._tmp = Path(log_path) if log_path else _BACKEND_DIR / "logs"
        self._tmp.mkdir(parents=True, exist_ok=True)
        self._log_path = self._tmp / "dlss_run.log"

        self._input_shape = (self.height, self.width, 4)
        self._output_shape = (self.output_height, self.output_width, 4)
        in_bytes = int(np.prod(self._input_shape))
        out_bytes = int(np.prod(self._output_shape))
        self._input_shms = [shared_memory.SharedMemory(create=True, size=in_bytes)
                            for _ in range(_RING_DEPTH)]
        self._output_shms = [shared_memory.SharedMemory(create=True, size=out_bytes)
                             for _ in range(_RING_DEPTH)]
        self._input_views = [np.ndarray(self._input_shape, dtype=np.uint8, buffer=shm.buf)
                             for shm in self._input_shms]
        self._output_views = [np.ndarray(self._output_shape, dtype=np.uint8, buffer=shm.buf)
                              for shm in self._output_shms]
        self._head = 0
        self._pending: deque[int] = deque()

        self._request = {
            "op": "init",
            "width": self.width,
            "height": self.height,
            "style": int(style),
            "intensity": float(intensity),
            "local_tone": float(local_tone),
            "local_struct": float(local_struct),
            "skin_struct": float(skin_struct),
            "use_auto_mask": bool(use_auto_mask),
            "super_resolution_scale": self.super_resolution_scale,
            "vsr_quality": max(1, min(4, int(vsr_quality))),
            "input_shm": [shm.name for shm in self._input_shms],
            "output_shm": [shm.name for shm in self._output_shms],
            "dll_dir": str(_DLL_DIR),
            "log_path": str(self._log_path),
        }
        self._start()

    # -- process plumbing -------------------------------------------------
    def _start(self) -> None:
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-u", str(_WORKER), "--request", json.dumps(self._request)],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        try:
            ready = self._read_event(timeout=600.0)  # NGX shader compile can take minutes on a new driver
        except DLSS5Error:
            self.close()
            raise
        if ready.get("event") != "ready":
            self.close()
            raise DLSS5Error(f"worker not ready: {ready}")

    def _read_event(self, timeout: float) -> dict:
        if self._proc is None or self._proc.stdout is None:
            raise DLSS5Error("worker pipe unavailable")
        deadline = time.monotonic() + timeout
        # readline blocks; poll liveness so a crashed worker fails fast instead
        # of blocking on a dead pipe.
        while True:
            line = self._proc.stdout.readline()
            if line:
                break
            code = self._proc.poll()
            if code is not None:
                detail = self._log_tail() or "no diagnostics"
                raise DLSS5Error(f"DLSS5 worker exited (code {code}): {detail}")
            if time.monotonic() > deadline:
                raise DLSS5Error(f"DLSS5 worker timed out after {timeout:.0f}s: {self._log_tail() or 'no diagnostics'}")
            time.sleep(0.05)
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            raise DLSS5Error("empty worker response")
        try:
            event = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DLSS5Error(f"bad worker response: {text[:400]}") from exc
        if not event.get("ok"):
            raise DLSS5Error(str(event.get("error", "unknown worker error")) + "\n" + str(event.get("log_tail", ""))[:1500])
        return event

    def _log_tail(self, limit: int = 4000) -> str:
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    # -- frame API ---------------------------------------------------------
    @property
    def output_width(self) -> int:
        return self.width * self.super_resolution_scale

    @property
    def output_height(self) -> int:
        return self.height * self.super_resolution_scale

    @property
    def inflight(self) -> int:
        """Frames handed to the worker whose result has not been pulled yet."""
        return len(self._pending)

    def next_input(self) -> np.ndarray:
        """Input buffer for the next frame. The caller must `push()` it."""
        if len(self._pending) >= _RING_DEPTH:
            raise DLSS5Error("ring full: pull() an in-flight frame before pushing another")
        return self._input_views[self._head]

    def push(self, reset: bool = False) -> None:
        """Hand the frame written into `next_input()` to the worker."""
        slot = self._head
        self._send({"op": "frame", "slot": slot, "reset": bool(reset)})
        self._pending.append(slot)
        self._head = (self._head + 1) % _RING_DEPTH

    def pull(self) -> np.ndarray:
        """Block until the oldest in-flight frame is done; return its RGBA8 output."""
        if not self._pending:
            raise DLSS5Error("pull() with no frame in flight")
        self._read_event(timeout=300.0)
        return self._output_views[self._pending.popleft()]

    def drain(self) -> None:
        """Wait out every in-flight frame (used before closing/rebuilding)."""
        while self._pending:
            self._read_event(timeout=300.0)
            self._pending.popleft()

    def _send(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise DLSS5Error(f"DLSS5 worker unavailable (exit={self._proc.poll() if self._proc else 'n/a'})")
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DLSS5Error(f"failed to send to DLSS5 worker: {exc}; {self._log_tail()}") from exc

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        proc, self._proc = self._proc, None
        try:
            if proc is not None:
                try:
                    if proc.poll() is None and proc.stdin is not None:
                        proc.stdin.write(b'{"op": "close"}\n')
                        proc.stdin.flush()
                        proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except OSError:
                        pass
        finally:
            self._pending.clear()
            for shms, views in ((self._input_shms, self._input_views),
                                (self._output_shms, self._output_views)):
                while shms:
                    shm, view = shms.pop(), views.pop()
                    del view
                    try:
                        shm.close()
                    except (BufferError, OSError):
                        pass
                    try:
                        shm.unlink()
                    except (BufferError, FileNotFoundError, OSError):
                        pass
