"""DLSS5 worker entry point. Runs inside a disposable subprocess.

Usage:
    python worker.py --request '<json>'

The request dict is produced by dlss5_backend/session.py. stdout is a strict
JSON-lines protocol channel; all diagnostics go to stderr and the NGX log.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import traceback
from multiprocessing import shared_memory

import numpy as np

_FRAME_FORMAT_RGBA8 = "rgba8"


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.dlssnr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_wchar_p]
    lib.dlssnr_init.restype = ctypes.c_int
    lib.dlssnr_create_feature.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.dlssnr_create_feature.restype = ctypes.c_int
    lib.dlssnr_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.dlssnr_process.restype = ctypes.c_int
    lib.dlssnr_set_options.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                       ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    lib.dlssnr_set_options.restype = None
    lib.dlssnr_shutdown.argtypes = []
    lib.dlssnr_shutdown.restype = None
    return lib


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log_tail(path: str, limit: int = 4000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:]
    except OSError:
        return ""


def _set_options(lib: ctypes.CDLL, request: dict) -> None:
    lib.dlssnr_set_options(
        int(request.get("preset", 1)),      # preset: fixed at 1 by DLSS5Tool (only value used)
        int(request.get("style", 0)),       # 0 default / 1 natural / 2 cinema
        float(request.get("intensity", 1.0)),
        float(request.get("local_tone", 1.0)),
        float(request.get("local_struct", 1.0)),
        float(request.get("skin_struct", 0.5)),
        1 if request.get("use_auto_mask") else 0,
        0,                                  # ui_correction: static content only
        0,
        2,                                  # depth_convention: zero-guidance contract
        1.0,                                # motion_scale_x (unused, zero guidance)
        1.0,                                # motion_scale_y
    )


def _load_vsr(dll_dir: str, scale: int):
    """Bind the RTX Video Super Resolution host. Contract from DLSS5Tool's
    super_resolution._load_host: vsr_init(w, h, scale, quality, is_hdr, base, log)
    then vsr_process(src_ptr, dst_ptr) per frame."""
    lib = ctypes.WinDLL(os.path.join(dll_dir, "vsr_host.dll"))
    lib.vsr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_wchar_p, ctypes.c_wchar_p]
    lib.vsr_init.restype = ctypes.c_int
    lib.vsr_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vsr_process.restype = ctypes.c_int
    lib.vsr_shutdown.argtypes = []
    lib.vsr_shutdown.restype = None
    return lib


def _load_cudart():
    """Locate a CUDA runtime for host-memory pinning. Optional fast path:
    returns None (and the buffer stays pageable) when no CUDA is present."""
    try:
        import glob
        import torch  # noqa: F401 - only to locate the bundled cudart DLL
        base = os.path.dirname(torch.__file__)
        for candidate in glob.glob(os.path.join(base, "lib", "cudart64_*.dll")) + \
                glob.glob(os.path.join(base, "..", "..", "bin", "cudart64_*.dll")) + \
                glob.glob(os.path.join(base, "bin", "cudart64_*.dll")):
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
    except ImportError:
        pass
    return None


def _pin_shared_buffer(shm) -> int | None:
    """cudaHostRegister the mapped view so the NGX host's per-frame D3D12
    uploads/downloads skip the pageable-memory staging copy. Measured on this
    machine: 0.71ms -> 0.32ms per 8MB 1080p frame. Best-effort: any failure
    leaves the buffer pageable with zero functional impact."""
    if shm.buf is None:
        return None
    cudart = _load_cudart()
    if cudart is None or not hasattr(cudart, "cudaHostRegister"):
        return None
    try:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        err = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(shm.size), 0x02)
        return ptr if err == 0 else None
    except Exception:  # noqa: BLE001 - optional fast path only
        return None


def _unpin_shared_buffer(cudart, ptr) -> None:
    if cudart is None or ptr is None:
        return
    try:
        cudart.cudaHostUnregister(ctypes.c_void_p(ptr))
    except Exception:  # noqa: BLE001
        pass


def _worker_loop(lib, vsr_lib, request: dict, log_path: str, sr_scale: int,
                 width: int, height: int) -> int:
    input_shm = shared_memory.SharedMemory(name=request["input_shm"])
    output_shm = shared_memory.SharedMemory(name=request["output_shm"])
    in_shape = (height, width, 4)
    out_shape = (height * sr_scale, width * sr_scale, 4)
    input_view = np.ndarray(in_shape, dtype=np.uint8, buffer=input_shm.buf)
    output_view = np.ndarray(out_shape, dtype=np.uint8, buffer=output_shm.buf)

    # Pin the ring buffers in THIS process: the NGX host's D3D12 engine does its
    # DMA uploads/downloads from this address space, and pinned registration is
    # tracked per-process.
    cudart = _load_cudart()
    in_pin = _pin_shared_buffer(input_shm)
    out_pin = _pin_shared_buffer(output_shm)

    mv = np.zeros(out_shape[:2], np.float32)
    dp = np.zeros(out_shape[:2], np.float32)
    out = np.empty(out_shape, np.uint8)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = message.get("op")
            if op == "close":
                break
            if op != "frame":
                _emit({"ok": False, "error": f"unknown op: {op!r}"})
                return 2
            try:
                rgba = input_view
                reset = 1 if message.get("reset") else 0
                if vsr_lib is not None:
                    if not vsr_lib.vsr_process(
                            rgba.ctypes.data_as(ctypes.c_void_p),
                            out.ctypes.data_as(ctypes.c_void_p)):
                        raise RuntimeError("vsr_process failed")
                    rgba = out
                ok = lib.dlssnr_process(
                    rgba.ctypes.data_as(ctypes.c_void_p),
                    mv.ctypes.data_as(ctypes.c_void_p),
                    dp.ctypes.data_as(ctypes.c_void_p),
                    output_view.ctypes.data_as(ctypes.c_void_p),
                    reset,
                )
                if not ok:
                    raise RuntimeError("dlssnr_process failed")
                _emit({"ok": True, "event": "frame"})
            except Exception as exc:  # noqa: BLE001 - protocol-level report
                _emit({"ok": False, "error": f"{exc}", "log_tail": _log_tail(log_path)})
                return 3
    finally:
        try:
            lib.dlssnr_shutdown()
        except Exception:  # noqa: BLE001 - shutdown best effort
            traceback.print_exc(file=sys.stderr)
        if vsr_lib is not None:
            try:
                vsr_lib.vsr_shutdown()
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
        for shm in (input_shm, output_shm):
            try:
                shm.close()
            except OSError:
                pass
        _unpin_shared_buffer(cudart, in_pin)
        _unpin_shared_buffer(cudart, out_pin)
        # Parent unlinks its own handles; worker never unlinks (it opened
        # with create=False), matching multiprocessing ownership guidance.
    _emit({"ok": True, "event": "closed"})
    return 0


def main() -> int:
    request = json.loads(sys.argv[sys.argv.index("--request") + 1])
    dll_dir = request["dll_dir"]
    log_path = request["log_path"]
    width = int(request["width"])
    height = int(request["height"])
    sr_scale = int(request.get("super_resolution_scale") or 1)
    if sr_scale not in (1, 2, 4):
        _emit({"ok": False, "error": f"super_resolution_scale must be 1/2/4, got {sr_scale}"})
        return 2

    os.add_dll_directory(dll_dir)
    lib = _bind(ctypes.CDLL(os.path.join(dll_dir, "dlssnr_host.dll")))

    # host tuning: v2 supports dlssnr_configure; legacy host does not export it
    if hasattr(lib, "dlssnr_configure"):
        lib.dlssnr_configure(1, 1, 1, 2, 1)
    _set_options(lib, request)
    lib.dlssnr_shutdown()

    # Optional front RTX Video Super Resolution stage (DLSS5Tool order:
    # upscale first, then neural-render at the target resolution).
    vsr_lib = None
    if sr_scale > 1:
        vsr_lib = _load_vsr(dll_dir, sr_scale)
        if not vsr_lib.vsr_init(width, height, sr_scale, int(request.get("vsr_quality", 4)),
                                0, dll_dir, os.path.join(dll_dir, "vsr_run.log")):
            _emit({"ok": False, "error": "vsr_init failed (RTX VSR gate)",
                   "log_tail": _log_tail(os.path.join(dll_dir, "vsr_run.log"))})
            return 1

    if not lib.dlssnr_init(width * sr_scale, height * sr_scale, int(request.get("preset", 1)),
                           os.path.join(dll_dir, "nvngx_dlssnr.dll"), log_path):
        _emit({"ok": False, "error": "dlssnr_init failed (D3D12/driver gate)",
               "log_tail": _log_tail(log_path)})
        return 1
    if not lib.dlssnr_create_feature(width * sr_scale, height * sr_scale, int(request.get("preset", 1))):
        _emit({"ok": False, "error": "Feature 18 create failed", "log_tail": _log_tail(log_path)})
        return 1

    _emit({"ok": True, "event": "ready"})
    return _worker_loop(lib, vsr_lib, request, log_path, sr_scale, width, height)


if __name__ == "__main__":
    raise SystemExit(main())
