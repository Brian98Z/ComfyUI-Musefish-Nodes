"""System memory guard for long-running downloads.

Watches TOTAL physical memory pressure (this process plus every other
process — the metric is free/total from the OS, so all processes count).
Before each chunk the caller asks :func:`wait_for_memory`:

- usage <= 95% of total -> return immediately, download proceeds
- usage >  95%          -> sleep silently (reporter still ticks) until the
  pressure drops, or the caller's deadline expires

This keeps bulk downloads from pushing the machine into swap when other
apps (browser, ComfyUI, the OS itself) already hold most of the RAM.
"""

from __future__ import annotations

import time

# Pause between re-checks while waiting for memory to free up.
_POLL_INTERVAL = 2.0
# Hard cap on a single wait so a stuck process cannot hang a node forever
# when the caller passes no deadline.
_MAX_SINGLE_WAIT = 300.0

_WATERMARK = 0.95


def system_memory_fraction() -> float:
    """Return used/total physical memory across ALL processes (0.0 .. 1.0)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return 1.0 - vm.available / vm.total
    except Exception:
        pass
    # psutil-free fallback (Windows): GlobalMemoryStatusEx via ctypes.
    if os_name() == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            if status.ullTotalPhys:
                used = status.ullTotalPhys - status.ullAvailPhys
                return used / status.ullTotalPhys
    # Last resort: /proc/meminfo (Linux).
    try:
        info = {}
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                info[key.strip()] = int(value.strip().split()[0]) * 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        if total:
            return (total - available) / total
    except Exception:
        pass
    return 0.0  # cannot measure: never block the caller


def os_name() -> str:
    import os

    return os.name


def wait_for_memory(
    watermark: float = _WATERMARK,
    deadline_seconds: float | None = None,
    progress=None,
    log: list[str] | None = None,
) -> bool:
    """Block while system memory pressure exceeds ``watermark``.

    Returns True if it had to wait (for the caller's logging), False when
    memory was fine on the first check. ``progress`` (the node's reporter)
    keeps ticking so the UI bar does not look frozen during the wait.
    """
    waited = False
    started = time.monotonic()
    while True:
        usage = system_memory_fraction()
        if usage <= watermark:
            if waited and log is not None:
                log.append(
                    f"memory resumed ({usage * 100:.1f}% <= {watermark * 100:.0f}%), "
                    f"waited {time.monotonic() - started:.0f}s"
                )
            return waited
        waited = True
        if progress is not None:
            # keep the bar alive; value itself unchanged
            try:
                progress.fraction(-1.0)  # fraction() clamps -> no-op move
            except Exception:
                pass
        if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
            if log is not None:
                log.append(
                    f"memory still above {watermark * 100:.0f}% after "
                    f"{deadline_seconds:.0f}s; continuing anyway"
                )
            return waited
        time.sleep(_POLL_INTERVAL)
