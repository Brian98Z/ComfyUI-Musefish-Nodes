"""Child-process lifecycle helpers (Windows-first, standard library only).

Windows has no POSIX process groups: a parent that dies hard (Task Manager,
crash, ``Stop-Process -Force``) leaves its children running — observed with the
headless ComfyUI instance outliving its Studio server for hours. A Job Object
with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ties the whole tree to this
process: when the last handle to the job closes, including on abnormal exit,
the kernel terminates every member.

``kill_tree`` is the cooperative path (taskkill /T) for deliberate stops, e.g.
cancelling a mastering run whose engine child is an ffmpeg process.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9

_job_handle = None
_job_failed = False


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(wintypes.ULONG)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _get_job():
    """Create (once) the kill-on-close job this server owns."""
    global _job_handle, _job_failed
    if os.name != "nt" or _job_failed:
        return None
    if _job_handle is not None:
        return _job_handle
    try:
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job_handle = job
    except Exception as exc:  # noqa: BLE001 - never block a job over this
        _job_failed = True
        print(f"[proc-tree] job object unavailable: {exc}", file=sys.stderr, flush=True)
        return None
    return _job_handle


def join(proc) -> bool:
    """Put a freshly spawned child (and everything it spawns) in the job.

    Returns False when the process cannot be assigned — most often because this
    server itself already runs inside a job that forbids nested assignment; the
    caller's cooperative ``kill_tree`` path still applies.
    """
    if os.name != "nt" or proc is None:
        return False
    job = _get_job()
    handle = getattr(proc, "_handle", None)
    if not job or handle is None:
        return False
    try:
        if _kernel32().AssignProcessToJobObject(job, wintypes.HANDLE(int(handle))):
            return True
        err = ctypes.get_last_error()
        if err != 5:  # 5 = ACCESS_DENIED (already in another job): expected, stay quiet
            print(f"[proc-tree] assign failed for pid {proc.pid}: {err}",
                  file=sys.stderr, flush=True)
    except (OSError, ctypes.ArgumentError) as exc:
        print(f"[proc-tree] assign error for pid {proc.pid}: {exc}",
              file=sys.stderr, flush=True)
    return False


def kill_tree(proc, timeout: float = 10.0, log=None) -> None:
    """Terminate a child and every process it started (Windows: taskkill /T)."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/pid", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        if log:
            log(f"pid {proc.pid} ignored terminate; killed")
