"""MusefishStudio local server.

Serves the workbench web UI and a JSON API that supervises a headless ComfyUI
instance (isolated base directory) and drives Musefish plugin workflows.
Standard library only — the interpreter is the H-disk python_embeded and we do
not install anything into it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request as _ur
import uuid
from urllib.parse import quote, unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from comfy_client import ComfyClient, ComfyError
import workflows
import asr as _asr
from asr import StopASR as _asr_StopASR
ROOT = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text("utf-8"))

HOST = CONFIG.get("host", "127.0.0.1")
PORT = int(CONFIG.get("port", 17995))
COMFY_HOST = CONFIG.get("comfy", {}).get("host", "127.0.0.1")
COMFY_PORT = int(CONFIG.get("comfy", {}).get("port", 17996))
COMFY_AUTOSTART = bool(CONFIG.get("comfy", {}).get("autostart", True))
COMFY_STARTUP_TIMEOUT = float(CONFIG.get("comfy", {}).get("startup_timeout", 300))
DATA_DIR = (HERE / CONFIG.get("data_dir", "../data")).resolve()
UPLOAD_DIR = (HERE / CONFIG.get("upload_dir", "../uploads")).resolve()
OUTPUT_DIR = (HERE / CONFIG.get("output_dir", "../data/output")).resolve()
LOG_DIR = (HERE / CONFIG.get("log_dir", "../logs")).resolve()
# ASR model root: config.asr_models_dir wins; empty → asr.py's resolution
# chain (env MUSEFISH_ASR_MODELS > ComfyUI folder_paths > <data>/models/asr).
# Dev machine pins it to the unified ComfyUI library via config.
if CONFIG.get("asr_models_dir"):
    _asr.set_models_dir(CONFIG["asr_models_dir"])
# ComfyUI 引擎目录解析链（首个含 main.py 的生效）：
#   1. config.comfy_dir        — 显式指定
#   2. data/comfyui            — 自包含形态（tools/env_setup.py 克隆/安装）
#   3. <root>/comfy junction   — 开发机快捷方式（指向外部 ComfyUI 安装）
_COMFY_CANDIDATES = [
    Path(CONFIG["comfy_dir"]) if CONFIG.get("comfy_dir") else None,
    DATA_DIR / "comfyui",
    ROOT / "comfy",
]
COMFY_DIR = next((p for p in _COMFY_CANDIDATES if p and (p / "main.py").is_file()),
                 _COMFY_CANDIDATES[1])
# 解释器解析链：runtime/venv（env_setup 产物，自包含形态）→ runtime/python junction（开发机）
_VENV_PY = ROOT / "runtime" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
PY = _VENV_PY if _VENV_PY.is_file() else ROOT / "runtime" / "python" / "python.exe"
# ffmpeg: config.ffmpeg_dir → <root>/ffmpeg/bin (portable bundle) → PATH.
# yt-dlp 与产物校验都会用到；PATH 兜底由调用处 (shutil.which) 完成。
_FFMPEG_CANDIDATES = [
    Path(CONFIG["ffmpeg_dir"]) if CONFIG.get("ffmpeg_dir") else None,
    ROOT / "ffmpeg" / "bin",
]
FFMPEG_DIR = next((p for p in _FFMPEG_CANDIDATES if p and (p / "ffmpeg.exe").is_file()),
                  _FFMPEG_CANDIDATES[-1])

for d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

client = ComfyClient(COMFY_HOST, COMFY_PORT)

_log_lock = threading.Lock()
_log_tail: list[str] = []


def log(line: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with _log_lock:
        _log_tail.append(f"[{stamp}] {line}")
        del _log_tail[:-500]
    print(f"[{stamp}] {line}", flush=True)


# ---------------------------------------------------------------- Comfy supervisor

class ComfySupervisor:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.want_running = False

    def extra_paths_yaml(self) -> Path:
        yaml = DATA_DIR / "extra_model_paths.yaml"
        models = DATA_DIR / "models"
        text = (
            "musefish_workbench:\n"
            f"  base_path: {models.as_posix()}\n"
            "  is_default: false\n"
            "  diffusion_models: diffusion_models\n"
            "  text_encoders: text_encoders\n"
            "  vae: vae\n"
            "  audio_encoders: audio_encoders\n"
        )
        yaml.write_text(text, "utf-8")
        return yaml

    def _spawn(self) -> None:
        cmd = [
            str(PY), "-s", str(COMFY_DIR / "main.py"),
            "--windows-standalone-build",
            "--base-directory", str(DATA_DIR),
            "--extra-model-paths-config", str(self.extra_paths_yaml()),
            "--port", str(COMFY_PORT),
            "--listen", COMFY_HOST,
            "--disable-auto-launch",
            "--cpu-vae",
        ]
        env = os.environ.copy()
        env["PATH"] = f"{PY.parent};{FFMPEG_DIR};" + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        # ASR node's model root: pass through the configured library when the
        # config pins one (dev machine); otherwise leave unset so the node's
        # own resolution chain (env > folder_paths > data dir) decides.
        if CONFIG.get("asr_models_dir"):
            env["MUSEFISH_ASR_MODELS"] = CONFIG["asr_models_dir"]
        log_file = LOG_DIR / "comfyui.log"
        with open(log_file, "ab") as lf:
            self.proc = subprocess.Popen(
                cmd, cwd=str(DATA_DIR), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        log(f"ComfyUI starting (pid {self.proc.pid}, port {COMFY_PORT})")

    def start(self, wait: bool = True) -> dict:
        with self.lock:
            if client.alive():
                self.want_running = True
                return {"running": True, "already": True}
            if self.proc and self.proc.poll() is None:
                pass
            else:
                self.want_running = True
                self._spawn()
        if wait:
            deadline = time.monotonic() + COMFY_STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                if client.alive():
                    log("ComfyUI is up")
                    return {"running": True}
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited with code {self.proc.returncode}; see logs/comfyui.log")
                time.sleep(1.0)
            raise RuntimeError("ComfyUI startup timeout")
        return {"running": client.alive()}

    def stop(self) -> dict:
        with self.lock:
            self.want_running = False
            p, self.proc = self.proc, None
        if p and p.poll() is None:
            subprocess.run(["taskkill", "/pid", str(p.pid), "/T", "/F"],
                           capture_output=True)
            log("ComfyUI stopped")
        return {"running": False}

    def watchdog(self) -> None:
        while True:
            time.sleep(5)
            if self.want_running and not client.alive():
                with self.lock:
                    if self.proc and self.proc.poll() is None:
                        continue
                    if self.want_running:
                        log("ComfyUI is down; restarting")
                        try:
                            self._spawn()
                        except Exception as e:  # noqa: BLE001
                            log(f"restart failed: {e}")


supervisor = ComfySupervisor()

# ---------------------------------------------------------------- job manager

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
# History survives a restart: the UI is the only record of what was produced,
# and a reboot used to erase it. Only terminal records are written (a crash
# mid-download is recovered from the plugin's own .part/.state files).
JOBS_FILE = DATA_DIR / "jobs.json"
JOBS_MAX = max(10, int(CONFIG.get("jobs_max", 200)))
_PERSIST = {"dirty": threading.Event(), "stop": threading.Event()}
_env_setup_proc: dict = {}  # {"proc","tail","done"} — one-shot env setup bookkeeping
WORK_Q: "queue.Queue[str]" = __import__("queue").Queue()
TASK_BUILDERS = {
    "pid": workflows.pid_upscale_graph,
    "pid_video": workflows.pid_video_graph,
    "dlss5_image": workflows.dlss5_image_graph,
    "dlss5_video": workflows.dlss5_video_graph,
    "universr": workflows.universr_graph,
}
# Tasks that bypass ComfyUI entirely (plugin python backends run in-process).
DIRECT_TASKS = {"video_download", "wechat_download", "file_download", "asr_transcribe"}


def _new_job(task: str, params: dict) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "task": task,
        "params": params,
        "status": "pending",
        "error": None,
        "created": time.time(),
        "started": None,
        "finished": None,
        "progress": None,
        "result": {},
    }
    with JOBS_LOCK:
        JOBS[job["id"]] = job
    _mark_dirty()
    WORK_Q.put(job["id"])
    return job

_TERMINAL = ("done", "error", "cancelled")
_JOB_FIELDS = ("id", "task", "status", "error", "created", "started", "finished",
               "progress", "result")


def _job_record(job: dict) -> dict:
    """The persisted shape is exactly what /api/jobs returns."""
    return {key: job.get(key) for key in _JOB_FIELDS}

_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _sanitise_result(result) -> dict:
    """Restored results are user-writable bytes: keep only local /files links."""
    if not isinstance(result, dict):
        return {}
    outputs = []
    for item in result.get("outputs") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        outputs.append({**item, "url": url if url.startswith("/files/") else ""})
    return {**result, "outputs": outputs}


def _contained_output(relative: str) -> Path | None:
    """Resolve a /files/ path inside OUTPUT_DIR, or None (no prefix tricks)."""
    try:
        dest = (OUTPUT_DIR / relative).resolve()
        dest.relative_to(OUTPUT_DIR.resolve())
    except (ValueError, OSError):
        return None
    return dest


def _mark_dirty() -> None:
    """Ask the writer thread to persist; callers never touch the disk."""
    _PERSIST["dirty"].set()


def _prune_jobs() -> list[str]:
    """Drop the oldest *terminal* records beyond JOBS_MAX (files stay on disk)."""
    with JOBS_LOCK:
        terminal = sorted((j for j in JOBS.values() if j.get("status") in _TERMINAL),
                          key=lambda j: j.get("created") or 0)
        excess = len(JOBS) - JOBS_MAX
        dropped = []
        for job in terminal:
            if excess <= 0:
                break
            if JOBS.pop(job["id"], None) is not None:
                dropped.append(job["id"])
                excess -= 1
    return dropped


_WRITE_LOCK = threading.Lock()   # serialises the file itself, not just the payload


def _write_jobs() -> None:
    with _WRITE_LOCK:
        _write_jobs_locked()


def _write_jobs_locked() -> None:
    with JOBS_LOCK:
        payload = {"version": 1, "jobs": [_job_record(j) for j in
                                          sorted(JOBS.values(), key=lambda j: j.get("created") or 0)]}
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, JOBS_FILE)


def _load_jobs() -> int:
    """Restore history; anything that was mid-flight is reported as interrupted."""
    if not JOBS_FILE.is_file():
        return 0
    try:
        payload = json.loads(JOBS_FILE.read_text("utf-8"))
        records = payload.get("jobs") or []
    except (OSError, ValueError) as exc:
        log(f"jobs history unreadable ({exc}); starting empty")
        return 0
    restored = 0
    with JOBS_LOCK:
        for record in records[-JOBS_MAX:]:
            if not isinstance(record, dict) or not _JOB_ID_RE.match(str(record.get("id") or "")):
                continue
            job = {key: record.get(key) for key in _JOB_FIELDS}
            job["result"] = _sanitise_result(job.get("result"))
            if job.get("status") not in _TERMINAL:
                job["status"] = "error"
                job["error"] = "服务重启，任务已中断（未完成）"
                job["finished"] = job.get("finished") or time.time()
            JOBS[job["id"]] = job
            restored += 1
    return restored


def _persist_loop() -> None:
    """Coalesce writes: many status changes, at most a few disk writes."""
    while not _PERSIST["stop"].is_set():
        if _PERSIST["dirty"].wait(timeout=1.0):
            _PERSIST["dirty"].clear()
            dropped = _prune_jobs()
            if dropped:
                log(f"jobs history pruned: {len(dropped)} old record(s) (files kept)")
            try:
                _write_jobs()
            except OSError as exc:
                log(f"jobs history write failed: {exc}")
            continue
        _PERSIST["stop"].wait(0.0)  # idle tick


def _collect_outputs(history_entry: dict, job_dir: Path) -> list[dict]:
    job_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for node_id, node_out in (history_entry.get("outputs") or {}).items():
        for kind in ("images", "video", "audio", "gifs"):
            for item in node_out.get(kind) or []:
                data = client.download(item["filename"], item.get("subfolder", ""),
                                       item.get("type", "output"))
                dest = job_dir / Path(str(item["filename"]).replace("\\", "/")).name
                dest.write_bytes(data)
                files.append({
                    "name": item["filename"],
                    "path": str(dest),
                    "url": f"/files/{job_dir.name}/{dest.name}",
                    "bytes": len(data),
                })
    return files


def _queue_snapshot() -> dict | None:
    try:
        q = client.queue()
        running = q.get("queue_running") or []
        pending = q.get("queue_pending") or []
        return {"queue_remaining": len(running) + len(pending)}
    except Exception:  # noqa: BLE001
        return None


def _worker() -> None:
    while True:
        job_id = WORK_Q.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                continue
            if job.get("status") == "cancelled":
                # cancelled while still queued: honor it atomically, never run
                job["finished"] = time.time()
                _mark_dirty()
                continue
            job["status"] = "running"
            job["started"] = time.time()
        task = job["task"]
        try:
            if task in DIRECT_TASKS:
                _run_direct_task(job)
                continue
            if not client.alive():
                supervisor.start(wait=True)
            # bind the engine at submit time: an endpoint switch mid-job must
            # not redirect this job's polling/uploads/downloads to another instance
            job_client = client
            if job["status"] == "cancelled":
                continue  # cancelled while still queued locally
            params = dict(job["params"])
            graph, out_node = _build_graph(task, params, cli=job_client)
            prompt_id = job_client.submit(graph, client_id=f"studio-{job_id}")
            job["prompt_id"] = prompt_id
            log(f"job {job_id} ({task}) submitted as {prompt_id}")

            def on_progress(snap: dict) -> None:
                # ComfyClient.wait() already fetched /queue for this tick; reuse it
                # instead of a second request every poll (was ~60 extra req/min).
                running = snap.get("queue_running") or []
                pending = snap.get("queue_pending") or []
                job["progress"] = {"queue_remaining": len(running) + len(pending)}

            history = job_client.wait(prompt_id, on_progress=on_progress)
            job_dir = OUTPUT_DIR / job_id
            files = _collect_outputs(history, job_dir)
            job["result"] = {"outputs": files}
            if job["status"] != "cancelled":  # user cancelled during wait
                job["status"] = "done"
            log(f"job {job_id} done: {len(files)} file(s)")
        except ComfyError as e:
            if job["status"] == "cancelled":
                log(f"job {job_id} cancelled")
            else:
                job["status"] = "error"
                job["error"] = str(e)[:2000]
                log(f"job {job_id} failed: {str(e)[:300]}")
        except _asr_StopASR:
            job["status"] = "cancelled"
            log(f"job {job_id} cancelled")
        except Exception as e:  # noqa: BLE001
            if job["status"] == "cancelled":
                log(f"job {job_id} cancelled")
            else:
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"
                log(f"job {job_id} failed: {job['error'][:300]}")
        finally:
            job["finished"] = time.time()
            _mark_dirty()


def _load_plugin_module(name: str):
    """Import a module from the plugin dir as part of a package so its
    relative imports (e.g. `from .memory_guard import ...`) resolve.

    We register the package with a neutral name and a spec whose loader does
    NOT execute __init__.py — importing the real package would pull in every
    ComfyUI node module (torch/comfy deps we don't want in the workbench
    server) and a stray `import app` that collides with server/app.py.
    """
    import importlib
    import importlib.util
    import types

    plugin_root = DATA_DIR / "custom_nodes"
    pkg_dir = plugin_root / "ComfyUI-Musefish-Nodes"
    pkg = "_musefish_plugin"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg, pkg_dir / "__init__.py",
            submodule_search_locations=[str(pkg_dir)])
        m = importlib.util.module_from_spec(spec)
        # skip exec_module: only alias __path__ for submodule resolution
        m.__path__ = [str(pkg_dir)]
        m.__spec__ = spec
        sys.modules[pkg] = m
    return importlib.import_module(f"{pkg}.{name}")


def _run_direct_task(job: dict) -> None:
    """Tasks that run a plugin python backend without ComfyUI."""
    params = dict(job["params"])
    if job["task"] in ("video_download", "wechat_download"):
        _run_video_download(job, params)
    elif job["task"] == "file_download":
        _run_file_download(job, params)
    elif job["task"] == "asr_transcribe":
        _run_asr_transcribe(job, params)
    else:
        raise ValueError(f"unknown direct task {job['task']}")


_WX_DECODE_KEY_JSON_KEYS = ("decode_key", "dec_key", "decrypt_key")
_WX_CDN_URL_RE = re.compile(
    r"https?://[^\s\"'<>]*finder\.video\.qq\.com/251/20302/stodownload[^\s\"'<>]*",
    re.IGNORECASE)


def _wechat_auto_extract(paste: str, logs: list[str]) -> tuple[str, str]:
    """Extract (tokened CDN URL, decode_key) from a user paste.

    Port of the plugin node's _auto_extract (kept Comfy-free here): Channels
    API JSON > direct CDN URL > share link hint. Empty strings mean the
    memory scan must supply the piece.
    """
    if not paste:
        return "", ""
    stripped = paste.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            def _find_first(node, keys):
                if isinstance(node, dict):
                    for k in keys:
                        v = node.get(k)
                        if isinstance(v, str) and v:
                            return v
                        if isinstance(v, (int, float)):
                            return str(int(v))
                    for value in node.values():
                        r = _find_first(value, keys)
                        if r:
                            return r
                elif isinstance(node, list):
                    for value in node:
                        r = _find_first(value, keys)
                        if r:
                            return r
                return None

            key = _find_first(data, _WX_DECODE_KEY_JSON_KEYS)
            url = None

            def _walk_media(node):
                if isinstance(node, dict):
                    if "url" in node and isinstance(node.get("url"), str):
                        yield node["url"], node.get("decode_key")
                    for value in node.values():
                        yield from _walk_media(value)
                elif isinstance(node, list):
                    for value in node:
                        yield from _walk_media(value)

            for media_url, media_key in _walk_media(data):
                if not media_url:
                    continue
                # video channel is 20302; 20304 is the cover stream whose
                # keys do NOT decrypt the video
                if "/20302/stodownload" in media_url or not media_url.startswith("http"):
                    url = media_url
                    key = key or media_key
                    break
            if url or key:
                if url:
                    logs.append("已从 Channels API JSON 解析出视频 URL")
                if key:
                    logs.append("已从 Channels API JSON 解析出 decode_key")
                return url or "", key or ""
            logs.append("粘贴内容像 JSON 但无视频号字段；转内存扫描")
    cdn = _WX_CDN_URL_RE.search(paste)
    if cdn:
        logs.append("粘贴内容含视频号 CDN URL")
        return cdn.group(0).rstrip(".,;)"), ""
    if re.search(r"weixin\.qq\.com/sph/|channels\.weixin\.qq\.com", paste, re.IGNORECASE):
        logs.append("粘贴内容为分享链接；将匹配微信客户端内存记录")
    else:
        logs.append("粘贴内容无链接；将按文本匹配微信客户端内存记录")
    return "", ""


def _run_asr_transcribe(job: dict, params: dict) -> None:
    """语音识别 (ported GPT-SoVITS ASR engines, in-process).

    input_path: 一个音频文件或一个装音频的目录；产物 <名称>.list 落
    OUTPUT_DIR/<job_id>/，可直接在标注页打开。params:
      model  — faster-whisper 尺寸 (small/large-v3/large-v3-turbo…) 或 "funasr"
      lang   — auto/en/ja/ko/zh/yue
      precision — float32/float16/int8
    """
    raw = (params.get("input_path") or "").strip().strip('"')
    src = Path(raw)
    if not src.is_absolute() or not src.exists():
        raise ValueError(f"输入不存在: {raw}")
    _AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    if src.is_file() and src.suffix.lower() not in _AUDIO_EXT:
        raise ValueError(f"仅支持音频文件: {', '.join(_AUDIO_EXT)}")
    if src.is_dir() and not any(src.glob("*.wav")) and not any(
            n.suffix.lower() in _AUDIO_EXT for n in src.iterdir() if n.is_file()):
        raise ValueError("目录内没有音频文件")

    # 单文件 → 建一个只含它的临时目录（引擎按目录枚举）；目录 → 原样
    import shutil
    if src.is_file():
        stage = OUTPUT_DIR / job["id"] / "_input"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, stage / src.name)
        in_dir = stage
    else:
        in_dir = src

    model = (params.get("model") or "large-v3-turbo").strip()
    lang = (params.get("lang") or "auto").strip().lower()
    if lang not in ("auto", "zh", "yue", "en", "ja", "ko"):
        lang = "auto"
    precision = (params.get("precision") or "float16").strip().lower()
    if precision not in ("float16", "int8"):  # float32 removed: largest VRAM/space for no visible gain here
        precision = "float16"
    out_dir = OUTPUT_DIR / job["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_line(done: int, total: int) -> None:
        job["progress"] = {"pct": round(done * 100 / total, 1) if total else None,
                           "text": f"{done}/{total} 条"}

    def should_stop() -> bool:
        return job.get("status") == "cancelled"

    log(f"ASR 开始：{in_dir.name} · {model} · {lang} · {precision}")
    if model == "funasr":
        # funasr 单模型逐文件（lang 固定 zh/yue）
        import soundfile  # noqa: F401  fail fast if missing
        names = sorted(n for n in in_dir.iterdir()
                       if n.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg"))
        fun_model = _asr.create_funasr_model("zh" if lang in ("", "auto", "zh") else lang)
        lines = []
        total = len(names)
        for k, n in enumerate(names):
            if should_stop():
                raise _asr.StopASR()
            text = fun_model.generate(input=str(n))[0]["text"]
            lines.append(f"{n.resolve()}|{in_dir.name}|{'ZH' if lang != 'yue' else 'YUE'}|{text}")
            on_line(k + 1, total)
        list_path = out_dir / f"{in_dir.name}.list"
        list_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        list_path = _asr.whisper_execute(
            str(in_dir), str(out_dir), model, lang, precision,
            on_line=on_line, should_stop=should_stop)
    if should_stop():
        raise _asr.StopASR()  # cancelled during the final file: never report done
    list_path = Path(list_path)
    # parse the .list into display rows: wav file name + recognized text
    rows = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) == 4:
            rows.append({"file": Path(parts[0]).name, "lang": parts[2], "text": parts[3].strip()})
    job["result"] = {
        "outputs": [{"name": list_path.name, "path": str(list_path),
                     "url": f"/files/{job['id']}/{quote(list_path.name)}",
                     "bytes": list_path.stat().st_size}],
        "asr_rows": rows,
    }
    job["status"] = "done"
    log(f"ASR 完成：{list_path} ({len(rows)} 条)")


def _run_video_download(job: dict, params: dict) -> None:
    mod = _load_plugin_module("musefish_video_download")

    out_dir = OUTPUT_DIR / job["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    reporter = mod.ProgressReporter(lambda kind, data=None: lines.append(f"{kind}: {data}") or None)

    def finish(path, info=None):
        size = path.stat().st_size
        job["result"] = {"outputs": [{
            "name": path.name, "path": str(path),
            "url": f"/files/{job['id']}/{path.name}", "bytes": size,
        }]}
        if info and info.get("title"):
            job["result"]["info"] = {"title": info["title"], "duration": info.get("duration")}
        job["status"] = "done"
        log(f"job {job['id']} done: {path.name} ({size/1048576:.1f} MB)")

    if job["task"] == "wechat_download":
        # WeChat Channels: memory-scan the running desktop client for the
        # decode_key + tokened URL, download ciphertext, verify via ftyp XOR,
        # decrypt (first 128 KiB Isaac64), ffprobe-verify. No ComfyUI needed:
        # the plugin's backend module is Comfy-free for this path.
        paste = (params.get("url") or "").strip().strip('"')
        max_scan = float(params.get("scan_seconds", 120))
        logs: list[str] = []
        url, decode_key = _wechat_auto_extract(paste, logs)
        for line in logs:
            log(line)

        def _scan():
            cands = mod.scan_wechat_decode_keys(max_seconds=max_scan)
            if not cands:
                raise mod.DownloadError(
                    "未在微信客户端内存中找到视频号记录：请先在本机微信里播放目标视频几秒，"
                    "保持微信运行后重试")
            return cands

        candidates: list[dict] = []
        if not decode_key or not url:
            # URL or key missing → need memory records (a pasted key alone
            # still requires the tokened URL from the client's memory)
            log("从微信客户端内存扫描记录（补齐密钥/URL）…")
            job["progress"] = {"pct": 5}
            candidates = _scan()
            if not decode_key:
                decode_key = candidates[0]["decode_key"]
                log(f"扫描到 {len(candidates)} 条记录，优先最新：decode_key={decode_key[:16]}…")
        if not url:
            log("从内存记录恢复带签名的 CDN URL…")
            job["progress"] = {"pct": 15}
            match = next((c for c in candidates if c["decode_key"] == decode_key), candidates[0])
            url = mod.complete_wechat_url(match, max_seconds=max_scan)

        attempts = [(url, decode_key)]
        for c in candidates[1:6]:
            try:
                attempts.append((mod.complete_wechat_url(c, max_seconds=max_scan), c["decode_key"]))
            except mod.DownloadError:
                continue

        job["progress"] = {"pct": 25}
        last_error = None
        path = info = None
        for i, (try_url, try_key) in enumerate(attempts, 1):
            try:
                path, info = mod.download_wechat_encrypted(try_url, out_dir, try_key, reporter, lines)
                break
            except mod.DownloadError as err:
                last_error = err
                if "ftyp mismatch" not in str(err):
                    raise
                log(f"候选 {i}：密钥/URL 不匹配，尝试下一条")
                path = info = None
        if path is None:
            raise mod.DownloadError(
                f"全部 {len(attempts)} 个候选密钥校验失败（签名可能已过期）。"
                f"请在微信里重新播放目标视频几秒后再试。（最后错误：{last_error}）")
        media = mod.verify_download(path, lines)
        finish(path, {"title": media.get("title") or "微信视频号"})
        return

    # ---- generic platforms (yt-dlp) ----
    url = params["url"]
    quality = params.get("quality", "best")

    def progress_hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            if total:
                job["progress"] = {"pct": round(done / total * 100, 1)}

    platform = mod.detect_platform(url)
    if platform == "wechat" or mod.looks_like_wechat_media(url):
        raise mod.DownloadError(
            "检测到微信视频号链接：请切到「微信视频号下载」功能提交（该链接需本机微信客户端内存解密，不走 yt-dlp）")

    # Bypass the plugin's format_map: prefer H.264 (widest playback support in
    # the workbench <video> preview), then HEVC, then unconstrained fallback.
    # NOTE: yt-dlp filter field is `vcodec` (not `v_codec`); Bilibili serves
    # video-only + audio-only streams, so pair with `+bestaudio`.
    height_cap = {"best": "", "1080p": "[height<=1080]", "720p": "[height<=720]",
                  "480p": "[height<=480]"}.get(quality, "")
    fmt = (
        f"bestvideo*[vcodec^=avc1]{height_cap}+bestaudio/"
        f"best[vcodec^=avc1]{height_cap}/"
        f"bestvideo*[vcodec^=hev1]{height_cap}+bestaudio/"
        f"best{height_cap}/best"
    )
    target = mod._unique_path(out_dir, "musefish_download", ".mp4")
    yt_dlp = mod._ytdlp_module()
    opts = {
        "outtmpl": str(target),
        "format": fmt,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "progress_hooks": [progress_hook],
        "ffmpeg_location": str(mod.FFMPEG) if mod.FFMPEG.exists() else mod.shutil.which("ffmpeg"),
        "concurrent_fragment_downloads": 8,
    }
    if mod.looks_like_x_url(url) and mod._proxy_reachable(mod._YTDLP_PROXY):
        opts["proxy"] = mod._YTDLP_PROXY
    elif __import__("re").search(r"(?:youtube\.com|youtu\.be)", url, __import__("re").IGNORECASE):
        if mod._proxy_reachable(mod._YTDLP_PROXY):
            opts["proxy"] = mod._YTDLP_PROXY
        else:
            raise mod.DownloadError("YouTube 需要本地代理（127.0.0.1:7897），当前不可达")

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp (2026.08+) rewrites opts["outtmpl"] into a dict after a run;
    # use the path we computed, falling back to whatever landed in out_dir.
    path = target if target.is_file() else None
    if path is None:
        cands = sorted(out_dir.glob("musefish_download*"), key=lambda p: p.stat().st_size, reverse=True)
        if not cands:
            raise mod.DownloadError("下载完成但未找到输出文件")
        path = cands[0]
    size = path.stat().st_size
    job["result"] = {"outputs": [{
        "name": path.name, "path": str(path),
        "url": f"/files/{job['id']}/{path.name}", "bytes": size,
    }]}
    if info.get("title"):
        job["result"]["info"] = {"title": info["title"], "duration": info.get("duration")}
    job["status"] = "done"
    log(f"job {job['id']} done: {path.name} ({size/1048576:.1f} MB)")


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """`name.ext`, `name (1).ext`, … never clobber an existing file."""
    cand = directory / f"{stem}{suffix}"
    i = 1
    while cand.exists():
        cand = directory / f"{stem} ({i}){suffix}"
        i += 1
    return cand


def _run_file_download(job: dict, params: dict) -> None:
    """Direct-link download with parallel ranges (shared musefish_fastdl module).

    Built for CDNs that throttle each connection (CivitAI b2, S3-compatible):
    a browser or single curl crawls at ~100 KB/s while 32-64 ranged requests
    aggregate to the link's real bandwidth. Resumes a `.part` file, and keeps
    signed query strings intact.
    """
    # 与 ComfyUI 节点共用同一份实现（插件包 musefish_fastdl），避免两套行为漂移
    fastdl = _load_plugin_module("musefish_fastdl")

    url = (params.get("url") or "").strip().strip('"').strip("'")
    if not url.lower().startswith(("http://", "https://")):
        raise fastdl.DownloadError("请填写 http(s) 直链地址（含完整查询串，如签名参数）")

    conns = max(1, min(int(params.get("conns") or 32), 64))
    want_proxy = (params.get("proxy") or "auto").strip()
    log(f"文件断点续传：{url[:180]}")
    try:
        info, proxy, route = fastdl.probe_with_route(
            url, want_proxy, config_proxy=CONFIG.get("proxy"), timeout=45,
            should_stop=lambda: job.get("status") == "cancelled", log=log)
    except fastdl.Cancelled:
        raise
    log(f"代理：{route}")

    total = info["size"]
    size_label = f"{total / 1048576:.1f} MB" if total else "大小未知"
    log(f"远端 {size_label} · 分片{'支持' if info['ranges'] else '不支持'} · "
        f"{info['content_type'] or '未知类型'} · {route}")

    save_dir = (params.get("save_dir") or "").strip()
    target_dir = Path(save_dir).expanduser() if save_dir else fastdl.desktop_dir()
    if not target_dir.is_absolute():
        target_dir = (ROOT / target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    name = fastdl.safe_name(params.get("filename") or info["filename"])
    dest = _unique_path(target_dir, Path(name).stem, Path(name).suffix)

    t0 = time.time()

    def on_progress(progress) -> None:
        """引擎已把 (done,total) 折算成速率/ETA，这里只负责搬进任务状态。"""
        job["progress"] = {
            "pct": round(progress.percent, 1) if progress.total else None,
            "speed": round(progress.speed / 1048576, 2),
            "done": progress.done,
            "total": progress.total,
            "eta": round(progress.eta, 1) if progress.eta is not None else None,
            "elapsed": round(progress.elapsed, 1),
            "resumed": progress.resumed,
            "text": progress.text(),
        }

    reporter = fastdl.ProgressReporter(on_progress, interval=0.4)
    res = fastdl.download(url, dest, conns=conns, proxy=proxy, timeout=45,
                          on_progress=reporter,
                          should_stop=lambda: job.get("status") == "cancelled",
                          probe_info=info)

    size = Path(res["path"]).stat().st_size
    elapsed = max(time.time() - t0, 1e-6)
    try:
        Path(res["path"]).resolve().relative_to(OUTPUT_DIR.resolve())
        inside = True
    except (ValueError, OSError):
        inside = False
    job["result"] = {
        "outputs": [{
            "name": Path(res["path"]).name,
            "path": res["path"],
            "url": f"/files/{job['id']}/{quote(Path(res['path']).name)}" if inside else "",
            "bytes": size,
        }],
        "info": {
            "avg_mbps": round(size / elapsed / 1048576, 2),
            "seconds": round(elapsed, 1),
            "conns": conns,
            "ranges": res.get("ranges", True),
            "proxy": proxy or "direct",
            "route": route,
        },
    }
    job["status"] = "done"
    log(f"文件断点续传完成：{res['path']} ({size / 1048576:.1f} MB, "
        f"{size / elapsed / 1048576:.2f} MB/s, {elapsed:.0f}s)")


def _resolve_input(input_path: str) -> Path:
    p = Path(input_path)
    if p.is_file():
        resolved = p.resolve()
        for base in (UPLOAD_DIR, ROOT):
            try:
                resolved.relative_to(base.resolve())
            except ValueError:
                continue
            return resolved
        raise PermissionError(f"input path outside the studio workspace: {input_path}")
    for base in (UPLOAD_DIR, ROOT):
        q = base / input_path
        if q.is_file():
            return q
    raise FileNotFoundError(f"input not found: {input_path}")


def _build_graph(task: str, params: dict, cli=None) -> tuple[dict, str]:
    cli = cli or client
    input_path = params.pop("input_path", "")
    if task in ("pid", "dlss5_image", "pid_video"):
        p = _resolve_input(input_path)
        if task in ("pid", "pid_video"):
            kw = {k: v for k, v in params.items() if k != "input_path"}
            kw.setdefault("unet", PID_DEFAULTS["unet"])
            kw.setdefault("clip", PID_DEFAULTS["clip"])
            kw.setdefault("vae", PID_DEFAULTS["vae"])
            if task == "pid_video":
                video_ref = cli.upload(str(p))
                return workflows.pid_video_graph(video_ref, **kw)
        image_ref = cli.upload(str(p))
        if task == "pid":
            return workflows.pid_upscale_graph(image_ref, **kw)
        return workflows.dlss5_image_graph(image_ref, **params)
    if task == "dlss5_video":
        video_ref = cli.upload(str(_resolve_input(input_path)))
        return workflows.dlss5_video_graph(video_ref, **params)
    if task == "universr":
        audio_ref = cli.upload(str(_resolve_input(input_path)))
        return workflows.universr_graph(audio_ref, **params)
    raise ValueError(f"unknown task {task}")


threading.Thread(target=_worker, daemon=True, name="workbench-jobs").start()

# ---------------------------------------------------------------- helpers

MODEL_EXTS = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".sft", ".pkl", ".pt2"}
SKIP_DIRS = {"Put%", "put_"}


def list_models(kind: str) -> list[dict]:
    base = DATA_DIR / "models" / kind
    if not base.is_dir():
        return []
    out = []
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
            continue
        rel = p.relative_to(base).as_posix()
        if any(part.startswith("put_") for part in p.parts):
            continue
        out.append({"name": rel, "bytes": p.stat().st_size})
    out.sort(key=lambda x: x["name"])
    return out


def check_links_inline() -> dict:
    targets = {
        "runtime/python": ROOT / "runtime" / "python",
        "comfy": ROOT / "comfy",
        "custom_node": DATA_DIR / "custom_nodes" / "ComfyUI-Musefish-Nodes",
        "models/dlss5": DATA_DIR / "models" / "dlss5",
        "models/UniverSR": DATA_DIR / "models" / "UniverSR",
        "models/diffusion_models": DATA_DIR / "models" / "diffusion_models",
        "models/text_encoders": DATA_DIR / "models" / "text_encoders",
        "models/vae": DATA_DIR / "models" / "vae",
    }
    return {k: {"ok": v.exists(), "path": str(v)} for k, v in targets.items()}


def _config_path() -> Path:
    return HERE / "config.json"


def save_config() -> None:
    cfg = dict(CONFIG)
    cfg["comfy"] = {"host": COMFY_HOST, "port": COMFY_PORT,
                    "autostart": COMFY_AUTOSTART, "startup_timeout": COMFY_STARTUP_TIMEOUT}
    _config_path().write_text(json.dumps(cfg, indent=2, ensure_ascii=False), "utf-8")


def apply_comfy_endpoint(host: str, port: int, *, persist: bool = True) -> dict:
    """Point the client at a (possibly different) local ComfyUI endpoint."""
    global COMFY_HOST, COMFY_PORT, client
    host = (host or "127.0.0.1").strip()
    port = int(port)
    if not (1 <= port <= 65535):
        return {"ok": False, "error": "port out of range"}
    if host not in ("127.0.0.1", "localhost", "::1"):
        # Only a local ComfyUI is a valid target: any other host would let the
        # studio upload local files to a third party.
        return {"ok": False, "error": "host must be loopback (127.0.0.1/localhost/::1)"}
    COMFY_HOST, COMFY_PORT = host, port
    client = ComfyClient(host, port)
    if persist:
        save_config()
    up = client.alive()
    log(f"comfy endpoint set to {host}:{port} (reachable={up})")
    return {"ok": True, "host": host, "port": port, "running": up}


def scan_comfy_ports(timeout: float = 0.35) -> list[dict]:
    """Probe well-known ComfyUI ports and a port range for a live instance."""
    import socket

    candidates = [8188, 8189, 8988, 17996, 8288, 8388]
    found = []
    seen = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        c = ComfyClient(COMFY_HOST, port)
        try:
            if c.alive():
                found.append({"host": COMFY_HOST, "port": port})
        except Exception:  # noqa: BLE001
            pass
    # quick sweep of 8100-8299 and 8970-8999 for user installs
    for start, end in ((8100, 8299), (8970, 8999)):
        for port in range(start, end + 1):
            if port in seen:
                continue
            seen.add(port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((COMFY_HOST, port)) != 0:
                    continue
            c = ComfyClient(COMFY_HOST, port)
            try:
                if c.alive():
                    found.append({"host": COMFY_HOST, "port": port})
            except Exception:  # noqa: BLE001
                pass
    return found


def env_info() -> dict:
    """Runtime environment snapshot for the Settings page."""
    py = PY
    torch = cuda = "未知"
    try:
        out = subprocess.run(
            [str(py), "-c", "import torch;print(torch.__version__);print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            lines = out.stdout.split()
            torch = lines[0] if lines else torch
            cuda = "可用" if lines[-1] == "True" else "不可用"
    except Exception:  # noqa: BLE001
        pass
    ffmpeg = (FFMPEG_DIR / "ffmpeg.exe").is_file()
    disk = shutil.disk_usage(ROOT)
    return {
        "python": {"path": str(py), "ok": py.is_file()},
        "torch": torch, "cuda": cuda,
        "ffmpeg": {"ok": ffmpeg, "dir": str(FFMPEG_DIR)},
        "comfy_dir": {"ok": (COMFY_DIR / "main.py").is_file(), "path": str(COMFY_DIR)},
        "plugin": {"ok": (DATA_DIR / "custom_nodes/ComfyUI-Musefish-Nodes/__init__.py").is_file(),
                   "path": str(DATA_DIR / "custom_nodes/ComfyUI-Musefish-Nodes")},
        "disk_free_gb": round(disk.free / 1e9, 1),
    }


# workbench model tree -> what it is for
MODEL_MANIFEST = {
    # 就绪判定对齐 PID_DEFAULTS（工作台实际默认使用的权重）：默认文件在 = 就绪
    "diffusion_models/PiD": {
        "desc": "PiD 超分 UNET（PiD 必需）",
        "required": [r"pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors"],
    },
    "text_encoders/PixelDiT": {
        "desc": "PixelDiT 文本编码器（PiD 必需）",
        "required": ["gemma_2_2b_it_elm_fp8_scaled.safetensors"],
    },
    "vae/Flux": {
        "desc": "Flux VAE（PiD 必需）",
        "required": ["UltraFlux-v1-vae.safetensors"],
    },
    "dlss5": {
        "desc": "NVIDIA NGX DLSS5 DLL（DLSS5 必需，RTX 50 系）",
        "required": ["nvngx_dlss.dll", "nvngx_dlssnr.dll"],
    },
    "UniverSR": {
        "desc": "UniverSR 音频超分权重（音频必需）",
        "required": [],  # 目录非空即视为就绪（权重经 models/huggingface 缓存布局）
    },
}


def model_manifest() -> dict:
    import time as _t

    items = []
    for rel, spec in MODEL_MANIFEST.items():
        desc = spec["desc"]
        required = spec.get("required") or []
        d = DATA_DIR / "models" / rel
        files = [p for p in d.rglob("*") if p.is_file()] if d.is_dir() else []
        total = sum(f.stat().st_size for f in files)
        if required:
            names = {p.name for p in files}
            ready = all(r in names for r in required)
        else:
            ready = bool(files)
        items.append({
            "dir": rel, "desc": desc, "present": bool(files),
            "ready": ready,
            "files": len(files), "bytes": total,
        })
    return {"models": items, "root": str(DATA_DIR / "models"),
            "checked_at": _t.time()}


PID_DEFAULTS = {
    "unet": r"PiD\pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors",
    "clip": r"PixelDiT\gemma_2_2b_it_elm_fp8_scaled.safetensors",
    "vae": r"Flux\UltraFlux-v1-vae.safetensors",
}

# ---------------------------------------------------------------- HTTP server

STATIC = HERE / "web"
MIME = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
    ".svg": "image/svg+xml", ".ico": "image/x-icon", ".json": "application/json",
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg", ".flac": "audio/flac", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError):
            self.close_connection = True  # browser media stack aborts ranges; not an error

    def log_message(self, fmt, *args):  # keep file-route traffic visible
        try:
            msg = fmt % args
            if "/files/" in msg:
                log(f"http {msg}")
        except Exception:
            pass

    # ---- plumbing
    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # local-only service: same-origin + loopback origins, never "*"
        self.send_header("Access-Control-Allow-Origin", f"http://{HOST}:{PORT}")
        self.end_headers()
        self.wfile.write(body)

    def _file_headers(self, ctype: str, download: bool) -> None:
        """Shared header block; `nosniff` keeps a mislabelled upload from being run."""
        self.send_header("Content-Type", ctype)
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", "attachment")

    def _file(self, path: Path, code: int = 200, download: bool = False) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
        if download:
            ctype = "application/octet-stream"
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            # media playback in Chromium requires 206 partial content
            try:
                start_s, end_s = rng[6:].split("-", 1)
                if not start_s:
                    # suffix form "bytes=-N": last N bytes of the file
                    n = int(end_s)
                    if n <= 0:
                        raise ValueError
                    start = max(0, size - n)
                    end = size - 1
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s else size - 1
                    end = min(end, size - 1)
                if start > end or start >= size:
                    raise ValueError
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self._file_headers(ctype, download)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        # Stream the body: read_bytes() held the whole file (a 64 MiB export cost
        # 64 MiB of peak allocation, and parallel downloads multiplied it).
        self.send_response(code)
        self._file_headers(ctype, download)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        remaining = size
        with open(path, "rb") as f:
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 32 * 1024 * 1024:  # JSON bodies here are tiny; reject abuse
            raise ValueError(f"request body too large ({length} bytes)")
        return self.rfile.read(length) if length > 0 else b""

    # ---- request guard
    def _guard(self) -> bool:
        """Loopback-only + same-origin + explicit client header.

        The server is a privileged local API: without these checks any web page
        could POST /api/jobs (arbitrary save_dir) or repoint the ComfyUI endpoint
        at an attacker host, and DNS rebinding could read /api/log.
        """
        host = (self.headers.get("Host") or "").strip().lower()
        allowed = {f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"}
        if host not in allowed:
            self._json({"error": "host not allowed"}, 403)
            return False
        origin = (self.headers.get("Origin") or "").strip().rstrip("/").lower()
        if origin and origin not in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}",
                                     f"http://[::1]:{PORT}"}:
            self._json({"error": "origin not allowed"}, 403)
            return False
        # Browser-initiated cross-site requests are the threat (a page the user
        # happens to have open). Local scripts/curl stay usable: no Origin and no
        # cross-site Sec-Fetch-Site means "not a browser page from elsewhere".
        if self.command in ("POST", "PUT", "DELETE"):
            site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
            if site and site not in ("same-origin", "none"):
                self._json({"error": "cross-site request blocked"}, 403)
                return False
            # A custom header cannot be sent cross-origin without a preflight, so
            # requiring it closes the "no-Origin plain form POST" hole this guard
            # documents. Same-origin UI and local scripts both send it.
            if (self.headers.get("X-Musefish-Client") or "").strip() != "1":
                self._json({"error": "X-Musefish-Client: 1 required for changes"}, 403)
                return False
        return True

    # ---- routes
    def do_GET(self):  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._file(STATIC / "index.html")
            elif path == "/api/status":
                self._json({
                    "server": {"host": HOST, "port": PORT},
                    "mode": "standalone" if (DATA_DIR / "comfyui" / "main.py").is_file() else "linked",
                    "comfy": {
                        "running": client.alive(),
                        "port": COMFY_PORT,
                        "endpoint": {"host": COMFY_HOST, "port": COMFY_PORT},
                        "autostart": COMFY_AUTOSTART,
                        "pid": supervisor.proc.pid if supervisor.proc and supervisor.proc.poll() is None else None,
                    },
                    "links": check_links_inline(),
                    "pid_defaults": PID_DEFAULTS,
                })
            elif path == "/api/env":
                self._json(env_info())
            elif path == "/api/models/manifest":
                self._json(model_manifest())
            elif path == "/api/download/options":
                fastdl = _load_plugin_module("musefish_fastdl")
                auto_url, auto_label = fastdl.auto_proxy_endpoint(CONFIG.get("proxy"))
                self._json({
                    "save_dir": str(fastdl.desktop_dir()),
                    "proxies": fastdl.detect_proxies(),
                    "auto": {"url": auto_url, "label": auto_label},
                })
            elif path == "/api/log":
                with _log_lock:
                    self._json({"lines": list(_log_tail)})
            elif path == "/api/models":
                kind = self.path.split("kind=", 1)[1].split("&")[0] if "kind=" in self.path else ""
                if kind not in ("diffusion_models", "text_encoders", "vae", "audio_encoders", "UniverSR", "dlss5"):
                    self._json({"error": "bad kind"}, 400)
                    return
                self._json({"models": list_models(kind)})
            elif path.startswith("/api/nodes/"):
                cls = path.rsplit("/", 1)[1]
                self._json(client.object_info(cls))
            elif path == "/api/asr/model/status":
                # /api/asr/model/status?model=large-v3
                import asr as _asr
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                mm = re.search(r"model=([\w\-.]+)", qs)
                model = mm.group(1) if mm else ""
                st = _asr.download_state()
                self._json({"ready": _asr.model_ready(model) if model else None,
                            "exists": _asr.model_ready(model) if model else None,
                            **st})
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
                    slim = [{k: j[k] for k in ("id", "task", "status", "error", "created",
                                               "started", "finished", "progress", "result")} for j in jobs[:100]]
                self._json({"jobs": slim})
            elif path.startswith("/api/jobs/"):
                jid = path.rsplit("/", 1)[1]
                with JOBS_LOCK:
                    job = JOBS.get(jid)
                    rec = _job_record(job) if job else None
                self._json(rec if rec else {"error": "no such job"}, 200 if rec else 404)
            elif path.startswith("/files/"):
                rel = unquote(path[len("/files/"):])
                dest = _contained_output(rel)
                if dest is None or not dest.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    self._file(dest, download=dest.suffix.lower() not in
                               (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm",
                                ".mov", ".mp3", ".wav", ".flac", ".m4a", ".ogg"))
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client closed mid-stream (normal for video seek)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    log(f"file {rel} FAILED: {type(e).__name__}: {e} | "
                        f"{traceback.format_exc().splitlines()[-1].strip()}")
                    raise
            else:
                self._json({"error": "unknown route"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # <video> aborts range streams on seek/pause — normal
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"GET {path} FAILED: {type(e).__name__}: {e} | "
                f"{traceback.format_exc().splitlines()[-1].strip()}")
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    def do_POST(self):  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/upload":
                fn = self._save_multipart()
                self._json(fn)
            elif path == "/api/jobs":
                req = json.loads(self._body() or b"{}")
                task = req.get("task")
                if task not in TASK_BUILDERS and task not in DIRECT_TASKS:
                    self._json({"error": f"task must be one of {sorted(set(TASK_BUILDERS) | DIRECT_TASKS)}"}, 400)
                    return
                params = dict(req.get("params") or {})
                if req.get("input_path"):
                    params.setdefault("input_path", req["input_path"])
                if (task not in DIRECT_TASKS or task == "asr_transcribe") and not params.get("input_path"):
                    self._json({"error": "input_path required"}, 400)
                    return
                if task in ("video_download", "file_download") and not params.get("url"):
                    self._json({"error": "url required"}, 400)
                    return
                if task == "asr_transcribe":
                    import asr as _asr
                    m = (params.get("model") or "large-v3-turbo").strip()
                    if not _asr.model_ready(m):
                        gb = _asr.WHISPER_SIZES_GB.get(m, _asr._FUNASR_TOTAL_GB)
                        self._json({
                            "error": f"识别模型 {m} 尚未下载到本地（约 {gb:.1f} GB）",
                            "model_missing": {"model": m, "size_gb": gb},
                        }, 409)
                        return
                job = _new_job(task, params)
                self._json({"job_id": job["id"], "status": job["status"]})
            elif path == "/api/asr/model/download":
                import asr as _asr
                req = json.loads(self._body() or b"{}")
                try:
                    self._json(_asr.start_download((req.get("model") or "").strip()))
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
            elif path == "/api/comfy/start":
                self._json(supervisor.start(wait=False))
            elif path == "/api/comfy/stop":
                self._json(supervisor.stop())
            elif path == "/api/models/relink":
                # filesystem side effect → POST only (GET never mutates)
                r = subprocess.run(
                    [str(PY), str(ROOT / "tools" / "get_models.py")],
                    capture_output=True, text=True, timeout=300)
                ok = r.returncode == 0
                self._json({"ok": ok, "output": (r.stdout + r.stderr)[-2000:]})
            elif path == "/api/env/setup":
                # one-shot env prep: venv + deps + ComfyUI core (long: pip ~GBs)
                # run detached; progress lands in the server log, poll /api/env
                proc = subprocess.Popen(
                    [str(PY), str(ROOT / "tools" / "env_setup.py")],
                    cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace")
                _env_setup_proc["proc"] = proc
                _env_setup_proc["tail"] = []
                def _pump_envsetup():
                    for line in proc.stdout or []:
                        with _log_lock:
                            _env_setup_proc["tail"].append(line.rstrip())
                            del _env_setup_proc["tail"][:-80]
                    proc.wait()  # ensure returncode is set before we report
                    with _log_lock:
                        _env_setup_proc["done"] = True
                    log(f"env_setup exited {proc.returncode}")
                threading.Thread(target=_pump_envsetup, daemon=True).start()
                self._json({"ok": True, "started": True})
            elif path == "/api/env/setup/status":
                with _log_lock:
                    tail = list(_env_setup_proc.get("tail") or [])
                    done = _env_setup_proc.get("done", False)
                self._json({"done": done, "tail": tail[-25:]})
            elif path == "/api/models/download":
                body = json.loads(self._body() or b"{}")
                only = body.get("only", "all")
                r = subprocess.run(
                    [str(PY), str(ROOT / "tools" / "download_models.py"),
                     "--only", str(only)],
                    capture_output=True, text=True, timeout=7200)
                self._json({"ok": r.returncode == 0,
                            "output": (r.stdout + r.stderr)[-6000:]})
            elif path == "/api/comfy/endpoint":
                req = json.loads(self._body() or b"{}")
                res = apply_comfy_endpoint(req.get("host", COMFY_HOST),
                                           req.get("port", COMFY_PORT))
                self._json(res, 200 if res.get("ok") else 400)
            elif path == "/api/comfy/scan":
                self._json({"found": scan_comfy_ports()})
            else:
                self._json({"error": "unknown route"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_DELETE(self):  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/outputs":
            # one-click cleanup: remove every job output dir AND the matching
            # job records (and their uploads) so the task list stays in sync
            removed = 0
            jobs_removed = 0
            with JOBS_LOCK:
                done_ids = [jid for jid, j in JOBS.items() if j.get("status") in ("done", "error")]
            for jid in done_ids:
                for d in (OUTPUT_DIR / jid, UPLOAD_DIR / jid):
                    try:
                        if d.is_dir():
                            shutil.rmtree(d)
                            removed += 1
                        elif d.is_file():
                            d.unlink()
                            removed += 1
                    except OSError:
                        pass
                with JOBS_LOCK:
                    if JOBS.pop(jid, None) is not None:
                        jobs_removed += 1
            # orphan dirs whose job record no longer exists
            for child in list(OUTPUT_DIR.iterdir()):
                with JOBS_LOCK:
                    known = child.name in JOBS
                if not known:
                    try:
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                        removed += 1
                    except OSError:
                        pass
            log(f"outputs cleaned: {removed} entries, {jobs_removed} job record(s)")
            self._json({"ok": True, "removed": removed, "jobs_removed": jobs_removed})
            return
        if path.startswith("/api/jobs/"):
            jid = path.rsplit("/", 1)[1]
            queued_cancelled = False
            with JOBS_LOCK:
                job = JOBS.get(jid)
                if not job:
                    self._json({"error": "no such job"}, 404)
                    return
                if job.get("status") == "pending" and not job.get("prompt_id"):
                    # still queued locally: mark cancelled INSIDE the lock so the
                    # worker's atomic dequeue check can never race us
                    job["status"] = "cancelled"
                    _mark_dirty()
                    log(f"job {jid} cancelled before submit")
                    queued_cancelled = True
            if queued_cancelled:
                self._json({"ok": True})
                return
            # running (or engine-queued): interrupt the engine, then delete from
            # its queue; status flips to cancelled only if still not terminal
            try:
                import urllib.request as _ur

                req = _ur.Request(
                    client.base + "/interrupt",
                    data=b"", method="POST")  # interrupt running execution
                _ur.urlopen(req, timeout=10).read()
            except Exception:
                pass  # not the running item on the engine — delete queued
            try:
                import urllib.request as _ur

                req = _ur.Request(
                    client.base + "/queue",
                    data=json.dumps({"delete": [job["prompt_id"]]}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                _ur.urlopen(req, timeout=10).read()
            except Exception as e:  # noqa: BLE001
                log(f"cancel failed: {e}")
            with JOBS_LOCK:
                if job["status"] not in _TERMINAL:
                    job["status"] = "cancelled"
                    _mark_dirty()
            self._json({"ok": True})
        else:
            self._json({"error": "unknown route"}, 404)

    # ---- multipart (single file, field name "file")
    def _save_multipart(self) -> dict:
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary=(.+)$', ctype.strip())
        if not m:
            raise ValueError("expected multipart/form-data")
        boundary = m.group(1).strip().strip('"').encode()
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty upload body")
        stream = self.rfile
        # Count every byte taken off the socket: the body length is only knowable
        # from Content-Length minus what the boundary/header reads already consumed.
        consumed = 0

        def read_line() -> bytes:
            nonlocal consumed
            line = stream.readline()
            consumed += len(line)
            return line

        def remaining() -> int:
            return max(0, length - consumed)

        start = read_line()
        if not start.startswith(b"--" + boundary):
            raise ValueError("bad multipart start")
        headers = {}
        while True:
            line = read_line()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("utf-8", "replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        disp = headers.get("content-disposition", "")
        name_m = re.search(r'filename="([^"]*)"', disp)
        orig = name_m.group(1) if name_m else "upload.bin"
        safe = re.sub(r'[\\/:*?"<>|]+', "_", Path(orig).name) or "upload.bin"
        marker = b"\r\n--" + boundary
        chunk_sz = 1 << 20
        window = b""
        found = False
        tmp_path = UPLOAD_DIR / (uuid.uuid4().hex + ".part")
        try:
            with open(tmp_path, "wb") as f:
                while remaining() > 0:
                    chunk = stream.read(min(chunk_sz, remaining()))
                    if not chunk:
                        break
                    consumed += len(chunk)
                    window += chunk
                    idx = window.find(marker)
                    if idx >= 0:
                        f.write(window[:idx])
                        found = True
                        break
                    # keep the tail that could be the start of a boundary
                    safe_keep = len(window) - len(marker)
                    if safe_keep > 0:
                        f.write(window[:safe_keep])
                        window = window[safe_keep:]
            if not found:
                raise ValueError("multipart end not found")
            # Drain the epilogue (\r\n--boundary--\r\n): leaving bytes unread
            # desyncs the next keep-alive request on the same connection.
            while remaining() > 0:
                chunk = stream.read(min(chunk_sz, remaining()))
                if not chunk:
                    break
                consumed += len(chunk)
        except Exception:
            tmp_path.unlink(missing_ok=True)  # never leave .part litter
            raise
        # unique final name: same-second same-name uploads must not overwrite
        final = UPLOAD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe}"
        os.replace(tmp_path, final)
        return {"path": str(final), "name": final.name, "bytes": final.stat().st_size}
def main() -> None:
    # Refuse to double-start: if the port already serves OUR workbench, exit
    # so the new copy never fights the old one for requests (random routing).
    try:
        r = _ur.urlopen(f"http://{HOST}:{PORT}/api/status", timeout=2)
        body = r.read(4096).decode("utf-8", "replace")
        if r.status == 200 and "comfy" in body:
            log(f"another MusefishStudio server already on {HOST}:{PORT} — exiting")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass  # port free or foreign service; proceed (bind will fail loudly)
    log(f"MusefishStudio server on http://{HOST}:{PORT}")
    log(f"data={DATA_DIR} comfy_dir={COMFY_DIR} py={PY}")
    if not PY.is_file():
        log("FATAL: 未找到 Python 解释器（runtime/venv 或 runtime/python）。")
        log("       自包含形态: python tools/env_setup.py   开发机: python tools/link_models.py")
        raise SystemExit(3)
    if not (COMFY_DIR / "main.py").is_file():
        log("FATAL: 未找到 ComfyUI 引擎（data/comfyui 或 comfy junction）。")
        log("       自包含形态: python tools/env_setup.py   开发机: python tools/link_models.py")
        raise SystemExit(3)
    restored = _load_jobs()
    if restored:
        log(f"jobs history restored: {restored} record(s) from {JOBS_FILE.name}")
    persist_thread = threading.Thread(target=_persist_loop, daemon=True, name="jobs-persist")
    persist_thread.start()
    threading.Thread(target=supervisor.watchdog, daemon=True, name="comfy-watchdog").start()
    if COMFY_AUTOSTART:
        def boot() -> None:
            try:
                supervisor.start(wait=True)
                log("autostart complete")
            except Exception as e:  # noqa: BLE001
                log(f"autostart failed: {e}")
        threading.Thread(target=boot, daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.allow_reuse_address = False  # refuse double-bind: two servers on one port = random routing
    print(json.dumps({"server_port": PORT}), flush=True)  # client discovery
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        supervisor.stop()
    finally:
        _PERSIST["stop"].set()
        _PERSIST["dirty"].set()
        persist_thread.join(timeout=5.0)   # the writer finishes before the last flush
        try:
            _write_jobs()          # last state, including anything just finished
        except OSError:
            pass


if __name__ == "__main__":
    main()
