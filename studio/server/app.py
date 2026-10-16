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
import subprocess
import sys
import threading
import time
import urllib.request as _ur
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from comfy_client import ComfyClient, ComfyError
import workflows
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
COMFY_DIR = ROOT / "comfy"          # junction -> source ComfyUI install
PY = ROOT / "runtime" / "python" / "python.exe"
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
WORK_Q: "queue.Queue[str]" = __import__("queue").Queue()
TASK_BUILDERS = {
    "pid": workflows.pid_upscale_graph,
    "pid_video": workflows.pid_video_graph,
    "dlss5_image": workflows.dlss5_image_graph,
    "dlss5_video": workflows.dlss5_video_graph,
    "universr": workflows.universr_graph,
}
# Tasks that bypass ComfyUI entirely (plugin python backends run in-process).
DIRECT_TASKS = {"video_download", "wechat_download"}


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
    WORK_Q.put(job["id"])
    return job


def _collect_outputs(history_entry: dict, job_dir: Path) -> list[dict]:
    job_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for node_id, node_out in (history_entry.get("outputs") or {}).items():
        for kind in ("images", "video", "audio", "gifs"):
            for item in node_out.get(kind) or []:
                data = client.download(item["filename"], item.get("subfolder", ""),
                                       item.get("type", "output"))
                dest = job_dir / item["filename"].replace("/", "_")
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

            def on_progress(_snap: dict) -> None:
                s = _queue_snapshot()
                if s:
                    job["progress"] = s

            history = job_client.wait(prompt_id, on_progress=on_progress)
            job_dir = OUTPUT_DIR / job_id
            files = _collect_outputs(history, job_dir)
            job["result"] = {"outputs": files}
            if job["status"] != "cancelled":  # user cancelled during wait
                job["status"] = "done"
            log(f"job {job_id} done: {len(files)} file(s)")
        except ComfyError as e:
            job["status"] = "error"
            job["error"] = str(e)[:2000]
            log(f"job {job_id} failed: {str(e)[:300]}")
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            log(f"job {job_id} failed: {job['error'][:300]}")
        finally:
            job["finished"] = time.time()


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


def _resolve_input(input_path: str) -> Path:
    p = Path(input_path)
    if p.is_file():
        return p
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
    "diffusion_models/PiD": "PiD 超分 UNET（PiD 必需）",
    "text_encoders/PixelDiT": "PixelDiT 文本编码器（PiD 必需）",
    "vae/Flux": "Flux VAE（PiD 必需）",
    "dlss5": "NVIDIA NGX DLSS5 DLL（DLSS5 必需，RTX 50 系）",
    "UniverSR": "UniverSR 音频超分权重（音频必需）",
}


def model_manifest() -> dict:
    import time as _t

    items = []
    for rel, desc in MODEL_MANIFEST.items():
        d = DATA_DIR / "models" / rel
        files = [p for p in d.rglob("*") if p.is_file()] if d.is_dir() else []
        total = sum(f.stat().st_size for f in files)
        items.append({
            "dir": rel, "desc": desc, "present": bool(files),
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

    def _file(self, path: Path, code: int = 200) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
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
            self.send_header("Content-Type", ctype)
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
        try:
            body = path.read_bytes()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    # ---- routes
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._file(STATIC / "index.html")
            elif path == "/api/status":
                self._json({
                    "server": {"host": HOST, "port": PORT},
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
                    self._json(dict(job) if job else {"error": "no such job"}, 200 if job else 404)
            elif path.startswith("/files/"):
                rel = path[len("/files/"):]
                dest = (OUTPUT_DIR / rel).resolve()
                if not str(dest).startswith(str(OUTPUT_DIR)) or not dest.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    self._file(dest)
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
                if task not in DIRECT_TASKS and not params.get("input_path"):
                    self._json({"error": "input_path required"}, 400)
                    return
                if task == "video_download" and not params.get("url"):
                    self._json({"error": "url required"}, 400)
                    return
                job = _new_job(task, params)
                self._json({"job_id": job["id"], "status": job["status"]})
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
            elif path == "/api/comfy/endpoint":
                req = json.loads(self._body() or b"{}")
                self._json(apply_comfy_endpoint(req.get("host", COMFY_HOST),
                                                req.get("port", COMFY_PORT)))
            elif path == "/api/comfy/scan":
                self._json({"found": scan_comfy_ports()})
            else:
                self._json({"error": "unknown route"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_DELETE(self):  # noqa: N802
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
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if not job:
                self._json({"error": "no such job"}, 404)
                return
            if job.get("status") in ("pending", "running"):
                if not job.get("prompt_id"):
                    # still queued locally: mark cancelled; worker checks the
                    # terminal state before submit and skips it
                    job["status"] = "cancelled"
                    log(f"job {jid} cancelled before submit")
                    self._json({"ok": True})
                    return
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
                job["status"] = "cancelled"
            self._json({"ok": True})
        else:
            self._json({"error": "unknown route"}, 404)

    # ---- multipart (single file, field name "file")
    def _save_multipart(self) -> dict:
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=(.+)", ctype)
        if not m:
            raise ValueError("expected multipart/form-data")
        boundary = m.group(1).strip().encode()
        length = int(self.headers.get("Content-Length") or 0)
        stream = self.rfile
        # stream-parse: read headers line, then payload up to boundary
        line = stream.readline()
        if not line.startswith(b"--" + boundary):
            raise ValueError("bad multipart start")
        headers = {}
        while True:
            line = stream.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("utf-8", "replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        disp = headers.get("content-disposition", "")
        name_m = re.search(r'filename="([^"]*)"', disp)
        orig = name_m.group(1) if name_m else "upload.bin"
        safe = re.sub(r'[\\/:*?"<>|]+', "_", Path(orig).name) or "upload.bin"
        marker = b"\r\n--" + boundary
        buf = bytearray()
        chunk_sz = 1 << 20
        end = -1
        # bytes already consumed from the socket by the boundary/header reads:
        # only read what actually remains, or HTTP/1.1 keep-alive stalls forever
        preamble = len(b"--" + boundary) + len(line) + 1 + 2  # start line + CRLF terminator line
        remaining_body = max(0, length - preamble)
        # read until boundary found
        window = b""
        tmp_path = UPLOAD_DIR / (uuid.uuid4().hex + ".part")
        total = 0
        try:
            with open(tmp_path, "wb") as f:
                while remaining_body > 0:
                    chunk = stream.read(min(chunk_sz, remaining_body))
                    if not chunk:
                        break
                    total += len(chunk)
                    remaining_body -= len(chunk)
                    window += chunk
                    idx = window.find(marker)
                    if idx >= 0:
                        keep = window[:idx]
                        f.write(keep)
                        end = total - (len(window) - idx)
                        break
                    # keep last len(marker) bytes in window to catch boundary spanning chunks
                    safe_keep = len(window) - len(marker)
                    if safe_keep > 0:
                        f.write(window[:safe_keep])
                        window = window[safe_keep:]
                if end < 0:
                    raise ValueError("multipart end not found")
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
        log("FATAL: runtime/python junction missing — run: python tools/link_models.py")
        raise SystemExit(3)
    if not (COMFY_DIR / "main.py").is_file():
        log("FATAL: comfy junction missing — run: python tools/link_models.py")
        raise SystemExit(3)
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


if __name__ == "__main__":
    main()
