"""Musefish social-media video download backend.

Brings the ``video-download`` skill workflow into a ComfyUI node: yt-dlp for
platform URLs (YouTube, Bilibili, X/Twitter via proxy), WC encrypted-video
decryption through the vendor's own wasm module, and the
ftyp/ffprobe verification the skill requires for every finished file.

Module layout:

- yt-dlp path      : ``download_with_ytdlp`` (in-process python module)
- WeChat URL path  : ``download_wechat_encrypted`` + ``decrypt_wechat_video``
                     (raw HTTPS + node keystream daemon + XOR)
- verification     : ``verify_download`` (ftyp magic + ffprobe)
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .memory_guard import wait_for_memory

# ---------------------------------------------------------------- paths

_FFMPEG_DIR = Path(r"H:\ComfyUI_Mie_2026_V8.0\ffmpeg\bin")
FFMPEG = _FFMPEG_DIR / "ffmpeg.exe"
FFPROBE = _FFMPEG_DIR / "ffprobe.exe"

# Desktop is the fixed download target from the skill contract. When it does
# not exist (portable profiles, non-Administrator accounts) the node's
# output_directory fallback keeps downloads alive.
_DESKTOP_CANDIDATES = (
    Path(r"C:\Users\Administrator\Desktop"),
    Path(os.path.expanduser("~")) / "Desktop",
)


def default_download_dir() -> Path:
    for candidate in _DESKTOP_CANDIDATES:
        if candidate.is_dir():
            return candidate
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd()


def resolve_output_dir(value: str) -> Path:
    if value and str(value).strip():
        return Path(str(value).strip())
    return default_download_dir()


class DownloadError(RuntimeError):
    """Raised for every recoverable download/decrypt/verify failure."""


# ---------------------------------------------------------------- detection

WECHAT_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:[^/?#\"']*\.)?finder\.video\.qq\.com/251/20302/stodownload"
    r"|(?:https?://)?weixin\.qq\.com/sph/"
    r"|(?:https?://)?channels\.weixin\.qq\.com/",
    re.IGNORECASE
)
_X_URL_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/", re.IGNORECASE)


def looks_like_wechat_media(url: str) -> bool:
    # Only the video channel (20302) counts: 20304 stodownload links are
    # cover images whose keys do not decrypt videos.
    return bool(WECHAT_URL_PATTERN.search(url or ""))


_DOUYIN_SHORT_RE = re.compile(r"https?://v\.douyin\.com/[A-Za-z0-9_\-]+/?", re.IGNORECASE)
_DOUYIN_ID_RE = re.compile(r"/video/(\d{15,25})", re.IGNORECASE)
_DOUYIN_COOKIES = Path(__file__).with_name("_dy_cookies.txt")
_DOWNLOAD_CHUNK = 4 * 1024 * 1024


def looks_like_douyin_url(url: str) -> bool:
    return bool(re.search(r"douyin\.com|iesdouyin\.com", url or "", re.IGNORECASE))


def recommend_segments(url: str) -> int:
    """Best parallel-segment count for the URL's platform.

    Douyin: the resolved CDN (aweme.snssdk.com) allows ranged parallelism.
    YouTube/Bilibili: yt-dlp fragments download concurrently (DASH/HLS).
    Unknown platforms: 4 is a safe default; 1 only when explicitly chosen.
    """
    url = url or ""
    if looks_like_douyin_url(url):
        return 8
    if detect_platform(url) in ("youtube", "bilibili"):
        return 8
    return 4


def resolve_douyin_share(text: str, max_seconds: float = 240.0) -> tuple[str, str] | None:
    """Resolve a Douyin share text / short link to (direct_mp4_url, title).

    Douyin's web API needs request signatures yt-dlp cannot forge, but the
    mobile share page (iesdouyin.com/share/video/<id>) embeds the whole
    item JSON — including an unencrypted play_addr CDN URL — as long as we
    present the douyin.com cookies captured from the user's browser
    (``_dy_cookies.txt``, refreshed by the bundled CDP capture script).

    Returns None when the text carries no douyin link or resolution fails.
    """
    text = text or ""
    short = _DOUYIN_SHORT_RE.search(text)
    explicit = _DOUYIN_ID_RE.search(text)
    video_id = None
    if explicit:
        video_id = explicit.group(1)
    elif short:
        video_id = None  # resolve via redirect below
    else:
        return None

    if not video_id:
        import urllib.request

        req = urllib.request.Request(
            short.group(0), headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            final = resp.url
        match = _DOUYIN_ID_RE.search(final)
        if not match:
            return None
        video_id = match.group(1)

    import urllib.request

    if not _DOUYIN_COOKIES.exists():
        raise DownloadError(
            "Douyin requires browser cookies (_dy_cookies.txt missing). "
            "Run the bundled capture script once while logged into douyin."
        )
    yt_dlp = _ytdlp_module()
    jar = yt_dlp.utils.YoutubeDLCookieJar(str(_DOUYIN_COOKIES))
    jar.load(ignore_discard=True, ignore_expires=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    share = f"https://www.iesdouyin.com/share/video/{video_id}"
    req = urllib.request.Request(
        share,
        headers={
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15"
        },
    )
    deadline = time.monotonic() + max_seconds
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    with opener.open(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    match = re.search(
        r"_ROUTER_DATA\s*=\s*(\{.+?\})\s*;?\s*</script>", body, re.DOTALL
    )
    if not match:
        return None
    data = json.loads(match.group(1))
    items = (
        data.get("loaderData", {})
        .get("video_(id)/page", {})
        .get("videoInfoRes", {})
        .get("item_list")
        or []
    )
    if not items:
        return None
    item = items[0]
    play = item.get("video", {}).get("play_addr", {})
    urls = play.get("url_list") or []
    if not urls:
        return None
    # playwm serves a watermarked 720p transcode; play/ is the clean file.
    direct = urls[0].replace("/playwm/", "/play/")
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", item.get("desc") or video_id)[:80]
    return direct, title


def _stream_to_file(
    url: str,
    target: Path,
    cookies_file: Path | None,
    reporter: "ProgressReporter",
    parallel: int = 1,
) -> None:
    """Download a direct media URL to ``target``.

    ``parallel`` > 1 splits the file into that many ranged segments fetched
    concurrently (CDN servers allow it; single-connection throttling is the
    usual long-video bottleneck), then assembles them. ``parallel == 1``
    streams sequentially. Cookie auth via ``cookies_file`` (douyin).
    """
    import urllib.request

    def _opener() -> urllib.request.OpenerDirector:
        handlers = []
        if cookies_file is not None and cookies_file.exists():
            yt_dlp = _ytdlp_module()
            jar = yt_dlp.utils.YoutubeDLCookieJar(str(cookies_file))
            jar.load(ignore_discard=True, ignore_expires=True)
            handlers.append(urllib.request.HTTPCookieProcessor(jar))
        return urllib.request.build_opener(*handlers)

    opener = _opener()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        },
    )
    with opener.open(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else 0

    parallel = max(1, min(16, int(parallel)))
    if total > _DOWNLOAD_CHUNK and parallel > 1:
        # Ranged segments fetched by a thread pool, assembled in order.
        from concurrent.futures import ThreadPoolExecutor

        bounds = []
        seg = total // parallel
        for i in range(parallel):
            start = i * seg
            end = (total - 1) if i == parallel - 1 else (start + seg - 1)
            bounds.append((start, end))

        done = [0]

        def fetch(span):
            start, end = span
            status, _, data = _http_get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=180.0
            )
            return start, data

        parts: dict[int, bytes] = {}
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for start, data in pool.map(fetch, bounds):
                parts[start] = data
                done[0] += 1
                reporter.fraction(min(0.95, done[0] / parallel))

        with open(target, "wb") as sink:
            for start in sorted(parts):
                sink.write(parts[start])
        received = total
    else:
        received = 0
        with opener.open(request, timeout=60) as response, open(target, "wb") as sink:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                sink.write(chunk)
                received += len(chunk)
                if total:
                    reporter.fraction(min(0.95, received / total))
    if total and received < total:
        raise DownloadError(
            f"incomplete download: {received}/{total} bytes from {url[:100]}"
        )


def looks_like_x_url(url: str) -> bool:
    return bool(_X_URL_PATTERN.search(url or ""))


# Platform registry: combo order = UI order. "auto" first so the default
# never rejects anything; every other entry carries the URL patterns that
# positively identify the platform. WeChat Channels is intentionally NOT a
# selector option: those URLs (and share links) belong to the dedicated
# MusefishWeChatChannels node, which owns the decrypt pipeline; Auto still
# detects them here only to redirect the user with a clear error.
_PLATFORM_PATTERNS: dict[str, list[re.Pattern]] = {
    "youtube": [
        re.compile(r"(?:youtube\.com/(?:watch|shorts|live|embed)|youtu\.be/)", re.IGNORECASE),
    ],
    "bilibili": [
        re.compile(r"(?:bilibili\.com/(?:video|festival|bangumi)|b23\.tv/)", re.IGNORECASE),
    ],
    "douyin": [
        re.compile(r"(?:douyin\.com/(?:video|note)|v\.douyin\.com/|iesdouyin\.com/)", re.IGNORECASE),
    ],
    "xiaohongshu": [
        re.compile(r"(?:xiaohongshu\.com/(?:explore|discovery|user)|xhslink\.com/)", re.IGNORECASE),
    ],
    "x_twitter": [
        re.compile(r"(?:^|//)(?:www\.)?(?:x|twitter)\.com/.+/status/", re.IGNORECASE),
    ],
}
_PLATFORMS = ["auto"] + list(_PLATFORM_PATTERNS)
_PLATFORM_LABELS = {
    "auto": "Auto",
    "youtube": "YouTube",
    "bilibili": "Bilibili",
    "douyin": "Douyin",
    "xiaohongshu": "Xiaohongshu",
    "x_twitter": "X / Twitter",
}
_WECHAT_PATTERNS = [
    re.compile(r"finder\.video\.qq\.com/251/20302/stodownload", re.IGNORECASE),
    re.compile(r"weixin\.qq\.com/sph/", re.IGNORECASE),
    re.compile(r"channels\.weixin\.qq\.com", re.IGNORECASE),
]


def detect_platform(url: str) -> str | None:
    """Return the platform key whose patterns match ``url``, else None."""
    url = url or ""
    for platform, patterns in _PLATFORM_PATTERNS.items():
        if any(pattern.search(url) for pattern in patterns):
            return platform
    return None


def assert_platform_matches(url: str, platform: str) -> None:
    """Raise when the user-forced platform cannot have produced ``url``."""
    if platform == "auto" or not url:
        return
    expected = _PLATFORM_PATTERNS[platform]
    if not any(pattern.search(url) for pattern in expected):
        if detect_platform(url) is None:
            detail = (
                "the URL matches no supported platform either; leave the "
                "selector on auto to try yt-dlp anyway"
            )
        else:
            actual = detect_platform(url)
            detail = f"the URL is {_PLATFORM_LABELS[actual]}"
        raise DownloadError(
            f"platform mismatch: selector is {_PLATFORM_LABELS[platform]} "
            f"but {detail}. URL: {url[:120]}"
        )


def _sanitize_filename(name: str, fallback: str = "musefish_video") -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip()) or fallback
    return name[:120]


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


# ---------------------------------------------------------------- progress


class ProgressReporter:
    """Throttled progress publication shared by all backends.

    ComfyUI's ProgressBar covers a single node execution with an integer
    value range; the downloader calls :meth:`fraction` from download chunks
    and decrypt stages, so ``0.0 .. 1.0`` maps onto that range.
    """

    def __init__(self, bar=None, interval: float = 0.5):
        self._bar = bar
        self._interval = interval
        self._last = 0.0

    def fraction(self, value: float, note: str = "") -> None:
        value = max(0.0, min(1.0, float(value)))
        if self._bar is None:
            return
        now = time.monotonic()
        if value < 1.0 and now - self._last < self._interval:
            return
        self._last = now
        total = getattr(self._bar, "total", None) or 100
        try:
            self._bar.update_absolute(int(value * total))
        except Exception:
            pass


# ---------------------------------------------------------------- yt-dlp


def _ytdlp_module():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise DownloadError(
            "yt-dlp is not installed in the ComfyUI Python environment "
            f"({error}); install it with `<python> -m pip install yt-dlp`"
        ) from error


_YTDLP_PROXY = "http://127.0.0.1:7897"  # Clash Verge mixed port (local)
_YTDLP_COOKIES = Path(__file__).with_name("_yt_cookies.txt")  # Netscape format


def _proxy_reachable(proxy: str, timeout: float = 2.0) -> bool:
    try:
        host, port = proxy.rsplit(":", 1)
        host = host.split("//", 1)[-1]
        with __import__("socket").create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _ytdlp_progress_hook(reporter: ProgressReporter, start: float, end: float):
    state = {"fraction": 0.0}

    def hook(entry):
        total = entry.get("total_bytes") or entry.get("total_bytes_estimate") or 0
        done = entry.get("downloaded_bytes") or 0
        if total > 0:
            state["fraction"] = done / total
        reporter.fraction(start + (end - start) * state["fraction"], entry.get("status", ""))

    return hook


def download_with_ytdlp(
    url: str,
    output_dir: Path,
    reporter: ProgressReporter,
    quality: str = "best",
    log: list[str] | None = None,
    cookies_file: Path | None = None,
    parallel_fragments: int = 8,
) -> tuple[Path, dict]:
    """Download ``url`` with the in-process yt-dlp module.

    Returns ``(path, info)``; ``info`` holds title/duration/width/height.
    X/Twitter URLs route through the skill's fixed proxy when reachable and
    fall back to a direct attempt when it is not.
    """
    yt_dlp = _ytdlp_module()

    target = _unique_path(output_dir, "musefish_download", ".mp4")
    format_map = {
        "best": "bestvideo*+bestaudio/best",
        "1080p": "bestvideo*[height<=1080]+bestaudio/best[height<=1080]/best",
        "720p": "bestvideo*[height<=720]+bestaudio/best[height<=720]/best",
        "480p": "bestvideo*[height<=480]+bestaudio/best[height<=480]/best",
    }
    opts = {
        "outtmpl": str(target),
        "format": format_map.get(quality, format_map["best"]),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "progress_hooks": [_ytdlp_progress_hook(reporter, 0.0, 0.9)],
        # Keep ffmpeg next to the plugin toolchain instead of PATH dependence.
        "ffmpeg_location": str(FFMPEG) if FFMPEG.exists() else shutil.which("ffmpeg"),
        # Concurrent fetch of HLS/DASH fragments and video+audio streams:
        # the single-connection speed of long YouTube/Bilibili videos is
        # usually the bottleneck; 8 parallel fragments multiply throughput.
        "concurrent_fragment_downloads": max(1, int(parallel_fragments)),
    }
    # YouTube and other blocked sites need the local proxy; route when it
    # is up (X always, YouTube when reachable — both are bot-walled from
    # CN/datacenter IPs without it).
    if looks_like_x_url(url) and _proxy_reachable(_YTDLP_PROXY):
        opts["proxy"] = _YTDLP_PROXY
        if log is not None:
            log.append(f"X/Twitter URL: routing via proxy {_YTDLP_PROXY}")
    elif re.search(r"(?:youtube\.com|youtu\.be)", url, re.IGNORECASE):
        if _proxy_reachable(_YTDLP_PROXY):
            opts["proxy"] = _YTDLP_PROXY
            if log is not None:
                log.append(f"YouTube URL: routing via proxy {_YTDLP_PROXY}")
        else:
            raise DownloadError(
                "YouTube requires the local proxy but it is not reachable "
                f"({_YTDLP_PROXY}). Start Clash/verge and retry."
            )
    # YouTube bot-walls datacenter/region IPs; browser cookies solve it.
    # The cookie file is refreshed by the bundled capture script (CDP).
    if _YTDLP_COOKIES.exists() and re.search(
        r"(?:youtube\.com|youtu\.be)", url, re.IGNORECASE
    ):
        opts["cookiefile"] = str(_YTDLP_COOKIES)
    # Douyin cookies: passed via cookiefile when present (helps other
    # douyin URLs the resolver does not handle).
    if cookies_file is not None and cookies_file.exists():
        opts["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        requested = info.get("requested_downloads") or []
        for entry in requested:
            path = entry.get("filepath") or entry.get("filename")
            if path:
                return Path(path), info
        # Some extractors report the final file only through prepare_filename.
        final = Path(ydl.prepare_filename(info))
        if final.suffix != ".mp4" and final.with_suffix(".mp4").exists():
            final = final.with_suffix(".mp4")
        if final.exists():
            return final, info
        raise DownloadError(f"yt-dlp finished but no output file was found for {url}")


# ---------------------------------------------------------------- HTTP


def _http_get(
    url: str,
    headers: dict | None = None,
    timeout: float = 60.0,
    proxy: str | None = None,
) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    with opener.open(request, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


def _http_range(
    url: str,
    start: int,
    end: int,
    headers: dict | None = None,
    timeout: float = 120.0,
    proxy: str | None = None,
) -> tuple[int, bytes]:
    """Fetch ``url[start..end]``. Returns ``(status, body)``; status 200 means
    the server ignored the Range header and ``body`` is the whole file."""
    merged = {"Range": f"bytes={start}-{end}"}
    merged.update(headers or {})
    request = urllib.request.Request(url, headers=merged)
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    with opener.open(request, timeout=timeout) as response:
        _capture_headers(dict(response.headers))
        return response.status, response.read()


# ------------------------------------------------------------------ WC


_WECHAT_ENC_BYTES = 131072  # only the first 128 KiB of a Channels video is encrypted

_FLOW_ENTRY_PATTERN = re.compile(
    r"https?://finder\.video\.qq\.com/251/20302/stodownload\?encfilekey="
    r"([A-Za-z0-9_\-]+)[^\"]{0,700}?&m=[^\",]*?,(\d{6,12}),(\d+),(\d+),"
    r"\[v?\d[^\]]*\]\[Flow\]\[[^\]]*\]\[timestamp:(\d{13})\]"
)
# The escaped JSON continuation after "[timestamp:...]" holds the FULL
# tokened URL of that same playback row: [{\"url\":\"https://finder...\".
# Each playback of a feed item writes a new row; the tokened URL pins the
# content the CDN served for THAT token, so per encfilekey the newest
# row's URL is the one that serves the most recently played video.
_URL_ROW_BRIDGE = '[{\\"url\\":\\"'
_FLOW_URL_TS_PATTERN = re.compile(
    r"\[timestamp:(\d{13})\]" + re.escape(_URL_ROW_BRIDGE) +
    r"(https?://finder\.video\.qq\.com/251/20302/stodownload\?encfilekey="
    r"[A-Za-z0-9_\-]+[A-Za-z0-9_\-%&=\.:/~\+]{50,2500})",
    re.DOTALL,
)
_COMPLETE_URL_PATTERN = re.compile(
    r"https?://finder\.video\.qq\.com/251/20302/stodownload\?encfilekey="
    r"[A-Za-z0-9_\-]+[A-Za-z0-9_\-%&=\.:/~\+]{300,2500}"
)


class KeystreamDaemon:
    """Warm node-side ``keystream_daemon.js`` process.

    The wasm module takes seconds to initialize and milliseconds to answer,
    so the process is kept alive for the life of the ComfyUI run and is
    restarted transparently if node exits.
    """

    _GENERATE_TIMEOUT = 60.0
    _READY_TIMEOUT = 150.0

    def __init__(self):
        self._proc = None
        self._next_id = 1

    def _start(self):
        import sys

        script = Path(__file__).resolve().parent / "wxdec_toolchain" / "keystream_daemon.js"
        if not script.exists():
            raise DownloadError(f"keystream daemon missing: {script}")
        node = shutil.which("node") or r"C:\nodejs\node.exe"
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [node, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=creationflags,
        )

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None
        if self._proc is None:
            self._start()
        return self._proc

    def _call(self, request: dict, timeout: float) -> dict:
        proc = self._ensure()
        request = dict(request)
        request["id"] = self._next_id
        self._next_id += 1
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise DownloadError(f"keystream daemon pipe broke: {error}") from error
        deadline = time.monotonic() + timeout
        while True:
            line = proc.stdout.readline()
            if not line:
                raise DownloadError("keystream daemon exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request["id"]:
                continue
            return message

    def generate(self, decode_key: str) -> bytes:
        """Return the 128 KiB keystream for ``decode_key`` (cached daemon-side)."""
        deadline = time.monotonic() + self._READY_TIMEOUT
        while True:
            reply = self._call({"op": "ping"}, timeout=self._READY_TIMEOUT)
            if reply.get("ready"):
                break
            if time.monotonic() > deadline:
                raise DownloadError(f"wasm runtime not ready: {reply.get('error')}")
            time.sleep(0.5)
        reply = self._call({"op": "generate", "key": str(decode_key)}, timeout=self._GENERATE_TIMEOUT)
        if not reply.get("ok"):
            raise DownloadError(f"keystream generation failed: {reply.get('error')}")
        return base64.b64decode(reply["keystream"])

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._call({"op": "shutdown"}, timeout=5.0)
            except DownloadError:
                self._proc.kill()
        self._proc = None


_SHARED_DAEMON: KeystreamDaemon | None = None


def get_shared_daemon() -> KeystreamDaemon:
    global _SHARED_DAEMON
    if _SHARED_DAEMON is None:
        _SHARED_DAEMON = KeystreamDaemon()
    return _SHARED_DAEMON


def _decrypt_xor(encrypted: bytes, keystream: bytes) -> bytes:
    head = bytes(a ^ b for a, b in zip(encrypted[: len(keystream)], keystream))
    return head + encrypted[len(keystream):]


def verify_encrypted_header(encrypted_head: bytes, keystream: bytes) -> bool:
    """Skill gate: decrypting bytes 4..8 with the keystream must yield 'ftyp'."""
    if len(encrypted_head) < 12 or len(keystream) < 12:
        return False
    check = bytes(a ^ b for a, b in zip(encrypted_head[:12], keystream[:12]))
    return check[4:8] == b"ftyp"


def download_wechat_encrypted(
    url: str,
    output_dir: Path,
    decode_key: str,
    reporter: ProgressReporter,
    log: list[str] | None = None,
) -> tuple[Path, dict]:
    """Download one Channels CDN ciphertext URL and decrypt it in place.

    ``decode_key`` is bound to this exact URL (the skill contract); the caller
    obtains both from a Channels response JSON or the memory-scan node.
    """
    if not decode_key or not str(decode_key).strip().isdigit():
        raise DownloadError(
            "WC URLs need the matching numeric decode_key "
            "(URL-bound; the node recovers it automatically from the "
            "logged-in WC desktop client running locally, "
            "otherwise the key does not exist and decryption is impossible)"
        )
    decode_key = str(decode_key).strip()
    if "://" not in url:
        full_url = "https://" + url
    else:
        full_url = url

    # Validate the key against a small ranged fetch before pulling everything.
    daemon = get_shared_daemon()
    keystream = daemon.generate(decode_key)
    if log is not None:
        log.append(f"keystream ready for decode_key {decode_key} ({len(keystream)} bytes)")

    _probe_status, probe_head = _http_range(full_url, 0, 262143, timeout=60.0)
    if not verify_encrypted_header(probe_head, keystream):
        raise DownloadError(
            "decode_key does not decrypt this URL (ftyp mismatch). "
            "Keys are one-to-one with the CDN URL: re-fetch the URL/key pair."
        )
    if log is not None:
        log.append("decode_key verified against ciphertext header (ftyp ok)")

    stem = _sanitize_filename(f"wx_channels_{decode_key}")
    target = _unique_path(output_dir, stem, ".mp4")

    # Range support is decided by the Content-Range header, not the status
    # value: file:// and some proxies answer with status None or 200.
    probe_headers = _last_http_headers or {}
    content_range = next(
        (v for k, v in probe_headers.items() if k.lower() == "content-range"), None
    )

    # Stream straight to disk: only the first 128 KiB need XOR, so peak
    # memory stays at one chunk no matter the file size. Buffering the whole
    # ciphertext plus a plaintext copy tripled RAM for nothing.
    received = 0
    ranged = bool(content_range and "/" in content_range)
    with open(target, "wb") as sink:
        def _write_chunk(data: bytes) -> None:
            """Append data, XOR-decrypting whatever falls in the first 128 KiB."""
            nonlocal received
            offset = received
            sink.seek(offset)
            if offset < _WECHAT_ENC_BYTES:
                cut = min(len(data), _WECHAT_ENC_BYTES - offset)
                sink.write(bytes(a ^ b for a, b in
                                 zip(data[:cut], keystream[offset:offset + cut])))
                sink.seek(offset + cut)
                sink.write(data[cut:])
            else:
                sink.write(data)
            received += len(data)

        if ranged:
            try:
                total = int(content_range.rsplit("/", 1)[1])
            except ValueError:
                total = None
            _write_chunk(probe_head)
            while total is None or received < total:
                wait_for_memory(log=log)
                chunk_start = received
                chunk_end = min(chunk_start + _DOWNLOAD_CHUNK - 1,
                                (total or chunk_start + _DOWNLOAD_CHUNK) - 1)
                _status, data = _http_range(full_url, chunk_start, chunk_end, timeout=180.0)
                if not data:
                    break
                if len(data) > (chunk_end - chunk_start + 1):
                    # Server ignored this Range too; keep only unseen bytes.
                    data = data[: (chunk_end - chunk_start + 1)]
                _write_chunk(data)
                if total:
                    reporter.fraction(0.05 + 0.85 * received / total)
                if received >= total:
                    break
                if len(data) < (chunk_end - chunk_start + 1):
                    break
        else:
            # Server ignored Range: the probe response IS the whole file.
            sink.write(probe_head)
            received = len(probe_head)

    # Rewrite the decrypted head in one bounded 128 KiB read-modify-write.
    # The ranged stream already XOR-decrypted the head inside _write_chunk;
    # only the non-ranged fallback (200, raw whole-body write) still needs
    # the explicit head decryption.
    if not ranged:
        with open(target, "r+b") as fixup:
            head = fixup.read(_WECHAT_ENC_BYTES)
            fixup.seek(0)
            fixup.write(_decrypt_xor(head, keystream))
    if log is not None:
        log.append(f"written {received:,} bytes; head decrypted "
                   f"(only the first {_WECHAT_ENC_BYTES} bytes were XORed)")
    reporter.fraction(0.92)

    info = {
        "title": target.stem,
        "decode_key": decode_key,
        "verified_key": True,
    }
    return target, info


_last_http_headers: dict | None = None


def _capture_headers(headers: dict):
    global _last_http_headers
    _last_http_headers = headers


def decrypt_wechat_video(encrypted_path: Path, output_dir: Path, decode_key: str,
                         log: list[str] | None = None) -> Path:
    """Decrypt an already-downloaded ciphertext file with ``decode_key``.

    Streams: a 128 KiB head read-modify-write plus a chunked tail copy, so
    a multi-GB file never loads into RAM.
    """
    daemon = get_shared_daemon()
    keystream = daemon.generate(str(decode_key).strip())
    source = Path(encrypted_path)
    stem = _sanitize_filename(f"{source.stem}_decrypted")
    target = _unique_path(Path(output_dir), stem, ".mp4")
    with source.open("rb") as src, open(target, "wb") as dst:
        head = src.read(_WECHAT_ENC_BYTES)
        if not verify_encrypted_header(head[:4096], keystream):
            raise DownloadError("decode_key does not decrypt this file (ftyp mismatch)")
        dst.write(_decrypt_xor(head, keystream))
        while True:
            wait_for_memory(log=log)
            chunk = src.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            dst.write(chunk)
    if log is not None:
        log.append(f"decrypted {source.name} -> {target.name}")
    return target


# ------------------------------------------------- memory scan (Channels)


_MEMORY_PROTECTIONS = {0x02, 0x04, 0x08, 0x20, 0x40, 0x80}


def scan_wechat_decode_keys(max_seconds: float = 300.0) -> list[dict]:
    """Scan running WC client processes for Flow-log URL/decode_key pairs.

    Mirrors the skill's x64 user-mode scan: MEM_COMMIT private regions only,
    reads clamped to VirtualQueryEx region bounds, read access bits only.
    Returns newest-first candidates; the caller re-verifies via ftyp XOR.
    """
    import ctypes
    from ctypes import wintypes

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("RegionSize", ctypes.c_size_t),  # offset 24 on x64
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_VM_READ = 0x0010
    PROCESS_QUERY_INFORMATION = 0x0400
    MEM_COMMIT = 0x1000
    MEM_PRIVATE = 0x20000

    def process_ids():
        count = 4096
        while True:
            buf = (wintypes.DWORD * count)()
            returned = wintypes.DWORD()
            if not enum_processes(buf, ctypes.sizeof(buf), ctypes.byref(returned)):
                return []
            ids = buf[: returned.value // ctypes.sizeof(wintypes.DWORD)]
            if len(ids) < count:
                return ids
            count *= 2

    # EnumProcesses is exported by PSAPI on modern Windows (kernel32 only
    # forwards it; the export is not always resolvable there).
    psapi = ctypes.WinDLL("psapi")
    if not hasattr(psapi, "EnumProcesses"):
        raise DownloadError("EnumProcesses unavailable; memory scan requires Windows x64")
    enum_processes = psapi.EnumProcesses

    targets = []
    for pid in process_ids():
        try:
            handle = kernel32.OpenProcess(
                PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, int(pid)
            )
        except Exception:
            continue
        if not handle:
            continue
        try:
            name_buffer = ctypes.create_unicode_buffer(260)
            psapi = ctypes.WinDLL("psapi")
            if psapi.GetModuleBaseNameW(handle, None, name_buffer, 260):
                base = name_buffer.value.lower()
                if base in {"weixin.exe", "wechatappex.exe", "xworker.exe"}:
                    targets.append((int(pid), base))
        finally:
            kernel32.CloseHandle(handle)

    if not targets:
        raise DownloadError(
            "no running WC client processes found (Weixin.exe / WeChatAppEx.exe). "
            "WC downloads require the logged-in WC desktop client "
            "running locally: start it, play the target video for a few "
            "seconds, keep it running, and rescan."
        )

    scan_deadline = time.monotonic() + max_seconds
    candidates: dict[str, dict] = {}
    chunk_size = 8 * 1024 * 1024

    for pid, name in targets:
        if time.monotonic() > scan_deadline:
            break
        handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            continue
        try:
            address = 0
            while address < 0x7FFFFFFEFFFF and time.monotonic() < scan_deadline:
                mbi = MEMORY_BASIC_INFORMATION()
                result = kernel32.VirtualQueryEx(
                    handle,
                    ctypes.c_void_p(address),
                    ctypes.byref(mbi),
                    ctypes.sizeof(mbi),
                )
                if not result:
                    break
                region_base = int(mbi.BaseAddress or 0)
                region_size = int(mbi.RegionSize)
                if region_size == 0:
                    break
                region_end = region_base + region_size
                if (
                    mbi.State == MEM_COMMIT
                    and mbi.Type == MEM_PRIVATE
                    and (mbi.Protect & 0xFF) in _MEMORY_PROTECTIONS
                    and region_size <= 64 * 1024 * 1024
                ):
                    offset = 0
                    while offset < region_size and time.monotonic() < scan_deadline:
                        take = min(chunk_size, region_size - offset)
                        buffer = ctypes.create_string_buffer(take)
                        read = ctypes.c_size_t()
                        if not kernel32.ReadProcessMemory(
                            handle,
                            ctypes.c_void_p(region_base + offset),
                            buffer,
                            take,
                            ctypes.byref(read),
                        ):
                            break
                        data = buffer.raw[: read.value]
                        text = data.decode("utf-8", errors="ignore")
                        for match in _FLOW_ENTRY_PATTERN.finditer(text):
                            url_head, key, errno, flag, timestamp = match.groups()
                            if errno != "0":
                                continue
                            entry = candidates.get(key)
                            if entry is None or int(timestamp) > int(entry["timestamp"]):
                                candidates[key] = {
                                    "decode_key": key,
                                    "url_head": url_head,
                                    "timestamp": timestamp,
                                    "pid": pid,
                                    "process": name,
                                }
                        # Harvest complete tokened URLs (with row
                        # timestamps) into the persistent cache: the
                        # kvReport rows holding them are compacted by the
                        # client within minutes, but a cached URL stays
                        # valid until its token expires (~hours).
                        _load_url_cache()
                        for match in _FLOW_URL_TS_PATTERN.finditer(text):
                            url, ts = match.group(2), int(match.group(1))
                            marker = re.match(
                                r"https?://[^/]+/251/20302/stodownload\?encfilekey=([A-Za-z0-9_\-]+)",
                                url,
                            )
                            if not marker:
                                continue
                            head = marker.group(1)[:16]
                            prior = _COMPLETE_URL_CACHE.get(head)
                            if prior is None or ts > prior[1]:
                                _COMPLETE_URL_CACHE[head] = (url, ts)
                                _store_url_cache()
                        offset += take
                address = region_end
        finally:
            kernel32.CloseHandle(handle)

    ordered = sorted(candidates.values(), key=lambda item: -int(item["timestamp"]))
    return ordered


# Complete tokened URLs seen in Flow rows: marker -> (url, row_timestamp).
# Memory compaction erases the rows within minutes; the cached URL remains
# valid until its token expires, so complete_wechat_url falls back to it.
# Persisted to disk so the cache survives ComfyUI restarts.
_COMPLETE_URL_CACHE: dict[str, tuple[str, int]] = {}
_URL_CACHE_PATH = Path(__file__).with_name("_wc_url_cache.json")


def _load_url_cache() -> None:
    global _COMPLETE_URL_CACHE
    if _COMPLETE_URL_CACHE:
        return
    try:
        raw = json.loads(_URL_CACHE_PATH.read_text(encoding="utf-8"))
        _COMPLETE_URL_CACHE = {
            marker: (entry["url"], int(entry["ts"])) for marker, entry in raw.items()
        }
    except (OSError, ValueError, AttributeError, TypeError):
        _COMPLETE_URL_CACHE = {}


def _store_url_cache() -> None:
    try:
        payload = {
            marker: {"url": url, "ts": ts}
            for marker, (url, ts) in _COMPLETE_URL_CACHE.items()
        }
        _URL_CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def complete_wechat_url(candidate: dict, max_seconds: float = 240.0) -> str:
    """Re-scan for the tokened full URL matching a candidate's encfilekey.

    The Flow log only stores a truncated URL head; the tokened 1000+ char
    URL lives elsewhere in the same process memories and must be captured
    while the CDN entry is fresh.
    """
    import ctypes
    from ctypes import wintypes

    encfilekey = candidate["url_head"].split("encfilekey=", 1)[-1]
    marker = encfilekey[:16] if len(encfilekey) >= 16 else encfilekey
    if not marker:
        raise DownloadError("candidate has no encfilekey to match against")

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_VM_READ = 0x0010
    PROCESS_QUERY_INFORMATION = 0x0400
    MEM_COMMIT = 0x1000
    MEM_PRIVATE = 0x20000

    handle = kernel32.OpenProcess(
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, int(candidate["pid"])
    )
    if not handle:
        raise DownloadError(f"cannot open pid {candidate['pid']} for full-URL scan")
    deadline = time.monotonic() + max_seconds
    best: str | None = None
    try:
        address = 0
        chunk_size = 8 * 1024 * 1024
        while address < 0x7FFFFFFEFFFF and time.monotonic() < deadline:
            mbi = MEMORY_BASIC_INFORMATION()
            if not kernel32.VirtualQueryEx(
                handle, ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)
            ):
                break
            region_base = int(mbi.BaseAddress or 0)
            region_size = int(mbi.RegionSize)
            if region_size == 0:
                break
            region_end = region_base + region_size
            if (
                mbi.State == MEM_COMMIT
                and mbi.Type == MEM_PRIVATE
                and (mbi.Protect & 0xFF) in _MEMORY_PROTECTIONS
                and region_size <= 64 * 1024 * 1024
            ):
                offset = 0
                while offset < region_size and time.monotonic() < deadline:
                    take = min(chunk_size, region_size - offset)
                    buffer = ctypes.create_string_buffer(take)
                    read = ctypes.c_size_t()
                    if not kernel32.ReadProcessMemory(
                        handle,
                        ctypes.c_void_p(region_base + offset),
                        buffer,
                        take,
                        ctypes.byref(read),
                    ):
                        break
                    data = buffer.raw[: read.value].decode("utf-8", errors="ignore")
                    for match in _FLOW_URL_TS_PATTERN.finditer(data):
                        ts = int(match.group(1))
                        url = match.group(2)
                        # Identity tokens: encfilekey + basedata + sign.
                        # taskid/_pUid_ exist only in some CDN URL shapes
                        # (newer feeds omit _pUid_ entirely), so requiring
                        # them hides valid tokened URLs of newer videos.
                        if marker in url and all(
                            token in url for token in ("basedata=", "sign=")
                        ):
                            if best is None or ts > best[1]:
                                best = (url, ts)
                    offset += take
            address = region_end
    finally:
        kernel32.CloseHandle(handle)

    if not best:
        _load_url_cache()
        cached = _COMPLETE_URL_CACHE.get(marker[:16])
        if cached:
            return cached[0] if cached[0].startswith("http") else "https://" + cached[0]
        raise DownloadError(
            "no tokened full URL found in memory — replay the video for a few "
            "seconds in the WC client and rescan (URLs expire with their token)"
        )
    url = best[0]
    return url if url.startswith("http") else "https://" + url


def resolve_share_link(link: str, max_seconds: float = 120.0) -> dict | None:
    """Bind a Channels share link (weixin.qq.com/sph/<id>) to its record.

    Two chained memory searches in the WC client processes:
      1. sph id -> object_id: feed responses store
         ``"feed_h5_url":".../sph/<id>" ... "object_id":"<digits>"``.
      2. object_id -> encfilekey: Flow-log rows store
         ``<object_id>,https://finder.video.qq.com/251/20302/stodownload?
         encfilekey=<key>...``.
    Returns the candidate dict whose url_head carries that encfilekey, or
    None when any hop fails (caller falls back to newest-record behaviour).
    """
    link = (link or "").strip()
    match = re.search(r"/sph/([A-Za-z0-9_\-]+)", link)
    if not match:
        return None
    sph_id = match.group(1)
    if not sph_id or len(sph_id) < 10:
        return None

    import ctypes
    from ctypes import wintypes

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi")
    PROCESS_VM_READ = 0x0010
    PROCESS_QUERY_INFORMATION = 0x0400

    targets = []
    count = 4096
    while True:
        buf = (wintypes.DWORD * count)()
        returned = wintypes.DWORD()
        if not psapi.EnumProcesses(buf, ctypes.sizeof(buf), ctypes.byref(returned)):
            return None
        ids = buf[: returned.value // ctypes.sizeof(wintypes.DWORD)]
        if len(ids) < count:
            break
        count *= 2
    for pid in ids:
        handle = kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, int(pid)
        )
        if not handle:
            continue
        try:
            name_buffer = ctypes.create_unicode_buffer(260)
            if psapi.GetModuleBaseNameW(handle, None, name_buffer, 260):
                base = name_buffer.value.lower()
                if base in {"weixin.exe", "wechatappex.exe", "xworker.exe"}:
                    targets.append(int(pid))
        finally:
            kernel32.CloseHandle(handle)

    def _memory_search(patterns: list[re.Pattern], collect) -> set:
        """Walk WC client memories once, applying every pattern; ``collect``
        merges each pattern's matches into one set."""
        found: set = set()
        chunk_size = 8 * 1024 * 1024
        scan_deadline = time.monotonic() + max_seconds
        for pid in targets:
            if time.monotonic() > scan_deadline:
                break
            handle = kernel32.OpenProcess(
                PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid
            )
            if not handle:
                continue
            try:
                address = 0
                while address < 0x7FFFFFFEFFFF and time.monotonic() < scan_deadline:
                    mbi = MEMORY_BASIC_INFORMATION()
                    if not kernel32.VirtualQueryEx(
                        handle, ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)
                    ):
                        break
                    region_base = int(mbi.BaseAddress or 0)
                    region_size = int(mbi.RegionSize)
                    if region_size == 0:
                        break
                    if (
                        mbi.State == 0x1000
                        and mbi.Type == 0x20000
                        and (mbi.Protect & 0xFF) in _MEMORY_PROTECTIONS
                        and region_size <= 64 * 1024 * 1024
                    ):
                        offset = 0
                        while offset < region_size and time.monotonic() < scan_deadline:
                            take = min(chunk_size, region_size - offset)
                            buffer = ctypes.create_string_buffer(take)
                            read = ctypes.c_size_t()
                            if not kernel32.ReadProcessMemory(
                                handle,
                                ctypes.c_void_p(region_base + offset),
                                buffer,
                                take,
                                ctypes.byref(read),
                            ):
                                break
                            text = buffer.raw[: read.value].decode("utf-8", errors="ignore")
                            for pattern in patterns:
                                for hit in pattern.finditer(text):
                                    collect(found, hit)
                            offset += take
                    address = region_base + region_size
            finally:
                kernel32.CloseHandle(handle)
        return found

    # Hop 1: sph id -> object id(s).
    id_pattern = re.compile(
        r"sph/" + re.escape(sph_id) + r".{0,400}?object_id[\"':=\s]+([0-9]{10,25})",
        re.DOTALL,
    )
    object_ids = _memory_search([id_pattern], lambda acc, hit: acc.add(hit.group(1)))
    if not object_ids:
        return None

    # Hop 2: object id -> encfilekey marker(s) from Flow-log binding rows.
    bound_keys: set[str] = set()
    for object_id in object_ids:
        bind_pattern = re.compile(
            re.escape(object_id)
            + r",https?://finder\.video\.qq\.com/251/20302/stodownload\?encfilekey=([A-Za-z0-9_\-]+)"
        )
        bound_keys |= _memory_search(
            [bind_pattern], lambda acc, hit: acc.add(hit.group(1)[:16])
        )
    if not bound_keys:
        return None

    candidates = scan_wechat_decode_keys(max_seconds=max_seconds)
    for candidate in candidates:
        head_marker = candidate["url_head"].split("encfilekey=", 1)[-1][:16]
        if head_marker in bound_keys:
            return candidate
    return None


# ---------------------------------------------------------------- verify


def verify_download(path: Path, log: list[str] | None = None) -> dict:
    """Skill contract: ftyp magic + ffprobe codec/duration/resolution."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise DownloadError(f"downloaded file missing or empty: {path}")
    with path.open("rb") as handle:
        head = handle.read(12)
    if head[4:8] != b"ftyp":
        raise DownloadError(
            f"{path.name}: file header is not MP4 ('ftyp' missing at offset 4)"
        )
    if not FFPROBE.exists():
        if log is not None:
            log.append(f"ffprobe not found at {FFPROBE}; skipped media probe")
        return {"size": path.stat().st_size}
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise DownloadError(f"ffprobe failed on {path.name}: {result.stderr.strip()[:300]}")
    meta = json.loads(result.stdout or "{}")
    summary = {"size": path.stat().st_size}
    fmt = meta.get("format") or {}
    if fmt.get("duration"):
        summary["duration"] = float(fmt["duration"])
    streams = meta.get("streams") or []
    for stream in streams:
        if stream.get("codec_type") == "video":
            summary["video_codec"] = stream.get("codec_name")
            summary["width"] = stream.get("width")
            summary["height"] = stream.get("height")
            # r_frame_rate is a fraction string like "30000/1001".
            rate = stream.get("r_frame_rate") or ""
            try:
                num, _, den = rate.partition("/")
                summary["fps"] = round(float(num) / float(den or 1), 3)
            except (ValueError, ZeroDivisionError):
                summary["fps"] = None
        elif stream.get("codec_type") == "audio":
            summary["audio_codec"] = stream.get("codec_name")
    if log is not None:
        log.append(
            "verified {name}: {vc}+{ac} {w}x{h} {dur:.1f}s {size:,}B".format(
                name=path.name,
                vc=summary.get("video_codec", "?"),
                ac=summary.get("audio_codec", "-"),
                w=summary.get("width", "?"),
                h=summary.get("height", "?"),
                dur=summary.get("duration", 0.0),
                size=summary["size"],
            )
        )
    return summary
