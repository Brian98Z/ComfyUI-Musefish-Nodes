"""Parallel-range HTTP downloader shared by the ComfyUI nodes and Studio.

Same file, two hosts: ``MusefishFileDownload`` (ComfyUI) and the Studio's
「文件断点续传」task both import this module, so behaviour cannot drift.
Standard library only — neither host may install packages.

Why this exists: several model/CDN hosts (CivitAI b2, S3-compatible, some
ModelScope mirrors) throttle *per connection* to ~100 KB/s, so a browser or a
single-stream curl crawls while 32-64 ranged requests saturate the link. This
module also resumes from a `.part`/`.state.json` pair and refuses to trust a
short read.

Deliberately Comfy-free and dependency-free: the workbench server runs on the
bundled interpreter and installs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

CHUNK = 4 * 1024 * 1024
# Checkpoint cadence: state is rewritten only after the payload is durable, so a
# per-chunk fsync serialised every worker (4 ms each, lock held). Batching costs at
# most this much re-download after a hard crash, and nothing on the normal path.
CKPT_INTERVAL = 0.25      # seconds between checkpoints
CKPT_MAX_CHUNKS = 16      # ...or this many completed chunks, whichever first
MAX_TOTAL = 256 * 1024 ** 3   # refuse a Content-Range total beyond this (OOM guard)
MAX_REDIRECTS = 5             # a longer chain is either broken or hostile
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def format_bytes(count: float) -> str:
    """`1.2 GB` / `320.1 MB` / `813 KB` — sizes as a human reads them."""
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds > 359_999:
        return "--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class Progress:
    """One progress snapshot handed to a callback."""

    done: int                 # bytes on disk (resumed bytes included)
    total: int                # expected size, 0 when the server sent none
    percent: float            # 0..100
    speed: float              # bytes/second, smoothed
    eta: float | None         # seconds, None while the rate is unknown
    elapsed: float            # seconds since the reporter started
    resumed: int              # bytes that were already on disk at start

    def text(self) -> str:
        """`160.0 MB / 320.1 MB · 50.0% · 3.36 MB/s · ETA 00:47`"""
        if self.total:
            head = f"{format_bytes(self.done)} / {format_bytes(self.total)} · {self.percent:.1f}%"
        else:
            head = f"{format_bytes(self.done)} · 大小未知"
        return f"{head} · {format_bytes(self.speed)}/s · ETA {format_seconds(self.eta)}"


class ProgressReporter:
    """Throttle fastdl's ``(done, total)`` callback into :class:`Progress` snapshots.

    Pass any callable — a progress bar, a websocket, a dict write, a logger::

        reporter = ProgressReporter(lambda p: print(p.text()))
        download(url, dest, on_progress=reporter)

    The first non-zero call always passes (it is the resume baseline, so the
    first rate is not a spike), the final call always passes, and everything
    else is limited to one snapshot per ``interval``. The callback runs on
    worker threads, so a sink must not block the download for long.
    """

    def __init__(self, sink, *, interval: float = 0.25, clock=time.monotonic, smoothing: float = 0.3):
        self._sink = sink
        self._interval = max(0.0, float(interval))
        self._clock = clock
        self._smoothing = min(max(float(smoothing), 0.0), 1.0)
        self._started: float | None = None
        self._at: float | None = None
        self._done = 0
        self._resumed = 0
        self._speed = 0.0
        self._last: Progress | None = None
        self._lock = threading.Lock()

    def __call__(self, done: int, total: int) -> None:
        done = int(done)
        total = int(total)
        with self._lock:
            now = self._clock()
            first = self._last is None
            if self._started is None:
                self._started = now
                self._at = now
            # Resume jumps to a non-zero `done`: that is a baseline, not a rate.
            if not first and done < total and now - (self._at or now) < self._interval:
                return
            window = max(now - (self._at or now), 1e-6)
            if first:
                self._resumed = done
                self._speed = 0.0
            elif done > total:                      # server sent a short total
                self._speed = 0.0
            else:
                instant = max(done - self._done, 0) / window
                self._speed = (instant if self._speed <= 0
                               else self._smoothing * instant + (1 - self._smoothing) * self._speed)
            self._done = done
            self._at = now
            percent = (done / total * 100.0) if total else 0.0
            remaining = max(total - done, 0)
            eta = (remaining / self._speed) if (total and self._speed > 0) else None
            snapshot = Progress(done=done, total=total, percent=percent, speed=self._speed,
                                eta=eta, elapsed=now - (self._started or now), resumed=self._resumed)
            self._last = snapshot
        try:
            self._sink(snapshot)
        except Exception:  # noqa: BLE001 — a broken sink must never fail the download
            pass

    @property
    def last(self) -> Progress | None:
        return self._last


class DownloadError(Exception):
    """Any download failure that should be surfaced to the user."""


class AuthError(DownloadError):
    """401/403: the signed URL or token is invalid or expired."""


class RemoteChanged(DownloadError):
    """The object behind the URL changed mid-transfer: never splice two files."""


class Cancelled(DownloadError):
    """The user cancelled the job."""


_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _is_loopback(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalize_proxy(value: str | None) -> str | None:
    """'auto'/'direct'/'' or an explicit 'http://127.0.0.1:7897' → 'host:port'.

    Only a loopback proxy is accepted: the signed URL of a model download is
    forwarded through it, so a workflow-supplied proxy must not be able to point
    at someone else's machine. ``MUSEFISH_ALLOW_REMOTE_PROXY=1`` opts out.
    """
    v = (value or "").strip()
    if not v or v.lower() in ("auto", "none", "direct", "直连", "自动"):
        return None
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.rstrip("/") or None
    if v and os.environ.get("MUSEFISH_ALLOW_REMOTE_PROXY") != "1":
        host = v.rsplit(":", 1)[0] if ":" in v else v
        if not _is_loopback(host):
            raise DownloadError(f"代理只允许本机地址（{v}）：本机代理客户端通常是 127.0.0.1:<端口>")
    return v


def auto_proxy(value: str | None, fallback: str | None) -> str | None:
    """'auto' resolves to the server-config proxy, anything explicit wins."""
    v = (value or "auto").strip()
    if v.lower() in ("auto", "自动"):
        return normalize_proxy(fallback) or None
    return normalize_proxy(v)


def _transient(err: Exception) -> bool:
    """本地代理/链路偶发的 TLS EOF、连接重置：值得立刻重试一次。

    连接超时（WinError 10060 / timed out）不算——重试只是再等一遍。
    """
    text = f"{type(err).__name__}: {err}"
    if "10060" in text or "timed out" in text.lower():
        return False
    return any(k in text for k in ("SSL", "UNEXPECTED_EOF", "EOF occurred", "10054",
                                   "Connection reset", "ConnectionReset", "handshake"))


_LOCAL = threading.local()   # per worker thread: one reusable connection per origin


class _KeepAlive:
    """One reusable HTTP(S) connection per worker thread, proxy tunnels included.

    Without this, every 4 MiB chunk paid a full TCP+TLS+TTFB round trip: measured
    2.4-3.4 s per new connection through a local proxy, i.e. a real share of a
    large download. A server that closes the connection (``Connection: close``)
    simply costs one reconnect on the next chunk.
    """

    __slots__ = ("key", "conn")

    def __init__(self, key, conn):
        self.key = key
        self.conn = conn

    @staticmethod
    def _slot() -> dict:
        slot = getattr(_LOCAL, "conns", None)
        if slot is None:
            slot = {}
            _LOCAL.conns = slot
        return slot

    @classmethod
    def take(cls, url: str, proxy: str | None, timeout: float):
        """Return ``(connection, reused, target)`` for this URL."""
        import http.client
        import ssl

        parsed = urlparse(url)
        scheme = (parsed.scheme or "http").lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        key = (scheme, host, port, proxy)
        slot = cls._slot()
        entry = slot.get(key)
        if entry is not None:
            return entry.conn, True, (scheme, host, port, parsed)
        context = ssl.create_default_context()
        if proxy:
            proxy_host, _, proxy_port = proxy.partition(":")
            proxy_port = int(proxy_port or 80)
            if scheme == "https":
                conn = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=timeout, context=context)
                conn.set_tunnel(host, port)
            else:
                conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout)
        else:
            conn = (http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
                    if scheme == "https" else http.client.HTTPConnection(host, port, timeout=timeout))
        slot[key] = cls(key, conn)
        return conn, False, (scheme, host, port, parsed)

    @classmethod
    def discard(cls, key) -> None:
        entry = cls._slot().pop(key, None)
        if entry is not None:
            try:
                entry.conn.close()
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def reset(cls) -> None:
        slot = cls._slot()
        for entry in list(slot.values()):
            try:
                entry.conn.close()
            except Exception:  # noqa: BLE001
                pass
        slot.clear()


class _KeepAliveResponse:
    """Response that returns its connection to the pool once fully read."""

    __slots__ = ("_key", "_conn", "_resp", "headers", "status", "reason", "final_url")

    def __init__(self, key, conn, resp):
        self._key = key
        self._conn = conn
        self._resp = resp
        self.headers = resp.headers
        self.status = resp.status
        self.reason = getattr(resp, "reason", "")
        self.final_url = None

    def read(self, size: int = -1) -> bytes:
        try:
            data = self._resp.read(size)
        except Exception:
            _KeepAlive.discard(self._key)
            raise
        if size is not None and size > 0 and len(data) < size:
            # short read: the peer closed or the chunk ended. Reusable only when
            # the response knows it delivered everything it promised.
            if not self._resp.isclosed() or self._resp.will_close:
                _KeepAlive.discard(self._key)
        return data

    def close(self) -> None:
        complete = False
        try:
            complete = self._resp.isclosed() and not self._resp.will_close
        except Exception:  # noqa: BLE001
            complete = False
        if not complete:
            _KeepAlive.discard(self._key)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _open(url: str, proxy: str | None, headers: dict | None, timeout: float, attempts: int = 3,
          hops: int = MAX_REDIRECTS):
    """Open a ranged request, reusing this thread's connection when possible."""
    import http.client

    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        raise DownloadError(f"只支持 http(s) 直链：{url[:120]}")

    req_headers = {"User-Agent": "Musefish/1.7 (+fastdl)", **(headers or {})}
    last: Exception | None = None
    for i in range(attempts):
        key = None
        reused = False
        try:
            conn, reused, (scheme, host, port, parsed) = _KeepAlive.take(url, proxy, timeout)
            key = (scheme, host, port, proxy)
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            if proxy and scheme == "http":
                path = url                       # forward proxies want an absolute URI
            if "Host" not in req_headers:
                req_headers["Host"] = parsed.netloc
            conn.request("GET", path, headers=req_headers)
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader("Location") or ""
                resp.read(64 * 1024)          # bounded: never buffer an arbitrary 3xx body
                if location:
                    _KeepAlive.discard(key)
                    from urllib.parse import urljoin
                    target = urljoin(url, location)
                    hops_left = hops - 1
                    if hops_left < 0:
                        raise DownloadError(f"重定向超过 {MAX_REDIRECTS} 次，已停止：{url}")
                    target_parsed = urlparse(target)
                    if target_parsed.scheme not in ("http", "https"):
                        raise DownloadError(f"重定向到非 http(s) 地址，已拒绝：{target[:120]}")
                    if not _is_loopback(parsed.hostname or "") and _is_loopback(target_parsed.hostname or ""):
                        raise DownloadError("重定向指向本机地址，已拒绝（SSRF 防护）")
                    # caller headers (auth/cookies) never follow a redirect to another origin
                    same_origin = (target_parsed.hostname or "") == (parsed.hostname or "")
                    return _open(target, proxy, headers if same_origin else None, timeout,
                                 attempts, hops_left)
            if resp.status in (401, 403):
                resp.read()
                _KeepAlive.discard(key)
                raise AuthError(
                    f"HTTP {resp.status}：链接未授权（签名/令牌可能已过期，或缺少完整查询串）")
            if resp.status == 416:
                resp.read()
                _KeepAlive.discard(key)
                raise DownloadError("HTTP 416：Range 请求超出文件范围（远端文件可能已变化）")
            if resp.status >= 400:
                detail = resp.reason
                resp.read()
                _KeepAlive.discard(key)
                raise DownloadError(f"HTTP {resp.status} {detail}")
            response = _KeepAliveResponse(key, conn, resp)
            response.final_url = url      # what actually served this body
            return response
        except AuthError:
            raise
        except DownloadError:
            raise
        except (http.client.HTTPException, OSError) as e:
            if key is not None:
                _KeepAlive.discard(key)
            last = e
            if reused:
                continue                        # stale keep-alive socket: retry once, fresh
            if i < attempts - 1 and _transient(e):
                time.sleep(0.4 * (i + 1))
                continue
            break
    raise DownloadError(
        f"无法连接：{last}（如需代理请在「代理」里填写，例如 http://127.0.0.1:7897）") from last


def _filename_from(resp, url: str) -> str:
    cd = resp.headers.get("Content-Disposition") or ""
    m = re.search(r"filename\*\s*=\s*([^']*)''([^;]+)", cd, re.IGNORECASE)
    if m:
        return unquote(m.group(2).strip().strip('"'))
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd, re.IGNORECASE)
    if m:
        return unquote(m.group(1).strip())
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1]) or "download.bin"
    return name


def safe_name(name: str, fallback: str = "download.bin") -> str:
    name = _BAD_NAME.sub("_", (name or "").strip()).strip(". ")
    return name[:180] or fallback


def probe(url: str, *, proxy: str | None = None, headers: dict | None = None,
          timeout: float = 30.0) -> dict:
    """Size / range support / filename, using GET with a 1-byte range.

    Never uses HEAD: signed CDN links routinely answer HEAD with 401 while the
    identical GET returns 200.
    """
    resp = _open(url, proxy, {**(headers or {}), "Range": "bytes=0-0"}, timeout)
    try:
        status = getattr(resp, "status", 200)
        size = None
        ranges = False
        cr = resp.headers.get("Content-Range") or ""
        # 总长以 Content-Range 为准，且不要求 206：ModelScope/部分对象存储
        # 会以 200 + 正确的 Content-Range 返回请求的那一段（每个分片都 200）。
        if "/" in cr:
            tail = cr.rsplit("/", 1)[1].strip()
            if tail.isdigit():
                size = int(tail)
                ranges = True
                if size > MAX_TOTAL:
                    raise DownloadError(
                        f"远端声明大小 {size / 2**30:.0f} GiB，超过 {MAX_TOTAL / 2**30:.0f} GiB 上限，已拒绝")
        elif resp.headers.get("Content-Length"):
            size = int(resp.headers["Content-Length"])
            ranges = (resp.headers.get("Accept-Ranges") or "").lower() == "bytes"
        validator = (resp.headers.get("ETag") or resp.headers.get("Last-Modified") or "").strip()
        return {
            "size": size,
            "ranges": ranges,
            "filename": safe_name(_filename_from(resp, url)),
            "content_type": resp.headers.get("Content-Type") or "",
            "validator": validator or None,
            "url": url,
            "final_url": getattr(resp, "final_url", None) or url,
        }
    finally:
        resp.close()


class _RangeIgnored(Exception):
    """The server answered 200 without Content-Range: ranges are unusable."""


def _content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """Parse ``Content-Range: bytes start-end/total`` (``*`` -> None total)."""
    m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", (value or "").strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), (None if m.group(3) == "*" else int(m.group(3)))


def url_digest(url: str) -> str:
    """State files sit next to model weights: never persist the signed URL itself."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _write_state(state: Path, payload: dict) -> None:
    """Atomic replace: a torn state file would claim chunks that never landed."""
    tmp = state.with_suffix(state.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), "utf-8")
    os.replace(tmp, state)


def _load_state(part: Path, state: Path, size: int, chunk: int, url: str,
                validator: str | None) -> set[int]:
    """Chunks that provably belong to *this* remote object.

    An ETag/Last-Modified pins the object even when the signed URL changes on
    every retry; without a validator only an identical URL may be resumed —
    matching the length alone is what silently splices two different files.
    """
    if not (part.is_file() and state.is_file() and part.stat().st_size == size):
        return set()
    try:
        st = json.loads(state.read_text("utf-8"))
    except (OSError, ValueError):
        return set()
    if st.get("size") != size or st.get("chunk") != chunk:
        return set()
    if validator:
        if st.get("validator") != validator:
            return set()
    elif st.get("url_hash") != url_digest(url):
        return set()
    return set(st.get("done") or [])


def download(url: str, dest: Path, *, conns: int = 32, chunk: int = CHUNK,
             proxy: str | None = None, headers: dict | None = None,
             timeout: float = 30.0, on_progress=None, should_stop=None,
             retries: int = 8, overwrite: bool = False,
             probe_info: dict | None = None) -> dict:
    """Download `url` to `dest`, resuming any matching `.part` file.

    on_progress(done_bytes, total_bytes) is called from worker threads;
    should_stop() may return True to abort (raises Cancelled). With
    ``overwrite=False`` a same-sized destination is reported as skipped.
    ``probe_info`` lets a caller that already probed this exact URL skip the
    second metadata request (size/validator mismatches are still caught while
    transferring).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = probe_info or probe(url, proxy=proxy, headers=headers, timeout=timeout)
    total, ranges, name = info["size"], info["ranges"], info["filename"]
    validator = info.get("validator")
    # Signed/redirecting links (CivitAI -> r2, ModelScope -> CDN) answered every
    # chunk with a 307 first; pinning the resolved URL costs one redirect in total.
    transport = {"url": info.get("final_url") or url, "refreshed": False}
    transport_lock = threading.Lock()

    def transport_url() -> str:
        return transport["url"]

    def refresh_transport() -> None:
        """A pinned signed URL expired (403): re-probe the original once."""
        with transport_lock:
            if transport["refreshed"]:
                return
            transport["refreshed"] = True
            try:
                fresh = probe(url, proxy=proxy, headers=headers, timeout=timeout)
            except DownloadError:
                return
            new_size = fresh.get("size") or 0
            new_validator = fresh.get("validator")
            if new_size != (total or 0) or (validator and new_validator and new_validator != validator):
                raise RemoteChanged("远端对象在下载中发生变化（尺寸/ETag 不一致），已中止以免拼接两个文件")
            transport["url"] = fresh.get("final_url") or url
    conns = max(1, min(int(conns or 1), 64))
    part = dest.with_suffix(dest.suffix + ".part")
    state = dest.with_suffix(dest.suffix + ".state.json")

    if not overwrite and dest.is_file() and total and dest.stat().st_size == total:
        return {"path": str(dest), "bytes": total, "filename": name, "skipped": True}

    def check_stop():
        if should_stop and should_stop():
            raise Cancelled("已取消")

    def single_stream() -> dict:
        check_stop()
        done = 0
        resp = _open(transport_url(), proxy, headers, timeout)
        try:
            with open(part, "wb") as f:
                while True:
                    check_stop()
                    block = resp.read(256 * 1024)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if on_progress:
                        on_progress(done, total or 0)   # 0 = unknown size, not "finished"
                f.flush()
                os.fsync(f.fileno())
        finally:
            resp.close()
        if total and done != total:
            part.unlink(missing_ok=True)  # never publish a truncated body
            raise DownloadError(f"单流下载不完整：{done}/{total} bytes（远端提前结束）")
        os.replace(part, dest)
        state.unlink(missing_ok=True)
        return {"path": str(dest), "bytes": done, "filename": name, "ranges": False}

    if not ranges or not total:
        return single_stream()

    if total > MAX_TOTAL:
        raise DownloadError(f"远端大小 {total / 2**30:.0f} GiB 超出上限，已拒绝")
    n = (total + chunk - 1) // chunk
    done_chunks = _load_state(part, state, total, chunk, url, validator)
    if not done_chunks:
        # These bytes belong to some other attempt/object; keeping them lets a
        # later matching resume skip a chunk that was never written from *this*
        # source. Start clean instead of trusting the size alone.
        state.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    got = (sum(min(chunk, total - i * chunk) for i in done_chunks)
           if len(done_chunks) < n else total)
    doc = open(part, "r+b") if done_chunks and part.is_file() and part.stat().st_size == total else open(part, "w+b")
    doc.truncate(total)
    if on_progress and got:  # 续传：先报一次已有进度，避免首帧速度出现异常尖峰
        on_progress(min(got, total), total)
    lock = threading.Lock()
    stop = threading.Event()
    range_ignored = threading.Event()
    failures: list[str] = []
    # `durable` is what the state file may claim; `pending` holds chunks whose
    # bytes are on disk but not yet fsynced.
    durable: set[int] = set(done_chunks)
    pending: set[int] = set()
    last_checkpoint = [0.0]

    def checkpoint(force: bool = False) -> None:
        if not pending:
            return
        now = time.time()
        if not force and now - last_checkpoint[0] < CKPT_INTERVAL and len(pending) < CKPT_MAX_CHUNKS:
            return
        with lock:
            if not pending:
                return
            doc.flush()
            os.fsync(doc.fileno())
            durable.update(pending)
            pending.clear()
            last_checkpoint[0] = time.time()
            _write_state(state, {"size": total, "chunk": chunk, "url_hash": url_digest(url),
                                 "validator": validator, "done": sorted(durable)})

    def bump(n_bytes: int) -> None:
        nonlocal got
        got += n_bytes
        if on_progress:
            on_progress(min(got, total), total)

    written: dict[int, int] = {}   # bytes of chunk i currently on disk (this run)

    def fetch_chunk(i: int) -> bool:
        """Fetch chunk i, keeping every byte that arrives.

        A flaky proxy drops connections mid-stream; re-asking only for the
        missing tail (instead of the whole 4 MiB) turns that into progress
        rather than a retry storm. `stalled` counts only zero-byte attempts.
        """
        nonlocal got
        start, end = i * chunk, min((i + 1) * chunk, total) - 1
        pos, stalled = start, 0
        with lock:
            # a retry pass restarts the chunk from its first byte: drop the bytes
            # it already reported, otherwise progress double-counts them
            previous = written.pop(i, 0)
            if previous:
                got -= previous
        while pos <= end:
            if should_stop and should_stop():
                raise Cancelled("已取消")
            before = pos
            try:
                try:
                    resp = _open(transport_url(), proxy, {**(headers or {}),
                                                          "Range": f"bytes={pos}-{end}"}, timeout)
                except AuthError:
                    refresh_transport()
                    resp = _open(transport_url(), proxy, {**(headers or {}),
                                                          "Range": f"bytes={pos}-{end}"}, timeout)
                try:
                    # 200 + 带着我们请求区间 Content-Range 的服务器（ModelScope
                    # 网关、部分 OSS 代理）仍然算分片成功；没有 Content-Range
                    # 的 200 才是真忽略了 Range（会把整个文件塞回来）。
                    if (getattr(resp, "status", 206) == 200
                            and not (resp.headers.get("Content-Range") or "")):
                        raise _RangeIgnored()
                    span = _content_range(resp.headers.get("Content-Range"))
                    if span is not None:
                        span_start, _span_end, span_total = span
                        if span_total is not None and span_total != total:
                            raise DownloadError(
                                f"远端文件尺寸在下载中变化（{span_total} != {total}）")
                        if span_start != pos:
                            raise DownloadError(
                                f"Content-Range 起点不符：要求 {pos}，返回 {span_start}")
                    while pos <= end:
                        if should_stop and should_stop():
                            raise Cancelled("已取消")
                        block = resp.read(min(262144, end - pos + 1))
                        if not block:
                            break
                        with lock:
                            doc.seek(pos)
                            doc.write(block)
                            pos += len(block)
                            written[i] = written.get(i, 0) + len(block)
                            bump(len(block))
                finally:
                    resp.close()
            except Cancelled:
                raise
            except AuthError:
                raise
            except _RangeIgnored:
                range_ignored.set()
                stop.set()
                return False
            except (DownloadError, OSError) as e:
                if isinstance(e, DownloadError) and "Content-Range" in str(e):
                    raise  # protocol violation: retrying cannot help
                if pos == before:  # connection died before a single byte
                    stalled += 1
                    if stalled > retries:
                        raise
                    time.sleep(min(4.0, 0.5 * stalled))
                continue
            if pos == before:  # connected, no bytes: treat as a stalled request
                stalled += 1
                if stalled > retries:
                    raise DownloadError(f"分片 {i} 无数据（{retries} 次空响应）")
                time.sleep(min(4.0, 0.5 * stalled))
        with lock:
            done_chunks.add(i)
            pending.add(i)
            written.pop(i, None)      # now accounted for by done_chunks
        checkpoint()
        return True

    def worker() -> None:
        while not stop.is_set():
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            try:
                fetch_chunk(i)
            except Cancelled:
                stop.set()
                return
            except AuthError as e:
                with lock:
                    failures.append(str(e))
                stop.set()
                return
            except Exception as e:  # noqa: BLE001
                with lock:
                    failures.append(f"第 {i} 片失败：{e}")
                stop.set()
                return

    # up to 3 passes: a poisoned keep-alive connection is fixed by fresh
    # sessions, and every completed chunk is already on disk + in the state
    for _pass in range(3):
        missing = [i for i in range(n) if i not in done_chunks]
        if not missing:
            break
        failures.clear()
        stop.clear()
        q = queue.Queue()
        for i in missing:
            q.put(i)
        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(max(1, min(conns, n)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if range_ignored.is_set():
            break

    if range_ignored.is_set():
        # Advertised ranges but stopped honouring them: restart through the
        # single-stream path instead of stitching offset-zero bodies together.
        doc.close()
        part.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
        return single_stream()

    checkpoint(force=True)
    doc.flush()
    os.fsync(doc.fileno())
    doc.close()

    if should_stop and should_stop():
        if not done_chunks and part.is_file():
            part.unlink(missing_ok=True)   # 预分配但一个分片都没完成：不留假进度
            state.unlink(missing_ok=True)
        raise Cancelled("已取消")
    missing = [i for i in range(n) if i not in done_chunks]
    if missing:
        raise DownloadError(
            f"{failures[0] if failures else '网络中断'}（已保留 {len(done_chunks)}/{n} 分片，"
            f"用相同参数重新提交即可续传）")
    if not part.is_file() or part.stat().st_size != total:
        raise DownloadError("文件大小校验失败")
    os.replace(part, dest)
    if state.is_file():
        state.unlink()
    return {"path": str(dest), "bytes": total, "filename": name, "ranges": True}


# ---------------------------------------------------------------- 代理探测

# 主流代理客户端与其默认端口。先看进程名，再看端口是否真的在监听。
_PROXY_CLIENTS = [
    ("Clash Verge / Mihomo", ("clash", "mihomo", "verge"),
     [(7897, "混合端口"), (7890, "混合端口(旧)"), (7891, "SOCKS5")]),
    ("Clash for Windows", ("clash",), [(7890, "混合端口"), (7891, "SOCKS5")]),
    ("v2rayN / Xray", ("v2ray", "xray", "qv2ray", "nekoray"),
     [(10809, "HTTP"), (10808, "SOCKS5")]),
    ("sing-box / Hiddify", ("sing-box", "singbox", "hiddify"), [(2080, "混合端口"), (12334, "Hiddify")]),
    ("Shadowsocks", ("shadowsocks", "ss-local", "ssr"), [(1080, "混合端口"), (1081, "备选")]),
    ("Netch / SSTap", ("netch", "sstap"), [(2801, "HTTP"), (2802, "SOCKS5")]),
]
_PROXY_FALLBACK_PORTS = (7897, 7890, 10809, 10808, 1080, 2080, 8889, 8118, 12334, 2801)
_proxy_cache: dict = {"at": 0.0, "items": []}
_proxy_lock = threading.Lock()


def _port_open(port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout):
            return True
    except OSError:
        return False


def _running_procs() -> str:
    if os.name != "nt":
        return ""
    try:
        return subprocess.run(["tasklist"], capture_output=True, text=True, timeout=8).stdout.lower()
    except Exception:  # noqa: BLE001
        return ""


def detect_proxies(force: bool = False) -> list[dict]:
    """本机正在监听的代理端点（并行探测 0.25 s 上限，结果缓存 30 s）。"""
    now = time.time()
    with _proxy_lock:
        # Cache empty results too: on a machine without any local proxy the old
        # condition reran tasklist + 26 port probes on every single call.
        cached = _proxy_cache["at"] > 0 and now - _proxy_cache["at"] < 30
        if not force and cached:
            return list(_proxy_cache["items"])

    wanted: dict[int, list[tuple[str, str, tuple]]] = {}
    for label, hints, ports in _PROXY_CLIENTS:
        for port, tag in ports:
            wanted.setdefault(port, []).append((label, tag, hints))
    for port in _PROXY_FALLBACK_PORTS:
        wanted.setdefault(port, [])

    opened: dict[int, bool] = {}

    def probe_port(port: int) -> None:
        opened[port] = _port_open(port)

    threads = [threading.Thread(target=probe_port, args=(p,), daemon=True) for p in wanted]
    for t in threads:
        t.start()
    procs = _running_procs()
    for t in threads:
        t.join()

    found: list[dict] = []
    for port, hints in wanted.items():
        if not opened.get(port):
            continue
        hit = next((h for h in hints if any(name in procs for name in h[2])), None)
        if hit is None and hints:
            continue  # 该客户端的端口，但那个客户端没在跑
        found.append({
            "label": f"{hit[0]} · {hit[1]}" if hit else f"本机 HTTP 代理 · {port}",
            "url": f"http://127.0.0.1:{port}",
        })
    with _proxy_lock:
        _proxy_cache["at"], _proxy_cache["items"] = time.time(), found
    return list(found)


def auto_proxy_endpoint(config_proxy: str | None = None) -> tuple[str | None, str]:
    """`自动` 用的端点：配置里的代理（若在监听）→ 探测到的首个代理 → 直连。"""
    found = detect_proxies()

    def label_for(url: str, fallback: str) -> str:
        return next((f["label"] for f in found if f["url"].rstrip("/") == url.rstrip("/")), fallback)

    raw = (config_proxy or "").strip()
    if raw:
        url = raw if "://" in raw else f"http://{raw}"
        matched = next((f for f in found if f["url"].rstrip("/") == url.rstrip("/")), None)
        if matched:
            return matched["url"], matched["label"]
        body = url.split("://", 1)[1].rstrip("/")
        host, _, port = body.partition(":")
        if not port.isdigit() or host not in ("127.0.0.1", "localhost"):
            return url, f"配置代理 {body}"
        if _port_open(int(port)):
            return f"http://127.0.0.1:{port}", label_for(url, f"配置代理 {port}")
    for item in found:
        return item["url"], item["label"]
    return None, "直连"


_route_memo: dict = {"value": None}   # 上次探测成功的通路（None=直连）


def resolve_route(want: str | None, config_proxy: str | None = None) -> tuple[str, str | None, str]:
    """把界面上的「代理」值归一成 (kind, proxy, label)。

    kind: "auto"（记忆→直连→本机代理）、"direct"、"manual"。
    """
    v = (want or "auto").strip()
    low = v.lower()
    if low in ("auto", "自动") or v.startswith("自动"):
        return "auto", None, auto_proxy_endpoint(config_proxy)[1]
    if low in ("direct", "none", "直连") or v.startswith("直连"):
        return "direct", None, "直连"
    return "manual", normalize_proxy(v), v


def probe_with_route(url: str, want: str | None = "auto", *, config_proxy: str | None = None,
                     timeout: float = 30.0, headers: dict | None = None,
                     should_stop=None, log=None) -> tuple[dict, str | None, str]:
    """探测远端（尺寸/分片/文件名），自动模式按「记忆→直连→本机代理」依次尝试。

    返回 (probe_info, proxy, label)；proxy 为 None 表示直连。
    """
    kind, proxy, label = resolve_route(want, config_proxy)
    if kind == "manual":
        return probe(url, proxy=proxy, timeout=timeout, headers=headers), proxy, label
    if kind == "direct":
        return probe(url, proxy=None, timeout=timeout, headers=headers), None, "直连"

    auto_url, auto_label = auto_proxy_endpoint(config_proxy)
    candidates: list[str | None] = []
    for cand in (_route_memo["value"], None, normalize_proxy(auto_url)):
        if cand not in candidates:
            candidates.append(cand)
    last: Exception | None = None
    for idx, cand in enumerate(candidates):
        if should_stop and should_stop():
            raise Cancelled("已取消")
        where = "直连" if cand is None else f"代理 {cand}"
        try:
            info = probe(url, proxy=cand, headers=headers,
                         timeout=10 if cand is None else timeout)
        except DownloadError as err:
            last = err
            if idx < len(candidates) - 1 and log:
                log(f"{where}失败（{err}）")
            continue
        _route_memo["value"] = cand
        if log:
            log(f"通路：{where}")
        return info, cand, where
    raise last or DownloadError("无法建立到该地址的连接")


def guard_dest_dir(directory: Path, allowed_roots: list[Path], allow_any: bool = False) -> Path:
    """Refuse a destination outside the caller's allowlist.

    A shared workflow can set ``save_dir``; without this check the node wrote
    attacker-chosen bytes anywhere the ComfyUI process can write (e.g. the
    Startup folder). The caller passes the roots it considers legitimate.
    """
    resolved = directory.expanduser().resolve()
    if allow_any:
        return resolved
    for root in allowed_roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
            return resolved
        except (ValueError, OSError):
            continue
    shown = "、".join(str(root) for root in allowed_roots[:4])
    raise DownloadError(
        f"保存目录 {resolved} 不在允许范围（{shown} 等）；确需写入其它位置请在节点上打开 allow_any_dir")


def desktop_dir() -> Path:
    """用户的真实桌面（跟随 OneDrive 重定向）。"""
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0 and buf.value:
                return Path(buf.value)
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Desktop"
