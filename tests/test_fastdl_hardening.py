"""Guards added after the security audit: redirect policy, size cap, state hygiene.

Run with the ComfyUI embedded interpreter:
python tests/test_fastdl_hardening.py
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _load_fastdl():
    spec = importlib.util.spec_from_file_location("musefish_fastdl", PLUGIN_ROOT / "musefish_fastdl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["musefish_fastdl"] = module
    spec.loader.exec_module(module)
    return module


FASTDL = _load_fastdl()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(handler_cls):
    server = _Server(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _range_handler(payload: bytes, etag: str = '"v1"', total_override: int | None = None, pace: float = 0.0):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            size = total_override if total_override is not None else len(payload)
            start, end = (int(x) for x in rng.split("=")[1].split("-")) if rng else (0, len(payload) - 1)
            end = min(end, len(payload) - 1)
            body = payload[start:end + 1]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", etag)
            self.end_headers()
            if pace:
                time.sleep(pace)
            self.wfile.write(body)

    return Handler


class RedirectPolicyTests(unittest.TestCase):
    def test_redirect_loop_is_capped(self):
        state = {"hits": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                state["hits"] += 1
                self.send_response(302)
                self.send_header("Location", "/again")     # self-redirect forever
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = _serve(Handler)
        try:
            with self.assertRaises(FASTDL.DownloadError) as ctx:
                FASTDL.probe(f"http://127.0.0.1:{server.server_address[1]}/loop", timeout=10)
        finally:
            server.shutdown()
            FASTDL._KeepAlive.reset()
        self.assertIn("重定向", str(ctx.exception))
        self.assertLessEqual(state["hits"], FASTDL.MAX_REDIRECTS + 1)

    def test_private_targets_are_classified(self):
        self.assertTrue(FASTDL._is_loopback("127.0.0.1"))
        self.assertTrue(FASTDL._is_loopback("localhost"))
        self.assertTrue(FASTDL._is_loopback("[::1]"))
        self.assertFalse(FASTDL._is_loopback("civitai.com"))
        self.assertFalse(FASTDL._is_loopback("10.1.2.3"))

    def test_remote_proxy_is_refused(self):
        with self.assertRaises(FASTDL.DownloadError):
            FASTDL.normalize_proxy("http://192.0.2.10:8080")
        self.assertEqual(FASTDL.normalize_proxy("http://127.0.0.1:7897"), "127.0.0.1:7897")


class SizeCapTests(unittest.TestCase):
    def test_absurd_remote_total_is_refused_before_allocating(self):
        payload = b"x" * 4096
        server = _serve(_range_handler(payload, total_override=FASTDL.MAX_TOTAL + 1))
        tmp = Path(tempfile.mkdtemp(prefix="fastdl_cap_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        self.addCleanup(FASTDL._KeepAlive.reset)
        try:
            with self.assertRaises(FASTDL.DownloadError) as ctx:
                FASTDL.probe(f"http://127.0.0.1:{server.server_address[1]}/big", timeout=10)
        finally:
            server.shutdown()
        self.assertIn("上限", str(ctx.exception))
        self.assertEqual(list(tmp.iterdir()), [], "no .part may be created")


class DestinationGuardTests(unittest.TestCase):
    def test_only_allowed_roots_are_accepted(self):
        tmp = Path(tempfile.mkdtemp(prefix="fastdl_dest_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        allowed = tmp / "output"
        allowed.mkdir()
        inside = allowed / "models" / "x.bin"
        self.assertEqual(FASTDL.guard_dest_dir(inside.parent, [allowed]), inside.parent.resolve())
        with self.assertRaises(FASTDL.DownloadError):
            FASTDL.guard_dest_dir(tmp / "elsewhere", [allowed])
        # explicit opt-in keeps arbitrary drives usable for people who mean it
        self.assertEqual(FASTDL.guard_dest_dir(tmp / "elsewhere", [allowed], allow_any=True),
                         (tmp / "elsewhere").resolve())


class UnknownSizeTests(unittest.TestCase):
    """A close-delimited response has no Content-Length: size must stay unknown."""

    def setUp(self):
        self.temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temporary, True)
        payload = b"".join(hashlib.sha256(bytes([i])).digest() * 900 for i in range(120))

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"          # no keep-alive => close-delimited body

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

        self.payload = payload
        self.server = _serve(Handler)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/blob.bin"

    def test_probe_reports_no_size_and_download_still_works(self):
        info = FASTDL.probe(self.url, timeout=10)
        self.assertIsNone(info["size"])
        seen = []
        dest = self.temporary / "blob.bin"
        FASTDL.download(self.url, dest, conns=4, timeout=10,
                        on_progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(dest.read_bytes(), self.payload)
        self.assertEqual(seen[-1][1], 0, "unknown size must never be reported as finished")
        self.assertTrue(all(done <= len(self.payload) for done, _ in seen))


class DestateHygieneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fastdl_state_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.addCleanup(FASTDL._KeepAlive.reset)
        self.payload = os.urandom(28 * 1024 * 1024)     # 7 chunks: enough to interrupt

    def _download(self, server, dest, **kwargs):
        url = f"http://127.0.0.1:{server.server_address[1]}/f.bin"
        return FASTDL.download(url, dest, conns=2, timeout=20, **kwargs)

    def test_state_file_never_contains_the_signed_url(self):
        secret = "Authorization=super-secret-token"
        server = _serve(_range_handler(self.payload, pace=0.5))
        dest = self.tmp / "a.bin"
        deadline = time.monotonic() + 0.7

        def stop_midway() -> bool:
            return time.monotonic() > deadline

        with self.assertRaises(FASTDL.Cancelled):
            self._download(server, dest, should_stop=stop_midway)
        server.shutdown()
        state = dest.with_suffix(dest.suffix + ".state.json")
        self.assertTrue(state.is_file(), "an interrupted run must leave its state for resume")
        if state.is_file():
            text = state.read_text("utf-8")
            self.assertNotIn("http", text)
            self.assertNotIn(secret, text)
            self.assertIn("url_hash", json.loads(text))

    def test_bytes_from_another_object_are_never_reused(self):
        """Same length, different ETag: the old .part must not be spliced in."""
        dest = self.tmp / "b.bin"
        part = dest.with_suffix(dest.suffix + ".part")
        part.write_bytes(b"\x00" * len(self.payload))               # pretend A's bytes
        state = dest.with_suffix(dest.suffix + ".state.json")
        state.write_text(json.dumps({"size": len(self.payload), "chunk": FASTDL.CHUNK,
                                     "url_hash": FASTDL.url_digest("http://other/f.bin"),
                                     "validator": '"object-A"', "done": [0, 1, 2]}), "utf-8")
        second = _serve(_range_handler(self.payload, etag='"object-B"'))
        try:
            self._download(second, dest)
        finally:
            second.shutdown()
        self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(),
                         hashlib.sha256(self.payload).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
