"""fastdl connection reuse: chunk requests must not pay a handshake each.

Run with the ComfyUI embedded interpreter:
python tests/test_fastdl_keepalive.py
"""
from __future__ import annotations

import hashlib
import http.server
import json
import importlib.util
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


class _CountingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(payload: bytes, *, keep_alive: bool = True, ignore_range: bool = False, pace: float = 0.0):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if keep_alive else "HTTP/1.0"

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.server.accepts += 1            # one TCP connection, whatever the chunk count

        def do_GET(self):
            rng = self.headers.get("Range")
            if ignore_range:
                body = payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
            elif rng:
                start, end = (int(x) for x in rng.split("=")[1].split("-"))
                end = min(end, len(payload) - 1)
                body = payload[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
            else:
                body = payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "none" if ignore_range else "bytes")
            self.send_header("ETag", '"fixed"')
            if not keep_alive:
                self.send_header("Connection", "close")
            self.end_headers()
            if pace:
                time.sleep(pace)
            self.wfile.write(body)

    server = _CountingServer(("127.0.0.1", 0), Handler)
    server.accepts = 0
    server.payload = payload
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class KeepAliveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fastdl_ka_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.payload = os.urandom(40 * 1024 * 1024)

    def _url(self, server) -> str:
        return f"http://127.0.0.1:{server.server_address[1]}/model.bin"

    def test_chunks_share_connections(self):
        server = _serve(self.payload)
        try:
            dest = self.tmp / "a.bin"
            result = FASTDL.download(self._url(server), dest, conns=2, timeout=20)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        finally:
            server.shutdown()
            FASTDL._KeepAlive.reset()
        chunks = (len(self.payload) + FASTDL.CHUNK - 1) // FASTDL.CHUNK
        self.assertEqual(result["bytes"], len(self.payload))
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())
        # probe + at most one connection per worker, not one per chunk
        self.assertLessEqual(server.accepts, 4, f"{server.accepts} accepts for {chunks} chunks")
        self.assertLess(server.accepts, chunks, "connections must be reused across chunks")

    def test_server_sending_connection_close_still_downloads(self):
        server = _serve(self.payload, keep_alive=False)
        try:
            dest = self.tmp / "b.bin"
            FASTDL.download(self._url(server), dest, conns=3, timeout=20)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        finally:
            server.shutdown()
            FASTDL._KeepAlive.reset()
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())

    def test_range_ignored_falls_back_to_single_stream(self):
        server = _serve(self.payload, ignore_range=True)
        try:
            dest = self.tmp / "c.bin"
            result = FASTDL.download(self._url(server), dest, conns=4, timeout=20)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        finally:
            server.shutdown()
            FASTDL._KeepAlive.reset()
        self.assertFalse(result["ranges"])
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())

    def test_resume_after_interrupt_skips_finished_chunks(self):
        server = _serve(self.payload, pace=0.25)      # slow enough to interrupt mid-file
        try:
            dest = self.tmp / "d.bin"
            deadline = time.monotonic() + 1.2

            def stop_later() -> bool:
                return time.monotonic() > deadline

            with self.assertRaises(FASTDL.Cancelled):
                FASTDL.download(self._url(server), dest, conns=2, timeout=20, should_stop=stop_later)
            claimed = len(json.loads((self.tmp / "d.bin.state.json").read_text("utf-8"))["done"]) \
                if (self.tmp / "d.bin.state.json").is_file() else 0
            self.assertGreater(claimed, 0, "the interrupted run must leave durable chunks")
            before = server.accepts
            FASTDL._KeepAlive.reset()
            FASTDL.download(self._url(server), dest, conns=2, timeout=20)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        finally:
            server.shutdown()
            FASTDL._KeepAlive.reset()
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())
        self.assertGreater(server.accepts, before, "the resumed run must reconnect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
