"""Redirecting links (CivitAI -> r2, ModelScope -> CDN): resolve once, then pin.

Run with the ComfyUI embedded interpreter:
python tests/test_fastdl_redirect.py
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import os
import shutil
import socketserver
import sys
import tempfile
import threading
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


def _redirect_server(state: dict, target_port: int):
    """A: rotates a signed target on each probe; B: serves ranged bytes for a valid sig."""
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            state["a_requests"] += 1
            state["sig"] = f"s{state['a_requests']}"
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/blob?sig={state['sig']}")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _blob_server(payload: bytes, state: dict, expire_after: int | None = None):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            state["b_requests"] += 1
            sig = (self.path.split("sig=", 1) + [""])[1]
            if (expire_after is not None and state["b_requests"] > expire_after
                    and not state.get("rotated")):
                # the origin rotated its signature: the pinned URL is now stale
                state["rotated"] = True
                state["sig"] = "rotated"
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if sig != state["sig"]:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            rng = self.headers.get("Range")
            start, end = (int(x) for x in rng.split("=")[1].split("-")) if rng else (0, len(payload) - 1)
            end = min(end, len(payload) - 1)
            body = payload[start:end + 1]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"blob"')
            self.end_headers()
            self.wfile.write(body)

    server = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class RedirectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fastdl_redirect_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.addCleanup(FASTDL._KeepAlive.reset)
        self.payload = os.urandom(14 * 1024 * 1024)      # 4 chunks

    def _pair(self, expire_after=None):
        state = {"a_requests": 0, "b_requests": 0, "sig": ""}
        blob = _blob_server(self.payload, state, expire_after=expire_after)
        redirect = _redirect_server(state, blob.server_address[1])
        self.addCleanup(blob.shutdown)
        self.addCleanup(redirect.shutdown)
        return redirect, state

    def test_redirect_is_resolved_once_then_pinned(self):
        redirect, state = self._pair()
        url = f"http://127.0.0.1:{redirect.server_address[1]}/api/download/models/1"
        dest = self.tmp / "pinned.bin"
        FASTDL.download(url, dest, conns=4, timeout=20)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        self.assertEqual(digest, hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(state["a_requests"], 1, "chunks must not re-resolve the redirect")
        self.assertEqual(state["b_requests"], 5,
                         f"probe + one request per chunk, saw {state['b_requests']}")

    def test_expired_pinned_link_re_resolves_and_finishes(self):
        redirect, state = self._pair(expire_after=1)   # signature dies after the first chunk
        url = f"http://127.0.0.1:{redirect.server_address[1]}/api/download/models/2"
        dest = self.tmp / "expired.bin"
        FASTDL.download(url, dest, conns=2, timeout=20)
        self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(),
                         hashlib.sha256(self.payload).hexdigest())
        self.assertGreaterEqual(state["a_requests"], 2, "a stale signature must trigger a re-probe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
