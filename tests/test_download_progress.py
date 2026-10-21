"""Progress reporting: reporter math, node wiring, Studio job payload.

Run with the ComfyUI embedded interpreter:
python tests/test_download_progress.py
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import socketserver
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _load_package():
    sys.path.insert(0, str(PLUGIN_ROOT.parent))
    spec = importlib.util.spec_from_file_location(
        PLUGIN_ROOT.name, PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PLUGIN_ROOT.name] = package
    spec.loader.exec_module(package)
    return sys.modules[PLUGIN_ROOT.name]


PACKAGE = _load_package()
FASTDL = PACKAGE.musefish_fastdl
NODE = PACKAGE.musefish_file_download


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class ReporterMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.snapshots = []
        self.reporter = FASTDL.ProgressReporter(self.snapshots.append, interval=0.25, clock=self.clock)

    def test_throttles_to_interval_but_always_reports_completion(self) -> None:
        total = 1000
        self.reporter(0, total)                 # first snapshot: baseline
        for done in range(0, 1001, 50):
            self.clock.tick(0.05)
            self.reporter(done, total)
        self.assertLess(len(self.snapshots), 10, "0.05 s calls must not each emit")
        self.assertEqual(self.snapshots[-1].done, total)
        self.assertAlmostEqual(self.snapshots[-1].percent, 100.0)
        percent = [s.percent for s in self.snapshots]
        self.assertEqual(percent, sorted(percent), "progress must never go backwards")

    def test_resume_baseline_is_not_a_speed_spike(self) -> None:
        total = 1000
        self.reporter(400, total)               # resumed: 400 bytes already on disk
        first = self.snapshots[-1]
        self.assertEqual(first.resumed, 400)
        self.assertEqual(first.speed, 0.0, "the resume jump is a baseline, not a rate")
        self.assertIsNone(first.eta)
        self.clock.tick(1.0)
        self.reporter(500, total)
        second = self.snapshots[-1]
        self.assertAlmostEqual(second.speed, 100.0, delta=1.0)
        self.assertAlmostEqual(second.eta, 5.0, delta=0.05)

    def test_speed_is_smoothed_and_eta_follows(self) -> None:
        total = 10_000
        self.reporter(0, total)
        self.clock.tick(1.0)
        self.reporter(1000, total)              # 1000 B/s
        self.clock.tick(1.0)
        self.reporter(1000, total)              # stalled window
        stalled = self.snapshots[-1]
        self.assertLess(stalled.speed, 1000.0, "a stalled window must drag the rate down")
        self.assertGreater(stalled.speed, 0.0, "smoothing keeps the rate from collapsing")

    def test_sink_failure_does_not_break_the_download(self) -> None:
        def boom(_progress):
            raise RuntimeError("frontend gone")

        reporter = FASTDL.ProgressReporter(boom, clock=self.clock)
        reporter(0, 100)
        reporter(100, 100)                      # must not raise

    def test_text_is_human_readable(self) -> None:
        self.reporter(0, 320 * 1024 * 1024)
        self.clock.tick(2.0)
        self.reporter(20 * 1024 * 1024, 320 * 1024 * 1024)
        text = self.snapshots[-1].text()
        self.assertIn("/ 320.0 MB", text)
        self.assertIn("%", text)
        self.assertIn("ETA", text)
        self.assertEqual(FASTDL.format_seconds(3671), "1:01:11")
        self.assertEqual(FASTDL.format_seconds(None), "--:--")
        self.assertEqual(FASTDL.format_bytes(1024 ** 3 * 1.5), "1.5 GB")


class _RangeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(payload: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 - silence
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            if rng == "bytes=0-0":
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-0/{len(payload)}")
                self.send_header("Content-Length", "1")
                self.send_header("ETag", '"p"')
                self.end_headers()
                self.wfile.write(payload[:1])
                return
            if not rng:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            start, end = (int(part) for part in rng.split("=")[1].split("-"))
            body = payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _RangeServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class NodeProgressTests(unittest.TestCase):
    def test_node_pushes_live_progress_text_and_bar_positions(self) -> None:
        payload = os.urandom(6 * 1024 * 1024)
        server = _serve(payload)
        port = server.server_address[1]
        texts: list[str] = []
        pushed: list[tuple[str, str]] = []
        bar_updates: list[int] = []

        class FakeBar:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, value, total=None, preview=None):
                bar_updates.append(int(value))

        class Cls(NODE.MusefishFileDownload):
            """ComfyUI clones the node per prompt and attaches UNIQUE_ID here."""

            class hidden:
                unique_id = "node-7"

        fake_server = type(sys)("server")
        fake_server.PromptServer = type("PS", (), {
            "instance": type("I", (), {"send_progress_text": staticmethod(lambda text, node: pushed.append((text, node)))})(),
        })
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(NODE.comfy.utils, "ProgressBar", FakeBar), \
             patch.dict(sys.modules, {"server": fake_server}), \
             patch.object(NODE, "print", lambda *a, **k: texts.append(" ".join(str(x) for x in a)), create=True):
            result = Cls.execute(
                url=f"http://127.0.0.1:{port}/model.bin", save_dir=temporary,
                filename="model.bin", conns=4, proxy=NODE.DIRECT_LABEL, overwrite=False,
                allow_any_dir=True)   # a temp save_dir is outside the default allowlist
            saved = Path(result[0])
            # the temp directory disappears with the context, so assert while it exists
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.stat().st_size, len(payload))
            self.assertIn("并发 4", result[1])
        server.shutdown()
        self.assertGreaterEqual(len(pushed), 2, "progress text must be pushed while downloading")
        self.assertTrue(all(node == "node-7" for _text, node in pushed))
        self.assertIn("/ 6.0 MB", pushed[0][0])
        # the first snapshot lands on the first 256 KiB read of a 6 MiB file
        self.assertIn("256.0 KB / 6.0 MB", pushed[0][0])
        self.assertTrue(any("ETA" in text for text, _ in pushed))
        self.assertEqual(bar_updates[-1], 100)
        self.assertEqual(bar_updates, sorted(bar_updates), "bar percent must increase")
        self.assertTrue(any(line.startswith("[MusefishFileDownload] ") and "/ 6.0 MB" in line for line in texts),
                        f"throttled console line missing: {texts[:3]}")


class SaveDirGuardTests(unittest.TestCase):
    def test_foreign_save_dir_needs_an_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(NODE.DownloadError) as ctx:
                NODE.MusefishFileDownload.execute(
                    url="http://127.0.0.1:9/never", save_dir=temporary,
                    filename="x.bin", conns=1, proxy=NODE.DIRECT_LABEL)
            self.assertIn("allow_any_dir", str(ctx.exception))


class StudioPayloadTests(unittest.TestCase):
    def test_studio_job_progress_carries_rate_and_eta(self) -> None:
        """The Studio task must publish the same numbers the node shows."""
        source = (PLUGIN_ROOT / "studio" / "server" / "app.py").read_text(encoding="utf-8")
        self.assertIn("fastdl.ProgressReporter(", source)
        self.assertIn('"eta"', source)
        for key in ("pct", "speed", "done", "total", "text"):
            self.assertIn(f'"{key}"', source)
        # and the deployed copy must be byte-identical to the packaged one
        deployed = Path(r"D:\MusefishStudio\server\app.py")
        if deployed.is_file():
            self.assertEqual(deployed.read_bytes(), (PLUGIN_ROOT / "studio" / "server" / "app.py").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
