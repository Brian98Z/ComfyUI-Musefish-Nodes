"""Batch audio must load the UniverSR checkpoint once per node call.

Run with the ComfyUI embedded interpreter:
python tests/test_audio_batch_worker.py
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 48_000


def _load_package():
    sys.path.insert(0, str(PLUGIN_ROOT.parent))
    spec = importlib.util.spec_from_file_location(
        PLUGIN_ROOT.name, PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PLUGIN_ROOT.name] = package
    spec.loader.exec_module(package)
    return sys.modules[PLUGIN_ROOT.name]


PACKAGE = _load_package()
MUSEFISH = PACKAGE.musefish_audio


def _audio(batch: int, seconds: float = 1.0) -> dict:
    samples = int(SAMPLE_RATE * seconds)
    wave = (torch.randn(batch, 2, samples) * 0.05).clamp(-1, 1)
    return {"waveform": wave, "sample_rate": SAMPLE_RATE}


class WorkerBatchProtocolTests(unittest.TestCase):
    """``--requests`` keeps one process for the whole batch and reports per index."""

    def test_batch_requests_emit_ordered_events(self) -> None:
        from importlib import import_module
        worker = import_module(f"{PACKAGE.__name__}.audio_backend.worker")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = []
            for index in range(2):
                path = root / f"request_{index}.json"
                path.write_text(json.dumps({"input_path": f"in{index}", "output_path": str(root / f"out{index}.wav")}),
                                encoding="utf-8")
                requests.append(str(path))
            calls = []

            def fake_process(request, progress=None, cancel=None):
                calls.append(request["input_path"])
                out = Path(request["output_path"])
                out.write_bytes(b"RIFF")
                if progress:
                    progress(50, "half")
                return out

            buffer = io.StringIO()
            with patch.object(worker, "process_request", side_effect=fake_process), \
                 patch.object(worker, "_PROTOCOL_OUT", buffer), \
                 patch.object(sys, "argv", ["worker.py", "--requests", *requests]), \
                 redirect_stdout(buffer):
                code = worker.main()
            events = [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["in0", "in1"])
        self.assertEqual([e["type"] for e in events],
                         ["start", "progress", "result", "start", "progress", "result"])
        self.assertEqual([e["index"] for e in events], [0, 0, 0, 1, 1, 1])
        self.assertEqual([Path(e["output_path"]).name for e in events if e["type"] == "result"],
                         ["out0.wav", "out1.wav"])

    def test_request_paths_are_mutually_exclusive(self) -> None:
        from importlib import import_module
        worker = import_module(f"{PACKAGE.__name__}.audio_backend.worker")
        for argv in (["worker.py"], ["worker.py", "--request", "a.json", "--requests", "b.json"]):
            stderr = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                worker._parse_args()


class NodeBatchDispatchTests(unittest.TestCase):
    def _run(self, batch: int, **overrides):
        single, batched = [], []

        def fake_worker(request, scope, batch_index, batch_total, log_lines, progress_bar):
            single.append(request)
            stub = scope / f"stub_{batch_index}.wav"
            stub.write_bytes(b"")
            return stub

        def fake_batch(requests, scope, batch_total, log_lines, progress_bar):
            batched.append(list(requests))
            # Write into the scope like the real implementation: a batch that runs
            # after the temporary directory is gone must fail here, not in production.
            paths = []
            for index in range(len(requests)):
                path = scope / f"stub_{index}.wav"
                path.write_bytes(b"")
                paths.append(path)
            return paths

        arguments = dict(
            audio=_audio(batch), mode="auto", model="general",
            allowed_modes=MUSEFISH._UNIVERSR_MODES, node_name="musefish_universr_general_audio",
            input_sr="16000", channel_mode="stereo", ode_method="midpoint", ode_steps=4,
            guidance=1.5, chunk_sec=15, seed=1, deess=True, auto_params=False, model_cache="",
        )
        arguments.update(overrides)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(MUSEFISH, "_universr_release_comfy_models"), \
             patch.object(MUSEFISH, "_universr_write_wav"), \
             patch.object(MUSEFISH, "_universr_run_worker", side_effect=fake_worker), \
             patch.object(MUSEFISH, "_universr_run_worker_batch", side_effect=fake_batch), \
             patch.object(MUSEFISH, "_universr_read_wav",
                          return_value=(torch.zeros(2, 512), SAMPLE_RATE)):
            result = MUSEFISH._universr_execute_audio(**arguments)
        return result, single, batched

    def test_single_clip_keeps_one_worker_per_item(self) -> None:
        result, single, batched = self._run(1)
        self.assertEqual(len(single), 1)
        self.assertEqual(batched, [])
        waveform = result[0]["waveform"]
        self.assertEqual(tuple(waveform.shape), (1, 2, 512))

    def test_multi_clip_batch_uses_one_worker_for_all_items(self) -> None:
        result, single, batched = self._run(3)
        self.assertEqual(single, [])
        self.assertEqual(len(batched), 1, "one worker per node call, not per clip")
        self.assertEqual(len(batched[0]), 3)
        self.assertEqual([Path(request["input_path"]).name for request in batched[0]],
                         ["input_0000.wav", "input_0001.wav", "input_0002.wav"])
        self.assertEqual(tuple(result[0]["waveform"].shape), (3, 2, 512))


if __name__ == "__main__":
    unittest.main(verbosity=2)
