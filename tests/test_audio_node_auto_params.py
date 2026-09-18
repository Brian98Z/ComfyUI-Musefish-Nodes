"""Node-level contract for the UniverSR ``auto_params`` switch.

Run with the ComfyUI embedded interpreter:
python tests/test_audio_node_auto_params.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 48_000


def _load_node_package():
    """Import the pack the way ComfyUI's loader does (package spec for a dashed name)."""
    sys.path.insert(0, str(PLUGIN_ROOT.parent))
    spec = importlib.util.spec_from_file_location(
        PLUGIN_ROOT.name, PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PLUGIN_ROOT.name] = package
    spec.loader.exec_module(package)
    return sys.modules[PLUGIN_ROOT.name].musefish_audio


MUSEFISH = _load_node_package()


def _low_bandwidth_audio(seconds: float = 3.0, seed: int = 5) -> dict:
    """Pink noise low-passed at 8 kHz — material the matcher scores as heavy restoration."""
    from scipy.signal import butter, sosfiltfilt
    n = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spectrum /= np.maximum(freqs, 20.0) ** 0.5
    wave = np.fft.irfft(spectrum, n=n)
    wave = sosfiltfilt(butter(8, 8000, btype="low", fs=SAMPLE_RATE, output="sos"), wave)
    wave *= 0.3 / max(float(np.abs(wave).max()), 1e-9)
    stereo = np.stack([wave, wave * 0.9], axis=0).astype(np.float32)
    return {"waveform": torch.from_numpy(stereo)[None, ...], "sample_rate": SAMPLE_RATE}


class AutoParameterWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[dict] = []
        self.logs: list[str] = []

    def _execute(self, omit: tuple[str, ...] = (), **overrides) -> dict:
        captured = self.requests

        def fake_worker(request, scope, batch_index, batch_total, log_lines, progress_bar):
            captured.append(dict(request))
            self.logs.extend(log_lines)
            stub = scope / "stub.wav"
            stub.write_bytes(b"")
            return stub

        arguments = dict(
            audio=overrides.pop("audio", _low_bandwidth_audio()),
            mode="auto",
            model="general",
            allowed_modes=MUSEFISH._UNIVERSR_MODES,
            node_name="MusefishUniverSRGeneralAudio",
            input_sr="16000",
            channel_mode="stereo",
            ode_method="midpoint",
            ode_steps=4,
            guidance=1.5,
            chunk_sec=15,
            seed=1,
            deess=True,
            auto_params=False,
            model_cache="",
        )
        arguments.update(overrides)
        for key in omit:
            arguments.pop(key, None)
        waveform = arguments["audio"]["waveform"]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(MUSEFISH, "_universr_release_comfy_models"), \
             patch.object(MUSEFISH, "_universr_write_wav"), \
             patch.object(MUSEFISH, "_universr_run_worker", side_effect=fake_worker), \
             patch.object(MUSEFISH, "_universr_read_wav",
                          return_value=(torch.zeros(2, int(waveform.shape[2])), SAMPLE_RATE)):
            MUSEFISH._universr_execute_audio(**arguments)
        return self.requests[-1]

    def test_declared_defaults_turn_auto_params_on(self) -> None:
        schema = MUSEFISH.MusefishUniverSRGeneralAudio.define_schema()
        defaults = {item.id: item.default for item in schema.inputs if item.id in {"deess", "auto_params"}}
        self.assertEqual(defaults, {"deess": True, "auto_params": True})

    def test_auto_params_is_the_required_first_widget(self) -> None:
        """The frontend renders widget inputs in input_order.required before optional, so being the
        only required widget input is what puts the switch above the parameters it governs."""
        schema = MUSEFISH.MusefishUniverSRGeneralAudio.define_schema()
        widget_inputs = [item for item in schema.inputs if item.id != "audio"]
        required = [item for item in widget_inputs if not getattr(item, "optional", False)]
        self.assertEqual([item.id for item in required], ["auto_params"])
        self.assertEqual(widget_inputs[0].id, "auto_params")
        self.assertFalse(getattr(required[0], "display_name", None))

    def test_speech_node_also_leads_with_the_switch(self) -> None:
        schema = MUSEFISH.MusefishUniverSRSpeechAudio.define_schema()
        widget_inputs = [item for item in schema.inputs if item.id != "audio"]
        self.assertEqual(widget_inputs[0].id, "auto_params")
        self.assertNotIn("mode", [item.id for item in widget_inputs])

    def test_governed_inputs_sit_directly_below_the_switch(self) -> None:
        """The switch hides these three, so they have to follow it - that adjacency is what makes the
        hidden set obvious on the node."""
        for node in (MUSEFISH.MusefishUniverSRGeneralAudio, MUSEFISH.MusefishUniverSRSpeechAudio):
            with self.subTest(node=node.__name__):
                ids = [item.id for item in node.define_schema().inputs if item.id not in {"audio", "model_cache"}]
                self.assertEqual(ids[:4], ["auto_params", "input_sr", "ode_steps", "guidance"])
                self.assertIn("mode" if node is MUSEFISH.MusefishUniverSRGeneralAudio else "deess", ids)

    def _extension(self) -> dict:
        source = (PLUGIN_ROOT / "web" / "musefish_universr_auto_params.js").read_text(encoding="utf-8")
        lists = {}
        governed = re.search(r"const GOVERNED_WIDGETS = \[(.*?)\];", source, re.S).group(1)
        lists["GOVERNED_WIDGETS"] = re.findall(r'"([^"]+)"', governed)
        for name in ("LEGACY_WIDGET_ORDER", "CURRENT_WIDGET_ORDER"):
            block = re.search(rf"const {name} = \{{(.*?)\n\}};", source, re.S).group(1)
            lists[name] = {node_id: re.findall(r'"([^"]+)"', body)
                           for node_id, body in re.findall(r"(\w+): \[(.*?)\]", block, re.S)}
        defaults_block = re.search(r"const WIDGET_DEFAULTS = \{(.*?)\n\};", source, re.S).group(1)
        defaults = {}
        for line in defaults_block.strip().splitlines():
            key, _, value = line.strip().rstrip(",").partition(": ")
            defaults[key] = json.loads(value)
        lists["WIDGET_DEFAULTS"] = defaults
        return lists

    def test_extension_lists_track_the_schema(self) -> None:
        """Widget order and defaults are duplicated in the extension; drift there would hide the wrong
        widget or restore a stale value."""
        ext = self._extension()
        for node_id, node in (("MusefishUniverSRGeneralAudio", MUSEFISH.MusefishUniverSRGeneralAudio),
                              ("MusefishUniverSRSpeechAudio", MUSEFISH.MusefishUniverSRSpeechAudio)):
            with self.subTest(node=node_id):
                schema_ids = [item.id for item in node.define_schema().inputs
                              if item.id not in {"audio", "model_cache"}]
                current = ext["CURRENT_WIDGET_ORDER"][node_id]
                # The frontend appends the seed control widget, which the schema does not declare.
                self.assertEqual(current, [name for item in schema_ids for name in
                                           ([item, "control_after_generate"] if item == "seed" else [item])])
                # Only `accel` was introduced after the reorder; anything else appearing on one
                # side but not the other means a name drifted and the migration would miss it.
                self.assertEqual(set(current) - set(ext["LEGACY_WIDGET_ORDER"][node_id]), {"accel"})
                self.assertEqual(set(ext["LEGACY_WIDGET_ORDER"][node_id]) - set(current), set())
                defaults = {item.id: item.default for item in node.define_schema().inputs
                            if item.id in ext["WIDGET_DEFAULTS"]}
                self.assertEqual({k: ext["WIDGET_DEFAULTS"][k] for k in defaults}, defaults)

    def test_extension_governs_exactly_the_hidden_three(self) -> None:
        ext = self._extension()
        self.assertEqual(ext["GOVERNED_WIDGETS"], ["input_sr", "ode_steps", "guidance"])
        self.assertEqual(ext["CURRENT_WIDGET_ORDER"]["MusefishUniverSRGeneralAudio"][:4],
                         ["auto_params", *ext["GOVERNED_WIDGETS"]])
        self.assertIn('WEB_DIRECTORY = "./web"', (PLUGIN_ROOT / "__init__.py").read_text(encoding="utf-8"))

    def test_accel_offers_exactly_the_backend_modes(self) -> None:
        """The node must offer exactly the modes the worker can apply, nothing inert, and its
        default must be the shipped backend default (cuDNN TF32)."""
        sys.path.insert(0, str(PLUGIN_ROOT))
        from audio_backend import processing as backend
        schema = MUSEFISH.MusefishUniverSRGeneralAudio.define_schema()
        accel = next(item for item in schema.inputs if item.id == "accel")
        self.assertEqual(tuple(accel.options), tuple(backend.ACCEL_MODES))
        self.assertEqual(accel.default, backend.DEFAULT_ACCEL)
        self.assertTrue(accel.optional)

    def test_accel_reaches_the_worker_request(self) -> None:
        request = self._execute(accel="cuDNN TF32")
        self.assertEqual(request["accel"], "cuDNN TF32")

    def test_unknown_accel_is_rejected_before_any_work(self) -> None:
        with self.assertRaises(ValueError):
            self._execute(accel="flash")

    def test_omitting_auto_params_still_matches_the_material(self) -> None:
        request = self._execute(omit=("auto_params",))
        self.assertEqual(request["mode"], "sr_master")
        self.assertEqual(request["ode_steps"], 16)
        self.assertTrue(any("auto params →" in line for line in self.logs), self.logs)

    def test_auto_params_rewrites_the_sampling_parameters(self) -> None:
        request = self._execute(auto_params=True)
        self.assertEqual(request["mode"], "sr_master")
        self.assertEqual(request["ode_steps"], 16)
        self.assertEqual(request["guidance"], 2.0)
        self.assertIs(request["deess"], True)
        self.assertTrue(any("auto params →" in line for line in self.logs), self.logs)

    def test_auto_params_keeps_the_node_mode_when_the_matcher_disagrees(self) -> None:
        request = self._execute(auto_params=True, allowed_modes=("sr",))
        self.assertEqual(request["mode"], "sr")
        self.assertEqual(request["ode_steps"], 16)

    def test_without_auto_params_the_node_values_stand(self) -> None:
        request = self._execute(auto_params=False, mode="sr", ode_steps=5, guidance=1.2, deess=False)
        self.assertEqual(request["mode"], "sr")
        self.assertEqual(request["ode_steps"], 5)
        self.assertEqual(request["guidance"], 1.2)
        self.assertIs(request["deess"], False)
        self.assertFalse(any("auto params →" in line for line in self.logs), self.logs)

    def test_speech_matching_keeps_super_resolution_and_slow_ladder(self) -> None:
        request = self._execute(model="speech", allowed_modes=("sr",), auto_params=True)
        self.assertEqual(request["mode"], "sr")
        self.assertEqual(request["guidance"], 1.5)
        self.assertEqual(request["ode_steps"], 16)
        self.assertIs(request["deess"], False)


if __name__ == "__main__":
    unittest.main()
