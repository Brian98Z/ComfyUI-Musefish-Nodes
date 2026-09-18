"""Focused behavior checks for UniverSR automatic input-rate and parameter matching.

Run with the ComfyUI embedded interpreter from the repository root, for example:
python -m unittest audio_backend.test_processing_bandwidth
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audio_backend import processing


SAMPLE_RATE = 48_000


def _band_limited_music(cutoff_hz: float | None = None, seconds: float = 6.0, seed: int = 5) -> np.ndarray:
    """Pink noise plus two tones, optionally low-passed — program material, not a test tone.

    A tone body with a detached high band is not representative: the content-cutoff probe
    (first frequency >20 dB below the rolloff level) reads the gap and reports the body.
    """
    n = int(SAMPLE_RATE * seconds)
    time = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spectrum /= np.maximum(freqs, 20.0) ** 0.5
    wave = np.fft.irfft(spectrum, n=n)
    wave *= 0.30 / max(float(np.abs(wave).max()), 1e-9)
    wave = wave + 0.10 * np.sin(2 * np.pi * 120 * time) + 0.05 * np.sin(2 * np.pi * 3000 * time)
    if cutoff_hz:
        wave = sosfiltfilt(butter(8, cutoff_hz, btype="low", fs=SAMPLE_RATE, output="sos"), wave)
    return wave


class AutoInputRateBehaviorTests(unittest.TestCase):
    def test_analysis_reports_content_cutoff_next_to_the_rolloff(self) -> None:
        _, reason, _, _, _, _, content_cutoff = processing._bandwidth_analysis(_band_limited_music(6000.0), SAMPLE_RATE)
        self.assertIn("content cutoff", reason)
        self.assertGreater(content_cutoff, 4000.0)

    def test_low_bandwidth_keeps_8k_and_skips_the_upward_correction(self) -> None:
        wave = _band_limited_music(3500.0)
        selected, reason = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(selected, 8_000)
        self.assertIn("upward correction skipped", reason)
        self.assertIn("<= 5000 Hz", reason)

    def test_edge_band_trials_8k_then_corrects_up_on_evidence(self) -> None:
        wave = _band_limited_music(4500.0)
        selected, reason = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(selected, 12_000)
        self.assertIn("base 8000", reason)
        self.assertIn("persistent high-frequency evidence", reason)
        speech, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="speech")
        self.assertEqual(speech, 8_000)

    def test_mid_bandwidth_maps_to_16k(self) -> None:
        wave = _band_limited_music(5000.0)
        for model in ("general", "speech"):
            with self.subTest(model=model):
                selected, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type=model)
                self.assertEqual(selected, 16_000)

    def test_high_bandwidth_corrects_to_24k_for_general_only(self) -> None:
        wave = _band_limited_music(8000.0)
        selected, reason = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(selected, 24_000)
        self.assertIn("corrected 24000", reason)
        speech, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="speech")
        self.assertEqual(speech, 16_000)

    def test_full_bandwidth_takes_the_top_tier(self) -> None:
        wave = _band_limited_music(12000.0)
        for model in ("general", "speech"):
            with self.subTest(model=model):
                selected, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type=model)
                self.assertEqual(selected, 24_000)

    def test_stereo_layouts_have_identical_selection(self) -> None:
        mono = _band_limited_music(8000.0)
        stereo = np.column_stack((mono, mono * 0.8))
        for wave in (stereo, stereo.T):
            with self.subTest(shape=wave.shape):
                selected, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="general")
                self.assertEqual(selected, 24_000)

    def test_silence_noise_and_single_spike_stay_at_the_lowest_rate(self) -> None:
        silence, silence_reason = processing.select_input_sample_rate(np.zeros(SAMPLE_RATE), SAMPLE_RATE, model_type="general")
        self.assertEqual(silence, 8_000)
        self.assertIn("silent", silence_reason)

        noise = np.random.default_rng(9).normal(0.0, 0.03, SAMPLE_RATE * 2)
        noisy, noisy_reason = processing.select_input_sample_rate(noise, SAMPLE_RATE, model_type="general")
        self.assertEqual(noisy, 8_000)
        self.assertIn("no spectrum >10 dB above noise floor", noisy_reason)

        spike_wave = np.zeros(SAMPLE_RATE * 2)
        spike_wave[SAMPLE_RATE // 3] = 0.8
        spike, _ = processing.select_input_sample_rate(spike_wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(spike, 8_000)

    def test_manual_rate_does_not_invoke_auto_detector(self) -> None:
        source = np.zeros((SAMPLE_RATE, 1), dtype=np.float64)
        rendered = np.zeros((SAMPLE_RATE, 2), dtype=np.float64)
        request = {
            "input_path": "manual-input.wav",
            "output_path": "manual-output.wav",
            "mode": "sr",
            "model": "general",
            "input_sr": 16_000,
            "channel_mode": "stereo",
            "universr_root": str(Path(__file__).parent / "vendor"),
        }
        with patch.object(processing, "_read_audio", return_value=(source, SAMPLE_RATE)), \
             patch.object(processing, "_resample_stereo", return_value=source), \
             patch.object(processing, "_configure_environment", return_value=Path(".")), \
             patch.object(processing, "_seed_everything"), \
             patch.object(processing, "_load_model", return_value=object()), \
             patch.object(processing, "_sr_render", return_value=rendered), \
             patch.object(processing, "_write_audio"), \
             patch.object(processing, "select_input_sample_rate") as detector:
            processing.process_request(request)
        detector.assert_not_called()


def _flat_material(seconds: float = 6.0, seed: int = 5) -> np.ndarray:
    """Full-bandwidth material with a level high band (need 0 → nothing to restore)."""
    n = int(SAMPLE_RATE * seconds)
    wave = np.random.default_rng(seed).standard_normal(n)
    return wave * (0.30 / max(float(np.abs(wave).max()), 1e-9))


def _boosted_band(wave: np.ndarray, low_hz: float, high_hz: float, gain_db: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(len(wave), 1.0 / SAMPLE_RATE)
    weight = np.clip(np.minimum((freqs - (low_hz - 500)) / 500, (high_hz + 500 - freqs) / 500), 0.0, 1.0)
    return np.fft.irfft(np.fft.rfft(wave) * (1 + (10 ** (gain_db / 20) - 1) * weight), n=len(wave))


class MaterialParameterMatchTests(unittest.TestCase):
    def test_full_bandwidth_material_skips_super_resolution(self) -> None:
        matched = processing.recommend_processing(_flat_material(), SAMPLE_RATE, model_type="general")
        self.assertEqual(matched["mode"], "master")
        self.assertEqual(matched["metrics"]["need"], 0.0)
        self.assertEqual((matched["guidance"], matched["steps"], matched["deess"]), (2.0, 16, True))

    def test_speech_keeps_super_resolution_and_takes_the_fast_ladder(self) -> None:
        matched = processing.recommend_processing(_flat_material(), SAMPLE_RATE, model_type="speech")
        self.assertEqual(matched["mode"], "sr")
        self.assertEqual((matched["guidance"], matched["steps"], matched["deess"]), (1.0, 8, False))

    def test_low_bandwidth_material_takes_the_restoration_ladder(self) -> None:
        wave = _band_limited_music(8000.0)
        general = processing.recommend_processing(wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(general["mode"], "sr_master")
        self.assertEqual(general["metrics"]["need"], 1.0)
        speech = processing.recommend_processing(wave, SAMPLE_RATE, model_type="speech")
        self.assertEqual((speech["mode"], speech["guidance"], speech["steps"]), ("sr", 1.5, 16))
        self.assertFalse(speech["deess"])

    def test_sibilant_material_enables_the_de_esser_for_speech(self) -> None:
        wave = _boosted_band(_band_limited_music(None), 6500.0, 7500.0, 8.0)
        matched = processing.recommend_processing(wave, SAMPLE_RATE, model_type="speech")
        self.assertGreaterEqual(matched["metrics"]["d6"], processing.SIBILANCE_D6_DB)
        self.assertTrue(matched["deess"])

    def test_unreadable_material_falls_back_to_per_model_defaults(self) -> None:
        silence = np.zeros(SAMPLE_RATE * 3)
        general = processing.recommend_processing(silence, SAMPLE_RATE, model_type="general")
        speech = processing.recommend_processing(silence, SAMPLE_RATE, model_type="speech")
        self.assertEqual((general["mode"], general["guidance"], general["steps"], general["deess"]),
                         ("sr_master", 2.0, 16, True))
        self.assertEqual((speech["mode"], speech["guidance"], speech["steps"], speech["deess"]),
                         ("sr", 1.5, 8, False))
        self.assertIn("unreadable", general["reason"])


class AccelerationSelectionTests(unittest.TestCase):
    """Only the acceleration paths that measured faster are offered; attention backends are not.

    Measured on one 9.5 s chunk at 16 steps (RTX 5070 Ti): cuDNN TF32 1.18x, bf16 1.35x (with an
    audibly different waveform), fp16 NaN, channels_last 0.69x, cudnn.benchmark noise.
    """

    def test_offered_modes_are_the_measured_ones(self) -> None:
        self.assertEqual(processing.ACCEL_MODES, ("fp32", "cuDNN TF32", "bf16"))
        self.assertEqual(processing.DEFAULT_ACCEL, "cuDNN TF32")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            processing.resolve_accel("flash")
        with self.assertRaises(ValueError):
            processing.resolve_accel("Kitchen")

    def test_missing_cuda_drops_back_to_fp32(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(processing.resolve_accel("bf16"), "fp32")
            self.assertEqual(processing.resolve_accel(processing.DEFAULT_ACCEL), "fp32")

    def test_the_shipped_default_turns_tf32_on(self) -> None:
        """The default must actually enable both TF32 matmul and cuDNN flags, and restore them
        (matmul TF32 is off in a fresh torch, cuDNN TF32 is on, so "back to the previous value"
        is the only correct expectation)."""
        import torch
        self.assertEqual(processing.DEFAULT_ACCEL, "cuDNN TF32")
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with processing.accel_context(processing.DEFAULT_ACCEL):
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(
            (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32), before
        )

    def test_fp32_mode_leaves_torch_untouched(self) -> None:
        import torch
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with processing.accel_context("fp32"):
            self.assertEqual((torch.backends.cudnn.allow_tf32,
                              torch.backends.cuda.matmul.allow_tf32), before)
        self.assertEqual((torch.backends.cudnn.allow_tf32,
                          torch.backends.cuda.matmul.allow_tf32), before)

    def test_tf32_mode_enables_the_two_flags_for_its_scope_only(self) -> None:
        import torch
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with processing.accel_context("cuDNN TF32"):
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual((torch.backends.cudnn.allow_tf32,
                          torch.backends.cuda.matmul.allow_tf32), before)

    def test_bf16_mode_autocasts_inside_the_context(self) -> None:
        import torch
        with processing.accel_context("bf16"):
            inside = torch.is_autocast_enabled("cuda")
        self.assertTrue(inside)
        self.assertFalse(torch.is_autocast_enabled("cuda"))


if __name__ == "__main__":
    unittest.main()
