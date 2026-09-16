"""Focused behavior checks for UniverSR automatic input-rate selection.

Run with the ComfyUI embedded interpreter from the repository root, for example:
python -m unittest audio_backend.test_processing_bandwidth
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_backend import processing


SAMPLE_RATE = 48_000


def _weak_wideband_music(seconds: float = 2.0, low_hz: float = 8_300.0, high_hz: float = 11_800.0) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    time = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    body = 0.55 * np.sin(2 * np.pi * 440 * time) + 0.22 * np.sin(2 * np.pi * 880 * time)
    rng = np.random.default_rng(17)
    bins = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spectrum = np.zeros(len(bins), dtype=np.complex128)
    band = (bins >= low_hz) & (bins <= high_hz)
    spectrum[band] = rng.normal(size=int(band.sum())) + 1j * rng.normal(size=int(band.sum()))
    high = np.fft.irfft(spectrum, n=n)
    high *= 0.012 / max(float(np.std(high)), 1e-12)
    return body + high


class AutoInputRateBehaviorTests(unittest.TestCase):
    def test_general_weak_persistent_high_band_corrects_upward(self) -> None:
        selected, reason = processing.select_input_sample_rate(_weak_wideband_music(), SAMPLE_RATE, model_type="general")
        self.assertEqual(selected, 24_000)
        self.assertIn("base 8000", reason)
        self.assertIn("corrected 24000", reason)
        self.assertIn("persistent high-frequency evidence", reason)

    def test_high_band_maps_to_the_rate_that_preserves_it(self) -> None:
        mid, _ = processing.select_input_sample_rate(_weak_wideband_music(low_hz=6_500.0, high_hz=7_500.0), SAMPLE_RATE, model_type="general")
        self.assertEqual(mid, 16_000)
        high, _ = processing.select_input_sample_rate(_weak_wideband_music(low_hz=8_500.0, high_hz=11_000.0), SAMPLE_RATE, model_type="general")
        self.assertEqual(high, 24_000)

    def test_lower_sample_rate_containers_can_correct_up_to_their_bandwidth(self) -> None:
        for low, high, stride, expected in ((4_500, 5_500, 4, 12_000), (6_500, 7_500, 3, 16_000), (8_500, 11_000, 2, 24_000)):
            with self.subTest(sample_rate=SAMPLE_RATE // stride):
                wave = _weak_wideband_music(low_hz=low, high_hz=high)[::stride]
                selected, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE // stride, model_type="general")
                self.assertEqual(selected, expected)

    def test_noise_floor_and_isolated_high_tone_do_not_promote(self) -> None:
        time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        low = 0.6 * np.sin(2 * np.pi * 600 * time)
        noise = np.random.default_rng(11).normal(0, 1e-5, len(time))
        tone = 0.002 * np.sin(2 * np.pi * 10_000 * time)
        for residual in (noise, tone):
            selected, _ = processing.select_input_sample_rate(low + residual, SAMPLE_RATE, model_type="general")
            self.assertEqual(selected, 8_000)

    def test_stereo_layouts_have_identical_selection(self) -> None:
        mono = _weak_wideband_music()
        stereo = np.column_stack((mono, mono * 0.8))
        for wave in (stereo, stereo.T):
            selected, _ = processing.select_input_sample_rate(wave, SAMPLE_RATE, model_type="general")
            self.assertEqual(selected, 24_000)

    def test_true_low_bandwidth_stays_at_lowest_rate(self) -> None:
        time = np.arange(SAMPLE_RATE * 2, dtype=np.float64) / SAMPLE_RATE
        low = 0.6 * np.sin(2 * np.pi * 600 * time)
        selected, _ = processing.select_input_sample_rate(low, SAMPLE_RATE, model_type="general")
        self.assertEqual(selected, 8_000)

    def test_silence_noise_and_single_spike_do_not_upward_correct(self) -> None:
        silence, silence_reason = processing.select_input_sample_rate(np.zeros(SAMPLE_RATE), SAMPLE_RATE, model_type="general")
        self.assertEqual(silence, 8_000)
        self.assertIn("silent", silence_reason)

        noise = np.random.default_rng(9).normal(0.0, 0.03, SAMPLE_RATE * 2)
        noisy, noisy_reason = processing.select_input_sample_rate(noise, SAMPLE_RATE, model_type="general")
        self.assertLessEqual(noisy, 16_000)
        self.assertIn("no upward correction", noisy_reason)

        spike_wave = np.zeros(SAMPLE_RATE * 2)
        spike_wave[SAMPLE_RATE // 3] = 0.8
        spike, _ = processing.select_input_sample_rate(spike_wave, SAMPLE_RATE, model_type="general")
        self.assertEqual(spike, 8_000)

    def test_speech_keeps_historical_rolloff_selection(self) -> None:
        selected, reason = processing.select_input_sample_rate(_weak_wideband_music(), SAMPLE_RATE, model_type="speech")
        self.assertEqual(selected, 8_000)
        self.assertIn("disabled for non-general model", reason)

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


if __name__ == "__main__":
    unittest.main()
