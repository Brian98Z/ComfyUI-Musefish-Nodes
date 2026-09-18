"""Behavior checks for the V8 mastering DSP: de-esser band split, side decorrelation, headroom.

Run with the ComfyUI embedded interpreter from the repository root, for example:
python -m unittest audio_backend.test_dsp_chain
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audio_backend import dsp


SAMPLE_RATE = 48_000


def _band_level_db(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(10.0 * np.log10(float(np.mean(spectrum[band])) + 1e-20))


def _sibilant_material(seconds: float = 3.0, level: float = 1.0, seed: int = 21) -> np.ndarray:
    """Body plus sparse sibilance bursts (5.5-8.5k) and steady 12-20k content, at ``level``.

    The bursts have to be sparse (a fifth of the time): the detector's threshold is the
    band's own median +6 dB, so a fixture that is sibilant most of the time has no peaks
    left to catch.
    """
    n = int(SAMPLE_RATE * seconds)
    time = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    body = 0.30 * np.sin(2 * np.pi * 220 * time) + 0.12 * np.sin(2 * np.pi * 3000 * time)
    gate = np.zeros(n)
    for onset in range(int(seconds * 2.5)):
        start = int(onset * 0.4 * SAMPLE_RATE)
        gate[start:start + int(0.06 * SAMPLE_RATE)] = 1.0
    sibilance = 0.35 * np.sin(2 * np.pi * 7000 * time) * gate
    air = rng.normal(0.0, 0.02, n)
    air = sosfiltfilt(butter(4, [12000, 20000], btype="band", fs=SAMPLE_RATE, output="sos"), air)
    return np.stack([level * (body + sibilance + air)] * 2, axis=1)


class DeEsserTests(unittest.TestCase):
    def test_only_the_sibilance_band_is_attenuated(self) -> None:
        source = _sibilant_material()
        processed = dsp.de_esser(source, SAMPLE_RATE, max_cut_db=8.0)
        sibilance = _band_level_db(processed, 6000.0, 8000.0) - _band_level_db(source, 6000.0, 8000.0)
        self.assertLess(sibilance, -1.5)
        for low, high in ((12000.0, 16000.0), (16000.0, 20000.0)):
            with self.subTest(band=(low, high)):
                change = _band_level_db(processed, low, high) - _band_level_db(source, low, high)
                self.assertAlmostEqual(change, 0.0, delta=0.3)

    def test_attenuation_is_level_independent(self) -> None:
        # An absolute -16 dBFS threshold never fires on a chain that does not normalize;
        # the band's own median +6 dB keeps the same action at either level.
        loud = _sibilant_material(level=1.0)
        quiet = _sibilant_material(level=0.5)
        drop_loud = _band_level_db(dsp.de_esser(loud, SAMPLE_RATE), 6000.0, 8000.0) - _band_level_db(loud, 6000.0, 8000.0)
        drop_quiet = _band_level_db(dsp.de_esser(quiet, SAMPLE_RATE), 6000.0, 8000.0) - _band_level_db(quiet, 6000.0, 8000.0)
        self.assertLess(drop_loud, -1.5)
        self.assertLess(drop_quiet, -1.5)
        self.assertAlmostEqual(drop_loud, drop_quiet, delta=1.5)


class SideDecorrelationTests(unittest.TestCase):
    def test_band_limited_side_gains_a_decorrelated_14k_plus_band(self) -> None:
        n = SAMPLE_RATE * 3
        rng = np.random.default_rng(7)
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        mid = self._pink(rng, n)
        side = self._pink(rng, n)
        # The side is band-limited to the input rate, so its 14k+ is only the in-phase
        # excitation noise — L and R would stay identical up there.
        side = sosfiltfilt(butter(8, 12000, btype="low", fs=SAMPLE_RATE, output="sos"), side)
        # The render path hands the side over as a flat (n,) array; both ranks must work.
        for layout in (side, side[:, None]):
            with self.subTest(rank=layout.ndim):
                decorated = dsp.decorrelate_side_hf(layout, mid)
                self.assertEqual(decorated.ndim, layout.ndim)
                self.assertGreater(float(np.abs(decorated).max()), 0.0)
                flat = decorated if decorated.ndim == 1 else decorated[:, 0]

                band = (freqs >= 15500.0) & (freqs <= 19500.0)
                mid_band = np.abs(np.fft.rfft(mid))[band] ** 2
                side_band = np.abs(np.fft.rfft(flat))[band] ** 2
                self.assertAlmostEqual(float(np.mean(side_band) / np.mean(mid_band)), 0.16, delta=0.06)

                # With the +6 dB excitation the mix must be neither a mono top (corr 1.0)
                # nor anti-phase (negative); real recordings measure ~0.7.
                high = sosfiltfilt(butter(2, 6000, btype="high", fs=SAMPLE_RATE, output="sos"), flat)
                excited = flat + high * (10 ** (6.0 / 20) - 1)
                sos = butter(4, 14000, btype="high", fs=SAMPLE_RATE, output="sos")
                left = sosfiltfilt(sos, mid + excited / 2)
                right = sosfiltfilt(sos, mid - excited / 2)
                correlation = float(np.corrcoef(left, right)[0, 1])
                self.assertGreater(correlation, 0.4)
                self.assertLess(correlation, 0.95)

    @staticmethod
    def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        spectrum /= np.maximum(freqs, 20.0) ** 0.5
        wave = np.fft.irfft(spectrum, n=n)
        return wave * (0.3 / max(float(np.abs(wave).max()), 1e-9))


class MasterChainTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        n = SAMPLE_RATE * 3
        time = np.arange(n) / SAMPLE_RATE
        body = 0.25 * np.sin(2 * np.pi * 110 * time) + 0.10 * np.sin(2 * np.pi * 2200 * time)
        body = body + 0.05 * rng.standard_normal(n)
        self.signal = np.stack([body, body * 0.9 + 0.01 * rng.standard_normal(n)], axis=1)

    def test_requested_headroom_lowers_the_delivered_true_peak(self) -> None:
        default = dsp.master_chain(self.signal)
        reserved = dsp.master_chain(self.signal, peak_headroom_db=4.0)
        self.assertLessEqual(dsp.true_peak_db(default, SAMPLE_RATE), -0.35)
        self.assertLessEqual(dsp.true_peak_db(reserved, SAMPLE_RATE), -4.1)

    def test_hf_shape_never_lifts_an_empty_high_band(self) -> None:
        low_only = self.signal[:, 0]
        low_only = sosfiltfilt(butter(8, 8000, btype="low", fs=SAMPLE_RATE, output="sos"), low_only)
        low_only = np.stack([low_only, low_only], axis=1)
        before = dsp._band_energy_db(low_only, SAMPLE_RATE, 12000, 16000)
        shaped = dsp.hf_shape(low_only, SAMPLE_RATE, drop1216_target=27.0, d812_now=12.0)
        after = dsp._band_energy_db(shaped, SAMPLE_RATE, 12000, 16000)
        self.assertLessEqual(after - before, 0.5)


class BoxWindowTests(unittest.TestCase):
    """The box-window RMS and the shared band STFT are hot paths; both must stay exact."""

    def test_moving_rms_matches_the_direct_convolve(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(SAMPLE_RATE) * 0.3
        block = 2400
        reference = np.sqrt(np.convolve(signal * signal, np.ones(block) / block, mode="same"))
        np.testing.assert_allclose(dsp._moving_rms(signal, block), reference, rtol=1e-12, atol=1e-12)

    def test_moving_rms_stays_finite_on_digital_silence(self) -> None:
        """The FFT window sum can land a few ULP below zero where the signal is silent; without a
        clamp the square root is NaN and the whole chain (de-esser, multiband) turns to NaN."""
        silence = np.zeros((SAMPLE_RATE, 2))
        envelope = dsp._moving_rms(silence, 480)
        self.assertTrue(np.isfinite(envelope).all())
        np.testing.assert_array_equal(envelope, np.zeros_like(envelope))

    def test_shared_band_stft_matches_single_band_calls(self) -> None:
        rng = np.random.default_rng(11)
        signal = rng.standard_normal(SAMPLE_RATE * 2) * 0.2
        bands = [(0.0, 8000.0), (8000.0, 12000.0), (12000.0, 16000.0), (16000.0, 20000.0)]
        shared = dsp._band_energies_db(signal, SAMPLE_RATE, bands)
        separate = [dsp._band_energy_db(signal, SAMPLE_RATE, f0, f1) for f0, f1 in bands]
        self.assertEqual(shared, separate)


class BandMatchTests(unittest.TestCase):
    def test_missing_high_band_is_restored_up_to_the_cap(self) -> None:
        rng = np.random.default_rng(13)
        n = SAMPLE_RATE * 3
        source = rng.standard_normal(n) * 0.2
        dark = sosfiltfilt(butter(8, 7000, btype="low", fs=SAMPLE_RATE, output="sos"), source)
        restored = dsp.band_match(dark[:, None], source[:, None], 8000.0, 12000.0, 5.0)
        gain = dsp._band_energy_db(restored, SAMPLE_RATE, 8000, 12000) - dsp._band_energy_db(dark[:, None], SAMPLE_RATE, 8000, 12000)
        self.assertGreater(gain, 3.0)
        self.assertLessEqual(gain, 5.4)

    def test_brighter_material_is_left_alone(self) -> None:
        rng = np.random.default_rng(17)
        n = SAMPLE_RATE * 3
        source = rng.standard_normal(n) * 0.2
        bright = source + 0.3 * sosfiltfilt(butter(4, [10000, 20000], btype="band", fs=SAMPLE_RATE, output="sos"), source)
        result = dsp.band_match(bright[:, None], source[:, None], 8000.0, 12000.0, 12.0)
        np.testing.assert_array_equal(result, bright[:, None])


if __name__ == "__main__":
    unittest.main()
