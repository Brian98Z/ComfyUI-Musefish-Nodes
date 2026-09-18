"""Portable mastering DSP copied from UniverSR's MIT-licensed scripts.

The functions here deliberately do not import the Gradio application.  They use
[Torchaudio/NumPy/SciPy] only and operate on float arrays shaped ``(samples,
channels)`` at 48 kHz.  The upstream UniverSR repository is MIT licensed:
https://github.com/woongzip1/UniverSR
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt, sosfilt

TARGET_SR = 48_000


def _moving_rms(x: np.ndarray, block: int) -> np.ndarray:
    """Box-window RMS.

    ``np.convolve`` with a 2400-tap box is O(n·block) and was 62% of the mastering chain on a
    30 s stereo file; ``fftconvolve`` computes the same sum in O(n log n) (0.54 s → 0.09 s) and
    agrees with the direct sum to ~1e-15 relative, far below the 24-bit floor.
    """
    block = max(1, int(block))
    window = np.full(block, 1.0 / block)
    # The window mean of squares is non-negative by construction, but the FFT sum can land a few
    # ULP below zero on digitally silent stretches, and sqrt() of that would poison the chain.
    if x.ndim == 1:
        return np.sqrt(np.maximum(fftconvolve(x * x, window, mode="same"), 0.0))
    return np.sqrt(
        np.maximum(
            np.stack(
                [fftconvolve(x[:, c] * x[:, c], window, mode="same") for c in range(x.shape[1])],
                axis=1,
            ),
            0.0,
        )
    )


def eq_tone(final: np.ndarray, sr: int = TARGET_SR, gentle: bool = False) -> np.ndarray:
    """30 Hz HPF / 80 Hz low shelf +0.5 dB / 3k peak +0.5 dB / 10k high shelf +1.2 dB.

    ``gentle`` halves the peak and shelf gains for vocal-sensitive material.  Time-domain
    IIR, not frequency-domain gains: the FFT version rang on transients (Gibbs spikes on
    drum hits), so the shelves stay zero-phase but no longer multiply a whole spectrum.
    """
    final = sosfiltfilt(butter(4, 30, btype="high", fs=sr, output="sos"), final, axis=0)
    low = sosfiltfilt(butter(2, 80, btype="low", fs=sr, output="sos"), final, axis=0)
    final = final + low * (10 ** (0.5 / 20) - 1)
    peak_db = 0.25 if gentle else 0.5
    mid = sosfiltfilt(butter(2, [1800, 4200], btype="band", fs=sr, output="sos"), final, axis=0)
    final = final + mid * (10 ** (peak_db / 20) - 1)
    shelf_db = 0.6 if gentle else 1.2
    high = sosfiltfilt(butter(2, 7000, btype="high", fs=sr, output="sos"), final, axis=0)
    return final + high * (10 ** (shelf_db / 20) - 1)


def de_esser(final: np.ndarray, sr: int = TARGET_SR, thr_db: float | None = None,
             max_cut_db: float = 5.0, f_lo: float = 5500.0,
             f_hi: float = 8500.0) -> np.ndarray:
    """Attenuate only the ``f_lo``-``f_hi`` component while its envelope exceeds threshold.

    Scaling the whole spectrum (the earlier implementation) dragged 12-16k and 16-20k down
    along with the sibilance; subtracting the filtered band and re-adding it scaled leaves
    the rest of the spectrum untouched.  ``thr_db=None`` derives the threshold from the
    band's own envelope (median +6 dB) so the detector is level independent — an absolute
    -16 dBFS threshold never triggers on a chain that does not normalize loudness.
    """
    sos_band = butter(4, [f_lo, f_hi], btype="band", fs=sr, output="sos")
    band = sosfiltfilt(sos_band, final, axis=0)
    env = _moving_rms(band, int(0.01 * sr)).max(axis=1)
    if thr_db is None:
        thr = max(float(np.percentile(env, 50)) * 10 ** (6.0 / 20), 1e-6)
    else:
        thr = 10 ** (thr_db / 20)
    over_db = 20 * np.log10(np.maximum(env, 1e-12) / thr)
    cut_db = np.minimum(np.maximum(over_db, 0), max_cut_db)
    g = 10 ** (-cut_db / 20)
    sos_g = butter(2, 10.0, btype="low", fs=sr, output="sos")
    g = np.clip(sosfiltfilt(sos_g, g), 0.0, 1.0)
    return final - band + band * g[:, None]


def multiband_comp(final: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    c1, c2 = 200.0, 6000.0
    sos_l = butter(4, c1, btype="low", fs=sr, output="sos")
    sos_m1 = butter(4, c1, btype="high", fs=sr, output="sos")
    sos_m2 = butter(4, c2, btype="low", fs=sr, output="sos")
    sos_h = butter(4, c2, btype="high", fs=sr, output="sos")
    low = sosfiltfilt(sos_l, final, axis=0)
    mid = sosfiltfilt(sos_m2, sosfiltfilt(sos_m1, final, axis=0), axis=0)
    high = sosfiltfilt(sos_h, final, axis=0)
    out = np.zeros_like(final)
    knee_w = 3.0
    # high threshold -8 → -4 dB: bell/cymbal transients were being squashed into spikes.
    for sig, thr_db, ratio in ((low, -12.0, 1.5), (mid, -10.0, 1.5), (high, -4.0, 1.3)):
        env = _moving_rms(sig, int(0.05 * sr))
        thr = 10 ** (thr_db / 20)
        over_db = 20 * np.log10(np.maximum(env, 1e-12) / thr)
        gr_db = np.where(
            over_db > knee_w,
            over_db * (1 - 1 / ratio),
            np.maximum(over_db, 0) ** 2 / (2 * knee_w) * (1 - 1 / ratio),
        )
        g = 10 ** (-gr_db / 20)
        sos_g = butter(2, 10.0, btype="low", fs=sr, output="sos")
        g = np.clip(sosfiltfilt(sos_g, g, axis=0), 0.0, 1.0)
        out += sig * g
    return out


def stereo_width_bands(final: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mid = (final[:, 0] + final[:, 1]) / 2
    side = (final[:, 0] - final[:, 1]) / 2
    sos_sm1 = butter(4, 200, btype="high", fs=sr, output="sos")
    sos_sm2 = butter(4, 6000, btype="low", fs=sr, output="sos")
    sos_sh = butter(4, 6000, btype="high", fs=sr, output="sos")
    side_mid = sosfiltfilt(sos_sm2, sosfiltfilt(sos_sm1, side, axis=0), axis=0)
    side_hi = sosfiltfilt(sos_sh, side, axis=0)
    side_out = side_mid * 0.5 + side_hi * 1.2
    return np.stack([mid + side_out, mid - side_out], axis=1)


def early_reflections(final: np.ndarray, sr: int = TARGET_SR, wet_db: float = -30.0) -> np.ndarray:
    wet = 10 ** (wet_db / 20)
    damp = butter(2, 7000, btype="low", fs=sr, output="sos")
    n = final.shape[0]
    er = np.zeros_like(final)
    for d_ms, g in ((5.0, 0.6), (11.0, 0.4), (23.0, 0.25)):
        d = int(d_ms * sr / 1000)
        tap = np.zeros(n)
        if d < n:
            tap[d:] = final[:-d].mean(axis=1) * g
        er[:, 0] += tap
        er[:, 1] += tap
    er = sosfiltfilt(damp, er, axis=0)
    return final + er * wet


def lufs_gated(x: np.ndarray, sr: int = TARGET_SR) -> float:
    sos_hs = butter(2, 1681, btype="high", fs=sr, output="sos")
    sos_hp = butter(2, 38, btype="high", fs=sr, output="sos")
    ys = [sosfilt(sos_hp, sosfilt(sos_hs, x[:, c])) for c in range(x.shape[1])]
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    n = max(1, (x.shape[0] - blk) // hop + 1)
    e = np.array([sum(np.mean(y[i * hop:i * hop + blk] ** 2) for y in ys) for i in range(n)])
    z = -0.691 + 10 * np.log10(e + 1e-12)
    m1 = z > -70
    if not m1.any():
        return float(z.max())
    rel = -0.691 + 10 * np.log10(e[m1].mean())
    m2 = m1 & (z > rel - 10)
    return float(-0.691 + 10 * np.log10(e[m2].mean()) if m2.any() else rel)


def true_peak_db(x: np.ndarray, sr: int = TARGET_SR) -> float:
    pk = 0.0
    for c in range(x.shape[1]):
        up = resample_poly(x[:, c].astype(np.float32), 4, 1)
        pk = max(pk, float(np.abs(up).max()))
    return 20 * np.log10(pk + 1e-12)


def soft_limit(x: np.ndarray, ceiling_lin: float) -> np.ndarray:
    env = np.abs(x).max(axis=1)
    g = np.ones_like(env)
    over = env > ceiling_lin
    if not over.any():
        return x
    g[over] = ceiling_lin / env[over]
    g = fftconvolve(g, np.full(480, 1.0 / 480), mode="same")
    g = np.minimum(g, 1.0)
    need = np.ones_like(env)
    need[over] = ceiling_lin / env[over]
    g = np.minimum(g, need)
    return x * g[:, None]


def spectral_cut(final: np.ndarray, sr: int = TARGET_SR, cut_hz: float = 19200.0) -> np.ndarray:
    spec = np.fft.rfft(final, axis=0)
    freqs = np.fft.rfftfreq(final.shape[0], 1.0 / sr)
    spec[freqs > cut_hz, :] = 0
    return np.fft.irfft(spec, n=final.shape[0], axis=0)


def _band_energies_db(sig: np.ndarray, sr: int, bands: Sequence[tuple[float, float]]) -> list[float]:
    """Mean power per band from a single STFT.

    ``hf_shape`` measured four bands per iteration with four separate transforms; sharing one
    transform is exact (each band is the same mean over the same power spectrum) and cuts those
    transforms by 4x, which matters because the loop can run eight iterations.
    """
    import librosa
    samples = np.asarray(sig)
    if samples.ndim == 1:
        samples = samples[:, None]
    S = np.abs(librosa.stft(samples.T, n_fft=2048, hop_length=512)) ** 2
    S = S.mean(axis=0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    energies = []
    for f0, f1 in bands:
        m = (freqs >= f0) & (freqs < f1)
        energies.append(float(10 * np.log10(S[m].mean() + 1e-12)))
    return energies


def _band_energy_db(sig: np.ndarray, sr: int, f0: float, f1: float) -> float:
    return _band_energies_db(sig, sr, [(f0, f1)])[0]


def hf_shape(final: np.ndarray, sr: int = TARGET_SR, drop1216_target: float = 27.0,
             d812_now: float | None = None, max_iter: int = 8) -> np.ndarray:
    """Drive the 12-16k / 16-20k falls toward 27 / 45 dB below 0-8k, measured per pass.

    Cut-only guard: a band's cumulative gain never exceeds the input level, so an already
    dark band is never lifted.  Without it a near-empty 12-16k band (a low step count
    measures a 68 dB fall) is boosted 20-40 dB chasing the target, which amplifies the
    model's noise floor, inflates true peak, and forces a deep, dynamic-flattening limit.
    """
    n = final.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    y = final
    prev1216 = prev1620 = 1.0            # cumulative band gain already applied
    for it in range(max_iter):
        e08, e812, e1216, e1620 = _band_energies_db(
            y, sr, [(0, 8000), (8000, 12000), (12000, 16000), (16000, 20000)])
        d812, d1216, d1620 = e08 - e812, e08 - e1216, e08 - e1620
        err1216, err1620 = d1216 - drop1216_target, d1620 - 45.0
        if abs(err1216) < 0.4 and abs(err1620) < 0.4:
            break
        # err < 0: band is brighter than target → cut (half step toward the target).
        # err > 0: band is already darker → only a restore that stays at or below the
        # input level is allowed, never a lift of an empty band.
        g1216 = min(10 ** (err1216 / 40), 1.0 / prev1216)
        g1620 = min(10 ** (err1620 / 40), 1.0 / prev1620)
        # 8-12k: cut only when the band is too bright; a dark band is left alone.
        g812 = 10 ** ((d812 - 15.0) / 40) if d812_now is not None and it == 0 and d812 < 15.0 else 1.0
        if max(abs(20 * np.log10(g1216)), abs(20 * np.log10(g1620)),
               abs(20 * np.log10(g812))) < 0.05:
            break                        # every band already at or below its target

        def ramp(fc: float, half: float):
            m = (freqs >= fc - half) & (freqs < fc + half)
            w = 0.5 * (1 - np.cos(np.pi * (freqs[m] - (fc - half)) / (2 * half)))
            return m, w

        gain = np.ones_like(freqs)
        m, w = ramp(11000, 1000)
        base = g812 * g1216
        gain[m] = 1 + w * (base - 1)
        gain[freqs >= 12000] = base
        m, w = ramp(16000, 1000)
        gain[m] = base + w * (base * g1620 - base)
        gain[freqs >= 17000] = base * g1620
        y = np.fft.irfft(np.fft.rfft(y, axis=0) * gain[:, None], n=n, axis=0)
        prev1216 *= base
        prev1620 *= base * g1620
    return y


def band_match(y48: np.ndarray, src48: np.ndarray, f0: float = 8000.0, f1: float = 12000.0,
               max_boost_db: float = 12.0) -> np.ndarray:
    """Lift the ``f0``-``f1`` band of ``y48`` toward the level in ``src48``, boost only."""
    n = min(len(y48), len(src48))
    sos = butter(4, [f0, f1], btype="band", fs=TARGET_SR, output="sos")
    e_y = float(np.mean(sosfiltfilt(sos, y48[:n], axis=0) ** 2)) + 1e-20
    e_s = float(np.mean(sosfiltfilt(sos, src48[:n], axis=0) ** 2)) + 1e-20
    gain_db = float(np.clip(10 * np.log10(e_s / e_y), 0.0, max_boost_db))
    if gain_db < 0.05:
        return y48
    freqs = np.fft.rfftfreq(len(y48), 1.0 / TARGET_SR)
    w = np.clip((freqs - (f0 - 1000.0)) / 1000.0, 0.0, 1.0) * np.clip(((f1 + 1000.0) - freqs) / 1000.0, 0.0, 1.0)
    gain = 1 + (10 ** (gain_db / 20) - 1) * np.clip(w, 0.0, 1.0)
    return np.fft.irfft(np.fft.rfft(y48, axis=0) * gain[:, None], n=len(y48), axis=0)


def soft_limit_tp2(x: np.ndarray, ceiling_lin: float, sr: int = TARGET_SR,
                   lookahead_ms: float = 1.5, release_ms: float = 150.0,
                   passes: int = 4) -> np.ndarray:
    """True-peak limiter: 4x-oversampled envelope, block lookahead, block-domain release.

    0.25 ms blocks keep a block minimum pinned to the real peaks (1 ms blocks poison ~60%
    of blocks and over-limit by ~1.9 dB), the gain curve steps down *before* the peak
    arrives, and fast-attack/slow-release beats the symmetric 10 ms window's pumping on
    deep gain reduction.  Each pass re-measures residual true peak and lowers the ceiling
    the next pass works against.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n0 = len(x)
    blk1 = max(1, int(0.00025 * sr))
    ceil_db = 20 * np.log10(ceiling_lin)
    ceil_cur = ceiling_lin
    la_blk = max(1, int(round(lookahead_ms / 0.25)))
    rel_k = 1.0 - np.exp(-0.25 / max(release_ms, 1.0))
    for _ in range(passes):
        up = np.stack([resample_poly(x[:, c].astype(np.float32), 4, 1) for c in range(x.shape[1])], axis=1)
        env = np.abs(up).max(axis=1)
        tp_now = 20 * np.log10(float(env.max()) + 1e-12)
        if tp_now <= ceil_db + 0.05:
            break
        over_db = tp_now - ceil_db
        ceil_cur = min(ceil_cur, ceil_cur * 10 ** (-over_db / 20 * 0.9))
        need = np.minimum(1.0, ceil_cur / np.maximum(env, 1e-12))
        nb = int(np.ceil(len(need) / (blk1 * 4)))
        pad = nb * blk1 * 4 - len(need)
        need_b = np.pad(need, (0, pad), constant_values=1.0).reshape(nb, blk1 * 4).min(axis=1)
        need_b = np.roll(need_b, -la_blk)
        need_b[-la_blk:] = 1.0
        g = np.empty_like(need_b)
        prev = 1.0
        for i in range(nb):
            rec = prev + (1.0 - prev) * rel_k
            v = need_b[i]
            prev = v if v < rec else rec
            g[i] = prev
        g_1x = np.repeat(g, blk1)[:n0]
        if len(g_1x) < n0:
            g_1x = np.pad(g_1x, (0, n0 - len(g_1x)), mode="edge")
        k1 = np.ones(blk1) / blk1
        g_1x = np.minimum(np.convolve(g_1x, k1, mode="same"), g_1x)
        x = x * g_1x[:, None]
    return x


def master_chain(st48: np.ndarray, peak_headroom_db: float = 0.0,
                 vocal_gentle: bool = False) -> np.ndarray:
    """The V8 master path; deliberately does not loudness-normalize.

    ``peak_headroom_db`` reserves true-peak margin for a downstream sum (the +6 dB side
    excitation and the M/S phase interference can push L/R back over a ceiling the mid
    alone already met).  ``vocal_gentle`` runs the band-split de-esser after limiting, the
    way the sr_master / stem_mix vocal paths do.
    """
    final = np.asarray(st48, dtype=np.float64).copy()
    if final.ndim == 1:
        final = np.stack([final, final], axis=1)
    sr = TARGET_SR
    n_fade = int(0.005 * sr)
    if len(final) > 2 * n_fade:
        fade = 0.5 * (1 - np.cos(np.pi * np.arange(n_fade) / n_fade))
        final[:n_fade] *= fade[:, None]
        final[-n_fade:] *= fade[::-1][:, None]
    final = eq_tone(final, sr, gentle=True)
    final = multiband_comp(final, sr)
    final = early_reflections(final, sr, wet_db=-30.0)
    e08, e812 = _band_energies_db(final, sr, [(0, 8000), (8000, 12000)])
    d812 = e08 - e812
    final = hf_shape(final, sr, drop1216_target=27.0, d812_now=d812)
    ceil_db = -0.3 - float(peak_headroom_db)
    tp_rise = 0.0
    for _ in range(4):
        tp = true_peak_db(final, sr)
        over = tp - ceil_db
        if over <= 0:
            break
        tp_rise += over
        final = soft_limit(final, 10 ** ((ceil_db - tp_rise) / 20))
    if vocal_gentle:
        final = de_esser(final, sr)
    e08_f, e812_f = _band_energies_db(final, sr, [(0, 8000), (8000, 12000)])
    d812_f = e08_f - e812_f
    final = hf_shape(final, sr, drop1216_target=27.0, d812_now=d812_f)
    final = spectral_cut(final, sr, 19200)
    limit_db = min(-0.4, ceil_db - 0.1)
    tp = true_peak_db(final, sr)
    if tp > limit_db:
        final *= 10 ** ((limit_db - tp) / 20)
    final = np.clip(final, -1, 1)
    if len(final) > 2 * n_fade:
        final[:n_fade] *= fade[:, None]
        final[-n_fade:] *= fade[::-1][:, None]
    return final


def denoise_gate(y48: np.ndarray, src48: np.ndarray) -> np.ndarray:
    sr = TARGET_SR
    sos_lp = butter(8, 20000, btype="low", fs=sr, output="sos")
    y = sosfiltfilt(sos_lp, y48, axis=0)
    n = len(y)
    sp = np.fft.rfft(y, axis=0)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    sp[fr > 19200, :] = 0
    y = np.fft.irfft(sp, n=n, axis=0)
    m = min(len(src48), n)
    sos_band = butter(4, [15000, 20000], btype="band", fs=sr, output="sos")
    hf_src = np.nan_to_num(sosfiltfilt(sos_band, src48[:m], axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    hf_y = np.nan_to_num(sosfiltfilt(sos_band, y[:m], axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    env = _moving_rms(hf_src, int(0.02 * sr))
    k = np.ones(int(0.05 * sr)) / int(0.05 * sr)
    env = np.clip(np.stack([np.convolve(env[:, c], k, "same") for c in range(2)], axis=1), 0.0, 10.0)
    ref = np.maximum(np.percentile(env, 90, axis=0, keepdims=True), 1e-8)
    gdb = np.clip((20 * np.log10(np.maximum(env, 1e-8)) - (ref - 20.0)) / 4.0, -20.0, 0.0)
    out = y[:m] - hf_y + hf_y * 10 ** (gdb / 20)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def brighten(y48: np.ndarray, gain_db: float = 1.5) -> np.ndarray:
    n = len(y48)
    sp = np.fft.rfft(y48, axis=0)
    fr = np.fft.rfftfreq(n, 1.0 / TARGET_SR)
    w = np.clip((fr - 2500) / 1500, 0, 1) * np.clip((9000 - fr) / 1000, 0, 1)
    g = 1 + (10 ** (gain_db / 20) - 1) * np.clip(w, 0, 1)
    return np.fft.irfft(sp * g[:, None], n=n, axis=0)


def clean_sr_output(st: np.ndarray, orig_sr: int, input_sr: int) -> np.ndarray:
    # The model output is always TARGET_SR. Do not cap it at the low-rate input Nyquist:
    # that deletes the very band the SR model reconstructed.  What is above the container's
    # own Nyquist is generated by definition, so the cut sits at the smaller of the two
    # (4% transition) — this keeps generated noise out without touching rebuilt content.
    keep_hz = min(max(int(input_sr), int(orig_sr)) / 2.0, 20000.0)
    sos_lp = butter(8, keep_hz, btype="low", fs=TARGET_SR, output="sos")
    out = sosfiltfilt(sos_lp, st, axis=0)
    n = len(out)
    sp = np.fft.rfft(out, axis=0)
    fr = np.fft.rfftfreq(n, 1.0 / TARGET_SR)
    sp[fr > keep_hz * 0.96, :] = 0
    out = np.fft.irfft(sp, n=n, axis=0)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if np.isfinite(peak) and peak > 1.0:
        out = out / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def decorrelate_side_hf(side_bl: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Replace the side's 14k+ content with noise shaped and scaled from ``mid``.

    Band-limiting the side to the input rate leaves 14k+ with nothing but the in-phase
    excitation noise, so L and R stay identical up there (measured correlation 0.998 —
    a mono high end).  Noise is shaped by the side's own 6k+ envelope and calibrated to
    E[decor @15.5-19.5k] = 0.16 x E[mid] over the same band; with the +6 dB excitation that
    lands at corr(L,R) ~ +0.72, matching real recordings instead of a fake mono top.
    Accepts ``(n,)`` or ``(n, 1)`` and returns the same rank.
    """
    side = np.asarray(side_bl, dtype=np.float64)
    one_dimensional = side.ndim == 1
    if one_dimensional:
        side = side[:, None]
    n_bl = len(side)
    freqs = np.fft.rfftfreq(n_bl, 1.0 / TARGET_SR)
    sos_hs = butter(2, 6000, btype="high", fs=TARGET_SR, output="sos")
    envelope = np.abs(sosfiltfilt(sos_hs, side, axis=0))
    window = np.ones(int(0.05 * TARGET_SR)) / int(0.05 * TARGET_SR)
    envelope = np.stack([np.convolve(envelope[:, c], window, mode="same") for c in range(envelope.shape[1])], axis=1)
    rng = np.random.default_rng(1970)
    noise = sosfiltfilt(sos_hs, rng.standard_normal(n_bl))
    noise = sosfiltfilt(butter(4, 19000, btype="low", fs=TARGET_SR, output="sos"), noise)
    decor = (noise * envelope.mean(axis=1) * 0.5)[:, None]
    reference = np.asarray(mid, dtype=np.float64)
    if reference.ndim > 1:
        reference = reference[:, 0]
    if len(reference) < n_bl:
        reference = np.pad(reference, (0, n_bl - len(reference)))
    band = (freqs >= 15500.0) & (freqs <= 19500.0)
    e_mid = float(np.mean(np.abs(np.fft.rfft(reference[:n_bl]))[band] ** 2)) + 1e-20
    e_dec = float(np.mean(np.abs(np.fft.rfft(decor[:, 0]))[band] ** 2)) + 1e-20
    decor *= np.sqrt(0.16 * e_mid / e_dec)
    weight = np.clip((freqs - 14000.0) / 1500.0, 0.0, 1.0)[:, None]
    out = np.fft.irfft(np.fft.rfft(side, axis=0) * (1 - weight) + np.fft.rfft(decor, axis=0) * weight,
                       n=n_bl, axis=0)
    return out[:, 0] if one_dimensional else out
