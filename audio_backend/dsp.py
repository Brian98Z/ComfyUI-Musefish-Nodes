"""Portable mastering DSP copied from UniverSR's MIT-licensed scripts.

The functions here deliberately do not import the Gradio application.  They use
[Torchaudio/NumPy/SciPy] only and operate on float arrays shaped ``(samples,
channels)`` at 48 kHz.  The upstream UniverSR repository is MIT licensed:
https://github.com/woongzip1/UniverSR
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt, sosfilt

TARGET_SR = 48_000


def _moving_rms(x: np.ndarray, block: int) -> np.ndarray:
    block = max(1, int(block))
    if x.ndim == 1:
        return np.sqrt(np.convolve(x * x, np.ones(block) / block, mode="same"))
    return np.sqrt(
        np.stack(
            [np.convolve(x[:, c] * x[:, c], np.ones(block) / block, mode="same") for c in range(x.shape[1])],
            axis=1,
        )
    )


def eq_tone(final: np.ndarray, sr: int = TARGET_SR, gentle: bool = False) -> np.ndarray:
    sos_hpf = butter(4, 30, btype="high", fs=sr, output="sos")
    final = sosfiltfilt(sos_hpf, final, axis=0)
    n = final.shape[0]
    spec = np.fft.rfft(final, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.ones_like(freqs)
    w = np.clip((freqs - 40.0) / 80.0, 0, 1)
    gain *= 10 ** (0.5 / 20 * w)
    pk = np.zeros_like(freqs)
    pk_amp = 0.25 if gentle else 0.5
    m = (freqs >= 1800) & (freqs <= 4200)
    pk[m] = pk_amp * (1 + np.cos(np.pi * (freqs[m] - 3000) / 1200.0))
    gain *= 10 ** (pk / 20)
    w2 = np.clip((freqs - 7000) / 6000, 0, 1)
    gain *= 10 ** ((0.6 if gentle else 1.2) / 20 * w2)
    return np.fft.irfft(spec * gain[:, None], n=n, axis=0)


def de_esser(final: np.ndarray, sr: int = TARGET_SR, thr_db: float = -16.0,
             max_cut_db: float = 5.0, f_lo: float = 5500.0,
             f_hi: float = 8500.0) -> np.ndarray:
    sos_band = butter(4, [f_lo, f_hi], btype="band", fs=sr, output="sos")
    band = sosfiltfilt(sos_band, final, axis=0)
    env = _moving_rms(band, int(0.01 * sr)).max(axis=1)
    thr = 10 ** (thr_db / 20)
    over_db = 20 * np.log10(np.maximum(env, 1e-12) / thr)
    cut_db = np.minimum(np.maximum(over_db, 0), max_cut_db)
    g = 10 ** (-cut_db / 20)
    sos_g = butter(2, 10.0, btype="low", fs=sr, output="sos")
    g = np.clip(sosfiltfilt(sos_g, g), 0.0, 1.0)
    return final * g[:, None]


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
    for sig, thr_db, ratio in ((low, -12.0, 1.5), (mid, -10.0, 1.5), (high, -8.0, 1.3)):
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
    from scipy.signal import resample_poly
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
    g = np.convolve(g, np.ones(480) / 480, mode="same")
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


def _band_energy_db(sig: np.ndarray, sr: int, f0: float, f1: float) -> float:
    import librosa
    S = np.abs(librosa.stft(sig.T, n_fft=2048, hop_length=512)) ** 2
    S = S.mean(axis=0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    m = (freqs >= f0) & (freqs < f1)
    return float(10 * np.log10(S[m].mean() + 1e-12))


def hf_shape(final: np.ndarray, sr: int = TARGET_SR, drop1216_target: float = 27.0,
             d812_now: float | None = None, max_iter: int = 8) -> np.ndarray:
    n = final.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    y = final
    for it in range(max_iter):
        e08 = _band_energy_db(y, sr, 0, 8000)
        d812 = e08 - _band_energy_db(y, sr, 8000, 12000)
        d1216 = e08 - _band_energy_db(y, sr, 12000, 16000)
        d1620 = e08 - _band_energy_db(y, sr, 16000, 20000)
        err1216, err1620 = d1216 - drop1216_target, d1620 - 45.0
        if abs(err1216) < 0.4 and abs(err1620) < 0.4:
            break
        g1216 = 10 ** (err1216 / 40)
        g1620 = 10 ** (err1620 / 40)
        g812 = 10 ** ((15.0 - d812) / 40) if d812_now is not None and it == 0 and d812 > 15.0 else 1.0

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
    return y


def master_chain(st48: np.ndarray) -> np.ndarray:
    """The app's V8 master path; deliberately does not loudness-normalize."""
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
    d812 = _band_energy_db(final, sr, 0, 8000) - _band_energy_db(final, sr, 8000, 12000)
    final = hf_shape(final, sr, drop1216_target=27.0, d812_now=d812)
    tp_rise = 0.0
    for _ in range(4):
        tp = true_peak_db(final, sr)
        over = tp - (-0.3)
        if over <= 0:
            break
        tp_rise += over
        final = soft_limit(final, 10 ** ((-0.3 - tp_rise) / 20))
    d812_f = _band_energy_db(final, sr, 0, 8000) - _band_energy_db(final, sr, 8000, 12000)
    final = hf_shape(final, sr, drop1216_target=27.0, d812_now=d812_f)
    final = spectral_cut(final, sr, 19200)
    tp = true_peak_db(final, sr)
    if tp > -0.4:
        final *= 10 ** ((-0.4 - tp) / 20)
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
    # The model output is always TARGET_SR. Do not cap it at the low-rate
    # input Nyquist: that deletes the very band the SR model reconstructed.
    # Keep only the known generated-noise region above 19.2 kHz.
    keep_hz = 20000.0
    sos_lp = butter(8, keep_hz, btype="low", fs=TARGET_SR, output="sos")
    out = sosfiltfilt(sos_lp, st, axis=0)
    n = len(out)
    sp = np.fft.rfft(out, axis=0)
    fr = np.fft.rfftfreq(n, 1.0 / TARGET_SR)
    sp[fr > 19200.0, :] = 0
    out = np.fft.irfft(sp, n=n, axis=0)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if np.isfinite(peak) and peak > 1.0:
        out = out / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
