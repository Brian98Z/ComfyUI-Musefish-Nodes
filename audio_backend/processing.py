"""Portable four-mode UniverSR processing, independent of Gradio/universr_app."""
from __future__ import annotations
import os, random, signal, sys, tempfile, time
from contextlib import contextmanager
from math import gcd
from pathlib import Path
from typing import Callable
import numpy as np
import soundfile as sf
from . import dsp

TARGET_SR = 48000
SUPPORTED_INPUT_SR = (8000, 12000, 16000, 24000)
MODEL_REPOS = {"general": "woongzip1/universr-audio", "speech": "woongzip1/universr-speech"}
# Measured on one 9.5 s chunk, 16 ODE steps, RTX 5070 Ti: fp32 is the bit-exact baseline,
# cuDNN TF32 is 1.18x with a max sample diff of 3.9e-3 (~ -48 dBFS) and is therefore the default,
# bf16 autocast is 1.35x with a max diff of 0.26 (~ -11.7 dB, opt-in only). fp16 autocast
# overflowed to NaN, channels_last was 0.69x (slower), cudnn.benchmark was noise, and the model
# calls no attention at all (0 scaled_dot_product_attention calls per enhance), so the attention
# backends ComfyUI exposes cannot accelerate this model.
ACCEL_MODES = ("fp32", "cuDNN TF32", "bf16")
DEFAULT_ACCEL = "cuDNN TF32"
FALLBACK_ACCEL = "fp32"
Progress = Callable[[int, str], None]
_MODEL_CACHE: dict[tuple[str, str], object] = {}

def _noop_progress(percent: int, message: str) -> None: pass

def resolve_accel(name: str) -> str:
    """Validate the acceleration choice and drop what this machine cannot run."""
    mode = str(name or DEFAULT_ACCEL)
    if mode not in ACCEL_MODES:
        raise ValueError(f"unknown accel {mode!r}; choose one of {ACCEL_MODES}")
    import torch
    if mode != FALLBACK_ACCEL and not torch.cuda.is_available():
        print(f"accel {mode!r} needs CUDA; using {FALLBACK_ACCEL}", file=sys.stderr, flush=True)
        return FALLBACK_ACCEL
    return mode


@contextmanager
def accel_context(mode: str):
    """Apply the acceleration setting for the duration of the block, then restore it."""
    import torch
    if mode == "cuDNN TF32":
        previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield mode
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous
    elif mode == "bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield mode
    else:
        yield mode

def _seed_everything(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) & 0xffffffff)
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except Exception as exc:
        print(f"deterministic torch setup unavailable: {exc}", file=sys.stderr, flush=True)

def _configure_environment(request: dict) -> Path:
    package_root = Path(__file__).resolve().parent
    root_value = request.get("universr_root") or os.environ.get("UNIVERSR_ROOT")
    root = Path(str(root_value)).expanduser() if root_value else package_root / "vendor"
    if not root.exists():
        raise FileNotFoundError(f"universr_root does not exist: {root}")
    cache = Path(str(request.get("model_cache") or os.environ.get("UNIVERSR_MODEL_CACHE") or root / "models")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    hf_cache = cache if cache.name == "huggingface" else cache / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache)); os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if str(root.resolve()) not in sys.path: sys.path.insert(0, str(root.resolve()))
    return cache

def _model_directory(cache: Path, model_type: str) -> Path | None:
    name = "universr-audio" if model_type == "general" else "universr-speech"
    # "general"/"speech" are the directory names MusefishUniverSRModel writes, so a cache
    # root holding both of them serves sr and stem_mix (which needs both models at once).
    candidates = (cache, cache / name, cache / model_type, cache / f"models--woongzip1--{name}",
                  cache / "huggingface" / name, cache / "huggingface" / f"models--woongzip1--{name}")
    for candidate in candidates:
        if (candidate / "config.yaml").is_file() and (candidate / "pytorch_model.bin").is_file(): return candidate
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            found = sorted((p for p in snapshots.iterdir() if (p / "config.yaml").is_file() and (p / "pytorch_model.bin").is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
            if found: return found[0]
    return None

def _load_model(model_type: str, cache: Path):
    import torch
    from universr import UniverSR
    device = os.environ.get("UNIVERSR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    local = _model_directory(cache, model_type); source = str(local) if local else MODEL_REPOS[model_type]
    key = (model_type, device + "|" + source)
    if key in _MODEL_CACHE: return _MODEL_CACHE[key]
    print(f"loading UniverSR-{model_type} from {source} on {device}", file=sys.stderr, flush=True)
    model = UniverSR.from_pretrained(source, device=device)
    model.eval(); _MODEL_CACHE[key] = model
    return model

def _resample(signal_1d: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if int(orig_sr) == int(new_sr): return np.asarray(signal_1d, dtype=np.float64).copy()
    from scipy.signal import resample_poly
    g = gcd(int(orig_sr), int(new_sr))
    return resample_poly(np.asarray(signal_1d, dtype=np.float64), int(new_sr) // g, int(orig_sr) // g).astype(np.float64, copy=False)

def _resample_stereo(x: np.ndarray, orig_sr: int, new_sr: int = TARGET_SR) -> np.ndarray:
    if x.ndim == 1: x = x[:, None]
    if int(orig_sr) == int(new_sr): return x.astype(np.float64, copy=True)
    channels = [_resample(x[:, c], orig_sr, new_sr) for c in range(x.shape[1])]; n = min(map(len, channels))
    return np.stack([c[:n] for c in channels], axis=1)
def _bandwidth_analysis(waveform: np.ndarray, sample_rate: int) -> tuple[float, str, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return robust bandwidth facts and the already-computed frame spectra."""
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim == 2:
        # Accept both Comfy AUDIO [C,T] and decoded soundfile [T,C] layouts.
        if values.shape[0] <= 2 and values.shape[1] > values.shape[0]:
            values = values.T
        values = values.mean(axis=1)
    elif values.ndim != 1:
        raise ValueError(f"bandwidth estimator expects 1-D or 2-D waveform, got {values.shape}")
    if not values.size or not np.isfinite(values).all():
        raise ValueError("bandwidth estimator requires finite, non-empty waveform")
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-8:
        empty = np.empty((0, 0), dtype=np.float64)
        return 0.0, "silent/near-silent waveform", empty, empty, empty, 0.0, 0.0

    n_fft = min(4096, 1 << max(8, int(np.floor(np.log2(max(256, values.size))))) )
    hop = max(n_fft // 2, 1)
    if values.size < n_fft:
        frames = np.pad(values, (0, n_fft - values.size))[None, :]
    else:
        starts = np.arange(0, values.size - n_fft + 1, hop)
        if starts.size > 96:
            starts = np.linspace(starts[0], starts[-1], 96, dtype=np.int64)
        frames = np.stack([values[int(start):int(start) + n_fft] for start in starts])
    window = np.hanning(n_fft)
    spectrum_frames = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) ** 2
    power = np.median(spectrum_frames, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid = freqs <= sr / 2.0
    power = np.maximum(power[valid], 1e-20)
    freqs = freqs[valid]
    spectrum_frames = np.maximum(spectrum_frames[:, valid], 1e-20)
    if power.size < 4:
        return sr / 2.0, "short waveform; using available Nyquist", freqs, spectrum_frames, power, 0.0, sr / 2.0

    # Smooth at roughly 100 Hz, then estimate a robust broadband noise floor.
    smooth_bins = max(1, int(round(100.0 * n_fft / sr)))
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
        power_smooth = np.convolve(power, kernel, mode="same")
    else:
        power_smooth = power
    log_power = 10.0 * np.log10(np.maximum(power_smooth, 1e-20))
    noise_db = float(np.percentile(log_power, 20.0))
    significant = log_power - noise_db >= 10.0
    excess = np.where(significant, np.maximum(power_smooth - 10 ** ((noise_db + 6.0) / 10.0), 0.0), 0.0)
    total = float(excess.sum())
    if total <= 1e-18:
        return 0.0, f"no spectrum >10 dB above noise floor ({noise_db:.1f} dB)", freqs, spectrum_frames, power_smooth, noise_db, 0.0
    cumulative = np.cumsum(excess)
    rolloff_index = int(np.searchsorted(cumulative, total * 0.99))
    cutoff = float(freqs[min(rolloff_index, len(freqs) - 1)])
    # Ignore tiny isolated spikes: require meaningful energy in the final band.
    band_width = max(250.0, cutoff * 0.08)
    tail = excess[freqs >= max(0.0, cutoff - band_width)]
    if tail.size and float(tail.sum()) < total * 0.005:
        cutoff = float(freqs[max(0, rolloff_index - max(1, int(band_width * n_fft / sr)))])
    # Content cutoff: the first frequency past the rolloff where the spectrum drops more
    # than 20 dB below the rolloff level.  An mp3 can roll off near 9.6 kHz while carrying
    # usable content to ~15 kHz through a noisy extension; tiering on the rolloff alone
    # picks the 24k tier and has the model generate 8-12 kHz that was already there.
    rolloff_db = 10.0 * np.log10(power_smooth[min(rolloff_index, len(power_smooth) - 1)] + 1e-20)
    content_cutoff = cutoff
    for index in range(rolloff_index, len(freqs)):
        if 10.0 * np.log10(power_smooth[index] + 1e-20) < rolloff_db - 20.0:
            content_cutoff = float(freqs[index])
            break
    else:
        content_cutoff = float(freqs[-1])
    return (cutoff, f"99% spectral rolloff {cutoff:.0f} Hz; noise floor {noise_db:.1f} dB; "
                    f"content cutoff {content_cutoff:.0f} Hz",
            freqs, spectrum_frames, power_smooth, noise_db, content_cutoff)


def estimate_effective_bandwidth(waveform: np.ndarray, sample_rate: int) -> tuple[float, str]:
    """Estimate decoded waveform bandwidth using robust Welch-style spectra."""
    cutoff, reason, _, _, _, _, _ = _bandwidth_analysis(waveform, sample_rate)
    return cutoff, reason


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run in a one-dimensional mask."""
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _persistent_high_band_evidence(
    freqs: np.ndarray,
    spectrum_frames: np.ndarray,
    power_smooth: np.ndarray,
    noise_db: float,
    total_power: float,
    lower_hz: float,
    upper_hz: float,
) -> tuple[bool, float, float, int]:
    """Check contiguous spectral width plus repeated frame-level energy."""
    band = (freqs >= float(lower_hz)) & (freqs <= float(upper_hz))
    if int(band.sum()) < 2:
        return False, 0.0, 0.0, 0
    band_energy_ratio = float(np.sum(power_smooth[band]) / max(float(total_power), 1e-20))
    if band_energy_ratio < 1e-5:
        return False, 0.0, 0.0, 0
    band_freqs = freqs[band]
    profile_db = 10.0 * np.log10(np.maximum(power_smooth[band], 1e-20))
    # A narrow tone's sidelobes can exceed an almost-zero global floor.
    # Require the contiguous band to also stay within 20 dB of its own peak.
    threshold_db = max(float(noise_db) + 6.0, float(np.max(profile_db)) - 20.0)
    significant = profile_db >= threshold_db
    padded = np.concatenate(([False], significant, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    widest_hz = float(np.max(band_freqs[ends - 1] - band_freqs[starts])) if starts.size else 0.0
    noise_power = 10 ** (float(noise_db) / 10.0)
    frame_band_power = np.mean(spectrum_frames[:, band], axis=1)
    active = 10.0 * np.log10(np.maximum(frame_band_power, 1e-20) / noise_power) >= 6.0
    persistence = float(np.mean(active)) if active.size else 0.0
    run = _longest_true_run(active)
    min_run = 2 if active.size < 12 else 3
    accepted = widest_hz >= 300.0 and persistence >= 0.25 and run >= min_run
    return accepted, widest_hz, persistence, run


def select_input_sample_rate(waveform: np.ndarray, sample_rate: int, model_type: str = "general") -> tuple[int, str]:
    """Map effective bandwidth to a supported rate, correcting general auto upward.

    Tiering uses ``content_cutoff`` (the highest frequency still within 20 dB of the
    rolloff level) rather than the 99% rolloff: an mp3 rolls off near 9.6 kHz while its
    content runs to ~15 kHz, and a rolloff-only tier would pick 24k and have the model
    generate 8-12 kHz that was already present.
    """
    cutoff, reason, freqs, spectrum_frames, power_smooth, noise_db, content_cutoff = _bandwidth_analysis(waveform, sample_rate)
    effective = max(cutoff, content_cutoff)
    if effective <= 5400.0:
        # 8k→48k is 70% of the model's training batches and its strongest condition, so
        # anything at or below the 8k tier's own visible band (4 kHz) stays there; the
        # 5000-5400 Hz edge band trials 8k and relies on the upward correction below if
        # the band really carries sustained high-frequency content.
        base = 8000
    elif effective <= 7200.0:
        base = 16000
    elif effective <= 12000.0:
        base = 16000   # content to 12k: the 16k condition cleans up without inventing sibilance
    else:
        base = 24000

    if model_type != "general" or not spectrum_frames.size:
        suffix = " (high-frequency correction disabled for non-general model)" if model_type != "general" else ""
        return base, (f"auto input_sr={base} from effective cutoff {effective:.0f} Hz "
                      f"(rolloff {cutoff:.0f}, content {content_cutoff:.0f}; {reason}){suffix}")
    source_nyquist = float(int(sample_rate)) / 2.0
    # 8k hard floor: at or below 5000 Hz return the base tier without the upward correction.
    # The correction window (4150-5850 Hz) counts filter transition energy as "sustained
    # high-frequency evidence" and promoted every measured 3400-5000 Hz source to 12k/16k.
    if effective <= 5000.0:
        return base, (f"auto input_sr={base} (base {base}; upward correction skipped for effective "
                      f"{effective:.0f} Hz <= 5000 Hz; {reason})")
    corrected = base
    evidence = ""
    total_power = float(np.sum(power_smooth))
    for previous, candidate in zip(SUPPORTED_INPUT_SR, SUPPORTED_INPUT_SR[1:]):
        if candidate <= base:
            continue
        candidate_nyquist = candidate / 2.0
        # Each candidate protects the band above the previous supported tier:
        # 12k: >4k, 16k: >6k, 24k: >8k (with transition margins).
        previous_nyquist = previous / 2.0
        lower = previous_nyquist + 150.0
        upper = min(source_nyquist - 150.0, candidate_nyquist - 150.0)
        accepted, width, persistence, run = _persistent_high_band_evidence(freqs, spectrum_frames, power_smooth, noise_db, total_power, lower, upper)
        if accepted:
            corrected = candidate
            evidence = f"{width:.0f} Hz contiguous band, {persistence * 100:.0f}% frames, run {run}"
    if corrected > base:
        return corrected, f"auto input_sr={corrected} (base {base}; corrected {corrected}; persistent high-frequency evidence: {evidence}; {reason})"
    return base, f"auto input_sr={base} (base {base}; no upward correction: no persistent contiguous high-frequency band; {reason})"


# ── Material-driven parameter matching (2026-09-18 cross-material measurements) ──────────
# Four materials x four configurations at a fixed seed, each scored against its own
# full-band reference: guidance is the "restore highs vs invent them" knob (16-20k
# overgeneration climbs monotonically, +8.6 → +17.5 dB) and its best value moves with how
# much high band the source is missing.  These rules hang the same measurement
# (content_cutoff + band falls) onto mode/guidance/steps/de-esser:
#     restoration need = clip((-23 - d12) / 20, 0, 1),  d12 = B(12-16k) - B(4-6k)
# Measured: full mixes -13.6 / -13.5 / -8.4 dB → need 0 (SR would only add hiss);
# vocals -29.6 → 0.33; low-bandwidth -39.0 → 0.80 (genuine restoration).
FULL_BAND_D12_DB = -23.0        # d12 above this = the source already carries the top band
RESTORATION_SPAN_DB = 20.0      # -23 … -43 dB spans "nothing to fix" … "heavy restoration"
REFERENCE_BAND_HZ = (4000.0, 6000.0)   # band every level is referenced against
SPEECH_GUIDANCE_LO, SPEECH_GUIDANCE_HI = 1.0, 1.5
GENERAL_GUIDANCE = 2.0          # music range was fixed by listening, not re-measured here
SIBILANCE_D6_DB = 2.0           # 6-8k must exceed 4-6k; real sibilance measures -1.7…-2.5
STEPS_SPEECH_FAST, STEPS_SPEECH_RESTORE = 8, 16
STEPS_BY_MODEL = {"speech": STEPS_SPEECH_FAST, "general": 16}
NEED_FOR_QUALITY_STEPS = 0.5


def _band_level_db(freqs: np.ndarray, power: np.ndarray, lower_hz: float, upper_hz: float) -> float | None:
    band = (freqs >= float(lower_hz)) & (freqs < float(upper_hz))
    if not band.any():
        return None
    return float(10.0 * np.log10(float(np.mean(power[band])) + 1e-20))


def recommend_processing(waveform: np.ndarray, sample_rate: int, model_type: str = "general") -> dict:
    """Match mode / guidance / steps / de-esser to what the material itself measures.

    Shares ``_bandwidth_analysis`` with ``select_input_sample_rate``.  Returns
    ``{"mode", "guidance", "steps", "deess", "metrics", "reason"}``; ``mode`` uses the same
    keys as the node's ``mode`` input ("sr"/"master"/"sr_master").  The speech model only
    offers super-resolution, so a full-bandwidth speech input still reports "sr" and the
    reason says the material needs no enhancement.  Unreadable or near-silent material
    falls back to the per-model defaults.
    """
    model = "speech" if str(model_type).lower() == "speech" else "general"
    fallback = {
        "mode": "sr" if model == "speech" else "sr_master",
        "guidance": SPEECH_GUIDANCE_HI if model == "speech" else GENERAL_GUIDANCE,
        "steps": STEPS_BY_MODEL[model],
        "deess": model == "general",
        "metrics": {},
        "reason": "material unreadable (silent/too short) — keeping per-model defaults",
    }
    try:
        _, _, freqs, _, power, _, content_cutoff = _bandwidth_analysis(waveform, sample_rate)
    except (ValueError, TypeError, FloatingPointError):
        return fallback
    if freqs.size < 4 or power.size < 4 or not np.isfinite(power).all() or float(np.max(power)) <= 1e-20:
        return fallback
    reference = _band_level_db(freqs, power, *REFERENCE_BAND_HZ)
    if reference is None:
        return fallback
    levels = {name: _band_level_db(freqs, power, low, high) for name, low, high in (
        ("d6", 6000.0, 8000.0), ("d8", 8000.0, 12000.0),
        ("d12", 12000.0, 16000.0), ("d16", 16000.0, 20000.0))}
    visible = {name: (None if value is None else round(value - reference, 1)) for name, value in levels.items()}
    if visible["d12"] is None or float(freqs[-1]) < 16000.0:
        need = 1.0
        basis = f"container band stops below 16k (visible to {freqs[-1] / 1000:.1f}k)"
    else:
        need = float(np.clip((FULL_BAND_D12_DB - visible["d12"]) / RESTORATION_SPAN_DB, 0.0, 1.0))
        basis = (f"12-16k sits {visible['d12']:+.1f} dB below 4-6k (full-band threshold "
                 f"{FULL_BAND_D12_DB:.0f}) → restoration need {need * 100:.0f}%")
    if need <= 0.0:
        mode = "sr" if model == "speech" else "master"
    else:
        mode = "sr" if model == "speech" else "sr_master"
    if model == "speech":
        guidance = SPEECH_GUIDANCE_LO if need < 0.5 else SPEECH_GUIDANCE_HI
        steps = STEPS_SPEECH_RESTORE if need >= NEED_FOR_QUALITY_STEPS else STEPS_SPEECH_FAST
        deess = visible["d6"] is not None and visible["d6"] >= SIBILANCE_D6_DB
    else:
        guidance = GENERAL_GUIDANCE
        steps = STEPS_BY_MODEL["general"]
        deess = True
    mode_name = {"sr": "sr", "master": "master", "sr_master": "sr_master"}[mode]
    bands = " / ".join(f"{name} {value:+.1f}" for name, value in visible.items() if value is not None)
    reason = (f"{basis} · content cutoff {content_cutoff:.0f} Hz · {bands} dB rel 4-6k · "
              f"mode={mode_name} · guidance={guidance:g} · steps={steps} · deess={deess}")
    return {"mode": mode, "guidance": guidance, "steps": steps, "deess": deess,
            "metrics": {"content_cutoff": content_cutoff, "need": round(need, 3), **visible},
            "reason": reason}


def _bandlimit(signal_1d: np.ndarray, effective_sr: int) -> np.ndarray:
    return _resample(_resample(signal_1d, TARGET_SR, effective_sr), effective_sr, TARGET_SR)

def _save_temp_wav(directory: Path, audio: np.ndarray) -> Path:
    path = directory / f"segment_{time.time_ns()}.wav"; sf.write(str(path), np.asarray(audio, dtype=np.float32), TARGET_SR, subtype="FLOAT"); return path

def _sr_mid_chunks(model, mid48: np.ndarray, input_sr: int, request: dict, temp_dir: Path, progress: Progress, p0: int, p1: int, cancel: Callable[[], bool]) -> np.ndarray:
    import torch
    chunk = max(1, int(float(request.get("chunk_sec", 15)) * TARGET_SR)); xf = min(int(.05 * TARGET_SR), chunk // 4)
    length = len(mid48); step = max(chunk - xf, 1); starts = list(range(0, max(length - xf, 1), step)) or [0]
    if starts[-1] + chunk < length: starts.append(max(length - chunk, 0))
    result = np.zeros(length); weights = np.zeros(length); ramp = np.linspace(0., 1., max(xf, 1)); total = len(starts)
    kwargs = dict(input_sr=input_sr, ode_method=str(request.get("ode_method", "midpoint")), ode_steps=int(request.get("ode_steps", 4)), guidance_scale=float(request.get("guidance", 1.5)))
    for index, start in enumerate(starts):
        if cancel(): raise KeyboardInterrupt("cancelled")
        progress(p0 + int((p1 - p0) * index / max(total, 1)), f"super-resolution chunk {index + 1}/{total}")
        segment = mid48[start:start + chunk]; path = _save_temp_wav(temp_dir, segment)
        try:
            with torch.no_grad(): out = model.enhance(path.as_posix(), **kwargs)
            out_np = out.detach().cpu().numpy(); out_np = out_np[0] if out_np.ndim > 1 else out_np
        finally:
            try: path.unlink()
            except FileNotFoundError: pass
        expected = len(segment)
        if len(out_np) < expected: out_np = np.pad(out_np, (0, expected - len(out_np)))
        used = min(len(out_np), expected, length - start)
        if used <= 0: continue
        local = np.ones(used)
        if used > 2 * xf: local[:xf] = ramp; local[-xf:] = 1 - ramp[::-1]
        elif used > xf: local[:xf] = ramp
        result[start:start + used] += out_np[:used] * local; weights[start:start + used] += local
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    progress(p1, f"super-resolution complete ({total} chunks)")
    result /= np.maximum(weights, 1e-8)
    gain = np.sqrt(np.mean(mid48 * mid48) + 1e-12) / (np.sqrt(np.mean(result * result)) + 1e-12)
    return result * np.clip(gain, .5, 2.)

def _sr_render(model, source48: np.ndarray, input_sr: int, request: dict, temp_dir: Path, progress: Progress, p0: int, p1: int, cancel: Callable[[], bool], stereo: bool, master_mid: bool = False) -> np.ndarray:
    if source48.ndim == 1: source48 = source48[:, None]
    mid = _sr_mid_chunks(model, _bandlimit(source48.mean(axis=1), input_sr), input_sr, request, temp_dir, progress, p0, p1, cancel)
    if master_mid:
        # The V8 chain runs on the mid alone.  4 dB of true-peak headroom is reserved: the
        # +6 dB side excitation plus the M/S sum can push L/R back over a ceiling the mid
        # already met on its own.  The de-esser rides inside the chain (as in the app).
        mid = dsp.master_chain(np.stack([mid, mid], axis=1), peak_headroom_db=4.0, vocal_gentle=True).mean(axis=1)
    if not stereo: return mid[:, None]
    side = source48[:, 0] - source48[:, 1] if source48.shape[1] >= 2 else np.zeros(len(source48)); side = _bandlimit(side, input_sr)
    side = dsp.decorrelate_side_hf(side, mid)
    from scipy.signal import butter, sosfiltfilt
    side += sosfiltfilt(butter(2, 6000, btype="high", fs=TARGET_SR, output="sos"), side) * (10 ** (6. / 20) - 1)
    n = min(len(mid), len(side)); return np.stack([mid[:n] + side[:n] / 2, mid[:n] - side[:n] / 2], axis=1)

def _sr_stem(model, source48: np.ndarray, input_sr: int, request: dict, temp_dir: Path, progress: Progress, p0: int, p1: int, cancel: Callable[[], bool]) -> np.ndarray:
    mid = source48.mean(axis=1); side = source48[:, 0] - source48[:, 1] if source48.shape[1] > 1 else np.zeros(len(source48))
    mid = _sr_mid_chunks(model, _bandlimit(mid, input_sr), input_sr, request, temp_dir, progress, p0, p1, cancel); n = min(len(mid), len(side))
    return dsp.denoise_gate(np.stack([mid[:n] + side[:n] / 2, mid[:n] - side[:n] / 2], axis=1), source48[:n])


def _terminate_process(proc) -> None:
    if proc.poll() is not None: return
    try:
        if os.name == "nt":
            import subprocess; subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else: os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try: proc.kill()
        except Exception: pass
    try: proc.wait(timeout=5)
    except Exception:
        try: proc.kill(); proc.wait(timeout=2)
        except Exception: pass

def _demucs_two_stems(input_path: Path, out_dir: Path, request: dict, progress: Progress, cancel: Callable[[], bool]) -> tuple[Path, Path]:
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    supplied = request.get("demucs_executable")
    command = [str(Path(str(supplied)).expanduser())] if supplied else [sys.executable, "-m", "demucs"]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    so = (out_dir / "demucs.stdout").open("w", encoding="utf-8")
    se = (out_dir / "demucs.stderr").open("w", encoding="utf-8")
    kwargs = dict(stdout=so, stderr=se, text=True, creationflags=flags)
    child = subprocess.Popen(command + ["-n", "htdemucs", "--two-stems", "vocals", "-o", str(out_dir), str(input_path)], **kwargs)
    try:
        while child.poll() is None:
            if cancel(): _terminate_process(child); raise KeyboardInterrupt("cancelled during Demucs")
            progress(12, "Demucs separating vocals and instrumental"); time.sleep(.5)
        if child.returncode != 0: raise RuntimeError(f"demucs failed: {(out_dir / 'demucs.stderr').read_text(encoding='utf-8', errors='replace')[-1000:]}")
    finally:
        so.close(); se.close()
    root = out_dir / "htdemucs" / input_path.stem; vocals, instrumental = root / "vocals.wav", root / "no_vocals.wav"
    if not vocals.is_file() or not instrumental.is_file(): raise RuntimeError(f"demucs output missing under {root}")
    return vocals, instrumental

def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float64", always_2d=True)
    if data.size == 0: raise ValueError("input audio is empty")
    if not np.isfinite(data).all(): raise ValueError("input audio contains non-finite samples")
    return data, int(rate)

def _fit_headroom(data: np.ndarray) -> np.ndarray:
    """Scale an over-range render down instead of hard-clipping its samples."""
    arr = np.asarray(data, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return arr * (1.0 / peak) if np.isfinite(peak) and peak > 1.0 else arr

def _write_audio(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); arr = np.asarray(data, dtype=np.float64)
    if not np.isfinite(arr).all(): raise ValueError("rendered audio contains non-finite samples")
    if arr.ndim == 2 and arr.shape[1] == 1: arr = arr[:, 0]
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0: arr = arr / peak
    sf.write(str(path), arr.astype(np.float32), TARGET_SR, subtype="FLOAT")
    check, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != TARGET_SR or check.shape[0] == 0: raise IOError(f"output verification failed: {path}")

def process_request(request: dict, progress: Progress = _noop_progress, cancel: Callable[[], bool] = lambda: False) -> str:
    missing = [key for key in ("input_path", "output_path", "mode") if not request.get(key)]
    if missing: raise ValueError("missing request fields: " + ", ".join(missing))
    mode = str(request["mode"])
    if mode not in {"sr", "master", "sr_master", "stem_mix"}: raise ValueError(f"unsupported mode: {mode}")
    input_sr_value = request.get("input_sr", "auto")
    auto_input_sr = isinstance(input_sr_value, str) and input_sr_value.strip().lower() == "auto"
    if auto_input_sr:
        input_sr = None
    else:
        try:
            input_sr = int(input_sr_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid input_sr: {input_sr_value!r}") from exc
        if input_sr not in SUPPORTED_INPUT_SR: raise ValueError(f"input_sr must be one of {SUPPORTED_INPUT_SR}")
    model_type = str(request.get("model", "general"))
    if model_type not in MODEL_REPOS: raise ValueError("model must be general or speech")
    deess = bool(request.get("deess", True))
    if mode != "sr" and model_type != "general": raise ValueError("master and stem_mix require general model")
    source, source_sr = _read_audio(Path(str(request["input_path"])).expanduser()); source48 = _resample_stereo(source, source_sr)
    progress(3, f"prepared decoded audio ({source_sr} Hz container, {len(source) / source_sr:.2f}s)")
    if auto_input_sr:
        input_sr, bandwidth_reason = select_input_sample_rate(source, source_sr, model_type=model_type)
        progress(4, bandwidth_reason)
    stereo = str(request.get("channel_mode", "stereo")).lower() in {"stereo", "ms", "stereo (立体声)"}
    accel = resolve_accel(request.get("accel", DEFAULT_ACCEL))
    _seed_everything(int(request.get("seed", 0))); cache = _configure_environment(request)
    if accel != FALLBACK_ACCEL:
        progress(5, f"accel {accel}" if mode != "master"
                 else f"accel {accel} unused: master mode runs no model inference")
    with accel_context(accel), tempfile.TemporaryDirectory(prefix="musefish_universr_") as temp_name:
        temp_dir = Path(temp_name)
        if mode == "master":
            progress(10, "running V8 mastering chain"); rendered = dsp.master_chain(source48)
            if not stereo: rendered = rendered.mean(axis=1, keepdims=True)
            progress(94, "mastering complete")
        elif mode == "stem_mix":
            general = _load_model("general", cache); speech = _load_model("speech", cache); stem_input = temp_dir / "demucs_input.wav"
            sf.write(str(stem_input), source, source_sr, subtype="FLOAT"); progress(8, "loading Demucs")
            vocal_path, instrumental_path = _demucs_two_stems(stem_input, temp_dir / "demucs", request, progress, cancel)
            vocal, vocal_sr = _read_audio(vocal_path); instrumental, instrumental_sr = _read_audio(instrumental_path)
            vocal48, instrumental48 = _resample_stereo(vocal, vocal_sr), _resample_stereo(instrumental, instrumental_sr)
            # Vocals: speech SR → denoise gate → 8-12k and 12-16k restored toward the stem,
            # brightened at 1.0 dB (1.5 rang), then peak-shaved 1.5 dB — demucs vocal crest
            # is 21-22 dB, so shaving makes the voice sit forward instead of spiking.
            vocal_out = _sr_stem(speech, vocal48, input_sr, request, temp_dir, progress, 25, 52, cancel)
            vocal_out = dsp.band_match(vocal_out, vocal48, 8000.0, 12000.0, 12.0)
            vocal_out = dsp.band_match(vocal_out, vocal48, 12000.0, 16000.0, 5.0)
            vocal_out = dsp.brighten(vocal_out, 1.0)
            vocal_tp = dsp.true_peak_db(vocal_out, TARGET_SR)
            vocal_out = dsp.soft_limit(vocal_out, 10 ** ((vocal_tp - 1.5) / 20))
            # Instrumental: general SR → V8 master → 12-16k restored.
            instrumental_out = dsp.master_chain(_sr_stem(general, instrumental48, input_sr, request, temp_dir, progress, 54, 80, cancel))
            instrumental_out = dsp.band_match(instrumental_out, instrumental48, 12000.0, 16000.0, 5.0)
            rv0 = np.sqrt(np.mean(vocal48 * vocal48) + 1e-12); ri0 = np.sqrt(np.mean(instrumental48 * instrumental48) + 1e-12)
            rv = np.sqrt(np.mean(vocal_out * vocal_out) + 1e-12); ri = np.sqrt(np.mean(instrumental_out * instrumental_out) + 1e-12)
            rel_db = 20 * np.log10(rv0 / ri0 + 1e-12) + 2.; gain = 10 ** (rel_db / 20) * ri / max(rv, 1e-12)
            n = min(len(vocal_out), len(instrumental_out)); rendered = vocal_out[:n] * gain + instrumental_out[:n]
            # Anchor to the source's own loudness (the two stems' sum is 1.5-3 dB off the
            # original, so anchoring to the stems was dragging the whole mix down), then
            # close on true peak with limit-then-restore rounds: what the limiter eats is
            # gained back, so the mix keeps its loudness instead of losing it to a trim.
            reference_lufs = dsp.lufs_gated(source48, TARGET_SR)
            goal_lufs = float(np.clip(reference_lufs, -23.0, -18.0)) if np.isfinite(reference_lufs) else -18.0
            rendered *= 10 ** (float(np.clip(goal_lufs - dsp.lufs_gated(rendered, TARGET_SR), -3.0, 3.0)) / 20)
            for _ in range(4 if dsp.true_peak_db(rendered, TARGET_SR) > -1.0 else 0):
                rendered = dsp.soft_limit_tp2(rendered, 10 ** (-1.0 / 20), TARGET_SR, release_ms=50.0)
                back_db = float(np.clip(goal_lufs - dsp.lufs_gated(rendered, TARGET_SR), 0.0, 3.0))
                if back_db > 0.05: rendered *= 10 ** (back_db / 20)
                if dsp.true_peak_db(rendered, TARGET_SR) <= -1.0: break
            if not stereo: rendered = rendered.mean(axis=1, keepdims=True)
            progress(94, f"mixed stems; vocal lift {rel_db:+.1f} dB, loudness {dsp.lufs_gated(rendered, TARGET_SR):.1f} LUFS")
        else:
            model = _load_model(model_type, cache)
            enhanced = _sr_render(model, source48, input_sr, request, temp_dir, progress, 20, 88, cancel, stereo, master_mid=(mode == "sr_master"))
            rendered = dsp.clean_sr_output(enhanced, source_sr, input_sr)
            # Band-split de-esser on the raw SR output: 16 steps at guidance 2.0 measured a
            # +8.4 dB sibilance band while 12-16k/16-20k stayed intact (Δ0.0 dB).  General
            # only — the speech model's sibilance already sits below the source, and cutting
            # it there removes 12.7 dB of real detail.  sr_master runs its own in-chain one.
            if deess and mode == "sr" and model_type == "general":
                rendered = dsp.de_esser(rendered, TARGET_SR, max_cut_db=8.0)
                progress(92, "de-esser: band-split 5.5-8.5k applied")
            rendered = _fit_headroom(rendered)
            true_peak = dsp.true_peak_db(rendered, TARGET_SR)
            if true_peak > -1.05:
                # sr_master arrives here after a loudness-carrying master, so shave the peaks
                # (gain riding, no new harmonics) before falling back to a linear trim; a
                # full trim would throw away the ~1.5 dB of loudness the peaks cost.
                if mode == "sr_master":
                    rendered = dsp.soft_limit(rendered, 10 ** (-1.05 / 20))
                    residual = dsp.true_peak_db(rendered, TARGET_SR)
                    if residual > -1.05: rendered *= 10 ** ((-1.05 - residual) / 20)
                else:
                    rendered *= 10 ** ((-1.05 - true_peak) / 20)
                progress(93, f"true peak {true_peak:.2f} dBTP → {dsp.true_peak_db(rendered, TARGET_SR):.2f} dBTP")
            if not stereo: rendered = rendered.mean(axis=1, keepdims=True)
            progress(94, "super-resolution complete")
        _write_audio(Path(str(request["output_path"])).expanduser(), rendered)
    progress(100, "output written at 48 kHz"); return str(Path(str(request["output_path"])).expanduser())
