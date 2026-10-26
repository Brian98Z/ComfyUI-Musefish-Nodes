"""ASR engines for Musefish Studio, ported from GPT-SoVITS tools/asr.

Code port of funasr_asr.py + fasterwhisper_asr.py into the Studio server
process: same model names and behaviour, but models resolve from the Studio
model library (data/models/asr) and downloads go straight there. No import
of, or subprocess into, the GPT-SoVITS install.

.list line format (unchanged, consumed by the annotate page):
    wav_path|speaker_name|language|text
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------- model paths

# Model library root, resolved lazily at call time. Resolution order:
#   1. explicit set_models_dir() (Studio server / ComfyUI node call it)
#   2. MUSEFISH_ASR_MODELS env var (injected by the Studio supervisor)
#   3. ComfyUI's folder_paths models_dir /audio_encoders/faster-whisper
#   4. <this file>/../data/models/asr (bare import fallback)
ASR_MODELS_DIR: Path | None = None


def set_models_dir(path) -> None:
    global ASR_MODELS_DIR
    ASR_MODELS_DIR = Path(path)


def _default_models_dir() -> Path:
    env = os.environ.get("MUSEFISH_ASR_MODELS")
    if env:
        return Path(env)
    try:
        import folder_paths  # present only inside ComfyUI
        return Path(folder_paths.models_dir) / "audio_encoders" / "faster-whisper"
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent / "data" / "models" / "asr"


def models_dir() -> Path:
    """Effective model root, resolved at call time."""
    return ASR_MODELS_DIR if ASR_MODELS_DIR is not None else _default_models_dir()


FUNASR_ZH = {
    "vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "asr": "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
}

WHISPER_LANGS = ("auto", "en", "ja", "ko", "zh", "yue")
FUNASR_LANGS = ("zh", "yue")

_funasr_models: dict = {}
_WHISPER_CACHE: dict = {}  # (model_path, device, precision) -> WhisperModel
_WHISPER_CACHE_MAX = 1     # single worker: keep only the hottest model loaded


class StopASR(Exception):
    """Cooperative cancel for in-process ASR (never KeyboardInterrupt:
    that would kill the single shared worker thread)."""
    pass


def ensure_funasr_models(language: str = "zh") -> None:
    """Download the FunASR zh bundle once, into the Studio library."""
    from modelscope import snapshot_download

    if language not in ("zh", "yue"):
        language = "zh"
    for key, name in FUNASR_ZH.items():
        target = models_dir() / name
        if not (target / "model.pt").exists() if key == "asr" else not target.exists():
            snapshot_download(f"iic/{name}", local_dir=str(target))


def create_funasr_model(language: str = "zh"):
    """Port of funasr_asr.create_model: cached AutoModel per language."""
    from funasr import AutoModel

    language = "zh" if language not in ("zh", "yue") else language
    if language in _funasr_models:
        return _funasr_models[language]
    ensure_funasr_models(language)
    model = AutoModel(
        model=str(models_dir() / FUNASR_ZH["asr"]),
        vad_model=str(models_dir() / FUNASR_ZH["vad"]),
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=str(models_dir() / FUNASR_ZH["punc"]),
        disable_update=True,
        disable_pbar=True,  # tqdm redraw cost inside a headless service
    )
    _funasr_models[language] = model
    return model


def funasr_only(input_file: str, language: str = "zh") -> str:
    """Port of funasr_asr.only_asr: transcribe one file."""
    try:
        model = create_funasr_model(language)
        return model.generate(input=input_file, batch_size_s=60)[0]["text"]
    except Exception:
        traceback.print_exc()
        return ""


def funasr_only_array(audio_16k: "np.ndarray", language: str = "zh") -> str:
    """funasr_only for an in-memory mono float32 array already at 16 kHz."""
    try:
        model = create_funasr_model(language)
        return model.generate(input=audio_16k, batch_size_s=60)[0]["text"]
    except Exception:
        traceback.print_exc()
        return ""


def _resolve_whisper_model(model_size: str) -> str:
    """Return the local directory for a whisper model, downloading if needed.
    Progress-visible downloads go through /api/asr/model/download; this is the
    blocking fallback used inside jobs (e.g. CLI or a race after 409)."""
    if whisper_model_ready(model_size):
        return str(models_dir() / f"faster-whisper-{model_size}")
    global _DL_GEN
    with _DL_LOCK:
        _DL_GEN += 1
        gen = _DL_GEN
        _DL_STATE.update({"active": True, "model": model_size, "pct": 0.0,
                          "done_gb": 0.0, "total_gb": WHISPER_SIZES_GB.get(model_size, 2.0),
                          "speed": None, "error": None, "finished": None, "gen": gen})
    _download_loop(model_size, gen)
    if not whisper_model_ready(model_size):
        raise RuntimeError(f"模型下载后仍不完整: faster-whisper-{model_size}")
    return str(models_dir() / f"faster-whisper-{model_size}")


# ------------------------------------------------------------ download w/ progress

# approx on-disk sizes for the user-facing hint (GB)
WHISPER_SIZES_GB = {
    "small": 0.5, "medium": 1.5, "large-v2": 2.9, "large-v3": 2.9,
    "large-v3-turbo": 1.6,
}
_FUNASR_TOTAL_GB = 1.14

# single download slot (the UI allows one at a time); guarded by _DL_LOCK.
# gen increments on every start: progress threads only write while their gen
# matches, so a finished/stale downloader can never clobber a newer slot.
_DL_STATE: dict = {"active": None, "model": None, "pct": 0.0, "done_gb": 0.0,
                   "total_gb": 0.0, "speed": None, "error": None, "finished": None,
                   "gen": 0}
_DL_LOCK = threading.Lock()
_DL_GEN = 0

# per-sub-model readiness for the funasr bundle (dir existence ≠ complete)
_FUNASR_READY_MARKERS = {
    "speech_fsmn_vad_zh-cn-16k-common-pytorch": ["model.pt"],
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch": ["model.pt"],
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": ["model.pt"],
}


def whisper_model_ready(model_size: str) -> bool:
    local = models_dir() / f"faster-whisper-{model_size}"
    return (local / "model.bin").exists() and (local / "tokenizer.json").exists() \
        and ((local / "vocabulary.json").exists() or (local / "vocabulary.txt").exists())


def funasr_ready() -> bool:
    for name, markers in _FUNASR_READY_MARKERS.items():
        d = models_dir() / name
        if not all((d / m).exists() for m in markers):
            return False
    return True


def model_ready(model: str) -> bool:
    if model == "funasr":
        return funasr_ready()
    return whisper_model_ready(model)


def download_state() -> dict:
    with _DL_LOCK:
        return dict(_DL_STATE)


def _set_dl(**kw) -> None:
    with _DL_LOCK:
        _DL_STATE.update(kw)


def _dir_size_gb(path: Path) -> float:
    total = 0
    for root, _dirs, fs in os.walk(path):
        for f in fs:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / 1024 ** 3


def _download_loop(model: str, gen: int) -> None:
    """Blocking download with polled progress into _DL_STATE (bg thread).
    Writes only while _DL_STATE.gen == gen (slot ownership)."""
    try:
        if model == "funasr":
            from modelscope import snapshot_download as ms_dl
            _set_dl(total_gb=_FUNASR_TOTAL_GB, done_gb=_dir_size_gb(models_dir()))
            stop = threading.Event()
            def _poll():
                prev = [0.0, time.time()]
                while not stop.is_set():
                    cur = _dir_size_gb(models_dir())
                    dt = time.time() - prev[1]
                    speed = max(0.0, (cur - prev[0]) / dt) if dt > 0.5 else None
                    prev = [cur, time.time()]
                    _set_dl(done_gb=cur,
                            pct=min(99.9, cur / _FUNASR_TOTAL_GB * 100),
                            speed=speed)
                    stop.wait(1.0)
            t = threading.Thread(target=_poll, daemon=True)
            t.start()
            try:
                for name in FUNASR_ZH.values():
                    markers = _FUNASR_READY_MARKERS.get(name, ["model.pt"])
                    target = models_dir() / name
                    if all((target / m).exists() for m in markers):
                        continue  # genuinely complete; a stale dir gets re-downloaded
                    ms_dl(f"iic/{name}", local_dir=str(target))
            finally:
                stop.set()
                t.join(timeout=2)
            if not funasr_ready():
                raise RuntimeError("funasr 模型下载后仍不完整")
            _set_dl(pct=100.0, finished=time.time(), active=False)
        else:
            from huggingface_hub import snapshot_download as hf_dl
            local = models_dir() / f"faster-whisper-{model}"
            total = WHISPER_SIZES_GB.get(model, 2.0)
            _set_dl(total_gb=total, done_gb=0.0, pct=0.0)
            repo_id = ("mobiuslabsgmbh/faster-whisper-large-v3-turbo"
                       if model == "large-v3-turbo" else f"Systran/faster-whisper-{model}")
            stop = threading.Event()
            def _poll():
                prev = [0.0, time.time()]
                while not stop.is_set():
                    cur = _dir_size_gb(local)
                    dt = time.time() - prev[1]
                    speed = max(0.0, (cur - prev[0]) / dt) if dt > 0.5 else None
                    prev = [cur, time.time()]
                    _set_dl(done_gb=cur, pct=min(99.9, cur / total * 100), speed=speed)
                    stop.wait(1.0)
            t = threading.Thread(target=_poll, daemon=True)
            t.start()
            try:
                hf_dl(repo_id,
                      local_dir=str(local), local_dir_use_symlinks=False,
                      allow_patterns=["config.json", "model.bin", "tokenizer.json",
                                      "vocabulary.txt", "vocabulary.json",
                                      "preprocessor_config.json"])
            finally:
                stop.set()
                t.join(timeout=2)
            if not whisper_model_ready(model):
                raise RuntimeError(f"模型下载后仍不完整: faster-whisper-{model}")
            _set_dl(pct=100.0, finished=time.time(), active=False)
    except Exception as e:  # noqa: BLE001
        _set_dl(error=f"{type(e).__name__}: {e}", active=False, finished=time.time())


def start_download(model: str) -> dict:
    """Kick off a background download; returns current state. Refuses to
    double-start and validates the engine name. An already-installed model is
    reported ready WITHOUT touching the slot (another download may own it)."""
    global _DL_GEN
    if model != "funasr" and model not in WHISPER_SIZES_GB:
        raise ValueError(f"unknown ASR model {model}")
    with _DL_LOCK:
        if model_ready(model):
            # do not mutate the slot: a different model may be mid-download
            if not _DL_STATE.get("active"):
                _DL_STATE.update({"active": False, "pct": 100.0, "finished": time.time(),
                                  "error": None, "model": model})
            return dict(_DL_STATE)
        if _DL_STATE.get("active"):
            return dict(_DL_STATE)  # never nest lock acquisition
        _DL_GEN += 1
        gen = _DL_GEN
        _DL_STATE.update({"active": True, "model": model, "pct": 0.0,
                          "done_gb": 0.0, "total_gb": WHISPER_SIZES_GB.get(model, _FUNASR_TOTAL_GB),
                          "speed": None, "error": None, "finished": None, "gen": gen})
    threading.Thread(target=_download_loop, args=(model, gen), daemon=True,
                     name=f"asr-dl-{model}").start()
    with _DL_LOCK:
        return dict(_DL_STATE)


def whisper_execute(input_folder: str, output_folder: str, model_size: str,
                    language: str, precision: str,
                    on_line=None, should_stop=None,
                    audio_map: dict | None = None) -> str:
    """Port of fasterwhisper_asr.execute_asr: transcribe every media file in
    input_folder, write <folder-name>.list, return its absolute path.

    audio_map ({name: mono float32 np.ndarray @16k}) bypasses disk reads —
    the ComfyUI node passes tensors straight through; when omitted, files are
    read from input_folder as usual.

    zh/yue detections are re-decoded through FunASR for punctuation, matching
    upstream behaviour. on_line(done, total) reports progress; should_stop()
    aborts by raising StopASR (a plain Exception the worker handles).
    """
    import torch
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    if language == "auto":
        language = None

    model_path = _resolve_whisper_model(model_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cache loaded models process-wide: reloading a 1.6 GB turbo on every job
    # cost ~18 s of the ~20 s wall time for a 4 s clip. Single worker → keep
    # at most _WHISPER_CACHE_MAX entries; evicting lets a switched-away model
    # free its VRAM/RAM instead of pinning every combination forever.
    cache_key = (model_path, device, precision)
    entry = _WHISPER_CACHE.get(cache_key)
    if entry is None:
        while len(_WHISPER_CACHE) >= _WHISPER_CACHE_MAX:
            _WHISPER_CACHE.pop(next(iter(_WHISPER_CACHE)))
        print(f"[asr] loading faster-whisper {model_size} ({device}, {precision})")
        base = WhisperModel(model_path, device=device, compute_type=precision)
        # BatchedInferencePipeline chunks the audio and runs batches through
        # the encoder — roughly VRAM-bound speedup over sequential decode.
        entry = {"model": base,
                 "batched": BatchedInferencePipeline(model=base)}
        _WHISPER_CACHE[cache_key] = entry
    model = entry["model"]
    batched = entry["batched"]

    in_dir = Path(input_folder)
    if audio_map is not None:
        # in-memory mode: no files are staged on disk; the map's keys are the
        # clip names (each value is a mono float32 array already at 16 kHz)
        names = sorted(audio_map)
    else:
        names = sorted(n for n in os.listdir(in_dir)
                       if n.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")))
    out_dir = Path(output_folder) if output_folder else models_dir().parent / "asr_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (out_dir / f"{in_dir.name}.list").resolve()

    lines: list[str] = []
    total = len(names)
    import numpy as _np
    for i, name in enumerate(names):
        if should_stop and should_stop():
            raise StopASR()
        path = in_dir / name
        audio_arg = audio_map.get(name) if audio_map else None
        try:
            # beam_size=1 (greedy) ≈2-3× faster than beam 5 on large models with
            # negligible accuracy change for transcription use; vad still on.
            # float32 array direct feed skips a disk decode round-trip.
            # Batched pipeline chunks long audio through the encoder in
            # parallel (batch_size=8): ~VRAM-bound speedup, biggest on large
            # models; falls back internally if a chunk fails.
            audio_ref = audio_arg if audio_arg is not None else str(path)
            if device == "cuda" and len(names) > 1:
                segments, info = batched.transcribe(
                    audio=audio_ref,
                    beam_size=1, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    language=language, batch_size=8)
            else:
                # single clip (or CPU): batched wrapper only pays off with
                # several clips and hallucinated punctuation artifacts on
                # short input (verified: same wav clean via model.transcribe)
                segments, info = model.transcribe(
                    audio=audio_ref,
                    beam_size=1, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    language=language)
            text = ""
            if info.language in ("zh", "yue"):
                # FunASR gives zh better punctuation; consume segments only as
                # fallback. Iterator must be consumed inside (generator).
                if audio_arg is not None:
                    text = funasr_only_array(audio_arg, language=info.language.lower())
                else:
                    text = funasr_only(str(path), language=info.language.lower())
                if "<unk>" in text:
                    # punc model mangles short resampled clips into
                    # <unk>-littered pseudo punctuation; whisper's own text is
                    # clean — drop the repunctuated result
                    text = ""
            if text == "":
                for segment in segments:
                    text += segment.text
            lines.append(f"{path}|{in_dir.name}|{info.language.upper()}|{text}")
        except Exception:
            traceback.print_exc()
        if on_line:
            on_line(i + 1, total)

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[asr] done -> {out_file}")
    return str(out_file)
