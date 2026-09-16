"""Musefish UniverSR audio nodes and isolated worker bridge."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import Input, io


try:
    import folder_paths
except ImportError:
    folder_paths = None

if folder_paths is not None:
    _UNIVERSR_MODEL_DIR = Path(folder_paths.models_dir) / "universr"
    folder_paths.add_model_folder_path("universr", str(_UNIVERSR_MODEL_DIR))
else:
    _UNIVERSR_MODEL_DIR = Path(__file__).resolve().parent / "models" / "universr"

_UNIVERSR_MODEL_ROOT = (Path(folder_paths.models_dir) / "UniverSR" / "models" / "huggingface") if folder_paths is not None else (Path(__file__).resolve().parent / "models" / "UniverSR" / "models" / "huggingface")
_UNIVERSR_MODEL_REPOS = {"general": "woongzip1/universr-audio", "speech": "woongzip1/universr-speech"}
_UNIVERSR_MODEL_DIR_NAMES = {"general": "general", "speech": "speech"}

_UNIVERSR_SAMPLE_RATES = (8000, 12000, 16000, 24000)
_UNIVERSR_MODES = ["sr", "master", "sr_master", "stem_mix"]
_UNIVERSR_MODELS = ["general", "speech"]
_UNIVERSR_ODE_METHODS = ["euler", "midpoint", "rk4"]


def _universr_write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> int:
    """Write a Comfy AUDIO [C,T] tensor as floating-point WAV without quantization."""
    import soundfile as sf
    data = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if data.ndim != 2:
        raise ValueError(f"UniverSR expects AUDIO waveform [C,T], got shape {tuple(data.shape)}")
    channels = int(data.shape[0])
    if channels not in (1, 2):
        raise ValueError(f"UniverSR supports mono or stereo audio, got {channels} channels")
    if not bool(torch.isfinite(data).all()):
        raise ValueError("UniverSR input AUDIO contains NaN or infinite samples")
    sf.write(str(path), data.transpose(0, 1).numpy(), int(sample_rate), format="WAV", subtype="FLOAT")
    return channels


def _universr_read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Read worker WAV output into Comfy's [C,T] float32 AUDIO layout."""
    import soundfile as sf
    values, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if values.shape[1] not in (1, 2):
        raise ValueError(f"UniverSR worker returned unsupported channel count: {values.shape[1]}")
    tensor = torch.from_numpy(values.transpose(1, 0).copy())
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"UniverSR worker returned NaN or infinite samples: {path}")
    return tensor.clamp(-1.0, 1.0), int(sample_rate)


def _universr_tail(path: Path, limit: int = 16 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read(limit).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _universr_terminate_tree(process: subprocess.Popen) -> None:
    """Terminate only the process group rooted at our verified Popen child."""
    if process.poll() is not None:
        return
    pid = int(process.pid)
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, creationflags=flags)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _universr_release_comfy_models() -> None:
    """Release Comfy model allocations before the isolated worker starts."""
    try:
        import gc
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gc.collect()
    except Exception:
        pass


def _universr_run_worker(request: dict, scope: Path, batch_index: int, batch_total: int, log_lines: list[str], progress_bar) -> Path:
    worker = Path(__file__).resolve().parent / "audio_backend" / "worker.py"
    request_path = scope / f"request_{batch_index:04d}.json"
    stdout_path = scope / f"stdout_{batch_index:04d}.log"
    stderr_path = scope / f"stderr_{batch_index:04d}.log"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, "-X", "utf8", "-u", str(worker), "--request", str(request_path)]
    environment = os.environ.copy()
    environment.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = None
    offset = 0
    pending = ""
    seen_error = ""

    def append_log(message: str) -> None:
        if message:
            log_lines.append(message[-1000:])
            del log_lines[:-200]

    def consume(text: str, final: bool = False) -> None:
        nonlocal pending, seen_error
        text = pending + text
        pending = ""
        rows = text.splitlines(keepends=True)
        if rows and not rows[-1].endswith(("\n", "\r")) and not final:
            pending = rows.pop()
        for raw in rows:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                append_log(line)
                continue
            kind = event.get("type")
            if kind == "progress":
                percent = max(0, min(100, int(event.get("percent", 0))))
                append_log(f"batch {batch_index + 1}/{batch_total}: {percent}% {str(event.get('message', '')).strip()}".strip())
                try:
                    progress_bar.update_absolute(batch_index * 100 + percent)
                except Exception:
                    pass
            elif kind == "error":
                seen_error = str(event.get("message", "")).strip()
                append_log(f"worker error: {seen_error}")
            elif kind == "result":
                append_log(f"batch {batch_index + 1}/{batch_total}: complete")

    try:
        with stdout_path.open("w", encoding="utf-8", newline="") as stdout, stderr_path.open("w", encoding="utf-8", newline="") as stderr:
            process = subprocess.Popen(command, cwd=None, env=environment, stdout=stdout, stderr=stderr, creationflags=creationflags, start_new_session=(os.name != "nt"))
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    with stdout_path.open("r", encoding="utf-8", errors="replace") as progress:
                        progress.seek(offset)
                        chunk = progress.read(256 * 1024)
                        offset = progress.tell()
                except OSError:
                    chunk = ""
                consume(chunk)
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(0.2)
        try:
            with stdout_path.open("r", encoding="utf-8", errors="replace") as progress:
                progress.seek(offset)
                consume(progress.read(256 * 1024), final=True)
        except OSError:
            consume("", final=True)
        if process.returncode != 0:
            detail = seen_error or _universr_tail(stderr_path) or _universr_tail(stdout_path)
            raise RuntimeError(f"UniverSR worker failed (exit {process.returncode}): {detail or 'no diagnostics'}")
        output = Path(str(request["output_path"]))
        if not output.is_file():
            detail = seen_error or _universr_tail(stderr_path) or _universr_tail(stdout_path)
            raise RuntimeError(f"UniverSR worker produced no output WAV: {detail or output}")
        return output
    except BaseException:
        if process is not None:
            _universr_terminate_tree(process)
        raise




class MusefishUniverSRModel(io.ComfyNode):
    """Resolve or explicitly download a UniverSR checkpoint into Comfy models."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MusefishUniverSRModel",
            display_name="Musefish UniverSR Model",
            search_aliases=["universr model", "download universr", "audio model cache"],
            category="Musefish/Audio",
            description="Resolve a local UniverSR model cache, or download it only when requested.",
            inputs=[
                io.Combo.Input("model", options=_UNIVERSR_MODELS, default="general"),
                io.Boolean.Input("download", default=False),
            ],
            outputs=[io.String.Output("model_cache")],
        )

    @classmethod
    def execute(cls, model: str = "general", download: bool = False) -> io.NodeOutput:
        if model not in _UNIVERSR_MODEL_REPOS:
            raise ValueError(f"Unsupported UniverSR model: {model!r}")
        target = _UNIVERSR_MODEL_ROOT / _UNIVERSR_MODEL_DIR_NAMES[model]
        config_files = list(target.rglob("config.yaml")) if target.is_dir() else []
        weight_files = list(target.rglob("pytorch_model.bin")) if target.is_dir() else []
        if not config_files or not weight_files:
            if not download:
                raise FileNotFoundError(
                    f"UniverSR {model} model is missing config.yaml/pytorch_model.bin under {target}; "
                    "enable download on Musefish UniverSR Model to fetch it."
                )
            try:
                import urllib.request
                target.mkdir(parents=True, exist_ok=True)
                mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
                repo = _UNIVERSR_MODEL_REPOS[model]
                for filename in ("config.yaml", "pytorch_model.bin"):
                    url = f"{mirror}/{repo}/resolve/main/{filename}"
                    destination = target / filename
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Musefish-Nodes/1.0"})
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        raise RuntimeError(f"Downloaded empty UniverSR file: {filename}")
                    temporary.replace(destination)
            except Exception as exc:
                raise RuntimeError(f"Unable to download UniverSR {model} from mirror to {target}: {exc}") from exc
            config_files = list(target.rglob("config.yaml"))
            weight_files = list(target.rglob("pytorch_model.bin"))
            if not config_files or not weight_files:
                raise RuntimeError(f"Downloaded UniverSR {model} model is incomplete under {target}")
        return io.NodeOutput(str(target))


def _universr_execute_audio(
    *,
    audio: Input.Audio,
    mode: str,
    model: str,
    allowed_modes: tuple[str, ...],
    node_name: str,
    input_sr: str = "auto",
    channel_mode: str = "auto",
    ode_method: str = "midpoint",
    ode_steps: int = 4,
    guidance: float = 1.5,
    chunk_sec: int = 15,
    seed: int = 0,
    model_cache: str = "",
) -> io.NodeOutput:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(f"{node_name} requires a Comfy AUDIO dict with waveform and sample_rate")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{node_name} expects waveform [B,C,T], got {getattr(waveform, 'shape', None)}")
    batch_count, channels, _ = (int(waveform.shape[0]), int(waveform.shape[1]), int(waveform.shape[2]))
    if batch_count < 1 or channels not in (1, 2):
        raise ValueError(f"{node_name} supports AUDIO batches with mono/stereo channels, got [B={batch_count}, C={channels}]")
    source_rate = int(audio["sample_rate"])
    auto_input_sr = isinstance(input_sr, str) and input_sr.strip().lower() == "auto"
    if auto_input_sr:
        chosen_rate = None
    else:
        try:
            chosen_rate = int(input_sr)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid UniverSR input_sr: {input_sr!r}") from exc
        if chosen_rate not in _UNIVERSR_SAMPLE_RATES:
            raise ValueError(f"input_sr must be one of {_UNIVERSR_SAMPLE_RATES} or auto, got {input_sr!r}")
    if mode not in allowed_modes:
        raise ValueError(f"Unsupported {node_name} mode: {mode!r}; expected one of {allowed_modes}")
    if model not in _UNIVERSR_MODELS:
        raise ValueError(f"Unsupported UniverSR model: {model!r}")
    if mode != "sr" and model != "general":
        raise ValueError(f"UniverSR mode {mode!r} requires the general model")
    if ode_method not in _UNIVERSR_ODE_METHODS:
        raise ValueError(f"Unsupported UniverSR ODE method: {ode_method!r}")
    if channel_mode not in {"auto", "mono", "stereo"}:
        raise ValueError(f"Unsupported UniverSR channel_mode: {channel_mode!r}")
    effective_channel_mode = ("mono" if channels == 1 else "stereo") if channel_mode == "auto" else channel_mode
    _universr_release_comfy_models()
    cache_value = str(model_cache or os.environ.get("MUSEFISH_UNIVERSR_MODEL_CACHE", str(_UNIVERSR_MODEL_ROOT))).strip()
    result_batches: list[torch.Tensor] = []
    logs: list[str] = [f"UniverSR: mode={mode}, model={model}, input_sr={'auto' if auto_input_sr else chosen_rate}, batches={batch_count}"]
    progress_bar = comfy.utils.ProgressBar(batch_count * 100)
    with tempfile.TemporaryDirectory(prefix="musefish_universr_") as temporary:
        scope = Path(temporary)
        for index in range(batch_count):
            comfy.model_management.throw_exception_if_processing_interrupted()
            current = waveform[index].detach().to(device="cpu", dtype=torch.float32)
            if effective_channel_mode == "mono" and current.shape[0] == 2:
                current = current.mean(dim=0, keepdim=True)
            elif effective_channel_mode == "stereo" and current.shape[0] == 1:
                current = current.repeat(2, 1)
            batch_rate = chosen_rate
            if auto_input_sr:
                from .audio_backend.processing import select_input_sample_rate
                batch_rate, bandwidth_reason = select_input_sample_rate(current.numpy(), source_rate, model_type=model)
                logs.append(f"batch {index + 1}/{batch_count}: {bandwidth_reason}")
            input_path = scope / f"input_{index:04d}.wav"
            output_path = scope / f"output_{index:04d}.wav"
            _universr_write_wav(input_path, current, source_rate)
            request = {"input_path": str(input_path), "output_path": str(output_path), "mode": mode, "model": model, "channel_mode": effective_channel_mode, "input_sr": batch_rate, "ode_method": ode_method, "ode_steps": int(ode_steps), "guidance": float(guidance), "chunk_sec": int(chunk_sec), "seed": int(seed), "model_cache": cache_value, "demucs_executable": ""}
            produced = _universr_run_worker(request, scope, index, batch_count, logs, progress_bar)
            rendered, rendered_rate = _universr_read_wav(produced)
            if rendered_rate != 48000:
                raise RuntimeError(f"UniverSR worker returned {rendered_rate} Hz; expected 48000 Hz")
            result_batches.append(rendered)
    max_samples = max(int(item.shape[1]) for item in result_batches)
    output = torch.zeros((batch_count, max(int(item.shape[0]) for item in result_batches), max_samples), dtype=torch.float32)
    for index, item in enumerate(result_batches):
        output[index, : item.shape[0], : item.shape[1]] = item
    logs.append(f"UniverSR complete: {batch_count} batch(es), output 48000 Hz")
    return io.NodeOutput({"waveform": output, "sample_rate": 48000}, "\n".join(logs))


class _MusefishUniverSRAudioBase(io.ComfyNode):
    """Shared schema and worker dispatch for the fixed-model audio nodes."""

    _MODEL = "general"
    _GUIDANCE_DEFAULT = 1.5
    _MODES: tuple[str, ...] = tuple(_UNIVERSR_MODES)
    _NODE_ID = "MusefishUniverSRGeneralAudio"
    _DISPLAY_NAME = "Musefish UniverSR General Audio"
    _SEARCH_ALIASES = ["universr", "audio super resolution", "audio mastering", "demucs"]
    _EXPOSE_MODE = True

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [io.Audio.Input("audio")]
        if cls._EXPOSE_MODE:
            inputs.append(io.Combo.Input("mode", options=list(cls._MODES), default="sr"))
        inputs.extend(
            [
                io.Combo.Input("input_sr", options=["auto", "8000", "12000", "16000", "24000"], default="auto"),
                io.Combo.Input("channel_mode", options=["auto", "mono", "stereo"], default="auto"),
                io.Combo.Input("ode_method", options=_UNIVERSR_ODE_METHODS, default="midpoint"),
                io.Int.Input("ode_steps", default=4, min=1, max=25, step=1),
                io.Float.Input("guidance", default=cls._GUIDANCE_DEFAULT, min=0.0, max=5.0, step=0.1),
                io.Int.Input("chunk_sec", default=15, min=1, max=120, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1),
                io.String.Input("model_cache", default="", optional=True, force_input=True),
            ]
        )
        return io.Schema(
            node_id=cls._NODE_ID,
            display_name=cls._DISPLAY_NAME,
            search_aliases=cls._SEARCH_ALIASES,
            category="Musefish/Audio",
            description="Enhance standard AUDIO through isolated UniverSR processing.",
            inputs=inputs,
            outputs=[io.Audio.Output("audio"), io.String.Output("log")],
        )

    @classmethod
    def execute(
        cls,
        audio: Input.Audio,
        mode: str = "sr",
        input_sr: str = "auto",
        channel_mode: str = "auto",
        ode_method: str = "midpoint",
        ode_steps: int = 4,
        guidance: float = 1.5,
        chunk_sec: int = 15,
        seed: int = 0,
        model_cache: str = "",
    ) -> io.NodeOutput:
        return _universr_execute_audio(
            audio=audio,
            mode=mode,
            model=cls._MODEL,
            allowed_modes=cls._MODES,
            node_name=cls._NODE_ID,
            input_sr=input_sr,
            channel_mode=channel_mode,
            ode_method=ode_method,
            ode_steps=ode_steps,
            guidance=guidance,
            chunk_sec=chunk_sec,
            seed=seed,
            model_cache=model_cache,
        )


class MusefishUniverSRGeneralAudio(_MusefishUniverSRAudioBase):
    """Run general UniverSR modes (sr, mastering, and stem mixing)."""

    _MODEL = "general"
    _MODES = tuple(_UNIVERSR_MODES)
    _NODE_ID = "MusefishUniverSRGeneralAudio"
    _DISPLAY_NAME = "Musefish UniverSR General Audio"


class MusefishUniverSRSpeechAudio(_MusefishUniverSRAudioBase):
    """Run speech UniverSR super-resolution with the speech model."""

    _MODEL = "speech"
    _MODES = ("sr",)
    _NODE_ID = "MusefishUniverSRSpeechAudio"
    _GUIDANCE_DEFAULT = 1.5
    _DISPLAY_NAME = "Musefish UniverSR Speech Audio"
    _EXPOSE_MODE = False
