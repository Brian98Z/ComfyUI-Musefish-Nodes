"""Musefish ASR nodes: speech recognition inside ComfyUI.

Reuses the exact engine code ported for Musefish Studio (server/asr.py) so the
ComfyUI node and the Studio workbench share one implementation, one model
library (data/models/asr) and one behaviour. The plugin tree does NOT vendor a
second copy: the module is loaded from the Studio checkout via the
MUSEFISH_STUDIO_SERVER env var, falling back to a junction-relative
../../server (the studio/ bundle inside this repo).

Inputs are Comfy AUDIO batches; every clip is written to a temp WAV,
transcribed, and the texts are returned as a newline-joined STRING plus the
raw .list path (wav|speaker|lang|text) for downstream tools.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import comfy.utils
from comfy_api.latest import io


def _import_studio_asr():
    """Locate the Studio server tree and import its asr module (single source
    of truth for engine code + model dir resolution)."""
    candidates = []
    env = os.environ.get("MUSEFISH_STUDIO_SERVER")
    if env:
        candidates.append(Path(env))
    # studio/ bundle inside this plugin: <plugin>/studio/server
    candidates.append(Path(__file__).resolve().parent / "studio" / "server")
    # dev checkout next to the plugin's sibling tree
    candidates.append(Path(r"D:\MusefishStudio\server"))
    for base in candidates:
        if (base / "asr.py").is_file():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            import asr  # noqa: PLC0415
            return asr
    raise RuntimeError(
        "Musefish ASR: 未找到 Studio server 目录（asr.py）。"
        "设置环境变量 MUSEFISH_STUDIO_SERVER 指向 MusefishStudio/server 后重启 ComfyUI。")


_asr = _import_studio_asr()

# In ComfyUI the model root is the user-mandated
# <ComfyUI>/models/audio_encoders/faster-whisper — take it from folder_paths
# so it follows whatever instance this plugin runs in. set_models_dir has the
# highest priority in asr's resolution chain, overriding set_data_dir.
try:
    import folder_paths as _fp
    _asr.set_models_dir(Path(_fp.models_dir) / "audio_encoders" / "faster-whisper")
except Exception:
    pass

_ENGINE_OPTIONS = [
    "large-v3-turbo（多语种·推荐）",
    "large-v3（多语种·最强）",
    "small（多语种·最快）",
    "funasr（中文·带标点）",
]


def _engine_value(label: str) -> str:
    if "funasr" in label:
        return "funasr"
    if "turbo" in label:
        return "large-v3-turbo"
    if "large-v3" in label:
        return "large-v3"
    return "small"


class MusefishASR(io.ComfyNode):
    """语音识别：AUDIO 批次 → 文本（funasr 中文带标点 / faster-whisper 多语种）。

    模型与 Musefish Studio 共用 data/models/asr；缺失时自动下载（首次耗时
    取决于模型大小：small 0.5GB、turbo 1.6GB、large-v3 2.9GB、funasr 1.1GB）。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="musefish_asr",
            display_name="Musefish ASR",
            search_aliases=["asr", "语音识别", "transcribe", "whisper", "funasr",
                            "字幕", "subtitle", "转写"],
            category="Musefish/Audio",
            is_output_node=True,
            description=(
                "Transcribe AUDIO batches to text. Engines share Musefish "
                "Studio's model library (data/models/asr); missing models "
                "download automatically on first use. Chinese output is "
                "re-decoded through FunASR for punctuation, matching Studio."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="任意长度的 AUDIO 批次；每条 clip 单独转写。",
                ),
                io.Combo.Input(
                    "engine",
                    options=_ENGINE_OPTIONS,
                    default=_ENGINE_OPTIONS[0],
                    tooltip="funasr=中文带标点；其余为 faster-whisper 多语种。",
                ),
                io.Combo.Input(
                    "language",
                    options=["auto（自动）", "zh（中文）", "yue（粤语）",
                             "en（英语）", "ja（日语）", "ko（韩语）"],
                    default="auto（自动）",
                    tooltip="whisper 的语言提示；funasr 只认 zh/yue。",
                ),
                io.Combo.Input(
                    "precision",
                    options=["float16", "int8"],
                    default="float16",
                    tooltip="int8 显存/内存减半，速度相近。",
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.String.Output("text", tooltip="全部识别文本，多 clip 按行拼接。"),
                io.String.Output("list_path", tooltip=".list 明细文件绝对路径（wav|speaker|lang|text）。"),
            ],
        )

    @classmethod
    def execute(cls, audio: dict, engine: str = _ENGINE_OPTIONS[0],
                language: str = "auto（自动）", precision: str = "float16") -> io.NodeOutput:
        model = _engine_value(engine)
        lang = language.split("（", 1)[0] or "auto"
        waveform = audio["waveform"]      # [B, C, T] float32
        sample_rate = int(audio["sample_rate"])
        batch = int(waveform.shape[0])
        if batch < 1:
            raise ValueError("AUDIO 批次为空")

        import numpy as np
        import torchaudio.functional as AF
        workdir = Path(tempfile.mkdtemp(prefix="musefish_asr_"))
        in_dir = workdir / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        audio_map: dict[str, np.ndarray] = {}
        for i in range(batch):
            data = waveform[i].detach().to(device="cpu", dtype=torch.float32)  # [C,T]
            mono = data.mean(dim=0)
            if sample_rate != 16000:
                mono = AF.resample(mono, sample_rate, 16000)
            name = f"clip_{i:04d}.wav"
            # in-memory feed: engines receive the 16 kHz array directly; the
            # wav on disk exists only as the .list row's playable reference
            audio_map[name] = mono.contiguous().numpy()
            names.append(name)

        bar = comfy.utils.ProgressBar(100)
        done_count = {"n": 0}

        def on_line(done: int, total: int) -> None:
            done_count["n"] = done
            try:
                bar.update_absolute(int(done * 100 / max(total, 1)))
            except Exception:  # noqa: BLE001
                pass

        def should_stop() -> bool:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
            except Exception:  # noqa: BLE001
                return True
            return False

        print(f"[MusefishASR] {model} | {lang} | {precision} | {batch} clip(s)", flush=True)
        if model == "funasr":
            fun_model = _asr.create_funasr_model("zh" if lang in ("", "auto", "zh") else lang)
            lines = []
            for k, name in enumerate(names):
                if should_stop():
                    raise KeyboardInterrupt  # comfy 用 InterruptProcessing 表达取消
                text = fun_model.generate(input=audio_map[name], batch_size_s=60)[0]["text"]
                lines.append(f"{in_dir / name}|{model}|{'ZH' if lang != 'yue' else 'YUE'}|{text}")
                on_line(k + 1, batch)
            list_path = workdir / f"{in_dir.name}.list"
            list_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            list_path = Path(_asr.whisper_execute(
                str(in_dir), str(workdir), model, lang, precision,
                on_line=on_line, should_stop=should_stop, audio_map=audio_map))

        rows = []
        for line in Path(list_path).read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) == 4:
                rows.append(parts[3].strip())
        text = "\n".join(rows)
        try:
            bar.update_absolute(100)
        except Exception:  # noqa: BLE001
            pass
        try:
            print(f"[MusefishASR] done: {text[:80]}{'...' if len(text) > 80 else ''}", flush=True)
        except UnicodeEncodeError:  # GBK console + non-ASCII text
            print("[MusefishASR] done (text contains non-ASCII, see node output)", flush=True)
        return io.NodeOutput(text, str(list_path),
                             ui={"text": [text], "list_path": [str(list_path)]})
