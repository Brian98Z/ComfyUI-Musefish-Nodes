"""AI Mastering: shared engine driver (server/mastering.py) + ComfyUI node contract.

Run with the ComfyUI embedded interpreter (or the Studio venv):
python tests/test_mastering.py

The engine run tests are real: they synthesize a 3 s WAV, master it with
PhaseLimiter and measure the result with ffmpeg. They skip when the engine or
ffmpeg is not installed.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
STUDIO_SERVER = Path(r"D:\MusefishStudio\server")
COMFY_ROOT = PLUGIN_ROOT.parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MASTERING = _load(STUDIO_SERVER / "mastering.py", "mastering_under_test")
INSTALL = MASTERING.install()
ENGINE_READY = bool(INSTALL["ok"]) and MASTERING.ffmpeg_bin() is not None


def _probe_wav(path: Path, seconds: float = 3.0) -> Path:
    """Deterministic 44.1 kHz stereo test signal (two tones + tremolo)."""
    import numpy as np
    import soundfile as sf

    rate = 44100
    t = np.arange(int(rate * seconds)) / rate
    left = 0.22 * np.sin(2 * np.pi * 220 * t) * (0.75 + 0.25 * np.sin(2 * np.pi * 2 * t))
    right = 0.18 * np.sin(2 * np.pi * 330 * t) * (0.75 + 0.25 * np.sin(2 * np.pi * 2 * t))
    sf.write(str(path), np.stack([left, right], axis=1), rate, subtype="PCM_24")
    return path


class ParamTests(unittest.TestCase):
    def test_install_resolves_engine_and_resources(self) -> None:
        if not INSTALL["ok"]:
            self.skipTest("PhaseLimiter not installed")
        self.assertTrue(Path(INSTALL["exe"]).is_file())
        self.assertTrue(Path(INSTALL["cache"]).is_file())
        self.assertTrue(Path(INSTALL["reference"]).is_file())

    def test_db_param_reads_labels_units_and_off(self) -> None:
        p = MASTERING.db_param
        self.assertEqual(p("-14 LUFS（流媒体）", -14.0, -40.0, -5.0), -14.0)
        self.assertEqual(p("-23 LUFS（广播）", -14.0, -40.0, -5.0), -23.0)
        self.assertEqual(p("-2 dBTP（保守）", -1.0, -6.0, 0.0), -2.0)
        self.assertIsNone(p("off（不统一）", -14.0, -40.0, -5.0))
        self.assertIsNone(p("不设限", -1.0, -6.0, 0.0))
        self.assertEqual(p(None, -14.0, -40.0, -5.0), -14.0)
        self.assertEqual(p("garbage", -1.0, -6.0, 0.0), -1.0)
        # 越界收敛到 [lo, hi]
        self.assertEqual(p("-80 LUFS", -14.0, -40.0, -5.0), -40.0)

    def test_normalize_params_defaults_and_clamps(self) -> None:
        # 空参数 → 与 Studio 界面默认一致
        p = MASTERING.normalize_params({})
        self.assertEqual((p["mode"], p["format"], p["bit_depth"]), ("mastering5", "wav", 16))
        self.assertEqual((p["level"], p["matching"]), (0.5, 1.0))
        self.assertEqual((p["loudness_target"], p["ceiling"]), (-14.0, -1.0))
        # 非法枚举回落，数值收敛
        q = MASTERING.normalize_params({
            "mastering_mode": "nonsense", "output_format": "flac", "bit_depth": "32",
            "mastering5_mastering_level": 5, "mastering_matching_level": -3,
            "loudness_target": "off（不统一）", "ceiling": "-9 dBTP",
        })
        # 位深越界收敛到上界 24；非法枚举回落默认
        self.assertEqual((q["mode"], q["format"], q["bit_depth"]), ("mastering5", "wav", 24))
        self.assertEqual((q["level"], q["matching"]), (1.0, 0.0))
        self.assertIsNone(q["loudness_target"])
        self.assertEqual(q["ceiling"], -6.0)


@unittest.skipUnless(ENGINE_READY, "PhaseLimiter engine or ffmpeg missing")
class EngineRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="musefish_music_mastering_test_"))
        self.src = _probe_wav(self.root / "probe.wav")

    def test_mp3_delivery_hits_loudness_target_and_ceiling(self) -> None:
        dest = self.root / "out" / "mastered.mp3"
        keep = self.root / "kept.wav"
        events: list[tuple[int, int, str]] = []
        info = MASTERING.run(
            self.src, dest,
            {"mastering_mode": "classic", "loudness_target": "-14 LUFS（流媒体）",
             "ceiling": "-1 dBTP（推荐）", "output_format": "mp3", "bit_depth": 24},
            workdir=self.root / "job", keep_wav=keep,
            on_progress=lambda done, total, label: events.append((done, total, label)))
        self.assertFalse(info["cancelled"])
        self.assertEqual(info["format"], "mp3")
        self.assertTrue(dest.is_file() and dest.stat().st_size > 10_000)
        # 响度统一实测误差 <0.5 LU；真峰不超上限（留 0.5 dB 测量余量）
        self.assertAlmostEqual(info["lufs"], -14.0, delta=0.5)
        self.assertLessEqual(info["true_peak"], -0.5)
        # 中间 WAV 保留给节点回填 AUDIO（mp3 不可作张量来源）
        self.assertTrue(keep.is_file())
        self.assertEqual(keep.read_bytes()[:4], b"RIFF")
        # 临时转码目录必须清掉，不能留下与输入等大的 wav
        self.assertFalse((self.root / "job" / "tmp").exists())

        # 进度回调：首帧 0、末帧满分、单调不回退、分母等于 progress_total
        total = MASTERING.progress_total({"output_format": "mp3"})
        self.assertEqual(events[0][:2], (0, total))
        self.assertEqual(events[-1][:2], (total, total))
        steps = [done for done, _total, _label in events]
        self.assertEqual(steps, sorted(steps), f"进度回退: {steps}")
        self.assertTrue(all(label.strip() for _d, _t, label in events))
        self.assertTrue(any("转码" in label for _d, _t, label in events))

    def test_progress_total_counts_transcode_only_for_lossy(self) -> None:
        wav = MASTERING.progress_total({"output_format": "wav"})
        mp3 = MASTERING.progress_total({"output_format": "mp3"})
        self.assertEqual(mp3, wav + 1)
        # 引擎阶段 + 测量（+ 非 WAV 的转码步骤）
        self.assertEqual(wav, len(MASTERING._STAGES) + 1)

    def test_wav_bit_depth_and_off_target(self) -> None:
        dest = self.root / "mastered16.wav"
        info = MASTERING.run(
            self.src, dest,
            {"output_format": "wav", "bit_depth": 16, "loudness_target": "off（不统一）",
             "mastering_mode": "mastering2"},
            workdir=self.root / "job")
        self.assertTrue(dest.is_file())
        self.assertEqual(info["bit_depth"], 16)
        self.assertIsNone(info["loudness_target"])
        # off 时不套参考电平：输出电平由引擎默认决定，不强制 -14
        self.assertTrue(info["lufs"] is None or info["lufs"] > -20)


@unittest.skipUnless(ENGINE_READY, "PhaseLimiter engine or ffmpeg missing")
class NodeTests(unittest.TestCase):
    """Node-level contract: AUDIO in → AUDIO out + saved file + measured info."""

    @classmethod
    def setUpClass(cls) -> None:
        if str(COMFY_ROOT) not in sys.path:
            sys.path.insert(0, str(COMFY_ROOT))
        try:
            cls.node = _load(PLUGIN_ROOT / "musefish_music_mastering.py", "musefish_music_mastering_test")
        except Exception as exc:  # noqa: BLE001 - comfy_api only exists inside ComfyUI
            raise unittest.SkipTest(f"ComfyUI runtime unavailable: {exc}")
        import comfy.model_management as mm
        # 非队列环境：打断检查没有运行时上下文，视作“未取消”
        cls._orig_interrupt = mm.throw_exception_if_processing_interrupted
        mm.throw_exception_if_processing_interrupted = lambda: None
        cls.out_dir = Path(tempfile.mkdtemp(prefix="musefish_music_mastering_node_"))
        cls.node._output_dir = lambda: cls.out_dir

    @classmethod
    def tearDownClass(cls) -> None:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted = cls._orig_interrupt

    def _batch(self, seconds: list[float], rate: int = 48000):
        import torch
        clips = []
        for sec in seconds:
            t = torch.arange(int(rate * sec)) / rate
            tone = torch.sin(2 * torch.pi * 220 * t) * 0.2
            clips.append(torch.stack([tone, tone * 0.8]))  # [C, T]
        lengths = {int(c.shape[-1]) for c in clips}
        target = max(lengths)
        padded = [torch.nn.functional.pad(c, (0, target - c.shape[-1])) for c in clips]
        return {"waveform": torch.stack(padded, dim=0), "sample_rate": rate}

    def test_node_returns_44100_stereo_and_saves_file(self) -> None:
        out = self.node.MusefishMusicMastering.execute(
            self._batch([2.0]), mode="classic（经典）", level=0.5, matching=1.0,
            loudness_target="-14 LUFS（流媒体）", ceiling="-1 dBTP（推荐）",
            format="mp3", bit_depth="24", filename_prefix="audio/node_test")
        audio, path, info = out.args
        self.assertEqual(audio["sample_rate"], 44100)
        self.assertEqual(tuple(audio["waveform"].shape)[:2], (1, 2))
        saved = Path(path)
        self.assertTrue(saved.is_file() and saved.suffix == ".mp3")
        self.assertIn("LUFS", info)
        self.assertTrue(out.ui["audio"][0]["filename"].endswith(".mp3"))

    def test_node_batch_stacks_clips_and_pads(self) -> None:
        out = self.node.MusefishMusicMastering.execute(
            self._batch([1.0, 2.0]), mode="mastering3（母带3 · 均衡）", level=1.0,
            matching=0.0, loudness_target="-16 LUFS（播客）", ceiling="-2 dBTP（保守）",
            format="wav", bit_depth="16", filename_prefix="audio/node_batch")
        audio, path, info = out.args
        wave = audio["waveform"]
        self.assertEqual(tuple(wave.shape)[0], 2)
        self.assertEqual(int(wave.shape[-1]), int(2.0 * 44100))  # 两段补齐到最长
        self.assertEqual(Path(path).suffix, ".wav")
        self.assertEqual(len(out.ui["audio"]), 2)
        self.assertIn("-16", info)

    def test_node_progress_spans_batch_times_stages(self) -> None:
        """进度条刻度 = 批次 × 单文件阶段数；结束时打满，中途不回退。"""
        import comfy.utils as cu

        class Recorder:
            last = None

            def __init__(self, total):
                self.total = total
                self.values = []
                Recorder.last = self

            def update_absolute(self, value, total=None, preview=None):
                if total is not None:
                    self.total = total
                self.values.append(value)

        original = cu.ProgressBar
        cu.ProgressBar = Recorder
        try:
            self.node.MusefishMusicMastering.execute(
                self._batch([1.0, 1.0]), mode="classic（经典）", level=0.5, matching=1.0,
                loudness_target="-14 LUFS（流媒体）", ceiling="-1 dBTP（推荐）",
                format="mp3", bit_depth="24", filename_prefix="audio/node_progress")
        finally:
            cu.ProgressBar = original

        rec = Recorder.last
        stage_total = MASTERING.progress_total({"output_format": "mp3"})
        self.assertEqual(rec.total, 2 * stage_total)
        self.assertEqual(rec.values, sorted(rec.values), f"进度回退: {rec.values}")
        self.assertEqual(rec.values[-1], 2 * stage_total)
        self.assertGreater(len(rec.values), 3, "阶段推进应产生多帧进度")


if __name__ == "__main__":
    unittest.main(verbosity=2)
