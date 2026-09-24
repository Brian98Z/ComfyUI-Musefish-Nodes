"""Single-session DLSS5 video processing without a Comfy IMAGE batch."""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from comfy_api.latest import Input, InputImpl, io

from .dlss5_backend.session import DLSS5Session, backend_available
from .musefish_dlss5 import _STYLES, _VSR_QUALITY, _VSR_QUALITY_IDS, _SR_SCALES, _scale_plan



def _binaries() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must both be available on PATH")
    return ffmpeg, ffprobe


def _video_info(ffprobe: str, source: Path) -> tuple[int, int, Fraction, int]:
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames", "-of", "json", str(source)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream in {source}")
    video = streams[0]
    fps = Fraction(video["r_frame_rate"])
    if fps <= 0:
        raise ValueError("Could not determine source frame rate")
    return int(video["width"]), int(video["height"]), fps, int(video.get("nb_frames") or 0)


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


class MusefishDLSS5VideoStream(io.ComfyNode):
    """Decode one frame at a time, preserve NGX history, encode one MP4."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MusefishDLSS5VideoStream",
            display_name="Musefish DLSS5 Video Stream",
            category="Musefish/Video",
            description=("Connect Load Video -> Musefish DLSS5 Video Stream -> Save Video. "
                         "Processes the file-backed VIDEO with bounded memory and one "
                         "continuous DLSS5 session; returns VIDEO without an IMAGE batch."),
            inputs=[
                io.Combo.Input("super_resolution", display_name="放大参数", options=_SR_SCALES,
                               default="2× (Balance)",
                               tooltip="Labels name output scales, not VSR quality. Native 2x/4x VSR; 1.5x/3x use VSR plus downsampling. 4K from 720p uses a 5120x2880 intermediate; 8K requires at least a 1080p source."),
                io.Video.Input("video", tooltip="Connect Load Video (file-backed VIDEO)"),
                io.Combo.Input("style", options=_STYLES, default="default"),
                io.Combo.Input("vsr_quality", options=_VSR_QUALITY, default="ultra"),
                io.Float.Input("intensity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_tone", default=0.94, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_struct", default=0.84, min=0.0, max=1.0, step=0.01),
                io.Float.Input("skin_struct", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("use_auto_mask", default=True),
                io.Int.Input("crf", default=19, min=0, max=51, step=1,
                             tooltip="Quality for libx264 only; h264_nvenc uses fixed CQ 27."),
                io.Combo.Input("encoder", options=["libx264（CPU 编码）", "h264_nvenc（GPU 编码加速）"],
                               default="libx264（CPU 编码）",
                               tooltip="GPU NVENC can reduce CPU load; CQ 27 differs from x264 CRF quality."),
            ],
            outputs=[io.Video.Output("video")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, style: str, intensity: float,
                local_tone: float, local_struct: float, skin_struct: float,
                use_auto_mask: bool, super_resolution: str, vsr_quality: str,
                crf: int, encoder: str = "libx264") -> io.NodeOutput:
        import folder_paths
        import comfy.utils
        from .musefish_dlss5 import MusefishDLSS5NeuralRender

        stream_source = video.get_stream_source()
        if not isinstance(stream_source, str):
            raise ValueError("video must be file-backed; connect the Load Video node")
        source = Path(stream_source).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("video must reference an existing file")
        ffmpeg, ffprobe = _binaries()
        width, height, fps, expected_frames = _video_info(ffprobe, source)
        start_time, duration = video.get_active_trim_window()
        if start_time < 0 or duration < 0:
            raise ValueError("Invalid video trim window")
        trim_start = ["-ss", str(start_time)] if start_time else []
        trim_duration = ["-t", str(duration)] if duration else []
        frame_limit = ["-frames:v", str(round(duration * fps))] if duration else []
        total_frames = round(duration * fps) if duration else expected_frames
        if not total_frames:
            total_frames = round(video.get_duration() * fps)
        if total_frames <= 0:
            raise ValueError("Could not determine video frame count for progress")
        factor, out_w, out_h = _scale_plan(super_resolution, width, height)
        ok, detail = backend_available(factor)
        if not ok:
            raise RuntimeError(f"DLSS5 backend unavailable: {detail}")
        progress = comfy.utils.ProgressBar(total_frames)
        progress.update_absolute(0)
        # SaveVideo receives a file-backed VIDEO and performs the final save.
        destination = Path(folder_paths.get_temp_directory()) / "Musefish"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"DLSS5_stream_{uuid.uuid4().hex[:12]}.mp4"
        partial = target.with_suffix(".part.mp4")
        decoder = encoder_proc = None
        session = None
        frame_bytes = width * height * 3
        raw = bytearray(frame_bytes)
        output_rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
        try:
            # Rawvideo from stdout enforces backpressure: no decoded batch exists.
            decoder = subprocess.Popen(
                [ffmpeg, "-v", "error", *trim_start, *trim_duration, "-i", str(source), "-map", "0:v:0",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", *frame_limit, "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            # Input 0 is the frame pipe; input 1 supplies original audio. There
            # is one continuous encode, no chunk files or manual concatenation.
            codecs = {"libx264（CPU 编码）": "libx264", "h264_nvenc（GPU 编码加速）": "h264_nvenc",
                      "libx264": "libx264", "h264_nvenc": "h264_nvenc"}
            if encoder not in codecs:
                raise ValueError(f"Unsupported video encoder: {encoder}")
            codec = codecs[encoder]
            if out_w > 4096 or out_h > 4096:
                if codec == "h264_nvenc":
                    codec = "hevc_nvenc"  # NVENC H.264 is limited to 4096px on this GPU
                elif codec == "libx264":
                    raise ValueError("8K requires GPU encoding: select h264_nvenc (uses HEVC at 8K)")
            video_codec = (["-c:v", "libx264", "-crf", str(crf)] if codec == "libx264" else
                           ["-c:v", codec, "-preset", "p5", "-rc", "vbr",
                            "-cq", "27", "-b:v", "0"])
            encoder_proc = subprocess.Popen(
                [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
                 *trim_start, *trim_duration, "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                 *video_codec, "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-tag:v", "hvc1" if codec == "hevc_nvenc" else "avc1", "-movflags", "+faststart",
                 "-f", "mp4", str(partial)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            session = DLSS5Session(
                width, height, style=_STYLES.index(style), intensity=intensity,
                local_tone=local_tone, local_struct=local_struct,
                skin_struct=skin_struct, use_auto_mask=use_auto_mask,
                super_resolution_scale=factor,
                vsr_quality=_VSR_QUALITY_IDS[vsr_quality],
            )
            count = 0
            completed = 0

            def consume() -> None:
                nonlocal completed
                enhanced = session.pull()
                if enhanced.shape[:2] != (out_h, out_w):
                    enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_AREA)
                cv2.cvtColor(enhanced, cv2.COLOR_RGBA2RGB, dst=output_rgb)
                encoder_proc.stdin.write(memoryview(output_rgb).cast("B"))
                completed += 1
                progress.update_absolute(min(completed, total_frames - 1))
                MusefishDLSS5NeuralRender._check_cancel()

            frame_view = memoryview(raw)
            while True:
                MusefishDLSS5NeuralRender._check_cancel()
                got = 0
                while got < frame_bytes:
                    read = decoder.stdout.readinto(frame_view[got:])
                    if not read:
                        break
                    got += read
                if got == 0:
                    break
                if got != frame_bytes:
                    raise RuntimeError(f"Truncated decoded frame: {got}/{frame_bytes} bytes")
                if session.inflight >= 2:
                    consume()
                cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(height, width, 3),
                             cv2.COLOR_RGB2RGBA, dst=session.next_input())
                session.push(reset=count == 0)
                count += 1
            while session.inflight:
                consume()
            if count == 0:
                raise RuntimeError("No frames decoded")
            decoder.stdout.close()
            if decoder.wait() != 0:
                raise RuntimeError("Video decoder failed")
            encoder_proc.stdin.close()
            if encoder_proc.wait() != 0:
                raise RuntimeError("Video encoder failed")
            info_w, info_h, encoded_fps, encoded_frames = _video_info(ffprobe, partial)
            if ((info_w, info_h) != (out_w, out_h) or encoded_fps != fps
                    or encoded_frames != count or (not duration and expected_frames and count != expected_frames)
                    or partial.stat().st_size == 0):
                raise RuntimeError(f"Encoded output verification failed: {encoded_frames}/{count} frames")
            partial.replace(target)
            progress.update_absolute(total_frames)
            return io.NodeOutput(InputImpl.VideoFromFile(str(target)))
        finally:
            if session is not None:
                session.close()
            _stop(decoder)
            _stop(encoder_proc)
            partial.unlink(missing_ok=True)
