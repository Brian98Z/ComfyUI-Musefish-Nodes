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
from aiohttp import web
import folder_paths
from comfy_api.latest import Input, InputImpl, io
from PIL import Image
from server import PromptServer

from .dlss5_backend.session import DLSS5Session, backend_available
from .musefish_dlss5 import _STYLES, _VSR_QUALITY, _VSR_QUALITY_IDS, _SR_SCALES, _scale_plan, _stable_render_factor



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


def _input_source(filename: str) -> Path | None:
    if (not filename or filename.startswith(("/", "\\")) or
            ".." in filename.replace("\\", "/").split("/") or ":" in filename):
        return None
    root = Path(folder_paths.get_input_directory()).resolve()
    source = (root / filename.replace("\\", "/")).resolve()
    return source if source.is_relative_to(root) else None


async def video_dimensions(request: web.Request) -> web.Response:
    """Probe a Comfy input video without relying on browser codec support."""
    source = _input_source(request.rel_url.query.get("file", ""))
    if source is None:
        return web.Response(status=400)
    if not source.is_file():
        return web.Response(status=404)
    try:
        _ffmpeg, ffprobe = _binaries()
        width, height, _fps, _frames = _video_info(ffprobe, source)
    except (OSError, ValueError, subprocess.SubprocessError, KeyError, json.JSONDecodeError):
        return web.Response(status=422)
    return web.json_response({"width": width, "height": height})


async def image_dimensions(request: web.Request) -> web.Response:
    """Probe a LoadImage source without loading the raster into RAM."""
    source = _input_source(request.rel_url.query.get("file", ""))
    if source is None:
        return web.Response(status=400)
    if not source.is_file():
        return web.Response(status=404)
    try:
        with Image.open(source) as image:
            width, height = image.size
    except (OSError, ValueError):
        return web.Response(status=422)
    return web.json_response({"width": width, "height": height})


if getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.get("/musefish/dlss5/video-dimensions")(video_dimensions)
    PromptServer.instance.routes.get("/musefish/dlss5/image-dimensions")(image_dimensions)


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
                io.Combo.Input("super_resolution", display_name="Upscale", options=_SR_SCALES,
                               default="2× (Balance)",
                               tooltip="Output scale, not VSR quality. Native 2x/4x VSR; 1.5x/3x use VSR plus downsampling. Unavailable scales are hidden when the source resolution is known."),
                io.Video.Input("video", tooltip="Connect Load Video (file-backed VIDEO)"),
                io.Combo.Input("style", options=_STYLES, default="default"),
                io.Combo.Input("vsr_quality", options=_VSR_QUALITY, default="ultra"),
                io.Float.Input("intensity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_tone", default=0.94, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_struct", default=0.84, min=0.0, max=1.0, step=0.01),
                io.Float.Input("skin_struct", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("use_auto_mask", default=True),
                io.Int.Input("crf", default=19, min=0, max=51, step=1,
                             tooltip="Quality for CPU encoding (H.264 under 4K, HEVC above 4096px); GPU NVENC uses fixed CQ 27."),
                io.Combo.Input("encoder", options=["libx264（CPU 编码）", "h264_nvenc（GPU 编码加速）"],
                               default="libx264（CPU 编码）",
                               tooltip="At widths/heights above 4096px both CPU and GPU modes use HEVC; GPU NVENC uses fixed CQ 27."),
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
        # Feature 18 at a full 8K render develops strong color/noise artifacts.
        # Keep neural rendering at <=4K for 8K exports, then enlarge the
        # enhanced result; VSR/NGX never sees an unstable 8K surface.
        render_factor = _stable_render_factor(super_resolution, width, height, factor, out_w, out_h)
        ok, detail = backend_available(render_factor)
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
        render_rgb = (np.empty((height * render_factor, width * render_factor, 3), dtype=np.uint8)
                      if (height * render_factor, width * render_factor) != (out_h, out_w) else output_rgb)
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
                    codec = "libx265"  # H.264 cannot encode an 8K frame
            video_codec = (["-c:v", codec, "-crf", str(crf)] if codec.startswith("libx") else
                           ["-c:v", codec, "-preset", "p5", "-rc", "vbr",
                            "-cq", "27", "-b:v", "0"])
            encoder_proc = subprocess.Popen(
                [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
                 *trim_start, *trim_duration, "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                 *video_codec, "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-tag:v", "hvc1" if codec in ("hevc_nvenc", "libx265") else "avc1", "-movflags", "+faststart",
                 "-f", "mp4", str(partial)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            session = DLSS5Session(
                width, height, style=_STYLES.index(style), intensity=intensity,
                local_tone=local_tone, local_struct=local_struct,
                skin_struct=skin_struct, use_auto_mask=use_auto_mask,
                super_resolution_scale=render_factor,
                vsr_quality=_VSR_QUALITY_IDS[vsr_quality],
            )
            count = 0
            completed = 0

            def consume() -> None:
                nonlocal completed
                enhanced = session.pull()
                cv2.cvtColor(enhanced, cv2.COLOR_RGBA2RGB, dst=render_rgb)
                if render_rgb is not output_rgb:
                    interpolation = (cv2.INTER_AREA if render_rgb.shape[0] > out_h or render_rgb.shape[1] > out_w
                                     else cv2.INTER_CUBIC)
                    cv2.resize(render_rgb, (out_w, out_h), dst=output_rgb, interpolation=interpolation)
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
