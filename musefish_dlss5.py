"""Musefish DLSS5 neural-render node.

Wraps DLSS5Tool's Feature 18 (DLSS 5 Neural Rendering) as a ComfyUI IMAGE
node. The NGX host runs in a disposable subprocess per node execution:
Comfy's Python never loads the NVIDIA DLLs, so a driver fault cannot take
down the server. One worker session handles the whole frame batch — the
temporal model keeps its per-execution history, matching DLSS5Tool's
"strict timing (single session)" export mode.

Style mapping mirrors DLSS5Tool: 0 default / 1 natural / 2 cinema.
Intensity / local tone / local structure / skin mask are the same sliders;
values outside [0,1] need DLSS5Tool's "5x experimental" flag, so the node
clamps at 1.0 to stay within the supported range.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from comfy_api.latest import Input, io
from typing_extensions import override

from .dlss5_backend.session import DLSS5Error, DLSS5Session, backend_available

_STYLES = ["default", "natural", "cinema"]
_SR_SCALES = ["off", "1.5x", "2x", "4x"]
# Internal VSR engine factor for each user-facing scale. NGX VSR scale is an
# enum {2, 4}; 1.5x is a composite (VSR 2x + Feature 18 at the 2x target, then
# cv2 INTER_AREA downscale by 2/3 in the parent).
_SR_VSR_FACTOR = {"off": 1, "1.5x": 2, "2x": 2, "4x": 4}
# NGX VSR PerfQuality levels: 1 Performance (fastest) .. 4 UltraQuality (DLSS5Tool default).
_VSR_QUALITY = ["performance", "balanced", "quality", "ultra"]
_VSR_QUALITY_IDS = {"performance": 1, "balanced": 2, "quality": 3, "ultra": 4}
_MAX_FRAME_PIXELS = 3840 * 2160


def _is_session_error(error: BaseException) -> bool:
    return isinstance(error, DLSS5Error)


def _downscale_1_5(frame_rgba, out_w: int, out_h: int):
    """Shrink a 2x-pipeline frame to 1.5x. cv2 INTER_AREA is the exact
    area-average for downscaling and measured 5.2ms at 1080p->1620p
    (24x faster than PIL Lanczos on this machine)."""
    import cv2
    return cv2.resize(frame_rgba, (out_w, out_h), interpolation=cv2.INTER_AREA)


class _SessionCache:
    """Keep at most one warm DLSS5 worker across node executions.

    A worker session costs ~3.2s to boot (per-process D3D12/NGX runtime bring-up,
    measured on this machine) versus ~20-90ms per frame of actual compute. Video
    workflows that call this node once per segment at the same resolution pay the
    boot tax every execution; caching one keyed session removes it.

    The cache key is the full engine contract — anything that would require a
    different NGX feature (resolution, style, sliders, mask, VSR factor/quality)
    misses. A worker whose *session* differs only in PNG-era fields is not
    distinguishable, so the key includes everything passed to the worker request.
    """

    def __init__(self) -> None:
        self._lock = __import__("threading").Lock()
        self._entry: tuple[tuple, DLSS5Session, float] | None = None
        self._idle_ttl = 300.0  # recycle an idle worker after 5 minutes

    @staticmethod
    def _key(width, height, style, intensity, local_tone, local_struct,
             skin_struct, use_auto_mask, vsr_factor, vsr_quality) -> tuple:
        return (int(width), int(height), int(style), round(float(intensity), 4),
                round(float(local_tone), 4), round(float(local_struct), 4),
                round(float(skin_struct), 4), bool(use_auto_mask),
                int(vsr_factor), int(vsr_quality))

    def acquire(self, width, height, style, intensity, local_tone, local_struct,
                skin_struct, use_auto_mask, vsr_factor, vsr_quality):
        key = self._key(width, height, style, intensity, local_tone, local_struct,
                        skin_struct, use_auto_mask, vsr_factor, vsr_quality)
        with self._lock:
            if self._entry is not None:
                cached_key, session, _parked_at = self._entry
                if cached_key == key and session._proc is not None and session._proc.poll() is None:
                    return session
                # key mismatch or dead worker: drop it
                self._discard_locked()
            return None

    def release(self, session: DLSS5Session, width, height, style, intensity,
                local_tone, local_struct, skin_struct, use_auto_mask,
                vsr_factor, vsr_quality) -> None:
        key = self._key(width, height, style, intensity, local_tone, local_struct,
                        skin_struct, use_auto_mask, vsr_factor, vsr_quality)
        with self._lock:
            self._discard_locked()
            self._entry = (key, session, time.monotonic())

    def discard(self, width, height, style, intensity, local_tone, local_struct,
                skin_struct, use_auto_mask, vsr_factor, vsr_quality) -> None:
        """Drop the cached session only if it matches the key (contract changed)."""
        key = self._key(width, height, style, intensity, local_tone, local_struct,
                        skin_struct, use_auto_mask, vsr_factor, vsr_quality)
        with self._lock:
            if self._entry is not None and self._entry[0] == key:
                self._discard_locked()

    def invalidate(self) -> None:
        """Drop whatever is cached regardless of key (worker failed)."""
        with self._lock:
            self._discard_locked()

    def _discard_locked(self) -> None:
        if self._entry is None:
            return
        _key, session, _parked = self._entry
        self._entry = None
        session.close()

    def reap_idle(self) -> None:
        """Close a worker parked longer than the TTL. Cheap; called on acquire."""
        with self._lock:
            if self._entry is not None and time.monotonic() - self._entry[2] > self._idle_ttl:
                self._discard_locked()


_SESSION_CACHE = _SessionCache()


class MusefishDLSS5NeuralRender(io.ComfyNode):
    """DLSS 5 neural rendering on an IMAGE batch (same-resolution enhance)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available, _detail = backend_available()
        return io.Schema(
            node_id="MusefishDLSS5NeuralRender",
            display_name="Musefish DLSS5 Neural Render",
            search_aliases=["dlss5", "dlss", "neural render", "神经渲染", "DLSS5Tool"],
            category="Musefish/Video",
            description=(
                "DLSS 5 Neural Rendering (Feature 18) on an IMAGE batch, via an "
                "isolated worker driving NVIDIA's nvngx runtime — the same engine "
                "as DLSS5Tool's strict single-session export. RTX 50-series "
                "(Blackwell) ONLY — the feature and its runtime ship exclusively "
                "with Blackwell drivers; older cards fail feature creation. One "
                "session per execution keeps temporal coherence across frames."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("style", options=_STYLES, default="default",
                               tooltip="0 default / 1 natural / 2 cinema — same mapping as DLSS5Tool"),
                io.Float.Input("intensity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_tone", default=0.94, min=0.0, max=1.0, step=0.01),
                io.Float.Input("local_struct", default=0.84, min=0.0, max=1.0, step=0.01),
                io.Float.Input("skin_struct", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("use_auto_mask", default=True,
                                 tooltip="Skin-protection mask (DLSS5Tool 'auto skin mask')"),
                io.Int.Input(
                    "reset_every_n_frames",
                    default=0, min=0, max=999, step=1,
                    tooltip=(
                        "Re-initialize the temporal history every N frames (0 = never, "
                        "strict single-session like DLSS5Tool export). Use 1 for "
                        "temporally independent stills."
                    ),
                ),
                io.Combo.Input(
                    "super_resolution",
                    options=_SR_SCALES, default="off",
                    tooltip=(
                        "Front RTX Video Super Resolution stage (DLSS5Tool order: "
                        "upscale first, then neural-render at the target size). "
                        "Output becomes width/height x scale."
                    ),
                ),
                io.Combo.Input(
                    "vsr_quality",
                    options=_VSR_QUALITY, default="ultra",
                    tooltip=(
                        "RTX VSR quality/performance operator (NGX PerfQuality): "
                        "performance is fastest, ultra is DLSS5Tool's default. "
                        "Only used when super_resolution is not off."
                    ),
                ),
                io.Combo.Input(
                    "keep_session",
                    options=["auto", "off"], default="auto",
                    tooltip=(
                        "auto: reuse one warm worker across executions with the "
                        "same engine contract (skips the ~3s NGX boot per run; "
                        "recommended for segment-per-run video workflows). "
                        "off: always boot a fresh engine (strict isolation). "
                        "The first frame of a cached run is history-reset, so "
                        "output matches a fresh session."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        style: str,
        intensity: float,
        local_tone: float,
        local_struct: float,
        skin_struct: float,
        use_auto_mask: bool = True,
        reset_every_n_frames: int = 0,
        super_resolution: str = "off",
        vsr_quality: str = "ultra",
        keep_session: str = "auto",
    ) -> io.NodeOutput:
        ok, detail = backend_available()
        if not ok:
            raise RuntimeError(f"DLSS5 backend unavailable: {detail}")
        if images is None or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an RGB(A) frame batch [N,H,W,C]")
        if not torch.cuda.is_available():
            raise RuntimeError("DLSS5 neural rendering requires an NVIDIA RTX GPU (no CUDA device visible)")

        vsr_factor = _SR_VSR_FACTOR[super_resolution]
        vsr_quality_id = _VSR_QUALITY_IDS.get(vsr_quality, 4)
        user_scale = float(super_resolution.rstrip("x"))

        frame_count, height, width, _ = images.shape
        if frame_count == 0:
            return io.NodeOutput(images)
        out_w, out_h = int(width * user_scale), int(height * user_scale)
        if out_w * out_h > _MAX_FRAME_PIXELS:
            raise ValueError(
                f"target frame {out_w}x{out_h} exceeds the 4K budget "
                f"({_MAX_FRAME_PIXELS // 1000000}MP); lower super_resolution or input size"
            )

        # Comfy images are RGB float [0,1]; the NGX contract is RGBA8. Convert
        # PER FRAME — materializing rgb8 for the whole batch costs width*height
        # *frames bytes on top of Comfy's float32 input (a 670-frame 1080p
        # video: 3.9GB extra for zero benefit).
        has_alpha = images.shape[-1] == 4

        reset_every = max(0, int(reset_every_n_frames))
        # The output tensor is the node's unavoidable return value; everything
        # else stays per-frame so peak RAM ~= upstream input + output.
        out_tensor = torch.empty((frame_count, out_h, out_w, 3), dtype=torch.float32)
        session: DLSS5Session | None = None
        cached = False
        processed = 0

        if keep_session != "off":
            _SESSION_CACHE.reap_idle()
            session = _SESSION_CACHE.acquire(
                width, height, _STYLES.index(style), intensity, local_tone,
                local_struct, skin_struct, use_auto_mask, vsr_factor,
                vsr_quality_id,
            )
            cached = session is not None

        def _open_session() -> DLSS5Session:
            return DLSS5Session(
                width, height,
                style=_STYLES.index(style),
                intensity=float(intensity),
                local_tone=float(local_tone),
                local_struct=float(local_struct),
                skin_struct=float(skin_struct),
                use_auto_mask=bool(use_auto_mask),
                super_resolution_scale=vsr_factor,
                vsr_quality=vsr_quality_id,
            )

        try:
            for index in range(frame_count):
                if session is None:
                    session = _open_session()
                frame = images[index]
                rgba = np.empty((height, width, 4), dtype=np.uint8)
                rgba[:, :, :3] = (frame[:, :, :3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
                rgba[:, :, 3] = 255 if not has_alpha else (
                    frame[:, :, 3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
                # First frame resets the temporal history — mandatory for a
                # cached session, which carries history from the previous run.
                reset = index == 0
                if reset_every > 0 and index > 0 and index % reset_every == 0:
                    # Mid-batch reset contract requires a fresh engine; a
                    # rebuilt session is not cache-worthy for this run.
                    if cached:
                        _SESSION_CACHE.invalidate()
                        cached = False
                    session.close()
                    session = _open_session()
                    reset = True
                enhanced = session.process(rgba)
                del rgba
                if vsr_factor == 2 and super_resolution == "1.5x":
                    enhanced = _downscale_1_5(enhanced, out_w, out_h)
                # Write straight into the float32 output slice; no full-batch
                # numpy intermediate, no astype copy of the whole result.
                out_tensor[index] = torch.from_numpy(enhanced[:, :, :3]).to(torch.float32).div_(255.0)
                del enhanced
                processed += 1
                cls._check_cancel()
            if cached and session is not None and keep_session == "auto":
                _SESSION_CACHE.release(session, width, height, _STYLES.index(style),
                                       intensity, local_tone, local_struct, skin_struct,
                                       use_auto_mask, vsr_factor, vsr_quality_id)
                session = None
        except DLSS5Error:
            # A failed worker must never be reused from cache.
            if session is not None:
                session.close()
                session = None
            _SESSION_CACHE.invalidate()
            raise
        finally:
            if session is not None:
                session.close()

        if processed != frame_count:
            raise RuntimeError(f"DLSS5 processed {processed}/{frame_count} frames")
        if has_alpha:
            alpha_f32 = images[:, :, :, 3]
            if alpha_f32.shape[1:3] != (out_h, out_w):
                alpha_f32 = torch.nn.functional.interpolate(
                    alpha_f32.unsqueeze(1), size=(out_h, out_w), mode="area"
                ).squeeze(1)
            out_tensor = torch.cat([out_tensor, alpha_f32.unsqueeze(-1)], dim=-1)
        return io.NodeOutput(out_tensor)

    @classmethod
    def _check_cancel(cls) -> None:
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()


class MusefishDLSS5Extension(io.ComfyNode):
    """Marker kept for symmetry with the package's extension list."""
