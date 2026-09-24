"""Musefish DLSS5 neural-render node.

Wraps DLSS5Tool's Feature 18 (DLSS 5 Neural Rendering) as a ComfyUI IMAGE
node. NGX runs in isolated subprocesses, never inside Comfy's Python process.
Parallel workers process contiguous frame blocks; each block retains temporal
history within its own session. A warm pool can be reused across executions,
with history reset at each block boundary.

Style mapping mirrors DLSS5Tool: 0 default / 1 natural / 2 cinema.
Intensity / local tone / local structure / skin mask are the same sliders;
values outside [0,1] need DLSS5Tool's "5x experimental" flag, so the node
clamps at 1.0 to stay within the supported range.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np
import comfy.utils
import torch

from comfy_api.latest import Input, io
from typing_extensions import override

from .dlss5_backend.session import DLSS5Error, DLSS5Session, backend_available

_STYLES = ["default", "natural", "cinema"]
_SR_SCALES = ["off", "1× (Native)", "1.5× (Quality)", "2× (Balance)",
              "3× (Performance)", "4× (Ultra)", "1K", "2K", "4K", "8K"]
# NGX VSR PerfQuality levels: 1 Performance (fastest) .. 4 UltraQuality (DLSS5Tool default).
_VSR_QUALITY = ["performance", "balanced", "quality", "ultra"]
_VSR_QUALITY_IDS = {"performance": 1, "balanced": 2, "quality": 3, "ultra": 4}


_FIXED_SCALES = {"off": 1.0, "1× (Native)": 1.0,
                 "1.5× (Quality)": 1.5, "2× (Balance)": 2.0,
                 "3× (Performance)": 3.0, "4× (Ultra)": 4.0}
_BUCKET_SHORT_EDGES = {"1K": 1080, "2K": 1440, "4K": 2160, "8K": 4320}
_MAX_VSR_PIXELS = 7680 * 4320  # 1080p 4x verified on this machine


def _scale_plan(mode: str, width: int, height: int) -> tuple[int, int, int]:
    """Resolve output scale; 1.5/3x downsample a 2x/4x VSR render."""
    if mode in _BUCKET_SHORT_EDGES:
        short_edge = _BUCKET_SHORT_EDGES[mode]
        for candidate in (1.0, 1.5, 2.0, 3.0, 4.0):
            factor = 1 if candidate == 1 else 2 if candidate <= 2 else 4
            out_w = round(width * candidate / 2) * 2
            out_h = round(height * candidate / 2) * 2
            if (min(out_w, out_h) >= short_edge and
                    out_w * out_h <= _MAX_VSR_PIXELS and
                    width * height * factor * factor <= _MAX_VSR_PIXELS):
                return factor, out_w, out_h
        raise ValueError(f"{mode} cannot be reached from {width}x{height} with up to 4x VSR and 7680x4320 output")
    scale = _FIXED_SCALES[mode]
    out_w = round(width * scale / 2) * 2
    out_h = round(height * scale / 2) * 2
    if out_w * out_h > _MAX_VSR_PIXELS or min(out_w, out_h) <= 0:
        raise ValueError(f"{mode} from {width}x{height} exceeds the 7680x4320 output limit")
    factor = 1 if scale == 1 else 2 if scale <= 2 else 4
    if width * height * factor * factor > _MAX_VSR_PIXELS:
        raise ValueError(f"{mode} needs a {width * factor}x{height * factor} VSR intermediate above the 7680x4320 limit")
    return factor, out_w, out_h

def _is_session_error(error: BaseException) -> bool:
    return isinstance(error, DLSS5Error)




class _SessionCache:
    """Keep one warm pool only while a DLSS5 job is running or queued.

    A short idle grace period lets VHS Meta Batch requeue its next segment
    without paying the NGX startup cost. Once the queue has no DLSS5 work,
    close the child processes and release their engine allocations.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: tuple[tuple, list[DLSS5Session], float] | None = None
        self._idle_ttl = 2.0
        self._timer: threading.Timer | None = None

    @staticmethod
    def _key(width, height, style, intensity, local_tone, local_struct,
             skin_struct, use_auto_mask, vsr_factor, vsr_quality, workers) -> tuple:
        return (int(width), int(height), int(style), round(float(intensity), 4),
                round(float(local_tone), 4), round(float(local_struct), 4),
                round(float(skin_struct), 4), bool(use_auto_mask),
                int(vsr_factor), int(vsr_quality), int(workers))

    def acquire(self, width, height, style, intensity, local_tone, local_struct,
                skin_struct, use_auto_mask, vsr_factor, vsr_quality, workers):
        """Return the cached pool for this contract, or None on a miss."""
        key = self._key(width, height, style, intensity, local_tone, local_struct,
                        skin_struct, use_auto_mask, vsr_factor, vsr_quality, workers)
        with self._lock:
            if self._entry is not None:
                cached_key, sessions, _parked_at = self._entry
                alive = all(s._proc is not None and s._proc.poll() is None for s in sessions)
                if cached_key == key and alive:
                    self._entry = None  # caller owns this pool until release()
                    return sessions
                # key mismatch or a dead worker: drop the whole pool
                self._discard_locked()
            return None

    def release(self, sessions, width, height, style, intensity,
                local_tone, local_struct, skin_struct, use_auto_mask,
                vsr_factor, vsr_quality, workers) -> None:
        key = self._key(width, height, style, intensity, local_tone, local_struct,
                        skin_struct, use_auto_mask, vsr_factor, vsr_quality, workers)
        with self._lock:
            self._discard_locked()
            self._entry = (key, list(sessions), time.monotonic())
            self._schedule_locked()

    def invalidate(self) -> None:
        """Drop whatever is cached regardless of key (worker failed)."""
        with self._lock:
            self._discard_locked()

    def _discard_locked(self) -> None:
        if self._entry is None:
            return
        _key, sessions, _parked = self._entry
        self._entry = None
        for session in sessions:
            session.close()

    def _schedule_locked(self) -> None:
        if self._timer is None and self._entry is not None:
            self._timer = threading.Timer(1.0, self._check_idle)
            self._timer.daemon = True
            self._timer.start()

    def _check_idle(self) -> None:
        try:
            import server
            running, pending = server.PromptServer.instance.prompt_queue.get_current_queue_volatile()
            # A current execution may still be encoding; VHS Meta Batch
            # requeues subsequent chunks before the current one completes.
            has_dlss_work = any(
                any(node.get("class_type") == "MusefishDLSS5NeuralRender"
                    for node in item[2].values())
                for item in (*running, *pending)
            )
        except (ImportError, AttributeError, IndexError, TypeError):
            has_dlss_work = False  # queue unavailable: fall back to idle TTL
        with self._lock:
            self._timer = None
            if self._entry is None:
                return
            if not has_dlss_work and time.monotonic() - self._entry[2] >= self._idle_ttl:
                self._discard_locked()
            else:
                self._schedule_locked()

    def reap_idle(self) -> None:
        """Sweep expired workers even if the timer has not fired yet."""
        with self._lock:
            if self._entry is not None and time.monotonic() - self._entry[2] > 300.0:
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
                "DLSS 5 Neural Rendering on IMAGE batches for still images and "
                "short clips, using one isolated worker and continuous temporal "
                "history. For high-resolution or long videos use Musefish DLSS5 "
                "Video Stream (video path to MP4) to avoid an in-memory batch. "
                "RTX 50-series (Blackwell) ONLY."
            ),
            inputs=[
                io.Combo.Input("super_resolution", display_name="放大参数", options=_SR_SCALES,
                               default="off", tooltip="Output scale, not VSR quality. 1.5× and 3× combine VSR with downsampling; IMAGE batches at 8K require substantial RAM."),
                io.Image.Input("images"),
                io.Combo.Input("style", options=_STYLES, default="default",
                               tooltip="0 default / 1 natural / 2 cinema — same mapping as DLSS5Tool"),
                io.Combo.Input("vsr_quality", options=_VSR_QUALITY, default="ultra",
                               tooltip="Actual RTX VSR quality; only used when scale is above native."),
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
                    "keep_session",
                    options=["auto", "off"], default="auto",
                    tooltip=(
                        "auto: reuse warm workers between consecutive segments "
                        "of a running job; release their GPU resources shortly "
                        "after the job completes or is cancelled. "
                        "off: always close workers after this node execution. "
                        "The first cached frame resets temporal history."
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
        if images is None or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an RGB(A) frame batch [N,H,W,C]")
        if not torch.cuda.is_available():
            raise RuntimeError("DLSS5 neural rendering requires an NVIDIA RTX GPU (no CUDA device visible)")


        frame_count, height, width, _ = images.shape
        if frame_count == 0:
            return io.NodeOutput(images)
        vsr_factor, out_w, out_h = _scale_plan(super_resolution, width, height)
        vsr_quality_id = _VSR_QUALITY_IDS.get(vsr_quality, 4)
        ok, detail = backend_available(vsr_factor)
        if not ok:
            raise RuntimeError(f"DLSS5 backend unavailable: {detail}")

        # Comfy images are RGB float [0,1]; NGX consumes RGBA8. OpenCV's
        # saturating conversion and RGB->RGBA packing avoid four NumPy passes
        # over each frame. Keep the alpha path below unchanged.
        has_alpha = images.shape[-1] == 4
        # IMAGE returns the entire float32 batch. Reject oversized batches before
        # allocating rather than exhausting host RAM; long clips use VIDEO Stream.
        output_bytes = frame_count * out_w * out_h * (4 if has_alpha else 3) * 4
        if output_bytes > 1024 * 1024 * 1024:
            raise ValueError(f"IMAGE output needs {output_bytes / 2**30:.1f} GiB; use Musefish DLSS5 Video Stream for this size")

        reset_every = max(0, int(reset_every_n_frames))
        pool_size = 1
        # The output tensor is the node's unavoidable return value; everything
        # else stays per-frame so peak RAM ~= upstream input + output.
        out_tensor = torch.empty((frame_count, out_h, out_w, 3), dtype=torch.float32)
        sessions: list[DLSS5Session] | None = None
        counts = [0] * pool_size
        one = np.float32(1.0)
        progress = comfy.utils.ProgressBar(frame_count)
        progress.update_absolute(0)
        scale = np.float32(255.0)

        if keep_session != "off":
            _SESSION_CACHE.reap_idle()
            sessions = _SESSION_CACHE.acquire(
                width, height, _STYLES.index(style), intensity, local_tone,
                local_struct, skin_struct, use_auto_mask, vsr_factor,
                vsr_quality_id, pool_size,
            )
            if sessions is not None:
                pool_size = len(sessions)

        def _open_sessions(count: int) -> list[DLSS5Session]:
            """Boot `count` isolated workers; a pool that cannot grow degrades."""
            opened: list[DLSS5Session] = []
            try:
                for _ in range(count):
                    try:
                        opened.append(DLSS5Session(
                            width, height,
                            style=_STYLES.index(style),
                            intensity=float(intensity),
                            local_tone=float(local_tone),
                            local_struct=float(local_struct),
                            skin_struct=float(skin_struct),
                            use_auto_mask=bool(use_auto_mask),
                            super_resolution_scale=vsr_factor,
                            vsr_quality=vsr_quality_id,
                        ))
                    except DLSS5Error:
                        if not opened:
                            raise
                        break  # e.g. a VRAM limit: use the workers already opened
                return opened
            except BaseException:
                for session in opened:
                    session.close()
                raise

        def _consume(session: DLSS5Session, index: int, slot: int, output_rgb: np.ndarray) -> None:
            """Claim the engine result for `index` and write the output slice."""
            enhanced = session.pull()
            if enhanced.shape[:2] != (out_h, out_w):
                enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_AREA)
            # Compact RGBA to RGB before scaling; the strided NumPy cast is
            # substantially slower than an OpenCV channel copy plus dense cast.
            cv2.cvtColor(enhanced, cv2.COLOR_RGBA2RGB, dst=output_rgb)
            np.divide(output_rgb, scale, out=out_tensor[index].numpy(), dtype=np.float32)
            counts[slot] += 1
            cls._check_cancel()
            progress.update_absolute(min(sum(counts), frame_count - 1))

        def _run_block(session: DLSS5Session, lo: int, hi: int, slot: int) -> None:
            """Drive one contiguous frame block through one worker."""
            awaiting: deque[int] = deque()
            output_rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
            rgb = np.empty((height, width, 3), dtype=np.float32)
            rgb8 = np.empty((height, width, 3), dtype=np.uint8)
            alpha = np.empty((height, width), dtype=np.float32) if has_alpha else None
            for index in range(lo, hi):
                # Keep the ring 2 deep: claim the oldest result before handing
                # over another frame, which also frees the slot it occupied.
                if session.inflight >= 2:
                    _consume(session, awaiting.popleft(), slot, output_rgb)
                # A block always opens with a history reset, so a cached session
                # never leaks history into this run; reset_every adds in-place
                # resets on top (reset_every_n_frames=1 => independent frames).
                reset = index == lo or (reset_every > 0 and index % reset_every == 0)
                view = session.next_input()
                frame = images[index]
                np.clip(frame[:, :, :3].numpy(), 0.0, one, out=rgb)
                cv2.convertScaleAbs(rgb, alpha=255, dst=rgb8)
                cv2.cvtColor(rgb8, cv2.COLOR_RGB2RGBA, dst=view)
                if has_alpha:
                    np.clip(frame[:, :, 3].numpy(), 0.0, one, out=alpha)
                    np.multiply(alpha, scale, out=alpha)
                    np.rint(alpha, out=alpha)
                    np.copyto(view[:, :, 3], alpha, casting="unsafe")
                else:
                    view[:, :, 3] = 255
                session.push(reset)
                awaiting.append(index)
            while awaiting:
                _consume(session, awaiting.popleft(), slot, output_rgb)

        def _guarded(session: DLSS5Session, lo: int, hi: int, slot: int,
                     errors: list[BaseException]) -> None:
            try:
                _run_block(session, lo, hi, slot)
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                errors.append(exc)

        errors: list[BaseException] = []
        try:
            if sessions is None:
                sessions = _open_sessions(pool_size)
                pool_size = len(sessions)
                counts = [0] * pool_size
            # Contiguous blocks: each worker owns one stretch of frames, so an
            # uninterrupted session still accumulates history inside its block.
            edges = [round(i * frame_count / pool_size) for i in range(pool_size + 1)]
            threads = [
                threading.Thread(
                    target=_guarded,
                    args=(sessions[slot], edges[slot], edges[slot + 1], slot, errors),
                    name=f"dlss5-block{slot}",
                    daemon=True,
                )
                for slot in range(pool_size)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            if sum(counts) != frame_count:
                raise RuntimeError(f"DLSS5 processed {sum(counts)}/{frame_count} frames")
            if keep_session == "auto":
                _SESSION_CACHE.release(sessions, width, height, _STYLES.index(style),
                                       intensity, local_tone, local_struct, skin_struct,
                                       use_auto_mask, vsr_factor, vsr_quality_id, pool_size)
                sessions = None
        except BaseException:  # includes interruption; never cache a failed pool
            _SESSION_CACHE.invalidate()
            raise
        finally:
            if sessions is not None:
                for session in sessions:
                    session.close()

        if has_alpha:
            alpha_f32 = images[:, :, :, 3]
            if alpha_f32.shape[1:3] != (out_h, out_w):
                alpha_f32 = torch.nn.functional.interpolate(
                    alpha_f32.unsqueeze(1), size=(out_h, out_w), mode="area"
                ).squeeze(1)
            out_tensor = torch.cat([out_tensor, alpha_f32.unsqueeze(-1)], dim=-1)
        progress.update_absolute(frame_count)
        return io.NodeOutput(out_tensor)

    @classmethod
    def _check_cancel(cls) -> None:
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()


class MusefishDLSS5Extension(io.ComfyNode):
    """Marker kept for symmetry with the package's extension list."""
