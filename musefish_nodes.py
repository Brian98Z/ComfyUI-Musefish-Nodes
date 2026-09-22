"""PiD batch video upscaling node.

The node performs VAE encode -> PiD conditioning -> custom sampling -> VAE decode
inside one execution. Frames are sampled in bounded batches, while the model is
preloaded once for the execution and reused for every batch.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional


import torch

import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io
from typing_extensions import override
from .musefish_audio import MusefishUniverSRGeneralAudio, MusefishUniverSRModel, MusefishUniverSRSpeechAudio
from .musefish_dlss5 import MusefishDLSS5NeuralRender
from .musefish_video_nodes import MusefishVideoDownload, MusefishWeChatChannels
from .pid_runtime import (
    DEFAULT_ATTENTION_BACKEND,
    ATTENTION_BACKENDS,
    pid_lq_projection_cache,
    pid_pixel_runtime,
    resolve_attention_backend,
)
_MEMORY_HEADROOM = 0.70


def _is_oom_error(error: BaseException) -> bool:
    """Return true only for recognizable device/system out-of-memory errors."""
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return "out of memory" in message or ("cuda error" in message and "memory" in message)
    return False


def _release_after_oom(device: torch.device) -> None:
    """Use Comfy's cache lifecycle, without bypassing its memory manager."""
    if device.type == "cuda":
        comfy.model_management.soft_empty_cache()




_LATENT_FORMATS = ["flux", "sd3", "sdxl", "qwenimage"]
_MODEL_LONG_EDGE = 1024
_MODEL_SCALE = 4


def _latent_format(name: str):
    if name == "flux":
        return comfy.latent_formats.Flux
    if name == "sd3":
        return comfy.latent_formats.SD3
    if name == "sdxl":
        return comfy.latent_formats.SDXL
    if name == "qwenimage":
        return comfy.latent_formats.Wan21
    raise ValueError(f"Unknown latent format: {name}")


def _pid_conditioning(conditioning, latent: torch.Tensor, latent_format: str, degrade_sigma: float):
    samples = latent
    fmt = _latent_format(latent_format)()
    lq_latent = fmt.process_in(samples)
    if lq_latent.ndim == 5:
        lq_latent = lq_latent[:, :, 0]
    sigma = torch.tensor([float(degrade_sigma)], dtype=torch.float32)
    return node_helpers.conditioning_set_values(
        conditioning,
        {"lq_latent": lq_latent, "degrade_sigma": sigma},
    )


_LQ_CACHE_UNSAFE_CONDITIONING_KEYS = frozenset({
    "area",
    "mask",
    "hooks",
    "control",
    "start_percent",
    "end_percent",
    "timestep_start",
    "timestep_end",
    "sigma_start",
    "sigma_end",
    "model_function_wrapper",
    "model_function_wrapper_inner",
})


def _pid_lq_cache_safe_path(conditioning, model) -> bool:
    """Allow reuse only for one plain conditioning entry without wrappers."""
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        return False
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
        return False
    keys = {str(key).lower() for key in entry[1]}
    if keys & _LQ_CACHE_UNSAFE_CONDITIONING_KEYS:
        return False
    return not bool(getattr(model, "wrappers", {}))


def _conditioning_zero_out(conditioning):
    result = []
    for tensor, values in conditioning:
        copied = values.copy()
        for key in ("pooled_output", "conditioning_lyrics", "conditioning_scale"):
            value = copied.get(key)
            if value is not None:
                copied[key] = torch.zeros_like(value)
        result.append([torch.zeros_like(tensor), copied])
    return result


def _round_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, (value // multiple) * multiple)


def _rgb_to_yuv601(rgb: torch.Tensor):
    """RGB [...,3] float 0-1 -> Y/U/V each [...,1], U/V centered at 0.5 (full-range BT.601)."""
    y = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
    u = (rgb[..., 2:3] - y) * 0.492 + 0.5
    v = (rgb[..., 0:1] - y) * 0.877 + 0.5
    return y, u, v


def _yuv601_to_rgb(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = y + (v - 0.5) / 0.877
    g = y - 0.194 * (u - 0.5) / 0.492 - 0.509 * (v - 0.5) / 0.877
    b = y + (u - 0.5) / 0.492
    return torch.cat([r, g, b], dim=-1).clamp(0.0, 1.0)


def _temporal_bilateral(
    seq: torch.Tensor,
    strength: float,
    guide: Optional[torch.Tensor] = None,
    guide_strength: Optional[float] = None,
) -> torch.Tensor:
    """Symmetric two-sided temporal bilateral filter.

    Uses only the immediately adjacent *source* frames, never a recursively
    filtered history. Each neighbor is weighted by its photometric distance to
    the current frame, so motion boundaries receive near-zero weight instead
    of leaving a one-directional trail. For chroma planes, the source luma
    plane is an additional motion guide that blocks color bleeding at edges.

    Args:
        seq: [N, H, W] float tensor in temporal order.
        strength: Similarity width on the hqdn3d 0-255 scale; zero disables
            filtering.
        guide: Optional [N, H, W] luma guide.
        guide_strength: Similarity width for the luma guide, 0-255.
    """
    if strength <= 0.0 or seq.shape[0] < 2:
        return seq.clone()

    sigma = max(float(strength) / 255.0, torch.finfo(seq.dtype).eps)
    guide_sigma = None
    if guide is not None and guide_strength is not None and guide_strength > 0.0:
        guide_sigma = max(float(guide_strength) / 255.0, torch.finfo(seq.dtype).eps)

    # Vectorize the two-sided neighborhood. Invalid boundary neighbors are
    # duplicated only to keep tensor shapes; their weights are explicitly zero.
    previous = torch.cat((seq[:1], seq[:-1]), dim=0)
    following = torch.cat((seq[1:], seq[-1:]), dim=0)
    previous_valid = torch.ones((seq.shape[0], 1, 1), device=seq.device, dtype=seq.dtype)
    following_valid = previous_valid.clone()
    previous_valid[0] = 0.0
    following_valid[-1] = 0.0


    prev_weight = torch.exp(-((previous - seq) / sigma).square()) * previous_valid
    next_weight = torch.exp(-((following - seq) / sigma).square()) * following_valid
    if guide is not None and guide_sigma is not None:
        guide_previous = torch.cat((guide[:1], guide[:-1]), dim=0)
        guide_following = torch.cat((guide[1:], guide[-1:]), dim=0)
        prev_weight *= torch.exp(-((guide_previous - guide) / guide_sigma).square())
        next_weight *= torch.exp(-((guide_following - guide) / guide_sigma).square())

    weight_sum = 1.0 + prev_weight + next_weight
    return (seq + previous * prev_weight + following * next_weight) / weight_sum


# Antiflicker processes frames in bounded chunks sized to the free memory of
# the selected device (VRAM for gpu, system RAM for cpu) so a long video never
# materializes the whole sequence (plus its neighbor/weight copies) at once.
# Interior chunk frames see their true temporal neighbors because every chunk
# carries one extra frame per side.
_ANTIFLICKER_PEAK_COPIES = 8.0  # rgb + y/u/v + bilateral temporaries
_ANTIFLICKER_MAX_BLOCK = 16
_ANTIFLICKER_CPU_MAX_BLOCK = 64


def _antiflicker_gpu_block(source: torch.Tensor) -> int:
    """Choose a block with headroom for one-frame temporal halos on each side."""
    n = source.shape[0]
    per_frame = source.shape[1] * source.shape[2] * 3 * 4
    free = comfy.model_management.get_free_memory()
    if free <= 0:
        return min(_ANTIFLICKER_MAX_BLOCK, n)
    budget = free * _MEMORY_HEADROOM
    block = int(budget / (per_frame * _ANTIFLICKER_PEAK_COPIES)) - 2
    return min(_ANTIFLICKER_MAX_BLOCK, max(0, block), n)


def _antiflicker_cpu_block(source: torch.Tensor) -> int:
    """Choose a float32 CPU block with headroom for temporal halo copies."""
    import psutil

    n = source.shape[0]
    per_frame = source.shape[1] * source.shape[2] * 3 * 4
    free = psutil.virtual_memory().available
    if free <= 0:
        return min(_ANTIFLICKER_CPU_MAX_BLOCK, n)
    budget = free * _MEMORY_HEADROOM
    block = int(budget / (per_frame * _ANTIFLICKER_PEAK_COPIES)) - 2
    return max(1, min(_ANTIFLICKER_CPU_MAX_BLOCK, max(1, block), n))


def _antiflicker_process_chunk(
    source: torch.Tensor,
    start: int,
    block: int,
    work: torch.device,
    luma_tmp: float,
    chroma_tmp: float,
) -> torch.Tensor:
    """Process one chunk including its source-frame halo and return CPU frames."""
    lo = max(0, start - 1)
    count = min(block, source.shape[0] - start)
    hi = min(source.shape[0], start + count + 1)
    rgb = source[lo:hi].to(work)
    y, u, v = _rgb_to_yuv601(rgb)
    y_plane = y.squeeze(-1)
    yf = _temporal_bilateral(y_plane, luma_tmp).unsqueeze(-1)
    uf = _temporal_bilateral(u.squeeze(-1), chroma_tmp, y_plane, luma_tmp).unsqueeze(-1)
    vf = _temporal_bilateral(v.squeeze(-1), chroma_tmp, y_plane, luma_tmp).unsqueeze(-1)
    rgb_f = _yuv601_to_rgb(yf, uf, vf)
    keep_lo = start - lo
    return rgb_f[keep_lo : keep_lo + count].cpu()


class AutoBatchAntiflicker(io.ComfyNode):
    """Ghost-resistant temporal flicker suppression on an IMAGE frame batch.

    The node uses a symmetric bilateral filter over the immediately adjacent
    source frames. Unlike the previous one-sided hqdn3d-style recursion, it
    cannot propagate a previous-frame residual into subsequent frames.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AutoBatchAntiflicker",
            display_name="AutoBatch Antiflicker",
            search_aliases=["antiflicker", "flicker removal", "去频闪", "频闪抑制", "symmetric bilateral", "musefish antiflicker"],
            category="Musefish/Video",
            description=(
                "Symmetric, luma-guided temporal bilateral filtering on an IMAGE "
                "frame batch. Smooths local flicker while rejecting motion edges to "
                "avoid one-directional ghost trails. Processing is auto-batched to "
                "fit the selected device's memory; 'auto' falls back to CPU when "
                "the GPU cannot even fit one frame."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("luma_tmp", default=15.0, min=0.0, max=255.0, step=0.5),
                io.Float.Input("chroma_tmp", default=20.0, min=0.0, max=255.0, step=0.5),
                io.Int.Input("frames_per_batch", default=0, min=0, max=128, step=1),
                io.Combo.Input("device", options=["auto", "gpu", "cpu"], default="auto"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        luma_tmp: float,
        chroma_tmp: float,
        frames_per_batch: int = 0,
        device: str = "auto",
    ) -> io.NodeOutput:
        if images is None or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an RGB frame batch [N,H,W,3]")
        source = images[:, :, :, :3].float().cpu()
        n = source.shape[0]
        if n == 0:
            raise ValueError("IMAGE batch contains no frames")
        out_device = images.device
        if n < 2 or (float(luma_tmp) <= 0.0 and float(chroma_tmp) <= 0.0):
            # Nothing to filter: identity pass, no device round trip.
            return io.NodeOutput(source.to(out_device))

        fixed = int(frames_per_batch)
        if device == "cpu":
            work = torch.device("cpu")
            block = fixed if fixed > 0 else _antiflicker_cpu_block(source)
        else:
            work = comfy.model_management.get_torch_device()
            block = fixed if fixed > 0 else _antiflicker_gpu_block(source)
            if block < 1:
                if device == "auto":
                    work = torch.device("cpu")
                    block = fixed if fixed > 0 else _antiflicker_cpu_block(source)
                else:
                    block = 1

        result = torch.empty_like(source)  # CPU; the full result never occupies VRAM
        preferred_block = block
        start = 0
        while start < n:
            active = min(preferred_block, n - start)
            while True:
                try:
                    processed = _antiflicker_process_chunk(
                        source, start, active, work, float(luma_tmp), float(chroma_tmp)
                    )
                except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    if not _is_oom_error(error):
                        raise
                    processed = None
                    error.__traceback__ = None
                    _release_after_oom(comfy.model_management.get_torch_device())
                    if active > 1:
                        active = max(1, active // 2)
                        preferred_block = active
                        continue
                    if device == "auto" and work.type != "cpu":
                        work = torch.device("cpu")
                        preferred_block = fixed if fixed > 0 else _antiflicker_cpu_block(source)
                        active = min(preferred_block, n - start)
                        continue
                    raise RuntimeError(
                        "AutoBatchAntiflicker could not process a single frame without OOM"
                    ) from error
                result[start : start + active] = processed
                del processed
                start += active
                comfy.model_management.throw_exception_if_processing_interrupted()
                break

        return io.NodeOutput(result.to(out_device))


class MusefishPiDBatchVideoUpscale(io.ComfyNode):
    """Batch PiD upscaler with one model lifecycle per node execution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MusefishPiDBatchVideoUpscale",
            display_name="Musefish PiD Batch Video Upscale",
            search_aliases=["pid video upscale", "batch video upscale", "video super resolution"],
            category="Musefish/Video",
            description=(
                "Upscale an IMAGE frame batch with a PiD model in bounded batches. "
                "Weights are reused during the execution, while each batch still runs DynamicVRAM preparation; "
                "optional audio and FPS are preserved in VIDEO. "
                "Kitchen is the default attention backend and falls back to cuDNN when unavailable; "
                "the backend override is local to this node execution."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio", optional=True),
                io.Float.Input("frame_rate", default=24.0, min=1.0, max=240.0, step=0.01),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("encode_vae"),
                io.String.Input("positive_prompt", default="high quality, ultra detailed, sharp details", multiline=True),
                io.Int.Input("batch_size", default=2, min=1, max=64, step=1),
                io.Int.Input("upscale_factor", default=4, min=2, max=4, step=1),
                io.Combo.Input("latent_format", options=_LATENT_FORMATS, default="flux"),
                io.Float.Input("degrade_sigma", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="lcm"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
                io.Int.Input("steps", default=4, min=1, max=100, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input(
                    "pixel_chunk_size",
                    default=1024,
                    min=0,
                    max=65536,
                    step=1,
                    tooltip=(
                        "Internal PiD MLP chunk size, not spatial tiling. "
                        "1024 is the recommended default; 0 disables chunking, "
                        "and smaller values may increase overhead."
                    ),
                ),
                io.Combo.Input(
                    "attention_backend",
                    options=list(ATTENTION_BACKENDS),
                    default=DEFAULT_ATTENTION_BACKEND,
                    tooltip=(
                        "Attention implementation for this node execution only. "
                        "Kitchen is the default and falls back to cuDNN when unavailable. "
                        "Choose cuDNN explicitly to use that backend; the two paths are not pixel-identical."
                    ),
                ),
            ],
            outputs=[io.Video.Output(), io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        audio: Optional[Input.Audio],
        frame_rate: float,
        model,
        clip,
        encode_vae,
        positive_prompt: str,
        batch_size: int,
        upscale_factor: int,
        latent_format: str,
        degrade_sigma: float,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        steps: int,
        seed: int,
        pixel_chunk_size: int = 1024,
        attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    ) -> io.NodeOutput:
        attention_backend = resolve_attention_backend(str(attention_backend))
        if images is None:
            raise ValueError("images are required")
        if model is None or clip is None or encode_vae is None:
            raise ValueError("model, clip, and encode_vae are required")

        source_images = images
        if source_images.ndim != 4 or source_images.shape[-1] < 3:
            raise ValueError("IMAGE must be an RGB frame batch")
        source_images = source_images[:, :, :, :3].float()
        frame_count, source_h, source_w, _ = source_images.shape
        if frame_count == 0:
            raise ValueError("IMAGE batch contains no frames")

        # PiD is trained for a fixed 1024 -> 4096 path. The user-facing
        # factor only controls the final delivery resize after decoding.
        scale = float(_MODEL_LONG_EDGE) / max(source_h, source_w)
        model_h = _round_multiple(int(round(source_h * scale)))
        model_w = _round_multiple(int(round(source_w * scale)))
        model_target_h = model_h * _MODEL_SCALE
        model_target_w = model_w * _MODEL_SCALE
        output_h = model_h * int(upscale_factor)
        output_w = model_w * int(upscale_factor)
        # Comfy's Lanczos path round-trips through PIL uint8. Bicubic retains
        # floating-point gradients before VAE conditioning; downscaling uses area.
        input_resize_method = "bicubic" if model_w > source_w or model_h > source_h else "area"

        # VAE encoding stays a separate phase so the PiD model is not repeatedly
        # swapped in and out. These CPU latents are much smaller than 4K output.
        lowres_latents: list[torch.Tensor] = []
        encode_start = 0
        encode_batch = max(1, int(batch_size))
        with torch.inference_mode():
            while encode_start < frame_count:
                current_encode = min(encode_batch, frame_count - encode_start)
                while True:
                    chunk = None
                    try:
                        chunk = comfy.utils.common_upscale(
                            source_images[encode_start : encode_start + current_encode].movedim(-1, 1),
                            model_w,
                            model_h,
                            input_resize_method,
                            "center",
                        ).movedim(1, -1)
                        chunk.clamp_(0.0, 1.0)
                        lowres_latents.append(encode_vae.encode(chunk).detach().cpu())
                    except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
                        if not _is_oom_error(error):
                            raise
                        chunk = None
                        error.__traceback__ = None
                        _release_after_oom(comfy.model_management.get_torch_device())
                        if current_encode > 1:
                            current_encode = max(1, current_encode // 2)
                            encode_batch = current_encode
                            continue
                        raise RuntimeError(
                            "Musefish PiD could not VAE-encode a single frame without OOM"
                        ) from error
                    finally:
                        del chunk
                    encode_start += current_encode
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    break
        del source_images
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt))
        negative = _conditioning_zero_out(positive)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        ).cpu()

        with pid_pixel_runtime(
            model,
            pixel_chunk_size,
            attention_backend=attention_backend,
        ) as sampling_model:
            # Use Comfy's normal patcher residency and eviction policy.  The
            # runtime clone owns only the scoped PiD pixel-block forward patches.
            comfy.model_management.load_models_gpu([sampling_model])

            output_images = torch.empty(
                (frame_count, output_h, output_w, 3), dtype=torch.float32, device="cpu"
            )
            noise_template: torch.Tensor | None = None
            latent_device = comfy.model_management.intermediate_device()
            output_start = 0
            preferred_batch = max(1, int(batch_size))
            with torch.inference_mode():
                for cache_index, lowres_cpu in enumerate(lowres_latents):
                    cache_offset = 0
                    while cache_offset < lowres_cpu.shape[0]:
                        current_batch = min(preferred_batch, lowres_cpu.shape[0] - cache_offset)
                        while True:
                            lowres = latent_image = positive_pid = noise = samples = decoded = None
                            try:
                                lowres = lowres_cpu[cache_offset : cache_offset + current_batch].to(latent_device)
                                latent_image = torch.zeros(
                                    (current_batch, 3, model_target_h, model_target_w),
                                    device=latent_device,
                                    dtype=lowres.dtype,
                                )
                                positive_pid = _pid_conditioning(
                                    positive, lowres, latent_format, degrade_sigma
                                )
                                if noise_template is None:
                                    # A single seed-derived frame is repeated for every
                                    # frame, so changing batch boundaries cannot change noise.
                                    noise_template = comfy.sample.prepare_noise(latent_image[:1], int(seed))
                                noise = noise_template.repeat(current_batch, 1, 1, 1)
                                # cfg=1 has one fixed LQ branch. The cache
                                # scope is exactly this sample batch, so a
                                # retry, next batch, or other cfg cannot reuse
                                cache_enabled = (
                                    float(cfg) == 1.0
                                    and _pid_lq_cache_safe_path(positive_pid, sampling_model)
                                )
                                with pid_lq_projection_cache(
                                    sampling_model, enabled=cache_enabled
                                ):
                                    samples = comfy.sample.sample_custom(
                                        sampling_model,
                                        noise,
                                        cfg,
                                        sampler,
                                        sigmas,
                                        positive_pid,
                                        negative,
                                        latent_image,
                                        disable_pbar=False,
                                        seed=int(seed),
                                    )
                                # PiD predicts pixels in [-1, 1], not a learned VAE
                                # latent. Convert on CPU in float32: a pixel_space
                                # VAE call would offload PiD and round through BF16.
                                decoded = samples.detach().to(device="cpu", dtype=torch.float32, copy=True)
                                decoded.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
                                decoded = decoded.movedim(1, -1)
                                if int(upscale_factor) != _MODEL_SCALE:
                                    decoded = comfy.utils.common_upscale(
                                        decoded.movedim(-1, 1), output_w, output_h, "area", "disabled"
                                    ).movedim(1, -1)
                                output_images[output_start + cache_offset : output_start + cache_offset + current_batch] = (
                                    decoded[:, :, :, :3].clamp(0.0, 1.0)
                                )
                            except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
                                if not _is_oom_error(error):
                                    raise
                                lowres = latent_image = positive_pid = noise = samples = decoded = None
                                error.__traceback__ = None
                                _release_after_oom(comfy.model_management.get_torch_device())
                                if current_batch > 1:
                                    current_batch = max(1, current_batch // 2)
                                    preferred_batch = current_batch
                                    continue
                                raise RuntimeError(
                                    "Musefish PiD could not process a single frame without OOM"
                                ) from error
                            finally:
                                del lowres, latent_image, positive_pid, noise, samples, decoded
                            cache_offset += current_batch
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            break
                    output_start += lowres_cpu.shape[0]
                    lowres_latents[cache_index] = None
                    del lowres_cpu

        del lowres_latents, noise_template, positive, negative, sigmas
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=output_images,
                audio=audio,
                frame_rate=Fraction(round(float(frame_rate) * 1000), 1000),
            ),
            bit_depth=8,
        )
        return io.NodeOutput(output_video, output_images)


# ---------------------------------------------------------------------------
# AutoBatch Image Sharpen FS — frequency-separation sharpening with bounded
# float32 GPU batches sized to free VRAM; the low-pass blur stays on CPU and
# preserves float precision for long 4K sequences.
# ---------------------------------------------------------------------------


def _fs_color_burn_blend(base, blend):
    return torch.clamp(1 - (1 - base) / (blend + 1e-8), 0, 1)


def _fs_divide_blend(base, blend):
    return torch.clamp(base / (blend + 1e-8), 0, 1)


def _fs_hard_light_freq_sep(original, low_pass):
    high_pass = (_fs_color_burn_blend(original, 1 - low_pass) + _fs_divide_blend(original, low_pass)) / 2
    return high_pass


def _fs_hard_light_blend(base, blend):
    return torch.where(blend <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))


def _fs_linear_light_freq_sep(base, blend):
    return (base + (1 - blend)) / 2


def _fs_linear_light_blend(base, blend):
    return torch.where(blend <= 0.5, base + 2 * blend - 1, base + 2 * (blend - 0.5))


def _fs_low_pass_cpu(chunk: torch.Tensor, blur_type: str, intensity: int) -> torch.Tensor:
    import cv2
    import numpy as np

    ksize = max(3, int(intensity) - 1)
    if ksize % 2 == 0:
        ksize += 1
    # Keep the source in float32 all the way through. In particular, never use
    # the old uint8 round-trip, which introduced visible 1/255 stair-stepping.
    arr = np.asarray(chunk.detach().cpu(), dtype=np.float32).clip(0.0, 1.0)
    out = np.empty_like(arr, dtype=np.float32)
    if blur_type == "median":
        if ksize <= 5:
            for i in range(arr.shape[0]):
                out[i] = cv2.medianBlur(arr[i], ksize)
        else:
            # OpenCV only accepts CV_8U for median kernels larger than 5. Use
            # scipy's float32 implementation instead of silently quantizing.
            try:
                from scipy.ndimage import median_filter
            except ImportError as error:
                raise RuntimeError(
                    "Float32 median blur with intensity > 6 requires scipy"
                ) from error
            for i in range(arr.shape[0]):
                out[i] = median_filter(arr[i], size=(ksize, ksize, 1), mode="reflect")
    elif blur_type == "gaussian":
        for i in range(arr.shape[0]):
            out[i] = cv2.GaussianBlur(arr[i], (ksize, ksize), 0)
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}")
    return torch.from_numpy(out)


_FS_PEAK_COPIES = 6.0  # float32 freq-sep temporaries
_FS_GPU_MAX_BATCH = 64
_FS_CPU_MAX_BATCH = 32


def _fs_gpu_batch(n_frames: int, frame_bytes: float, max_frames: int) -> int:
    """Choose a GPU batch with headroom; 0 means even one frame is too large."""
    if n_frames == 0:
        return 0
    free = comfy.model_management.get_free_memory()
    if free <= 0:
        return max(1, min(max_frames, n_frames))
    block = int(free * _MEMORY_HEADROOM / (frame_bytes * _FS_PEAK_COPIES))
    return min(max_frames, max(0, block), n_frames)


def _fs_cpu_batch(n_frames: int, frame_bytes: float, max_frames: int) -> int:
    """Choose a float32 CPU batch with memory headroom."""
    import psutil

    if n_frames == 0:
        return 0
    free = psutil.virtual_memory().available
    if free <= 0:
        return max(1, min(_FS_CPU_MAX_BATCH, max_frames, n_frames))
    block = int(free * _MEMORY_HEADROOM / (frame_bytes * _FS_PEAK_COPIES))
    return max(1, min(_FS_CPU_MAX_BATCH, max_frames, max(1, block), n_frames))


def _fs_process_chunk(
    chunk: torch.Tensor,
    method: str,
    blur_type: str,
    intensity: int,
    work: torch.device,
    amount: float,
    noise_threshold: float,
) -> torch.Tensor:
    """Run one frequency-separation chunk and return a finite CPU float32 tensor."""
    low_pass = _fs_low_pass_cpu(chunk, blur_type, intensity)
    orig = chunk[:, :, :, :3].to(device=work, dtype=torch.float32).permute(0, 3, 1, 2)
    lp = low_pass.to(device=work, dtype=torch.float32).permute(0, 3, 1, 2)
    if method == "hard":
        hp = _fs_hard_light_freq_sep(orig, lp)
        sharp = _fs_hard_light_blend(orig, hp)
    elif method == "linear":
        hp = _fs_linear_light_freq_sep(orig, lp)
        sharp = _fs_linear_light_blend(orig, hp)
    else:
        raise ValueError(f"Unknown method: {method}")
    # Soft-threshold the residual so low-amplitude grain is removed rather than
    # amplified. A conservative luma-gradient guard protects hairlines,
    # eyelids, and silhouettes from contour ringing while retaining flat detail.
    detail = sharp - orig
    threshold = max(0.0, float(noise_threshold))
    if threshold > 0.0:
        detail = torch.sign(detail) * torch.relu(detail.abs() - threshold)
    luma = 0.299 * orig[:, 0:1] + 0.587 * orig[:, 1:2] + 0.114 * orig[:, 2:3]
    edge = torch.zeros_like(luma)
    edge[:, :, :, 1:] = torch.maximum(edge[:, :, :, 1:], (luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs())
    edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs())
    detail = detail * (1.0 / (1.0 + 4.0 * edge)).clamp_min(0.2)
    output = orig + float(amount) * detail
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("AutoBatchImageSharpenFS produced non-finite output")
    output = output.clamp(0.0, 1.0)
    return output.permute(0, 2, 3, 1).float().cpu()


class AutoBatchImageSharpenFS(io.ComfyNode):
    """Frequency-separation sharpening with auto-batched GPU execution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AutoBatchImageSharpenFS",
            display_name="AutoBatch Image Sharpen FS",
            search_aliases=["autobatch sharpen", "频率分离锐化", "自动分批锐化"],
            category="Musefish/Image",
            description=(
                "Frequency-separation sharpening (hard/linear light) using float32 "
                "bounded batches. Gaussian and median low-pass filtering preserve "
                "float precision; auto mode may fall back to CPU after GPU OOM."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("method", options=["hard", "linear"], default="hard"),
                io.Combo.Input("blur_type", options=["median", "gaussian"], default="median"),
                io.Int.Input("intensity", default=6, min=1, max=31, step=1),
                io.Int.Input("frames_per_batch", default=0, min=0, max=128, step=1),
                io.Combo.Input("device", options=["auto", "gpu", "cpu"], default="auto"),
                io.Float.Input("amount", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("noise_threshold", default=0.0, min=0.0, max=1.0, step=0.005),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        method: str,
        blur_type: str,
        intensity: int,
        frames_per_batch: int = 0,
        device: str = "auto",
        amount: float = 1.0,
        noise_threshold: float = 0.0,
    ) -> io.NodeOutput:
        if images is None or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an RGB frame batch [N,H,W,3]")
        n, height, width = images.shape[:3]
        if n == 0:
            return io.NodeOutput(images)
        frame_bytes = height * width * 3 * 4
        fixed = int(frames_per_batch)

        if device == "cpu":
            work = torch.device("cpu")
            batch = fixed if fixed > 0 else _fs_cpu_batch(n, frame_bytes, _FS_CPU_MAX_BATCH)
        else:
            work = comfy.model_management.get_torch_device()
            batch = fixed if fixed > 0 else _fs_gpu_batch(n, frame_bytes, _FS_GPU_MAX_BATCH)
            if batch < 1:
                if device == "auto":
                    work = torch.device("cpu")
                    batch = fixed if fixed > 0 else _fs_cpu_batch(n, frame_bytes, _FS_CPU_MAX_BATCH)
                else:
                    batch = 1

        result = torch.empty((n, height, width, 3), dtype=torch.float32, device="cpu")
        preferred_batch = max(1, batch)
        start = 0
        while start < n:
            active = min(preferred_batch, n - start)
            while True:
                try:
                    processed = _fs_process_chunk(
                        images[start : start + active],
                        method,
                        blur_type,
                        int(intensity),
                        work,
                        float(amount),
                        float(noise_threshold),
                    )
                except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    if not _is_oom_error(error):
                        raise
                    processed = None
                    error.__traceback__ = None
                    _release_after_oom(comfy.model_management.get_torch_device())
                    if active > 1:
                        active = max(1, active // 2)
                        preferred_batch = active
                        continue
                    if device == "auto" and work.type != "cpu":
                        work = torch.device("cpu")
                        preferred_batch = fixed if fixed > 0 else _fs_cpu_batch(
                            n, frame_bytes, _FS_CPU_MAX_BATCH
                        )
                        active = min(preferred_batch, n - start)
                        continue
                    raise RuntimeError(
                        "AutoBatchImageSharpenFS could not process a single frame without OOM"
                    ) from error
                result[start : start + active] = processed
                del processed
                start += active
                comfy.model_management.throw_exception_if_processing_interrupted()
                break
        return io.NodeOutput(result)
class MusefishExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MusefishPiDBatchVideoUpscale, AutoBatchAntiflicker, AutoBatchImageSharpenFS, MusefishUniverSRModel, MusefishUniverSRGeneralAudio, MusefishUniverSRSpeechAudio, MusefishDLSS5NeuralRender, MusefishVideoDownload, MusefishWeChatChannels]


async def comfy_entrypoint() -> MusefishExtension:
    return MusefishExtension()
