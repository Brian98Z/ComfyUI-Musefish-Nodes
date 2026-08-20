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
_ANTIFLICKER_PEAK_COPIES = 8.0  # rgb + y/u/v + yf/uf/vf + prev/following/result
_ANTIFLICKER_MAX_BLOCK = 16
_ANTIFLICKER_CPU_MAX_BLOCK = 64


def _antiflicker_gpu_block(source: torch.Tensor) -> int:
    """Frame block fitting the currently free VRAM; 0 when even 1 frame does
    not fit (caller falls back to CPU in auto mode)."""
    n = source.shape[0]
    per_frame = source.shape[1] * source.shape[2] * 3 * 4  # float32 NHWC bytes
    free = comfy.model_management.get_free_memory()
    if free <= 0:
        return min(_ANTIFLICKER_MAX_BLOCK, n)
    block = int(free / (per_frame * _ANTIFLICKER_PEAK_COPIES))
    return min(_ANTIFLICKER_MAX_BLOCK, block, n)  # may be 0


def _antiflicker_cpu_block(source: torch.Tensor) -> int:
    """Frame block fitting the currently free system RAM (float32 copies)."""
    import psutil

    n = source.shape[0]
    per_frame = source.shape[1] * source.shape[2] * 3 * 4
    free = psutil.virtual_memory().available
    if free <= 0:
        return min(_ANTIFLICKER_CPU_MAX_BLOCK, n)
    block = int(free / (per_frame * _ANTIFLICKER_PEAK_COPIES))
    return max(1, min(_ANTIFLICKER_CPU_MAX_BLOCK, block, n))


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
            gpu_block = fixed if fixed > 0 else _antiflicker_gpu_block(source)
            if gpu_block >= 1:
                work = comfy.model_management.get_torch_device()
                block = gpu_block
            elif device == "auto":
                work = torch.device("cpu")
                block = fixed if fixed > 0 else _antiflicker_cpu_block(source)
            else:  # explicit "gpu": force at least one frame per chunk
                work = comfy.model_management.get_torch_device()
                block = 1

        result = torch.empty_like(source)  # CPU, never materialized on device whole

        for start in range(0, n, block):
            # One extra source frame on both sides so interior frames of the
            # chunk still see their true temporal neighbors. The two boundary
            # frames of each chunk are computed but discarded below.
            lo = max(0, start - 1)
            hi = min(n, start + block + 1)
            rgb = source[lo:hi].to(work)
            y, u, v = _rgb_to_yuv601(rgb)
            y_plane = y.squeeze(-1)
            yf = _temporal_bilateral(y_plane, float(luma_tmp)).unsqueeze(-1)
            uf = _temporal_bilateral(
                u.squeeze(-1), float(chroma_tmp), y_plane, float(luma_tmp)
            ).unsqueeze(-1)
            vf = _temporal_bilateral(
                v.squeeze(-1), float(chroma_tmp), y_plane, float(luma_tmp)
            ).unsqueeze(-1)
            rgb_f = _yuv601_to_rgb(yf, uf, vf)
            keep_lo = start - lo
            result[start : start + block] = rgb_f[keep_lo : keep_lo + block].cpu()
            del rgb, y, u, v, y_plane, yf, uf, vf, rgb_f
            comfy.model_management.throw_exception_if_processing_interrupted()

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
                "The model is loaded once and reused; optional audio and FPS are preserved in VIDEO."
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
    ) -> io.NodeOutput:
        if images is None:
            raise ValueError("images are required")
        if model is None or clip is None or encode_vae is None:
            raise ValueError("model, clip, and encode_vae are required")
        from nodes import VAELoader
        decode_vae = VAELoader().load_vae("pixel_space")[0]

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
        input_resize_method = "lanczos" if model_w > source_w or model_h > source_h else "area"

        lowres_latents: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, frame_count, batch_size):
                chunk = comfy.utils.common_upscale(
                    source_images[start : start + batch_size].movedim(-1, 1),
                    model_w,
                    model_h,
                    input_resize_method,
                    "center",
                ).movedim(1, -1)
                lowres_latents.append(encode_vae.encode(chunk).detach().cpu())
                comfy.model_management.throw_exception_if_processing_interrupted()

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt))
        negative = _conditioning_zero_out(positive)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        ).cpu()

        # Explicit preload prevents a model initialization for each frame batch.
        # sample_custom may still perform normal ComfyUI patcher bookkeeping,
        # but the same ModelPatcher object is reused throughout this call.
        comfy.model_management.load_models_gpu([model], force_full_load=True)

        output_chunks: list[torch.Tensor] = []
        noise_template: torch.Tensor | None = None
        latent_device = comfy.model_management.intermediate_device()
        with torch.inference_mode():
            for batch_index, lowres_cpu in enumerate(lowres_latents):
                lowres = lowres_cpu.to(latent_device)
                current_batch = lowres.shape[0]
                latent_image = torch.zeros(
                    (current_batch, 3, model_target_h, model_target_w),
                    device=latent_device,
                    dtype=lowres.dtype,
                )
                positive_pid = _pid_conditioning(
                    positive, lowres, latent_format, degrade_sigma
                )
                if noise_template is None:
                    noise_template = comfy.sample.prepare_noise(latent_image[:1], int(seed))
                noise = noise_template.repeat(current_batch, 1, 1, 1)
                samples = comfy.sample.sample_custom(
                    model,
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
                decoded = decode_vae.decode(samples).detach().float().cpu()
                if int(upscale_factor) != _MODEL_SCALE:
                    decoded = comfy.utils.common_upscale(
                        decoded.movedim(-1, 1), output_w, output_h, "area", "disabled"
                    ).movedim(1, -1)
                output_chunks.append(decoded[:, :, :, :3].clamp(0.0, 1.0))
                comfy.model_management.throw_exception_if_processing_interrupted()

        output_images = torch.cat(output_chunks, dim=0)
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
# GPU batches. Mirrors RES4LYF "Image Sharpen FS" behavior (hard/linear light)
# but sizes the float64 CUDA math to the free VRAM so long 4K sequences no
# longer OOM. The low-pass blur runs on CPU and never touches VRAM.
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
    arr = (chunk.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    out = np.empty_like(arr)
    if blur_type == "median":
        for i in range(arr.shape[0]):
            out[i] = cv2.medianBlur(arr[i], ksize)
    else:
        for i in range(arr.shape[0]):
            out[i] = cv2.GaussianBlur(arr[i], (ksize, ksize), 0)
    return torch.from_numpy(out).to(torch.float32) / 255.0


_FS_PEAK_COPIES = 6.0  # float64 freq-sep temps: original + low_pass + high_pass + blends
_FS_GPU_MAX_BATCH = 64
_FS_CPU_MAX_BATCH = 32


def _fs_gpu_batch(n_frames: int, frame_bytes64: float, max_frames: int) -> int:
    """GPU batch fitting the free VRAM; 0 when even 1 frame does not fit."""
    if n_frames == 0:
        return 0
    free = comfy.model_management.get_free_memory()
    if free <= 0:
        return max(1, min(max_frames, n_frames))
    block = int(free / (frame_bytes64 * _FS_PEAK_COPIES))
    return min(max_frames, block, n_frames)  # may be 0


def _fs_cpu_batch(n_frames: int, frame_bytes64: float, max_frames: int) -> int:
    """Batch fitting the free system RAM (float64 copies, CPU execution)."""
    import psutil

    if n_frames == 0:
        return 0
    free = psutil.virtual_memory().available
    if free <= 0:
        return max(1, min(_FS_CPU_MAX_BATCH, max_frames, n_frames))
    block = int(free / (frame_bytes64 * _FS_PEAK_COPIES))
    return max(1, min(_FS_CPU_MAX_BATCH, max_frames, block, n_frames))


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
                "Frequency-separation sharpening (hard/linear light) that runs the "
                "float64 GPU math in bounded batches sized to the free VRAM, so long "
                "4K sequences no longer OOM. The low-pass blur runs on CPU."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("method", options=["hard", "linear"], default="hard"),
                io.Combo.Input("blur_type", options=["median", "gaussian"], default="median"),
                io.Int.Input("intensity", default=6, min=1, max=31, step=1),
                io.Int.Input("frames_per_batch", default=0, min=0, max=128, step=1),
                io.Combo.Input("device", options=["auto", "gpu", "cpu"], default="auto"),
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
    ) -> io.NodeOutput:
        n = images.shape[0]
        if n == 0:
            return io.NodeOutput(images)
        frame_bytes64 = images.shape[1] * images.shape[2] * 3 * 8  # float64 NHWC bytes
        fixed = int(frames_per_batch)

        if device == "cpu":
            work = torch.device("cpu")
            batch = fixed if fixed > 0 else _fs_cpu_batch(n, frame_bytes64, _FS_CPU_MAX_BATCH)
        elif fixed > 0:
            work = comfy.model_management.get_torch_device()
            batch = fixed
        else:
            batch = _fs_gpu_batch(n, frame_bytes64, _FS_GPU_MAX_BATCH)
            if batch >= 1:
                work = comfy.model_management.get_torch_device()
            elif device == "auto":
                work = torch.device("cpu")
                batch = _fs_cpu_batch(n, frame_bytes64, _FS_CPU_MAX_BATCH)
            else:  # explicit "gpu": force one frame per chunk
                work = comfy.model_management.get_torch_device()
                batch = 1

        chunks = []
        for start in range(0, n, batch):
            chunk = images[start : start + batch]
            low_pass = _fs_low_pass_cpu(chunk, blur_type, int(intensity))  # CPU [B,H,W,3]
            orig = chunk.to(device=work, dtype=torch.float64).permute(0, 3, 1, 2)
            lp = low_pass.to(device=work, dtype=torch.float64).permute(0, 3, 1, 2)
            if method == "hard":
                hp = _fs_hard_light_freq_sep(orig, lp)
                sharp = _fs_hard_light_blend(orig, hp)
            else:
                hp = _fs_linear_light_freq_sep(orig, lp)
                sharp = _fs_linear_light_blend(orig, hp)
            chunks.append(sharp.permute(0, 2, 3, 1).float().cpu())
            del low_pass, orig, lp, hp, sharp
            comfy.model_management.throw_exception_if_processing_interrupted()
        return io.NodeOutput(torch.cat(chunks, dim=0))


class MusefishExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MusefishPiDBatchVideoUpscale, AutoBatchAntiflicker, AutoBatchImageSharpenFS]


async def comfy_entrypoint() -> MusefishExtension:
    return MusefishExtension()
