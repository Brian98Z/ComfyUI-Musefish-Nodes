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
                io.Int.Input("model_long_edge", default=1024, min=256, max=4096, step=16),
                io.Int.Input("upscale_factor", default=4, min=2, max=8, step=1),
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
        model_long_edge: int,
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

        # Normalize once. All batches use identical dimensions, so the model and
        # VAE never receive a changing shape during this execution.
        scale = float(model_long_edge) / max(source_h, source_w)
        model_h = _round_multiple(int(round(source_h * scale)))
        model_w = _round_multiple(int(round(source_w * scale)))
        target_h = model_h * int(upscale_factor)
        target_w = model_w * int(upscale_factor)

        lowres_latents: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, frame_count, batch_size):
                chunk = comfy.utils.common_upscale(
                    source_images[start : start + batch_size].movedim(-1, 1),
                    model_w,
                    model_h,
                    "bicubic",
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
        latent_device = comfy.model_management.intermediate_device()
        with torch.inference_mode():
            for batch_index, lowres_cpu in enumerate(lowres_latents):
                lowres = lowres_cpu.to(latent_device)
                current_batch = lowres.shape[0]
                latent_image = torch.zeros(
                    (current_batch, 3, target_h, target_w),
                    device=latent_device,
                    dtype=lowres.dtype,
                )
                positive_pid = _pid_conditioning(
                    positive, lowres, latent_format, degrade_sigma
                )
                noise = comfy.sample.prepare_noise(
                    latent_image, int(seed) + batch_index
                )
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
                    seed=int(seed) + batch_index,
                )
                decoded = decode_vae.decode(samples).detach().float().cpu()
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


class MusefishExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MusefishPiDBatchVideoUpscale]


async def comfy_entrypoint() -> MusefishExtension:
    return MusefishExtension()
