"""Workflow graph builders for the Musefish nodes.

Graphs mirror the plugin's node schemas exactly (verified against
INPUT_TYPES/define_schema on 2026-10-16). The plugin on H: remains the single
source of truth for behavior; this module only orchestrates it.
"""
from __future__ import annotations

import random
from typing import Any

# Exact enum strings from musefish_dlss5.py
SR_SCALES = ["off", "1× (Native)", "1.5× (Quality)", "2× (Balance)",
             "3× (Performance)", "4× (Ultra)", "1K", "2K", "4K", "8K"]
VSR_QUALITY = ["performance", "balanced", "quality", "ultra"]
STYLES = ["default", "natural", "cinema"]
VIDEO_ENCODERS = ["libx264（CPU 编码）", "h264_nvenc（GPU 编码加速）"]
# Exact values from musefish_audio.py
UNIVERSR_MODES = ["auto", "sr", "master", "sr_master", "stem_mix"]
UNIVERSR_ODE_METHODS = ["euler", "midpoint", "rk4"]
UNIVERSR_ACCEL = ["fp32", "cuDNN TF32", "bf16"]
UNIVERSR_INPUT_SR = ["auto", "8000", "12000", "16000", "24000"]


def _seed() -> int:
    return random.randint(0, 2**31 - 1)


def pid_video_graph(
    video_ref: str,
    *,
    unet: str,
    clip: str,
    vae: str,
    positive_prompt: str = "high quality, ultra detailed, sharp details",
    upscale_factor: int = 4,
    batch_size: int = 1,
    steps: int = 4,
    cfg: float = 1.0,
    latent_format: str = "flux",
    sampler_name: str = "lcm",
    scheduler: str = "simple",
    degrade_sigma: float = 0.16,
    pixel_chunk_size: int = 1024,
    attention_backend: str = "Kitchen",
    frame_rate: float = 24.0,
    seed: int | None = None,
    postprocess: bool = True,
    sharpen_intensity: int = 12,
    luma_tmp: float = 15.0,
    chroma_tmp: float = 20.0,
    frames_per_batch: int = 0,
    pp_device: str = "auto",
    sharpen_method: str = "hard",
    sharpen_blur: str = "median",
    sharpen_amount: float = 1.0,
    sharpen_noise: float = 0.0,
) -> tuple[dict, str]:
    """PiD on a video: LoadVideo -> GetVideoComponents (frames+audio+fps)
    -> PiD batch upscale -> optional postprocess -> SaveVideo (mp4 with audio)."""
    seed = _seed() if seed is None else seed
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip, "type": "pixeldit", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadVideo", "inputs": {"file": video_ref}},
        "5": {"class_type": "GetVideoComponents",
              "inputs": {"video": ["4", 0]}},
        "6": {
            "class_type": "MusefishPiDBatchVideoUpscale",
            "inputs": {
                "images": ["5", 0],
                "audio": ["5", 1],
                "frame_rate": ["5", 2],
                "model": ["1", 0],
                "clip": ["2", 0],
                "encode_vae": ["3", 0],
                "positive_prompt": positive_prompt,
                "batch_size": batch_size,
                "upscale_factor": upscale_factor,
                "latent_format": latent_format,
                "degrade_sigma": degrade_sigma,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "seed": seed,
                "pixel_chunk_size": pixel_chunk_size,
                "attention_backend": attention_backend,
            },
        },
    }
    images_src = "6"
    if postprocess:
        g["7"] = {
            "class_type": "AutoBatchAntiflicker",
            "inputs": {"images": ["6", 1], "luma_tmp": luma_tmp, "chroma_tmp": chroma_tmp,
                       "frames_per_batch": frames_per_batch, "device": pp_device},
        }
        g["8"] = {
            "class_type": "AutoBatchImageSharpenFS",
            "inputs": {"images": ["7", 0], "method": sharpen_method, "blur_type": sharpen_blur,
                       "intensity": sharpen_intensity, "frames_per_batch": frames_per_batch,
                       "device": pp_device, "amount": sharpen_amount,
                       "noise_threshold": sharpen_noise},
        }
        images_src = "8"
    g["10"] = {
        "class_type": "CreateVideo",   # images + audio + fps -> VIDEO
        "inputs": {"images": [images_src, 0], "fps": ["5", 2], "audio": ["5", 1]},
    }
    g["11"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["10", 0], "filename_prefix": "MusefishStudio/PiDVideo",
                   "format": "auto"},
    }
    return g, "11"


def pid_upscale_graph(
    image_ref: str,
    *,
    unet: str,
    clip: str,
    vae: str,
    positive_prompt: str = "high quality, ultra detailed, sharp details",
    upscale_factor: int = 4,
    batch_size: int = 1,
    steps: int = 4,
    cfg: float = 1.0,
    latent_format: str = "flux",
    sampler_name: str = "lcm",
    scheduler: str = "simple",
    degrade_sigma: float = 0.16,
    pixel_chunk_size: int = 1024,
    attention_backend: str = "Kitchen",
    frame_rate: float = 24.0,
    seed: int | None = None,
    postprocess: bool = True,
    sharpen_intensity: int = 12,
    luma_tmp: float = 15.0,
    chroma_tmp: float = 20.0,
    frames_per_batch: int = 0,
    pp_device: str = "auto",
    sharpen_method: str = "hard",
    sharpen_blur: str = "median",
    sharpen_amount: float = 1.0,
    sharpen_noise: float = 0.0,
) -> tuple[dict, str]:
    """PiD upscale -> optional antiflicker + FS sharpen -> SaveImage."""
    seed = _seed() if seed is None else seed
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip, "type": "pixeldit", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage",
              "inputs": {"image": image_ref, "upload": "image"}},
        "5": {
            "class_type": "MusefishPiDBatchVideoUpscale",
            "inputs": {
                "images": ["4", 0],
                "audio": None,
                "frame_rate": frame_rate,
                "model": ["1", 0],
                "clip": ["2", 0],
                "encode_vae": ["3", 0],
                "positive_prompt": positive_prompt,
                "batch_size": batch_size,
                "upscale_factor": upscale_factor,
                "latent_format": latent_format,
                "degrade_sigma": degrade_sigma,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "seed": seed,
                "pixel_chunk_size": pixel_chunk_size,
                "attention_backend": attention_backend,
            },
        },
    }
    out = "5"
    if postprocess:
        g["6"] = {
            "class_type": "AutoBatchAntiflicker",
            "inputs": {
                "images": ["5", 1],
                "luma_tmp": luma_tmp,
                "chroma_tmp": chroma_tmp,
                "frames_per_batch": frames_per_batch,
                "device": pp_device,
            },
        }
        g["7"] = {
            "class_type": "AutoBatchImageSharpenFS",
            "inputs": {
                "images": ["6", 0],
                "method": sharpen_method,
                "blur_type": sharpen_blur,
                "intensity": sharpen_intensity,
                "frames_per_batch": frames_per_batch,
                "device": pp_device,
                "amount": sharpen_amount,
                "noise_threshold": sharpen_noise,
            },
        }
        out = "7"
    g["9"] = {"class_type": "SaveImage",
              "inputs": {"images": [out, 0], "filename_prefix": "MusefishStudio/PiD"}}
    return g, "9"


def dlss5_image_graph(
    image_ref: str,
    *,
    super_resolution: str = "2× (Balance)",
    style: str = "default",
    vsr_quality: str = "ultra",
    intensity: float = 1.0,
    local_tone: float = 0.94,
    local_struct: float = 0.84,
    skin_struct: float = 1.0,
    use_auto_mask: bool = True,
    reset_every_n_frames: int = 1,
    keep_session: str = "off",
) -> tuple[dict, str]:
    g = {
        "1": {"class_type": "LoadImage",
              "inputs": {"image": image_ref, "upload": "image"}},
        "2": {
            "class_type": "MusefishDLSS5NeuralRender",
            "inputs": {
                "images": ["1", 0],
                "super_resolution": super_resolution,
                "style": style,
                "vsr_quality": vsr_quality,
                "intensity": intensity,
                "local_tone": local_tone,
                "local_struct": local_struct,
                "skin_struct": skin_struct,
                "use_auto_mask": use_auto_mask,
                "reset_every_n_frames": reset_every_n_frames,
                "keep_session": keep_session,
            },
        },
        "3": {"class_type": "SaveImage",
              "inputs": {"images": ["2", 0], "filename_prefix": "MusefishStudio/DLSS5"}},
    }
    return g, "3"


def dlss5_video_graph(
    video_ref: str,
    *,
    super_resolution: str = "2× (Balance)",
    style: str = "default",
    vsr_quality: str = "ultra",
    intensity: float = 1.0,
    local_tone: float = 0.94,
    local_struct: float = 0.84,
    skin_struct: float = 1.0,
    use_auto_mask: bool = True,
    crf: int = 19,
    encoder: str = "libx264（CPU 编码）",
) -> tuple[dict, str]:
    """Host LoadVideo -> MusefishDLSS5VideoStream -> host SaveVideo."""
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_ref}},
        "2": {
            "class_type": "MusefishDLSS5VideoStream",
            "inputs": {
                "video": ["1", 0],
                "super_resolution": super_resolution,
                "style": style,
                "vsr_quality": vsr_quality,
                "intensity": intensity,
                "local_tone": local_tone,
                "local_struct": local_struct,
                "skin_struct": skin_struct,
                "use_auto_mask": use_auto_mask,
                "crf": crf,
                "encoder": encoder,
            },
        },
        "3": {"class_type": "SaveVideo",
              "inputs": {"video": ["2", 0], "filename_prefix": "MusefishStudio/DLSS5Video",
                         "format": "auto"}},
    }
    return g, "3"


def universr_graph(
    audio_ref: str,
    *,
    model: str = "general",
    mode: str = "sr",
    download: bool = False,
    auto_params: bool = True,
    input_sr: str = "auto",
    ode_steps: int = 4,
    ode_method: str = "midpoint",
    guidance: float = 1.5,
    chunk_sec: int = 15,
    seed: int = 0,
    deess: bool = True,
    channel_mode: str = "auto",
    accel: str = "cuDNN TF32",
) -> tuple[dict, str]:
    """LoadAudio -> UniverSR model + process -> SaveAudio."""
    node_id = "MusefishUniverSRGeneralAudio" if model == "general" else "MusefishUniverSRSpeechAudio"
    g = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}},
        "2": {"class_type": "MusefishUniverSRModel",
              "inputs": {"model": model, "download": download}},
        "3": {
            "class_type": node_id,
            "inputs": {
                "audio": ["1", 0],
                "model_cache": ["2", 0],
                "auto_params": auto_params,
                "input_sr": input_sr,
                "ode_steps": ode_steps,
                "ode_method": ode_method,
                "guidance": guidance,
                "chunk_sec": chunk_sec,
                "seed": seed,
                "deess": deess,
                "channel_mode": channel_mode,
                "accel": accel,
                "mode": mode,
            },
        },
        "4": {"class_type": "SaveAudio",
              "inputs": {"audio": ["3", 0], "filename_prefix": "MusefishStudio/UniverSR"}},
    }
    return g, "4"
