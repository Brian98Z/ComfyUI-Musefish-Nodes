"""Deterministic behavioral checks for Musefish video nodes.

These tests intentionally import the real ComfyUI packages.  The PiD checks only
stub the model/sampler *boundary* so that batch/seed/error policy can be tested
without pretending that a synthetic model is an end-to-end quality result.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch


# The plugin directory contains a hyphen, so import it through importlib rather
# than using Python syntax.  Running under an actual ComfyUI checkout is a
# prerequisite: do not replace this with fake ``comfy`` modules.
PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_DIR.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
import comfy
musefish = importlib.import_module(
    "custom_nodes.ComfyUI-Musefish-Nodes.musefish_nodes"
)



def _tensor_output(node_output, index: int = 0) -> torch.Tensor:
    """Extract a tensor from Comfy's real NodeOutput without depending on internals."""
    value = node_output[index]
    assert isinstance(value, torch.Tensor)
    return value


def _antiflicker(images: torch.Tensor, *, batch: int = 0, luma: float = 15.0, chroma: float = 20.0):
    return _tensor_output(
        musefish.AutoBatchAntiflicker.execute(
            images=images,
            luma_tmp=luma,
            chroma_tmp=chroma,
            frames_per_batch=batch,
            device="cpu",
        )
    )


def _sharpen(
    images: torch.Tensor,
    *,
    batch: int = 0,
    method: str = "hard",
    blur_type: str = "median",
    intensity: int = 6,
    amount: float = 1.0,
    noise_threshold: float = 0.0,
):
    kwargs = {
        "images": images,
        "method": method,
        "blur_type": blur_type,
        "intensity": intensity,
        "frames_per_batch": batch,
        "device": "cpu",
        "amount": amount,
        "noise_threshold": noise_threshold,
    }
    # The optional controls are a trailing extension.  Filtering here lets the
    # test compare a checked-out baseline while still requiring the new module
    # to expose them (see test_sharpen_optional_controls_are_trailing).
    kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in inspect.signature(musefish.AutoBatchImageSharpenFS.execute).parameters
    }
    return _tensor_output(musefish.AutoBatchImageSharpenFS.execute(**kwargs))


def test_video_node_ids_and_trailing_optional_controls():
    assert musefish.MusefishPiDBatchVideoUpscale.GET_SCHEMA().node_id == "MusefishPiDBatchVideoUpscale"
    assert musefish.AutoBatchAntiflicker.GET_SCHEMA().node_id == "AutoBatchAntiflicker"
    assert musefish.AutoBatchImageSharpenFS.GET_SCHEMA().node_id == "AutoBatchImageSharpenFS"

    pid_parameters = list(inspect.signature(musefish.MusefishPiDBatchVideoUpscale.execute).parameters)
    assert pid_parameters[:4] == ["images", "audio", "frame_rate", "model"]
    assert pid_parameters[-4:] == ["steps", "seed", "pixel_chunk_size", "attention_backend"]
    assert inspect.signature(musefish.MusefishPiDBatchVideoUpscale.execute).parameters[
        "attention_backend"
    ].default == "Kitchen"
    sharpen_parameters = list(inspect.signature(musefish.AutoBatchImageSharpenFS.execute).parameters)
    assert sharpen_parameters[:7] == [
        "images", "method", "blur_type", "intensity", "frames_per_batch", "device", "amount"
    ]
    assert sharpen_parameters[-1] == "noise_threshold"
    assert inspect.signature(musefish.AutoBatchImageSharpenFS.execute).parameters["amount"].default == 1.0
    assert inspect.signature(musefish.AutoBatchImageSharpenFS.execute).parameters["noise_threshold"].default == 0.0


def test_empty_and_invalid_image_batches_are_rejected_or_preserved_as_contract():
    empty = torch.empty((0, 8, 8, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        _antiflicker(empty)
    # Sharpen is an identity for an empty batch so graph construction can pass
    # through an empty slice without manufacturing a frame.
    sharpen_empty = _sharpen(empty)
    assert sharpen_empty.shape == empty.shape

    for bad in (None, torch.zeros((8, 8, 3)), torch.zeros((1, 8, 8, 2))):
        with pytest.raises(Exception):
            musefish.AutoBatchAntiflicker.execute(
                images=bad, luma_tmp=15.0, chroma_tmp=20.0, frames_per_batch=1, device="cpu"
            )
        with pytest.raises(Exception):
            musefish.AutoBatchImageSharpenFS.execute(
                images=bad,
                method="hard",
                blur_type="median",
                intensity=6,
                frames_per_batch=1,
                device="cpu",
            )


def test_antiflicker_manual_batches_are_neighbor_halo_equivalent():
    generator = torch.Generator().manual_seed(123)
    images = torch.rand((7, 12, 16, 3), generator=generator)
    whole = _antiflicker(images, batch=7)
    one_by_one = _antiflicker(images, batch=1)
    assert one_by_one.shape == images.shape
    assert torch.allclose(one_by_one, whole, atol=2e-6, rtol=2e-6)


def test_sharpen_manual_batches_are_equivalent_and_preserve_rgb_shape():
    generator = torch.Generator().manual_seed(456)
    images = torch.rand((5, 16, 20, 3), generator=generator)
    whole = _sharpen(images, batch=5, blur_type="gaussian", intensity=8)
    one_by_one = _sharpen(images, batch=1, blur_type="gaussian", intensity=8)
    assert one_by_one.shape == images.shape
    assert torch.allclose(one_by_one, whole, atol=1e-6, rtol=1e-6)


def test_antiflicker_rejects_shot_switch_and_moving_edge_ghosts():
    # A hard cut must not be averaged into a gray transition.
    cut = torch.cat(
        [
            torch.zeros((2, 8, 8, 3)),
            torch.ones((2, 8, 8, 3)),
        ],
        dim=0,
    )
    cut_out = _antiflicker(cut, batch=1, luma=2.0, chroma=2.0)
    assert float(cut_out[1].max()) < 0.1
    assert float(cut_out[2].min()) > 0.9

    # The bright bar moves by two pixels per frame.  The center frame may
    # smooth its true neighbors, but it must not leave a one-sided old bar.
    moving = torch.zeros((5, 12, 12, 3))
    for frame_index in range(5):
        x = frame_index * 2
        moving[frame_index, 3:9, x : x + 2] = 1.0
    moving_out = _antiflicker(moving, batch=1, luma=5.0, chroma=5.0)
    assert float(moving_out[2, 3:9, 2].mean()) < 0.1
    assert float(moving_out[2, 3:9, 4].mean()) > 0.9
def test_antiflicker_reduces_high_frequency_edge_noise_without_blurring_contour():
    height = width = 48
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    diagonal = (xx > yy).float().unsqueeze(-1)
    noise = 0.04 * torch.randn((5, height, width, 3), generator=torch.Generator().manual_seed(42))
    images = (diagonal.unsqueeze(0) + noise).clamp(0.0, 1.0)
    output = _antiflicker(images, batch=1, luma=15.0, chroma=20.0)

    black_region = (xx < 10) & (yy > 28)
    input_noise = images[:, black_region, :].std()
    output_noise = output[:, black_region, :].std()
    assert float(output_noise) < float(input_noise) * 0.9

    # Sample points well away from the diagonal: the contour must remain a
    # high-contrast boundary rather than becoming a globally blurred ramp.
    assert float(output[2, 8, 40].mean()) > 0.85
    assert float(output[2, 40, 8].mean()) < 0.15




def test_postprocess_does_not_create_uint8_gradient_staircase():
    gradient = torch.linspace(0.0, 1.0, 1024, dtype=torch.float32).view(1, 1, 1024, 1)
    gradient = gradient.expand(3, 8, 1024, 3).clone()
    anti_out = _antiflicker(gradient, batch=1, luma=20.0, chroma=20.0)
    assert torch.unique(anti_out[1, 0, :, 0]).numel() > 256

    sharp_out = _sharpen(gradient, batch=1, blur_type="gaussian", intensity=6)
    assert torch.unique(sharp_out[1, 0, :, 0]).numel() > 256


def test_sharpen_noise_and_overshoot_are_finite_and_clamped():
    generator = torch.Generator().manual_seed(789)
    noisy = torch.rand((3, 20, 20, 3), generator=generator)
    for method in ("hard", "linear"):
        out = _sharpen(noisy, batch=1, method=method, blur_type="gaussian", intensity=12, amount=2.0)
        assert torch.isfinite(out).all()
        assert float(out.min()) >= -1e-6
        assert float(out.max()) <= 1.0 + 1e-6


def test_postprocess_honors_processing_interrupt(monkeypatch):
    images = torch.rand((4, 8, 8, 3), generator=torch.Generator().manual_seed(11))
    calls = 0

    def interrupt_after_first_chunk():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("test cancellation")

    monkeypatch.setattr(
        musefish.comfy.model_management,
        "throw_exception_if_processing_interrupted",
        interrupt_after_first_chunk,
    )
    with pytest.raises(InterruptedError, match="test cancellation"):
        _antiflicker(images, batch=1)
    calls = 0
    with pytest.raises(InterruptedError, match="test cancellation"):
        _sharpen(images, batch=1)


def _install_pid_boundary_stubs(monkeypatch, *, oom_once: bool = False, non_oom: BaseException | None = None):
    """Install tiny deterministic VAE/sampler boundaries, never a fake model result."""
    monkeypatch.setattr(musefish, "_MODEL_LONG_EDGE", 4)
    monkeypatch.setattr(musefish, "_MODEL_SCALE", 1)
    monkeypatch.setattr(musefish.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(musefish.comfy.model_management, "load_models_gpu", lambda models, force_full_load=True: None)
    monkeypatch.setattr(musefish.comfy.model_management, "throw_exception_if_processing_interrupted", lambda: None)
    monkeypatch.setattr(musefish.comfy.samplers, "sampler_object", lambda name: name)
    monkeypatch.setattr(musefish.comfy.samplers, "calculate_sigmas", lambda sampling, scheduler, steps: torch.ones(2))

    encoded_means = []

    class FakeEncodeVAE:
        def encode(self, chunk):
            encoded_means.extend(chunk.mean(dim=(1, 2, 3)).tolist())
            return chunk


    class FakeClip:
        def tokenize(self, prompt):
            return prompt

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones((1, 1)), {"pooled_output": torch.ones((1, 1))}]]

    class FakeModel:
        def __init__(self):
            # Runtime detection must see a real ModelPatcher-like object path,
            # but an empty block list intentionally remains a non-PiD boundary.
            self.diffusion_model = types.SimpleNamespace(pixel_blocks=[])

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self.diffusion_model
            assert name == "model_sampling"
            return object()

    noise_calls = []
    sample_calls = []

    def prepare_noise(latent_image, seed):
        noise_calls.append((tuple(latent_image.shape), int(seed)))
        return torch.full_like(latent_image, 0.25)

    def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, **kwargs):
        sample_calls.append((int(noise.shape[0]), int(kwargs["seed"])))
        if non_oom is not None:
            raise non_oom
        if oom_once and len(sample_calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test sampler")
        frame_signal = torch.tensor(
            encoded_means[: noise.shape[0]], dtype=latent_image.dtype, device=latent_image.device
        ).view(-1, 1, 1, 1)
        del encoded_means[: noise.shape[0]]
        return latent_image + noise + frame_signal

    monkeypatch.setattr(musefish.comfy.sample, "prepare_noise", prepare_noise)
    monkeypatch.setattr(musefish.comfy.sample, "sample_custom", sample_custom)
    return FakeModel(), FakeClip(), FakeEncodeVAE(), noise_calls, sample_calls


def _pid_kwargs(images, *, batch_size):
    return {
        "images": images,
        "audio": None,
        "frame_rate": 24.0,
        "model": None,  # replaced by caller
        "clip": None,  # replaced by caller
        "encode_vae": None,  # replaced by caller
        "positive_prompt": "deterministic boundary probe",
        "batch_size": batch_size,
        "upscale_factor": 2,
        "latent_format": "flux",
        "degrade_sigma": 0.0,
        "cfg": 1.0,
        "sampler_name": "lcm",
        "scheduler": "simple",
        "steps": 2,
        "seed": 1234,
        "pixel_chunk_size": 256,
    }


def test_pid_batch_boundary_reuses_seed_noise_and_preserves_order(monkeypatch):
    model, clip, encode_vae, noise_calls, sample_calls = _install_pid_boundary_stubs(monkeypatch)
    images = torch.arange(5 * 2 * 2 * 3, dtype=torch.float32).reshape(5, 2, 2, 3) / 64.0
    kwargs = _pid_kwargs(images, batch_size=2)
    kwargs.update(model=model, clip=clip, encode_vae=encode_vae)
    output = musefish.MusefishPiDBatchVideoUpscale.execute(**kwargs)
    output_images = _tensor_output(output, 1)

    assert output_images.shape[0] == images.shape[0]
    assert noise_calls == [((1, 3, 16, 16), 1234)]
    assert sample_calls == [(2, 1234), (2, 1234), (1, 1234)]
    # This is a mocked sampler-boundary check, not a claim about model fidelity.
    frame_means = output_images.mean(dim=(1, 2, 3))
    assert torch.all(frame_means[1:] > frame_means[:-1])


def test_pid_oom_reduces_only_the_active_batch_and_retries(monkeypatch):
    model, clip, encode_vae, _, sample_calls = _install_pid_boundary_stubs(monkeypatch, oom_once=True)
    images = torch.rand((5, 2, 2, 3), generator=torch.Generator().manual_seed(12))
    kwargs = _pid_kwargs(images, batch_size=4)
    kwargs.update(model=model, clip=clip, encode_vae=encode_vae)
    output = musefish.MusefishPiDBatchVideoUpscale.execute(**kwargs)
    assert _tensor_output(output, 1).shape[0] == 5
    assert sample_calls[0][0] == 4
    assert sample_calls[1][0] < sample_calls[0][0]
    assert sum(batch for batch, _ in sample_calls[1:]) == 5

def test_pid_pixel_decode_is_float32_affine_and_does_not_mutate_input(monkeypatch):
    model, clip, encode_vae, _, _ = _install_pid_boundary_stubs(monkeypatch)
    images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
    before = images.clone()
    kwargs = _pid_kwargs(images, batch_size=2)
    kwargs.update(model=model, clip=clip, encode_vae=encode_vae, upscale_factor=4)

    output = musefish.MusefishPiDBatchVideoUpscale.execute(**kwargs)
    output_images = _tensor_output(output, 1)
    assert output_images.dtype == torch.float32
    # The sampler boundary emits 0.25 in the PiD pixel range [-1, 1].
    # A learned pixel-space VAE would not produce this exact affine value.
    torch.testing.assert_close(output_images, torch.full_like(output_images, 0.625))
    assert torch.equal(images, before)


def test_pid_non_oom_exception_is_not_swallowed(monkeypatch):
    error = RuntimeError("sampler configuration is invalid")
    model, clip, encode_vae, _, _ = _install_pid_boundary_stubs(monkeypatch, non_oom=error)
    kwargs = _pid_kwargs(torch.rand((2, 2, 2, 3), generator=torch.Generator().manual_seed(13)), batch_size=2)
    kwargs.update(model=model, clip=clip, encode_vae=encode_vae)
    with pytest.raises(RuntimeError, match="sampler configuration is invalid"):
        musefish.MusefishPiDBatchVideoUpscale.execute(**kwargs)
