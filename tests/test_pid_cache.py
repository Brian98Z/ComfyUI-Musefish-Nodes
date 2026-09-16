"""CPU contract tests for the scoped PiD LQ projection cache."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_DIR.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from comfy import ops  # noqa: E402
from comfy.ldm.pixeldit.modules import PiTBlock  # noqa: E402
from comfy.ldm.pixeldit.pid import LQProjection2D  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

pid_runtime = importlib.import_module(
    "custom_nodes.ComfyUI-Musefish-Nodes.pid_runtime"
)


class _ToyPiD(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.lq_proj = LQProjection2D(
            latent_channels=1,
            hidden_dim=1,
            out_dim=1,
            patch_size=2,
            sr_scale=4,
            latent_spatial_down_factor=8,
            num_res_blocks=0,
            num_outputs=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=ops.disable_weight_init,
        )
        self.diffusion_model.pixel_blocks = nn.ModuleList([
            PiTBlock(
                pixel_hidden_size=4,
                patch_hidden_size=4,
                patch_size=2,
                num_heads=2,
                mlp_ratio=2.0,
                attn_hidden_size=8,
                attn_num_heads=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
                operations=ops.disable_weight_init,
            )
        ])
        self.device = torch.device("cpu")
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.2, 0.2)


def _make_patcher() -> ModelPatcher:
    return ModelPatcher(_ToyPiD(), torch.device("cpu"), torch.device("cpu"))


def _run_projection(patcher, latent, *, enabled=True):
    with pid_runtime.pid_pixel_runtime(patcher, 1) as runtime_model:
        runtime_model.patch_model(load_weights=False)
        projection = runtime_model.get_model_object("diffusion_model").lq_proj
        with pid_runtime.pid_lq_projection_cache(runtime_model, enabled=enabled):
            with torch.inference_mode():
                first = projection(latent, 2, 2)
                second = projection(latent.clone(), 2, 2)
            hits = runtime_model._pid_lq_projection_cache.hits
    return first, second, hits


def test_lq_projection_cache_hits_for_same_batch_without_value_sync(monkeypatch):
    patcher = _make_patcher()
    projection = patcher.get_model_object("diffusion_model").lq_proj
    calls = 0
    original = projection.latent_proj.forward

    def count_projection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(projection.latent_proj, "forward", count_projection)
    latent = torch.randn(2, 1, 2, 2)
    _, _, hits = _run_projection(patcher, latent)
    assert calls == 1
    assert hits == 1


def test_lq_projection_cache_isolated_between_batches_and_cfg(monkeypatch):
    patcher = _make_patcher()
    projection = patcher.get_model_object("diffusion_model").lq_proj
    calls = 0
    original = projection.latent_proj.forward

    def count_projection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(projection.latent_proj, "forward", count_projection)
    latent = torch.randn(2, 1, 2, 2)
    _run_projection(patcher, latent, enabled=True)
    _run_projection(patcher, latent, enabled=True)
    _run_projection(patcher, latent, enabled=False)
    assert calls == 4


def test_lq_projection_cache_releases_after_exception_and_runtime_restores_model(monkeypatch):
    patcher = _make_patcher()
    projection = patcher.get_model_object("diffusion_model").lq_proj
    original_align = projection._align_latent_to_patch_grid
    latent = torch.randn(2, 1, 2, 2)
    with pytest.raises(RuntimeError, match="stop"):
        with pid_runtime.pid_pixel_runtime(patcher, 1) as runtime_model:
            runtime_model.patch_model(load_weights=False)
            patched_projection = runtime_model.get_model_object("diffusion_model").lq_proj
            with pytest.raises(RuntimeError, match="stop"):
                with pid_runtime.pid_lq_projection_cache(runtime_model):
                    with torch.inference_mode():
                        patched_projection(latent, 2, 2)
                    assert runtime_model._pid_lq_projection_cache.features is not None
                    raise RuntimeError("stop")
            assert runtime_model._pid_lq_projection_cache.features is None
            raise RuntimeError("stop")
    assert projection._align_latent_to_patch_grid.__func__ is original_align.__func__
    assert patcher.object_patches == {}


def test_lq_projection_cache_is_mathematically_equivalent_to_uncached():
    patcher = _make_patcher()
    projection = patcher.get_model_object("diffusion_model").lq_proj
    latent = torch.randn(2, 1, 2, 2)
    with torch.inference_mode():
        baseline = projection(latent, 2, 2)
    actual, repeated, hits = _run_projection(patcher, latent)
    assert hits == 1
    for expected, result in zip(baseline, actual):
        torch.testing.assert_close(result, expected)
    for first, result in zip(actual, repeated):
        torch.testing.assert_close(result, first)
