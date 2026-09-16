"""CPU contract tests for the scoped PiD pixel-block runtime.

These tests exercise the real ComfyUI PiTBlock and ModelPatcher.  The block is
small enough for CPU execution, but all parameters are explicitly initialized
because ``disable_weight_init`` intentionally skips initialization.
"""

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

import comfy  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.ldm.pixeldit.modules import PiTBlock  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

pid_runtime = importlib.import_module(
    "custom_nodes.ComfyUI-Musefish-Nodes.pid_runtime"
)


class _ToyPiD(nn.Module):
    """Minimal ModelPatcher payload with PiD's diffusion-model object path."""

    def __init__(self, block: PiTBlock):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.lq_proj = nn.Identity()
        self.diffusion_model.pixel_blocks = nn.ModuleList([block])
        self.device = torch.device("cpu")


def _make_patcher() -> ModelPatcher:
    torch.manual_seed(1937)
    block = PiTBlock(
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
    # disable_weight_init creates storage without filling it.  Every parameter
    # must be initialized explicitly for deterministic numerical comparisons.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1938)
        for parameter in block.parameters():
            nn.init.uniform_(parameter, -0.2, 0.2)
    model = _ToyPiD(block)
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def _inputs(*, batch: int = 2):
    generator = torch.Generator(device="cpu").manual_seed(1940)
    # H=W=4 and P=2 gives L=4 global attention tokens and BL=8 pixel rows.
    x = torch.randn((batch * 4, 4, 4), generator=generator, dtype=torch.float32)
    s_cond = torch.randn((batch * 4, 4), generator=generator, dtype=torch.float32)
    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[0, 3] = -7.0
    options = {"rope_options": {"scale_x": 1.13, "scale_y": 0.87, "shift_x": 0.2}}
    return x, s_cond, mask, options


def _forward_original(patcher: ModelPatcher, x, s_cond, mask, options):
    block = patcher.get_model_object("diffusion_model").pixel_blocks[0]
    with torch.inference_mode():
        return block.forward(
            x.clone(), s_cond, 4, 4, 2, mask=mask, transformer_options=options
        )


def _forward_chunked(patcher: ModelPatcher, chunk_size: int, x, s_cond, mask, options):
    with pid_runtime.pid_pixel_runtime(patcher, chunk_size) as runtime_model:
        # Object patches are installed on the clone and become live through the
        # same ModelPatcher lifecycle used by Comfy's sampler.
        runtime_model.patch_model(load_weights=False)
        block = runtime_model.get_model_object("diffusion_model").pixel_blocks[0]
        with torch.inference_mode():
            result = block.forward(
                x.clone(), s_cond, 4, 4, 2, mask=mask, transformer_options=options
            )
    return result


def test_real_pitblock_matches_baseline_for_chunk_boundaries_and_global_attention(monkeypatch):
    patcher = _make_patcher()
    x, s_cond, mask, options = _inputs()
    baseline = _forward_original(patcher, x, s_cond, mask, options)

    attention_shapes: list[tuple[int, ...]] = []
    block = patcher.get_model_object("diffusion_model").pixel_blocks[0]
    original_attention_forward = block.attn.forward

    def capture_attention(*args, **kwargs):
        attention_shapes.append(tuple(args[0].shape))
        return original_attention_forward(*args, **kwargs)

    monkeypatch.setattr(block.attn, "forward", capture_attention)
    for chunk_size in (1, 3, 5, 256):
        actual = _forward_chunked(patcher, chunk_size, x, s_cond, mask, options)
        torch.testing.assert_close(actual, baseline, rtol=2e-5, atol=2e-6)

    # The compressed attention call is always [B, L, D], never a chunk-sized
    # prefix.  This guards the global-L invariant independently of output values.
    assert attention_shapes
    assert all(shape == (2, 4, 8) for shape in attention_shapes)


def test_nonpositive_chunk_and_non_pid_are_true_noops():
    for chunk_size in (0, -1):
        patcher = _make_patcher()
        original_forward = patcher.get_model_object("diffusion_model").pixel_blocks[0].forward
        with pid_runtime.pid_pixel_runtime(patcher, chunk_size) as runtime_model:
            assert runtime_model is patcher
            assert runtime_model.object_patches == {}
        assert patcher.get_model_object("diffusion_model").pixel_blocks[0].forward.__func__ is original_forward.__func__

    non_pid = ModelPatcher(nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
    with pid_runtime.pid_pixel_runtime(non_pid, 3) as runtime_model:
        assert runtime_model is non_pid
        assert runtime_model.object_patches == {}


def test_patch_is_restored_after_interruption_and_shared_clone_is_not_polluted(monkeypatch):
    patcher = _make_patcher()
    sibling = patcher.clone()
    block = patcher.get_model_object("diffusion_model").pixel_blocks[0]
    original_forward = block.forward
    calls = 0

    def interrupt_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("cancelled")

    monkeypatch.setattr(
        comfy.model_management,
        "throw_exception_if_processing_interrupted",
        interrupt_once,
    )
    x, s_cond, mask, options = _inputs()
    with pytest.raises(InterruptedError, match="cancelled"):
        _forward_chunked(patcher, 3, x, s_cond, mask, options)

    assert block.forward.__func__ is original_forward.__func__
    assert sibling.object_patches == {}
    assert sibling.get_model_object("diffusion_model").pixel_blocks[0].forward.__func__ is original_forward.__func__
    assert patcher.object_patches == {}

def test_patch_is_restored_after_generic_exception(monkeypatch):
    patcher = _make_patcher()
    block = patcher.get_model_object("diffusion_model").pixel_blocks[0]
    original_forward = block.forward

    def fail():
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", fail)
    x, s_cond, mask, options = _inputs()
    with pytest.raises(RuntimeError, match="sampler failed"):
        _forward_chunked(patcher, 3, x, s_cond, mask, options)
    assert block.forward.__func__ is original_forward.__func__
    assert patcher.object_patches == {}


def test_chunked_runtime_preserves_grad_enabled_semantics():
    patcher = _make_patcher()
    x, s_cond, mask, options = _inputs()
    block = patcher.get_model_object("diffusion_model").pixel_blocks[0]
    before = x.clone()
    # PiTBlock's established training path is the oracle.  It currently has
    # an in-place residual and is not required to support backward here; the
    # runtime must preserve that exact grad-enabled forward behavior.
    baseline_input = (x.clone() * 1.0).requires_grad_()
    baseline = block.forward(
        baseline_input, s_cond, 4, 4, 2, mask=mask, transformer_options=options
    )
    with pid_runtime.pid_pixel_runtime(patcher, 1) as runtime_model:
        runtime_model.patch_model(load_weights=False)
        patched_block = runtime_model.get_model_object("diffusion_model").pixel_blocks[0]
        patched_input = (x.clone() * 1.0).requires_grad_()
        actual = patched_block.forward(
            patched_input, s_cond, 4, 4, 2, mask=mask, transformer_options=options
        )
        assert actual.requires_grad
    torch.testing.assert_close(actual, baseline)
    assert torch.equal(x, before)
    assert block.forward.__func__ is PiTBlock.forward


def test_attention_backend_validation_and_kitchen_fallback(monkeypatch, caplog):
    assert pid_runtime.DEFAULT_ATTENTION_BACKEND == "Kitchen"
    assert pid_runtime.ATTENTION_BACKENDS == ("Kitchen", "cuDNN")
    assert pid_runtime.resolve_attention_backend("cuDNN") == "cuDNN"
    with pytest.raises(ValueError, match="Unknown attention_backend"):
        pid_runtime.resolve_attention_backend("invalid")

    monkeypatch.setattr(
        pid_runtime.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        False,
    )
    with caplog.at_level("WARNING"):
        assert pid_runtime.resolve_attention_backend("Kitchen") == "cuDNN"
    assert "falling back to cuDNN" in caplog.text


def test_attention_backend_override_is_clone_scoped(monkeypatch):
    monkeypatch.setattr(
        pid_runtime.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )
    patcher = _make_patcher()
    assert "optimized_attention_override" not in patcher.model_options["transformer_options"]
    with pid_runtime.pid_pixel_runtime(
        patcher, pixel_chunk_size=0, attention_backend="Kitchen"
    ) as runtime_model:
        assert runtime_model is not patcher
        assert callable(
            runtime_model.model_options["transformer_options"]["optimized_attention_override"]
        )
        assert "optimized_attention_override" not in patcher.model_options["transformer_options"]
    assert "optimized_attention_override" not in patcher.model_options["transformer_options"]


def test_cudnn_override_preserves_attention_callback_arguments(monkeypatch):
    seen = {}

    def fake_attention(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return args[0]

    monkeypatch.setattr(pid_runtime.comfy_attention, "attention_pytorch", fake_attention)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    q, k, v = (torch.ones((1, 2, 3, 4)) for _ in range(3))
    result = pid_runtime._cudnn_attention_override(q, k, v, 2, scale=0.5)
    assert result is q
    assert seen["args"][:3] == (q, k, v)
    assert seen["args"][3] == 2
    assert seen["kwargs"] == {"scale": 0.5}
