"""Scoped low-memory PiD pixel-block runtime.

This module deliberately lives in the Musefish plugin rather than ComfyUI core.  A
ModelPatcher clone owns the temporary ``forward`` object patches; the shared model
attributes are restored by the patcher's normal unpatch lifecycle when the context
exits.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
from torch import nn

import comfy.model_management
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.pixeldit.modules import PiTBlock, apply_adaln_
from comfy.ldm.pixeldit.pid import LQProjection2D


ATTENTION_BACKENDS = ("Kitchen", "cuDNN")
DEFAULT_ATTENTION_BACKEND = "Kitchen"


def resolve_attention_backend(name: str) -> str:
    """Validate the node option and resolve unavailable Kitchen to cuDNN."""
    if name not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unknown attention_backend {name!r}; choose one of {ATTENTION_BACKENDS}"
        )
    if name == "Kitchen" and not bool(
        getattr(comfy_attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
    ):
        import logging

        logging.warning(
            "Musefish PiD attention backend Kitchen is unavailable; falling back to cuDNN"
        )
        return "cuDNN"
    return name


def _cudnn_attention_override(q, *args, **kwargs):
    """Run Comfy's PyTorch attention while permitting the cuDNN SDPA backend."""
    attention_pytorch = comfy_attention.attention_pytorch
    if not torch.cuda.is_available():
        return attention_pytorch(q, *args, **kwargs)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        return attention_pytorch(q, *args, **kwargs)
    # Keep this context local to the clone's attention callback.  MATH remains
    # allowed for shapes cuDNN cannot implement; unlike a global backend toggle,
    # this cannot affect another model or a concurrent node execution.
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
        return attention_pytorch(q, *args, **kwargs)


def _install_attention_backend(patcher: object, backend: str) -> bool:
    """Install the selected backend through ModelPatcher's supported hook."""
    setter = getattr(patcher, "set_model_optimized_attention", None)
    if not callable(setter):
        return False
    if backend == "Kitchen":
        setter(comfy_attention.attention_comfy_kitchen_int8)
    else:
        setter(_cudnn_attention_override)
    return True


# The cached latent projection is much smaller than the per-head outputs, but
# a hard bound keeps a large batch from pinning an unbounded activation tensor.
MAX_LQ_PROJECTION_CACHE_BYTES = 256 * 1024 * 1024


class _LQProjectionCache:
    __slots__ = ("active", "key", "features", "hits")

    def __init__(self):
        self.active = False
        self.key = None
        self.features = None
        self.hits = 0

    def clear(self):
        self.key = None
        self.features = None
        self.hits = 0


def _find_lq_projection(patcher: object):
    """Return PiD's unpatched LQ projection module, or None for other models."""
    get_object = getattr(patcher, "get_model_object", None)
    if not callable(get_object):
        return None
    try:
        diffusion_model = get_object("diffusion_model")
    except AttributeError:
        return None
    projection = getattr(diffusion_model, "lq_proj", None)
    if projection is None:
        return None
    align = getattr(projection, "_align_latent_to_patch_grid", None)
    if getattr(align, "__func__", None) is not LQProjection2D._align_latent_to_patch_grid:
        return None
    return projection


def _projection_cache_key(lq_latent: torch.Tensor, target_pH: int, target_pW: int):
    """Describe a call without synchronizing GPU values to the host.

    The node's enclosing context is the batch identity: it resets for every
    sample_custom invocation and is enabled only for cfg=1. Shape/device/dtype
    metadata still distinguishes legitimate projection-shape changes.
    """
    device = lq_latent.device
    return (
        tuple(lq_latent.shape),
        lq_latent.dtype,
        device.type,
        device.index,
        int(target_pH),
        int(target_pW),
    )


def _make_lq_projection_align(cache: _LQProjectionCache):
    def align(self, lq_latent, pH, pW):
        if not cache.active or torch.is_grad_enabled():
            return LQProjection2D._align_latent_to_patch_grid(self, lq_latent, pH, pW)
        key = _projection_cache_key(lq_latent, pH, pW)
        if cache.features is not None and cache.key == key:
            cache.hits += 1
            return cache.features
        features = LQProjection2D._align_latent_to_patch_grid(self, lq_latent, pH, pW)
        cache.clear()
        if features.numel() * features.element_size() <= MAX_LQ_PROJECTION_CACHE_BYTES:
            cache.key = key
            cache.features = features
        return features
    return align


@contextmanager
def pid_lq_projection_cache(model, enabled: bool = True):
    """Enable one-batch LQ projection reuse on a ``pid_pixel_runtime`` clone."""
    cache = getattr(model, "_pid_lq_projection_cache", None)
    if cache is None:
        yield model
        return
    previous_active = cache.active
    cache.clear()
    cache.active = bool(enabled)
    try:
        yield model
    finally:
        cache.clear()
        cache.active = previous_active

# Balance per-patch MLP activation memory against kernel launch overhead.
DEFAULT_PIXEL_CHUNK_SIZE = 1024


def _check_supported_block(block: object) -> bool:
    """Accept only the released PiTBlock structure this patch reproduces."""
    if type(block) is not PiTBlock:
        return False
    required = (
        "pixel_dim",
        "attn_dim",
        "num_heads",
        "norm1",
        "norm2",
        "attn",
        "mlp",
        "compress_to_attn",
        "expand_from_attn",
        "adaLN_modulation_msa",
        "adaLN_modulation_mlp",
        "_fetch_pos",
    )
    if any(not hasattr(block, name) for name in required):
        return False
    forward = getattr(block, "forward", None)
    return getattr(forward, "__func__", None) is PiTBlock.forward


def _find_pixel_blocks(patcher: object) -> list[PiTBlock] | None:
    """Return PiD's complete pixel block list, or None for non-PiD models."""
    get_object = getattr(patcher, "get_model_object", None)
    if not callable(get_object):
        return None
    try:
        diffusion_model = get_object("diffusion_model")
    except AttributeError:
        return None
    # Plain PixelDiT has pixel_blocks too; PiD's lq_proj is the unambiguous marker.
    if getattr(diffusion_model, "lq_proj", None) is None:
        return None
    blocks = getattr(diffusion_model, "pixel_blocks", None)
    if not isinstance(blocks, (nn.ModuleList, list, tuple)) or not blocks:
        return None
    if not all(_check_supported_block(block) for block in blocks):
        return None
    return list(blocks)


def _pit_chunked_forward(
    self: PiTBlock,
    x: torch.Tensor,
    s_cond: torch.Tensor,
    image_height: int,
    image_width: int,
    patch_size: int,
    mask=None,
    transformer_options={},
    *,
    chunk_size: int,
):
    """PiTBlock.forward with bounded per-patch activation materialization.

    Attention remains globally vectorized; its temporaries are released before
    modulation, normalization, and the complete MLP run in bounded BL chunks.
    MLP patches are independent, so in-place residual updates cannot affect a
    neighboring patch.  This function is installed only on an inference-scoped
    ModelPatcher clone.
    """
    # Never alter training/autograd semantics if a caller uses the clone outside
    # the node's inference scope.
    if torch.is_grad_enabled():
        return PiTBlock.forward(
            self, x, s_cond, image_height, image_width, patch_size, mask=mask,
            transformer_options=transformer_options,
        )

    BL, P2, pixel_dim = x.shape
    Hs, Ws = image_height // patch_size, image_width // patch_size
    L = Hs * Ws
    B = BL // L
    if chunk_size <= 0 or BL == 0 or P2 != patch_size * patch_size or pixel_dim != self.pixel_dim:
        return PiTBlock.forward(
            self, x, s_cond, image_height, image_width, patch_size, mask=mask,
            transformer_options=transformer_options,
        )

    # Keep the globally coupled attention path vectorized, and release its
    # full-resolution temporaries before entering the independent MLP branch.
    msa_params = self.adaLN_modulation_msa(s_cond).view(BL, P2, 3 * self.pixel_dim)
    shift_msa, scale_msa, gate_msa = msa_params.chunk(3, dim=-1)
    x_norm = apply_adaln_(self.norm1(x), shift_msa, scale_msa)
    x_comp = self.compress_to_attn(x_norm.view(BL, P2 * self.pixel_dim)).view(B, L, self.attn_dim)
    del x_norm
    pos_comp = self._fetch_pos(
        Hs, Ws, x.device, x.dtype,
        **(transformer_options.get("rope_options") or {}),
    )
    attn_out = self.attn(x_comp, pos_comp, mask=mask, transformer_options=transformer_options)
    del x_comp, pos_comp
    attn_exp = self.expand_from_attn(attn_out.view(BL, self.attn_dim)).view(BL, P2, self.pixel_dim)
    x = torch.addcmul(x, gate_msa, attn_exp)
    del msa_params, shift_msa, scale_msa, gate_msa, attn_exp, attn_out

    # Every MLP operation, including its modulation and norm input, is bounded to
    # one BL chunk.  No full-resolution lq/pixel feature cache is retained.
    for start in range(0, BL, chunk_size):
        end = min(start + chunk_size, BL)
        mlp_params = self.adaLN_modulation_mlp(s_cond[start:end]).view(
            end - start, P2, 3 * self.pixel_dim
        )
        shift_mlp, scale_mlp, gate_mlp = mlp_params.chunk(3, dim=-1)
        gate_mlp = gate_mlp.contiguous()
        mlp_input = apply_adaln_(self.norm2(x[start:end]), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mlp_input)
        x[start:end].addcmul_(gate_mlp, mlp_out)
        del mlp_params, shift_mlp, scale_mlp, gate_mlp, mlp_input, mlp_out
        comfy.model_management.throw_exception_if_processing_interrupted()
    return x

def _make_chunked_forward(chunk_size: int):
    def forward(self, x, s_cond, image_height, image_width, patch_size, mask=None, transformer_options={}):
        return _pit_chunked_forward(
            self, x, s_cond, image_height, image_width, patch_size,
            mask=mask, transformer_options=transformer_options, chunk_size=chunk_size,
        )
    return forward


@contextmanager
def pid_pixel_runtime(
    model,
    pixel_chunk_size: int = DEFAULT_PIXEL_CHUNK_SIZE,
    attention_backend: str | None = None,
) -> Iterator[object]:
    """Yield a scoped PiD clone with chunking, LQ hooks, and attention override.

    ``pixel_chunk_size == 0`` disables only the internal MLP chunking.  When
    ``attention_backend`` is supplied, its transformer-options override still
    runs on a clone so backend selection remains node-local.  Passing no backend
    preserves the historical true no-op path used by direct callers.
    """
    chunk_size = int(pixel_chunk_size)
    backend = None
    if attention_backend is not None:
        backend = resolve_attention_backend(str(attention_backend))
    if chunk_size <= 0 and backend is None:
        yield model
        return

    blocks = _find_pixel_blocks(model)
    if blocks is None:
        yield model
        return

    clone = model.clone()
    clone_blocks = _find_pixel_blocks(clone)
    clone_projection = _find_lq_projection(clone)
    if clone_blocks is None or len(clone_blocks) != len(blocks):
        yield model
        return

    projection_patch_name = "diffusion_model.lq_proj._align_latent_to_patch_grid"
    if clone_projection is not None and projection_patch_name in getattr(clone, "object_patches", {}):
        yield model
        return

    for index, block in enumerate(clone_blocks):
        patch_name = f"diffusion_model.pixel_blocks.{index}.forward"
        # Composing an unknown pre-existing forward patch would change
        # semantics; leave this model untouched instead of replacing an owner.
        if patch_name in getattr(clone, "object_patches", {}):
            yield model
            return

    if backend is not None and not _install_attention_backend(clone, backend):
        yield model
        return

    cache = _LQProjectionCache() if clone_projection is not None else None
    if cache is not None:
        clone._pid_lq_projection_cache = cache
        clone.add_object_patch(
            projection_patch_name,
            MethodType(_make_lq_projection_align(cache), clone_projection),
        )
    if chunk_size > 0:
        for index, block in enumerate(clone_blocks):
            clone.add_object_patch(
                f"diffusion_model.pixel_blocks.{index}.forward",
                MethodType(_make_chunked_forward(chunk_size), block),
            )

    try:
        yield clone
    finally:
        # ModelPatcher's normal object-patch backup/restore path is the
        # authority; do not mutate shared model attributes directly.
        if cache is not None:
            cache.clear()
        clone.unpatch_model(unpatch_weights=False)
        if cache is not None:
            del clone._pid_lq_projection_cache
