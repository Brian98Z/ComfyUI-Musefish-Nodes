"""Musefish PiD batch video upscaling nodes."""

from .musefish_nodes import MusefishExtension, MusefishPiDBatchVideoUpscale, comfy_entrypoint

__all__ = ["comfy_entrypoint", "MusefishExtension", "MusefishPiDBatchVideoUpscale"]
