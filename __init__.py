"""Musefish PiD video and UniverSR audio nodes."""

from .musefish_audio import MusefishUniverSRGeneralAudio, MusefishUniverSRModel, MusefishUniverSRSpeechAudio
from .musefish_nodes import (
    AutoBatchAntiflicker,
    AutoBatchImageSharpenFS,
    MusefishExtension,
    MusefishPiDBatchVideoUpscale,
    comfy_entrypoint,
)

__all__ = [
    "comfy_entrypoint",
    "MusefishExtension",
    "MusefishPiDBatchVideoUpscale",
    "MusefishUniverSRGeneralAudio",
    "MusefishUniverSRModel",
    "MusefishUniverSRSpeechAudio",
    "AutoBatchAntiflicker",
    "AutoBatchImageSharpenFS",
]
