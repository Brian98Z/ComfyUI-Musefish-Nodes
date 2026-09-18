"""Musefish PiD video and UniverSR audio nodes."""

from .musefish_audio import MusefishUniverSRGeneralAudio, MusefishUniverSRModel, MusefishUniverSRSpeechAudio
from .musefish_nodes import (
    AutoBatchAntiflicker,
    AutoBatchImageSharpenFS,
    MusefishExtension,
    MusefishPiDBatchVideoUpscale,
    comfy_entrypoint,
)

WEB_DIRECTORY = "./web"

__all__ = [
    "WEB_DIRECTORY",
    "comfy_entrypoint",
    "MusefishExtension",
    "MusefishPiDBatchVideoUpscale",
    "MusefishUniverSRGeneralAudio",
    "MusefishUniverSRModel",
    "MusefishUniverSRSpeechAudio",
    "AutoBatchAntiflicker",
    "AutoBatchImageSharpenFS",
]
