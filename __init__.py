"""Musefish PiD video and UniverSR audio nodes."""

from . import _startup_update  # noqa: F401  (background yt-dlp self-update)
from .musefish_audio import MusefishUniverSRGeneralAudio, MusefishUniverSRModel, MusefishUniverSRSpeechAudio
from .musefish_dlss5 import MusefishDLSS5NeuralRender
from .musefish_dlss5_stream import MusefishDLSS5VideoStream
from .musefish_nodes import (
    AutoBatchAntiflicker,
    AutoBatchImageSharpenFS,
    MusefishExtension,
    MusefishPiDBatchVideoUpscale,
    comfy_entrypoint,
)
from .musefish_video_nodes import (
    MusefishVideoDownload,
    MusefishWeChatChannels,
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
    "MusefishDLSS5NeuralRender",
    "MusefishDLSS5VideoStream",
    "MusefishVideoDownload",
    "MusefishWeChatChannels",
]
