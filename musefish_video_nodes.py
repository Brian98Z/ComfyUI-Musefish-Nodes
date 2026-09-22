"""Musefish social-media video download nodes.

Two nodes cover the ``video-download`` skill's workflows:

- ``MusefishVideoDownload``     : URL -> file. Platform URLs run through
  yt-dlp; WC CDN URLs run through the wasm decrypt path and
  need the URL-bound ``decode_key``.
- ``MusefishWeChatChannels``    : one-shot WC pipeline (auto key/URL or
  manual key -> tokened URL -> download -> ftyp gate -> decrypt -> verify).

Both return the downloaded media as real ComfyUI types so downstream save
nodes connect directly: VIDEO (SaveVideo / AudioVideoCombine / VHS-style
consumers), IMAGE ([N,H,W,C] float frames, lazily decoded), AUDIO
(dict with waveform/sample_rate, lazily decoded) and a STRING report.
The heavy IMAGE/AUDIO decode only runs when a downstream node actually
reads those outputs.
"""

from __future__ import annotations

from typing_extensions import override

import comfy.utils
import json
import re
import time
import torch
from comfy_api.latest import ComfyExtension, Input, InputImpl, io

from .musefish_video_download import (
    DownloadError,
    ProgressReporter,
    _DOUYIN_COOKIES,
    _DOUYIN_SHORT_RE,
    _PLATFORM_LABELS,
    _PLATFORMS,
    _stream_to_file,
    _unique_path,
    assert_platform_matches,
    detect_platform,
    complete_wechat_url,
    download_wechat_encrypted,
    download_with_ytdlp,
    looks_like_douyin_url,
    looks_like_wechat_media,
    resolve_douyin_share,
    resolve_output_dir,
    resolve_share_link,
    scan_wechat_decode_keys,
    verify_download,
)

_QUALITIES = ["best", "1080p", "720p", "480p"]
_SEGMENT_CHOICES = ["1", "2", "4", "8", "16"]


def _resolve_dir():
    """Downloads always land in ComfyUI's output folder (under musefish/)."""
    try:
        import folder_paths

        directory = _path(folder_paths.get_output_directory()) / "musefish"
    except Exception:
        directory = resolve_output_dir("")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _path(value) -> "Path":
    from pathlib import Path

    return Path(value)


def _summarize(path, info: dict, logs: list[str]) -> str:
    lines = [f"saved: {path}", *[f"· {line}" for line in logs]]
    keys = ("duration", "width", "height", "video_codec", "audio_codec", "size")
    facts = {key: info[key] for key in keys if info.get(key) is not None}
    if facts:
        lines.append("verified: " + ", ".join(f"{k}={facts[k]}" for k in facts))
    return "\n".join(lines)


# Paste parsing: users paste whatever they happen to have. Recognized, in
# order of specificity: a Channels API JSON response (carries url + key), a
# tokened/plain CDN URL (key must come from memory scan), a share message
# (only identifies the video: scan resolves it), or anything at all (scan).
_DECODE_KEY_JSON_KEYS = ("decode_key", "dec_key", "decrypt_key")
_CDN_URL_RE = re.compile(r"https?://[^\s\"'<>]*finder\.video\.qq\.com/251/20302/stodownload[^\s\"'<>]*", re.IGNORECASE)
_SHARE_LINK_RE = re.compile(r"https?://[^\s\"'<>【】（）()，,。;；]+", re.IGNORECASE)


def _find_first(obj, keys):
    """Depth-first search for the first dict key in ``keys``.

    Values may be strings or numbers (the Channels API returns decode_key
    as a bare integer); anything found is stringified.
    """
    from collections import deque

    queue = deque([obj])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            for key in keys:
                value = node.get(key)
                if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                    return str(value).strip()
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


def _auto_extract(paste: str, logs: list[str]) -> tuple[str, str]:
    """Extract (tokened CDN URL, decode_key) from whatever the user pasted.

    Returns empty strings for pieces the paste does not carry — the caller
    memory-scans the WC desktop client for those. Never raises on garbage input:
    unknown text just means "scan for everything".
    """
    if not paste:
        return "", ""

    # 1. Channels API JSON response: richest source (url + decode_key).
    stripped = paste.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            key = _find_first(data, _DECODE_KEY_JSON_KEYS)
            url = None

            def _walk_media(node):
                if isinstance(node, dict):
                    if "url" in node and isinstance(node.get("url"), str):
                        yield node["url"], node.get("decode_key")
                    for value in node.values():
                        yield from _walk_media(value)
                elif isinstance(node, list):
                    for value in node:
                        yield from _walk_media(value)

            for media_url, media_key in _walk_media(data):
                if not media_url:
                    continue
                # Only the video channel (20302). 20304 is the cover/thumb
                # stream whose keys do NOT decrypt the video — the upstream
                # docs call this exact mismatch out as a trap. Non-HTTP
                # URLs (local file paths in tests) pass through untouched.
                is_video_url = (
                    "/20302/stodownload" in media_url
                    or not media_url.startswith("http")
                )
                if is_video_url:
                    url = media_url
                    key = key or media_key
                    break
            if url or key:
                if url:
                    logs.append("parsed Channels API JSON: found video URL")
                if key:
                    logs.append("parsed Channels API JSON: found decode_key")
                return url or "", key or ""
            logs.append("paste looked like JSON but carried no Channels fields; falling back to scan")

    # 2. Direct CDN URL (with or without token).
    cdn = _CDN_URL_RE.search(paste)
    if cdn:
        logs.append("found Channels CDN URL in paste")
        return cdn.group(0).rstrip(".,;)"), ""

    # 3. Share link / message text: identifies the video; scan supplies
    #    key + URL. Do not treat arbitrary links as resolvable.
    share = _SHARE_LINK_RE.search(paste)
    if share:
        logs.append("paste carries a share link; matching it against WC client memory")
    else:
        logs.append("paste carries no link; matching text against WC client memory")
    return "", ""


class MusefishVideoDownload(io.ComfyNode):
    """Download a social-media video URL and verify it as playable MP4."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MusefishVideoDownload",
            display_name="Musefish Video Download",
            search_aliases=["download", "视频下载", "yt-dlp", "视频号", "bilibili", "douyin"],
            category="Musefish/Video",
            description=(
                "Download a social-media video to disk. Platform URLs "
                "(YouTube, Bilibili, X/Twitter, Douyin, Xiaohongshu) use the "
                "in-process yt-dlp extractor; WC CDN URLs "
                "(finder.video.qq.com) take the automatic decrypt path — "
                "the decode_key is recovered on its own from the running "
                "desktop client (play the video there first and keep it "
                "running; for Channels links the Musefish WC Video Channels "
                "node is the more complete route). Every file is verified "
                "(MP4 header + ffprobe) before the node returns."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    multiline=True,
                    placeholder="Paste a video CDN URL / share link / share text / API JSON (optional)",
                    tooltip="Optional. Media CDN URL, share link, share message text, or an API JSON "
                            "response. Anything that identifies the video; the node auto-extracts URL/key and "
                            "memory-scans the desktop client for missing pieces. "
                            "Autoplay-safety: copy link A first, open A in WeChat and let it play a few "
                            "seconds, then copy any other link B, and paste A here before running. "
                            "Copying B last pins your target video as the most recently opened link, "
                            "immune to the client's autoplay.",
                ),
                io.Combo.Input(
                    "platform",
                    options=[_PLATFORM_LABELS[name] for name in _PLATFORMS],
                    default=_PLATFORM_LABELS["auto"],
                    tooltip=(
                        "Source platform. Auto detects from the URL; a "
                        "specific choice verifies the URL actually belongs "
                        "to that platform and errors on mismatch (WeChat "
                        "Channels URLs decrypt via the running desktop "
                        "client; all others go through yt-dlp)."
                    ),
                ),
                io.Combo.Input("quality", options=_QUALITIES, default="best",
                               tooltip="yt-dlp format selection; ignored on direct CDN URLs."),
                io.Combo.Input(
                    "chunks",
                    options=_SEGMENT_CHOICES,
                    default="4",
                    tooltip=(
                        "Long-video speed-up: split the download into N "
                        "concurrent ranged segments (direct CDN URLs like "
                        "Douyin) or fetch N fragments concurrently "
                        "(YouTube/Bilibili via yt-dlp). Higher = faster "
                        "when the CDN allows it; 1 = plain sequential."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video", tooltip="Downloaded media as VIDEO; connect to SaveVideo or any VIDEO consumer."),
                io.Float.Output("fps", tooltip="Frame rate of the downloaded video, in frames per second."),
            ],
        )

    @classmethod
    def validate_inputs(cls, quality: str = "", platform: str = "",
                        max_scan_seconds=None, **kwargs) -> bool | str:
        """Empty widget values from stale workflows fall back to defaults.

        Listing these inputs here exempts them from the framework's strict
        INT conversion / combo membership checks, which would otherwise
        reject the whole prompt on legacy workflow JSON that stored ''.
        """
        return True

    @classmethod
    def execute(cls, url: str, platform: str, quality: str, chunks: str = "4") -> io.NodeOutput:
        url = (url or "").strip()
        # Users often paste "【标题】 https://... &vd_source=..." — the
        # share text wraps the real link. Extract the first URL; the rest
        # is decoration yt-dlp would choke on.
        embedded = re.search(r"https?://[^\s\"'<>【】（）()，,]+", url)
        if embedded:
            url = embedded.group(0).rstrip("。．.，,;;、】»>\"'")
        if not url:
            raise DownloadError("url is empty: paste a video page or CDN URL")
        # Stale workflows may carry '' for combo widgets; map to the label
        # they saved (pre-platform workflows carry no platform at all).
        if platform not in _PLATFORM_LABELS.values():
            platform = _PLATFORM_LABELS["auto"]
        platform_key = next(
            (name for name, label in _PLATFORM_LABELS.items() if label == platform),
            "auto",
        )
        # WeChat Channels URLs never download here: the decrypt pipeline
        # (decode_key from the desktop client) lives in the dedicated
        # MusefishWeChatChannels node. Redirect instead of failing obscurely.
        if looks_like_wechat_media(url):
            raise DownloadError(
                "WeChat Channels URL detected: download it with the "
                "Musefish WC Video Channels node (it resolves decode_key "
                "and decryption automatically). URL: " + url[:120]
            )
        assert_platform_matches(url, platform_key)
        if quality not in _QUALITIES:
            quality = "best"
        try:
            chunks = max(1, min(16, int(str(chunks).strip() or 4)))
        except (TypeError, ValueError):
            chunks = 4
        # Unknown platforms gain nothing from ranged parallelism: collapse
        # to 1 unless the user explicitly picked a larger count.
        if detect_platform(url) is None and not looks_like_douyin_url(url):
            chunks = max(1, chunks if chunks != 4 else 1)
        directory = _resolve_dir()
        logs: list[str] = [f"platform: {platform}"]
        progress_bar = comfy.utils.ProgressBar(100)
        reporter = ProgressReporter(progress_bar)

        # Douyin share texts carry short links whose yt-dlp extractor is
        # signature-blocked. The resolver picks the real URL out of the
        # paste and streams the unencrypted CDN MP4 directly.
        if looks_like_douyin_url(url) or _DOUYIN_SHORT_RE.search(url or ""):
            resolved = resolve_douyin_share(url, max_seconds=120.0)
            if resolved:
                direct, title = resolved
                logs = [f"douyin: resolved {title[:40]}"]
                target = _unique_path(directory, "musefish_douyin", ".mp4")
                _stream_to_file(direct, target, _DOUYIN_COOKIES, reporter, parallel=chunks)
                media = verify_download(target, logs)
                media.update({"title": title})
                bundle = InputImpl.VideoFromFile(str(target))
                return io.NodeOutput(bundle, float(media.get("fps") or 0.0))
            raise DownloadError(
                "douyin link detected but resolution failed (cookies expired "
                "or video unavailable). Refresh _dy_cookies.txt and retry."
            )

        path, info = download_with_ytdlp(
            url, directory, reporter, quality, logs,
            parallel_fragments=chunks,
        )

        media = verify_download(path, logs)
        media.update({key: value for key, value in info.items() if key in ("title",)})
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(path)), float(media.get("fps") or 0.0)
        )


class MusefishWeChatChannels(io.ComfyNode):
    """One-shot encrypted-channels pipeline: key -> URL -> download -> verify -> decrypt.

    A single linear flow with no mode switch. Each stage feeds the next and
    the ftyp gate stops execution before any wasted work: the decode_key is
    verified against the ciphertext header before the full download, and the
    decrypted file is ffprobe-verified before the node returns.

    Everything is automatic: paste whatever identifies the video (CDN URL,
    share link, share text, or an API JSON response) — or nothing at all —
    and the node extracts the URL/key it can, recovering missing pieces
    from the running messaging client's memory (the client must be running
    locally and the video played there; the key only exists in its memory).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MusefishWeChatChannels",
            display_name="Musefish WC Video Channels",
            search_aliases=["视频号", "视频号下载", "视频号解密", "decode_key", "内存扫描", "key scan", "wx decrypt", "channels"],
            category="Musefish/Video",
            description=(
                "Download and decrypt one encrypted Channels video in a "
                "single pass — fully automatic, no manual keys: paste "
                "anything you have into link_or_url (CDN URL, share link, "
                "share message text, or an API JSON response) and the node "
                "extracts what it needs, recovering the decode_key from the "
                "running desktop client's memory when the paste doesn't "
                "carry one. Then: download ciphertext, verify the key "
                "against the ftyp header, decrypt (only the first 128 KiB "
                "are Isaac64-encrypted) and ffprobe-verify. The desktop "
                "client must be running locally and the video played there "
                "at least once: the key only exists in its memory. "
                "Verification failures abort before the next stage. "
                "IMPORTANT (autoplay safety): the client auto-plays the "
                "NEXT video after one finishes, so the newest memory record "
                "is not always the video you linked. Reliable sequence: "
                "(1) copy share link A, (2) open link A and let it play a "
                "few seconds, (3) copy share link B, (4) paste link A and "
                "run. Copying B right before running pins your target as "
                "the last-opened link, immune to the client's autoplay."
            ),
            inputs=[
                io.String.Input(
                    "link_or_url", multiline=True, default="",
                    placeholder="Paste a Channels share link",
                    tooltip="Optional. Media CDN URL, share link, share message text, or an API JSON "
                            "response. Anything that identifies the video; the node auto-extracts URL/key and "
                            "memory-scans the desktop client for missing pieces. "
                            "Autoplay-safety: copy link A first, open A in WeChat and let it play a few "
                            "seconds, then copy any other link B, and paste A here before running. "
                            "Copying B last pins your target video as the most recently opened link, "
                            "immune to the client's autoplay.",
                ),
            ],
            outputs=[
                io.Video.Output("video", tooltip="Decrypted media as VIDEO; connect to SaveVideo or any VIDEO consumer."),
                io.Float.Output("fps", tooltip="Frame rate of the decrypted video, in frames per second."),
            ],
        )

    @classmethod
    def validate_inputs(cls, link_or_url: str = "", **kwargs) -> bool | str:
        """Empty widget values from stale workflows fall back to defaults
        (same rationale as MusefishVideoDownload.validate_inputs)."""
        return True

    @classmethod
    def execute(
        cls,
        link_or_url: str,
    ) -> io.NodeOutput:
        logs: list[str] = ["stage 1: resolve decode_key + URL automatically"]
        progress_bar = comfy.utils.ProgressBar(100)
        reporter = ProgressReporter(progress_bar)
        reporter.fraction(0.02)
        max_scan_seconds = 120

        paste = (link_or_url or "").strip().strip('"')
        url, decode_key = _auto_extract(paste, logs)
        scanned = False

        # A share link (weixin.qq.com/sph/<id>) binds to a memory record as
        # a HINT: the client's memory is volatile (entries compact, tokens
        # rotate, one encfilekey can be recycled by a later playback), so
        # the bound record is tried FIRST but never exclusively — the full
        # newest-first candidate list always remains as fallback.
        bound_candidate = None
        if paste and not url and not decode_key:
            try:
                bound_candidate = resolve_share_link(paste, max_seconds=float(max_scan_seconds) / 2)
            except Exception:
                bound_candidate = None
            if bound_candidate is not None:
                logs.append(
                    "share link hint bound to memory record: "
                    f"decode_key={bound_candidate['decode_key']} (encfilekey "
                    f"{bound_candidate['url_head'].split('encfilekey=', 1)[-1][:16]}...); "
                    "other fresh records remain as fallback"
                )
                decode_key = bound_candidate["decode_key"]
                scanned = True
            elif _SHARE_LINK_RE.search(paste):
                logs.append(
                    "share link not yet bound in client memory; using the "
                    "newest record. Make sure the linked video was the LAST "
                    "one played in the WC desktop client before running."
                )

        def _scan():
            candidates = scan_wechat_decode_keys(max_seconds=float(max_scan_seconds))
            if not candidates:
                raise DownloadError(
                    "no WC video records found in the client's memory. "
                    "Play the target video in the WC desktop client for a few "
                    "seconds, keep the client running, then run again."
                )
            return candidates

        # ---- stage 1a: decode_key ------------------------------------
        if not decode_key:
            logs.append("no decode_key in paste; scanning WC client memory")
            reporter.fraction(0.08)
            candidates = _scan()
            scanned = True
            chosen = candidates[0]
            decode_key = chosen["decode_key"]
            age_min = (time.time() * 1000 - int(chosen["timestamp"])) / 60000
            logs.append(
                f"scanned {len(candidates)} record(s); trying newest first: "
                f"decode_key={decode_key} (played {age_min:.0f} min ago)"
            )
            for position, candidate in enumerate(candidates[:5], start=1):
                if position != 1:
                    logs.append(f"fallback decode_key[{position}]: {candidate['decode_key']}")
        elif bound_candidate is not None:
            # hint first; every other record stays in the fallback list
            candidates = [bound_candidate]
        else:
            candidates = []

        # ---- stage 1b: tokened URL ------------------------------------
        if not url:
            logs.append("no tokened URL in paste; re-scanning memory for it")
            reporter.fraction(0.18)
            if not candidates:
                candidates = _scan()
                scanned = True
            elif scanned and bound_candidate is not None:
                # enrich the hint with all fresh records as fallbacks
                try:
                    fresh = _scan()
                except DownloadError:
                    fresh = []
                seen_keys = {c["decode_key"] for c in candidates}
                candidates.extend(c for c in fresh if c["decode_key"] not in seen_keys)
            match = next((c for c in candidates if c["decode_key"] == decode_key), candidates[0])
            url = complete_wechat_url(match, max_seconds=float(max_scan_seconds))
            logs.append(f"recovered tokened URL ({len(url)} chars)")
        else:
            logs.append("stage 2: using URL from paste")
        logs.append(
            "note: the URL is a temporary (~48h) access credential stored in "
            "workflow JSON when saved — do not share workflows containing it"
        )
        reporter.fraction(0.25)

        # ---- stage 3: download + key verification + decrypt --------------------
        # download_wechat_encrypted probes 256 KiB, gates on ftyp (bad key ->
        # abort before the full download), then downloads and decrypts.
        # Memory scan order is newest-first; a newer record's token may be
        # stale while an older one still works, so on ftyp mismatch we skip
        # to the next candidate instead of failing the whole node.
        directory = _resolve_dir()

        attempts: list[tuple[str, str]] = [(url, decode_key)]
        if scanned and candidates:
            # alternates from the same scan: their own key + recovered URL
            for position, candidate in enumerate(candidates, start=1):
                if candidate["decode_key"] == decode_key:
                    continue
                try:
                    alt_url = complete_wechat_url(candidate, max_seconds=float(max_scan_seconds))
                    attempts.append((alt_url, candidate["decode_key"]))
                except DownloadError:
                    continue

        last_error: Exception | None = None
        path = info = None
        for attempt_index, (try_url, try_key) in enumerate(attempts, start=1):
            try:
                path, info = download_wechat_encrypted(try_url, directory, try_key, reporter, logs)
                break
            except DownloadError as error:
                last_error = error
                if "ftyp mismatch" not in str(error):
                    raise
                logs.append(f"candidate {attempt_index}: key/URL mismatch, trying next")
                path = info = None
        if path is None:
            raise DownloadError(
                f"all {len(attempts)} candidate(s) failed key verification. "
                "The client's tokens are stale: replay the target video for a "
                "few seconds in the WC desktop client, then run again."
                f" (last error: {last_error})"
            )
        reporter.fraction(0.9)

        # ---- stage 4: output verification --------------------------------------
        media = verify_download(path, logs)
        media["title"] = info["title"]
        logs.append("all stages passed: key verified, decrypted, ffprobe OK")
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(path)), float(media.get("fps") or 0.0)
        )


class MusefishVideoDownloadExtension(io.ComfyNode):
    """Marker kept for symmetry with the package's extension list."""
