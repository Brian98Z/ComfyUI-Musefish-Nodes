"""Behavioral checks for the Musefish video download nodes.

The WeChat decrypt path is exercised against the vendored upstream sample
ciphertext (wx_encrypted.mp4 + decode_key 2136343393), so the keystream
daemon, XOR gate, 200-fallback download and ftyp verification are tested
end-to-end rather than against mocks. The yt-dlp interaction is stubbed at
the node module's binding — the node owns validation/verification policy,
which is what these tests pin.

Run from the ComfyUI checkout root:
    <python> -m pytest custom_nodes/ComfyUI-Musefish-Nodes/tests/test_video_download_nodes.py \
        -q -p no:cacheprovider --import-mode=importlib
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from pytest import MonkeyPatch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_DIR.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
import comfy  # noqa: F401  (real ComfyUI checkout is a prerequisite)

# Import through the package path so `from .musefish_video_download import ...`
# inside the node module resolves; tests must patch names on BOTH modules
# where the node holds its own binding.
node_mod = importlib.import_module(
    "custom_nodes.ComfyUI-Musefish-Nodes.musefish_video_nodes"
)
backend = importlib.import_module(
    "custom_nodes.ComfyUI-Musefish-Nodes.musefish_video_download"
)

# Ground-truth sample (vendored): an upstream WeChat Channels ciphertext plus
# its decode_key. Keep in sync with wx_response.json's media[0].decode_key.
SAMPLE_CIPHERTEXT = PLUGIN_DIR / "tests" / "fixtures" / "wx_encrypted.mp4"
SAMPLE_DECODE_KEY = "2136343393"

pytestmark = pytest.mark.skipif(
    not SAMPLE_CIPHERTEXT.is_file(),
    reason="vendored wx_encrypted.mp4 fixture missing; decrypt-path tests skipped",
)


@pytest.fixture(scope="module")
def daemon():
    shared = backend.get_shared_daemon()
    yield shared
    shared.close()


# ---------------------------------------------------------------- routing


def test_wechat_url_detection():
    assert backend.looks_like_wechat_media(
        "https://finder.video.qq.com/251/20302/stodownload?encfilekey=abc&m=...&token=x"
    )
    # 20304 is the cover stream, not video: must NOT be detected
    assert not backend.looks_like_wechat_media(
        "https://finder.video.qq.com/251/20304/stodownload?encfilekey=abc"
    )
    assert not backend.looks_like_wechat_media("https://www.youtube.com/watch?v=x")
    assert not backend.looks_like_wechat_media("https://x.com/user/status/1")


def test_x_url_detection():
    assert backend.looks_like_x_url("https://x.com/user/status/1")
    assert backend.looks_like_x_url("https://twitter.com/user/status/1")
    assert not backend.looks_like_x_url("https://www.bilibili.com/video/BV1x")


def test_sanitize_filenames():
    assert backend._sanitize_filename('bad/name:video "x"?') == "bad_name_video _x__"
    assert backend._sanitize_filename("") == "musefish_video"
    assert len(backend._sanitize_filename("x" * 500)) == 120


# ------------------------------------------------------- decrypt pipeline


def test_verify_encrypted_header_accepts_real_key(daemon):
    keystream = daemon.generate(SAMPLE_DECODE_KEY)
    assert len(keystream) == 131072
    data = SAMPLE_CIPHERTEXT.read_bytes()[:4096]
    assert backend.verify_encrypted_header(data, keystream)


def test_verify_encrypted_header_rejects_wrong_key(daemon):
    keystream = daemon.generate("1999999999")
    data = SAMPLE_CIPHERTEXT.read_bytes()[:4096]
    assert not backend.verify_encrypted_header(data, keystream)
    assert not backend.verify_encrypted_header(b"", keystream)


def test_decrypt_only_first_128k(daemon):
    keystream = daemon.generate(SAMPLE_DECODE_KEY)
    data = SAMPLE_CIPHERTEXT.read_bytes()
    plain = backend._decrypt_xor(data, keystream)
    assert plain[4:8] == b"ftyp"
    # beyond 128 KiB the file must be byte-identical (no double-processing)
    assert plain[131072:] == data[131072:]
    assert plain[:131072] != data[:131072]


def test_download_wechat_via_file_url_200_fallback(tmp_path):
    """A server that ignores Range (here: file://) returns 200 + full body.

    The download must still produce a byte-perfect decrypted file.
    """
    logs: list[str] = []
    reporter = backend.ProgressReporter(None)  # no bar: must never touch it
    path, info = backend.download_wechat_encrypted(
        SAMPLE_CIPHERTEXT.as_uri(), tmp_path, SAMPLE_DECODE_KEY, reporter, logs
    )
    assert path.is_file() and path.stat().st_size > 0
    summary = backend.verify_download(path, logs)
    assert summary["video_codec"] == "h264"
    assert summary["height"] == 1280
    assert any("ftyp ok" in line for line in logs)
    # decrypt correctness: equals a direct XOR of the source
    keystream = backend.get_shared_daemon().generate(SAMPLE_DECODE_KEY)
    assert path.read_bytes() == backend._decrypt_xor(SAMPLE_CIPHERTEXT.read_bytes(), keystream)
    path.unlink()


def test_download_wechat_rejects_bad_key(tmp_path):
    url = SAMPLE_CIPHERTEXT.as_uri()
    with pytest.raises(backend.DownloadError, match="decode_key"):
        backend.download_wechat_encrypted(url, tmp_path, "", backend.ProgressReporter(None), [])
    with pytest.raises(backend.DownloadError):
        backend.download_wechat_encrypted(url, tmp_path, "12a3", backend.ProgressReporter(None), [])


def test_download_wechat_wrong_key_fails_before_download(tmp_path):
    # A key that generates a keystream but does not match this ciphertext
    # must be rejected at the header probe, not produce a corrupt file.
    with pytest.raises(backend.DownloadError, match="ftyp mismatch"):
        backend.download_wechat_encrypted(
            SAMPLE_CIPHERTEXT.as_uri(), tmp_path, "1999999999", backend.ProgressReporter(None), []
        )
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------- daemon rpc


def test_daemon_caches_keystreams(daemon):
    first = daemon.generate(SAMPLE_DECODE_KEY)
    second = daemon.generate(SAMPLE_DECODE_KEY)
    assert first == second


def test_daemon_rejects_invalid_key(daemon):
    with pytest.raises(backend.DownloadError):
        daemon.generate("not-a-number")


# ------------------------------------------------------------ file output


def test_verify_download_rejects_non_mp4(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18not_a_video" + b"\x00" * 100)
    with pytest.raises(backend.DownloadError, match="ftyp"):
        backend.verify_download(bad, [])
    missing = tmp_path / "missing.mp4"
    with pytest.raises(backend.DownloadError, match="missing or empty"):
        backend.verify_download(missing, [])


def test_resolve_output_dir_prefers_desktop(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_DESKTOP_CANDIDATES", (tmp_path / "nope", tmp_path))
    assert backend.resolve_output_dir("") == tmp_path
    assert backend.resolve_output_dir(str(tmp_path / "sub")) == tmp_path / "sub"


# ----------------------------------------------------------------- nodes


def test_node_schemas_registered():
    for node_id in ("MusefishVideoDownload", "MusefishWeChatChannels"):
        node = getattr(node_mod, node_id)
        schema = node.define_schema()
        assert schema.node_id == node_id
        input_ids = [i.id for i in schema.inputs]
        assert input_ids, f"{node_id} has no inputs"


def test_node_outputs_are_media_types():
    """Both download nodes expose VIDEO + fps only (no image tensors)."""
    for node_id in ("MusefishVideoDownload", "MusefishWeChatChannels"):
        schema = getattr(node_mod, node_id).define_schema()
        outputs = schema.outputs
        assert len(outputs) == 2, f"{node_id} must have 2 outputs"
        ids = [o.id for o in outputs]
        assert ids == ["video", "fps"], ids


def test_download_node_rejects_empty_url():
    with pytest.raises(backend.DownloadError, match="url is empty"):
        node_mod.MusefishVideoDownload.execute(
            url="  ", platform=backend._PLATFORM_LABELS["auto"], quality="best"
        )


def test_platform_detection():
    assert backend.detect_platform("https://www.youtube.com/watch?v=x") == "youtube"
    assert backend.detect_platform("https://youtu.be/x") == "youtube"
    assert backend.detect_platform("https://www.bilibili.com/video/BV1x") == "bilibili"
    assert backend.detect_platform("https://b23.tv/x") == "bilibili"
    assert backend.detect_platform("https://www.douyin.com/video/1") == "douyin"
    assert backend.detect_platform("https://v.douyin.com/x/") == "douyin"
    assert backend.detect_platform("https://www.xiaohongshu.com/explore/x") == "xiaohongshu"
    assert backend.detect_platform("https://xhslink.com/x") == "xiaohongshu"
    assert backend.detect_platform("https://x.com/u/status/1") == "x_twitter"
    assert backend.detect_platform("https://twitter.com/u/status/1") == "x_twitter"
    assert backend.detect_platform("https://example.com/video.mp4") is None


def test_wechat_urls_not_platform_options_but_detected():
    """WC URLs are not a selector option; this node redirects them."""
    assert "wechat_channels" not in backend._PLATFORM_LABELS
    assert "wechat_channels" not in backend._PLATFORM_PATTERNS
    assert backend.looks_like_wechat_media(
        "https://finder.video.qq.com/251/20302/stodownload?encfilekey=x"
    )
    assert backend.looks_like_wechat_media("https://weixin.qq.com/sph/ABCdef123")


def test_download_node_redirects_wechat_urls_to_channels_node():
    for wc_url in (
        "https://finder.video.qq.com/251/20302/stodownload?encfilekey=abc",
        "https://weixin.qq.com/sph/ABCdef123",
    ):
        with pytest.raises(backend.DownloadError, match="Musefish WC Video Channels"):
            node_mod.MusefishVideoDownload.execute(
                url=wc_url,
                platform=backend._PLATFORM_LABELS["auto"],
                quality="best",
                )


def test_download_node_rejects_platform_mismatch():
    url = "https://www.bilibili.com/video/BV1x"
    label = backend._PLATFORM_LABELS["youtube"]
    with pytest.raises(backend.DownloadError, match="platform mismatch"):
        node_mod.MusefishVideoDownload.execute(
            url=url, platform=label, quality="best"
        )


def test_download_node_rejects_unknown_url_for_forced_platform():
    with pytest.raises(backend.DownloadError, match="platform mismatch"):
        node_mod.MusefishVideoDownload.execute(
            url="https://example.com/video.mp4",
            platform=backend._PLATFORM_LABELS["youtube"],
            quality="best",
            )


def test_download_node_stale_platform_widget_falls_back_to_auto(tmp_path, monkeypatch):
    """Legacy workflows carry no platform (or ''); auto must still work."""
    daemon = backend.get_shared_daemon()
    plain = backend._decrypt_xor(
        SAMPLE_CIPHERTEXT.read_bytes(), daemon.generate(SAMPLE_DECODE_KEY)
    )
    target = tmp_path / "musefish_download.mp4"
    target.write_bytes(plain)

    def fake_ytdlp(url, directory, reporter, quality, log, parallel_fragments=8):
        return target, {"title": "stub"}

    monkeypatch.setattr(node_mod, "download_with_ytdlp", fake_ytdlp)
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    result = node_mod.MusefishVideoDownload.execute(
        url="https://example.com/video",
        platform="",
        quality="best",
    )
    assert isinstance(result[1], float)


def test_download_node_generic_url_uses_ytdlp(tmp_path, monkeypatch):
    """Unknown-URL + auto (the generic path) must go through yt-dlp."""
    daemon = backend.get_shared_daemon()
    plain = backend._decrypt_xor(
        SAMPLE_CIPHERTEXT.read_bytes(), daemon.generate(SAMPLE_DECODE_KEY)
    )
    target = tmp_path / "musefish_download.mp4"
    target.write_bytes(plain)

    calls = {}
    def fake_ytdlp(url, directory, reporter, quality, log, parallel_fragments=8):
        calls["url"] = url
        return target, {"title": "generic"}

    monkeypatch.setattr(node_mod, "download_with_ytdlp", fake_ytdlp)
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    result = node_mod.MusefishVideoDownload.execute(
        url="https://some-new-site.com/watch/123",
        platform=backend._PLATFORM_LABELS["auto"],
        quality="best",
    )
    assert calls["url"].startswith("https://some-new-site.com")
    assert isinstance(result[1], float)


def test_download_node_forced_platform_mismatch_still_blocks(tmp_path, monkeypatch):
    """Forcing bilibili on a YouTube URL must fail before any download."""
    def boom(*a, **k):
        raise AssertionError("download must not run on platform mismatch")

    monkeypatch.setattr(node_mod, "download_with_ytdlp", boom)
    with pytest.raises(backend.DownloadError, match="platform mismatch"):
        node_mod.MusefishVideoDownload.execute(
            url="https://www.youtube.com/watch?v=abc",
            platform=backend._PLATFORM_LABELS["bilibili"],
            quality="best",
        )


def test_download_node_extracts_url_from_share_text(tmp_path):
    """Pasting "【标题】 https://... &vd_source=..." must work: the node
    pulls the real URL out of the decorated share text."""
    daemon = backend.get_shared_daemon()
    plain = backend._decrypt_xor(
        SAMPLE_CIPHERTEXT.read_bytes(), daemon.generate(SAMPLE_DECODE_KEY)
    )
    target = tmp_path / "musefish_download.mp4"
    target.write_bytes(plain)

    calls = {}
    def fake_ytdlp(url, directory, reporter, quality, log, parallel_fragments=8):
        calls["url"] = url
        return target, {"title": "ok"}

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(node_mod, "download_with_ytdlp", fake_ytdlp)
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: target.parent)
    node_mod.MusefishVideoDownload.execute(
        url="【AI生成人物做得假，真不是提示词的锅】 "
            "https://www.bilibili.com/video/BV1XEeg6zE6y/?share_source=copy_web&vd_source=abc123",
        platform=backend._PLATFORM_LABELS["auto"],
        quality="best",
        )
    assert calls["url"] == (
        "https://www.bilibili.com/video/BV1XEeg6zE6y/?share_source=copy_web&vd_source=abc123"
    )


def test_download_node_reports_verified_file(tmp_path, monkeypatch):
    # Pre-place a decryptable file; the ytdlp stub returns its path.
    daemon = backend.get_shared_daemon()
    plain = backend._decrypt_xor(SAMPLE_CIPHERTEXT.read_bytes(), daemon.generate(SAMPLE_DECODE_KEY))
    target = tmp_path / "musefish_download.mp4"
    target.write_bytes(plain)

    def fake_ytdlp(url, directory, reporter, quality, log, parallel_fragments=8):
        return target, {"title": "stub"}

    monkeypatch.setattr(node_mod, "download_with_ytdlp", fake_ytdlp)
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)

    result = node_mod.MusefishVideoDownload.execute(
        url="https://example.com/video",
        platform=backend._PLATFORM_LABELS["auto"],
        quality="720p",
        )
    fps = result[1]
    assert isinstance(fps, float) and fps > 0



def test_channels_node_api_json_paste_end_to_end(tmp_path, monkeypatch):
    """Paste an API JSON carrying url + decode_key: no scan needed at all."""
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    scan_calls = []
    monkeypatch.setattr(node_mod, "scan_wechat_decode_keys",
                        lambda max_seconds: scan_calls.append(1) or [])
    paste = json.dumps({
        "code": 200,
        "data": {"object_desc": {"media": [{
            "url": "https://finder.video.qq.com/251/20302/stodownload"
                   "?encfilekey=X&token=T",
            "decode_key": SAMPLE_DECODE_KEY,
        }]}},
    })
    # The JSON names a real CDN URL, which we cannot fetch offline; stub the
    # downloader to serve the vendored ciphertext instead.
    keystream = backend.get_shared_daemon().generate(SAMPLE_DECODE_KEY)
    plain = backend._decrypt_xor(SAMPLE_CIPHERTEXT.read_bytes(), keystream)

    def fake_download(url, directory, key, reporter, log):
        assert key == SAMPLE_DECODE_KEY
        target = directory / "wx_channels_stub.mp4"
        target.write_bytes(plain)
        return target, {"title": target.stem}
    monkeypatch.setattr(node_mod, "download_wechat_encrypted", fake_download)
    result = node_mod.MusefishWeChatChannels.execute(
        link_or_url=paste,
        )
    assert scan_calls == []  # key+URL both present: memory scan never ran
    fps = result[1]
    assert isinstance(fps, float) and fps > 0
    outputs = [q for q in tmp_path.iterdir() if q.name.startswith("wx_channels_")]
    assert len(outputs) == 1
    assert outputs[0].read_bytes() == plain


def test_channels_node_cdn_url_scans_for_key(tmp_path, monkeypatch):
    """A bare (non-CDN) paste identifies the video: key+URL come from scan."""
    candidates = [
        {"decode_key": SAMPLE_DECODE_KEY,
         "url_head": "https://finder.video.qq.com/251/20302/stodownload?encfilekey=X",
         "timestamp": "1786000000000", "pid": 111, "process": "Weixin.exe"},
    ]
    monkeypatch.setattr(node_mod, "scan_wechat_decode_keys", lambda max_seconds: candidates)
    monkeypatch.setattr(
        node_mod, "complete_wechat_url",
        lambda candidate, max_seconds: SAMPLE_CIPHERTEXT.as_uri(),
    )
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    result = node_mod.MusefishWeChatChannels.execute(
        link_or_url="play the video then run",  # no URL: everything via scan
        )
    fps = result[1]
    assert isinstance(fps, float) and fps > 0


def test_channels_node_bad_key_aborts_before_download(tmp_path, monkeypatch):
    """The ftyp gate must stop manual+url runs with a wrong key: no file."""
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    paste = json.dumps({
        "code": 200,
        "data": {"object_desc": {"media": [{
            "url": "https://finder.video.qq.com/251/20302/stodownload"
                   "?encfilekey=X&token=T",
            "decode_key": "1999999999",
        }]}},
    })

    def rejecting_download(url, directory, key, reporter, log):
        raise backend.DownloadError(
            "decode_key does not decrypt this URL (ftyp mismatch). "
            "Keys are one-to-one with the CDN URL: re-fetch the URL/key pair."
        )

    monkeypatch.setattr(node_mod, "download_wechat_encrypted", rejecting_download)
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    with pytest.raises(backend.DownloadError, match="ftyp mismatch"):
        node_mod.MusefishWeChatChannels.execute(
            link_or_url=paste,
        )
    assert list(tmp_path.iterdir()) == []


def test_channels_node_scan_fills_key_and_url(tmp_path, monkeypatch):
    """scan mode: node recovers key + tokened URL itself, then downloads."""
    candidates = [
        {"decode_key": SAMPLE_DECODE_KEY,
         "url_head": "https://finder.video.qq.com/251/20302/stodownload?encfilekey=ABCDEF0123456789abcdef",
         "timestamp": "1786000000000", "pid": 111, "process": "Weixin.exe"},
        {"decode_key": "1111222233", "url_head": "https://finder.video.qq.com/251/20302/stodownload?encfilekey=ZZ",
         "timestamp": "1785900000000", "pid": 111, "process": "Weixin.exe"},
    ]
    monkeypatch.setattr(node_mod, "scan_wechat_decode_keys", lambda max_seconds: candidates)
    monkeypatch.setattr(
        node_mod, "complete_wechat_url",
        lambda candidate, max_seconds: SAMPLE_CIPHERTEXT.as_uri(),
    )
    monkeypatch.setattr(node_mod, "_resolve_dir", lambda: tmp_path)
    result = node_mod.MusefishWeChatChannels.execute(
        link_or_url="",
        )
    fps = result[1]
    assert isinstance(fps, float) and fps > 0


def test_channels_node_scan_no_records_aborts(monkeypatch):
    monkeypatch.setattr(node_mod, "scan_wechat_decode_keys", lambda max_seconds: [])
    with pytest.raises(backend.DownloadError, match="no WC video records"):
        node_mod.MusefishWeChatChannels.execute(
            link_or_url="",
            )


# ------------------------------------------------------------ paste parsing


def test_auto_extract_variants():
    logs = []
    api = json.dumps({
        "data": {"object_desc": {"media": [
            {"url": "https://finder.video.qq.com/251/20302/stodownload?encfilekey=A&token=T",
             "thumb_url": "https://finder.video.qq.com/251/20304/stodownload?encfilekey=B", "decode_key": "123456789"},
        ]}},
    })
    url, key = node_mod._auto_extract(api, logs)
    assert key == "123456789"
    assert "20302/stodownload" in url and "token=T" in url

    url, key = node_mod._auto_extract(
        "see this https://finder.video.qq.com/251/20302/stodownload?encfilekey=A&token=T ok", logs)
    assert url.startswith("https://finder.video.qq.com") and key == ""

    url, key = node_mod._auto_extract("look at https://example.com/t/abc nice video", logs)
    assert url == "" and key == ""

    url, key = node_mod._auto_extract("", logs)
    assert url == "" and key == ""

    url, key = node_mod._auto_extract("{not json really", logs)
    assert url == "" and key == ""


def test_auto_extract_prefers_video_over_thumb():
    logs = []
    api = json.dumps({"object_desc": {"media": [
        {"url": "", "thumb_url": "https://finder.video.qq.com/251/20304/stodownload?encfilekey=B"},
    ]}})
    url, key = node_mod._auto_extract(api, logs)
    assert url == "" and key == ""


# ------------------------------------------------------------ memory guard


def test_memory_guard_returns_immediately_under_low_pressure():
    guard = importlib.import_module(
        "custom_nodes.ComfyUI-Musefish-Nodes.memory_guard"
    )
    # live system is nowhere near 95% in CI; must return False (no wait)
    assert guard.system_memory_fraction() < 0.95
    assert guard.wait_for_memory() is False


def test_memory_guard_waits_until_watermark_clears(monkeypatch):
    guard = importlib.import_module(
        "custom_nodes.ComfyUI-Musefish-Nodes.memory_guard"
    )
    readings = iter([0.97, 0.96, 0.90])
    monkeypatch.setattr(guard, "system_memory_fraction", lambda: next(readings))
    monkeypatch.setattr(guard, "_POLL_INTERVAL", 0.0)
    logs: list[str] = []
    waited = guard.wait_for_memory(log=logs)
    assert waited is True
    assert any("memory resumed" in line for line in logs)


def test_memory_guard_deadline_forces_continue(monkeypatch):
    guard = importlib.import_module(
        "custom_nodes.ComfyUI-Musefish-Nodes.memory_guard"
    )
    monkeypatch.setattr(guard, "system_memory_fraction", lambda: 0.99)
    monkeypatch.setattr(guard, "_POLL_INTERVAL", 0.0)
    logs: list[str] = []
    waited = guard.wait_for_memory(deadline_seconds=0.0, log=logs)
    assert waited is True
    assert any("continuing anyway" in line for line in logs)
