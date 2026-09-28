# Video Download Nodes

This guide documents the `MusefishVideoDownload` and `MusefishWeChatChannels` nodes in ComfyUI-Musefish-Nodes. A ready-to-run workflow is available at [`workflows/Musefish_Video_Downloader.json`](../workflows/Musefish_Video_Downloader.json).

## Features and quick start

- **`Musefish Video Download`** (node ID: `MusefishVideoDownload`) downloads videos from supported platforms. YouTube, Bilibili, and X/Twitter use the in-process yt-dlp integration. Douyin uses the built-in `iesdouyin` share-page parser; yt-dlp's Douyin extractor is blocked by the platform's `a_bogus` signature checks and is not usable. Xiaohongshu is attempted through yt-dlp; if that fails, the node reports an error and recommends extracting the video through a browser. Before returning a file, the node verifies both the MP4 `ftyp` header and the file with ffprobe.
- **`Musefish WC Video Channels`** (node ID: `MusefishWeChatChannels`) downloads and decrypts WeChat Channels videos in one automatic, linear flow; you do not need to enter a key manually:
  **Parse pasted input automatically → scan the client process memory for URL/decode_key → download encrypted video → check `ftyp` header → decrypt → validate with ffprobe.**
  The operation stops immediately if validation fails. With a wrong key, it will not download the entire file or emit an unvalidated file.

## `Musefish Video Download` inputs

| Input | Type | Description |
| --- | --- | --- |
| `url` | STRING (multiline) | Direct video CDN URL, share link, or share text. Text combining a title and link is supported; the real URL is extracted automatically. |
| `platform` | COMBO | `Auto` (default; detected from the URL), YouTube, Bilibili, Douyin, Xiaohongshu, or X-Twitter. If you select a platform explicitly and the URL does not match it, the node reports an error. |
| `quality` | COMBO | `best` (default), `1080p`, `720p`, or `480p`. Ignored when downloading a direct URL. |
| `chunks` | COMBO | Number of parallel download chunks: `1`, `2`, `4`, `8`, or `16`; default `4`. A Douyin direct URL is split into N concurrent HTTP Range requests; YouTube and Bilibili use yt-dlp's `concurrent_fragment_downloads`. **The UI hides or adjusts this control automatically:** it is hidden when the URL is empty or the platform does not support chunks. For detected Douyin, YouTube, or Bilibili URLs, it appears and automatically selects the optimal value `8` (you can change it manually). Unknown platforms are forced to `1` at execution time. |

## Outputs and downstream connections

Both download nodes provide the same outputs:

| Output | Type | Description |
| --- | --- | --- |
| `video` | VIDEO | Connect directly to `SaveVideo` or any other VIDEO consumer. |
| `fps` | FLOAT | Frame rate of the downloaded video. |

> Full `images` or `audio` outputs are intentionally **not** provided. Decoding a 194-second 1080p video into frame tensors would create a float32 tensor larger than 112 GB and exhaust system memory. Use a dedicated sampling node if a downstream consumer needs frames.

## Save location

Files are saved under `ComfyUI/output/musefish/` with names in the form `<platform>_<title/video ID>_<timestamp>.mp4`. A numeric suffix is added automatically if a filename already exists.

## Usage notes

- **X/Twitter and YouTube use a local proxy when it is reachable.** The configured default is `http://127.0.0.1:7897` (the Clash Verge mixed port); this port is fixed in the node, so change it in the plugin code if your local proxy listens elsewhere. If the proxy is unreachable, X falls back to a direct connection; YouTube reports an error. YouTube also requires cookies: the node reads `_yt_cookies.txt` (Netscape format) from the package directory. Without cookies, YouTube's bot wall may reject the request with “Sign in to confirm you're not a bot.”
- **Douyin cookies:** `_dy_cookies.txt` in the package directory must contain cookies captured from a logged-in browser through CDP, including `ttwid` and `s_v_web_id`. Without cookies, the `iesdouyin` mobile endpoint returns `video_layout: null` due to bot detection.
- **yt-dlp self-update:** A background thread checks for updates when the package is imported, at most once every 24 hours, using the Tsinghua pip mirror. Startup is not blocked. An outdated yt-dlp is the most common reason for YouTube or Douyin extractors to stop working.
- **WeChat Channels input:** `link_or_url` on `Musefish WC Video Channels` accepts a media CDN URL, share link, share text, or JSON response from the Channels API. It may also be left blank to use the most recent playback record. Missing information (URL/decode_key) is recovered automatically from the memory of the locally running client. **Before using the node, play the target video in the client for a few seconds and keep the client running.**
- **Avoid auto-playback mismatches with the two-link method:** After a video ends, the client automatically plays the next one, so the latest memory record may not correspond to the link you intend to download. For a reliable sequence: (1) copy share link A; (2) open A and play it for a few seconds; (3) copy any other share link B; (4) paste A into the node and run it. Copying B last pins the target video in the “recently opened” position so auto-playback does not displace it.
- **Memory guard:** Between download chunks, the node checks total system memory usage. If usage exceeds 95%, it pauses and waits to avoid pushing the machine into swap.
- **Sensitive URL cache:** A recovered tokenized URL is cached in `_wc_url_cache.json` in the package directory. The WeChat client may compact the URL row out of memory within a few minutes; the cache lets you download it again afterward. This URL is a temporary access credential valid for about 48 hours. **Do not share workflow JSON or screenshots containing it.**

## Related files

- `musefish_video_nodes.py`: schemas for both download nodes, automatic parsing of pasted content, and execution orchestration.
- `musefish_video_download.py`: yt-dlp downloads (including parallel chunks), encrypted Channels download/decryption, client-memory scanning, `ftyp`/ffprobe validation, and URL caching.
- `memory_guard.py`: system memory pressure guard; pauses downloads above 95% usage.
- `_startup_update.py`: background yt-dlp self-updater (once every 24 hours).
- `wxdec_toolchain/`: resident keystream daemon, including WASM and a Node daemon script; no global dependencies are required. The WASM comes from [Evil0ctal/WeChat-Channels-Video-File-Decryption](https://github.com/Evil0ctal/WeChat-Channels-Video-File-Decryption) (MIT); this repository's daemon is an original rewrite.
- `web/musefish_video_download_segments.js`: platform-aware UI logic that shows/hides the `chunks` control and adjusts its value automatically.
- `tests/test_video_download_nodes.py`: tests for the decryption pipeline (using upstream sample ciphertext and key as ground truth in `tests/fixtures/`), key validation gate, Range fallback, paste parsing, platform detection, chunk concurrency parameters, and node behavior.
