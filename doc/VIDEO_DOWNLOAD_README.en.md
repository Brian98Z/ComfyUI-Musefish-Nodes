# Video Download Nodes

## Nodes
- **Musefish Video Download** (`musefish_video_downloader`): downloads YouTube, Bilibili, X/Twitter, Douyin, and Xiaohongshu videos. Uses yt-dlp for YouTube/Bilibili/X, the built-in `iesdouyin` share-page parser for Douyin, and yt-dlp for Xiaohongshu (failure reports an error and recommends browser extraction). Verifies the MP4 `ftyp` header and ffprobe before returning.
- **Musefish WC Video Channels** (`musefish_wc_video_channels`): downloads and decrypts WeChat Channels videos by parsing pasted input, scanning client memory for URL/decode_key, downloading ciphertext, checking `ftyp`, decrypting, and validating with ffprobe. Failure stops processing; a wrong key does not download the full file or emit an unvalidated file.

## Musefish Video Download inputs
| Parameter | Type | Description |
| --- | --- | --- |
| `url` | STRING (multiline) | CDN URL, share link/text; may include a title, from which the URL is extracted. |
| `platform` | COMBO | `Auto` (default; detect from URL) / YouTube / Bilibili / Douyin / Xiaohongshu / X-Twitter; mismatch with an explicitly selected platform reports an error. |
| `quality` | COMBO | `best` (default) / `1080p` / `720p` / `480p`; ignored for direct URLs. |
| `chunks` | COMBO | `1` / `2` / `4` / `8` / `16`, default `4`. Douyin direct URLs use N parallel Range requests; YouTube/Bilibili use yt-dlp `concurrent_fragment_downloads`. Hidden for empty URLs or unsupported platforms; Douyin/YouTube/Bilibili automatically select `8` (editable); unknown platforms are forced to `1` at execution. |

## Outputs and saving
Both nodes output `video` (file-backed VIDEO; connect to `SaveVideo` or another consumer) and `fps` (FLOAT frame rate). Files go to `ComfyUI/output/musefish/` as `<platform>_<title/video ID>_<timestamp>.mp4`; collisions get a numeric suffix. Full `images`/`audio` outputs are omitted: decoding a 194s/1080p video into float32 frame tensors exceeds 112GB; use a sampling node when frames are needed.

## Dependencies and usage
- yt-dlp: `pip install -U yt-dlp`. A background update check runs on plugin import every 24 hours using the Tsinghua pip mirror and does not block startup; outdated yt-dlp commonly breaks YouTube/Douyin extraction.
- X/Twitter and YouTube use local proxy `http://127.0.0.1:7897` (Clash Verge mixed port). If unreachable, X falls back to direct connection and YouTube errors. YouTube also requires package-directory `_yt_cookies.txt` (Netscape format); without it, the bot wall may reject requests.
- Douyin requires package-directory `_dy_cookies.txt` containing `ttwid` and `s_v_web_id` captured through CDP from a logged-in browser; without it, the iesdouyin mobile endpoint returns `video_layout: null` (bot detection).
- Xiaohongshu is attempted with yt-dlp; failure reports an error and recommends browser extraction.
- WeChat Channels `link_or_url` accepts a CDN URL, share link/text, or Channels API JSON; leave blank to use the latest playback record. Missing URL/decode_key are recovered from local client memory. Before use, open and play the target video in the local client for a few seconds and keep the client running. To avoid auto-playback mismatch: copy and play target link A for a few seconds, copy another link B, then paste A and run.
- Between `chunks`, system memory usage is checked; above 95% the node pauses. Recovered token URLs are cached in package-directory `_wc_url_cache.json` and are valid for about 48 hours. Cookies and URLs containing tokens are credentials: never commit or share them, including in shared workflow JSON or screenshots.
- Workflow: `workflows/Musefish_Video_Downloader.json`. Load it, enter a link and select platform/quality/chunks as needed, then connect downstream `video` and `fps`; before running the Channels node, play the target video in the local client as above.
