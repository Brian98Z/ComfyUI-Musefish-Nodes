# File download (direct links / signed CDNs, with resume)

Node: `musefish_file_downloader` (`Musefish/Download`).
Studio's "其他工具 → 文件断点续传" task shares `musefish_fastdl.py` with this node.

## Purpose and inputs

CDNs such as CivitAI `b2`, S3-compatible storage and ModelScope often throttle per
connection (20–150 KB/s); concurrent independent Range requests bypass the single-
connection bottleneck. Measured: a 320 MB LoRA improved from ~0.1 MB/s to ~10 MB/s
peak; a 988 MB file reached ~22 MB/s at 8 connections.

| Input | Meaning |
|---|---|
| `url` | Full http(s) direct link; preserve every signature parameter (e.g. `Authorization=`, `b2ContentDisposition=`). Signed links return 401 to `HEAD`; **use `GET` to check validity**. URL credentials are sensitive; do not publish or log them. |
| `save_dir` | Empty defaults to `ComfyUI/output/musefish/`; accepts absolute paths or paths relative to the ComfyUI root. |
| `filename` | Empty detects from `Content-Disposition` / URL; illegal characters are sanitized. |
| `conns` | Concurrent chunks, 1–64, default 32. Increase for throttled CDNs; 4–8 is usually enough on a LAN. |
| `proxy` | `自动（auto）` / `直连（direct）` or a local proxy. Auto tries direct first, then detects local proxies and remembers the successful route. Ports: Clash Verge·Mihomo 7897/7890, v2rayN 10809/10808, sing-box/Hiddify 2080/12334, Shadowsocks 1080. Custom port: `http://127.0.0.1:<port>`. |
| `overwrite` | If target exists, off saves as `name (1).ext`; on overwrites. |
| `allow_any_dir` | Off by default; allows ComfyUI output/input/models and Desktop/Downloads/Documents. Add roots with `MUSEFISH_DOWNLOAD_DIRS` (Windows separator: `;`); enable explicitly for other locations. |

## Outputs

- `path`: absolute output path.
- `report`: filename, size, connections and route summary; also shown in task history via `ui.text`.

## Resume

- Each 4 MiB chunk is written to `<name>.part`; state is stored in `<name>.state.json`.
  After interruption, rerun with the same `url` and `save_dir` to resume; both files are
  removed on completion.
- State is updated about every 0.25 s or 16 chunks, after `flush`+`fsync`; it records
  only chunks persisted to disk, so a crash can require redownloading at most one
  checkpoint. If no chunk completed, `.part` is deleted.
- Resume checks `ETag`/`Last-Modified` and total size. After refreshing a signed URL,
  resume only if the object's ETag and size match. A changed object or 401/403
  authentication response invalidates old state and restarts from zero, preventing
  mixed-object files.

## Behaviour and caveats

- Probe size and range support with `GET Range: bytes=0-0`, **not HEAD**. `200 +
  Content-Range` is still treated as ranged; only `200` without `Content-Range` falls
  back to a single stream. Unknown size also falls back to a single stream.
- Auto proxy order is direct, then local proxy. Transient TLS `UNEXPECTED_EOF` and
  connection resets retry with short backoff; connect timeouts (WinError 10060) switch
  route without retry. 401/403 report an invalid or expired signature/token.
- Each worker reuses one keep-alive connection (proxy via CONNECT), avoiding a
  handshake for every 4 MiB chunk. Through local port 7897, a new connection measured
  2.4–3.4 s (DNS+TLS+TTFB). In an 800 ms/connection comparison, 20 chunks / 4 workers
  went from 21 connections and 5.18 s to 5 connections and 1.97 s (−62%).
  `Connection: close` or a reclaimed connection reconnects on the next request.
- Redirects are resolved once (CivitAI 307 → r2.cloudflarestorage; ModelScope → cdn-lfs).
  If the final URL expires with 403, the original URL is re-resolved once; resume state
  is unchanged. The final byte count is verified; a downstream loader can confirm
  `safetensors` readability.
