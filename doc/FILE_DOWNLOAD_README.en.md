# File download (direct links / signed CDNs, with resume)

Node: `musefish_file_downloader` (category `Musefish/Download`; search aliases: file download, 断点续传, safetensors, civitai, resume)

Studio's "其他工具 → 文件断点续传" task and this node **share one implementation**
(`musefish_fastdl.py`), so chunking, resume state, proxy detection and error wording
cannot drift between the two hosts.

## What it solves

CivitAI `b2`, S3-compatible storage and ModelScope throttle **per connection**
(typically 20–150 KB/s). A browser, `curl`, or even `aria2c -x16` cannot fill the
link, because they split one throttled connection. This node issues **independent
ranged requests**: 32 concurrently by default (up to 64). Measured: a 320 MB LoRA
went from ~0.1 MB/s to ~10 MB/s peak; a 988 MB file reached ~22 MB/s at 8 connections.

## Inputs

| Input | Meaning |
|---|---|
| `url` | Full http(s) direct link. **Keep every query parameter** (`Authorization=`, `b2ContentDisposition=`). Such links answer `HEAD` with 401 forever — test expiry with `GET`. |
| `save_dir` | Empty = `ComfyUI/output/musefish/`; absolute paths allowed (e.g. `H:/ComfyUI/models/loras`); relative paths resolve against the ComfyUI root. |
| `filename` | Empty = detect from `Content-Disposition` / URL; illegal characters are sanitized. |
| `conns` | Concurrent ranged requests, 1–64, default 32. Helps on throttled CDNs; 4–8 is enough for LAN/local. |
| `proxy` | `自动（auto）` / `直连（direct）` plus proxy clients actually listening on this machine (Clash Verge·Mihomo 7897/7890, v2rayN 10809/10808, sing-box/Hiddify 2080/12334, Shadowsocks 1080). Auto = try direct first, then the local proxy, and remember the winning route. |
| `overwrite` | If the target exists: off = save as `name (1).ext`; on = overwrite. |
| `allow_any_dir` | Off by default: `save_dir` must be inside ComfyUI's output/input/models or Desktop/Downloads/Documents (extra roots via `MUSEFISH_DOWNLOAD_DIRS`, `;`-separated on Windows). Any other location needs this switch. |

## Outputs

- `path`: absolute path on disk.
- `report`: `name · size · connections · route`, also surfaced through `ui.text` in the
  job history for later verification.

## Live progress and callbacks

While downloading, the node pushes one live status line to its own widget
(`send_progress_text`), prints the same line to the console every 2 s, and
advances a percentage bar:

```
160.0 MB / 320.1 MB · 50.0% · 10.7 MB/s · ETA 00:15
```

To consume progress from your own code, hand a `ProgressReporter` to the engine:

```python
from musefish_fastdl import ProgressReporter, download

reporter = ProgressReporter(lambda p: print(p.text(), p.percent, p.speed, p.eta))
download(url, dest, on_progress=reporter)
```

- The callback receives `Progress(done, total, percent, speed, eta, elapsed, resumed)`;
  `p.text()` is the line above, and `format_bytes()` / `format_seconds()` are reusable.
- Throttling: one snapshot per 0.25 s by default (`interval=` to change it). The first
  non-zero callback always fires (on resume it is the existing-bytes baseline, so it is
  not reported as a speed spike) and the final one always fires (100% is guaranteed).
- Speed is an exponentially smoothed window rate (`smoothing=0.3`); `eta` is `None`
  while the rate is still unknown.
- Callbacks run on the download worker threads: keep sinks short, and a raising sink is
  swallowed rather than failing the download.
- Studio's 「文件断点续传」 task writes the same snapshot into `job.progress`
  (`pct` / `speed` / `eta` / `done` / `total` / `elapsed` / `resumed` / `text`), and its
  UI shows percent, transferred size, rate and time remaining.

## Resume

Every completed 4 MiB chunk is written to `<name>.part` and recorded in
`<name>.state.json`. After an interruption (cancel, network drop, ComfyUI restart),
re-run with **the same url and save_dir** and it continues instead of restarting;
both files are removed once the download completes. The state file is written in
batches (roughly every 0.25 s or 16 chunks, only after `flush`+`fsync`), so it never
claims a chunk that did not reach the disk; a crash costs at most one checkpoint.
If nothing was downloaded yet,
the pre-allocated `.part` is deleted on cancel so it cannot look like progress.

## Behaviour notes

- Length and range support come from `GET Range: bytes=0-0`, **never HEAD**. Some
  gateways (ModelScope) answer `200 + Content-Range` for the requested span; that is
  treated as ranged too. Only "200 without Content-Range" means Range was ignored, and
  the download then falls back to a single stream.
- Transient proxy failures (TLS `UNEXPECTED_EOF`, connection reset) are retried with a
  short backoff; connect timeouts (WinError 10060) are not retried — it switches route
- Connection reuse: each download thread keeps one keep-alive connection (through the proxy
  via CONNECT) instead of paying a fresh handshake per 4 MiB chunk. A new connection measured
  2.4–3.4 s through the local proxy (DNS+TLS+TTFB); in a controlled A/B with 800 ms per
  connection, 20 chunks over 4 workers went from 21 connections / 5.18 s to 5 connections /
  1.97 s (**−62%**). A `Connection: close` reply, or an idle-timed-out socket, costs one
  reconnect on the next request.
- Redirects are resolved once: for links that 307 (CivitAI → r2.cloudflarestorage,
  ModelScope → cdn-lfs) the probe pins the final URL, so chunks no longer pay a redirect
  first. If that pinned URL expires (403 on a stale signature) the original link is
  re-probed once and the download continues; resume state is unaffected.
  instead of waiting twice.
- 401/403 surface as "signature or token invalid or expired", never as a vague network
  error.
- The final byte count is verified; for `safetensors` a downstream loader is the last
  sanity check.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 401 on a freshly copied link | Re-test with `GET`, not `HEAD`; make sure `Authorization` and `b2ContentDisposition` are both present. |
| Stuck at a few hundred KB/s | Raise `conns` to 48–64; check the route shown in `report` (a slow proxy caps everything). |
| Want the file inside the models tree | Put that absolute path in `save_dir`, then refresh the loader. |
| Non-default proxy port | Enter `http://127.0.0.1:<port>` in `proxy`; auto mode only uses detected default ports. |
