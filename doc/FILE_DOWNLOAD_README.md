# 文件下载（直链 / 签名 CDN，断点续传）

节点：`musefish_file_downloader`（`Musefish/Download`）。
Studio「其他工具 → 文件断点续传」与本节点共用 `musefish_fastdl.py`。

## 用途与输入

CDN（如 CivitAI `b2`、S3 兼容存储、ModelScope）常按单连接限速（20–150 KB/s）；
独立 Range 分片并发可绕过单连接瓶颈。
实测：320 MB LoRA 从约 0.1 MB/s 提升至峰值约 10 MB/s；
988 MB 文件、8 并发约 22 MB/s。

| 参数 | 说明 |
|---|---|
| `url` | 完整 http(s) 直链；保留全部签名参数（如 `Authorization=`、`b2ContentDisposition=`）。签名链接 `HEAD` 会返回 401，必须用 `GET` 判断有效性。链接中的凭据属于敏感信息，勿公开或写入日志。 |
| `save_dir` | 空值默认 `ComfyUI/output/musefish/`；绝对路径或相对 ComfyUI 根目录路径。 |
| `filename` | 空值按 `Content-Disposition` / URL 识别，非法字符会清洗。 |
| `conns` | 并发分片数 1–64，默认 32；限速 CDN 可增加，局域网通常 4–8 足够。 |
| `proxy` | `自动（auto）` / `直连（direct）` 或本机代理。自动按直连优先，失败再探测本机代理并记住成功通路。探测端口：Clash Verge·Mihomo 7897/7890、v2rayN 10809/10808、sing-box/Hiddify 2080/12334、Shadowsocks 1080。自定义端口填 `http://127.0.0.1:<端口>`。 |
| `overwrite` | 目标存在时关：另存 `name (1).ext`；开：覆盖。 |
| `allow_any_dir` | 默认关；允许 ComfyUI 的 output/input/models、桌面/下载/文档目录。用 `MUSEFISH_DOWNLOAD_DIRS` 追加根目录（Windows 以 `;` 分隔）；其他位置须显式开启。 |

## 输出

- `path`：落盘绝对路径。
- `report`：文件名、大小、并发、通路摘要；同时显示于 `ui.text` 任务历史。

## 断点续传

- 每个 4 MiB 分片写入 `<name>.part`；状态存于 `<name>.state.json`。
  中断后用相同 `url`、`save_dir` 重跑续传，完成后两文件自动清理。
- 状态约每 0.25 s 或每 16 个分片，经 `flush`+`fsync` 后更新；
  只记录已落盘分片，崩溃最多重下一个检查点内分片。
  未完成任何分片即中断时删除 `.part`。
- 续传会校验 `ETag`/`Last-Modified` 与总大小。
  签名 URL 刷新后仅在对象的 ETag、size 匹配时续传。
  对象变化或鉴权返回 401/403 时废弃旧状态，从头下载，避免拼接不同对象。

## 行为与注意事项

- 用 `GET Range: bytes=0-0` 探测大小和分片支持，**不用 HEAD**。
  `200 + Content-Range` 仍按分片处理；只有 `200` 且无 `Content-Range` 才退回单流。
  大小未知时也退回单流。
- 自动代理顺序为先直连、再本机代理。
  TLS `UNEXPECTED_EOF`、连接重置短退避重试；连接超时（WinError 10060）不重试而切换通路。
  401/403 报签名或令牌失效。
- 每个下载线程复用一条 keep-alive 连接（代理经 CONNECT），避免每个 4 MiB 分片重新握手。
  本机 7897 代理新建连接实测 2.4–3.4 s（DNS+TLS+TTFB）。
  800 ms/连接对照实验中，20 分片、4 并发由 21 条连接、5.18 s 降至 5 条、1.97 s（−62%）。
  `Connection: close` 或连接回收后下次请求重连。
- 重定向只解析一次（CivitAI 307 → r2.cloudflarestorage；ModelScope → cdn-lfs）。
  最终地址过期返回 403 时从原始 URL 重解析一次，续传状态不变。
  完成后校验总字节数；`safetensors` 可由下游加载器确认可读。
