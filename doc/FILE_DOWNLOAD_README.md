# 文件下载（直链 / 签名 CDN，断点续传）

节点：`musefish_file_downloader`（分类 `Musefish/Download`，搜索别名 file download / 断点续传 / safetensors / civitai / resume）

Studio 的「其他工具 → 文件断点续传」与本节点**共用同一份实现**（`musefish_fastdl.py`），
所以两侧的分片策略、续传状态、代理探测和报错文案完全一致，不会各自漂移。

## 它解决什么

CivitAI `b2`、S3 兼容存储、ModelScope 一类 CDN 会**按单连接限速**（常见 20–150 KB/s）。
浏览器、`curl`、甚至 `aria2c -x16` 都吃不满带宽，因为它们只是在一条被限速的连接上切文件。
本节点改用「多条独立 Range 请求」：默认 32 并发（可到 64），实测把 320 MB 的 LoRA
从 ~0.1 MB/s 提到峰值 ~10 MB/s，988 MB 文件 8 并发也有 ~22 MB/s。

## 输入

| 参数 | 说明 |
|---|---|
| `url` | 完整 http(s) 直链。**签名参数必须整串保留**（`Authorization=`、`b2ContentDisposition=`）。这类链接 `HEAD` 恒返回 401，判断是否过期要用 `GET`。 |
| `save_dir` | 留空 = `ComfyUI/output/musefish/`；可填绝对路径（例如 `H:/ComfyUI/models/loras`），相对路径按 ComfyUI 根目录解析。 |
| `filename` | 留空按 `Content-Disposition` / URL 自动识别；非法字符会被清洗。 |
| `conns` | 并发分片数 1–64，默认 32。限速 CDN 上加大有效；本地/内网 4–8 足够。 |
| `proxy` | `自动（auto）` / `直连（direct）` + 本机正在监听的代理客户端（Clash Verge·Mihomo 7897/7890、v2rayN 10809/10808、sing-box/Hiddify 2080/12334、Shadowsocks 1080）。自动 = 先直连，失败改用本机代理并记住该通路。 |
| `overwrite` | 目标已存在时：关 = 另存 `name (1).ext`；开 = 覆盖。 |
| `allow_any_dir` | 默认关：`save_dir` 只允许 ComfyUI 的 output/input/models 与桌面/下载/文档目录（另可用环境变量 `MUSEFISH_DOWNLOAD_DIRS` 追加，Windows 用 `;` 分隔）。要写别处必须显式打开。 |

## 输出

- `path`：落盘的绝对路径。
- `report`：`文件名 · 大小 · 并发 · 通路` 摘要，同时通过 `ui.text` 出现在任务历史里，便于事后核对。

## 进度显示与回调

下载过程中节点会把一行实时状态推送到自己的部件上（`send_progress_text`），
控制台每 2 s 同步打印一行，进度条按百分比推进：

```
160.0 MB / 320.1 MB · 50.0% · 10.7 MB/s · ETA 00:15
```

要在自己的代码里接收进度，把 `ProgressReporter` 直接当 `on_progress` 传给引擎：

```python
from musefish_fastdl import ProgressReporter, download

reporter = ProgressReporter(lambda p: print(p.text(), p.percent, p.speed, p.eta))
download(url, dest, on_progress=reporter)
```

- 回调收到 `Progress(done, total, percent, speed, eta, elapsed, resumed)`；
  `p.text()` 就是上面那行，`format_bytes()` / `format_seconds()` 可单独复用。
- 节流：默认每 0.25 s 一次（`ProgressReporter(sink, interval=...)` 可调）；
  **首次非零回调一定放行**（续传时它是已有进度基线，不会被算成速度尖峰），
  **最后一次一定放行**（保证 100% 必达）。
- 速度是窗口速率的指数平滑（`smoothing=0.3`），`eta` 在速率未知时为 `None`。
- 回调运行在下载线程池里，sink 不要做长阻塞；sink 抛异常会被吞掉，不会中断下载。
- Studio 的「文件断点续传」把同样的快照写进 `job.progress`
  （`pct` / `speed` / `eta` / `done` / `total` / `elapsed` / `resumed` / `text`），前端显示百分比、已下载量、速率与剩余时间。

## 断点续传

下载过程中每完成一个 4 MiB 分片就写到 `<name>.part` 并更新 `<name>.state.json`。
中断（取消、断网、关掉 ComfyUI）后用**完全相同的 url 与 save_dir 再跑一次**即可续传，
不会重下已完成分片；全部完成后 `.part`/`.state.json` 自动清理。
状态文件按需落盘：每约 0.25 s 或每 16 个分片才把 `flush`+`fsync` 后仍持久的分片合并写入一次，
因此它只会声明真正落盘的分片——崩溃后最多重下最近一个检查点内的分片，不会出现“状态说有、磁盘没有”。
若一个分片都没完成就中断，`.part` 会被删除，避免留下“看起来有进度”的空文件。
续传前会核对 `ETag`/`Last-Modified` 与总长：地址背后的对象换了（签名链接重签、CDN 换源），
或该链接改要求鉴权（`401`/`403`）时，旧状态立即作废并从零重下，绝不把两个对象的内容拼成一个文件。

## 行为细节

- 总长度与分片支持来自 `GET Range: bytes=0-0`，**不用 HEAD**。部分网关（ModelScope）以
  `200 + Content-Range` 返回请求的区间，这种情况同样按分片处理；只有「200 且没有
  Content-Range」才判定服务端忽略 Range，此时自动退化为单流下载。
- 代理偶发 TLS `UNEXPECTED_EOF`、连接重置会小退避重试；连接超时（WinError 10060）不重试，
  直接换下一条通路，避免白等。
- 连接复用：每个下载线程保持 1 条 keep-alive 连接（代理走 CONNECT 隧道），不再为每个
  4 MiB 分片重新握手。本机经 7897 代理实测**新建连接 2.4–3.4 s**（DNS+TLS+TTFB）；
  本地按 800 ms/连接 的对照实验：20 分片、4 并发下连接数 21→5，耗时 5.18 s→1.97 s（**−62%**）。
  服务端返回 `Connection: close` 或连接被回收时，下一次请求自动重连一次，行为不变。
- 401/403 直接报「签名或令牌已失效」，不会伪装成网络错误。
- 重定向只解析一次：CivitAI（307 → r2.cloudflarestorage）、ModelScope（→ cdn-lfs）这类链接，
  探测时就把最终地址固定下来，分片直接打最终地址，不再每片先付一次 307；固定地址若因签名
  过期返回 403，会自动回到原始链接重新解析一次再继续（续传状态不受影响）。
- 完整下载后校验总字节数；`safetensors` 等格式可再用下游加载器确认可读。

## 常见问题

| 现象 | 处理 |
|---|---|
| 报 401 但链接是刚复制的 | 用 `GET` 而不是 `HEAD` 复测；确认 `Authorization`、`b2ContentDisposition` 都在。 |
| 速度只有几百 KB/s | `conns` 提到 48–64；确认不是走了很慢的代理（`report` 里会显示通路）。 |
| 想直接落到模型目录 | `save_dir` 填 `ComfyUI/models/...` 下的绝对路径；回头在加载器里刷新即可看到。 |
| 保存目录被拒绝 | 提示里的 `allow_any_dir` 就是开关：默认只允许 ComfyUI 的三类目录与桌面/下载/文档，避免任意路径写入。 |
| 进度停在 100% 但还没结束 | 服务器没给 `Content-Length`（`probe` 报告「大小未知」）。此时没有总长可比，进度条只表示已收字节；完成以最终文件落盘为准。 |
| 代理端口不是默认值 | 在 `proxy` 里填 `http://127.0.0.1:<端口>`；自动模式只会用探测到的默认端口。 |
