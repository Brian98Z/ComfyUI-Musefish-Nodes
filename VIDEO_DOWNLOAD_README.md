# 视频下载节点

### 功能与快速上手

- **`Musefish Video Download`**（节点 ID：`MusefishVideoDownload`）：通用平台视频下载。YouTube / B站 / X/Twitter 走进程内 yt-dlp；抖音走内置 iesdouyin 分享页解析（yt-dlp 的 Douyin 提取器被 a_bogus 签名墙挡死，不可用）；小红书走 yt-dlp 尝试、失败即报错并建议改走浏览器提取。每个文件在节点返回前都经过 MP4 `ftyp` 头 + ffprobe 双重验证。
- **`Musefish WC Video Channels`**（节点 ID：`MusefishWeChatChannels`）：微信视频号一键下载解密。单一直线流程，全自动、无需手动填 key：
  **自动解析粘贴内容 → 扫描客户端内存获取 URL/decode_key → 下载密文 → ftyp 头校验 → 解密 → ffprobe 验证**。
  校验不通过立即中止：key 错误时不会下载完整文件，也不会输出未验证的文件。

### Musefish Video Download 输入

| 输入 | 类型 | 说明 |
| --- | --- | --- |
| `url` | STRING（多行） | 视频 CDN 直链 / 分享链接 / 分享文本，支持"标题+链接"混合文本（自动提取真实 URL） |
| `platform` | COMBO | `Auto`（默认，从 URL 自动检测）/ YouTube / Bilibili / Douyin / Xiaohongshu / X-Twitter；选定平台后 URL 不符直接报错 |
| `quality` | COMBO | best（默认）/ 1080p / 720p / 480p；直链下载时忽略 |
| `chunks` | COMBO | 分块并行下载：1 / 2 / 4 / 8 / 16，默认 4。抖音直链切成 N 段 Range 并发；YouTube/B站走 yt-dlp `concurrent_fragment_downloads`。**前端自动隐藏/吸附**：URL 为空或平台不支持分块时控件隐藏；检测到抖音/YouTube/B站链接时出现并自动吸附最优值 8（可手动改回）；未知平台执行时强制 1 |

### 输出与下游连接

两个下载节点输出一致：

| 输出 | 类型 | 说明 |
| --- | --- | --- |
| `video` | VIDEO | 直接接 `SaveVideo` 等所有 VIDEO 消费节点 |
| `fps` | FLOAT | 下载视频的帧率 |

> 故意**不提供** `images`/`audio` 全量输出：把 194s/1080p 视频解码成帧张量会产生 112GB+ 的 float32 张量，直接撑爆内存。需要帧的消费方请用专用采样节点。

### 保存位置

固定 `ComfyUI/output/musefish/`，文件名 `<平台>_<标题/视频ID>_<时间戳>.mp4`，重名自动加序号。

### 使用注意

- **X/Twitter 与 YouTube 必须走本地代理**（默认 `http://127.0.0.1:7897`，Clash Verge 混合端口；探测不可达时 X 回退直连、YouTube 直接报错）。YouTube 另需 cookie：节点从包目录 `_yt_cookies.txt`（Netscape 格式）读取，无 cookie 会被 bot 墙拒绝（"Sign in to confirm you're not a bot"）。
- **抖音 cookie**：包目录 `_dy_cookies.txt`（从登录浏览器 CDP 抓取，含 ttwid/s_v_web_id）；无 cookie 时 iesdouyin 移动接口返回 `video_layout: null`（bot 检测）。
- **yt-dlp 自更新**：包导入时后台线程检查更新（24h 一次，清华 pip 镜像，不阻塞启动）——YouTube/抖音提取器失效的第一原因就是 yt-dlp 过期。
- **`Musefish WC Video Channels`** 的 `link_or_url` 支持粘贴：媒体 CDN 链接、分享链接、分享文本、Channels API 的 JSON 响应，也可以完全留空（= 取最新播放记录）。缺失的信息（URL/decode_key）自动从本机运行中的客户端内存获取——**使用前先在客户端里播放目标视频几秒，并保持客户端运行**。
- **防自动连播错位（双链接法）**：客户端播完会自动播放下一个视频，最新内存记录未必是你链接的视频。可靠顺序：① 复制分享链接 A → ② 打开 A 播放几秒 → ③ 复制任一其他链接 B → ④ 粘贴 A 运行。最后复制 B 会把目标视频钉在"最近打开"位置，不受连播影响。
- **内存守护**：下载分块间检查整机内存占用，>95% 时暂停等待，避免把机器推进 swap。
- **URL 缓存**：恢复出的带 token URL 缓存在包目录 `_wc_url_cache.json`（微信客户端几分钟内会压实内存中的 URL 行，缓存保证之后仍可重下）；该 URL 是约 48 小时的临时访问凭证，**不要外发含它的工作流 JSON 或截图**。

### 下载相关文件

- `musefish_video_nodes.py`：两个下载节点的 schema、粘贴内容自动解析与执行编排。
- `musefish_video_download.py`：yt-dlp 下载（含分块并发）、加密通道下载/解密、客户端内存扫描、ftyp/ffprobe 验证、URL 缓存。
- `memory_guard.py`：整机内存压力守护（>95% 暂停下载）。
- `_startup_update.py`：yt-dlp 后台自更新（24h 一次）。
- `wxdec_toolchain/`：常驻密钥流守护进程（内含 wasm + Node 守护脚本，无需全局装依赖）。wasm 取自 [Evil0ctal/WeChat-Channels-Video-File-Decryption](https://github.com/Evil0ctal/WeChat-Channels-Video-File-Decryption)（MIT），本仓库的 daemon 为原创重写。
- `web/musefish_video_download_segments.js`：`chunks` 控件的平台感知显隐与自动吸附前端逻辑。
- `tests/test_video_download_nodes.py`：解密管线（以上游样本密文+key 为真值，`tests/fixtures/`）、key 校验门、Range 回退、粘贴解析、平台检测、分块并发参数、节点行为测试。
