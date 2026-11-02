# 视频下载节点

## 节点
- **Musefish Video Download**（`musefish_video_downloader`）：下载 YouTube、Bilibili、X/Twitter、抖音、小红书视频；分别使用 yt-dlp（YouTube/Bilibili/X）、内置 iesdouyin 分享页解析（抖音）、yt-dlp（小红书，失败时报错并建议浏览器提取）。返回前验证 MP4 `ftyp` 头及 ffprobe。
- **Musefish WC Video Channels**（`musefish_wc_video_channels`）：微信视频号下载解密；自动解析粘贴内容、扫描客户端内存取得 URL/decode_key、下载密文、校验 `ftyp`、解密并用 ffprobe 验证。失败即停止；错误 key 不会下载完整文件或输出未验证文件。

## Musefish Video Download 输入
| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `url` | STRING（多行） | CDN 直链、分享链接/文本；可含标题，自动提取 URL。 |
| `platform` | COMBO | `Auto`（默认，自动检测）/ YouTube / Bilibili / Douyin / Xiaohongshu / X-Twitter；手动选择与 URL 不符时报错。 |
| `quality` | COMBO | `best`（默认）/ `1080p` / `720p` / `480p`；直链忽略。 |
| `chunks` | COMBO | `1` / `2` / `4` / `8` / `16`，默认 `4`。抖音直链用 N 段并发 Range；YouTube/Bilibili 用 yt-dlp `concurrent_fragment_downloads`。URL 为空或平台不支持时隐藏；抖音/YouTube/Bilibili 自动选 `8`（可改），未知平台执行时强制 `1`。 |

## 输出与保存
两个节点均输出 `video`（file-backed VIDEO，可接 `SaveVideo` 等）和 `fps`（FLOAT，视频帧率）。文件保存至 `ComfyUI/output/musefish/`，命名 `<平台>_<标题/视频ID>_<时间戳>.mp4`，重名加序号。不输出全量 `images`/`audio`：194s/1080p 解码成 float32 帧张量超过 112GB；需帧请用采样节点。

## 依赖与使用
- yt-dlp：`pip install -U yt-dlp`；导入插件时后台每 24h 检查更新，使用清华 pip 镜像，不阻塞启动。过期常导致 YouTube/抖音提取失败。
- X/Twitter、YouTube 使用本地代理 `http://127.0.0.1:7897`（Clash Verge 混合端口）；不可达时 X 回退直连，YouTube 报错。YouTube 还需包目录 `_yt_cookies.txt`（Netscape 格式），否则可能遇 bot 墙。
- 抖音需包目录 `_dy_cookies.txt`，含登录浏览器 CDP 获取的 `ttwid`、`s_v_web_id`；缺失时 iesdouyin 移动接口返回 `video_layout: null`（bot 检测）。
- 小红书用 yt-dlp 尝试，失败则报错并建议浏览器提取。
- 视频号 `link_or_url` 接受 CDN 链接、分享链接/文本、Channels API JSON，也可留空使用最新播放记录；缺失 URL/decode_key 从本机客户端内存恢复。使用前须在本机客户端打开并播放目标视频几秒，且保持客户端运行。为防自动连播错位：复制并播放目标链接 A 几秒，再复制其他链接 B，粘贴 A 运行。
- `chunks` 间检查整机内存，使用率 >95% 时暂停等待。恢复的 token URL 缓存在包目录 `_wc_url_cache.json`，约 48 小时有效；cookie 与含 token 的 URL 都是凭据，勿提交、外发或放入共享工作流 JSON/截图。
- 工作流：`workflows/Musefish_Video_Downloader.json`。加载后按需填写下载链接、选择平台/质量/分块，并连接 `video`、`fps` 下游；运行视频号节点前按上文在本机客户端播放目标视频。
