# Musefish Studio

ComfyUI-Musefish-Nodes 插件的本地客户端。开箱即用，无需打开 ComfyUI 网页端——
图片/视频/音频增强与视频下载全部在工作台界面完成。

**功能**：PiD 图片超分 · PiD 视频超分 · DLSS5 图片增强 · DLSS5 视频增强 ·
UniverSR 音频超分 · 视频下载（通用平台 yt-dlp + 微信视频号内存解密）

## 架构

```
Electron 客户端(client/) ──┐
浏览器 http://127.0.0.1:17995 ──┴→ Studio 服务(server/, 17995)
                                        ├→ headless ComfyUI(17996, 隔离 base-directory)
                                        └→ 直连任务(yt-dlp / 微信解密, 不经引擎)
```

- **单一事实源**：插件算法代码零复制，`data/custom_nodes/ComfyUI-Musefish-Nodes`
  是指向源插件目录的 junction；客户端只做编排，参数枚举与插件源码逐字对齐。
- **全链接映射**：Python 环境、ComfyUI 内核、插件全部经 NTFS junction 引入
  （`tools/link_models.py`，无需管理员权限），项目目录内没有 GB 级拷贝。
- **模型两种部署形态**（`tools/get_models.py`）：
  - 开发机（默认）：模型目录 junction 到本机 ComfyUI 安装，零拷贝；
  - 终端用户（`--copy`）：`data/models/` 内是真实模型文件——分发即拷贝该文件夹。
- **隔离运行**：headless ComfyUI 以 `--base-directory data/` 启动，只加载 Musefish
  一个插件，独立 `input/ output/ temp/ user/`，与本机其他 ComfyUI 实例互不干扰。
- **引擎可换**：设置面板可连接本机已装的任意 ComfyUI（手动填写或自动扫描端口），
  选择持久化到 `server/config.json`。

## 目录

| 目录 | 内容 |
|---|---|
| `server/` | Studio HTTP 服务（stdlib，17995）+ ComfyUI 监督器 + 工作流构建 |
| `client/` | Electron 壳：健康探测复用既有服务，否则拉起 server |
| `runtime/` | `python/` 内嵌解释器 junction（`._pth` 约束的兄弟布局） |
| `comfy/` | 源 ComfyUI 内核 junction（引擎启动入口） |
| `data/` | 隔离 base-directory：custom_nodes、models、input/output/temp/user |
| `tools/link_models.py` | 代码/运行时 junction 建链（幂等，`--check` 校验） |
| `tools/get_models.py` | 模型填充：默认 junction；`--copy` 生成终端用户形态 |
| `uploads/`、`logs/` | 上传暂存、运行日志（运行时生成） |

## 使用

**获取客户端（推荐）**：从 [GitHub Releases](https://github.com/Brian98Z/ComfyUI-Musefish-Nodes/releases/latest) 下载
`MusefishStudio-Client-<版本>-win-x64.exe`（免安装 portable）；完整性用同发布页的
`SHA256SUMS.txt` 校验（PowerShell：`Get-FileHash .\MusefishStudio-Client-*-win-x64.exe`）。

**终端用户（自包含形态，无需安装 ComfyUI）**

```
python tools\env_setup.py        # 一键准备：venv + ComfyUI 内核 + 全部依赖（首装约 15-20 GB 下载）
python tools\download_models.py  # 下载缺失模型（ModelScope/HuggingFace，约 12.7 GB）
start_server.bat                 # 启动服务（或由客户端自动拉起）
```

也可在界面 **设置 → ❷ 环境准备** 里点击两个按钮完成同样的事（进度显示在按钮旁）。
需要本机已装 Python 3.11+ x64（含 venv 组件）与 Git；venv 与引擎都落在
`runtime/venv`、`data/comfyui`，不依赖任何外部 ComfyUI 安装。

环境准备细节：
- ComfyUI 内核 pin 在 commit `b0b743566`（v0.39.0，与开发机验证版本一致）——主线后续版本
  已移除 `comfy.ldm.pixeldit`，而插件 PiD 节点硬依赖该模块，勿随意升级内核。
- pip 分两阶段：torch 单独走 pytorch.org cu130 源（2 GB 级 wheel 失败不连累其余），
  依赖与插件走镜像（默认清华，超时 60s 重试 5 次，防单包挂死拖垮整轮）。
- 克隆完成后会安装内核自带的 `requirements.txt`，与 bundle 清单做权威对齐
  （如 `comfyui-frontend-package` 版本必须跟随内核要求）。
- 模型目录 `data/models/` 下按类别放子目录（或指向外部模型库的 NTFS 符号链接/junction，
  即「映射链接方式引入」，不复制文件）；`data/extra_model_paths.yaml` 会把它挂进引擎。

**开发机（链接形态，零拷贝）**

```
python tools\link_models.py     # 首次：建立全部映射（校验：--check）
start_server.bat                # 启动：Studio 17995 + headless ComfyUI 17996
client\start_client.bat         # 桌面客户端（首次需 client\> npm install）
```

引擎目录解析链：`config.comfy_dir` → `data/comfyui`（自包含）→ `comfy` junction（开发机）；
解释器解析链：`runtime/venv`（自包含）→ `runtime/python` junction（开发机）。

## API（本机）

| 方法 | 路由 | 说明 |
|---|---|---|
| GET | `/api/status` | server/引擎/链接状态 |
| POST | `/api/upload` | multipart 字段 `file`，返回 `{path,name,bytes}` |
| POST | `/api/jobs` | `{task, input_path, params}`；task ∈ pid / pid_video / dlss5_image / dlss5_video / universr / video_download / wechat_download |
| GET | `/api/jobs` · `/api/jobs/<id>` | 任务列表 / 详情（pending→running→done/error） |
| DELETE | `/api/jobs/<id>` | 取消运行中任务 |
| DELETE | `/api/outputs` | 一键清理：已完成任务连同产物/记录一并删除（运行中保留） |
| GET | `/api/models?kind=…` | 枚举映射后的模型文件 |
| GET | `/api/nodes/<Class>` | 透传引擎 `/object_info/<Class>`（参数 schema 单一来源） |
| POST | `/api/comfy/start` · `/api/comfy/stop` | 引擎手动启停 |
| POST | `/api/comfy/endpoint` | 切换引擎 `{host,port}`，持久化 |
| POST | `/api/comfy/scan` | 自动扫描本机 ComfyUI 端口 |
| GET | `/files/<job_id>/<file>` | 产物下载/内嵌播放（支持 HTTP Range）；路径解析后必须落在 output 目录内，非媒体类型按附件返回 |

**写操作鉴权**：`POST`/`PUT`/`DELETE` 必须带 `X-Musefish-Client: 1`。Studio 前端一直携带，
作用是拦住浏览器里跨站页面伪造的简单请求（跨域自定义头会触发预检，预检不放行）。
自己写脚本调用时同样要加这个头。

## 冒烟清单

1. `python tools/link_models.py --check` 全 ok
2. 启动后 `/api/status` 中 `comfy.running=true`（首启 30-90 s）
3. 视频下载：B 站链接 → done → 客户端内可直接播放（Range + MIME 已处理）
4. DLSS5 图片：上传任意图 → done → 出图（秒级）
5. UniverSR：上传音频 → done → 48 kHz FLAC
6. PiD 图片：上传图 → done → 4× 超分（首启加载 UNET+CLIP 约 1-2 分钟）
7. PiD 视频：上传短视频 → done → 4K mp4（含原音频；耗时随帧数线性）
8. 微信视频号：本机微信播放目标视频后，留空提交 → 自动解密下载

## 已知边界

- DLSS5 需 RTX 50 系（Blackwell）+ NGX 驱动；DLL 由 `data/models/dlss5` 提供。
- 微信视频号下载依赖本机微信桌面客户端（密钥仅存在于其进程内存）。
- ffmpeg：默认从 PATH 解析；可在 `server/config.json` 的 `ffmpeg_dir` 指定
  （开发机指向 H 盘 ComfyUI 旁置 ffmpeg）。
- 8K PiD/DLSS5 任务显存/内存需求高；UniverSR `stem_mix` 需 Demucs 权重在位。
