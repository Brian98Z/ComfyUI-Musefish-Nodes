# Musefish DLSS5 超分节点 — 使用说明

> **硬件要求：NVIDIA RTX 50 系（Blackwell）独占。** DLSS 5 超分
> （NGX Feature 18）与配套的 RTX 视频超分运行库只随 Blackwell 驱动
> 提供；40 系及更早显卡无法创建该 Feature，节点会直接报错。
> 已验证环境：RTX 5070 Ti + 616.56 驱动。

## 这是什么

把 DLSS5Tool（`D:\DLSS5Tool`，MIT 源码 + NVIDIA 专有 DLL）的 DLSS 5 超分
（NGX Feature 18）封装成的 ComfyUI 节点，已集成到
`ComfyUI-Musefish-Nodes`，节点名 **`MusefishDLSS5NeuralRender`**
（分类 `Musefish/Video`）。

## 文件

- `musefish_dlss5.py` — 节点本体（schema + execute）
- `dlss5_backend/session.py` — 会话管理（子进程生命周期、共享内存帧协议）
- `dlss5_backend/worker.py` — 子进程入口（ctypes 驱动 DLL + 共享内存 pinned 注册）
- `dlss5_backend/dlls/` — NVIDIA 专有运行库（**不入库**，`.gitignore` 已排除）。手动部署：从 DLSS5Tool 安装目录的 `_internal/` 复制 `dlssnr_host.dll`、`nvngx_dlssnr.dll`、`vsr_host.dll`、`nvngx_vsr.dll` 四个 DLL 到该目录即可。
- `workflows/Musefish_DLSS5_Video_Segments.json` — 长视频分段增强案例工作流

## 架构

NGX 宿主在**每次节点执行时拉起的一次性子进程**里运行，ComfyUI 主进程
永不加载 NVIDIA DLL——驱动崩溃不会带垮服务。一次执行 = 一个会话 =
一段完整时序历史（等同 DLSS5Tool「严格时序（单会话）」导出模式）。

参数与 DLSS5Tool 一一对应：

| 节点参数 | DLSS5Tool 对应 |
|---|---|
| style: default/natural/cinema | 默认/自然/电影 |
| intensity (0–1) | 强度 |
| local_tone (0–1) | 本地色调 |
| local_struct (0–1) | 本地结构 |
| skin_struct (0–1) | 皮肤蒙版强度 |
| use_auto_mask | 皮肤蒙版开关 |
| reset_every_n_frames (0=never) | 时序历史重置（0 为单会话连续） |
| super_resolution: off/1.5x/2x/4x | RTX 视频超分（先超分再增强，输出 = 输入 × 倍数；1.5x 为合成档：VSR 2× + 增强 + INTER_AREA 缩回，输出 = 输入 × 1.5） |
| vsr_quality: performance/balanced/quality/ultra | VSR 质量档（NGX PerfQuality 1–4；ultra 为 DLSS5Tool 默认） |
| keep_session: auto/off | auto 跨执行保留同契约的热 worker（省 ~3s NGX 引导/次）；off 每次冷启动严格隔离 |

超出 [0,1] 的「5× 实验范围」参数已被夹紧（对应 GUI 默认关闭实验范围）。

## 已知限制

1. **必须 v2 宿主的现象**：本机（RTX 5070 Ti + 616.56 驱动）实测
   `dlssnr_host_v2.dll` 首次 `dlssnr_process` 内部等待不返回
   （疑似与 NVIDIA App overlay 的 NGX 组件冲突），**legacy 宿主
   `dlssnr_host.dll` 完全正常**（256px 约 4ms/帧），故 worker 固定用
   legacy。若日后升级驱动想试 v2，改回 `worker.py` 里的 DLL 名即可。
2. 仅支持 SDR RGBA8 路径；HDR 高精度（RGBA16F/PQ/HLG）路径未封装。
3. 单帧上限 4K（3840×2160，含超分后目标尺寸）。
4. 首次运行每个分辨率会触发 NGX 权重准备，首帧可能显著变慢，
   之后缓存生效。
5. 帧传输走共享内存，worker 侧把两个环缓冲 `cudaHostRegister` 为页锁定
   （pinned）内存，NGX 宿主的 D3D12 每帧 DMA 上传/下载跳过分页暂存拷贝
   （1080p off 档实测稳态 23.0ms → 21.1ms/帧，约 +8%；找不到 CUDA runtime
   时自动退回普通模式，功能无影响）。GPU 独占时实测：off 1080p ≈21ms/帧、
   2× 超分到 4K ≈70ms/帧。GPU 被其他任务占用时按占比变慢；
   attention/PyTorch 加速对该引擎不适用（NGX 为 D3D12 闭源推理）。
6. 节点内部按帧流式处理（逐帧转换 RGBA、直接写入输出张量），
   自身峰值内存 ≈ 上游输入批次 + 输出批次。上游 VHS_LoadVideo
   不设 frame_load_cap 时会把整段视频装进内存，长视频请配合
   frame_load_cap/skip_first_frames 分段。

## 长视频推荐用法（分段工作流）

`workflows/Musefish_DLSS5_Video_Segments.json` 演示了长视频的推荐接法：

```
VHS_LoadVideoFFmpegPath (frame_load_cap=每段帧数, start_time=段起点)
  → MusefishDLSS5NeuralRender (模板示例 keep_session=off；相邻段批量跑可改 auto)
    → VHS_VideoCombine (audio 直通, frame_rate 取自 VHS_VideoInfoSource)
```

- **每段独立执行**：内存峰值 ≈ 段帧数 × 单帧，而非整片；长视频不会爆内存。
- **keep_session=auto**（模板示例为 `off`）：相邻段复用热 worker，跳过 ~3s NGX 冷启动；
  段首帧自动重置时序历史，段间接缝与冷启动输出一致。
- 想跑全片：把 `start_time` 递增（秒），段长保持不变，最后用
  VHS_VideoCombine 的文件或外部工具拼接。
- `frame_load_cap=0` 表示不限制（小视频可一把跑完）。

## 适用范围与案例

**图像超分、视频超分均可使用**：节点输入是标准 IMAGE 批次——
LoadImage 出来的单张/多张图、VHS 系列节点解出的视频帧序列都能直接接入；
输出保持批次结构，图像接 SaveImage、视频接 VHS_VideoCombine 即可。

案例效果见 [`assets/DLSS5超分案例.mp4`](assets/DLSS5超分案例.mp4)
（1472×1280@24fps，10 秒；经节点 2× RTX VSR + Feature 18 处理）。

## 验证记录（2026-09-20）

- 8988 实例 API 提交 LoadImage → MusefishDLSS5NeuralRender → SaveImage
  全链路 success，输出 `output/dlss5_test_00001_.png`。
- 与原图逐像素对比：54.4% 像素变化 >8/255，mean abs diff 8.18，
  视觉上细节增强符合 Feature 18 预期。
- 2× RTX VSR + Feature 18 全链路 success，输出 1728×2304
  （`output/dlss5_sr2x_00001_.png`），四个 VSR 质量档均实测通过。
- 共享内存 IPC 替换 PNG 后复测：8988 全链 off/2x 均 success
  （`shm_off_00001_.png` / `shm_2x_00001_.png`），尺寸正确
  （864×1152 / 1728×2304）；GPU 独占时 off 1080p ≈18.5ms/帧、
  2×到 4K ≈70ms/帧。
- 1.5x 合成档 + 会话缓存：864×1152 输入 1.5× 输出 1296×1728 success；
  连续两次执行 2.3s → 1.0s（缓存命中省 NGX 冷启动）；离线基准
  热会话首帧 34ms vs 冷启动 1271ms（37×）。
- pinned 共享内存上线后回归：39 passed / 4 subtests（pytest，嵌入式
  Python，`--import-mode=importlib`）；开/关会话 3 轮循环无泄漏；
  真实用户负载（1080p→2x，keep_session=auto）在新代码下连续成功出片。
