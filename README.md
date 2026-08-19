# ComfyUI-Musefish-Nodes

ComfyUI 节点集合，当前提供 PiD 视频批处理超分节点。

## 节点

### Musefish PiD Batch Video Upscale

节点 ID：`MusefishPiDBatchVideoUpscale`

功能：将视频加载节点输出的 `IMAGE` 帧批量送入 PiD 模型，按原始顺序输出超分后的 `VIDEO` 与 `IMAGE`。可选音频和输入帧率会写入 `VIDEO` 输出。

节点会在一次执行中完成：

1. 按 `model_long_edge` 统一输入帧尺寸；
2. 使用输入 VAE 编码低分辨率帧；
3. 按 `batch_size` 分批执行 PiD 采样；
4. 使用代码内固定的 `pixel_space` VAE 解码超分结果；
5. 合并所有帧并保留音频、FPS。

扩散模型在节点执行开始时预加载，并在所有帧批次间复用同一个模型对象。

## 推荐连接

```text
VHS_LoadVideo
  ├── IMAGE ───────────────┐
  ├── AUDIO ───────────────┤
  └── VHS_VIDEOINFO.FPS ───┤
                            ▼
Musefish PiD Batch Video Upscale
  ├── MODEL      ← UNETLoader
  ├── CLIP       ← CLIPLoader(type=pixeldit)
  ├── encode_vae ← VAELoader(Flux\\UltraFlux-v1-vae.safetensors)
  └── 解码 VAE   ← 代码内固定为 pixel_space
                            │
                            ├── VIDEO → SaveVideo
                            └── IMAGE → 预览或视频合并节点
```

`encode_vae` 是唯一需要连接的 VAE 输入。解码端固定使用 ComfyUI 的 `pixel_space` VAE，不需要额外的 VAE 节点。

## PiD Flux 4x 推荐参数

| 参数 | 推荐值 |
| --- | ---: |
| `batch_size` | `1` 起步；显存足够时提高到 `2` 或 `4` |
| `model_long_edge` | `1024` |
| `upscale_factor` | `4` |
| `latent_format` | `flux` |
| `degrade_sigma` | `0.0` |
| `cfg` | `1.0` |
| `sampler_name` | `lcm` |
| `scheduler` | `simple` |
| `steps` | `4` |
| `positive_prompt` | `high quality, ultra detailed, sharp details` |

推荐模型：

```text
UNET:
PiD\\pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors

CLIP:
PixelDiT\\gemma_2_2b_it_elm_fp8_scaled.safetensors

encode_vae:
Flux\\UltraFlux-v1-vae.safetensors

decode VAE:
代码内固定为 `pixel_space`，无需连接节点
```

## 长视频处理建议

- 先将 `VHS_LoadVideo.frame_load_cap` 设为少量帧验证，例如 `2` 或 `4`。
- 确认输出尺寸和模型参数正确后，再增加帧数。
- 显存不足时优先降低 `batch_size`，不要改变帧顺序。
- `model_long_edge=1024`、`upscale_factor=4` 会产生约 4096 长边输出，显存与编码时间都明显增加。
- 通过 `VIDEO` 输出连接 `SaveVideo`，由 ComfyUI 统一编码和保存音频。

## 安装

将目录放入：

```text
ComfyUI/custom_nodes/ComfyUI-Musefish-Nodes
```

重启 ComfyUI 后，在节点搜索中查找：

```text
Musefish PiD Batch Video Upscale
```

## 文件

- `__init__.py`：ComfyUI 扩展入口
- `musefish_nodes.py`：节点实现
- `workflows/Musefish_PiD_Batch_Video_Upscale.json`：示例工作流
