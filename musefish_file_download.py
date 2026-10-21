"""Musefish direct-link downloader node (parallel ranges + resume).

Same engine as Studio's 「文件断点续传」 task: both import
``musefish_fastdl``, so a download started in ComfyUI and one started in
Studio behave identically (same chunking, same resume state, same proxy
detection, same error wording).

Why parallel ranges: CivitAI b2 / S3-compatible / ModelScope throttle *per
connection* to ~100 KB/s. 32-64 ranged requests aggregate to the link's real
bandwidth, and every completed 4 MiB chunk is recorded so a cancelled or
interrupted download continues instead of restarting.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import comfy.utils
from comfy_api.latest import io

from .musefish_fastdl import (
    DownloadError,
    guard_dest_dir,
    Progress,
    ProgressReporter,
    desktop_dir,
    detect_proxies,
    download,
    probe_with_route,
    safe_name,
)

def _send_progress_text(text: str, node_cls) -> None:
    """Push a live status line to the node's own widget (ComfyUI >= 0.3.60).

    Failure is silent on purpose: an older frontend or a headless run must
    still download, and the same text also goes to the console every 2 s.
    """
    try:
        from server import PromptServer

        node_id = getattr(getattr(node_cls, "hidden", None), "unique_id", None)
        if node_id:
            PromptServer.instance.send_progress_text(text, node_id)
    except Exception:  # noqa: BLE001
        pass


AUTO_LABEL = "自动（auto）"
DIRECT_LABEL = "直连（direct）"


def _proxy_choices() -> list[str]:
    """自动 / 直连 + 本机实际在监听的代理客户端（Clash、v2rayN、sing-box…）。"""
    choices = [AUTO_LABEL, DIRECT_LABEL]
    try:
        for item in detect_proxies():
            if item["label"] not in choices:
                choices.append(item["label"])
    except Exception:  # noqa: BLE001 — detection must never break node registration
        pass
    return choices


def _proxy_value(label: str) -> str:
    """把下拉里的标签还原成 musefish_fastdl 认识的值。

    ComfyUI 会把保存过的 combo 值直接回灌；旧工作流里的空串、未知标签
    都退回 `auto`，不因为一个下拉值让整个 prompt 失败。
    """
    text = (label or "").strip()
    if not text or text.startswith("自动"):
        return "auto"
    if text.startswith("直连"):
        return "direct"
    for item in detect_proxies():
        if item["label"] == text:
            return item["url"]
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return "auto"


def _allowed_roots() -> list[Path]:
    """Directories a workflow may write into without asking the user."""
    roots: list[Path] = []
    try:
        import folder_paths

        for getter in ("get_output_directory", "get_input_directory"):
            value = getattr(folder_paths, getter, None)
            if callable(value):
                roots.append(Path(value()))
        models = getattr(folder_paths, "models_dir", None)
        if models:
            roots.append(Path(models))
    except Exception:  # noqa: BLE001 — folder_paths is optional outside ComfyUI
        pass
    home = Path.home()
    roots.extend([home / "Desktop", home / "Downloads", home / "Documents"])
    roots.extend(Path(item) for item in os.environ.get("MUSEFISH_DOWNLOAD_DIRS", "").split(os.pathsep)
                 if item.strip())
    return [root for root in roots if str(root)]


def _output_dir(value: str) -> Path:
    """空值 → ComfyUI 的 output/musefish；相对路径按 ComfyUI 根目录解析。"""
    text = (value or "").strip().strip('"')
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            try:
                import folder_paths

                path = Path(folder_paths.get_output_directory()) / path
            except Exception:  # noqa: BLE001
                path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    try:
        import folder_paths

        path = Path(folder_paths.get_output_directory()) / "musefish"
    except Exception:  # noqa: BLE001
        path = desktop_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


class MusefishFileDownload(io.ComfyNode):
    """下载任意 http(s) 直链（签名 CDN 也行），并发分片 + 断点续传。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="musefish_file_downloader",
            display_name="Musefish File Download",
            search_aliases=["download", "文件下载", "断点续传", "safetensors", "civitai",
                            "checkpoint", "lora", "resume", "aria2"],
            category="Musefish/Download",
            # 下载本身就是终点：允许不接任何下游节点直接执行（schema.is_output_node）
            is_output_node=True,
            description=(
                "Download a direct http(s) URL — signed CDN links (CivitAI b2, "
                "S3, ModelScope, HuggingFace) included — with parallel ranged "
                "requests and resume. Per-connection throttled CDNs go from "
                "~100 KB/s to the link's real bandwidth; an interrupted run "
                "leaves a .part + state file, so re-running the same URL "
                "continues instead of restarting. Proxy=auto tries the local "
                "proxy client (Clash/v2rayN/sing-box/Shadowsocks) only when "
                "the direct route fails."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    multiline=True,
                    placeholder="https://b2.civitai.com/file/.../model.safetensors?Authorization=...",
                    tooltip=(
                        "完整直链，签名参数必须整串保留（缺 Authorization / "
                        "b2ContentDisposition 会 401）。注意：这类链接 HEAD 恒 401，"
                        "GET 才是有效判据。"
                    ),
                ),
                io.String.Input(
                    "save_dir",
                    default="",
                    multiline=False,
                    placeholder="留空 = output/musefish；也可填绝对路径，如 H:/ComfyUI/models/loras",
                    tooltip="留空下载到 ComfyUI 的 output/musefish；相对路径按 ComfyUI 根目录解析。",
                ),
                io.String.Input(
                    "filename",
                    default="",
                    multiline=False,
                    placeholder="留空则按 Content-Disposition / URL 自动识别",
                    tooltip="文件名覆盖（自动识别会做非法字符清洗）。",
                ),
                io.Int.Input(
                    "conns",
                    default=32,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="并发分片请求数。被限速的 CDN 上 32-64 才能吃满带宽；直连本地/内网时 4-8 足够。",
                ),
                io.Combo.Input(
                    "proxy",
                    options=_proxy_choices(),
                    default=AUTO_LABEL,
                    tooltip=(
                        "自动：先直连，失败改用本机代理并记住该通路；直连：绝不用代理；"
                        "其余为本机正在监听的代理客户端端口。"
                    ),
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    tooltip="目标文件已存在时：关=另存为「name (1).ext」；开=覆盖。",
                ),
                io.Boolean.Input(
                    "allow_any_dir",
                    default=False,
                    tooltip=(
                        "关（默认）：save_dir 只允许 ComfyUI 输出/模型目录、桌面、下载、文档；"
                        "开：允许任意可写路径。共享工作流里请保持关闭。"
                    ),
                ),
            ],
            # UNIQUE_ID lets the node push live progress text to its own widget.
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.String.Output("path", tooltip="下载完成后的绝对路径。"),
                io.String.Output("report", tooltip="尺寸/速度/并发/通路等一行摘要，便于日志核对。"),
            ],
        )

    @classmethod
    def validate_inputs(cls, url: str = "", proxy: str = "", conns=None, **kwargs) -> bool | str:
        """空 url 之外一律放行：旧工作流里的空 combo/数字值不应让 prompt 整体失败。"""
        if not (url or "").strip():
            return "url 不能为空：请粘贴 http(s) 直链"
        return True

    @classmethod
    def execute(cls, url: str, save_dir: str = "", filename: str = "",
                conns: int = 32, proxy: str = AUTO_LABEL, overwrite: bool = False,
                allow_any_dir: bool = False) -> io.NodeOutput:
        target_url = (url or "").strip().strip('"').strip("'")
        if not target_url.lower().startswith(("http://", "https://")):
            raise DownloadError("url 必须是 http(s) 直链")

        try:
            want_conns = max(1, min(64, int(conns or 32)))
        except (TypeError, ValueError):
            want_conns = 32

        bar = comfy.utils.ProgressBar(100)
        logged = {"at": 0.0}

        def on_progress(progress: Progress) -> None:
            """Live progress: bar position + a one-line text pushed to the node."""
            try:
                bar.update_absolute(int(progress.percent))
            except Exception:  # noqa: BLE001
                pass
            text = progress.text()
            _send_progress_text(text, cls)
            now = time.monotonic()
            if now - logged["at"] >= 2.0 or progress.done >= progress.total:
                logged["at"] = now
                print(f"[MusefishFileDownload] {text}", flush=True)

        reporter = ProgressReporter(on_progress)

        directory = guard_dest_dir(_output_dir(save_dir), _allowed_roots(), allow_any_dir)
        want = _proxy_value(proxy)
        info, used_proxy, route = probe_with_route(
            target_url, want, timeout=30,
            log=lambda line: print(f"[MusefishFileDownload] {line}", flush=True),
        )
        total = info["size"]
        print(
            f"[MusefishFileDownload] {(f'{total / 1048576:.1f} MB' if total else '大小未知')} · "
            f"分片{'支持' if info['ranges'] else '不支持'} · {route} · 并发 {want_conns}",
            flush=True,
        )

        name = safe_name(filename or info["filename"])
        dest = directory / name
        if dest.exists() and not overwrite:
            from .musefish_video_download import _unique_path

            dest = _unique_path(directory, Path(name).stem, Path(name).suffix)

        # ComfyUI 的 Cancel/Interrupt 必须真的中断下载（否则节点会闷头下完），
        # 中断后 .part + state 留着，重跑同参数即续传。
        try:
            from comfy.model_management import processing_interrupted
        except Exception:  # noqa: BLE001
            processing_interrupted = None
        result = download(target_url, dest, conns=want_conns, proxy=used_proxy,
                          timeout=45, on_progress=reporter,
                          should_stop=processing_interrupted,
                          overwrite=overwrite, probe_info=info)
        size = Path(result["path"]).stat().st_size
        report = (
            f"{result['path']} · {size / 1048576:.1f} MB · 并发 {want_conns} · {route}"
            + ("" if result.get("ranges", True) else " · 单流（服务端不支持 Range）")
            + (" · 已存在（跳过）" if result.get("skipped") else "")
        )
        print(f"[MusefishFileDownload] 完成 {result['path']} ({size} bytes)", flush=True)
        # ui.text 让摘要出现在 ComfyUI 的历史/节点面板里，便于事后核对
        return io.NodeOutput(str(result["path"]), report, ui={"text": [report]})

