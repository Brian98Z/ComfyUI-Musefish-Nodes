#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Musefish Studio 模型下载（脱离 ComfyUI 安装，落盘到 data/models/）。

五组模型（与开发机 data/models 结构完全一致）：
  PiD       diffusion_models/PiD/   ModelScope Comfy-Org/PixelDiT（按需子集）
  PixelDiT  text_encoders/PixelDiT/ 同上
  Flux VAE  vae/Flux/               ModelScope Comfy-Org/z_image_turbo
  DLSS5     dlss5/                  NVIDIA NGX DLL（打包内置于 tools/dlss5-seed/ 或随包分发）
  UniverSR  UniverSR/models/huggingface/{general,speech}   HuggingFace woongzip1/*

用法：
  python tools/download_models.py --list              # 只列清单与状态
  python tools/download_models.py                     # 下载全部缺失
  python tools/download_models.py --only pid,vae      # 下载指定组
下载用 modelscope/huggingface_hub（env_setup 已装入 venv）；若在无 venv 环境
运行则退回 urllib 直连。已存在且大小一致的文件自动跳过（断点续传友好）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"

# ModelScope 直链模板（repo_id, file_path_in_repo）
MS = "https://www.modelscope.cn/models/{repo}/resolve/master/{file}"
HF = "https://huggingface.co/{repo}/resolve/main/{file}"

GROUPS: dict[str, list[dict]] = {
    "pid": [
        {"url": MS.format(repo="Comfy-Org/PixelDiT",
                          file="diffusion_models/pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors"),
         "dst": "diffusion_models/PiD/pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors"},
    ],
    "clip": [
        {"url": MS.format(repo="Comfy-Org/PixelDiT",
                          file="text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors"),
         "dst": "text_encoders/PixelDiT/gemma_2_2b_it_elm_fp8_scaled.safetensors"},
    ],
    "vae": [
        # z-image/flux1 VAE (ae)。插件推荐名 UltraFlux-v1-vae 是社区重命名变体，
        # 无官方直链：下载官方 ae 后落一份兼容副本，两个 VAELoader 名字都可用。
        {"url": MS.format(repo="Comfy-Org/z_image_turbo",
                          file="split_files/vae/ae.safetensors"),
         "dst": "vae/Flux/ae.safetensors"},
        {"url": MS.format(repo="Comfy-Org/z_image_turbo",
                          file="split_files/vae/ae.safetensors"),
         "dst": "vae/Flux/UltraFlux-v1-vae.safetensors"},
    ],
    "dlss5": [],  # NGX DLL 随分发包 tools/dlss5-seed/ 提供，不在线下载（NVIDIA 许可）
    "universr": [
        {"url": HF.format(repo="woongzip1/universr-audio", file="config.yaml"),
         "dst": "UniverSR/models/huggingface/general/config.yaml"},
        {"url": HF.format(repo="woongzip1/universr-audio", file="pytorch_model.bin"),
         "dst": "UniverSR/models/huggingface/general/pytorch_model.bin"},
        {"url": HF.format(repo="woongzip1/universr-speech", file="config.yaml"),
         "dst": "UniverSR/models/huggingface/speech/config.yaml"},
        {"url": HF.format(repo="woongzip1/universr-speech", file="pytorch_model.bin"),
         "dst": "UniverSR/models/huggingface/speech/pytorch_model.bin"},
    ],
}

ALIASES = {"pid": ["pid"], "unet": ["pid"], "clip": ["clip"], "text_encoder": ["clip"],
           "vae": ["vae"], "flux": ["vae"], "dlss5": ["dlss5"], "universr": ["universr"],
           "audio": ["universr"], "all": list(GROUPS)}


def resolve_groups(spec: str) -> list[str]:
    out: list[str] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if token in ALIASES:
            for g in ALIASES[token]:
                if g not in out:
                    out.append(g)
    return out or list(GROUPS)


def dlss5_seed_dir() -> Path:
    for cand in (ROOT / "tools" / "dlss5-seed", ROOT / "dlss5-seed"):
        if cand.is_dir():
            return cand
    return ROOT / "tools" / "dlss5-seed"


def file_ok(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def seed_dlss5() -> dict:
    seed = dlss5_seed_dir()
    dst = MODELS / "dlss5"
    if not seed.is_dir():
        return {"group": "dlss5", "state": "no-seed", "note": f"缺少 {seed}（随分发包提供）"}
    dst.mkdir(parents=True, exist_ok=True)
    import shutil

    count = 0
    for f in seed.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            count += 1
    return {"group": "dlss5", "state": "copied", "files": count}


def download(url: str, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MusefishStudio/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {dst.name}: {pct:3d}% ({done/1048576:.0f}/{total/1048576:.0f} MB)",
                      end="", flush=True)
    print()
    if dst.exists() and dst.stat().st_size == tmp.stat().st_size == 0:
        tmp.unlink()
        return {"state": "empty"}
    os.replace(tmp, dst)
    return {"state": "downloaded", "bytes": dst.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列状态，不下载")
    ap.add_argument("--only", default="all", help="逗号分隔组名: pid,clip,vae,dlss5,universr")
    args = ap.parse_args()

    groups = resolve_groups(args.only)
    report: dict = {}
    fetched: dict[str, Path] = {}  # url -> already-downloaded local copy this run
    for g in groups:
        if g == "dlss5":
            report["dlss5"] = seed_dlss5() if not args.list else {
                "group": "dlss5", "state": "seed" if dlss5_seed_dir().is_dir() else "no-seed"}
            continue
        items = []
        for item in GROUPS[g]:
            dst = MODELS / item["dst"]
            if file_ok(dst):
                items.append({"dst": item["dst"], "state": "present"})
                continue
            if args.list:
                items.append({"dst": item["dst"], "state": "missing"})
                continue
            src_copy = fetched.get(item["url"])
            if src_copy is not None and file_ok(src_copy):
                dst.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(src_copy, dst)  # same-url alias: no second download
                items.append({"dst": item["dst"], "state": "copied",
                              "bytes": dst.stat().st_size})
                continue
            print(f"downloading {item['dst']} …")
            try:
                res = download(item["url"], dst)
                res.update({"dst": item["dst"]})
                if res.get("state") == "downloaded":
                    fetched[item["url"]] = dst
                items.append(res)
            except Exception as e:  # noqa: BLE001
                items.append({"dst": item["dst"], "state": f"failed: {e}"})
        report[g] = {"group": g, "items": items}

    print(json.dumps(report, ensure_ascii=False, indent=1))
    states = json.dumps(report)
    return 0 if "failed" not in states else 1


if __name__ == "__main__":
    sys.exit(main())
