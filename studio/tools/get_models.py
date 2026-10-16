"""Populate data/models/ for a dev machine: link model dirs from a source install.

End users get REAL model files inside data/models/ (downloaded by an installer).
On a dev machine we instead junction each model LEAF directory into the source
ComfyUI install, so nothing is copied and both installs read the same bytes.

Per-file linking is impossible without admin (file symlinks need the
SeCreateSymbolicLinkPrivilege); junctions need no privileges but only link
directories — so we link the leaf folder that contains exactly the needed files.

Usage:
  python tools/get_models.py            # junction model dirs from H: (dev mode)
  python tools/get_models.py --copy     # real files (what end users get)
  python tools/get_models.py --check    # report only
  python tools/get_models.py --from X   # custom source models root
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path(r"H:\ComfyUI_Mie_2026_V8.0\ComfyUI\models")

# workbench model dir (leaf) -> source dir that should back it
DIR_LINKS = {
    "diffusion_models/PiD": "diffusion_models/PiD",
    "text_encoders/PixelDiT": "text_encoders/PixelDiT",
    "vae/Flux": "vae/Flux",
    "dlss5": "dlss5",
    "UniverSR": "UniverSR",
}


def is_link(p: Path) -> bool:
    if not p.exists():
        return False
    return p.is_symlink() or os.path.realpath(p) != str(p)


def link_dir(dst: Path, src: Path) -> dict:
    entry = {"src": str(src), "mode": "junction"}
    if dst.exists():
        if is_link(dst):
            entry["state"] = "ok" if os.path.realpath(dst).lower() == str(src).lower() else "stale"
        else:
            entry["state"] = "real-dir"  # end-user layout: leave untouched
        return entry
    if not src.is_dir():
        entry["state"] = "missing-src"
        return entry
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True)
    entry["state"] = "ok" if dst.exists() else "failed: " + r.stdout.decode("gbk", "replace")
    return entry


def copy_dir(dst: Path, src: Path, check: bool) -> dict:
    entry = {"src": str(src), "mode": "copy"}
    if not src.is_dir():
        entry["state"] = "missing-src"
        return entry
    need = any(
        not (dst / f.relative_to(src)).is_file()
        or (dst / f.relative_to(src)).stat().st_size != f.stat().st_size
        for f in src.rglob("*") if f.is_file()
    )
    if not need:
        entry["state"] = "ok"
        return entry
    if check:
        entry["state"] = "absent"
        return entry
    shutil.copytree(src, dst, dirs_exist_ok=True)
    entry["state"] = "copied"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy real files instead of junctions (end-user mode)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--from", dest="src", default=str(DEFAULT_SRC))
    args = ap.parse_args()
    src_root = Path(args.src)

    report, ok = {}, True
    for rel_dst, rel_src in DIR_LINKS.items():
        dst, src = ROOT / "data" / "models" / rel_dst, src_root / rel_src
        if args.copy:
            entry = copy_dir(dst, src, args.check)
        elif args.check:
            # --check is report-only in every mode: never create junctions
            entry = link_dir(dst, src) if False else {"src": str(src), "mode": "junction"}
            if dst.exists() and is_link(dst):
                entry["state"] = "ok" if os.path.realpath(dst).lower() == str(src).lower() else "stale"
            elif dst.exists():
                entry["state"] = "real-dir"
            elif not src.is_dir():
                entry["state"] = "missing-src"
            else:
                entry["state"] = "missing-link"
        else:
            entry = link_dir(dst, src)
        report[rel_dst] = entry
        ok = ok and entry["state"] in ("ok", "copied", "real-dir")

    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
