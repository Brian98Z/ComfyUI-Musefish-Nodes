"""Create directory junctions mapping H-disk ComfyUI assets into this workbench.

Junctions (mklink /J) need no admin rights and work across local NTFS drives.
Everything heavy (Python env, plugin code, models) stays on H:; this project only
holds thin links plus its own mutable data (input/output/temp/user).

Usage:
  python tools/link_models.py            # create all missing links
  python tools/link_models.py --check    # report only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

H_COMFY = Path(r"H:\ComfyUI_Mie_2026_V8.0")
H_ROOT = H_COMFY / "ComfyUI"

# target (under this repo) -> source (on H:)
# NOTE: only *code/runtime* assets are junctioned. Models under data/models/
# are real files owned by the workbench (tools/get_models.py populates them),
# mirroring what an end user has after downloading.
LINKS = {
    "runtime/python": H_COMFY / "python_embeded",
    # python_embeded's python313._pth lists "../ComfyUI": the interpreter must
    # sit beside a ComfyUI checkout, so mirror that layout under runtime/.
    "runtime/ComfyUI": H_ROOT,
    "comfy": H_ROOT,
    "data/custom_nodes/ComfyUI-Musefish-Nodes": H_ROOT / "custom_nodes/ComfyUI-Musefish-Nodes",
}

# Created as real directories (workbench-private mutable state).
MKDIRS = [
    "data/input",
    "data/output",
    "data/temp",
    "data/user",
    "data/models/audio_encoders",  # placeholder dir so folder_paths finds nothing extra
    "server/web",
    "client",
    "logs",
    "uploads",
]

def _is_junction(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(path.stat().st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except OSError:
        return False


def junction_create(link: Path, target: Path) -> None:
    """Create a directory junction via mklink /J (no admin required on NTFS)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    if r.returncode != 0 or not link.exists():
        raise OSError(r.stdout.decode("gbk", "replace") + r.stderr.decode("gbk", "replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    ok = True
    for d in MKDIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    status = {}
    for rel, target in LINKS.items():
        link = ROOT / rel
        entry = {"target": str(target), "ok": False}
        if link.exists():
            entry["ok"] = True
            entry["existing"] = _is_junction(link)
        elif not target.exists():
            entry["error"] = f"source missing: {target}"
            ok = False
        elif args.check:
            entry["error"] = "missing link"
            ok = False
        else:
            try:
                junction_create(link, target)
                entry["ok"] = link.exists()
            except OSError as e:
                entry["error"] = str(e)
                ok = False
        status[rel] = entry

    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
