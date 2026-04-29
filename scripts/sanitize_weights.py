"""Strip private training metadata from shipped Ultralytics YOLO checkpoints
in-place (.pt only; ONNX exports are sanitized manually if needed).

What it removes:
  - top-level ``git`` block (training-host git remote)
  - any string field in ``train_args`` / ``model.args`` that looks like an
    absolute path (``/home/...``, ``/Users/...``, ``C:\\...``, etc.)
  - common path-bearing keys (``model``, ``data``, ``project``, ``name``,
    ``save_dir``, ``source``, ``weights``, ``cfg``, ``tracker``)
  - ``model.pt_path`` / ``model.ckpt_path`` / ``model.yaml_file``

What it keeps: network parameters, class names, ``imgsz``, ``task`` — i.e.
everything inference actually needs. Idempotent.

Usage::

    python3 scripts/sanitize_weights.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO_ROOT / "models/hose_aging/hose_aging_best.pt",
    REPO_ROOT / "models/hose_clamp/hose_clamp_best.pt",
    REPO_ROOT / "models/hose_type_metal/hose_type_metal_best.pt",
    REPO_ROOT / "models/hose_wall/hose_wall_best.pt",
]
PATH_KEYS = {"model", "data", "project", "name", "save_dir", "source",
             "weights", "cfg", "tracker"}
ABS_PATH_RE = re.compile(r"^(/|[A-Za-z]:\\)|/home/|/Users/|/root/|/mnt/|/opt/")


def looks_like_path(value: str) -> bool:
    return bool(ABS_PATH_RE.search(value))


def sanitize_args(args: dict) -> dict:
    cleaned = {}
    for k, v in args.items():
        if k in PATH_KEYS:
            cleaned[k] = ""
        elif isinstance(v, str) and looks_like_path(v):
            cleaned[k] = ""
        else:
            cleaned[k] = v
    return cleaned


def sanitize_pt(path: Path) -> None:
    import torch  # imported lazily so the script still imports without torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"  skipping {path.name}: not a dict checkpoint")
        return

    ckpt.pop("git", None)
    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = sanitize_args(ckpt["train_args"])

    model = ckpt.get("model")
    if model is not None:
        if hasattr(model, "args") and isinstance(model.args, dict):
            model.args = sanitize_args(model.args)
        for attr in ("pt_path", "ckpt_path", "yaml_file"):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, "")
                except Exception:
                    pass

    torch.save(ckpt, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    for p in TARGETS:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        before = p.stat().st_size
        sanitize_pt(p)
        after = p.stat().st_size
        print(f"sanitized {p.relative_to(REPO_ROOT)} ({before} -> {after} bytes, "
              f"sha256={sha256(p)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
