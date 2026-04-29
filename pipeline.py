"""Run all four detectors on one image and emit a combined report.

Each model is loaded once and reused across images. Per-model runtime
thresholds come from the bundled model_card.json files (no flag tuning
needed). Output is a JSON document keyed by model name.

Usage:
    python pipeline.py --image path/to/img.jpg
    python pipeline.py --image img.jpg --save-dir runs/demo
    python pipeline.py --image img.jpg --models hose_wall hose_clamp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference import (
    KNOWN_MODELS,
    MODELS_DIR,
    filter_detections,
    load_card,
    resolve_weights,
    runtime_thresholds,
)

REPO_ROOT = Path(__file__).resolve().parent


class HosePipeline:
    """Holds one YOLO instance per model and runs them in series."""

    def __init__(self, models: list[str], weights_pref: str = "pytorch"):
        from ultralytics import YOLO

        self._models = {}
        self._cards = {}
        for name in models:
            card = load_card(name)
            weights = resolve_weights(name, weights_pref)
            self._models[name] = YOLO(str(weights))
            self._cards[name] = card

    def run(self, image_path: Path, save_dir: Path | None = None) -> dict:
        from inference import _draw_and_save  # reuse

        report = {"image": str(image_path), "results": {}}
        for name, model in self._models.items():
            card = self._cards[name]
            imgsz = card.get("input_size", 640)
            results = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=0.05,
                verbose=False,
            )
            if not results:
                report["results"][name] = []
                continue
            r = results[0]
            h, w = r.orig_shape
            area_total = h * w
            raw = []
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                raw.append({
                    "class_index": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                    "conf": float(box.conf[0]),
                    "xyxy": xyxy,
                    "area": area,
                })
            kept = filter_detections(raw, runtime_thresholds(card), area_total)
            report["results"][name] = kept

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                out = save_dir / f"{image_path.stem}_{name}.jpg"
                _draw_and_save(image_path, out, kept)

        report["summary"] = {
            name: {
                "hits": len(dets),
                "top_conf": max((d["conf"] for d in dets), default=None),
            }
            for name, dets in report["results"].items()
        }
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="if set, writes annotated images per model into this directory")
    parser.add_argument("--models", nargs="+", default=KNOWN_MODELS,
                        choices=KNOWN_MODELS)
    parser.add_argument("--weights", choices=["pytorch", "onnx"], default="pytorch")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    pipeline = HosePipeline(args.models, args.weights)
    report = pipeline.run(args.image, args.save_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
