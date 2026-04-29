"""Single-model inference CLI for the hose-anomaly-detection bundle.

Loads one of the four YOLO detectors, applies the model card's recommended
runtime profile (confidence + min/max area ratio), and prints / saves the
filtered detections.

Usage:
    python inference.py --model hose_wall --image path/to/img.jpg
    python inference.py --model hose_type_metal --image img.jpg --save out.jpg
    python inference.py --model hose_aging --image img.jpg --conf 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
KNOWN_MODELS = ["hose_aging", "hose_clamp", "hose_type_metal", "hose_wall"]


def load_card(model_name: str) -> dict:
    card_path = MODELS_DIR / model_name / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"missing model card: {card_path}")
    return json.loads(card_path.read_text())


def resolve_weights(model_name: str, prefer: str) -> Path:
    card = load_card(model_name)
    weights = card["weights"]
    if prefer in weights:
        return MODELS_DIR / model_name / weights[prefer]
    fallback = next(iter(weights.values()))
    return MODELS_DIR / model_name / fallback


def runtime_thresholds(card: dict) -> dict:
    profile = dict(card.get("recommended_runtime", {}))
    if card["name"] == "hose_type_metal":
        return {
            "per_class_conf": {
                "metal_clad_hose": profile.get("metal_clad_hose_conf", 0.5),
                "non_standard_hose": profile.get("non_standard_hose_conf", 0.45),
            },
            "min_area_ratio": profile.get("min_area_ratio", 0.001),
            "max_area_ratio": profile.get("max_area_ratio", 0.85),
            "prefer_high_risk_min_conf_gap": profile.get(
                "prefer_high_risk_min_conf_gap", 0.2
            ),
        }
    return {
        "conf": profile.get("min_conf", profile.get("conf", 0.25)),
        "min_area_ratio": profile.get("min_area_ratio", 0.0),
        "max_area_ratio": profile.get("max_area_ratio", 1.0),
    }


def filter_detections(detections, thresholds, image_area):
    """Apply per-class confidence + area-ratio gating used in production."""
    kept = []
    metal_clad = []
    non_standard = []
    for det in detections:
        area_ratio = det["area"] / image_area
        if area_ratio < thresholds.get("min_area_ratio", 0.0):
            continue
        if area_ratio > thresholds.get("max_area_ratio", 1.0):
            continue

        if "per_class_conf" in thresholds:
            min_conf = thresholds["per_class_conf"].get(det["class_name"], 1.01)
            if det["conf"] < min_conf:
                continue
            if det["class_name"] == "metal_clad_hose":
                metal_clad.append(det)
            elif det["class_name"] == "non_standard_hose":
                non_standard.append(det)
            else:
                kept.append(det)
        else:
            if det["conf"] < thresholds["conf"]:
                continue
            kept.append(det)

    if "per_class_conf" in thresholds:
        gap = thresholds["prefer_high_risk_min_conf_gap"]
        kept.extend(_merge_two_class(metal_clad, non_standard, gap))
    return kept


def _merge_two_class(safer, risky, gap):
    """For overlapping safer/risky boxes, prefer risky only if it beats safer by `gap`."""
    out = []
    used_safer = set()
    for r in risky:
        best_overlap = -1.0
        best_idx = None
        for i, s in enumerate(safer):
            if i in used_safer:
                continue
            iou = _iou(r["xyxy"], s["xyxy"])
            if iou > best_overlap:
                best_overlap = iou
                best_idx = i
        if best_idx is not None and best_overlap > 0.4:
            s = safer[best_idx]
            if r["conf"] - s["conf"] >= gap:
                out.append(r)
                used_safer.add(best_idx)
            else:
                out.append(s)
                used_safer.add(best_idx)
        else:
            out.append(r)
    for i, s in enumerate(safer):
        if i not in used_safer:
            out.append(s)
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter)


def run(model_name: str, image_path: Path, save_path: Path | None,
        conf_override: float | None, weights_pref: str) -> list[dict]:
    from ultralytics import YOLO

    card = load_card(model_name)
    weights_path = resolve_weights(model_name, weights_pref)
    model = YOLO(str(weights_path))
    imgsz = card.get("input_size", 640)

    base_conf = 0.05  # let the area/threshold layer downstream do real gating
    if conf_override is not None:
        base_conf = conf_override

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=base_conf,
        verbose=False,
    )
    if not results:
        return []

    r = results[0]
    h, w = r.orig_shape
    area_total = h * w
    raw = []
    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_idx = int(box.cls[0])
        cls_name = r.names[cls_idx]
        area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
        raw.append({
            "class_index": cls_idx,
            "class_name": cls_name,
            "conf": conf,
            "xyxy": xyxy,
            "area": area,
        })

    thresholds = runtime_thresholds(card)
    if conf_override is not None:
        if "per_class_conf" in thresholds:
            for k in thresholds["per_class_conf"]:
                thresholds["per_class_conf"][k] = conf_override
        else:
            thresholds["conf"] = conf_override
    kept = filter_detections(raw, thresholds, area_total)

    if save_path:
        _draw_and_save(image_path, save_path, kept)
    return kept


def _draw_and_save(image_path: Path, save_path: Path, detections: list[dict]) -> None:
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed; skipping visualization", file=sys.stderr)
        return
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"could not read image: {image_path}", file=sys.stderr)
        return
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['class_name']} {det['conf']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=KNOWN_MODELS)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--save", type=Path, default=None,
                        help="optional output path for the annotated image")
    parser.add_argument("--conf", type=float, default=None,
                        help="override the model card's recommended confidence")
    parser.add_argument("--weights", choices=["pytorch", "onnx"], default="pytorch")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    detections = run(args.model, args.image, args.save, args.conf, args.weights)
    print(json.dumps({"model": args.model,
                      "image": str(args.image),
                      "detections": detections},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
