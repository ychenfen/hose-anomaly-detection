# hose-anomaly-detection

Four YOLO detectors for inspecting flexible hoses in indoor inspection imagery.
Each model targets one anomaly type and ships with a recommended runtime
profile (confidence threshold and area gates) tuned on hand-reviewed truth
sets, so callers do not have to discover thresholds themselves.

| Model | Classes | Input | What it flags | Runtime conf |
|---|---|---|---|---|
| `hose_aging` | `hose aging` | 832 | Visible aging / oxidation / cracking / corrosion / tearing | 0.50 |
| `hose_clamp` | `clamp_abnormal` | 640 | Loose, missing, or otherwise abnormal hose clamps | 0.35 |
| `hose_type_metal` | `metal_clad_hose`, `non_standard_hose` | 640 | Metal-clad hose vs. non-standard hose (two-class) | 0.50 / 0.45 |
| `hose_wall` | `hose penetrating the wall` | 640 | Hose routed through a wall penetration | 0.45 |

All four are Ultralytics YOLO detection models. Three of the four also ship
ONNX exports next to the PyTorch weights for non-Python deployments. Total
bundle size is ~130 MB.

## Layout

```
hose-anomaly-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── inference.py            # single-model CLI
├── pipeline.py             # run all four detectors at once
├── verify_checksums.py     # SHA-256 verifier
├── SHA256SUMS.txt
└── models/
    ├── hose_aging/         # .pt + classes.txt + model_card.json
    ├── hose_clamp/         # .pt + .onnx + classes.txt + model_card.json
    ├── hose_type_metal/    # .pt + .onnx + classes.txt + model_card.json
    └── hose_wall/          # .pt + .onnx + classes.txt + model_card.json
```

`model_card.json` is the source of truth for each model: classes,
recommended thresholds, training notes, and held-out metrics. The CLIs read
it directly so you do not have to repeat numbers in scripts.

## Install

```bash
git clone https://github.com/ychenfen/hose-anomaly-detection.git
cd hose-anomaly-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 verify_checksums.py
```

`verify_checksums.py` is optional but a good sanity check after a fresh clone
- a corrupted `.pt` will silently produce nonsense detections rather than
crash.

## Usage

Single model, print JSON to stdout:

```bash
python3 inference.py --model hose_wall --image path/to/img.jpg
```

Single model, also save an annotated image:

```bash
python3 inference.py --model hose_clamp --image img.jpg --save out/clamp.jpg
```

Override the recommended threshold (rare; the model card defaults are tuned):

```bash
python3 inference.py --model hose_aging --image img.jpg --conf 0.6
```

All four detectors on one image, write per-model overlays:

```bash
python3 pipeline.py --image img.jpg --save-dir runs/demo
```

Pick a subset of detectors:

```bash
python3 pipeline.py --image img.jpg --models hose_wall hose_clamp
```

The pipeline output is a single JSON document keyed by model name, plus a
`summary` block with the hit count and top confidence per model — handy for
stitching into a downstream rule engine.

## Runtime profile, by model

The recommended profile is what we actually run in production after a few
rounds of threshold sweeping on reviewed truth. Notable nuances:

- **`hose_aging`** runs strictly anomaly-only at `conf=0.50`,
  `min_area_ratio=0.0005`. Normal old hose, blue/green hose and pure
  negatives are intentionally left empty so this model behaves more like a
  classifier-style alarm than a permissive detector.
- **`hose_clamp`** is single-class (`clamp_abnormal` only — there is no
  `clamp_normal` head). It runs at `conf=0.35`, `min_area_ratio=0.001`. The
  near-zero clean-negative false rate (~0.5%) comes from explicit
  direct-negative mining during fine-tuning, not just thresholding.
- **`hose_type_metal`** is the only two-class detector. The two heads use
  different thresholds (0.50 for the safer `metal_clad_hose`, 0.45 for the
  higher-risk `non_standard_hose`). When both fire on the same crop,
  `inference.py` only switches to the higher-risk class if it beats the
  safer class by `0.20` confidence — see `_merge_two_class` in
  `inference.py`. This stops normal-looking metal-clad regions from being
  upgraded to non-standard on a tiny score margin.
- **`hose_wall`** runs production at `conf=0.45`, `min_area_ratio=0.0005`,
  with an extra `viewer_review_min_conf=0.25` reserved for human-in-the-loop
  review UIs (it surfaces low-score true positives without raising the
  production false-positive budget).

`min_area_ratio` is computed against the original image area, not the
resized model input.

## Headline metrics

These are reviewed-truth combined numbers, not raw validation. Reviewed
truth means images were re-labeled from scratch by humans rather than
trusted from training labels.

| Model | Eval set | Precision | Recall | F1 |
|---|---|---|---|---|
| `hose_aging` | 7-bucket reviewed truth, 335 imgs | 0.859 | 0.743 | 0.797 |
| `hose_clamp` | runtime eval, 219 + 358 imgs | — | — | positive_rate 0.963, clean-neg false 0.005 |
| `hose_type_metal` | 5-bucket reviewed truth, 306 imgs | 0.857 | 0.803 | 0.829 |
| `hose_wall` | 4-bucket reviewed truth, 404 imgs | 0.909 | 0.859 | 0.883 |

The full per-bucket numbers — including the buckets where each model still
loses — live in each model's `model_card.json`.

## Known limitations

- `hose_aging` is anomaly-only by design. It under-recalls on narrow-style
  buckets where aging is subtle (one bucket sits at F1 ≈ 0.18). If you need
  to surface borderline aging for human review, drop `conf` to ~0.30 and
  expect the false-positive rate to roughly triple.
- `hose_type_metal` recall on `non_standard_hose` is 0.80 combined. A
  minority of hard cases that look visually close to normal metal-clad hose
  remain missed — the two-class gap rule above is a deliberate
  precision-favoring trade.
- `hose_wall` keeps one acknowledged hard case where the wall penetration
  is partially occluded; that single image still produces no stable box
  even at the lowest review threshold. Worth flagging if you build an
  automated pipeline that assumes 100% wall-penetration recall.
- Shipped weights were trained at the input sizes listed above (832 for
  aging, 640 for the others). If you run them at a different `imgsz`, the
  recommended thresholds no longer apply — re-sweep before deploying.

## License

MIT. See [LICENSE](LICENSE).
