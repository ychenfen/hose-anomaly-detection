# examples

`architecture.png` — pipeline overview used in the top-level README. Generated
deterministically from a small Matplotlib script (no real images, no
training-set imagery).

## Reproducing the demo run

```bash
# 1. Drop a hose photo here (jpg/png). Anything you have on hand works.
cp /path/to/your/hose.jpg examples/demo.jpg

# 2. Run all four detectors and write annotated overlays to examples/runs/
python3 pipeline.py --image examples/demo.jpg --save-dir examples/runs
```

The output is a single JSON document on stdout, plus four annotated images
under `examples/runs/` (one per model). Empty-result models simply emit an
empty `[]` and produce no overlay boxes.

A typical `summary` block looks like:

```json
{
  "summary": {
    "hose_aging":      {"hits": 1, "top_conf": 0.71},
    "hose_clamp":      {"hits": 0, "top_conf": null},
    "hose_type_metal": {"hits": 1, "top_conf": 0.58},
    "hose_wall":       {"hits": 0, "top_conf": null}
  }
}
```

If you push real demo images here, keep them small (< 1 MB each) so the repo
stays under 200 MB. `examples/runs/` is gitignored — overlays do not get
committed.
