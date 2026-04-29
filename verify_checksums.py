"""Verify SHA-256 checksums for the bundled model weights.

Reads SHA256SUMS.txt at the repo root (format: ``<sha256>  <relative-path>``)
and compares against the on-disk weights. Exits non-zero on any mismatch.
"""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SUMS_FILE = REPO_ROOT / "SHA256SUMS.txt"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SUMS_FILE.exists():
        print(f"missing {SUMS_FILE}", file=sys.stderr)
        return 2

    bad = 0
    for line in SUMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, _, rel = line.partition("  ")
        target = REPO_ROOT / rel
        if not target.exists():
            print(f"MISSING {rel}")
            bad += 1
            continue
        actual = hash_file(target)
        ok = actual == expected
        print(f"{'OK     ' if ok else 'FAIL   '}{rel}")
        if not ok:
            print(f"  expected {expected}")
            print(f"  actual   {actual}")
            bad += 1

    if bad:
        print(f"\n{bad} file(s) failed verification", file=sys.stderr)
        return 1
    print("\nall checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
