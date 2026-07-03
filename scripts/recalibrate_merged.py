#!/usr/bin/env python3
"""
recalibrate_merged.py — batch re-run inter-quadrant normalization on orphaned
merged parquets (those whose per-quadrant/intermediate products were deleted).

For each *_merged.parquet found, applies the in-place per-magnitude
inter-quadrant correction (renorm_merged_parquet.renorm): for every secondary
quadrant it measures dominant - secondary over common sources as a function of
magnitude and shifts all MAG_* columns onto the dominant quadrant's scale.

This is the ONLY recalibration possible from a merged file alone.  The
per-quadrant absolute calibration (linear ZP, faint correction, flatfield)
cannot be redone here — those need the intermediate products that were deleted.
Single-quadrant merged files are left unchanged.

The merged files are typically the only surviving copy, so:
  * --dry-run reports the corrections without writing (run this first);
  * --backup writes a one-time <file>.bak before the first modification.

Usage:
    python recalibrate_merged.py [--mag-bin 0.1] [--dry-run] [--backup] <root|file> [...]

    # preview everything under the project, then apply with backups:
    python recalibrate_merged.py --dry-run .
    python recalibrate_merged.py --backup .
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renorm_merged_parquet import renorm


def _iter_targets(arg: str):
    p = Path(arg)
    if p.is_file():
        yield p
    elif p.is_dir():
        yield from sorted(p.rglob("*_merged.parquet"))
    else:
        yield from sorted(Path('.').glob(arg))


def main(argv):
    mag_bin = 0.1
    dry_run = False
    backup  = False
    rest    = []
    it = iter(argv)
    for a in it:
        if a == "--mag-bin":
            mag_bin = float(next(it))
        elif a == "--dry-run":
            dry_run = True
        elif a == "--backup":
            backup = True
        else:
            rest.append(a)

    if not rest:
        rest = ["."]

    files, seen = [], set()
    for a in rest:
        for f in _iter_targets(a):
            rp = f.resolve()
            if rp not in seen:
                seen.add(rp)
                files.append(f)

    if not files:
        sys.exit("No *_merged.parquet files found.")

    mode = "DRY-RUN" if dry_run else ("write + backup" if backup else "write")
    print(f"recalibrate_merged: {len(files)} file(s)  mag_bin={mag_bin}  mode={mode}\n")

    ok = skipped = err = 0
    for f in files:
        try:
            if backup and not dry_run:
                bak = f.with_suffix(f.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(f, bak)
            renorm(f, mag_bin, write=not dry_run)
            ok += 1
        except Exception as exc:                       # keep going across the batch
            err += 1
            print(f"  ERROR {f}: {exc}")

    print(f"\nDone: {ok} processed, {err} errors"
          + ("  (dry-run, nothing written)" if dry_run else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
