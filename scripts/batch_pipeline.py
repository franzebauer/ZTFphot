#!/usr/bin/env python3
"""
batch_pipeline.py
-----------------
Run ZTFphot for a list of RA/Dec targets, keeping only the final LC parquet
and diagnostic plots for each object and deleting all other products.

Usage:
    python batch_pipeline.py empty_noncal_stars_coords5.txt

Coordinates file: one target per line, comma-separated RA,Dec (degrees):
    152.808792,50.387500
    161.423500,8.563278
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

BAND_TO_FC = {"g": "zg", "r": "zr", "i": "zi"}


def find_final_parquets(work_dir: Path, ra: float, dec: float, bands: list,
                        both: bool = False) -> list:
    """
    Return (parquet_path, label) for the best available LC per band.
    Prefers merged over single-quadrant.  When both=True also finds sci-pos variants.
    """
    results = []
    ra_tag  = f"{ra:.5f}"
    dec_tag = f"{dec:+.5f}"

    suffixes = ["", "_sci"] if both else [""]

    for band in bands:
        fc = BAND_TO_FC.get(band, f"z{band}")
        for suffix in suffixes:
            # Merged parquet (multiple quadrants cross-calibrated)
            merged = (work_dir / "LightCurves" / "merged"
                      / f"{ra_tag}_{dec_tag}" / f"{fc}{suffix}" / "lightcurves_merged.parquet")
            if merged.exists():
                results.append((merged, f"{fc}{suffix}_merged"))
                continue

            # Single-quadrant parquets for this filtercode
            filename = f"lightcurves{suffix}.parquet"
            singles = sorted((work_dir / "LightCurves").rglob(filename))
            band_singles = [p for p in singles if f"/{fc}/" in str(p)]
            for p in band_singles:
                parts = p.parts
                try:
                    q_part   = parts[-2]
                    ccd_part = parts[-3]
                    fc_part  = parts[-4]
                    fld_part = parts[-5]
                    label = f"{fld_part}_{fc_part}_{ccd_part}_{q_part}{suffix}"
                except IndexError:
                    label = f"{fc}{suffix}"
                results.append((p, label))

    return results


def run_pipeline(pipeline: Path, ra: float, dec: float, work_dir: Path,
                 bands: list, workers: int, purge_batch: int,
                 min_maglim: float, max_seeing: float,
                 both: bool, extra_args: list) -> int:
    cmd = [
        sys.executable, str(pipeline),
        "--ra",          str(ra),
        "--dec",         str(dec),
        "--base-dir",    str(work_dir),
        "--purge-batch", str(purge_batch),
        "--workers",     str(workers),
        "--min-maglim",  str(min_maglim),
        "--max-seeing",  str(max_seeing),
        "--bands",       *bands,
    ]
    if both:
        cmd.append("--both")
    cmd += extra_args

    print(f"  CMD: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    return ret.returncode


def save_results(work_dir: Path, ra: float, dec: float, bands: list,
                 results_dir: Path, both: bool = False) -> bool:
    """Copy final parquets and plots to results_dir. Returns True if any parquet found."""
    ra_tag  = f"{ra:.5f}"
    dec_tag = f"{dec:+.5f}"
    tag = f"{ra_tag}_{dec_tag}"

    success = False

    # ── parquets ──────────────────────────────────────────────────────────────
    parquets = find_final_parquets(work_dir, ra, dec, bands, both=both)
    if not parquets:
        print(f"  WARNING: no LC parquet found for {tag}")
    for pq_path, label in parquets:
        dest = results_dir / f"{tag}_{label}.parquet"
        shutil.copy2(pq_path, dest)
        size_mb = pq_path.stat().st_size / 1e6
        print(f"  Saved parquet ({size_mb:.1f} MB) → {dest}")
        success = True

    # ── plots ─────────────────────────────────────────────────────────────────
    for plots_root, dest_suffix in ([("Plots", ""), ("Plots_sci", "_sci")] if both
                                    else [("Plots", "")]):
        plots_src = work_dir / plots_root / tag
        if plots_src.exists():
            plots_dest = results_dir / f"plots{dest_suffix}" / tag
            shutil.copytree(plots_src, plots_dest, dirs_exist_ok=True)
            n_plots = sum(1 for _ in plots_dest.rglob("*.png"))
            print(f"  Saved {n_plots} plot(s) → {plots_dest}")
        else:
            all_plots = work_dir / plots_root
            if all_plots.exists():
                plots_dest = results_dir / f"plots{dest_suffix}" / tag
                shutil.copytree(all_plots, plots_dest, dirs_exist_ok=True)
                n_plots = sum(1 for _ in plots_dest.rglob("*.png"))
                print(f"  Saved {n_plots} plot(s) → {plots_dest}")
            elif plots_root == "Plots":
                print(f"  WARNING: no plots directory found")

    return success


def cleanup(work_dir: Path) -> None:
    """Delete the entire per-target working directory."""
    if work_dir.exists():
        size = sum(f.stat().st_size for f in work_dir.rglob("*") if f.is_file())
        shutil.rmtree(work_dir)
        print(f"  Deleted {work_dir}  (freed {size/1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch ZTFphot: process a list of targets, keep only LC parquet + plots.")
    parser.add_argument("coords_file", type=Path,
                        help="Text file with RA,Dec per line (degrees, comma-separated)")
    parser.add_argument("--pipeline", type=Path,
                        default=Path("ZTFphot/scripts/run_pipeline.py"),
                        help="Path to run_pipeline.py (default: ZTFphot/scripts/run_pipeline.py)")
    parser.add_argument("--base-dir", default="IMBH",
                        help="Root working directory; a per-target subdirectory is "
                             "created inside it (default: IMBH)")
    parser.add_argument("--results-dir", type=Path, default=Path("IMBH_results"),
                        help="Directory where final parquets and plots are collected "
                             "(default: IMBH_results)")
    parser.add_argument("--bands", nargs="+", default=["g"],
                        metavar="BAND", help="Bands to process (default: g)")
    parser.add_argument("--workers",     type=int,   default=20)
    parser.add_argument("--purge-batch", type=int,   default=20)
    parser.add_argument("--min-maglim",  type=float, default=19.5)
    parser.add_argument("--max-seeing",  type=float, default=4.0)
    parser.add_argument("--both",         action="store_true",
                        help="Run both ref-pos and sci-pos photometry for each target")
    parser.add_argument("--no-cleanup",  action="store_true",
                        help="Skip deletion of working directory after each target")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip targets that already have a parquet in results-dir")
    args, extra = parser.parse_known_args()

    # Read coordinates
    coords = []
    with open(args.coords_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                print(f"WARNING: line {lineno} malformed, skipping: {line!r}",
                      file=sys.stderr)
                continue
            try:
                ra, dec = float(parts[0].strip()), float(parts[1].strip())
                coords.append((ra, dec))
            except ValueError:
                print(f"WARNING: line {lineno} not numeric, skipping: {line!r}",
                      file=sys.stderr)

    if not coords:
        sys.exit("No valid coordinates found.")

    print(f"Loaded {len(coords)} target(s) from {args.coords_file}")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    if args.both:
        (results_dir / "plots_sci").mkdir(exist_ok=True)

    base_dir = Path(args.base_dir)
    n_ok = n_fail = n_skip = 0

    for i, (ra, dec) in enumerate(coords, 1):
        tag = f"{ra:.5f}_{dec:+.5f}"
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Target {i}/{len(coords)}:  RA={ra}  Dec={dec}  ({tag})")
        print(sep)

        # Optional: skip if results already present
        if args.skip_existing:
            existing = list(results_dir.glob(f"{tag}_*.parquet"))
            if existing:
                print(f"  Already done ({len(existing)} parquet(s)) — skipping")
                n_skip += 1
                continue

        work_dir = base_dir / tag

        # Run pipeline
        rc = run_pipeline(
            pipeline=args.pipeline,
            ra=ra, dec=dec,
            work_dir=work_dir,
            bands=args.bands,
            workers=args.workers,
            purge_batch=args.purge_batch,
            min_maglim=args.min_maglim,
            max_seeing=args.max_seeing,
            both=args.both,
            extra_args=extra,
        )

        if rc != 0:
            print(f"  WARNING: pipeline exit code {rc} — saving whatever exists")

        # Save results
        ok = save_results(work_dir, ra, dec, args.bands, results_dir, both=args.both)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

        # Clean up working directory
        if not args.no_cleanup:
            cleanup(work_dir)

    print(f"\n{'=' * 70}")
    print(f"Batch complete:  {n_ok} succeeded,  {n_fail} no parquet,  {n_skip} skipped")
    print(f"Results in:  {results_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
