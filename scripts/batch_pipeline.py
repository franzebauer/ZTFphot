#!/usr/bin/env python3
"""
batch_pipeline.py
-----------------
Run ZTFphot for a list of targets, keeping only the final LC parquet
and diagnostic plots for each object and deleting all other products.

Two input modes, auto-detected by column count:

  RA/Dec mode (2 columns) — full pipeline including lookup and merge:
      152.808792,50.387500
      161.423500,8.563278

  Quadrant mode (4 columns) — skips merge; field/ccd/qid/filtercode are
  passed directly to run_pipeline; the quadrant center is derived from
  the ztfquery field grid and used for lookup. Target plots are suppressed.
  After the run the brightest source RA/Dec is saved alongside the parquet.
      field,ccdid,qid,filtercode
      000443,16,2,zg
      001389,12,3,zg
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

BAND_TO_FC = {"g": "zg", "r": "zr", "i": "zi",
              "zg": "zg", "zr": "zr", "zi": "zi"}


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
                 bands: list, workers: int, download_workers: int, purge_batch: int,
                 min_maglim: float, max_seeing: float,
                 both: bool, extra_args: list) -> int:
    cmd = [
        sys.executable, str(pipeline),
        "--ra",               str(ra),
        "--dec",              str(dec),
        "--base-dir",         str(work_dir),
        "--purge-batch",      str(purge_batch),
        "--workers",          str(workers),
        "--download-workers", str(download_workers),
        "--min-maglim",       str(min_maglim),
        "--max-seeing",       str(max_seeing),
        "--bands",            *bands,
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


def _quadrant_center(field: int, ccdid: int, qid: int) -> tuple:
    """
    Return approximate (RA, Dec) of a ZTF CCD quadrant center.
    Uses ztfquery's bundled field grid and CCD/quad corner offset tables.
    """
    import numpy as np
    import pandas as pd
    import ztfquery
    data_dir = Path(ztfquery.__file__).parent / "data"

    fields_tbl = pd.read_csv(data_dir / "ztf_fields.txt")
    row = fields_tbl[fields_tbl["ID"].astype(int) == int(field)]
    if row.empty:
        raise ValueError(f"Field {field} not found in ZTF field table")
    ra0  = float(row["RA"].iloc[0])
    dec0 = float(row["Dec"].iloc[0])

    # RCID = (ccdid-1)*4 + (qid-1); layout table uses 0-based quad = RCID
    rcid   = (ccdid - 1) * 4 + (qid - 1)
    layout = pd.read_csv(data_dir / "ztf_ccd_quad_layout.tbl")
    corners = layout[layout["Quad"] == rcid]
    if corners.empty:
        raise ValueError(f"RCID {rcid} (ccdid={ccdid}, qid={qid}) not found in layout table")
    dew = float(corners["EW"].mean())
    dns = float(corners["NS"].mean())

    ra  = (ra0 + dew / np.cos(np.radians(dec0))) % 360.0
    dec = dec0 + dns
    return ra, dec


_QUAD_STEPS = [
    "lookup", "download", "catalog", "simulate", "sex",
    "vet", "calibrate", "flatfield", "recalibrate", "lightcurves", "plots",
]


def run_pipeline_quad(pipeline: Path, field: int, ccdid: int, qid: int,
                      band: str, ra: float, dec: float, work_dir: Path,
                      workers: int, download_workers: int, purge_batch: int,
                      min_maglim: float, max_seeing: float,
                      both: bool, extra_args: list) -> int:
    cmd = [
        sys.executable, str(pipeline),
        "--ra",               str(ra),
        "--dec",              str(dec),
        "--base-dir",         str(work_dir),
        "--field",            str(field),
        "--ccdid",            str(ccdid),
        "--qid",              str(qid),
        "--bands",            band,
        "--steps",            *_QUAD_STEPS,
        "--purge-batch",      str(purge_batch),
        "--workers",          str(workers),
        "--download-workers", str(download_workers),
        "--min-maglim",       str(min_maglim),
        "--max-seeing",       str(max_seeing),
    ]
    if both:
        cmd.append("--both")
    cmd.append("--no-target")
    cmd += extra_args
    print(f"  CMD: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    return ret.returncode


def find_quad_parquets(work_dir: Path, field: int, fc: str, ccdid: int,
                       qid: int, both: bool = False) -> list:
    results = []
    for suffix in (["", "_sci"] if both else [""]):
        pq = (work_dir / "LightCurves" / f"{field:06d}" / fc
              / f"ccd{ccdid:02d}" / f"q{qid}" / f"lightcurves{suffix}.parquet")
        if pq.exists():
            results.append((pq, f"{fc}{suffix}"))
    return results


def save_results_quad(work_dir: Path, field: int, fc: str, ccdid: int, qid: int,
                      ra: float, dec: float, results_dir: Path,
                      both: bool = False) -> bool:
    tag = f"{field:06d}_{fc}_c{ccdid:02d}_q{qid}"
    success = False

    parquets = find_quad_parquets(work_dir, field, fc, ccdid, qid, both)
    if not parquets:
        print(f"  WARNING: no LC parquet found for {tag}")
    for pq_path, label in parquets:
        dest = results_dir / f"{tag}_{label}.parquet"
        shutil.copy2(pq_path, dest)
        print(f"  Saved parquet ({pq_path.stat().st_size/1e6:.1f} MB) → {dest}")
        success = True

    coord_tag = f"{ra:.5f}_{dec:+.5f}"
    for plots_root, dest_suffix in ([("Plots", ""), ("Plots_sci", "_sci")] if both
                                    else [("Plots", "")]):
        plots_src = work_dir / plots_root / coord_tag
        if plots_src.exists():
            plots_dest = results_dir / f"plots{dest_suffix}" / tag
            shutil.copytree(plots_src, plots_dest, dirs_exist_ok=True)
            n = sum(1 for _ in plots_dest.rglob("*.png"))
            print(f"  Saved {n} plot(s) → {plots_dest}")

    return success


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
    parser.add_argument("--bands", nargs="+", default=["zg"],
                        metavar="BAND", help="Bands to process (default: zg)")
    parser.add_argument("--workers",          type=int, default=4,
                        help="Parallel workers for simulate/sex/calibrate (default: 4)")
    parser.add_argument("--download-workers", type=int, default=50,
                        help="Parallel threads for image downloads (default: 50)")
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

    # Read targets — auto-detect mode from first valid line's column count
    # RA/Dec mode:   2 columns  →  ra, dec
    # Quadrant mode: 4 columns  →  field, ccdid, qid, filtercode
    targets = []
    quad_mode = None
    with open(args.coords_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if quad_mode is None:
                quad_mode = len(parts) >= 4
            if not quad_mode:
                if len(parts) < 2:
                    print(f"WARNING: line {lineno} malformed, skipping: {line!r}", file=sys.stderr)
                    continue
                try:
                    targets.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    print(f"WARNING: line {lineno} not numeric, skipping: {line!r}", file=sys.stderr)
            else:
                if len(parts) < 4:
                    print(f"WARNING: line {lineno} malformed, skipping: {line!r}", file=sys.stderr)
                    continue
                try:
                    targets.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
                except ValueError:
                    print(f"WARNING: line {lineno} not parseable, skipping: {line!r}", file=sys.stderr)

    if not targets:
        sys.exit("No valid targets found.")

    mode_str = "quadrant" if quad_mode else "RA/Dec"
    print(f"Loaded {len(targets)} target(s) from {args.coords_file} [{mode_str} mode]")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    if args.both:
        (results_dir / "plots_sci").mkdir(exist_ok=True)

    base_dir = Path(args.base_dir)
    n_ok = n_fail = n_skip = 0

    for i, target in enumerate(targets, 1):
        sep = "=" * 70

        if quad_mode:
            field, ccdid, qid, fc = target
            band = fc  # filtercode is already in zg/zr/zi form
            try:
                ra, dec = _quadrant_center(field, ccdid, qid)
            except ValueError as e:
                print(f"  ERROR: {e} — skipping", file=sys.stderr)
                n_fail += 1
                continue
            tag = f"{field:06d}_{fc}_c{ccdid:02d}_q{qid}"
            print(f"\n{sep}")
            print(f"  Target {i}/{len(targets)}:  field={field:06d}  ccdid={ccdid}  "
                  f"qid={qid}  fc={fc}  ({tag})")
            print(f"  Quadrant center: RA={ra:.5f}  Dec={dec:+.5f}")
            print(sep)
        else:
            ra, dec = target
            tag = f"{ra:.5f}_{dec:+.5f}"
            print(f"\n{sep}")
            print(f"  Target {i}/{len(targets)}:  RA={ra}  Dec={dec}  ({tag})")
            print(sep)

        # Optional: skip if results already present
        if args.skip_existing:
            existing = list(results_dir.glob(f"{tag}_*.parquet"))
            if existing:
                print(f"  Already done ({len(existing)} parquet(s)) — skipping")
                n_skip += 1
                continue

        work_dir = base_dir / tag

        if quad_mode:
            rc = run_pipeline_quad(
                pipeline=args.pipeline,
                field=field, ccdid=ccdid, qid=qid, band=band,
                ra=ra, dec=dec,
                work_dir=work_dir,
                workers=args.workers,
                download_workers=args.download_workers,
                purge_batch=args.purge_batch,
                min_maglim=args.min_maglim,
                max_seeing=args.max_seeing,
                both=args.both,
                extra_args=extra,
            )
            if rc != 0:
                print(f"  WARNING: pipeline exit code {rc} — saving whatever exists")
            ok = save_results_quad(work_dir, field, fc, ccdid, qid, ra, dec,
                                   results_dir, both=args.both)
        else:
            rc = run_pipeline(
                pipeline=args.pipeline,
                ra=ra, dec=dec,
                work_dir=work_dir,
                bands=args.bands,
                workers=args.workers,
                download_workers=args.download_workers,
                purge_batch=args.purge_batch,
                min_maglim=args.min_maglim,
                max_seeing=args.max_seeing,
                both=args.both,
                extra_args=extra,
            )
            if rc != 0:
                print(f"  WARNING: pipeline exit code {rc} — saving whatever exists")
            ok = save_results(work_dir, ra, dec, args.bands, results_dir, both=args.both)

        if ok:
            n_ok += 1
        else:
            n_fail += 1

        if not args.no_cleanup:
            cleanup(work_dir)

    print(f"\n{'=' * 70}")
    print(f"Batch complete:  {n_ok} succeeded,  {n_fail} no parquet,  {n_skip} skipped")
    print(f"Results in:  {results_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
