"""
compare_scipos.py
-----------------
Compare ref-pos vs sci-pos photometry for matched sources.

For each quadrant that has both lightcurves.parquet and lightcurves_sci.parquet,
plots:
  Left panel:  median(MAG_4_TOT_AB_sci - MAG_4_TOT_AB_ref) vs magnitude
               (systematic offset per source)
  Right panel: sigma_sci - sigma_ref vs magnitude
               (negative = sci-pos has lower scatter = better)

Sources are matched by object_index (same reference ASSOC catalog in both runs).
Only clean epochs (INFOBITS_DIF == 0) with valid MAG_4_TOT_AB are used.

Usage:
    python compare_scipos.py --base-dir J1025 --ra 156.37621 --dec 14.03539
    python compare_scipos.py --base-dir J1025 --ra 156.37621 --dec 14.03539 --band zg
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MIN_EPOCHS = 5   # minimum clean epochs per source to include


def _per_source_stats(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[(df["INFOBITS_DIF"] == 0) & df["MAG_4_TOT_AB"].notna()].copy()
    stats = (clean.groupby("object_index")
             .agg(
                 n=("MAG_4_TOT_AB", "count"),
                 mag=("MAG_4_TOT_AB", "median"),
                 std=("MAG_4_TOT_AB", "std"),
             )
             .reset_index())
    return stats[stats["n"] >= MIN_EPOCHS]


def compare_quadrant(ref_pq: Path, sci_pq: Path, out_path: Path, tag: str) -> None:
    ref = pd.read_parquet(ref_pq)
    sci = pd.read_parquet(sci_pq)

    stats_ref = _per_source_stats(ref).set_index("object_index")
    stats_sci = _per_source_stats(sci).set_index("object_index")

    common = stats_ref.index.intersection(stats_sci.index)
    if len(common) < 5:
        print(f"  {tag}: too few common sources ({len(common)}) — skipping")
        return

    r = stats_ref.loc[common]
    s = stats_sci.loc[common]

    mag_ref  = r["mag"].values
    d_mag    = s["mag"].values - r["mag"].values          # sci - ref median mag
    d_sigma  = s["std"].values - r["std"].values          # sci - ref sigma

    def binned_median(x, y, nbins=20):
        """Return bin centres and median y per bin, dropping empty bins."""
        bins = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), nbins + 1)
        centres, meds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.sum() >= 3:
                centres.append(0.5 * (lo + hi))
                meds.append(np.median(y[mask]))
        return np.array(centres), np.array(meds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{tag}   N={len(common)} sources", fontsize=11)

    kw = dict(s=6, alpha=0.4, rasterized=True)

    # ── Left: systematic offset ─────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(mag_ref, d_mag, c="steelblue", **kw)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    bx, by = binned_median(mag_ref, d_mag)
    ax.plot(bx, by, color="tomato", lw=1.8, label="binned median")

    ax.set_xlabel("MAG_4_TOT_AB (ref-pos, median)")
    ax.set_ylabel("Δmag  (sci − ref)  [mag]")
    ax.set_title("Magnitude offset (sci − ref)")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_ylim(-0.02,0.02)
    
    # ── Right: scatter improvement ──────────────────────────────────────────
    ax = axes[1]
    c = np.where(d_sigma < 0, "forestgreen", "tomato")
    ax.scatter(mag_ref, d_sigma * 1000, c=c, **kw)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    bx, by = binned_median(mag_ref, d_sigma * 1000)
    ax.plot(bx, by, color="purple", lw=1.8, label="binned median")

    n_better = int((d_sigma < 0).sum())
    ax.set_xlabel("MAG_4_TOT_AB (ref-pos, median)")
    ax.set_ylabel("Δσ  (sci − ref)  [mmag]")
    ax.set_title(f"Scatter change (sci − ref)   {n_better}/{len(common)} sources improved")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_ylim(-10,10)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag}: N={len(common)}  improved={n_better}/{len(common)} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--ra",  type=float, default=None)
    ap.add_argument("--dec", type=float, default=None)
    ap.add_argument("--band", default=None, help="Restrict to one filtercode, e.g. zg")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: base_dir/Plots/compare_scipos/)")
    args = ap.parse_args()

    lc_root = args.base_dir / "LightCurves"
    if not lc_root.exists():
        print(f"LightCurves directory not found: {lc_root}"); return

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.base_dir / "Plots" / "compare_scipos"
        if args.ra is not None and args.dec is not None:
            out_dir = out_dir / f"{args.ra:.5f}_{args.dec:+.5f}"

    pairs: list[tuple[Path, Path, str]] = []
    for ref_pq in sorted(lc_root.rglob("lightcurves.parquet")):
        sci_pq = ref_pq.with_name("lightcurves_sci.parquet")
        if not sci_pq.exists():
            continue
        parts = ref_pq.parts
        # path: .../LightCurves/{field}/{fc}/ccd{ccd}/q{qid}/lightcurves.parquet
        try:
            field, fc, ccd_s, qid_s = parts[-5], parts[-4], parts[-3], parts[-2]
            tag = f"{field}_{fc}_{ccd_s}_{qid_s}"
        except IndexError:
            tag = ref_pq.parent.name
        if args.band and fc != args.band:
            continue
        pairs.append((ref_pq, sci_pq, tag))

    if not pairs:
        print("No quadrant pairs found (both lightcurves.parquet and lightcurves_sci.parquet required).")
        return

    print(f"Comparing {len(pairs)} quadrant(s)...")
    for ref_pq, sci_pq, tag in pairs:
        compare_quadrant(ref_pq, sci_pq,
                         out_dir / f"compare_{tag}.png",
                         tag)


if __name__ == "__main__":
    main()
