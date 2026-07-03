#!/usr/bin/env python3
"""
plot_quad_offsets.py — diagnose systematic magnitude offsets between quadrants.

For each pair of quadrants in a merged parquet, finds sources detected in both,
computes the per-source median magnitude difference, and plots:
  - Histogram of differences per quad pair
  - Differences vs magnitude (to detect slope/faint-end issues)
  - Spatial map of differences on sky

Usage:
    python plot_quad_offsets.py <merged.parquet> [<merged.parquet> ...]
    python plot_quad_offsets.py /path/to/*_merged.parquet
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq

MAG_COL    = "MAG_4_TOT_AB"
MERR_COL   = "MERR_4_TOT_AB"
OUT_SUFFIX = "_quad_offsets"
MAG_BRIGHT = 14.0
MAG_FAINT  = 21.5
MAX_ERR    = 0.1
HARD_REJECT = (1 << 0) | (1 << 1) | (1 << 25)


def _quad_label(field, ccdid, qid):
    return f"{int(field)}/c{int(ccdid)}/q{int(qid)}"


def _clean(df):
    """Remove hard-rejected epochs and outlier magnitudes."""
    mask = (
        ((df["INFOBITS_DIF"].fillna(0).astype(int) & HARD_REJECT) == 0) &
        df[MAG_COL].notna() &
        (df[MAG_COL] > MAG_BRIGHT) &
        (df[MAG_COL] < MAG_FAINT)
    )
    if MERR_COL in df.columns:
        mask &= df[MERR_COL].fillna(999) < MAX_ERR
    return df[mask]


def _median_mags(df, field, ccdid, qid):
    """Per-source median magnitude for one quadrant."""
    mask = (df["field"] == field) & (df["ccdid"] == ccdid) & (df["qid"] == qid)
    sub = _clean(df[mask])
    return (sub.groupby("object_index")[MAG_COL]
               .agg(median="median", std="std", count="count"))


def plot_offsets(path: Path) -> None:
    pf   = pq.read_table(path)
    df   = pf.to_pandas()
    band = Path(path).stem.split("_")[-1].replace("merged", "").strip("_") or "?"

    quads = (df[["field", "ccdid", "qid"]]
             .drop_duplicates()
             .sort_values(["field", "ccdid", "qid"])
             .reset_index(drop=True))

    if len(quads) < 2:
        print(f"SKIP {path.name}: only one quadrant")
        return

    # ── Compute per-source median mag per quadrant ─────────────────────────────
    medians = {}
    for _, r in quads.iterrows():
        key = (int(r.field), int(r.ccdid), int(r.qid))
        medians[key] = _median_mags(df, *key)

    pairs = list(combinations(list(medians.keys()), 2))
    n_pairs = len(pairs)

    fig = plt.figure(figsize=(6 * n_pairs, 12))
    gs  = gridspec.GridSpec(3, n_pairs, figure=fig, hspace=0.45, wspace=0.35)

    for col, (ka, kb) in enumerate(pairs):
        lbl_a = _quad_label(*ka)
        lbl_b = _quad_label(*kb)

        # Common sources with enough epochs in both quadrants
        common = medians[ka].join(medians[kb], lsuffix="_a", rsuffix="_b", how="inner")
        common = common[(common["count_a"] >= 5) & (common["count_b"] >= 5)]
        diff   = common["median_a"] - common["median_b"]
        mag_ax = common["median_a"]  # x-axis: magnitude in quadrant A

        if len(diff) < 3:
            print(f"  {lbl_a} vs {lbl_b}: too few common sources ({len(diff)})")
            continue

        med_off = float(np.median(diff))
        std_off = float(np.std(diff))

        # Magnitude-bin breakdown to separate ZP (bright) from faint-correction residual
        bins_report = [(14.0, 18.5, "bright"), (18.5, 19.5, "ramp"), (19.5, 21.5, "faint")]
        bin_strs = []
        for lo, hi, label in bins_report:
            sel = (mag_ax >= lo) & (mag_ax < hi)
            if sel.sum() >= 3:
                bin_strs.append(f"{label}[{lo:.0f}–{hi:.1f}]={np.median(diff[sel]):+.3f}({sel.sum()})")
        print(f"  {lbl_a} vs {lbl_b}: {len(diff)} sources, "
              f"median offset = {med_off:+.4f} mag, σ = {std_off:.4f} mag"
              + (f"  [{', '.join(bin_strs)}]" if bin_strs else ""))

        # ── Row 0: histogram ───────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        bins = np.linspace(np.percentile(diff, 1), np.percentile(diff, 99), 50)
        ax0.hist(diff, bins=bins, color="steelblue", alpha=0.8)
        ax0.axvline(0,       color="black",  lw=1.2, ls="--")
        ax0.axvline(med_off, color="crimson", lw=1.5, ls="-",
                   label=f"median = {med_off:+.4f}")
        ax0.set_xlabel(f"Δmag  ({lbl_a} − {lbl_b})")
        ax0.set_ylabel("N sources")
        ax0.set_title(f"{lbl_a}\nvs {lbl_b}   (n={len(diff)})")
        ax0.legend(fontsize=8)

        # ── Row 1: Δmag vs magnitude ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.scatter(mag_ax, diff, s=3, alpha=0.3, color="steelblue", rasterized=True)
        ax1.axhline(0,       color="black",  lw=1.2, ls="--")
        ax1.axhline(med_off, color="crimson", lw=1.5, ls="-",
                   label=f"median = {med_off:+.4f}")
        # Binned running median
        _bins = np.linspace(MAG_BRIGHT, MAG_FAINT, 20)
        _bc   = 0.5 * (_bins[:-1] + _bins[1:])
        _bmed = [np.median(diff[(mag_ax >= lo) & (mag_ax < hi)].values)
                 for lo, hi in zip(_bins[:-1], _bins[1:])]
        _ok   = [~np.isnan(v) for v in _bmed]
        ax1.plot(np.array(_bc)[_ok], np.array(_bmed)[_ok],
                 color="orange", lw=2, label="binned median")
        ax1.set_xlabel(f"mag ({lbl_a})")
        ax1.set_ylabel(f"Δmag  ({lbl_a} − {lbl_b})")
        ax1.legend(fontsize=8)

        # ── Row 2: spatial map ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        # Attach sky positions from quadrant A rows
        pos_a = (df[(df["field"] == ka[0]) & (df["ccdid"] == ka[1]) & (df["qid"] == ka[2])]
                 .groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first())
        plot_df = common.join(pos_a, how="left")
        plot_df = plot_df.assign(diff=diff)
        vmax = max(abs(np.percentile(diff, 2)), abs(np.percentile(diff, 98)), 0.05)
        sc = ax2.scatter(plot_df["ALPHAWIN_REF"], plot_df["DELTAWIN_REF"],
                         c=plot_df["diff"], cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, s=4, alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax2, label="Δmag")
        ax2.set_xlabel("RA (deg)")
        ax2.set_ylabel("Dec (deg)")
        ax2.set_title("Spatial distribution of offsets")
        ax2.invert_xaxis()

    band_label = band if band else path.stem
    fig.suptitle(f"Quadrant magnitude offsets — {path.stem}", fontsize=13, y=1.01)

    out = path.parent / (path.stem + OUT_SUFFIX + ".png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--org" in args:
        args.remove("--org")
        MAG_COL    = "MAG_4_TOT_AB_org"
        OUT_SUFFIX = "_quad_offsets_org"
    if not args:
        sys.exit("Usage: python plot_quad_offsets.py [--org] <merged.parquet> [...]")
    for arg in args:
        p = Path(arg)
        if p.exists():
            plot_offsets(p)
        else:
            for p2 in sorted(Path(".").glob(arg)):
                plot_offsets(p2)
