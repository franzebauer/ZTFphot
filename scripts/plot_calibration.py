"""
plot_calibration.py
-------------------
Fig 2 — Calibration RMS improvement & corrections (4 panels):
  top-left  — boxplot of calibrator RMS at each pipeline stage (mmag)
  top-right — median ± IQR per stage, line plot
  bot-left  — linear ZP correction at mag 17 vs seeing, coloured by MAGLIM
  bot-right — per-epoch faint correction curve vs source mag, coloured by MAGLIM

Reads *_cal.fits primary headers from Calibrated/.
Header keywords used:
    NC_RMS0 … NC_RMS4, NC_RMSFC  — calibrator RMS at each stage (mmag)
    CALIB_N, CALIB_M              — linear fit intercept and slope
    NC_FC_00 … NC_FC_06           — faint correction per mag bin (mmag)
    SEEING, MAGLIM
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)

_SENTINEL = -999.0


def _load_epoch_headers(cal_dir: Path) -> "pd.DataFrame":
    import pandas as pd
    from astropy.io import fits
    keys = ["OBSMJD", "SEEING", "MAGLIM", "num_stars", "NC_N",
            "NC_RMS0", "NC_RMS1", "NC_RMS2", "NC_RMSFC", "NC_RMS3", "NC_RMS4",
            "CALIB_N", "CALIB_M", "CALIB_ZP",
            "TGT_MRAW", "TGT_DCLIN", "TGT_DCPOL", "TGT_DCFF",
            "NC_FC_00", "NC_FC_01", "NC_FC_02", "NC_FC_03",
            "NC_FC_04", "NC_FC_05", "NC_FC_06"]
    rows = []
    for p in sorted(cal_dir.glob("*_cal.fits")):
        try:
            with fits.open(p, memmap=False) as h:
                hdr = h[0].header
            row = {k: float(hdr.get(k, np.nan)) for k in keys}
            row["fname"] = p.name
            rows.append(row)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).replace(_SENTINEL, np.nan)
    return df.sort_values("OBSMJD").reset_index(drop=True)


def make_rms(cal_dir: Path, out_path: Path, tag: str = "") -> None:
    df = _load_epoch_headers(cal_dir)
    if df.empty:
        logger.warning(f"No calibrated epochs in {cal_dir}")
        return

    stages = ["NC_RMS0", "NC_RMS1", "NC_RMS2", "NC_RMSFC", "NC_RMS3", "NC_RMS4"]
    labels = ["Before\ncal", "After\nlinear", "After\n3σ clip",
              "After\nfaint corr.", "After\n2D poly", "After\nflatfield"]
    colors = ["#888888", "C0", "C1", "C2", "C3", "C4"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Calibration RMS improvement & corrections — {tag}", fontsize=11)

    # ── Panel 1: boxplot ──
    ax = axes[0, 0]
    data_box = [df[s].dropna().values for s in stages]
    bp = ax.boxplot(data_box, patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Calibrator RMS (mmag)")
    ax.set_title("Distribution per calibration step")
    ax.set_ylim(bottom=0)

    # ── Panel 2: median ± IQR line ──
    ax = axes[0, 1]
    meds = [float(df[s].median()) for s in stages]
    p25  = [float(df[s].quantile(0.25)) for s in stages]
    p75  = [float(df[s].quantile(0.75)) for s in stages]
    xs   = list(range(len(stages)))
    total_delta = meds[0] - meds[-1]
    ax.plot(xs, meds, "ko-", lw=2, ms=6, label=f"Total Δ = {total_delta:.1f} mmag")
    ax.fill_between(xs, p25, p75, alpha=0.25, color="steelblue", label="IQR")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median ± IQR per step (mmag)")
    ax.set_title("Median ± IQR per step")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # ── Panel 3: ZP correction vs seeing ──
    ax = axes[1, 0]
    cn  = df["CALIB_N"].dropna()
    cm  = df["CALIB_M"].dropna()
    see = df["SEEING"]
    ml  = df["MAGLIM"]
    idx = cn.index.intersection(cm.index)
    idx = idx[np.isfinite(see[idx].values) & np.isfinite(ml[idx].values)]
    if len(idx) > 5:
        s   = see[idx].values
        mlv = ml[idx].values
        zpc = (cn[idx].values + cm[idx].values * 17.0) * 1000
        sc  = ax.scatter(s, zpc, c=mlv, cmap="plasma", s=10, alpha=0.7,
                         vmin=np.nanpercentile(mlv, 5), vmax=np.nanpercentile(mlv, 95))
        plt.colorbar(sc, ax=ax, label="MAGLIM (mag)")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.text(0.03, 0.97,
                f"med = {float(np.nanmedian(zpc)):.1f} mmag\nσ = {float(np.nanstd(zpc)):.1f} mmag",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xlabel("Seeing (arcsec)")
    ax.set_ylabel("Linear ZP correction @ mag 17 (mmag)")
    ax.set_title("ZP correction vs seeing\n(coloured by limiting magnitude)")

    # ── Panel 4: faint correction curves ──
    ax = axes[1, 1]
    fc_cols = [f"NC_FC_{i:02d}" for i in range(7)]
    fc_mags = [18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75]
    fc_data = df[fc_cols].replace(_SENTINEL, np.nan)
    ml_vals = df["MAGLIM"].values
    if fc_data.notna().any().any():
        ml_min = np.nanpercentile(ml_vals, 5)
        ml_max = np.nanpercentile(ml_vals, 95)
        norm   = mcolors.Normalize(vmin=ml_min, vmax=ml_max)
        cmap   = plt.cm.plasma
        for i, row in fc_data.iterrows():
            vals = row.values.astype(float)
            ok   = np.isfinite(vals)
            if ok.sum() >= 3:
                color = cmap(norm(ml_vals[i])) if np.isfinite(ml_vals[i]) else "gray"
                ax.plot(np.array(fc_mags)[ok], vals[ok], color=color, lw=0.5, alpha=0.4)
        med_fc = fc_data.median(axis=0).values
        ok = np.isfinite(med_fc)
        if ok.sum() >= 3:
            ax.plot(np.array(fc_mags)[ok], med_fc[ok], "k-", lw=2, label="Median")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="MAGLIM (mag)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Source magnitude (AB)")
    ax.set_ylabel("Faint correction applied (mmag)")
    ax.set_title("Per-epoch faint correction vs magnitude\n(coloured by limiting magnitude)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"  rms → {out_path}")
