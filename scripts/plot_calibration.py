"""
plot_calibration.py
-------------------
Fig 2 — Calibration RMS improvement & corrections (4 panels).
Left column = calibrators, right column = full sample (correction → corrected,
top to bottom):
  top-left  — boxplot of calibrator RMS at each pipeline stage (mmag)
  bot-left  — linear ZP correction at mag 17 vs seeing, coloured by MAGLIM
  top-right — per-epoch faint correction curve vs source mag, coloured by MAGLIM
  bot-right — full-sample residual vs calibrated magnitude after calibration,
              with per-bin clipped-median (the correction target) / raw median /
              mean / mode overplotted. The clipped-median line ≈0 shows the
              correction hit its target; the offset between raw median and mode
              is the irreducible faint-end skew.

Reads *_cal.fits primary headers from Calibrated/, and (for the top-right panel)
mag_all + dm_all_post from *_resid.npz in FlatfieldResiduals/.
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
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)

_SENTINEL = -999.0


def _load_epoch_headers(cal_dir: Path) -> "pd.DataFrame":
    import pandas as pd
    from astropy.io import fits
    keys = ["OBSMJD", "SEEING", "MAGLIM", "num_stars", "NC_N",
            "NC_RMS0", "NC_RMS1", "NC_RMS2", "NC_RMSFC", "NC_RMS3", "NC_RMS4",
            "CALIB_N", "CALIB_M", "CALIB_ZP", "APCORR46",
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


def _binned_center_curves(mag, resid, edges):
    """Per magnitude-bin median, mean, (histogram) mode, and 3σ-MAD-clipped
    median of the residual. The clipped median is the statistic the faint
    correction actually subtracts, so it should sit ≈0 after calibration."""
    cen  = 0.5 * (edges[:-1] + edges[1:])
    med  = np.full(len(cen), np.nan)
    mean = np.full(len(cen), np.nan)
    mode = np.full(len(cen), np.nan)
    cmed = np.full(len(cen), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        v = resid[(mag >= lo) & (mag < hi)]
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        med[i]  = float(np.median(v))
        mean[i] = float(np.mean(v))
        _mad = float(np.median(np.abs(v - med[i])))
        vg   = v[np.abs(v - med[i]) < 3.0 * 1.4826 * _mad] if _mad > 0 else v
        if len(vg) >= 3:
            cmed[i] = float(np.median(vg))
        c_lo, c_hi = np.percentile(v, [2, 98])
        if c_hi > c_lo:
            h, e = np.histogram(v, bins=41, range=(c_lo, c_hi))
            h = np.convolve(h.astype(float), np.ones(3) / 3, mode="same")
            bc = 0.5 * (e[:-1] + e[1:])
            mode[i] = float(bc[int(np.argmax(h))])
    return cen, med, mean, mode, cmed


def _faint_residual_panel(ax, resid_dir) -> None:
    """Full-sample residual (mmag) vs calibrated magnitude, with per-bin
    median / mean / mode overplotted.  If median≈0 after calibration but the
    mode is offset at the faint end, the median faint correction is centring a
    skewed distribution on the wrong point (over/under-correcting the bulk).
    Reads mag_all + dm_all_post from *_resid.npz."""
    ax.set_title("Full-sample residual vs magnitude\n(after calib; median / mean / mode)")
    ax.set_xlabel("Calibrated magnitude (AB)")
    ax.set_ylabel("Residual: measured − ZTF ref (mmag)")

    mags, res = [], []
    if resid_dir is not None:
        for p in sorted(Path(resid_dir).glob("*_resid.npz")):
            try:
                d = np.load(str(p))
            except Exception:
                continue
            if "mag_all" not in d or "dm_all_post" not in d:
                continue
            mags.append(np.asarray(d["mag_all"], float))
            res.append(np.asarray(d["dm_all_post"], float) * 1000.0)  # mag → mmag
    if not mags:
        ax.text(0.5, 0.5, "no mag_all in resid.npz\n(re-run recalibrate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    mag   = np.concatenate(mags)
    resid = np.concatenate(res)
    ok    = np.isfinite(mag) & np.isfinite(resid)
    mag, resid = mag[ok], resid[ok]
    if len(mag) < 100:
        ax.text(0.5, 0.5, "too few sources", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return

    m_lo, m_hi = np.percentile(mag, [1, 99.5])
    ylim = 120.0

    # Per-magnitude-column normalised density: each 0.1-mag column is scaled to its
    # own peak, so the residual distribution *shape* is visible at every magnitude
    # (faint bins don't drown the plot in black) and skew shows as an asymmetric
    # column with its bright core off the median line.
    xbins = np.arange(np.floor(m_lo * 2) / 2, m_hi + 0.1, 0.1)
    ybins = np.linspace(-ylim, ylim, 121)
    H, xe, ye = np.histogram2d(mag, resid, bins=[xbins, ybins])
    colmax = H.max(axis=1, keepdims=True)
    Hn = np.divide(H, colmax, out=np.zeros_like(H), where=colmax > 0)
    ax.imshow(Hn.T, origin="lower", aspect="auto",
              extent=[xe[0], xe[-1], ye[0], ye[-1]],
              cmap="magma", vmin=0, vmax=1, interpolation="nearest")

    edges = np.arange(np.floor(m_lo), np.ceil(m_hi) + 0.25, 0.25)
    cen, med, mean, mode, cmed = _binned_center_curves(mag, resid, edges)
    _stroke = [pe.withStroke(linewidth=2.6, foreground="black")]
    ax.plot(cen, cmed, "-",  color="white",       lw=2.4, label="clipped median (correction target)", path_effects=_stroke)
    ax.plot(cen, med,  "-",  color="0.7",         lw=1.2, label="median (raw)",        path_effects=_stroke)
    ax.plot(cen, mean, "--", color="deepskyblue", lw=1.6, label="mean",               path_effects=_stroke)
    ax.plot(cen, mode, ":",  color="lime",        lw=2.2, label="mode (bulk)",        path_effects=_stroke)
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)

    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(xe[0], xe[-1])
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.text(0.98, 0.03, "density normalised per magnitude column",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.8")


def make_rms(cal_dir: Path, out_path: Path, tag: str = "",
             resid_dir: Path | None = None) -> None:
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
    ax.set_title("Calibrators: distribution per calibration step")
    ax.set_ylim(bottom=0)

    # ── Panel 2 (bottom-right): full-sample residual vs magnitude ──
    # (below the correction it results from, so the right column reads
    #  correction → corrected top-to-bottom)
    _faint_residual_panel(axes[1, 1], resid_dir)

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
        # APCORR46 (mmag) was absorbed into CALIB_N; add it back so zpc is
        # centred on the true photometric ZP variation, not the aperture offset.
        apcorr_vals = df["APCORR46"].reindex(idx).values if "APCORR46" in df.columns else np.zeros(len(idx))
        apcorr_vals = np.where(np.isfinite(apcorr_vals), apcorr_vals, 0.0)
        zpc = zpc + apcorr_vals
        med_apcorr  = float(np.nanmedian(apcorr_vals))
        sc  = ax.scatter(s, zpc, c=mlv, cmap="plasma", s=10, alpha=0.7,
                         vmin=np.nanpercentile(mlv, 5), vmax=np.nanpercentile(mlv, 95))
        plt.colorbar(sc, ax=ax, label="MAGLIM (mag)")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.text(0.03, 0.97,
                f"med = {float(np.nanmedian(zpc)):.1f} mmag\nσ = {float(np.nanstd(zpc)):.1f} mmag",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.text(0.97, 0.03,
                f"AperCorr 4→6px = {med_apcorr:.1f} mmag",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xlabel("Seeing (arcsec)")
    ax.set_ylabel("Linear ZP correction @ mag 17 (mmag)")
    ax.set_title("ZP correction vs seeing\n(coloured by limiting magnitude)")

    # ── Panel 4 (top-right): faint correction curves ──
    ax = axes[0, 1]
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
