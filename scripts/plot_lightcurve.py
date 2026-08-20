"""
plot_lightcurve.py
------------------
Per-quadrant 2-panel light curve figure:

  Top:    target MJD vs MAG_4_TOT_AB, scatter coloured by MAGLIM
  Bottom: 5 IS_GOOD calibration stars nearest in magnitude to the target,
          excluding high-sigma outliers (σ > 4× median of vet stars)

Both panels share the same y-axis range (10–90th percentile of target mags ± 0.3).
Panel widths are equal (colorbar occupies a dedicated narrow column).

Key parquet columns used:
    MAG_4_TOT_AB, MERR_4_TOT_AB  — calibrated magnitude and error
    INFOBITS_DIF                  — quality flag (== 0 for clean epochs)
    OBSMJD, MAGLIM                — epoch metadata
    CLASS_STAR, CLASS_STAR_OBJ    — stellarity
    ALPHAWIN_REF, DELTAWIN_REF    — reference-catalog positions
    object_index                  — source identifier
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

_MAG_COL  = "MAG_4_TOT_AB"
_MERR_COL = "MERR_4_TOT_AB"
_COMP_COLORS = ["C1", "C2", "C3", "C4", "C5"]


def _find_target(clean: pd.DataFrame, tgt_coord: SkyCoord):
    """Return (object_index, median_mag, sep_arcsec) of closest source to tgt_coord.

    sep_arcsec is always the nearest-source distance regardless of the 3\" threshold,
    so callers can report it even when the target is not found.
    """
    if "ALPHAWIN_REF" not in clean.columns:
        return None, np.nan, np.nan
    srcs = clean.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        return None, np.nan, np.nan
    cats = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                    dec=srcs["DELTAWIN_REF"].values * u.deg)
    idx, sep, _ = tgt_coord.match_to_catalog_sky(cats)
    sep_arcsec = float(sep[0].arcsec)
    if sep_arcsec > 3.0:
        return None, np.nan, sep_arcsec
    tgt_obj  = srcs.index[int(idx)]
    tgt_mags = pd.to_numeric(
        clean.loc[clean["object_index"] == tgt_obj, _MAG_COL], errors="coerce")
    return tgt_obj, float(tgt_mags.median()), sep_arcsec


def _make_no_target_plot(out_path: Path, tag: str,
                         target_ra: float, target_dec: float,
                         sep_arcsec: float) -> None:
    """Generate a placeholder figure when the target is not found in the parquet."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_visible(False)
    sep_str = f"{sep_arcsec:.1f}\"" if np.isfinite(sep_arcsec) else "unknown"
    msg = (f"Target not found in {tag}\n"
           f"RA = {target_ra:.5f}   Dec = {target_dec:+.5f}\n"
           f"Nearest reference source: {sep_str} away\n"
           f"Target likely falls on masked or edge pixels — no light curve available.")
    fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12,
             color="firebrick", transform=fig.transFigure,
             bbox=dict(boxstyle="round,pad=0.6", fc="mistyrose", ec="firebrick", lw=1.5))
    fig.suptitle(f"Light curve — {tag}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  no-target placeholder → {out_path}")


def _load_vet_good_indices(vet_catalog: Path, clean: pd.DataFrame) -> set:
    """Return set of object_index for sources marked IS_GOOD=True in the vet catalog."""
    if vet_catalog is None or not vet_catalog.exists():
        return set()
    try:
        from astropy.io import fits
        with fits.open(str(vet_catalog)) as h:
            vd = h[1].data
        vet_ra   = vd["ALPHAWIN_J2000"].astype(float)
        vet_dec  = vd["DELTAWIN_J2000"].astype(float)
        vet_good = vd["IS_GOOD"].astype(bool)
        good_ra  = vet_ra[vet_good]
        good_dec = vet_dec[vet_good]
        ok = np.isfinite(good_ra) & np.isfinite(good_dec)
        if not ok.any():
            return set()
        srcs = clean.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
        if srcs.empty:
            return set()
        cat_src  = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                            dec=srcs["DELTAWIN_REF"].values * u.deg)
        cat_good = SkyCoord(ra=good_ra[ok] * u.deg, dec=good_dec[ok] * u.deg)
        idx, sep, _ = cat_good.match_to_catalog_sky(cat_src)
        matched = sep.arcsec < 3.0
        return set(srcs.index[idx[matched]].tolist())
    except Exception as e:
        logger.warning(f"Could not load vet catalog {vet_catalog}: {e}")
        return set()


def _pick_comps(clean: pd.DataFrame, tgt_obj, tgt_med: float,
                vet_good: set, n_comp: int = 5) -> list:
    """
    Select n_comp IS_GOOD calibration stars nearest in magnitude to tgt_med,
    excluding tgt_obj and high-sigma outliers (sigma > 4× median of vet stars).
    Falls back to all CLASS_STAR > 0.7 sources if vet catalog unavailable.
    """
    grp     = clean.groupby("object_index")
    med_mag = grp[_MAG_COL].median()
    std_mag = grp[_MAG_COL].std()

    per_src = pd.DataFrame({"med_mag": med_mag, "std_mag": std_mag})
    per_src = per_src[per_src["med_mag"].between(10, 26) & (per_src.index != tgt_obj)]

    if vet_good:
        # Select from IS_GOOD calibration stars; drop high-sigma outliers
        cands = per_src[per_src.index.isin(vet_good)].copy()
        if not cands.empty:
            med_sigma = float(cands["std_mag"].median())
            cands = cands[cands["std_mag"] <= 4.0 * med_sigma]
    else:
        # No vet catalog: fall back to CLASS_STAR > 0.7
        cs_col = ("CLASS_STAR" if "CLASS_STAR" in clean.columns
                  else "CLASS_STAR_OBJ" if "CLASS_STAR_OBJ" in clean.columns
                  else None)
        if cs_col:
            med_cs = pd.to_numeric(grp[cs_col].median(), errors="coerce")
            per_src["med_cs"] = med_cs
            cands = per_src[per_src["med_cs"] > 0.7].copy()
        else:
            cands = per_src.copy()

    if cands.empty:
        logger.warning("  _pick_comps: no candidates found")
        return []

    cands["delta"] = (cands["med_mag"] - tgt_med).abs()
    return cands.sort_values("delta").index[:n_comp].tolist()


def _bin_series(mjd, mag, merr, bin_days: float):
    """Bin a light curve into fixed bin_days windows anchored at the first epoch.

    Returns (bin_mjd, bin_mag, bin_err). Each bin's magnitude is an inverse-variance
    weighted mean (unweighted mean where errors are missing); bin_mjd is the mean
    epoch in the bin; bin_err is the weighted error 1/sqrt(Σ 1/σ²), or the standard
    error of the mean, whichever is larger. Only bins that contain at least one epoch
    are returned — empty windows are dropped, not plotted.
    """
    mjd = np.asarray(mjd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    merr = (np.asarray(merr, dtype=float) if merr is not None
            else np.full(mag.shape, np.nan))
    ok = np.isfinite(mjd) & np.isfinite(mag)
    mjd, mag, merr = mjd[ok], mag[ok], merr[ok]
    if mjd.size == 0 or bin_days <= 0:
        return np.array([]), np.array([]), np.array([])
    bin_idx = np.floor((mjd - mjd.min()) / bin_days).astype(int)
    bmjd, bmag, berr = [], [], []
    for b in np.unique(bin_idx):
        m = bin_idx == b
        x, y, e = mjd[m], mag[m], merr[m]
        w = np.where(np.isfinite(e) & (e > 0), 1.0 / e ** 2, 0.0)
        if w.sum() > 0:
            ymean = float(np.sum(w * y) / np.sum(w))
            ew = float(1.0 / np.sqrt(np.sum(w)))
        else:
            ymean = float(np.mean(y))
            ew = np.nan
        sem = float(np.std(y) / np.sqrt(len(y))) if len(y) > 1 else 0.0
        bmjd.append(float(np.mean(x)))
        bmag.append(ymean)
        berr.append(float(np.nanmax([ew, sem])))
    return np.array(bmjd), np.array(bmag), np.array(berr)


def make_lightcurves(lc_path: Path, out_path: Path,
                          target_ra: float, target_dec: float,
                          tag: str = "",
                          vet_catalog: Path | None = None,
                          n_comp: int = 5,
                          bin_days: float = 50.0) -> None:
    """
    Two-panel light curve figure for one quadrant.

    Top panel:    target LC coloured by MAGLIM
    Bottom panel: n_comp stellar (CLASS_STAR > 0.7) comparison objects,
                  excluding any source in the vet calibration catalog
    """
    try:
        df = pd.read_parquet(lc_path)
    except Exception as e:
        logger.warning(f"  cannot read {lc_path}: {e}")
        return

    if _MAG_COL not in df.columns:
        logger.warning(f"  {_MAG_COL} missing — skipping lightcurve plot")
        return

    df[_MAG_COL] = pd.to_numeric(df[_MAG_COL], errors="coerce")
    clean = df[df["INFOBITS_DIF"] == 0].copy()

    tgt_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    tgt_obj, tgt_med, sep_arcsec = _find_target(clean, tgt_coord)
    if tgt_obj is None:
        logger.warning(f"  target not found within 3\" in {tag} (nearest: {sep_arcsec:.1f}\")")
        _make_no_target_plot(out_path, tag, target_ra, target_dec, sep_arcsec)
        return

    tgt_rows = (clean[clean["object_index"] == tgt_obj]
                .sort_values("OBSMJD").copy())
    tgt_rows[_MAG_COL] = pd.to_numeric(tgt_rows[_MAG_COL], errors="coerce")
    tgt_rows = tgt_rows[tgt_rows[_MAG_COL].between(10, 26)]
    if tgt_rows.empty:
        return

    vet_good  = _load_vet_good_indices(vet_catalog, clean)
    comp_idxs = _pick_comps(clean, tgt_obj, tgt_med, vet_good, n_comp)

    # ── shared y-axis range ───────────────────────────────────────────────────
    tgt_mag_vals = tgt_rows[_MAG_COL].values
    tgt_mag_vals = tgt_mag_vals[np.isfinite(tgt_mag_vals)]
    if len(tgt_mag_vals) >= 2:
        ylo = float(np.percentile(tgt_mag_vals, 10)) - 0.3
        yhi = float(np.percentile(tgt_mag_vals, 90)) + 0.3
    else:
        ylo, yhi = tgt_med - 0.5, tgt_med + 0.5
    # magnitude convention: faint (large) at bottom, bright (small) at top
    ylim = (yhi, ylo)

    # ── figure: 2 rows × 2 cols, narrow col-1 for colorbar ───────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[20, 1],
                           hspace=0.32, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    cax    = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).set_visible(False)   # spacer to balance widths

    fig.suptitle(f"Light curve — {tag}  (target RA={target_ra:.4f} Dec={target_dec:+.4f})",
                 fontsize=11)

    # ── Top panel: target LC coloured by MAGLIM ───────────────────────────────
    mjd  = tgt_rows["OBSMJD"].values
    mag  = tgt_rows[_MAG_COL].values
    merr = (pd.to_numeric(tgt_rows[_MERR_COL], errors="coerce").values
            if _MERR_COL in tgt_rows.columns else np.full(len(mag), np.nan))
    ml   = (pd.to_numeric(tgt_rows["MAGLIM"], errors="coerce").values
            if "MAGLIM" in tgt_rows.columns else np.full(len(mag), np.nan))

    ok = np.isfinite(mag) & np.isfinite(mjd)
    if np.any(np.isfinite(ml[ok])):
        c_norm = mcolors.Normalize(vmin=np.nanpercentile(ml[ok], 5),
                                   vmax=np.nanpercentile(ml[ok], 95))
        sc = ax_top.scatter(mjd[ok], mag[ok], c=ml[ok], cmap="plasma",
                            norm=c_norm, s=20, zorder=3)
        fig.colorbar(sc, cax=cax, label="MAGLIM (mag)")
        if np.any(np.isfinite(merr[ok])):
            ax_top.errorbar(mjd[ok], mag[ok], yerr=merr[ok],
                            fmt="none", ecolor="grey", elinewidth=0.6,
                            alpha=0.5, zorder=2)
    else:
        cax.set_visible(False)
        ax_top.errorbar(mjd[ok], mag[ok],
                        yerr=merr[ok] if np.any(np.isfinite(merr[ok])) else None,
                        fmt=".", color="black", ms=5, elinewidth=0.7, alpha=0.85)

    tgt_std = float(np.nanstd(mag[ok]))

    # binned target LC: weighted mean per bin_days window, black squares (no line)
    bmjd, bmag, berr = _bin_series(mjd[ok], mag[ok], merr[ok], bin_days)
    if bmjd.size:
        ax_top.errorbar(bmjd, bmag, yerr=berr, fmt="s-", mfc="none",
                        mec="black", ecolor="black", color="black",
                        mew=1.0, lw=1.2,
                        ms=7, elinewidth=1.4, capsize=2, zorder=5,
                        label=f"{bin_days:g}-day binned")
        ax_top.legend(fontsize=8, loc="upper right")

    ax_top.set_ylim(ylim)
    ax_top.set_ylabel("Calibrated magnitude (AB)", fontsize=10)
    ax_top.set_xlabel("MJD", fontsize=10)
    ax_top.set_title(
        f"Target  med={tgt_med:.2f}  σ={tgt_std*1000:.0f} mmag  N={int(ok.sum())}",
        fontsize=10)
    ax_top.tick_params(labelsize=9)
    ax_top.grid(True, alpha=0.2)

    # ── Bottom panel: stellar comparison objects ───────────────────────────────
    if comp_idxs:
        for ci, (comp_oi, color) in enumerate(zip(comp_idxs, _COMP_COLORS)):
            crow = (clean[clean["object_index"] == comp_oi]
                    .sort_values("OBSMJD").copy())
            crow[_MAG_COL] = pd.to_numeric(crow[_MAG_COL], errors="coerce")
            crow = crow[crow[_MAG_COL].between(10, 26)]
            if crow.empty:
                continue
            comp_med = float(crow[_MAG_COL].median())
            cerr = (pd.to_numeric(crow[_MERR_COL], errors="coerce").values
                    if _MERR_COL in crow.columns else None)
            ax_bot.errorbar(crow["OBSMJD"].values, crow[_MAG_COL].values,
                            yerr=cerr, fmt=".", color=color,
                            ms=5, elinewidth=0.6, alpha=0.4,
                            label=f"Star {ci + 1}  med={comp_med:.2f}")

            # binned comparison LC: squares in the star's colour, thicker errorbars
            bcmjd, bcmag, bcerr = _bin_series(
                crow["OBSMJD"].values, crow[_MAG_COL].values, cerr, bin_days)
            if bcmjd.size:
                ax_bot.errorbar(bcmjd, bcmag, yerr=bcerr, fmt="s-", mfc="none",
                                mec=color, ecolor=color, color=color,
                                mew=1.0, lw=1.2,
                                ms=7, elinewidth=1.4, capsize=2, zorder=5)

        ax_bot.set_ylim(ylim)
        ax_bot.set_ylabel("Calibrated magnitude (AB)", fontsize=10)
        ax_bot.set_xlabel("MJD", fontsize=10)
        ax_bot.set_title(
            f"Nearest {len(comp_idxs)} IS_GOOD calibration stars  "
            f"(excl. σ > 4× median)", fontsize=10)
        ax_bot.tick_params(labelsize=9)
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.grid(True, alpha=0.2)
    else:
        ax_bot.set_visible(False)
        logger.warning(f"  no comparison stars found for {tag}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info(f"  lightcurves → {out_path}")


_FLUX_COL = "FLUX_4_TOT_AB"
_FERR_COL = "FERR_4_TOT_AB"


def make_lightcurves_flux(lc_path: Path, out_path: Path,
                          target_ra: float, target_dec: float,
                          tag: str = "",
                          vet_catalog: Path | None = None,
                          n_comp: int = 5,
                          bin_days: float = 50.0) -> None:
    """
    Flux (μJy) equivalent of make_lightcurves. Difference photometry is bipolar, so
    this shows the fainter-than-reference (negative-flux) epochs the magnitude drops
    to NaN. Top: target FLUX_4_TOT_AB vs MJD with a zero line; bottom: comparison
    stars. Each panel auto-scales (the target sits near 0 μJy, comps near their
    reference flux).
    """
    try:
        df = pd.read_parquet(lc_path)
    except Exception as e:
        logger.warning(f"  cannot read {lc_path}: {e}")
        return
    if _FLUX_COL not in df.columns:
        logger.info(f"  {_FLUX_COL} missing — skipping flux lightcurve plot")
        return

    df[_FLUX_COL] = pd.to_numeric(df[_FLUX_COL], errors="coerce")
    clean = df[df["INFOBITS_DIF"] == 0].copy()

    tgt_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    tgt_obj, tgt_med, sep_arcsec = _find_target(clean, tgt_coord)
    if tgt_obj is None:
        logger.warning(f"  target not found within 3\" in {tag} (nearest: {sep_arcsec:.1f}\") [flux]")
        _make_no_target_plot(out_path, tag, target_ra, target_dec, sep_arcsec)
        return

    tgt_rows = (clean[clean["object_index"] == tgt_obj].sort_values("OBSMJD").copy())
    tgt_rows = tgt_rows[np.isfinite(pd.to_numeric(tgt_rows[_FLUX_COL], errors="coerce"))]
    if tgt_rows.empty:
        return

    vet_good  = _load_vet_good_indices(vet_catalog, clean)
    comp_idxs = _pick_comps(clean, tgt_obj, tgt_med, vet_good, n_comp)

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[20, 1], hspace=0.32, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    cax    = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).set_visible(False)
    fig.suptitle(f"Flux light curve — {tag}  (target RA={target_ra:.4f} Dec={target_dec:+.4f})",
                 fontsize=11)

    mjd  = tgt_rows["OBSMJD"].values
    flux = pd.to_numeric(tgt_rows[_FLUX_COL], errors="coerce").values
    ferr = (pd.to_numeric(tgt_rows[_FERR_COL], errors="coerce").values
            if _FERR_COL in tgt_rows.columns else np.full(len(flux), np.nan))
    ml   = (pd.to_numeric(tgt_rows["MAGLIM"], errors="coerce").values
            if "MAGLIM" in tgt_rows.columns else np.full(len(flux), np.nan))
    ok = np.isfinite(flux) & np.isfinite(mjd)

    ax_top.axhline(0.0, color="grey", lw=0.8, ls="--", zorder=1)
    if np.any(np.isfinite(ml[ok])):
        c_norm = mcolors.Normalize(vmin=np.nanpercentile(ml[ok], 5),
                                   vmax=np.nanpercentile(ml[ok], 95))
        sc = ax_top.scatter(mjd[ok], flux[ok], c=ml[ok], cmap="plasma",
                            norm=c_norm, s=20, zorder=3)
        fig.colorbar(sc, cax=cax, label="MAGLIM (mag)")
        if np.any(np.isfinite(ferr[ok])):
            ax_top.errorbar(mjd[ok], flux[ok], yerr=ferr[ok], fmt="none",
                            ecolor="grey", elinewidth=0.6, alpha=0.5, zorder=2)
    else:
        cax.set_visible(False)
        ax_top.errorbar(mjd[ok], flux[ok],
                        yerr=ferr[ok] if np.any(np.isfinite(ferr[ok])) else None,
                        fmt=".", color="black", ms=5, elinewidth=0.7, alpha=0.85)

    fv = flux[ok]
    if len(fv) >= 2:
        lo, hi = float(np.percentile(fv, 2)), float(np.percentile(fv, 98))
        pad = 0.1 * (hi - lo + 1e-9)
        ax_top.set_ylim(lo - pad, hi + pad)

    bmjd, bflux, bferr = _bin_series(mjd[ok], flux[ok], ferr[ok], bin_days)
    if bmjd.size:
        ax_top.errorbar(bmjd, bflux, yerr=bferr, fmt="s-", mfc="none",
                        mec="black", ecolor="black", color="black", mew=1.0, lw=1.2,
                        ms=7, elinewidth=1.4, capsize=2, zorder=5,
                        label=f"{bin_days:g}-day binned")
        ax_top.legend(fontsize=8, loc="upper right")

    ax_top.set_ylabel("Flux (μJy)", fontsize=10)
    ax_top.set_xlabel("MJD", fontsize=10)
    ax_top.set_title(
        f"Target  mean={np.nanmean(fv):.1f} μJy  σ={np.nanstd(fv):.1f} μJy  "
        f"N={int(ok.sum())}  (finite mag: {int(np.isfinite(pd.to_numeric(tgt_rows['MAG_4_TOT_AB'], errors='coerce')).sum()) if 'MAG_4_TOT_AB' in tgt_rows.columns else 0})",
        fontsize=10)
    ax_top.tick_params(labelsize=9)
    ax_top.grid(True, alpha=0.2)

    if comp_idxs:
        ax_bot.axhline(0.0, color="grey", lw=0.8, ls="--", zorder=1)
        for ci, (comp_oi, color) in enumerate(zip(comp_idxs, _COMP_COLORS)):
            crow = clean[clean["object_index"] == comp_oi].sort_values("OBSMJD").copy()
            cfv = pd.to_numeric(crow[_FLUX_COL], errors="coerce").values
            keep = np.isfinite(cfv)
            if not keep.any():
                continue
            cmjd = crow["OBSMJD"].values[keep]; cflux = cfv[keep]
            cferr = (pd.to_numeric(crow[_FERR_COL], errors="coerce").values[keep]
                     if _FERR_COL in crow.columns else None)
            ax_bot.errorbar(cmjd, cflux, yerr=cferr, fmt=".", color=color,
                            ms=5, elinewidth=0.6, alpha=0.4,
                            label=f"Star {ci + 1}  med={np.nanmedian(cflux):.0f}")
            bcmjd, bcf, bcfe = _bin_series(cmjd, cflux, cferr, bin_days)
            if bcmjd.size:
                ax_bot.errorbar(bcmjd, bcf, yerr=bcfe, fmt="s-", mfc="none",
                                mec=color, ecolor=color, color=color, mew=1.0, lw=1.2,
                                ms=7, elinewidth=1.4, capsize=2, zorder=5)
        ax_bot.set_ylabel("Flux (μJy)", fontsize=10)
        ax_bot.set_xlabel("MJD", fontsize=10)
        ax_bot.set_title(f"Nearest {len(comp_idxs)} IS_GOOD calibration stars", fontsize=10)
        ax_bot.tick_params(labelsize=9)
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.grid(True, alpha=0.2)
    else:
        ax_bot.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info(f"  flux lightcurves → {out_path}")
