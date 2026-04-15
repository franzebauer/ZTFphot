"""
plot_lightcurve.py
------------------
Per-quadrant 2-panel light curve figure:

  Top:    target MJD vs MAG_4_TOT_AB, scatter coloured by MAGLIM
  Bottom: 5 IS_GOOD calibration stars nearest in magnitude to the target,
          excluding high-sigma outliers (σ > 2× median of vet stars)

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
    """Return (object_index, median_mag) of the closest source to tgt_coord."""
    if "ALPHAWIN_REF" not in clean.columns:
        return None, np.nan
    srcs = clean.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        return None, np.nan
    cats = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                    dec=srcs["DELTAWIN_REF"].values * u.deg)
    idx, sep, _ = tgt_coord.match_to_catalog_sky(cats)
    if sep[0].arcsec > 3.0:
        return None, np.nan
    tgt_obj  = srcs.index[int(idx)]
    tgt_mags = pd.to_numeric(
        clean.loc[clean["object_index"] == tgt_obj, _MAG_COL], errors="coerce")
    return tgt_obj, float(tgt_mags.median())


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
    excluding tgt_obj and high-sigma outliers (sigma > 2× median of vet stars).
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
            cands = cands[cands["std_mag"] <= 2.0 * med_sigma]
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


def make_fig4_lightcurves(lc_path: Path, out_path: Path,
                          target_ra: float, target_dec: float,
                          tag: str = "",
                          vet_catalog: Path | None = None,
                          n_comp: int = 5) -> None:
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
    tgt_obj, tgt_med = _find_target(clean, tgt_coord)
    if tgt_obj is None:
        logger.warning(f"  target not found within 3\" in {tag}")
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
                            ms=5, elinewidth=0.6, alpha=0.8,
                            label=f"Star {ci + 1}  med={comp_med:.2f}")

        ax_bot.set_ylim(ylim)
        ax_bot.set_ylabel("Calibrated magnitude (AB)", fontsize=10)
        ax_bot.set_xlabel("MJD", fontsize=10)
        ax_bot.set_title(
            f"Nearest {len(comp_idxs)} IS_GOOD calibration stars  "
            f"(excl. σ > 2× median)", fontsize=10)
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
