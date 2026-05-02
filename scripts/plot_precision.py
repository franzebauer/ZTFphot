"""
plot_precision.py
-----------------
Photometric precision and astrometric scatter (2×2):

  Top-left:  σ_mag vs median calibrated magnitude, coloured by N clean detections
  Top-right: σ_mag vs median calibrated magnitude, coloured by CLASS_STAR (0=gal, 1=star)
             Both top panels:
               • vertical dotted lines at calibration-star magnitude limits (14–19 mag)
               • target marked with red star
               • vet-rejected sources shown as open grey circles

  Bot-left:  nearest-neighbour separation histogram by magnitude bin
  Bot-right: Δposition histogram by magnitude bin (13–17, 17–19, 19–21, 21–23)

Key parquet columns used:
    MAG_4_TOT_AB, MERR_4_TOT_AB  — calibrated magnitude and error
    INFOBITS_DIF                  — quality flag (== 0 for clean epochs)
    CLASS_STAR                    — stellarity
    ALPHAWIN_REF, DELTAWIN_REF    — reference-catalog positions (for target match)
    ALPHA_OBJ, DELTA_OBJ          — per-epoch measured positions (for Δpos)
    SEEING, MAGLIM                — epoch metadata
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

logger = logging.getLogger(__name__)

_MAG_COL  = "MAG_4_TOT_AB"
_CALIB_MAG_LO = 14.0
_CALIB_MAG_HI = 19.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_target_obj(df: pd.DataFrame, target_ra: float, target_dec: float):
    if "ALPHAWIN_REF" not in df.columns:
        return None
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        return None
    cats = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                    dec=srcs["DELTAWIN_REF"].values * u.deg)
    tgt  = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    idx, sep, _ = tgt.match_to_catalog_sky(cats)
    if sep[0].arcsec < 2.0:
        return srcs.index[int(idx)]
    return None


def _load_vet_rejected(vet_catalog: Path, df: pd.DataFrame) -> set:
    """Return set of object_index values flagged IS_GOOD=False in the vet catalog."""
    if vet_catalog is None or not vet_catalog.exists():
        return set()
    try:
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        with fits.open(str(vet_catalog)) as h:
            vd = h[1].data
        vet_ra   = vd["ALPHAWIN_J2000"].astype(float)
        vet_dec  = vd["DELTAWIN_J2000"].astype(float)
        vet_good = vd["IS_GOOD"].astype(bool)
        bad_ra   = vet_ra[~vet_good]
        bad_dec  = vet_dec[~vet_good]
        if len(bad_ra) == 0:
            return set()
        srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
        if srcs.empty:
            return set()
        cat_src = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                           dec=srcs["DELTAWIN_REF"].values * u.deg)
        cat_bad = SkyCoord(ra=bad_ra * u.deg, dec=bad_dec * u.deg)
        idx, sep, _ = cat_bad.match_to_catalog_sky(cat_src)
        matched = sep.arcsec < 3.0
        return set(srcs.index[idx[matched]].tolist())
    except Exception as e:
        logger.warning(f"Could not load vet catalog {vet_catalog}: {e}")
        return set()


def _running_median(grp: pd.DataFrame, edges: np.ndarray):
    cx, my = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = grp[(grp["med"] >= lo) & (grp["med"] < hi)]["std"] * 1000
        if len(s) >= 5:
            cx.append(0.5 * (lo + hi))
            my.append(float(np.median(s)))
    return cx, my


# ── main function ─────────────────────────────────────────────────────────────

def make_precision(lc_path: Path, out_path: Path, tag: str = "",
                        target_ra: float | None = None,
                        target_dec: float | None = None,
                        vet_catalog: Path | None = None) -> None:
    df = pd.read_parquet(lc_path)

    if _MAG_COL not in df.columns:
        logger.warning(f"  {_MAG_COL} missing — skipping precision plot")
        return

    df[_MAG_COL] = pd.to_numeric(df[_MAG_COL], errors="coerce")

    clean = df[df["INFOBITS_DIF"] == 0].copy()

    # ── per-source summary for top panels ─────────────────────────────────────
    agg = clean.groupby("object_index").agg(
        n=(_MAG_COL, "count"),
        med=(_MAG_COL, "median"),
        std=(_MAG_COL, "std"),
    ).dropna(subset=["std"])
    cs_col = "CLASS_STAR" if "CLASS_STAR" in clean.columns else \
             "CLASS_STAR_OBJ" if "CLASS_STAR_OBJ" in clean.columns else None
    if cs_col is not None:
        agg["cs"] = clean.groupby("object_index")[cs_col].median()
    else:
        agg["cs"] = np.nan
    grp = agg[agg["n"] >= 5]

    # ── per-source summary from pre-calibration mags (aperture-corrected maginst)
    _ORG_COL = "MAG_4_TOT_AB_org"
    grp_org = None
    if _ORG_COL in clean.columns:
        clean[_ORG_COL] = pd.to_numeric(clean[_ORG_COL], errors="coerce")
        agg_org = clean.groupby("object_index").agg(
            n_org=(_ORG_COL, "count"),
            med=(_ORG_COL,   "median"),
            std=(_ORG_COL,   "std"),
        ).dropna(subset=["std"])
        grp_org = agg_org[agg_org["n_org"] >= 5]

    tgt_obj_idx  = _find_target_obj(df, target_ra, target_dec) if target_ra is not None else None
    vet_rejected = _load_vet_rejected(vet_catalog, df)

    edges = np.arange(13, 23.5, 0.5)
    nc_rms4_med = float(df["NC_RMS4"].median()) if "NC_RMS4" in df.columns else np.nan
    med_maglim  = float(df["MAGLIM"].median())  if "MAGLIM"  in df.columns else None

    # ── per-epoch Δposition for bottom panels ─────────────────────────────────
    pos_data = None

    _ra_col  = "ALPHA_OBJ"  if "ALPHA_OBJ"  in clean.columns else \
               "ALPHA_SCI" if "ALPHA_SCI" in clean.columns else None
    _dec_col = "DELTA_OBJ"  if "DELTA_OBJ"  in clean.columns else \
               "DELTA_SCI" if "DELTA_SCI" in clean.columns else None
    req_cols = {_ra_col, _dec_col, "OBSMJD"} if _ra_col else set()
    if req_cols and req_cols.issubset(clean.columns):
        pos = clean[["object_index", _MAG_COL, "OBSMJD",
                     _ra_col, _dec_col]].copy()
        pos = pos.rename(columns={_ra_col: "ALPHA_OBJ", _dec_col: "DELTA_OBJ"})
        for c in pos.columns:
            pos[c] = pd.to_numeric(pos[c], errors="coerce")
        pos = pos.dropna(subset=["ALPHA_OBJ", "DELTA_OBJ", _MAG_COL])

        # per-source Δpos relative to median epoch position
        med_pos = pos.groupby("object_index")[["ALPHA_OBJ", "DELTA_OBJ"]].median()
        pos = pos.join(med_pos, on="object_index", rsuffix="_med")
        cos_dec = np.cos(np.radians(pos["DELTA_OBJ_med"].values))
        dra  = (pos["ALPHA_OBJ"].values - pos["ALPHA_OBJ_med"].values) * cos_dec * 3600
        ddec = (pos["DELTA_OBJ"].values - pos["DELTA_OBJ_med"].values) * 3600
        pos["dpos"] = np.sqrt(dra**2 + ddec**2)
        pos = pos.join(grp[["med"]], on="object_index")
        pos_data = pos.dropna(subset=["dpos", "med"])

    # ── nearest-neighbour separations ────────────────────────────────────────
    nn_data = None  # (sep_arcsec, med_mag) per source
    _pos_col = ("ALPHAWIN_REF", "DELTAWIN_REF") if "ALPHAWIN_REF" in df.columns \
               else ("ALPHA_OBJ",  "DELTA_OBJ")  if "ALPHA_OBJ"  in df.columns \
               else ("ALPHA_SCI",  "DELTA_SCI")  if "ALPHA_SCI"  in df.columns \
               else (None, None)
    if _pos_col[0] is not None:
        srcs = df.groupby("object_index")[list(_pos_col)].first().dropna()
        srcs.columns = ["ra", "dec"]
        if len(srcs) >= 2:
            from scipy.spatial import KDTree
            cos_dec = np.cos(np.radians(srcs["dec"].values.mean()))
            xy  = np.column_stack([srcs["ra"].values * cos_dec, srcs["dec"].values])
            dists, _ = KDTree(xy).query(xy, k=2)
            sep = dists[:, 1] * 3600  # deg → arcsec
            nn_df = pd.DataFrame({"sep": sep}, index=srcs.index)
            nn_df = nn_df.join(grp[["med"]], how="left")
            nn_data = nn_df.dropna(subset=["sep"])

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             gridspec_kw=dict(wspace=0.25, hspace=0.32))
    fig.suptitle(f"Photometric precision & astrometry — {tag}", fontsize=13)

    # ── top panels ────────────────────────────────────────────────────────────
    for col_idx, (color_col, cmap_name, clabel) in enumerate([
        ("n",  "viridis", "N clean detections"),
        ("cs", "RdYlGn",  "CLASS_STAR  (0=gal, 1=star)"),
    ]):
        ax = axes[0, col_idx]
        c_vals = grp[color_col].values
        if color_col == "n":
            c_norm = mcolors.Normalize(vmin=np.nanpercentile(c_vals, 5),
                                       vmax=np.nanpercentile(c_vals, 95))
        else:
            c_norm = mcolors.Normalize(vmin=0, vmax=1)

        # vet-rejected: open grey circles (50% larger radius → 2.25× area)
        vet_mask = grp.index.isin(vet_rejected)
        if vet_mask.any():
            ax.scatter(grp.loc[vet_mask, "med"],
                       grp.loc[vet_mask, "std"] * 1000,
                       s=27, facecolors="none", edgecolors="grey",
                       linewidths=0.6, alpha=0.7, zorder=2, label="Vet-rejected")

        sc = ax.scatter(grp["med"], grp["std"] * 1000,
                        c=c_vals, cmap=cmap_name, norm=c_norm,
                        s=4, alpha=0.5, rasterized=True, zorder=3)
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.88)

        # running median locus — calibrated (solid) and pre-calibration (dotted)
        cx, my = _running_median(grp, edges)
        if cx:
            ax.plot(cx, my, "k-", lw=2, zorder=4, label="Median σ (calibrated)")
        if grp_org is not None:
            cx_org, my_org = _running_median(grp_org, edges)
            if cx_org:
                ax.plot(cx_org, my_org, "k--", lw=1.5, zorder=4,
                        label="Median σ (pre-calibration)")

        # calibration star range
        ax.axvline(_CALIB_MAG_LO, color="steelblue", lw=1.2, ls=":", alpha=0.8)
        ax.axvline(_CALIB_MAG_HI, color="steelblue", lw=1.2, ls=":", alpha=0.8,
                   label=f"Cal range {_CALIB_MAG_LO:.0f}–{_CALIB_MAG_HI:.0f} mag")

        # median MAGLIM
        if med_maglim is not None:
            ax.axvline(med_maglim, color="orange", lw=1, ls="--", alpha=0.7,
                       label=f"Median MAGLIM={med_maglim:.1f}")

        # median NC_RMS4
        if np.isfinite(nc_rms4_med):
            ax.axhline(nc_rms4_med, color="gray", lw=1, ls=":", alpha=0.6)

        # target
        if tgt_obj_idx is not None and tgt_obj_idx in grp.index:
            r = grp.loc[tgt_obj_idx]
            ax.plot(r["med"], r["std"] * 1000, "*", ms=14, color="red",
                    zorder=5, label=f"Target  σ={r['std']*1000:.1f} mmag")

        ax.set_yscale("log")
        ax.set_ylim(1, 1000)
        ax.set_xlim(13, 23)
        ax.set_xlabel("Median calibrated magnitude (AB)", fontsize=10)
        ax.set_ylabel("σ_mag across epochs (mmag)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper left")

        n_obj = len(grp)
        ax.set_title(f"N={n_obj:,} sources  |  {clabel}", fontsize=10)

    # ── bottom panels ─────────────────────────────────────────────────────────
    if pos_data is not None and not pos_data.empty:
        # Panel bot-left: nearest-neighbour distance histogram by magnitude bin
        ax = axes[1, 0]
        if nn_data is not None and not nn_data.empty:
            sel = nn_data["sep"]
            sel = sel[(sel >= 0.5) & (sel <= 100)]
            hbins = np.logspace(np.log10(0.5), np.log10(100), 60)
            ax.hist(sel, bins=hbins, histtype="step", color="steelblue",
                    lw=1.5, density=True, label=f"N={len(sel):,} sources")
            # target
            if tgt_obj_idx is not None and tgt_obj_idx in nn_data.index:
                tgt_sep = nn_data.loc[tgt_obj_idx, "sep"]
                ax.axvline(tgt_sep, color="red", lw=1.5, ls="--",
                           label=f"Target  NN={tgt_sep:.1f}\"")
            ax.set_xscale("log")
            ax.set_xlabel("Distance to nearest source (arcsec)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title("Nearest-neighbour separation", fontsize=10)
        else:
            axes[1, 0].set_visible(False)

        # Panel bot-right: histogram of Δpos by magnitude bin
        ax = axes[1, 1]
        mag_bins = [(13, 17, "13–17", "steelblue"),
                    (17, 19, "17–19", "darkorange"),
                    (19, 21, "19–21", "mediumseagreen"),
                    (21, 23, "21–23", "crimson")]
        hist_bins = np.logspace(np.log10(0.1), np.log10(5.0), 60)
        for (mlo, mhi, mlabel, color) in mag_bins:
            sel = pos_data[(pos_data["med"] >= mlo) & (pos_data["med"] < mhi)]["dpos"]
            sel = sel[(sel >= 0.1) & (sel <= 5.0)]
            if len(sel) < 10:
                continue
            ax.hist(sel, bins=hist_bins, histtype="step", color=color,
                    lw=1.5, density=True, label=f"{mlabel} mag (N={len(sel):,})")
        ax.set_xscale("log")
        ax.set_xlabel("Δposition from median (arcsec)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Δpos distribution by magnitude bin", fontsize=10)
    else:
        logger.info("  No per-epoch position column (ref-pos parquet) — astrometry panels skipped")
        for col_idx in range(2):
            axes[1, col_idx].set_visible(False)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info(f"  precision → {out_path}")
