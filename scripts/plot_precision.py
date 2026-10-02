"""
plot_precision.py
-----------------
Photometric precision (2×2) — make_precision:

  Rows are the two unit systems, columns the two colourings. Layout is fixed and
  identical in ref-pos and sci-pos, so no axes are ever left empty.

  Row 1 (magnitude): σ_mag vs brightness — censored, since epochs fainter than the
                     reference have total flux ≤0 and a NaN magnitude.
  Row 2 (flux):      σ_flux vs brightness — unbiased, keeps those epochs.
  Col 1 coloured by N clean detections; col 2 by CLASS_STAR (0=gal, 1=star).
  All panels: dotted lines at the calibration-star limits (14–19 mag), target as a
  red star, vet-rejected sources as open grey circles.

Astrometry (1×2) — make_position, sci-pos only:

  Left:  nearest-neighbour separation histogram
  Right: Δposition histogram by magnitude bin (13–17, 17–19, 19–21, 21–23)

  Ref-pos has no per-epoch positions (the centroid is measured on the simulated
  PSF and is meaningless), so this figure is simply not produced in that mode.

Key parquet columns used:
    MAG_4_TOT_AB, MERR_4_TOT_AB  — calibrated magnitude and error
    INFOBITS_DIF                  — quality flag (== 0 for clean epochs)
    CLASS_STAR                    — stellarity
    ALPHAWIN_REF, DELTAWIN_REF    — reference-catalog positions (for target match)
    DRA_SCI, DDEC_SCI             — per-epoch offsets from the reference position,
                                    arcsec (sci-pos only; for Δpos). Absent in
                                    ref-pos, where the Δpos panel is skipped.
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


def _running_median(grp: pd.DataFrame, edges: np.ndarray, scale: float = 1000.0):
    cx, my = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = grp[(grp["med"] >= lo) & (grp["med"] < hi)]["std"] * scale
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

    # Both unit systems are built: σ_mag (censored — negative-flux epochs are NaN) and
    # σ_flux from the bipolar FLUX column (unbiased, keeps them). The brightness axis is
    # MAG_4_REF (reference mag) when available, else the biased median mag; it is shared
    # by both rows so the panels stack meaningfully.
    _FLUX_STD = "FLUX_4_TOT_AB"
    _BRT = "MAG_4_REF" if "MAG_4_REF" in df.columns else _MAG_COL
    if _MAG_COL not in df.columns:
        logger.warning(f"  {_MAG_COL} missing — skipping precision plot")
        return

    for _c in {_FLUX_STD, _BRT, _MAG_COL}:
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")

    clean = df[df["INFOBITS_DIF"] == 0].copy()

    cs_col = "CLASS_STAR" if "CLASS_STAR" in clean.columns else \
             "CLASS_STAR_OBJ" if "CLASS_STAR_OBJ" in clean.columns else None

    def _summarise(std_col):
        """Per-source N / brightness / scatter summary for one unit system."""
        if std_col not in clean.columns:
            return None
        a = clean.groupby("object_index").agg(
            n=(std_col, "count"),
            med=(_BRT, "median"),
            std=(std_col, "std"),
        ).dropna(subset=["std"])
        a["cs"] = (clean.groupby("object_index")[cs_col].median()
                   if cs_col is not None else np.nan)
        return a[a["n"] >= 5]

    grp      = _summarise(_MAG_COL)
    grp_flux = _summarise(_FLUX_STD)

    # ── per-source summary from pre-calibration mags (aperture-corrected maginst)
    _ORG_COL = "MAG_4_TOT_AB_org"
    grp_org = grp_org_flux = None
    if _ORG_COL in clean.columns:
        clean[_ORG_COL] = pd.to_numeric(clean[_ORG_COL], errors="coerce")
        agg_org = clean.groupby("object_index").agg(
            n_org=(_ORG_COL, "count"),
            med=(_ORG_COL,   "median"),
            std=(_ORG_COL,   "std"),
        ).dropna(subset=["std"])
        grp_org = agg_org[agg_org["n_org"] >= 5]

        # Flux-row equivalent. There is no pre-calibration FLUX column, so convert the
        # magnitude scatter: σ_flux ≈ flux · σ_mag · 0.4·ln10, with flux from the
        # source's own pre-calibration median magnitude (AB: 10^(-0.4(m-23.9)) μJy).
        # NOTE this curve is CENSORED — MAG_4_TOT_AB_org is NaN wherever total flux was
        # ≤0, so those epochs never enter σ_mag. At the faint end it therefore sits low
        # relative to a true unbiased pre-calibration flux scatter; it is labelled as
        # such. Recovering the unbiased version would need the per-epoch ZP/poly/
        # flatfield factors, which live in the FITS headers, not the parquet.
        _org_med_mag = agg_org["med"].values
        _org_flux    = 10.0 ** (-0.4 * (_org_med_mag - 23.9))     # μJy
        g = agg_org.copy()
        g["std"] = g["std"].values * _org_flux * (0.4 * np.log(10.0))
        # x-axis must match the flux row (reference mag when available)
        if _BRT != _MAG_COL:
            g["med"] = clean.groupby("object_index")[_BRT].median()
        grp_org_flux = g[(g["n_org"] >= 5) & np.isfinite(g["std"])]

    tgt_obj_idx  = _find_target_obj(df, target_ra, target_dec) if target_ra is not None else None
    vet_rejected = _load_vet_rejected(vet_catalog, df)

    edges = np.arange(13, 23.5, 0.5)
    nc_rms4_med = float(df["NC_RMS4"].median()) if "NC_RMS4" in df.columns else np.nan
    med_maglim  = float(df["MAGLIM"].median())  if "MAGLIM"  in df.columns else None


    # ── figure ────────────────────────────────────────────────────────────────
    # Fixed 2x2: rows are the two unit systems (magnitude on top, flux below, sharing
    # the x-axis), columns are the two colourings. Layout is identical in ref-pos and
    # sci-pos — the astrometric panels live in their own figure (make_position), so no
    # axes are ever left empty.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True,
                             gridspec_kw=dict(wspace=0.25, hspace=0.22))
    fig.suptitle(f"Photometric precision — {tag}", fontsize=13)

    for row_idx, (_row_grp, _row_org, _row_flux) in enumerate([
        (grp,      grp_org,      False),
        (grp_flux, grp_org_flux, True),
    ]):
        if _row_grp is None or _row_grp.empty:
            for _c in range(2):
                axes[row_idx, _c].text(0.5, 0.5,
                                       "FLUX_4_TOT_AB not in parquet"
                                       if _row_flux else "no data",
                                       ha="center", va="center",
                                       transform=axes[row_idx, _c].transAxes,
                                       fontsize=10, color="grey")
                axes[row_idx, _c].set_xticks([]); axes[row_idx, _c].set_yticks([])
            continue
        _YS = 1.0 if _row_flux else 1000.0
        _YU = "μJy" if _row_flux else "mmag"
        for col_idx, (color_col, cmap_name, clabel) in enumerate([
            ("n",  "viridis", "N clean detections"),
            ("cs", "RdYlGn",  "CLASS_STAR  (0=gal, 1=star)"),
        ]):
            ax  = axes[row_idx, col_idx]
            grp = _row_grp
            grp_org = _row_org
            flux = _row_flux
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
                           grp.loc[vet_mask, "std"] * _YS,
                           s=27, facecolors="none", edgecolors="grey",
                           linewidths=0.6, alpha=0.7, zorder=2, label="Vet-rejected")

            sc = ax.scatter(grp["med"], grp["std"] * _YS,
                            c=c_vals, cmap=cmap_name, norm=c_norm,
                            s=4, alpha=0.5, rasterized=True, zorder=3)
            plt.colorbar(sc, ax=ax, label=clabel, shrink=0.88)

            # running median locus — calibrated (solid) and pre-calibration (dotted)
            cx, my = _running_median(grp, edges, scale=_YS)
            if cx:
                ax.plot(cx, my, "k-", lw=2, zorder=4, label="Median σ (calibrated)")
            if grp_org is not None:
                cx_org, my_org = _running_median(grp_org, edges, scale=_YS)
                if cx_org:
                    ax.plot(cx_org, my_org, "k--", lw=1.5, zorder=4,
                            label=("Median σ (pre-calibration, censored)" if flux
                                   else "Median σ (pre-calibration)"))

            # calibration star range
            ax.axvline(_CALIB_MAG_LO, color="steelblue", lw=1.2, ls=":", alpha=0.8)
            ax.axvline(_CALIB_MAG_HI, color="steelblue", lw=1.2, ls=":", alpha=0.8,
                       label=f"Cal range {_CALIB_MAG_LO:.0f}–{_CALIB_MAG_HI:.0f} mag")

            # median MAGLIM
            if med_maglim is not None:
                ax.axvline(med_maglim, color="orange", lw=1, ls="--", alpha=0.7,
                           label=f"Median MAGLIM={med_maglim:.1f}")

            # median NC_RMS4 (calibrator RMS, mmag — magnitude plot only)
            if np.isfinite(nc_rms4_med) and not flux:
                ax.axhline(nc_rms4_med, color="gray", lw=1, ls=":", alpha=0.6)

            # target
            if tgt_obj_idx is not None and tgt_obj_idx in grp.index:
                r = grp.loc[tgt_obj_idx]
                ax.plot(r["med"], r["std"] * _YS, "*", ms=14, color="red",
                        zorder=5, label=f"Target  σ={r['std']*_YS:.1f} {_YU}")

            ax.set_yscale("log")
            ax.set_ylim((0.3, 3000) if flux else (1, 1000))
            ax.set_xlim(13, 23)
            ax.set_xlabel(("Reference magnitude (AB)" if _BRT == "MAG_4_REF"
                           else "Median calibrated magnitude (AB)"), fontsize=10)
            ax.set_ylabel(f"σ_flux across epochs ({_YU})" if flux
                          else f"σ_mag across epochs ({_YU})", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="upper left")

            n_obj = len(grp)
            ax.set_title(f"N={n_obj:,} sources  |  {clabel}", fontsize=10)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info(f"  precision → {out_path}")


def make_position(lc_path: Path, out_path: Path, tag: str = "",
                  target_ra: float | None = None,
                  target_dec: float | None = None) -> None:
    """Astrometric diagnostics — sci-pos only (needs DRA_SCI/DDEC_SCI).

    Ref-pos parquets carry no per-epoch positions (the centroid is measured on the
    simulated PSF), so this returns without writing anything in that mode.
    """
    df = pd.read_parquet(lc_path)
    if not {"DRA_SCI", "DDEC_SCI"} <= set(df.columns):
        logger.info(f"  [{tag}] no DRA_SCI/DDEC_SCI (ref-pos) — skipping position plot")
        return

    for _c in ("DRA_SCI", "DDEC_SCI", _MAG_COL):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    _BRT  = "MAG_4_REF" if "MAG_4_REF" in df.columns else _MAG_COL
    clean = df[df["INFOBITS_DIF"] == 0].copy()

    # per-source median brightness, for binning both panels by magnitude
    med = clean.groupby("object_index")[_BRT].median().rename("med")

    pos = clean[["object_index", "DRA_SCI", "DDEC_SCI"]].dropna()
    pos["dpos"] = np.hypot(pos["DRA_SCI"].values, pos["DDEC_SCI"].values)
    pos = pos.join(med, on="object_index").dropna(subset=["med"])

    # nearest-neighbour separation between distinct sources
    nn_data = None
    if "ALPHAWIN_REF" in df.columns:
        srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
        srcs.columns = ["ra", "dec"]
        if len(srcs) >= 2:
            from scipy.spatial import KDTree
            cos_dec = np.cos(np.radians(srcs["dec"].values.mean()))
            xy = np.column_stack([srcs["ra"].values * cos_dec, srcs["dec"].values])
            dists, _ = KDTree(xy).query(xy, k=2)
            nn_data = pd.DataFrame({"sep": dists[:, 1] * 3600}, index=srcs.index)
            nn_data = nn_data.join(med, how="left").dropna(subset=["sep"])

    tgt_obj_idx = (_find_target_obj(df, target_ra, target_dec)
                   if target_ra is not None else None)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw=dict(wspace=0.22))
    fig.suptitle(f"Astrometry (sci-pos) — {tag}", fontsize=13)

    # ── left: nearest-neighbour separation ──
    ax = axes[0]
    if nn_data is not None and not nn_data.empty:
        sel = nn_data["sep"]
        sel = sel[(sel >= 0.5) & (sel <= 100)]
        ax.hist(sel, bins=np.logspace(np.log10(0.5), np.log10(100), 60),
                histtype="step", color="steelblue", lw=1.5, density=True,
                label=f"N={len(sel):,} sources")
        if tgt_obj_idx is not None and tgt_obj_idx in nn_data.index:
            ax.axvline(float(nn_data.loc[tgt_obj_idx, "sep"]), color="red",
                       lw=1.5, ls="--",
                       label=f"Target  NN={float(nn_data.loc[tgt_obj_idx, 'sep']):.1f}\"")
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Distance to nearest source (arcsec)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_title("Nearest-neighbour separation", fontsize=10)

    # ── right: Δposition from the reference position, by magnitude bin ──
    ax = axes[1]
    hist_bins = np.logspace(np.log10(0.01), np.log10(5.0), 60)
    for (mlo, mhi, mlabel, color) in [(13, 17, "13–17", "steelblue"),
                                      (17, 19, "17–19", "darkorange"),
                                      (19, 21, "19–21", "mediumseagreen"),
                                      (21, 23, "21–23", "crimson")]:
        sel = pos[(pos["med"] >= mlo) & (pos["med"] < mhi)]["dpos"]
        sel = sel[(sel >= 0.01) & (sel <= 5.0)]
        if len(sel) < 10:
            continue
        ax.hist(sel, bins=hist_bins, histtype="step", color=color, lw=1.5,
                density=True, label=f"{mlabel} mag (N={len(sel):,})")
    ax.set_xscale("log")
    # Offsets are measured from the REFERENCE position, so this shows astrometric
    # bias as well as scatter — a population centred away from ~0 is a systematic.
    ax.set_xlabel("Δposition from reference (arcsec)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Per-epoch offset from reference position", fontsize=10)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info(f"  position → {out_path}")
