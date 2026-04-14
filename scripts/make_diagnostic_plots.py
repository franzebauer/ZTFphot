"""
make_diagnostic_plots.py
------------------------
Diagnostic plots for the ZTF photometry pipeline.

Functions (called by run_pipeline.py step_plots):
  make_fig_calibration  — per-epoch calibration quality timeline
  make_fig_precision    — photometric precision locus (σ vs mag)
  make_fig_spatial      — spatial residual map (binned in RA/Dec)
  make_fig_lightcurve   — target + comparison star light curves
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

_SENTINEL = -999.0   # value written by calib_catalogs for missing header quantities


def _load_epoch_headers(cal_dir: Path) -> pd.DataFrame:
    """Read primary headers from all *_cal.fits in cal_dir into a DataFrame."""
    from astropy.io import fits

    keys = ["OBSMJD", "SEEING", "MAGLIM", "NC_RMS0", "NC_RMS1", "NC_RMS2",
            "NC_RMSFC", "NC_RMS3", "NC_RMS4", "CALIB_ZP", "num_stars", "NC_N"]
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
    df = pd.DataFrame(rows)
    # Replace sentinel with NaN
    df = df.replace(_SENTINEL, np.nan)
    return df.sort_values("OBSMJD").reset_index(drop=True)


# ── Fig 1: Calibration quality timeline ──────────────────────────────────────

def make_fig_calibration(cal_dir: Path, out_path: Path, tag: str = "") -> None:
    """
    4-panel figure showing per-epoch calibration quality vs MJD:
      top-left   — calibration RMS at each pipeline stage (mmag)
      top-right  — number of calibration stars used
      bottom-left — seeing FWHM (arcsec)
      bottom-right — 5σ limiting magnitude
    """
    df = _load_epoch_headers(cal_dir)
    if df.empty:
        logger.warning(f"No calibrated epochs found in {cal_dir}")
        return

    mjd = df["OBSMJD"].values
    mjd0 = int(np.nanmin(mjd))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Calibration quality — {tag}", fontsize=12)

    # ── Panel 1: RMS progression ──
    ax = axes[0, 0]
    stages = [("NC_RMS0", "linear ZP",      "C0"),
              ("NC_RMS1", "3σ clip",         "C1"),
              ("NC_RMS2", "faint corr.",     "C2"),
              ("NC_RMS3", "polynomial",      "C3"),
              ("NC_RMS4", "flatfield",       "C4")]
    for col, label, color in stages:
        vals = df[col].values * 1000   # mag → mmag
        ok   = np.isfinite(vals)
        if ok.sum() > 0:
            ax.plot(mjd[ok] - mjd0, vals[ok], ".", ms=3, color=color,
                    alpha=0.6, label=label)
    ax.set_xlabel(f"MJD − {mjd0}")
    ax.set_ylabel("Calibration RMS (mmag)")
    ax.set_title("Per-epoch calibration RMS")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # ── Panel 2: N calibration stars ──
    ax = axes[0, 1]
    n_stars = df["num_stars"].values
    ok = np.isfinite(n_stars)
    ax.plot(mjd[ok] - mjd0, n_stars[ok], ".", ms=3, color="C0", alpha=0.6)
    ax.set_xlabel(f"MJD − {mjd0}")
    ax.set_ylabel("N calibration stars")
    ax.set_title("Calibration star count per epoch")

    # ── Panel 3: Seeing ──
    ax = axes[1, 0]
    seeing = df["SEEING"].values
    ok = np.isfinite(seeing)
    sc = ax.scatter(mjd[ok] - mjd0, seeing[ok], c=df["NC_RMS4"].values[ok] * 1000,
                    s=5, cmap="RdYlGn_r", vmin=0, vmax=100)
    plt.colorbar(sc, ax=ax, label="Final RMS (mmag)")
    ax.set_xlabel(f"MJD − {mjd0}")
    ax.set_ylabel("Seeing FWHM (arcsec)")
    ax.set_title("Seeing (coloured by final RMS)")

    # ── Panel 4: Limiting magnitude ──
    ax = axes[1, 1]
    maglim = df["MAGLIM"].values
    ok = np.isfinite(maglim)
    ax.plot(mjd[ok] - mjd0, maglim[ok], ".", ms=3, color="C2", alpha=0.6)
    ax.set_xlabel(f"MJD − {mjd0}")
    ax.set_ylabel("5σ limiting magnitude")
    ax.set_title("Limiting magnitude per epoch")
    ax.invert_yaxis()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"  calib quality → {out_path}")


# ── Fig 2: Photometric precision locus ───────────────────────────────────────

def make_fig_precision(lc_path: Path, out_path: Path, tag: str = "",
                       target_ra: float | None = None,
                       target_dec: float | None = None) -> None:
    """
    σ_mag vs median mag for all sources with ≥5 clean detections.
    Running median locus shown in black. Target marked if ra/dec given.
    """
    df = pd.read_parquet(lc_path)

    # Calibrated mag column (prefer MAG_4_TOT_AB, fall back to flux conversion)
    if "MAG_4_TOT_AB" in df.columns:
        mag_col = "MAG_4_TOT_AB"
    else:
        logger.warning("MAG_4_TOT_AB not in lightcurves — skipping precision plot")
        return

    clean = df[df["FLAG_CLEAN"].astype(bool) & df["FLAG_DET"].astype(bool)].copy()
    clean[mag_col] = pd.to_numeric(clean[mag_col], errors="coerce")
    clean = clean[clean[mag_col].notna() & clean[mag_col].between(14, 23)]

    grp = clean.groupby("object_index")[mag_col].agg(
        n="count", med="median", std="std"
    ).dropna()
    grp = grp[grp["n"] >= 5]

    if grp.empty:
        logger.warning(f"No sources with ≥5 clean detections for precision plot")
        return

    # Running-median locus in 0.5-mag bins
    edges   = np.arange(14, 23, 0.5)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = grp[(grp["med"] >= lo) & (grp["med"] < hi)]["std"] * 1000
        if len(m) >= 5:
            centers.append(0.5 * (lo + hi))
            medians.append(np.median(m))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(grp["med"], grp["std"] * 1000,
                    c=np.log10(np.clip(grp["n"], 1, None)),
                    s=4, alpha=0.5, cmap="viridis", rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("log₁₀(N epochs)")
    cb.set_ticks([1, 1.5, 2, 2.5, 3])
    cb.set_ticklabels(["10", "32", "100", "316", "1000"])

    if centers:
        ax.plot(centers, medians, "k-", lw=2, label="Median locus")
        ax.plot(centers, medians, "k.", ms=6)

    # Mark target
    if target_ra is not None and target_dec is not None and "ALPHAWIN_REF" in df.columns:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        tgt  = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
        srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
        if not srcs.empty:
            cats = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                            dec=srcs["DELTAWIN_REF"].values * u.deg)
            idx, sep, _ = tgt.match_to_catalog_sky(cats)
            if sep[0].arcsec < 2.0:
                tgt_idx = srcs.index[idx]
                if tgt_idx in grp.index:
                    r = grp.loc[tgt_idx]
                    ax.plot(r["med"], r["std"] * 1000, "*", ms=14, color="red",
                            label=f"Target  σ={r['std']*1000:.1f} mmag  N={int(r['n'])}")

    ax.set_xlabel("Median calibrated magnitude")
    ax.set_ylabel("σ_mag (mmag)")
    ax.set_title(f"Photometric precision — {tag}")
    ax.set_yscale("log")
    ax.set_ylim(1, 500)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"  precision locus → {out_path}")


# ── Fig 3: Spatial residual map ───────────────────────────────────────────────

def make_fig_spatial(lc_path: Path, out_path: Path, tag: str = "",
                     nbins: int = 30) -> None:
    """
    Median per-source residual (mag − source_median) binned in RA/Dec.
    Shows spatial structure remaining after all calibration steps.
    """
    df = pd.read_parquet(lc_path)

    if "MAG_4_TOT_AB" not in df.columns:
        logger.warning("MAG_4_TOT_AB not in lightcurves — skipping spatial plot")
        return

    mag_col = "MAG_4_TOT_AB"
    clean = df[df["FLAG_CLEAN"].astype(bool) & df["FLAG_DET"].astype(bool)].copy()
    clean[mag_col] = pd.to_numeric(clean[mag_col], errors="coerce")
    # Restrict to calibration-quality magnitude range
    clean = clean[clean[mag_col].between(15, 19.5)]

    if clean.empty:
        logger.warning("No sources in 15–19.5 mag range for spatial plot")
        return

    # Per-source median mag
    src_med = clean.groupby("object_index")[mag_col].median().rename("src_med")
    clean   = clean.join(src_med, on="object_index")
    clean["residual"] = (clean[mag_col] - clean["src_med"]) * 1000   # mmag

    ra_col  = "ALPHAWIN_REF" if "ALPHAWIN_REF" in clean.columns else "ALPHAWIN_OBJ"
    dec_col = "DELTAWIN_REF" if "DELTAWIN_REF" in clean.columns else "DELTAWIN_OBJ"
    clean   = clean[[ra_col, dec_col, "residual"]].dropna()

    ra  = clean[ra_col].values
    dec = clean[dec_col].values
    res = clean["residual"].values

    # Clip extreme outliers before binning
    lo, hi = np.nanpercentile(res, [1, 99])
    mask = (res >= lo) & (res <= hi)
    ra, dec, res = ra[mask], dec[mask], res[mask]

    ra_edges  = np.linspace(ra.min(),  ra.max(),  nbins + 1)
    dec_edges = np.linspace(dec.min(), dec.max(), nbins + 1)

    # Bin: median residual per cell
    grid = np.full((nbins, nbins), np.nan)
    ra_idx  = np.clip(np.digitize(ra,  ra_edges)  - 1, 0, nbins - 1)
    dec_idx = np.clip(np.digitize(dec, dec_edges) - 1, 0, nbins - 1)
    for i in range(nbins):
        for j in range(nbins):
            sel = res[(ra_idx == i) & (dec_idx == j)]
            if len(sel) >= 3:
                grid[j, i] = np.median(sel)

    vmax = max(5.0, np.nanpercentile(np.abs(grid[np.isfinite(grid)]), 95))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[ra.min(), ra.max(), dec.min(), dec.max()],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Median residual (mmag)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Spatial residuals (15–19.5 mag) — {tag}")
    rms = np.nanstd(grid[np.isfinite(grid)])
    ax.text(0.02, 0.97, f"RMS = {rms:.1f} mmag", transform=ax.transAxes,
            va="top", fontsize=9, color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"  spatial residuals → {out_path}")


# ── Fig 4: Target light curve ─────────────────────────────────────────────────

def make_fig_lightcurve(lc_paths: list[tuple], out_path: Path,
                        target_ra: float, target_dec: float) -> None:
    """
    Light curve for the target source across all quadrants/bands.

    lc_paths: list of (lc_parquet_path, filtercode) tuples
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    tgt = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

    band_data: dict[str, list] = {}

    for lc_path, fc in lc_paths:
        try:
            df = pd.read_parquet(lc_path)
        except Exception as e:
            logger.warning(f"Could not read {lc_path}: {e}")
            continue

        if "ALPHAWIN_REF" not in df.columns:
            continue

        srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
        if srcs.empty:
            continue

        cats = SkyCoord(ra=srcs["ALPHAWIN_REF"].values * u.deg,
                        dec=srcs["DELTAWIN_REF"].values * u.deg)
        idx, sep, _ = tgt.match_to_catalog_sky(cats)
        if sep[0].arcsec > 2.0:
            logger.debug(f"  target not found in {lc_path.name} (sep={sep[0].arcsec:.1f}\")")
            continue

        tgt_obj_idx = srcs.index[idx]
        rows = df[df["object_index"] == tgt_obj_idx].copy()
        rows = rows[rows["FLAG_CLEAN"].astype(bool) & rows["FLAG_DET"].astype(bool)]

        if "MAG_4_TOT_AB" not in rows.columns:
            continue

        rows["MAG_4_TOT_AB"] = pd.to_numeric(rows["MAG_4_TOT_AB"], errors="coerce")
        rows["MERR_4_TOT_AB"] = pd.to_numeric(rows.get("MERR_4_TOT_AB", np.nan), errors="coerce")
        rows = rows[rows["MAG_4_TOT_AB"].between(10, 25)]

        if rows.empty:
            continue

        band = {"zg": "g", "zr": "r", "zi": "i"}.get(fc, fc)
        band_data.setdefault(band, []).append(rows[["OBSMJD", "MAG_4_TOT_AB", "MERR_4_TOT_AB"]])
        logger.debug(f"  target in {lc_path.name}: {len(rows)} clean detections, "
                     f"sep={sep[0].arcsec:.2f}\"")

    if not band_data:
        logger.warning(f"Target RA={target_ra:.4f} Dec={target_dec:.4f} not found in any quadrant")
        return

    bands   = sorted(band_data.keys())
    colors  = {"g": "C2", "r": "C3", "i": "C1"}
    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands),
                             sharex=True, squeeze=False)

    all_mjd = []
    for ax, band in zip(axes[:, 0], bands):
        rows = pd.concat(band_data[band]).sort_values("OBSMJD")
        mjd  = rows["OBSMJD"].values
        mag  = rows["MAG_4_TOT_AB"].values
        err  = rows["MERR_4_TOT_AB"].values
        all_mjd.extend(mjd.tolist())

        ok = np.isfinite(mag) & np.isfinite(err)
        ax.errorbar(mjd[ok], mag[ok], yerr=err[ok], fmt=".",
                    color=colors.get(band, "C0"), ms=4, elinewidth=0.7,
                    alpha=0.8, label=f"{band}-band  N={ok.sum()}")
        ax.invert_yaxis()
        ax.set_ylabel("Calibrated mag")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        med = np.nanmedian(mag[ok])
        std = np.nanstd(mag[ok])
        ax.text(0.01, 0.05, f"median={med:.3f}  σ={std*1000:.1f} mmag",
                transform=ax.transAxes, fontsize=8)

    if all_mjd:
        axes[-1, 0].set_xlabel("MJD")

    fig.suptitle(f"Target light curve  RA={target_ra:.4f}  Dec={target_dec:+.4f}",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"  target light curve → {out_path}")
