#!/usr/bin/env python3
"""
update_old_parquet.py
---------------------
Retrofit light-curve parquets written before 2026-10-21 (no raw FLUX_4_DIF / APCORR410)
to the current pipeline: same options and procedure as update_parquet.py (missing and
newer epochs are added; old epochs are rebuilt from the parquet and recalibrated,
without re-downloading them), with the old rows first converted to the current
conventions.

Raw 4 px difference flux of every old row:
  - FLUX_4_DIF where stored;
  - else from MAG_4_TOT_AB_org (exact):
        flux_tot = 10^(0.4 (MAGZP_DIF - APCORR46/1000 - (MAG_4_TOT_AB_org - norm_offset)))
  - rows with total flux <= 0 (MAG_4_TOT_AB_org NaN) from FLUX_4_TOT_AB divided by the
    epoch's calibration factor, which is fitted on the epoch's other rows as
    n + m*mag + quadratic in position + inverse-distance interpolated residual of the 9
    nearest rows, and evaluated without the m*mag term (the calibration used the
    intercept only where the magnitude is undefined). Validated on a parquet with
    stored FLUX_4_DIF: factor MAD 0.25 mmag, rebuilt flux within 0.002 sigma;
  - FLUX_4_DIF = flux_tot - reference flux, with the reference magnitude the file was
    built with (MAG_4_REF, else MAGZP_REF_{tag} metadata + the rebuilt catalog).
  Rows with neither a magnitude nor a flux carry no photometry and are dropped — except
  for the target: epochs where the target has none are measured again from the images
  (reused locally where present, else downloaded).
Other conventions: FLUX_6_TOT_AB is put on the merge-corrected scale (factor
10^(-0.4 norm_offset)); per-epoch flags come from FLAG_AP, or are marked AP_UNKNOWN.

MAG_3/6/10 of old epochs are shifted by the 4 px change, as in update_parquet.py: the
median is unbiased, but the reference zero-point correction rescales each row's
difference flux per aperture, which a common shift cannot reproduce (est_084892: MAD
12-22 mmag vs a fresh run). 4 px is exact. Parquets made before 2026-09-08 had the faint
correction in FLUX_4_TOT_AB; for their negative-flux rows it is not modelled.

Usage (ztf env, IRSA credentials in ~/.netrc): as update_parquet.py, e.g.
    python update_old_parquet.py RESULTS_DIR --from-updated --recalibrate --infer-cuts
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import update_parquet as up  # noqa: E402
from epoch_inventory import target_position  # noqa: E402

_MIN_FIT_ROWS = 30          # rows with a magnitude needed for the full factor model
TARGET_RADIUS = 1.0         # arcsec, the pipeline's target association radius


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return (pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df
            else np.full(len(df), np.nan))


def prepare(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Bring an old parquet's rows to the current conventions (see module docstring)."""
    from ztfphot_flags import flags_of
    meta = {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items()
            if not k.startswith(b"pandas")}
    if "MAG_4_TOT_AB_org" not in df:
        raise ValueError(f"{path.name}: no MAG_4_TOT_AB_org — cannot rebuild the raw flux")
    if "FLAGS" not in df:
        df["FLAGS"] = flags_of(df).to_numpy()
    if "flux_norm_applied" not in meta and "FLUX_6_TOT_AB" in df:
        df["FLUX_6_TOT_AB"] = (_num(df, "FLUX_6_TOT_AB")
                               * 10.0 ** (-0.4 * np.nan_to_num(_num(df, "norm_offset"))))
    # reference zero point the file was built with, per quadrant (used where MAG_4_REF
    # is absent)
    tags = (df["field"].astype(int).map("{:06d}".format) + "_" + df["filtercode"].astype(str)
            + "_c" + df["ccdid"].astype(int).map("{:02d}".format) + "_q" + df["qid"].astype(str))
    df["_MAGZP_REF_OLD"] = pd.to_numeric(tags.map(lambda t: meta.get(f"MAGZP_REF_{t}")),
                                         errors="coerce")
    # rows with no 4 px photometry at all
    has = np.isfinite(_num(df, "MAG_4_TOT_AB_org")) | np.isfinite(_num(df, "FLUX_4_TOT_AB"))
    if "FLUX_4_DIF" in df:
        has |= np.isfinite(_num(df, "FLUX_4_DIF"))
    # Epochs where the TARGET has no 4 px photometry are removed entirely: update_parquet
    # then treats them as missing and measures them again from the images.
    ra, dec = target_position(path, meta)
    sep = 3600 * np.hypot((_num(df, "ALPHAWIN_REF") - ra) * np.cos(np.radians(dec)),
                          _num(df, "DELTAWIN_REF") - dec)
    ep = df[up.QKEY].astype(str).agg("|".join, axis=1) + "|" + df["OBSMJD"].round(6).astype(str)
    redo = ep.isin(set(ep[(sep < TARGET_RADIUS) & ~has]))
    if redo.any():
        up.logger.info(f"  {path.name}: {ep[redo].nunique()} epoch(s) without target 4 px "
                       f"photometry — re-measured from the images")
    if (~has & ~redo).any():
        up.logger.info(f"  {path.name}: {int((~has & ~redo).sum()):,} rows of other sources "
                       f"without 4 px photometry — dropped")
    return df[has & ~redo].copy()


def _cal_log_factor(rows: pd.DataFrame, mraw: np.ndarray, flux: np.ndarray,
                    ftot: np.ndarray, zp: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Calibration factor (mag: flux_cal = flux_tot * 10^(0.4 (L + 23.9 - MAGZP_DIF)))
    at the rows `want`, from the rows of the same epoch that have a magnitude."""
    from scipy.spatial import cKDTree
    ra, dec = _num(rows, "ALPHAWIN_REF"), _num(rows, "DELTAWIN_REF")
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 2.5 * np.log10(np.abs(flux / ftot)) - (23.9 - zp)
    have = np.isfinite(mraw) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec)
    out = np.full(len(rows), np.nan)
    for idx in rows.groupby("OBSMJD").indices.values():
        idx = np.asarray(idx)
        q, g = idx[want[idx]], idx[have[idx]]
        if len(q) == 0 or len(g) < 3:
            continue
        if len(g) < _MIN_FIT_ROWS:
            out[q] = np.median(y[g])
            continue
        r0, d0 = ra[g].mean(), dec[g].mean()
        u, v = ra[g] - r0, dec[g] - d0
        A = np.column_stack([np.ones_like(u), mraw[g], u, v, u * u, u * v, v * v])
        c, *_ = np.linalg.lstsq(A, y[g], rcond=None)
        res = y[g] - A @ c
        uq, vq = ra[q] - r0, dec[q] - d0
        dist, j = cKDTree(np.column_stack([u, v])).query(np.column_stack([uq, vq]),
                                                         k=min(9, len(g)))
        dist, j = np.atleast_2d(dist), np.atleast_2d(j)
        w = 1.0 / np.maximum(dist, 1e-9)
        out[q] = (np.column_stack([np.ones_like(uq), uq, vq, uq * uq, uq * vq, vq * vq])
                  @ np.delete(c, 1) + (w * res[j]).sum(1) / w.sum(1))
    return out


def raw_flux(rows: pd.DataFrame, q_mag: np.ndarray, is_target: np.ndarray, ref: pd.DataFrame):
    """Raw 4 px difference flux and error (counts, MAGZP_DIF units) of old parquet rows."""
    zp = _num(rows, "MAGZP_DIF")
    # reference magnitude the file's total fluxes were formed with
    if "MAG_4_REF" in rows:
        q_old = _num(rows, "MAG_4_REF")                 # NaN / > 90: no reference source
    else:
        zp_new = float(pd.to_numeric(ref["MAGZP_REF"], errors="coerce").iloc[0])
        q_old = q_mag - zp_new + _num(rows, "_MAGZP_REF_OLD")
        if (np.isfinite(q_mag) & ~is_target & ~np.isfinite(q_old)).any():
            raise ValueError("old reference zero point unknown (no MAG_4_REF column, no "
                             "MAGZP_REF_{tag} metadata)")
    no_ref = is_target | ~(q_old < 90)
    fref = np.where(no_ref, 0.0, 10.0 ** (0.4 * (zp - np.where(no_ref, 0.0, q_old))))

    apc = np.nan_to_num(_num(rows, "APCORR46")) / 1000.0
    mraw = _num(rows, "MAG_4_TOT_AB_org") - np.nan_to_num(_num(rows, "norm_offset"))
    ftot = 10.0 ** (0.4 * (zp - apc - mraw))
    etot = _num(rows, "MERR_4_TOT_AB_org") * ftot / 1.0857

    flux, ferr = _num(rows, "FLUX_4_TOT_AB"), _num(rows, "FERR_4_TOT_AB")
    neg = ~np.isfinite(mraw) & np.isfinite(flux)
    if neg.any():
        L = _cal_log_factor(rows, mraw, flux, ftot, zp, neg)
        fac = 10.0 ** (0.4 * (L + 23.9 - zp))
        ftot = np.where(neg, flux / fac, ftot)
        etot = np.where(neg, ferr / fac, etot)

    f, e = ftot - fref, etot
    fs, es = _num(rows, "FLUX_4_DIF"), _num(rows, "FERR_4_DIF")     # stored raw flux wins
    return np.where(np.isfinite(fs), fs, f), np.where(np.isfinite(es), es, e)


if __name__ == "__main__":
    up.main(legacy=sys.modules[__name__])
