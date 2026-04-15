"""
merge_fields.py
---------------
Cross-calibrate and merge per-quadrant light curve parquets for one band.

Algorithm
---------
1.  Load all per-quadrant lightcurves.parquet files for the requested band.
2.  Compute calibrated AB magnitude per row:
        mag = -2.5 * log10(FLUX_4_TOT_AB) - 48.6
3.  Determine the **dominant quadrant**: the one whose sources have the most
    total FLAG_CLEAN & FLAG_DET epochs (i.e. the most calibration leverage).
4.  For every non-dominant quadrant, find sources common to both quadrants
    (matched within MAX_SEP arcsec).  Keep only stable non-variable sources
    (std_mag < STD_CUT, N_clean >= MIN_EPOCHS in both quadrants).
5.  Compute the median magnitude offset:
        offset = median( median_mag_dominant[source] - median_mag_minority[source] )
    over all common stable sources.  Apply offset to the minority parquet's
    calibrated magnitude column.
6.  Concatenate all quadrants (with a `quadrant_id` tag), sort by OBSMJD, and
    write to LightCurves/merged/{band}/lightcurves_merged.parquet.

Output parquet extra columns (beyond per-quadrant schema)
----------------------------------------------------------
    mag_calib       float64   -2.5*log10(FLUX_4_TOT_AB) - 48.6  (magnitude)
    mag_calib_err   float64   propagated from FERR_4_TOT_AB
    norm_offset     float64   offset applied to bring this quadrant onto dominant scale
    quadrant_id     str       "{field:06d}_{filtercode}_c{ccdid:02d}_q{qid}"
    is_dominant     bool      True for the dominant quadrant
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

# ── Tuning parameters ─────────────────────────────────────────────────────────
MAX_SEP_ARCSEC = 3.0      # cross-match radius for common sources
MIN_EPOCHS     = 5        # min FLAG_CLEAN+FLAG_DET epochs per source per quadrant
STD_CUT        = 0.15     # max mag std to qualify as a stable calibration source
MIN_COMMON     = 10       # min number of common stable sources required for offset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flux_to_mag(flux: np.ndarray) -> np.ndarray:
    """Convert AB flux density (erg/s/cm²/Hz) to AB magnitude."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux > 0, -2.5 * np.log10(flux) - 48.6, np.nan)


def _flux_err_to_mag_err(flux: np.ndarray, fluxerr: np.ndarray) -> np.ndarray:
    """Propagate flux error to magnitude error."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux > 0, 2.5 / np.log(10) * np.abs(fluxerr / flux), np.nan)


def _per_source_stats(df: pd.DataFrame, min_epochs: int = MIN_EPOCHS) -> pd.DataFrame:
    """
    Given a lightcurve parquet DataFrame (all sources, all epochs), return a
    summary with one row per source:
        ID_REF, ra, dec, n_clean, median_mag, std_mag
    Only sources with n_clean >= min_epochs are returned.
    """
    clean = df[df['FLAG_CLEAN'] & df['FLAG_DET'] & df['mag_calib'].notna()].copy()
    stats = (clean.groupby('ID_REF')
             .agg(
                 ra       =('ALPHAWIN_REF', 'first'),
                 dec      =('DELTAWIN_REF', 'first'),
                 n_clean  =('mag_calib', 'count'),
                 median_mag=('mag_calib', 'median'),
                 std_mag  =('mag_calib', 'std'),
             )
             .reset_index())
    return stats[stats['n_clean'] >= min_epochs]


def _compute_offset(stats_dom: pd.DataFrame,
                    stats_min: pd.DataFrame,
                    max_sep_arcsec: float = MAX_SEP_ARCSEC,
                    std_cut: float = STD_CUT,
                    min_common: int = MIN_COMMON) -> tuple[float, int]:
    """
    Match sources in dominant vs minority quadrant; compute median magnitude
    offset = median(mag_dominant - mag_minority) over stable common sources.

    Returns (offset, n_common_used).  Raises ValueError if too few common sources.
    """
    cat_dom = SkyCoord(ra=stats_dom['ra'].values * u.deg,
                       dec=stats_dom['dec'].values * u.deg)
    cat_min = SkyCoord(ra=stats_min['ra'].values * u.deg,
                       dec=stats_min['dec'].values * u.deg)

    idx, sep, _ = cat_min.match_to_catalog_sky(cat_dom)
    matched = sep.arcsec < max_sep_arcsec

    stats_min_m = stats_min[matched].copy()
    stats_dom_m = stats_dom.iloc[idx[matched]].copy()

    # Keep only stable sources in both
    stable = (
        (stats_dom_m['std_mag'].values < std_cut) &
        (stats_min_m['std_mag'].values < std_cut)
    )
    n_common = stable.sum()

    if n_common < min_common:
        raise ValueError(
            f"Only {n_common} stable common sources (need {min_common}). "
            "Cannot compute reliable normalization offset."
        )

    diff = stats_dom_m['median_mag'].values[stable] - stats_min_m['median_mag'].values[stable]

    # Robust: iterative 3σ MAD clipping
    for _ in range(3):
        med   = np.median(diff)
        mad   = np.median(np.abs(diff - med))
        sigma = 1.4826 * mad
        mask  = np.abs(diff - med) < 3.0 * sigma
        if mask.sum() < min_common // 2:
            break
        diff = diff[mask]

    return float(np.median(diff)), int(n_common)


# ── Public API ────────────────────────────────────────────────────────────────

def merge_band(
    lc_root: Path,
    band: str,
    quadrants: list[dict],
    force: bool = False,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Cross-calibrate and merge all quadrant light curves for one band.

    Parameters
    ----------
    lc_root   : base_dir / 'LightCurves'
    band      : filtercode string, e.g. 'zg'
    quadrants : list of dicts from find_quadrants() — filtered to the wanted band
    force     : overwrite existing merged parquet
    out_dir   : output directory; defaults to lc_root / 'merged' / band

    Returns
    -------
    Path to merged parquet, or None if fewer than 2 quadrants have data.
    """
    if out_dir is None:
        out_dir = lc_root / 'merged' / band
    out_path = out_dir / 'lightcurves_merged.parquet'

    if out_path.exists() and not force:
        logger.info(f"merge [{band}]: merged parquet already exists — skip (use --force to redo)")
        return out_path

    # ── 1. Load per-quadrant parquets ─────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for q in quadrants:
        if q['filtercode'] != band:
            continue
        field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
        pq = lc_root / f"{field:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid}" / 'lightcurves.parquet'
        if not pq.exists():
            logger.warning(f"merge [{band}]: parquet missing for {field:06d} {fc} c{ccd:02d} q{qid} — skipped")
            continue
        df = pd.read_parquet(pq)
        df['quadrant_id'] = f"{field:06d}_{fc}_c{ccd:02d}_q{qid}"
        df['norm_offset']  = 0.0
        frames.append(df)

    if len(frames) < 2:
        logger.warning(f"merge [{band}]: fewer than 2 quadrants with data — nothing to merge")
        return None

    # ── 2. Compute mag_calib ──────────────────────────────────────────────────
    for df in frames:
        df['mag_calib'] = _flux_to_mag(df['FLUX_4_TOT_AB'].values)
        ferr_col = 'FERR_4_TOT_AB'
        if ferr_col in df.columns:
            df['mag_calib_err'] = _flux_err_to_mag_err(
                df['FLUX_4_TOT_AB'].values, df[ferr_col].values)
        else:
            df['mag_calib_err'] = np.nan

    # ── 3. Determine dominant quadrant ───────────────────────────────────────
    def _n_clean_epochs(df: pd.DataFrame) -> int:
        return int((df['FLAG_CLEAN'] & df['FLAG_DET']).sum())

    n_clean = [_n_clean_epochs(df) for df in frames]
    dom_idx = int(np.argmax(n_clean))
    logger.info(f"merge [{band}]: dominant quadrant = {frames[dom_idx]['quadrant_id'].iloc[0]} "
                f"({n_clean[dom_idx]} clean det-epochs)")
    for i, (df, n) in enumerate(zip(frames, n_clean)):
        logger.info(f"  {'[DOM]' if i == dom_idx else '     '} "
                    f"{df['quadrant_id'].iloc[0]}: {n} clean det-epochs")

    frames[dom_idx]['is_dominant'] = True
    for i, df in enumerate(frames):
        if i != dom_idx:
            df['is_dominant'] = False

    # ── 4 & 5. Compute and apply normalization offsets ────────────────────────
    stats_dom = _per_source_stats(frames[dom_idx])

    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        qid_str = df['quadrant_id'].iloc[0]
        try:
            stats_min = _per_source_stats(df)
            offset, n_used = _compute_offset(stats_dom, stats_min)
            logger.info(f"merge [{band}]: offset {qid_str} → dominant = "
                        f"{offset:+.4f} mag  (N_common={n_used})")
            df['mag_calib']    += offset
            df['norm_offset']   = offset
        except ValueError as exc:
            logger.warning(f"merge [{band}]: cannot normalize {qid_str}: {exc}. "
                           "Merging without offset correction.")

    # ── 6. Concatenate and write ──────────────────────────────────────────────
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values('OBSMJD', inplace=True, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    n_obj   = merged['ID_REF'].nunique()   # note: IDs are unique per quadrant
    n_rows  = len(merged)
    logger.info(f"merge [{band}]: {n_rows} rows, {n_obj} source×quadrant combinations "
                f"→ {out_path}")
    return out_path
