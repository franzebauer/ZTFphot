"""
merge_fields.py
---------------
Cross-calibrate and merge per-quadrant light curve parquets for one band.

Algorithm
---------
1.  Load all per-quadrant lightcurves.parquet files for the requested band.
2.  Determine the **dominant quadrant**: the one with the most clean epochs
    (INFOBITS_DIF == 0).
3.  For every non-dominant quadrant, find sources common to both quadrants
    (matched within MAX_SEP arcsec).  Keep only stable non-variable sources
    (std_mag < STD_CUT, N_clean >= MIN_EPOCHS in both quadrants).
4.  Compute the median magnitude offset:
        offset = median( median_mag_dominant[source] - median_mag_minority[source] )
    over all common stable sources.  Apply offset directly to all MAG_* columns
    in the minority parquet.
5.  Concatenate all quadrants (with origin columns), sort by OBSMJD, and
    write to LightCurves/merged/{band}/lightcurves_merged.parquet.

Output parquet extra columns (beyond per-quadrant schema)
----------------------------------------------------------
    norm_offset   float32   offset applied to all MAG_* columns (0.0 for dominant quadrant)
    field         int64     origin field number
    filtercode    str       origin filter code
    ccdid         int64     origin CCD ID
    qid           int64     origin quadrant ID

File-level metadata key added:
    dominant_quadrant   "{field:06d}_{filtercode}_c{ccdid:02d}_q{qid}"
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
MAX_SEP_ARCSEC = 1.5      # cross-match radius for common sources
MIN_EPOCHS     = 5        # min clean epochs per source per quadrant
STD_CUT        = 0.15     # max mag std to qualify as a stable calibration source
MIN_COMMON     = 10       # min number of common stable sources required for offset


_MAG_COLS = ['MAG_3_TOT_AB', 'MAG_4_TOT_AB', 'MAG_6_TOT_AB', 'MAG_10_TOT_AB', 'MAG_4_TOT_AB_org']


def _per_source_stats(df: pd.DataFrame, min_epochs: int = MIN_EPOCHS) -> pd.DataFrame:
    """
    Return one row per object_index with ra, dec, n_clean, median_mag, std_mag.
    Only sources with n_clean >= min_epochs are returned.
    """
    clean = df[(df['INFOBITS_DIF'] == 0) & df['MAG_4_TOT_AB'].notna()].copy()
    stats = (clean.groupby('object_index')
             .agg(
                 ra        =('ALPHAWIN_REF', 'first'),
                 dec       =('DELTAWIN_REF', 'first'),
                 n_clean   =('MAG_4_TOT_AB', 'count'),
                 median_mag=('MAG_4_TOT_AB', 'median'),
                 std_mag   =('MAG_4_TOT_AB', 'std'),
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
    lc_suffix: str = "",
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
    import pyarrow.parquet as pq_io

    if out_dir is None:
        out_dir = lc_root / 'merged' / band
    out_path = out_dir / 'lightcurves_merged.parquet'

    if out_path.exists() and not force:
        logger.info(f"merge [{band}]: merged parquet already exists — skip (use --force to redo)")
        return out_path

    # ── 1. Load per-quadrant parquets ─────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    all_meta: dict = {}
    for q in quadrants:
        if q['filtercode'] != band:
            continue
        field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
        pq_path = lc_root / f"{field:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid}" / f'lightcurves{lc_suffix}.parquet'
        if not pq_path.exists():
            logger.warning(f"merge [{band}]: parquet missing for {field:06d} {fc} c{ccd:02d} q{qid} — skipped")
            continue

        # Read parquet and carry MAGZP_REF_* metadata forward
        pf = pq_io.read_table(pq_path)
        file_meta = pf.schema.metadata or {}
        all_meta.update(file_meta)

        df = pf.to_pandas()
        # Restore origin columns from quadrant dict (stored as metadata in per-quadrant files)
        df['field']       = field
        df['filtercode']  = fc
        df['ccdid']       = ccd
        df['qid']         = qid
        df['norm_offset'] = np.float32(0.0)
        frames.append(df)

    if len(frames) < 2:
        logger.warning(f"merge [{band}]: fewer than 2 quadrants with data — nothing to merge")
        return None

    # ── 2. Determine dominant quadrant ───────────────────────────────────────
    def _n_clean_epochs(df: pd.DataFrame) -> int:
        return int((df['INFOBITS_DIF'] == 0).sum())

    def _quad_tag(df: pd.DataFrame) -> str:
        return (f"{int(df['field'].iloc[0]):06d}_{df['filtercode'].iloc[0]}"
                f"_c{int(df['ccdid'].iloc[0]):02d}_q{int(df['qid'].iloc[0])}")

    n_clean = [_n_clean_epochs(df) for df in frames]
    dom_idx = int(np.argmax(n_clean))
    dom_tag = _quad_tag(frames[dom_idx])
    logger.info(f"merge [{band}]: dominant quadrant = {dom_tag} ({n_clean[dom_idx]} clean det-epochs)")
    for i, (df, n) in enumerate(zip(frames, n_clean)):
        logger.info(f"  {'[DOM]' if i == dom_idx else '     '} {_quad_tag(df)}: {n} clean det-epochs")

    # ── 3 & 4. Compute and apply normalization offsets to all MAG_* columns ──
    stats_dom = _per_source_stats(frames[dom_idx])

    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        qid_str = _quad_tag(df)
        try:
            stats_min = _per_source_stats(df)
            offset, n_used = _compute_offset(stats_dom, stats_min)
            logger.info(f"merge [{band}]: offset {qid_str} → dominant = "
                        f"{offset:+.4f} mag  (N_common={n_used})")
            for col in _MAG_COLS:
                if col in df.columns:
                    df[col] = (pd.to_numeric(df[col], errors='coerce') + offset).astype('float32')
            df['norm_offset'] = np.float32(offset)
        except ValueError as exc:
            logger.warning(f"merge [{band}]: cannot normalize {qid_str}: {exc}. "
                           "Merging without offset correction.")

    # ── 6. Concatenate and write ──────────────────────────────────────────────
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values('OBSMJD', inplace=True, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    table = pa.Table.from_pandas(merged, preserve_index=False)
    table = table.replace_schema_metadata({
        **all_meta,
        **(table.schema.metadata or {}),
        b'dominant_quadrant': dom_tag.encode(),
    })
    pq_io.write_table(table, out_path)

    n_src_quads = merged.groupby(['field', 'filtercode', 'ccdid', 'qid', 'object_index']).ngroups
    logger.info(f"merge [{band}]: {len(merged)} rows, {n_src_quads} source×quadrant combinations "
                f"→ {out_path}")
    return out_path
