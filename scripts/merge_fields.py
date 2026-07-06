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
    (matched within MAX_SEP arcsec, N_clean >= MIN_EPOCHS in both quadrants).
4.  Measure
        delta_mag(mag) = median_mag_dominant - median_mag_minority
    as a function of magnitude over all common sources, in bins of width
    MAG_BIN with a 3σ MAD clip per bin.  Apply the magnitude-dependent
    correction to all MAG_* columns by interpolating the curve at each row's
    MAG_4_TOT_AB.  Falls back to a scalar median offset if too few bins.
5.  Re-key object_index to be globally unique across quadrants, then
    link overlap sources to the dominant quadrant's object_index via
    coordinate matching.
6.  Concatenate all quadrants (with origin columns), sort by OBSMJD, and
    write to LightCurves/merged/{band}/lightcurves_merged.parquet.

Output parquet extra columns (beyond per-quadrant schema)
----------------------------------------------------------
    norm_offset   float32   per-row magnitude-dependent correction applied to
                            MAG_* columns (0.0 for dominant quadrant rows)
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
MIN_COMMON     = 10       # min number of common sources required for any offset

MAG_BIN        = 0.1      # width of magnitude bins for the inter-quadrant correction
MIN_BIN        = 5        # min common sources required to trust a magnitude bin
MAG_BRIGHT     = 14.0
MAG_FAINT      = 21.5
MAX_ERR        = 0.1      # max MERR_4_TOT_AB for a clean epoch


_MAG_COLS = ['MAG_3_TOT_AB', 'MAG_4_TOT_AB', 'MAG_6_TOT_AB', 'MAG_10_TOT_AB', 'MAG_4_TOT_AB_org']


def _per_source_stats(df: pd.DataFrame, min_epochs: int = MIN_EPOCHS) -> pd.DataFrame:
    """
    Return one row per object_index with ra, dec, n_clean, median_mag.
    Only sources with n_clean >= min_epochs are returned.

    Uses MAG_4_TOT_AB with the same clean-epoch filter as plot_quad_offsets.py
    and renorm_merged_parquet.py (INFOBITS_DIF == 0, magnitude in
    [MAG_BRIGHT, MAG_FAINT], error < MAX_ERR), over the full magnitude range so
    the inter-quadrant offset can be measured as a function of magnitude.
    """
    mag_col = 'MAG_4_TOT_AB'
    mask = (
        (df['INFOBITS_DIF'] == 0) &
        df[mag_col].notna() &
        (df[mag_col] > MAG_BRIGHT) &
        (df[mag_col] < MAG_FAINT)
    )
    if 'MERR_4_TOT_AB' in df.columns:
        mask &= df['MERR_4_TOT_AB'].fillna(999) < MAX_ERR
    clean = df[mask]
    stats = (clean.groupby('object_index')
             .agg(
                 ra        =('ALPHAWIN_REF', 'first'),
                 dec       =('DELTAWIN_REF', 'first'),
                 n_clean   =(mag_col, 'count'),
                 median_mag=(mag_col, 'median'),
             )
             .reset_index())
    return stats[stats['n_clean'] >= min_epochs]


def _compute_mag_correction(
        stats_dom: pd.DataFrame,
        stats_min: pd.DataFrame,
        mag_bin: float = MAG_BIN,
        max_sep_arcsec: float = MAX_SEP_ARCSEC,
        min_common: int = MIN_COMMON,
) -> tuple:
    """
    Match dominant and minority sources by position and measure
        delta_mag(mag) = median_mag_dominant - median_mag_minority
    as a function of (dominant) magnitude, in bins of width `mag_bin` with a
    3σ MAD clip per bin.  Returns (centers, deltas, n_common) suitable for
    np.interp.

    Falls back to a single-element (scalar) curve when fewer than two bins
    hold >= MIN_BIN sources.  Raises ValueError if too few common sources to
    compute even a scalar.
    """
    cat_dom = SkyCoord(ra=stats_dom['ra'].values * u.deg,
                       dec=stats_dom['dec'].values * u.deg)
    cat_min = SkyCoord(ra=stats_min['ra'].values * u.deg,
                       dec=stats_min['dec'].values * u.deg)

    idx, sep, _ = cat_min.match_to_catalog_sky(cat_dom)
    matched = sep.arcsec < max_sep_arcsec

    n_common = int(matched.sum())
    if n_common < min_common:
        raise ValueError(f"Only {n_common} common sources (need {min_common}).")

    dom_mag = stats_dom['median_mag'].values[idx[matched]]
    sec_mag = stats_min['median_mag'].values[matched]
    diff    = dom_mag - sec_mag           # dominant - minority
    mag_axis = dom_mag                    # the more reliable magnitude

    def _clip_median(d: np.ndarray) -> float:
        med  = float(np.median(d))
        mad  = float(np.median(np.abs(d - med)))
        sig  = 1.4826 * mad
        keep = np.abs(d - med) < 3.0 * sig if sig > 0 else np.ones(len(d), bool)
        return float(np.median(d[keep]))

    edges = np.arange(MAG_BRIGHT, MAG_FAINT + mag_bin, mag_bin)
    centers, deltas = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = diff[(mag_axis >= lo) & (mag_axis < hi)]
        if len(d) < MIN_BIN:
            continue
        centers.append(0.5 * (lo + hi))
        deltas.append(_clip_median(d))

    if len(centers) < 2:
        # too few populated bins → single scalar curve (flat with magnitude)
        return np.array([17.0]), np.array([_clip_median(diff)]), n_common

    return np.asarray(centers), np.asarray(deltas), n_common


# ── Public API ────────────────────────────────────────────────────────────────

def merge_band(
    lc_root: Path,
    band: str,
    quadrants: list[dict],
    force: bool = False,
    out_dir: Optional[Path] = None,
    lc_suffix: str = "",
    mag_bin: float = MAG_BIN,
    max_sep_arcsec: float = MAX_SEP_ARCSEC,
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

    # ── 3 & 4. Compute and apply magnitude-dependent normalization correction ──
    stats_dom = _per_source_stats(frames[dom_idx])

    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        qid_str = _quad_tag(df)
        try:
            stats_min = _per_source_stats(df)
            centers, deltas, n_used = _compute_mag_correction(
                stats_dom, stats_min, mag_bin=mag_bin, max_sep_arcsec=max_sep_arcsec)
            logger.info(
                f"merge [{band}]: magnitude correction {qid_str} → dominant  "
                f"bins={len(centers)}  N_common={n_used}  "
                f"bright={deltas[0]:+.4f}@{centers[0]:.2f}  "
                f"faint={deltas[-1]:+.4f}@{centers[-1]:.2f}  "
                f"median={float(np.median(deltas)):+.4f} mag"
            )
            row_mag = pd.to_numeric(df['MAG_4_TOT_AB'], errors='coerce').values
            correction = np.interp(row_mag, centers, deltas)  # flat extrapolation
            correction[~np.isfinite(row_mag)] = float(np.median(deltas))
            correction = correction.astype(np.float32)
            for col in _MAG_COLS:
                if col in df.columns:
                    df[col] = (pd.to_numeric(df[col], errors='coerce')
                               + correction).astype('float32')
            df['norm_offset'] = correction
        except ValueError as exc:
            logger.warning(f"merge [{band}]: cannot normalize {qid_str}: {exc}. "
                           "Merging without offset correction.")

    # ── 5. Re-key object_index to be globally unique, then link overlap sources ─
    # Step 5a: offset each secondary quadrant's indices above all previous blocks
    running_max = int(frames[dom_idx]['object_index'].max())
    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        df['object_index'] = df['object_index'] + running_max + 1
        running_max = int(df['object_index'].max())

    # Step 5b: replace matched secondary indices with the dominant's object_index
    # so that the same physical star shares one index across quadrants
    from astropy.coordinates import SkyCoord as _SkyCoord
    import astropy.units as _u

    dom_pos = (frames[dom_idx]
               .groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']]
               .first().reset_index())
    cat_dom = _SkyCoord(ra=dom_pos['ALPHAWIN_REF'].values * _u.deg,
                        dec=dom_pos['DELTAWIN_REF'].values * _u.deg)

    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        min_pos = (df.groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']]
                   .first().reset_index())
        cat_min = _SkyCoord(ra=min_pos['ALPHAWIN_REF'].values * _u.deg,
                            dec=min_pos['DELTAWIN_REF'].values * _u.deg)
        idx_arr, sep, _ = cat_min.match_to_catalog_sky(cat_dom)
        matched = sep.arcsec < max_sep_arcsec

        remap = {
            int(min_pos['object_index'].iloc[j]): int(dom_pos['object_index'].iloc[idx_arr[j]])
            for j in range(len(min_pos)) if matched[j]
        }
        if remap:
            df['object_index'] = (df['object_index']
                                  .map(remap)
                                  .fillna(df['object_index'])
                                  .astype('Int32'))
            logger.info(f"merge [{band}]: linked {len(remap)} overlap sources from "
                        f"{_quad_tag(df)} → dominant object_index")

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
