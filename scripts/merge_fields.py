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
4.  Fit a 2-D polynomial (default degree=1, i.e. a linear tilt) to
        delta_mag(RA, Dec) = median_mag_dominant - median_mag_minority
    over all common stable sources with iterative 3σ MAD clipping.
    Apply the position-dependent correction to all MAG_* columns.
    Falls back to a scalar median offset if the polynomial fit fails.
5.  Re-key object_index to be globally unique across quadrants, then
    link overlap sources to the dominant quadrant's object_index via
    coordinate matching.
6.  Concatenate all quadrants (with origin columns), sort by OBSMJD, and
    write to LightCurves/merged/{band}/lightcurves_merged.parquet.

Output parquet extra columns (beyond per-quadrant schema)
----------------------------------------------------------
    norm_offset   float32   per-row spatial correction applied to MAG_* columns
                            (0.0 for dominant quadrant rows)
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


def _poly2d_basis(ra: np.ndarray, dec: np.ndarray,
                  ra0: float, dec0: float, degree: int) -> np.ndarray:
    """Design matrix for a 2-D polynomial in (RA-ra0, Dec-dec0)."""
    u = ra  - ra0
    v = dec - dec0
    cols = [np.ones(len(ra)), u, v]
    if degree >= 2:
        cols += [u**2, u * v, v**2]
    return np.column_stack(cols)


def _compute_spatial_correction(
        stats_dom: pd.DataFrame,
        stats_min: pd.DataFrame,
        poly_degree: int = 1,
        max_sep_arcsec: float = MAX_SEP_ARCSEC,
        std_cut: float = STD_CUT,
        min_common: int = MIN_COMMON,
) -> tuple:
    """
    Match sources, sigma-clip, fit a 2-D polynomial to
        delta_mag(RA, Dec) = median_mag_dominant - median_mag_minority
    Returns (coeffs, ra0, dec0, n_used).

    Falls back to a scalar (degree=0) when fewer sources are available than
    needed for the requested degree.  Raises ValueError if even a scalar
    cannot be computed.
    """
    cat_dom = SkyCoord(ra=stats_dom['ra'].values * u.deg,
                       dec=stats_dom['dec'].values * u.deg)
    cat_min = SkyCoord(ra=stats_min['ra'].values * u.deg,
                       dec=stats_min['dec'].values * u.deg)

    idx, sep, _ = cat_min.match_to_catalog_sky(cat_dom)
    matched = sep.arcsec < max_sep_arcsec

    stats_min_m = stats_min[matched].copy()
    stats_dom_m = stats_dom.iloc[idx[matched]].copy()

    stable = (
        (stats_dom_m['std_mag'].values < std_cut) &
        (stats_min_m['std_mag'].values < std_cut)
    )
    n_common = int(stable.sum())
    if n_common < min_common:
        raise ValueError(
            f"Only {n_common} stable common sources (need {min_common})."
        )

    ra_s  = stats_min_m['ra'].values[stable]
    dec_s = stats_min_m['dec'].values[stable]
    diff  = stats_dom_m['median_mag'].values[stable] - stats_min_m['median_mag'].values[stable]

    # Iterative 3σ MAD clipping on residuals
    clip_mask = np.ones(len(diff), dtype=bool)
    for _ in range(3):
        med   = np.median(diff[clip_mask])
        mad   = np.median(np.abs(diff[clip_mask] - med))
        sigma = 1.4826 * mad
        new_mask = np.abs(diff - med) < 3.0 * sigma
        if new_mask.sum() < max(min_common // 2, 3):
            break
        clip_mask = new_mask

    ra_fit  = ra_s[clip_mask]
    dec_fit = dec_s[clip_mask]
    diff_fit = diff[clip_mask]
    n_used   = int(clip_mask.sum())

    ra0, dec0 = float(ra_fit.mean()), float(dec_fit.mean())

    # Degrade to scalar if not enough sources for the requested degree
    n_params = {0: 1, 1: 3, 2: 6}
    deg = poly_degree
    while deg > 0 and n_used < n_params[deg] * 5:
        deg -= 1

    A = _poly2d_basis(ra_fit, dec_fit, ra0, dec0, deg)
    coeffs, _, _, _ = np.linalg.lstsq(A, diff_fit, rcond=None)

    return coeffs, ra0, dec0, deg, n_used


# ── Public API ────────────────────────────────────────────────────────────────

def merge_band(
    lc_root: Path,
    band: str,
    quadrants: list[dict],
    force: bool = False,
    out_dir: Optional[Path] = None,
    lc_suffix: str = "",
    poly_degree: int = 1,
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

    # ── 3 & 4. Compute and apply spatial normalization correction ────────────
    stats_dom = _per_source_stats(frames[dom_idx])

    for i, df in enumerate(frames):
        if i == dom_idx:
            continue
        qid_str = _quad_tag(df)
        try:
            stats_min = _per_source_stats(df)
            coeffs, ra0, dec0, deg_used, n_used = _compute_spatial_correction(
                stats_dom, stats_min, poly_degree=poly_degree)
            scalar_summary = float(coeffs[0])
            logger.info(
                f"merge [{band}]: spatial correction {qid_str} → dominant  "
                f"degree={deg_used}  N_common={n_used}  "
                f"ZP={scalar_summary:+.4f} mag"
            )
            ra_arr  = pd.to_numeric(df['ALPHAWIN_REF'], errors='coerce').values
            dec_arr = pd.to_numeric(df['DELTAWIN_REF'], errors='coerce').values
            correction = (_poly2d_basis(ra_arr, dec_arr, ra0, dec0, deg_used)
                          @ coeffs).astype(np.float32)
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
        matched = sep.arcsec < MAX_SEP_ARCSEC

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
