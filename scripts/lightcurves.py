"""
lightcurves.py
--------------
Pipeline steps for assembling and merging per-object light curves:
  step_lightcurves — assemble per-source parquet from calibrated FITS
  step_merge       — cross-calibrate and merge multiple quadrants per band
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _cast_lc_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast light-curve columns to correct numeric types before writing Parquet.
    FITS LDAC header keywords are parsed as raw strings; this coerces them.

    float64: OBSMJD (MJD precision), ALPHAWIN_REF, DELTAWIN_REF, ALPHA_OBJ,
             DELTA_OBJ (astrometric precision at RA~330 deg)
    float32: all photometric quantities and per-epoch scalars
    Int32:   integer indices and flag words
    """
    float64_cols = [
        'OBSMJD',
        'ALPHAWIN_REF', 'DELTAWIN_REF',
        'ALPHA_OBJ', 'DELTA_OBJ',
    ]
    float32_cols = [
        'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
        'SEEING', 'MAGLIM', 'DISTANCE',
        'CLASS_STAR_OBJ',
        'MAG_3_TOT_AB', 'MERR_3_TOT_AB',
        'MAG_4_TOT_AB', 'MERR_4_TOT_AB',
        'MAG_6_TOT_AB', 'MERR_6_TOT_AB',
        'MAG_10_TOT_AB', 'MERR_10_TOT_AB',
        'MAG_4_TOT_AB_org', 'MERR_4_TOT_AB_org',
        'APCORR46',
    ]
    int32_cols = [
        'NMATCHES', 'INFOBITS_DIF', 'INFOBITS_REF',
        'FLAG_SE_REF', 'object_index',
    ]
    for col in float64_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    for col in float32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    for col in int32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int32')
    return df


# ── Step: build light curves ──────────────────────────────────────────────────

# Header keywords to broadcast from cal.fits HDU[0] to every source row
_EPOCH_KEYS = [
    'OBSMJD', 'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
    'SEEING', 'MAGLIM', 'NMATCHES', 'INFOBITS_DIF', 'APCORR46',
]

# Columns to drop before writing (redundant, derivable, or replaced by metadata)
_DROP_COLS = {
    'ALPHAWIN_J2000', 'DELTAWIN_J2000',   # renamed to ALPHA_OBJ / DELTA_OBJ
    'FLAGS',                               # = FLAG_SE_DIF; dropped
    'FLUX_3_TOT_AB', 'FERR_3_TOT_AB',
    'FLUX_4_TOT_AB', 'FERR_4_TOT_AB',
    'FLUX_6_TOT_AB', 'FERR_6_TOT_AB',
    'FLUX_10_TOT_AB', 'FERR_10_TOT_AB',
    'FLUX_AUTO_TOT_AB', 'FERR_AUTO_TOT_AB',
    'FLUX_3_DIF', 'FERR_3_DIF',
    'FLUX_4_DIF', 'FERR_4_DIF',
    'FLUX_6_DIF', 'FERR_6_DIF',
    'FLUX_10_DIF', 'FERR_10_DIF',
    'FLUX_AUTO_DIF', 'FERR_AUTO_DIF',
    'ALPHA_J2000', 'DELTA_J2000',          # reference positions — kept in ALPHAWIN_REF/DELTAWIN_REF
    'VECTOR_ASSOC',                        # carried as object_index; raw column not needed in parquet
    'MAG_AP_3_REF', 'MAG_AP_4_REF', 'MAG_AP_6_REF', 'MAG_AP_10_REF', 'MAG_AP_AUTO_REF',
    'SATURATE',
    'MAG_3_TOT_AB_org', 'MERR_3_TOT_AB_org',   # only 4px _org retained
    'MAG_6_TOT_AB_org', 'MERR_6_TOT_AB_org',
    'MAG_10_TOT_AB_org', 'MERR_10_TOT_AB_org',
}


def step_lightcurves(
    base_dir: Path, quadrants: list[dict],
    force: bool = False,
    use_calibrated: bool = True,
) -> int:
    """
    Assemble light curves from calibrated per-epoch FITS for each quadrant.
    Reads Calibrated/{field}/{fc}/{ccd}/{qid}/*_cal.fits directly.
    Saves one Parquet per quadrant under LightCurves/.

    Parquet metadata keys (not columns):
      field, filtercode, ccdid, qid
      MAGZP_REF_{field}_{fc}_c{ccd}_q{qid}
      MAGZPRMS_REF_{field}_{fc}_c{ccd}_q{qid}
    """
    import numpy as np
    from astropy.io import fits as pyfits
    import pyarrow as pa
    import pyarrow.parquet as pq

    cat_dir = base_dir / "Catalogs"
    lc_root = base_dir / "LightCurves"

    n_done = n_skip = 0
    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        tag = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"

        lc_dir = lc_root / f"{field:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid_}"
        lc_out = lc_dir / "lightcurves.parquet"

        if lc_out.exists() and not force:
            n_skip += 1
            continue

        ref_csv = cat_dir / f"{tag}(REFERENCE)[OBJECTS].csv"
        if not ref_csv.exists():
            logger.warning(f"Reference catalog not found: {ref_csv}")
            continue

        cal_dir   = base_dir / "Calibrated" / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_)
        cal_files = sorted(cal_dir.glob("*_cal.fits"))
        if not cal_files:
            logger.warning(f"No calibrated FITS found in {cal_dir}")
            continue

        logger.info(f"Building light curves for {tag} ({len(cal_files)} epochs)")

        # Load reference catalog — provides object positions and per-source metadata
        ref = pd.read_csv(ref_csv)
        # Per-source reference columns to carry into LC (MAGZP_REF/MAGZPRMS_REF → metadata)
        ref_cols = {}
        for src, dst in [('ALPHAWIN_J2000', 'ALPHAWIN_REF'), ('DELTAWIN_J2000', 'DELTAWIN_REF'),
                         ('FLAGS', 'FLAG_SE_REF'),
                         ('INFOBITS', 'INFOBITS_REF')]:
            if src in ref.columns:
                ref_cols[dst] = pd.to_numeric(ref[src], errors='coerce').values

        # Extract MAGZP_REF / MAGZPRMS_REF as scalars for parquet metadata
        magzp_ref_val    = float(pd.to_numeric(ref['MAGZP_REF'],    errors='coerce').iloc[0]) if 'MAGZP_REF'    in ref.columns else float('nan')
        magzprms_ref_val = float(pd.to_numeric(ref['MAGZPRMS_REF'], errors='coerce').iloc[0]) if 'MAGZPRMS_REF' in ref.columns else float('nan')

        frames = []
        for cal_path in cal_files:
            try:
                with pyfits.open(cal_path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[1].data
            except Exception as e:
                logger.warning(f"  Could not read {cal_path.name}: {e}")
                continue

            from astropy.table import Table
            tbl = Table(data).to_pandas()
            if tbl.empty:
                continue

            # VECTOR_ASSOC is 1-based; ASSOCSELEC_TYPE=MATCHED guarantees all rows > 0.
            assoc_id = pd.to_numeric(tbl['VECTOR_ASSOC'], errors='coerce').fillna(0).astype(int)
            matched  = assoc_id > 0
            if matched.sum() == 0:
                continue
            ep_rows    = tbl[matched].copy().reset_index(drop=True)
            ep_ref_idx = (assoc_id[matched].values - 1)  # 1-based to 0-based

            # Epoch metadata (broadcast to all sources)
            for key in _EPOCH_KEYS:
                ep_rows[key] = hdr.get(key, np.nan)

            # Reference catalog properties
            for dst, arr in ref_cols.items():
                ep_rows[dst] = arr[ep_ref_idx]

            ep_rows['object_index'] = ep_ref_idx

            # Rename cal.fits measurement columns to LC schema
            rename = {
                'ALPHAWIN_J2000': 'ALPHA_OBJ', 'DELTAWIN_J2000': 'DELTA_OBJ',
                'CLASS_STAR': 'CLASS_STAR_OBJ',
            }
            ep_rows = ep_rows.rename(columns={k: v for k, v in rename.items() if k in ep_rows.columns})

            frames.append(ep_rows)

        if not frames:
            logger.warning(f"No light curves built for {tag}")
            continue

        all_lcs = pd.concat(frames, ignore_index=True)

        # Drop redundant / removed columns
        drop = [c for c in _DROP_COLS if c in all_lcs.columns]
        if drop:
            all_lcs = all_lcs.drop(columns=drop)

        all_lcs = _cast_lc_dtypes(all_lcs)

        # Write parquet with quadrant identity and ZP info as file-level metadata
        lc_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(all_lcs, preserve_index=False)
        existing_meta = table.schema.metadata or {}
        extra_meta = {
            b'field':      str(field).encode(),
            b'filtercode': fc.encode(),
            b'ccdid':      str(ccd).encode(),
            b'qid':        str(qid_).encode(),
            f'MAGZP_REF_{tag}'.encode():    str(magzp_ref_val).encode(),
            f'MAGZPRMS_REF_{tag}'.encode(): str(magzprms_ref_val).encode(),
        }
        table = table.replace_schema_metadata({**existing_meta, **extra_meta})
        pq.write_table(table, lc_out)

        n_obj = all_lcs['object_index'].nunique()
        logger.info(f"  → {n_obj} objects, {len(all_lcs)} rows → {lc_out}")
        n_done += 1

    logger.info(f"lightcurves: {n_done} processed, {n_skip} already exist")
    return n_done


# ── Step: merge quadrants ─────────────────────────────────────────────────────

def step_merge(base_dir: Path, quadrants: list[dict], force: bool = False,
               target_ra: float | None = None, target_dec: float | None = None) -> None:
    """
    Cross-calibrate and merge per-quadrant light curves for each band.

    The quadrant with the most clean detection-epochs is adopted as the
    photometric reference. All others are shifted by a median offset from
    stable common sources, then merged into LightCurves/merged/{band}/.
    """
    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from merge_fields import merge_band

    lc_root = base_dir / "LightCurves"
    bands   = sorted({q['filtercode'] for q in quadrants})

    for band in bands:
        band_qs = [q for q in quadrants if q['filtercode'] == band]
        if len(band_qs) < 2:
            logger.info(f"merge [{band}]: only 1 quadrant — skipping cross-calibration")
            continue
        if target_ra is None or target_dec is None:
            logger.error("merge: --ra and --dec are required to determine output directory")
            continue
        out_dir = lc_root / "merged" / f"{target_ra:.5f}_{target_dec:+.5f}" / band
        merge_band(lc_root=lc_root, band=band, quadrants=band_qs, force=force,
                   out_dir=out_dir)
