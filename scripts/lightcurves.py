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

# ── Quality-flag constants ────────────────────────────────────────────────────

_HARD_REJECT_MASK = (1 << 0) | (1 << 1) | (1 << 25)          # = 33554435
_CAUTIONARY_MASK  = (
    (1 << 2)  | (1 << 3)  | (1 << 4)  | (1 << 5)  | (1 << 6)
  | (1 << 11) | (1 << 21) | (1 << 22) | (1 << 26) | (1 << 27)
)


def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean quality-flag columns based on INFOBITS_DIF.

    FLAG_HARD_REJECT — bits 0, 1, 25 set (no valid astrometric/photometric calib)
    FLAG_CAUTIONARY  — cautionary bits set but not hard-rejected
    FLAG_CLEAN       — INFOBITS_DIF == 0 (most conservative clean sample)
    FLAG_USABLE      — not hard-rejected (clean + cautionary)
    """
    bits = pd.to_numeric(df["INFOBITS_DIF"], errors="coerce").fillna(-1).astype("int64")

    hard_reject = (bits & _HARD_REJECT_MASK) != 0
    cautionary  = (~hard_reject) & ((bits & _CAUTIONARY_MASK) != 0)
    clean       = (~hard_reject) & (~cautionary) & (bits >= 0)

    df["FLAG_HARD_REJECT"] = hard_reject
    df["FLAG_CAUTIONARY"]  = cautionary
    df["FLAG_CLEAN"]       = clean
    df["FLAG_USABLE"]      = ~hard_reject
    return df


def _cast_lc_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all light-curve columns to correct numeric types before writing Parquet.

    FITS LDAC header keywords are parsed as raw strings; mixed string+numeric
    dtypes cannot be serialised to Parquet. Uses pd.to_numeric(errors='coerce').
    """
    float_cols = [
        'OBSMJD', 'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
        'SATURATE', 'SEEING', 'MAGLIM', 'DISTANCE',
        'ALPHAWIN_REF', 'DELTAWIN_REF', 'MAGZP_REF', 'MAGZPRMS_REF',
        'ALPHA_OBJ', 'DELTA_OBJ', 'ALPHAWIN_OBJ', 'DELTAWIN_OBJ',
        'CLASS_STAR_REF', 'CLASS_STAR_OBJ',
        'FLUX_3_DIF',  'FLUX_4_DIF',  'FLUX_6_DIF',  'FLUX_10_DIF',  'FLUX_AUTO_DIF',
        'FERR_3_DIF',  'FERR_4_DIF',  'FERR_6_DIF',  'FERR_10_DIF',  'FERR_AUTO_DIF',
        'FLUX_3_TOT_AB',  'FLUX_4_TOT_AB',  'FLUX_6_TOT_AB',
        'FLUX_10_TOT_AB', 'FLUX_AUTO_TOT_AB',
        'FERR_3_TOT_AB',  'FERR_4_TOT_AB',  'FERR_6_TOT_AB',
        'FERR_10_TOT_AB', 'FERR_AUTO_TOT_AB',
        'MAG_AP_3_REF', 'MAG_AP_4_REF', 'MAG_AP_6_REF',
        'MAG_AP_10_REF', 'MAG_AP_AUTO_REF',
    ]
    int_cols = [
        'NMATCHES', 'INFOBITS_DIF', 'INFOBITS_REF',
        'FLAG_SE_DIF', 'FLAG_SE_REF', 'object_index',
        'field', 'ccdid', 'qid',
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


# ── Step: build light curves ──────────────────────────────────────────────────

# Header keywords to broadcast from cal.fits HDU[0] to every source row
_EPOCH_KEYS = [
    'OBSMJD', 'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
    'SATURATE', 'SEEING', 'MAGLIM', 'NMATCHES', 'INFOBITS_DIF',
]


def step_lightcurves(
    base_dir: Path, quadrants: list[dict],
    force: bool = False,
    use_calibrated: bool = True,
) -> int:
    """
    Assemble light curves from calibrated per-epoch FITS for each quadrant.
    Reads Calibrated/{field}/{fc}/{ccd}/{qid}/*_cal.fits directly.
    Saves one Parquet per quadrant under LightCurves/.
    """
    import numpy as np
    from astropy.io import fits as pyfits
    from astropy.coordinates import SkyCoord

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
        ref_ra  = pd.to_numeric(ref.get('ALPHAWIN_J2000', ref.get('RA')), errors='coerce').values
        ref_dec = pd.to_numeric(ref.get('DELTAWIN_J2000', ref.get('DEC')), errors='coerce').values
        ref_coords = SkyCoord(ra=ref_ra, dec=ref_dec, unit='deg')

        # Reference-catalog per-source columns to carry into LC
        ref_cols = {}
        for src, dst in [('ALPHAWIN_J2000', 'ALPHAWIN_REF'), ('DELTAWIN_J2000', 'DELTAWIN_REF'),
                         ('CLASS_STAR', 'CLASS_STAR_REF'), ('FLAGS', 'FLAG_SE_REF'),
                         ('MAGZP_REF', 'MAGZP_REF'), ('MAGZPRMS_REF', 'MAGZPRMS_REF'),
                         ('INFOBITS', 'INFOBITS_REF')]:
            if src in ref.columns:
                ref_cols[dst] = pd.to_numeric(ref[src], errors='coerce').values

        frames = []
        for cal_path in cal_files:
            try:
                with pyfits.open(cal_path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[1].data
            except Exception as e:
                logger.warning(f"  Could not read {cal_path.name}: {e}")
                continue

            tbl = pd.DataFrame(np.array(data).byteswap().newbyteorder())
            if tbl.empty:
                continue

            # Cross-match epoch detections to reference catalog (1.5 arcsec)
            ep_ra  = pd.to_numeric(tbl.get('ALPHAWIN_J2000', tbl.get('ALPHA_J2000')), errors='coerce').values
            ep_dec = pd.to_numeric(tbl.get('DELTAWIN_J2000', tbl.get('DELTA_J2000')), errors='coerce').values
            valid  = np.isfinite(ep_ra) & np.isfinite(ep_dec)
            if valid.sum() == 0:
                continue

            ep_coords = SkyCoord(ra=ep_ra[valid], dec=ep_dec[valid], unit='deg')
            ref_idx, sep, _ = ep_coords.match_to_catalog_sky(ref_coords)
            matched = sep.arcsec < 1.5

            if matched.sum() == 0:
                continue

            ep_rows      = tbl[valid][matched].copy().reset_index(drop=True)
            ep_ref_idx   = ref_idx[matched]

            # Epoch metadata (broadcast to all sources)
            for key in _EPOCH_KEYS:
                ep_rows[key] = hdr.get(key, np.nan)

            # Reference catalog properties
            for dst, arr in ref_cols.items():
                ep_rows[dst] = arr[ep_ref_idx]

            ep_rows['object_index'] = ep_ref_idx
            ep_rows['ID_REF'] = [
                f"{i}_{field:06d}_{fc}_c{ccd:02d}_q{qid_}" for i in ep_ref_idx
            ]
            ep_rows['ALPHAWIN_OBJ'] = ep_ra[valid][matched]
            ep_rows['DELTAWIN_OBJ'] = ep_dec[valid][matched]
            ep_rows['FLAG_DET']     = True

            # Rename cal.fits measurement columns to LC schema
            rename = {
                'ALPHAWIN_J2000': 'ALPHA_OBJ', 'DELTAWIN_J2000': 'DELTA_OBJ',
                'FLAGS': 'FLAG_SE_DIF', 'CLASS_STAR': 'CLASS_STAR_OBJ',
                'FLUX_4_TOT_AB': 'FLUX_4_TOT_AB', 'FERR_4_TOT_AB': 'FERR_4_TOT_AB',
                'FLUX_3_TOT_AB': 'FLUX_3_TOT_AB', 'FERR_3_TOT_AB': 'FERR_3_TOT_AB',
                'FLUX_6_TOT_AB': 'FLUX_6_TOT_AB', 'FERR_6_TOT_AB': 'FERR_6_TOT_AB',
                'FLUX_10_TOT_AB': 'FLUX_10_TOT_AB', 'FERR_10_TOT_AB': 'FERR_10_TOT_AB',
                'MAG_4_TOT_AB': 'MAG_4_TOT_AB', 'MERR_4_TOT_AB': 'MERR_4_TOT_AB',
            }
            ep_rows = ep_rows.rename(columns={k: v for k, v in rename.items() if k in ep_rows.columns})

            frames.append(ep_rows)

        if not frames:
            logger.warning(f"No light curves built for {tag}")
            continue

        all_lcs = pd.concat(frames, ignore_index=True)
        all_lcs["field"]      = field
        all_lcs["filtercode"] = fc
        all_lcs["ccdid"]      = ccd
        all_lcs["qid"]        = qid_
        all_lcs = _cast_lc_dtypes(all_lcs)
        all_lcs = _add_quality_flags(all_lcs)

        lc_dir.mkdir(parents=True, exist_ok=True)
        all_lcs.to_parquet(lc_out, index=False)

        n_obj = all_lcs['object_index'].nunique()
        logger.info(f"  → {n_obj} objects, {len(all_lcs)} rows → {lc_out}")
        n_done += 1

    logger.info(f"lightcurves: {n_done} processed, {n_skip} already exist")
    return n_done


# ── Step: merge quadrants ─────────────────────────────────────────────────────

def step_merge(base_dir: Path, quadrants: list[dict], force: bool = False) -> None:
    """
    Cross-calibrate and merge per-quadrant light curves for each band.

    The quadrant with the most FLAG_CLEAN detection-epochs is adopted as the
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
        merge_band(lc_root=lc_root, band=band, quadrants=band_qs, force=force)
