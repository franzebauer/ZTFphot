"""
vet_calibration_stars.py
------------------------
Flag variable calibration stars using their lightcurve scatter relative to
the local photometric precision locus.

This directly corresponds to what is visible in Fig 3: sources whose
magnitude std across clean epochs exceeds threshold × the running-median
locus at that magnitude are flagged as unreliable calibrators.

Sources must pass all calibration eligibility criteria to be considered:
  CLASS_STAR_REF >= 0.7  (stellar morphology)
  FLAG_SE_REF == 0       (no SExtractor flags — same cut used in calib_catalog)
  14 < q_mag < 19        (calibration magnitude range)
  N >= min_epochs        (enough epochs for a reliable std estimate)

Algorithm
---------
1. Load lightcurves.parquet; compute per-source median mag and std across
   FLAG_CLEAN & FLAG_DET epochs.
2. Restrict to calibration-eligible sources (above criteria).
3. Match to reference catalog to get q_mag.
4. Fit locus: running median of std in 0.5-mag bins (robust against outliers).
5. Flag sources with std > threshold × locus(q_mag).
6. Write vet_calib_stars_{field}_{band}_c{ccd}_q{qid}.fits with IS_GOOD column.

Usage
-----
    python vet_calibration_stars.py \\
        --field 443 --band zg --ccd 16 --qid 2 \\
        [--base-dir ../data] [--threshold 2.0] [--min-epochs 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


def _flux_to_mag(flux):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux > 0, -2.5 * np.log10(flux) - 48.6, np.nan)


def fit_locus(q_mag, std_mmag, bin_width=0.5, mag_lo=14.0, mag_hi=19.0):
    """Running-median locus of std vs reference magnitude (in mmag).

    Returns (bin_centers, bin_medians) for use with np.interp.
    Uses median, so a minority of variable stars don't inflate the locus.
    """
    edges = np.arange(mag_lo, mag_hi + bin_width, bin_width)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q_mag >= lo) & (q_mag < hi) & np.isfinite(std_mmag) & (std_mmag > 0)
        if m.sum() >= 5:
            centers.append(0.5 * (lo + hi))
            medians.append(float(np.median(std_mmag[m])))
    return np.array(centers), np.array(medians)


def vet_stars(field, band, ccd, qid, base_dir, threshold, min_epochs):
    field_str = f"{field:06d}"
    ccd_str   = f"{ccd:02d}"

    parquet_path = (base_dir / "LightCurves" / field_str / band
                    / f"ccd{ccd_str}" / f"q{qid}" / "lightcurves.parquet")
    ref_csv_path = (base_dir / "Catalogs"
                    / f"{field_str}_{band}_c{ccd_str}_q{qid}(REFERENCE)[OBJECTS].csv")
    out_path = (base_dir / "Calibrated"
                / field_str / band / ccd_str / str(qid)
                / "vet_calib_stars.fits")

    if not parquet_path.exists():
        print(f"ERROR: lightcurves parquet not found — {parquet_path}")
        sys.exit(1)
    if not ref_csv_path.exists():
        print(f"ERROR: reference catalog not found — {ref_csv_path}")
        sys.exit(1)

    # ── Load and aggregate lightcurves ────────────────────────────────────────
    df = pd.read_parquet(parquet_path)
    clean = df[df['INFOBITS_DIF'] == 0].copy()
    clean['mag'] = pd.to_numeric(clean['MAG_4_TOT_AB'], errors='coerce')
    clean = clean[clean['mag'].notna()]

    grp   = clean.groupby('object_index')
    stats = grp.agg(
        ra        = ('ALPHAWIN_REF', 'first'),
        dec       = ('DELTAWIN_REF', 'first'),
        n         = ('mag', 'count'),
        med_mag   = ('mag', 'median'),
        std_mag   = ('mag', 'std'),
    ).reset_index()

    # FLAG_SE_REF still stored per-source in the parquet
    if 'FLAG_SE_REF' in clean.columns:
        stats = stats.join(clean.groupby('object_index')['FLAG_SE_REF'].first()
                           .rename('flag_se'), on='object_index')
    else:
        stats['flag_se'] = 0

    # ── Match to reference catalog to get q_mag ───────────────────────────────
    ref = pd.read_csv(ref_csv_path)
    for col in ['MAG_APER_4px', 'MAGZP_REF']:
        ref[col] = pd.to_numeric(ref[col], errors='coerce')
    ref['q_mag'] = ref['MAG_APER_4px'] + ref['MAGZP_REF']

    cat_ref   = SkyCoord(ra=ref['ALPHAWIN_J2000'].values * u.deg,
                         dec=ref['DELTAWIN_J2000'].values * u.deg)
    cat_stats = SkyCoord(ra=stats['ra'].values * u.deg,
                         dec=stats['dec'].values * u.deg)
    idx, sep, _ = cat_stats.match_to_catalog_sky(cat_ref)
    matched = sep.arcsec < 3.0
    stats = stats[matched].copy()
    stats['q_mag']    = ref['q_mag'].iloc[idx[matched]].values
    stats['ref_idx']  = idx[matched]
    # class_star from reference CSV (CLASS_STAR column)
    if 'CLASS_STAR' in ref.columns:
        stats['class_star'] = pd.to_numeric(ref['CLASS_STAR'].iloc[idx[matched]].values, errors='coerce')
    else:
        stats['class_star'] = np.nan

    # ── Restrict to calibration-eligible sources ──────────────────────────────
    calib = stats[
        (stats['class_star'] >= 0.7) &
        (stats['flag_se'] == 0) &
        (stats['q_mag'] > 14.0) &
        (stats['q_mag'] < 19.0) &
        (stats['n'] >= min_epochs) &
        stats['std_mag'].notna()
    ].copy()

    print(f"\nVetting {field_str}/{band}/ccd{ccd_str}/q{qid}")
    print(f"  Total sources with >= {min_epochs} clean epochs: "
          f"{(stats['n'] >= min_epochs).sum()}")
    print(f"  Calibration-eligible (CLASS_STAR≥0.7, FLAG_SE=0, "
          f"14<q_mag<19, N≥{min_epochs}): {len(calib)}")

    if len(calib) < 20:
        print("ERROR: too few calibration-eligible sources.")
        sys.exit(1)

    # ── Fit precision locus ───────────────────────────────────────────────────
    std_mmag = calib['std_mag'].values * 1000
    bin_centers, bin_medians = fit_locus(calib['q_mag'].values, std_mmag)

    if len(bin_centers) < 3:
        print("ERROR: too few populated magnitude bins to fit locus.")
        sys.exit(1)

    print(f"\n  Precision locus (q_mag, median std):")
    for c, m in zip(bin_centers, bin_medians):
        print(f"    {c:.1f} mag  →  {m:.1f} mmag")

    mag_clip = np.clip(calib['q_mag'].values, bin_centers[0], bin_centers[-1])
    locus    = np.interp(mag_clip, bin_centers, bin_medians)   # mmag

    calib['locus']   = locus
    calib['ratio']   = std_mmag / np.where(locus > 0, locus, np.nan)
    calib['is_bad']  = calib['ratio'] > threshold

    n_bad   = int(calib['is_bad'].sum())
    n_total = len(calib)
    print(f"\n  Threshold : {threshold:.1f}× local median std")
    print(f"  Flagged   : {n_bad} / {n_total} ({100*n_bad/n_total:.1f}%)")

    if n_bad > 0:
        worst = calib[calib['is_bad']].nlargest(15, 'ratio')
        print(f"\n  Top flagged stars:")
        for _, row in worst.iterrows():
            print(f"    RA={row['ra']:.4f}  Dec={row['dec']:.4f}  "
                  f"q_mag={row['q_mag']:.2f}  "
                  f"std={row['std_mag']*1000:.1f} mmag  "
                  f"locus={row['locus']:.1f} mmag  "
                  f"ratio={row['ratio']:.2f}×  N={int(row['n'])}")

    # ── Build reference catalog with IS_GOOD flag ─────────────────────────────
    ref_out = ref[['ALPHAWIN_J2000', 'DELTAWIN_J2000']].copy()
    ref_out['IS_GOOD'] = True

    if n_bad > 0:
        bad_ref_indices = calib.loc[calib['is_bad'], 'ref_idx'].values
        ref_out.loc[bad_ref_indices, 'IS_GOOD'] = False

    n_good = int(ref_out['IS_GOOD'].sum())
    print(f"\n  Reference catalog: {len(ref_out)} total  "
          f"{n_good} good  {len(ref_out) - n_good} flagged")

    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='ALPHAWIN_J2000', format='D',
                    array=ref_out['ALPHAWIN_J2000'].values),
        fits.Column(name='DELTAWIN_J2000', format='D',
                    array=ref_out['DELTAWIN_J2000'].values),
        fits.Column(name='IS_GOOD', format='L',
                    array=ref_out['IS_GOOD'].values),
    ])
    hdu.writeto(str(out_path), overwrite=True)
    print(f"  Saved → {out_path}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Flag variable calibration stars using lightcurve scatter')
    p.add_argument('--field',       type=int,   required=True)
    p.add_argument('--band',                    required=True)
    p.add_argument('--ccd',         type=int,   required=True)
    p.add_argument('--qid',         type=int,   required=True)
    p.add_argument('--base-dir',    default=Path("data"),
                   help="Data directory (default: ./data in current working directory)")
    p.add_argument('--threshold',   type=float, default=2.0,
                   help='Flag sources with std > THRESHOLD × local locus (default: 2.0)')
    p.add_argument('--min-epochs',  type=int,   default=20,
                   help='Min clean epochs required (default: 20 for reliable std)')
    args = p.parse_args()

    vet_stars(args.field, args.band, args.ccd, args.qid,
              Path(args.base_dir), args.threshold, args.min_epochs)
