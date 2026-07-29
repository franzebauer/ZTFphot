#!/usr/bin/env python3
"""
recalibrate_parquet.py
----------------------
Re-derive the photometric calibration of existing *merged* light-curve parquet
files using the CURRENT pipeline calibration, without any of the original
intermediate products (SExtractor catalogs, calibrated FITS, downloads).

Why this is possible
--------------------
The merged parquet already carries the pre-calibration re-entry point:

    magQi = MAG_4_TOT_AB_org - norm_offset          (aperture 4, per source/epoch)

which is exactly the `maginst` that `calib_catalogs.calib_catalog` feeds into its
staged fit (linear ZP -> 3-sigma clip -> 2D poly -> flatfield -> faint).  That fit
uses only magQi, the reference magnitude q_mag, positions and errors -- no fluxes,
no image headers.  Everything except q_mag is in the parquet; q_mag is recovered
by re-fetching the small, static per-quadrant reference products from IRSA and
rebuilding the reference catalog (make_catalog), which reproduces the identical
object ordering.

What it does, per input parquet
-------------------------------
  1. read quadrant set (field/ccd/qid/filtercode) straight from the parquet
  2. fetch refimg.fits + refsexcat.fits per quadrant from IRSA (needs ~/.netrc)
  3. make_catalog -> reference CSVs (gives q_mag = MAG_APER_4px + MAGZP_REF)
  4. position-match each source to the reference catalog
  5. two-pass calibration re-using the pipeline steps:
        pass 1  : calibrate (no flatfield) -> lightcurves -> vet
        build flatfield
        pass 2  : recalibrate (with flatfield + vetting) -> lightcurves -> merge
  6. aperture 4 is recomputed from magQi; 3/6/10 are shifted by the same delta
     (MAG_k_new = MAG_k_old + (MAG_4_new - MAG_4_old)); MAG_4_REF is added
  7. write <stem>_recal.parquet next to the input, and delete ALL temporaries

Only the aperture-4 pre-calibration magnitude is retained in the parquet, so
apertures 3/6/10 are corrected by the aperture-4 shift (they do not carry an
independent recalibration).  Handles both ref-position and science-position
(`_sci`) products, merged and non-merged: `_sci` files are detected by name and
calibrated at the per-epoch science centroids (ALPHA_SCI/DELTA_SCI) through the
parallel `_sci` trees, with vetting seeded from the sci light curves.

Usage
-----
    conda activate ztf
    python recalibrate_parquet.py file1.parquet file2.parquet ...
    python recalibrate_parquet.py --list parquets.txt
    # options: --outdir DIR  --suffix _recal  --keep-temp  --match-tol 0.5

The script imports the pipeline modules from its own directory, so keep it in
ZTFphot/scripts/.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits as pyfits
from astropy.io.fits.verify import VerifyWarning
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Pure calibration helpers reused verbatim from the pipeline (no duplication)
from calib_catalogs import _poly2d_basis, _fit_poly2d, _apply_flatfield
# Pipeline steps reused unchanged for assembly / vetting / flatfield / merge
from calibrate import step_build_flatfield, step_vet
from lightcurves import step_lightcurves, step_merge
from make_catalog import make_catalog
from download_coordinator import ref_url, ref_local_path, get_auth, download_file

logger = logging.getLogger("recalibrate_parquet")

# Reference products needed to rebuild q_mag: image stack (carries MAGZP_REF in
# its header) and the SExtractor reference catalog (aperture magnitudes).
_REF_PRODUCTS = ["refimg.fits", "refsexcat.fits"]

_fit_fun = lambda x, n, m: m * x + n


# ── per-epoch calibration (faithful port of calib_catalog, aperture 4 only) ───

def _calibrate_epoch(maginst, q_mag, q_err, errinst, ra, dec, class_star,
                     is_good, ra0, dec0, flatfield=None, poly_degree=2,
                     faint_err_max=0.5):
    """Return (Q_cal, Q_err, (ra_c, dec_c, dm4)) for one epoch, or None if too
    few calibrators. `maginst` == magQi (already aperture-corrected). Mirrors the
    k=1 staged math of calib_catalogs.calib_catalog exactly."""
    maginst_all = np.asarray(maginst, dtype=float)
    errinst_all = np.asarray(errinst, dtype=float)
    q_mag_all   = np.asarray(q_mag, dtype=float)
    q_err_all   = np.asarray(q_err, dtype=float)
    ra          = np.asarray(ra, dtype=float)
    dec         = np.asarray(dec, dtype=float)
    class_star  = np.asarray(class_star, dtype=float)
    is_good     = np.asarray(is_good, dtype=bool)
    flags       = np.zeros(len(maginst_all))  # SIM detection frame: flags forced to 0

    # ── calibrator selection (identical cuts to calib_catalog) ────────────────
    fn = np.where(
        (class_star >= 0.7) & (flags == 0) &
        (q_mag_all > 14.) & (q_mag_all < 19.0) & (q_err_all < 0.3) &
        (maginst_all < 19.0) & (maginst_all > 14.) & (errinst_all < 0.3) &
        is_good
    )
    maginst = maginst_all[fn]
    ra_c, dec_c = ra[fn], dec[fn]
    if len(maginst) <= 15:
        return None
    errinst = errinst_all[fn]
    q_mag   = q_mag_all[fn]
    q_err   = q_err_all[fn]
    sigma   = np.sqrt(errinst**2 + q_err**2)
    errf    = sigma
    diff    = maginst - q_mag

    # ── Step 1: initial linear fit ────────────────────────────────────────────
    coeffs, _ = curve_fit(_fit_fun, maginst, diff, p0=[0, 0],
                          sigma=errf, absolute_sigma=True)
    fit  = _fit_fun(maginst, *coeffs)
    res1 = np.abs(diff - fit)
    rms  = np.sum((diff - fit)**2) / len(maginst)

    # ── Step 2: 3-sigma iterative rejection ──────────────────────────────────
    l1 = np.where((res1 / rms**0.5) <= 3)
    r1 = np.where((res1 / rms**0.5) > 3)
    diff2 = diff
    diff  = diff[l1]; diff_rej = diff2[r1]
    sigma = sigma[l1]; errf = errf[l1]; maginst = maginst[l1]; q_mag = q_mag[l1]
    ra_c = ra_c[l1]; dec_c = dec_c[l1]
    if len(diff) <= 10:
        return None

    while len(diff) > 10:
        if len(diff_rej) == 0:
            break
        coeffs, _ = curve_fit(_fit_fun, maginst, diff, p0=[0, 0],
                              sigma=errf, absolute_sigma=True)
        fit  = _fit_fun(maginst, *coeffs)
        res2 = np.abs(diff - fit)
        rms  = np.sum((diff - fit)**2) / len(maginst)
        l2 = np.where((res2 / rms**0.5) <= 3)
        r2 = np.where((res2 / rms**0.5) > 3)
        diff2 = diff
        diff  = diff[l2]; diff_rej = diff2[r2]
        sigma = sigma[l2]; errf = errf[l2]; maginst = maginst[l2]; q_mag = q_mag[l2]
        ra_c = ra_c[l2]; dec_c = dec_c[l2]

    coeffs, _ = curve_fit(_fit_fun, maginst, diff, p0=[0, 0],
                          sigma=errf, absolute_sigma=True)
    fit       = _fit_fun(maginst, *coeffs)
    final_fit = _fit_fun(maginst_all, *coeffs)

    # per-bin RMS for the error floor (identical bins to calib_catalog)
    _bin_edges = [14, 15.5, 17, 17.5, 18, 18.5, 19, 19.5]
    rms_per_bin, median_mag_per_bin = [], []
    for _lo, _hi in zip(_bin_edges[:-1], _bin_edges[1:]):
        _bm = (maginst >= _lo) & (maginst < _hi)
        if _bm.sum() < 2:
            continue
        _d = diff[_bm]; _f = fit[_bm]
        _res = np.abs(_d - _f)
        _rms = np.sum((_d - _f)**2) / _bm.sum()
        if _rms == 0:
            continue
        _l = (_res / _rms**0.5) <= 5
        if _l.sum() > 1:
            rms_per_bin.append(
                np.sqrt(np.sum((_d[_l] - np.mean(_d[_l]))**2) / (_l.sum() - 1)))
            median_mag_per_bin.append(float(np.median(maginst[_bm][_l])))

    Q_cal = maginst_all - final_fit

    # ── Step 3: 2-D polynomial spatial correction ────────────────────────────
    _dm_for_poly = diff - fit
    _poly_fitted = np.zeros(len(ra_c))
    try:
        _poly_coeffs, _poly_fitted = _fit_poly2d(ra_c, dec_c, _dm_for_poly,
                                                 ra0, dec0, poly_degree)
        _poly_corr = _poly2d_basis(ra, dec, ra0, dec0, poly_degree) @ _poly_coeffs
    except Exception as exc:
        logger.warning(f"    poly2d fit failed: {exc}")
        _poly_corr = np.zeros(len(ra))
    Q_cal = Q_cal - _poly_corr

    # ── Step 4: stacked flatfield correction ─────────────────────────────────
    # A sparse quadrant can yield a degenerate flatfield whose grid is entirely
    # (or partly) NaN — e.g. too few sources per bin, "bins=0/400". Applying it
    # verbatim would subtract NaN and poison every magnitude, so treat undefined
    # cells as a zero correction (no flatfield where it could not be measured).
    if flatfield is not None:
        try:
            _ff = _apply_flatfield(ra, dec, flatfield)
            Q_cal = Q_cal - np.where(np.isfinite(_ff), _ff, 0.0)
        except Exception as exc:
            logger.warning(f"    flatfield apply failed: {exc}")

    # ── Step 5: faint-source per-bin smoothed correction ─────────────────────
    _FC_EDGES   = np.arange(18.5, 22.0001, 0.25)
    _FC_CENTERS = 0.5 * (_FC_EDGES[:-1] + _FC_EDGES[1:])
    residual_all = Q_cal - q_mag_all
    _bin_med = np.full(len(_FC_CENTERS), np.nan)
    _all_fc_mask = (
        (maginst_all >= 18.5) & (maginst_all < 22.0) &
        (errinst_all < faint_err_max) & np.isfinite(residual_all)
    )
    for _ib, (_lo, _hi) in enumerate(zip(_FC_EDGES[:-1], _FC_EDGES[1:])):
        _bm = _all_fc_mask & (maginst_all >= _lo) & (maginst_all < _hi)
        if _bm.sum() >= 5:
            _r = residual_all[_bm]
            _med = np.nanmedian(_r)
            _mad = np.nanmedian(np.abs(_r - _med))
            _gd = (np.abs(_r - _med) < 3.0 * 1.4826 * _mad
                   if _mad > 0 else np.ones(len(_r), dtype=bool))
            if _gd.sum() >= 3:
                _bin_med[_ib] = float(np.nanmedian(_r[_gd]))

    faint_corr_curve = None
    _valid_fc = np.isfinite(_bin_med)
    if _valid_fc.sum() >= 3:
        _filled = np.interp(_FC_CENTERS, _FC_CENTERS[_valid_fc], _bin_med[_valid_fc])
        _emp = _FC_CENTERS >= 19.0
        _ctrl_mag = np.concatenate([[18.5], _FC_CENTERS[_emp], [24.0]])
        _ctrl_val = np.concatenate([[0.0],  _filled[_emp],     [_filled[_emp][-1]]])
        _grid  = np.arange(17.5, 24.0001, 0.05)
        _curve = np.interp(_grid, _ctrl_mag, _ctrl_val)
        _curve = gaussian_filter1d(_curve, sigma=0.2 / 0.05, mode='nearest')
        faint_corr_curve = (_grid, _curve)

    if faint_corr_curve is not None:
        _corr_all = np.interp(maginst_all, faint_corr_curve[0], faint_corr_curve[1],
                              left=0.0, right=faint_corr_curve[1][-1])
        Q_cal = Q_cal - _corr_all

    # error floor
    if median_mag_per_bin:
        interp = np.interp(Q_cal, median_mag_per_bin, rms_per_bin)
        Q_err  = np.maximum(interp, errinst_all)
    else:
        Q_err = errinst_all.copy()

    dm4 = _dm_for_poly - _poly_fitted
    return Q_cal, Q_err, (ra_c, dec_c, dm4)


# ── _cal.fits writer (columns/header that step_lightcurves reads) ─────────────

_CAL_HDR_KEYS = ['OBSMJD', 'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
                 'SEEING', 'MAGLIM', 'NMATCHES', 'INFOBITS_DIF', 'APCORR46']


def _write_cal_fits(path, hdr_vals, cols):
    """Write a per-epoch calibrated FITS in the format step_lightcurves expects."""
    prim = pyfits.PrimaryHDU()
    # Some header keys (MAGZP_DIF, MAGZPRMS_DIF, INFOBITS_DIF) exceed 8 chars and
    # become HIERARCH cards; step_lightcurves reads them back fine, so suppress
    # the (purely cosmetic) VerifyWarning to keep batch logs clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        for k in _CAL_HDR_KEYS:
            v = hdr_vals.get(k, np.nan)
            try:
                prim.header[k] = float(v)
            except (TypeError, ValueError):
                prim.header[k] = v
    fcols = [
        pyfits.Column(name='ALPHAWIN_J2000',   format='D', array=cols['ra']),
        pyfits.Column(name='DELTAWIN_J2000',   format='D', array=cols['dec']),
        pyfits.Column(name='MAG_3_TOT_AB',     format='D', array=cols['mag3']),
        pyfits.Column(name='MERR_3_TOT_AB',    format='D', array=cols['merr3']),
        pyfits.Column(name='MAG_4_TOT_AB',     format='D', array=cols['mag4']),
        pyfits.Column(name='MERR_4_TOT_AB',    format='D', array=cols['merr4']),
        pyfits.Column(name='MAG_6_TOT_AB',     format='D', array=cols['mag6']),
        pyfits.Column(name='MERR_6_TOT_AB',    format='D', array=cols['merr6']),
        pyfits.Column(name='MAG_10_TOT_AB',    format='D', array=cols['mag10']),
        pyfits.Column(name='MERR_10_TOT_AB',   format='D', array=cols['merr10']),
        pyfits.Column(name='MAG_4_TOT_AB_org', format='D', array=cols['mag4org']),
        pyfits.Column(name='MERR_4_TOT_AB_org',format='D', array=cols['merr4org']),
        pyfits.Column(name='CLASS_STAR',       format='D', array=cols['class_star']),
        pyfits.Column(name='VECTOR_ASSOC',     format='J', array=cols['assoc']),
    ]
    tbl = pyfits.BinTableHDU.from_columns(fcols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        pyfits.HDUList([prim, tbl]).writeto(path, overwrite=True)


# ── reference download + catalog build ────────────────────────────────────────

def _fetch_reference(base_dir, quadrants, auth):
    """Download refimg.fits + refsexcat.fits per quadrant into base_dir/Reference.
    Returns the set of quadrant keys that got both products."""
    ok = set()
    for q in quadrants:
        field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
        got = True
        for suffix in _REF_PRODUCTS:
            dest = ref_local_path(base_dir, field, fc, ccd, qid, suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = ref_url(field, fc, ccd, qid, suffix)
            try:
                _, status, msg = download_file(url, dest, auth)
            except Exception as exc:
                status, msg = "failed", str(exc)
            if status not in ("ok", "skipped"):
                logger.warning(f"  reference fetch {status} {field:06d}/{fc}/c{ccd:02d}/q{qid} "
                               f"{suffix}: {msg}")
                got = False
                break
        if got:
            ok.add((field, fc, ccd, qid))
    return ok


def _load_ref_catalog(base_dir, field, fc, ccd, qid):
    """Load the rebuilt reference CSV; return (DataFrame with q_mag/q_err/cstar,
    SkyCoord), or None."""
    tag = f"{field:06d}_{fc}_c{ccd:02d}_q{qid}"
    csv = base_dir / "Catalogs" / f"{tag}(REFERENCE)[OBJECTS].csv"
    if not csv.exists():
        logger.warning(f"  reference CSV missing for {tag}")
        return None
    ref = pd.read_csv(csv)
    for c in ('ALPHAWIN_J2000', 'DELTAWIN_J2000', 'MAG_APER_4px',
              'MAGERR_APER_4px', 'MAGZP_REF', 'CLASS_STAR'):
        if c not in ref.columns:
            logger.warning(f"  reference CSV {tag} lacks column {c}")
            return None
    ref = ref.reset_index(drop=True)
    ref['q_mag'] = (pd.to_numeric(ref['MAG_APER_4px'], errors='coerce')
                    + pd.to_numeric(ref['MAGZP_REF'], errors='coerce'))
    ref['q_err'] = pd.to_numeric(ref['MAGERR_APER_4px'], errors='coerce')
    ref['cstar'] = pd.to_numeric(ref['CLASS_STAR'], errors='coerce')
    coord = SkyCoord(
        ra=pd.to_numeric(ref['ALPHAWIN_J2000'], errors='coerce').values * u.deg,
        dec=pd.to_numeric(ref['DELTAWIN_J2000'], errors='coerce').values * u.deg)
    return ref, coord


# ── main per-file driver ──────────────────────────────────────────────────────

def _cal_quad_dir(base, kind, field, fc, ccd, qid):
    return base / kind / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid)


def _run_pass(base_dir, quadrants, df, refs, obj_match, ff_map, vet_masks,
              poly_degree, faint_err_max, suffix="",
              pos_ra='ALPHAWIN_REF', pos_dec='DELTAWIN_REF'):
    """Write _cal.fits (and pass-1 residual NPZ when ff_map is empty) for every
    quadrant/epoch. Returns nothing; results land on disk for the pipeline steps.
    `pos_ra`/`pos_dec` are the per-epoch positions used for the spatial fit and
    written to _cal.fits — reference positions in ref-pos mode, the science
    centroids (ALPHA_SCI/DELTA_SCI) in sci-pos mode."""
    pass1 = not ff_map
    for q in quadrants:
        field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
        key = (field, fc, ccd, qid)
        if key not in obj_match:
            continue
        cal_dir = _cal_quad_dir(base_dir, f"Calibrated{suffix}", field, fc, ccd, qid)
        cal_dir.mkdir(parents=True, exist_ok=True)
        for old in cal_dir.glob("*_cal.fits"):
            old.unlink()
        resid_dir = _cal_quad_dir(base_dir, f"FlatfieldResiduals{suffix}", field, fc, ccd, qid)
        if pass1:
            resid_dir.mkdir(parents=True, exist_ok=True)

        ref = refs[key]
        ra0 = float(np.mean(pd.to_numeric(ref['ALPHAWIN_J2000'], errors='coerce')))
        dec0 = float(np.mean(pd.to_numeric(ref['DELTAWIN_J2000'], errors='coerce')))
        flatfield = ff_map.get(key)
        vmask = vet_masks.get(key)  # bool array indexed by ref row, or None

        sub = df[(df['field'] == field) & (df['filtercode'] == fc)
                 & (df['ccdid'] == ccd) & (df['qid'] == qid)]
        omap = obj_match[key]  # object_index -> (csv_idx, q_mag, q_err, cstar)

        for mjd, ep in sub.groupby('OBSMJD'):
            oidx = ep['object_index'].values
            keep = np.array([o in omap for o in oidx])
            if keep.sum() <= 15:
                continue
            ep = ep[keep]
            oidx = oidx[keep]
            csv_idx = np.array([omap[o][0] for o in oidx])
            q_mag   = np.array([omap[o][1] for o in oidx])
            q_err   = np.array([omap[o][2] for o in oidx])
            cstar   = np.array([omap[o][3] for o in oidx])

            magQi = (pd.to_numeric(ep['MAG_4_TOT_AB_org'], errors='coerce').values
                     - pd.to_numeric(ep.get('norm_offset', 0.0), errors='coerce').fillna(0.0).values)
            errQi = pd.to_numeric(ep['MERR_4_TOT_AB_org'], errors='coerce').values
            ra    = pd.to_numeric(ep[pos_ra], errors='coerce').values
            dec   = pd.to_numeric(ep[pos_dec], errors='coerce').values

            is_good = (vmask[csv_idx] if vmask is not None
                       else np.ones(len(csv_idx), dtype=bool))

            out = _calibrate_epoch(magQi, q_mag, q_err, errQi, ra, dec, cstar,
                                   is_good, ra0, dec0, flatfield=flatfield,
                                   poly_degree=poly_degree, faint_err_max=faint_err_max)
            if out is None:
                continue
            Q_cal, Q_err, (ra_c, dec_c, dm4) = out

            m4_old = pd.to_numeric(ep['MAG_4_TOT_AB'], errors='coerce').values
            delta  = Q_cal - m4_old   # aperture-4 shift applied to 3/6/10
            cols = dict(
                ra=ra, dec=dec,
                mag4=Q_cal, merr4=Q_err,
                mag4org=magQi, merr4org=errQi,
                mag3=pd.to_numeric(ep['MAG_3_TOT_AB'], errors='coerce').values + delta,
                merr3=pd.to_numeric(ep['MERR_3_TOT_AB'], errors='coerce').values,
                mag6=pd.to_numeric(ep['MAG_6_TOT_AB'], errors='coerce').values + delta,
                merr6=pd.to_numeric(ep['MERR_6_TOT_AB'], errors='coerce').values,
                mag10=pd.to_numeric(ep['MAG_10_TOT_AB'], errors='coerce').values + delta,
                merr10=pd.to_numeric(ep['MERR_10_TOT_AB'], errors='coerce').values,
                class_star=cstar,
                assoc=(csv_idx + 1).astype(np.int32),
            )
            hdr_vals = {k: (ep[k].iloc[0] if k in ep.columns else np.nan)
                        for k in _CAL_HDR_KEYS}
            _write_cal_fits(cal_dir / f"{float(mjd):.6f}_cal.fits", hdr_vals, cols)

            if pass1:
                np.savez(str(resid_dir / f"{float(mjd):.6f}_resid.npz"),
                         ra_4=ra_c, dec_4=dec_c, dm_4=dm4)


def _load_vet_masks(base_dir, quadrants, refs, refs_coord):
    """Read vet_calib_stars.fits per quad; return {key: bool array over ref rows}."""
    masks = {}
    for q in quadrants:
        field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
        key = (field, fc, ccd, qid)
        if key not in refs:
            continue
        vf = _cal_quad_dir(base_dir, "Calibrated", field, fc, ccd, qid) / "vet_calib_stars.fits"
        if not vf.exists():
            continue
        try:
            with pyfits.open(vf) as h:
                vd = h[1].data
                vcoord = SkyCoord(ra=vd['ALPHAWIN_J2000'] * u.deg,
                                  dec=vd['DELTAWIN_J2000'] * u.deg)
                vgood = np.asarray(vd['IS_GOOD'], dtype=bool)
            ref = refs[key]
            idx, sep, _ = vcoord.match_to_catalog_sky(refs_coord[key])
            mask = np.ones(len(ref), dtype=bool)
            m = sep.arcsec < 1.0
            mask[idx[m]] = vgood[m]
            masks[key] = mask
        except Exception as exc:
            logger.warning(f"  could not load vet mask {field:06d}/{fc}/c{ccd:02d}/q{qid}: {exc}")
    return masks


def _seed_vet_input(base_dir, quadrants, suffix):
    """step_vet (and vet_calibration_stars.py) read the ref-named
    lightcurves.parquet; for a sci-position run, copy the sci light curves to that
    name so variable calibrators are flagged from the sci data being calibrated."""
    for q in quadrants:
        d = (base_dir / "LightCurves" / f"{q['field']:06d}" / q['filtercode']
             / f"ccd{q['ccdid']:02d}" / f"q{q['qid']}")
        src = d / f"lightcurves{suffix}.parquet"
        if src.exists():
            shutil.copy2(src, d / "lightcurves.parquet")


def recalibrate_file(path, args, auth):
    path = Path(path)
    logger.info(f"═══ {path.name} ═══")

    # Science-position products live in parallel _sci trees and are photometered at
    # the per-epoch science centroids (ALPHA_SCI/DELTA_SCI), which drive the spatial
    # fit and the output positions; ref-pos uses the fixed reference positions.
    is_sci = "_sci" in path.stem
    suffix = "_sci" if is_sci else ""
    pos_ra, pos_dec = ('ALPHA_SCI', 'DELTA_SCI') if is_sci else ('ALPHAWIN_REF', 'DELTAWIN_REF')

    df = pd.read_parquet(path)
    need = {'MAG_4_TOT_AB_org', 'MERR_4_TOT_AB_org', 'MAG_4_TOT_AB',
            'ALPHAWIN_REF', 'DELTAWIN_REF', 'object_index', 'OBSMJD', pos_ra, pos_dec}
    missing = need - set(df.columns)
    if missing:
        logger.warning(f"SKIP {path.name}: missing columns {sorted(missing)}")
        return False

    # Quadrant identity: merged parquets carry field/filtercode/ccdid/qid as
    # columns; per-quadrant (non-merged) parquets carry them in parquet metadata
    # and are single-quadrant (no merge, no norm_offset).
    merged_input = {'field', 'filtercode', 'ccdid', 'qid'} <= set(df.columns)
    if not merged_input:
        meta = pq.read_schema(path).metadata or {}
        try:
            df['field']      = int(meta[b'field'].decode())
            df['filtercode'] = meta[b'filtercode'].decode()
            df['ccdid']      = int(meta[b'ccdid'].decode())
            df['qid']        = int(meta[b'qid'].decode())
        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning(f"SKIP {path.name}: no quadrant columns and quadrant "
                           f"metadata unreadable ({exc})")
            return False
    if 'norm_offset' not in df.columns:
        df['norm_offset'] = np.float32(0.0)

    quadrants = [dict(field=int(f), filtercode=str(fc), ccdid=int(c), qid=int(qd))
                 for (f, fc, c, qd) in
                 df[['field', 'filtercode', 'ccdid', 'qid']].drop_duplicates().values]
    logger.info(f"  {len(quadrants)} quadrant(s), {len(df):,} rows")

    base_dir = Path(tempfile.mkdtemp(prefix="recal_"))
    try:
        # 1. reference products + catalogs
        good = _fetch_reference(base_dir, quadrants, auth)
        quadrants = [q for q in quadrants
                     if (q['field'], q['filtercode'], q['ccdid'], q['qid']) in good]
        if not quadrants:
            logger.warning(f"SKIP {path.name}: no reference products retrieved")
            return False
        (base_dir / "Catalogs").mkdir(parents=True, exist_ok=True)
        make_catalog(str(base_dir / "Catalogs"), str(base_dir / "Reference"))

        # 2. load reference catalogs + position-match sources -> csv row + q_mag
        refs, refs_coord, obj_match = {}, {}, {}
        for q in quadrants:
            field, fc, ccd, qid = q['field'], q['filtercode'], q['ccdid'], q['qid']
            key = (field, fc, ccd, qid)
            loaded = _load_ref_catalog(base_dir, field, fc, ccd, qid)
            if loaded is None:
                continue
            ref, ref_coord = loaded
            refs[key] = ref
            refs_coord[key] = ref_coord
            sub = df[(df['field'] == field) & (df['filtercode'] == fc)
                     & (df['ccdid'] == ccd) & (df['qid'] == qid)]
            uo = sub.drop_duplicates('object_index')
            scoord = SkyCoord(ra=pd.to_numeric(uo['ALPHAWIN_REF'], errors='coerce').values * u.deg,
                              dec=pd.to_numeric(uo['DELTAWIN_REF'], errors='coerce').values * u.deg)
            idx, sep, _ = scoord.match_to_catalog_sky(ref_coord)
            omap = {}
            for j, oi in enumerate(uo['object_index'].values):
                if sep.arcsec[j] < args.match_tol:
                    ci = int(idx[j])
                    omap[int(oi)] = (ci, float(ref['q_mag'].iloc[ci]),
                                     float(ref['q_err'].iloc[ci]), float(ref['cstar'].iloc[ci]))
            obj_match[key] = omap
            logger.info(f"  {field:06d}/{fc}/c{ccd:02d}/q{qid}: matched "
                        f"{len(omap)}/{len(uo)} sources to reference")
        quadrants = [q for q in quadrants
                     if (q['field'], q['filtercode'], q['ccdid'], q['qid']) in refs]

        # 3. PASS 1 : calibrate (no flatfield) -> lightcurves -> vet
        _run_pass(base_dir, quadrants, df, refs, obj_match,
                  ff_map={}, vet_masks={}, suffix=suffix,
                  pos_ra=pos_ra, pos_dec=pos_dec,
                  poly_degree=args.poly_degree, faint_err_max=args.faint_err_max)
        step_lightcurves(base_dir, quadrants, force=True, use_calibrated=True, suffix=suffix)
        # vet reads the ref-named lightcurves.parquet; for sci, seed it from the sci
        # light curves so variable calibrators are flagged from the same data.
        if is_sci:
            _seed_vet_input(base_dir, quadrants, suffix)
        step_vet(base_dir, quadrants)

        # 4. build flatfield, load vetting, PASS 2 : recalibrate -> lightcurves
        ff_map = step_build_flatfield(base_dir, quadrants, nbins=args.ff_bins,
                                      min_count=args.ff_min_count,
                                      edge_split=args.ff_edge_split, suffix=suffix)
        vet_masks = _load_vet_masks(base_dir, quadrants, refs, refs_coord)
        _run_pass(base_dir, quadrants, df, refs, obj_match,
                  ff_map=ff_map, vet_masks=vet_masks, suffix=suffix,
                  pos_ra=pos_ra, pos_dec=pos_dec,
                  poly_degree=args.poly_degree, faint_err_max=args.faint_err_max)
        step_lightcurves(base_dir, quadrants, force=True, use_calibrated=True, suffix=suffix)

        out_dir = Path(args.outdir) if args.outdir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}{args.suffix}.parquet"

        if merged_input:
            # 5. merge (needs a target tag only to name its output dir)
            tra = float(np.nanmedian(pd.to_numeric(df['ALPHAWIN_REF'], errors='coerce')))
            tdec = float(np.nanmedian(pd.to_numeric(df['DELTAWIN_REF'], errors='coerce')))
            step_merge(base_dir, quadrants, force=True, target_ra=tra, target_dec=tdec,
                       suffix=suffix)
            # 6. collect output (merged if >=2 quads/band, else single per-quad LC)
            out_df = _collect_output(base_dir, quadrants, suffix)
            if out_df is None or out_df.empty:
                logger.warning(f"FAIL {path.name}: no recalibrated output produced")
                return False
            out_df.to_parquet(out_path, index=False)
            logger.info(f"  → {out_path}  ({len(out_df):,} rows, "
                        f"{out_df['object_index'].nunique()} objects)")
        else:
            # Non-merged (single-quadrant) input: copy the per-quadrant lightcurve
            # verbatim, preserving its non-merged schema + parquet metadata (quad
            # identity, MAGZP_REF). MAG_4_REF is added by step_lightcurves.
            q = quadrants[0]
            lc = (base_dir / "LightCurves" / f"{q['field']:06d}" / q['filtercode']
                  / f"ccd{q['ccdid']:02d}" / f"q{q['qid']}" / f"lightcurves{suffix}.parquet")
            if not lc.exists():
                logger.warning(f"FAIL {path.name}: no recalibrated lightcurve produced")
                return False
            shutil.copy2(lc, out_path)
            logger.info(f"  → {out_path}  ({pq.read_metadata(out_path).num_rows:,} rows)")
        return True
    finally:
        if args.keep_temp:
            logger.info(f"  (kept temp dir {base_dir})")
        else:
            shutil.rmtree(base_dir, ignore_errors=True)


def _collect_output(base_dir, quadrants, suffix=""):
    """Return the merged parquet if present, else the single-quadrant lightcurve
    with merge-schema columns added."""
    merged = sorted((base_dir / "LightCurves" / "merged").rglob("lightcurves_merged.parquet"))
    frames = [pd.read_parquet(m) for m in merged]

    merged_bands = set()
    for m in merged:
        merged_bands.add(m.parent.name)  # {band}{suffix}

    # quadrants whose band was not merged (single-quadrant bands)
    for q in quadrants:
        if f"{q['filtercode']}{suffix}" in merged_bands:
            continue
        lc = (base_dir / "LightCurves" / f"{q['field']:06d}" / q['filtercode']
              / f"ccd{q['ccdid']:02d}" / f"q{q['qid']}" / f"lightcurves{suffix}.parquet")
        if not lc.exists():
            continue
        d = pd.read_parquet(lc)
        d['field'] = q['field']; d['filtercode'] = q['filtercode']
        d['ccdid'] = q['ccdid']; d['qid'] = q['qid']
        if 'norm_offset' not in d.columns:
            d['norm_offset'] = np.float32(0.0)
        frames.append(d)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquets", nargs="*", help="merged parquet files to recalibrate")
    p.add_argument("--list", help="text file with one parquet path per line")
    p.add_argument("--outdir", help="output directory (default: alongside each input)")
    p.add_argument("--suffix", default="_recal", help="output filename suffix (default: _recal)")
    p.add_argument("--match-tol", type=float, default=0.5,
                   help="source↔reference match radius, arcsec (default: 0.5)")
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--faint-err-max", type=float, default=0.5)
    p.add_argument("--ff-bins", type=int, default=20)
    p.add_argument("--ff-min-count", type=int, default=50)
    p.add_argument("--ff-edge-split", type=int, default=3)
    p.add_argument("--keep-temp", action="store_true", help="do not delete temp dirs (debug)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    files = list(args.parquets)
    if args.list:
        files += [ln.strip() for ln in Path(args.list).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
    if not files:
        p.error("no parquet files given (positional args or --list)")

    auth = get_auth()  # IRSA creds from ~/.netrc / env

    n_ok = 0
    for f in files:
        try:
            if recalibrate_file(f, args, auth):
                n_ok += 1
        except Exception as exc:
            logger.error(f"ERROR {f}: {exc}", exc_info=args.verbose)
    logger.info(f"\nDone: {n_ok}/{len(files)} recalibrated")


if __name__ == "__main__":
    main()
