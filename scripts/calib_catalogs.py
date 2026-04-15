import numpy as np
import pandas as pd
import astropy.io.fits as pf
import os
from pathlib import Path

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates.matching import match_coordinates_sky

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d


# ── Spatial-correction helpers ────────────────────────────────────────────────

def _poly2d_basis(ra, dec, ra0, dec0, degree):
    """Design matrix for 2-D polynomial in normalised (u, v) = (RA-ra0, Dec-dec0)."""
    u, v = ra - ra0, dec - dec0
    cols = [np.ones_like(u), u, v]
    if degree >= 2:
        cols += [u * u, u * v, v * v]
    if degree >= 3:
        cols += [u**3, u**2 * v, u * v**2, v**3]
    return np.column_stack(cols)


def _fit_poly2d(ra, dec, dm, ra0, dec0, degree):
    """Least-squares 2-D polynomial fit. Returns (coefficients, fitted_values)."""
    A = _poly2d_basis(ra, dec, ra0, dec0, degree)
    coeffs, _, _, _ = np.linalg.lstsq(A, dm, rcond=None)
    return coeffs, A @ coeffs


def _apply_flatfield(ra, dec, ff):
    """Bilinear interpolation of a pre-built flatfield grid at (RA, Dec) positions.

    Parameters
    ----------
    ff : dict with keys 'stat' (2-D array, shape [n_ra, n_dec]),
         'ra_edges', 'dec_edges'  (1-D bin-edge arrays)
    Returns correction in magnitude units (subtract from mag to apply).
    """
    ra_cen  = 0.5 * (ff['ra_edges'][:-1]  + ff['ra_edges'][1:])
    dec_cen = 0.5 * (ff['dec_edges'][:-1] + ff['dec_edges'][1:])
    interp  = RegularGridInterpolator(
        (ra_cen, dec_cen), ff['stat'],
        method='linear', bounds_error=False, fill_value=0.0)
    return interp(np.column_stack([ra, dec]))


# ── LDAC reader ───────────────────────────────────────────────────────────────

_HDR_KEYS = ['OBSMJD', 'AIRMASS', 'NMATCHES', 'MAGZP_DIF', 'MAGZPRMS_DIF',
             'CLRCOEFF', 'SATURATE', 'INFOBITS_DIF', 'SEEING', 'MAGLIM']


def _read_ldac(catalog_path):
    """Read SExtractor LDAC FITS file. Returns (header_dict, astropy Table)."""
    hdul = pf.open(catalog_path)
    data = {}
    for line in hdul[1].data[0][0]:
        key = line[0:line.find('=')].strip()
        if key in _HDR_KEYS:
            data[key] = line[line.find('=') + 1: line.find('/')].strip()
        if key.startswith("INFOBITS"):
            data['INFOBITS_DIF'] = line[line.find('=') + 1: line.find('/')].strip()
        if key.startswith("MAGZPRMS"):
            data['MAGZPRMS_DIF'] = line[line.find('=') + 1: line.find('/')].strip()
        elif key.startswith("MAGZP") and not key.startswith("MAGZPUNC"):
            data['MAGZP_DIF'] = line[line.find('=') + 1: line.find('/')].strip()
    table = Table(hdul[2].data)
    hdul.close()
    return data, table


# ── Main calibration entry point ──────────────────────────────────────────────

def calib_catalog(ref_catalog, input_catalog, output_catalog, img_kind, vet_catalog=None,
                  poly_degree=2, flatfield=None,
                  target_ra=None, target_dec=None,
                  residuals_out=None):

    table_ref = pd.read_csv(ref_catalog)

    raref  = table_ref['ALPHAWIN_J2000'].values
    decref = table_ref['DELTAWIN_J2000'].values

    # Load vetting mask if provided (bad calibration stars flagged by vet_calibration_stars.py)
    is_good_calib = np.ones(len(table_ref), dtype=bool)
    if vet_catalog is not None:
        if isinstance(vet_catalog, str):
            vet_catalog = Path(vet_catalog)
        if os.path.isfile(str(vet_catalog)):
            try:
                vet_hdul = pf.open(str(vet_catalog))
                vet_data = vet_hdul[1].data
                vet_ra   = vet_data['ALPHAWIN_J2000']
                vet_dec  = vet_data['DELTAWIN_J2000']
                vet_good = vet_data['IS_GOOD']
                vet_hdul.close()

                cat_ref_vet = SkyCoord(ra=raref, dec=decref, unit='deg')
                cat_vet     = SkyCoord(ra=vet_ra, dec=vet_dec, unit='deg')
                idx, sep, _ = match_coordinates_sky(cat_vet, cat_ref_vet, nthneighbor=1)
                matched_vet = sep.arcsec < 3.0
                is_good_calib[idx[matched_vet]] = vet_good[matched_vet]
                print(f"  Vet catalog: {np.sum(~is_good_calib)} bad calibration stars masked")
            except Exception as e:
                print(f"Warning: could not load vetting catalog {vet_catalog}: {e}")

    cat_coord = SkyCoord(ra=raref, dec=decref, unit='deg')

    ref_zp = table_ref['MAGZP_REF'].values
    mag_ref_tot     = [[], [], [], []]
    mag_ref_tot_err = [[], [], [], []]
    for k, px in enumerate(['3px', '4px', '6px', '10px']):
        mag_ref_tot[k]     = table_ref[f'MAG_APER_{px}'].values  + ref_zp
        mag_ref_tot_err[k] = table_ref[f'MAGERR_APER_{px}'].values

    fit_fun = lambda x, n, m: m * x + n
    p0 = [0, 0]

    data, table = _read_ldac(input_catalog)

    alpha = table['ALPHAWIN_J2000']
    delta = table['DELTAWIN_J2000']
    alp   = table['ALPHA_J2000']
    dlt   = table['DELTA_J2000']

    seeing      = data['SEEING']
    saturate    = data['SATURATE']
    airmass     = data['AIRMASS']
    maglim      = data['MAGLIM']
    nmatches    = data['NMATCHES']
    clrcoeff    = data['CLRCOEFF']
    magzp_dif   = data['MAGZP_DIF']
    magzprms_dif = data['MAGZPRMS_DIF']
    infobits_dif = data['INFOBITS_DIF']
    mjd         = data['OBSMJD']

    flags = table['FLAGS']
    if img_kind.casefold() == 'sim':
        for i in range(len(flags)):
            flags[i] = 0

    clas = table_ref['CLASS_STAR'].values

    flux_dif     = table['FLUX_APER']
    flux_dif_err = table['FLUXERR_APER']

    # Match reference catalog to SExtractor detections
    img_coord = SkyCoord(ra=alpha, dec=delta, unit='deg')
    ind, ang, _ = match_coordinates_sky(cat_coord, img_coord, nthneighbor=1)
    ang0 = np.array(ang)
    n    = np.where(ang0 < 0.00083333)[0]   # 3.0 arcsec

    # Reference frame centre for polynomial normalisation
    ra0  = float(np.mean(raref))
    dec0 = float(np.mean(decref))

    # Find target in matched-source array (same position for all apertures)
    tgt_in_matched = -1
    if target_ra is not None and target_dec is not None and len(n) > 0:
        _pos0   = ind[n]
        _ra_m   = np.array(alpha[_pos0])
        _dec_m  = np.array(delta[_pos0])
        _sep_sq = (_ra_m - target_ra)**2 + (_dec_m - target_dec)**2
        _j = int(np.argmin(_sep_sq))
        if np.sqrt(_sep_sq[_j]) < 3.0 / 3600.0:
            tgt_in_matched = _j

    # Per-step RMS accumulators (recorded for k=1, primary aperture)
    _nc_rms0 = _nc_rms1 = _nc_rms2 = _nc_rmsfc = _nc_rms3 = _nc_rms4 = np.nan
    _tgt_mraw = _tgt_dclin = _tgt_dcpoly = _tgt_dcff = np.nan
    _calib_n  = _calib_m = np.nan

    flux_ref_tot = [[], [], [], []]
    flux_tot     = [[], [], [], []]
    Q_cal        = [[], [], [], []]
    Q_err        = [[], [], [], []]
    flux_ab      = [[], [], [], []]
    fluxerr_ab   = [[], [], [], []]
    magQi        = [[], [], [], []]
    errmagQi     = [[], [], [], []]

    # Intermediate-stage snapshot arrays (k=1 only)
    _ra_c_pre = _dec_c_pre = None
    _dm_st0   = _dm_st1 = _dm_st2 = None

    write = False

    # ── Part 1: aperture correction from reference catalog (4px → 6px) ────────
    # Uses calibration stars in the refcat; ref_zp cancels in the difference so
    # this is purely MAG_APER_4px − MAG_APER_6px on the reference image PSF.
    _apcorr_4_6 = 0.0
    _ac_mask = (
        (clas >= 0.7) &
        (mag_ref_tot[1] > 14.) & (mag_ref_tot[1] < 19.0) &
        (mag_ref_tot_err[1] < 0.3) &
        is_good_calib
    )
    if _ac_mask.sum() >= 5:
        _apcorr_4_6 = float(np.median(
            mag_ref_tot[1][_ac_mask] - mag_ref_tot[2][_ac_mask]))
        print(f"  AperCorr 4→6px: {_apcorr_4_6*1000:.1f} mmag  ({_ac_mask.sum()} stars)")

    for k in range(4):
        if len(n) == 0:
            continue

        pos       = ind[n]
        flags_tmp = flags[pos]
        flagsfin  = flags_tmp
        alphafin  = alpha[pos]
        deltafin  = delta[pos]
        class_star = clas[n]
        alpfin    = alp[pos]
        dltfin    = dlt[pos]

        flux_dif_tmp     = flux_dif[:, k][pos]
        flux_dif_err_tmp = flux_dif_err[:, k][pos]

        q_mag = mag_ref_tot[k][n]
        q_err = mag_ref_tot_err[k][n]

        flux_ref_tot[k] = 10.0**(0.4 * (np.float64(magzp_dif) - np.float64(q_mag)))
        flux_tot[k]     = np.float64(flux_ref_tot[k]) + np.float64(flux_dif_tmp)

        maginst  = np.float64(magzp_dif) - 2.5 * np.log10(flux_tot[k])
        # ── Part 2: apply aperture correction (4px aperture only) ─────────────
        if k == 1:
            maginst = maginst - _apcorr_4_6
        magQi[k] = maginst

        errinst    = 1.0857 / (flux_tot[k] / flux_dif_err_tmp)
        errmagQi[k] = errinst

        # Save full matched arrays before calibrator selection (needed for faint correction)
        maginst_all  = maginst.copy()
        errinst_all  = errinst.copy()
        q_mag_all    = q_mag.copy()

        # ── Calibrator selection ────────────────────────────────────────────
        good_ref_mask = is_good_calib[n]
        fn = np.where(
            (class_star >= 0.7) & (flags_tmp == 0) &
            (q_mag > 14.) & (q_mag < 19.0) & (q_err < 0.3) &
            (maginst < 19.0) & (maginst > 14.) & (errinst < 0.3) &
            good_ref_mask
        )
        maginst = maginst[fn]
        ra_c    = alphafin[fn]
        dec_c   = deltafin[fn]

        if len(maginst) <= 15:
            continue

        errinst = errinst[fn]
        q_mag   = q_mag[fn]
        q_err   = q_err[fn]
        sigma   = np.sqrt((errinst**2) + (q_err**2))
        errf    = sigma
        diff    = maginst - q_mag

        _nc_rms0_k = float(np.std(diff))
        if k == 1:
            _ra_c_pre  = ra_c.copy()
            _dec_c_pre = dec_c.copy()
            _dm_st0    = diff.copy()

        # ── Step 1: initial linear fit ──────────────────────────────────────
        coefficients, pcov = curve_fit(fit_fun, maginst, diff, p0=p0,
                                       sigma=errf, absolute_sigma=True)
        pvar    = np.diagonal(pcov)
        pstd    = np.sqrt(pvar)
        fit     = fit_fun(maginst, *coefficients)
        var_fit = (maginst * pstd[1])**2 + pstd[0]**2

        res1       = np.abs(diff - fit)
        rms        = np.sum((diff - fit)**2) / len(maginst)
        _nc_rms1_k = float(np.sqrt(rms))
        if k == 1:
            _dm_st1 = (diff - fit).copy()

        # ── Step 2: 3σ iterative rejection ─────────────────────────────────
        l1 = np.where((res1 / (rms**0.5)) <= 3)
        r1 = np.where((res1 / (rms**0.5)) > 3)

        diff2    = diff
        diff     = diff[l1]
        diff_rej = diff2[r1]
        sigma    = sigma[l1]
        errf     = errf[l1]
        maginst  = maginst[l1]
        q_mag    = q_mag[l1]
        ra_c     = ra_c[l1]
        dec_c    = dec_c[l1]

        if len(diff) > 10:
            while len(diff) > 10:
                if len(diff_rej) == 0:
                    break
                coefficients, pcov = curve_fit(fit_fun, maginst, diff, p0=p0,
                                               sigma=errf, absolute_sigma=True)
                pvar    = np.diagonal(pcov)
                pstd    = np.sqrt(pvar)
                fit     = fit_fun(maginst, *coefficients)
                var_fit = (maginst * pstd[1])**2 + pstd[0]**2

                res2     = np.abs(diff - fit)
                rms      = np.sum((diff - fit)**2) / len(maginst)
                l2       = np.where((res2 / (rms**0.5)) <= 3)
                r2       = np.where((res2 / (rms**0.5)) > 3)
                diff2    = diff
                diff     = diff[l2]
                diff_rej = diff2[r2]
                sigma    = sigma[l2]
                errf     = errf[l2]
                maginst  = maginst[l2]
                q_mag    = q_mag[l2]
                ra_c     = ra_c[l2]
                dec_c    = dec_c[l2]

            coefficients, pcov = curve_fit(fit_fun, maginst, diff, p0=p0,
                                           sigma=errf, absolute_sigma=True)
            pvar    = np.diagonal(pcov)
            pstd    = np.sqrt(pvar)
            fit     = fit_fun(maginst, *coefficients)
            var_fit = (maginst * pstd[1])**2 + pstd[0]**2
            final_fit  = fit_fun(magQi[k], *coefficients)
            _nc_rms2_k = float(np.std(diff - fit))
            if k == 1:
                _dm_st2 = (diff - fit).copy()

            # Per-bin RMS for error estimation (7 bins, 14–20 mag)
            _bin_edges   = [14, 15.5, 17, 17.5, 18, 18.5, 19, 19.5]
            rms_per_bin        = []
            median_mag_per_bin = []
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

            Q_cal[k] = magQi[k] - final_fit

            # ── Step 3: faint-source per-bin smoothed correction ───────────
            # Residuals for all matched sources after the linear ZP fit.
            # Bin in 0.5-mag steps (18.5–22), take the median per bin (3σ-clipped),
            # interpolate across empty bins, Gaussian-smooth σ=0.2 mag (σ_bins=0.4).
            # Apply per-source by interpolating the smoothed curve at each source mag.
            # Correction is forced to 0 at the bright end (mag < 18.5).
            _FC_EDGES   = np.arange(18.5, 22.1, 0.5)
            _FC_CENTERS = 0.5 * (_FC_EDGES[:-1] + _FC_EDGES[1:])

            residual_all = maginst_all - q_mag_all - fit_fun(maginst_all, *coefficients)
            faint_nc_mask = (
                (maginst_all > 19.0) & (maginst_all < 21.0) &
                (errinst_all < 0.5) & np.isfinite(residual_all)
            )

            _bin_med = np.full(len(_FC_CENTERS), np.nan)
            _all_fc_mask = (
                (maginst_all >= 18.5) & (maginst_all < 22.0) &
                (errinst_all < 0.5) & np.isfinite(residual_all)
            )
            for _ib, (_lo, _hi) in enumerate(zip(_FC_EDGES[:-1], _FC_EDGES[1:])):
                _bm = _all_fc_mask & (maginst_all >= _lo) & (maginst_all < _hi)
                if _bm.sum() >= 5:
                    _r   = residual_all[_bm]
                    _med = np.nanmedian(_r)
                    _mad = np.nanmedian(np.abs(_r - _med))
                    _gd  = (np.abs(_r - _med) < 3.0 * 1.4826 * _mad
                            if _mad > 0 else np.ones(len(_r), dtype=bool))
                    if _gd.sum() >= 3:
                        _bin_med[_ib] = float(np.nanmedian(_r[_gd]))

            faint_corr_curve = None
            _valid_fc = np.isfinite(_bin_med)
            if _valid_fc.sum() >= 3:
                _filled   = np.interp(_FC_CENTERS, _FC_CENTERS[_valid_fc], _bin_med[_valid_fc])
                _smoothed = gaussian_filter1d(_filled, sigma=0.4, mode='nearest')
                _interp_mags = np.concatenate([[18.5], _FC_CENTERS])
                _interp_corr = np.concatenate([[0.0], _smoothed])
                faint_corr_curve = (_interp_mags, _interp_corr)

            faint_offset = 0.0
            if faint_corr_curve is not None:
                _hdr_bins = (_FC_CENTERS >= 19.0) & (_FC_CENTERS < 21.0) & _valid_fc
                if _hdr_bins.sum() > 0:
                    faint_offset = float(np.nanmean(_bin_med[_hdr_bins]))

            if faint_corr_curve is not None:
                _corr_all = np.interp(magQi[k], faint_corr_curve[0], faint_corr_curve[1],
                                      left=0.0, right=faint_corr_curve[1][-1])
                Q_cal[k] = Q_cal[k] - _corr_all
            # ── end faint-source correction ─────────────────────────────────

            # ── Step 4: 2-D polynomial spatial correction ───────────────────
            _faint_corr_c = (
                np.interp(maginst, faint_corr_curve[0], faint_corr_curve[1],
                          left=0.0, right=faint_corr_curve[1][-1])
                if faint_corr_curve is not None else np.zeros(len(maginst))
            )
            _dm_for_poly = (diff - fit) - _faint_corr_c
            if k == 1:
                _nc_rmsfc_k = float(np.std(_dm_for_poly))

            _poly_corr   = np.zeros(len(alphafin))
            _nc_rms3_k   = _nc_rms2_k
            _poly_fitted = np.zeros(len(ra_c))
            try:
                _poly_coeffs, _poly_fitted = _fit_poly2d(
                    ra_c, dec_c, _dm_for_poly, ra0, dec0, poly_degree)
                _poly_corr = _poly2d_basis(
                    alphafin, deltafin, ra0, dec0, poly_degree) @ _poly_coeffs
                _nc_rms3_k = float(np.std(_dm_for_poly - _poly_fitted))
            except Exception as _pe:
                print(f"  Warning: poly2d fit failed k={k}: {_pe}")
            Q_cal[k] = Q_cal[k] - _poly_corr
            # ── end polynomial correction ───────────────────────────────────

            # ── Step 5: stacked flatfield correction ────────────────────────
            _ff_corr   = np.zeros(len(alphafin))
            _nc_rms4_k = np.nan
            if flatfield is not None:
                try:
                    _ff_corr   = _apply_flatfield(alphafin, deltafin, flatfield)
                    _ff_at_c   = _apply_flatfield(ra_c, dec_c, flatfield)
                    _nc_rms4_k = float(np.std(_dm_for_poly - _poly_fitted - _ff_at_c))
                except Exception as _fe:
                    print(f"  Warning: flatfield apply failed k={k}: {_fe}")
                Q_cal[k] = Q_cal[k] - _ff_corr
            # ── end flatfield correction ────────────────────────────────────

            # ── Accumulate diagnostics for k=1 (primary aperture) ──────────
            if k == 1:
                _calib_n  = float(coefficients[0])
                _calib_m  = float(coefficients[1])
                _nc_rms0  = _nc_rms0_k * 1000
                _nc_rms1  = _nc_rms1_k * 1000
                _nc_rms2  = _nc_rms2_k * 1000
                _nc_rmsfc = _nc_rmsfc_k * 1000 if np.isfinite(_nc_rmsfc_k) else np.nan
                _nc_rms3  = _nc_rms3_k * 1000
                _nc_rms4  = _nc_rms4_k * 1000 if np.isfinite(_nc_rms4_k) else np.nan
                if tgt_in_matched >= 0:
                    _tgt_mraw    = float(magQi[k][tgt_in_matched])
                    _tgt_faint_c = (
                        float(np.interp(_tgt_mraw, faint_corr_curve[0], faint_corr_curve[1],
                                        left=0.0, right=faint_corr_curve[1][-1]))
                        if faint_corr_curve is not None else 0.0
                    )
                    _tgt_dclin  = float(final_fit[tgt_in_matched] + _tgt_faint_c) * 1000
                    _tgt_dcpoly = float(_poly_corr[tgt_in_matched]) * 1000
                    _tgt_dcff   = (float(_ff_corr[tgt_in_matched]) * 1000
                                   if flatfield is not None else np.nan)
                if residuals_out is not None:
                    _dm_st3  = _dm_for_poly.astype(float)
                    _dm_st4  = (_dm_for_poly - _poly_fitted).astype(float)
                    _ff_save = (_ff_at_c if flatfield is not None
                                else np.zeros(len(ra_c), dtype=float))
                    _dm_st5  = (_dm_st4 - _ff_save).astype(float)
                    np.savez(str(residuals_out),
                             ra_0=_ra_c_pre,  dec_0=_dec_c_pre,
                             dm_0=_dm_st0.astype(float),
                             ra_1=_ra_c_pre,  dec_1=_dec_c_pre,
                             dm_1=_dm_st1.astype(float),
                             ra_2=ra_c,        dec_2=dec_c,
                             dm_2=_dm_st2.astype(float),
                             ra_3=ra_c,        dec_3=dec_c,
                             dm_3=_dm_st3,
                             ra_4=ra_c,        dec_4=dec_c,
                             dm_4=_dm_st4,
                             ra_5=ra_c,        dec_5=dec_c,
                             dm_5=_dm_st5,
                             ra_all=alphafin.astype(float),
                             dec_all=deltafin.astype(float),
                             dm_all_pre=(maginst_all - q_mag_all).astype(float),
                             dm_all_post=(Q_cal[k] - q_mag_all).astype(float))

            interpolation = np.interp(Q_cal[k], median_mag_per_bin, rms_per_bin)
            Q_err[k]      = np.array([max(i, j) for i, j in zip(interpolation, errmagQi[k])])

            flux_ab[k]    = 10.0**(-0.4 * (np.float64(Q_cal[k]) + 48.6))
            fluxerr_ab[k] = (0.4 * np.log(10)
                             * 10.0**(-0.4 * (Q_cal[k] + 48.6)) * Q_err[k])

            rms     = np.sum((diff - fit)**2) / len(maginst)
            chi_red = (np.sum(((diff - fit)**2) / (errf**2 + var_fit)) / (len(maginst) - 2))
            write   = True

    # ── Write output catalog ──────────────────────────────────────────────────
    if not write:
        print(f'  {input_catalog}: too few calibration stars — skipped')
        return

    cols = [
        pf.Column(name='ALPHAWIN_J2000',    format='D', array=alphafin),
        pf.Column(name='DELTAWIN_J2000',    format='D', array=deltafin),
        pf.Column(name='MAG_3_TOT_AB',      format='D', array=Q_cal[0]),
        pf.Column(name='MERR_3_TOT_AB',     format='D', array=Q_err[0]),
        pf.Column(name='FLUX_3_TOT_AB',     format='D', array=flux_ab[0]),
        pf.Column(name='FERR_3_TOT_AB',     format='D', array=fluxerr_ab[0]),
        pf.Column(name='MAG_4_TOT_AB',      format='D', array=Q_cal[1]),
        pf.Column(name='MERR_4_TOT_AB',     format='D', array=Q_err[1]),
        pf.Column(name='FLUX_4_TOT_AB',     format='D', array=flux_ab[1]),
        pf.Column(name='FERR_4_TOT_AB',     format='D', array=fluxerr_ab[1]),
        pf.Column(name='MAG_6_TOT_AB',      format='D', array=Q_cal[2]),
        pf.Column(name='MERR_6_TOT_AB',     format='D', array=Q_err[2]),
        pf.Column(name='FLUX_6_TOT_AB',     format='D', array=flux_ab[2]),
        pf.Column(name='FERR_6_TOT_AB',     format='D', array=fluxerr_ab[2]),
        pf.Column(name='MAG_10_TOT_AB',     format='D', array=Q_cal[3]),
        pf.Column(name='MERR_10_TOT_AB',    format='D', array=Q_err[3]),
        pf.Column(name='FLUX_10_TOT_AB',    format='D', array=flux_ab[3]),
        pf.Column(name='FERR_10_TOT_AB',    format='D', array=fluxerr_ab[3]),
        pf.Column(name='FLAGS',             format='D', array=flagsfin),
        pf.Column(name='CLASS_STAR',        format='D', array=class_star),
        pf.Column(name='MAG_4_TOT_AB_org',  format='D', array=magQi[1]),
        pf.Column(name='MERR_4_TOT_AB_org', format='D', array=errmagQi[1]),
        pf.Column(name='ALPHA_J2000',       format='D', array=alpfin),
        pf.Column(name='DELTA_J2000',       format='D', array=dltfin),
    ]
    pf.BinTableHDU.from_columns(cols).writeto(output_catalog, overwrite=True)

    with pf.open(output_catalog, mode='update') as arch1:
        head = arch1[0].header

        def _hv(v):
            f = float(v)
            return -999.0 if np.isnan(f) else f

        head['fit_rms']      = np.sqrt(rms)
        head['chi_red']      = chi_red
        head['num_stars']    = len(maginst)
        head['NC_N']         = int(np.sum(faint_nc_mask))
        head['CALIB_N']      = _hv(_calib_n)
        head['CALIB_M']      = _hv(_calib_m)
        head['CALIB_ZP']     = _hv(_calib_n + _calib_m * 17.0)
        head['APCORR46']     = _hv(_apcorr_4_6 * 1000)        # mmag, 4px→6px

        head['NC_RMS0']      = _hv(_nc_rms0)
        head['NC_RMS1']      = _hv(_nc_rms1)
        head['NC_RMS2']      = _hv(_nc_rms2)
        head['NC_RMSFC']     = _hv(_nc_rmsfc)
        head['NC_RMS3']      = _hv(_nc_rms3)
        head['NC_RMS4']      = _hv(_nc_rms4)

        head['TGT_MRAW']     = _hv(_tgt_mraw)
        head['TGT_DCLIN']    = _hv(_tgt_dclin)
        head['TGT_DCPOL']    = _hv(_tgt_dcpoly)
        head['TGT_DCFF']     = _hv(_tgt_dcff)

        # Per-bin faint correction curve (7 bins, mmag; -999 if bin had no data)
        if faint_corr_curve is not None:
            _fc_mmag = np.interp(
                [18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75],
                faint_corr_curve[0], faint_corr_curve[1],
                left=0.0, right=faint_corr_curve[1][-1]) * 1000
            for _bi, _bv in enumerate(_fc_mmag):
                head[f'NC_FC_{_bi:02d}'] = float(_bv) if _valid_fc[_bi] else -999.0
        else:
            for _bi in range(7):
                head[f'NC_FC_{_bi:02d}'] = -999.0

        head['SEEING']       = seeing
        head['SATURATE']     = saturate
        head['AIRMASS']      = airmass
        head['MAGLIM']       = maglim
        head['NMATCHES']     = nmatches
        head['CLRCOEFF']     = clrcoeff
        head['MAGZP_DIF']    = magzp_dif
        head['MAGZPRMS_DIF'] = magzprms_dif
        head['INFOBITS_DIF'] = infobits_dif
        head['OBSMJD']       = mjd
