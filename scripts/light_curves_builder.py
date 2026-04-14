import os
import time
import sys

import numpy as np
import pandas as pd
import fastparquet as fp

import astropy.units as u
import astropy.units.quantity
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.stats import chi2


class Apertures:
    sizes = ["3", "4", "6", "10", "AUTO"]


class Headers_from_Reference:
    names = ['ID_REF', 'CLASS_STAR_REF', 'ALPHAWIN_REF',
             'DELTAWIN_REF', 'MAGZP_REF', 'MAGZPRMS_REF', 'INFOBITS_REF', 'FLAG_SE_REF']


class Fluxes_Ref:
    Flux = ['FLUX_X_TOT_AB'.replace("_X_", "_" + size + "_")
           for size in Apertures.sizes]
    Err = ['FERR_X_TOT_AB'.replace("_X_", "_" + size + "_")
           for size in Apertures.sizes]


class Mag_Ref:
    Mag = ['MAG_AP_X_REF'.replace("_X_", "_" + size + "_")
           for size in Apertures.sizes]


class Headers_from_Diference:
    names = ['OBSMJD', 'AIRMASS', 'NMATCHES', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',\
             'SATURATE', 'INFOBITS_DIF', 'SEEING',   'MAGLIM']


class Keys_from_object:
    names = ['ALPHA_OBJ', 'DELTA_OBJ', 'ALPHAWIN_OBJ', 'DELTAWIN_OBJ',
             'DISTANCE', "CLASS_STAR_OBJ", 'FLAG_SE_DIF', 'FLAG_DET']


class Fluxes_Dif:
    Flux = ['FLUX_X_DIF'.replace("_X_", "_" + size + "_")
            for size in Apertures.sizes]
    Err = ['FERR_X_DIF'.replace("_X_", "_" + size + "_")
           for size in Apertures.sizes]


# 'ID_REF', 'RA', 'DEC', 'CLASS_STAR', 'DELTA_T', 'COUNT', ('MEAN_FLUX', 'FLUX_VAR', 'EX_VAR', 'EX_VERR')[5]
class Preliminary_Features:
    names = ['ID_REF', 'RA', 'DEC', 'CLASS_STAR', 'DELTA_T', 'COUNT']
    mean = ['MEAN_X_FLUX'.replace("_X_", "_" + size + "_") for size in Apertures.sizes]
    var = ['VAR_X_FLUX'.replace("_X_", "_" + size + "_") for size in Apertures.sizes]
    pvar = ['PVAR_X_FLUX'.replace("_X_", "_" + size + "_") for size in Apertures.sizes]
    ex_var = ['EXVAR_X_FLUX'.replace("_X_", "_" + size + "_") for size in Apertures.sizes]
    exverr = ['EXVERR_X_FLUX'.replace("_X_", "_" + size + "_") for size in Apertures.sizes]
    class_star = ['DELTA_CS', 'MEAN_CS', 'MEDIAN_CS', 'STD_CS', 'VAR_CS']


class LightCurve:
    # Column order matches the original DataFrame definition so that downstream
    # code that relies on column positions continues to work unchanged.
    _COLUMNS = (  Headers_from_Reference.names + Headers_from_Diference.names
                + Keys_from_object.names + Mag_Ref.Mag
                + Fluxes_Dif.Flux + Fluxes_Dif.Err + Fluxes_Ref.Flux + Fluxes_Ref.Err)

    def __init__(self):
        # Accumulate rows as plain dicts; build the DataFrame once on first access.
        # This avoids the O(n²) copy cost of pd.concat-on-every-epoch.
        self._rows = []
        self._df   = None   # built lazily by the .data property

    @property
    def data(self):
        """Return a DataFrame of all accumulated light-curve points."""
        if self._df is None:
            if self._rows:
                self._df = pd.DataFrame(self._rows, columns=self._COLUMNS)
            else:
                self._df = pd.DataFrame(columns=self._COLUMNS)
        return self._df

    @data.setter
    def data(self, value):
        """Allow external code to replace .data directly (backward compatibility)."""
        self._df   = value
        self._rows = []   # rows list is now stale; discard it

    def add_point(self,
                  # From Reference
                  id_ref: str = None,
                  class_star_ref: np.float32 = np.nan,
                  alphawin_ref: np.float64 = np.nan,
                  deltawin_ref: np.float64 = np.nan,
                  infobits_ref: np.uint32 = np.uint32.max,
                  flag_SE_ref: np.uint8 = np.uint8.max,
                  magzp_ref: np.float32 = np.nan,
                  magzprms_ref: np.float32 = np.nan,
                  flux_ref=[np.nan for _ in Apertures.sizes],
                  fluxerr_ref=[np.nan for _ in Apertures.sizes],
                  mag_ref=[np.nan for _ in Apertures.sizes],
                  # From Sextractor Catalog
                  obsmjd: np.float64 = np.nan,
                  flag_detection: bool = False,
                  alpha_dif: np.float64 = np.nan,
                  delta_dif: np.float64 = np.nan,
                  alphawin_dif: np.float64 = np.nan,
                  deltawin_dif: np.float64 = np.nan,
                  class_star_dif: np.float32 = np.nan,
                  distance: np.float64 = np.nan,
                  seeing: np.float32 = np.nan,
                  saturate: np.float32 = np.nan,
                  airmass: np.float32 = np.nan,
                  maglim: np.float32 = np.nan,
                  nmatches: np.uint32 = np.uint32.min,
                  clrcoeff: np.float32 = np.nan,
                  magzp_dif: np.float32 = np.nan,
                  magzprms_dif: np.float32 = np.nan,
                  infobits_dif: np.uint32 = np.uint32.max,
                  flag_SE_dif: np.uint8 = np.uint8.max,
                  flux_dif=[np.nan for _ in Apertures.sizes],
                  fluxerr_dif=[np.nan for _ in Apertures.sizes]
                  ):

        row = {
                'ID_REF': id_ref,
                'CLASS_STAR_REF': class_star_ref,
                'OBSMJD': obsmjd,
                'FLAG_DET': flag_detection,
                'ALPHA_OBJ': alpha_dif,
                'DELTA_OBJ': delta_dif,
                'ALPHAWIN_OBJ': alphawin_dif,
                'DELTAWIN_OBJ': deltawin_dif,
                'CLASS_STAR_OBJ': class_star_dif,
                'DISTANCE': distance,
                'SEEING': seeing,
                'SATURATE': saturate,
                'AIRMASS': airmass,
                'MAGLIM': maglim,
                'NMATCHES': nmatches,
                'CLRCOEFF': clrcoeff,
                'MAGZP_DIF': magzp_dif,
                'MAGZPRMS_DIF': magzprms_dif,
                'INFOBITS_DIF': infobits_dif,
                'FLAG_SE_DIF': flag_SE_dif,
                'FLUX_3_DIF': flux_dif[0],
                'FLUX_4_DIF': flux_dif[1],
                'FLUX_6_DIF': flux_dif[2],
                'FLUX_10_DIF': flux_dif[3],
                'FLUX_AUTO_DIF': flux_dif[4],
                'FERR_3_DIF': fluxerr_dif[0],
                'FERR_4_DIF': fluxerr_dif[1],
                'FERR_6_DIF': fluxerr_dif[2],
                'FERR_10_DIF': fluxerr_dif[3],
                'FERR_AUTO_DIF': fluxerr_dif[4],
                'ALPHAWIN_REF': alphawin_ref,
                'DELTAWIN_REF': deltawin_ref,
                'INFOBITS_REF': infobits_ref,
                'FLAG_SE_REF': flag_SE_ref,
                'MAGZP_REF': magzp_ref,
                'MAGZPRMS_REF': magzprms_ref,
                'FLUX_3_TOT_AB': flux_ref[0],
                'FLUX_4_TOT_AB': flux_ref[1],
                'FLUX_6_TOT_AB': flux_ref[2],
                'FLUX_10_TOT_AB': flux_ref[3],
                'FLUX_AUTO_TOT_AB': flux_ref[4],
                'FERR_3_TOT_AB': fluxerr_ref[0],
                'FERR_4_TOT_AB': fluxerr_ref[1],
                'FERR_6_TOT_AB': fluxerr_ref[2],
                'FERR_10_TOT_AB': fluxerr_ref[3],
                'FERR_AUTO_TOT_AB': fluxerr_ref[4],
                "MAG_AP_3_REF": mag_ref[0],
                "MAG_AP_4_REF": mag_ref[1],
                "MAG_AP_6_REF": mag_ref[2],
                "MAG_AP_10_REF": mag_ref[3],
                "MAG_AP_AUTO_REF": mag_ref[4],
        }

        # Append to the row list and invalidate any cached DataFrame.
        self._rows.append(row)
        self._df = None

        return


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            for file in files:
                r.append(os.path.join(subdir, file))

    return r


def list_folders(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            r.append(subdir)

    return r


def id_to_subfolder(catalog_id) -> str:
    return catalog_id.replace('_o', "").replace('q', "")\
        .replace('c', "").replace('_', "/")


def id_to_path(catalog_id, catalog_home) -> str:
    # 000468_zg_c03_o_q2
    return catalog_home + '/' + catalog_id[0:3] + '/' + id_to_subfolder(catalog_id)


def obtain_catalog_from_catalog(catalog_path):
    data = {}
    table = []

    if catalog_path.endswith('.cat'):
        for key in Headers_from_Diference.names:
            data = "0"
        table = ascii.read(catalog_path)

    if catalog_path.endswith('.fits'):

        hdul = fits.open(catalog_path)

        if '/Calibrated/' in catalog_path:
            for entry in hdul[0].header:
                if entry in Headers_from_Diference.names:
                    data[entry] = hdul[0].header[entry]

            table = Table(hdul[1].data)
        else:
            for line in hdul[1].data[0][0]:
                key = line[0:line.find('=')].strip()
                if key in Headers_from_Diference.names:
                    data[key] = line[line.find('=') + 1: line.find('/')].strip()
                #Missing infobits and flag
                if key.startswith("INFOBITS"):
                    data['INFOBITS_DIF'] = line[line.find('=') + 1:
                                                line.find('/')].strip()
                if key.startswith("MAGZPRMS"):
                    data['MAGZPRMS_DIF'] = line[line.find('=') + 1:
                                                line.find('/')].strip()
                elif key.startswith("MAGZP") and not key.startswith("MAGZPUNC"):
                    data['MAGZP_DIF'] = line[line.find('=') + 1:
                                                line.find('/')].strip()
            table = Table(hdul[2].data)
        hdul.close()

    return data, table


def build_lightcurve_catalog(reference_catalog_path='', search_catalogs_home='',
                             max_separation: astropy.units.quantity.Quantity = 1.5 * u.arcsec,
                             min_points: int = 10, flag_filter: int = 0, verbose: bool = False, simulated: bool = False):
    """
    Build a flat per-source-per-epoch light curve table.

    Memory strategy: pre-allocate one numpy array per output column and fill
    by epoch slice (row r = epoch_k * n_refs + source_i).  This uses ~8x less
    RAM than accumulating Python dicts inside per-source LightCurve objects,
    because numpy stores floats at 8 bytes each vs ~56 bytes for a boxed Python
    float inside a dict.

    Returns
    -------
    dict with key 'LIGHTCURVES_DF' -> a flat pandas DataFrame with
    n_refs * n_epochs rows.  The 'object_index' column identifies the source
    (0 .. n_refs-1); rows for the same source can be recovered with
    df[df['object_index'] == i].
    """
    cat_id = reference_catalog_path[reference_catalog_path.rfind('/')+1:
                                    reference_catalog_path.rfind('(')]

    objects = pd.read_csv(reference_catalog_path)
    if 'ALPHAWIN_J2000' in objects.columns:
        objects.rename(columns={'ALPHAWIN_J2000': 'RA',
                       'DELTAWIN_J2000': 'DEC'}, inplace=True)

    ref_cat = SkyCoord(ra=objects['RA'], dec=objects['DEC'], unit='deg')
    n_refs  = len(objects)

    fits_list = list_files(id_to_path(cat_id, search_catalogs_home))
    n_epochs  = len(fits_list)
    n_rows    = n_refs * n_epochs

    if n_rows == 0:
        empty_cols = LightCurve._COLUMNS + ['object_index']
        return {'LIGHTCURVES_DF': pd.DataFrame(columns=empty_cols)}

    # ------------------------------------------------------------------
    # Pre-allocate output arrays.
    # Source-static columns are tiled (same value for every epoch of a source).
    # Epoch-specific / detection-specific columns start at NaN / sentinels.
    # ------------------------------------------------------------------
    NaN = np.nan
    UINT32_MAX = np.iinfo(np.uint32).max
    UINT8_MAX  = np.iinfo(np.uint8).max

    def _tile(col, dtype):
        return np.tile(objects[col].values.astype(dtype), n_epochs)

    def _full(val, dtype=np.float64):
        return np.full(n_rows, val, dtype=dtype)

    # Source-static (tiled across all epochs)
    a_id_ref       = (_tile('ID', object)         if 'ID'           in objects.columns
                      else np.full(n_rows, " ", object))
    a_obj_idx      = np.tile(np.arange(n_refs, dtype=np.int32), n_epochs)
    a_cs_ref       = (_tile('CLASS_STAR', np.float32) if 'CLASS_STAR'   in objects.columns
                      else _full(NaN, np.float32))
    a_ra_ref       = _tile('RA',  np.float64)
    a_dec_ref      = _tile('DEC', np.float64)
    a_magzp_ref    = (_tile('MAGZP_REF',    np.float64) if 'MAGZP_REF'    in objects.columns else _full(NaN))
    a_magzprms_ref = (_tile('MAGZPRMS_REF', np.float64) if 'MAGZPRMS_REF' in objects.columns else _full(NaN))
    a_ibits_ref    = (_tile('INFOBITS_REF', np.uint32)  if 'INFOBITS_REF' in objects.columns
                      else _full(UINT32_MAX, np.uint32))
    a_flag_se_ref  = (_tile('FLAG_SE_REF',  np.uint8)   if 'FLAG_SE_REF'  in objects.columns
                      else _full(UINT8_MAX, np.uint8))
    a_mag3_ref     = (_tile('MAG_APER_3px',  np.float64) if 'MAG_APER_3px'  in objects.columns else _full(NaN))
    a_mag4_ref     = (_tile('MAG_APER_4px',  np.float64) if 'MAG_APER_4px'  in objects.columns else _full(NaN))
    a_mag6_ref     = (_tile('MAG_APER_6px',  np.float64) if 'MAG_APER_6px'  in objects.columns else _full(NaN))
    a_mag10_ref    = (_tile('MAG_APER_10px', np.float64) if 'MAG_APER_10px' in objects.columns else _full(NaN))
    a_magA_ref     = (_tile('MAG_AUTO',      np.float64) if 'MAG_AUTO'      in objects.columns else _full(NaN))

    # Epoch-level metadata (filled per epoch block; non-detections revert to sentinels)
    a_obsmjd    = _full(NaN)
    a_airmass   = _full(NaN)
    a_nmatches  = _full(0,          np.uint32)
    a_magzp_dif = _full(NaN)
    a_mzprms_dif= _full(NaN)
    a_clrcoeff  = _full(NaN)
    a_saturate  = _full(NaN)
    a_ibits_dif = _full(0xFFFFFFFF, np.uint32)
    a_seeing    = _full(NaN)
    a_maglim    = _full(NaN)

    # Detection-conditional columns
    a_alpha_obj   = _full(NaN); a_delta_obj   = _full(NaN)
    a_awin_obj    = _full(NaN); a_dwin_obj    = _full(NaN)
    a_distance    = _full(NaN); a_cs_obj      = _full(NaN)
    a_flag_se_dif = _full(UINT8_MAX, np.uint8)
    a_flag_det    = _full(False, bool)

    # Aperture fluxes — difference image (NaN = no detection)
    a_f3d  = _full(NaN); a_f4d  = _full(NaN); a_f6d  = _full(NaN)
    a_f10d = _full(NaN); a_fAd  = _full(NaN)
    a_e3d  = _full(NaN); a_e4d  = _full(NaN); a_e6d  = _full(NaN)
    a_e10d = _full(NaN); a_eAd  = _full(NaN)

    # Aperture fluxes — total AB (ref + diff combined)
    a_f3t  = _full(NaN); a_f4t  = _full(NaN); a_f6t  = _full(NaN)
    a_f10t = _full(NaN); a_fAt  = _full(NaN)
    a_e3t  = _full(NaN); a_e4t  = _full(NaN); a_e6t  = _full(NaN)
    a_e10t = _full(NaN); a_eAt  = _full(NaN)

    # ------------------------------------------------------------------
    # Main loop — one FITS file per epoch
    # ------------------------------------------------------------------
    process = time.time()

    for (k, path) in enumerate(fits_list):
        if verbose:
            print("file ", k + 1, " of ", len(fits_list), " | ",
                  time.time() - process, "elapsed")

        data, table = obtain_catalog_from_catalog(path)

        tmp_cat = SkyCoord(table["ALPHAWIN_J2000"], table["DELTAWIN_J2000"], unit="deg")
        idx_m, dis_m, _ = ref_cat.match_to_catalog_sky(tmp_cat)

        r0 = k * n_refs        # first row index for this epoch
        r1 = r0 + n_refs       # last row index (exclusive)
        det_arr = dis_m < max_separation   # boolean array, length n_refs

        # --- Epoch-level metadata (broadcast to whole epoch block) ---
        def _safe_float(key, default=NaN):
            try:
                return float(data[key])
            except (KeyError, TypeError, ValueError):
                return default

        def _safe_int(key, default=0):
            try:
                return int(float(data[key]))
            except (KeyError, TypeError, ValueError):
                return default

        a_obsmjd[r0:r1]     = _safe_float('OBSMJD')
        a_airmass[r0:r1]    = _safe_float('AIRMASS')
        a_nmatches[r0:r1]   = _safe_int('NMATCHES')
        a_magzp_dif[r0:r1]  = _safe_float('MAGZP_DIF')
        a_mzprms_dif[r0:r1] = _safe_float('MAGZPRMS_DIF')
        a_clrcoeff[r0:r1]   = _safe_float('CLRCOEFF')
        a_saturate[r0:r1]   = _safe_float('SATURATE')
        a_ibits_dif[r0:r1]  = _safe_int('INFOBITS_DIF', 0xFFFFFFFF)

        seeing_val = _safe_float('SEEING')
        maglim_val = _safe_float('MAGLIM')
        magzp_dif_val = _safe_float('MAGZP_DIF')

        # Revert sentinel values for non-detection rows (matching original behaviour)
        nd_idx = r0 + np.where(~det_arr)[0]
        if len(nd_idx):
            a_ibits_dif[nd_idx] = 0xFFFFFFFF
            a_nmatches[nd_idx]  = 0
            a_magzp_dif[nd_idx] = NaN
            a_mzprms_dif[nd_idx]= NaN
            a_clrcoeff[nd_idx]  = NaN

        has_flux_aper = 'FLUX_APER' in table.columns

        # --- Per-source fill ---
        for i in range(n_refs):
            r      = r0 + i
            is_det = bool(det_arr[i])
            a_flag_det[r] = is_det

            if is_det:
                a_seeing[r]    = seeing_val
                a_maglim[r]    = maglim_val
                a_alpha_obj[r] = float(table[idx_m[i]]['ALPHA_J2000'])
                a_delta_obj[r] = float(table[idx_m[i]]['DELTA_J2000'])
                if 'ALPHAWIN_J2000' in table.columns:
                    a_awin_obj[r] = float(table[idx_m[i]]['ALPHAWIN_J2000'])
                if 'DELTAWIN_J2000' in table.columns:
                    a_dwin_obj[r] = float(table[idx_m[i]]['DELTAWIN_J2000'])
                a_cs_obj[r]    = float(table[idx_m[i]]['CLASS_STAR'])
                a_distance[r]  = float(dis_m[i].value)
                a_flag_se_dif[r] = np.uint8(0x00 if simulated else int(table[idx_m[i]]['FLAGS']))

            if has_flux_aper:
                # Raw LDAC catalog: read raw aperture fluxes and compute total AB fluxes.
                if is_det:
                    a_f3d[r]  = float(table[idx_m[i]]['FLUX_APER'][0])
                    a_f4d[r]  = float(table[idx_m[i]]['FLUX_APER'][1])
                    a_f6d[r]  = float(table[idx_m[i]]['FLUX_APER'][2])
                    a_f10d[r] = float(table[idx_m[i]]['FLUX_APER'][3])
                    a_fAd[r]  = float(table[idx_m[i]]['FLUX_AUTO'])
                    a_e3d[r]  = float(table[idx_m[i]]['FLUXERR_APER'][0])
                    a_e4d[r]  = float(table[idx_m[i]]['FLUXERR_APER'][1])
                    a_e6d[r]  = float(table[idx_m[i]]['FLUXERR_APER'][2])
                    a_e10d[r] = float(table[idx_m[i]]['FLUXERR_APER'][3])
                    a_eAd[r]  = float(table[idx_m[i]]['FLUXERR_AUTO'])

                mzp_r = float(a_magzp_ref[r])
                mzp_d = magzp_dif_val
                for (fd, fe, mg, ft_arr, et_arr) in [
                    (a_f3d[r],  a_e3d[r],  a_mag3_ref[r],  a_f3t,  a_e3t),
                    (a_f4d[r],  a_e4d[r],  a_mag4_ref[r],  a_f4t,  a_e4t),
                    (a_f6d[r],  a_e6d[r],  a_mag6_ref[r],  a_f6t,  a_e6t),
                    (a_f10d[r], a_e10d[r], a_mag10_ref[r], a_f10t, a_e10t),
                    (a_fAd[r],  a_eAd[r],  a_magA_ref[r],  a_fAt,  a_eAt),
                ]:
                    fv, ev = calculate_flux_from_magnitude(mg, fd, fe, mzp_r, mzp_d)
                    ft_arr[r] = fv
                    et_arr[r] = ev

            else:
                # Calibrated FITS: pre-computed AB fluxes; diff fluxes remain NaN.
                a_f3t[r]  = float(table[idx_m[i]]['FLUX_3_TOT_AB'])
                a_f4t[r]  = float(table[idx_m[i]]['FLUX_4_TOT_AB'])
                a_f6t[r]  = float(table[idx_m[i]]['FLUX_6_TOT_AB'])
                a_f10t[r] = float(table[idx_m[i]]['FLUX_10_TOT_AB'])
                a_e3t[r]  = float(table[idx_m[i]]['FERR_3_TOT_AB'])
                a_e4t[r]  = float(table[idx_m[i]]['FERR_4_TOT_AB'])
                a_e6t[r]  = float(table[idx_m[i]]['FERR_6_TOT_AB'])
                a_e10t[r] = float(table[idx_m[i]]['FERR_10_TOT_AB'])

    if verbose:
        print(" Building Done in ", time.time() - process)

    # ------------------------------------------------------------------
    # Assemble the flat DataFrame from pre-allocated arrays — one allocation,
    # no per-source DataFrames, no pd.concat of 2000+ frames.
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'ID_REF':            a_id_ref,
        'CLASS_STAR_REF':    a_cs_ref,
        'ALPHAWIN_REF':      a_ra_ref,
        'DELTAWIN_REF':      a_dec_ref,
        'MAGZP_REF':         a_magzp_ref,
        'MAGZPRMS_REF':      a_magzprms_ref,
        'INFOBITS_REF':      a_ibits_ref,
        'FLAG_SE_REF':       a_flag_se_ref,
        'OBSMJD':            a_obsmjd,
        'AIRMASS':           a_airmass,
        'NMATCHES':          a_nmatches,
        'MAGZP_DIF':         a_magzp_dif,
        'MAGZPRMS_DIF':      a_mzprms_dif,
        'CLRCOEFF':          a_clrcoeff,
        'SATURATE':          a_saturate,
        'INFOBITS_DIF':      a_ibits_dif,
        'SEEING':            a_seeing,
        'MAGLIM':            a_maglim,
        'ALPHA_OBJ':         a_alpha_obj,
        'DELTA_OBJ':         a_delta_obj,
        'ALPHAWIN_OBJ':      a_awin_obj,
        'DELTAWIN_OBJ':      a_dwin_obj,
        'DISTANCE':          a_distance,
        'CLASS_STAR_OBJ':    a_cs_obj,
        'FLAG_SE_DIF':       a_flag_se_dif,
        'FLAG_DET':          a_flag_det,
        'MAG_AP_3_REF':      a_mag3_ref,
        'MAG_AP_4_REF':      a_mag4_ref,
        'MAG_AP_6_REF':      a_mag6_ref,
        'MAG_AP_10_REF':     a_mag10_ref,
        'MAG_AP_AUTO_REF':   a_magA_ref,
        'FLUX_3_DIF':        a_f3d,
        'FLUX_4_DIF':        a_f4d,
        'FLUX_6_DIF':        a_f6d,
        'FLUX_10_DIF':       a_f10d,
        'FLUX_AUTO_DIF':     a_fAd,
        'FERR_3_DIF':        a_e3d,
        'FERR_4_DIF':        a_e4d,
        'FERR_6_DIF':        a_e6d,
        'FERR_10_DIF':       a_e10d,
        'FERR_AUTO_DIF':     a_eAd,
        'FLUX_3_TOT_AB':     a_f3t,
        'FLUX_4_TOT_AB':     a_f4t,
        'FLUX_6_TOT_AB':     a_f6t,
        'FLUX_10_TOT_AB':    a_f10t,
        'FLUX_AUTO_TOT_AB':  a_fAt,
        'FERR_3_TOT_AB':     a_e3t,
        'FERR_4_TOT_AB':     a_e4t,
        'FERR_6_TOT_AB':     a_e6t,
        'FERR_10_TOT_AB':    a_e10t,
        'FERR_AUTO_TOT_AB':  a_eAt,
        'object_index':      a_obj_idx,
    })

    return {'LIGHTCURVES_DF': df}


def calculate_flux_from_magnitude(mag_ref, flux_dif, fluxerr_dif, magzp_ref, magzp_dif, optional:str = ""):
    if flux_dif is np.nan:
        return (np.nan, np.nan)
    else:

        mag_ref_tot = np.float64(mag_ref) + np.float64(magzp_ref)
        flux_ref_tot = 10.0**(0.4*(np.float64(magzp_dif) - np.float64(mag_ref_tot)))
        flux_tot = np.float64(flux_ref_tot) + np.float64(flux_dif)

        if(flux_tot) < 0:
            return (np.nan, np.nan)

        mag_tot = np.float64(magzp_dif) - 2.5*np.log10(flux_tot)
        flux_ab = 10.0**(-0.4*(np.float64(mag_tot) + 48.6))

        mag_err = 1.0857 / (flux_tot / fluxerr_dif)

        fluxerr_ab = 0.4 * np.log(10) * (10.0**(-0.4*(mag_tot+48.6))) * mag_err

        if(flux_tot) < 0:
            print("WARNING! ", optional, " - flux_tot = ", flux_tot, "(", flux_ref_tot, " + ", flux_dif, ")")
            print("IN:")
            print("mag_ref : ", mag_ref)
            print("flux_dif : ", flux_dif)
            print("fluxerr_dif : ", fluxerr_dif)
            print("magzp_ref : ", magzp_ref)
            print("magzp_dif : ", magzp_dif)
            print("")
            print("mag_ref_tot : ", mag_ref_tot)
            print("flux_ref_tot : ", flux_ref_tot)
            print("flux_tot : ", flux_tot)
            print("mag_tot : ", mag_tot)
            print("flux_ab : ", flux_ab)
            print("mag_err : ", mag_err)
            print("fluxerr_ab : ", fluxerr_ab)
            print("")
            print("OUT:")
            print("flux_ab : ", flux_ab)
            print("fluxerr_ab : ", fluxerr_ab)

        return (flux_ab, fluxerr_ab)


def calculate_preliminary_features(dict_lightcurves, min_epochs = 1):

    if min_epochs < 1:
        min_epochs = 1

    features = pd.DataFrame(columns=(Preliminary_Features.names + Preliminary_Features.mean
                                    + Preliminary_Features.var + Preliminary_Features.ex_var
                                    + Preliminary_Features.exverr + Preliminary_Features.class_star))

    for object in dict_lightcurves['ID_REF'].unique():

        filter = np.logical_and(dict_lightcurves['ID_REF'] == object, dict_lightcurves['INFOBITS_REF'] == 0)
        filter = np.logical_and(filter, dict_lightcurves['INFOBITS_DIF'] == 0)
        filter = np.logical_and(filter, dict_lightcurves['FLAG_SE_REF'] == 0)
        filter = np.logical_and(filter, dict_lightcurves['FLAG_SE_DIF'] == 0)

        # filter = dict_lightcurves['ID_REF'] == object

        if len(dict_lightcurves[filter]) > min_epochs:

            v3 = var_parameters(dict_lightcurves[filter]['OBSMJD'], dict_lightcurves[filter]['FLUX_3_TOT_AB'], dict_lightcurves[filter]['FERR_3_TOT_AB'])
            v4 = var_parameters(dict_lightcurves[filter]['OBSMJD'], dict_lightcurves[filter]['FLUX_4_TOT_AB'], dict_lightcurves[filter]['FERR_4_TOT_AB'])
            v6 = var_parameters(dict_lightcurves[filter]['OBSMJD'], dict_lightcurves[filter]['FLUX_6_TOT_AB'], dict_lightcurves[filter]['FERR_6_TOT_AB'])
            v10 = var_parameters(dict_lightcurves[filter]['OBSMJD'], dict_lightcurves[filter]['FLUX_10_TOT_AB'], dict_lightcurves[filter]['FERR_10_TOT_AB'])
            vA = var_parameters(dict_lightcurves[filter]['OBSMJD'], dict_lightcurves[filter]['FLUX_AUTO_TOT_AB'], dict_lightcurves[filter]['FERR_AUTO_TOT_AB'])

            dict = {
                'ID_REF': object,
                'RA': dict_lightcurves[filter].iloc[0, dict_lightcurves.columns.get_loc('ALPHAWIN_REF')],
                'DEC': dict_lightcurves[filter].iloc[0, dict_lightcurves.columns.get_loc('DELTAWIN_REF')],
                'CLASS_STAR': dict_lightcurves[filter].iloc[0, dict_lightcurves.columns.get_loc('CLASS_STAR_REF')],
                'DELTA_T': max(dict_lightcurves[filter]['OBSMJD']) - min(dict_lightcurves[filter]['OBSMJD']),
                'COUNT': np.count_nonzero(filter),
                'MEAN_3_FLUX': np.mean(dict_lightcurves[filter]['FLUX_3_TOT_AB']),
                'MEAN_4_FLUX': np.mean(dict_lightcurves[filter]['FLUX_4_TOT_AB']),
                'MEAN_6_FLUX': np.mean(dict_lightcurves[filter]['FLUX_6_TOT_AB']),
                'MEAN_10_FLUX': np.mean(dict_lightcurves[filter]['FLUX_10_TOT_AB']),
                'MEAN_AUTO_FLUX': np.mean(dict_lightcurves[filter]['FLUX_AUTO_TOT_AB']),
                'VAR_3_FLUX': np.var(dict_lightcurves[filter]['FLUX_3_TOT_AB']),
                'VAR_4_FLUX': np.var(dict_lightcurves[filter]['FLUX_4_TOT_AB']),
                'VAR_6_FLUX': np.var(dict_lightcurves[filter]['FLUX_6_TOT_AB']),
                'VAR_10_FLUX': np.var(dict_lightcurves[filter]['FLUX_10_TOT_AB']),
                'VAR_AUTO_FLUX': np.var(dict_lightcurves[filter]['FLUX_AUTO_TOT_AB']),
                'PVAR_3_FLUX': v3[0],
                'PVAR_4_FLUX': v4[0],
                'PVAR_6_FLUX': v6[0],
                'PVAR_10_FLUX': v10[0],
                'PVAR_AUTO_FLUX': vA[0],
                'EXVAR_3_FLUX': v3[1],
                'EXVAR_4_FLUX': v4[1],
                'EXVAR_6_FLUX': v6[1],
                'EXVAR_10_FLUX': v10[1],
                'EXVAR_AUTO_FLUX': vA[1],
                'EXVERR_3_FLUX': v3[2],
                'EXVERR_4_FLUX': v4[2],
                'EXVERR_6_FLUX': v6[2],
                'EXVERR_10_FLUX': v10[2],
                'EXVERR_AUTO_FLUX': vA[2],
                # Class Star features
                'DELTA_CS': np.amax(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0) - np.amin(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0),
                'MEAN_CS': np.nanmean(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0),
                'MEDIAN_CS': np.nanmedian(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0),
                'STD_CS': np.nanstd(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0),
                'VAR_CS': np.nanvar(dict_lightcurves[filter]['CLASS_STAR_OBJ'], axis=0),
            }

            features = pd.concat([features, pd.DataFrame([dict])])
        #
        # else:
        #     continue

    return features


def var_parameters(jd, mag, err) -> (np.float64, np.float64, np.float64):

    # Calculate the probability of a light curve to be variable
    # and the excess variance.
    #
    # inputs:
    # jd: julian days array
    # mag: magnitudes array
    # err: error of magnitudes array
    #
    # outputs:
    # p_chi: Probability that the source is intrinsically variable (Pvar)
    # ex_var: excess variance, a measure of the intrinsic variability amplitude
    # ex_verr: error of the excess variance

    mean = np.mean(mag)
    nepochs = np.float64(len(jd))

    chi = np.sum((mag - mean)**2. / err**2.)
    p_chi = chi2.cdf(chi, (nepochs-1))

    a = (mag-mean)**2
    ex_var = (np.sum(a-err**2)/((nepochs*(mean**2))))
    sd = np.sqrt((1./(nepochs-1))*np.sum(((a-err**2)-ex_var*(mean**2))**2))
    ex_verr = sd/((mean**2)*np.sqrt(nepochs))

    return (p_chi, ex_var, ex_verr)


def save_as_parquet(savepath_lightcurves: str, savepath_preliminary_features: str, dict_lightcurves):
    # need for every table.data need to calculate class_star things

    lc = pd.concat([table.data for table in dict_lightcurves['LIGHTCURVES']])

    lc = lc.astype({'CLASS_STAR_REF': 'float32',
                    'ALPHAWIN_REF': 'float64',
                    'DELTAWIN_REF': 'float64',
                    'MAGZP_REF': 'float32',
                    'MAGZPRMS_REF': 'float32',
                    'INFOBITS_REF': 'uint32',
                    'FLAG_SE_REF': 'uint8',
                    'FLUX_3_TOT_AB': 'float64',
                    'FLUX_4_TOT_AB': 'float64',
                    'FLUX_6_TOT_AB': 'float64',
                    'FLUX_10_TOT_AB': 'float64',
                    'FLUX_AUTO_TOT_AB': 'float64',
                    'FERR_3_TOT_AB': 'float64',
                    'FERR_4_TOT_AB': 'float64',
                    'FERR_6_TOT_AB': 'float64',
                    'FERR_10_TOT_AB': 'float64',
                    'FERR_AUTO_TOT_AB': 'float64',
                    'OBSMJD': 'float32',
                    'AIRMASS': 'float32',
                    'NMATCHES': 'uint32',
                    'MAGZP_DIF': 'float32',
                    'MAGZPRMS_DIF': 'float32',
                    'CLRCOEFF': 'float32',
                    'SATURATE': 'float32',
                    'INFOBITS_DIF': 'uint32',
                    'SEEING': 'float32',
                    'MAGLIM': 'float32',
                    'ALPHA_OBJ': 'float64',
                    'DELTA_OBJ': 'float64',
                    'ALPHAWIN_OBJ': 'float64',
                    'DELTAWIN_OBJ': 'float64',
                    'DISTANCE': 'float64',
                    'CLASS_STAR_OBJ': 'float32',
                    'FLAG_SE_DIF': 'uint8',
                    'FLAG_DET': 'bool',
                    'FLUX_3_DIF': 'float64',
                    'FLUX_4_DIF': 'float64',
                    'FLUX_6_DIF': 'float64',
                    'FLUX_10_DIF': 'float64',
                    'FLUX_AUTO_DIF': 'float64',
                    'FERR_3_DIF': 'float64',
                    'FERR_4_DIF': 'float64',
                    'FERR_6_DIF': 'float64',
                    'FERR_10_DIF': 'float64',
                    'FERR_AUTO_DIF': 'float64',
                    'MAG_AP_3_REF': 'float64',
                    'MAG_AP_4_REF': 'float64',
                    'MAG_AP_6_REF': 'float64',
                    'MAG_AP_10_REF': 'float64',
                    'MAG_AP_AUTO_REF': 'float64',
                    })

    if len(lc) > 0:

        stats = calculate_preliminary_features(lc)

        fp.write(savepath_lightcurves, lc, compression="GZIP")
        fp.write(savepath_preliminary_features, stats, compression="GZIP")

        print(stats)

        print("saved: ", len(lc['ID_REF'].unique()),"Lightcurves in", savepath_lightcurves)
        print("saved: ", len(stats),"stats in", savepath_preliminary_features)
    else:
        print("did not save, no curves in file")

# In []
def main():
    # Builds a parquet file with the data from multiple sextractions.
    # Give the arguments as aditional arguments when calling this script
    # First Argument is  the catalog from witch we are choosing the Objects
    # Second argument is the parent folder on witch the catalogs lives
    # Third argument is the save path
    # Fourth argument indicates if we are bulding from 'SCI'ence or 'SIM'ulated images

    # Example:

    # reference_catalog_path = "../Catalogs/Objects/000468_zg_c03_q1(REFERENCE)[OBJECTS].csv"
    # search_cat_home = "../Catalogs/Calibrated/Science"
    # save_path = "."
    # cat_type = "SIM"

    reference_catalog_path = sys.argv[1]
    search_cat_home = sys.argv[2]
    save_path = sys.argv[3]
    cat_type = sys.argv[4]
    print("python", sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    simulated = cat_type.casefold()=='SIM'.casefold()

    file = build_lightcurve_catalog(reference_catalog_path=reference_catalog_path,
                                    search_catalogs_home=search_cat_home,
                                    verbose=True,
                                    simulated=simulated)
    #
    filename = reference_catalog_path[reference_catalog_path.rfind('/') + 1: reference_catalog_path.rfind('(')]


    filename_data = save_path + '/' + filename + "_LightCurves(" + str(len(file['LIGHTCURVES'])) + ").parquet"
    filename_stat = save_path + '/' + filename + "_Preliminary_Features" + ".parquet"

    save_as_parquet(filename_data, filename_stat, file)
    print(filename_stat, "and", filename_data, "saved")


# In []
if __name__ == '__main__':
    main()
