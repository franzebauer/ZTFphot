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
from astropy.table import Table as AstropyTable

logger = logging.getLogger(__name__)


def _cast_lc_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast light-curve columns to correct numeric types before writing Parquet.
    FITS LDAC header keywords are parsed as raw strings; this coerces them.

    float64: OBSMJD (MJD precision), ALPHAWIN_REF, DELTAWIN_REF, ALPHA_OBJ,
             DELTA_OBJ (astrometric precision at RA~330 deg)
    float32: all photometric quantities and per-epoch scalars, plus the sci-pos
             epoch offsets DRA_SCI/DDEC_SCI (arcsec from the reference position,
             2 dp — absolute degrees would need float64 to hold 0.01")
    Int32:   integer indices and flag words
    """
    float64_cols = [
        'OBSMJD',
        'ALPHAWIN_REF', 'DELTAWIN_REF',
    ]
    float32_cols = [
        'AIRMASS', 'MAGZP_DIF', 'MAGZPRMS_DIF', 'CLRCOEFF',
        'SEEING', 'MAGLIM', 'DISTANCE',
        'CLASS_STAR_OBJ',
        *_PHOT_COLS['full'],
        'DRA_SCI', 'DDEC_SCI',                               # sci-pos offsets, arcsec
    ]
    int32_cols = [
        'NMATCHES', 'INFOBITS_DIF', 'FLAGS', 'object_index',
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
    'SEEING', 'MAGLIM', 'NMATCHES', 'INFOBITS_DIF', 'APCORR410',
]

# Columns to drop before writing (redundant, derivable, or replaced by metadata)
_DROP_COLS = {
    'ALPHAWIN_J2000', 'DELTAWIN_J2000',   # dropped in ref-pos; renamed to ALPHA_SCI/DELTA_SCI in sci-pos
    'ALPHA_J2000', 'DELTA_J2000',          # reference positions — kept in ALPHAWIN_REF/DELTAWIN_REF
    'VECTOR_ASSOC',                        # carried as object_index; raw column not needed in parquet
    'SATURATE',
}

# Photometric columns written per --lc-columns preset (all others of that kind are
# dropped). TOT: calibrated, 4 px aperture-corrected to 10 px (APCORR410), bipolar
# flux in μJy. DIF: raw difference-image flux converted to μJy with ZTF's per-epoch
# MAGZP_DIF — neither recalibrated nor aperture-corrected.
_PHOT_COLS = {
    'slim': ['MAG_4_TOT_AB', 'MERR_4_TOT_AB', 'FLUX_4_TOT_uJy', 'FERR_4_TOT_uJy',
             'FLUX_4_DIF_uJy', 'FERR_4_DIF_uJy', 'APCORR410', 'MAG_4_REF'],
}
_PHOT_COLS['full'] = _PHOT_COLS['slim'] + [
    'MAG_10_TOT_AB', 'MERR_10_TOT_AB', 'FLUX_10_TOT_uJy', 'FERR_10_TOT_uJy',
    'FLUX_10_DIF_uJy', 'FERR_10_DIF_uJy']


def _photometry_columns(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Rename the calibrated fluxes to μJy names, convert the raw difference fluxes
    (counts) to μJy with MAGZP_DIF, and keep only the preset's photometric columns."""
    for k in ('4', '10'):
        df = df.rename(columns={f'FLUX_{k}_TOT_AB': f'FLUX_{k}_TOT_uJy',
                                f'FERR_{k}_TOT_AB': f'FERR_{k}_TOT_uJy'})
        if f'FLUX_{k}_DIF' in df.columns:
            ujy = 10.0 ** (-0.4 * (pd.to_numeric(df['MAGZP_DIF'], errors='coerce') - 23.9))
            df[f'FLUX_{k}_DIF_uJy'] = pd.to_numeric(df[f'FLUX_{k}_DIF'], errors='coerce') * ujy
            df[f'FERR_{k}_DIF_uJy'] = pd.to_numeric(df[f'FERR_{k}_DIF'], errors='coerce') * ujy
    keep = set(_PHOT_COLS[preset])
    phot = [c for c in df.columns if c.split('_')[0] in ('MAG', 'MERR', 'FLUX', 'FERR')
            or c == 'APCORR410']
    return df.drop(columns=[c for c in phot if c not in keep])


def step_lightcurves(
    base_dir: Path, quadrants: list[dict],
    force: bool = False,
    use_calibrated: bool = True,
    suffix: str = "",
    target_ra: float | None = None,
    target_dec: float | None = None,
    target_match_radius: float = 1.0,
    run_meta: dict | None = None,
    lc_columns: str = "slim",
) -> int:
    """
    Assemble light curves from calibrated per-epoch FITS for each quadrant.
    Reads Calibrated/{field}/{fc}/{ccd}/{qid}/*_cal.fits directly.
    Saves one Parquet per quadrant under LightCurves/, with the photometric columns of
    the `lc_columns` preset (_PHOT_COLS: slim or full).

    Parquet metadata keys (not columns):
      field, filtercode, ccdid, qid
      MAGZP_REF_{field}_{fc}_c{ccd}_q{qid}
      MAGZPRMS_REF_{field}_{fc}_c{ccd}_q{qid}
      target_sep_arcsec  (nearest ref source to target; NaN if target not given)
      target_ref_mag     (reference AB mag of the target's matched source, or NaN if
                          reference-less — then FLUX_4_TOT_uJy is difference flux)
      epoch_summary_{field}_{fc}_c{ccd}_q{qid}
                         JSON, from the data: n_epochs, MJD range, min MAGLIM, max
                         SEEING, epochs with INFOBITS set — the effective epoch cuts
      + every key of run_meta (run_pipeline passes ztfphot_run = this invocation and
        ztfphot_download = the latest download-time invocation for the target, both
        JSON). merge_fields carries all of these into the merged parquet.
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
        lc_out = lc_dir / f"lightcurves{suffix}.parquet"

        _tgt_key = (f"{target_ra},{target_dec}" if target_ra is not None and target_dec is not None
                    else "none")
        if lc_out.exists() and not force:
            # light curves made for another target carry its metadata (target_sep_arcsec,
            # target_ref_mag, ztfphot_run): rebuild them (the photometry is shared)
            _old = (pq.read_schema(lc_out).metadata or {}).get(b'target_radec', b'').decode()
            if _old == _tgt_key:
                n_skip += 1
                continue
            logger.info(f"  {lc_out.name} was built for target '{_old or 'unrecorded'}' — rebuilding")

        ref_csv = cat_dir / f"{tag}(REFERENCE)[OBJECTS].csv"
        if not ref_csv.exists():
            logger.warning(f"Reference catalog not found: {ref_csv}")
            continue

        cal_dir   = base_dir / f"Calibrated{suffix}" / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_)
        cal_files = sorted(cal_dir.glob("*_cal.fits"))
        if not cal_files:
            logger.warning(f"No calibrated FITS found in {cal_dir}")
            continue

        logger.info(f"Building light curves for {tag} ({len(cal_files)} epochs)")

        # Load reference catalog — provides object positions and per-source metadata
        ref = pd.read_csv(ref_csv)
        # Per-source reference columns to carry into LC (MAGZP_REF/MAGZPRMS_REF → metadata)
        ref_cols = {}
        # reference flags as written by make_catalog
        for src, dst in [('ALPHAWIN_J2000', 'ALPHAWIN_REF'), ('DELTAWIN_J2000', 'DELTAWIN_REF'),
                         ('FLAG_SE_REF', 'FLAG_SE_REF'),
                         ('INFOBITS_REF', 'INFOBITS_REF')]:
            if src in ref.columns:
                ref_cols[dst] = pd.to_numeric(ref[src], errors='coerce').values
        # 4px reference AB magnitude per source (aperture mag + reference ZP). Enables
        # a per-source measured-vs-reference calibration check from the parquet alone,
        # and marks injected sources, which carry MAG_APER_4px = 99 (so MAG_4_REF > 90).
        if 'MAG_APER_4px' in ref.columns and 'MAGZP_REF' in ref.columns:
            ref_cols['MAG_4_REF'] = (
                pd.to_numeric(ref['MAG_APER_4px'], errors='coerce').values
                + pd.to_numeric(ref['MAGZP_REF'], errors='coerce').values
            )

        # Append one sentinel row for the target ASSOC entry (index = len(ref), 0-based).
        # Needed when the target is appended to the ASSOC catalog beyond the ref catalog.
        if target_ra is not None and target_dec is not None:
            _tgt_extra = {
                'ALPHAWIN_REF': np.float64(target_ra),
                'DELTAWIN_REF': np.float64(target_dec),
                'FLAG_SE_REF':  np.int32(0),
                'INFOBITS_REF': np.int32(0),
                'MAG_4_REF':    np.float64('nan'),   # target has no reference magnitude
            }
            for dst in list(ref_cols):
                ref_cols[dst] = np.append(ref_cols[dst], _tgt_extra[dst])

        # Extract MAGZP_REF / MAGZPRMS_REF as scalars for parquet metadata
        magzp_ref_val    = float(pd.to_numeric(ref['MAGZP_REF'],    errors='coerce').iloc[0]) if 'MAGZP_REF'    in ref.columns else float('nan')
        magzprms_ref_val = float(pd.to_numeric(ref['MAGZPRMS_REF'], errors='coerce').iloc[0]) if 'MAGZPRMS_REF' in ref.columns else float('nan')

        # Nearest ref source to target — stored as metadata for downstream diagnostics.
        # target_ref_mag = that source's reference AB mag if it is within the match
        # radius (i.e. the target HAS a reference detection); NaN when reference-less
        # (then FLUX_4_TOT_uJy is bipolar difference flux, not total — see README).
        target_sep_val = float('nan')
        target_ref_mag = float('nan')
        if target_ra is not None and target_dec is not None:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            ra_col  = 'ALPHAWIN_J2000' if 'ALPHAWIN_J2000' in ref.columns else 'RA'
            dec_col = 'DELTAWIN_J2000' if 'DELTAWIN_J2000' in ref.columns else 'DEC'
            _ra  = pd.to_numeric(ref[ra_col],  errors='coerce').values
            _dec = pd.to_numeric(ref[dec_col], errors='coerce').values
            _ok  = np.isfinite(_ra) & np.isfinite(_dec)
            if _ok.any():
                _tgt  = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
                _cats = SkyCoord(ra=_ra[_ok] * u.deg,  dec=_dec[_ok] * u.deg)
                _seps = _tgt.separation(_cats).arcsec
                _jmin = int(np.argmin(_seps))
                target_sep_val = float(_seps[_jmin])
                if (target_sep_val <= target_match_radius
                        and 'MAG_APER_4px' in ref.columns and 'MAGZP_REF' in ref.columns):
                    _rm = (pd.to_numeric(ref['MAG_APER_4px'], errors='coerce').values[_ok]
                           + pd.to_numeric(ref['MAGZP_REF'], errors='coerce').values[_ok])
                    target_ref_mag = float(_rm[_jmin])

        frames = []
        for cal_path in cal_files:
            try:
                with pyfits.open(cal_path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[1].data
            except Exception as e:
                logger.warning(f"  Could not read {cal_path.name}: {e}")
                continue

            tbl = AstropyTable(data).to_pandas()
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

            # Reference catalog properties. A row beyond the catalog (the --ra/--dec target
            # injected by the photometry) needs that target here too.
            n_ref = len(next(iter(ref_cols.values()))) if ref_cols else 0
            beyond = ep_ref_idx >= n_ref
            if beyond.any():
                logger.warning(f"  {cal_path.name}: {int(beyond.sum())} row(s) beyond the reference "
                               f"catalog (injected target; run lightcurves with --ra/--dec) — skipped")
                ep_rows, ep_ref_idx = ep_rows[~beyond].reset_index(drop=True), ep_ref_idx[~beyond]
            for dst, arr in ref_cols.items():
                ep_rows[dst] = arr[ep_ref_idx]

            ep_rows['object_index'] = ep_ref_idx

            # Rename cal.fits measurement columns to LC schema.
            # Sci-pos: keep epoch positions as ALPHA_SCI/DELTA_SCI.
            # Ref-pos: drop them (they equal the reference positions); _DROP_COLS handles removal.
            pos_rename = {'ALPHAWIN_J2000': 'ALPHA_SCI', 'DELTAWIN_J2000': 'DELTA_SCI'} if suffix == "_sci" else {}
            # per-epoch aperture flags (ztfphot_flags bits 0-7) -> composed into FLAGS
            rename = {'CLASS_STAR': 'CLASS_STAR_OBJ', 'FLAGS': '_EPOCH_FLAGS', **pos_rename}
            ep_rows = ep_rows.rename(columns={k: v for k, v in rename.items() if k in ep_rows.columns})

            frames.append(ep_rows)

        if not frames:
            logger.warning(f"No light curves built for {tag}")
            continue

        all_lcs = pd.concat(frames, ignore_index=True)
        # A target without a reference source is measured only if the photometry injected
        # it (catalogs made for another target in a reused base dir do not contain it)
        if (target_ra is not None and np.isfinite(target_sep_val)
                and target_sep_val > target_match_radius
                and not (all_lcs['object_index'] == len(ref)).any()):
            logger.warning(f"  {tag}: the target has no reference source and is not in the "
                           f"photometry catalogs (made for another target?) — re-run the phot "
                           f"step with --force to measure it")

        # One quality bitmask, FLAGS (ztfphot_flags.py): per-epoch aperture bits +
        # reference-source flag + no-reference marker + cautionary INFOBITS summaries.
        # FLAG_SE_REF / INFOBITS_REF are folded in; INFOBITS_REF (constant per quadrant)
        # goes to the file metadata. INFOBITS_DIF stays as ZTF's raw per-epoch bits.
        from ztfphot_flags import compose as _compose_flags
        _ep_flags = (pd.to_numeric(all_lcs['_EPOCH_FLAGS'], errors='coerce').fillna(0)
                     .astype('int64').to_numpy() if '_EPOCH_FLAGS' in all_lcs else None)
        all_lcs['FLAGS'] = _compose_flags(all_lcs, epoch_flags=_ep_flags)
        _infobits_ref = (int(pd.to_numeric(all_lcs['INFOBITS_REF'], errors='coerce').fillna(0).max())
                         if 'INFOBITS_REF' in all_lcs else None)
        all_lcs = all_lcs.drop(columns=[c for c in ('_EPOCH_FLAGS', 'FLAG_SE_REF', 'INFOBITS_REF')
                                        if c in all_lcs.columns])

        # Drop redundant / removed columns
        drop = [c for c in _DROP_COLS if c in all_lcs.columns]
        if drop:
            all_lcs = all_lcs.drop(columns=drop)
        all_lcs = _photometry_columns(all_lcs, lc_columns)

        # Sci-pos: store epoch positions as offsets FROM THE REFERENCE position in
        # arcsec (2 dp, float32) rather than absolute degrees. The absolute values
        # need float64 to preserve 0.01" (float32 degrees only resolves ~0.03" at
        # these declinations), so this is ~4x cheaper and directly plottable. Using
        # the reference as the baseline — not the per-source median epoch position —
        # keeps any astrometric *bias* visible, not just the scatter.
        if {'ALPHA_SCI', 'DELTA_SCI', 'ALPHAWIN_REF', 'DELTAWIN_REF'} <= set(all_lcs.columns):
            _cd = np.cos(np.radians(pd.to_numeric(all_lcs['DELTAWIN_REF'], errors='coerce')))
            _dra = ((pd.to_numeric(all_lcs['ALPHA_SCI'], errors='coerce')
                     - pd.to_numeric(all_lcs['ALPHAWIN_REF'], errors='coerce')) * _cd * 3600.0)
            _ddec = ((pd.to_numeric(all_lcs['DELTA_SCI'], errors='coerce')
                      - pd.to_numeric(all_lcs['DELTAWIN_REF'], errors='coerce')) * 3600.0)
            all_lcs['DRA_SCI']  = _dra.round(2).astype('float32')
            all_lcs['DDEC_SCI'] = _ddec.round(2).astype('float32')
            all_lcs = all_lcs.drop(columns=['ALPHA_SCI', 'DELTA_SCI'])

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
            b'target_sep_arcsec':           str(target_sep_val).encode(),
            b'target_ref_mag':              str(target_ref_mag).encode(),
            b'target_radec':                _tgt_key.encode(),
        }
        if _infobits_ref is not None:
            extra_meta[f'INFOBITS_REF_{tag}'.encode()] = str(_infobits_ref).encode()
        # Effective epoch selection, measured from the epochs actually present
        _ep = all_lcs.drop_duplicates('OBSMJD')
        _num = lambda c: pd.to_numeric(_ep[c], errors='coerce') if c in _ep.columns else pd.Series(dtype=float)
        _summary = {
            "n_epochs": int(len(_ep)),
            "mjd_min": round(float(_num('OBSMJD').min()), 5),
            "mjd_max": round(float(_num('OBSMJD').max()), 5),
            "maglim_min": round(float(_num('MAGLIM').min()), 3),
            "seeing_max": round(float(_num('SEEING').max()), 3),
            "n_infobits_set": int((_num('INFOBITS_DIF').fillna(0) != 0).sum()),
        }
        import json as _json
        extra_meta[f'epoch_summary_{tag}'.encode()] = _json.dumps(_summary).encode()
        for _k, _v in (run_meta or {}).items():
            extra_meta[_k.encode()] = str(_v).encode()
        table = table.replace_schema_metadata({**existing_meta, **extra_meta})
        _tmp = lc_out.with_name(lc_out.name + ".tmp")   # atomic: no truncated file
        pq.write_table(table, _tmp)
        _tmp.replace(lc_out)

        n_obj = all_lcs['object_index'].nunique()
        logger.info(f"  → {n_obj} objects, {len(all_lcs)} rows → {lc_out}")
        n_done += 1

    logger.info(f"lightcurves: {n_done} processed, {n_skip} already exist")
    return n_done


# ── Step: merge quadrants ─────────────────────────────────────────────────────

def step_merge(base_dir: Path, quadrants: list[dict], force: bool = False,
               target_ra: float | None = None, target_dec: float | None = None,
               suffix: str = "", mag_bin: float = 0.1,
               max_sep_arcsec: float = 1.5) -> None:
    """
    Cross-calibrate and merge per-quadrant light curves for each band.

    The quadrant with the most clean detection-epochs is adopted as the
    photometric reference. All others are corrected with a magnitude-dependent
    offset (per-mag-bin median of dominant - secondary, bin width=mag_bin)
    against common sources, then merged into LightCurves/merged/{band}/.
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
        out_dir = lc_root / "merged" / f"{target_ra:.5f}_{target_dec:+.5f}" / f"{band}{suffix}"
        merge_band(lc_root=lc_root, band=band, quadrants=band_qs, force=force,
                   out_dir=out_dir, lc_suffix=suffix, mag_bin=mag_bin,
                   max_sep_arcsec=max_sep_arcsec)
