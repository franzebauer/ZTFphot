"""
calibrate.py
------------
Pipeline steps for photometric calibration and flatfield construction:
  step_vet           — flag multi-epoch variable calibration stars
  step_calibrate     — per-epoch linear ZP + faint + poly + flatfield
  step_build_flatfield — stack per-epoch residuals into a spatial flatfield
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _cal_quad_dir(base_dir: Path, field: int, fc: str, ccd: int, qid: int,
                  suffix: str = "") -> Path:
    """Return the Calibrated per-quadrant directory."""
    return (base_dir / f"Calibrated{suffix}"
            / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid))


# ── Step: vet calibration stars ───────────────────────────────────────────────

def step_vet(base_dir: Path, quadrants: list[dict]) -> int:
    """
    Flag multi-epoch outlier calibration stars.
    Requires lightcurves to have been built first.
    """
    import subprocess

    n_done = 0
    vet_script = Path(__file__).parent / "vet_calibration_stars.py"

    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        cat_name = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"

        parquet = (base_dir / "LightCurves" / f"{field:06d}" / fc
                   / f"ccd{ccd:02d}" / f"q{qid_}" / "lightcurves.parquet")
        ref_csv = base_dir / "Catalogs" / f"{cat_name}(REFERENCE)[OBJECTS].csv"

        if not parquet.exists():
            logger.debug(f"vet: lightcurves not found for {cat_name} — run lightcurves first")
            continue
        if not ref_csv.exists():
            logger.warning(f"vet: reference catalog missing — {ref_csv}")
            continue

        cmd = [
            sys.executable, str(vet_script),
            "--field", str(field), "--band", fc,
            "--ccd", str(ccd), "--qid", str(qid_),
            "--base-dir", str(base_dir), "--threshold", "2.5",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"vet failed for {cat_name}: {result.stderr.strip()[-200:]}")
                continue
            logger.info(f"vet: {cat_name}")
            n_done += 1
        except Exception as exc:
            logger.error(f"vet failed for {cat_name}: {exc}")

    logger.info(f"vet: completed for {n_done} quadrant(s)")
    return n_done


# ── Step: per-epoch calibration ───────────────────────────────────────────────

def _calibrate_one(args: tuple) -> tuple[str, bool, str]:
    """Worker: run calib_catalog() for a single LDAC epoch catalog."""
    ref_csv, in_ldac, out_cal, img_kind, vet_catalog, extra_kw = args
    import contextlib, io, warnings
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    try:
        from calib_catalogs import calib_catalog
        Path(out_cal).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib_catalog(str(ref_csv), str(in_ldac), str(out_cal), img_kind,
                          vet_catalog=vet_catalog, **extra_kw)
        ok  = Path(out_cal).exists()
        msg = "" if ok else "too few calibrators — calib_catalog wrote nothing"
        return str(out_cal), ok, msg
    except Exception as exc:
        return str(out_cal), False, str(exc)


def step_calibrate(
    base_dir: Path, quadrants: list[dict],
    workers: int = 4, force: bool = False,
    vet_catalog: Optional[Path] = None,
    poly_degree: int = 2,
    flatfield: Optional[dict] = None,
    target_ra: Optional[float] = None,
    target_dec: Optional[float] = None,
    save_residuals: bool = False,
    suffix: str = "",
) -> int:
    """
    Apply per-epoch photometric calibration (linear ZP → 3σ clip → faint
    correction → 2D polynomial → flatfield) to every SExtractor LDAC catalog.

    Reads from  SExCatalogs{suffix}/{field}/{fc}/{ccd}/{qid}/*_sexout.fits
    Writes to   Calibrated{suffix}/{field}/{fc}/{ccd}/{qid}/*_cal.fits
    """
    cat_dir  = base_dir / "Catalogs"
    sex_root = base_dir / f"SExCatalogs{suffix}"
    cal_root = base_dir / f"Calibrated{suffix}"

    tasks = []
    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        cat_name = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"
        ref_csv  = cat_dir / f"{cat_name}(REFERENCE)[OBJECTS].csv"

        if not ref_csv.exists():
            logger.warning(f"calibrate: reference catalog missing — {ref_csv}")
            continue

        sex_subdir = sex_root / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_)
        cal_subdir = cal_root / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_)

        ldac_files = sorted(sex_subdir.glob("*_sexout.fits"))
        if not ldac_files:
            logger.debug(f"calibrate: no LDAC catalogs in {sex_subdir}")
            continue

        for ldac in ldac_files:
            out_cal = cal_subdir / ldac.name.replace("_sexout.fits", "_cal.fits")
            if out_cal.exists() and not force:
                continue

            # Auto-discover vet catalog if not explicitly provided
            # Always look in the standard Calibrated/ dir (not _sci) for vet catalog
            vet_cat = None
            if vet_catalog is not None and vet_catalog.exists():
                vet_cat = vet_catalog
            else:
                vet_fn = _cal_quad_dir(base_dir, field, fc, ccd, qid_) / "vet_calib_stars.fits"
                if vet_fn.exists():
                    vet_cat = vet_fn

            resid_out = None
            if save_residuals:
                resid_dir = (base_dir / f"FlatfieldResiduals{suffix}"
                             / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_))
                resid_dir.mkdir(parents=True, exist_ok=True)
                resid_out = resid_dir / ldac.name.replace("_sexout.fits", "_resid.npz")

            extra_kw = dict(
                poly_degree=poly_degree, flatfield=flatfield,
                target_ra=target_ra, target_dec=target_dec,
                residuals_out=str(resid_out) if resid_out else None,
            )
            tasks.append((ref_csv, ldac, out_cal, "SIM", vet_cat, extra_kw))

    if not tasks:
        logger.info("calibrate: all catalogs already calibrated (use --force to redo)")
        return 0

    logger.info(f"calibrate: {len(tasks)} epochs with {workers} workers")
    n_done = n_sparse = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_calibrate_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            out_cal, ok, msg = fut.result()
            if ok:
                n_done += 1
            elif "too few" in msg:
                n_sparse += 1
            else:
                n_fail += 1
                logger.warning(f"calibrate failed: {Path(out_cal).name}: {msg}")
            if i % 50 == 0 or i == len(tasks):
                logger.info(f"calibrate: {i}/{len(tasks)} — "
                            f"{n_done} done, {n_sparse} sparse, {n_fail} failed")

    logger.info(f"calibrate: {n_done} written, {n_sparse} sparse, {n_fail} failures")
    return n_done


# ── Step: build spatial flatfield ─────────────────────────────────────────────

def step_build_flatfield(
    base_dir: Path, quadrants: list[dict],
    nbins: int = 20, min_count: int = 50,
    suffix: str = "",
) -> dict:
    """
    Stack per-epoch NPZ residual files from step_calibrate(save_residuals=True)
    and build a spatial flatfield per quadrant.

    Returns dict keyed by (field, fc, ccd, qid) → flatfield dict.
    """
    from scipy.stats import binned_statistic_2d

    flatfields = {}

    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]

        resid_dir = (base_dir / f"FlatfieldResiduals{suffix}"
                     / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_))
        resid_files = sorted(resid_dir.glob("*.npz")) if resid_dir.exists() else []

        if not resid_files:
            logger.warning(f"flatfield: no residual files for "
                           f"{field:06d}/{fc}/c{ccd:02d}/q{qid_}")
            continue

        all_ra, all_dec, all_dm = [], [], []
        for f in resid_files:
            try:
                d = np.load(f)
                if 'dm_4' in d:
                    all_ra.extend(d['ra_4']); all_dec.extend(d['dec_4']); all_dm.extend(d['dm_4'])
                elif 'dm_3' in d:
                    all_ra.extend(d['ra_3']); all_dec.extend(d['dec_3']); all_dm.extend(d['dm_3'])
                else:
                    all_ra.extend(d['ra']); all_dec.extend(d['dec']); all_dm.extend(d['dm'])
            except Exception:
                pass

        all_ra  = np.array(all_ra,  dtype=float)
        all_dec = np.array(all_dec, dtype=float)
        all_dm  = np.array(all_dm,  dtype=float)

        finite = np.isfinite(all_dm) & (np.abs(all_dm) < 1.0)
        if finite.sum() < 100:
            logger.warning(f"flatfield: too few residuals ({finite.sum()})")
            continue

        all_ra, all_dec, all_dm = all_ra[finite], all_dec[finite], all_dm[finite]
        ra_lo, ra_hi   = np.percentile(all_ra,  [1, 99])
        dec_lo, dec_hi = np.percentile(all_dec, [1, 99])

        stat, ra_e, dec_e, _ = binned_statistic_2d(
            all_ra, all_dec, all_dm, statistic='median', bins=nbins,
            range=[[ra_lo, ra_hi], [dec_lo, dec_hi]])
        nobs, _, _, _ = binned_statistic_2d(
            all_ra, all_dec, all_dm, statistic='count', bins=nbins,
            range=[[ra_lo, ra_hi], [dec_lo, dec_hi]])

        global_med  = float(np.nanmedian(stat[nobs >= min_count]))
        stat_filled = np.where(nobs >= min_count, stat - global_med, -global_med)

        vals = stat[nobs >= min_count]
        rms  = float(np.std(vals)) * 1000 if len(vals) > 0 else np.nan
        pp   = float(np.nanmax(vals) - np.nanmin(vals)) * 1000 if len(vals) > 1 else np.nan
        logger.info(f"flatfield {field:06d}/{fc}/c{ccd:02d}/q{qid_}: "
                    f"RMS={rms:.1f} mmag  P-P={pp:.1f} mmag  "
                    f"N={finite.sum():,}  bins={int(np.sum(nobs>=min_count))}/{nbins**2}")

        ff = dict(stat=stat_filled.astype(np.float32),
                  ra_edges=ra_e.astype(np.float64),
                  dec_edges=dec_e.astype(np.float64))

        out_path = _cal_quad_dir(base_dir, field, fc, ccd, qid_, suffix) / "flatfield.npz"
        np.savez(str(out_path), **ff, nobs=nobs.astype(np.int32))
        logger.info(f"flatfield saved → {out_path}")

        flatfields[(field, fc, ccd, qid_)] = ff

    return flatfields
