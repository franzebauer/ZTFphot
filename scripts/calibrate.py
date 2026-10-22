"""
calibrate.py
------------
Pipeline steps for photometric calibration and flatfield construction:
  step_vet           — flag multi-epoch variable calibration stars
  step_calibrate     — per-epoch linear ZP + 3σ clip + 2D poly + flatfield
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
            "--base-dir", str(base_dir), "--threshold", "1.5",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                why = (result.stderr.strip() or result.stdout.strip()).splitlines()
                logger.warning(f"vet skipped for {cat_name}: {why[-1][-200:] if why else 'no message'}"
                               " — calibrating with all eligible stars")
                continue
            logger.info(f"vet: {cat_name}")
            n_done += 1
        except Exception as exc:
            logger.error(f"vet failed for {cat_name}: {exc}")

    logger.info(f"vet: completed for {n_done} quadrant(s)")
    return n_done


# ── Step: per-epoch calibration ───────────────────────────────────────────────

_VET_OK: dict = {}


def _vet_readable(path: Path) -> bool:
    """True if the vetting catalog can be read (checked once per file); otherwise warn —
    the quadrant is then calibrated with all eligible stars."""
    key = (str(path), path.stat().st_mtime)
    if key not in _VET_OK:
        try:
            from astropy.io import fits
            d = fits.getdata(path, 1)
            _ = d['IS_GOOD'], d['ALPHAWIN_J2000'], d['DELTAWIN_J2000']
            _VET_OK[key] = True
        except Exception as exc:
            logger.warning(f"calibrate: vetting catalog {path} unreadable ({exc}) — "
                           f"calibrating without vetting; re-run the vet step")
            _VET_OK[key] = False
    return _VET_OK[key]


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
        before = Path(out_cal).stat().st_mtime_ns if Path(out_cal).exists() else None
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calib_catalog(str(ref_csv), str(in_ldac), str(out_cal), img_kind,
                          vet_catalog=vet_catalog, **extra_kw)
        if not Path(out_cal).exists():
            return str(out_cal), False, "too few calibrators — calib_catalog wrote nothing"
        if before is not None and Path(out_cal).stat().st_mtime_ns == before:
            # Too few calibrators on this pass: the previous calibration is kept
            return str(out_cal), True, "too few calibrators — previous calibration kept"
        return str(out_cal), True, ""
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
    target_match_radius: float = 1.0,
    save_residuals: bool = False,
    suffix: str = "",
) -> int:
    """
    Apply per-epoch photometric calibration (linear ZP → 3σ clip → 2D polynomial →
    flatfield) to every per-epoch photometry catalog.

    Reads from  PhotCatalogs{suffix}/{field}/{fc}/{ccd}/{qid}/*_phot.fits
    Writes to   Calibrated{suffix}/{field}/{fc}/{ccd}/{qid}/*_cal.fits
    """
    from aperture_photometry import CATALOG_DIRS, CATALOG_SUFFIX
    cat_dir  = base_dir / "Catalogs"
    sex_root = base_dir / CATALOG_DIRS[suffix]
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

        ldac_files = sorted(sex_subdir.glob(f"*{CATALOG_SUFFIX}"))
        if not ldac_files:
            logger.debug(f"calibrate: no LDAC catalogs in {sex_subdir}")
            continue

        for ldac in ldac_files:
            out_cal = cal_subdir / ldac.name.replace(CATALOG_SUFFIX, "_cal.fits")
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
            if vet_cat is not None and not _vet_readable(vet_cat):
                vet_cat = None

            resid_out = None
            if save_residuals:
                resid_dir = (base_dir / f"FlatfieldResiduals{suffix}"
                             / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_))
                resid_dir.mkdir(parents=True, exist_ok=True)
                resid_out = resid_dir / ldac.name.replace(CATALOG_SUFFIX, "_resid.npz")

            extra_kw = dict(
                poly_degree=poly_degree, flatfield=flatfield,
                target_ra=target_ra, target_dec=target_dec,
                target_match_radius=target_match_radius,
                residuals_out=str(resid_out) if resid_out else None,
            )
            tasks.append((ref_csv, ldac, out_cal, "SIM", vet_cat, extra_kw))

    if not tasks:
        logger.info("calibrate: all catalogs already calibrated (use --force to redo)")
        return 0

    logger.info(f"calibrate: {len(tasks)} epochs with {workers} workers")
    n_done = n_sparse = n_fail = n_kept = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_calibrate_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            out_cal, ok, msg = fut.result()
            if ok and msg:
                n_kept += 1
                logger.warning(f"calibrate: {Path(out_cal).name}: {msg}")
            elif ok:
                n_done += 1
            elif "too few" in msg:
                n_sparse += 1
            else:
                n_fail += 1
                logger.warning(f"calibrate failed: {Path(out_cal).name}: {msg}")
            if i % 50 == 0 or i == len(tasks):
                logger.info(f"calibrate: {i}/{len(tasks)} — "
                            f"{n_done} done, {n_sparse} sparse, {n_fail} failed")

    logger.info(f"calibrate: {n_done} written, {n_sparse} sparse, {n_fail} failures"
                + (f", {n_kept} kept from the previous pass (too few calibrators now)"
                   if n_kept else ""))
    if n_kept:
        logger.warning(f"calibrate: {n_kept} epoch(s) keep their previous calibration")
    return n_done + n_kept


# ── Step: build spatial flatfield ─────────────────────────────────────────────

def _refined_edges(lo: float, hi: float, nbins: int, edge_split: int) -> np.ndarray:
    """Bin edges: `nbins` uniform interior bins, with the first and last bin on
    each axis subdivided into `edge_split` thinner bins. This produces elongated
    edge cells — full interior width *along* the edge but 1/edge_split as deep
    *perpendicular* to it — to resolve the steep vignetting gradient at the field
    boundary without starving the cells of sources. edge_split=1 → uniform grid.
    """
    w = (hi - lo) / nbins
    first = np.linspace(lo,     lo + w, edge_split + 1)
    last  = np.linspace(hi - w, hi,     edge_split + 1)
    mid   = np.linspace(lo + w, hi - w, nbins - 1)
    return np.unique(np.concatenate([first, mid, last]))


def step_build_flatfield(
    base_dir: Path, quadrants: list[dict],
    nbins: int = 20, min_count: int = 50,
    edge_split: int = 3,
    suffix: str = "",
) -> dict:
    """
    Stack per-epoch NPZ residual files from step_calibrate(save_residuals=True)
    and build a spatial flatfield per quadrant.

    The grid spans the full source range with subdivided outer bins (see
    _refined_edges) so the vignetted field edges are covered and resolved.

    Returns dict keyed by (field, fc, ccd, qid) → flatfield dict.
    """
    from scipy.stats import binned_statistic_2d
    from calib_catalogs import _unwrap_ra

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

        # post-polynomial calibrator residuals (dm_4) of every epoch
        all_ra, all_dec, all_dm, n_bad = [], [], [], 0
        for f in resid_files:
            try:
                with np.load(f) as d:
                    all_ra.append(d['ra_4']); all_dec.append(d['dec_4']); all_dm.append(d['dm_4'])
            except Exception as exc:
                n_bad += 1
                logger.warning(f"flatfield: skipping unreadable {Path(f).name}: {exc}")
        if not all_dm:
            logger.warning(f"flatfield: no readable residual files")
            continue

        all_ra  = np.concatenate(all_ra).astype(float)
        all_dec = np.concatenate(all_dec).astype(float)
        all_dm  = np.concatenate(all_dm).astype(float)

        finite = np.isfinite(all_dm) & (np.abs(all_dm) < 1.0)
        if finite.sum() < 100:
            logger.warning(f"flatfield: too few residuals ({finite.sum()})")
            continue

        all_ra, all_dec, all_dm = all_ra[finite], all_dec[finite], all_dm[finite]
        all_ra = _unwrap_ra(all_ra, all_ra[0])          # continuous across RA = 0/360
        # Full spatial range (was 1–99 pct): ref-pos positions are exact
        # reference-catalog coordinates — no astrometric outliers to clip — and the
        # extreme edge rows/columns are exactly where the vignetting correction is
        # needed. Outer bins are subdivided into elongated edge cells.
        ra_edges  = _refined_edges(all_ra.min(),  all_ra.max(),  nbins, edge_split)
        dec_edges = _refined_edges(all_dec.min(), all_dec.max(), nbins, edge_split)

        stat, ra_e, dec_e, _ = binned_statistic_2d(
            all_ra, all_dec, all_dm, statistic='median', bins=[ra_edges, dec_edges])
        nobs, _, _, _ = binned_statistic_2d(
            all_ra, all_dec, all_dm, statistic='count', bins=[ra_edges, dec_edges])

        global_med  = float(np.nanmedian(stat[nobs >= min_count]))
        stat_filled = np.where(nobs >= min_count, stat - global_med, 0.0)   # empty cell: no correction

        vals = stat[nobs >= min_count]
        rms  = float(np.std(vals)) * 1000 if len(vals) > 0 else np.nan
        pp   = float(np.nanmax(vals) - np.nanmin(vals)) * 1000 if len(vals) > 1 else np.nan
        logger.info(f"flatfield {field:06d}/{fc}/c{ccd:02d}/q{qid_}: "
                    f"RMS={rms:.1f} mmag  P-P={pp:.1f} mmag  "
                    f"N={finite.sum():,}  bins={int(np.sum(nobs>=min_count))}/{stat.size}")

        ff = dict(stat=stat_filled.astype(np.float32),
                  ra_edges=ra_e.astype(np.float64),
                  dec_edges=dec_e.astype(np.float64))

        out_path = _cal_quad_dir(base_dir, field, fc, ccd, qid_, suffix) / "flatfield.npz"
        with open(str(out_path) + ".tmp", "wb") as _fh:                 # atomic write
            np.savez(_fh, **ff, nobs=nobs.astype(np.int32))
        Path(str(out_path) + ".tmp").replace(out_path)
        logger.info(f"flatfield saved → {out_path}")

        flatfields[(field, fc, ccd, qid_)] = ff

    return flatfields
