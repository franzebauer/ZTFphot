"""
photometry.py
-------------
Pipeline steps for image preparation and source detection:
  step_funpack   — decompress .fits.fz difference images
  step_make_catalog — build reference CSV catalogs from refsexcat.fits
  step_simulate  — build simulated detection images (PSFs at reference positions)
  step_sextractor — run SExtractor in dual-image mode
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits

logger = logging.getLogger(__name__)


def _ffd_from_path(p: Path) -> str:
    """Extract filefracday from a ZTF filename: ztf_{ffd}_{field}_..."""
    return p.name.split("_")[1]


# ── Constants ─────────────────────────────────────────────────────────────────

# Aperture diameters in pixels.
# Match original Rust pipeline (structs.rs: PHOT_APERTURES = [3,4,6,10])
# k=0→3px, k=1→4px (primary, stored as FLUX_4_TOT_AB), k=2→6px, k=3→10px
PHOT_APERTURES    = "3,4,6,10"
DETECT_THRESH     = 1.0
ANALYSIS_THRESH   = 1.0
DETECT_MINAREA    = 3
DEFAULT_PIXEL_SCALE = 1.01   # arcsec/pixel for ZTF

_SEX_DIR = Path(__file__).parent / "SExtractor"


# ── Step 0: funpack ───────────────────────────────────────────────────────────

def step_funpack(base_dir: Path, force: bool = False,
                 filefracdays: set | None = None) -> int:
    """Decompress .fits.fz files under base_dir/Science/.
    filefracdays: if given, only process files matching those epoch IDs."""
    sci_root = base_dir / "Science"
    fz_files = sorted(sci_root.rglob("*.fits.fz"))
    if filefracdays is not None:
        fz_files = [f for f in fz_files if _ffd_from_path(f) in filefracdays]

    if not fz_files:
        logger.info("No .fits.fz files found — nothing to funpack")
        return 0

    n_done = n_skip = 0
    for fz in fz_files:
        unpacked = fz.with_name(fz.name.replace(".fits.fz", ".fits"))
        if unpacked.exists() and not force:
            n_skip += 1
            continue
        result = subprocess.run(["funpack", "-D", str(fz)], capture_output=True)
        if result.returncode == 0:
            n_done += 1
            logger.debug(f"Funpacked: {fz.name}")
        else:
            logger.warning(f"funpack failed on {fz.name}: {result.stderr.decode().strip()}")

    logger.info(f"Funpack: {n_done} files unpacked, {n_skip} already exist")
    return n_done


# ── Step 1: reference CSV catalogs ───────────────────────────────────────────

def step_make_catalog(base_dir: Path, quadrants: list[dict], force: bool = False) -> int:
    """Build a reference CSV catalog for each quadrant from its refsexcat.fits."""
    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from make_catalog import make_catalog

    cat_dir = base_dir / "Catalogs"
    cat_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = 0
    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        cat_name = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"
        out_csv  = cat_dir / f"{cat_name}(REFERENCE)[OBJECTS].csv"
        if out_csv.exists() and not force:
            n_skip += 1
            continue
        logger.info(f"Building reference catalog for {cat_name}")
        try:
            make_catalog(save_path=str(cat_dir), refcats=str(q["ref_dir"]))
            n_done += 1
        except Exception as exc:
            logger.error(f"make_catalog failed for {cat_name}: {exc}")

    logger.info(f"make_catalog: {n_done} catalogs built, {n_skip} already exist")
    return n_done


# ── Step 2: simulated detection images ───────────────────────────────────────

def _write_assoc_catalog(ref_csv_path: Path, assoc_path: Path,
                         target_ra: float | None = None,
                         target_dec: float | None = None) -> None:
    """Write a SExtractor ASSOC catalog (world coords, 1-based object_index).

    Format per line: ``object_index RA Dec``
    Index is 1-based so that SExtractor's unmatched default of 0 is
    distinguishable from a real source.  If target_ra/target_dec is given and
    lies ≥3″ from every reference source, the target is appended as the last
    entry (index = N+1).
    """
    import numpy as np
    import pandas as pd
    ref = pd.read_csv(ref_csv_path)
    ra_col  = 'ALPHAWIN_J2000' if 'ALPHAWIN_J2000' in ref.columns else 'RA'
    dec_col = 'DELTAWIN_J2000' if 'DELTAWIN_J2000' in ref.columns else 'DEC'
    ra  = pd.to_numeric(ref[ra_col],  errors='coerce').values.astype(float)
    dec = pd.to_numeric(ref[dec_col], errors='coerce').values.astype(float)
    with open(assoc_path, 'w') as f:
        for i, (r, d) in enumerate(zip(ra, dec)):
            f.write(f"{i + 1} {r:.8f} {d:.8f}\n")
        if target_ra is not None and target_dec is not None:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            tgt = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
            valid = np.isfinite(ra) & np.isfinite(dec)
            if valid.sum() == 0 or tgt.separation(
                    SkyCoord(ra=ra[valid], dec=dec[valid], unit='deg')
            ).min().arcsec >= 3.0:
                f.write(f"{len(ref) + 1} {target_ra:.8f} {target_dec:.8f}\n")


def _simulate_one(args: tuple) -> tuple[str, bool, str]:
    """Worker function for parallel simulate step."""
    diff_path, refcat_path, sim_path, target_ra, target_dec = args
    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    try:
        from simulate_science import build_simulated_image
        build_simulated_image(
            source_img=str(diff_path),
            source_cat=str(refcat_path),
            save_name=str(sim_path),
            target_ra=target_ra,
            target_dec=target_dec,
        )
        return (str(sim_path), True, "ok")
    except Exception as exc:
        return (str(sim_path), False, str(exc))


def step_simulate(
    base_dir: Path, quadrants: list[dict],
    workers: int = 4, force: bool = False,
    filefracdays: set | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
) -> int:
    """Build a simulated detection image for each science epoch.
    filefracdays: if given, only process files matching those epoch IDs."""
    tasks = []
    for q in quadrants:
        sci_dir = q["sci_dir"]; ref_dir = q["ref_dir"]
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]

        refcat_path = ref_dir / f"ztf_{field:06d}_{fc}_c{ccd:02d}_q{qid_}_refsexcat.fits"
        if not refcat_path.exists():
            logger.warning(f"refsexcat not found: {refcat_path} — skipping quadrant")
            continue

        for diff_path in sorted(sci_dir.glob("*_scimrefdiffimg.fits")):
            if filefracdays is not None and _ffd_from_path(diff_path) not in filefracdays:
                continue
            sim_path = diff_path.with_name(diff_path.stem + "_simulated.fits")
            if sim_path.exists() and not force:
                continue
            tasks.append((diff_path, refcat_path, sim_path, target_ra, target_dec))

    if not tasks:
        logger.info("simulate: all simulated images already exist")
        return 0

    logger.info(f"simulate: {len(tasks)} images with {workers} workers")
    n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_simulate_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            _, ok, msg = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                logger.warning(f"simulate failed: {msg}")
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"simulate: {i}/{len(tasks)} — {n_done} done, {n_fail} failed")

    logger.info(f"simulate: {n_done} built, {n_fail} failures")
    return n_done


# ── Step 3: SExtractor ───────────────────────────────────────────────────────

def _sex_header_params(diff_path: Path) -> dict:
    """Read MAGZP, SATURATE, SEEING, PIXSCALE from a diff image header."""
    params = {"MAGZP": 25.0, "SATURATE": 50000.0,
              "SEEING": 2.5, "PIXSCALE": DEFAULT_PIXEL_SCALE}
    try:
        with fits.open(diff_path, memmap=False) as hdul:
            hdr = hdul[0].header
            for key in ["MAGZP", "SATURATE", "SEEING"]:
                if key in hdr:
                    params[key] = float(hdr[key])
            if "PIXSCALE" in hdr:
                params["PIXSCALE"] = abs(float(hdr["PIXSCALE"]))
            elif "CD1_1" in hdr:
                params["PIXSCALE"] = abs(float(hdr["CD1_1"])) * 3600.0
    except Exception as exc:
        logger.debug(f"Header read failed for {diff_path.name}: {exc} — using defaults")
    return params


def _sex_one(args: tuple) -> tuple[str, bool, str]:
    """Worker function for parallel SExtractor step."""
    sim_path, diff_path, out_cat, sex_conf, sex_param, sex_nnw, verbose, assoc_path = args

    hdr = _sex_header_params(diff_path)
    cmd = [
        "sex", f"{sim_path},{diff_path}",
        "-c",               str(sex_conf),
        "-CATALOG_NAME",    str(out_cat),
        "-CATALOG_TYPE",    "FITS_LDAC",
        "-PARAMETERS_NAME", str(sex_param),
        "-STARNNW_NAME",    str(sex_nnw),
        "-FILTER_NAME",     str(sex_conf.parent / "default.conv"),
        "-DETECT_THRESH",   str(DETECT_THRESH),
        "-ANALYSIS_THRESH", str(ANALYSIS_THRESH),
        "-DETECT_MINAREA",  str(DETECT_MINAREA),
        "-PHOT_APERTURES",  PHOT_APERTURES,
        "-MAG_ZEROPOINT",   str(hdr["MAGZP"]),
        "-SATUR_LEVEL",     str(hdr["SATURATE"]),
        "-SEEING_FWHM",     str(hdr["SEEING"]),
        "-PIXEL_SCALE",     str(hdr["PIXSCALE"]),
        "-VERBOSE_TYPE",    "FULL" if verbose else "QUIET",
    ]
    if assoc_path is not None and Path(assoc_path).exists():
        cmd += [
            "-ASSOC_NAME",       str(assoc_path),
            "-ASSOC_DATA",       "1",
            "-ASSOC_PARAMS",     "2,3",
            "-ASSOCCOORD_TYPE",  "WORLD",
            "-ASSOC_RADIUS",     "3.0",
            "-ASSOC_TYPE",       "NEAREST",
            "-ASSOCSELEC_TYPE",  "MATCHED",
        ]
    out_cat.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            return (str(out_cat), False,
                    result.stderr.decode(errors="replace").strip()[-300:])
        return (str(out_cat), True, "ok")
    except subprocess.TimeoutExpired:
        return (str(out_cat), False, "SExtractor timed out after 120s")
    except FileNotFoundError:
        return (str(out_cat), False,
                "'sex' not found. Install: mamba install -c conda-forge astromatic-source-extractor")


def step_sextractor(
    base_dir: Path, quadrants: list[dict],
    workers: int = 4, force: bool = False, verbose: bool = False,
    filefracdays: set | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
) -> int:
    """Run SExtractor dual-image mode for every epoch.
    filefracdays: if given, only process files matching those epoch IDs."""
    sex_conf  = _SEX_DIR / "clean.sex"
    sex_param = _SEX_DIR / "default.param"
    sex_nnw   = _SEX_DIR / "default.nnw"

    if not sex_conf.exists():
        raise FileNotFoundError(f"SExtractor config not found: {sex_conf}")

    tasks = []
    for q in quadrants:
        sci_dir = q["sci_dir"]
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]

        cat_subdir = (
            base_dir / "SExCatalogs"
            / f"{field:06d}"
            / fc / f"{ccd:02d}" / str(qid_)
        )

        # Build (or reuse) the per-quadrant ASSOC catalog
        ref_csv = (base_dir / "Catalogs"
                   / f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(REFERENCE)[OBJECTS].csv")
        assoc_path = (base_dir / "Catalogs"
                      / f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(ASSOC).cat")
        if ref_csv.exists():
            _write_assoc_catalog(ref_csv, assoc_path,
                                 target_ra=target_ra, target_dec=target_dec)
        else:
            assoc_path = None

        for sim_path in sorted(sci_dir.glob("*_simulated.fits")):
            if filefracdays is not None and _ffd_from_path(sim_path) not in filefracdays:
                continue
            diff_path = sim_path.with_name(
                sim_path.name.replace("_simulated.fits", ".fits"))
            if not diff_path.exists():
                continue
            out_cat = cat_subdir / sim_path.name.replace("_simulated.fits", "_sexout.fits")
            if out_cat.exists() and not force:
                continue
            tasks.append((sim_path, diff_path, out_cat,
                          sex_conf, sex_param, sex_nnw, verbose, assoc_path))

    if not tasks:
        logger.info("SExtractor: all catalogs already exist")
        return 0

    logger.info(f"SExtractor: {len(tasks)} epochs with {workers} workers")
    n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_sex_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            out_cat, ok, msg = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                logger.warning(f"sex failed: {Path(out_cat).name}: {msg}")
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"SExtractor: {i}/{len(tasks)} — {n_done} done, {n_fail} failed")

    logger.info(f"SExtractor: {n_done} catalogs built, {n_fail} failures")
    return n_done
