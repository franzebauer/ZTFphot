"""
photometry_scipos.py
--------------------
Alternative pipeline steps using per-epoch science image sexcat positions:
  step_simulate_scipos  — build simulated detection images from per-epoch sexcat.fits
  step_sex_scipos       — run SExtractor with science-position simulated images
                          and reference ASSOC catalog at 1.5 arcsec radius

Output catalogs go to SExCatalogs_sci/ to keep separate from reference-position
products.  All downstream steps (calib_catalogs, calibrate, lightcurves) are
used unchanged; point them at SExCatalogs_sci/ via base_dir or a wrapper.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

_SEX_DIR = Path(__file__).parent / "SExtractor"
_ASSOC_RADIUS = "1.5"   # arcsec; wider than ref-pos approach (0.5) to allow for epoch-to-epoch position shifts


# ── Simulate from per-epoch science sexcat ────────────────────────────────────

def _simulate_scipos_one(args: tuple) -> tuple[str, bool, str]:
    diff_path, sexcat_path, sim_path = args
    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    try:
        from simulate_science import build_simulated_image
        build_simulated_image(
            source_img=str(diff_path),
            source_cat=str(sexcat_path),
            save_name=str(sim_path),
        )
        return (str(sim_path), True, "ok")
    except Exception as exc:
        return (str(sim_path), False, str(exc))


def step_simulate_scipos(
    base_dir: Path, quadrants: list[dict],
    workers: int = 4, force: bool = False,
    filefracdays: set | None = None,
) -> int:
    """Build simulated detection images from per-epoch science sexcat positions."""
    def _ffd(p: Path) -> str:
        return p.name.split("_")[1]

    tasks = []
    for q in quadrants:
        sci_dir = q["sci_dir"]
        for diff_path in sorted(sci_dir.glob("*_scimrefdiffimg.fits")):
            if filefracdays is not None and _ffd(diff_path) not in filefracdays:
                continue
            sexcat_path = diff_path.with_name(
                diff_path.name.replace("_scimrefdiffimg.fits", "_sexcat.fits"))
            if not sexcat_path.exists():
                logger.debug(f"sexcat missing: {sexcat_path.name} — skipping epoch")
                continue
            sim_path = diff_path.with_name(diff_path.stem + "_simulated_sci.fits")
            if sim_path.exists() and not force:
                continue
            tasks.append((diff_path, sexcat_path, sim_path))

    if not tasks:
        logger.info("simulate_scipos: all simulated images already exist")
        return 0

    logger.info(f"simulate_scipos: {len(tasks)} images with {workers} workers")
    n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_simulate_scipos_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            _, ok, msg = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                logger.warning(f"simulate_scipos failed: {msg}")
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"simulate_scipos: {i}/{len(tasks)} — {n_done} done, {n_fail} failed")

    logger.info(f"simulate_scipos: {n_done} built, {n_fail} failures")
    return n_done


# ── SExtractor with science positions + reference ASSOC ───────────────────────

def _sex_scipos_one(args: tuple) -> tuple[str, bool, str]:
    sim_path, diff_path, out_cat, sex_conf, sex_param, sex_nnw, verbose, assoc_path = args

    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from photometry import _sex_header_params, PHOT_APERTURES, DETECT_THRESH, ANALYSIS_THRESH, DETECT_MINAREA

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
        "-ASSOC_NAME",      str(assoc_path),
        "-ASSOC_DATA",      "1",
        "-ASSOC_PARAMS",    "2,3",
        "-ASSOCCOORD_TYPE", "WORLD",
        "-ASSOC_RADIUS",    _ASSOC_RADIUS,
        "-ASSOC_TYPE",      "NEAREST",
        "-ASSOCSELEC_TYPE", "MATCHED",
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


def step_sex_scipos(
    base_dir: Path, quadrants: list[dict],
    workers: int = 4, force: bool = False, verbose: bool = False,
    filefracdays: set | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
) -> int:
    """Run SExtractor using science-position simulated images + reference ASSOC catalog."""
    import sys
    _scripts = Path(__file__).parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from photometry import _write_assoc_catalog

    def _ffd(p: Path) -> str:
        return p.name.split("_")[1]

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

        cat_subdir = (base_dir / "SExCatalogs_sci"
                      / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_))

        ref_csv = (base_dir / "Catalogs"
                   / f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(REFERENCE)[OBJECTS].csv")
        assoc_path = (base_dir / "Catalogs"
                      / f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(ASSOC).cat")
        if ref_csv.exists():
            _write_assoc_catalog(ref_csv, assoc_path,
                                 target_ra=target_ra, target_dec=target_dec)
        else:
            logger.warning(f"Reference catalog missing for {field:06d}_{fc}_c{ccd:02d}_q{qid_} "
                           f"— skipping (run catalog step first)")
            continue

        for sim_path in sorted(sci_dir.glob("*_simulated_sci.fits")):
            if filefracdays is not None and _ffd(sim_path) not in filefracdays:
                continue
            diff_path = sim_path.with_name(
                sim_path.name.replace("_scimrefdiffimg_simulated_sci.fits", "_scimrefdiffimg.fits"))
            if not diff_path.exists():
                continue
            out_cat = cat_subdir / sim_path.name.replace("_simulated_sci.fits", "_sexout.fits")
            if out_cat.exists() and not force:
                continue
            tasks.append((sim_path, diff_path, out_cat,
                          sex_conf, sex_param, sex_nnw, verbose, assoc_path))

    if not tasks:
        logger.info("sex_scipos: all catalogs already exist")
        return 0

    logger.info(f"sex_scipos: {len(tasks)} epochs with {workers} workers")
    n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_sex_scipos_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            out_cat, ok, msg = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                logger.warning(f"sex_scipos failed: {Path(out_cat).name}: {msg}")
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"sex_scipos: {i}/{len(tasks)} — {n_done} done, {n_fail} failed")

    logger.info(f"sex_scipos: {n_done} catalogs built, {n_fail} failures")
    return n_done
