"""
photometry.py
-------------
Reference-catalog step of the pipeline:
  step_make_catalog — build the per-quadrant reference CSV catalog (the source list for
                      forced photometry) from IRSA's refsexcat.fits + refimg.fits header.
The photometry itself is aperture_photometry.py.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def _catalog_zp_ok(csv: Path) -> bool:
    """False for a reference CSV whose MAGZP_REF is all NaN (no reference zero-point)
    — such a catalog must be rebuilt."""
    import pandas as pd
    try:
        zp = pd.to_numeric(pd.read_csv(csv, usecols=["MAGZP_REF"], nrows=5)["MAGZP_REF"],
                           errors="coerce")
        return bool(zp.notna().any())
    except Exception:
        return True


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
            if _catalog_zp_ok(out_csv):
                n_skip += 1
                continue
            logger.warning(f"{out_csv.name}: no reference zero-point (built without the "
                           f"reference image) — rebuilding")
        logger.info(f"Building reference catalog for {cat_name}")
        try:
            make_catalog(save_path=str(cat_dir), refcats=str(q["ref_dir"]))
            n_done += 1
        except Exception as exc:
            logger.error(f"make_catalog failed for {cat_name}: {exc}")

    logger.info(f"make_catalog: {n_done} catalogs built, {n_skip} already exist")
    return n_done
