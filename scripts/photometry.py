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


def _catalog_zp_ok(csv: Path, refimg: Path | None = None) -> bool:
    """False for a reference CSV that must be rebuilt: MAGZP_REF all NaN (no reference
    zero-point), or different from the MAGZP keyword of `refimg` when that exists."""
    import pandas as pd
    try:
        zp = pd.to_numeric(pd.read_csv(csv, usecols=["MAGZP_REF"], nrows=5)["MAGZP_REF"],
                           errors="coerce").dropna()
        if zp.empty:
            return False
        if refimg is not None and refimg.exists():
            from astropy.io import fits
            hzp = fits.getheader(refimg).get("MAGZP")
            if hzp is not None and abs(float(zp.iloc[0]) - float(hzp)) > 1e-6:
                return False
        return True
    except Exception:                               # unreadable catalog: rebuild it
        return False


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
        refimg   = Path(q["ref_dir"]) / f"ztf_{cat_name}_refimg.fits"
        if out_csv.exists() and not force:
            if _catalog_zp_ok(out_csv, refimg):
                n_skip += 1
                continue
            logger.warning(f"{out_csv.name}: MAGZP_REF missing or not the reference "
                           f"image's MAGZP — rebuilding")
        logger.info(f"Building reference catalog for {cat_name}")
        try:
            n_done += make_catalog(save_path=str(cat_dir), refcats=str(q["ref_dir"]))
        except Exception as exc:
            logger.error(f"make_catalog failed for {cat_name}: {exc}")

    logger.info(f"make_catalog: {n_done} catalogs built, {n_skip} already exist")
    return n_done
