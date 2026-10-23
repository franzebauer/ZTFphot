"""
run_pipeline.py — ZTF difference-image photometry pipeline

Steps (run in order, or select with --steps):
    lookup       — query IRSA for field/epoch coverage and cache results
    download     — fetch science and reference images from IRSA
    catalog      — build reference CSV catalogs from refsexcat.fits
    phot         — forced aperture photometry at the reference positions (photutils)
    phot_sci     — forced aperture photometry at the epoch's science-catalog positions
                   (runs with --both / --scipos)
    vet          — flag variable/bad calibration stars by multi-epoch RMS
    calibrate    — linear ZP → 3σ clip → 2D polynomial → flatfield
    flatfield    — rebuild spatial flatfield from post-polynomial residuals
    lightcurves  — assemble per-object light curves from calibrated FITS
    merge        — cross-calibrate and merge multiple quadrants per band
    plots        — diagnostic plots (spatial, rms, precision, quality, light curves)

Usage:
    # Full run from scratch (steps 1–12):
    python run_pipeline.py --ra 330.34158 --dec 0.72143

    # Re-run calibration and plots on existing data:
    python run_pipeline.py --steps calibrate lightcurves plots \\
        --ra 330.34158 --dec 0.72143 --field 443 --band zg --ccdid 16 --qid 2 --force

    # Status check:
    python run_pipeline.py --status
"""

from __future__ import annotations
import argparse, logging, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*DataFrame concatenation with empty or all-NA entries.*")

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

logger = logging.getLogger(__name__)

ALL_STEPS = ["lookup", "download", "catalog", "phot", "phot_sci",
             "vet", "calibrate", "flatfield", "recalibrate", "lightcurves", "merge", "plots"]


def _print_status(base_dir: Path, quadrants: list[dict]) -> None:
    from shutil import get_terminal_size
    w = min(get_terminal_size().columns, 100)
    print("=" * w)
    print(f"{'Quadrant':<32} {'Diff imgs*':>10} {'PhotCats':>10} {'Calibrated':>12} {'LC':>4}")
    print("-" * w)
    for q in quadrants:
        f, fc, ccd, qid = q["field"], q["filtercode"], q["ccdid"], q["qid"]
        sci = q["sci_dir"]
        n_diff = len(list(sci.glob("*_scimrefdiffimg.fits")))
        from aperture_photometry import catalog_dir, CATALOG_SUFFIX
        sex_d  = catalog_dir(base_dir, "", f, fc, ccd, qid)
        n_sex  = len(list(sex_d.glob(f"*{CATALOG_SUFFIX}"))) if sex_d.exists() else 0
        cal_d  = base_dir / "Calibrated" / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid)
        n_cal  = len(list(cal_d.glob("*_cal.fits"))) if cal_d.exists() else 0
        lc     = base_dir / "LightCurves" / f"{f:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid}" / "lightcurves.parquet"
        print(f"{f:06d} {fc} ccd{ccd:02d} q{qid:<24} {n_diff:>10} {n_sex:>10} {n_cal:>12} {'✓' if lc.exists() else '–':>4}")
    print("=" * w)
    print("  * Diff imgs = 0 is expected after --purge-batch or --clean-up")


def _write_target_status(base_dir, quadrants, target_ra, target_dec, match_radius):
    """Classify why the target does/does not have photometry and write one CSV line
    to base_dir/target_status.csv (ra,dec,reason). Coarse reason, best outcome across
    quadrants: OK, LACKS_REFIMG_DETECTION, MATCHED_NEIGHBOR, NOT_RECOVERED, ON_MASK,
    OUTSIDE_FOOTPRINT, NO_EPOCHS.

    OK means the target has a light curve, i.e. at least one finite FLUX measurement —
    positive OR negative (a fainter-than-reference detection is still a detection).
    LACKS_REFIMG_DETECTION is OK but with no reference source (MAG_4_REF > 90), so its
    FLUX_4_TOT_uJy is bipolar *difference* flux, not total — treat with care (blends).
    NOT_RECOVERED means it was measurable ("painted": in frame, unmasked) in >= 1 epoch
    but no row reached the light curve.
    Aggregates the per-epoch target fate written under TargetStatus/ by the photometry
    step (survives --purge-batch), then upgrades to OK from the final light curve."""
    import json, numpy as np, pandas as pd

    matched = False
    n_in_frame = n_painted = n_epochs = 0
    for q in quadrants:
        tag = f"{q['field']:06d}_{q['filtercode']}_c{q['ccdid']:02d}_q{q['qid']}"
        sdir = base_dir / "TargetStatus" / tag
        for jf in (sdir.glob("*.json") if sdir.exists() else []):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            n_epochs   += 1
            matched     = matched or bool(d.get("matched"))
            n_in_frame += int(bool(d.get("in_frame")))
            n_painted  += int(bool(d.get("painted")))

    # OK if the target (real neighbour or injected catalog entry) is in a light curve.
    # Also record whether the target has a reference-image detection in any quadrant: a
    # MAG_4_REF NaN or > 90 (injected) means no reference source, so FLUX_4_TOT_uJy is
    # the (bipolar) *difference* flux, not a total — the user must treat it accordingly
    # (and watch for blends). All quadrants are checked.
    found = False
    tgt_has_ref = False
    for q in quadrants:
        _qd = (base_dir / "LightCurves" / f"{q['field']:06d}" / q['filtercode']
               / f"ccd{q['ccdid']:02d}" / f"q{q['qid']}")
        lc = _qd / "lightcurves.parquet"
        if not lc.exists():
            lc = _qd / "lightcurves_sci.parquet"         # sci-pos-only runs
        if not lc.exists():
            continue
        # A finite FLUX is a real measurement — including the negative-flux
        # (fainter-than-reference) detections the magnitude drops to NaN. Difference
        # photometry is bipolar, so "detected" must not require a positive magnitude.
        try:
            import pyarrow.parquet as _pq
            _cols = set(_pq.read_schema(lc).names)
            _meas = 'FLUX_4_TOT_uJy'
            _want = ['ALPHAWIN_REF', 'DELTAWIN_REF', _meas, 'object_index']
            if 'MAG_4_REF' in _cols:
                _want.append('MAG_4_REF')
            df = pd.read_parquet(lc, columns=_want)
        except Exception:
            continue
        pos = df.groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']].first().dropna()
        if pos.empty:
            continue
        sep = np.hypot((pd.to_numeric(pos['ALPHAWIN_REF'], errors='coerce') - target_ra)
                       * np.cos(np.radians(target_dec)),
                       pd.to_numeric(pos['DELTAWIN_REF'], errors='coerce') - target_dec) * 3600.0
        j = int(np.nanargmin(sep.values))
        if sep.values[j] <= match_radius:
            oi = pos.index[j]
            _trow = df.loc[df['object_index'] == oi]
            if np.isfinite(pd.to_numeric(_trow[_meas], errors='coerce')).any():
                found = True
                if 'MAG_4_REF' not in df.columns or (
                        pd.to_numeric(_trow['MAG_4_REF'], errors='coerce') < 90).any():
                    tgt_has_ref = True

    if found and not tgt_has_ref:
                           reason = "LACKS_REFIMG_DETECTION"  # has flux, but no reference source
    elif found:            reason = "OK"
    elif matched:          reason = "MATCHED_NEIGHBOR"
    elif n_painted > 0:    reason = "NOT_RECOVERED"   # painted but no measurement produced
    elif n_in_frame > 0:   reason = "ON_MASK"
    elif n_epochs > 0:     reason = "OUTSIDE_FOOTPRINT"
    else:                  reason = "NO_EPOCHS"

    out = base_dir / "target_status.csv"
    out.write_text(f"{target_ra},{target_dec},{reason}\n")
    logger.info(f"target status: {reason} "
                f"(epochs={n_epochs}, in_frame={n_in_frame}, painted={n_painted}) → {out}")
    return reason


def _filter_quadrants_by_wcs(
    base_dir: Path, quadrants: list[dict],
    target_ra: float, target_dec: float,
    margin: int = 3,
) -> list[dict]:
    """
    Drop quadrants whose refimg.fits WCS places the target outside the active
    detector area.  Handles the case where ztfquery returns a field/quadrant
    whose nominal grid footprint contains the target but the actual image does
    not (e.g. CCD gap, edge roll-off).

    margin : pixels to subtract from each edge before declaring out-of-bounds.
             Default 3 px ≈ the 4 px primary-aperture radius, so a source that is
             mostly on the detector is kept; edge artefacts within a few px of the
             boundary may still affect its photometry.
    """
    from astropy.wcs import WCS
    from astropy.io import fits as _fits

    kept = []
    for q in quadrants:
        field, fc = q["field"], q["filtercode"]
        ccd, qid_ = q["ccdid"], q["qid"]
        tag = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"
        refimg = q["ref_dir"] / f"ztf_{tag}_refimg.fits"
        if not refimg.exists():
            logger.debug(f"  {tag}: refimg not found — keeping quadrant (not yet downloaded)")
            kept.append(q)
            continue
        try:
            with _fits.open(refimg, memmap=False) as hdul:
                hdr = hdul[0].header
                wcs = WCS(hdr)
                naxis1 = int(hdr.get("NAXIS1", 0))
                naxis2 = int(hdr.get("NAXIS2", 0))
            x, y = wcs.world_to_pixel_values(target_ra, target_dec)
            if margin <= x <= naxis1 - margin and margin <= y <= naxis2 - margin:
                kept.append(q)
            else:
                logger.warning(
                    f"  {tag}: target pixel ({x:.0f}, {y:.0f}) outside active area "
                    f"[{margin}:{naxis1 - margin}, {margin}:{naxis2 - margin}] — dropping quadrant"
                )
        except Exception as exc:
            logger.warning(f"  {tag}: WCS check failed ({exc}) — keeping quadrant")
            kept.append(q)
    return kept


def _inject_targets_into_quadrants(base_dir: Path, quadrants: list[dict],
                                   transients: list, match_radius: float) -> None:
    """Augment each quadrant's reference catalog with the given transient sources
    that (a) lie inside the quadrant footprint and (b) are not already within
    match_radius of a reference source. Writes non-destructive *_refsexcat_augmented.fits;
    make_catalog prefers it automatically. Injected sources carry the
    reference-mag sentinel (MAG_APER=99 → MAG_4_REF > 90 in the light curves)."""
    from transient_catalog import augment_sexcat

    logger.info(f"inject-targets: {len(transients)} candidate source(s)")
    n_quad = 0
    for q in quadrants:
        f, fc, ccd, qid_ = q["field"], q["filtercode"], q["ccdid"], q["qid"]
        refcat = q["ref_dir"] / f"ztf_{f:06d}_{fc}_c{ccd:02d}_q{qid_}_refsexcat.fits"
        if not refcat.exists():
            continue
        diffs = sorted(q["sci_dir"].glob("*_scimrefdiffimg.fits"))
        refimg = refcat.with_name(refcat.name.replace("_refsexcat.fits", "_refimg.fits"))
        augment_sexcat(
            refsexcat_path=refcat,
            transients=transients,
            # footprint from a difference image, else the reference image (same WCS
            # footprint; the only image on disk before purge-batch downloads epochs)
            diff_img_path=(diffs[0] if diffs else refimg if refimg.exists() else None),
            output_path=refcat.with_name(refcat.stem + "_augmented.fits"),
            match_radius_arcsec=match_radius,
            footprint_filter=True,
        )
        n_quad += 1
    logger.info(f"inject-targets: augmented {n_quad} quadrant catalog(s)")


def _injection_pending(ref_csv: Path, transients: list, match_radius: float) -> bool:
    """True if a source to inject lies within the reference catalog's extent but has no
    catalog row (real or injected) within match_radius, i.e. the catalog must be rebuilt."""
    if not transients:
        return False
    import pandas as pd
    if not ref_csv.exists():
        return True
    ref = pd.read_csv(ref_csv, usecols=["ALPHAWIN_J2000", "DELTAWIN_J2000"])
    ra, dec = ref["ALPHAWIN_J2000"].to_numpy(float), ref["DELTAWIN_J2000"].to_numpy(float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return True
    ra, dec = ra[ok], dec[ok]
    for t in transients:
        dra = (((ra - t.ra) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(t.dec))
        inside = (dra.min() < 0 < dra.max()) and (dec.min() < t.dec < dec.max())
        if inside and np.hypot(dra, dec - t.dec).min() * 3600.0 >= match_radius:
            return True
    return False


def _epoch_cache_path(base_dir: Path, args) -> Path:
    """The epoch cache the lookup step writes for --ra/--dec/--bands."""
    from ztf_field_lookup import _cache_path
    return _cache_path(args.ra, args.dec, args.bands or ["g", "r", "i"], base_dir / "Epochs")


def _injection_list(args, base_dir: Path) -> list:
    """Sources to inject: the --ra/--dec target (unless --no-target), --grabTNSobjs and
    --inject-targets. Each becomes a reference-catalog row (MAG_APER = 99) wherever no
    reference source lies within --target-match-radius, so every target is an ordinary
    catalog source with a fixed index and several targets can share a base dir."""
    _transients = []
    if args.ra is not None and args.dec is not None and not args.no_target:
        from transient_catalog import TransientSource
        _transients.append(TransientSource(ra=float(args.ra), dec=float(args.dec),
                                           name="TARGET", source="target"))
    if args.grabTNSobjs:
        from transient_catalog import load_tns_catalog
        _transients += load_tns_catalog(
            args.tns_cred_file, args.tns_key_file,
            args.tns_cache_dir or (base_dir / "TNS"),
            force_download=args.tns_force_download)
    if args.inject_targets:
        from transient_catalog import load_user_catalog
        _transients += load_user_catalog(args.inject_targets)
    return _transients


def _warn_target_coverage(base_dir: Path, quadrants: list[dict],
                          target_ra: float, target_dec: float,
                          threshold: float = 30.0) -> None:
    """
    For each quadrant, log a WARNING if the nearest reference catalog source
    is more than `threshold` arcsec from the target — a sign the target falls
    on masked or edge pixels and forced photometry will silently fail.
    """
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    tgt = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    for q in quadrants:
        field, fc = q['field'], q['filtercode']
        ccd, qid_ = q['ccdid'], q['qid']
        tag = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"
        ref_csv = (base_dir / "Catalogs"
                   / f"{tag}(REFERENCE)[OBJECTS].csv")
        if not ref_csv.exists():
            continue
        ref = pd.read_csv(ref_csv)
        ra_col  = 'ALPHAWIN_J2000' if 'ALPHAWIN_J2000' in ref.columns else 'RA'
        dec_col = 'DELTAWIN_J2000' if 'DELTAWIN_J2000' in ref.columns else 'DEC'
        ra  = pd.to_numeric(ref[ra_col],  errors='coerce').values
        dec = pd.to_numeric(ref[dec_col], errors='coerce').values
        ok  = np.isfinite(ra) & np.isfinite(dec)
        if 'MAG_APER_4px' in ref.columns:            # real sources only, not injected ones
            ok &= ~(pd.to_numeric(ref['MAG_APER_4px'], errors='coerce').values > 90)
        if not ok.any():
            logger.warning(f"  {tag}: ref catalog empty — target coverage unknown")
            continue
        sep = float(tgt.separation(SkyCoord(ra=ra[ok]*u.deg, dec=dec[ok]*u.deg)).arcsec.min())
        if sep >= threshold:
            logger.warning(
                f"  {tag}: nearest ref source {sep:.1f}\" from target "
                f"(threshold {threshold:.0f}\") — target likely on masked/edge pixels; "
                f"forced photometry will produce no light curve")
        else:
            logger.info(f"  {tag}: nearest ref source {sep:.2f}\" from target — OK")


# Quadrants this run could not complete (reference products unavailable, no reference
# catalog): main() then exits with code 3. They are also logged to
# Epochs/deferred_quadrants.log, from which the light curves record them in their metadata
# (quadrants_deferred) for update_parquet to add later.
_INCOMPLETE: list[str] = []


def _defer_quadrant(base_dir: Path, field, fc, ccd, qid) -> None:
    """Record a quadrant not completed this run (_INCOMPLETE + deferred_quadrants.log)."""
    tag = f"{int(field):06d}_{fc}_c{int(ccd):02d}_q{int(qid)}"
    _INCOMPLETE.append(tag)
    log = base_dir / "Epochs" / "deferred_quadrants.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(tag + "\n")


class _QuadrantIncomplete(RuntimeError):
    """A quadrant cannot be completed this run (recorded in _INCOMPLETE)."""

_FFD_RE = __import__("re").compile(r"ztf_(\d+)_(\d{6})_(z[gri])_c(\d\d)_o_q(\d)_")


def _record_deferred(base_dir: Path, urls_or_keys) -> None:
    """Append epochs not measured this run (download failures / given up) to
    Epochs/deferred_epochs.log as `filefracday field filtercode ccdid qid`; the light
    curves report those still without a photometry catalog as epochs_deferred, which
    update_parquet --retry-deferred fills in later."""
    rows = set()
    for u in urls_or_keys:
        if isinstance(u, tuple):
            rows.add(u)
        elif (m := _FFD_RE.search(str(u))):
            rows.add((m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))))
    if rows:
        log = base_dir / "Epochs" / "deferred_epochs.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            for r in sorted(rows):
                f.write("\t".join(map(str, r)) + "\n")


def _refs_ok(ref_dir: Path, field, fc, ccd, qid) -> bool:
    """Both reference products (refimg, refsexcat) present and non-empty."""
    tag = f"ztf_{int(field):06d}_{fc}_c{int(ccd):02d}_q{int(qid)}"
    return all((Path(ref_dir) / f"{tag}_{s}").exists()
               and (Path(ref_dir) / f"{tag}_{s}").stat().st_size > 0
               for s in ("refimg.fits", "refsexcat.fits"))


# Deprecated step names, mapped to the current ones (None = ignored)
_DEPRECATED_STEPS = {"sex": "phot", "sex_sci": "phot_sci", "simulate": None, "simulate_sci": None}


def _normalise_steps(args) -> None:
    """Map deprecated step names (sex/sex_sci -> phot/phot_sci; simulate/simulate_sci
    dropped) with a warning, and run phot_sci only when a sci-pos variant is requested
    (--both / --scipos) unless it was listed explicitly."""
    explicit = any(a == "--steps" or a.startswith("--steps=") for a in sys.argv)
    steps = []
    for s in args.steps:
        if s in _DEPRECATED_STEPS:
            new = _DEPRECATED_STEPS[s]
            logger.warning(f"step '{s}' is deprecated" + (f" — use '{new}'" if new else
                           " — ignored"))
            s = new
        if s and s not in steps:
            steps.append(s)
    if not explicit:
        if not (args.both or args.scipos):
            steps = [s for s in steps if s != "phot_sci"]
        elif args.scipos and not args.both:
            steps = [s for s in steps if s != "phot"]
    args.steps = steps


def _photometry_steps(args):
    """(phot, phot_sci) step functions. With ref + sci requested, both position sets are
    measured in one pass over each image (the phot step writes the sci catalogs too) and
    the separate sci step is None."""
    from functools import partial
    from aperture_photometry import step_aperture, step_aperture_scipos
    kw = dict(bkg=args.aper_bkg, bkg_mesh=args.aper_bkg_mesh)
    both = "phot" in args.steps and "phot_sci" in args.steps
    ref_step = partial(step_aperture, also_sci=both,
                       assoc_radius_sci=args.assoc_radius_sci, **kw)
    sci_step = None if both else partial(step_aperture_scipos, **kw)
    return ref_step, sci_step


def _run_phot(steps, step_phot, step_phot_sci, base_dir, quads, args, filefracdays=None):
    """Run the requested photometry steps (phot / phot_sci) on `quads`."""
    common = dict(workers=args.workers, force=args.force, filefracdays=filefracdays,
                  target_ra=args.ra, target_dec=args.dec,
                  match_radius=args.target_match_radius)
    if "phot" in steps:
        step_phot(base_dir, quads, **common)
    if "phot_sci" in steps and step_phot_sci is not None:
        step_phot_sci(base_dir, quads, assoc_radius=args.assoc_radius_sci, **common)


def _run_purge_batch(base_dir: Path, epochs, quadrants: list[dict], args) -> None:
    """
    Run funpack → (catalog once) → photometry in batches of --purge-batch N,
    deleting all imaging products after each batch's photometry step.
    Reference products are deleted after the first batch (catalog no longer needs them).
    """
    import math
    import pandas as pd
    from download_coordinator import (download_all, purge_images, filter_epochs,
                                      load_permanent_404s)
    from photometry import step_make_catalog
    from aperture_photometry import catalog_ok, catalog_dir, catalog_name
    step_phot, step_phot_sci = _photometry_steps(args)

    N = args.purge_batch
    steps = set(args.steps)
    if not steps & {"phot", "phot_sci"}:
        logger.info("purge-batch without a photometry step: no epoch images are downloaded "
                    "(they would be deleted unused); only the reference catalogs are built")
    _transients = _injection_list(args, base_dir) if "catalog" in steps else []

    # Load epochs from cache if not already in memory (e.g. lookup not in steps)
    if epochs is None and args.ra is not None and args.dec is not None:
        cache_path = _epoch_cache_path(base_dir, args)
        if cache_path.exists():
            epochs = pd.read_parquet(cache_path)
            logger.info(f"purge-batch: loaded {len(epochs)} epochs from cache")
        else:
            logger.warning(f"purge-batch: no epoch cache found at {cache_path} — "
                           "will process existing files on disk only")

    # Pre-filter epochs once (avoid re-filtering inside download_all per batch)
    if epochs is not None and not epochs.empty:
        epochs = filter_epochs(
            epochs,
            skip_cautionary=args.skip_flagged,
            max_seeing=args.max_seeing,
            min_maglim=args.min_maglim,
            mjd_min=args.mjd_min,
            mjd_max=args.mjd_max,
            min_epochs_per_quad=args.min_epochs_per_quad,
        )

    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        tag = f"{field:06d}/{fc}/ccd{ccd:02d}/q{qid_}"

        # Epochs for this quadrant (from pre-filtered cache)
        q_epochs = pd.DataFrame()
        if epochs is not None and not epochs.empty:
            mask = (
                (epochs["field"].astype(int)    == field) &
                (epochs["filtercode"]            == fc)   &
                (epochs["ccdid"].astype(int)     == ccd)  &
                (epochs["qid"].astype(int)       == qid_)
            )
            q_epochs = epochs[mask].reset_index(drop=True)

        if q_epochs.empty:
            logger.warning(f"{tag}: no epochs in cache — skipping download, "
                           "processing existing files only")
            # Still run catalog + photometry on whatever is already on disk
            if "catalog" in steps:
                step_make_catalog(base_dir, [q], force=args.force)
            _run_phot(steps, step_phot, step_phot_sci, base_dir, [q], args)
            continue

        # Resume: a quadrant whose photometry is already complete (reference CSV and every
        # epoch's catalog; no pending injection) needs no reference download or check
        _wanted = [sfx for sfx, st in (("", "phot"), ("_sci", "phot_sci")) if st in steps]
        _q_csv = base_dir / "Catalogs" / f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(REFERENCE)[OBJECTS].csv"
        _complete = (not args.force and bool(_wanted) and _q_csv.exists()
                     and not _injection_pending(_q_csv, _transients, args.target_match_radius)
                     and all(catalog_ok(catalog_dir(base_dir, sfx, field, fc, ccd, qid_) / catalog_name(
                                 f"ztf_{int(float(v))}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}_scimrefdiffimg"))
                             for v in q_epochs["filefracday"] for sfx in _wanted))
        if _complete:
            logger.info(f"{tag}: photometry complete — reference products not needed")

        # Footprint gate: fetch only the (small) reference products first, then skip
        # the entire quadrant if the target falls outside the reference detector area
        # (chip edge/gap). This avoids downloading any of this quadrant's science
        # images — the costly part — for a target it can never photometer.
        if (not _complete and args.ra is not None and args.dec is not None
                and ("download" in steps or args.purge_batch)):
            download_all(q_epochs, base_dir=base_dir, bands=[fc[1:]],
                         max_workers=args.download_workers, products="ref",
                         retry_404=args.retry_404,
                         perm404_max_age_days=args.perm404_max_age)
            if not _filter_quadrants_by_wcs(base_dir, [q], args.ra, args.dec,
                                            margin=args.footprint_margin_px):
                logger.info(f"{tag}: target outside reference footprint — skipping quadrant")
                continue

        # (references are fetched up front only in target mode; otherwise with batch 1)
        if (not _complete and args.ra is not None and args.dec is not None
                and ("download" in steps or args.purge_batch)
                and not _refs_ok(q["ref_dir"], field, fc, ccd, qid_)):
            logger.warning(f"{tag}: reference products unavailable (download failed) — "
                           f"quadrant skipped this run; it is retried on the next run")
            _defer_quadrant(base_dir, field, fc, ccd, qid_)
            continue

        n_batches = math.ceil(len(q_epochs) / N)
        logger.info(f"─── purge-batch {tag}: {len(q_epochs)} epochs → "
                    f"{n_batches} batches of {N} ───")

        # injection needs only the reference products, so it precedes the catalog step;
        # it changes the source list, so the reference CSV is then rebuilt
        _injected = bool(_transients)
        if _injected:
            _inject_targets_into_quadrants(base_dir, [q], _transients, args.target_match_radius)
        cat_csv = (base_dir / "Catalogs" /
                   f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(REFERENCE)[OBJECTS].csv")
        catalog_done = cat_csv.exists() and not (args.force or _injected)
        if not steps & {"phot", "phot_sci"}:
            if "catalog" in steps:
                step_make_catalog(base_dir, [q], force=args.force or _injected)
            continue

        def _cat_path(ffd, suffix):
            stem = f"ztf_{ffd}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}_scimrefdiffimg"
            return catalog_dir(base_dir, suffix, field, fc, ccd, qid_) / catalog_name(stem)

        # Load permanent 404s so epochs with no IRSA diff image are treated as done
        # Recorded 404s still within --perm404-max-age (none with --retry-404)
        _perm404_urls = load_permanent_404s(base_dir, args.perm404_max_age,
                                            ignore=args.retry_404)

        def _epoch_done(ffd):
            needle = f"ztf_{ffd}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}_scimrefdiffimg"
            perm404 = any(needle in url for url in _perm404_urls)
            # catalog_ok, not exists(): a header-only/truncated catalog must not mark the
            # epoch done (it would never be redone)
            if "phot" in steps and not (catalog_ok(_cat_path(ffd, "")) or perm404):
                return False
            if "phot_sci" in steps:
                return catalog_ok(_cat_path(ffd, "_sci")) or perm404
            return True

        import re as _re
        _MAX_DL_RETRIES = 3          # discard an epoch after this many consecutive download-fail cycles

        def _process_batch(this_batch, is_first, label):
            """Download + photometry + purge one batch. Returns the set of ffds whose
            science download failed transiently (503/timeout), so the caller can carry
            them into the next cycle. Permanent 404s are NOT returned (never retried)."""
            nonlocal catalog_done
            ffds_set = {str(int(float(v))) for v in this_batch["filefracday"]}

            if not args.force and all(_epoch_done(f) for f in ffds_set):
                logger.info(f"  {label}: all photometry catalogs exist — skipping")
                # the reference catalog may still need (re)building (e.g. injection)
                # before this batch's purge deletes the reference products
                if "catalog" in steps and not catalog_done:
                    step_make_catalog(base_dir, [q], force=args.force or _injected)
                    catalog_done = cat_csv.exists()
                    if not catalog_done:
                        raise _QuadrantIncomplete("no reference catalog could be built")
                purge_images(base_dir, [q], sci=True, ref=is_first,
                             filefracdays=ffds_set, dry_run=args.dry_run)
                return set()

            logger.info(f"  {label}: {len(this_batch)} epochs")

            failed = set()
            if "download" in steps or args.purge_batch:
                band = fc[1:]   # zg→g, zr→r, zi→i
                # reference products come from the footprint gate (target mode) or, without
                # a target, with the first batch; later batches fetch science products only
                res = download_all(this_batch, base_dir=base_dir, bands=[band],
                                   products=("all" if is_first and (args.ra is None or args.dec is None)
                                             else "sci"),
                                   max_workers=args.download_workers,
                                   retry_404=args.retry_404,
                                   perm404_max_age_days=args.perm404_max_age)
                for url in res.get("failed_urls", []):     # transient failures only
                    m = _re.search(r"ztf_(\d+)_", url)
                    if m:
                        failed.add(m.group(1))

            if "catalog" in steps and not catalog_done:
                step_make_catalog(base_dir, [q], force=args.force or _injected)
                catalog_done = cat_csv.exists()
                if not catalog_done:             # e.g. a truncated refsexcat
                    raise _QuadrantIncomplete("no reference catalog could be built")

            _run_phot(steps, step_phot, step_phot_sci, base_dir, [q], args,
                      filefracdays=ffds_set)

            # Purge the measured epochs (sci products; refs after the first batch).
            # Downloaded but unmeasured ones (funpack / photometry failure) keep their
            # images and are retried in the next cycle, like download failures.
            measured = {f for f in ffds_set if _epoch_done(f)}
            failed |= ffds_set - measured
            purge_images(base_dir, [q],
                         sci=True, ref=is_first,
                         filefracdays=measured, dry_run=args.dry_run)
            return failed

        def _update_carry(this_batch, failed, fail_count):
            """Increment per-ffd failure counts, drop epochs that hit _MAX_DL_RETRIES,
            and return the DataFrame of epochs to retry in the next cycle."""
            keep = []
            for _, row in this_batch.iterrows():
                ffd = str(int(float(row["filefracday"])))
                if ffd in failed:
                    fail_count[ffd] = fail_count.get(ffd, 0) + 1
                    if fail_count[ffd] >= _MAX_DL_RETRIES:
                        logger.warning(f"  {field:06d}/{fc}/c{ccd:02d}/q{qid_} ffd {ffd}: "
                                       f"deferred after {_MAX_DL_RETRIES} failed cycles "
                                       f"(recorded; fill in later with update_parquet --retry-deferred)")
                        _record_deferred(base_dir, [(ffd, field, fc, ccd, qid_)])
                        purge_images(base_dir, [q], sci=True, ref=False,
                                     filefracdays={ffd}, dry_run=args.dry_run)
                    else:
                        keep.append(row)
                else:
                    fail_count.pop(ffd, None)   # succeeded (or already done): reset streak
            return pd.DataFrame(keep) if keep else q_epochs.iloc[0:0]

        # Carry-forward retry: a batch's transient download failures are prepended to the
        # NEXT batch (downloaded at the same rate), so IRSA gets time to recover between
        # attempts. An epoch is only given up after _MAX_DL_RETRIES cycles in a row.
        carry = q_epochs.iloc[0:0]
        fail_count: dict[str, int] = {}
        try:
            for bi, start in enumerate(range(0, len(q_epochs), N)):
                batch      = q_epochs.iloc[start:start + N]
                this_batch = pd.concat([carry, batch]).drop_duplicates(subset="filefracday")
                failed     = _process_batch(this_batch, is_first=(bi == 0),
                                            label=f"batch {bi+1}/{n_batches}"
                                                  + (f" (+{len(carry)} carried)" if len(carry) else ""))
                carry = _update_carry(this_batch, failed, fail_count)

            # Drain any epochs still failing from the final batches, giving each its
            # remaining retry cycles (back to back) before it is deferred.
            drain = 0
            while len(carry) > 0:
                drain += 1
                failed = _process_batch(carry, is_first=False, label=f"retry-drain {drain}")
                carry = _update_carry(carry, failed, fail_count)
        except _QuadrantIncomplete as exc:
            logger.error(f"{tag}: {exc} — quadrant not completed this run")
            _defer_quadrant(base_dir, field, fc, ccd, qid_)
            purge_images(base_dir, [q], sci=True, ref=False, dry_run=args.dry_run)


RUN_LOG = Path("Epochs") / "ztfphot_runs.jsonl"


def _git_info() -> dict:
    """Commit of the ZTFphot checkout running this script (None outside git)."""
    import subprocess
    repo = _SCRIPTS.parent
    def git(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None}
    return {"commit": commit, "commit_date": git("show", "-s", "--format=%cI", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _record_run(base_dir: Path, args) -> dict:
    """Describe this invocation (code version, time, full command line and options,
    package versions) and append it to <base>/Epochs/ztfphot_runs.jsonl. The log keeps
    the settings of earlier invocations — notably the download-time epoch cuts, which a
    later calibrate/lightcurves-only run does not repeat on its command line."""
    import json, platform
    from datetime import datetime, timezone
    from importlib import metadata as _md
    vers = {}
    for pkg in ("numpy", "astropy", "photutils", "scipy", "pandas", "pyarrow", "ztfquery"):
        try:
            vers[pkg] = _md.version(pkg)
        except Exception:
            pass
    rec = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ztfphot": _git_info(),
        "argv": sys.argv,
        "args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                 "base_dir": str(Path(base_dir).resolve())},      # absolute: image reuse
        "python": platform.python_version(),
        "packages": vers,
        "host": platform.node(),
    }
    log = base_dir / RUN_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def _latest_download_run(base_dir: Path, args) -> dict | None:
    """Most recent logged invocation that downloaded for this target (same --ra/--dec,
    or the same --field/--ccd/--qid in quadrant mode) — its epoch cuts define which
    epochs are in the light curves."""
    import json
    log = base_dir / RUN_LOG
    if not log.exists():
        return None
    def same_target(a):
        if args.ra is not None and args.dec is not None:
            return (a.get("ra") is not None and a.get("dec") is not None
                    and abs(a["ra"] - args.ra) < 1e-5 and abs(a["dec"] - args.dec) < 1e-5)
        return all(a.get(k) == getattr(args, k, None) for k in ("field", "ccdid", "qid"))
    best = None
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        a = rec.get("args", {})
        if ("download" in (a.get("steps") or []) or a.get("purge_batch")) and same_target(a):
            best = rec
    return best


def main() -> None:
    p = argparse.ArgumentParser(
        description="ZTF difference-image photometry pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir",  type=Path, default=Path("data"),
                   help="Data directory (default: ./data)")
    p.add_argument("--ra",        type=float, default=None,
                   help="Target RA (deg) — required for lookup/download steps")
    p.add_argument("--dec",       type=float, default=None,
                   help="Target Dec (deg) — required for lookup/download steps")
    p.add_argument("--bands",     nargs="+", default=None,
                   help="Bands to process: g r i (default: all present)")
    p.add_argument("--field",     type=int,   default=None)
    p.add_argument("--ccdid",     type=int,   default=None)
    p.add_argument("--qid",       type=int,   default=None)
    p.add_argument("--steps",     nargs="+",  default=ALL_STEPS, choices=ALL_STEPS + list(_DEPRECATED_STEPS),
                   metavar="STEP")
    p.add_argument("--workers",          type=int, default=4,
                   help="Parallel workers for photometry/calibrate (default: 4)")
    p.add_argument("--download-workers", type=int, default=50,
                   help="Parallel threads for image downloads (default: 50)")
    p.add_argument("--retry-404", action="store_true",
                   help="Ignore Epochs/permanent_404s.log and request every file again")
    p.add_argument("--perm404-max-age", type=float, default=30.0, metavar="DAYS",
                   help="Recorded 404s older than this are retried (default 30; 0 = never "
                        "expire). Entries without a timestamp (old logs) count as expired.")
    p.add_argument("--force",     action="store_true")
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--status",    action="store_true", help="Print file-existence summary and exit")
    p.add_argument("--dry-run",   action="store_true")
    # Calibration
    p.add_argument("--vet-catalog",  type=Path, default=None)
    p.add_argument("--poly-degree",       type=int,  default=2)
    p.add_argument("--merge-mag-bin",        type=float, default=0.1)
    p.add_argument("--ff-bins",              type=int,   default=20)
    p.add_argument("--ff-min-count",         type=int,   default=50)
    p.add_argument("--ff-edge-split",        type=int,   default=3,
                   help="Subdivide the outer flatfield bin on each axis into this "
                        "many thinner bins (elongated edge cells; 1 = uniform grid)")
    p.add_argument("--target-match-radius",  type=float, default=1.0, metavar="ARCSEC",
                   help="Target association radius (arcsec, default 1.0): the target is "
                        "injected as a new source if no catalog source lies within this "
                        "radius, and the same radius is used to match the input position "
                        "to a source in the calibrated catalog. One radius for both, so "
                        "a target with no nearby reference source is injected and then "
                        "recovered.")
    p.add_argument("--footprint-margin-px",  type=int, default=3, metavar="PIX",
                   help="Reference-image edge margin (pixels, default 3 ≈ the 4 px "
                        "aperture radius): a quadrant is dropped before science download "
                        "if the target lands within this many pixels of the refimg edge "
                        "(or outside it). Larger = more conservative against edge artefacts.")
    p.add_argument("--assoc-radius-sci",     type=float, default=1.5, metavar="ARCSEC",
                   help="sci-pos photometry: radius tying each science-catalog source to its "
                        "nearest reference source (default 1.5; allows epoch-to-epoch "
                        "science-centroid shifts)")
    p.add_argument("--aper-bkg", choices=["box", "local", "mesh", "none"], default="box",
                   help="Photometry background: 'box' (default) = 3σ-clipped mean "
                        "in a square annulus (1.58×SEEING .. +35 px) via integral images; "
                        "'local' = clipped median in the equivalent circular annulus "
                        "(same precision, ~2.5x slower photometry), "
                        "'mesh' = 64 px "
                        "Background2D level, 'none' = no subtraction")
    p.add_argument("--aper-bkg-mesh", type=int, default=64, metavar="PX",
                   help="Mesh box size (px) for --aper-bkg mesh (default 64)")
    p.add_argument("--merge-match-radius",   type=float, default=1.5, metavar="ARCSEC",
                   help="Cross-match radius (arcsec) for common sources between "
                        "overlapping quadrants in the merge step (default 1.5)")
    p.add_argument("--inject-targets",       type=Path, default=None, metavar="CSV",
                   help="CSV of targets (ra,dec[,name]) to inject as sources absent "
                        "from the reference catalog. During the catalog step, targets "
                        "inside each quadrant footprint and farther than "
                        "--target-match-radius from any reference source are added to a "
                        "non-destructive augmented catalog that the photometry then uses. "
                        "Injected sources carry MAG_4_REF > 90 in the light curves.")
    p.add_argument("--grabTNSobjs",          action="store_true",
                   help="Inject real transients from the TNS public bulk catalog that "
                        "overlap the processed quadrants (footprint-filtered per "
                        "quadrant). Same injection path as --inject-targets; may be "
                        "combined with it. Requires --tns-cred-file and --tns-key-file.")
    p.add_argument("--tns-cred-file",        type=Path, default=None, metavar="JSON",
                   help="TNS marker credentials JSON (keys: tns_id, type, name)")
    p.add_argument("--tns-key-file",         type=Path, default=None, metavar="FILE",
                   help="Text file containing the TNS API key")
    p.add_argument("--tns-cache-dir",        type=Path, default=None, metavar="DIR",
                   help="Where to cache the TNS bulk catalog (default: <base-dir>/TNS)")
    p.add_argument("--tns-force-download",   action="store_true",
                   help="Re-download the TNS bulk catalog even if a cached copy exists")
    # Download filters (used when "download" is in steps)
    p.add_argument("--max-seeing",          type=float, default=None, metavar="ARCSEC")
    p.add_argument("--min-maglim",          type=float, default=None, metavar="MAG")
    p.add_argument("--skip-flagged",        action="store_true")
    p.add_argument("--mjd-min",             type=float, default=None)
    p.add_argument("--mjd-max",             type=float, default=None)
    p.add_argument("--min-epochs-per-quad", type=int,   default=None)
    # Purge utility
    p.add_argument("--purge-hard-reject", action="store_true",
                   help="Delete on-disk files for hard-rejected epochs and exit")
    p.add_argument("--epochs-parquet",    type=Path, default=None)
    # Low-disk mode
    p.add_argument("--purge-batch",  type=int, default=None, metavar="N",
                   help="Process N epochs at a time, deleting imaging products after each "
                        "batch's photometry. Keeps only the photometry catalogs on disk. "
                        "Use --dry-run to preview what would be deleted.")
    p.add_argument("--scipos",        action="store_true",
                   help="Use the epoch's science-catalog (sexcat) positions instead of the "
                        "reference positions. Writes PhotCatalogs_sci/, Calibrated_sci/, "
                        "lightcurves_sci.parquet.")
    p.add_argument("--both",          action="store_true",
                   help="Run calibrate/lightcurves/merge/plots for both ref-pos and sci-pos. "
                        "Both position sets are measured in one pass over each image.")
    p.add_argument("--clean-up",     action="store_true",
                   help="Delete all imaging products (Science/, Reference/) for discovered "
                        "quadrants and exit. Safe once the sex step is complete.")
    p.add_argument("--no-target",   action="store_true",
                   help="Suppress target-source plots (lightcurves, precision target marker). "
                        "Use when processing a whole quadrant without a specific target.")
    p.add_argument("--lc-columns", choices=["slim", "full"], default="slim",
                   help="photometric columns of the light-curve parquets: slim (default) = "
                        "4 px calibrated magnitude and μJy flux (aperture-corrected to "
                        "10 px), raw 4 px difference flux, APCORR410, MAG_4_REF; full = "
                        "slim + the same for the 3, 6 and 10 px apertures")
    p.add_argument("--lc-bin-days", type=float, default=50.0,
                   help="Bin width (days) for the binned light-curve overlay in the "
                        "lightcurves plot (default: 50).")
    args = p.parse_args()

    if args.grabTNSobjs and (args.tns_cred_file is None or args.tns_key_file is None):
        p.error("--grabTNSobjs requires --tns-cred-file and --tns-key-file")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    base_dir = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Deprecated step names; phot_sci only with a sci-pos variant; --both with an explicit
    # 'phot' also measures sci-pos (same images, one pass)
    _normalise_steps(args)
    if args.both and "phot" in args.steps and "phot_sci" not in args.steps:
        args.steps.insert(args.steps.index("phot") + 1, "phot_sci")

    # Rename SExCatalogs*/ *_sexout.fits trees to PhotCatalogs*/ *_phot.fits
    from aperture_photometry import migrate_legacy_layout
    migrate_legacy_layout(base_dir)

    from download_coordinator import find_quadrants, download_all, purge_hard_reject

    # ── Purge utility ─────────────────────────────────────────────────────────
    if args.purge_hard_reject:
        if not args.epochs_parquet:
            sys.exit("--purge-hard-reject requires --epochs-parquet")
        purge_hard_reject(base_dir, args.epochs_parquet.resolve(), dry_run=args.dry_run)
        return

    steps = args.steps
    bands = args.bands or ["g", "r", "i"]
    epochs = None

    # Provenance: log this invocation; light-curve parquets embed it, the latest
    # download-time invocation for this target (its epoch cuts), and per-quadrant
    # epoch summaries computed from the data.
    import json as _json
    # --status / --clean-up / --dry-run produce nothing and are not recorded (a
    # recorded run counts as the target's latest download run)
    _run_rec = ({} if (args.status or args.clean_up or args.dry_run)
                else _record_run(base_dir, args))

    # ── Step: lookup ──────────────────────────────────────────────────────────
    if "lookup" in steps:
        if args.ra is None or args.dec is None:
            logger.warning("lookup: --ra and --dec required — skipping")
        else:
            logger.info("─── lookup ───")
            from ztf_field_lookup import lookup_target
            plot_dir = base_dir / "Plots" / f"{args.ra:.5f}_{args.dec:+.5f}"
            plot_dir.mkdir(parents=True, exist_ok=True)
            epochs = lookup_target(
                ra=args.ra, dec=args.dec, bands=bands,
                cache_dir=base_dir / "Epochs",
                plot_out=plot_dir / "coverage.png",
                min_maglim=args.min_maglim,
                max_seeing=args.max_seeing,
            )
            if epochs is not None and epochs.empty:
                # IRSA answered with no epochs: no coverage, or an IRSA-side problem. Not
                # cached, so a later run queries again (README: re-running IRSA_EMPTY).
                (base_dir / "target_status.csv").write_text(f"{args.ra},{args.dec},IRSA_EMPTY\n")
                logger.warning("lookup: IRSA returned no epochs for this position — "
                               "target status IRSA_EMPTY")
                return

    # ── Step: download ────────────────────────────────────────────────────────
    if "download" in steps and not args.purge_batch:
        if args.ra is None or args.dec is None:
            logger.warning("download: --ra and --dec required — skipping")
        else:
            if epochs is None:
                # Load from cache (lookup already ran previously)
                import pandas as pd
                cache_path = _epoch_cache_path(base_dir, args)
                if not cache_path.exists():
                    sys.exit(f"Epochs cache not found: {cache_path}\nRun the lookup step first.")
                epochs = pd.read_parquet(cache_path)

            _dl_kw = dict(
                base_dir=base_dir, bands=bands,
                max_workers=args.download_workers,
                skip_flagged=args.skip_flagged,
                max_seeing=args.max_seeing,
                min_maglim=args.min_maglim,
                mjd_min=args.mjd_min,
                mjd_max=args.mjd_max,
                min_epochs_per_quad=args.min_epochs_per_quad,
                retry_404=args.retry_404,
                perm404_max_age_days=args.perm404_max_age,
            )
            logger.info("─── download: reference products ───")
            download_all(epochs, products="ref", **_dl_kw)

            # Drop quadrants whose target falls outside the *reference* footprint
            # BEFORE the costly science download. ztfquery returns every quadrant
            # near the target; some do not actually contain it (chip edge / gap), and
            # their science images are wasted downloads + spurious photometry.
            if args.ra is not None and args.dec is not None:
                _uq = epochs[["field", "filtercode", "ccdid", "qid"]].drop_duplicates()
                _quads = [dict(field=int(r.field), filtercode=str(r.filtercode),
                               ccdid=int(r.ccdid), qid=int(r.qid),
                               ref_dir=(base_dir / "Reference" / f"{int(r.field):06d}"
                                        / str(r.filtercode) / f"ccd{int(r.ccdid):02d}"
                                        / f"q{int(r.qid)}"))
                          for r in _uq.itertuples(index=False)]
                _kept = _filter_quadrants_by_wcs(base_dir, _quads, args.ra, args.dec,
                                                 margin=args.footprint_margin_px)
                _keep = {(q["field"], q["filtercode"], q["ccdid"], q["qid"]) for q in _kept}
                if len(_kept) < len(_quads):
                    _mask = [(int(f), str(fc), int(c), int(qd)) in _keep
                             for f, fc, c, qd in zip(epochs["field"], epochs["filtercode"],
                                                     epochs["ccdid"], epochs["qid"])]
                    epochs = epochs[_mask]
                    logger.info(f"footprint filter: dropped {len(_quads) - len(_kept)} "
                                f"out-of-footprint quadrant(s) before science download")

            # Quadrants whose reference products could not be fetched cannot be
            # calibrated: skip their science download this run (retried next run).
            _qk = epochs[["field", "filtercode", "ccdid", "qid"]].drop_duplicates()
            _noref = {(int(r.field), str(r.filtercode), int(r.ccdid), int(r.qid))
                      for r in _qk.itertuples(index=False)
                      if not _refs_ok(base_dir / "Reference" / f"{int(r.field):06d}" / str(r.filtercode)
                                      / f"ccd{int(r.ccdid):02d}" / f"q{int(r.qid)}",
                                      int(r.field), str(r.filtercode), int(r.ccdid), int(r.qid))}
            if _noref:
                logger.warning(f"reference products unavailable for {sorted(_noref)} — "
                               f"skipping their science download this run (retried next run)")
                for _q in sorted(_noref):
                    _defer_quadrant(base_dir, *_q)
                epochs = epochs[np.array([(int(f), str(fc), int(c), int(qd)) not in _noref
                                          for f, fc, c, qd in zip(epochs["field"], epochs["filtercode"],
                                                                  epochs["ccdid"], epochs["qid"])],
                                         dtype=bool)]

            logger.info("─── download: science products ───")
            _res = download_all(epochs, products="sci", **_dl_kw)
            _record_deferred(base_dir, [u for u in _res.get("failed_urls", [])
                                        if "scimrefdiffimg" in u])

    # ── Discover quadrants on disk ────────────────────────────────────────────
    quadrants = find_quadrants(base_dir, bands=args.bands,
                               field=args.field, ccdid=args.ccdid, qid=args.qid)

    # Filter to fields belonging to this target using the epoch cache
    if args.ra is not None and args.dec is not None:
        import pandas as _pd
        cache_path = _epoch_cache_path(base_dir, args)
        if cache_path.exists():
            _cache     = _pd.read_parquet(cache_path)
            _fields    = set(_cache["field"].dropna().astype(int).tolist())
            _before    = len(quadrants)
            quadrants  = [q for q in quadrants if q["field"] in _fields]
            logger.info(f"Epoch cache filter: {_before} → {len(quadrants)} quadrants "
                        f"(keeping fields {sorted(_fields)})")
        else:
            logger.warning(f"No epoch cache found at {cache_path} — using all quadrants on disk")

    if args.ra is not None and args.dec is not None:
        quadrants = _filter_quadrants_by_wcs(base_dir, quadrants, args.ra, args.dec,
                                             margin=args.footprint_margin_px)

    if args.status:
        _print_status(base_dir, quadrants)
        return

    # ── Clean-up: delete all imaging products and exit ────────────────────────
    if args.clean_up:
        logger.info("─── clean-up: deleting imaging products ───")
        from download_coordinator import purge_images
        purge_images(base_dir, quadrants, sci=True, ref=True, dry_run=args.dry_run)
        return

    # ── purge-batch: supplement on-disk quadrants with any missing from epoch cache ─
    # find_quadrants() only finds files already on disk. With --purge-batch some or
    # all quadrants may not have been downloaded yet. Always merge the cache list so
    # every expected quadrant is processed (whether partially done or brand-new).
    if args.purge_batch and epochs is None and args.ra is not None and args.dec is not None:
        import pandas as _pd2
        _cp = _epoch_cache_path(base_dir, args)
        if _cp.exists():
            epochs = _pd2.read_parquet(_cp)
    if args.purge_batch and epochs is not None:
        _on_disk = {(q["field"], q["filtercode"], q["ccdid"], q["qid"]) for q in quadrants}
        _added: list = []
        for _, _row in epochs[["field", "filtercode", "ccdid", "qid"]].drop_duplicates().iterrows():
            _key = (int(_row["field"]), str(_row["filtercode"]),
                    int(_row["ccdid"]),  int(_row["qid"]))
            _f, _fc, _ccd, _qid = _key
            if ((args.field is not None and _f != args.field)
                    or (args.ccdid is not None and _ccd != args.ccdid)
                    or (args.qid is not None and _qid != args.qid)
                    or (args.bands and _fc[1:] not in {x.lstrip("z") for x in args.bands})):
                continue
            if _key not in _on_disk:
                _added.append({
                    "field": _f, "filtercode": _fc, "ccdid": _ccd, "qid": _qid,
                    "ref_dir": (base_dir / "Reference"
                                / f"{_f:06d}" / _fc / f"ccd{_ccd:02d}" / f"q{_qid}"),
                    "sci_dir": (base_dir / "Science"
                                / f"{_f:06d}" / _fc / f"ccd{_ccd:02d}" / f"q{_qid}"),
                })
        if _added:
            quadrants.extend(_added)
            logger.info(f"purge-batch: added {len(_added)} quadrant(s) from epoch cache "
                        f"not yet on disk: "
                        + ", ".join(f"{q['filtercode']}" for q in _added))

    remaining = [s for s in steps if s not in ("lookup", "download")]
    if remaining and not quadrants and not args.purge_batch:
        sys.exit("No quadrants found on disk. Run lookup and download first.")

    logger.info(f"Quadrants ({len(quadrants)}):")
    for q in quadrants:
        logger.info(f"  {q['field']:06d} {q['filtercode']} ccd{q['ccdid']:02d} q{q['qid']}")

    # ── Import step modules ───────────────────────────────────────────────────
    from photometry  import step_make_catalog
    step_phot, step_phot_sci = _photometry_steps(args)
    from calibrate   import step_vet, step_calibrate, step_build_flatfield
    from lightcurves import step_lightcurves, step_merge

    t0 = time.time()

    # --both → both suffixes; --scipos → sci only; neither → ref only
    if args.both:
        suffixes = ["", "_sci"]
    elif args.scipos:
        suffixes = ["_sci"]
    else:
        suffixes = [""]

    if args.purge_batch and any(s in steps for s in ("catalog", "phot", "phot_sci")):
        _run_purge_batch(base_dir, epochs, quadrants, args)
    else:
        if "catalog"    in steps:
            _injected = False
            _transients = _injection_list(args, base_dir)
            if _transients:
                _inject_targets_into_quadrants(base_dir, quadrants, _transients,
                                               args.target_match_radius)
                _injected = True
            # injecting changes the source list, so the reference CSV must be rebuilt
            step_make_catalog(base_dir, quadrants, force=args.force or _injected)
            if args.ra is not None and args.dec is not None:
                _warn_target_coverage(base_dir, quadrants, args.ra, args.dec)
        _run_phot(steps, step_phot, step_phot_sci, base_dir, quadrants, args)
    # Two-pass calibration in one invocation: pass 1 calibrates without a flatfield, its
    # light curves feed the calibration-star vetting and its residuals feed the
    # flatfield; pass 2 (the loop below: flatfield -> recalibrate) applies both.
    two_pass = "calibrate" in steps and ("recalibrate" in steps or "flatfield" in steps)
    if two_pass:
        logger.info("─── calibration pass 1 (no flatfield) → light curves → vetting ───")
        for suffix in suffixes:
            for q in quadrants:
                # when this run vets, pass 1 is unvetted (unless --vet-catalog): a vetting
                # catalog left by an earlier run in this base dir would otherwise steer
                # this run's vetting, so the result would depend on the directory's
                # history. Without the vet step an existing vetting catalog is the input.
                step_calibrate(base_dir, [q], workers=args.workers, force=True,
                               vet_catalog=args.vet_catalog, auto_vet="vet" not in steps,
                               poly_degree=args.poly_degree,
                               flatfield=None, target_ra=args.ra, target_dec=args.dec,
                               target_match_radius=args.target_match_radius,
                               save_residuals=True,
                               suffix=suffix)
        if "vet" in steps:
            # vetting reads the ref-pos light curves; a sci-only run vets from its sci
            # light curves, seeded under the ref name for the duration of the vet step
            vsuf = "" if "" in suffixes else "_sci"
            step_lightcurves(base_dir, quadrants, force=True,
                             suffix=vsuf, target_ra=args.ra, target_dec=args.dec,
                             target_match_radius=args.target_match_radius)
            seeded = []
            if vsuf:
                import shutil as _sh
                for q in quadrants:
                    d = (base_dir / "LightCurves" / f"{q['field']:06d}" / q['filtercode']
                         / f"ccd{q['ccdid']:02d}" / f"q{q['qid']}")
                    if (d / "lightcurves_sci.parquet").exists() and not (d / "lightcurves.parquet").exists():
                        _sh.copy2(d / "lightcurves_sci.parquet", d / "lightcurves.parquet")
                        seeded.append(d / "lightcurves.parquet")
            try:
                step_vet(base_dir, quadrants)
            finally:                                # never leave a sci-pos copy under the ref name
                for f in seeded:
                    f.unlink(missing_ok=True)
        logger.info("─── calibration pass 2 (flatfield + vetting) ───")
    elif "vet" in steps:
        step_vet(base_dir, quadrants)

    for suffix in suffixes:
        logger.info(f"─── suffix='{suffix}' pass ───" if len(suffixes) > 1 else "")

        # Load flatfield from disk for each quadrant (used in calibrate)
        _ff_map: dict = {}
        for q in quadrants:
            ff = (base_dir / f"Calibrated{suffix}" / f"{q['field']:06d}"
                  / q['filtercode'] / f"{q['ccdid']:02d}" / str(q['qid']) / "flatfield.npz")
            if ff.exists():
                try:
                    d = np.load(str(ff))
                    _ff_map[(q['field'], q['filtercode'], q['ccdid'], q['qid'])] = dict(
                        stat=d['stat'], ra_edges=d['ra_edges'], dec_edges=d['dec_edges'])
                except Exception as e:
                    logger.warning(f"Could not load flatfield {ff}: {e}")

        if "flatfield"  in steps:
            _ff_map = step_build_flatfield(base_dir, quadrants,
                                           nbins=args.ff_bins, min_count=args.ff_min_count,
                                           edge_split=args.ff_edge_split,
                                           suffix=suffix)

        # recalibrate is the second, flatfield-applied pass; it shares output paths
        # with calibrate, so it must overwrite that first pass regardless of --force.
        # Otherwise step_calibrate skips the existing *_cal.fits and the flatfield is
        # built but never applied (dm_5 == dm_4, "flatfield does nothing").
        _cal_force = args.force or ("recalibrate" in steps) or two_pass
        if "calibrate" in steps or "recalibrate" in steps:
            for q in quadrants:
                key = (q['field'], q['filtercode'], q['ccdid'], q['qid'])
                step_calibrate(base_dir, [q], workers=args.workers, force=_cal_force,
                               vet_catalog=args.vet_catalog, poly_degree=args.poly_degree,
                               flatfield=_ff_map.get(key),
                               target_ra=args.ra, target_dec=args.dec,
                               target_match_radius=args.target_match_radius,
                               save_residuals=True,
                               suffix=suffix)

        if "lightcurves" in steps:
            # after a two-pass calibration the pass-1 light curves (written for vetting)
            # must be replaced by the final, flatfield-applied ones
            step_lightcurves(base_dir, quadrants, force=args.force or two_pass or "recalibrate" in steps,
                             suffix=suffix, target_ra=args.ra, target_dec=args.dec,
                             target_match_radius=args.target_match_radius,
                             lc_columns=args.lc_columns,
                             run_meta={
                                 "ztfphot_run": _json.dumps(_run_rec, default=str),
                                 "ztfphot_download": _json.dumps(
                                     _latest_download_run(base_dir, args), default=str),
                             })

        if "merge"      in steps: step_merge(base_dir, quadrants, force=args.force or two_pass or "recalibrate" in steps,
                                              target_ra=args.ra, target_dec=args.dec,
                                              suffix=suffix,
                                              mag_bin=args.merge_mag_bin,
                                              max_sep_arcsec=args.merge_match_radius)

        if "plots"      in steps:
            logger.info("─── plots ───")
            from plot_diagnostics import (
                make_spatial_rms, make_spatial_iqr,
                make_rms, make_precision, make_position,
                make_lightcurves, make_lightcurves_flux,
            )

            plot_root = base_dir / f"Plots{suffix}"
            if args.ra is not None and args.dec is not None:
                plot_root = plot_root / f"{args.ra:.5f}_{args.dec:+.5f}"
            plot_root.mkdir(parents=True, exist_ok=True)

            for q in quadrants:
                f, fc, ccd, qid_ = q["field"], q["filtercode"], q["ccdid"], q["qid"]
                tag = f"{f:06d}_{fc}_c{ccd:02d}_q{qid_}"

                cal_dir   = base_dir / f"Calibrated{suffix}"         / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid_)
                resid_dir = base_dir / f"FlatfieldResiduals{suffix}"  / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid_)
                lc_path   = (base_dir / "LightCurves" / f"{f:06d}" / fc
                             / f"ccd{ccd:02d}" / f"q{qid_}" / f"lightcurves{suffix}.parquet")

                has_cal   = cal_dir.exists()   and any(cal_dir.glob("*_cal.fits"))
                has_resid = resid_dir.exists() and any(resid_dir.glob("*_resid.npz"))
                has_lc    = lc_path.exists()

                if has_resid:
                    make_spatial_rms(resid_dir,
                                     plot_root / f"spatial_rms_{tag}.png", tag)
                    make_spatial_iqr(resid_dir,
                                     plot_root / f"spatial_IQR_{tag}.png", tag)
                    # (flux variants intentionally not emitted — the fractional-flux
                    # residual diverges for faint sources; mmag is the right unit here)
                else:
                    logger.info(f"  [{tag}] no residual NPZ files — skipping spatial_rms/IQR")

                if has_cal:
                    make_rms(cal_dir,
                             plot_root / f"rms_{tag}.png", tag,
                             resid_dir=resid_dir if has_resid else None)
                else:
                    logger.info(f"  [{tag}] no calibrated FITS — skipping rms")

                if has_lc:
                    vet_cat = (base_dir / "Calibrated" / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid_)) / "vet_calib_stars.fits"
                    vet_cat_arg = vet_cat if vet_cat.exists() else None
                    _tgt_ra  = None if args.no_target else args.ra
                    _tgt_dec = None if args.no_target else args.dec
                    make_precision(lc_path,
                                   plot_root / f"precision_{tag}.png",
                                   tag, _tgt_ra, _tgt_dec,
                                   vet_catalog=vet_cat_arg)
                    # Astrometry: sci-pos only — returns without writing in ref-pos,
                    # where there are no per-epoch positions.
                    make_position(lc_path,
                                  plot_root / f"position_{tag}.png",
                                  tag, _tgt_ra, _tgt_dec)
                    if not args.no_target and args.ra is not None and args.dec is not None:
                        make_lightcurves(lc_path,
                                         plot_root / f"lightcurves_{tag}.png",
                                         args.ra, args.dec,
                                         tag=tag,
                                         vet_catalog=vet_cat_arg,
                                         bin_days=args.lc_bin_days)
                        make_lightcurves_flux(lc_path,
                                              plot_root / f"lightcurves_{tag}_flux.png",
                                              args.ra, args.dec,
                                              tag=tag,
                                              vet_catalog=vet_cat_arg,
                                              bin_days=args.lc_bin_days)
                else:
                    logger.info(f"  [{tag}] no light-curve parquet — skipping precision/lightcurves")

    if not args.no_target and args.ra is not None and args.dec is not None:
        _write_target_status(base_dir, quadrants, args.ra, args.dec,
                             args.target_match_radius)

    logger.info(f"Done in {time.time() - t0:.1f}s")
    _print_status(base_dir, quadrants)
    if _INCOMPLETE:
        logger.error(f"incomplete this run (retried by the next run): {sorted(set(_INCOMPLETE))}")
        sys.exit(3)


if __name__ == "__main__":
    main()
