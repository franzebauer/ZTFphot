"""
run_pipeline.py — ZTF difference-image photometry pipeline

Steps (run in order, or select with --steps):
    lookup       — query IRSA for field/epoch coverage and cache results
    download     — fetch science and reference images from IRSA
    funpack      — decompress .fits.fz difference images
    catalog      — build reference CSV catalogs from refsexcat.fits
    simulate     — build simulated detection images (PSF at reference positions)
    sex          — SExtractor dual-image aperture photometry
    vet          — flag variable/bad calibration stars by multi-epoch RMS
    calibrate    — linear ZP → 3σ clip → faint correction → polynomial → flatfield
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

_SCRIPTS = Path(__file__).parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

logger = logging.getLogger(__name__)

ALL_STEPS = ["lookup", "download", "funpack", "catalog", "simulate", "sex",
             "vet", "calibrate", "flatfield", "lightcurves", "merge", "plots"]


def _print_status(base_dir: Path, quadrants: list[dict]) -> None:
    from shutil import get_terminal_size
    w = min(get_terminal_size().columns, 100)
    print("=" * w)
    print(f"{'Quadrant':<32} {'Diff imgs*':>10} {'SExCats':>10} {'Calibrated':>12} {'LC':>4}")
    print("-" * w)
    for q in quadrants:
        f, fc, ccd, qid = q["field"], q["filtercode"], q["ccdid"], q["qid"]
        sci = q["sci_dir"]
        n_diff = len(list(sci.glob("*_scimrefdiffimg.fits")))
        sex_d  = base_dir / "SExCatalogs" / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid)
        n_sex  = len(list(sex_d.glob("*_sexout.fits"))) if sex_d.exists() else 0
        cal_d  = base_dir / "Calibrated" / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid)
        n_cal  = len(list(cal_d.glob("*_cal.fits"))) if cal_d.exists() else 0
        lc     = base_dir / "LightCurves" / f"{f:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid}" / "lightcurves.parquet"
        print(f"{f:06d} {fc} ccd{ccd:02d} q{qid:<24} {n_diff:>10} {n_sex:>10} {n_cal:>12} {'✓' if lc.exists() else '–':>4}")
    print("=" * w)
    print("  * Diff imgs = 0 is expected after --purge-batch or --clean-up")


def _run_purge_batch(base_dir: Path, epochs, quadrants: list[dict], args) -> None:
    """
    Run funpack → (catalog once) → simulate → sex in batches of --purge-batch N,
    deleting all imaging products after each batch's sex step.
    Reference products are deleted after the first batch (catalog no longer needs them).
    """
    import math
    import pandas as pd
    from download_coordinator import download_all, purge_images, filter_epochs
    from photometry import step_funpack, step_make_catalog, step_simulate, step_sextractor

    N = args.purge_batch
    steps = set(args.steps)

    # Load epochs from cache if not already in memory (e.g. lookup not in steps)
    if epochs is None and args.ra is not None and args.dec is not None:
        import pandas as pd
        bands = args.bands or ["g", "r", "i"]
        band_str   = "-".join(sorted(bands))
        cache_path = (base_dir / "Epochs"
                      / f"lookup_{args.ra:.5f}_{args.dec:.5f}_{band_str}.epochs.parquet")
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
            # Still run catalog/simulate/sex on whatever is already on disk
            if "catalog" in steps:
                step_make_catalog(base_dir, [q], force=args.force)
            if "simulate" in steps:
                step_simulate(base_dir, [q], workers=args.workers, force=args.force)
            if "sex" in steps:
                step_sextractor(base_dir, [q], workers=args.workers,
                                force=args.force, verbose=args.verbose)
            continue

        n_batches = math.ceil(len(q_epochs) / N)
        logger.info(f"─── purge-batch {tag}: {len(q_epochs)} epochs → "
                    f"{n_batches} batches of {N} ───")

        cat_csv = (base_dir / "Catalogs" /
                   f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}(REFERENCE)[OBJECTS].csv")
        catalog_done = cat_csv.exists() and not args.force

        sex_dir = (base_dir / "SExCatalogs" / f"{field:06d}" / fc
                   / f"{ccd:02d}" / str(qid_))

        def _sexcat_path(ffd):
            fname = (f"ztf_{ffd}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}"
                     f"_scimrefdiffimg_sexout.fits")
            return sex_dir / fname

        # Load permanent 404s so epochs with no IRSA diff image are treated as done
        _perm404_log = base_dir / "Epochs" / "permanent_404s.log"
        _perm404_urls = set()
        if _perm404_log.exists():
            _perm404_urls = set(_perm404_log.read_text().splitlines())

        def _epoch_done(ffd):
            if _sexcat_path(ffd).exists():
                return True
            needle = f"ztf_{ffd}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}_scimrefdiffimg"
            return any(needle in url for url in _perm404_urls)

        for bi, start in enumerate(range(0, len(q_epochs), N)):
            batch    = q_epochs.iloc[start:start + N]
            # Normalise filefracday to integer string to match filenames on disk
            ffds     = [str(int(float(v))) for v in batch["filefracday"]]
            is_first = (bi == 0)

            ffds_set = set(ffds)

            # Skip batch entirely if all SEx catalogs exist (or epoch was a permanent 404)
            if not args.force and all(_epoch_done(ffd) for ffd in ffds):
                logger.info(f"  batch {bi+1}/{n_batches}: all SEx catalogs exist — skipping")
                purge_images(base_dir, [q], sci=True, ref=is_first,
                             filefracdays=ffds_set, dry_run=args.dry_run)
                continue

            logger.info(f"  batch {bi+1}/{n_batches}: {len(batch)} epochs")

            # Download this batch (ref products skipped automatically if already on disk)
            if "download" in steps or args.purge_batch:
                band = fc[1:]   # zg→g, zr→r, zi→i
                download_all(batch, base_dir=base_dir, bands=[band],
                             max_workers=args.workers)

            if "funpack" in steps:
                step_funpack(base_dir, force=args.force, filefracdays=ffds_set)

            if "catalog" in steps and not catalog_done:
                step_make_catalog(base_dir, [q], force=args.force)
                catalog_done = True

            if "simulate" in steps:
                step_simulate(base_dir, [q], workers=args.workers,
                              force=args.force, filefracdays=ffds_set)

            if "sex" in steps:
                step_sextractor(base_dir, [q], workers=args.workers,
                                force=args.force, verbose=args.verbose,
                                filefracdays=ffds_set)

            # Purge: always remove sci products; ref only after first batch
            purge_images(base_dir, [q],
                         sci=True, ref=is_first,
                         filefracdays=ffds_set, dry_run=args.dry_run)


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
    p.add_argument("--steps",     nargs="+",  default=ALL_STEPS, choices=ALL_STEPS,
                   metavar="STEP")
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--force",     action="store_true")
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--status",    action="store_true", help="Print file-existence summary and exit")
    p.add_argument("--dry-run",   action="store_true")
    # Calibration
    p.add_argument("--vet-catalog",  type=Path, default=None)
    p.add_argument("--poly-degree",  type=int,  default=2)
    p.add_argument("--ff-bins",      type=int,  default=20)
    p.add_argument("--ff-min-count", type=int,  default=50)
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
                        "batch's sex step. Keeps only SEx catalogs on disk. "
                        "Use --dry-run to preview what would be deleted.")
    p.add_argument("--clean-up",     action="store_true",
                   help="Delete all imaging products (Science/, Reference/) for discovered "
                        "quadrants and exit. Safe once the sex step is complete.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    base_dir = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

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
            )

    # ── Step: download ────────────────────────────────────────────────────────
    if "download" in steps and not args.purge_batch:
        if args.ra is None or args.dec is None:
            logger.warning("download: --ra and --dec required — skipping")
        else:
            if epochs is None:
                # Load from cache (lookup already ran previously)
                import pandas as pd
                band_str   = "-".join(sorted(bands))
                cache_path = (base_dir / "Epochs"
                              / f"lookup_{args.ra:.5f}_{args.dec:.5f}_{band_str}.epochs.parquet")
                if not cache_path.exists():
                    sys.exit(f"Epochs cache not found: {cache_path}\nRun the lookup step first.")
                epochs = pd.read_parquet(cache_path)

            logger.info("─── download ───")
            download_all(
                epochs, base_dir=base_dir, bands=bands,
                max_workers=args.workers,
                skip_flagged=args.skip_flagged,
                max_seeing=args.max_seeing,
                min_maglim=args.min_maglim,
                mjd_min=args.mjd_min,
                mjd_max=args.mjd_max,
                min_epochs_per_quad=args.min_epochs_per_quad,
            )

    # ── Discover quadrants on disk ────────────────────────────────────────────
    quadrants = find_quadrants(base_dir, bands=args.bands,
                               field=args.field, ccdid=args.ccdid, qid=args.qid)

    # Filter to fields belonging to this target using the epoch cache
    if args.ra is not None and args.dec is not None:
        import pandas as _pd
        bands_str  = "-".join(sorted(args.bands or ["g", "i", "r"]))
        cache_path = base_dir / "Epochs" / f"lookup_{args.ra:.5f}_{args.dec:.5f}_{bands_str}.epochs.parquet"
        if cache_path.exists():
            _cache     = _pd.read_parquet(cache_path)
            _fields    = set(_cache["field"].dropna().astype(int).tolist())
            _before    = len(quadrants)
            quadrants  = [q for q in quadrants if q["field"] in _fields]
            logger.info(f"Epoch cache filter: {_before} → {len(quadrants)} quadrants "
                        f"(keeping fields {sorted(_fields)})")
        else:
            logger.warning(f"No epoch cache found at {cache_path} — using all quadrants on disk")

    if args.status:
        _print_status(base_dir, quadrants)
        return

    # ── Clean-up: delete all imaging products and exit ────────────────────────
    if args.clean_up:
        logger.info("─── clean-up: deleting imaging products ───")
        from download_coordinator import purge_images
        purge_images(base_dir, quadrants, sci=True, ref=True, dry_run=args.dry_run)
        return

    remaining = [s for s in steps if s not in ("lookup", "download")]
    if remaining and not quadrants and not args.purge_batch:
        sys.exit("No quadrants found on disk. Run lookup and download first.")

    logger.info(f"Quadrants ({len(quadrants)}):")
    for q in quadrants:
        logger.info(f"  {q['field']:06d} {q['filtercode']} ccd{q['ccdid']:02d} q{q['qid']}")

    # ── Import step modules ───────────────────────────────────────────────────
    from photometry  import step_funpack, step_make_catalog, step_simulate, step_sextractor
    from calibrate   import step_vet, step_calibrate, step_build_flatfield
    from lightcurves import step_lightcurves, step_merge

    t0 = time.time()

    if args.purge_batch and any(s in steps for s in ("funpack", "catalog", "simulate", "sex")):
        _run_purge_batch(base_dir, epochs, quadrants, args)
    else:
        if "funpack"    in steps: step_funpack(base_dir, force=args.force)
        if "catalog"    in steps: step_make_catalog(base_dir, quadrants, force=args.force)
        if "simulate"   in steps: step_simulate(base_dir, quadrants, workers=args.workers, force=args.force)
        if "sex"        in steps: step_sextractor(base_dir, quadrants, workers=args.workers,
                                                  force=args.force, verbose=args.verbose)
    if "vet"        in steps: step_vet(base_dir, quadrants)

    # Load flatfield from disk for each quadrant (used in calibrate)
    _ff_map: dict = {}
    for q in quadrants:
        ff = (base_dir / "Calibrated" / f"{q['field']:06d}"
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
                                       nbins=args.ff_bins, min_count=args.ff_min_count)

    if "calibrate"  in steps:
        for q in quadrants:
            key = (q['field'], q['filtercode'], q['ccdid'], q['qid'])
            step_calibrate(base_dir, [q], workers=args.workers, force=args.force,
                           vet_catalog=args.vet_catalog, poly_degree=args.poly_degree,
                           flatfield=_ff_map.get(key),
                           target_ra=args.ra, target_dec=args.dec,
                           save_residuals=True)

    if "lightcurves" in steps:
        step_lightcurves(base_dir, quadrants, force=args.force,
                         use_calibrated="calibrate" in steps)

    if "merge"      in steps: step_merge(base_dir, quadrants, force=args.force,
                                          target_ra=args.ra, target_dec=args.dec)

    if "plots"      in steps:
        logger.info("─── plots ───")
        import sys as _sys
        if str(_SCRIPTS) not in _sys.path:
            _sys.path.insert(0, str(_SCRIPTS))
        from make_diagnostic_plots import (
            make_spatial_rms, make_spatial_iqr,
            make_fig2_rms, make_fig3_precision,
            make_fig4_lightcurves,
        )

        plot_root = base_dir / "Plots"
        if args.ra is not None and args.dec is not None:
            plot_root = plot_root / f"{args.ra:.5f}_{args.dec:+.5f}"
        plot_root.mkdir(parents=True, exist_ok=True)

        for q in quadrants:
            f, fc, ccd, qid_ = q["field"], q["filtercode"], q["ccdid"], q["qid"]
            tag = f"{f:06d}_{fc}_c{ccd:02d}_q{qid_}"

            cal_dir   = base_dir / "Calibrated"          / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid_)
            resid_dir = base_dir / "FlatfieldResiduals"  / f"{f:06d}" / fc / f"{ccd:02d}" / str(qid_)
            lc_path   = (base_dir / "LightCurves" / f"{f:06d}" / fc
                         / f"ccd{ccd:02d}" / f"q{qid_}" / "lightcurves.parquet")

            has_cal   = cal_dir.exists()   and any(cal_dir.glob("*_cal.fits"))
            has_resid = resid_dir.exists() and any(resid_dir.glob("*_resid.npz"))
            has_lc    = lc_path.exists()

            if has_resid:
                make_spatial_rms(resid_dir,
                                 plot_root / f"spatial_rms_{tag}.png", tag)
                make_spatial_iqr(resid_dir,
                                 plot_root / f"spatial_IQR_{tag}.png", tag)
            else:
                logger.info(f"  [{tag}] no residual NPZ files — skipping spatial_rms/IQR")

            if has_cal:
                make_fig2_rms(cal_dir,
                              plot_root / f"rms_{tag}.png", tag)
            else:
                logger.info(f"  [{tag}] no calibrated FITS — skipping rms")

            if has_lc:
                vet_cat = cal_dir / "vet_calib_stars.fits"
                vet_cat_arg = vet_cat if vet_cat.exists() else None
                make_fig3_precision(lc_path,
                                    plot_root / f"precision_{tag}.png",
                                    tag, args.ra, args.dec,
                                    vet_catalog=vet_cat_arg)
                if args.ra is not None and args.dec is not None:
                    make_fig4_lightcurves(lc_path,
                                          plot_root / f"lightcurves_{tag}.png",
                                          args.ra, args.dec,
                                          tag=tag,
                                          vet_catalog=vet_cat_arg)
            else:
                logger.info(f"  [{tag}] no light-curve parquet — skipping precision/lightcurves")

    logger.info(f"Done in {time.time() - t0:.1f}s")
    _print_status(base_dir, quadrants)


if __name__ == "__main__":
    main()
