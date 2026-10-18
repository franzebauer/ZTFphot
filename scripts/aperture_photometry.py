"""
aperture_photometry.py
----------------------
Forced circular-aperture photometry on ZTF difference images (photutils), replacing
the simulated-image + SExtractor dual-image route.

Positions are forced, not detected:
  ref-pos  — every reference-catalog source (Catalogs/*(REFERENCE)[OBJECTS].csv), plus
             the --ra/--dec target appended as index N+1 when no reference source lies
             within the target match radius (same rule as photometry._write_assoc_catalog).
  sci-pos  — every source of the epoch's ZTF science catalog (*_sexcat.fits), tied to
             the nearest reference entry within --assoc-radius-sci (unmatched dropped).

A source is measured when its centre is on the frame and > 50% of the pixels within
0.75*SEEING of it are finite (the rule the simulated image used to decide what to paint).

Apertures: PHOT_APERTURES DIAMETERS (3,4,6,10 px -> radii 1.5,2,3,5), exact overlap.
Masked (NaN) pixels are excluded from the sums.

Background (--aper-bkg):
  local           — sigma-clipped median in a circular annulus [r_in, r_in + 35 px],
                    r_in = 1.58*SEEING (the detection isophote of the old simulated
                    source); mimics SExtractor BACKPHOTO_TYPE LOCAL / THICK 35.
  box (default)   — 3-sigma-clipped mean in a SQUARE annulus (half-sizes 1.58*SEEING ..
                    +35 px) via integral images: same pixel-level local estimate, ~5x
                    cheaper (pixels clipped against the 64 px background level/RMS boxes).
  mesh            — Background2D level (--aper-bkg-mesh px mesh, default 64; 3x3 filter)
                    at the source position.
  none            — no subtraction (difference images are already background-subtracted).
Errors: FLUXERR = sigma_rms(x, y) * sqrt(unmasked aperture area), sigma_rms from a
64 px Background2D RMS mesh evaluated at the source positions (cubic spline) — the
background-only error SExtractor gave with GAIN=inf.

With ref + sci requested, both position sets are measured in one pass per image.

FLAGS (bit values aligned with SExtractor's meanings; carried to the parquet as FLAG_AP):
   1 — another catalog source within the largest aperture radius (≈ SExtractor 1, neighbour)
   8 — an aperture extends beyond the image edge                 (≈ SExtractor 8, truncated)
  16 — an aperture contains masked (NaN) pixels                  (≈ SExtractor 16, incomplete
       aperture data)
Calibration rejects calibrator measurements with bits 8|16 (CAL_REJECT_BITS); blends are
already handled at reference level (FLAG_SE_REF == 0 in vetting).

Output: one LDAC catalog per epoch with the same name, layout and columns as the
SExtractor *_sexout.fits (LDAC_IMHEAD = the difference-image header; LDAC_OBJECTS with
FLUX_APER(4), FLUXERR_APER(4), VECTOR_ASSOC, positions, FLAGS), so calibration and
everything downstream is unchanged. Written to a temporary file and renamed, so a
crash never leaves a partial catalog.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BKG_ANNULUS_THICK = 35.0      # px, = SExtractor BACKPHOTO_THICK in clean.sex
BKG_MESH = 64                 # px, = SExtractor BACK_SIZE
BKG_FILTER = 3                # = SExtractor BACK_FILTERSIZE
ISO_FACTOR = 1.58             # r_iso / FWHM of the old simulated source (amp 1000, thresh 1)

FLAG_NEIGHBOUR, FLAG_EDGE, FLAG_MASKED = 1, 8, 16
CAL_REJECT_BITS = FLAG_EDGE | FLAG_MASKED     # calib_catalogs rejects calibrators with these
PHOTCODE = "ztfphot-aper"                     # LDAC header tag identifying these catalogs


def _ffd_from_path(p: Path) -> str:
    return p.name.split("_")[1]


def _aperture_radii() -> list[float]:
    from photometry import PHOT_APERTURES
    return [float(d) / 2.0 for d in PHOT_APERTURES.split(",")]


def catalog_ok(path: Path) -> bool:
    """True if `path` is a complete LDAC catalog (guards against the header-only /
    empty / truncated files a failed SExtractor run leaves behind)."""
    try:
        if path.stat().st_size <= 2880:
            return False
        from astropy.io import fits
        with fits.open(path) as h:
            if len(h) < 3:
                return False
            _ = h[2].data                       # raises on a truncated table
        return True
    except Exception:
        return False


def _assoc_entries(ref_csv: Path, target_ra, target_dec, match_radius):
    """(index, ra, dec, injected): reference sources (1-based index) plus the target as
    the last entry (index N+1) when it is injected."""
    import pandas as pd
    ref = pd.read_csv(ref_csv)
    ra = pd.to_numeric(ref["ALPHAWIN_J2000"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(ref["DELTAWIN_J2000"], errors="coerce").to_numpy(float)
    idx = np.arange(1, len(ref) + 1)
    injected = False
    if target_ra is not None and target_dec is not None:
        from astropy.coordinates import SkyCoord
        ok = np.isfinite(ra) & np.isfinite(dec)
        sep = (SkyCoord(target_ra, target_dec, unit="deg")
               .separation(SkyCoord(ra[ok], dec[ok], unit="deg")).arcsec) if ok.any() else np.array([np.inf])
        if sep.min() >= match_radius:
            idx = np.append(idx, len(ref) + 1)
            ra = np.append(ra, target_ra)
            dec = np.append(dec, target_dec)
            injected = True
    return idx, ra, dec, injected


def _write_ldac(out_cat: Path, header, cols: dict) -> None:
    from astropy.io import fits
    cards = [c.image for c in header.cards]
    n = len(cards)
    imhead = fits.BinTableHDU.from_columns(
        [fits.Column(name="Field Header Card", format=f"{80 * n}A", dim=f"(80,{n})",
                     array=np.array([cards], dtype="S80"))], name="LDAC_IMHEAD")
    fmt = {"NUMBER": "J", "NUMBER_ASSOC": "J", "FLAGS": "I", "VECTOR_ASSOC": "D",
           "FLUX_APER": "4E", "FLUXERR_APER": "4E", "BKG_LOCAL": "E",
           "X_IMAGE": "E", "Y_IMAGE": "E", "CLASS_STAR": "E"}
    objs = fits.BinTableHDU.from_columns(
        [fits.Column(name=k, format=fmt.get(k, "D"), array=v) for k, v in cols.items()],
        name="LDAC_OBJECTS")
    out_cat.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_cat.with_name(out_cat.name + ".tmp")
    fits.HDUList([fits.PrimaryHDU(), imhead, objs]).writeto(tmp, overwrite=True)
    os.replace(tmp, out_cat)


class _Epoch:
    """One difference image plus its background maps, shared by every position set
    measured on it (ref-pos and sci-pos). Background RMS / level are evaluated only at
    the source positions from the Background2D mesh (cubic spline, as photutils' own
    zoom interpolator), not as full-resolution images."""

    def __init__(self, diff_path: Path, bkg_mesh: int):
        from astropy.io import fits
        from astropy.wcs import WCS
        with fits.open(diff_path) as h:
            hdu = h[0] if h[0].data is not None else h[1]
            self.data = hdu.data.astype(np.float64)
            self.header = hdu.header.copy()
        self.wcs = WCS(self.header)
        self.mask = ~np.isfinite(self.data)
        self.data0 = np.where(self.mask, 0.0, self.data)
        self.fin = (~self.mask).astype(np.float64)
        self.fwhm = float(self.header["SEEING"])
        self.bkg_mesh = bkg_mesh
        self._bk = {}
        self._integ = None

    def _background(self, box: int):
        if box not in self._bk:
            from astropy.stats import SigmaClip
            from photutils.background import (Background2D, SExtractorBackground,
                                              StdBackgroundRMS)
            self._bk[box] = Background2D(
                self.data, box, filter_size=BKG_FILTER, mask=self.mask,
                sigma_clip=SigmaClip(sigma=3.0), exclude_percentile=50.0,
                bkg_estimator=SExtractorBackground(), bkgrms_estimator=StdBackgroundRMS())
        return self._bk[box]

    @staticmethod
    def _at(mesh, box, x, y):
        from scipy.ndimage import map_coordinates
        return map_coordinates(mesh, [(y - (box - 1) / 2) / box, (x - (box - 1) / 2) / box],
                               order=3, mode="nearest")

    def rms_at(self, x, y):
        return self._at(self._background(BKG_MESH).background_rms_mesh, BKG_MESH, x, y)

    def level_at(self, x, y):
        return self._at(self._background(self.bkg_mesh).background_mesh, self.bkg_mesh, x, y)

    def box_bkg_at(self, x, y, r_in, r_out):
        """Clipped mean in a square annulus (half-sizes ceil(r_in) .. ceil(r_out) px around
        the rounded position) for all sources at once, from integral images. Pixels more
        than 3 sigma from the 64 px background level (mesh boxes, no interpolation) are
        excluded before the sums, so the estimate is pixel-level and local while costing
        two cumulative sums per image instead of one clipped cutout per source."""
        H, W = self.data.shape
        if self._integ is None:
            bk = self._background(BKG_MESH)
            up = lambda m: np.repeat(np.repeat(m, BKG_MESH, 0), BKG_MESH, 1)[:H, :W]
            good = (~self.mask) & (np.abs(self.data - up(bk.background_mesh))
                                   < 3.0 * up(bk.background_rms_mesh))
            S = np.zeros((H + 1, W + 1))
            S[1:, 1:] = np.where(good, self.data, 0.0).cumsum(0).cumsum(1)
            N = np.zeros((H + 1, W + 1), np.int64)
            N[1:, 1:] = good.astype(np.int32).cumsum(0).cumsum(1)
            self._integ = (S, N)
        S, N = self._integ
        cx, cy = np.round(x).astype(int), np.round(y).astype(int)

        def box_sum(T, h):
            x0, x1 = np.clip(cx - h, 0, W), np.clip(cx + h + 1, 0, W)
            y0, y1 = np.clip(cy - h, 0, H), np.clip(cy + h + 1, 0, H)
            return T[y1, x1] - T[y0, x1] - T[y1, x0] + T[y0, x0]

        ri, ro = int(np.ceil(r_in)), int(np.ceil(r_out))
        n = box_sum(N, ro) - box_sum(N, ri)
        return np.where(n > 0, (box_sum(S, ro) - box_sum(S, ri)) / np.maximum(n, 1), 0.0)


def _measure(ep: _Epoch, out_cat: Path, ref_csv: Path, sexcat_path, assoc_radius,
             target_ra, target_dec, match_radius, bkg: str):
    """Forced photometry of one position set on an already-loaded epoch; writes out_cat.
    Returns the target fate dict (ref-pos with a target only)."""
    from astropy.stats import SigmaClip
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry, ApertureStats)
    nrows, ncols = ep.data.shape
    fwhm = ep.fwhm

    idx, ra, dec, injected = _assoc_entries(ref_csv, target_ra, target_dec, match_radius)
    if sexcat_path is not None:                               # sci-pos
        from astropy.io import fits
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        with fits.open(sexcat_path) as sc:
            t = sc[1].data
            sra = np.asarray(t["ALPHAWIN_J2000"], float)
            sdec = np.asarray(t["DELTAWIN_J2000"], float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        j, sep, _ = match_coordinates_sky(SkyCoord(sra, sdec, unit="deg"),
                                          SkyCoord(ra[ok], dec[ok], unit="deg"))
        keep = sep.arcsec <= assoc_radius
        idx, ra, dec = idx[ok][j[keep]], sra[keep], sdec[keep]

    x, y = ep.wcs.all_world2pix(ra, dec, 0)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    on = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < ncols) & (iy >= 0) & (iy < nrows)

    # > 50% finite pixels within 0.75*SEEING (the old "paint" criterion)
    good = np.zeros(len(x), bool)
    if on.any():
        ap_v = CircularAperture(np.column_stack([x[on], y[on]]), r=0.75 * fwhm)
        n_fin = aperture_photometry(ep.fin, ap_v, method="center")["aperture_sum"]
        n_all = aperture_photometry(np.ones_like(ep.fin), ap_v, method="center")["aperture_sum"]
        good[on] = np.asarray(n_fin) > 0.5 * np.asarray(n_all)

    # Target fate (same keys the simulate step wrote): matched = has a reference source
    fate = None
    if target_ra is not None and target_dec is not None and sexcat_path is None:
        fate = {"matched": not injected, "in_frame": bool(injected and on[-1]),
                "painted": bool(injected and good[-1])}

    idx, ra, dec, x, y = idx[good], ra[good], dec[good], x[good], y[good]
    radii = _aperture_radii()
    n = len(idx)
    flux = np.zeros((n, len(radii)), np.float32)
    ferr = np.zeros((n, len(radii)), np.float32)
    flags = np.zeros(n, np.int16)
    bkg_loc = np.zeros(n, np.float32)

    if n:
        pos = np.column_stack([x, y])
        rms = ep.rms_at(x, y)
        if bkg == "local":
            r_in = ISO_FACTOR * fwhm
            ann = CircularAnnulus(pos, r_in=r_in, r_out=r_in + BKG_ANNULUS_THICK)
            st = ApertureStats(ep.data, ann, mask=ep.mask, sigma_clip=SigmaClip(sigma=3.0))
            bkg_loc = np.nan_to_num(np.asarray(st.median, float)).astype(np.float32)
        elif bkg == "box":
            r_in = ISO_FACTOR * fwhm
            bkg_loc = ep.box_bkg_at(x, y, r_in, r_in + BKG_ANNULUS_THICK).astype(np.float32)
        elif bkg == "mesh":
            bkg_loc = ep.level_at(x, y).astype(np.float32)

        for k, r in enumerate(radii):
            ap = CircularAperture(pos, r=r)
            s = np.asarray(aperture_photometry(ep.data0, ap, method="exact")["aperture_sum"], float)
            a_eff = np.asarray(aperture_photometry(ep.fin, ap, method="exact")["aperture_sum"], float)
            flux[:, k] = s - bkg_loc * a_eff
            ferr[:, k] = rms * np.sqrt(a_eff)
            flags |= np.where(a_eff < ap.area - 1e-6, FLAG_MASKED, 0).astype(np.int16)
        rmax = max(radii)
        edge = (x - rmax < -0.5) | (x + rmax > ncols - 0.5) | (y - rmax < -0.5) | (y + rmax > nrows - 0.5)
        flags |= np.where(edge, FLAG_EDGE, 0).astype(np.int16)
        flags[edge] &= ~np.int16(FLAG_MASKED)                 # off-edge pixels aren't "masked"
        if n > 1:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pos).query(pos, k=2)
            flags |= np.where(d[:, 1] < rmax, FLAG_NEIGHBOUR, 0).astype(np.int16)

    header = ep.header.copy()
    header["PHOTCODE"] = (PHOTCODE, "forced aperture photometry (photutils)")
    header["APBKG"] = (bkg, "aperture background mode")
    if bkg == "mesh":
        header["APBKGBOX"] = (ep.bkg_mesh, "background mesh box [px]")
    _write_ldac(out_cat, header, {
        "NUMBER": np.arange(1, n + 1, dtype=np.int32),
        "ALPHA_J2000": ra, "DELTA_J2000": dec,
        "ALPHAWIN_J2000": ra, "DELTAWIN_J2000": dec,
        "X_IMAGE": (x + 1).astype(np.float32), "Y_IMAGE": (y + 1).astype(np.float32),
        "CLASS_STAR": np.zeros(n, np.float32),
        "FLAGS": flags, "NUMBER_ASSOC": np.ones(n, np.int32),
        "VECTOR_ASSOC": idx.astype(np.float64),
        "FLUX_APER": flux, "FLUXERR_APER": ferr, "BKG_LOCAL": bkg_loc,
    })
    return fate


def measure_jobs(diff_path: Path, ref_csv: Path, jobs: list, *, target_ra=None,
                 target_dec=None, match_radius: float = 1.0, bkg: str = "box",
                 bkg_mesh: int = BKG_MESH) -> dict | None:
    """Measure several position sets on one difference image, reading it and building
    its background maps once. jobs = [(out_cat, sexcat_path or None, assoc_radius), ...];
    sexcat_path None = ref-pos. Returns the ref-pos target fate (or None)."""
    ep = _Epoch(diff_path, bkg_mesh)
    fate = None
    for out_cat, sexcat_path, assoc_radius in jobs:
        f = _measure(ep, out_cat, ref_csv, sexcat_path, assoc_radius,
                     target_ra, target_dec, match_radius, bkg)
        fate = fate if f is None else f
    return fate


def measure_epoch(diff_path: Path, out_cat: Path, ref_csv: Path, *,
                  sexcat_path: Path | None = None, assoc_radius: float = 1.5,
                  target_ra=None, target_dec=None, match_radius: float = 1.0,
                  bkg: str = "box", bkg_mesh: int = BKG_MESH) -> dict | None:
    """Forced aperture photometry of one difference image. sexcat_path=None -> ref-pos
    (reference positions); else sci-pos (science-catalog positions tied to the
    reference by nearest match within assoc_radius). Returns the target fate dict."""
    return measure_jobs(diff_path, ref_csv, [(out_cat, sexcat_path, assoc_radius)],
                        target_ra=target_ra, target_dec=target_dec,
                        match_radius=match_radius, bkg=bkg, bkg_mesh=bkg_mesh)


def _aper_one(args: tuple) -> tuple[str, bool, str]:
    (diff_path, jobs, ref_csv, target_ra, target_dec, match_radius, bkg, bkg_mesh,
     status_dir) = args
    try:
        fate = measure_jobs(diff_path, ref_csv, jobs, target_ra=target_ra,
                            target_dec=target_dec, match_radius=match_radius,
                            bkg=bkg, bkg_mesh=bkg_mesh)
        # Persist the target fate under TargetStatus/ (survives --purge-batch), as the
        # simulate step did, so run_pipeline can classify missing target photometry.
        if fate is not None and status_dir is not None:
            Path(status_dir).mkdir(parents=True, exist_ok=True)
            (Path(status_dir) / f"{_ffd_from_path(diff_path)}.json").write_text(json.dumps(fate))
        return (diff_path.name, True, "ok")
    except Exception as exc:
        return (diff_path.name, False, f"{type(exc).__name__}: {exc}")


def _run(base_dir: Path, quadrants: list[dict], *, ref: bool, sci: bool, workers: int,
         force: bool, filefracdays, target_ra, target_dec, match_radius,
         assoc_radius_sci, bkg, bkg_mesh) -> int:
    """Measure the requested position sets (ref and/or sci) per epoch in one image pass."""
    label = "aperture" + ("_sci" if sci and not ref else "+sci" if sci else "")
    tasks = []
    for q in quadrants:
        field, fc, ccd, qid_ = q["field"], q["filtercode"], q["ccdid"], q["qid"]
        tag = f"{field:06d}_{fc}_c{ccd:02d}_q{qid_}"
        ref_csv = base_dir / "Catalogs" / f"{tag}(REFERENCE)[OBJECTS].csv"
        if not ref_csv.exists():
            logger.warning(f"Reference catalog missing for {tag} — skipping {label} "
                           f"(run catalog step first)")
            continue
        sub = Path(f"{field:06d}") / fc / f"{ccd:02d}" / str(qid_)
        status_dir = base_dir / "TargetStatus" / tag if ref else None
        for diff_path in sorted(q["sci_dir"].glob("*_scimrefdiffimg.fits")):
            if filefracdays is not None and _ffd_from_path(diff_path) not in filefracdays:
                continue
            name = diff_path.name.replace(".fits", "_sexout.fits")
            jobs = []
            if ref:
                out = base_dir / "SExCatalogs" / sub / name
                if force or not catalog_ok(out):
                    jobs.append((out, None, None))
            if sci:
                out = base_dir / "SExCatalogs_sci" / sub / name
                sexcat = diff_path.with_name(diff_path.name.replace("_scimrefdiffimg.fits",
                                                                    "_sexcat.fits"))
                if not sexcat.exists():
                    logger.debug(f"sexcat missing: {sexcat.name} — no sci-pos catalog")
                elif force or not catalog_ok(out):
                    jobs.append((out, sexcat, assoc_radius_sci))
            if jobs:
                tasks.append((diff_path, jobs, ref_csv, target_ra, target_dec,
                              match_radius, bkg, bkg_mesh, status_dir))

    if not tasks:
        logger.info(f"{label}: all catalogs already exist")
        return 0
    logger.info(f"{label}: {len(tasks)} epochs with {workers} workers (bkg={bkg})")
    n_done = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_aper_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            name, ok, msg = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                logger.warning(f"{label} failed: {name}: {msg}")
            if i % 20 == 0 or i == len(tasks):
                logger.info(f"{label}: {i}/{len(tasks)} — {n_done} done, {n_fail} failed")
    logger.info(f"{label}: {n_done} epochs measured, {n_fail} failures")
    return n_done


def step_aperture(base_dir: Path, quadrants: list[dict], workers: int = 4,
                  force: bool = False, verbose: bool = False, filefracdays: set | None = None,
                  target_ra: float | None = None, target_dec: float | None = None,
                  match_radius: float = 1.0, assoc_radius: float = 0.5,
                  bkg: str = "box", bkg_mesh: int = BKG_MESH,
                  also_sci: bool = False, assoc_radius_sci: float = 1.5) -> int:
    """Ref-pos forced photometry; drop-in for photometry.step_sextractor (assoc_radius
    is unused: positions are the reference positions themselves). also_sci=True also
    writes the sci-pos catalogs from the same image pass (the later sci step then finds
    them complete and skips)."""
    return _run(base_dir, quadrants, ref=True, sci=also_sci, workers=workers, force=force,
                filefracdays=filefracdays, target_ra=target_ra, target_dec=target_dec,
                match_radius=match_radius, assoc_radius_sci=assoc_radius_sci,
                bkg=bkg, bkg_mesh=bkg_mesh)


def step_aperture_scipos(base_dir: Path, quadrants: list[dict], workers: int = 4,
                         force: bool = False, verbose: bool = False,
                         filefracdays: set | None = None,
                         target_ra: float | None = None, target_dec: float | None = None,
                         match_radius: float = 1.0, assoc_radius: float = 1.5,
                         bkg: str = "box", bkg_mesh: int = BKG_MESH) -> int:
    """Sci-pos forced photometry; drop-in for photometry_scipos.step_sex_scipos."""
    return _run(base_dir, quadrants, ref=False, sci=True, workers=workers, force=force,
                filefracdays=filefracdays, target_ra=target_ra, target_dec=target_dec,
                match_radius=match_radius, assoc_radius_sci=assoc_radius,
                bkg=bkg, bkg_mesh=bkg_mesh)


def step_noop(*args, **kwargs) -> int:
    """Stands in for the simulate steps: forced photometry needs no detection image."""
    return 0
