"""
aperture_photometry.py
----------------------
Forced circular-aperture photometry on ZTF difference images (photutils) — the
pipeline's photometry step (`phot` / `phot_sci`).

Positions are forced, not detected:
  ref-pos  — every reference-catalog source (Catalogs/*(REFERENCE)[OBJECTS].csv),
             including injected ones: the --ra/--dec target and --inject-targets sources
             with no reference source within the target match radius are added to the
             catalog by the catalog step (MAG_APER = 99: no reference flux).
  sci-pos  — every source of the epoch's ZTF science catalog (*_sexcat.fits), tied to
             the nearest reference entry within --assoc-radius-sci (unmatched dropped).

A source is measured when its centre is on the frame and > 50% of the pixels within
0.75*SEEING of it are finite.

Apertures: PHOT_APERTURES DIAMETERS (3,4,6,10 px -> radii 1.5,2,3,5), exact overlap.
Masked (NaN) pixels are excluded from the sums.

Background (--aper-bkg):
  local           — sigma-clipped median in a circular annulus [r_in, r_in + 35 px],
                    r_in = 1.58*SEEING.
  box (default)   — 3-sigma-clipped mean in a SQUARE annulus (half-sizes 1.58*SEEING ..
                    +35 px) via integral images: same pixel-level local estimate, ~5x
                    cheaper (pixels clipped against the 64 px background level/RMS boxes).
  mesh            — Background2D level (--aper-bkg-mesh px mesh, default 64; 3x3 filter)
                    at the source position.
  none            — no subtraction (difference images are already background-subtracted).
Errors: FLUXERR = sigma_rms(x, y) * sqrt(unmasked aperture area), sigma_rms from a
64 px Background2D RMS mesh evaluated at the source positions (cubic spline) — the
background-only error.

With ref + sci requested, both position sets are measured in one pass per image.

FLAGS (per-epoch bits of the parquet FLAGS bitmask — see ztfphot_flags.py):
   1 — another catalog source within the largest aperture radius
   8 — the primary 4 px aperture extends beyond the image edge
  16 — masked (NaN) pixels in the primary 4 px aperture
  32 — edge or masked pixels only in the larger 6 / 10 px apertures
8 and 16 corrupt the 4 px measurement (default rejection); calibration rejects
calibrator measurements with 8 | 16 | 32 (all apertures are calibrated).

Output: one LDAC catalog per epoch, PhotCatalogs{,_sci}/{field}/{fc}/{ccd}/{qid}/
*_phot.fits (LDAC_IMHEAD = the difference-image header; LDAC_OBJECTS with FLUX_APER(4),
FLUXERR_APER(4), VECTOR_ASSOC = 1-based reference index, positions, FLAGS), written to a
temporary file and renamed, so a crash never leaves a partial catalog.
migrate_legacy_layout() renames SExCatalogs{,_sci}/ *_sexout.fits trees to this layout.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import functools

import numpy as np

logger = logging.getLogger(__name__)

# Aperture DIAMETERS in pixels — do not change (match the original Rust pipeline;
# k=0 3 px, k=1 4 px = primary -> FLUX_4_TOT_uJy / MAG_4_TOT_AB, k=2 6 px, k=3 10 px)
PHOT_APERTURES = "3,4,6,10"

BKG_ANNULUS_THICK = 35.0      # px, local-background annulus thickness
BKG_MESH = 64                 # px, Background2D mesh (RMS map, mesh background)
BKG_FILTER = 3                # Background2D median-filter size (meshes)
ISO_FACTOR = 1.58             # annulus inner radius / SEEING

# On-disk layout of the per-epoch photometry catalogs, and the names migrated to it
CATALOG_DIRS = {"": "PhotCatalogs", "_sci": "PhotCatalogs_sci"}
CATALOG_SUFFIX = "_phot.fits"
_LEGACY_DIRS = {"": "SExCatalogs", "_sci": "SExCatalogs_sci"}
_LEGACY_SUFFIX = "_sexout.fits"

from ztfphot_flags import F as _F
FLAG_NEIGHBOUR, FLAG_EDGE, FLAG_MASKED, FLAG_OUTER = (_F.NEIGHBOUR, _F.EDGE, _F.MASKED,
                                                      _F.OUTER_APERTURES)
PRIMARY_APERTURE = 1                          # index of the 4 px aperture in PHOT_APERTURES
PHOTCODE = "ztfphot-aper"                     # LDAC header tag identifying these catalogs


def _ffd_from_path(p: Path) -> str:
    return p.name.split("_")[1]


def _aperture_radii() -> list[float]:
    return [float(d) / 2.0 for d in PHOT_APERTURES.split(",")]


_ONES: dict = {}


def _ones(shape) -> np.ndarray:
    """A cached image of ones (the in-image pixel count in one aperture pass)."""
    if shape not in _ONES:
        _ONES.clear()
        _ONES[shape] = np.ones(shape)
    return _ONES[shape]


def catalog_dir(base_dir: Path, suffix: str, field: int, fc: str, ccd: int, qid: int) -> Path:
    """PhotCatalogs{suffix}/{field:06d}/{fc}/{ccd:02d}/{qid} under base_dir."""
    return (Path(base_dir) / CATALOG_DIRS[suffix] / f"{int(field):06d}" / fc
            / f"{int(ccd):02d}" / str(int(qid)))


def catalog_name(diff_or_stem: str) -> str:
    """Catalog file name for a difference image name (ztf_..._scimrefdiffimg.fits) or
    for an epoch stem (ztf_..._scimrefdiffimg)."""
    stem = diff_or_stem[:-5] if diff_or_stem.endswith(".fits") else diff_or_stem
    return stem + CATALOG_SUFFIX


def migrate_legacy_layout(base_dir: Path) -> int:
    """Rename a tree in place: SExCatalogs{,_sci}/ -> PhotCatalogs{,_sci}/ and
    *_sexout.fits -> *_phot.fits (merging into an existing new-style directory). The
    catalog contents are unchanged (the LDAC format is the same). Returns files renamed."""
    import shutil
    n = 0
    for suffix, old in _LEGACY_DIRS.items():
        src = Path(base_dir) / old
        if not src.is_dir():
            continue
        dst = Path(base_dir) / CATALOG_DIRS[suffix]
        n0 = n
        if not dst.exists():
            src.rename(dst)
        for f in sorted((dst if not src.exists() else src).rglob(f"*{_LEGACY_SUFFIX}")):
            rel = f.relative_to(dst if not src.exists() else src)
            target = (dst / rel).with_name(f.name[:-len(_LEGACY_SUFFIX)] + CATALOG_SUFFIX)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                f.unlink()                     # a new-style catalog already exists: keep it
            else:
                f.rename(target)
                n += 1
        if src.exists():                       # move any other file, then drop the tree
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    t = dst / f.relative_to(src)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    if not t.exists():
                        f.rename(t)
            shutil.rmtree(src, ignore_errors=True)
        logger.info(f"migrated {old}/ -> {CATALOG_DIRS[suffix]}/ ({n - n0} catalogs renamed to *{CATALOG_SUFFIX})")
    return n


def catalog_ok(path: Path) -> bool:
    """True if `path` is a complete LDAC catalog (guards against the header-only /
    empty / truncated catalog files)."""
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


def _catalog_card(path: Path, key: str):
    """Value of header card `key` of an existing catalog (None when absent)."""
    from astropy.io import fits
    try:
        with fits.open(path) as h:
            for c in h[1].data["Field Header Card"][0]:
                c = c.decode() if isinstance(c, bytes) else str(c)
                if c.startswith(f"{key:<8}="):
                    return fits.Card.fromstring(c).value
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=4)
def _ref_positions(ref_csv: str, mtime: float):
    """Reference catalog read once per worker process (the same file for every epoch):
    positions, and the 4 px reference magnitude (> 90: injected source)."""
    import pandas as pd
    cols = pd.read_csv(ref_csv, nrows=0).columns
    return pd.read_csv(ref_csv, usecols=["ALPHAWIN_J2000", "DELTAWIN_J2000"]
                       + (["MAG_APER_4px"] if "MAG_APER_4px" in cols else []))


def _ref_fingerprint(ref_csv: Path) -> str:
    """REFCAT header value: number of catalog rows and a checksum of their positions.
    A photometry catalog's VECTOR_ASSOC indexes this catalog, so it is valid only for
    the catalog it was measured with."""
    import zlib
    ref = _ref_positions(str(ref_csv), Path(ref_csv).stat().st_mtime)
    pos = np.round(ref[["ALPHAWIN_J2000", "DELTAWIN_J2000"]].to_numpy(float), 7)
    return f"{len(ref)} {zlib.crc32(np.ascontiguousarray(pos).tobytes()):08x}"


def _assoc_entries(ref_csv: Path):
    """(index, ra, dec): every reference-catalog row, 1-based index."""
    import pandas as pd
    ref = _ref_positions(str(ref_csv), Path(ref_csv).stat().st_mtime)
    ra = pd.to_numeric(ref["ALPHAWIN_J2000"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(ref["DELTAWIN_J2000"], errors="coerce").to_numpy(float)
    return np.arange(1, len(ref) + 1), ra, dec


def _target_row(ref_csv: Path, target_ra, target_dec, match_radius):
    """(matched, injected row): whether a real reference source lies within
    match_radius of the target, and otherwise the 0-based row of the target's injected
    entry (None if the catalog has none — the catalog step was not run for it)."""
    import pandas as pd
    ref = _ref_positions(str(ref_csv), Path(ref_csv).stat().st_mtime)
    ra = pd.to_numeric(ref["ALPHAWIN_J2000"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(ref["DELTAWIN_J2000"], errors="coerce").to_numpy(float)
    mag = (pd.to_numeric(ref["MAG_APER_4px"], errors="coerce").to_numpy(float)
           if "MAG_APER_4px" in ref else np.zeros(len(ref)))
    dra = (((ra - target_ra) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(target_dec))
    sep = np.nan_to_num(np.hypot(dra, dec - target_dec) * 3600.0, nan=np.inf)
    inj = mag > 90
    if (sep[~inj] < match_radius).any():
        return True, None
    j = np.where(inj & (sep < match_radius))[0]
    return False, (int(j[np.argmin(sep[j])]) if len(j) else None)


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

    idx, ra, dec = _assoc_entries(ref_csv)
    ref_card = _ref_fingerprint(ref_csv)
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
        # one science detection per reference source: the nearest
        order = np.lexsort((sep.arcsec, j))
        first = np.zeros(len(j), bool)
        first[order[np.r_[True, j[order][1:] != j[order][:-1]]]] = True
        keep &= first
        idx, ra, dec = idx[ok][j[keep]], sra[keep], sdec[keep]

    x, y = ep.wcs.all_world2pix(ra, dec, 0)
    # centre on the frame: within the pixel edges (pixel centres at integers, 0-based)
    on = (np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x < ncols - 0.5)
          & (y >= -0.5) & (y < nrows - 0.5))

    # > 50% finite pixels within 0.75*SEEING (the old "paint" criterion)
    good = np.zeros(len(x), bool)
    if on.any():
        ap_v = CircularAperture(np.column_stack([x[on], y[on]]), r=0.75 * fwhm)
        # one pass: valid pixels (sum of fin) and all in-image pixels (error=1 -> sqrt(count))
        t = aperture_photometry(ep.fin, ap_v, method="center", error=_ones(ep.fin.shape))
        good[on] = np.asarray(t["aperture_sum"]) > 0.5 * np.asarray(t["aperture_sum_err"]) ** 2

    # Target fate (read by run_pipeline's target status): matched = has a reference
    # source; in_frame / painted for its injected catalog entry
    fate = None
    if target_ra is not None and target_dec is not None and sexcat_path is None:
        matched, j = _target_row(ref_csv, target_ra, target_dec, match_radius)
        fate = {"matched": matched, "in_frame": bool(j is not None and on[j]),
                "painted": bool(j is not None and good[j])}

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

        outer = np.zeros(n, bool)
        for k, r in enumerate(radii):
            ap = CircularAperture(pos, r=r)
            # one exact pass: the flux, and with error = the 0/1 valid-pixel map the
            # propagated error is sqrt(sum overlap over valid pixels) = sqrt(effective area)
            t = aperture_photometry(ep.data0, ap, method="exact", error=ep.fin)
            s = np.asarray(t["aperture_sum"], float)
            a_eff = np.asarray(t["aperture_sum_err"], float) ** 2
            flux[:, k] = s - bkg_loc * a_eff
            ferr[:, k] = rms * np.sqrt(a_eff)
            edge_k = (x - r < -0.5) | (x + r > ncols - 0.5) | (y - r < -0.5) | (y + r > nrows - 0.5)
            masked_k = (a_eff < ap.area - 1e-6) & ~edge_k     # off-edge pixels aren't "masked"
            if k == PRIMARY_APERTURE:       # the 4 px measurement itself is affected
                flags |= np.where(edge_k, FLAG_EDGE, 0).astype(np.int16)
                flags |= np.where(masked_k, FLAG_MASKED, 0).astype(np.int16)
            elif r > radii[PRIMARY_APERTURE]:
                outer |= edge_k | masked_k
        # only the larger apertures affected (the 3 px one lies inside the 4 px one)
        flags |= np.where(outer & ((flags & (FLAG_EDGE | FLAG_MASKED)) == 0),
                          FLAG_OUTER, 0).astype(np.int16)
        rmax = max(radii)
        if n > 1:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pos).query(pos, k=2)
            flags |= np.where(d[:, 1] < rmax, FLAG_NEIGHBOUR, 0).astype(np.int16)

    header = ep.header.copy()
    header["PHOTCODE"] = (PHOTCODE, "forced aperture photometry (photutils)")
    header["APBKG"] = (bkg, "aperture background mode")
    header["REFCAT"] = (ref_card, "reference catalog measured with: rows crc32")
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


def _aper_one(args: tuple) -> tuple[str, bool, str]:
    (diff_path, jobs, ref_csv, target_ra, target_dec, match_radius, bkg, bkg_mesh,
     status_dir) = args
    try:
        fate = measure_jobs(diff_path, ref_csv, jobs, target_ra=target_ra,
                            target_dec=target_dec, match_radius=match_radius,
                            bkg=bkg, bkg_mesh=bkg_mesh)
        # Persist the target fate under TargetStatus/ (survives --purge-batch), so
        # run_pipeline can classify missing target photometry.
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
        # A catalog measured with another reference catalog (e.g. before a target was
        # injected into it) indexes the wrong rows: re-measured. Catalogs written before
        # the REFCAT card are kept unless this run's target is an injected entry (older
        # versions appended it as a temporary row N+1).
        ref_card = _ref_fingerprint(ref_csv)
        tgt_injected = (target_ra is not None and target_dec is not None
                        and _target_row(ref_csv, target_ra, target_dec, match_radius)[1] is not None)

        def _stale(out):
            if force or not catalog_ok(out):
                return True
            card = _catalog_card(out, "REFCAT")
            return card != ref_card if card is not None else tgt_injected
        for diff_path in sorted(q["sci_dir"].glob("*_scimrefdiffimg.fits")):
            if filefracdays is not None and _ffd_from_path(diff_path) not in filefracdays:
                continue
            name = catalog_name(diff_path.name)
            jobs = []
            if ref:
                out = base_dir / CATALOG_DIRS[""] / sub / name
                if _stale(out):
                    jobs.append((out, None, None))
            if sci:
                out = base_dir / CATALOG_DIRS["_sci"] / sub / name
                sexcat = diff_path.with_name(diff_path.name.replace("_scimrefdiffimg.fits",
                                                                    "_sexcat.fits"))
                if not sexcat.exists():
                    logger.debug(f"sexcat missing: {sexcat.name} — no sci-pos catalog")
                elif _stale(out):
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
                  force: bool = False, filefracdays: set | None = None,
                  target_ra: float | None = None, target_dec: float | None = None,
                  match_radius: float = 1.0,
                  bkg: str = "box", bkg_mesh: int = BKG_MESH,
                  also_sci: bool = False, assoc_radius_sci: float = 1.5) -> int:
    """Ref-pos forced photometry (step `phot`) at the reference positions. also_sci=True also
    writes the sci-pos catalogs from the same image pass (the later sci step then finds
    them complete and skips)."""
    return _run(base_dir, quadrants, ref=True, sci=also_sci, workers=workers, force=force,
                filefracdays=filefracdays, target_ra=target_ra, target_dec=target_dec,
                match_radius=match_radius, assoc_radius_sci=assoc_radius_sci,
                bkg=bkg, bkg_mesh=bkg_mesh)


def step_aperture_scipos(base_dir: Path, quadrants: list[dict], workers: int = 4,
                         force: bool = False,
                         filefracdays: set | None = None,
                         target_ra: float | None = None, target_dec: float | None = None,
                         match_radius: float = 1.0, assoc_radius: float = 1.5,
                         bkg: str = "box", bkg_mesh: int = BKG_MESH) -> int:
    """Sci-pos forced photometry (step `phot_sci`)."""
    return _run(base_dir, quadrants, ref=False, sci=True, workers=workers, force=force,
                filefracdays=filefracdays, target_ra=target_ra, target_dec=target_dec,
                match_radius=match_radius, assoc_radius_sci=assoc_radius,
                bkg=bkg, bkg_mesh=bkg_mesh)
