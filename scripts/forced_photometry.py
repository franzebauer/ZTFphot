"""
forced_photometry.py
--------------------
Aperture photometry fallback for sources that SExtractor fails to detect in
the simulated detection image.

Background
----------
The photometry pipeline works in dual-image mode: SExtractor detects objects
in the simulated science image and measures their flux in the difference image.
At high S/N, SExtractor occasionally merges adjacent sources or uses a
detection threshold that excludes faint residuals, leaving some reference
catalog positions with no matched SExtractor detection.

When the light curve builder finds `detection = dis[i] < max_separation` is
False for a given reference source, it currently stores NaN for all flux
entries in that epoch — creating a gap in the light curve.  This module
provides a fallback: for any position where SExtractor found nothing, compute
aperture photometry directly on the difference image using photutils.

Apertures match the four sizes used in the main pipeline:
    3, 4, 6, 10 pixels in radius (plus a pseudo-AUTO using 6px as a proxy)

Sky background is estimated from a local annulus (inner: 12px, outer: 20px)
using sigma-clipped median, consistent with how SExtractor estimates local sky.

Integration with light_curves_builder.py
-----------------------------------------
After the existing match loop in `build_lightcurve_catalog`, if `detection`
is False, call `forced_aperture_photometry()` to fill in the flux columns:

    if not detection:
        forced = forced_aperture_photometry(
            diff_img_path = diff_img_path,
            ra = objects['RA'][i],
            dec = objects['DEC'][i],
        )
        if forced is not None:
            flux_dif    = forced['flux']
            fluxerr_dif = forced['fluxerr']
            detection   = False    # remains False — but fluxes are now filled
            flag_forced = True
        else:
            flux_dif    = [np.NAN for _ in Apertures.sizes]
            fluxerr_dif = [np.NAN for _ in Apertures.sizes]
            flag_forced = False

The `FLAG_FORCED` column in the light curve (True = photutils measurement,
False = SExtractor detection) lets downstream code treat forced-photometry
epochs differently if needed (e.g. for variability classifiers that require
clean detections).

Aperture sizes
--------------
ZTF refsexcat uses a fixed pixel scale of ~1.01 arcsec/pixel.  The four
aperture radii (3, 4, 6, 10 px) correspond to ~3.0, 4.0, 6.1, 10.1 arcsec
diameters respectively.  We use the same radii here so that forced-photometry
fluxes are on the same scale as SExtractor fluxes.

Notes on noise
--------------
The difference image has already been sky-subtracted by the ZTF pipeline,
so the sky level should be near zero.  However, the subtraction residuals
vary spatially.  We estimate local background from an annulus around each
source to account for residual gradients.  For most ZTF difference images,
the local background estimate is small relative to source flux, but it
matters for non-detections (negative flux = source faded relative to reference).

Pixel units
-----------
Flux is returned in the same ADU units as FLUX_APER / FLUX_AUTO in the
SExtractor catalogs (counts in the difference image per aperture), so that
the light curve builder's `calculate_flux_from_magnitude()` function can be
used without modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aperture configuration — must match make_catalog.py and light_curves_builder.py
# ---------------------------------------------------------------------------

# Aperture radii in pixels, matching the four-aperture scheme
APERTURE_RADII_PX = [3.0, 4.0, 6.0, 10.0]    # 3px, 4px, 6px, 10px
APERTURE_LABELS   = ["3", "4", "6", "10"]      # used as dict keys

# Sky annulus: inner / outer radii in pixels
SKY_INNER_PX = 12.0
SKY_OUTER_PX = 20.0

# Sigma-clipping iterations and threshold for local sky estimation
SKY_SIGMA_CLIP = 3.0
SKY_SIGMA_ITERS = 5


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

def forced_aperture_photometry(
    diff_img_path: str | Path,
    ra: float,
    dec: float,
    aperture_radii_px: list[float] = None,
    sky_inner_px: float = SKY_INNER_PX,
    sky_outer_px: float = SKY_OUTER_PX,
    gain: Optional[float] = None,
    wcs: Optional[WCS] = None,
    data: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    Measure aperture photometry at a fixed sky position on a difference image.

    This function is designed to be called for sources that SExtractor did not
    detect, providing a photutils-based forced measurement at the reference
    catalog position.

    Parameters
    ----------
    diff_img_path : str or Path
        Path to the difference image (.fits, funpacked).  Not required if
        `data` and `wcs` are provided (useful when processing many sources
        per epoch to avoid re-reading the image).
    ra, dec : float
        J2000 sky position in decimal degrees.
    aperture_radii_px : list of float, optional
        Aperture radii in pixels.  Defaults to [3, 4, 6, 10].
    sky_inner_px, sky_outer_px : float
        Inner and outer radii of the sky annulus (pixels).
    gain : float, optional
        CCD gain (e-/ADU) for Poisson noise contribution.  If None, the gain
        is read from the image header (keyword GAIN); falls back to 1.0.
    wcs : astropy.wcs.WCS, optional
        Pre-loaded WCS.  Pass to avoid re-parsing the header on every call.
    data : np.ndarray, optional
        Pre-loaded image data.  Pass alongside `wcs` to avoid re-reading.

    Returns
    -------
    dict with keys:
        'flux'       : list[float] — aperture sums (ADU) for each radius
        'fluxerr'    : list[float] — total flux uncertainties (ADU)
        'sky_per_px' : float       — local sky background per pixel (ADU)
        'sky_rms'    : float       — sky RMS in annulus (ADU)
        'x_pix'      : float       — source x-pixel (0-indexed)
        'y_pix'      : float       — source y-pixel (0-indexed)
        'radii_px'   : list[float] — aperture radii used
    Returns None if the source falls outside the image boundary or the
    sky annulus contains fewer than 10 finite pixels.
    """
    from photutils.aperture import (
        CircularAperture,
        CircularAnnulus,
        aperture_photometry,
    )
    from astropy.stats import sigma_clipped_stats

    if aperture_radii_px is None:
        aperture_radii_px = APERTURE_RADII_PX

    # ---- Load image if not pre-provided ----
    if data is None or wcs is None:
        diff_img_path = Path(diff_img_path)
        if not diff_img_path.exists():
            logger.warning(f"Diff image not found: {diff_img_path}")
            return None
        with fits.open(diff_img_path) as hdul:
            header = hdul[0].header
            raw_data = hdul[0].data.astype(float)
            wcs_local = WCS(header)
            gain_header = header.get("GAIN", 1.0)
    else:
        raw_data = data
        wcs_local = wcs
        gain_header = gain if gain is not None else 1.0

    if gain is None:
        gain = float(gain_header) if gain_header else 1.0

    # ---- World → pixel coordinates ----
    # astropy WCS world_to_pixel returns (x, y) for a SkyCoord
    from astropy.coordinates import SkyCoord
    sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    x_pix, y_pix = wcs_local.world_to_pixel(sky_coord)
    x_pix = float(x_pix)
    y_pix = float(y_pix)

    ny, nx = raw_data.shape

    # Guard: source off the image
    pad = sky_outer_px + 2
    if (x_pix < pad or x_pix > nx - pad - 1 or
            y_pix < pad or y_pix > ny - pad - 1):
        logger.debug(f"Source at ({ra:.5f}, {dec:.5f}) maps to pixel "
                     f"({x_pix:.1f}, {y_pix:.1f}) — too close to edge, skipping")
        return None

    position = (x_pix, y_pix)

    # ---- Sky background estimation from annulus ----
    sky_ann = CircularAnnulus(position, r_in=sky_inner_px, r_out=sky_outer_px)

    # Create a mask for NaN/inf pixels
    nan_mask = ~np.isfinite(raw_data)

    # Extract annulus pixel values via a simple bitmask
    sky_mask = sky_ann.to_mask(method="center")
    sky_data_full = sky_mask.multiply(raw_data)
    sky_pixels = sky_data_full[sky_mask.data > 0]
    sky_pixels = sky_pixels[np.isfinite(sky_pixels)]

    if sky_pixels.size < 10:
        logger.debug(f"Too few sky pixels ({sky_pixels.size}) at "
                     f"({ra:.5f}, {dec:.5f}) — skipping forced photometry")
        return None

    _, sky_med, sky_std = sigma_clipped_stats(
        sky_pixels, sigma=SKY_SIGMA_CLIP, maxiters=SKY_SIGMA_ITERS
    )
    sky_per_px = float(sky_med)
    sky_rms    = float(sky_std)

    # ---- Aperture photometry ----
    # Replace NaNs with sky background for photutils (it doesn't handle NaN natively)
    clean_data = np.where(np.isfinite(raw_data), raw_data, sky_per_px)

    apertures = [CircularAperture(position, r=r) for r in aperture_radii_px]

    phot_table = aperture_photometry(
        clean_data,
        apertures,
        mask=nan_mask,
    )

    fluxes   = []
    fluxerrs = []

    for j, (aper, radius) in enumerate(zip(apertures, aperture_radii_px)):
        col_name = f"aperture_sum_{j}" if len(apertures) > 1 else "aperture_sum"

        raw_sum = float(phot_table[col_name][0])
        n_pix   = np.pi * radius**2   # area in pixels (before masking correction)

        # Sky-subtracted flux
        sky_sum = sky_per_px * n_pix
        flux    = raw_sum - sky_sum

        # Uncertainty: sky variance + Poisson noise from source counts
        # σ²_sky   = n_pix × sky_rms²
        # σ²_poisson = |flux| / gain   (abs because diff image can go negative)
        sky_var     = n_pix * sky_rms**2
        poisson_var = max(abs(flux), 0.0) / gain
        fluxerr     = float(np.sqrt(sky_var + poisson_var))

        fluxes.append(float(flux))
        fluxerrs.append(fluxerr)

    return {
        "flux":       fluxes,
        "fluxerr":    fluxerrs,
        "sky_per_px": sky_per_px,
        "sky_rms":    sky_rms,
        "x_pix":      x_pix,
        "y_pix":      y_pix,
        "radii_px":   aperture_radii_px,
    }


# ---------------------------------------------------------------------------
# Batch version: measure all reference positions on one diff image at once
# ---------------------------------------------------------------------------

def forced_photometry_batch(
    diff_img_path: str | Path,
    ra_list: list[float],
    dec_list: list[float],
    aperture_radii_px: list[float] = None,
    sky_inner_px: float = SKY_INNER_PX,
    sky_outer_px: float = SKY_OUTER_PX,
) -> list[Optional[dict]]:
    """
    Measure forced aperture photometry for a list of positions on one image.

    More efficient than calling forced_aperture_photometry() in a loop because
    the image is loaded and the WCS is parsed only once.

    Parameters
    ----------
    diff_img_path : str or Path
        Path to the difference image.
    ra_list, dec_list : list of float
        Sky positions to measure (decimal degrees, J2000).
    aperture_radii_px : list of float, optional
        Radii in pixels.  Defaults to [3, 4, 6, 10].

    Returns
    -------
    list of dict or None — one entry per (ra, dec) pair.
    None entries indicate the source was off-chip or had bad sky coverage.
    """
    diff_img_path = Path(diff_img_path)
    if not diff_img_path.exists():
        logger.warning(f"Diff image not found: {diff_img_path}")
        return [None] * len(ra_list)

    # Load image once
    with fits.open(diff_img_path) as hdul:
        header = hdul[0].header
        data   = hdul[0].data.astype(float)
        wcs    = WCS(header)
        gain   = float(header.get("GAIN", 1.0))

    results = []
    for ra, dec in zip(ra_list, dec_list):
        result = forced_aperture_photometry(
            diff_img_path=diff_img_path,   # unused since data/wcs are passed
            ra=ra,
            dec=dec,
            aperture_radii_px=aperture_radii_px,
            sky_inner_px=sky_inner_px,
            sky_outer_px=sky_outer_px,
            gain=gain,
            wcs=wcs,
            data=data,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Patch for light_curves_builder.py
# ---------------------------------------------------------------------------
#
# The following is a drop-in patch for the inner loop of
# build_lightcurve_catalog() in light_curves_builder.py.
#
# Replace the block:
#
#     detection = dis[i] < max_separation
#     if 'FLUX_APER' in table.columns:
#         if detection:
#             ...
#         else:
#             flux_dif    = [np.NAN for _ in Apertures.sizes]
#             fluxerr_dif = [np.NAN for _ in Apertures.sizes]
#
# With the patched version in `patched_detection_block()` below.
# Or, more practically, import and call `fill_missing_epochs()` once per
# epoch before entering the per-source loop.
#
# ---------------------------------------------------------------------------

def fill_missing_epochs(
    reference_catalog: "pd.DataFrame",
    detections: list[bool],
    diff_img_path: str | Path,
    aperture_radii_px: list[float] = None,
) -> tuple[list[list[float]], list[list[float]], list[bool]]:
    """
    For any reference source where SExtractor did not make a detection,
    compute forced aperture photometry on the difference image.

    Parameters
    ----------
    reference_catalog : pd.DataFrame
        Reference catalog with 'RA' and 'DEC' columns (as produced by
        make_catalog.py after the ALPHAWIN_J2000/DELTAWIN_J2000 rename).
    detections : list of bool
        SExtractor detection flag per source (True = detected, False = missed).
    diff_img_path : str or Path
        Difference image for this epoch.
    aperture_radii_px : list of float, optional
        Aperture radii to use (default: [3, 4, 6, 10]).

    Returns
    -------
    flux_dif_list    : list of lists — flux per source per aperture [+AUTO proxy]
    fluxerr_dif_list : list of lists — uncertainty per source per aperture
    flag_forced_list : list of bool  — True where forced photometry was used

    Notes
    -----
    AUTO aperture (5th element in Apertures.sizes) has no direct equivalent
    in forced photometry.  We use the 6px aperture result as a proxy for AUTO,
    since ZTF FWHM ~2" (≈2px) makes AUTO and 6px very similar in practice.
    AUTO_proxy_index = 2  (index of the 6px aperture in the radii list)
    """
    import pandas as pd

    n_sources = len(reference_catalog)
    n_aper    = len(APERTURE_RADII_PX) + 1   # +1 for AUTO proxy

    flux_dif_list    = [[np.nan] * n_aper for _ in range(n_sources)]
    fluxerr_dif_list = [[np.nan] * n_aper for _ in range(n_sources)]
    flag_forced_list = [False] * n_sources

    # Collect positions that need forced photometry
    missing_idx  = [i for i, det in enumerate(detections) if not det]
    if not missing_idx:
        return flux_dif_list, fluxerr_dif_list, flag_forced_list

    missing_ra   = [reference_catalog['RA'].iloc[i]  for i in missing_idx]
    missing_dec  = [reference_catalog['DEC'].iloc[i] for i in missing_idx]

    forced_results = forced_photometry_batch(
        diff_img_path=diff_img_path,
        ra_list=missing_ra,
        dec_list=missing_dec,
        aperture_radii_px=aperture_radii_px or APERTURE_RADII_PX,
    )

    AUTO_PROXY_IDX = 2   # 6px aperture as AUTO proxy

    for list_idx, src_idx in enumerate(missing_idx):
        result = forced_results[list_idx]
        if result is None:
            continue

        fluxes   = result["flux"]    # 4 values: 3px, 4px, 6px, 10px
        fluxerrs = result["fluxerr"]

        # Append AUTO proxy (6px repeated at index 4)
        flux_with_auto    = fluxes   + [fluxes[AUTO_PROXY_IDX]]
        fluxerr_with_auto = fluxerrs + [fluxerrs[AUTO_PROXY_IDX]]

        flux_dif_list[src_idx]    = flux_with_auto
        fluxerr_dif_list[src_idx] = fluxerr_with_auto
        flag_forced_list[src_idx] = True

    n_filled = sum(flag_forced_list)
    if n_filled:
        logger.debug(f"Forced photometry filled {n_filled}/{len(missing_idx)} "
                     f"missed sources in {Path(diff_img_path).name}")

    return flux_dif_list, fluxerr_dif_list, flag_forced_list


# ---------------------------------------------------------------------------
# Standalone diagnostic: test forced photometry on a known source
# ---------------------------------------------------------------------------

def diagnose_forced_vs_sextractor(
    diff_img_path: str | Path,
    sexcat_path: str | Path,
    ra: float,
    dec: float,
    match_radius_arcsec: float = 1.5,
) -> dict:
    """
    Compare forced photometry flux with the SExtractor measurement at the
    nearest matched source, to validate the calibration of the fallback.

    Useful for sanity-checking on a known bright source before trusting the
    forced photometry for faint transients.

    Returns a dict with:
        'sextractor_flux' : list[float]  — FLUX_APER[0..3] + FLUX_AUTO from sexcat
        'forced_flux'     : list[float]  — forced photometry at matched position
        'separation_arcsec': float       — angular distance to nearest SExtractor source
        'forced_result'   : dict         — full forced photometry output
    """
    from astropy.table import Table
    from astropy.coordinates import SkyCoord

    # Load diff image SExtractor catalog
    with fits.open(sexcat_path) as hdul:
        cat = Table(hdul[2].data) if len(hdul) > 2 else Table(hdul[1].data)

    cat_coords = SkyCoord(
        ra=cat["ALPHAWIN_J2000"] * u.deg,
        dec=cat["DELTAWIN_J2000"] * u.deg,
    )
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    idx, sep, _ = target.match_to_catalog_sky(cat_coords)

    sep_arcsec = float(sep.arcsec)
    row = cat[idx]

    # SExtractor fluxes (FLUX_APER may be a 4- or 5-element vector)
    if "FLUX_APER" in cat.colnames:
        se_flux = list(row["FLUX_APER"]) + [float(row.get("FLUX_AUTO", np.nan))]
    else:
        se_flux = [np.nan] * 5

    # Forced photometry at the given position
    forced = forced_aperture_photometry(diff_img_path, ra, dec)

    return {
        "sextractor_flux":   se_flux,
        "forced_flux":       (forced["flux"] + [forced["flux"][2]]) if forced else None,
        "separation_arcsec": sep_arcsec,
        "forced_result":     forced,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Forced aperture photometry at a fixed sky position on a ZTF diff image."
    )
    parser.add_argument("--diffimg", type=Path, required=True,
                        help="Difference image FITS file (funpacked)")
    parser.add_argument("--ra",  type=float, required=True,
                        help="Target RA (J2000 decimal degrees)")
    parser.add_argument("--dec", type=float, required=True,
                        help="Target Dec (J2000 decimal degrees)")
    parser.add_argument("--sexcat", type=Path, default=None,
                        help="Optional: SExtractor catalog to compare against (diagnostic mode)")
    parser.add_argument("--radii", nargs="+", type=float, default=APERTURE_RADII_PX,
                        help=f"Aperture radii in pixels (default: {APERTURE_RADII_PX})")

    args = parser.parse_args()

    if args.sexcat:
        result = diagnose_forced_vs_sextractor(
            diff_img_path=args.diffimg,
            sexcat_path=args.sexcat,
            ra=args.ra,
            dec=args.dec,
        )
        print("\n=== SExtractor vs Forced Photometry comparison ===")
        print(f"Nearest SExtractor source: {result['separation_arcsec']:.2f}\"")
        print(f"SExtractor fluxes : {result['sextractor_flux']}")
        print(f"Forced     fluxes : {result['forced_flux']}")
        if result["forced_result"]:
            print(f"Sky per pixel     : {result['forced_result']['sky_per_px']:.4f} ADU")
            print(f"Sky RMS           : {result['forced_result']['sky_rms']:.4f} ADU")
            print(f"Pixel position    : x={result['forced_result']['x_pix']:.2f}, "
                  f"y={result['forced_result']['y_pix']:.2f}")
    else:
        result = forced_aperture_photometry(
            diff_img_path=args.diffimg,
            ra=args.ra,
            dec=args.dec,
            aperture_radii_px=args.radii,
        )
        if result is None:
            print("ERROR: source off-chip or bad sky coverage", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({
            "ra":         args.ra,
            "dec":        args.dec,
            "radii_px":   result["radii_px"],
            "flux":       result["flux"],
            "fluxerr":    result["fluxerr"],
            "sky_per_px": result["sky_per_px"],
            "sky_rms":    result["sky_rms"],
            "x_pix":      result["x_pix"],
            "y_pix":      result["y_pix"],
        }, indent=2))
