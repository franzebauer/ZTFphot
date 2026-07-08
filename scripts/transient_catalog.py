"""
transient_catalog.py
--------------------
Inject additional sources (transients not present in the ZTF reference image)
into the reference SExtractor catalog before the simulation step, so that
simulate_science.py will paint them into the simulated detection image and
SExtractor will recover their positions in the difference image.

Background
----------
The ZTF reference image is a deep stack of historical science images.  Any
object that brightened *after* the reference epoch (supernovae, TDEs, novae,
AGN flares, etc.) will be absent from the reference SExtractor catalog
(refsexcat.fits).  Because simulate_science.py uses the reference catalog to
build the simulated detection image, SExtractor running in dual-image mode
will never attempt to measure flux at the transient's position, resulting in
missing epochs for that object.

This module solves the problem by:
  1. Loading a list of additional positions (from a user CSV or TNS API query).
  2. Converting them into synthetic rows compatible with the refsexcat.fits
     FITS binary table format.
  3. Writing an augmented catalog file that simulate_science.py can accept
     alongside (or instead of) the original refsexcat.fits.

Injected sources are marked with:
  - MAG_AUTO / MAG_APER = 99  (physically impossible sentinel; they have no real
    reference-image flux). This surfaces as MAG_4_REF > 90 in the light curves,
    so injected sources are distinguishable in the output with no extra column,
    and it also selects the boosted detection amplitude in simulate_science.py.
  - FLAGS = 1 if a reference neighbour lies within 3 arcsec (SExtractor "close
    neighbour" bit, so blends are flagged like real crowded sources), else 0.

Modes
-----
Mode A — User-supplied CSV
    A CSV with at minimum 'ra' and 'dec' columns (decimal degrees, J2000).
    Optional columns: 'name', 'mag_estimate', 'redshift', 'classification'.

    python transient_catalog.py --mode user \\
        --input my_targets.csv \\
        --refsexcat ../data/Reference/000443/zr/ccd16/q2/ztf_000443_zr_c16_q2_refsexcat.fits \\
        --diffimg   ../data/Science/000443/zr/ccd16/q2/ztf_20210101123456_000443_zr_c16_o_q2_scimrefdiffimg.fits \\
        --output    ztf_000443_zr_c16_q2_refsexcat_augmented.fits

Mode B — TNS (Transient Name Server) query
    Queries the TNS Bot API for all reported transients within the quadrant
    footprint and injects them.  Requires a TNS Bot API key (free registration
    at https://www.wis-tns.org/bots).

    Set env var TNS_API_KEY=<your key> or pass --tns-api-key.

    python transient_catalog.py --mode tns \\
        --ra 330.34158 --dec 0.72143 --radius 0.35 \\
        --refsexcat ../data/Reference/000443/zr/ccd16/q2/ztf_000443_zr_c16_q2_refsexcat.fits \\
        --diffimg   ../data/Science/000443/zr/ccd16/q2/ztf_20210101123456_000443_zr_c16_o_q2_scimrefdiffimg.fits \\
        --output    ztf_000443_zr_c16_q2_refsexcat_augmented.fits \\
        --tns-api-key $TNS_API_KEY

Programmatic API
----------------
    from transient_catalog import load_user_catalog, query_tns, augment_sexcat

    # Mode A
    transients = load_user_catalog("my_targets.csv")

    # Mode B
    transients = query_tns(ra_center=330.34158, dec_center=0.72143,
                           radius_deg=0.35, api_key="...")

    augmented_path = augment_sexcat(
        refsexcat_path = "ztf_000443_zr_c16_q2_refsexcat.fits",
        transients     = transients,
        diff_img_path  = "ztf_..._scimrefdiffimg.fits",  # WCS for footprint filter
        output_path    = "ztf_000443_zr_c16_q2_refsexcat_augmented.fits",
    )

Notes
-----
- The augmented catalog is a drop-in replacement for the original refsexcat.fits:
  all existing rows are preserved unchanged; injected rows are appended.
- Injected rows carry MAG_APER = 99 as their only marker (no extra column).
- Injection flux is derived from the sky noise of the *difference* image, not
  the reference image, because that is what the photometry measures.
- If a user CSV row already matches a reference catalog source within 1 arcsec,
  it is not injected (the source is already in the catalog).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TransientSource:
    """
    A single source to be injected into the reference catalog.

    ra, dec     : J2000 decimal degrees
    name        : human-readable identifier (e.g. "SN 2021xyz", "AT 2021abc")
    source      : provenance — "user" or "tns"
    mag_estimate: rough r/g magnitude for reference; used only to set injection
                  flux when no diff-image noise estimate is available.
                  Set None to use the default sigma-based flux.
    redshift    : optional; carried through for bookkeeping
    classification: e.g. "SN Ia", "TDE", "AGN"
    """
    ra: float
    dec: float
    name: str = "INJECTED"
    source: str = "user"
    mag_estimate: Optional[float] = None
    redshift: Optional[float] = None
    classification: Optional[str] = None


# ---------------------------------------------------------------------------
# Mode A — user CSV
# ---------------------------------------------------------------------------

def load_user_catalog(csv_path: str | Path) -> list[TransientSource]:
    """
    Load transient positions from a user-supplied CSV file.

    Required columns : ra, dec  (decimal degrees, J2000)
    Optional columns : name, mag_estimate, redshift, classification

    Returns a list of TransientSource objects.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"User catalog not found: {path}")

    df = pd.read_csv(path)

    required = {"ra", "dec"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"User catalog {path} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Normalise column names to lowercase for robust matching
    df.columns = df.columns.str.lower()

    sources = []
    for i, row in df.iterrows():
        src = TransientSource(
            ra=float(row["ra"]),
            dec=float(row["dec"]),
            name=str(row.get("name", f"USER_{i}")),
            source="user",
            mag_estimate=float(row["mag_estimate"]) if "mag_estimate" in row and not pd.isna(row.get("mag_estimate")) else None,
            redshift=float(row["redshift"]) if "redshift" in row and not pd.isna(row.get("redshift")) else None,
            classification=str(row["classification"]) if "classification" in row and not pd.isna(row.get("classification")) else None,
        )
        sources.append(src)

    logger.info(f"Loaded {len(sources)} sources from {path}")
    return sources


# ---------------------------------------------------------------------------
# Mode C — TNS public bulk catalog (whole-sky daily dump; filtered per quadrant)
# ---------------------------------------------------------------------------

TNS_PUBLIC_URL = ("https://www.wis-tns.org/system/files/tns_public_objects/"
                  "tns_public_objects.csv.zip")


def download_tns_public_catalog(cred_file: str | Path, key_file: str | Path,
                                cache_dir: str | Path, force: bool = False):
    """Download and extract the TNS public objects bulk catalog, returning a
    cleaned DataFrame with ra_deg / dec_deg columns. A cached copy in cache_dir is
    reused unless force=True.

    cred_file : JSON with keys 'tns_id', 'type', 'name' (the TNS marker identity)
    key_file  : text file containing the TNS API key
    """
    import json
    import zipfile
    import pandas as pd
    import requests

    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "tns_public_objects.csv"

    if force or not csv_path.exists():
        cred = json.loads(Path(cred_file).read_text())
        api_key = Path(key_file).read_text().strip()
        marker = ('tns_marker{"tns_id": "%s", "type": "%s", "name": "%s"}'
                  % (cred["tns_id"], cred["type"], cred["name"]))
        logger.info("Downloading TNS public objects bulk catalog ...")
        resp = requests.post(TNS_PUBLIC_URL, headers={"User-Agent": marker},
                             data={"api_key": api_key})
        resp.raise_for_status()
        zip_path = cache_dir / "tns_public_objects.csv.zip"
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        zip_path.unlink()
        logger.info(f"TNS catalog cached at {csv_path}")

    # The bulk CSV carries a one-line title above the header row.
    df = pd.read_csv(csv_path, skiprows=1, skip_blank_lines=True)
    df = df.rename(columns={"ra": "ra_deg", "declination": "dec_deg"})
    return df


def load_tns_catalog(cred_file: str | Path, key_file: str | Path,
                     cache_dir: str | Path, force_download: bool = False,
                     dec_min: float = -30.0) -> list[TransientSource]:
    """Fetch the TNS public bulk catalog and return it as a list of
    TransientSource. Per-quadrant footprint filtering is applied later at
    injection time, so this returns the whole (dec-limited) catalog."""
    import pandas as pd

    df = download_tns_public_catalog(cred_file, key_file, cache_dir, force=force_download)
    ra  = pd.to_numeric(df["ra_deg"],  errors="coerce")
    dec = pd.to_numeric(df["dec_deg"], errors="coerce")
    name_col = next((c for c in ("objname", "name") if c in df.columns), None)
    keep = ra.notna() & dec.notna() & (dec >= dec_min)
    sub = df[keep]
    sources = [
        TransientSource(ra=float(r_ra), dec=float(r_dec),
                        name=(str(r[name_col])[:64] if name_col else "TNS"),
                        source="tns")
        for (_, r), r_ra, r_dec in zip(sub.iterrows(), ra[keep], dec[keep])
    ]
    logger.info(f"Loaded {len(sources)} TNS sources (dec >= {dec_min}) for injection")
    return sources


# ---------------------------------------------------------------------------
# Mode B — TNS cone-search query (Bot API)
# ---------------------------------------------------------------------------

TNS_BASE_URL = "https://www.wis-tns.org/api/get"
TNS_SEARCH_URL = f"{TNS_BASE_URL}/search"
TNS_REQUEST_TIMEOUT = (10, 60)  # (connect, read) seconds


def query_tns(
    ra_center: float,
    dec_center: float,
    radius_deg: float = 0.35,
    api_key: Optional[str] = None,
    tns_bot_id: Optional[str] = None,
    tns_bot_name: Optional[str] = None,
    max_results: int = 500,
) -> list[TransientSource]:
    """
    Query the Transient Name Server (TNS) for objects within a circular region.

    Parameters
    ----------
    ra_center, dec_center : float
        Centre of the search cone (J2000 decimal degrees).
    radius_deg : float
        Search radius in degrees.  A ZTF quadrant is ~0.26°×0.26°; use 0.35
        to cover the full diagonal.
    api_key : str
        TNS Bot API key.  If not provided, falls back to the TNS_API_KEY
        environment variable.
    tns_bot_id, tns_bot_name : str
        Bot credentials from your TNS bot registration page.
        Falls back to env vars TNS_BOT_ID and TNS_BOT_NAME.
    max_results : int
        Maximum number of objects to retrieve (TNS default page size is 100;
        this function handles pagination automatically).

    Returns
    -------
    list of TransientSource

    Notes
    -----
    You need a free TNS account and a Bot registration to get an API key.
    See https://www.wis-tns.org/bots for instructions.

    TNS search radius is in arcseconds; we convert radius_deg internally.
    """
    # Resolve credentials
    key = api_key or os.environ.get("TNS_API_KEY")
    bot_id = tns_bot_id or os.environ.get("TNS_BOT_ID", "")
    bot_name = tns_bot_name or os.environ.get("TNS_BOT_NAME", "ZTF_pipeline")

    if not key:
        raise ValueError(
            "TNS API key is required for Mode B. "
            "Provide --tns-api-key, or set the TNS_API_KEY environment variable. "
            "Register a free bot at https://www.wis-tns.org/bots"
        )

    radius_arcsec = radius_deg * 3600.0

    headers = {
        "User-Agent": f'tns_marker{{"tns_id":{bot_id!r},"type":"bot","name":{bot_name!r}}}',
    }

    all_sources: list[TransientSource] = []
    page = 1

    while True:
        search_params = {
            "ra": ra_center,
            "dec": dec_center,
            "radius": radius_arcsec,
            "units": "arcsec",
            "num_page": min(100, max_results - len(all_sources)),
            "page": page,
        }

        payload = {
            "api_key": key,
            "data": json.dumps(search_params),
        }

        logger.debug(f"TNS query page {page}: cone({ra_center:.5f}, {dec_center:.5f}, "
                     f"{radius_arcsec:.0f}\")")

        try:
            resp = requests.post(
                TNS_SEARCH_URL,
                headers=headers,
                data=payload,
                timeout=TNS_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"TNS API request failed: {exc}") from exc

        try:
            result = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"TNS returned non-JSON response (HTTP {resp.status_code}): "
                f"{resp.text[:200]}"
            ) from exc

        # TNS wraps data in {"data": {"reply": [...]}} or similar
        data = result.get("data", {})
        if isinstance(data, dict):
            reply = data.get("reply", [])
        else:
            reply = data if isinstance(data, list) else []

        if not reply:
            break

        for obj in reply:
            try:
                src = TransientSource(
                    ra=float(obj.get("ra", 0)),
                    dec=float(obj.get("dec", 0)),
                    name=obj.get("name_prefix", "") + obj.get("name", "UNKNOWN"),
                    source="tns",
                    mag_estimate=_tns_mag(obj),
                    redshift=float(obj["redshift"]) if obj.get("redshift") else None,
                    classification=obj.get("type", {}).get("name") if isinstance(obj.get("type"), dict) else obj.get("type"),
                )
                all_sources.append(src)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(f"Skipping malformed TNS object {obj.get('name', '?')}: {exc}")

        logger.info(f"TNS: retrieved {len(reply)} objects on page {page} "
                    f"(total so far: {len(all_sources)})")

        # Stop if we got fewer than a full page or reached the limit
        if len(reply) < 100 or len(all_sources) >= max_results:
            break
        page += 1

    logger.info(f"TNS query complete: {len(all_sources)} transients within "
                f"{radius_arcsec:.0f}\" of ({ra_center:.5f}, {dec_center:.5f})")
    return all_sources


def _tns_mag(obj: dict) -> Optional[float]:
    """Extract a representative magnitude from a TNS object dict, if available."""
    # TNS may provide discovery magnitude or last-non-detection magnitude
    for key in ("discoveryabsmag", "discoverymag", "lastnondetlimit"):
        val = obj.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# Reference catalog augmentation
# ---------------------------------------------------------------------------

def _filter_already_in_catalog(
    transients: list[TransientSource],
    ref_table: Table,
    match_radius_arcsec: float = 1.0,
) -> list[TransientSource]:
    """
    Remove transients that already appear in the reference catalog within
    match_radius_arcsec.  These sources don't need injection — SExtractor
    will already recover them.
    """
    if not transients:
        return transients

    ref_coords = SkyCoord(
        ra=ref_table["ALPHAWIN_J2000"] * u.deg,
        dec=ref_table["DELTAWIN_J2000"] * u.deg,
    )

    to_inject = []
    n_already_present = 0

    for src in transients:
        target = SkyCoord(ra=src.ra * u.deg, dec=src.dec * u.deg)
        sep = target.separation(ref_coords)
        if sep.min().arcsec <= match_radius_arcsec:
            logger.debug(f"{src.name} is already in the reference catalog "
                         f"(nearest match: {sep.min().arcsec:.2f}\" away) — skipping injection")
            n_already_present += 1
        else:
            to_inject.append(src)

    if n_already_present:
        logger.info(f"{n_already_present} sources already in reference catalog — "
                    f"not re-injecting")
    return to_inject


def _filter_in_footprint(sources, image_path):
    """Keep only sources whose sky position falls inside the image frame.
    Cheap even for a full TNS dump. Avoids injecting a target into a quadrant whose
    footprint does not contain it (which would otherwise bloat every quadrant's
    catalog with sources that are never painted)."""
    from astropy.wcs import WCS
    if not sources:
        return sources
    with fits.open(image_path) as h:
        wcs = WCS(h[0].header)
        ny, nx = h[0].data.shape
    ra  = np.array([s.ra  for s in sources], dtype=float)
    dec = np.array([s.dec for s in sources], dtype=float)

    # Coarse angular pre-filter around the image centre. The gnomonic WCS inverse
    # does not converge far from the tangent point, so we must not feed it whole-sky
    # coordinates; the forward pixel->world transform is always well-defined.
    c_ra, c_dec = wcs.pixel_to_world_values(nx / 2.0, ny / 2.0)
    dra  = ((ra - float(c_ra) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(float(c_dec)))
    ddec = dec - float(c_dec)
    idx = np.where(dra**2 + ddec**2 < 1.0**2)[0]   # within ~1 deg of centre
    if idx.size == 0:
        return []

    def _in_frame(x, y):
        return np.isfinite(x) and np.isfinite(y) and 0 <= x < nx and 0 <= y < ny

    kept = []
    try:
        x, y = wcs.world_to_pixel_values(ra[idx], dec[idx])
        for j, i in enumerate(idx):
            if _in_frame(x[j], y[j]):
                kept.append(sources[i])
    except Exception:
        # fall back to per-source so one bad coordinate cannot abort the batch
        for i in idx:
            try:
                x, y = wcs.world_to_pixel_values(float(ra[i]), float(dec[i]))
            except Exception:
                continue
            if _in_frame(x, y):
                kept.append(sources[i])
    return kept


def augment_sexcat(
    refsexcat_path: str | Path,
    transients: list[TransientSource],
    diff_img_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    match_radius_arcsec: float = 1.0,
    footprint_filter: bool = True,
) -> Path:
    """
    Append injected transient sources to a ZTF reference SExtractor catalog.

    The output is a FITS binary table that is a drop-in replacement for the
    original refsexcat.fits.  It contains:
      - All original rows, unchanged
      - One new row per injected transient (marked only by MAG_APER = 99)

    Parameters
    ----------
    refsexcat_path : str or Path
        Path to the original ZTF refsexcat.fits (or already-augmented version).
    transients : list of TransientSource
        Sources to inject.  Sources already present in the catalog (within
        match_radius_arcsec) are silently skipped.
    diff_img_path : str or Path, optional
        Difference image used only for the per-quadrant footprint filter (its WCS).
    output_path : str or Path, optional
        Where to write the augmented catalog.  Defaults to
        {refsexcat_stem}_augmented.fits in the same directory.
    match_radius_arcsec : float
        Sources within this radius of an existing reference source are not
        re-injected (default 1 arcsec).

    Returns
    -------
    Path to the written augmented catalog.
    """
    refsexcat_path = Path(refsexcat_path)
    if not refsexcat_path.exists():
        raise FileNotFoundError(f"Reference catalog not found: {refsexcat_path}")

    if output_path is None:
        output_path = refsexcat_path.with_name(
            refsexcat_path.stem + "_augmented.fits"
        )
    output_path = Path(output_path)

    # ---- Load original catalog ----
    with fits.open(refsexcat_path) as hdul:
        orig_table = Table(hdul[1].data)

    logger.info(f"Original catalog: {len(orig_table)} sources from {refsexcat_path.name}")

    # Injected sources are marked solely by their reference-magnitude sentinel
    # (MAG_APER = 99, surfaced as MAG_4_REF > 90 in the light curves); no separate
    # INJECTED flag column is added.

    # ---- Keep only those inside this quadrant's footprint (fast; do this first so
    #      the per-source catalog cross-match below only sees the few survivors) ----
    to_inject = transients
    if footprint_filter and diff_img_path is not None and to_inject:
        n_before = len(to_inject)
        to_inject = _filter_in_footprint(to_inject, diff_img_path)
        if len(to_inject) != n_before:
            logger.info(f"footprint: {len(to_inject)}/{n_before} transients fall in "
                        f"{Path(refsexcat_path).name}")

    # ---- Drop those already present in the reference catalog ----
    to_inject = _filter_already_in_catalog(to_inject, orig_table, match_radius_arcsec)

    if not to_inject:
        logger.info("No sources to inject after footprint + cross-match — "
                    "no augmented catalog written for this quadrant.")
        return refsexcat_path

    # ---- Reference positions, for the neighbour-blend flag ----
    # The simulated detection image paints every source at a fixed amplitude (only
    # positions matter), so no per-source injection flux is computed. If an injected
    # source has a reference neighbour within _BLEND_ARCSEC, we set SExtractor FLAGS
    # bit 0 (value 1, "close neighbour biasing photometry") so it is flagged exactly
    # like a real crowded source (carried to the light curves as FLAG_SE_REF).
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _ref_coords = SkyCoord(ra=np.asarray(orig_table["ALPHAWIN_J2000"], float) * u.deg,
                           dec=np.asarray(orig_table["DELTAWIN_J2000"], float) * u.deg)
    _BLEND_ARCSEC = 3.0

    # ---- Build synthetic rows for injected sources ----
    # Create a template row from the original table with all columns set to
    # sensible defaults, then override the position, magnitude and flag columns.
    injected_rows = []

    for src in to_inject:
        row = {}
        for col in orig_table.colnames:
            dtype = orig_table[col].dtype
            shape = orig_table[col].shape[1:]   # per-row shape for vector cols

            if np.issubdtype(dtype, np.floating):
                row[col] = np.full(shape, np.nan) if shape else np.nan
            elif np.issubdtype(dtype, np.integer):
                row[col] = np.zeros(shape, dtype=dtype) if shape else dtype.type(0)
            elif np.issubdtype(dtype, np.bool_):
                row[col] = False
            else:
                row[col] = ""

        # Mandatory position columns
        row["ALPHAWIN_J2000"] = src.ra
        row["DELTAWIN_J2000"] = src.dec

        # Magnitude sentinel (99) is the sole injected-source marker: it drives the
        # brighter detection-image amplitude in simulate_science and surfaces as
        # MAG_4_REF > 90 in the light curves. No injection flux is stored.
        for mc in ["MAG_BEST", "MAG_AUTO", "MAG_APER"]:
            if mc in orig_table.colnames:
                if orig_table[mc].shape[1:]:
                    row[mc] = np.full(orig_table[mc].shape[1:], 99.0)
                else:
                    row[mc] = 99.0
        for ec in ["MAGERR_BEST", "MAGERR_AUTO", "MAGERR_APER"]:
            if ec in orig_table.colnames:
                if orig_table[ec].shape[1:]:
                    row[ec] = np.full(orig_table[ec].shape[1:], 99.0)
                else:
                    row[ec] = 99.0

        # SExtractor quality flags: co-opt bit 0 (close neighbour) if a reference
        # source lies within _BLEND_ARCSEC, so injected sources are blend-flagged
        # like real crowded sources; 0 (clean) otherwise.
        _nn = SkyCoord(ra=src.ra * u.deg, dec=src.dec * u.deg).separation(_ref_coords).arcsec.min() \
            if len(_ref_coords) else np.inf
        row["FLAGS"] = 1 if _nn < _BLEND_ARCSEC else 0

        injected_rows.append(row)

    # ---- Assemble into an astropy Table ----
    injected_table = Table(rows=injected_rows, names=orig_table.colnames)

    # Copy column metadata (units, descriptions) from original
    for col in orig_table.colnames:
        if col in injected_table.colnames:
            injected_table[col].unit = orig_table[col].unit
            injected_table[col].description = orig_table[col].description

    # ---- Merge and write ----
    augmented = vstack([orig_table, injected_table])
    augmented.write(str(output_path), format="fits", overwrite=True)

    logger.info(f"Augmented catalog: {len(augmented)} sources "
                f"({len(orig_table)} original + {len(injected_rows)} injected) "
                f"→ {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Convenience: augment all refsexcats for a set of quadrants
# ---------------------------------------------------------------------------

def augment_all_refsexcats(
    base_dir: str | Path,
    transients: list[TransientSource],
    bands: list[str] = None,
    match_radius_arcsec: float = 1.0,
    footprint_filter: bool = True,
) -> dict[str, Path]:
    """
    Find all refsexcat.fits files under base_dir/Reference and augment each one.
    A representative difference image per quadrant supplies the WCS for the
    footprint filter. Returns a dict mapping original refsexcat → augmented path.
    """
    from ztf_field_lookup import BAND_TO_FILTERCODE

    base_dir = Path(base_dir)
    ref_root = base_dir / "Reference"
    sci_root = base_dir / "Science"

    if bands:
        filtercodes = {BAND_TO_FILTERCODE.get(b, b) for b in bands}
    else:
        filtercodes = None   # all bands

    results = {}
    for refcat in sorted(ref_root.rglob("*_refsexcat.fits")):
        # Skip already-augmented catalogs
        if "_augmented" in refcat.name:
            continue

        # Extract (filtercode, field, ccdid, qid) from path
        # Path: .../Reference/{field}/{filtercode}/ccd{ccdid}/q{qid}/ztf_*_refsexcat.fits
        parts = refcat.parts
        try:
            q_idx   = next(i for i, p in enumerate(parts) if p.startswith("q"))
            ccd_idx = q_idx - 1
            fc_idx  = ccd_idx - 1
            fld_idx = fc_idx - 1
            filtercode = parts[fc_idx]
            field_str  = parts[fld_idx]
        except (StopIteration, IndexError):
            logger.warning(f"Cannot parse path structure: {refcat} — skipping")
            continue

        if filtercodes and filtercode not in filtercodes:
            continue

        # Find a representative diff image (WCS for the footprint filter)
        sci_quadrant_dir = (sci_root / field_str / filtercode
                            / parts[ccd_idx] / parts[q_idx])
        diff_imgs = sorted(sci_quadrant_dir.glob("*_scimrefdiffimg.fits")) if sci_quadrant_dir.exists() else []
        diff_img = diff_imgs[0] if diff_imgs else None
        if diff_img is None and footprint_filter:
            logger.warning(f"No diff image for {refcat.parent.relative_to(base_dir)} — "
                           f"footprint filter skipped for this quadrant")

        out_path = refcat.with_name(refcat.stem.replace("_refsexcat", "_refsexcat_augmented") + ".fits")

        try:
            augmented = augment_sexcat(
                refsexcat_path=refcat,
                transients=transients,
                diff_img_path=diff_img,
                output_path=out_path,
                match_radius_arcsec=match_radius_arcsec,
                footprint_filter=footprint_filter,
            )
            results[str(refcat)] = augmented
        except Exception as exc:
            logger.error(f"Failed to augment {refcat.name}: {exc}")

    logger.info(f"Augmented {len(results)} reference catalogs under {base_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Inject additional sources into a ZTF reference SExtractor catalog "
            "so that simulate_science.py can detect them in difference images."
        )
    )
    parser.add_argument(
        "--mode", choices=["user", "tns"], required=True,
        help="Source of transient positions: 'user' (CSV file) or 'tns' (TNS API query)"
    )

    # Mode A arguments
    parser.add_argument("--input", type=Path,
                        help="[Mode A] CSV file with ra, dec columns")

    # Mode B arguments
    parser.add_argument("--ra",     type=float, help="[Mode B] Field centre RA (deg)")
    parser.add_argument("--dec",    type=float, help="[Mode B] Field centre Dec (deg)")
    parser.add_argument("--radius", type=float, default=0.35,
                        help="[Mode B] Search radius in degrees (default: 0.35)")
    parser.add_argument("--tns-api-key", type=str, default=None,
                        help="[Mode B] TNS Bot API key (or set TNS_API_KEY env var)")
    parser.add_argument("--tns-bot-id",  type=str, default=None,
                        help="[Mode B] TNS Bot ID (or set TNS_BOT_ID env var)")
    parser.add_argument("--tns-bot-name", type=str, default=None,
                        help="[Mode B] TNS Bot name (or set TNS_BOT_NAME env var)")

    # Catalog / image arguments
    parser.add_argument("--refsexcat", type=Path, required=False,
                        help="Path to a single refsexcat.fits to augment")
    parser.add_argument("--diffimg",   type=Path, required=False,
                        help="Path to diff image (WCS for the footprint filter)")
    parser.add_argument("--output",    type=Path, required=False,
                        help="Output path for augmented catalog")
    parser.add_argument("--base-dir",  type=Path, required=False,
                        help="Base data directory: augment ALL refsexcats found there")
    parser.add_argument("--bands", nargs="+", default=None,
                        help="Bands to process when using --base-dir (default: all)")
    parser.add_argument("--match-radius",    type=float, default=1.0,
                        help="Skip injection if within N arcsec of existing source (default: 1)")

    args = parser.parse_args()

    # ---- Load transient list ----
    if args.mode == "user":
        if not args.input:
            parser.error("--input is required for --mode user")
        transients = load_user_catalog(args.input)

    elif args.mode == "tns":
        if args.ra is None or args.dec is None:
            parser.error("--ra and --dec are required for --mode tns")
        transients = query_tns(
            ra_center=args.ra,
            dec_center=args.dec,
            radius_deg=args.radius,
            api_key=args.tns_api_key,
            tns_bot_id=args.tns_bot_id,
            tns_bot_name=args.tns_bot_name,
        )

    if not transients:
        print("No transients found. Nothing to inject.")
        sys.exit(0)

    print(f"Found {len(transients)} transient(s) to inject.")

    # ---- Augment catalog(s) ----
    if args.base_dir:
        results = augment_all_refsexcats(
            base_dir=args.base_dir,
            transients=transients,
            bands=args.bands,
            match_radius_arcsec=args.match_radius,
        )
        print(f"\nAugmented {len(results)} reference catalog(s).")
        for orig, out in results.items():
            print(f"  {Path(orig).name} → {out.name}")

    elif args.refsexcat:
        out = augment_sexcat(
            refsexcat_path=args.refsexcat,
            transients=transients,
            diff_img_path=args.diffimg,
            output_path=args.output,
            match_radius_arcsec=args.match_radius,
        )
        print(f"\nAugmented catalog written to: {out}")

    else:
        parser.error("Provide either --refsexcat or --base-dir")
