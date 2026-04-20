"""
download_coordinator.py
-----------------------
Takes the output of ztf_field_lookup.py and downloads the ZTF image products
needed to run the photometry pipeline.

Downloads per epoch (science products):
    scimrefdiffimg.fits.fz  — difference image (F-packed, needs funpack after)
    sexcat.fits             — ZTF's own SExtractor catalog for the science image

Downloads per quadrant (reference products, static — only once per field/ccd/quad/band):
    refimg.fits             — reference image stack
    refsexcat.fits          — SExtractor catalog for the reference image

URL formats (from previous definition):
    Science:
        https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/
          {YYYY}/{MM}{DD}/{fracday}/
          ztf_{filefracday}_{field:06d}_{filtercode}_c{ccdid:02d}_o_q{qid}_{suffix}

    Reference:
        https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/
          {fieldprefix:03d}/field{field:06d}/{filtercode}/ccd{ccdid:02d}/q{qid}/
          ztf_{field:06d}_{filtercode}_c{ccdid:02d}_q{qid}_{suffix}

Authentication:
    IRSA requires credentials to download image products (metadata queries are public).
    Pass via --username / --password flags, or set environment variables:
        IRSA_USERNAME, IRSA_PASSWORD
    Or store in ~/.netrc (recommended for scripts):
        machine irsa.ipac.caltech.edu login <user> password <pass>

Usage:
    from ztf_field_lookup import lookup_target
    from download_coordinator import download_all

    epochs = lookup_target(ra=330.34158, dec=0.72143, bands=['g', 'r'])
    download_all(epochs, base_dir='../data', username='...', password='...')

    # Or from CLI:
    python scripts/download_coordinator.py \
        --epochs ../data/Epochs/lookup_330.34158_0.72143_g-r.epochs.parquet \
        --base-dir ../data \
        --bands g r \
        --workers 20 \
        --max-seeing 3.0 \
        --min-maglim 19.5 \
        --min-epochs-per-quad 30

Notes:
    - Files already present on disk are skipped (idempotent).
    - Permanent 404s (files that will never exist on IRSA) are recorded in
      {base_dir}/Epochs/permanent_404s.log and silently skipped on future runs.
      Common case: i-band reference files for quadrants with no i-band coverage.
    - Retriable failures (timeouts, incomplete reads) are logged to
      {base_dir}/Epochs/failed_downloads.log. Re-running the script will retry these.
    - Missing science epochs (404 on diff image) are logged to
      {base_dir}/Epochs/missing_epochs.log — these epochs simply have no processed
      difference image on IRSA and will not appear in light curves.
    - .fits.fz files are automatically funpacked after download if funpack is available.
    - The infobits quality flag filter from ztf_field_lookup is respected:
      flagged epochs are downloaded by default but tagged; use --skip-flagged to omit.
"""

import os
import logging
import subprocess
import netrc
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IRSA_BASE = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"

# Files to fetch per epoch (science / difference image products)
SCI_SUFFIXES = [
    "scimrefdiffimg.fits.fz",   # difference image — primary photometry input
    "sexcat.fits",              # ZTF SExtractor catalog (used for position reference)
]

# Files to fetch per quadrant (reference products — static, download once)
REF_SUFFIXES = [
    "refimg.fits",              # reference image stack
    "refsexcat.fits",           # SExtractor catalog for reference
]

# Max parallel download threads. IRSA handles ~40 concurrently without rate-limiting.
# Each thread gets its own Session+ConnectionPool so they don't contend.
DEFAULT_WORKERS = 50

# ---------------------------------------------------------------------------
# Infobits quality masks
# ---------------------------------------------------------------------------
# These epochs have no valid astrometric or photometric calibration and cannot
# be recovered regardless of science case — never worth downloading.
#   Bit  0 (       1): no astrometric solution
#   Bit  1 (       2): no photometric solution
#   Bit 25 (33554432): composite "bad quality" flag
HARD_REJECT_MASK = (1 << 0) | (1 << 1) | (1 << 25)   # = 33554435

# Cautionary bits — data may still be usable depending on science requirements.
# Skipped only when --skip-flagged is passed.
# Well-documented bits (ZSDS Sec 10.4):
#   Bit  2 (       4): astrometric fit rms above threshold
#   Bit  3 (       8): photometric fit rms above threshold
#   Bit  4 (      16): moon in/near FOV, elevated background
#   Bit  5 (      32): satellite/aircraft trail present
#   Bit  6 (      64): high sky background (non-moon)
# Observed in data but not fully documented (treated as cautionary until confirmed):
#   Bit 11 (    2048): unknown — rare (few epochs), treat cautionary
#   Bit 21 ( 2097152): unknown — very rare, treat cautionary
#   Bit 22 ( 4194304): PSF matching / subtraction quality below threshold
#                      (most common mystery bit — 158 epochs in r-band field 443;
#                       appears WITHOUT bit 25, so ZTF did not consider globally bad)
#   Bit 26 (67108864): reference image quality / normalization issue
#   Bit 27 (134217728): possibly non-standard reference used
# TODO: confirm bits 11, 21, 22, 26, 27 against ZSDS Explanatory Supplement Sec 10.4
CAUTIONARY_MASK = (
    (1 << 2)  | (1 << 3)  | (1 << 4)  | (1 << 5)  | (1 << 6)   # documented
  | (1 << 11) | (1 << 21) | (1 << 22) | (1 << 26) | (1 << 27)   # observed, unconfirmed
)  # = 234881148

# Request timeout (seconds): connect, read
# Connect timeout is generous because IRSA can be slow to accept connections
# under load. Read timeout is long because diff images can be ~100 MB.
REQUEST_TIMEOUT = (30, 300)

# Retry config for transient failures
RETRY_CONFIG = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def sci_url(filefracday: str, field: int, filtercode: str,
            ccdid: int, qid: int, suffix: str) -> str:
    """
    Build a science product URL from metadata fields.

    filefracday is a 14-character string: YYYYMMDDffffff
        [0:4]  = year
        [4:6]  = month
        [6:8]  = day
        [8:]   = fractional day (6 chars)
    """
    ffd = str(filefracday)
    year    = ffd[0:4]
    mmdd    = ffd[4:8]        # e.g. "1030"
    fracday = ffd[8:]         # e.g. "211019"
    fname   = f"ztf_{ffd}_{field:06d}_{filtercode}_c{ccdid:02d}_o_q{qid}_{suffix}"
    return f"{IRSA_BASE}/sci/{year}/{mmdd}/{fracday}/{fname}"


def ref_url(field: int, filtercode: str, ccdid: int, qid: int, suffix: str) -> str:
    """
    Build a reference product URL.

    Reference products are static per (field, ccd, quad, band) —
    there is only one reference image stack per combination.
    """
    padded   = f"{field:06d}"
    prefix   = padded[:3]           # first 3 digits of zero-padded field
    fname    = f"ztf_{padded}_{filtercode}_c{ccdid:02d}_q{qid}_{suffix}"
    return f"{IRSA_BASE}/ref/{prefix}/field{padded}/{filtercode}/ccd{ccdid:02d}/q{qid}/{fname}"


# ---------------------------------------------------------------------------
# Local path layout
# ---------------------------------------------------------------------------

def sci_local_path(base_dir: Path, field: int, filtercode: str,
                   ccdid: int, qid: int, filefracday: str, suffix: str) -> Path:
    """
    Local path mirrors the quadrant hierarchy, with per-epoch files nested inside.
    Structure: base_dir/Science/{field:06d}/{filtercode}/ccd{ccdid:02d}/q{qid}/{filename}
    """
    ffd   = str(filefracday)
    fname = f"ztf_{ffd}_{field:06d}_{filtercode}_c{ccdid:02d}_o_q{qid}_{suffix}"
    return (base_dir / "Science"
            / f"{field:06d}" / filtercode
            / f"ccd{ccdid:02d}" / f"q{qid}" / fname)


def ref_local_path(base_dir: Path, field: int, filtercode: str,
                   ccdid: int, qid: int, suffix: str) -> Path:
    """
    Local path for reference products (matches IRSA directory layout).
    Structure: base_dir/Reference/{field:06d}/{filtercode}/ccd{ccdid:02d}/q{qid}/{filename}
    """
    padded = f"{field:06d}"
    fname  = f"ztf_{padded}_{filtercode}_c{ccdid:02d}_q{qid}_{suffix}"
    return (base_dir / "Reference"
            / padded / filtercode
            / f"ccd{ccdid:02d}" / f"q{qid}" / fname)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_auth(username: Optional[str] = None,
             password: Optional[str] = None) -> tuple[str, str]:
    """
    Resolve IRSA credentials in priority order:
        1. Explicit username / password arguments
        2. Environment variables IRSA_USERNAME / IRSA_PASSWORD
        3. ~/.netrc entry for irsa.ipac.caltech.edu
    Raises RuntimeError if no credentials are found.
    """
    if username and password:
        return (username, password)

    u = os.environ.get("IRSA_USERNAME")
    p = os.environ.get("IRSA_PASSWORD")
    if u and p:
        return (u, p)

    try:
        n = netrc.netrc()
        entry = n.authenticators("irsa.ipac.caltech.edu")
        if entry:
            return (entry[0], entry[2])   # (login, password)
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    raise RuntimeError(
        "No IRSA credentials found. Provide --username / --password, "
        "set IRSA_USERNAME / IRSA_PASSWORD env vars, "
        "or add irsa.ipac.caltech.edu to ~/.netrc"
    )


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

def _make_session(auth: tuple[str, str]) -> requests.Session:
    """Create a requests Session with auth, retries, and timeouts configured."""
    s = requests.Session()
    s.auth = auth
    # pool_connections / pool_maxsize = one connection pool per host per session.
    # Since each worker thread owns its own session, this gives each thread a
    # dedicated connection pool and eliminates lock contention on the shared pool.
    adapter = HTTPAdapter(max_retries=RETRY_CONFIG, pool_connections=1, pool_maxsize=1)
    s.mount("https://", adapter)
    return s


# Thread-local storage: one Session per worker thread, created on first use.
_thread_local = __import__("threading").local()


def _get_thread_session(auth: tuple[str, str]) -> requests.Session:
    """Return the calling thread's own Session, creating it on first call."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session(auth)
    return _thread_local.session


def download_file(url: str, dest: Path, auth: tuple[str, str],
                  skip_if_exists: bool = True) -> tuple[str, str, str]:
    """
    Download a single file.

    Returns (url, status, message) where status is one of:
        "ok"          — downloaded successfully
        "skipped"     — already exists on disk
        "404"         — permanent not-found (file will never exist on IRSA)
        "failed"      — retriable error (timeout, incomplete read, server error)
    """
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        return (url, "skipped", "exists")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    session = _get_thread_session(auth)

    try:
        with session.get(url, timeout=REQUEST_TIMEOUT, stream=True) as resp:
            # 404 → permanent absence; distinguish from retriable server errors
            if resp.status_code == 404:
                return (url, "404", f"404 Not Found")

            resp.raise_for_status()

            # IRSA occasionally returns HTML for missing files instead of 404 status
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                return (url, "404", "404 (HTML response — file not found on IRSA)")

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)

        tmp.rename(dest)
        return (url, "ok", f"downloaded ({dest.stat().st_size / 1e6:.1f} MB)")

    except requests.RequestException as exc:
        if tmp.exists():
            tmp.unlink()
        return (url, "failed", str(exc))


def funpack_file(path: Path) -> bool:
    """
    Decompress a .fits.fz file in-place using funpack.
    Returns True on success, False if funpack is not available or fails.
    """
    if not str(path).endswith(".fits.fz"):
        return True

    unpacked = path.with_name(path.name.replace(".fits.fz", ".fits"))
    if unpacked.exists():
        return True

    try:
        result = subprocess.run(
            ["funpack", "-D", str(path)],   # -D deletes the .fz after unpacking
            capture_output=True, timeout=120
        )
        if result.returncode == 0:
            logger.debug(f"Funpacked: {path.name}")
            return True
        else:
            logger.warning(f"funpack failed on {path.name}: {result.stderr.decode()}")
            return False
    except FileNotFoundError:
        logger.warning("funpack not found in PATH — skipping decompression. "
                       "Install with: conda install -c conda-forge cfitsio")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"funpack timed out on {path.name}")
        return False


# ---------------------------------------------------------------------------
# Epoch filtering
# ---------------------------------------------------------------------------

def filter_epochs(
    epochs: pd.DataFrame,
    *,
    max_seeing: float | None = None,
    min_maglim: float | None = None,
    skip_cautionary: bool = False,
    mjd_min: float | None = None,
    mjd_max: float | None = None,
    min_epochs_per_quad: int | None = None,
) -> pd.DataFrame:
    """
    Apply quality and selection cuts to an epochs DataFrame before downloading.

    Hard-rejected epochs (bits 0, 1, 25) are always removed — they have no
    valid calibration. All other cuts are optional and default to disabled.

    Parameters
    ----------
    epochs : pd.DataFrame
        Epoch list from ztf_field_lookup.lookup_target().
    max_seeing : float, optional
        Drop epochs with seeing (arcsec FWHM) above this value.
    min_maglim : float, optional
        Drop epochs with 5σ limiting magnitude below this value.
    skip_cautionary : bool
        If True, also drop epochs with any cautionary infobit set.
    mjd_min, mjd_max : float, optional
        Restrict to epochs within this MJD range (inclusive).
    min_epochs_per_quad : int, optional
        After all other cuts, drop entire (field, filtercode, ccdid, qid)
        groups that have fewer than this many remaining epochs.

    Returns
    -------
    pd.DataFrame
        Filtered epochs, index reset.
    """
    df = epochs.copy()
    n_start = len(df)

    # --- Hard reject (always applied) ---
    mask_hard = (df["infobits"].fillna(0).astype(int) & HARD_REJECT_MASK) != 0
    n_hard = mask_hard.sum()
    df = df[~mask_hard]

    # --- Cautionary bits ---
    n_cautionary = 0
    if skip_cautionary:
        mask_caut = (df["infobits"].fillna(0).astype(int) & CAUTIONARY_MASK) != 0
        n_cautionary = mask_caut.sum()
        df = df[~mask_caut]

    # --- Seeing ---
    n_seeing = 0
    if max_seeing is not None:
        mask_see = df["seeing"].notna() & (df["seeing"] > max_seeing)
        n_seeing = mask_see.sum()
        df = df[~mask_see]

    # --- Limiting magnitude ---
    n_maglim = 0
    if min_maglim is not None:
        mask_mag = df["maglimit"].notna() & (df["maglimit"] < min_maglim)
        n_maglim = mask_mag.sum()
        df = df[~mask_mag]

    # --- MJD range ---
    n_mjd = 0
    if mjd_min is not None or mjd_max is not None:
        mjd_col = "obsmjd" if "obsmjd" in df.columns else "obsjd"
        mask_mjd = pd.Series(False, index=df.index)
        if mjd_min is not None:
            mask_mjd |= df[mjd_col] < mjd_min
        if mjd_max is not None:
            mask_mjd |= df[mjd_col] > mjd_max
        n_mjd = mask_mjd.sum()
        df = df[~mask_mjd]

    # --- Minimum epochs per quadrant ---
    n_sparse = 0
    if min_epochs_per_quad is not None:
        group_keys = ["field", "filtercode", "ccdid", "qid"]
        counts = df.groupby(group_keys)["obsmjd" if "obsmjd" in df.columns else "obsjd"].transform("count")
        mask_sparse = counts < min_epochs_per_quad
        n_sparse = mask_sparse.sum()
        df = df[~mask_sparse]

    df = df.reset_index(drop=True)
    n_kept = len(df)

    # Summary log
    parts = [f"hard-rejected={n_hard}"]
    if n_cautionary:
        parts.append(f"cautionary={n_cautionary}")
    if n_seeing:
        parts.append(f"seeing>{max_seeing}\"={n_seeing}")
    if n_maglim:
        parts.append(f"maglim<{min_maglim}={n_maglim}")
    if n_mjd:
        parts.append(f"mjd-range={n_mjd}")
    if n_sparse:
        parts.append(f"sparse-quad={n_sparse}")
    removed = n_start - n_kept
    logger.info(f"filter_epochs: {n_start} → {n_kept} epochs "
                f"(removed {removed}: {', '.join(parts)})")

    return df


# ---------------------------------------------------------------------------
# Batch download builders
# ---------------------------------------------------------------------------

def _build_sci_tasks(epochs: pd.DataFrame, base_dir: Path,
                     bands: list[str], skip_flagged: bool) -> list[tuple]:
    """
    Build list of (url, local_path) tuples for all science epochs.
    Each epoch contributes len(SCI_SUFFIXES) files.

    Hard-rejected epochs (bits 0, 1, or 25 set) are always skipped —
    they have no valid calibration and cannot be recovered.
    Cautionary epochs (bits 2-6) are skipped only if skip_flagged=True.
    """
    from ztf_field_lookup import BAND_TO_FILTERCODE
    filtercodes = {BAND_TO_FILTERCODE[b] for b in bands if b in BAND_TO_FILTERCODE}

    n_hard_rejected = 0
    n_cautionary_skipped = 0
    tasks = []

    for _, row in epochs.iterrows():
        if str(row.get("filtercode", "")) not in filtercodes:
            continue

        infobits = int(row.get("infobits", 0) or 0)

        # Always skip unrecoverable epochs
        if infobits & HARD_REJECT_MASK:
            n_hard_rejected += 1
            continue

        # Optionally skip cautionary epochs
        if skip_flagged and (infobits & CAUTIONARY_MASK):
            n_cautionary_skipped += 1
            continue

        for suffix in SCI_SUFFIXES:
            url  = sci_url(row.filefracday, int(row.field), row.filtercode,
                           int(row.ccdid), int(row.qid), suffix)
            dest = sci_local_path(base_dir, int(row.field), row.filtercode,
                                  int(row.ccdid), int(row.qid),
                                  row.filefracday, suffix)
            tasks.append((url, dest))

    if n_hard_rejected > 0:
        logger.info(f"Skipped {n_hard_rejected} hard-rejected epochs "
                    f"(bits 0/1/25 set — no valid calibration)")
    if n_cautionary_skipped > 0:
        logger.info(f"Skipped {n_cautionary_skipped} cautionary epochs "
                    f"(--skip-flagged is set)")

    return tasks


def _build_ref_tasks(epochs: pd.DataFrame, base_dir: Path,
                     bands: list[str]) -> list[tuple]:
    """
    Build list of (url, local_path) for reference products.
    One set per unique (field, filtercode, ccdid, qid) — not per epoch.
    """
    from ztf_field_lookup import BAND_TO_FILTERCODE
    filtercodes = {BAND_TO_FILTERCODE[b] for b in bands if b in BAND_TO_FILTERCODE}

    seen  = set()
    tasks = []
    for _, row in epochs.iterrows():
        if str(row.get("filtercode", "")) not in filtercodes:
            continue
        key = (int(row.field), row.filtercode, int(row.ccdid), int(row.qid))
        if key in seen:
            continue
        seen.add(key)

        for suffix in REF_SUFFIXES:
            url  = ref_url(*key, suffix)
            dest = ref_local_path(base_dir, *key, suffix)
            tasks.append((url, dest))

    return tasks


# ---------------------------------------------------------------------------
# Main download entry points
# ---------------------------------------------------------------------------

def download_all(
    epochs: pd.DataFrame,
    base_dir: str | Path,
    bands: list[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    max_workers: int = DEFAULT_WORKERS,
    skip_flagged: bool = False,
    funpack: bool = True,
    max_seeing: float | None = None,
    min_maglim: float | None = None,
    mjd_min: float | None = None,
    mjd_max: float | None = None,
    min_epochs_per_quad: int | None = None,
) -> dict:
    """
    Download all science and reference products for the given epochs.

    Parameters
    ----------
    epochs : pd.DataFrame
        Full epoch list from ztf_field_lookup.lookup_target().
    base_dir : str or Path
        Root directory for downloads. Science and Reference subdirs are created.
    bands : list of str
        Which bands to download. Default: all bands present in epochs.
    username, password : str
        IRSA credentials. Falls back to env vars / ~/.netrc if not provided.
    max_workers : int
        Number of parallel download threads.
    skip_flagged : bool
        If True, also skip cautionary epochs (infobits with bits 2–6/11/21/22/26/27).
        Hard-rejected epochs (bits 0/1/25) are always skipped.
    funpack : bool
        If True, run funpack on .fits.fz files after download.
    max_seeing : float, optional
        Skip epochs with seeing above this value (arcsec FWHM).
    min_maglim : float, optional
        Skip epochs with 5σ limiting magnitude below this value.
    mjd_min, mjd_max : float, optional
        Restrict to epochs in this MJD range (inclusive).
    min_epochs_per_quad : int, optional
        Skip entire quadrants with fewer than this many passing epochs.

    Returns
    -------
    dict with keys: n_downloaded, n_skipped, n_failed, n_permanent_404, failed_urls
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if bands is None:
        from ztf_field_lookup import FILTERCODE_TO_BAND
        present_fcs  = epochs["filtercode"].unique()
        bands = [FILTERCODE_TO_BAND.get(fc, fc) for fc in present_fcs]

    # Apply epoch selection cuts before building the download queue
    epochs = filter_epochs(
        epochs,
        skip_cautionary=skip_flagged,
        max_seeing=max_seeing,
        min_maglim=min_maglim,
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        min_epochs_per_quad=min_epochs_per_quad,
    )

    auth = get_auth(username, password)

    # Build task lists (hard-reject already removed by filter_epochs; pass False here)
    sci_tasks = _build_sci_tasks(epochs, base_dir, bands, skip_flagged=False)
    ref_tasks = _build_ref_tasks(epochs, base_dir, bands)
    all_tasks = ref_tasks + sci_tasks   # refs first — smaller, quick to confirm auth works

    # Load previously recorded permanent 404s — skip them silently this run
    perm404_log     = base_dir / "Epochs" / "permanent_404s.log"
    missing_log     = base_dir / "Epochs" / "missing_epochs.log"
    failed_log      = base_dir / "Epochs" / "failed_downloads.log"

    known_404s: set[str] = set()
    if perm404_log.exists():
        known_404s = {line.strip() for line in perm404_log.read_text().splitlines() if line.strip()}

    # Filter out tasks whose URL is a known permanent 404
    filtered_tasks = [(url, dest) for url, dest in all_tasks if url not in known_404s]
    n_pre_skipped_404 = len(all_tasks) - len(filtered_tasks)

    n_total = len(filtered_tasks)
    logger.info(f"Download plan: {len(ref_tasks)} reference files + "
                f"{len(sci_tasks)} science files = {len(all_tasks)} total")
    if n_pre_skipped_404:
        logger.info(f"  (silently skipping {n_pre_skipped_404} known permanent 404s — "
                    f"see {perm404_log.name})")

    if n_total == 0:
        logger.warning("No files to download (all already exist or are known 404s).")
        return {"n_downloaded": 0, "n_skipped": 0, "n_failed": 0,
                "n_permanent_404": n_pre_skipped_404, "failed_urls": []}

    # Parallel download
    n_downloaded  = 0
    n_skipped     = 0
    n_failed      = 0
    n_404         = n_pre_skipped_404   # count known + newly discovered
    failed_urls   = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, auth): (url, dest)
            for url, dest in filtered_tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, dest = futures[future]
            try:
                _, status, msg = future.result()
            except Exception as exc:
                status, msg = "failed", str(exc)

            if status == "skipped":
                n_skipped += 1

            elif status == "ok":
                n_downloaded += 1
                logger.debug(f"[{i}/{n_total}] {dest.name}: {msg}")
                if funpack and dest.exists():
                    funpack_file(dest)

            elif status == "404":
                # Permanent absence — record and never retry
                n_404 += 1
                with open(perm404_log, "a") as f:
                    f.write(f"{url}\n")

                # Distinguish reference products (no coverage) from missing science epochs
                is_ref = "/products/ref/" in url
                if is_ref:
                    logger.warning(f"[{i}/{n_total}] MISSING REFERENCE (no IRSA product) "
                                   f"{dest.name} — recorded in {perm404_log.name}")
                else:
                    # Science epoch has no processed diff image — log but don't warn noisily
                    logger.debug(f"[{i}/{n_total}] MISSING EPOCH {dest.name}")
                    with open(missing_log, "a") as f:
                        f.write(f"{url}\n")

            else:  # "failed" — retriable
                n_failed += 1
                failed_urls.append(url)
                logger.warning(f"[{i}/{n_total}] FAILED {dest.name}: {msg}")
                with open(failed_log, "a") as f:
                    f.write(f"{url}\n")

            if i % 50 == 0 or i == n_total:
                logger.info(f"Progress: {i}/{n_total} — "
                            f"{n_downloaded} downloaded, {n_skipped} skipped, "
                            f"{n_404} permanent-404, {n_failed} failed")

    result = {
        "n_downloaded":    n_downloaded,
        "n_skipped":       n_skipped,
        "n_permanent_404": n_404,
        "n_failed":        n_failed,
        "failed_urls":     failed_urls,
    }
    logger.info(f"Download complete: {result}")
    return result


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
        description="Download ZTF image products for a set of epochs."
    )
    parser.add_argument("--epochs",   type=Path, required=True,
                        help="Parquet file of epochs from ztf_field_lookup.py")
    parser.add_argument("--base-dir", type=Path, default=Path("data"),
                        help="Data directory (default: ./data in current working directory)")
    parser.add_argument("--bands",    nargs="+", default=None,
                        help="Bands to download: g r i (default: all in epochs file)")
    parser.add_argument("--username", type=str, default=None,
                        help="IRSA username (or set IRSA_USERNAME env var)")
    parser.add_argument("--password", type=str, default=None,
                        help="IRSA password (or set IRSA_PASSWORD env var)")
    parser.add_argument("--workers",  type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--skip-flagged", action="store_true",
                        help="Also skip cautionary epochs (infobits bits 2-6/11/21/22/26/27). "
                             "Hard-rejected epochs (bits 0/1/25) are always skipped.")
    parser.add_argument("--no-funpack", action="store_true",
                        help="Do not run funpack after downloading .fits.fz files")

    # Selection cuts — all optional; defaults download all non-hard-rejected epochs
    cuts = parser.add_argument_group(
        "selection cuts",
        "All cuts are applied before downloading (files you won't use are not fetched). "
        "Defaults download all epochs except hard-rejected (bits 0/1/25)."
    )
    cuts.add_argument("--max-seeing", type=float, default=None, metavar="ARCSEC",
                      help="Skip epochs with seeing FWHM above this value (arcsec)")
    cuts.add_argument("--min-maglim", type=float, default=None, metavar="MAG",
                      help="Skip epochs with 5σ limiting magnitude below this value")
    cuts.add_argument("--mjd-min", type=float, default=None, metavar="MJD",
                      help="Skip epochs before this MJD")
    cuts.add_argument("--mjd-max", type=float, default=None, metavar="MJD",
                      help="Skip epochs after this MJD")
    cuts.add_argument("--min-epochs-per-quad", type=int, default=None, metavar="N",
                      help="Skip quadrants with fewer than N epochs after other cuts")

    args = parser.parse_args()

    if not args.epochs.exists():
        print(f"ERROR: epochs file not found: {args.epochs}", file=sys.stderr)
        sys.exit(1)

    epochs = pd.read_parquet(args.epochs)
    logger.info(f"Loaded {len(epochs)} epochs from {args.epochs}")

    result = download_all(
        epochs=epochs,
        base_dir=args.base_dir,
        bands=args.bands,
        username=args.username,
        password=args.password,
        max_workers=args.workers,
        skip_flagged=args.skip_flagged,
        funpack=not args.no_funpack,
        max_seeing=args.max_seeing,
        min_maglim=args.min_maglim,
        mjd_min=args.mjd_min,
        mjd_max=args.mjd_max,
        min_epochs_per_quad=args.min_epochs_per_quad,
    )

    if result["n_permanent_404"] > 0:
        print(f"\n{result['n_permanent_404']} permanent 404s (missing IRSA products). "
              f"See {args.base_dir}/permanent_404s.log")
    if result["n_failed"] > 0:
        print(f"\n{result['n_failed']} retriable failures. "
              f"See {args.base_dir}/failed_downloads.log — re-run to retry.")
        sys.exit(1)


# ── Disk inventory ────────────────────────────────────────────────────────────

# Inline band→filtercode mapping to avoid importing ztf_field_lookup
_BAND_TO_FC = {"g": "zg", "r": "zr", "i": "zi"}


def find_quadrants(
    base_dir: Path,
    bands: list[str] | None = None,
    field: int | None = None,
    ccdid: int | None = None,
    qid: int | None = None,
) -> list[dict]:
    """
    Walk base_dir/Reference/ to find all downloaded (field, filtercode, ccdid, qid)
    combinations that have a refsexcat.fits.

    Returns a list of dicts:
        {field, filtercode, ccdid, qid, ref_dir, sci_dir}
    """
    import logging
    _log = logging.getLogger(__name__)

    ref_root = base_dir / "Reference"
    sci_root = base_dir / "Science"

    allowed_fcs = (
        {_BAND_TO_FC.get(b, b) for b in bands} if bands else None
    )

    quadrants = []
    for refcat in sorted(ref_root.rglob("*_refsexcat.fits")):
        if "_augmented" in refcat.name:
            continue

        parts = refcat.parts
        try:
            q_part   = next(p for p in reversed(parts) if p.startswith("q") and p[1:].isdigit())
            c_part   = parts[list(parts).index(q_part) - 1]
            fc_part  = parts[list(parts).index(c_part) - 1]
            fld_part = parts[list(parts).index(fc_part) - 1]
        except (StopIteration, ValueError):
            _log.debug(f"Cannot parse quadrant from {refcat} — skipping")
            continue

        if allowed_fcs and fc_part not in allowed_fcs:
            continue

        try:
            this_field = int(fld_part)
            this_ccd   = int(c_part.replace("ccd", ""))
            this_qid   = int(q_part[1:])
        except ValueError:
            continue

        if field is not None and this_field != field:
            continue
        if ccdid is not None and this_ccd != ccdid:
            continue
        if qid is not None and this_qid != qid:
            continue

        quadrants.append({
            "field":      this_field,
            "filtercode": fc_part,
            "ccdid":      this_ccd,
            "qid":        this_qid,
            "ref_dir":    refcat.parent,
            "sci_dir":    sci_root / fld_part / fc_part / c_part / q_part,
        })

    return quadrants


def purge_images(
    base_dir: Path,
    quadrants: list[dict],
    *,
    sci: bool = True,
    ref: bool = True,
    filefracdays: set | None = None,
    dry_run: bool = False,
) -> int:
    """
    Delete large imaging products after SExtractor catalogs are built.

    sci=True  — deletes diff.fits.fz, diff.fits, *_simulated.fits
    ref=True  — deletes refimg.fits, refsexcat.fits, refpsfcat.fits, refcov.fits
    filefracdays — if given, restrict science deletion to these epoch IDs only;
                   ref products are always deleted in full (one per quadrant)
    dry_run   — log what would be deleted without touching the disk

    Returns total bytes freed (or that would be freed).
    """
    sci_patterns = [
        "*_scimrefdiffimg.fits.fz",
        "*_scimrefdiffimg.fits",
        "*_simulated.fits",
    ]
    ref_patterns = [
        "*_refimg.fits",
        "*_refsexcat.fits",
        "*_refpsfcat.fits",
        "*_refcov.fits",
    ]

    to_delete: list[Path] = []

    for q in quadrants:
        field = q["field"]; fc = q["filtercode"]
        ccd = q["ccdid"]; qid_ = q["qid"]
        sci_dir = q.get("sci_dir",
            base_dir / "Science" / f"{field:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid_}")
        ref_dir = q.get("ref_dir",
            base_dir / "Reference" / f"{field:06d}" / fc / f"ccd{ccd:02d}" / f"q{qid_}")

        if sci and sci_dir.exists():
            for pat in sci_patterns:
                for p in sorted(sci_dir.glob(pat)):
                    if filefracdays is not None:
                        ffd = p.name.split("_")[1]
                        if ffd not in filefracdays:
                            continue
                    to_delete.append(p)

        if ref and ref_dir.exists():
            for pat in ref_patterns:
                to_delete.extend(sorted(ref_dir.glob(pat)))

    total_bytes = 0
    n_deleted = 0
    for p in to_delete:
        try:
            size = p.stat().st_size
            total_bytes += size
            if dry_run:
                logger.info(f"  [dry-run] {p.name} ({size/1e6:.1f} MB)")
            else:
                p.unlink()
                n_deleted += 1
                logger.debug(f"  deleted {p.name} ({size/1e6:.1f} MB)")
        except FileNotFoundError:
            pass

    freed_mb = total_bytes / 1e6
    action = "Would free" if dry_run else "Freed"
    n = len(to_delete)
    logger.info(f"purge_images: {n} files — {action} {freed_mb:.0f} MB")
    return total_bytes


def purge_hard_reject(base_dir: Path, epochs_parquet: Path, dry_run: bool = True) -> int:
    """Delete on-disk files belonging to hard-rejected epochs (bits 0/1/25)."""
    import pandas as pd
    epochs   = pd.read_parquet(epochs_parquet)
    infobits = pd.to_numeric(epochs.get("infobits", 0), errors="coerce").fillna(0).astype(int)
    bad      = epochs[(infobits & HARD_REJECT_MASK) != 0]
    logger.info(f"Hard-rejected: {len(bad):,} / {len(epochs):,}")
    if len(bad) == 0:
        return 0

    to_delete = []
    for _, row in bad.iterrows():
        ffd = str(row["filefracday"]); field = int(row["field"])
        fc  = str(row["filtercode"]); ccd = int(row["ccdid"]); qid_ = int(row["qid"])
        for suffix in ["scimrefdiffimg.fits", "scimrefdiffimg.fits.fz",
                       "sexcat.fits", "scimrefdiffimg_simulated.fits"]:
            p = sci_local_path(base_dir, field, fc, ccd, qid_, ffd, suffix)
            if p.exists(): to_delete.append(p)
        sex_p = (base_dir / "SExCatalogs" / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid_)
                 / f"ztf_{ffd}_{field:06d}_{fc}_c{ccd:02d}_o_q{qid_}_scimrefdiffimg_sexout.fits")
        if sex_p.exists(): to_delete.append(sex_p)

    if dry_run:
        logger.info(f"DRY RUN — would delete {len(to_delete):,} files")
        for p in to_delete[:20]: logger.info(f"  {p.relative_to(base_dir)}")
        if len(to_delete) > 20: logger.info(f"  ... and {len(to_delete)-20} more")
        return len(to_delete)

    n = sum(1 for p in to_delete if (p.unlink(), True)[1])
    logger.info(f"Purge: {n:,} deleted")
    return n
