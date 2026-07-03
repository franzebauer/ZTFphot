#!/usr/bin/env python3
"""
match_ps1_reference.py — cross-match a reference-objects catalog to Pan-STARRS DR1.

Replicates the original pipeline's CDS cross-match (STILTS cdsskymatch against
II/349/ps1, radius 1.5") in Python so every quadrant can be checked against the
same absolute anchor.  Writes a sidecar CSV next to the input with the matched
PS1 g/r magnitudes appended; the original catalog is never overwritten.

Output columns (sidecar):
    ALPHAWIN_J2000, DELTAWIN_J2000, MAG_APER_4px, MAGZP_REF,
    g_ps1, g_ps1_err, r_ps1, r_ps1_err, sep_arcsec

Usage (run in the `ztf` env, needs network):
    python match_ps1_reference.py "<...(REFERENCE)[OBJECTS].csv>" [more.csv ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

PS1_TABLE   = "II/349/ps1"
MATCH_RADIUS = 1.5      # arcsec, matches the original cdsskymatch radius
GMAG_MAX     = 21.0     # limit PS1 rows pulled; fainter than our calib range


def _query_ps1(ra, dec) -> pd.DataFrame:
    """Cone search of II/349/ps1 covering all (ra, dec); returns g/r PSF mags."""
    ra0, dec0 = float(np.mean(ra)), float(np.mean(dec))
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    radius = c.separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).max() + 30 * u.arcsec

    v = Vizier(columns=["RAJ2000", "DEJ2000", "gmag", "e_gmag", "rmag", "e_rmag"],
               column_filters={"gmag": f"<{GMAG_MAX}"},
               row_limit=-1)
    res = v.query_region(SkyCoord(ra0 * u.deg, dec0 * u.deg),
                         radius=radius, catalog=PS1_TABLE)
    if not res:
        raise RuntimeError("PS1 cone search returned no rows")
    t = res[0]
    return pd.DataFrame({
        "ra":      np.asarray(t["RAJ2000"], dtype=float),
        "dec":     np.asarray(t["DEJ2000"], dtype=float),
        "g_ps1":   np.asarray(t["gmag"],    dtype=float),
        "g_ps1_err": np.asarray(t["e_gmag"], dtype=float),
        "r_ps1":   np.asarray(t["rmag"],    dtype=float),
        "r_ps1_err": np.asarray(t["e_rmag"], dtype=float),
    })


def match(path: Path) -> None:
    df = pd.read_csv(path)
    ra  = pd.to_numeric(df["ALPHAWIN_J2000"], errors="coerce").values
    dec = pd.to_numeric(df["DELTAWIN_J2000"], errors="coerce").values

    ps1 = _query_ps1(ra, dec)
    print(f"{path.name}: {len(df)} ref objects, PS1 cone returned {len(ps1)} rows")

    cat_ref = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    cat_ps1 = SkyCoord(ra=ps1["ra"].values * u.deg, dec=ps1["dec"].values * u.deg)
    idx, sep, _ = cat_ref.match_to_catalog_sky(cat_ps1)
    ok = sep.arcsec < MATCH_RADIUS

    out = pd.DataFrame({
        "ALPHAWIN_J2000": ra,
        "DELTAWIN_J2000": dec,
        "MAG_APER_4px":   pd.to_numeric(df["MAG_APER_4px"], errors="coerce").values,
        "MAGZP_REF":      pd.to_numeric(df["MAGZP_REF"], errors="coerce").values,
        "g_ps1":     np.where(ok, ps1["g_ps1"].values[idx],     np.nan),
        "g_ps1_err": np.where(ok, ps1["g_ps1_err"].values[idx], np.nan),
        "r_ps1":     np.where(ok, ps1["r_ps1"].values[idx],     np.nan),
        "r_ps1_err": np.where(ok, ps1["r_ps1_err"].values[idx], np.nan),
        "sep_arcsec": np.where(ok, sep.arcsec, np.nan),
    })
    print(f"  matched {int(ok.sum())}/{len(df)} within {MATCH_RADIUS}\"")

    out_path = path.with_name(path.stem + "_ps1match.csv")
    out.to_csv(out_path, index=False)
    print(f"  → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python match_ps1_reference.py "<...(REFERENCE)[OBJECTS].csv>" [...]')
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            match(p)
        else:
            for p2 in sorted(Path(".").glob(arg)):
                match(p2)
