"""
check_target_detection.py
-------------------------
Diagnostic: check whether the target RA/Dec appears in the reference sexcat,
the reference CSV catalog, the per-epoch SExtractor output catalogs, and the
assembled light-curve parquet.

Single target (pipeline work-dir):
    python check_target_detection.py --ra 330.34158 --dec 0.72143 --base-dir data

Multiple work-dirs (RA/Dec parsed from directory name):
    python check_target_detection.py data_*/
    python check_target_detection.py data_*/ --good good.txt --missing redo.txt

Results-dir mode (parquet files from batch_pipeline output):
    python check_target_detection.py BAT_results/*.parquet --good good.txt --missing redo.txt

In results-dir mode RA/Dec are parsed from the parquet filename.  A target is
classified as GOOD only if the target appears in EVERY ref-pos parquet for
that coordinate (sci-pos variants are skipped).  Any parquet that lacks the
target causes the coordinate to be written to the missing file.

Output files match the batch_pipeline.py input format: RA,Dec per line.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

MATCH_RADIUS_ARCSEC = 3.0

# Matches {ra}_{dec} as a full name (directory mode) or as a prefix (parquet mode)
_PREFIX_RE = re.compile(r'^(\d+(?:\.\d+)?)_([+-]\d+(?:\.\d+)?)')


def _parse_ra_dec(name: str):
    """Parse RA, Dec from a directory name or parquet filename."""
    stem = Path(name).name
    if stem.endswith('.parquet'):
        stem = stem[:-8]
    m = _PREFIX_RE.match(stem)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _is_sci(path: Path) -> bool:
    """True for sci-pos parquet variants (_sci.parquet or _sci_merged.parquet)."""
    return path.name.endswith('_sci.parquet') or path.name.endswith('_sci_merged.parquet')


# ── Pipeline-stage checks (work-dir mode) ────────────────────────────────────

def check_refsexcat(ref_dir: Path, field: int, fc: str, ccd: int, qid: int,
                    tgt: SkyCoord) -> None:
    path = ref_dir / f"ztf_{field:06d}_{fc}_c{ccd:02d}_q{qid}_refsexcat.fits"
    print(f"  refsexcat : {path.name}", end="  ")
    if not path.exists():
        print("[NOT FOUND]")
        return
    with fits.open(path) as h:
        tbl = Table(h[1].data)
    cat = SkyCoord(ra=tbl["ALPHAWIN_J2000"], dec=tbl["DELTAWIN_J2000"], unit="deg")
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    d = sep[0].arcsec
    if d < MATCH_RADIUS_ARCSEC:
        row = tbl[int(idx)]
        print(f"MATCH  sep={d:.2f}\"  FLAGS={row['FLAGS']}  FLUX_BEST={row['FLUX_BEST']:.1f}")
    else:
        print(f"NO MATCH  (nearest={d:.2f}\")")


def check_ref_csv(cat_dir: Path, field: int, fc: str, ccd: int, qid: int,
                  tgt: SkyCoord) -> None:
    import pandas as pd
    path = cat_dir / f"{field:06d}_{fc}_c{ccd:02d}_q{qid}(REFERENCE)[OBJECTS].csv"
    print(f"  ref csv   : {path.name}", end="  ")
    if not path.exists():
        print("[NOT FOUND]")
        return
    df = pd.read_csv(path)
    ra_col  = "ALPHAWIN_J2000" if "ALPHAWIN_J2000" in df.columns else "RA"
    dec_col = "DELTAWIN_J2000" if "DELTAWIN_J2000" in df.columns else "DEC"
    cat = SkyCoord(ra=df[ra_col].values, dec=df[dec_col].values, unit="deg")
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    d = sep[0].arcsec
    if d < MATCH_RADIUS_ARCSEC:
        print(f"MATCH  sep={d:.2f}\"  row_index={int(idx)}")
    else:
        print(f"NO MATCH  (nearest={d:.2f}\")")


def check_sexout_catalogs(sex_dir: Path, tgt: SkyCoord, max_show: int = 5) -> None:
    cats = sorted(sex_dir.glob("*_sexout.fits"))
    print(f"  sexout    : {len(cats)} catalogs in {sex_dir.relative_to(sex_dir.parents[4])}")
    if not cats:
        return
    n_match = 0
    match_examples = []
    no_match_examples = []
    for path in cats:
        try:
            with fits.open(path) as h:
                tbl = Table(h[2].data)
        except Exception:
            continue
        if len(tbl) == 0:
            no_match_examples.append((path.name, "empty"))
            continue
        ra_col  = "ALPHAWIN_J2000" if "ALPHAWIN_J2000" in tbl.colnames else "ALPHA_J2000"
        dec_col = "DELTAWIN_J2000" if "DELTAWIN_J2000" in tbl.colnames else "DELTA_J2000"
        try:
            cat = SkyCoord(ra=tbl[ra_col], dec=tbl[dec_col], unit="deg")
        except Exception:
            continue
        idx, sep, _ = tgt.match_to_catalog_sky(cat)
        d = sep[0].arcsec
        if d < MATCH_RADIUS_ARCSEC:
            n_match += 1
            if len(match_examples) < max_show:
                row = tbl[int(idx)]
                flux = row["FLUX_APER"][1] if "FLUX_APER" in tbl.colnames else float("nan")
                match_examples.append((path.name, d, flux))
        else:
            if len(no_match_examples) < max_show:
                no_match_examples.append((path.name, d))

    print(f"    matched in {n_match}/{len(cats)} epochs")
    if match_examples:
        print("    examples (matched):")
        for name, d, flux in match_examples:
            print(f"      {name}  sep={d:.2f}\"  FLUX_APER[k=1]={flux:.1f}")
    if no_match_examples and n_match < len(cats):
        print("    examples (no match):")
        for item in no_match_examples:
            if item[1] == "empty":
                print(f"      {item[0]}  (empty catalog)")
            else:
                print(f"      {item[0]}  nearest={item[1]:.2f}\"")


def check_lightcurve(lc_path: Path, tgt: SkyCoord) -> bool:
    """Returns True if the target is found within MATCH_RADIUS_ARCSEC."""
    import pandas as pd
    print(f"  parquet   : {lc_path.name}", end="  ")
    if not lc_path.exists():
        print("[NOT FOUND]")
        return False
    df = pd.read_parquet(lc_path, columns=["object_index", "ALPHAWIN_REF", "DELTAWIN_REF"])
    srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        print("NO SOURCES")
        return False
    cat = SkyCoord(ra=srcs["ALPHAWIN_REF"].values, dec=srcs["DELTAWIN_REF"].values, unit="deg")
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    d = sep[0].arcsec
    if d < MATCH_RADIUS_ARCSEC:
        obj_idx = srcs.index[int(idx)]
        n_epochs = (df["object_index"] == obj_idx).sum()
        print(f"MATCH  sep={d:.2f}\"  object_index={obj_idx}  N_epochs={n_epochs}")
        return True
    print(f"NO MATCH  (nearest={d:.2f}\")")
    return False


def check_target_workdir(ra, dec, base_dir, field, band, ccdid, qid):
    """Check all pipeline stages for one target in a work-dir. Returns True if LC found."""
    tgt = SkyCoord(ra=ra, dec=dec, unit="deg")

    ref_root = base_dir / "Reference"
    if not ref_root.exists():
        print(f"  [Reference directory not found: {ref_root}]")
        return False

    quadrants = []
    for field_dir in sorted(ref_root.iterdir()):
        try:
            f = int(field_dir.name)
        except ValueError:
            continue
        if field is not None and f != field:
            continue
        for fc_dir in sorted(field_dir.iterdir()):
            fc = fc_dir.name
            if band is not None and fc != band:
                continue
            for ccd_dir in sorted(fc_dir.iterdir()):
                try:
                    ccd = int(ccd_dir.name.replace("ccd", ""))
                except ValueError:
                    continue
                if ccdid is not None and ccd != ccdid:
                    continue
                for qid_dir in sorted(ccd_dir.iterdir()):
                    try:
                        q = int(qid_dir.name.replace("q", ""))
                    except ValueError:
                        continue
                    if qid is not None and q != qid:
                        continue
                    quadrants.append(dict(field=f, fc=fc, ccd=ccd, qid=q,
                                         ref_dir=qid_dir))

    if not quadrants:
        print("  [No quadrants found]")
        return False

    found = False
    for quad in quadrants:
        f, fc, ccd, q = quad["field"], quad["fc"], quad["ccd"], quad["qid"]
        print(f"\n{'='*60}")
        print(f"  {f:06d} {fc} ccd{ccd:02d} q{q}")
        print(f"{'='*60}")
        check_refsexcat(quad["ref_dir"], f, fc, ccd, q, tgt)
        check_ref_csv(base_dir / "Catalogs", f, fc, ccd, q, tgt)
        sex_dir = base_dir / "SExCatalogs" / f"{f:06d}" / fc / f"{ccd:02d}" / str(q)
        check_sexout_catalogs(sex_dir, tgt)
        lc_path = (base_dir / "LightCurves" / f"{f:06d}" / fc
                   / f"ccd{ccd:02d}" / f"q{q}" / "lightcurves.parquet")
        if check_lightcurve(lc_path, tgt):
            found = True

    return found


# ── Results-dir mode ──────────────────────────────────────────────────────────

def check_results_parquets(ra, dec, parquets):
    """
    Check every ref-pos parquet for this target.
    Returns True only if the target is found in ALL of them.
    """
    tgt = SkyCoord(ra=ra, dec=dec, unit="deg")
    all_found = True
    for p in parquets:
        if not check_lightcurve(p, tgt):
            all_found = False
    return all_found


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Check target detection across pipeline stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", type=Path,
                   help="Work-dirs or parquet files (shell glob supported). "
                        "RA/Dec are parsed from the name.")
    p.add_argument("--ra",       type=float, default=None)
    p.add_argument("--dec",      type=float, default=None)
    p.add_argument("--base-dir", type=Path,  default=Path("data"),
                   help="Base directory for single-target mode (default: data)")
    p.add_argument("--field",    type=int,   default=None)
    p.add_argument("--band",     type=str,   default=None, help="e.g. zg, zr, zi")
    p.add_argument("--ccdid",    type=int,   default=None)
    p.add_argument("--qid",      type=int,   default=None)
    p.add_argument("--good",     type=Path,  default=None, metavar="FILE",
                   help="Write RA,Dec of targets with a detected LC to this file")
    p.add_argument("--missing",  type=Path,  default=None, metavar="FILE",
                   help="Write RA,Dec of targets with a missing/incomplete LC to this file")
    args = p.parse_args()

    good_coords    = []
    missing_coords = []

    if args.paths:
        paths = [p_.resolve() for p_ in args.paths]

        # Separate into parquet files and directories
        parquet_files = [p_ for p_ in paths if p_.suffix == '.parquet']
        directories   = [p_ for p_ in paths if p_.is_dir()]

        # Group parquets by coord key
        groups    = {}
        coord_map = {}

        # ── Results-dir mode: parquet files given ────────────────────────────
        if parquet_files:
            skipped = 0
            for pq in sorted(parquet_files):
                if _is_sci(pq):
                    continue
                parsed = _parse_ra_dec(pq.name)
                if parsed is None:
                    print(f"WARNING: cannot parse RA/Dec from '{pq.name}' — skipping",
                          file=sys.stderr)
                    skipped += 1
                    continue
                ra, dec = parsed
                key = f"{ra:.5f}_{dec:+.5f}"
                groups.setdefault(key, []).append(pq)
                coord_map[key] = (ra, dec)

            if not groups:
                sys.exit("No parseable parquet files found.")

            for key in sorted(groups):
                ra, dec = coord_map[key]
                pqs = sorted(groups[key])
                print(f"\n{'#'*60}")
                print(f"  TARGET  RA={ra}  Dec={dec}")
                print(f"{'#'*60}")
                found = check_results_parquets(ra, dec, pqs)
                if found:
                    good_coords.append((ra, dec))
                else:
                    missing_coords.append((ra, dec))

        # ── Work-dir mode: directories given ─────────────────────────────────
        for d in directories:
            parsed = _parse_ra_dec(d.name)
            if parsed is None:
                print(f"WARNING: cannot parse RA/Dec from '{d.name}' — skipping",
                      file=sys.stderr)
                continue
            ra, dec = parsed
            print(f"\n{'#'*60}")
            print(f"  TARGET  RA={ra}  Dec={dec}  ({d.name})")
            print(f"{'#'*60}")
            found = check_target_workdir(ra, dec, d,
                                         field=args.field, band=args.band,
                                         ccdid=args.ccdid, qid=args.qid)
            if found:
                good_coords.append((ra, dec))
            else:
                missing_coords.append((ra, dec))

    else:
        # ── Single-target mode ────────────────────────────────────────────────
        if args.ra is None or args.dec is None:
            p.error("--ra and --dec are required when no paths are given")
        ra, dec = args.ra, args.dec
        base_dir = args.base_dir.resolve()
        print(f"\n{'#'*60}")
        print(f"  TARGET  RA={ra}  Dec={dec}")
        print(f"{'#'*60}")
        found = check_target_workdir(ra, dec, base_dir,
                                     field=args.field, band=args.band,
                                     ccdid=args.ccdid, qid=args.qid)
        if found:
            good_coords.append((ra, dec))
        else:
            missing_coords.append((ra, dec))

    print()

    if len(good_coords) + len(missing_coords) > 1:
        print(f"Summary: {len(good_coords)} good, {len(missing_coords)} missing "
              f"(out of {len(good_coords) + len(missing_coords)})")

    if args.good is not None:
        with open(args.good, "w") as f:
            for ra, dec in good_coords:
                f.write(f"{ra},{dec}\n")
        print(f"good    → {args.good}  ({len(good_coords)} targets)")

    if args.missing is not None:
        with open(args.missing, "w") as f:
            for ra, dec in missing_coords:
                f.write(f"{ra},{dec}\n")
        print(f"missing → {args.missing}  ({len(missing_coords)} targets)")


if __name__ == "__main__":
    main()
