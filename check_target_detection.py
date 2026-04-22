"""
check_target_detection.py
-------------------------
Diagnostic: check whether the target RA/Dec appears in the reference sexcat,
the reference CSV catalog, the per-epoch SExtractor output catalogs, and the
assembled light-curve parquet.

Usage (ztf environment):
    python check_target_detection.py --ra 330.34158 --dec 0.72143
    python check_target_detection.py --ra 330.34158 --dec 0.72143 \
        --field 443 --band zg --ccdid 16 --qid 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

MATCH_RADIUS_ARCSEC = 3.0


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


def check_sexout_catalogs(sex_dir: Path, tgt: SkyCoord,
                          max_show: int = 5) -> None:
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
                tbl = Table(h[2].data)   # LDAC: HDU[1]=header, HDU[2]=objects
        except Exception as e:
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


def check_lightcurve(lc_path: Path, tgt: SkyCoord) -> None:
    import pandas as pd
    print(f"  parquet   : {lc_path.name}", end="  ")
    if not lc_path.exists():
        print("[NOT FOUND]")
        return
    df = pd.read_parquet(lc_path, columns=["object_index", "ALPHAWIN_REF", "DELTAWIN_REF"])
    srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        print("NO SOURCES")
        return
    cat = SkyCoord(ra=srcs["ALPHAWIN_REF"].values, dec=srcs["DELTAWIN_REF"].values, unit="deg")
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    d = sep[0].arcsec
    if d < MATCH_RADIUS_ARCSEC:
        obj_idx = srcs.index[int(idx)]
        n_epochs = (df["object_index"] == obj_idx).sum()
        print(f"MATCH  sep={d:.2f}\"  object_index={obj_idx}  N_epochs={n_epochs}")
    else:
        print(f"NO MATCH  (nearest={d:.2f}\")")


def main() -> None:
    p = argparse.ArgumentParser(description="Check target detection across pipeline stages.")
    p.add_argument("--ra",       type=float, required=True)
    p.add_argument("--dec",      type=float, required=True)
    p.add_argument("--base-dir", type=Path,  default=Path("data"))
    p.add_argument("--field",    type=int,   default=None)
    p.add_argument("--band",     type=str,   default=None, help="e.g. zg, zr, zi")
    p.add_argument("--ccdid",    type=int,   default=None)
    p.add_argument("--qid",      type=int,   default=None)
    args = p.parse_args()

    base_dir = args.base_dir.resolve()
    tgt = SkyCoord(ra=args.ra, dec=args.dec, unit="deg")

    # Discover quadrants from Reference directory
    ref_root = base_dir / "Reference"
    quadrants = []
    for field_dir in sorted(ref_root.iterdir()):
        try:
            field = int(field_dir.name)
        except ValueError:
            continue
        if args.field is not None and field != args.field:
            continue
        for fc_dir in sorted(field_dir.iterdir()):
            fc = fc_dir.name
            if args.band is not None and fc != args.band:
                continue
            for ccd_dir in sorted(fc_dir.iterdir()):
                ccd_str = ccd_dir.name.replace("ccd", "")
                try:
                    ccd = int(ccd_str)
                except ValueError:
                    continue
                if args.ccdid is not None and ccd != args.ccdid:
                    continue
                for qid_dir in sorted(ccd_dir.iterdir()):
                    try:
                        qid = int(qid_dir.name.replace("q", ""))
                    except ValueError:
                        continue
                    if args.qid is not None and qid != args.qid:
                        continue
                    quadrants.append(dict(field=field, fc=fc, ccd=ccd, qid=qid,
                                         ref_dir=qid_dir))

    if not quadrants:
        sys.exit("No quadrants found. Check --base-dir and filter arguments.")

    for q in quadrants:
        field, fc, ccd, qid = q["field"], q["fc"], q["ccd"], q["qid"]
        tag = f"{field:06d} {fc} ccd{ccd:02d} q{qid}"
        print(f"\n{'='*60}")
        print(f"  {tag}")
        print(f"{'='*60}")

        check_refsexcat(q["ref_dir"], field, fc, ccd, qid, tgt)
        check_ref_csv(base_dir / "Catalogs", field, fc, ccd, qid, tgt)

        sex_dir = base_dir / "SExCatalogs" / f"{field:06d}" / fc / f"{ccd:02d}" / str(qid)
        check_sexout_catalogs(sex_dir, tgt)

        lc_path = (base_dir / "LightCurves" / f"{field:06d}" / fc
                   / f"ccd{ccd:02d}" / f"q{qid}" / "lightcurves.parquet")
        check_lightcurve(lc_path, tgt)

    print()


if __name__ == "__main__":
    main()
