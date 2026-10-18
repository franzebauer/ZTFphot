#!/usr/bin/env python
"""
compare_aperture_sex.py
-----------------------
Validate aperture_photometry against existing SExtractor catalogs, epoch by epoch.

For every difference image under <base>/Science that still has its SExtractor
catalog under <base>/SExCatalogs (ref-pos) — and, with --sci, SExCatalogs_sci —
re-measure it with aperture_photometry.measure_epoch into a scratch directory
and compare per source (matched on VECTOR_ASSOC = reference index):

  dF/sigma  = (FLUX_aper - FLUX_sex) / FLUXERR_sex    per aperture
  err ratio = FLUXERR_aper / FLUXERR_sex
  dpos      = offset of the SExtractor aperture centre from the forced position [px]

Sources are split into "clean" (SExtractor centre within 0.1 px of the forced
position, no blend flag) and the rest, which isolates the effect of the detection
step from the measurement itself.

    python compare_aperture_sex.py <base_dir> [--bkg local mesh none] [--sci] [--max-epochs N]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aperture_photometry as A          # noqa: E402
from calib_catalogs import _read_ldac    # noqa: E402

warnings.filterwarnings("ignore")


def _table(path):
    _, t = _read_ldac(str(path))
    va = np.asarray(t["VECTOR_ASSOC"]).reshape(len(t), -1)[:, 0].astype(int)
    col = lambda c: np.asarray(t[c]).astype(np.float64)       # native byte order for pandas
    return pd.DataFrame({
        "idx": va,
        "ra": col("ALPHAWIN_J2000"), "dec": col("DELTAWIN_J2000"),
        **{f"F{k}": col("FLUX_APER")[:, k] for k in range(4)},
        **{f"E{k}": col("FLUXERR_APER")[:, k] for k in range(4)},
        "FLAGS": np.asarray(t["FLAGS"]).astype(int)})


def compare(base: Path, bkg: str, sci: bool, max_epochs, scratch: Path) -> pd.DataFrame:
    from astropy.io import fits
    from astropy.wcs import WCS
    rows = []
    diffs = sorted(base.glob("Science/*/*/ccd*/q*/*_scimrefdiffimg.fits"))
    n = 0
    for d in diffs:
        field, fc, ccd, q = d.parts[-5], d.parts[-4], d.parts[-3][3:], d.parts[-2][1:]
        sub = "SExCatalogs_sci" if sci else "SExCatalogs"
        sex = base / sub / field / fc / ccd / q / d.name.replace(".fits", "_sexout.fits")
        ref = base / "Catalogs" / f"{field}_{fc}_c{ccd}_q{q}(REFERENCE)[OBJECTS].csv"
        if not (A.catalog_ok(sex) and ref.exists()):
            continue
        sexcat = d.with_name(d.name.replace("_scimrefdiffimg.fits", "_sexcat.fits")) if sci else None
        out = scratch / f"{bkg}_{'sci_' if sci else ''}{d.name}"
        A.measure_epoch(d, out, ref, sexcat_path=sexcat, bkg=bkg)
        s, a = _table(sex), _table(out)
        if sci:      # sci-pos: several detections can share an index — compare 1:1 pairs only
            s, a = s.drop_duplicates("idx", keep=False), a.drop_duplicates("idx", keep=False)
        m = s.merge(a, on="idx", suffixes=("_s", "_a"))
        w = WCS(fits.getheader(d))
        xs, ys = w.all_world2pix(m["ra_s"], m["dec_s"], 0)
        xa, ya = w.all_world2pix(m["ra_a"], m["dec_a"], 0)
        m["dpos"] = np.hypot(xs - xa, ys - ya)
        m["epoch"] = d.name[4:18]
        m["quad"] = f"{field}/{fc}/c{ccd}/q{q}"
        m["n_sex"], m["n_aper"] = len(s), len(a)
        rows.append(m)
        n += 1
        if max_epochs and n >= max_epochs:
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(m: pd.DataFrame, label: str):
    ep = m.groupby("epoch")[["n_sex", "n_aper"]].first()
    print(f"\n=== {label}: {m['epoch'].nunique()} epochs, {len(m)} matched source-epochs; "
          f"rows per epoch SEx {ep['n_sex'].sum()} vs aper {ep['n_aper'].sum()} ===")
    print(f"SExtractor centre offset from forced position [px]: median {m.dpos.median():.3f}, "
          f"90% {m.dpos.quantile(.9):.2f}, frac > 0.5 px {np.mean(m.dpos > 0.5):.1%}")
    clean = (m.dpos < 0.1) & ((m["FLAGS_a"] & 1) == 0)
    for name, sel in (("clean", clean), ("offset/blend", ~clean)):
        d = m[sel]
        if not len(d):
            continue
        out = []
        for k, dia in enumerate((3, 4, 6, 10)):
            z = (d[f"F{k}_a"] - d[f"F{k}_s"]) / d[f"E{k}_s"]
            r = d[f"E{k}_a"] / d[f"E{k}_s"]
            out.append((f"{dia}px", len(d), np.median(z), 1.4826 * np.median(np.abs(z - np.median(z))),
                        np.mean(np.abs(z) > 1), np.median(r)))
        print(f"  {name}:")
        print(pd.DataFrame(out, columns=["aper", "n", "median dF/sig", "MAD dF/sig",
                                         "frac |dF|>1sig", "median err ratio"])
              .round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("--bkg", nargs="+", default=["local"], choices=["local", "mesh", "none"])
    ap.add_argument("--sci", action="store_true", help="also compare sci-pos catalogs")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write matched table(s) as parquet here")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for bkg in a.bkg:
            for sci in ([False, True] if a.sci else [False]):
                m = compare(a.base, bkg, sci, a.max_epochs, Path(tmp))
                if m.empty:
                    print(f"no comparable epochs for bkg={bkg} sci={sci}")
                    continue
                summarize(m, f"bkg={bkg} {'sci-pos' if sci else 'ref-pos'}")
                if a.out:
                    a.out.mkdir(parents=True, exist_ok=True)
                    m.to_parquet(a.out / f"cmp_{bkg}_{'sci' if sci else 'ref'}.parquet")


if __name__ == "__main__":
    main()
