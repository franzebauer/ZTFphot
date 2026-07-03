#!/usr/bin/env python3
"""
audit_calib_slope.py — per-epoch calibration-slope audit, grouped by quadrant.

Scans calibrated catalogs (*_cal.fits) and reads the per-epoch calibration
diagnostics stored in each primary header by calib_catalogs.py:

    CALIB_M   slope of the per-epoch linear fit  diff = m*maginst + n
    CALIB_N   intercept
    CALIB_ZP  n + m*17   (zeropoint referenced to mag 17)
    fit_rms   calibration RMS
    num_stars number of calibration stars used

A non-zero / quadrant-dependent CALIB_M tilts the calibrated magnitudes
magnitude-dependently.  Quadrants with few epochs cannot average this out,
so a biased CALIB_M there appears as a magnitude-dependent inter-quadrant
offset in the merged light curves.

Usage:
    python audit_calib_slope.py <Calibrated_dir> [--band zg]
    python audit_calib_slope.py <Calibrated/FIELD/zg/CC/Q> [...]
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

_FN_RE = re.compile(r"_(\d{6})_(z[gri])_c(\d{2})_o_q(\d)_")
_TS_RE = re.compile(r"ztf_(\d{8})")


def _scan(cal_files):
    rows = []
    for f in cal_files:
        m = _FN_RE.search(f.name)
        if not m:
            continue
        field, band, ccd, qid = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        ts = _TS_RE.search(f.name)
        date = int(ts.group(1)) if ts else 0
        try:
            h = fits.getheader(f, 0)
        except Exception as e:
            print(f"  WARN unreadable header {f.name}: {e}")
            continue
        rows.append(dict(
            quad=f"{field}/c{ccd:02d}/q{qid}", band=band, date=date,
            m=h.get("CALIB_M", np.nan), n=h.get("CALIB_N", np.nan),
            zp=h.get("CALIB_ZP", np.nan), rms=h.get("fit_rms", np.nan),
            nstars=h.get("num_stars", np.nan),
        ))
    return rows


def _stat(label, vals):
    v = np.asarray([x for x in vals if np.isfinite(x) and x != -999.0], dtype=float)
    if len(v) == 0:
        return f"    {label:9s}  (no valid values)"
    return (f"    {label:9s}  median={np.median(v):+.5f}  mean={np.mean(v):+.5f}  "
            f"std={np.std(v):.5f}  min={v.min():+.4f}  max={v.max():+.4f}")


def main(args):
    band = "zg"
    if "--band" in args:
        i = args.index("--band")
        band = args[i + 1]
        args = args[:i] + args[i + 2:]

    cal_files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            cal_files += sorted(p.rglob(f"*{band}*_cal.fits"))
        elif p.exists():
            cal_files.append(p)
    if not cal_files:
        sys.exit("No *_cal.fits files found.")

    rows = _scan(cal_files)
    quads = sorted({r["quad"] for r in rows})

    print(f"\nScanned {len(rows)} calibrated epochs across {len(quads)} quadrants (band {band})\n")
    for q in quads:
        sub = [r for r in rows if r["quad"] == q]
        print(f"  {q}   ({len(sub)} epochs)")
        print(_stat("CALIB_M", [r["m"] for r in sub]))
        print(_stat("CALIB_N", [r["n"] for r in sub]))
        print(_stat("CALIB_ZP", [r["zp"] for r in sub]))
        print(_stat("fit_rms", [r["rms"] for r in sub]))
        print(_stat("num_star", [r["nstars"] for r in sub]))
        print()

    # ── Plot: CALIB_M distribution + vs time, per quadrant ─────────────────────
    fig, (axh, axt) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(quads)))
    for q, c in zip(quads, colors):
        sub = [r for r in rows if r["quad"] == q]
        mv = np.array([r["m"] for r in sub if np.isfinite(r["m"]) and r["m"] != -999.0])
        if len(mv):
            axh.hist(mv, bins=40, histtype="step", lw=1.8, color=c, density=True,
                     label=f"{q} (n={len(mv)}, med={np.median(mv):+.4f})")
            dv = np.array([r["date"] for r in sub if np.isfinite(r["m"]) and r["m"] != -999.0])
            axt.scatter(dv, mv, s=10, alpha=0.5, color=c, label=q)
    axh.axvline(0, color="black", lw=1, ls="--")
    axh.set_xlabel("CALIB_M  (per-epoch slope)")
    axh.set_ylabel("normalised density")
    axh.set_title("Calibration slope distribution")
    axh.legend(fontsize=8)
    axt.axhline(0, color="black", lw=1, ls="--")
    axt.set_xlabel("date (YYYYMMDD)")
    axt.set_ylabel("CALIB_M")
    axt.set_title("Calibration slope vs time")
    axt.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args[0]).parent / "calib_slope_audit.png" if Path(args[0]).is_file() \
        else Path(args[0]) / "calib_slope_audit.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python audit_calib_slope.py <Calibrated_dir> [--band zg]")
    main(sys.argv[1:])
