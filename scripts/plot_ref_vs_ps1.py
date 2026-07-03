#!/usr/bin/env python3
"""
plot_ref_vs_ps1.py — per-quadrant absolute calibration check against Pan-STARRS.

For each quadrant's *_ps1match.csv (from match_ps1_reference.py), plots the
reference-photometry residual against PS1:

    residual = (MAG_APER_4px + MAGZP_REF) - g_ps1

Rows (one column per quadrant):
  0. residual vs magnitude, NO colour correction
  1. residual vs magnitude, WITH the pipeline colour term applied
        m_cal = m_inst + ZP + CLRCOEFF*(g-r)  →  resid_corr = resid + CLRCOEFF*(g-r)
  2. residual vs colour g-r, showing the empirical fit and the pipeline slope

The pipeline CLRCOEFF is read per quadrant, preferring the reference image
(Reference/.../*_refimg.fits) and falling back to the median over the
calibrated epochs (Calibrated/.../*_cal.fits).  Comparing rows 0 and 1 shows
whether the pipeline colour term removes the colour-dependent part of the
residual; comparing the empirical and pipeline slopes in row 2 shows whether
the pipeline value matches the data.

Usage:
    python plot_ref_vs_ps1.py [--data-root DIR] <a_ps1match.csv> <b_ps1match.csv> ...

    --data-root  directory holding Reference/ and Calibrated/ (default: the
                 parent of the CSV's parent, i.e. the field root).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

_QRE = re.compile(r"(\d{6})_(z[gri])_c(\d{2})_q(\d)")


def _pipeline_clrcoeff(csv_path: Path, data_root: Path | None):
    """Pipeline CLRCOEFF for the quadrant: prefer the reference image, else the
    median over calibrated epochs.  Returns (value, source) or (None, reason)."""
    m = _QRE.search(csv_path.name)
    if not m:
        return None, "no-parse"
    field, band, ccd, qid = m.group(1), m.group(2), m.group(3), int(m.group(4))
    root = Path(data_root) if data_root else csv_path.resolve().parent.parent

    for rf in root.glob(f"Reference/**/ztf_{field}_{band}_c{ccd}_q{qid}_refimg.fits"):
        try:
            v = fits.getheader(rf, 0).get("CLRCOEFF")
            if v is not None:
                return float(v), "refimg"
        except Exception:
            pass

    cals = sorted(root.glob(f"Calibrated/**/ztf_*_{field}_{band}_c{ccd}_o_q{qid}_*_cal.fits"))
    vals = []
    for cf in cals[:200]:
        try:
            v = fits.getheader(cf, 0).get("CLRCOEFF")
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    if vals:
        return float(np.median(vals)), f"calib(n={len(vals)})"
    return None, "not-found"

MAG_BRIGHT, MAG_FAINT = 14.0, 21.0   # PS1 g saturates ~13.5-14.5; stay above it
MAX_PS1_ERR = 0.05
MAX_SEP     = 1.0


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ztf_ref"] = df["MAG_APER_4px"] + df["MAGZP_REF"]
    df["resid"]   = df["ztf_ref"] - df["g_ps1"]
    df["color"]   = df["g_ps1"] - df["r_ps1"]
    ok = (
        df["g_ps1"].notna() & df["r_ps1"].notna() &
        (df["g_ps1_err"].fillna(9) < MAX_PS1_ERR) &
        (df["r_ps1_err"].fillna(9) < MAX_PS1_ERR) &
        (df["sep_arcsec"].fillna(9) < MAX_SEP) &
        (df["g_ps1"] > MAG_BRIGHT) & (df["g_ps1"] < MAG_FAINT)
    )
    return df[ok]


def _binned(x, y, lo, hi, n=18):
    edges = np.linspace(lo, hi, n + 1)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() >= 5:
            cen.append(0.5 * (a + b))
            med.append(np.median(y[sel]))
    return np.array(cen), np.array(med)


def _fit_color_term(color, resid):
    """Robust linear fit resid = c*color + b; returns (c, b) with one 3σ clip."""
    m = np.isfinite(color) & np.isfinite(resid)
    c, r = color[m], resid[m]
    if len(c) < 10:
        return 0.0, float(np.median(r)) if len(r) else 0.0
    coef = np.polyfit(c, r, 1)
    res  = r - np.polyval(coef, c)
    s    = np.std(res)
    keep = np.abs(res) < 3 * s if s > 0 else np.ones(len(c), bool)
    coef = np.polyfit(c[keep], r[keep], 1)
    return float(coef[0]), float(coef[1])


def _mag_panel(ax, mag, resid, base, label):
    ax.scatter(mag, resid, s=4, alpha=0.25, color="steelblue", rasterized=True)
    bx, by = _binned(mag, resid, MAG_BRIGHT, MAG_FAINT)
    ax.plot(bx, by, color="orange", lw=2, label="binned median")
    ax.axhline(base, color="crimson", lw=1.2, ls="-", label=f"median = {base:+.3f}")
    ax.set_xlabel("ZTF ref mag  (MAG_APER_4px + MAGZP_REF)")
    ax.set_ylabel(label)
    ax.set_ylim(base - 0.4, base + 0.4)
    ax.legend(fontsize=8)
    br = (mag >= 14) & (mag < 17)
    if br.sum() >= 10:
        s = np.polyfit(mag[br], resid[br], 1)[0]
        ax.text(0.04, 0.04, f"slope[14-17] = {s:+.3f} mag/mag",
                transform=ax.transAxes, fontsize=9, color="black")
        return s
    return np.nan


def _quad_name(path: Path) -> str:
    s = path.name
    return s.split("(")[0] if "(" in s else path.stem


def main(paths, data_root=None):
    dfs = {_quad_name(p): (p, _load(p)) for p in paths}
    n = len(dfs)
    fig, axes = plt.subplots(3, n, figsize=(5.5 * n, 13), squeeze=False)

    for col, (name, (path, df)) in enumerate(dfs.items()):
        mag   = df["ztf_ref"].values
        resid = df["resid"].values
        color = df["color"].values
        base  = float(np.median(resid))
        cmed  = float(np.median(color[np.isfinite(color)]))

        c_emp, b = _fit_color_term(color, resid)             # empirical, for comparison
        clr, src = _pipeline_clrcoeff(path, data_root)       # pipeline CLRCOEFF

        if clr is None:
            clr_used, src = c_emp * 0.0, f"none({src})"
            resid_cc = resid.copy()
        else:
            clr_used = clr
            resid_cc = resid + clr_used * (color - cmed)     # m_cal = m_inst+ZP+CLRCOEFF*(g-r)

        # ── Row 0: no colour correction ──────────────────────────────────────
        s_raw = _mag_panel(axes[0][col], mag, resid, base, "ref - g_ps1  (mag)")
        axes[0][col].set_title(f"{name}  (n={len(df)})  — no colour corr")

        # ── Row 1: pipeline colour correction ────────────────────────────────
        s_cc = _mag_panel(axes[1][col], mag, resid_cc, base,
                          "ref - g_ps1 + CLRCOEFF·(g-r)  (mag)")
        axes[1][col].set_title(f"pipeline colour corr  "
                               f"(CLRCOEFF = {clr_used:+.4f}, {src})")

        # ── Row 2: residual vs colour: empirical fit vs pipeline slope ────────
        ax2 = axes[2][col]
        ax2.scatter(color, resid, s=4, alpha=0.25, color="seagreen", rasterized=True)
        cc, cm = _binned(color, resid, 0.0, 1.5)
        ax2.plot(cc, cm, color="orange", lw=2, label="binned median")
        xline = np.array([np.nanmin(color), np.nanmax(color)])
        ax2.plot(xline, c_emp * (xline - cmed) + base, color="purple", lw=1.8, ls="--",
                 label=f"empirical: {c_emp:+.4f}(g-r)")
        # pipeline predicts raw resid has slope -CLRCOEFF (so +CLRCOEFF cancels it)
        ax2.plot(xline, -clr_used * (xline - cmed) + base, color="black", lw=1.8, ls=":",
                 label=f"pipeline: {-clr_used:+.4f}(g-r)")
        ax2.axhline(base, color="crimson", lw=1.0, ls="-")
        ax2.set_xlabel("g - r  (PS1)")
        ax2.set_ylabel("ref - g_ps1  (mag)")
        ax2.set_ylim(base - 0.4, base + 0.4)
        ax2.legend(fontsize=8)

        print(f"  {name}: CLRCOEFF={clr_used:+.4f} ({src})  empirical c={c_emp:+.4f}  "
              f"slope[14-17] raw={s_raw:+.3f}  pipeline-corr={s_cc:+.3f}")

    fig.suptitle("Reference photometry vs Pan-STARRS  (per quadrant)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = Path(paths[0]).parent / "ref_vs_ps1.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    data_root = None
    if "--data-root" in args:
        i = args.index("--data-root")
        data_root = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if not args:
        sys.exit("Usage: python plot_ref_vs_ps1.py [--data-root DIR] <a_ps1match.csv> ...")
    main([Path(a) for a in args], data_root=data_root)
