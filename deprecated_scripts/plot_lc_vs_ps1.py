#!/usr/bin/env python3
"""
plot_lc_vs_ps1.py — per-quadrant light-curve vs reference vs Pan-STARRS.

For each quadrant in a merged parquet, overlays as a function of magnitude:

    light curve :  median(MAG_4_TOT_AB_org)  -  g_ps1     (clean epochs, n>=5)
    reference   :  (MAG_APER_4px + MAGZP_REF) - g_ps1      (the anchor, flat)

The gap between the two = median(_org) - q_mag = the magnitude-dependent bias
the science/difference-imaging chain adds on top of the (clean) reference
anchor.  The reference curve is known flat vs PS1, so any tilt in the light
curve isolates where and in which quadrant the inter-quadrant trend is born.

Usage:
    python plot_lc_vs_ps1.py <merged.parquet> <a_ps1match.csv> <b_...> <c_...>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u

MAG_COL  = "MAG_4_TOT_AB_org"
MERR_COL = "MERR_4_TOT_AB"
MAG_BRIGHT, MAG_FAINT = 14.0, 21.5
MAX_ERR, MIN_EPOCHS, MATCH = 0.1, 5, 1.0

_SC_RE = re.compile(r"(\d{6})_z[gri]_c(\d{2})_q(\d)")


def _binned(x, y, lo, hi, n=22):
    edges = np.linspace(lo, hi, n + 1)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b)
        if s.sum() >= 5:
            cx.append(0.5 * (a + b)); cy.append(np.median(y[s]))
    return np.array(cx), np.array(cy)


def _sidecar_map(paths):
    out = {}
    for p in paths:
        m = _SC_RE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = pd.read_csv(p)
    return out


def main(parquet, sidecars):
    df = pq.read_table(parquet).to_pandas()
    smap = _sidecar_map(sidecars)

    quads = (df[["field", "ccdid", "qid"]].drop_duplicates()
             .sort_values(["field", "ccdid", "qid"]).reset_index(drop=True))

    fig, axes = plt.subplots(1, len(quads), figsize=(6 * len(quads), 5.5), squeeze=False)

    for col, (_, r) in enumerate(quads.iterrows()):
        key = (int(r.field), int(r.ccdid), int(r.qid))
        ax = axes[0][col]
        sc = smap.get(key)
        if sc is None:
            ax.set_title(f"{key} — no PS1 sidecar"); continue

        # ── per-source clean median light-curve mag ─────────────────────────
        qmask = ((df.field == r.field) & (df.ccdid == r.ccdid) & (df.qid == r.qid))
        sub = df[qmask]
        clean = sub[(sub["INFOBITS_DIF"] == 0) & sub[MAG_COL].notna() &
                    (sub[MAG_COL] > MAG_BRIGHT) & (sub[MAG_COL] < MAG_FAINT) &
                    (sub[MERR_COL].fillna(9) < MAX_ERR)]
        g = clean.groupby("object_index")
        lc = pd.DataFrame({"mag": g[MAG_COL].median(), "n": g[MAG_COL].count(),
                           "ra": g["ALPHAWIN_REF"].first(), "dec": g["DELTAWIN_REF"].first()})
        lc = lc[lc["n"] >= MIN_EPOCHS]

        # ── match light-curve sources and PS1 by position ───────────────────
        scv = sc[sc["g_ps1"].notna() & (sc["sep_arcsec"].fillna(9) < MATCH)]
        cat_lc = SkyCoord(lc["ra"].values * u.deg, lc["dec"].values * u.deg)
        cat_sc = SkyCoord(scv["ALPHAWIN_J2000"].values * u.deg,
                          scv["DELTAWIN_J2000"].values * u.deg)
        idx, sep, _ = cat_lc.match_to_catalog_sky(cat_sc)
        ok = sep.arcsec < MATCH

        gps1  = scv["g_ps1"].values[idx][ok]
        qmag  = (scv["MAG_APER_4px"].values + scv["MAGZP_REF"].values)[idx][ok]
        lcmag = lc["mag"].values[ok]

        res_lc  = lcmag - gps1
        res_ref = qmag  - gps1
        base = float(np.median(res_ref))

        ax.scatter(gps1, res_lc, s=5, alpha=0.25, color="steelblue", rasterized=True)
        bx, by = _binned(gps1, res_lc, MAG_BRIGHT, MAG_FAINT)
        ax.plot(bx, by, color="orange", lw=2.2, label="light curve (median _org)")
        rx, ry = _binned(gps1, res_ref, MAG_BRIGHT, MAG_FAINT)
        ax.plot(rx, ry, color="crimson", lw=2.2, label="reference (q_mag)")
        ax.axhline(base, color="black", lw=1, ls="--")
        ax.set_xlabel("g_ps1 (mag)")
        ax.set_ylabel("mag - g_ps1")
        ax.set_ylim(base - 0.4, base + 0.4)
        ax.set_title(f"{key[0]}/c{key[1]:02d}/q{key[2]}   (n={ok.sum()})")
        ax.legend(fontsize=8)

        br = (gps1 >= 14) & (gps1 < 17)
        ft = (gps1 >= 19) & (gps1 < 21)
        sb = np.polyfit(gps1[br], res_lc[br], 1)[0] if br.sum() >= 10 else np.nan
        print(f"  {key}: n={int(ok.sum())}  "
              f"LC bright-slope[14-17]={sb:+.3f}  "
              f"LC-ref gap bright[14-16]={np.median((res_lc-res_ref)[(gps1>=14)&(gps1<16)]):+.3f}  "
              f"faint[19-21]={np.median((res_lc-res_ref)[ft]):+.3f}")

    fig.suptitle("Light curve vs reference vs PS1 (per quadrant)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(parquet).parent / "lc_vs_ps1.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python plot_lc_vs_ps1.py <merged.parquet> <ps1match.csv> ...")
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
