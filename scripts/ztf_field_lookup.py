"""
ztf_field_lookup.py
-------------------
Given a target RA/Dec, find all ZTF field/CCD/quadrant combinations covering
it and retrieve full epoch metadata from IRSA, using ztfquery.

ztfquery's load_metadata() does both the geometric lookup and the IRSA epoch
query in a single call — returning field, ccdid, qid, filtercode, filefracday,
seeing, maglim, infobits, etc. for every matching epoch.

Credentials
-----------
IRSA credentials are required. Add them to ~/.netrc:

    machine irsa.ipac.caltech.edu
    login your_username
    password your_password

Set file permissions: chmod 600 ~/.netrc

Usage
-----
    from ztf_field_lookup import lookup_target

    epochs = lookup_target(ra=0.2032, dec=-7.1532, bands=['g','r','i'])

    # CLI
    python scripts/ztf_field_lookup.py --ra 0.203234 --dec -7.153223 --plot coverage.png
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BAND_TO_FILTERCODE = {"g": "zg", "r": "zr", "i": "zi"}
FILTERCODE_TO_BAND = {v: k for k, v in BAND_TO_FILTERCODE.items()}

DEFAULT_CACHE_DIR   = Path("data") / "Epochs"
DEFAULT_SEARCH_DEG  = 0.0005

_HARD_REJECT = (1 << 0) | (1 << 1) | (1 << 25)
_CAUTIONARY  = (
    (1 << 2)  | (1 << 3)  | (1 << 4)  | (1 << 5)  | (1 << 6)
  | (1 << 11) | (1 << 21) | (1 << 22) | (1 << 26) | (1 << 27)
)


# ── Core lookup ───────────────────────────────────────────────────────────────

def lookup_target(
    ra: float,
    dec: float,
    bands: list[str] = None,
    search_radius_deg: float = DEFAULT_SEARCH_DEG,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    plot_out: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Find all ZTF quadrants covering (ra, dec) and retrieve epoch metadata.

    Returns
    -------
    epochs : pd.DataFrame
        One row per observation. Pass directly to download_coordinator.
    """
    if bands is None:
        bands = ["g", "r", "i"]

    invalid = [b for b in bands if b not in BAND_TO_FILTERCODE]
    if invalid:
        raise ValueError(f"Unknown band(s): {invalid}. Choose from: g, r, i")

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_path = _cache_path(ra, dec, bands, Path(cache_dir))

    if not force_refresh:
        epochs = _load_from_cache(cache_path)
        if epochs is not None:
            logger.info(f"Loaded {len(epochs)} epochs from cache")
            if plot_out:
                plot_coverage(epochs, out_path=plot_out)
            return epochs

    try:
        from ztfquery import query as ztfquery
    except ImportError:
        raise ImportError("ztfquery is required. Install with: pip install ztfquery")

    filtercodes = [BAND_TO_FILTERCODE[b] for b in bands]

    logger.info(f"Querying IRSA for RA={ra:.5f}, Dec={dec:.5f}, "
                f"radius={search_radius_deg:.4f} deg ...")
    zquery = ztfquery.ZTFQuery()
    zquery.load_metadata(radec=[ra, dec], size=search_radius_deg)

    meta = zquery.metatable
    if meta is None or meta.empty:
        logger.warning("ztfquery returned no metadata. Check credentials and coordinates.")
        return pd.DataFrame()

    meta = meta[meta["filtercode"].isin(filtercodes)].copy()
    meta = meta[[
        "field", "ccdid", "qid", "filtercode",
        "filefracday", "obsjd", "exptime",
        "seeing", "airmass", "maglimit", "moonesb", "infobits",
        "pid", "expid",
    ]].copy()
    meta["obsmjd"] = meta["obsjd"] - 2400000.5
    meta["band"]   = meta["filtercode"].map(FILTERCODE_TO_BAND)

    n_quads = meta.groupby(["field", "ccdid", "qid"]).ngroups
    logger.info(f"Found {len(meta)} epochs across {n_quads} quadrant(s)")

    _save_to_cache(meta, cache_path)

    if plot_out:
        plot_coverage(meta, out_path=plot_out)

    return meta


# ── Coverage diagnostic plot ──────────────────────────────────────────────────

def plot_coverage(
    epochs: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> None:
    """
    One figure per band. All quadrants overlaid in different colors.

    Layout per band
    ---------------
    Row 0 : infobits bar chart (compact epoch-count table as text overlay)
            | seeing histogram | MAGLIM histogram
    Row 1 : MJD vs seeing | MJD vs MAGLIM
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    mjd_col = "obsmjd" if "obsmjd" in epochs.columns else "obsjd"

    # ── One color per (field, ccdid, qid) — shared across all band figures ───
    quads = (
        epochs[["field", "ccdid", "qid"]]
        .drop_duplicates()
        .sort_values(["field", "ccdid", "qid"])
        .reset_index(drop=True)
    )
    n_q  = len(quads)
    cmap = plt.cm.tab10
    colors, labels = {}, {}
    for i, r in quads.iterrows():
        key         = (int(r.field), int(r.ccdid), int(r.qid))
        rgba        = cmap(i / max(n_q - 1, 1))
        colors[key] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        labels[key] = f"{int(r.field)}/{int(r.ccdid)}/{int(r.qid)}"

    # ── Compact summary table text (all bands, used in every figure) ─────────
    # Header + one row per quadrant, monospaced alignment
    tbl_data = quads.copy()
    for b in ["g", "r", "i"]:
        fc     = BAND_TO_FILTERCODE[b]
        counts = (
            epochs[epochs["filtercode"] == fc]
            .groupby(["field", "ccdid", "qid"]).size().rename(f"n_{b}")
        )
        tbl_data = tbl_data.join(counts, on=["field", "ccdid", "qid"])
    tbl_data = tbl_data.fillna(0).reset_index(drop=True)

    hdr  = "field/ccd/qid    ng   nr   ni"
    rows = []
    for _, r in tbl_data.iterrows():
        tag  = f"{int(r.field)}/{int(r.ccdid)}/{int(r.qid)}"
        rows.append(f"{tag:<16} {int(r.get('n_g',0)):>4} {int(r.get('n_r',0)):>4} {int(r.get('n_i',0)):>4}")
    table_text = "\n".join([hdr] + rows)

    # ── One figure per band ───────────────────────────────────────────────────
    bands_present = sorted(epochs["band"].dropna().unique(),
                           key=lambda b: ["g", "r", "i"].index(b) if b in ["g","r","i"] else 99)

    for band in bands_present:
        band_ep = epochs[epochs["band"] == band]
        if band_ep.empty:
            continue

        fig = plt.figure(figsize=(15, 10))
        gs  = gridspec.GridSpec(2, 6, figure=fig,
                                height_ratios=[2.2, 2.2],
                                hspace=0.38, wspace=0.38)
        ax_bits   = fig.add_subplot(gs[0, 0:2])
        ax_see    = fig.add_subplot(gs[0, 2:4])
        ax_mag    = fig.add_subplot(gs[0, 4:6])
        ax_mjdsee = fig.add_subplot(gs[1, 0:3])
        ax_mjdmag = fig.add_subplot(gs[1, 3:6])

        # ── Infobits bar chart ────────────────────────────────────────────────
        quad_keys = list(colors.keys())
        x         = np.arange(len(quad_keys))
        bar_w     = 0.25
        cat_specs = [
            ("clean",        "steelblue", "clean (ib=0)"),
            ("cautionary",   "orange",    "cautionary (PSF/sky/trail)"),
            ("hard_reject",  "crimson",   "hard reject (no calib)"),
        ]
        max_bar = 0
        for ci, (cat, bcol, cat_lbl) in enumerate(cat_specs):
            heights = []
            for key in quad_keys:
                mask = ((band_ep["field"]  == key[0]) &
                        (band_ep["ccdid"] == key[1]) &
                        (band_ep["qid"]   == key[2]))
                ib = pd.to_numeric(band_ep.loc[mask, "infobits"],
                                   errors="coerce").fillna(0).astype(int)
                if cat == "hard_reject":
                    heights.append(int(((ib & _HARD_REJECT) != 0).sum()))
                elif cat == "cautionary":
                    heights.append(int((((ib & _CAUTIONARY) != 0) &
                                        ((ib & _HARD_REJECT) == 0)).sum()))
                else:
                    heights.append(int((ib == 0).sum()))
            ax_bits.bar(x + ci * bar_w, heights, bar_w,
                        label=cat_lbl, color=bcol, alpha=0.8)
            max_bar = max(max_bar, max(heights) if heights else 0)

        ax_bits.set_xticks(x + bar_w)
        ax_bits.set_xticklabels([labels[k] for k in quad_keys],
                                rotation=15, ha="right", fontsize=8)
        ax_bits.set_ylabel("N epochs")
        ax_bits.legend(fontsize=7, loc="upper right")

        # Compact table as text in upper-left of infobits panel
        # Reserve top ~40% for text by boosting y-limit
        ax_bits.set_ylim(0, max_bar * 2.6)
        ax_bits.text(
            0.02, 0.98, table_text,
            transform=ax_bits.transAxes,
            va="top", ha="left",
            fontsize=7, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )

        # ── Seeing histogram ──────────────────────────────────────────────────
        all_see  = band_ep["seeing"].dropna()
        see_bins = np.linspace(np.percentile(all_see, 1), np.percentile(all_see, 99), 30)

        for _, row in quads.iterrows():
            key  = (int(row.field), int(row.ccdid), int(row.qid))
            col  = colors[key]
            mask = ((band_ep["field"]  == row.field) &
                    (band_ep["ccdid"] == row.ccdid) &
                    (band_ep["qid"]   == row.qid))
            sub  = band_ep[mask]
            if sub.empty:
                continue
            see = sub["seeing"].dropna().values
            med = float(np.median(see))
            ax_see.hist(see, bins=see_bins, alpha=0.5, color=col)
            ax_see.axvline(med, color=col, lw=1.8, ls="--",
                           label=f"{labels[key]}  med={med:.2f}\"")

        ax_see.set_xlabel("Seeing (arcsec)")
        ax_see.set_ylabel("N epochs")
        ax_see.legend(fontsize=7)

        # ── MAGLIM histogram ──────────────────────────────────────────────────
        all_mag  = band_ep["maglimit"].dropna()
        mag_bins = np.linspace(np.percentile(all_mag, 1), np.percentile(all_mag, 99), 30)

        for _, row in quads.iterrows():
            key  = (int(row.field), int(row.ccdid), int(row.qid))
            col  = colors[key]
            mask = ((band_ep["field"]  == row.field) &
                    (band_ep["ccdid"] == row.ccdid) &
                    (band_ep["qid"]   == row.qid))
            sub  = band_ep[mask]
            if sub.empty:
                continue
            mag = sub["maglimit"].dropna().values
            med = float(np.median(mag))
            ax_mag.hist(mag, bins=mag_bins, alpha=0.5, color=col)
            ax_mag.axvline(med, color=col, lw=1.8, ls="--",
                           label=f"{labels[key]}  med={med:.1f}")

        ax_mag.set_xlabel("MAGLIM")
        ax_mag.set_ylabel("N epochs")
        ax_mag.legend(fontsize=7)

        # ── MJD vs seeing ─────────────────────────────────────────────────────
        for _, row in quads.iterrows():
            key  = (int(row.field), int(row.ccdid), int(row.qid))
            col  = colors[key]
            mask = ((band_ep["field"]  == row.field) &
                    (band_ep["ccdid"] == row.ccdid) &
                    (band_ep["qid"]   == row.qid))
            sub  = band_ep[mask]
            if sub.empty:
                continue
            ax_mjdsee.scatter(sub[mjd_col].values, sub["seeing"].values,
                              s=6, alpha=0.5, color=col, label=labels[key])
            ax_mjdmag.scatter(sub[mjd_col].values, sub["maglimit"].values,
                              s=6, alpha=0.5, color=col, label=labels[key])

        ax_mjdsee.set_xlabel("MJD")
        ax_mjdsee.set_ylabel("Seeing (arcsec)")
        ax_mjdsee.legend(fontsize=7, markerscale=2)

        ax_mjdmag.set_xlabel("MJD")
        ax_mjdmag.set_ylabel("MAGLIM")
        ax_mjdmag.legend(fontsize=7, markerscale=2)

        fig.suptitle(f"ZTF coverage — {band}-band", fontsize=12, y=1.01)

        if out_path is not None:
            p      = Path(out_path)
            b_path = p.parent / f"{p.stem}_{band}{p.suffix}"
            fig.savefig(str(b_path), dpi=150, bbox_inches="tight")
            logger.info(f"Coverage plot ({band}) → {b_path}")
        else:
            plt.show()
        plt.close(fig)


# ── Caching ───────────────────────────────────────────────────────────────────

def _cache_path(ra: float, dec: float, bands: list[str], cache_dir: Path) -> Path:
    band_str = "-".join(sorted(bands))
    return cache_dir / f"lookup_{ra:.5f}_{dec:.5f}_{band_str}.epochs.parquet"


def _load_from_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        logger.info(f"Loading cached epochs from {path.name}")
        return pd.read_parquet(path)
    return None


def _save_to_cache(epochs: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.to_parquet(path, index=False)
    logger.info(f"Cached epochs → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Find ZTF field/CCD/quadrant coverage for a sky position."
    )
    parser.add_argument("--ra",            type=float, required=True)
    parser.add_argument("--dec",           type=float, required=True)
    parser.add_argument("--bands",         nargs="+",  default=["g", "r", "i"])
    parser.add_argument("--search-radius", type=float, default=DEFAULT_SEARCH_DEG,
                        metavar="DEG",
                        help=f"Search radius passed to ztfquery (default: {DEFAULT_SEARCH_DEG})")
    parser.add_argument("--cache-dir",     type=Path,  default=None)
    parser.add_argument("--refresh",       action="store_true",
                        help="Force re-query even if cached")
    parser.add_argument("--plot",          type=Path,  default=None, metavar="FILE",
                        help="Coverage plot path stem (band appended automatically). "
                             "Default: data/Plots/{ra}_{dec}/coverage.png")
    parser.add_argument("--no-plot",       action="store_true",
                        help="Skip coverage plot entirely")
    parser.add_argument("--out-epochs",    type=Path,  default=None,
                        help="Save full epoch list to this CSV path")
    args = parser.parse_args()

    # Resolve plot output path: default to data/Plots/{ra}_{dec}/coverage.png
    _data_dir = Path("data")
    if args.no_plot:
        plot_out = None
    elif args.plot is not None:
        plot_out = args.plot
    else:
        plot_dir = _data_dir / "Plots" / f"{args.ra:.5f}_{args.dec:+.5f}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_out = plot_dir / "coverage.png"

    epochs = lookup_target(
        ra=args.ra, dec=args.dec, bands=args.bands,
        search_radius_deg=args.search_radius,
        cache_dir=args.cache_dir, force_refresh=args.refresh,
        plot_out=plot_out,
    )

    if epochs.empty:
        print("No coverage found.", file=sys.stderr)
        sys.exit(1)

    if args.out_epochs:
        epochs.to_csv(args.out_epochs, index=False)
        print(f"Epochs → {args.out_epochs}")
