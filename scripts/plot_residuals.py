"""
plot_residuals.py
-----------------
Two paired spatial diagnostic plots, sharing the same 8-panel layout:

  spatial_rms_*.png — median residual per spatial bin (systematic offsets)
  spatial_IQR_*.png — IQR (75th−25th percentile) per bin (epoch-to-epoch scatter)

Panel order (2 rows × 4 cols):
  1. Before (all sources)
  2. Before (calibrators)
  3. After linear ZP (calibrators)
  4. After 3σ clip (calibrators)
  5. After faint corr (calibrators)
  6. After 2D poly (calibrators)
  7. After flatfield (calibrators)
  8. After calibration (all sources)

Each panel has an independent colorbar.
Per-panel title includes: σ_sample (std of stacked residuals),
σ_map (std of binned grid), N (total point-epoch pairs used).

Reads *_resid.npz from FlatfieldResiduals/.  NPZ arrays (all in magnitudes):
  ra_0/dec_0/dm_0       calibrators, before any correction
  ra_1/dec_1/dm_1       calibrators, after linear ZP fit
  ra_2/dec_2/dm_2       calibrators, after 3σ iterative clip
  ra_3/dec_3/dm_3       calibrators, after faint-source correction
  ra_4/dec_4/dm_4       calibrators, after 2D polynomial
  ra_5/dec_5/dm_5       calibrators, after spatial flatfield
  ra_all/dec_all/dm_all_pre   all matched sources, before calibration
  ra_all/dec_all/dm_all_post  all matched sources, after full calibration
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Panel definitions: (ra_key, dec_key, dm_key, label)
_STAGES = [
    ("ra_all", "dec_all", "dm_all_pre",  "Before (all sources)"),
    ("ra_0",   "dec_0",   "dm_0",        "Before (calibrators)"),
    ("ra_1",   "dec_1",   "dm_1",        "After linear ZP (calibrators)"),
    ("ra_2",   "dec_2",   "dm_2",        "After 3σ clip (calibrators)"),
    ("ra_3",   "dec_3",   "dm_3",        "After faint corr (calibrators)"),
    ("ra_4",   "dec_4",   "dm_4",        "After 2D poly (calibrators)"),
    ("ra_5",   "dec_5",   "dm_5",        "After flatfield (calibrators)"),
    ("ra_all", "dec_all", "dm_all_post", "After calibration (all sources)"),
]


def _load_resid_npz(resid_dir: Path) -> list[dict]:
    epochs = []
    for p in sorted(resid_dir.glob("*_resid.npz")):
        try:
            d = np.load(str(p))
            epochs.append({k: d[k] for k in d.files})
        except Exception:
            pass
    return epochs


def _stack_stage(epochs: list[dict], rk: str, dk: str, mk: str):
    """Stack arrays for one stage across all epochs. Returns (ra, dec, dm) in mmag."""
    ras, decs, dms = [], [], []
    for e in epochs:
        if rk in e and mk in e:
            ras.append(e[rk])
            decs.append(e[dk])
            dms.append(e[mk] * 1000)   # mag → mmag
    if not ras:
        return None, None, None
    return np.concatenate(ras), np.concatenate(decs), np.concatenate(dms)


def _bin_grid(ra, dec, dm, nbins, stat_fn):
    """
    Bin dm values into a (nbins × nbins) RA/Dec grid applying stat_fn per cell.
    Returns (grid, ra_edges, dec_edges).
    """
    ra_edges  = np.linspace(ra.min(),  ra.max(),  nbins + 1)
    dec_edges = np.linspace(dec.min(), dec.max(), nbins + 1)
    grid = np.full((nbins, nbins), np.nan)
    ri = np.clip(np.digitize(ra,  ra_edges)  - 1, 0, nbins - 1)
    di = np.clip(np.digitize(dec, dec_edges) - 1, 0, nbins - 1)
    for i in range(nbins):
        for j in range(nbins):
            sel = dm[(ri == i) & (di == j)]
            sel = sel[np.isfinite(sel)]
            if len(sel) >= 3:
                grid[j, i] = stat_fn(sel)
    return grid, ra_edges, dec_edges


def _panel(ax, ra, dec, dm, label, nbins, stat_fn, cmap, symmetric):
    """
    Plot one spatial panel. Returns (im, subtitle_str) — caller sets the title.
    """
    grid, ra_edges, dec_edges = _bin_grid(ra, dec, dm, nbins, stat_fn)
    finite = grid[np.isfinite(grid)]

    dm_fin   = dm[np.isfinite(dm)]
    n_pts    = len(dm_fin)
    sig_samp = float(np.std(dm_fin))  if len(dm_fin) > 1 else np.nan
    sig_map  = float(np.std(finite))  if len(finite)  > 1 else np.nan

    if symmetric:
        vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 95))) if len(finite) else 20.0
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = max(1.0, float(np.nanpercentile(finite, 95))) if len(finite) else 30.0

    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[ra_edges[0], ra_edges[-1], dec_edges[0], dec_edges[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax)

    sig_samp_str = f"{sig_samp:.1f}" if np.isfinite(sig_samp) else "—"
    sig_map_str  = f"{sig_map:.1f}"  if np.isfinite(sig_map)  else "—"
    subtitle = (
        f"$\\sigma_{{\\mathrm{{sample}}}}={sig_samp_str}\\,\\mathrm{{mmag}}\\quad "
        f"\\sigma_{{\\mathrm{{map}}}}={sig_map_str}\\,\\mathrm{{mmag}}\\quad "
        f"N={n_pts:,}$"
    )

    ax.set_title(f"{label}\n{subtitle}", fontsize=10, pad=3, linespacing=1.4)
    ax.set_xlabel("RA (deg)", fontsize=10)
    ax.set_ylabel("Dec (deg)", fontsize=10)
    ax.tick_params(labelsize=7)
    return im


def _make_spatial_fig(epochs, out_path, tag, nbins, stat_fn, cmap, symmetric, fig_title):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10),
                             gridspec_kw=dict(wspace=0.15, hspace=0.25))
    fig.suptitle(f"{fig_title} — {tag}", fontsize=14, y=0.995)

    for ax, (rk, dk, mk, label) in zip(axes.flatten(), _STAGES):
        ra, dec, dm = _stack_stage(epochs, rk, dk, mk)
        if ra is None or len(ra) == 0:
            ax.set_visible(False)
            continue
        im = _panel(ax, ra, dec, dm, label, nbins, stat_fn, cmap, symmetric)
        cb = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("mmag", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Public API ────────────────────────────────────────────────────────────────

def make_spatial_rms(resid_dir: Path, out_path: Path, tag: str = "",
                     nbins: int = 20) -> None:
    """
    Median residual per spatial bin — shows systematic calibration offsets.
    Diverging colormap (RdBu_r), symmetric about zero, independent per panel.
    """
    epochs = _load_resid_npz(resid_dir)
    if not epochs:
        logger.warning(f"No residual NPZ files in {resid_dir} — skipping spatial_rms")
        return
    _make_spatial_fig(epochs, out_path, tag, nbins,
                      stat_fn=np.median,
                      cmap="RdBu_r",
                      symmetric=True,
                      fig_title="Spatial calibration residuals (median)")
    logger.info(f"  spatial_rms → {out_path}")


def make_spatial_iqr(resid_dir: Path, out_path: Path, tag: str = "",
                     nbins: int = 20) -> None:
    """
    IQR (75th−25th percentile) per spatial bin — shows epoch-to-epoch scatter.
    Sequential colormap (YlOrRd), always positive, independent per panel.
    """
    epochs = _load_resid_npz(resid_dir)
    if not epochs:
        logger.warning(f"No residual NPZ files in {resid_dir} — skipping spatial_IQR")
        return

    def _iqr(arr):
        return float(np.percentile(arr, 75) - np.percentile(arr, 25))

    _make_spatial_fig(epochs, out_path, tag, nbins,
                      stat_fn=_iqr,
                      cmap="YlOrRd",
                      symmetric=False,
                      fig_title="Spatial calibration IQR (75th−25th percentile)")
    logger.info(f"  spatial_IQR → {out_path}")
