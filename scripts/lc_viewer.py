#!/usr/bin/env python3
"""
lc_viewer.py — interactive light curve viewer for merged parquet files.

Usage:
    python lc_viewer.py <merged_parquet.parquet>

Controls:
    - Click any point in the overview scatter (left) to load that object
    - Type an object index in the text box and press Enter
    - Prev / Next buttons step through objects ranked by RMS (highest first)
    - Checkboxes toggle individual field/ccdid/qid combinations on/off
    - Light curve points are coloured by MAGLIM (plasma colourmap)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as mw
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm

# ── Load ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    sys.exit("Usage: python lc_viewer.py <merged_parquet.parquet>")

df = pd.read_parquet(sys.argv[1])
df = df.dropna(subset=['MAG_4_TOT_AB', 'OBSMJD']).copy()

# Quadrant label for each row
df['_qlabel'] = (
    df['field'].astype(int).map(lambda x: f'{x:06d}') + '/' +
    df['filtercode'] + '/c' +
    df['ccdid'].astype(int).map(lambda x: f'{x:02d}') + '/q' +
    df['qid'].astype(int).astype(str)
)
all_quads = sorted(df['_qlabel'].unique())

# Per-object stats for overview
def _stats(g):
    mag = g['MAG_4_TOT_AB']
    if len(mag) < 5:
        return None
    return pd.Series({
        'med_mag': float(np.median(mag)),
        'rms':     float(np.std(mag)),
        'n':       len(mag),
        'ra':      float(g['ALPHAWIN_REF'].iloc[0]),
        'dec':     float(g['DELTAWIN_REF'].iloc[0]),
    })

stats = (df.groupby('object_index', group_keys=False)
           .apply(_stats, include_groups=False)
           .dropna())
stats = stats[stats['n'] >= 5].sort_values('rms', ascending=False)
ranked = stats.index.tolist()   # ordered by RMS descending

# ── Figure ────────────────────────────────────────────────────────────────────

BG    = '#12121f'
AX_BG = '#0a0a17'
FG    = '#ccd6f6'
ACC   = '#ff6e40'
GRID  = '#1e1e3a'

fig = plt.figure(figsize=(17, 8.5), facecolor=BG)
fig.suptitle(Path(sys.argv[1]).name, color=FG, fontsize=10, x=0.5, y=0.99)

n_q   = len(all_quads)
chk_h = min(0.28, n_q * 0.055 + 0.04)

# Fixed axes positions [left, bottom, width, height]
ax_ov  = fig.add_axes([0.04, 0.38, 0.27, 0.56])   # overview scatter
ax_lc  = fig.add_axes([0.37, 0.38, 0.56, 0.56])   # light curve
ax_cbar= fig.add_axes([0.945, 0.38, 0.013, 0.56]) # MAGLIM colourbar
ax_txt = fig.add_axes([0.04, 0.22, 0.12, 0.055])  # index text box
ax_prv = fig.add_axes([0.04, 0.10, 0.055, 0.055]) # prev button
ax_nxt = fig.add_axes([0.105, 0.10, 0.055, 0.055])# next button
ax_chk = fig.add_axes([0.20, 0.02, 0.15, chk_h])  # quad checkboxes
ax_inf = fig.add_axes([0.37, 0.24, 0.57, 0.08])   # info text

for ax in [ax_ov, ax_lc, ax_inf]:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)

ax_chk.set_facecolor('#101025')
ax_inf.axis('off')

# ── Overview scatter ──────────────────────────────────────────────────────────

ov_sc = ax_ov.scatter(
    stats['med_mag'], stats['rms'],
    s=5, alpha=0.45, color='#5577cc', picker=6, zorder=2,
)
sel_ring, = ax_ov.plot([], [], 'o', ms=11, mfc='none', mec=ACC, mew=2, zorder=3)
ax_ov.set_xlabel('Median MAG_4', fontsize=8)
ax_ov.set_ylabel('RMS (mag)', fontsize=8)
ax_ov.set_title('overview  (click to select)', fontsize=8, color=FG)
ax_ov.grid(True, color=GRID, lw=0.5)

# ── MAGLIM colorbar ───────────────────────────────────────────────────────────

ml_lo = float(df['MAGLIM'].quantile(0.02))
ml_hi = float(df['MAGLIM'].quantile(0.98))
ml_norm = Normalize(vmin=ml_lo, vmax=ml_hi)
ml_cmap = cm.plasma

cb = ColorbarBase(ax_cbar, cmap=ml_cmap, norm=ml_norm, orientation='vertical')
cb.set_label('MAGLIM', color=FG, fontsize=8)
ax_cbar.tick_params(colors=FG, labelsize=7)

# ── Widgets ───────────────────────────────────────────────────────────────────

txt = mw.TextBox(ax_txt, 'index: ', initial='', color='#181830',
                 hovercolor='#1e1e40')
txt.label.set_color(FG)
txt.text_disp.set_color(FG)

btn_prv = mw.Button(ax_prv, '◀ Prev', color='#1a1a35', hovercolor='#2a2a55')
btn_nxt = mw.Button(ax_nxt, 'Next ▶', color='#1a1a35', hovercolor='#2a2a55')
for b in [btn_prv, btn_nxt]:
    b.label.set_color(FG)
    b.label.set_fontsize(8)

chk = mw.CheckButtons(ax_chk, all_quads, [True] * n_q)
_rects = getattr(chk, 'rectangles', None) or getattr(chk, 'patches', [])
for rect in _rects:
    rect.set_facecolor('#1a1a35')
    rect.set_edgecolor('#4455aa')
for lbl in chk.labels:
    lbl.set_color(FG)
    lbl.set_fontsize(7)

info = ax_inf.text(0.01, 0.5, '', transform=ax_inf.transAxes,
                   color='#88aadd', fontsize=8, va='center',
                   fontfamily='monospace')

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    'idx':         None,
    'rank':        0,
    'active_quads': set(all_quads),
}

# ── Update ────────────────────────────────────────────────────────────────────

def update(obj_idx):
    obj_idx = int(obj_idx)
    state['idx'] = obj_idx

    ax_lc.cla()
    ax_lc.set_facecolor(AX_BG)
    ax_lc.tick_params(colors=FG, labelsize=8)
    for sp in ax_lc.spines.values():
        sp.set_edgecolor(GRID)
    ax_lc.xaxis.label.set_color(FG)
    ax_lc.yaxis.label.set_color(FG)
    ax_lc.grid(True, color=GRID, lw=0.5, zorder=0)

    sub = df[(df['object_index'] == obj_idx) &
             (df['_qlabel'].isin(state['active_quads']))].sort_values('OBSMJD')

    if sub.empty:
        ax_lc.text(0.5, 0.5, 'No data for selected quadrants',
                   transform=ax_lc.transAxes, color=FG, ha='center', va='center')
    else:
        colors = ml_cmap(ml_norm(sub['MAGLIM'].values))
        ax_lc.errorbar(
            sub['OBSMJD'], sub['MAG_4_TOT_AB'],
            yerr=sub['MERR_4_TOT_AB'],
            fmt='none', ecolor='#334455', alpha=0.5, zorder=2,
        )
        ax_lc.scatter(
            sub['OBSMJD'], sub['MAG_4_TOT_AB'],
            c=colors, s=18, alpha=0.85, zorder=3,
        )
        ax_lc.invert_yaxis()
        ax_lc.set_xlabel('MJD', fontsize=8)
        ax_lc.set_ylabel('MAG_4_TOT_AB', fontsize=8)

        ra  = float(sub['ALPHAWIN_REF'].iloc[0])
        dec = float(sub['DELTAWIN_REF'].iloc[0])
        med = float(sub['MAG_4_TOT_AB'].median())
        rms = float(sub['MAG_4_TOT_AB'].std())
        ax_lc.set_title(
            f'object_index = {obj_idx}      RA = {ra:.5f}   Dec = {dec:.5f}',
            fontsize=9, color=FG,
        )

        rank = ranked.index(obj_idx) if obj_idx in ranked else -1
        state['rank'] = rank

        # Per-quadrant breakdown for info line
        quad_info = []
        for ql in sorted(sub['_qlabel'].unique()):
            n = (sub['_qlabel'] == ql).sum()
            quad_info.append(f'{ql} ({n}ep)')

        info.set_text(
            f"mag={med:.3f}  rms={rms:.3f}  N={len(sub)}  "
            f"rank #{rank+1}/{len(ranked)}\n"
            + '  '.join(quad_info)
        )

    # Highlight in overview
    if obj_idx in stats.index:
        sel_ring.set_data([stats.loc[obj_idx, 'med_mag']],
                          [stats.loc[obj_idx, 'rms']])

    fig.canvas.draw_idle()


def on_submit(text):
    try:
        update(int(text.strip()))
    except (ValueError, TypeError):
        pass

def on_prev(_):
    r = max(0, state['rank'] - 1)
    state['rank'] = r
    update(ranked[r])

def on_next(_):
    r = min(len(ranked) - 1, state['rank'] + 1)
    state['rank'] = r
    update(ranked[r])

def on_pick(event):
    if event.artist is not ov_sc:
        return
    ind = event.ind[0]
    update(int(stats.iloc[ind].name))

def on_quad(label):
    if label in state['active_quads']:
        state['active_quads'].discard(label)
    else:
        state['active_quads'].add(label)
    if state['idx'] is not None:
        update(state['idx'])

txt.on_submit(on_submit)
btn_prv.on_clicked(on_prev)
btn_nxt.on_clicked(on_next)
fig.canvas.mpl_connect('pick_event', on_pick)
chk.on_clicked(on_quad)

# Start on highest-RMS object
update(ranked[0])

plt.show()
