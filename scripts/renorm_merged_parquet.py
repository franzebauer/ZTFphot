#!/usr/bin/env python3
"""
renorm_merged_parquet.py — recompute inter-quadrant normalization in-place.

Measures the residual offset between each secondary quadrant and the dominant
quadrant from MAG_4_TOT_AB, using the same clean-epoch filter as
plot_quad_offsets.py (INFOBITS_DIF == 0).  The offset is computed as a function
of magnitude (per-magnitude-bin median of dominant - secondary, over common
sources) and applied to every secondary row by interpolating the correction
curve at that row's magnitude.  All MAG columns are corrected together.  Can be
run iteratively.

A very large --mag-bin (covering the whole range in one bin) reduces to the
former single-scalar behaviour.

Usage:
    python renorm_merged_parquet.py [--mag-bin 0.1] <merged.parquet> [...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MIN_EPOCHS = 5
MAG_BRIGHT = 14.0
MAG_FAINT  = 21.5
MAX_ERR    = 0.1
MIN_BIN    = 5      # min common sources required to trust a magnitude bin

_MAG_COLS = ['MAG_3_TOT_AB', 'MAG_4_TOT_AB', 'MAG_6_TOT_AB',
             'MAG_10_TOT_AB', 'MAG_4_TOT_AB_org']


def _per_source_median(sub: pd.DataFrame) -> pd.Series:
    """Per-source median of MAG_4_TOT_AB using the same quality filter
    as plot_quad_offsets.py: INFOBITS_DIF == 0, magnitude in range, error < MAX_ERR.
    """
    mask = (
        (sub['INFOBITS_DIF'] == 0) &
        sub['MAG_4_TOT_AB'].notna() &
        (sub['MAG_4_TOT_AB'] > MAG_BRIGHT) &
        (sub['MAG_4_TOT_AB'] < MAG_FAINT)
    )
    if 'MERR_4_TOT_AB' in sub.columns:
        mask &= sub['MERR_4_TOT_AB'].fillna(999) < MAX_ERR
    clean   = sub[mask]
    counts  = clean.groupby('object_index')['MAG_4_TOT_AB'].count()
    medians = clean.groupby('object_index')['MAG_4_TOT_AB'].median()
    return medians[counts >= MIN_EPOCHS]


def _correction_curve(diff: np.ndarray, mag_axis: np.ndarray, mag_bin: float):
    """Per-magnitude-bin median of `diff` (dominant - secondary) on the
    `mag_axis` (dominant magnitude), with a 3σ MAD clip in each bin.

    Returns (centers, deltas) for bins holding >= MIN_BIN sources, or
    (None, None) if fewer than two bins are populated.
    """
    edges = np.arange(MAG_BRIGHT, MAG_FAINT + mag_bin, mag_bin)
    centers, deltas = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = diff[(mag_axis >= lo) & (mag_axis < hi)]
        if len(d) < MIN_BIN:
            continue
        med  = float(np.median(d))
        mad  = float(np.median(np.abs(d - med)))
        sig  = 1.4826 * mad
        keep = np.abs(d - med) < 3.0 * sig if sig > 0 else np.ones(len(d), bool)
        centers.append(0.5 * (lo + hi))
        deltas.append(float(np.median(d[keep])))
    if len(centers) < 2:
        return None, None
    return np.asarray(centers), np.asarray(deltas)


def renorm(path: Path, mag_bin: float, write: bool = True) -> None:
    pf   = pq.read_table(path)
    meta = pf.schema.metadata or {}
    df   = pf.to_pandas()

    if 'MAG_4_TOT_AB_org' not in df.columns:
        print(f"SKIP {path.name}: no MAG_4_TOT_AB_org column")
        return
    if 'norm_offset' not in df.columns:
        print(f"SKIP {path.name}: no norm_offset column")
        return

    quads = (df[['field', 'ccdid', 'qid']]
             .drop_duplicates()
             .sort_values(['field', 'ccdid', 'qid'])
             .reset_index(drop=True))

    if len(quads) < 2:
        print(f"OK     {path.name}: single quadrant, nothing to do")
        return

    # Dominant = quadrant with norm_offset == 0; fall back to largest
    dom_mask_bool = df['norm_offset'].abs() < 1e-6
    dom_quads = df[dom_mask_bool][['field', 'ccdid', 'qid']].drop_duplicates()
    if len(dom_quads) != 1:
        sizes   = df.groupby(['field', 'ccdid', 'qid']).size()
        dom_key = sizes.idxmax()
        dom_quads = pd.DataFrame([{'field': dom_key[0],
                                   'ccdid': dom_key[1],
                                   'qid':   dom_key[2]}])

    dom_row  = dom_quads.iloc[0]
    dom_mask = ((df['field']  == dom_row['field']) &
                (df['ccdid'] == dom_row['ccdid']) &
                (df['qid']   == dom_row['qid']))

    dom_med = _per_source_median(df[dom_mask])

    print(f"\n{path.name}")
    print(f"  dominant: f{int(dom_row.field)} c{int(dom_row.ccdid)} q{int(dom_row.qid)}"
          f"  ({dom_mask.sum()} rows,  {len(dom_med)} sources with n≥{MIN_EPOCHS})")

    norm_arr = df['norm_offset'].values.astype(np.float64)
    mag_arrs = {col: df[col].values.astype(np.float64)
                for col in _MAG_COLS if col in df.columns}

    for _, row in quads.iterrows():
        if (row['field']  == dom_row['field'] and
            row['ccdid'] == dom_row['ccdid'] and
            row['qid']   == dom_row['qid']):
            continue

        sec_mask = ((df['field']  == row['field']) &
                    (df['ccdid'] == row['ccdid']) &
                    (df['qid']   == row['qid']))
        sec_bool = sec_mask.values

        sec_med = _per_source_median(df[sec_mask])

        common_idx = dom_med.index.intersection(sec_med.index)
        if len(common_idx) < 3:
            print(f"  SKIP f{int(row.field)} c{int(row.ccdid)} q{int(row.qid)}: "
                  f"only {len(common_idx)} common sources")
            continue

        mag_axis = dom_med.loc[common_idx].values
        diff     = (dom_med.loc[common_idx] - sec_med.loc[common_idx]).values

        centers, deltas = _correction_curve(diff, mag_axis, mag_bin)
        sec_mag = mag_arrs['MAG_4_TOT_AB'][sec_bool]

        if centers is None:
            # too few populated bins → single 3σ-clipped scalar
            med  = float(np.median(diff))
            mad  = float(np.median(np.abs(diff - med)))
            sig  = 1.4826 * mad
            keep = np.abs(diff - med) < 3.0 * sig if sig > 0 else np.ones(len(diff), bool)
            scalar = float(np.median(diff[keep]))
            corr_rows = np.full(int(sec_bool.sum()), scalar, dtype=np.float64)
            print(f"  f{int(row.field)} c{int(row.ccdid)} q{int(row.qid)}: "
                  f"scalar fallback  delta={scalar:+.5f}  (n_common={len(common_idx)})")
        else:
            # interpolate the curve at each row's magnitude (flat extrapolation);
            # rows with no magnitude get the curve's median correction
            corr_rows = np.interp(sec_mag, centers, deltas)
            corr_rows[~np.isfinite(sec_mag)] = float(np.median(deltas))
            print(f"  f{int(row.field)} c{int(row.ccdid)} q{int(row.qid)}: "
                  f"per-mag curve over {len(centers)} bins  "
                  f"bright={deltas[0]:+.4f}@{centers[0]:.2f}  "
                  f"faint={deltas[-1]:+.4f}@{centers[-1]:.2f}  "
                  f"median={float(np.median(deltas)):+.4f}  (n_common={len(common_idx)})")

        for arr in mag_arrs.values():
            arr[sec_bool] += corr_rows
        norm_arr[sec_bool] += corr_rows

    for col, arr in mag_arrs.items():
        df[col] = arr.astype(np.float32)
    df['norm_offset'] = norm_arr.astype(np.float32)

    if not write:
        print(f"  (dry-run — {path.name} not modified)")
        return

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**meta, **(table.schema.metadata or {})})
    pq.write_table(table, path)
    print(f"  → written {path.name}")


if __name__ == '__main__':
    args = sys.argv[1:]
    mag_bin = 0.1
    if '--mag-bin' in args:
        i = args.index('--mag-bin')
        mag_bin = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if not args:
        sys.exit("Usage: python renorm_merged_parquet.py [--mag-bin 0.1] <merged.parquet> [...]")
    for arg in args:
        p = Path(arg)
        if p.exists():
            renorm(p, mag_bin)
        else:
            for p2 in sorted(Path('.').glob(arg)):
                renorm(p2, mag_bin)
