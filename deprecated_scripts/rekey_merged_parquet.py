#!/usr/bin/env python3
"""
rekey_merged_parquet.py — fix object_index collisions in existing merged parquets.

Each quadrant's object_index is assigned independently, so the same integer
can refer to different physical stars in different quadrants. This script:
  1. Offsets each secondary quadrant's indices above the dominant quadrant's
     block so all indices are globally unique.
  2. Coordinate-matches secondary sources to the dominant quadrant and reassigns
     matched sources to share the dominant's object_index, so the same physical
     star has one index across quadrants.

The dominant quadrant (most rows) keeps its original indices unchanged.

Usage:
    python rekey_merged_parquet.py <merged.parquet> [<merged.parquet> ...]
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u

MAX_SEP_ARCSEC = 1.5


def rekey(path: Path) -> None:
    pf   = pq.read_table(path)
    meta = pf.schema.metadata or {}
    df   = pf.to_pandas()

    if 'object_index' not in df.columns or 'field' not in df.columns:
        print(f"SKIP {path.name}: missing required columns")
        return

    n_rows  = df.groupby(['field', 'ccdid', 'qid']).size()
    if len(n_rows) < 2:
        print(f"OK     {path.name}: single quadrant, nothing to do")
        return

    dom_key  = n_rows.idxmax()
    dom_mask = ((df['field']  == dom_key[0]) &
                (df['ccdid'] == dom_key[1]) &
                (df['qid']   == dom_key[2]))

    print(f"fixing {path.name}  ({len(n_rows)} quadrants, "
          f"dominant: field={dom_key[0]} ccd={dom_key[1]} q={dom_key[2]})")

    # ── Step 1: offset each secondary quadrant's indices above all previous ───
    running_max = int(df.loc[dom_mask, 'object_index'].max())
    for key in n_rows.index:
        if key == dom_key:
            continue
        mask = ((df['field']  == key[0]) &
                (df['ccdid'] == key[1]) &
                (df['qid']   == key[2]))
        df.loc[mask, 'object_index'] = df.loc[mask, 'object_index'] + running_max + 1
        running_max = int(df.loc[mask, 'object_index'].max())
        print(f"  f{key[0]} c{key[1]} q{key[2]}: indices offset → max now {running_max}")

    # ── Step 2: link overlap sources to dominant object_index ─────────────────
    dom_pos = (df[dom_mask]
               .groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']]
               .first().reset_index())
    cat_dom = SkyCoord(ra=dom_pos['ALPHAWIN_REF'].values * u.deg,
                       dec=dom_pos['DELTAWIN_REF'].values * u.deg)

    for key in n_rows.index:
        if key == dom_key:
            continue
        mask = ((df['field']  == key[0]) &
                (df['ccdid'] == key[1]) &
                (df['qid']   == key[2]))
        min_pos = (df[mask]
                   .groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']]
                   .first().reset_index())
        cat_min = SkyCoord(ra=min_pos['ALPHAWIN_REF'].values * u.deg,
                           dec=min_pos['DELTAWIN_REF'].values * u.deg)
        idx_arr, sep, _ = cat_min.match_to_catalog_sky(cat_dom)
        matched = sep.arcsec < MAX_SEP_ARCSEC

        remap = {
            int(min_pos['object_index'].iloc[j]): int(dom_pos['object_index'].iloc[idx_arr[j]])
            for j in range(len(min_pos)) if matched[j]
        }
        if remap:
            df['object_index'] = (df['object_index']
                                  .map(remap)
                                  .fillna(df['object_index'])
                                  .astype('Int32'))
        print(f"  f{key[0]} c{key[1]} q{key[2]}: {len(remap)} overlap sources linked to dominant")

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**meta, **(table.schema.metadata or {})})
    pq.write_table(table, path)
    print(f"  → written {path.name}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: python rekey_merged_parquet.py <merged.parquet> [...]")
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            rekey(p)
        else:
            for p2 in sorted(Path('.').glob(arg)):
                rekey(p2)
