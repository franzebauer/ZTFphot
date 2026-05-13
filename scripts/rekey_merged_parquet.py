#!/usr/bin/env python3
"""
rekey_merged_parquet.py — fix object_index collisions in existing merged parquets.

Each quadrant's object_index is assigned independently, so the same integer
can refer to different physical stars in different quadrants. This script
re-keys non-dominant quadrants so all object_index values are globally unique.

The dominant quadrant (most rows) keeps its original indices unchanged.

Usage:
    python rekey_merged_parquet.py <merged.parquet> [<merged.parquet> ...]
    python rekey_merged_parquet.py path/to/merged/*.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def rekey(path: Path) -> None:
    pf   = pq.read_table(path)
    meta = pf.schema.metadata or {}
    df   = pf.to_pandas()

    if 'object_index' not in df.columns or 'field' not in df.columns:
        print(f"SKIP {path.name}: missing required columns")
        return

    quads = df.groupby(['field', 'ccdid', 'qid'])
    n_rows = quads.size()

    # dominant = most rows (same criterion as merge_fields.py)
    dom_key = n_rows.idxmax()

    # Check whether any collision exists before modifying
    collisions = (df.groupby('object_index')['field'].nunique() > 1).sum()
    if collisions == 0:
        print(f"OK     {path.name}: no collisions, skipping")
        return

    print(f"fixing {path.name}: {collisions} colliding object_index values  "
          f"(dominant: field={dom_key[0]} ccd={dom_key[1]} q={dom_key[2]})")

    running_max = int(df.loc[
        (df['field'] == dom_key[0]) &
        (df['ccdid'] == dom_key[1]) &
        (df['qid']   == dom_key[2]),
        'object_index'
    ].max())

    for key, _ in n_rows.items():
        if key == dom_key:
            continue
        mask = (df['field'] == key[0]) & (df['ccdid'] == key[1]) & (df['qid'] == key[2])
        df.loc[mask, 'object_index'] = df.loc[mask, 'object_index'] + running_max + 1
        running_max = int(df.loc[mask, 'object_index'].max())

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**meta, **(table.schema.metadata or {})})
    pq.write_table(table, path)
    print(f"  → written {path.name}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: python rekey_merged_parquet.py <merged.parquet> [...]")
    for arg in sys.argv[1:]:
        for p in sorted(Path('.').glob(arg)) or [Path(arg)]:
            rekey(p)
