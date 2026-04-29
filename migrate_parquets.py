"""
Migrate merged parquet files from the old schema to the new schema.

Old schema extras: mag_calib, mag_calib_err, is_dominant, quadrant_id
New schema: norm_offset applied directly to all MAG_* columns;
            dominant_quadrant stored as file-level metadata key.

Per-quadrant parquets are not affected and are skipped.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_MAG_COLS = ['MAG_3_TOT_AB', 'MAG_4_TOT_AB', 'MAG_6_TOT_AB', 'MAG_10_TOT_AB', 'MAG_4_TOT_AB_org']
_OLD_COLS = {'mag_calib', 'mag_calib_err', 'is_dominant', 'quadrant_id'}

args = [a for a in sys.argv[1:] if not a.startswith('--')]
dry_run = '--dry-run' in sys.argv
targets = [Path(a) for a in args] if args else [Path('../J1717'), Path('../J1025')]

merged_files = []
for t in targets:
    if t.is_file():
        merged_files.append(t)
    elif t.is_dir():
        merged_files.extend(sorted(t.rglob('lightcurves_merged.parquet')))
    else:
        print(f"WARNING: {t} does not exist — skipping")

print(f"Found {len(merged_files)} merged parquet file(s)")

n_migrated = n_skip = 0
for path in merged_files:
    pf = pq.read_table(path)
    cols = set(pf.schema.names)
    meta = pf.schema.metadata or {}

    if not (_OLD_COLS & cols):
        print(f"  SKIP (already new schema): {path}")
        n_skip += 1
        continue

    print(f"  MIGRATE: {path}")
    df = pf.to_pandas()

    # Derive dominant_quadrant from is_dominant before dropping it
    dom_tag = None
    if 'is_dominant' in df.columns and 'quadrant_id' in df.columns:
        dom_rows = df[df['is_dominant'] == True]
        if not dom_rows.empty:
            dom_tag = dom_rows['quadrant_id'].iloc[0]
    elif 'is_dominant' in df.columns:
        # quadrant_id already absent — build tag from identity columns
        dom_rows = df[df['is_dominant'] == True]
        if not dom_rows.empty:
            r = dom_rows.iloc[0]
            dom_tag = f"{int(r['field']):06d}_{r['filtercode']}_c{int(r['ccdid']):02d}_q{int(r['qid'])}"

    # Apply norm_offset to all MAG_* columns (old format left them unmodified)
    if 'norm_offset' in df.columns:
        for col in _MAG_COLS:
            if col in df.columns:
                df[col] = (pd.to_numeric(df[col], errors='coerce')
                           + df['norm_offset'].fillna(0).astype('float64')).astype('float32')

    # Drop old columns
    drop = [c for c in _OLD_COLS if c in df.columns]
    df = df.drop(columns=drop)
    print(f"    dropped columns: {drop}")
    if dom_tag:
        print(f"    dominant_quadrant = {dom_tag}")

    if dry_run:
        print(f"    (dry-run — not writing)")
        continue

    table = pa.Table.from_pandas(df, preserve_index=False)
    new_meta = {**meta, **(table.schema.metadata or {})}
    if dom_tag:
        new_meta[b'dominant_quadrant'] = dom_tag.encode()
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, path)
    print(f"    written.")
    n_migrated += 1

print(f"\nDone: {n_migrated} migrated, {n_skip} already up to date.")
