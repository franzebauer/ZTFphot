"""
replot_merged.py
----------------
Regenerate per-quadrant precision and light curve plots from a merged parquet.
Useful when the working directory has been deleted (e.g. after batch_pipeline cleanup)
but the merged parquet is still available.

Only precision and lightcurve plots are possible — rms/spatial plots require
calibrated FITS and residual NPZ files that are no longer available.

Usage:
    python replot_merged.py lightcurves_merged.parquet --ra RA --dec DEC --out-dir plots/
"""

from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path, help="Merged parquet file")
    ap.add_argument("--ra",      type=float, required=True)
    ap.add_argument("--dec",     type=float, required=True)
    ap.add_argument("--out-dir", type=Path,  required=True)
    ap.add_argument("--scripts", type=Path,  default=Path(__file__).parent,
                    help="Path to ZTFphot scripts directory")
    args = ap.parse_args()

    if str(args.scripts) not in sys.path:
        sys.path.insert(0, str(args.scripts))

    from plot_diagnostics import make_precision, make_lightcurves

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)

    quadrants = (df[["field", "filtercode", "ccdid", "qid"]]
                 .drop_duplicates()
                 .sort_values(["field", "filtercode", "ccdid", "qid"])
                 .itertuples(index=False))

    with tempfile.TemporaryDirectory() as tmpdir:
        for row in quadrants:
            f, fc, ccd, qid_ = int(row.field), row.filtercode, int(row.ccdid), int(row.qid)
            tag = f"{f:06d}_{fc}_c{ccd:02d}_q{qid_}"

            subset = df[(df["field"] == f) & (df["filtercode"] == fc) &
                        (df["ccdid"] == ccd) & (df["qid"] == qid_)].copy()

            tmp_pq = Path(tmpdir) / f"{tag}.parquet"
            pq.write_table(pa.Table.from_pandas(subset, preserve_index=False), tmp_pq)

            try:
                make_precision(tmp_pq,
                               args.out_dir / f"precision_{tag}.png",
                               tag, args.ra, args.dec)
                print(f"  precision_{tag}.png")
            except Exception as e:
                print(f"  WARNING: precision failed for {tag}: {e}")

            try:
                make_lightcurves(tmp_pq,
                                 args.out_dir / f"lightcurves_{tag}.png",
                                 args.ra, args.dec, tag=tag)
                print(f"  lightcurves_{tag}.png")
            except Exception as e:
                print(f"  WARNING: lightcurves failed for {tag}: {e}")

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
