"""Print columns and dtypes of per-quadrant and merged light curve parquet files."""
import pyarrow.parquet as pq

for label, path in [
    ("PER-QUADRANT", "/Users/franzbauer/Desktop/LC/J1025/LightCurves/000521/zg/ccd15/q3/lightcurves.parquet"),
    ("MERGED",       "/Users/franzbauer/Desktop/LC/J1025/LightCurves/merged/156.37621_+14.03539/zg/lightcurves_merged.parquet"),
]:
    f = pq.read_table(path)
    print(f"\n{'='*60}\n{label}: {path.split('LightCurves/')[-1]}")
    print("Columns and dtypes:")
    for name, dtype in zip(f.schema.names, f.schema.types):
        print(f"  {name:<30} {dtype}")
    print("File-level metadata:")
    for k, v in (f.schema.metadata or {}).items():
        print(f"  {k.decode():<40} {v.decode()}")
