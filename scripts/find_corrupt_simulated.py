"""Find simulated FITS files that are corrupt (truncated from disk-full).
Validates by actually opening each file rather than using a size heuristic."""
from pathlib import Path
from astropy.io import fits

base = Path('../J1717/Science')
corrupt = []
checked = 0
for f in sorted(base.rglob('*_simulated.fits')):
    checked += 1
    try:
        with fits.open(f, memmap=False) as hdul:
            _ = hdul[0].data  # force read; truncated files raise here
    except Exception as e:
        corrupt.append((f.stat().st_size, f, str(e)))

print(f"Checked {checked} files. Found {len(corrupt)} corrupt:")
for size, f, err in sorted(corrupt):
    print(f"  {size:>10} bytes  {f}")
    print(f"               {err[:80]}")
