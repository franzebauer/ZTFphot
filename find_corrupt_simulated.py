"""Find simulated FITS files that are likely corrupt (truncated from disk-full)."""
import os
from pathlib import Path

base = Path('../J1717/Science')
corrupt = []
for f in base.rglob('*_simulated.fits'):
    size = f.stat().st_size
    if size < 5_000_000:   # a valid ZTF quadrant simulated FITS is ~35 MB; <5 MB = truncated
        corrupt.append((size, f))

corrupt.sort()
print(f"Found {len(corrupt)} suspect files:")
for size, f in corrupt:
    print(f"  {size:>8} bytes  {f}")
