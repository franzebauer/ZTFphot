"""Compare columns in sexcat.fits (science) vs refsexcat.fits (reference)."""
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

base = Path('J1025')

sci_cat  = next(base.rglob('*_sexcat.fits'), None)
ref_cat  = next(base.rglob('*_refsexcat.fits'), None)

for label, path in [('SCIENCE sexcat', sci_cat), ('REFERENCE refsexcat', ref_cat)]:
    if path is None:
        print(f"{label}: NOT FOUND"); continue
    with fits.open(path) as hdul:
        print(f"\n{label}: {path.name}")
        print(f"  HDUs: {len(hdul)}")
        for i, hdu in enumerate(hdul):
            print(f"  [{i}] {hdu.name}  shape={getattr(hdu.data, 'shape', None)}")
        # find the first binary table HDU
        data_hdu = next((h for h in hdul if hasattr(h, 'columns') and h.columns), None)
        if data_hdu is None:
            print("  No table HDU found"); continue
        t = Table(data_hdu.data)
    print(f"  Rows: {len(t)}")
    print(f"  Columns: {list(t.colnames)}")
    import numpy as np
    for col in ['ALPHAWIN_J2000', 'DELTAWIN_J2000', 'FLUX_BEST', 'FLAGS']:
        if col in t.colnames:
            vals = t[col].data
            print(f"  {col}: min={np.nanmin(vals):.3f}  max={np.nanmax(vals):.3f}")
