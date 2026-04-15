"""
plot_diagnostics.py
-------------------
Thin re-export shim — actual implementations live in:
  plot_residuals.py   — spatial_rms / spatial_IQR
  plot_calibration.py — make_rms
  plot_precision.py   — make_precision (photometric precision locus)
  plot_lightcurve.py  — make_lightcurves (target + comparison-star light curves)
"""

from __future__ import annotations

import logging

from plot_residuals   import make_spatial_rms, make_spatial_iqr
from plot_calibration import make_rms
from plot_precision   import make_precision
from plot_lightcurve  import make_lightcurves

logger = logging.getLogger(__name__)

__all__ = [
    "make_spatial_rms", "make_spatial_iqr",
    "make_rms",
    "make_precision",
    "make_lightcurves",
]
