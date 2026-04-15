"""
make_diagnostic_plots.py
------------------------
Thin re-export shim — actual implementations live in:
  plot_residuals.py   — spatial_rms / spatial_IQR
  plot_calibration.py — rms (RMS improvement & corrections)
  plot_dispersion.py  — precision (photometric precision locus)
  plot_lightcurve.py  — lightcurves (target + comparison-star light curves)
"""

from __future__ import annotations

import logging

from plot_residuals   import make_spatial_rms, make_spatial_iqr
from plot_calibration import make_fig2_rms
from plot_dispersion  import make_fig3_precision
from plot_lightcurve  import make_fig4_lightcurves

logger = logging.getLogger(__name__)

__all__ = [
    "make_spatial_rms", "make_spatial_iqr",
    "make_fig2_rms",
    "make_fig3_precision",
    "make_fig4_lightcurves",
]
