"""
ztfphot_flags.py
----------------
The single quality-flag bitmask carried in every light-curve parquet as FLAGS (Int32),
next to ZTF's own raw per-epoch image bits (INFOBITS_DIF, kept verbatim).

Bits 0-7: per epoch, from the aperture photometry (bit values follow the SExtractor
flag convention: 1 neighbour, 8 truncated, 16 incomplete aperture data)
     1  NEIGHBOUR         another catalog source within the largest aperture radius
     8  EDGE              the primary 4 px aperture extends beyond the image edge   [corrupts]
    16  MASKED            masked / saturated (NaN) pixels in the 4 px aperture      [corrupts]
    32  OUTER_APERTURES   edge or masked pixels only in the larger 6 / 10 px apertures
Bits 8-15: per source, from the reference image
   256  REF_BLEND         reference neighbour / blend (IRSA refsexcat FLAGS != 0)
  1024  NO_REFERENCE      no reference source (injected target): FLUX is difference flux
Bits 11-14: provenance / summaries
  2048  AP_UNKNOWN        aperture flags not measured for this epoch (rebuilt from an older
                          parquet by update_parquet.py)
  4096  CAUTIONARY_EPOCH  the epoch has cautionary ZTF INFOBITS (see INFOBITS_DIF)
  8192  REF_INFOBITS      the quadrant's reference image has INFOBITS set
 16384  NOT_CROSS_CALIBRATED  merged file: this quadrant could not be cross-calibrated
                          to the dominant one (merged with norm_offset = 0)

Only EDGE and MASKED corrupt the difference-image photometry of the primary aperture, so
the default selection rejects just those (DEFAULT_REJECT); everything else is
informational — add bits to `reject` as needed:

    from ztfphot_flags import good, F
    ok    = good(lc)                                          # drops EDGE | MASKED
    tidy  = good(lc, reject=F.DEFAULT_REJECT | F.CAUTIONARY_EPOCH | F.REF_BLEND)

Older parquets (FLAG_AP / FLAG_SE_REF / INFOBITS_REF columns, or none) are handled by
flags_of(), which good() uses automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class F:
    NEIGHBOUR = 1
    EDGE = 8
    MASKED = 16
    OUTER_APERTURES = 32
    REF_BLEND = 256
    NO_REFERENCE = 1024
    AP_UNKNOWN = 2048
    CAUTIONARY_EPOCH = 4096
    REF_INFOBITS = 8192
    NOT_CROSS_CALIBRATED = 16384

    EPOCH_BITS = 0xFF                                  # bits written by the photometry step
    DEFAULT_REJECT = EDGE | MASKED                      # corrupt the 4 px measurement
    CAL_REJECT = EDGE | MASKED | OUTER_APERTURES        # calibrators: all apertures clean


# These epochs have no valid astrometric or photometric calibration and cannot
# be recovered regardless of science case — never worth downloading.
#   Bit  0 (       1): no astrometric solution
#   Bit  1 (       2): no photometric solution
#   Bit 25 (33554432): composite "bad quality" flag
# Cautionary bits — data may still be usable depending on science requirements.
# Skipped only when --skip-flagged is passed.
# Well-documented bits (ZSDS Sec 10.4):
#   Bit  2 (       4): astrometric fit rms above threshold
#   Bit  3 (       8): photometric fit rms above threshold
#   Bit  4 (      16): moon in/near FOV, elevated background
#   Bit  5 (      32): satellite/aircraft trail present
#   Bit  6 (      64): high sky background (non-moon)
# Observed in data but not fully documented (treated as cautionary until confirmed):
#   Bit 11 (    2048): unknown — rare (few epochs), treat cautionary
#   Bit 21 ( 2097152): unknown — very rare, treat cautionary
#   Bit 22 ( 4194304): PSF matching / subtraction quality below threshold
#                      (most common mystery bit — 158 epochs in r-band field 443;
#                       appears WITHOUT bit 25, so ZTF did not consider globally bad)
#   Bit 26 (67108864): reference image quality / normalization issue
#   Bit 27 (134217728): possibly non-standard reference used
# TODO: confirm bits 11, 21, 22, 26, 27 against ZSDS Explanatory Supplement Sec 10.4
HARD_REJECT_INFOBITS = (1 << 0) | (1 << 1) | (1 << 25)                       # = 33554435
CAUTIONARY_INFOBITS = ((1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)   # documented
                       | (1 << 11) | (1 << 21) | (1 << 22) | (1 << 26) | (1 << 27))  # = 207620220


def _int(s, df) -> np.ndarray:
    return (pd.to_numeric(df[s], errors="coerce").fillna(0).astype("int64").to_numpy()
            if s in df else np.zeros(len(df), dtype="int64"))


def compose(df: pd.DataFrame, epoch_flags=None) -> np.ndarray:
    """FLAGS from the components: per-epoch aperture bits (`epoch_flags`, or the
    FLAG_AP column of older files), FLAG_SE_REF, the no-reference marker (MAG_4_REF
    NaN or > 90), INFOBITS_DIF (cautionary bits) and INFOBITS_REF."""
    ep = (np.asarray(epoch_flags, dtype="int64") if epoch_flags is not None
          else _int("FLAG_AP", df))
    f = ep.copy()
    f |= np.where(_int("FLAG_SE_REF", df) != 0, F.REF_BLEND, 0)
    if "MAG_4_REF" in df:
        mref = pd.to_numeric(df["MAG_4_REF"], errors="coerce").to_numpy(float)
        f |= np.where(~np.isfinite(mref) | (mref > 90), F.NO_REFERENCE, 0)
    f |= np.where((_int("INFOBITS_DIF", df) & CAUTIONARY_INFOBITS) != 0, F.CAUTIONARY_EPOCH, 0)
    f |= np.where(_int("INFOBITS_REF", df) != 0, F.REF_INFOBITS, 0)
    return f


def flags_of(df: pd.DataFrame) -> pd.Series:
    """The FLAGS bitmask of a light-curve table — the column if present, otherwise
    composed from the older columns (FLAG_AP missing -> AP_UNKNOWN)."""
    if "FLAGS" in df:
        return pd.to_numeric(df["FLAGS"], errors="coerce").fillna(0).astype("int64")
    f = compose(df)
    if "FLAG_AP" not in df:
        f |= F.AP_UNKNOWN
    else:                                           # NA FLAG_AP: not measured, not clean
        f |= np.where(pd.to_numeric(df["FLAG_AP"], errors="coerce").isna(), F.AP_UNKNOWN, 0)
    return pd.Series(f, index=df.index, dtype="int64")


def good(df: pd.DataFrame, reject: int = F.DEFAULT_REJECT) -> pd.DataFrame:
    """Rows whose FLAGS have none of the `reject` bits set."""
    return df[(flags_of(df) & reject) == 0]



def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for diagnostics (plots, viewer): no ZTF INFOBITS_DIF bits for the epoch and
    none of the DEFAULT_REJECT bits for the source."""
    ib = pd.to_numeric(df["INFOBITS_DIF"], errors="coerce").fillna(0).to_numpy() if "INFOBITS_DIF" in df else 0
    return df[((flags_of(df) & F.DEFAULT_REJECT) == 0).to_numpy() & (np.asarray(ib) == 0)]
