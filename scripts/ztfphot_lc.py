"""
ztfphot_lc.py
-------------
read_lc(path): a light-curve parquet (per-quadrant or merged) with the current column
names and units, whichever pipeline version wrote it. Files written before 2026-10-21
are converted on load:

  FLUX_k_TOT_AB, FERR_k_TOT_AB (μJy)        -> FLUX_k_TOT_uJy, FERR_k_TOT_uJy
  FLUX_k_DIF, FERR_k_DIF (counts)           -> FLUX_k_DIF_uJy, FERR_k_DIF_uJy
                                               (x 10^(-0.4 (MAGZP_DIF - 23.9)))
  merged files without flux_norm_applied    -> calibrated fluxes x 10^(-0.4 norm_offset),
                                               the merge correction the magnitudes carry
  no FLAGS column                           -> FLAGS composed by ztfphot_flags.flags_of

This harmonises names and units only. The photometry of such files keeps the
calibration it was made with (reference zero point, 4->6 px aperture correction
APCORR46); update_old_parquet.py recalibrates them.

    from ztfphot_lc import read_lc
    lc = read_lc("X_zg_merged.parquet")
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq


# stored columns the conversions below need
_FLAG_SOURCES = ["FLAG_AP", "FLAG_SE_REF", "MAG_4_REF", "INFOBITS_DIF", "INFOBITS_REF"]


def _needed(names: set, columns: list[str]) -> list[str]:
    """Stored columns to read for the requested (current-name) `columns`."""
    want = set()
    for c in columns:
        if c in names:
            want.add(c)
        for k in ("4", "6", "10"):
            for kind in ("FLUX", "FERR"):
                if c == f"{kind}_{k}_TOT_uJy" and f"{kind}_{k}_TOT_AB" in names:
                    want |= {f"{kind}_{k}_TOT_AB", "norm_offset"}
                if c == f"{kind}_{k}_DIF_uJy" and f"{kind}_{k}_DIF" in names:
                    want |= {f"{kind}_{k}_DIF", "MAGZP_DIF"}
        if c == "FLAGS" and "FLAGS" not in names:
            want |= set(_FLAG_SOURCES)
    return [c for c in want if c in names]


def read_lc(path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a light-curve parquet with current column names/units. `columns` (current
    names) reads only what they need; requested columns the file lacks are omitted."""
    meta = pq.read_schema(path).metadata or {}
    if columns is not None:
        df = pd.read_parquet(path, columns=_needed(set(pq.read_schema(path).names), columns))
    else:
        df = pd.read_parquet(path)
    for k in ("4", "6", "10"):
        for kind in ("FLUX", "FERR"):
            old = f"{kind}_{k}_TOT_AB"
            if old in df.columns and f"{kind}_{k}_TOT_uJy" not in df.columns:
                df = df.rename(columns={old: f"{kind}_{k}_TOT_uJy"})
            old = f"{kind}_{k}_DIF"
            if old in df.columns and f"{kind}_{k}_DIF_uJy" not in df.columns:
                ujy = 10.0 ** (-0.4 * (pd.to_numeric(df["MAGZP_DIF"], errors="coerce") - 23.9))
                df[f"{kind}_{k}_DIF_uJy"] = (pd.to_numeric(df[old], errors="coerce")
                                             * ujy).astype("float32")
                df = df.drop(columns=old)
    if "norm_offset" in df.columns and b"flux_norm_applied" not in meta:
        f = 10.0 ** (-0.4 * pd.to_numeric(df["norm_offset"], errors="coerce").fillna(0.0))
        for c in [c for c in df.columns if c.endswith("_TOT_uJy")]:
            df[c] = (pd.to_numeric(df[c], errors="coerce") * f).astype("float32")
    if "FLAGS" not in df.columns:
        from ztfphot_flags import flags_of
        df["FLAGS"] = flags_of(df).astype("Int32")
    return df[[c for c in columns if c in df.columns]] if columns is not None else df
