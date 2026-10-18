"""
epoch_inventory.py
------------------
Shared helpers for comparing a light-curve parquet with the epochs IRSA holds for its
target: which epochs the parquet contains, the epoch cuts it was built with (recorded in
its metadata, or inferred for older files), and matching parquet epochs to the IRSA
epoch list. Used by update_parquet.py and the LC-level check_missing_epochs.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

QKEY = ["field", "filtercode", "ccdid", "qid"]


def read_parquet_epochs(path: Path) -> tuple[pd.DataFrame, dict]:
    """Unique (quadrant, OBSMJD) present in the parquet (+ per-epoch MAGLIM, SEEING,
    INFOBITS_DIF when available), and the file metadata."""
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cols = ["OBSMJD"] + [c for c in QKEY + ["MAGLIM", "SEEING", "INFOBITS_DIF"] if c in names]
    df = pf.read(columns=cols).to_pandas()
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()
            if not k.startswith(b"pandas")}
    for c in QKEY:                                  # single-quadrant file: identity in metadata
        if c not in df.columns:
            df[c] = meta[c]
    df = df.astype({"field": int, "ccdid": int, "qid": int, "filtercode": str})
    return df.drop_duplicates(QKEY + ["OBSMJD"]).reset_index(drop=True), meta


def target_position(path: Path, meta: dict) -> tuple[float, float]:
    """Target RA/Dec: recorded run settings (ztfphot_run) if present, else the
    {ra}_{dec} tag in the file name or a parent directory (pipeline-tree merged files
    are LightCurves/merged/{ra}_{dec}/{band}/lightcurves_merged.parquet)."""
    import json
    try:
        a = json.loads(meta.get("ztfphot_run") or "null")["args"]
        if a.get("ra") is not None and a.get("dec") is not None:
            return float(a["ra"]), float(a["dec"])
    except (TypeError, ValueError, KeyError):
        pass
    for part in [path.name] + [q.name for q in path.parents]:
        m = re.match(r"(\d+\.\d+)_([+-]?\d+\.\d+)(_|$)", part)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise SystemExit(f"cannot determine the target RA/Dec for {path}")


def resolve_cuts(a, meta: dict) -> tuple[dict, bool]:
    """Epoch cuts for the 'total possible': explicit command-line values win, otherwise
    the download-time settings recorded in the parquet metadata (ztfphot_download,
    written by run_pipeline). Returns (cuts, recorded) — recorded=False for older
    parquets, whose cuts are then inferred after matching (infer_cuts)."""
    import json
    cuts = {"max_seeing": a.max_seeing, "min_maglim": a.min_maglim,
            "skip_cautionary": a.skip_flagged, "mjd_min": None, "mjd_max": None,
            "min_epochs_per_quad": None}
    try:
        rec = json.loads(meta.get("ztfphot_download") or "null")
    except ValueError:
        rec = None
    if not rec:
        return cuts, False
    ra = rec.get("args", {})
    for k in ("max_seeing", "min_maglim", "mjd_min", "mjd_max", "min_epochs_per_quad"):
        if cuts[k] is None:
            cuts[k] = ra.get(k)
    cuts["skip_cautionary"] = cuts["skip_cautionary"] or bool(ra.get("skip_flagged"))
    print(f"  cuts from parquet metadata (download run {rec.get('time_utc')}, "
          f"ZTFphot {str((rec.get('ztfphot') or {}).get('commit'))[:8]}): "
          + (", ".join(f"{k}={v}" for k, v in cuts.items() if v not in (None, False)) or "none"))
    return cuts, True


def infer_cuts(possible: pd.DataFrame) -> dict:
    """Older parquets: the IRSA maglimit/seeing range of the epochs present (IRSA's own
    values — the parquet MAGLIM/SEEING header values are different quantities), rounded
    outward to 0.1: the loosest cuts consistent with what was downloaded."""
    m = possible[possible["in_parquet"]]
    hint = {}
    if len(m) and "maglimit" in m:
        hint["min_maglim"] = float(np.floor(m["maglimit"].min() * 10 + 1e-6) / 10)
    if len(m) and "seeing" in m:
        hint["max_seeing"] = float(np.ceil(m["seeing"].max() * 10 - 1e-6) / 10)
    return hint


def match_epochs(possible: pd.DataFrame, have: pd.DataFrame, tol: float) -> np.ndarray:
    """Boolean per possible epoch: an OBSMJD in the same quadrant within tol days."""
    found = np.zeros(len(possible), dtype=bool)
    for key, grp in possible.groupby(QKEY):
        h = have.loc[(have[QKEY] == pd.Series(key, index=QKEY)).all(axis=1), "OBSMJD"].to_numpy()
        if len(h) == 0:
            continue
        h = np.sort(h)
        m = grp["obsmjd"].to_numpy()
        k = np.clip(np.searchsorted(h, m), 1, len(h) - 1) if len(h) > 1 else np.zeros(len(m), int)
        d = np.minimum(np.abs(h[k] - m), np.abs(h[np.maximum(k - 1, 0)] - m))
        found[grp.index.to_numpy()] = d <= tol
    return found
