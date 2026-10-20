#!/usr/bin/env python3
"""
update_parquet.py
-----------------
Add missing (and, by default, newer) epochs to existing light-curve parquets, and
recalibrate all 4 px photometry consistently with the CURRENT pipeline — without
re-downloading the epochs already in the parquet.

Per target and band (a ref-pos parquet and its `_sci` sibling are updated together
when both exist):

  1. Inventory: quadrants and epochs in the parquet(s); IRSA epoch list for the
     target; epoch cuts from the parquet metadata (ztfphot_download) or, for older
     files, inferred from the epochs present (--infer-cuts) — see epoch_inventory.py.
     Missing = passes the cuts, not in the parquet; new = after the parquet's last
     epoch (skipped with --no-new).
  2. Temporary base dir: reference products + reference catalogs (same object order
     as the original run); download ONLY the missing/new epochs.
  3. Old epochs become per-epoch catalogs again (same LDAC layout the photometry step
     writes) from the parquet: raw 4 px difference flux = FLUX_4_DIF when present,
     else reconstructed exactly from MAG_4_TOT_AB_org
         flux_tot = 10^(0.4 (MAGZP_DIF - APCORR46/1000 - (MAG_4_TOT_AB_org - norm_offset)))
         FLUX_4_DIF = flux_tot - 10^(0.4 (MAGZP_DIF - q_mag_ref))
     (error from MERR_4_TOT_AB_org). The 4 px value fills all four aperture slots;
     only 4 px is used from these epochs.
  4. run_pipeline (current code) on the temp dir: forced photometry of the new epochs,
     then vet -> calibrate -> flatfield -> recalibrate -> lightcurves -> merge over
     old + new epochs together.
  5. Output <stem>_updated.parquet next to the input (or --outdir):
       old epochs : new 4 px values; MAG_3/6/10 (and FLUX_6) shifted by the same
                    4 px change (MAG_k_new = MAG_k_old + dMAG_4); MERR_k kept
       new epochs : full photometry (all apertures)
       old rows that cannot be reconstructed (NaN 4 px magnitude and no FLUX_4_DIF,
       or quadrants the current footprint check drops) are carried over unchanged.
     Metadata: the current pipeline's keys plus ztfphot_update (source, counts).

Usage (ztf env, IRSA credentials in ~/.netrc):
    python update_parquet.py TARGET_zg_merged.parquet [more.parquet ...]
           [--infer-cuts] [--no-new] [--dry-run] [--outdir DIR] [--keep-temp]
    python update_parquet.py RESULTS_DIR --infer-cuts [--plan] [--recursive] [--redo]
      (directory: per target/band/variant uses X_recal.parquet if present, else X.parquet;
       skips targets already updated)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from epoch_inventory import (QKEY, read_parquet_epochs, target_position,  # noqa: E402
                             resolve_cuts, infer_cuts, match_epochs)

logger = logging.getLogger("update_parquet")

_REF_PRODUCTS = ("refimg.fits", "refsexcat.fits")
_HDR_FROM_PARQUET = {            # LDAC header card <- parquet column (per epoch)
    "OBSMJD": "OBSMJD", "AIRMASS": "AIRMASS", "NMATCHES": "NMATCHES",
    "MAGZP": "MAGZP_DIF", "MAGZPRMS": "MAGZPRMS_DIF", "CLRCOEFF": "CLRCOEFF",
    "INFOBITS": "INFOBITS_DIF", "SEEING": "SEEING", "MAGLIM": "MAGLIM",
}


# ── inputs ────────────────────────────────────────────────────────────────────

def _sibling(path: Path) -> Path | None:
    """The other variant (ref-pos <-> sci-pos) of a light-curve parquet, if present.
    Derived names keep their suffix: X_zg_merged_recal <-> X_zg_sci_merged_recal."""
    s, cands = path.stem, []
    tail = next((t for t in ("_recal", "_updated") if s.endswith(t)), "")
    s = s[:len(s) - len(tail)]
    if path.name == "lightcurves_merged.parquet":            # pipeline tree: .../{band}{_sci}/
        b = path.parent.name
        other = b[:-4] if b.endswith("_sci") else b + "_sci"
        cands.append(path.parent.parent / other / path.name)
    elif "_sci" in s:
        cands.append(path.with_name(s.replace("_sci", "") + tail + ".parquet"))
    elif s.endswith("_merged"):
        cands.append(path.with_name(s[:-len("_merged")] + "_sci_merged" + tail + ".parquet"))
    else:
        cands.append(path.with_name(s + "_sci" + tail + ".parquet"))
    return next((c for c in cands if c.exists()), None)


def _is_sci(path: Path) -> bool:
    return ("_sci" in path.stem) or path.parent.name.endswith("_sci")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    meta = {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items()
            if not k.startswith(b"pandas")}
    for c in QKEY:                                  # single-quadrant files: identity in metadata
        if c not in df.columns:
            df[c] = meta[c]
    df = df.astype({"field": int, "ccdid": int, "qid": int, "filtercode": str})
    if "norm_offset" not in df.columns:
        df["norm_offset"] = 0.0
    return df


# ── reference catalogs + source matching ─────────────────────────────────────

def _fetch_references(base: Path, quads, auth, retries: int, wait: float) -> list:
    """refimg + refsexcat per quadrant, retrying transient (503/HTML) failures."""
    import time
    from download_coordinator import ref_url, ref_local_path, download_file
    ok = []
    for q in quads:
        good = True
        for suf in _REF_PRODUCTS:
            dest = ref_local_path(base, q[0], q[1], q[2], q[3], suf)
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(retries + 1):
                _, st, msg = download_file(ref_url(q[0], q[1], q[2], q[3], suf), dest, auth)
                if st in ("ok", "skipped", "404"):
                    break
                logger.warning(f"  reference {suf} for {q}: {msg} — retry {attempt + 1}/{retries}")
                time.sleep(wait * (attempt + 1))
            if st not in ("ok", "skipped"):
                logger.warning(f"  reference {suf} for {q}: giving up ({st})")
                good = False
                break
        if good:
            ok.append(q)
    return ok


def _download_epochs(dc, todo: pd.DataFrame, base: Path, fc: str, a) -> list[str]:
    """Science products of the epochs to add. Transient failures (IRSA 503s, dropped
    connections) get a few SHORT retry rounds (--retry-wait s x round); whatever still
    fails is deferred rather than waited for — the target is processed without those
    epochs and they are recorded in the output (epochs_deferred), for --retry-deferred.
    Returns the deferred filefracdays."""
    import time
    for attempt in range(a.download_retries + 1):
        res = dc.download_all(todo, base_dir=base, bands=[fc[1:]], products="sci",
                              max_workers=a.download_workers)
        if not res["failed_urls"]:
            break
        if attempt < a.download_retries:
            wait = a.retry_wait * (attempt + 1)
            logger.warning(f"  {len(res['failed_urls'])} downloads failed — retry "
                           f"{attempt + 1}/{a.download_retries} in {wait:.0f} s")
            time.sleep(wait)
    # Epochs without a difference image now: an HTTP 404 means IRSA has no such product
    # (never retried — it would never succeed); anything else is transient -> deferred.
    gone = dc.load_permanent_404s(base, max_age_days=0)
    deferred, unavailable = [], []
    for r in todo.itertuples():
        ffd = str(int(float(r.filefracday)))
        diff = dc.sci_local_path(base, int(r.field), r.filtercode, int(r.ccdid), int(r.qid),
                                 ffd, "scimrefdiffimg.fits")
        if diff.exists():
            continue
        url = dc.sci_url(ffd, int(r.field), r.filtercode, int(r.ccdid), int(r.qid),
                         "scimrefdiffimg.fits.fz")
        (unavailable if url in gone else deferred).append(ffd)
    if unavailable:
        logger.info(f"  {len(unavailable)} epoch(s) unavailable on IRSA (HTTP 404) — skipped")
    if deferred:
        logger.warning(f"  {len(deferred)} epoch(s) deferred (download still failing) — "
                       f"processed without them; add later with --retry-deferred")
    return deferred


def _ref_catalog(base: Path, q) -> pd.DataFrame:
    tag = f"{q[0]:06d}_{q[1]}_c{q[2]:02d}_q{q[3]}"
    ref = pd.read_csv(base / "Catalogs" / f"{tag}(REFERENCE)[OBJECTS].csv")
    ref["q_mag"] = (pd.to_numeric(ref["MAG_APER_4px"], errors="coerce")
                    + pd.to_numeric(ref["MAGZP_REF"], errors="coerce"))
    return ref


def _csv_index(ra, dec, ref: pd.DataFrame, target, match_radius, tol=0.5) -> np.ndarray:
    """Reference-catalog row (0-based) for each (ra, dec); the injected target gets
    len(ref) (no reference source within match_radius), as in the photometry step.
    Unmatched -> -1."""
    from astropy.coordinates import SkyCoord
    rc = SkyCoord(ref["ALPHAWIN_J2000"].values, ref["DELTAWIN_J2000"].values, unit="deg")
    out = np.full(len(ra), -1, int)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if ok.any():
        sc = SkyCoord(np.asarray(ra)[ok], np.asarray(dec)[ok], unit="deg")
        j, sep, _ = sc.match_to_catalog_sky(rc)
        out[np.where(ok)[0][sep.arcsec < tol]] = j[sep.arcsec < tol]
        if target is not None:
            t = SkyCoord(target[0], target[1], unit="deg")
            injected = t.separation(rc).arcsec.min() >= match_radius
            near_t = sc.separation(t).arcsec < match_radius
            fill = np.where(ok)[0][near_t & (sep.arcsec >= tol)]
            out[fill] = len(ref) if injected else int(t.match_to_catalog_sky(rc)[0])
    return out


# ── old epochs -> per-epoch catalogs ─────────────────────────────────────────

def _raw_flux(rows: pd.DataFrame, q_mag: np.ndarray, is_target: np.ndarray):
    """Raw 4 px difference flux and error (counts, MAGZP_DIF units) per parquet row."""
    zp = pd.to_numeric(rows["MAGZP_DIF"], errors="coerce").to_numpy(float)
    if "FLUX_4_DIF" in rows and "FERR_4_DIF" in rows:
        f = pd.to_numeric(rows["FLUX_4_DIF"], errors="coerce").to_numpy(float)
        e = pd.to_numeric(rows["FERR_4_DIF"], errors="coerce").to_numpy(float)
        if np.isfinite(f).all():
            return f, e
    apc = (pd.to_numeric(rows["APCORR46"], errors="coerce").fillna(0.0).to_numpy(float) / 1000.0
           if "APCORR46" in rows else np.zeros(len(rows)))
    mraw = (pd.to_numeric(rows["MAG_4_TOT_AB_org"], errors="coerce").to_numpy(float)
            - pd.to_numeric(rows["norm_offset"], errors="coerce").fillna(0.0).to_numpy(float))
    ftot = 10.0 ** (0.4 * (zp - apc - mraw))
    fref = np.where(is_target, 0.0, 10.0 ** (0.4 * (zp - q_mag)))
    ferr = pd.to_numeric(rows["MERR_4_TOT_AB_org"], errors="coerce").to_numpy(float) * ftot / 1.0857
    f, e = ftot - fref, ferr
    if "FLUX_4_DIF" in rows:                          # prefer stored raw flux where present
        fs = pd.to_numeric(rows["FLUX_4_DIF"], errors="coerce").to_numpy(float)
        es = pd.to_numeric(rows["FERR_4_DIF"], errors="coerce").to_numpy(float)
        f, e = np.where(np.isfinite(fs), fs, f), np.where(np.isfinite(es), es, e)
    return f, e


def _write_old_epochs(base: Path, df: pd.DataFrame, sci: bool, refs: dict, ffd_of,
                      target, match_radius) -> tuple[int, pd.Index]:
    """Write one LDAC catalog per old epoch and quadrant; return (n catalogs, index of
    the parquet rows written)."""
    from astropy.io import fits
    from aperture_photometry import _write_ldac, PHOTCODE
    written, used = 0, []
    sub = "SExCatalogs_sci" if sci else "SExCatalogs"
    for key, rows in df.groupby(QKEY):
        q = (int(key[0]), str(key[1]), int(key[2]), int(key[3]))
        if q not in refs:
            continue
        ref = refs[q]
        ci = rows["_csv"].to_numpy()
        is_t = ci == len(ref)
        q_mag = np.where(is_t, np.nan, ref["q_mag"].to_numpy()[np.clip(ci, 0, len(ref) - 1)])
        f, e = _raw_flux(rows, q_mag, is_t)
        keep = (ci >= 0) & np.isfinite(f) & np.isfinite(e)
        rows, ci, f, e = rows[keep], ci[keep], f[keep], e[keep]
        if sci:
            if {"ALPHA_SCI", "DELTA_SCI"} <= set(rows.columns):
                ra = pd.to_numeric(rows["ALPHA_SCI"], errors="coerce").to_numpy(float)
                dec = pd.to_numeric(rows["DELTA_SCI"], errors="coerce").to_numpy(float)
            else:
                d0 = pd.to_numeric(rows["DELTAWIN_REF"], errors="coerce").to_numpy(float)
                ra = (pd.to_numeric(rows["ALPHAWIN_REF"], errors="coerce").to_numpy(float)
                      + pd.to_numeric(rows["DRA_SCI"], errors="coerce").to_numpy(float)
                      / 3600.0 / np.cos(np.radians(d0)))
                dec = d0 + pd.to_numeric(rows["DDEC_SCI"], errors="coerce").to_numpy(float) / 3600.0
        else:
            ra = np.where(ci == len(ref), target[0] if target else np.nan,
                          ref["ALPHAWIN_J2000"].to_numpy()[np.clip(ci, 0, len(ref) - 1)])
            dec = np.where(ci == len(ref), target[1] if target else np.nan,
                           ref["DELTAWIN_J2000"].to_numpy()[np.clip(ci, 0, len(ref) - 1)])
        flags = (pd.to_numeric(rows["FLAG_AP"], errors="coerce").fillna(0).to_numpy(int)
                 if "FLAG_AP" in rows else np.zeros(len(rows), int))
        cat_dir = base / sub / f"{q[0]:06d}" / q[1] / f"{q[2]:02d}" / str(q[3])
        for mjd, idx in rows.groupby("OBSMJD").indices.items():
            r0 = rows.iloc[idx[0]]
            h = fits.Header()
            # Every card needs a comment: calib_catalogs._read_ldac takes the value as
            # the text between '=' and '/', so a comment-less card loses its last digit.
            for card, col in _HDR_FROM_PARQUET.items():
                v = pd.to_numeric(pd.Series([r0.get(col, np.nan)]), errors="coerce").iloc[0]
                h[card] = (float(v) if np.isfinite(v) else 0.0, f"from parquet {col}")
            h["OBSMJD"] = (float(mjd), "from parquet OBSMJD")
            h["INFOBITS"] = (int(h["INFOBITS"]), "from parquet INFOBITS_DIF")
            h["SATURATE"] = (0.0, "not stored in the parquet")
            h["PHOTCODE"] = (PHOTCODE, "rebuilt from parquet by update_parquet.py")
            n = len(idx)
            fa = np.repeat(f[idx, None], 4, axis=1).astype(np.float32)
            ea = np.repeat(e[idx, None], 4, axis=1).astype(np.float32)
            name = (f"ztf_{ffd_of(q, mjd)}_{q[0]:06d}_{q[1]}_c{q[2]:02d}_o_q{q[3]}"
                    f"_scimrefdiffimg_sexout.fits")
            _write_ldac(cat_dir / name, h, {
                "NUMBER": np.arange(1, n + 1, dtype=np.int32),
                "ALPHA_J2000": ra[idx], "DELTA_J2000": dec[idx],
                "ALPHAWIN_J2000": ra[idx], "DELTAWIN_J2000": dec[idx],
                "CLASS_STAR": np.zeros(n, np.float32),
                "FLAGS": flags[idx].astype(np.int16), "NUMBER_ASSOC": np.ones(n, np.int32),
                "VECTOR_ASSOC": (ci[idx] + 1).astype(np.float64),
                "FLUX_APER": fa, "FLUXERR_APER": ea,
            })
            written += 1
        used.append(rows.index)
    return written, (pd.Index(np.concatenate([u.to_numpy() for u in used])) if used else pd.Index([]))


# ── output assembly ──────────────────────────────────────────────────────────

def _collect(base: Path, ra, dec, fc, suffix) -> tuple[pd.DataFrame | None, dict]:
    p = base / "LightCurves" / "merged" / f"{ra:.5f}_{dec:+.5f}" / f"{fc}{suffix}" / "lightcurves_merged.parquet"
    if not p.exists():
        cands = sorted((base / "LightCurves").glob(f"[0-9]*/{fc}/ccd*/q*/lightcurves{suffix}.parquet"))
        if len(cands) != 1:
            return None, {}
        p = cands[0]
    meta = dict(pq.read_schema(p).metadata or {})
    return _load(p), meta


def _assemble(old: pd.DataFrame, new: pd.DataFrame, refs, target, match_radius,
              old_epochs: set, all_sources: bool = False) -> pd.DataFrame:
    """Old epochs: new 4 px, other apertures shifted by the 4 px change; new epochs as
    measured; unreconstructable old rows appended unchanged."""
    new = new.copy()
    new["_csv"] = -1
    for key, idx in new.groupby(QKEY).groups.items():
        q = (int(key[0]), str(key[1]), int(key[2]), int(key[3]))
        if q in refs:
            sel = new.loc[idx]
            new.loc[idx, "_csv"] = _csv_index(sel["ALPHAWIN_REF"].to_numpy(float),
                                              sel["DELTAWIN_REF"].to_numpy(float),
                                              refs[q], target, match_radius)
    k = QKEY + ["_csv", "_mjd"]
    old = old.assign(_mjd=old["OBSMJD"].round(6))
    new = new.assign(_mjd=new["OBSMJD"].round(6))
    # Same object set as the original file: drop sources the parquet never had (the
    # reference catalog may list more, e.g. if make_catalog's selection changed since)
    if not all_sources:
        known = pd.MultiIndex.from_frame(old[QKEY + ["_csv"]].drop_duplicates())
        in_old = pd.MultiIndex.from_frame(new[QKEY + ["_csv"]]).isin(known)
        if (~in_old).any():
            logger.info(f"    dropping {int((~in_old).sum()):,} new rows of "
                        f"{new.loc[~in_old, QKEY + ['_csv']].drop_duplicates().shape[0]} sources "
                        f"not in the original parquet (--all-sources keeps them)")
        new = new[in_old]
    is_old_ep = new.set_index(["field", "filtercode", "ccdid", "qid", "_mjd"]).index.isin(list(old_epochs))
    shift_cols = [c for c in ("MAG_3_TOT_AB", "MAG_6_TOT_AB", "MAG_10_TOT_AB") if c in old]
    keep_cols = shift_cols + [c.replace("MAG_", "MERR_") for c in shift_cols
                              if c.replace("MAG_", "MERR_") in old]
    o = old[k + ["MAG_4_TOT_AB"] + keep_cols + (["FLUX_6_TOT_AB"] if "FLUX_6_TOT_AB" in old else [])]
    o = o.drop_duplicates(k)
    m = new[is_old_ep].reset_index().merge(o, on=k, how="left", suffixes=("", "_old"))
    dm = m["MAG_4_TOT_AB"] - m["MAG_4_TOT_AB_old"]
    for c in shift_cols:
        new.loc[m["index"].to_numpy(), c] = (m[f"{c}_old"] + dm).to_numpy()
    for c in keep_cols:
        if c not in shift_cols:
            new.loc[m["index"].to_numpy(), c] = m[f"{c}_old"].to_numpy()
    if "FLUX_6_TOT_AB" in new:
        f6 = (m["FLUX_6_TOT_AB_old"] * 10 ** (-0.4 * dm)) if "FLUX_6_TOT_AB_old" in m else np.nan
        new.loc[m["index"].to_numpy(), "FLUX_6_TOT_AB"] = np.asarray(f6, float)
    # old rows not represented in the new output -> carried over unchanged
    # Quadrants the current pipeline no longer processes (e.g. the footprint check drops
    # them because the target is off-chip) cannot be recalibrated consistently: drop them
    qnew = pd.MultiIndex.from_frame(new[QKEY].drop_duplicates())
    gone = ~pd.MultiIndex.from_frame(old[QKEY]).isin(qnew)
    if gone.any():
        logger.info(f"    dropping {int(gone.sum()):,} rows of quadrant(s) "
                    f"{sorted(map(tuple, old.loc[gone, QKEY].drop_duplicates().to_numpy().tolist()))} "
                    f"that the current pipeline does not process (e.g. target off the reference footprint)")
        old = old[~gone]
    have = pd.MultiIndex.from_frame(new[k])
    carry = old[~pd.MultiIndex.from_frame(old[k]).isin(have)].copy()
    # carried rows take the object_index the new output uses for the same quadrant +
    # reference source (catalog ordering / merge re-keying may differ from the original)
    oid = new.drop_duplicates(QKEY + ["_csv"]).set_index(QKEY + ["_csv"])["object_index"]
    keyc = pd.MultiIndex.from_frame(carry[QKEY + ["_csv"]])
    mapped = oid.reindex(keyc).to_numpy(dtype=float)
    # objects with no row in the new output (e.g. only unreconstructable NaN rows) keep
    # their original object_index — exact whenever the reference catalog is unchanged;
    # a row that would then collide with an existing (quadrant, object, epoch) is dropped
    carry["object_index"] = np.where(np.isfinite(mapped), mapped,
                                     pd.to_numeric(carry["object_index"]).to_numpy(float))
    n_unm = int((carry["_csv"] < 0).sum())
    if n_unm:
        logger.info(f"    {n_unm:,} old rows match no reference-catalog source — dropped")
    carry = carry[carry["_csv"] >= 0]
    kk = QKEY + ["object_index", "_mjd"]
    clash = pd.MultiIndex.from_frame(carry[kk].astype({"object_index": float})).isin(
        pd.MultiIndex.from_frame(new[kk].astype({"object_index": float})))
    if clash.any():
        logger.info(f"    {int(clash.sum()):,} carried rows would duplicate an existing "
                    f"(quadrant, object_index, epoch) — dropped")
    carry = carry[~clash]
    out = pd.concat([new, carry], ignore_index=True, sort=False)
    out = out.drop(columns=["_csv", "_mjd"]).sort_values("OBSMJD", ignore_index=True)
    out["object_index"] = pd.to_numeric(out["object_index"]).astype("Int32")
    logger.info(f"    old-epoch rows recalibrated {len(m):,} (median dMAG_4 "
                f"{1000 * np.nanmedian(dm):+.1f} mmag), new-epoch rows {int((~is_old_ep).sum()):,}, "
                f"carried unchanged {len(carry):,}")
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def update_target(paths: list[Path], a, auth) -> str:
    """Returns "ok", "refs" (reference products unavailable — worth retrying later in
    the same run) or "fail"."""
    import download_coordinator as dc
    import ztf_field_lookup as zl
    from make_catalog import make_catalog

    variants = {("_sci" if _is_sci(p) else ""): p for p in paths}
    first = next(iter(variants.values()))
    frames = {s: _load(p) for s, p in variants.items()}
    have, meta = read_parquet_epochs(first)
    ra, dec = target_position(first, meta)
    fcs = sorted(have["filtercode"].unique())
    if len(fcs) != 1:
        logger.warning(f"SKIP {first.name}: expected one band, found {fcs}")
        return "fail"
    fc = fcs[0]
    logger.info(f"═══ {', '.join(p.name for p in variants.values())}: RA={ra} Dec={dec} {fc} ═══")

    base = Path(tempfile.mkdtemp(prefix="ztfupd_", dir=a.tmpdir))
    try:
        (base / "Epochs").mkdir(parents=True)
        epochs = zl.lookup_target(ra, dec, bands=[fc], cache_dir=base / "Epochs", force_refresh=True)
        epochs = epochs[epochs["filtercode"] == fc]

        def build(cuts):
            p = dc.filter_epochs(epochs, **cuts).astype({"field": int, "ccdid": int, "qid": int})
            p["filefracday"] = p["filefracday"].map(lambda v: str(int(float(v))))
            p["in_parquet"] = match_epochs(p, have, a.tol)
            return p
        cuts, recorded = resolve_cuts(a, meta)
        possible = build(cuts)
        if not recorded:
            hint = {k: v for k, v in infer_cuts(possible).items() if cuts.get(k) is None}
            logger.info(f"  no recorded download cuts; IRSA values of epochs present: {hint}"
                        + (" — applying" if a.infer_cuts else " — NOT applied (use --infer-cuts)"))
            if a.infer_cuts:
                cuts = {**cuts, **hint}
                possible = build(cuts)

        quads_in = sorted({(int(f), str(c), int(cc), int(qq)) for f, c, cc, qq
                           in pd.concat([d[QKEY] for d in frames.values()]).drop_duplicates().to_numpy()})
        t_end = have["OBSMJD"].max()
        todo = possible[~possible["in_parquet"]]
        todo = todo[np.array([(int(r.field), r.filtercode, int(r.ccdid), int(r.qid)) in quads_in
                              for r in todo.itertuples()], dtype=bool)]
        n_new = int((todo["obsmjd"] > t_end + a.tol).sum())
        if a.no_new:
            todo = todo[todo["obsmjd"] <= t_end + a.tol]
        logger.info(f"  {len(have)} epochs in parquet; to add: {len(todo)} "
                    f"({len(todo) - (0 if a.no_new else n_new)} missing, "
                    f"{0 if a.no_new else n_new} newer than the parquet)")
        if a.dry_run or len(todo) == 0:
            if len(todo) == 0:
                logger.info("  nothing to add")
                if not a.dry_run:
                    for path in variants.values():
                        _clear_deferred(path)
            return "ok"

        # 2. references + catalogs, then only the missing/new epochs
        # References are essential: a quadrant missing them would lose its rows, so a
        # (transient) reference failure fails the whole target — nothing is written and
        # the next run retries it — rather than dropping the quadrant.
        quads = _fetch_references(base, quads_in, auth, max(a.download_retries, 3), a.retry_wait)
        if len(quads) < len(quads_in):
            logger.warning(f"FAIL {first.name}: reference products unavailable for "
                           f"{sorted(set(quads_in) - set(quads))} — target re-queued for later")
            return "refs"
        # Same reference-footprint check run_pipeline applies: quadrants whose reference
        # image does not contain the target are dropped by the pipeline anyway, so don't
        # download their epochs (science exposures dither and can cover the target while
        # the reference image — which defines what is measured — does not).
        from run_pipeline import _filter_quadrants_by_wcs
        qd = [dict(field=q[0], filtercode=q[1], ccdid=q[2], qid=q[3],
                   ref_dir=base / "Reference" / f"{q[0]:06d}" / q[1] / f"ccd{q[2]:02d}" / f"q{q[3]}")
              for q in quads]
        on_ref = {(d["field"], d["filtercode"], d["ccdid"], d["qid"])
                  for d in _filter_quadrants_by_wcs(base, qd, ra, dec, margin=a.footprint_margin_px)}
        if len(on_ref) < len(quads):
            off = sorted(set(quads) - on_ref)
            n_skip = sum((int(r.field), r.filtercode, int(r.ccdid), int(r.qid)) in off
                         for r in todo.itertuples())
            logger.info(f"  target outside the reference footprint of {off} — those quadrants "
                        f"are dropped by the pipeline; not downloading their {n_skip} epoch(s)")
            todo = todo[np.array([(int(r.field), r.filtercode, int(r.ccdid), int(r.qid)) in on_ref
                                  for r in todo.itertuples()], dtype=bool)]
        (base / "Catalogs").mkdir(exist_ok=True)
        make_catalog(str(base / "Catalogs"), str(base / "Reference"))
        refs = {q: _ref_catalog(base, q) for q in quads}
        deferred = _download_epochs(dc, todo, base, fc, a)

        # 3. old epochs -> per-epoch catalogs
        all_ep = dc.filter_epochs(epochs).astype({"field": int, "ccdid": int, "qid": int})
        def ffd_of(q, mjd):
            e = all_ep[(all_ep["field"] == q[0]) & (all_ep["ccdid"] == q[2]) & (all_ep["qid"] == q[3])]
            if len(e):
                j = int(np.argmin(np.abs(e["obsmjd"].to_numpy() - mjd)))
                if abs(e["obsmjd"].iloc[j] - mjd) <= a.tol:
                    return str(int(float(e["filefracday"].iloc[j])))
            return f"m{int(round(mjd * 1e5))}"
        target = (ra, dec)
        old_epochs = set()
        for s, df in frames.items():
            df["_csv"] = -1
            for key, idx in df.groupby(QKEY).groups.items():
                q = (int(key[0]), str(key[1]), int(key[2]), int(key[3]))
                if q in refs:
                    sel = df.loc[idx]
                    df.loc[idx, "_csv"] = _csv_index(sel["ALPHAWIN_REF"].to_numpy(float),
                                                     sel["DELTAWIN_REF"].to_numpy(float),
                                                     refs[q], target, a.target_match_radius)
            n_cat, used = _write_old_epochs(base, df, bool(s), refs, ffd_of, target,
                                            a.target_match_radius)
            logger.info(f"  {variants[s].name}: {n_cat} old-epoch catalogs from "
                        f"{len(used):,}/{len(df):,} rows (unmatched {int((df['_csv'] < 0).sum())})")
            old_epochs |= set(map(tuple, df[QKEY].assign(_m=df["OBSMJD"].round(6)).to_numpy().tolist()))

        # Has the reference catalog changed since the parquet was built (e.g. make_catalog's
        # selection changed)? If unchanged, the original object_index equals the rebuilt
        # catalog index for the (dominant) quadrant's sources; only a changed catalog
        # warrants restricting the added epochs to the original object set.
        fracs = []
        for df in frames.values():
            for _, g in df[df["_csv"] >= 0].drop_duplicates(QKEY + ["object_index"]).groupby(QKEY):
                fracs.append(float((pd.to_numeric(g["object_index"]) == g["_csv"]).mean()))
        catalog_changed = (max(fracs) if fracs else 0.0) < 0.9
        logger.info("  reference catalog " + (
            "CHANGED since the parquet was built — added epochs restricted to its objects "
            "(--all-sources overrides)" if catalog_changed else "unchanged"))

        # provenance: the download cuts this parquet was built with
        rec = json.loads(meta.get("ztfphot_download") or "null") or {
            "time_utc": None, "inferred_by": "update_parquet.py",
            "args": {"ra": ra, "dec": dec, "steps": ["download"],
                     "min_maglim": cuts.get("min_maglim"), "max_seeing": cuts.get("max_seeing"),
                     "skip_flagged": bool(cuts.get("skip_cautionary"))}}
        (base / "Epochs" / "ztfphot_runs.jsonl").write_text(json.dumps(rec, default=str) + "\n")

        # 4. current pipeline over old + new epochs
        both = len(variants) == 2
        phot = ["sex"] if "" in variants else ["sex_sci"]
        cmd = [sys.executable, str(_SCRIPTS / "run_pipeline.py"), "--base-dir", str(base),
               "--ra", str(ra), "--dec", str(dec), "--bands", fc[1:],
               "--steps", *phot, "vet", "calibrate", "flatfield", "recalibrate", "lightcurves", "merge",
               "--workers", str(a.workers), "--target-match-radius", str(a.target_match_radius),
               "--footprint-margin-px", str(a.footprint_margin_px)]
        cmd += ["--both"] if both else (["--scipos"] if "" not in variants else [])
        logger.info("  running: " + " ".join(cmd[1:]))
        log = base / "run_pipeline.log"
        with open(log, "w") as fh:
            if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode != 0:
                logger.warning(f"FAIL: run_pipeline failed — see {log}")
                a.keep_temp = True
                return "fail"

        # 5. assemble + write
        for s, path in variants.items():
            new, new_meta = _collect(base, ra, dec, fc, s)
            if new is None:
                logger.warning(f"FAIL {path.name}: no light curve produced")
                return "fail"
            out = _assemble(frames[s], new, refs, target, a.target_match_radius, old_epochs,
                            all_sources=a.all_sources or not catalog_changed)
            out_dir = Path(a.outdir) if a.outdir else path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            # re-running on an updated file (--retry-deferred) refreshes it in place
            stem = path.stem if path.stem.endswith(a.suffix) else f"{path.stem}{a.suffix}"
            out_path = out_dir / f"{stem}.parquet"
            new_meta[b"ztfphot_update"] = json.dumps({
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(path), "cuts": cuts, "cuts_recorded": recorded,
                "epochs_added": int(len(todo) - len(deferred)),
                "epochs_deferred": int(len(deferred)), "deferred_filefracdays": deferred,
                "rows": int(len(out)),
            }, default=str).encode()
            table = pa.Table.from_pandas(out, preserve_index=False)
            table = table.replace_schema_metadata({**new_meta, **(table.schema.metadata or {})})
            tmp = out_path.with_name(out_path.name + ".tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, out_path)
            logger.info(f"  → {out_path} ({len(out):,} rows, {out['OBSMJD'].nunique()} epochs)")
        return "ok"
    finally:
        if a.keep_temp:
            logger.info(f"  (kept temp dir {base})")
        else:
            shutil.rmtree(base, ignore_errors=True)


def _plan_directory(d: Path, a) -> list[list[Path]]:
    """Target groups from a directory of light-curve parquets. Per target, band and
    variant (ref-pos / _sci) the input is X_recal.parquet if it exists, else X.parquet
    (*_updated outputs are never inputs); ref + sci of the same target form one group,
    whichever version each one is. Groups whose outputs all exist are skipped (--redo)."""
    files = sorted(d.rglob("*.parquet") if a.recursive else d.glob("*.parquet"))
    pick: dict[tuple, dict] = {}
    for f in files:
        if f.name == "lightcurves_merged.parquet":            # pipeline tree .../{band}{_sci}/
            band = f.parent.name
            sci, tail = band.endswith("_sci"), ""
            gkey = str(f.parent.parent / band.removesuffix("_sci"))
        else:
            stem = f.stem
            tail = next((t for t in ("_recal", "_updated") if stem.endswith(t)), "")
            canon = stem[:len(stem) - len(tail)]
            if tail == "_updated" or canon.endswith(a.suffix):
                continue
            sci = "_sci" in canon
            gkey = str(f.parent / canon.replace("_sci", ""))
        slot = pick.setdefault((gkey, sci), {})
        slot[tail] = f
    groups: dict[str, list[Path]] = {}
    for (gkey, sci), versions in sorted(pick.items()):
        chosen = versions.get("_recal", versions.get(""))
        if chosen is not None:
            groups.setdefault(gkey, []).append(chosen)
    out = []
    for gkey, members in groups.items():
        outs = [(Path(a.outdir) if a.outdir else m.parent) / f"{m.stem}{a.suffix}.parquet"
                for m in members]
        done = all(o.exists() for o in outs)
        if a.retry_deferred:
            # only targets whose update left deferred epochs; their _updated files are the
            # input (all other epochs already in them) and are refreshed in place
            if done and any(_n_deferred(o) for o in outs):
                out.append(outs)
            continue
        if not a.redo and done:
            logger.info(f"  skip {Path(gkey).name}: already updated ({', '.join(o.name for o in outs)})")
            continue
        out.append(members)
    return out


def _clear_deferred(p: Path) -> None:
    """An updated file whose deferred epochs turn out to need nothing (no longer
    missing / no longer passing the cuts): reset epochs_deferred so --retry-deferred
    stops selecting it."""
    if not _n_deferred(p):
        return
    t = pq.read_table(p)
    md = dict(t.schema.metadata or {})
    u = json.loads(md[b"ztfphot_update"])
    u.update(epochs_deferred=0, deferred_filefracdays=[])
    md[b"ztfphot_update"] = json.dumps(u, default=str).encode()
    tmp = p.with_name(p.name + ".tmp")
    pq.write_table(t.replace_schema_metadata(md), tmp)
    os.replace(tmp, p)
    logger.info(f"  {p.name}: deferred epochs no longer missing — flag cleared")


def _n_deferred(p: Path) -> int:
    try:
        meta = pq.read_schema(p).metadata or {}
        return int(json.loads(meta.get(b"ztfphot_update", b"{}")).get("epochs_deferred", 0))
    except Exception:
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquet", nargs="+", type=Path,
                   help="parquet files, or directories: per target/band/variant the "
                        "*_recal.parquet is used if present, else the plain one")
    p.add_argument("--recursive", action="store_true", help="search directories recursively")
    p.add_argument("--plan", action="store_true",
                   help="print which files would be used (no IRSA queries), then stop")
    p.add_argument("--redo", action="store_true",
                   help="directory mode: also process targets that already have an output")
    p.add_argument("--infer-cuts", action="store_true",
                   help="older parquets without recorded cuts: use the IRSA maglimit/seeing "
                        "range of the epochs present as the epoch cuts")
    p.add_argument("--min-maglim", type=float, default=None, help="override the epoch cut")
    p.add_argument("--max-seeing", type=float, default=None, help="override the epoch cut")
    p.add_argument("--skip-flagged", action="store_true", help="override: skip cautionary epochs")
    p.add_argument("--no-new", action="store_true",
                   help="only fill gaps; do not add epochs newer than the parquet")
    p.add_argument("--dry-run", action="store_true", help="report what would be added, then stop")
    p.add_argument("--all-sources", action="store_true",
                   help="if the reference catalog changed since the parquet was built, also "
                        "keep sources the original parquet lacks (they only get the added "
                        "epochs); with an unchanged catalog all sources are kept anyway")
    p.add_argument("--outdir", default=None)
    p.add_argument("--suffix", default="_updated")
    p.add_argument("--tol", type=float, default=1e-4, help="OBSMJD match tolerance [d]")
    p.add_argument("--target-match-radius", type=float, default=1.0)
    p.add_argument("--footprint-margin-px", type=int, default=3,
                   help="as run_pipeline: reference-image edge margin for the target")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--download-retries", type=int, default=2,
                   help="short extra rounds for failed downloads (default 2); epochs still "
                        "failing are deferred, not waited for")
    p.add_argument("--retry-wait", type=float, default=15.0, metavar="S",
                   help="wait before retry round n is n x S seconds (default 15)")
    p.add_argument("--retry-deferred", action="store_true",
                   help="directory mode: only targets whose earlier update deferred epochs; "
                        "re-run on their _updated files (downloads just those epochs)")
    p.add_argument("--ref-passes", type=int, default=2,
                   help="extra passes at the end of the batch for targets whose reference "
                        "products were unavailable (default 2)")
    p.add_argument("--ref-pass-wait", type=float, default=120.0, metavar="S",
                   help="pause before each extra pass (default 120 s; once per pass, not per target)")
    p.add_argument("--tmpdir", default=None, help="parent dir for the temporary base dir")
    p.add_argument("--keep-temp", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    # Build the list of target groups: directories via the recal-else-plain selection,
    # explicit files paired with their sibling as given.
    groups, done = [], set()
    for path in a.parquet:
        path = path.resolve()
        if path.is_dir():
            for g in _plan_directory(path, a):
                g = [m.resolve() for m in g]
                if not done & set(g):
                    groups.append(g)
                    done |= set(g)
        elif path not in done:
            g = [path] + ([s.resolve()] if (s := _sibling(path)) else [])
            groups.append(g)
            done |= set(g)
    logger.info(f"{len(groups)} target group(s) to process:")
    for g in groups:
        logger.info("  " + "  +  ".join(m.name for m in g))
    if a.plan or not groups:
        return

    from download_coordinator import get_auth
    import time
    auth = get_auth()
    status = {}
    queue = list(groups)
    for npass in range(a.ref_passes + 1):
        if npass:
            logger.info(f"── pass {npass + 1}: retrying {len(queue)} target(s) whose reference "
                        f"products were unavailable (after {a.ref_pass_wait:.0f} s) ──")
            time.sleep(a.ref_pass_wait)
        requeue = []
        for g in queue:
            try:
                st = update_target(g, a, auth)
            except Exception as exc:
                logger.exception(f"FAIL {g[0].name}: {exc}")
                st = "fail"
            status[g[0].name] = st
            if st == "refs":
                requeue.append(g)
        queue = requeue
        if not queue:
            break
    bad = {n: st for n, st in status.items() if st != "ok"}
    logger.info(f"done: {len(status) - len(bad)}/{len(status)} target groups"
                + (f"; FAILED: {', '.join(f'{n} ({st})' for n, st in bad.items())}" if bad else "")
                + (" — reference failures are retried on the next run" if "refs" in bad.values() else ""))


if __name__ == "__main__":
    main()
