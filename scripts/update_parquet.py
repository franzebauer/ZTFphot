#!/usr/bin/env python3
"""
update_parquet.py
-----------------
Add missing (and, by default, newer) epochs to existing light-curve parquets, and
recalibrate all 4 px photometry consistently with the CURRENT pipeline — without
re-downloading the epochs already in the parquet.

Inputs must be in the current format (FLUX_4_DIF_uJy and APCORR410 columns); parquets
written before 2026-10-21 are retrofitted with update_old_parquet.py, which uses the
same machinery.

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
     writes) from the parquet's raw difference flux FLUX_4_DIF_uJy / FERR_4_DIF_uJy
     (converted back to counts with MAGZP_DIF), and FLUX_k_DIF_uJy for the 3/6/10 px
     apertures in --lc-columns full files; the output keeps the input's preset.
  4. run_pipeline (current code) on the temp dir: forced photometry of the new epochs,
     then vet -> calibrate -> flatfield -> recalibrate -> lightcurves -> merge over
     old + new epochs together.
  5. Output <stem>_updated.parquet next to the input (or --outdir):
       old and new epochs alike: all photometric columns from the current pipeline
       old rows the current pipeline does not reproduce (e.g. an epoch that now has
       too few calibrators, a quadrant the footprint check drops) are dropped.
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
    if tail:                                        # sibling without the suffix
        cands += [c.with_name(c.name.replace(tail + ".parquet", ".parquet")) for c in list(cands)]
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


def _is_current(df: pd.DataFrame) -> bool:
    """Written by the current pipeline (raw 4 px flux stored, 4→10 px correction)."""
    return {"FLUX_4_DIF_uJy", "FERR_4_DIF_uJy", "APCORR410"} <= set(df.columns)


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


def _download_epochs(dc, todo: pd.DataFrame, base: Path, fc: str, a) -> tuple[list[str], list[str]]:
    """Science products of the epochs to add. Transient failures (IRSA 503s, dropped
    connections) get a few SHORT retry rounds (--retry-wait s x round); whatever still
    fails is deferred rather than waited for — the target is processed without those
    epochs and they are recorded in the output (epochs_deferred), for --retry-deferred.
    Returns (deferred, unavailable on IRSA (HTTP 404)) filefracdays."""
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
    return deferred, unavailable


def _image_dirs(a, meta: dict) -> list[Path]:
    """Directories that may already hold the images: --image-dir, then the base dir
    recorded in the parquet metadata (ztfphot_run / ztfphot_download), if present here."""
    cands = [Path(d) for d in (a.image_dir or [])]
    for key in ("ztfphot_run", "ztfphot_download"):
        try:
            cands.append(Path(json.loads(meta.get(key) or "null")["args"]["base_dir"]))
        except (TypeError, KeyError, ValueError):
            pass
    out = []
    for d in cands:
        d = d.expanduser().resolve()
        if d not in out and ((d / "Science").is_dir() or (d / "Reference").is_dir()):
            out.append(d)
    return out


def _reuse_local(dc, base: Path, todo: pd.DataFrame, quads, src_dirs: list[Path]) -> tuple[int, int]:
    """Put images that already exist in `src_dirs` into the temporary base dir, so they
    are not downloaded again: an unpacked .fits is symlinked; a .fits.fz is copied and
    unpacked (funpack deletes its input). Returns (epochs, reference products) reused."""
    def put(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    ok = lambda f: f.is_file() and f.stat().st_size > 0
    n_ref = 0
    for q in quads:
        for suf in _REF_PRODUCTS:
            src = next((s for d in src_dirs if ok(s := dc.ref_local_path(d, *q, suf))), None)
            if src is not None:
                put(src, dc.ref_local_path(base, *q, suf))
                n_ref += 1
    n_sci = 0
    for r in todo.itertuples():
        e = (int(r.field), r.filtercode, int(r.ccdid), int(r.qid), str(int(float(r.filefracday))))
        for d in src_dirs:
            diff, fz = (dc.sci_local_path(d, *e, "scimrefdiffimg.fits"),
                        dc.sci_local_path(d, *e, "scimrefdiffimg.fits.fz"))
            if ok(diff):
                put(diff, dc.sci_local_path(base, *e, "scimrefdiffimg.fits"))
            elif ok(fz):
                dst = dc.sci_local_path(base, *e, "scimrefdiffimg.fits.fz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fz, dst)
                dc.funpack_file(dst)
            else:
                continue
            if ok(cat := dc.sci_local_path(d, *e, "sexcat.fits")):
                put(cat, dc.sci_local_path(base, *e, "sexcat.fits"))
            n_sci += 1
            break
    return n_sci, n_ref


def _ref_catalog(base: Path, q) -> pd.DataFrame:
    tag = f"{q[0]:06d}_{q[1]}_c{q[2]:02d}_q{q[3]}"
    ref = pd.read_csv(base / "Catalogs" / f"{tag}(REFERENCE)[OBJECTS].csv")
    ref["q_mag"] = (pd.to_numeric(ref["MAG_APER_4px"], errors="coerce")
                    + pd.to_numeric(ref["MAGZP_REF"], errors="coerce"))
    return ref


def _complete_catalog(dc, base: Path, q, positions: np.ndarray, target, match_radius,
                      tol: float = 0.5) -> tuple[int, int]:
    """Append to the rebuilt reference CSV of quadrant `q` every source of the old
    parquet (reference positions `positions`) it lacks, so no source loses its light
    curve: one in IRSA's refsexcat (removed by make_catalog's selection) with its
    reference photometry; otherwise (injected in the original run) as an injected
    entry, as transient_catalog writes them (99 mag sentinel; FLAG_SE_REF = 1 with a
    reference neighbour within 3"). The --ra/--dec target is left to the pipeline,
    which injects it itself. Returns (from refsexcat, injected)."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    tag = f"{q[0]:06d}_{q[1]}_c{q[2]:02d}_q{q[3]}"
    csv = base / "Catalogs" / f"{tag}(REFERENCE)[OBJECTS].csv"
    ref = pd.read_csv(csv)
    pos = np.unique(np.round(positions[np.isfinite(positions).all(1)], 7), axis=0)
    if not len(pos):
        return 0, 0
    sc = SkyCoord(pos[:, 0], pos[:, 1], unit="deg")
    rc = SkyCoord(ref["ALPHAWIN_J2000"].to_numpy(float), ref["DELTAWIN_J2000"].to_numpy(float), unit="deg")
    miss = sc.match_to_catalog_sky(rc)[1].arcsec >= tol
    if target is not None:
        miss &= sc.separation(SkyCoord(target[0], target[1], unit="deg")).arcsec >= match_radius
    if not miss.any():
        return 0, 0
    sc = sc[miss]
    raw = fits.getdata(dc.ref_local_path(base, *q, "refsexcat.fits"), 1)
    j, sep, _ = sc.match_to_catalog_sky(SkyCoord(raw["ALPHAWIN_J2000"], raw["DELTAWIN_J2000"], unit="deg"))
    in_raw = sep.arcsec < tol
    rows = []
    for k in sorted(set(j[in_raw])):                 # as make_catalog converts refsexcat rows
        r = raw[k]
        rows.append({"CLASS_STAR": r["CLASS_STAR"], "ALPHAWIN_J2000": r["ALPHAWIN_J2000"],
                     "DELTAWIN_J2000": r["DELTAWIN_J2000"], "MAG_AUTO": r["MAG_AUTO"],
                     "MAGERR_AUTO": r["MAGERR_AUTO"], "FLAG_SE_REF": r["FLAGS"],
                     **{f"MAG_APER_{px}": r["MAG_APER"][i] for i, px in enumerate(("3px", "4px", "6px", "10px"), 1)},
                     **{f"MAGERR_APER_{px}": r["MAGERR_APER"][i] for i, px in enumerate(("3px", "4px", "6px", "10px"), 1)}})
    n_raw = len(rows)
    for c in sc[~in_raw]:
        rows.append({"CLASS_STAR": 0.0, "ALPHAWIN_J2000": c.ra.deg, "DELTAWIN_J2000": c.dec.deg,
                     "MAG_AUTO": 99.0, "MAGERR_AUTO": 99.0,
                     "FLAG_SE_REF": int(len(ref) > 0 and c.separation(rc).arcsec.min() < 3.0),
                     **{f"MAG_APER_{px}": 99.0 for px in ("3px", "4px", "6px", "10px")},
                     **{f"MAGERR_APER_{px}": 99.0 for px in ("3px", "4px", "6px", "10px")}})
    add = pd.DataFrame(rows)
    for c in ("MAGZP_REF", "MAGZPRMS_REF", "INFOBITS_REF"):
        add[c] = ref[c].iloc[0]
    add["ID"] = [f"{len(ref) + i}_{tag}" for i in range(len(add))]
    pd.concat([ref, add[ref.columns]], ignore_index=True).to_csv(csv, index=False)
    return n_raw, len(rows) - n_raw


def _csv_index(ra, dec, ref: pd.DataFrame, target, match_radius, tol=0.5) -> np.ndarray:
    """Reference-catalog row (0-based) for each (ra, dec); the injected target gets
    len(ref) (no reference source within match_radius), as in the photometry step.
    Unmatched -> -1. Positions repeat for every epoch of a source, so the match is done
    once per distinct position."""
    from astropy.coordinates import SkyCoord
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    pos, inv = np.unique(np.column_stack([ra, dec]), axis=0, return_inverse=True)
    if len(pos) < len(ra):
        return _csv_index(pos[:, 0], pos[:, 1], ref, target, match_radius, tol)[inv.ravel()]
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

def _raw_flux(rows: pd.DataFrame, q_mag: np.ndarray, is_target: np.ndarray, ref: pd.DataFrame):
    """Raw 4 px difference flux and error (counts, MAGZP_DIF units) per parquet row."""
    return _dif_counts(rows, "4")


def _dif_counts(rows: pd.DataFrame, k: str):
    """FLUX/FERR_{k}_DIF_uJy in counts (MAGZP_DIF units); NaN where absent."""
    if f"FLUX_{k}_DIF_uJy" not in rows:
        return np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    c = 10.0 ** (0.4 * (pd.to_numeric(rows["MAGZP_DIF"], errors="coerce").to_numpy(float) - 23.9))
    return (pd.to_numeric(rows[f"FLUX_{k}_DIF_uJy"], errors="coerce").to_numpy(float) * c,
            pd.to_numeric(rows[f"FERR_{k}_DIF_uJy"], errors="coerce").to_numpy(float) * c)


def _write_old_epochs(base: Path, df: pd.DataFrame, sci: bool, refs: dict, ffd_of,
                      target, match_radius, raw_flux=_raw_flux) -> tuple[int, pd.Index]:
    """Write one LDAC catalog per old epoch and quadrant; return (n catalogs, index of
    the parquet rows written)."""
    from astropy.io import fits
    from aperture_photometry import _write_ldac, PHOTCODE, CATALOG_DIRS, catalog_name
    written, used = 0, []
    sub = CATALOG_DIRS["_sci" if sci else ""]
    for key, rows in df.groupby(QKEY):
        q = (int(key[0]), str(key[1]), int(key[2]), int(key[3]))
        if q not in refs:
            continue
        ref = refs[q]
        ci = rows["_csv"].to_numpy()
        is_t = ci == len(ref)
        q_mag = np.where(is_t, np.nan, ref["q_mag"].to_numpy()[np.clip(ci, 0, len(ref) - 1)])
        f, e = raw_flux(rows, q_mag, is_t, ref)
        # the other aperture slots from their own raw flux (full preset); NaN where the
        # parquet has none, so that aperture is simply not calibrated for these epochs
        other = {slot: _dif_counts(rows, k) for slot, k in ((0, "3"), (2, "6"), (3, "10"))}
        keep = (ci >= 0) & np.isfinite(f) & np.isfinite(e)
        rows, ci, f, e = rows[keep], ci[keep], f[keep], e[keep]
        other = {slot: (fo[keep], eo[keep]) for slot, (fo, eo) in other.items()}
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
        # per-epoch aperture bits of the old epochs, carried into the rebuilt FLAGS
        from ztfphot_flags import F
        flags = (pd.to_numeric(rows["FLAGS"], errors="coerce").fillna(0).astype("int64")
                 .to_numpy() & (F.EPOCH_BITS | F.AP_UNKNOWN))
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
            for slot, (fo, eo) in other.items():
                fa[:, slot], ea[:, slot] = fo[idx], eo[idx]
            name = catalog_name(f"ztf_{ffd_of(q, mjd)}_{q[0]:06d}_{q[1]}_c{q[2]:02d}_o_q{q[3]}"
                                f"_scimrefdiffimg")
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
    """The new output (old and new epochs), restricted to the original object set unless
    `all_sources`; old rows absent from it are dropped (and counted)."""
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
    o = old[k + ["MAG_4_TOT_AB"] + (["APCORR410"] if "APCORR410" in old else [])].drop_duplicates(k)
    m = new[is_old_ep].reset_index().merge(o, on=k, how="left", suffixes=("", "_old"))
    # change of the 4 px magnitude before its aperture correction (for the log)
    _apc = lambda c: (pd.to_numeric(m[c], errors="coerce").fillna(0.0) / 1000.0
                      if c in m else 0.0)
    dm = ((m["MAG_4_TOT_AB"] + _apc("APCORR410"))
          - (m["MAG_4_TOT_AB_old"] + _apc("APCORR410_old")))
    n_drop = int((~pd.MultiIndex.from_frame(old[k]).isin(pd.MultiIndex.from_frame(new[k]))).sum())
    out = new
    out["FLAGS"] = pd.to_numeric(out["FLAGS"]).astype("Int32")
    out = out.drop(columns=["_csv", "_mjd"]).sort_values("OBSMJD", ignore_index=True)
    out["object_index"] = pd.to_numeric(out["object_index"]).astype("Int32")
    logger.info(f"    old-epoch rows recalibrated {len(m):,} (median dMAG_4 "
                f"{1000 * np.nanmedian(dm):+.1f} mmag), new-epoch rows {int((~is_old_ep).sum()):,}, "
                f"old rows not reproduced by the current pipeline (dropped) {n_drop:,}")
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def update_target(paths: list[Path], a, auth, legacy=None) -> str:
    """Returns "ok", "refs" (reference products unavailable — worth retrying later in
    the same run) or "fail". `legacy` (update_old_parquet) converts parquets written
    before the current format: legacy.prepare(df, path) and legacy.raw_flux."""
    import download_coordinator as dc
    import ztf_field_lookup as zl
    from make_catalog import make_catalog

    variants = {("_sci" if _is_sci(p) else ""): p for p in paths}
    first = next(iter(variants.values()))
    frames = {s: _load(p) for s, p in variants.items()}
    if legacy is not None and all(_is_current(df) for df in frames.values()):
        logger.info(f"SKIP {', '.join(p.name for p in variants.values())}: already in the current "
                    f"format — refresh it with update_parquet.py")
        return "current"
    if legacy is not None:
        frames = {s: legacy.prepare(df, variants[s], a.target_match_radius) for s, df in frames.items()}
        # epochs the conversion removed count as missing (measured again)
        kept = set.intersection(*(set(map(tuple, df[QKEY].assign(_m=df["OBSMJD"].round(6))
                                          .drop_duplicates().to_numpy().tolist()))
                                  for df in frames.values()))
    elif not all(_is_current(df) for df in frames.values()):
        logger.warning(f"SKIP {', '.join(p.name for p in variants.values())}: written before the "
                       f"current format (no FLUX_4_DIF / APCORR410) — use update_old_parquet.py")
        return "fail"
    have, meta = read_parquet_epochs(first)
    # an epoch counts as present only if every variant has it (else it is missing for
    # one: downloaded, and measured for that variant)
    for p in variants.values():
        if p != first:
            h2 = read_parquet_epochs(p)[0][QKEY + ["OBSMJD"]].assign(_m=lambda d: d["OBSMJD"].round(6))
            have = (have.assign(_m=have["OBSMJD"].round(6))
                    .merge(h2.drop(columns="OBSMJD"), on=QKEY + ["_m"]).drop(columns="_m"))
    if legacy is not None:
        have = have[np.array([k in kept for k in map(tuple, have[QKEY].assign(
            _m=have["OBSMJD"].round(6)).to_numpy().tolist())], dtype=bool)].reset_index(drop=True)
    ra, dec = target_position(first, meta)
    fcs = sorted(have["filtercode"].unique())
    if len(fcs) != 1:
        logger.warning(f"SKIP {first.name}: expected one band, found {fcs}")
        return "fail"
    fc = fcs[0]
    logger.info(f"═══ {', '.join(p.name for p in variants.values())}: RA={ra} Dec={dec} {fc} ═══")

    base = Path(tempfile.mkdtemp(prefix="ztfupd_", dir=a.tmpdir))
    keep = a.keep_temp                              # also kept for this target if it fails
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
        if a.dry_run or (len(todo) == 0 and not a.recalibrate):
            if len(todo) == 0:
                logger.info("  nothing to add")
                if not a.dry_run:
                    for path in variants.values():
                        _clear_deferred(path)
            return "ok"
        if len(todo) == 0:
            logger.info("  nothing to add — recalibrating with the current pipeline (--recalibrate)")

        # 2. references + catalogs, then only the missing/new epochs
        # References are essential: a quadrant missing them would lose its rows, so a
        # (transient) reference failure fails the whole target — nothing is written and
        # the next run retries it — rather than dropping the quadrant.
        src_dirs = _image_dirs(a, meta)
        if src_dirs:
            n_s, n_r = _reuse_local(dc, base, todo, quads_in, src_dirs)
            logger.info(f"  reused from {', '.join(map(str, src_dirs))}: {n_s}/{len(todo)} epoch "
                        f"image(s), {n_r} reference product(s) — not downloaded")
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
        old_pos = pd.concat([df[QKEY + ["ALPHAWIN_REF", "DELTAWIN_REF"]] for df in frames.values()])
        for q in quads:
            sel = old_pos[(old_pos["field"] == q[0]) & (old_pos["filtercode"] == q[1])
                          & (old_pos["ccdid"] == q[2]) & (old_pos["qid"] == q[3])]
            n_raw, n_inj = _complete_catalog(
                dc, base, q, sel[["ALPHAWIN_REF", "DELTAWIN_REF"]].to_numpy(float),
                (ra, dec), a.target_match_radius)
            if n_raw or n_inj:
                logger.info(f"  {q}: reference catalog completed with the parquet's sources — "
                            f"{n_raw} from refsexcat, {n_inj} injected")
        refs = {q: _ref_catalog(base, q) for q in quads}
        deferred, unavailable = _download_epochs(dc, todo, base, fc, a)

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
                                            a.target_match_radius,
                                            raw_flux=legacy.raw_flux if legacy else _raw_flux)
            logger.info(f"  {variants[s].name}: {n_cat} old-epoch catalogs from "
                        f"{len(used):,}/{len(df):,} rows (unmatched {int((df['_csv'] < 0).sum())})")
            old_epochs |= set(map(tuple, df[QKEY].assign(_m=df["OBSMJD"].round(6))
                                  .drop_duplicates().to_numpy().tolist()))

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
        phot = ["phot"] if "" in variants else ["phot_sci"]
        cmd = [sys.executable, str(_SCRIPTS / "run_pipeline.py"), "--base-dir", str(base),
               "--ra", str(ra), "--dec", str(dec), "--bands", fc[1:],
               "--steps", *phot, "vet", "calibrate", "flatfield", "recalibrate", "lightcurves", "merge",
               "--workers", str(a.workers), "--target-match-radius", str(a.target_match_radius),
               "--footprint-margin-px", str(a.footprint_margin_px)]
        cmd += ["--both"] if both else (["--scipos"] if "" not in variants else [])
        full = legacy is None and all("FLUX_10_DIF_uJy" in df for df in frames.values())
        cmd += ["--lc-columns", "full" if full else "slim"]
        logger.info("  running: " + " ".join(cmd[1:]))
        log = base / "run_pipeline.log"
        with open(log, "w") as fh:
            if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode != 0:
                logger.warning(f"FAIL: run_pipeline failed — see {log}")
                keep = True
                return "fail"

        # 5. assemble + write
        for s, path in variants.items():
            new, new_meta = _collect(base, ra, dec, fc, s)
            if new is None:
                logger.warning(f"FAIL {path.name}: no light curve produced")
                return "fail"
            out = _assemble(frames[s], new, refs, target, a.target_match_radius, old_epochs,
                            all_sources=a.all_sources or not catalog_changed)
            out_path = _output_path(path, a)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_meta[b"ztfphot_update"] = json.dumps({
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(path), "cuts": cuts, "cuts_recorded": recorded,
                "epochs_added": int(len(todo) - len(deferred) - len(unavailable)),
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
        if keep:
            logger.info(f"  (kept temp dir {base})")
        else:
            shutil.rmtree(base, ignore_errors=True)


def _output_path(path: Path, a) -> Path:
    """<stem><suffix>.parquet next to the input, or in --outdir; an input that is already
    an output (stem ends with the suffix) is refreshed in place. Pipeline-tree inputs
    (LightCurves/merged/{ra}_{dec}/{band}/lightcurves_merged.parquet) are named
    {ra}_{dec}_{band}_merged in --outdir, so targets, bands and variants stay distinct."""
    stem = path.stem
    if a.outdir and stem.startswith("lightcurves_merged"):
        stem = f"{path.parent.parent.name}_{path.parent.name}_merged" + stem[len("lightcurves_merged"):]
    if not stem.endswith(a.suffix):
        stem += a.suffix
    return (Path(a.outdir) if a.outdir else path.parent) / f"{stem}.parquet"


_THIS_COMMIT: list = []


def _made_by_this_code(path: Path) -> bool:
    """The parquet's ztfphot_run records the commit of the ZTFphot checkout running now."""
    if not _THIS_COMMIT:
        from run_pipeline import _git_info
        _THIS_COMMIT.append(_git_info().get("commit"))
    if _THIS_COMMIT[0] is None:
        return False
    try:
        run = json.loads((pq.read_schema(path).metadata or {}).get(b"ztfphot_run", b"null") or "null")
        return bool(run) and (run.get("ztfphot") or {}).get("commit") == _THIS_COMMIT[0]
    except (ValueError, OSError):
        return False


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
        outs = [_output_path(m, a) for m in members]
        done = all(o.exists() for o in outs)
        if a.from_updated and done:
            # refresh the existing outputs in place (they already hold every epoch); with
            # --recalibrate, outputs already made by this code version are skipped, so an
            # interrupted refresh resumes (--redo refreshes them anyway)
            if a.recalibrate and not a.redo and all(_made_by_this_code(o) for o in outs):
                logger.info(f"  skip {Path(gkey).name}: already refreshed by this code version")
                continue
            out.append(outs)
            continue
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


def main(legacy=None):
    p = argparse.ArgumentParser(description=legacy.__doc__ if legacy else __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--recalibrate", action="store_true",
                   help="also process targets with nothing to add, i.e. recalibrate them with "
                        "the current pipeline (only reference products are downloaded)")
    p.add_argument("--from-updated", action="store_true",
                   help="directory mode: where an _updated output exists, use it as the input "
                        "and refresh it in place (combine with --recalibrate to re-calibrate "
                        "outputs made by an older pipeline version without re-downloading)")
    p.add_argument("--retry-deferred", action="store_true",
                   help="directory mode: only targets whose earlier update deferred epochs; "
                        "re-run on their _updated files (downloads just those epochs)")
    p.add_argument("--ref-passes", type=int, default=2,
                   help="extra passes at the end of the batch for targets whose reference "
                        "products were unavailable (default 2)")
    p.add_argument("--ref-pass-wait", type=float, default=120.0, metavar="S",
                   help="pause before each extra pass (default 120 s; once per pass, not per target)")
    p.add_argument("--tmpdir", default=None, help="parent dir for the temporary base dir")
    p.add_argument("--image-dir", action="append", default=None, metavar="DIR",
                   help="pipeline base dir whose Science/ and Reference/ images are reused "
                        "instead of downloaded (repeatable); the base dir recorded in the "
                        "parquet metadata is also checked")
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
                st = update_target(g, a, auth, legacy)
            except (Exception, SystemExit) as exc:
                logger.exception(f"FAIL {g[0].name}: {exc}")
                st = "fail"
            status[str(g[0])] = st
            if st == "refs":
                requeue.append(g)
        queue = requeue
        if not queue:
            break
    n_cur = sum(st == "current" for st in status.values())
    bad = {n: st for n, st in status.items() if st not in ("ok", "current")}
    logger.info(f"done: {len(status) - len(bad) - n_cur}/{len(status)} target groups"
                + (f"; {n_cur} already current format (skipped)" if n_cur else "")
                + (f"; FAILED: {', '.join(f'{n} ({st})' for n, st in bad.items())}" if bad else "")
                + (" — reference failures are retried on the next run" if "refs" in bad.values() else ""))


if __name__ == "__main__":
    main()
