#!/usr/bin/env python3
"""
check_epoch_completeness.py
---------------------------
Compare the epochs photometered in a ZTF light-curve parquet against every
science epoch IRSA has for the same quadrant(s), and report the fraction present
plus a by-year breakdown of what's missing.

Uses IRSA's public TAP metadata (ztf.ztf_current_meta_sci) — no credentials.

The "possible" set can be narrowed to match your pipeline's download cuts with
--max-seeing / --min-maglim; hard-rejected infobits are always excluded (never
usable). Without those cuts, "missing" includes epochs the pipeline may have
deliberately skipped on quality, not just failures.

Usage:
    python check_epoch_completeness.py lightcurves.parquet
    python check_epoch_completeness.py --max-seeing 4.0 --min-maglim 19.5 *.parquet
"""
import argparse
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
HARD_REJECT = (1 << 0) | (1 << 1) | (1 << 25)      # 33554435 — never usable
CAUTIONARY = ((1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)
              | (1 << 11) | (1 << 21) | (1 << 22) | (1 << 26) | (1 << 27))  # pipeline --skip-flagged
MJD0 = pd.Timestamp("1858-11-17")                  # MJD epoch


def tap_query(adql, attempts=3):
    url = TAP + "?" + urllib.parse.urlencode(
        {"LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return pd.read_csv(r)
        except Exception as e:              # transient IRSA timeout/5xx — retry
            last = e
            if i < attempts - 1:
                time.sleep(5 * (i + 1))
    raise last


def mjd_year(mjd):
    return (MJD0 + pd.to_timedelta(np.asarray(mjd, float), unit="D")).year


def parse_target(path):
    """Target RA/Dec = first two '_'-separated tokens of the filename."""
    toks = os.path.basename(path).split("_")
    try:
        return float(toks[0]), float(toks[1])
    except (IndexError, ValueError):
        return None, None


def _present_by_year(pool, phot, tol):
    """Return (present_mask, missing_mjd) for photometered epochs `phot` against the
    possible-epoch pool `pool`, via 1:1 nearest-neighbour claim within `tol` days."""
    present = np.zeros(len(pool), dtype=bool)
    if len(phot) and len(pool):
        nearest = np.abs(pool[:, None] - phot[None, :]).argmin(axis=0)
        ok = np.abs(pool[nearest] - phot) < tol
        present[nearest[ok]] = True
    return present, pool[~present]


def quadrants_and_mjd(path):
    """Return (DataFrame of quadrants, DataFrame with field/fc/ccd/qid/OBSMJD).
    Handles merged parquets (quad columns present) and single-quadrant parquets
    (quad identity in parquet metadata)."""
    names = set(pq.read_schema(path).names)
    want = (["OBSMJD"] + [c for c in ("SEEING", "MAGLIM") if c in names]
            + [c for c in ("ALPHAWIN_REF", "DELTAWIN_REF", "object_index", "MAG_4_TOT_AB")
               if c in names])
    if {"field", "filtercode", "ccdid", "qid"} <= names:
        df = pd.read_parquet(path, columns=["field", "filtercode", "ccdid", "qid"] + want)
    else:
        meta = pq.read_schema(path).metadata or {}
        df = pd.read_parquet(path, columns=want)
        df["field"]      = int(meta[b"field"].decode())
        df["filtercode"] = meta[b"filtercode"].decode()
        df["ccdid"]      = int(meta[b"ccdid"].decode())
        df["qid"]        = int(meta[b"qid"].decode())
    quads = df[["field", "filtercode", "ccdid", "qid"]].drop_duplicates()
    return quads, df


def check(path, args):
    print(f"\n=== {path} ===")
    try:
        quads, df = quadrants_and_mjd(path)
    except Exception as e:
        print(f"  cannot read parquet: {e}")
        return

    tot_av = tot_ph = 0
    miss_year = {}
    possible = {}          # (field, fc, ccd, qid) -> possible-epoch MJD pool (post-cut)
    print("  -- general (all sources in quadrant) --")
    for r in quads.itertuples(index=False):
        f, fc, c, q = int(r.field), str(r.filtercode), int(r.ccdid), int(r.qid)
        m = ((df.field == f) & (df.filtercode == fc) & (df.ccdid == c) & (df.qid == q))
        phot = pd.to_numeric(df.loc[m, "OBSMJD"], errors="coerce").dropna().unique()

        # Effective quality cuts: explicit flags win; otherwise autodetect the envelope
        # of the epochs actually kept (worst seeing / faintest maglim among photometered
        # epochs), i.e. "of epochs at least as good as the ones I accepted, how many?".
        max_see, min_mag, auto = args.max_seeing, args.min_maglim, []
        if max_see is None and "SEEING" in df.columns:
            s = pd.to_numeric(df.loc[m, "SEEING"], errors="coerce").dropna()
            if len(s): max_see = float(s.max()); auto.append(f"seeing≤{max_see:.2f}")
        if min_mag is None and "MAGLIM" in df.columns:
            ml = pd.to_numeric(df.loc[m, "MAGLIM"], errors="coerce").dropna()
            if len(ml): min_mag = float(ml.min()); auto.append(f"maglim≥{min_mag:.2f}")

        adql = ("SELECT obsjd, seeing, maglimit, infobits FROM ztf.ztf_current_meta_sci "
                f"WHERE field={f} AND ccdid={c} AND qid={q} AND filtercode='{fc}'")
        try:
            meta = tap_query(adql)
        except Exception as e:
            print(f"  {f:06d}/{fc}/c{c:02d}/q{q}: IRSA query failed: {e}")
            continue
        if meta.empty:
            print(f"  {f:06d}/{fc}/c{c:02d}/q{q}: no IRSA epochs found")
            continue

        # Work on hard-reject-cleaned, de-duplicated IRSA epochs.
        M = meta.copy()
        M["mjd"] = M["obsjd"] - 2400000.5
        M["ib"] = M["infobits"].fillna(0).astype(np.int64)
        M = M[(M["ib"] & HARD_REJECT) == 0].drop_duplicates("mjd").reset_index(drop=True)
        mjd = M["mjd"].values

        # 1:1 claim: each photometered epoch marks its single nearest IRSA epoch
        # present (within tol). Avoids double-counting closely-spaced IRSA exposures.
        present = np.zeros(len(mjd), dtype=bool)
        if len(phot) and len(mjd):
            nearest = np.abs(mjd[:, None] - phot[None, :]).argmin(axis=0)
            ok = np.abs(mjd[nearest] - phot) < args.tol
            present[nearest[ok]] = True

        # Quality cuts define which UN-photometered epochs count as "possible"; an
        # epoch that was actually photometered is always kept and always present
        # (so header/metadata seeing differences never drop a real detection).
        keep = np.ones(len(mjd), dtype=bool)
        if args.skip_flagged:
            keep &= (M["ib"].values & CAUTIONARY) == 0
        if max_see is not None:
            keep &= pd.to_numeric(M["seeing"], errors="coerce").fillna(1e9).values <= max_see
        if min_mag is not None:
            keep &= pd.to_numeric(M["maglimit"], errors="coerce").fillna(-1e9).values >= min_mag
        keep |= present
        mjd, present = mjd[keep], present[keep]
        possible[(f, fc, c, q)] = mjd

        n_av, n_ph = len(mjd), int(present.sum())
        frac = (n_ph / n_av) if n_av else float("nan")
        tag = (" [auto: " + ", ".join(auto) + "]") if auto else ""
        print(f"  {f:06d}/{fc}/c{c:02d}/q{q}: {n_ph}/{n_av} photometered "
              f"({frac:.1%}) — {n_av - n_ph} missing{tag}")

        miss = mjd[~present]
        if len(miss):
            yrs = mjd_year(miss)
            for y in sorted(set(int(v) for v in yrs)):
                cnt = int((yrs == y).sum())
                print(f"        {y}: {cnt} missing")
                miss_year[y] = miss_year.get(y, 0) + cnt
        tot_av += n_av
        tot_ph += n_ph

    if len(quads) > 1 and tot_av:
        print(f"  --- total: {tot_ph}/{tot_av} ({tot_ph / tot_av:.1%}) — {tot_av - tot_ph} missing")
        for y in sorted(miss_year):
            print(f"        {y}: {miss_year[y]} missing")

    # ---- target-specific completeness ----
    print("  -- target --")
    tra, tdec = parse_target(path)
    need = {"ALPHAWIN_REF", "DELTAWIN_REF", "object_index", "MAG_4_TOT_AB"}
    if tra is None:
        print("    (could not parse target RA/Dec from filename)")
        return
    if not need <= set(df.columns):
        print("    (parquet lacks position/mag columns — cannot isolate target)")
        return

    pos = (df.groupby(["field", "filtercode", "ccdid", "qid", "object_index"])
             [["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna())
    if pos.empty:
        print("    (no usable source positions)")
        return
    pra = pd.to_numeric(pos["ALPHAWIN_REF"], errors="coerce").values
    pdc = pd.to_numeric(pos["DELTAWIN_REF"], errors="coerce").values
    sep = np.hypot((pra - tra) * np.cos(np.radians(tdec)), pdc - tdec) * 3600.0
    j = int(np.nanargmin(sep))
    f, fc, c, q, oi = pos.index[j]
    f, c, q = int(f), int(c), int(q)

    if sep[j] > args.target_radius:
        print(f"    TARGET ({tra:.5f},{tdec:+.5f}): NOT FOUND within {args.target_radius}\" "
              f"(nearest source {sep[j]:.1f}\") — no photometry")
        return

    tm = ((df.field == f) & (df.filtercode == fc) & (df.ccdid == c) & (df.qid == q)
          & (df.object_index == oi)
          & np.isfinite(pd.to_numeric(df["MAG_4_TOT_AB"], errors="coerce")))
    tphot = pd.to_numeric(df.loc[tm, "OBSMJD"], errors="coerce").dropna().unique()
    pool = possible.get((f, fc, c, q))
    if pool is None or not len(pool):
        print(f"    TARGET matched in {f:06d}/{fc}/c{c:02d}/q{q} but no possible-epoch pool")
        return
    _, tmiss = _present_by_year(pool, tphot, args.tol)
    n_av, n_ph = len(pool), len(pool) - len(tmiss)
    print(f"    TARGET ({tra:.5f},{tdec:+.5f}, {sep[j]:.2f}\" in {f:06d}/{fc}/c{c:02d}/q{q}): "
          f"{n_ph}/{n_av} epochs ({n_ph / n_av:.1%}) — {n_av - n_ph} missing")
    if len(tmiss):
        yrs = mjd_year(tmiss)
        for y in sorted(set(int(v) for v in yrs)):
            print(f"        {y}: {int((yrs == y).sum())} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--max-seeing", type=float, default=None,
                    help="seeing cut (default: autodetect from the parquet's SEEING)")
    ap.add_argument("--min-maglim", type=float, default=None,
                    help="maglim cut (default: autodetect from the parquet's MAGLIM)")
    ap.add_argument("--skip-flagged", action="store_true",
                    help="also exclude cautionary-infobit epochs (match pipeline --skip-flagged)")
    ap.add_argument("--target-radius", type=float, default=1.0,
                    help="arcsec: match the filename RA/Dec to a source for target stats (default 1.0)")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="MJD match tolerance in days (default 0.005 ~ 7 min)")
    args = ap.parse_args()
    for p in args.parquets:
        check(p, args)


if __name__ == "__main__":
    main()
