# ZTFphot

Difference-image aperture photometry pipeline for ZTF data. Given a sky position, it queries IRSA for field coverage, downloads science and reference images, runs forced aperture photometry (photutils) on the difference images, calibrates light curves per epoch, and assembles per-object light curves across multiple field/CCD/quadrant combinations.

---

## Requirements

- Python 3.10+
- [Miniconda or Mambaforge](https://github.com/conda-forge/miniforge)
- An [IRSA account](https://irsa.ipac.caltech.edu) (free) for image downloads

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/franzebauer/ZTFphot.git
cd ZTFphot
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate ztf
```

This installs Python, numpy, astropy, scipy, matplotlib, pandas, photutils, and ztfquery.

> **Note:** an existing env gets photutils with `conda install -n ztf -c conda-forge photutils`.

### 3. Configure IRSA credentials

IRSA credentials are required to download image products. An IRSA account is free at [irsa.ipac.caltech.edu](https://irsa.ipac.caltech.edu).

```bash
read -p "IRSA username: " IRSA_USER && read -sp "IRSA password: " IRSA_PASS && echo && \
printf 'machine irsa.ipac.caltech.edu\nlogin %s\npassword %s\n' "$IRSA_USER" "$IRSA_PASS" >> ~/.netrc && \
chmod 600 ~/.netrc && echo "Credentials written to ~/.netrc"
```

Then register them with ztfquery:

```bash
python -c "from ztfquery import io; io.set_account('irsa')"
```

### 4. Data directory

By default, all pipeline data is written to `data/` in your current working directory, created automatically on first use. To use a different location pass `--base-dir /path/to/data` to any command.

---

## Quick start

### Full pipeline from scratch

```bash
python ZTFphot/scripts/run_pipeline.py --ra 182.635755 --dec 39.405849
```

This runs all steps in order: field lookup → image download → reference catalog → photometry → vet → calibrate → flatfield → recalibrate → light curves → merge → plots. The `recalibrate` step is identical to `calibrate` but runs after `flatfield` so the spatial correction is applied; it force-overwrites the first pass automatically (see the step table).

### Common quality cuts for download (all optional)

```bash
python ZTFphot/scripts/run_pipeline.py --ra 182.635755 --dec 39.405849 \
    --max-seeing 3.0          \  # skip epochs with seeing FWHM > 3.0 arcsec
    --min-maglim 19.5         \  # skip epochs with 5σ limiting mag < 19.5
    --mjd-min 59000           \  # skip epochs before this MJD
    --mjd-max 60000           \  # skip epochs after this MJD
    --skip-flagged            \  # also skip cautionary-flagged epochs
    --min-epochs-per-quad 30
```

### Re-run specific steps on existing data

Steps are idempotent — existing outputs are skipped unless `--force` is passed.

```bash
# Re-calibrate and rebuild light curves for one quadrant
python ZTFphot/scripts/run_pipeline.py --steps calibrate lightcurves \
    --ra 182.635755 --dec 39.405849 \
    --field 443 --bands zg --ccdid 16 --qid 2 --force

# Check what's on disk
python ZTFphot/scripts/run_pipeline.py --status
```

### Recommended workflow for a new field

```bash
# 1. Lookup + download
python ZTFphot/scripts/run_pipeline.py --steps lookup download \
    --ra 182.635755 --dec 39.405849 --download-workers 50

# 2. Reference catalog and photometry
python ZTFphot/scripts/run_pipeline.py --steps catalog phot \
    --ra 182.635755 --dec 39.405849

# 3. Calibrate → flatfield → recalibrate → light curves
python ZTFphot/scripts/run_pipeline.py --steps calibrate flatfield recalibrate lightcurves \
    --ra 182.635755 --dec 39.405849 --force
```

### Low-disk mode

On systems with limited disk space, use `--purge-batch N` to process images in batches of N epochs. After each batch's photometry step, the large imaging products (difference images, `refimg.fits`) are deleted. The photometry catalogs (~5 MB/epoch), the reference CSV catalogs (`Catalogs/`), and the small reference source catalog (`refsexcat.fits`) are kept.

```bash
# Full pipeline, 5 epochs at a time (~4 GB peak disk usage per quadrant):
python ZTFphot/scripts/run_pipeline.py --ra 182.635755 --dec 39.405849 --purge-batch 10 --download-workers 50

# Preview what would be deleted without touching disk:
python ZTFphot/scripts/run_pipeline.py --ra 182.635755 --dec 39.405849 --purge-batch 10 --dry-run
```

After a full run (without `--purge-batch`), you can also clean up imaging products retroactively:

```bash
python ZTFphot/scripts/run_pipeline.py --clean-up          # delete all imaging products
python ZTFphot/scripts/run_pipeline.py --clean-up --dry-run  # preview first
```

**Typical disk budget** (10 quadrants, ~400 epochs each):

| Stage | Peak disk |
|-------|-----------|
| During `--purge-batch 5` | ~4 GB |
| Photometry catalogs (end state) | ~20 GB |
| Calibrated FITS (end state) | ~8 GB |
| **Total at completion** | **~28 GB** |

### Re-calibrating after a purge (no re-download needed)

Calibration reads no image pixels. Every input it needs survives `--purge-batch`/`--clean-up`:

- **Per-epoch photometry and headers** — the LDAC catalogs in `PhotCatalogs/` embed a full copy of the diff-image header (`OBSMJD`, `MAGZP_DIF`, `SEEING`, `MAGLIM`, `INFOBITS_DIF`, …) in their `LDAC_IMHEAD` extension, alongside the source fluxes. These are never purged.
- **Reference magnitudes** — the `Catalogs/*.csv` files (built once by the `catalog` step) hold `MAGZP_REF` and the aperture magnitudes; purge never touches `Catalogs/`.
- **Discovery key** — `refsexcat.fits` is retained through purge (it is small), so quadrants stay discoverable via the primary `Reference/` scan.

So to test changes to the calibration procedure, re-run **only** the calibration-and-later steps — never `catalog` or `phot`, which are the image-dependent steps:

```bash
python ZTFphot/scripts/run_pipeline.py --base-dir data \
    --steps calibrate flatfield recalibrate lightcurves \
    --field 509 --no-target --force
```

Quadrant discovery (`find_quadrants`) additionally falls back to `LightCurves/` and then to `PhotCatalogs/`, so a quadrant remains discoverable even if `refsexcat.fits` and the light-curve parquets were both removed. If `--status` reports quadrants but a re-run says "No quadrants found", make sure the retained `PhotCatalogs/` tree is present under `--base-dir`.

---

## Directory layout

```
ZTFphot/                    ← this repository
  scripts/
    run_pipeline.py         ← main entry point
    batch_pipeline.py       ← batch processing for lists of targets
    ztf_field_lookup.py
    download_coordinator.py
    photometry.py
    aperture_photometry.py
    make_catalog.py
    calib_catalogs.py
    calibrate.py
    lightcurves.py
    merge_fields.py
    vet_calibration_stars.py
    transient_catalog.py
    lc_viewer.py
    plot_rms.py
    plot_diagnostics.py
    plot_lightcurve.py
    plot_precision.py
    plot_spatial.py
  deprecated_scripts/       ← unmaintained one-off diagnostics and parquet fix-up tools

data/                       ← created in your working directory on first run
  Epochs/                   ← IRSA epoch cache and download logs
  Science/                  ← difference images per field/fc/ccd/qid
  Reference/                ← reference images and catalogs
  Catalogs/                 ← reference star CSV catalogs
  PhotCatalogs/             ← per-epoch photometry LDAC catalogs (ref-pos)
  PhotCatalogs_sci/         ← per-epoch photometry LDAC catalogs (sci-pos)
  Calibrated/               ← calibrated FITS catalogs per epoch (ref-pos)
  Calibrated_sci/           ← calibrated FITS catalogs per epoch (sci-pos)
  FlatfieldResiduals/       ← per-epoch NPZ residual maps (ref-pos)
  FlatfieldResiduals_sci/   ← per-epoch NPZ residual maps (sci-pos)
  LightCurves/              ← assembled light curves (parquet); both variants share this root
  Plots/                    ← diagnostic plots (ref-pos)
  Plots_sci/                ← diagnostic plots (sci-pos)
```

### Output file locations

| Output | Path |
|--------|------|
| Epoch metadata | `data/Epochs/lookup_{ra}_{dec}_{bands}.epochs.parquet` |
| Reference catalog | `data/Catalogs/{field}_{fc}_c{ccd}_q{qid}(REFERENCE)[OBJECTS].csv` |
| Photometry LDAC catalogs | `data/PhotCatalogs/000/{field}/{fc}/{ccd}/{qid}/*_phot.fits` |
| Calibrated FITS | `data/Calibrated/000/{field}/{fc}/{ccd}/{qid}/*_cal.fits` |
| NPZ residuals | `data/FlatfieldResiduals/000/{field}/{fc}/{ccd}/{qid}/*_resid.npz` |
| Flatfield map | `data/Calibrated/000/{field}/{fc}/{ccd}/{qid}/flatfield.npz` |
| Vet catalog | `data/Calibrated/000/{field}/{fc}/{ccd}/{qid}/vet_calib_stars.fits` |
| Light curves | `data/LightCurves/{field}/{fc}/ccd{ccd}/q{qid}/lightcurves.parquet` |
| Merged LCs | `data/LightCurves/merged/{ra}_{dec}/{fc}/*.parquet` |
| Plots | `data/Plots/{ra}_{dec}/{field}_{fc}_c{ccd}_q{qid}/*.png` |

---

## Pipeline steps

| Step | CLI name | Description |
|------|----------|-------------|
| Field lookup | `lookup` | Query IRSA for ZTF field/CCD/quadrant coverage at the target position |
| Download | `download` | Fetch science and reference images from IRSA. With `--ra/--dec`, the small reference products are fetched first and any quadrant whose target falls outside the reference detector area is **dropped before the costly science download** (see [Target coverage diagnostics](#target-coverage-diagnostics)) |
| Reference catalog | `catalog` | Build reference CSV from `refsexcat.fits`; checks target coverage (see [Target coverage diagnostics](#target-coverage-diagnostics)) |
| Photometry (ref-pos) | `phot` | Forced aperture photometry on the difference images at the reference positions (plus the target if absent from the reference catalog). Writes `PhotCatalogs/…_phot.fits` |
| Photometry (sci-pos) | `phot_sci` | Forced aperture photometry at the per-epoch science-catalog (`*_sexcat.fits`) positions, tied to the reference within `--assoc-radius-sci`. Writes `PhotCatalogs_sci/…_phot.fits`. Runs with `--both` / `--scipos` |
| Vet | `vet` | Flag variable/bad calibration stars by multi-epoch RMS |
| Calibrate | `calibrate` | Linear ZP → 3σ clip → faint correction → 2D polynomial → flatfield |
| Flatfield | `flatfield` | Rebuild spatial flatfield from post-polynomial residuals |
| Recalibrate | `recalibrate` | Second calibration pass; identical to `calibrate` but runs after `flatfield` to apply the spatial correction. Because it shares output paths with `calibrate`, it **always force-overwrites** them (no `--force` needed) — otherwise the existing `_cal.fits` would be skipped and the flatfield never applied. |
| Light curves | `lightcurves` | Assemble per-object parquet light curves from calibrated FITS |
| Merge | `merge` | Cross-calibrate and merge multiple quadrants per band |
| Plots | `plots` | Diagnostic plots (spatial residuals, RMS, precision, light curves). The light-curve plot overlays a binned series (open squares, `--lc-bin-days`, default 50 d) on the target and on each comparison star. Generates a placeholder if the target is not found (see [Target coverage diagnostics](#target-coverage-diagnostics)) |

---

## Key CLI options

### Target and scope

| Option | Description |
|--------|-------------|
| `--base-dir PATH` | Data directory (default: `./data`) |
| `--ra / --dec` | Target RA/Dec (deg) — required for lookup, download, merge, and plots - forced if no match |
| `--bands zg zr zi` | Bands to process (default: all present on disk) |
| `--field N` | Restrict to a specific ZTF field number |
| `--ccdid N` | Restrict to a specific CCD |
| `--qid N` | Restrict to a specific quadrant |
| `--steps STEP ...` | Run only named steps (default: all) |
| `--workers N` | Parallel workers for photometry/calibrate (default: 4) |
| `--download-workers N` | Parallel threads for image downloads (default: 50) |
| `--retry-404` | Ignore `Epochs/permanent_404s.log` and request every file again |
| `--perm404-max-age DAYS` | A recorded HTTP 404 is skipped for this many days, then retried (default 30; 0 = never expire). Only real 404s are recorded — an HTML response (IRSA error/maintenance page) is a retriable failure. Log entries without a timestamp count as expired |
| `--force` | Re-run even if output files already exist (does **not** bypass the 404 log — use `--retry-404`) |
| `--verbose` | Increase log verbosity |
| `--status` | Print file-existence summary and exit |
| `--no-target` | Suppress target-source plots (`make_lightcurves`, target marker in precision plot). Use when processing a whole quadrant without a specific target of interest. |
| `--footprint-margin-px PIX` | Reference-image edge margin for the footprint gate (default: 3 px ≈ the 4 px primary-aperture radius). A quadrant is dropped before its science download if the target lands within this many pixels of the `refimg` edge (or outside it). Larger = more conservative against edge artefacts; see [Target coverage diagnostics](#target-coverage-diagnostics). |
| `--lc-bin-days DAYS` | Bin width (days) for the binned overlay in the light-curve plot (default: 50). |

### Download quality cuts

Applied when `download` is in `--steps`. All optional; default is no cuts.

| Option | Description |
|--------|-------------|
| `--max-seeing ARCSEC` | Skip epochs with seeing FWHM above this value |
| `--min-maglim MAG` | Skip epochs with 5σ limiting magnitude below this value |
| `--mjd-min MJD` | Skip epochs before this MJD |
| `--mjd-max MJD` | Skip epochs after this MJD |
| `--skip-flagged` | Skip cautionary-flagged epochs (in addition to hard-rejected ones) |
| `--min-epochs-per-quad N` | Skip quadrants with fewer than N epochs after filtering |

### Calibration options

| Option | Description |
|--------|-------------|
| `--poly-degree N` | Degree of the spatial calibration polynomial (default: 2) |
| `--ff-bins N` | Number of spatial bins per axis for the flatfield grid (default: 16) |
| `--ff-min-count N` | Minimum detections per flatfield bin to use (default: 5) |
| `--ff-edge-split N` | Subdivide the outer flatfield bin on each axis into N thinner bins — elongated edge cells (full width along the edge, 1/N deep perpendicular) to resolve the steep vignetting gradient at the field boundary. Default 3; 1 = uniform grid |
| `--faint-err-max E` | Max magnitude error for sources that define the per-bin faint correction (default 0.5). Near MAGLIM a tight cut keeps only high-SNR sources — a different population from the one the correction is applied to, which under-corrects those bins. Loosen (e.g. 1.0) so the correction is built from the population it corrects |
| `--vet-catalog PATH` | Path to a vet catalog FITS file (overrides the default location in `Calibrated/`) |
| `--target-match-radius ARCSEC` | Target association radius (default: 1.0 arcsec). One radius for two coupled operations: the target is **injected** as a new source if no catalog source lies within it, and the input RA/Dec is **matched** to a source in the calibrated catalog within it. Because a target with no nearby reference source is injected (at the input position) and then recovered by the match, this radius never causes a spurious "target not found". |
| `--aper-bkg {box,local,mesh,none}` | Aperture-photometry background: `box` (default) = 3σ-clipped mean in a square annulus (1.58×SEEING to +35 px) computed with integral images; `local` = clipped median in the equivalent circular annulus (same precision, ~2.5× slower photometry); `mesh` = Background2D level (`--aper-bkg-mesh`, default 64 px; degrades i band — fringing); `none` |
| `--assoc-radius-sci ARCSEC` | Radius tying each **sci-pos** science-catalog source to its reference entry (default: 1.5; allows epoch-to-epoch science-centroid shifts) |
| `--merge-match-radius ARCSEC` | Cross-match radius for common sources between overlapping quadrants in the merge step (default: 1.5) |
| `--inject-targets CSV` | CSV of targets (`ra,dec[,name]`) to inject as sources absent from the reference catalog. In the `catalog` step, targets inside each quadrant footprint and farther than `--target-match-radius` from any reference source are added to a non-destructive `*_refsexcat_augmented.fits` that `make_catalog` and the photometry step then prefer automatically. Injected sources appear in the light curves with `MAG_4_REF > 90`. Forces the `catalog` step; use `--force` so `phot`/`calibrate`/`lightcurves` also rebuild if outputs already exist. |
| `--grabTNSobjs` | Inject real transients from the TNS public bulk catalog that overlap the processed quadrants (same injection path as `--inject-targets`; may be combined with it). Requires `--tns-cred-file` (JSON with `tns_id`/`type`/`name`) and `--tns-key-file` (API key). The bulk catalog is cached under `--tns-cache-dir` (default `<base-dir>/TNS`); pass `--tns-force-download` to refresh. |
| `--merge-mag-bin W` | Magnitude-bin width (mag) for the per-magnitude quadrant cross-calibration in the merge step (default: 0.1; a very large value reduces to a single scalar offset) |

### Photometry variant

| Option | Description |
|--------|-------------|
| `--scipos` | Run calibrate/lightcurves/plots using sci-pos photometry (reads `PhotCatalogs_sci/`, writes `Calibrated_sci/`, `Plots_sci/`) |
| `--both` | Run photometry for both ref-pos and sci-pos (sharing the same diff images), then run calibrate/lightcurves/plots for both variants in sequence |

With `--both` and `--purge-batch`, sci-pos steps run within each batch before the diff images are deleted.

### Disk management

| Option | Description |
|--------|-------------|
| `--purge-batch N` | Low-disk mode: process N epochs at a time, deleting imaging products after each batch |
| `--clean-up` | Delete all imaging products (Science/, Reference/) and exit |
| `--dry-run` | Preview what `--purge-batch` or `--clean-up` would delete, without touching disk |
| `--purge-hard-reject` | Delete on-disk files for hard-rejected epochs and exit |
| `--epochs-parquet PATH` | Path to epoch cache parquet (used with `--purge-hard-reject`) |

---

## Calibration sequence

Within `calibrate`, each epoch is corrected in five stages:

1. **Linear ZP fit** — per-epoch magnitude offset + slope fit against calibrators (14–19 mag, CLASS_STAR ≥ 0.7, FLAGS=0, err < 0.3 mag)
2. **3σ iterative clip** — outlier calibrators removed, fit repeated
3. **2D polynomial** — low-order spatial polynomial fitted to calibrator residuals and applied to all sources
4. **Spatial flatfield** — stacked median residual map applied if a saved flatfield exists on disk. The grid spans the full source range (ref-pos positions are exact, so no percentile clipping) with **elongated edge cells** (`--ff-edge-split`, outer bins subdivided perpendicular to the edge) to resolve the steep vignetting gradient at the boundary. Query positions are clamped to the grid-centre range, so the outer half-cell receives the nearest cell's correction instead of zero.
5. **Faint correction** — smoothed empirical offset for faint sources, applied **last** (on the residuals surviving all spatial corrections). The all-source residual is binned in 0.25 mag steps over 18.5–22 with a 3σ-clipped median per bin. The correction curve is 0 at 18.5 mag, rises linearly (taper) to the empirical profile starting at the 19.0–19.25 bin, follows the bin medians to the last bin, and is held flat out to 24 mag; the whole curve is then Gaussian-smoothed at σ = 0.2 mag. Sources below 18.5 mag (the 14–19 mag calibrators, and hence the poly/flatfield fits built from them) are left untouched.

> The faint correction runs after the flatfield so it removes the magnitude-dependent offset that remains in faint sources *after* the spatial corrections, rather than feeding it into the poly/flatfield fits. The spatial fits are built from bright calibrators the faint ramp barely touches.

---

## Optional: calibration star vetting

Bad reference stars inflate calibration scatter. After an initial calibration run:

```bash
python ZTFphot/scripts/vet_calibration_stars.py \
    --field 443 --band zg --ccd 16 --qid 2

python ZTFphot/scripts/run_pipeline.py --steps calibrate flatfield recalibrate lightcurves plots \
    --field 443 --bands zg --ccdid 16 --qid 2 --force
```

The vet catalog is discovered automatically from its standard location in `data/Calibrated/`.

---

## Light curve parquet columns

Two variants are produced per quadrant, sharing the same `LightCurves/` root but with different filenames:

| File | Positions used | Epoch position column |
|------|----------------|-----------------------|
| `lightcurves.parquet` | Reference catalog (fixed) | — (forced positions) |
| `lightcurves_sci.parquet` | Per-epoch science sexcat | `ALPHA_SCI`, `DELTA_SCI` |

Both files share the same `object_index` and can be joined on `(object_index, OBSMJD)`. Sources not detected in the science image for a given epoch are absent from `lightcurves_sci.parquet` for that epoch (no forced photometry).

### Per-quadrant (`LightCurves/{field}/{fc}/ccd{ccd}/q{qid}/lightcurves.parquet`)

#### Source identity (reference catalog)
| Column | Type | Description |
|--------|------|-------------|
| `ALPHAWIN_REF` | float64 | Reference catalog RA (deg) |
| `DELTAWIN_REF` | float64 | Reference catalog Dec (deg) |
| `object_index` | int32 | 0-based index into the reference catalog |
| `MAG_4_REF` | float32 | 4 px reference AB magnitude of the source (`MAG_APER_4px + MAGZP_REF`). Enables a per-source measured-vs-reference calibration check from the parquet alone. Injected/target sources have no reference: injected list targets carry `> 90` (sentinel `MAG_APER = 99`); the `--ra/--dec` target carries `NaN`. |

#### Per-detection measurement
| Column | Type | Description |
|--------|------|-------------|
| `CLASS_STAR_OBJ` | float32 | Reference-catalog star/galaxy classifier (0=galaxy, 1=star) |
| `FLAGS` | Int32 | **The** quality bitmask (`scripts/ztfphot_flags.py`). Per epoch: 1 neighbour within 5 px, **8 the 4 px aperture crosses the image edge**, **16 masked/saturated pixels in the 4 px aperture**, 32 edge/masked only in the larger 6/10 px apertures. Per source: 256 reference neighbour/blend (IRSA refsexcat FLAGS ≠ 0), 1024 no reference source (`FLUX` is difference flux). Summaries: 2048 aperture flags unknown (epoch rebuilt from an older parquet by `update_parquet.py`), 4096 cautionary ZTF INFOBITS this epoch, 8192 reference image has INFOBITS. Only **8 and 16 corrupt the 4 px measurement** → default rejection: `from ztfphot_flags import good; good(lc)`; add bits as needed, e.g. `good(lc, reject=F.DEFAULT_REJECT \| F.CAUTIONARY_EPOCH \| F.REF_BLEND)`. `good()` also reads parquets that carry `FLAG_AP` / `FLAG_SE_REF` columns instead. The reference-image INFOBITS are file metadata `INFOBITS_REF_{tag}` |
| `MAG_3_TOT_AB` | float32 | Calibrated AB mag, 3 px diameter aperture |
| `MERR_3_TOT_AB` | float32 | Magnitude error, 3 px aperture |
| `MAG_4_TOT_AB` | float32 | Calibrated AB mag, 4 px diameter aperture (primary), **aperture-corrected to 10 px** (`APCORR410`; add it back for the 4 px value). **NaN when total flux ≤ 0** (fainter-than-reference epoch) — use `FLUX_4_TOT_AB` there |
| `MERR_4_TOT_AB` | float32 | Magnitude error, 4 px aperture |
| `FLUX_4_TOT_AB` | float32 | **Bipolar** total flux (μJy), 4 px aperture aperture-corrected to 10 px (same scale as `MAG_4_TOT_AB`) — retains the negative (fainter-than-reference) epochs the magnitude drops to NaN. For a reference-less target (`target_ref_mag` = NaN) this is the *difference* flux, not total |
| `FERR_4_TOT_AB` | float32 | Flux error (μJy), 4 px aperture |
| `FLUX_6_TOT_AB` | float32 | Bipolar total flux (μJy), 6 px aperture (carried for the aperture-correction check) |
| `MAG_6_TOT_AB` | float32 | Calibrated AB mag, 6 px diameter aperture |
| `MERR_6_TOT_AB` | float32 | Magnitude error, 6 px aperture |
| `MAG_10_TOT_AB` | float32 | Calibrated AB mag, 10 px diameter aperture |
| `MERR_10_TOT_AB` | float32 | Magnitude error, 10 px aperture |
| `MAG_4_TOT_AB_org` | float32 | Raw (pre-calibration, not aperture-corrected) 4 px magnitude |
| `MERR_4_TOT_AB_org` | float32 | Raw magnitude error, 4 px aperture |

#### Per-epoch metadata (broadcast from image header)
| Column | Type | Description |
|--------|------|-------------|
| `OBSMJD` | float64 | Observation MJD |
| `AIRMASS` | float32 | Airmass at time of observation |
| `SEEING` | float32 | Seeing FWHM (arcsec) |
| `MAGLIM` | float32 | 5σ limiting magnitude |
| `MAGZP_DIF` | float32 | ZTF photometric zero point |
| `MAGZPRMS_DIF` | float32 | Zero point RMS |
| `CLRCOEFF` | float32 | Color coefficient (already applied internally by ZTF) |
| `NMATCHES` | int32 | Number of calibration stars used |
| `INFOBITS_DIF` | int32 | ZTF's raw per-epoch image quality bits, verbatim (see ZTF documentation); summarised in `FLAGS` bit 4096 |
| `APCORR410` | float32 | Aperture correction from 4 px to 10 px (mmag), median `MAG_APER_4px − MAG_APER_10px` of reference calibrators; subtracted from `MAG_4_TOT_AB` (flux × 10^(0.4·APCORR410/1000)) |

File-level metadata keys: `field`, `filtercode`, `ccdid`, `qid`, `MAGZP_REF_{tag}`, `MAGZPRMS_REF_{tag}`, `target_sep_arcsec` (arcsec distance from target to the nearest reference catalog source; `nan` if `--ra`/`--dec` not provided), `target_ref_mag` (reference AB mag of the target's matched source, or `nan` if the target has **no** reference detection — see [Target coverage diagnostics §2](#2-target_sep_arcsec--target_ref_mag-metadata-in-the-parquet)).

### Per-quadrant sci-pos (`LightCurves/{field}/{fc}/ccd{ccd}/q{qid}/lightcurves_sci.parquet`)

Identical schema to the ref-pos file above, with one addition:

| Column | Type | Description |
|--------|------|-------------|
| `ALPHA_SCI` | float64 | Science-image RA of detection (deg) |
| `DELTA_SCI` | float64 | Science-image Dec of detection (deg) |

---

### Merged (`LightCurves/merged/{ra}_{dec}/{fc}/lightcurves_merged.parquet`)

Contains all per-quadrant columns above, with `norm_offset` already applied to all `MAG_*` columns (and as a factor 10^(−0.4·norm_offset) to `FLUX_4_TOT_AB`, `FERR_4_TOT_AB`, `FLUX_6_TOT_AB`) so that magnitudes from different field/ccdid/qids are on a common photometric scale. Plus:

| Column        | Type    | Description |
|---------------|---------|-------------|
| `field`       | int64   | ZTF field number |
| `filtercode`  | str     | Filter code (`zg`, `zr`, `zi`) |
| `ccdid`       | int64   | CCD number |
| `qid`         | int64   | Quadrant number |
| `norm_offset` | float32 | Per-row magnitude-dependent correction (mag) applied to all `MAG_*` columns to bring this quadrant onto the dominant photometric scale; 0.0 for dominant-quadrant rows |

**`object_index` in the merged file is globally unique across quadrants.** Each secondary quadrant's indices are offset above the dominant quadrant's block so no two different physical stars share the same index. Sources detected in multiple quadrants (spatial overlap) are additionally coordinate-matched at 1.5 arcsec and remapped to the dominant quadrant's `object_index`, so the same physical star has a single index throughout the merged file.

The cross-calibration measures the per-source median-magnitude difference between each secondary quadrant and the dominant quadrant (`dominant − secondary`, on `MAG_4_TOT_AB`, matched within 1.5 arcsec, ≥5 clean epochs in both quadrants, ≥10 common sources required) **as a function of magnitude**. The difference is binned by the dominant magnitude in bins of width `--merge-mag-bin` (default: 0.1 mag), with a 3σ MAD clip per bin (≥5 sources required per bin), forming a correction curve. The curve is applied to every secondary row by interpolating it at that row's `MAG_4_TOT_AB` (flat extrapolation beyond the populated range). It falls back to a single scalar median offset when fewer than two bins are populated. Inter-quadrant offsets in this data set are magnitude-dependent (driven by the per-epoch science calibration of shallow, few-epoch quadrants), not spatial, and a scalar or spatial fit left a residual tilt across magnitude.

Quadrant discovery (`find_quadrants`) is primarily driven by `Reference/*_refsexcat.fits` (retained through purge), then supplements from `LightCurves/.../lightcurves*.parquet`, then from `PhotCatalogs/.../*_phot.fits`, so quadrants whose reference products were purged are still picked up by the merge (and `--status`). See [Re-calibrating after a purge](#re-calibrating-after-a-purge-no-re-download-needed).

File-level metadata includes `MAGZP_REF_{tag}`, `MAGZPRMS_REF_{tag}` and `INFOBITS_REF_{tag}` for each contributing quadrant, per-quadrant copies `target_sep_arcsec_{tag}`, `target_ref_mag_{tag}`, `ztfphot_run_{tag}`, `ztfphot_download_{tag}`, plus `dominant_quadrant` (e.g. `000521_zg_c15_q3`) identifying the photometric reference quadrant and `flux_norm_applied`. The top-level `field`/`filtercode`/`ccdid`/`qid`, `ztfphot_run` and `ztfphot_download` are those of the dominant quadrant; the top-level `target_sep_arcsec`/`target_ref_mag` those of the quadrant whose nearest reference source is closest to the target.

**Run provenance** (JSON strings in the file metadata, per-quadrant and merged files):

| Key | Content |
|-----|---------|
| `ztfphot_run` | The invocation that built the light curves: ZTFphot git commit (+ `dirty` if uncommitted changes), UTC time, full `argv`, every CLI option, Python/package versions, host |
| `ztfphot_download` | The most recent logged invocation that ran `download` (or `--purge-batch`) for the same target — its options are the epoch selection cuts (`min_maglim`, `max_seeing`, `skip_flagged`, MJD range, …), which a later calibrate/lightcurves-only run does not repeat. `null` if no such run was logged |
| `epoch_summary_{tag}` | From the data, per quadrant: `n_epochs`, `mjd_min`/`mjd_max`, `maglim_min` and `seeing_max` (difference-image header values), `n_infobits_set` |

Every `run_pipeline` invocation is appended to `<base-dir>/Epochs/ztfphot_runs.jsonl`, which is where `ztfphot_download` is looked up. Read them with e.g. `json.loads(pq.read_schema(path).metadata[b"ztfphot_download"])["args"]["min_maglim"]`. `check_missing_epochs.py` (LC tools) uses `ztfphot_download` automatically; for parquets without it, `--infer-cuts` derives the cuts from the IRSA maglimit/seeing range of the epochs present.

---

## Calibrated FITS header keywords

| Keyword | Description |
|---------|-------------|
| `OBSMJD` | Observation MJD |
| `SEEING` | Seeing FWHM (arcsec) |
| `MAGLIM` | 5σ limiting magnitude |
| `num_stars` | Number of calibration stars used |
| `NC_RMS0` | Calibrator RMS before any correction (mmag) |
| `NC_RMS1` | Calibrator RMS after linear ZP fit (mmag) |
| `NC_RMS2` | Calibrator RMS after 3σ clip (mmag) |
| `NC_RMS3` | Calibrator RMS after 2D polynomial (mmag) |
| `NC_RMS4` | Calibrator RMS after spatial flatfield (mmag) |
| `NC_RMSFC` | Calibrator RMS entering the 2D polynomial (≈ after 3σ clip) (mmag) |
| `CALIB_N` | Linear fit intercept (mag) |
| `CALIB_M` | Linear fit slope |
| `CALIB_ZP` | ZP correction evaluated at mag 17 (mag) |
| `NC_FC_00–06` | Per-bin faint correction at mag 18.75–21.75 (mmag; −999 if empty bin) |
| `TGT_MRAW` | Raw instrumental mag of target (mmag; −999 if not detected) |
| `TGT_DCLIN` | Linear ZP correction applied to target (mmag) |
| `TGT_DCPOL` | 2D polynomial correction applied to target (mmag) |
| `TGT_DCFF` | Flatfield correction applied to target (mmag) |

---

## Aperture photometry

`aperture_photometry.py` measures forced circular apertures with photutils directly on each difference image; there is no detection step.

- **Positions**: ref-pos = every reference-catalog source (plus the `--ra/--dec` target as index N+1 when no reference source lies within `--target-match-radius`); sci-pos = every source of the epoch's ZTF `*_sexcat.fits`, tied to the nearest reference entry within `--assoc-radius-sci` (unmatched dropped). `VECTOR_ASSOC`/`object_index` is the 1-based reference index.
- **Which sources**: centre on the frame and > 50% finite pixels within 0.75×SEEING.
- **Apertures**: diameters 3, 4, 6, 10 px (radii 1.5, 2, 3, 5), exact pixel overlap; masked (NaN) pixels excluded. Primary aperture k=1 (4 px) → `MAG_4_TOT_AB` / `FLUX_4_TOT_AB`.
- **Background** (`--aper-bkg`): `box` (default) = 3σ-clipped mean in a square annulus from 1.58×SEEING to +35 px, for all sources at once via integral images (pixels clipped against the 64 px background level/RMS first); `local` = clipped median in the circular annulus. Tested on J1025 (642 epochs, g/r/i): per-star σ identical to `local` in g/r (ratio 1.000) and within 0–1.2% in i; photometry 2.5× faster. A 64 px `mesh` matches in g/r but degrades i (fringes: +6% faint σ, +20 mmag faint bias).
- **Errors**: background RMS (64 px `Background2D` mesh, evaluated at the source positions) × √(unmasked aperture area) (background-only error).
- **Per-epoch flags** (bits 0–7 of the parquet `FLAGS`): 1 = another catalog source within 5 px, 8 = the 4 px aperture crosses the image edge, 16 = masked (NaN) pixels in the 4 px aperture, 32 = edge/masked pixels only in the larger 6/10 px apertures. Calibration rejects calibrator measurements with 8, 16 or 32 (all apertures are calibrated) and requires the calibrator to be clean in the reference (`FLAG_SE_REF == 0` in the reference catalog). On est_084892 0.1% edge, 0.3% masked, 0.6% outer-only.
- **Output**: LDAC catalogs `PhotCatalogs/…/*_phot.fits` (`LDAC_IMHEAD` = difference-image header, `LDAC_OBJECTS` with `FLUX_APER(4)`, `FLUXERR_APER(4)`, `VECTOR_ASSOC`, …), written to a temp file and renamed, so a crash cannot leave a partial catalog. Directories `SExCatalogs{,_sci}/` and files `*_sexout.fits` found under `--base-dir` are renamed to this layout at startup.

---

## Target coverage diagnostics

ZTF field/CCD/quadrant assignments are based on nominal grid footprints, and ztfquery returns every quadrant *near* the target — not only those that contain it. Occasionally a target that nominally falls within a quadrant's bounding box actually lands outside the active **reference** detector area — in a CCD gap or right at the chip boundary. The pipeline detects and handles this at four levels.

### 1. Reference-footprint gate before the science download

Whenever `--ra/--dec` is given, the pipeline fetches each candidate quadrant's small reference products first, opens `refimg.fits`, converts the target RA/Dec to pixel coordinates via the WCS, and drops any quadrant where the target lands outside the active area — within `--footprint-margin-px` pixels of the edge (default 3 ≈ the 4 px aperture radius):

```
000782_zi_c07_q3: target pixel (1420, 3277) outside active area [3:3069, 3:3077] — dropping quadrant
000782/zi/ccd07/q3: target outside reference footprint — skipping quadrant
```

This runs in **both** the normal and `--purge-batch` download paths, **before any science images are fetched**, so an out-of-footprint quadrant costs neither the (bulk) science download nor spurious photometry. The discriminator is deliberately the *reference* footprint: individual science epochs dither and can cover a target that the reference coadd — which defines where sources are actually measured — does not. The same WCS check also re-runs at startup on any quadrants already on disk. No manual intervention is needed.

### 2. `target_sep_arcsec` / `target_ref_mag` metadata in the parquet

Every per-quadrant light curve parquet records target metadata as file-level keys:

```python
import pyarrow.parquet as pq
meta = pq.read_schema("lightcurves.parquet").metadata
print(meta[b"target_sep_arcsec"])   # nearest ref source to target, arcsec, e.g. b"117.968"
print(meta[b"target_ref_mag"])      # reference AB mag of the target, or b"nan" if NONE
```

**You must check `target_ref_mag`.** If it is `nan`, the target has **no reference-image
detection** — so its `FLUX_4_TOT_AB` is the **bipolar *difference* flux** (`flux_ref = 0`,
total = difference), *not* a total flux, and the magnitude columns are NaN whenever that
difference is negative. Such a target may also be **blended** with a nearby reference
source (its flux then contaminated by that source's subtraction residual). To recover a
clean light curve you would need the correct deblended reference magnitude and to confirm
the blended neighbour is non-variable. Batch runs also surface this: the target's
`sources_status.csv` reason is **`LACKS_REFIMG_DETECTION`** (rather than `OK`).

A value much larger than the typical inter-source spacing (~5–10 arcsec in a ZTF field) indicates the target position is in an unpopulated or masked region that the WCS check did not catch (e.g. a vignetted corner with valid pixel coordinates but no real detections).

### 3. Placeholder lightcurves plot

When the plots step cannot find any parquet source within 3 arcsec of the target, instead of producing no file it writes a red-text placeholder:

```
lightcurves_000396_zg_c15_q1.png   ← "Target not found: nearest source 117.9" away — likely masked/edge pixels"
```

This makes the failure **visible in the plots directory** rather than requiring users to notice a missing file.

### 4. Per-target reason codes (`target_status.csv` / `sources_status.csv`)

For a `--ra/--dec` run the pipeline records **why** the target did or did not get photometry. The photometry step (`phot`) writes each epoch's target fate under `TargetStatus/` (a directory `--purge-batch` never deletes), and at the end of the run `<base-dir>/target_status.csv` is written as one line `ra,dec,reason` with the coarse, best-across-quadrants reason:

| Reason | Meaning |
|--------|---------|
| `OK` | target has a light curve — ≥1 finite **flux** measurement, **positive or negative** (a fainter-than-reference detection is still a detection; difference photometry is bipolar) — *and* has a reference detection |
| `LACKS_REFIMG_DETECTION` | has a flux light curve but **no reference source** (`target_ref_mag` = NaN) — `FLUX_4_TOT_AB` is bipolar *difference* flux, not total; check for blends (see [§2](#2-target_sep_arcsec--target_ref_mag-metadata-in-the-parquet)) |
| `MATCHED_NEIGHBOR` | within `--target-match-radius` of a catalog source (photometered as that source) |
| `NOT_RECOVERED` | measurable (in frame, unmasked) in ≥1 epoch but no row reached the light curve |
| `ON_MASK` | in-frame but never measurable — `_valid_frac ≤ 0.5` every epoch (target pixel persistently masked) |
| `OUTSIDE_FOOTPRINT` | never in-frame in any epoch |
| `NO_EPOCHS` | no epochs were measured |

`batch_pipeline.py` collects these into `<results-dir>/sources_status.csv` (one `ra,dec,reason` line per target), so a whole batch's good/failed targets — and the reason for each failure — are summarised in one file.

---

## Standalone utilities

These scripts are not wired into `run_pipeline.py` but can be run directly from the `scripts/` directory (unmaintained one-off tools are in `deprecated_scripts/`):

| Script | Function |
|--------|----------|
| `transient_catalog.py` | Augments the reference catalog with additional sources (e.g. from TNS or a user CSV) before the photometry step. Needed when the target brightened after the ZTF reference epoch and is therefore absent from the reference catalog. Pass `--refsexcat` and `--input` (or `--ra/--dec` for a TNS cone search) to produce an augmented `*_refsexcat_augmented.fits` that the pipeline then prefers. |
| `update_parquet.py` | Adds missing (and, unless `--no-new`, newer) epochs to existing light-curve parquets **without re-downloading the epochs already in them**, recalibrating all photometry with the current pipeline. Inputs must be in the current format (`FLUX_4_DIF`, `APCORR410`); older files are refused — use `update_old_parquet.py`. A ref-pos parquet and its `_sci` sibling are updated together. Epoch cuts come from the `ztfphot_download` metadata, or `--infer-cuts`. The old epochs are rebuilt as per-epoch catalogs from `FLUX_4_DIF`; the rebuilt reference catalog is completed with every source of the old parquet it lacks (from refsexcat, or as an injected source), so no light curve is lost; then vet → calibrate → flatfield → recalibrate → lightcurves → merge run over old + new epochs. Old rows get the new 4 px values with MAG_3/6/10 (and FLUX_6) shifted by the same 4 px change (before the 4 px aperture correction); new epochs have all apertures. Images already on disk are reused instead of downloaded: `--image-dir DIR` (repeatable) and the `base_dir` recorded in the parquet metadata. Writes `<stem>_updated.parquet` (+ `ztfphot_update` metadata). Validated on est_084892: restoring 5 removed epochs reproduces a fresh run row for row in all apertures (≤0.004 mmag). Run: `python ZTFphot/scripts/update_parquet.py X_zg_merged.parquet [--dry-run]`, or on a whole directory: `update_parquet.py RESULTS_DIR [--plan] [--recursive] [--redo]` — per target/band/variant it uses `X_recal.parquet` if present, else `X.parquet`, pairs ref + `_sci`, skips targets that already have an `_updated` output, continues past failures and prints a summary. Downloads that still fail after short retries (`--download-retries 2`, `--retry-wait 15` s) are **deferred** rather than waited for (recorded as `epochs_deferred` in the output metadata); `update_parquet.py RESULTS_DIR --retry-deferred` later re-runs just those targets from their `_updated` files, downloading only the deferred epochs and refreshing the files in place |
| `update_old_parquet.py` | Same options and procedure as `update_parquet.py`, for parquets written before 2026-10-21 (no `FLUX_4_DIF` / `APCORR410`). Every old row's raw 4 px difference flux is rebuilt: from `MAG_4_TOT_AB_org` (exact), or — for epochs with total flux ≤ 0 — from `FLUX_4_TOT_AB` and that epoch's calibration factor fitted on its other rows; the reference flux is removed with the zero point the file was built with (`MAG_4_REF`, else `MAGZP_REF_{tag}` metadata). Epochs where the **target** has no 4 px photometry are measured again from the images; rows of other sources without any 4 px photometry are dropped. Validated on est_084892 (old file without `FLUX_4_DIF`): 4 px identical to a fresh run (MAG_4 p99 0.006 mmag; negative-flux rows within 0.0015σ); MAG_6/10 of old epochs unbiased but with 12–22 mmag MAD, since the zero-point fix rescales each row's difference flux per aperture. Run: `python ZTFphot/scripts/update_old_parquet.py RESULTS_DIR --from-updated --recalibrate --infer-cuts` |
| `batch_pipeline.py` | Process a list of targets in sequence, keeping only the final LC parquets and plots for each. Supports two input modes auto-detected by column count: **RA/Dec mode** (2 columns: `ra,dec`) runs the full pipeline including merge; **quadrant mode** (4 columns: `field,ccdid,qid,filtercode`) skips merge, derives the quadrant center from ztfquery field tables for lookup, suppresses target plots (`--no-target`). `--skip-existing` is band-aware in RA/Dec mode (a target is skipped only when a parquet exists for *every* requested band, and the sci variant with `--both`). Writes a coarse good/failed summary to `<results-dir>/sources_status.csv` (see [Target coverage diagnostics](#target-coverage-diagnostics)). Run: `python ZTFphot/scripts/batch_pipeline.py targets.txt [--both] [--bands zg zr] [--workers 20] [--skip-existing]` |
| `replot_merged.py` | Regenerate precision and light curve plots from a merged parquet when the original working directory has been deleted (e.g. after `batch_pipeline.py` cleanup). The light-curve plot includes the binned overlay (`--bin-days`, default 50). Run: `python ZTFphot/scripts/replot_merged.py lightcurves_merged.parquet --ra RA --dec DEC --out-dir plots/ [--bin-days 50]` |
| `lc_viewer.py` | Interactive GUI for browsing merged (or single-quadrant) light curve parquets. Left panel: overview scatter (median mag vs. RMS); click any point to load that object. Right panel: light curve coloured by MAGLIM. Controls: text box for direct object index entry, Prev/Next buttons (ranked by RMS), checkboxes to toggle individual quadrants, MAGLIM ≥ slider to discard shallow epochs, SEEING ≤ slider to discard poor-seeing epochs. Run: `python ZTFphot/scripts/lc_viewer.py path/to/lightcurves_merged.parquet` |
