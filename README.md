# ZTFphot

Difference-image aperture photometry pipeline for ZTF data. Given a sky position, it queries IRSA for field coverage, downloads science and reference images, runs SExtractor in dual-image mode, calibrates light curves per epoch, and assembles per-object light curves across multiple field/CCD/quadrant combinations.

---

## Requirements

- Python 3.10+
- [Miniconda or Mambaforge](https://github.com/conda-forge/miniforge)
- [SExtractor](https://www.astromatic.net/software/sextractor/) (available via conda-forge)
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

This installs Python, numpy, astropy, scipy, matplotlib, pandas, ztfquery, and SExtractor.

> **Note:** The `sex` step requires SExtractor. On most systems: `conda install -n ztf -c conda-forge source-extractor`. On HPC clusters it is often available via `module load sextractor` (or `module avail sex` to check).

### 3. Configure IRSA credentials

IRSA credentials are required to download image products. Add them to `~/.netrc`:

```
machine irsa.ipac.caltech.edu
login your_username
password your_password
```

Then restrict permissions:

```bash
chmod 600 ~/.netrc
```

An IRSA account is free at [irsa.ipac.caltech.edu](https://irsa.ipac.caltech.edu).

### 4. Data directory

By default, all pipeline data is written to `data/` in your current working directory, created automatically on first use. To use a different location pass `--base-dir /path/to/data` to any command.

---

## Quick start

### Full pipeline from scratch

```bash
python /path/to/ZTFphot/run_pipeline.py --ra 182.635755 --dec 39.405849
```

This runs all steps in order: field lookup → image download → reference catalog → simulate → SExtractor → vet → calibrate → flatfield → re-calibrate → light curves → merge → plots.

### Common quality cuts for download (all optional)

```bash
python run_pipeline.py --ra 182.635755 --dec 39.405849 \
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
python run_pipeline.py --steps calibrate lightcurves \
    --ra 182.635755 --dec 39.405849 \
    --field 443 --bands zg --ccdid 16 --qid 2 --workers 8 --force

# Check what's on disk
python run_pipeline.py --status
```

### Recommended workflow for a new field

```bash
# 1. Lookup + download (no SExtractor needed)
python run_pipeline.py --steps lookup download \
    --ra 182.635755 --dec 39.405849 --workers 8

# 2. Prepare images and run SExtractor (requires ztf env)
python run_pipeline.py --steps catalog simulate sex \
    --ra 182.635755 --dec 39.405849 --workers 8

# 3. Calibrate → flatfield → re-calibrate → light curves
python run_pipeline.py --steps calibrate flatfield calibrate lightcurves \
    --ra 182.635755 --dec 39.405849 --workers 8 --force
```

### Low-disk mode

On systems with limited disk space, use `--purge-batch N` to process images in batches of N epochs. After each batch's SExtractor step, all imaging products (difference images, simulated images, reference images) are deleted. Only the SExtractor catalogs (~5 MB/epoch) are kept for the calibration stage.

```bash
# Full pipeline, 5 epochs at a time (~4 GB peak disk usage per quadrant):
python run_pipeline.py --ra 182.635755 --dec 39.405849 --purge-batch 10 --workers 8

# Preview what would be deleted without touching disk:
python run_pipeline.py --ra 182.635755 --dec 39.405849 --purge-batch 10 --dry-run
```

After a full run (without `--purge-batch`), you can also clean up imaging products retroactively:

```bash
python run_pipeline.py --clean-up          # delete all imaging products
python run_pipeline.py --clean-up --dry-run  # preview first
```

**Typical disk budget** (10 quadrants, ~400 epochs each):

| Stage | Peak disk |
|-------|-----------|
| During `--purge-batch 5` | ~4 GB |
| SEx catalogs (end state) | ~20 GB |
| Calibrated FITS (end state) | ~8 GB |
| **Total at completion** | **~28 GB** |

---

## Directory layout

```
ZTFphot/                    ← this repository
  run_pipeline.py
  scripts/
    ztf_field_lookup.py
    download_coordinator.py
    photometry.py
    simulate_science.py
    make_catalog.py
    calib_catalogs.py
    calibrate.py
    lightcurves.py
    merge_fields.py
    vet_calibration_stars.py
    transient_catalog.py
    plot_calibration.py
    plot_diagnostics.py
    plot_lightcurve.py
    plot_precision.py
    plot_residuals.py
    SExtractor/

data/                       ← created in your working directory on first run
  Epochs/                   ← IRSA epoch cache and download logs
  Science/                  ← difference images per field/fc/ccd/qid
  Reference/                ← reference images and catalogs
  Catalogs/                 ← reference star CSV catalogs and ASSOC position catalogs
  SExCatalogs/              ← SExtractor LDAC output
  Calibrated/               ← calibrated FITS catalogs per epoch
  FlatfieldResiduals/       ← per-epoch NPZ residual maps
  LightCurves/              ← assembled light curves (parquet)
  Plots/                    ← diagnostic plots per target and quadrant
```

### Output file locations

| Output | Path |
|--------|------|
| Epoch metadata | `data/Epochs/lookup_{ra}_{dec}_{bands}.epochs.parquet` |
| Reference catalog | `data/Catalogs/{field}_{fc}_c{ccd}_q{qid}(REFERENCE)[OBJECTS].csv` |
| ASSOC position catalog | `data/Catalogs/{field}_{fc}_c{ccd}_q{qid}(ASSOC).cat` |
| SEx LDAC catalogs | `data/SExCatalogs/000/{field}/{fc}/{ccd}/{qid}/*.fits` |
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
| Download | `download` | Fetch science and reference images from IRSA |
| Reference catalog | `catalog` | Build reference CSV from `refsexcat.fits` |
| Simulate | `simulate` | Build simulated detection images (PSF at reference positions; injects target if absent from reference catalog) |
| SExtractor | `sex` | Dual-image aperture photometry on difference images |
| Vet | `vet` | Flag variable/bad calibration stars by multi-epoch RMS |
| Calibrate | `calibrate` | Linear ZP → 3σ clip → faint correction → 2D polynomial → flatfield |
| Flatfield | `flatfield` | Rebuild spatial flatfield from post-polynomial residuals |
| Light curves | `lightcurves` | Assemble per-object parquet light curves from calibrated FITS |
| Merge | `merge` | Cross-calibrate and merge multiple quadrants per band |
| Plots | `plots` | Diagnostic plots (spatial residuals, RMS, precision, light curves) |

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
| `--workers N` | Parallel workers (default: 4) |
| `--force` | Re-run even if output files already exist |
| `--verbose` | Increase log verbosity |
| `--status` | Print file-existence summary and exit |

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
| `--vet-catalog PATH` | Path to a vet catalog FITS file (overrides the default location in `Calibrated/`) |

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
3. **Faint correction** — smoothed empirical offset for faint sources (residuals binned in 0.5 mag steps, Gaussian-smoothed, tapering to zero at mag < 18.5)
4. **2D polynomial** — low-order spatial polynomial fitted to calibrator residuals and applied to all sources
5. **Spatial flatfield** — stacked median residual map applied if a saved flatfield exists on disk

---

## Optional: calibration star vetting

Bad reference stars inflate calibration scatter. After an initial calibration run:

```bash
python scripts/vet_calibration_stars.py \
    --field 443 --band zg --ccd 16 --qid 2

python run_pipeline.py --steps calibrate flatfield lightcurves plots \
    --field 443 --bands zg --ccdid 16 --qid 2 --workers 8 --force
```

The vet catalog is discovered automatically from its standard location in `data/Calibrated/`.

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
| `NC_RMS3` | Calibrator RMS after faint correction (mmag) |
| `NC_RMS4` | Calibrator RMS after 2D polynomial (mmag) |
| `NC_RMSFC` | Calibrator RMS after spatial flatfield (mmag) |
| `CALIB_N` | Linear fit intercept (mag) |
| `CALIB_M` | Linear fit slope |
| `CALIB_ZP` | ZP correction evaluated at mag 17 (mag) |
| `NC_FC_00–06` | Per-bin faint correction at mag 18.75–21.75 (mmag; −999 if empty bin) |
| `TGT_MRAW` | Raw instrumental mag of target (mmag; −999 if not detected) |
| `TGT_DCLIN` | Linear ZP correction applied to target (mmag) |
| `TGT_DCPOL` | 2D polynomial correction applied to target (mmag) |
| `TGT_DCFF` | Flatfield correction applied to target (mmag) |

---

## SExtractor configuration

- **Aperture diameters**: 3, 4, 6, 10 pixels (radii 1.5, 2, 3, 5 px) — these are diameters, matching the original pipeline
- **Primary aperture**: k=1 (4 px diameter) → stored as `MAG_4_TOT_AB` / `FLUX_4_TOT_AB`
- **Detection**: dual-image mode — detect on simulated image, measure on difference image
- **Background**: `BACK_TYPE=MANUAL, BACK_VALUE=0.0` — difference images have zero mean background
- **Source identification**: SExtractor ASSOC mode with `ASSOCSELEC_TYPE=MATCHED`; each detection is tagged with a 1-based reference catalog index (`VECTOR_ASSOC`) carried through to the calibrated FITS and parquet light curves as `object_index`. Match radius is 0.5 arcsec (PSFs are painted at exact reference positions, so the WCS round-trip error is the only tolerance needed).

---

## Standalone utilities

These scripts are not wired into `run_pipeline.py` but can be run directly:

| Script | Function |
|--------|----------|
| `transient_catalog.py` | Augments the reference SExtractor catalog with additional sources (e.g. from TNS or a user CSV) before the simulate step. Needed when the target brightened after the ZTF reference epoch and is therefore absent from the reference catalog. Pass `--refsexcat` and `--input` (or `--ra/--dec` for a TNS cone search) to produce an augmented catalog for `simulate_science.py`. |
