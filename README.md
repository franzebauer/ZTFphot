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

All pipeline data is written to `data/` in your current working directory, created automatically on first use. To use a different location pass `--base-dir /path/to/data` to any command.

---

## Quick start

Run from any directory — `data/` is created automatically in your current working directory.

### Full pipeline from scratch

```bash
python /path/to/ZTFphot/run_pipeline.py --ra 330.34158 --dec 0.72143
```

This runs all steps in order: field lookup → image download → decompress → reference catalog → simulate → SExtractor → vet → calibrate → flatfield → re-calibrate → light curves → merge → plots.

> **Note:** The `sex` step requires SExtractor. On most systems: `conda install -n ztf -c conda-forge source-extractor`. On HPC clusters it is often available via `module load sextractor` (or `module avail sex` to check).

### Common quality cuts for download (all optional)

```bash
python run_pipeline.py --ra 330.34158 --dec 0.72143 \
    --max-seeing 3.0        \  # skip epochs with seeing FWHM > 3.0 arcsec
    --min-maglim 19.5       \  # skip epochs with 5σ limiting mag < 19.5
    --skip-flagged          \  # also skip cautionary-flagged epochs
    --min-epochs-per-quad 30
```

### Re-run specific steps on existing data

Steps are idempotent — existing outputs are skipped unless `--force` is passed.

```bash
# Re-calibrate and rebuild light curves for one quadrant
python run_pipeline.py --steps calibrate lightcurves \
    --ra 330.34158 --dec 0.72143 \
    --field 443 --bands zg --ccdid 16 --qid 2 --workers 8 --force

# Check what's on disk
python run_pipeline.py --status
```

### Recommended workflow for a new field

```bash
# 1. Lookup + download (no SExtractor needed)
python run_pipeline.py --steps lookup download \
    --ra 330.34158 --dec 0.72143 --workers 8

# 2. Prepare images and run SExtractor (requires ztf env)
python run_pipeline.py --steps funpack catalog simulate sex \
    --ra 330.34158 --dec 0.72143 --workers 8

# 3. Calibrate → flatfield → re-calibrate → light curves
python run_pipeline.py --steps calibrate flatfield calibrate lightcurves \
    --ra 330.34158 --dec 0.72143 --workers 8 --force
```

---

## Directory layout

```
ZTFphot/                    ← this repository
  run_pipeline.py
  scripts/
    ztf_field_lookup.py
    download_coordinator.py
    photometry.py
    calibrate.py
    lightcurves.py
    calib_catalogs.py
    vet_calibration_stars.py
    SExtractor/

data/                       ← created in your working directory on first run
  Epochs/                   ← IRSA epoch cache and download logs
  Science/                  ← difference images per field/fc/ccd/qid
  Reference/                ← reference images and catalogs
  Catalogs/                 ← reference star CSV catalogs
  SExCatalogs/              ← SExtractor LDAC output
  Calibrated/               ← calibrated catalogs, flatfield, vet catalog
  FlatfieldResiduals/       ← per-epoch residual maps
  LightCurves/              ← assembled light curves (parquet)
  Plots/                    ← diagnostic plots per target and quadrant
```

---

## Pipeline steps

| Step | CLI name | Description |
|------|----------|-------------|
| Decompress | `funpack` | Decompress `.fits.fz` difference images |
| Reference catalog | `catalog` | Build reference CSV from `refsexcat.fits` |
| Simulate | `simulate` | Build simulated detection images (PSF at reference positions) |
| SExtractor | `sex` | Dual-image aperture photometry on difference images |
| Vet | `vet` | Flag variable/bad calibration stars by multi-epoch RMS |
| Calibrate | `calibrate` | Linear ZP → 3σ clip → faint correction → 2D polynomial → flatfield |
| Flatfield | `flatfield` | Rebuild spatial flatfield from post-polynomial residuals |
| Light curves | `lightcurves` | Assemble per-object parquet light curves from calibrated FITS |
| Merge | `merge` | Cross-calibrate and merge multiple quadrants per band |
| Plots | `plots` | Diagnostic plots (spatial residuals, RMS, precision, quality, light curves) |

---

## Key CLI options

| Option | Description |
|--------|-------------|
| `--base-dir PATH` | Data directory (default: `./data` in current working directory) |
| `--ra / --dec` | Target RA/Dec (deg) — required for lookup and download steps |
| `--field N` | ZTF field number (filter to specific field) |
| `--bands zg zr zi` | Filter(s) to process (default: all present) |
| `--ccdid N` | CCD number (filter to specific CCD) |
| `--qid N` | Quadrant ID (filter to specific quadrant) |
| `--workers N` | Parallel workers (default: 4) |
| `--force` | Re-run even if output files exist |
| `--target-ra / --target-dec` | Track target source through calibration headers and plots |
| `--poly-degree N` | Spatial polynomial degree for calibration (default: 2) |
| `--status` | Print file-existence summary and exit |

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
