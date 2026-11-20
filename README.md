# Avian community dissimilarity

Code and archived results for Adam Ali's study, *Predicting Avian Community Dissimilarity from Soundscape Indices: Improvements within Datasets and Limits to Transfer*.

This study predicts differences in bird community composition from 60 soundscape indices. Four datasets are evaluated in turn as withheld datasets. Models improve on constant predictions within familiar datasets, but these improvements do not persist in the withheld datasets. The processed observations come from [Sethi et al. (2023)](https://doi.org/10.1038/s41559-023-02148-z); see [attribution and licenses](NOTICE.md).

## Setup

The recorded environment is Python 3.13.3 and R 4.6.1, with the package versions in `requirements.txt` and R package `gdm` version 1.6.0-7. CPU execution was used. Other versions or hardware can produce numerical differences.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript -e 'install.packages("gdm", repos="https://cloud.r-project.org")'
```

Check the installed GDM version with `Rscript -e 'packageVersion("gdm")'`. For the original version, use the CRAN archive if the current release differs.

## Data and saved results

The permanent data archive is [Zenodo 10.5281/zenodo.24394649](https://doi.org/10.5281/zenodo.24394649).

Download the review archive from the [v1.0.0 release](https://github.com/adamhaniali1-ctrl/avian-community-dissimilarity/releases/tag/v1.0.0), then extract and verify it:

```sh
python prepare_data.py --download
python check_results.py --checkpoints
```

Alternatively, use `python prepare_data.py --archive /path/to/review_data_v1.0.0.zip` for a local download. The archive contains derived inputs, fixed pair manifests, saved predictions and model checkpoints. SHA256 checksums identify every archived file. The source dataset is also available separately at [Zenodo](https://doi.org/10.5281/zenodo.7446640). Original audio files are not needed for this analysis.

## Reproduction

```sh
python prepare_data.py --regenerate-manifests
python run_models.py
python evaluate.py
python make_figures.py
```

New outputs go into `reproduced/`; archived research results remain unchanged. `run_models.py` refits the baselines and final models using the recorded selected settings, then reports differences from saved predictions. Use `--holdout india --condition filtered` to run one experiment, or `--models forest` to run one model family. See each script's `--help` for options.

`evaluate.py` reconstructs all evaluation targets from the bird lists and verifies the saved metrics. It also calculates descriptive mean signed error and checks KS distribution differences. To repeat the original 2000 event bootstrap draws per group, run `python evaluate.py --bootstrap`.

The numerical figures are regenerated from saved tables. Figure 1 is an editable SVG diagram, copied from the archive with its matching PDF and PNG.

Model selection can be repeated separately:

```sh
python select_models.py --family forest
python select_models.py --family preprocessing
python select_models.py --family architecture
```

These commands use development observations only. The architecture step reuses the nine selected preprocessing fits for architecture A in each experiment, as in the study. If fresh preprocessing results exist in the same output folder, it uses those; otherwise it uses the archived fits. `--smoke` runs one neural fit for checking the setup and does not constitute model selection. The published biological conclusions use SSI features; the incompatible LF representation is retained only in the archived diagnostic records.

## Organization and verification

`analysis/` contains the calculation functions. `config/` records experimental settings. `provenance/` links the reorganized functions to the historical scripts and records release checks. The downloadable historical source archive preserves the original first party scripts and configuration; these contain stage-specific inventory checks and are a historical record rather than the public entry points.

See [release checks](RELEASE_CHECKS.md) for the tested scope and numerical limitations. All primary final model reruns matched saved predictions within floating point precision. Three sensitivity neural refits were not bit identical; all 40 saved neural checkpoints and eight GDM checkpoints reproduce their archived predictions. The archived results, rather than a new refit, are the reference for the manuscript. This release does not replace the paper's limitations or establish transfer to new ecosystems.
