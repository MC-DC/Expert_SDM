# Expert-Informed Bayesian Species Distribution Model

A two-step Bayesian species distribution model (SDM) exemplified using the monarch butterfly (*Danaus plexippus*) in Canada. This model integrates expert knowledge as informative priors into a species distribution model.

## Overview

The workflow is split into two steps:

**Step 1: Expert-knowledge model (Stan)**  
Expert-elicited information is used do fit a species distribution model, using point observations from experts and range limits of environmental variables to define informative priors. Two different approaches are used, 1) a classic regression where point observations are used as presence/absence, and 2) a weighted likelihood model that takes into consideration the level of confidence the expert reported when providing the points.

| Variant | Script | Description |
|---|---|---|
| Baseline | `Step1-Expert_model.stan` | Standard logistic regression with expert priors |
| Weighted likelihood | `Step1-Expert_model_weighted.stan` | Down-weights low-confidence points |

**Step 2: Spatial update (R-INLA)**  
Stan posterior means and SDs are converted to Gaussian precision priors and passed into an R-INLA binomial model (`Expert_informed_SDM.Rmd`). The INLA model adds a Matérn spatial random effect via an SPDE mesh over Canada. Results are compared against a non-informative INLA baseline run on the same data.

## Repository Structure

```
Expert_SDM/
├── scripts/
│   ├── Expert_informed_SDM.Rmd       # Main workflow (Steps 1 & 2)
│   ├── model_sampling.Rmd            # Model simulations using data sampling
│   ├── Data_clean.Rmd                # Occurrence data cleaning
│   ├── mean_env_layers.R             # Environmental raster preprocessing
│   ├── Step1-Expert_model.stan       # Stan baseline model
│   └── Step1-Expert_model_weighted.stan  # Stan weighted-likelihood variant
├── data/
│   ├── study_area/                   # Canada boundary shapefile
│   ├── species/
│   │   ├── PA_monarch_CAN.shp        # Presence/absence occurrence records for the monarch in Canada
│   │   ├── records_clean_gbif.csv    # Cleaned GBIF records for the monarch 
│   │   └── Danaus_plexippus-GBIF-30-03-26/  # Raw GBIF download for the monarch 
│   ├── expert_knowledge/
│   │   ├── Points.csv                # Expert-labelled points (presence/avoid + confidence)
│   │   └── Responses.csv            # Expert range responses (min/max temp & precip)
│   └── predictors/
│       ├── tas_mean.tif              # Mean annual temperature (Ouranos)
│       └── pr_mean.tif              # Mean annual precipitation (Ouranos)
└── results/                          # Prediction and uncertainty maps (PNG)
```

## Dependencies

**R packages:**

```r
install.packages(c("dplyr", "ggplot2", "tidyr", "sf", "terra", "geodata"))
```

R-INLA must be installed from its own repository:

```r
install.packages("INLA",
  repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable"),
  dep = TRUE)
```

RStan installation: see the [RStan Getting Started guide](https://mc-stan.org/rstan/).

## Usage

1. Update the absolute data paths in `scripts/Expert_informed_SDM.Rmd` and `scripts/Data_clean.Rmd` to match your local directory.
2. Run `scripts/Data_clean.Rmd` to prepare occurrence records.
3. Run `scripts/mean_env_layers.R` to preprocess the climate rasters.
4. Knit `scripts/Expert_informed_SDM.Rmd` to reproduce the full analysis and output maps.
5. Run `scripts/model_sampling.Rmd` for results in low-data scenario simulations

## Results

Prediction maps and uncertainty (SD) maps are saved to `results/` for both the expert-informed model (`pred_*`, `SD_*`) and the non-informative baseline (`NI_pred_*`, `NI_SD_*`).

## Data Sources

- **Occurrence records:** GBIF (*Danaus plexippus*, downloaded 2026-03-30)
- **Climate covariates:** Ouranos regional climate dataset (mean annual temperature and precipitation)
- **Expert knowledge:** Elicited range limits and point labels collected as part of this project

## Project documentation

The complete workflow, code, figures, and example results are available in the
[rendered HTML documentation](https://mc-dc.github.io/Expert_SDM/).

## Author

M. Camila Diaz-Corzo
