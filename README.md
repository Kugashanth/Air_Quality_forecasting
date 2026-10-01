# Air_Quality_forecasting
## Forecasting Air Quality in Kolkata: A Time Series Model Comparison Under Extreme Pollution Events

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Dataset licence](https://img.shields.io/badge/Dataset-CC0--1.0-lightgrey)
![Status](https://img.shields.io/badge/Status-Capstone%20Project-teal)

Capstone Project in Data Science II (DS3206), Faculty of Computing, Sabaragamuwa University of Sri Lanka.

This project compares four time series models for **PM2.5 forecasting in Kolkata** and, unlike most studies, scores every model **separately on Normal days and Extreme days** (Diwali, COVID-19 lockdown, cyclones, heatwaves) instead of using one pooled accuracy number.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Important Note About the Data](#important-note-about-the-data)
- [Workflow](#workflow)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Real-Data Validation and Uncertainty](#real-data-validation-and-uncertainty)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Author and Supervision](#author-and-supervision)
- [License](#license)

---

## Overview

Kolkata regularly records PM2.5 levels well above World Health Organization guidelines, with sharp short-lived spikes during Diwali fireworks, the 2020 COVID-19 lockdown, tropical cyclones and pre-monsoon heatwaves. A model that looks accurate on average can still fail on exactly the days when a health warning matters most.

**Research question:** how much does each model's accuracy drop when pollution moves from ordinary to extreme conditions, and which model offers the best trade-off?

**What this project does**

- Builds and compares four models: **SARIMA**, **Prophet**, **XGBoost** and **LSTM**.
- Labels every hour as *Normal* or *Extreme* using a calendar of 15 event windows plus an automatic anomaly detector.
- Evaluates each model on Normal and Extreme test data with **MAE, RMSE, MAPE and R²**, and measures the **RMSE degradation** between the two.
- Validates the XGBoost baseline against **real CPCB sensor data**.
- Adds an **event-aware adaptive XGBoost** and **conformal prediction intervals (90%)** with a high-risk flag above 250 µg/m³.

---

## Key Findings

| Finding | Detail |
|---|---|
| Best on Normal days | **Prophet** (MAE 10.37, RMSE 12.20, R² 0.726) |
| All models degrade in extreme events | RMSE rises by **114% to 381%** |
| Smallest relative degradation | **SARIMA** (+113.8%), but it has the weakest absolute accuracy |
| Most balanced trade-off | **LSTM** (+186.5% degradation, strong Extreme MAE and R²) |
| Largest relative degradation | **Prophet** (+381.1%) |
| No single winner | No model is best in both regimes, so an event-aware combination suits early warning |

---

## Important Note About the Data

The main dataset is **synthetic but statistically calibrated** to published CPCB and IMD reference statistics. It is **not** a direct record from physical sensors. This is disclosed throughout the project, and conclusions about specific real events should be read as **illustrative**. A real-data check with 271 hourly CPCB observations is included, but it is a small sample and should be treated as indicative only.

---

## Workflow

```mermaid
flowchart LR
    A[Data acquisition<br/>Hugging Face + CPCB] --> B[Cleaning and validation]
    B --> C[EDA<br/>trend, seasonality, correlation]
    C --> D[Event labelling<br/>15 windows + z-score detector]
    D --> E[Feature engineering<br/>lags, rolling, calendar]
    E --> F[Models<br/>SARIMA, Prophet, XGBoost, LSTM]
    F --> G[Evaluation<br/>Normal vs Extreme]
    G --> H[Conformal intervals<br/>and risk flag]
```

---

## Dataset

| Item | Value |
|---|---|
| Main source | [`neuralsorcerer/air-quality`](https://huggingface.co/datasets/neuralsorcerer/air-quality) on Hugging Face (CC0-1.0) |
| Records | 87,672 hourly rows, 1 Jan 2015 to 31 Dec 2024, Kolkata |
| Columns (11) | timestamp, PM2.5, PM10, NO2, CO, SO2, O3, temperature, humidity, wind speed, rainfall |
| Missing values | None |
| PM2.5 range | min 15.00, max 926.15, mean 57.83 µg/m³ |
| Real validation data | CPCB Kolkata PM2.5 (`cpcb_kolkata.csv`), 271 hourly observations after filtering and resampling |

**Exploratory highlights**

- PM2.5 is highest from November to February and lowest from May to August.
- Within a day, it peaks around 04:00 to 06:00 and is lowest around 14:00 to 15:00.
- Correlations: PM2.5 vs PM10 = 0.89, vs humidity = -0.46, vs wind speed = -0.40.

---

## Methodology

### 1. Preprocessing

- Remove duplicate rows and timestamps.
- Check the hourly index for gaps.
- Clip impossible values (pollutants and rainfall at 0 or above, humidity between 0 and 100).
- No imputation was needed.

### 2. Extreme-event labelling

- **Event calendar (15 windows):** Diwali 2015 to 2024, COVID-19 lockdown (25 Mar to 31 May 2020), Cyclones Amphan and Yaas, Heatwaves 2023 and 2024.
- **Automatic detector:** 24-hour rolling median and standard deviation, z-threshold 3.0 (849 flagged hours).
- **Cross-check:** PELT change-point detection (`ruptures`) found 46 change-points.
- **Result:** 84,394 Normal hours (96.26%) and 3,278 Extreme hours (3.74%).

### 3. Feature engineering (XGBoost)

- Calendar: hour, day of week, month, weekend flag.
- Lags of PM2.5: 1, 3, 6, 24 and 168 hours.
- 24-hour rolling mean and standard deviation, computed on past values only (shifted 1 hour) to avoid leakage.
- Weather: temperature, humidity, wind speed, rainfall.

### 4. Train, validation and test split

Chronological **70% / 15% / 15%** split (no shuffling). The test set is further divided into Normal and Extreme subsets so each model is scored per regime.

### 5. Models

| Model | Resolution | Setup |
|---|---|---|
| **SARIMA** | Daily | statsmodels SARIMAX, order (1,1,1)(1,1,1,7), 3,032 training days, AIC 23,442.5 |
| **Prophet** | Daily | Yearly and weekly seasonality, all 15 event windows supplied as custom holidays |
| **XGBoost** | Hourly | 400 trees, depth 6, learning rate 0.05, 80% row and column subsampling |
| **LSTM** | Hourly | 24-hour windows, LSTM 64 then 32 units (tanh), dropout 0.2, Adam, MSE, early stopping |

**Adaptive XGBoost:** a Normal model (60,687 rows) and an Event model (565 rows). At inference the detected regime selects which model forecasts.

### 6. Metrics

MAE, RMSE, MAPE and R², reported separately for Normal and Extreme periods. **RMSE** is the primary robustness metric.

Degradation = (Extreme RMSE - Normal RMSE) / Normal RMSE x 100.

---

## Results

All errors are in µg/m³ (MAPE in %).

| Model | Period | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| Prophet | Normal | 10.37 | 12.20 | 33.17 | 0.726 |
| LSTM | Normal | 16.92 | 21.48 | 50.86 | 0.558 |
| SARIMA | Normal | 27.78 | 36.47 | 49.12 | -1.446 |
| XGBoost | Normal | n/a* | n/a* | n/a* | n/a* |
| XGBoost | Extreme | 17.19 | 39.48 | 34.01 | 0.901 |
| Prophet | Extreme | 33.12 | 58.69 | 53.22 | 0.186 |
| LSTM | Extreme | 24.73 | 61.53 | 59.66 | 0.712 |
| SARIMA | Extreme | 42.26 | 77.97 | 45.50 | -0.437 |

\* XGBoost has no valid Normal-period test rows in this run because of a data-partitioning irregularity. It is reported openly, and its Extreme figures are **not directly comparable** to the other three models.

### RMSE degradation (Normal to Extreme)

| Model | Normal RMSE | Extreme RMSE | Change |
|---|---|---|---|
| SARIMA | 36.47 | 77.97 | +113.8% |
| LSTM | 21.48 | 61.53 | +186.5% |
| Prophet | 12.20 | 58.69 | +381.1% |
| XGBoost | n/a* | 39.48 | n/a* |

---

## Real-Data Validation and Uncertainty

| Check | Result |
|---|---|
| XGBoost on real CPCB data | RMSE **2.42**, MAE **2.34** (synthetic data RMSE: 23.19) |
| Conformal residual quantile | 31.723 |
| Target coverage | 90% |
| Observed test coverage | **88.6%** |
| Mean interval width | 63.45 µg/m³ |
| Risk threshold | 250 µg/m³ |
| Flagged hours | 24 High-Risk, 13,102 Low-Risk |

The real-data sample is small (271 points), so it is supporting evidence rather than final proof.

---

## Repository Structure

> Update this section to match your repository.

```
.
├── README.md
├── notebooks/
│   └── <your_notebook_name>.ipynb      # full pipeline: data, labelling, models, evaluation
├── data/
│   └── cpcb_kolkata.csv                # real CPCB validation data (if you include it)
├── figures/                            # exported charts used in the report and slides
├── report/
│   └── 22CDS0446_Final_Report.docx
├── slides/
│   └── Final_Presentation_22CDS0446.pptx
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Example `requirements.txt` (adjust to match the imports in your notebook):

```text
pandas
numpy
matplotlib
scikit-learn
huggingface_hub
statsmodels
prophet
xgboost
ruptures
pmdarima
jupyter
# Add the deep learning framework used for the LSTM (for example tensorflow)
```

### Download the main dataset

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="neuralsorcerer/air-quality",
    filename="<dataset_file_name>",   # check the dataset page for the exact file name
    repo_type="dataset",
)
```

### Run

```bash
jupyter notebook notebooks/<your_notebook_name>.ipynb
```

Run the cells from top to bottom. The notebook follows the workflow above: load and clean the data, label events, build features, train the four models, evaluate on Normal and Extreme periods, then run the real-data check and conformal intervals.

---

## Limitations

- The main dataset is synthetic but calibrated, so event-level conclusions are illustrative.
- XGBoost lacks a valid Normal-period test sample, so it cannot be compared fairly with the other models.
- Only 271 real CPCB observations were used for validation.
- Single city, single pollutant, single chronological split.
- SARIMA and Prophet run at daily resolution, XGBoost and LSTM at hourly resolution, so raw error values are not strictly like-for-like.
- Hyperparameters are fixed defaults and were not tuned.
- Weather variables are used as measured; a real forecasting system would need weather forecasts or lagged weather.
- Conformal intervals have a fixed width and are slightly below the 90% target coverage.

---

## Future Work

1. **Automatic hyperparameter search:** `auto_arima` for SARIMA, Optuna or grid search with time series cross-validation for XGBoost and LSTM.
2. **Rolling (walk-forward) evaluation** across several extreme events for a more reliable degradation estimate.
3. **Model ensembling:** a stable baseline (SARIMA or Prophet) blended with, or switched to, XGBoost or LSTM when an extreme event is detected.
4. **Interactive dashboard** with live forecasts, intervals and event flags for non-technical users.
5. Repeat the comparison for **PM10, NO2, CO, SO2 and O3**.

---

## References

1. neuralsorcerer, "Air Quality & Meteorology Dataset (Kolkata)," Hugging Face Datasets, 2025, doi: 10.57967/hf/5729.
2. D. Sharma and D. Mauzerall, CPCB continuous monitoring air quality data for India, 2015 to 2019, Princeton DataSpace, 2021, doi: 10.34770/60j3-yp02.
3. G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*. Holden-Day, 1970.
4. S. J. Taylor and B. Letham, "Forecasting at scale," *The American Statistician*, vol. 72, no. 1, pp. 37 to 45, 2018.
5. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785 to 794.
6. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735 to 1780, 1997.
7. World Health Organization, "Ambient (outdoor) air pollution," WHO Fact Sheets, 2022.
8. O. Vallis, J. Hochenbaum and A. Kejariwal, "A novel technique for long-term anomaly detection in the cloud," in *Proc. HotCloud 14*, 2014.
9. T. Biswas, D. Saha, et al., "Strict lockdown measures reduced PM2.5 concentrations during the COVID-19 pandemic in Kolkata, India," *Sustainable Water Resources Management*, vol. 8, no. 2, art. 45, 2022.

---

## Author and Supervision

- **Author:** T. Kugashanth (Index No: 22CDS0446)
- **Internal supervisor:** Professor S. Vasanthapriyan, Ph.D, SMIEEE
- **Institution:** Department of Data Science, Faculty of Computing, Sabaragamuwa University of Sri Lanka


