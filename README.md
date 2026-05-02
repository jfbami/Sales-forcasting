# Store Sales — Time Series Forecasting

Forecasting daily unit sales for thousands of product families across Favorita grocery stores in Ecuador.
Built on the [Kaggle "Store Sales — Time Series Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6F00?logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-1f9c4d" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" />
</p>

---

## Overview

This project forecasts the next **16 days** of sales for 33 product families across 54 stores. The workflow is split into two stages:

| Script | Purpose |
| --- | --- |
| `eda.py` | Loads the raw competition data and produces a full set of diagnostic plots saved to `plots/`. |
| `pipeline.py` | Builds 50+ features, tunes and trains an **XGBoost + LightGBM + LinearSVR** ensemble, and writes `submission.csv`. |

The exploratory analysis isn't an afterthought — every modelling decision in the pipeline (lags, payday flags, oil-price lag, the 2016 earthquake window) comes directly from a pattern visible in one of the plots below.

---

## Exploratory Analysis

The plots below tell the story of the data. They're saved in [`plots/`](plots) and are the foundation for the engineered features.

### 1. The macro picture — trend, seasonality, growth

| Overall trend (with 30-day MA + earthquake) | Year-over-year growth |
| :---: | :---: |
| ![Overall trend](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/01_overall_trend.png) | ![Yearly growth](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/04_yearly_growth.png) |

Sales grow steadily year-over-year with strong end-of-year peaks. The April 2016 earthquake produces a clear, dateable shock — captured later as a dedicated feature.

### 2. Calendar effects — weekday, month, payday

| Day of week | Monthly seasonality | Payday effect |
| :---: | :---: | :---: |
| ![Day of week](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/02_day_of_week.png) | ![Monthly seasonality](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/03_monthly_seasonality.png) | ![Payday](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/10_payday_effect.png) |

Weekends dominate, December spikes, and pay-cycle days (15th and end-of-month) lift sales meaningfully — all encoded as binary features.

### 3. Product mix and store segmentation

| Sales by family | Sales by store type |
| :---: | :---: |
| ![Family sales](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/05_family_sales.png) | ![Store type](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/06_store_type.png) |

Sales are heavily concentrated in a handful of families (GROCERY I, BEVERAGES, PRODUCE), and store type drives a large share of variance.

### 4. External signals — oil, promotions, transactions

| Oil vs sales | Promotion effect | Transactions vs sales |
| :---: | :---: | :---: |
| ![Oil vs sales](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/07_oil_vs_sales.png) | ![Promotions](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/08_promotion_effect.png) | ![Transactions](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/12_transactions_vs_sales.png) |

Ecuador's oil-dependent economy shows up clearly: the oil price moves inversely to sales with a lag, motivating the **14-day lagged oil feature**. Promotions roughly double family-level sales when active.

### 5. Shocks and autocorrelation

| 2016 Earthquake impact | Autocorrelation (lag-7) |
| :---: | :---: |
| ![Earthquake](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/09_earthquake.png) | ![Autocorrelation](https://raw.githubusercontent.com/jfbami/Sales-forcasting/main/plots/11_autocorrelation.png) |

The post-earthquake demand surge is sharp and short-lived — a one-month flag (Apr 16 – May 15, 2016) handles it. The lag-7 autocorrelation is the single strongest signal in the data, which is why the pipeline leans heavily on weekly-seasonal lags.

---

## The Pipeline

### Feature engineering (50+ features)

- **Calendar** — day-of-week, month, year, week, weekend, month start/end, payday (15th & last), New Year, Christmas season, post-earthquake window.
- **Lags** — sales lags from **16–56 days and 364 days** (kept ≥16 to avoid leakage into the test horizon), plus 1–4 week day-of-week lags.
- **Rolling statistics** — 7 / 14 / 28 / 91-day rolling means and standard deviations.
- **External** — oil price (14-day lagged), national holiday flags, store-level transaction averages.
- **Promotions** — `onpromotion` rolling counts.
- **Categoricals** — encoded family, city, state, store type.

### Model — weighted ensemble

| Model | Notes |
| --- | --- |
| **XGBoost** | `reg:squaredlogerror` objective, matches the competition's RMSLE metric directly. |
| **LightGBM** | RMSE on log-target, fast and complementary to XGBoost. |
| **LinearSVR** | Trained in log space with feature scaling — adds a linear baseline that smooths tree-model variance. |

Predictions are combined with **inverse-RMSLE weighting** — better validation models get higher weight.

### Training strategy

1. **Hyperparameter tuning** — `RandomizedSearchCV` over a 15% subsample with 3-fold `TimeSeriesSplit`.
2. **Full training** with the chosen hyperparameters.
3. **Validation** on the final 16 days (mirrors the test horizon).
4. **Retrain on all data** before generating test predictions.

---

## Project Structure

```
.
├── eda.py                    # Generates all 12 EDA plots
├── pipeline.py               # Feature engineering + ensemble training
├── plots/                    # All EDA visualisations (PNG)
├── feature_importances.csv   # XGBoost feature importance ranking
└── submission.csv            # Final test-set predictions (generated)
```

## Reproducing

```bash
# 1. Grab the competition data from Kaggle
kaggle competitions download -c store-sales-time-series-forecasting

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

# 3. Run EDA → produces plots/
python eda.py

# 4. Train + predict → produces submission.csv
python pipeline.py
```

## Data

From the [Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting):
`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`.
