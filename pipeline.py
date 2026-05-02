"""
Store Sales - Time Series Forecasting Pipeline
Ensemble forecasting: XGBoost + LightGBM + LinearSVR
with lag features, rolling averages, and external data.

Metric: RMSLE 

my Strategy:
  1. Hyperparameter tuning via RandomizedSearchCV + TimeSeriesSplit
     on a 20% subsample for speed.
  2. Final training on full data with best params.
  3. Inverse-RMSLE weighted ensemble.
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

#configs
# Colab: all CSVs are flat in /content/
DATA_DIR = "/content"
OUTPUT_DIR = "/content"
SEED = 42
np.random.seed(SEED)

# Subsample fraction for hyperparameter search (full data used for final train)
TUNE_FRAC = 0.15


# ─── RMSLE Metric ────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error (competition metric)."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def neg_rmsle_scorer(y_true, y_pred):
    return -rmsle(y_true, y_pred)


rmsle_scorer = make_scorer(neg_rmsle_scorer, greater_is_better=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data():
    """Load and merge all datasets."""
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(
        os.path.join(DATA_DIR, "holidays_events.csv"), parse_dates=["date"]
    )
    transactions = pd.read_csv(
        os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["date"]
    )

 
    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")

  
    oil_full = (
        oil.set_index("date")
        .resample("D")
        .mean()
        .interpolate(method="linear")
        .reset_index()
    )
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()

    train = train.merge(oil_full, on="date", how="left")
    test = test.merge(oil_full, on="date", how="left")

    #National holidays
    national_holidays = holidays[
        (holidays["locale"] == "National")
        & (holidays["type"].isin(["Holiday", "Bridge", "Transfer", "Additional"]))
        & (holidays["transferred"] == False)
    ][["date"]].drop_duplicates()
    national_holidays["is_national_holiday"] = 1

    train = train.merge(national_holidays, on="date", how="left")
    test = test.merge(national_holidays, on="date", how="left")
    train["is_national_holiday"] = train["is_national_holiday"].fillna(0).astype(int)
    test["is_national_holiday"] = test["is_national_holiday"].fillna(0).astype(int)

    # avhg store transactions
    avg_trans = transactions.groupby("store_nbr")["transactions"].mean().reset_index()
    avg_trans.columns = ["store_nbr", "avg_store_transactions"]
    train = train.merge(avg_trans, on="store_nbr", how="left")
    test = test.merge(avg_trans, on="store_nbr", how="left")

    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test


#new featuyres!
def create_date_features(df):
    """Extract calendar features from date."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Payday: wages paid on 15th and last day of month in Ecuador
    df["is_payday"] = ((df["day_of_month"] == 15) | (df["is_month_end"] == 1)).astype(int)

    # New Year's Day (stores closed)
    df["is_new_year"] = ((df["month"] == 1) & (df["day_of_month"] == 1)).astype(int)

    # Christmas season (Dec 15-31)
    df["is_christmas_season"] = (
        (df["month"] == 12) & (df["day_of_month"] >= 15)
    ).astype(int)

    # Earthquake aftermath (Apr 16 - May 15, 2016)
    df["is_earthquake_aftermath"] = (
        (df["date"] >= "2016-04-16") & (df["date"] <= "2016-05-15")
    ).astype(int)

    return df


def create_lag_features(df, group_cols, target_col, lags):
    """Lag features grouped by store/family. Only lags >= 16 for leakage ."""
    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def create_rolling_features(df, group_cols, target_col, windows, lag_offset=16):
    """Rolling mean/std with lag offset to prevent leakage."""
    for w in windows:
        shifted = df.groupby(group_cols)[target_col].shift(lag_offset)
        rolling = shifted.rolling(window=w, min_periods=1)
        df[f"rolling_mean_{w}"] = rolling.mean()
        df[f"rolling_std_{w}"] = rolling.std().fillna(0)
    return df


def create_ewm_features(df, group_cols, target_col, spans, lag_offset=16):
    """Exponential weighted moving average features."""
    for span in spans:
        shifted = df.groupby(group_cols)[target_col].shift(lag_offset)
        df[f"ewm_{span}"] = shifted.ewm(span=span, min_periods=1).mean()
    return df


def create_promo_features(df, group_cols):
    """Rolling promotion counts."""
    for w in [7, 14, 30]:
        shifted = df.groupby(group_cols)["onpromotion"].shift(1)
        df[f"promo_rolling_{w}"] = shifted.rolling(w, min_periods=1).mean()
    return df


def encode_categoricals(train, test):
    """Label-encode categorical columns."""
    cat_cols = ["family", "city", "state", "type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        train[col + "_enc"] = le.transform(train[col])
        test[col + "_enc"] = le.transform(test[col])
        encoders[col] = le
    return train, test, encoders


def build_features(train, test):
    """Full feature engineering pipeline."""
    print("Engineering features...")
    t0 = time.time()

    test["sales"] = np.nan
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Date features
    combined = create_date_features(combined)

    # Oil lag (14-day — effect takes ~2 weeks per EDA)
    combined["oil_lag_14"] = combined.groupby(["store_nbr", "family"])["dcoilwtico"].shift(14)
    combined["oil_lag_14"] = combined["oil_lag_14"].ffill()

    # Lag features (>= 16 days to avoid leakage into 16-day test window)
    group = ["store_nbr", "family"]
    combined = create_lag_features(
        combined, group, "sales", lags=[16, 17, 18, 19, 20, 21, 28, 35, 42, 49, 56, 364]
    )

    # Rolling features (offset=16)
    combined = create_rolling_features(combined, group, "sales", windows=[7, 14, 28, 56, 91])

    # EWM features
    combined = create_ewm_features(combined, group, "sales", spans=[7, 14, 28])

    # Promotion features
    combined = create_promo_features(combined, group)

    # Same-day-of-week lags (weekly seasonality)
    for w in [1, 2, 3, 4]:
        lag_days = 7 * w + 16
        combined[f"sales_dow_lag_{w}w"] = combined.groupby(group)["sales"].shift(lag_days)

    # Split back
    train_out = combined[combined["id"].isin(train["id"])].copy()
    test_out = combined[~combined["id"].isin(train["id"])].copy()

    # Encode categoricals
    train_out, test_out, encoders = encode_categoricals(train_out, test_out)

    print(f"  Features done in {time.time() - t0:.1f}s")
    return train_out, test_out, encoders


#akk the features and cols
FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "year", "week_of_year",
    "day_of_year", "is_weekend", "is_month_start", "is_month_end",
    "is_payday", "is_new_year", "is_christmas_season", "is_earthquake_aftermath",
    "store_nbr", "family_enc", "city_enc", "state_enc", "type_enc", "cluster",
    "onpromotion", "avg_store_transactions",
    "dcoilwtico", "oil_lag_14", "is_national_holiday",
    "sales_lag_16", "sales_lag_17", "sales_lag_18", "sales_lag_19",
    "sales_lag_20", "sales_lag_21", "sales_lag_28", "sales_lag_35",
    "sales_lag_42", "sales_lag_49", "sales_lag_56", "sales_lag_364",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_mean_56", "rolling_mean_91",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "ewm_7", "ewm_14", "ewm_28",
    "promo_rolling_7", "promo_rolling_14", "promo_rolling_30",
    "sales_dow_lag_1w", "sales_dow_lag_2w", "sales_dow_lag_3w", "sales_dow_lag_4w",
]


# train
def tune_and_train_xgboost(X_train, y_train, X_val, y_val):
    """Tune XGBoost on subsample, then train on full training set."""
    print("  [XGBoost] Tuning on subsample...")

    # Subsample for tuning
    n_tune = int(len(X_train) * TUNE_FRAC)
    # Take last TUNE_FRAC of training data (preserves temporal order)
    X_tune = X_train.iloc[-n_tune:]
    y_tune = y_train[-n_tune:]

    param_dist = {
        "n_estimators": [400, 600, 800],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.02, 0.05, 0.08],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [5, 10],
        "reg_lambda": [1, 5],
    }

    base = xgb.XGBRegressor(
        objective="reg:squaredlogerror", tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=12, scoring=rmsle_scorer,
        cv=tscv, random_state=SEED, n_jobs=1, verbose=0,
    )
    search.fit(X_tune, y_tune)
    best_params = search.best_params_
    print(f"    Best params: {best_params}")

    # Retrain on full training data with best params
    print("  [XGBoost] Training on full data...")
    model = xgb.XGBRegressor(
        **best_params, objective="reg:squaredlogerror",
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_pred = np.clip(model.predict(X_val), 0, None)
    score = rmsle(y_val, val_pred)
    print(f"    Val RMSLE: {score:.5f}")
    return model, best_params, score


def tune_and_train_lightgbm(X_train, y_train, X_val, y_val):
    """Tune LightGBM on subsample, then train on full training set."""
    print("  [LightGBM] Tuning on subsample...")

    n_tune = int(len(X_train) * TUNE_FRAC)
    X_tune = X_train.iloc[-n_tune:]
    y_tune = y_train[-n_tune:]

    param_dist = {
        "n_estimators": [400, 600, 800],
        "max_depth": [6, 8, -1],
        "learning_rate": [0.02, 0.05, 0.08],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [20, 50],
        "reg_lambda": [1, 5],
    }

    base = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=12, scoring=rmsle_scorer,
        cv=tscv, random_state=SEED, n_jobs=1, verbose=0,
    )
    search.fit(X_tune, y_tune)
    best_params = search.best_params_
    print(f"    Best params: {best_params}")

    print("  [LightGBM] Training on full data...")
    model = lgb.LGBMRegressor(
        **best_params, objective="regression", metric="rmse",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    val_pred = np.clip(model.predict(X_val), 0, None)
    score = rmsle(y_val, val_pred)
    print(f"    Val RMSLE: {score:.5f}")
    return model, best_params, score


def tune_and_train_svr(X_train, y_train, X_val, y_val):
    """Tune LinearSVR in log space on subsample, then train on full set."""
    print("  [LinearSVR] Tuning on subsample...")

    n_tune = int(len(X_train) * TUNE_FRAC)
    X_tune = X_train.iloc[-n_tune:]
    y_tune = y_train[-n_tune:]

    scaler = StandardScaler()
    X_tune_sc = scaler.fit_transform(X_tune)
    y_tune_log = np.log1p(y_tune)

    param_dist = {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1],
        "max_iter": [3000],
    }

    base = LinearSVR(random_state=SEED, dual=True)
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=6, cv=tscv,
        random_state=SEED, n_jobs=-1, verbose=0,
    )
    search.fit(X_tune_sc, y_tune_log)
    best_params = search.best_params_
    print(f"    Best params: {best_params}")

    # Retrain scaler and model on full data
    print("  [LinearSVR] Training on full data...")
    scaler_full = StandardScaler()
    X_train_sc = scaler_full.fit_transform(X_train)
    y_train_log = np.log1p(y_train)

    model = LinearSVR(**best_params, random_state=SEED, dual=True)
    model.fit(X_train_sc, y_train_log)

    X_val_sc = scaler_full.transform(X_val)
    val_pred_log = model.predict(X_val_sc)
    val_pred = np.clip(np.expm1(np.clip(val_pred_log, 0, None)), 0, None)
    score = rmsle(y_val, val_pred)
    print(f"    Val RMSLE: {score:.5f}")
    return model, scaler_full, best_params, score


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Load & engineer features
    train, test = load_data()
    train_fe, test_fe, encoders = build_features(train, test)

    # ── Train/Validation Split ──
    # Validate on last 16 days (mirrors test horizon)
    val_cutoff = train_fe["date"].max() - pd.Timedelta(days=15)
    train_mask = (train_fe["date"] <= val_cutoff) & (train_fe["date"] >= "2014-06-01")
    val_mask = train_fe["date"] > val_cutoff

    X_train = train_fe.loc[train_mask, FEATURE_COLS].fillna(-1)
    y_train = train_fe.loc[train_mask, "sales"].values
    X_val = train_fe.loc[val_mask, FEATURE_COLS].fillna(-1)
    y_val = train_fe.loc[val_mask, "sales"].values
    X_test = test_fe[FEATURE_COLS].fillna(-1)

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Val period: {val_cutoff.date()} to {train_fe['date'].max().date()}\n")

    # ── Train Models ──
    print("=" * 60)
    print("TRAINING ENSEMBLE MODELS")
    print("=" * 60)

    xgb_model, xgb_params, xgb_score = tune_and_train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_params, lgb_score = tune_and_train_lightgbm(X_train, y_train, X_val, y_val)
    svr_model, svr_scaler, svr_params, svr_score = tune_and_train_svr(X_train, y_train, X_val, y_val)

    # ── Ensemble (inverse-RMSLE weights) ──
    scores = np.array([xgb_score, lgb_score, svr_score])
    inv_scores = 1.0 / scores
    weights = inv_scores / inv_scores.sum()

    print(f"\n{'=' * 60}")
    print("ENSEMBLE WEIGHTS")
    print(f"{'=' * 60}")
    print(f"  XGBoost:   w={weights[0]:.3f}  RMSLE={xgb_score:.5f}")
    print(f"  LightGBM:  w={weights[1]:.3f}  RMSLE={lgb_score:.5f}")
    print(f"  LinearSVR: w={weights[2]:.3f}  RMSLE={svr_score:.5f}")

    # Validate ensemble
    val_xgb = np.clip(xgb_model.predict(X_val), 0, None)
    val_lgb = np.clip(lgb_model.predict(X_val), 0, None)
    val_svr_log = svr_model.predict(svr_scaler.transform(X_val))
    val_svr = np.clip(np.expm1(np.clip(val_svr_log, 0, None)), 0, None)

    val_ensemble = weights[0] * val_xgb + weights[1] * val_lgb + weights[2] * val_svr
    val_ensemble = np.clip(val_ensemble, 0, None)
    ensemble_score = rmsle(y_val, val_ensemble)
    print(f"\n  ENSEMBLE Val RMSLE: {ensemble_score:.5f}")

    # ── Retrain on ALL data for submission ──
    print("\nRetraining on full data for submission...")
    all_mask = train_fe["date"] >= "2014-06-01"
    X_all = train_fe.loc[all_mask, FEATURE_COLS].fillna(-1)
    y_all = train_fe.loc[all_mask, "sales"].values

    xgb_final = xgb.XGBRegressor(
        **xgb_params, objective="reg:squaredlogerror",
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )
    xgb_final.fit(X_all, y_all)

    lgb_final = lgb.LGBMRegressor(
        **lgb_params, objective="regression", metric="rmse",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb_final.fit(X_all, y_all)

    svr_scaler_final = StandardScaler()
    X_all_sc = svr_scaler_final.fit_transform(X_all)
    svr_final = LinearSVR(**svr_params, random_state=SEED, dual=True)
    svr_final.fit(X_all_sc, np.log1p(y_all))

    # ── Generate Predictions ──
    print("Generating predictions...")
    pred_xgb = np.clip(xgb_final.predict(X_test), 0, None)
    pred_lgb = np.clip(lgb_final.predict(X_test), 0, None)
    pred_svr_log = svr_final.predict(svr_scaler_final.transform(X_test))
    pred_svr = np.clip(np.expm1(np.clip(pred_svr_log, 0, None)), 0, None)

    final_pred = weights[0] * pred_xgb + weights[1] * pred_lgb + weights[2] * pred_svr
    final_pred = np.clip(final_pred, 0, None)

    # ── Submission ──
    submission = pd.DataFrame({"id": test_fe["id"].values, "sales": final_pred})
    submission = submission.sort_values("id").reset_index(drop=True)
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    # ── Feature Importance ──
    print("\nTop 20 Feature Importances (XGBoost):")
    importances = pd.Series(
        xgb_final.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, imp in importances.head(20).items():
        print(f"  {feat:30s} {imp:.4f}")
    importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"))

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"Submission: {os.path.abspath(sub_path)}")
    print(f"Ensemble Val RMSLE: {ensemble_score:.5f}")
    print(f"{'=' * 60}")

    return {
        "ensemble_val_rmsle": ensemble_score,
        "xgb_val_rmsle": xgb_score,
        "lgb_val_rmsle": lgb_score,
        "svr_val_rmsle": svr_score,
        "weights": weights.tolist(),
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
        "svr_params": svr_params,
    }


if __name__ == "__main__":
    results = main()
