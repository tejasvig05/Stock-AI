"""
train_final_model.py
Trains the final chosen model (Random Forest, class-balanced) on the FULL
training dataset and saves it + the scaler to disk, so the dashboard can
load them instantly instead of retraining every time.

Run from project root:
    python src/train_final_model.py
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "BB_width", "ATR_14",
    "daily_return", "volatility_20", "price_vs_SMA20",
]
TARGET_COLUMN = "signal_label"
RETURN_COLUMN = "future_return"  # continuous target for the regression model


if __name__ == "__main__":
    print("Loading dataset ...")
    df = pd.read_csv("data/processed/training_dataset.csv")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Keep a small holdout just to sanity-check the saved model works as
    # expected -- final production model below still trains on ALL data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest (class-balanced) ...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("\n=== Holdout sanity check ===")
    print(classification_report(y_test, preds))

    # Refit on ALL available data for the final production model (more data
    # = better generalization, now that we've validated the approach above)
    print("\nRefitting on full dataset for production model ...")
    scaler_final = StandardScaler()
    X_scaled_full = scaler_final.fit_transform(X)
    model_final = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
    )
    model_final.fit(X_scaled_full, y)

    # Save model + scaler + feature list together
    joblib.dump(model_final, "data/processed/rf_model.pkl")
    joblib.dump(scaler_final, "data/processed/scaler.pkl")
    joblib.dump(FEATURE_COLUMNS, "data/processed/feature_columns.pkl")

    print("\nSaved:")
    print("  data/processed/rf_model.pkl")
    print("  data/processed/scaler.pkl")
    print("  data/processed/feature_columns.pkl")

    # ================================================================
    # REGRESSION MODEL -- predicts expected % return over the horizon,
    # giving a concrete "likely gain/loss of X%" number, not just a
    # category. Trained and evaluated the same disciplined way.
    # ================================================================
    print("\n" + "=" * 60)
    print("Training regression model for expected return ...")

    df_reg = df.dropna(subset=FEATURE_COLUMNS + [RETURN_COLUMN])
    X_reg = df_reg[FEATURE_COLUMNS]
    y_reg = df_reg[RETURN_COLUMN]  # e.g. 0.03 = +3% over the horizon

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.15, random_state=42
    )
    scaler_reg = StandardScaler()
    Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
    Xr_test_scaled = scaler_reg.transform(Xr_test)

    reg_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    reg_model.fit(Xr_train_scaled, yr_train)

    reg_preds = reg_model.predict(Xr_test_scaled)
    mae = mean_absolute_error(yr_test, reg_preds)
    r2 = r2_score(yr_test, reg_preds)

    print(f"\n=== Regression holdout check ===")
    print(f"MAE (return): {mae:.4f}  (i.e. avg error of {mae*100:.2f} percentage points)")
    print(f"R^2 score: {r2:.4f}")
    if r2 < 0.05:
        print("NOTE: Low R^2 is expected and common for stock return prediction --")
        print("this mirrors the naive-baseline finding from the Prophet/LSTM experiments.")
        print("Report this honestly: the expected-return number is a rough directional")
        print("estimate from historical patterns, not a precise forecast.")

    # Refit on full data for production
    scaler_reg_final = StandardScaler()
    Xr_scaled_full = scaler_reg_final.fit_transform(X_reg)
    reg_model_final = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    reg_model_final.fit(Xr_scaled_full, y_reg)

    joblib.dump(reg_model_final, "data/processed/rf_regressor.pkl")
    joblib.dump(scaler_reg_final, "data/processed/scaler_reg.pkl")
    print("\nSaved:")
    print("  data/processed/rf_regressor.pkl")
    print("  data/processed/scaler_reg.pkl")