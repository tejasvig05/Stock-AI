"""
prophet_forecast.py
Uses Facebook Prophet to forecast future stock price trend. This is a
different problem than the classification model -- here we're predicting
an actual price trajectory, mainly for trend visualization on the dashboard.

IMPORTANT (know this for interviews): Prophet is designed for data with
trend + seasonality (e.g. retail sales, web traffic). Stock prices behave
close to a random walk and don't have clean seasonality, so Prophet tends
to just extrapolate the recent trend smoothly. We benchmark it against a
naive baseline (tomorrow's price = today's price) to see if it actually
adds value beyond that.

Run from project root:
    python src/prophet_forecast.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_fetcher import fetch_stock_data


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet requires exactly two columns: 'ds' (date) and 'y' (value).
    Also strips timezone info, since Prophet doesn't handle tz-aware
    datetimes well.
    """
    prophet_df = df[["Date", "Close"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    return prophet_df


def train_test_split_timeseries(df: pd.DataFrame, test_days: int = 30):
    """Time-series split -- last `test_days` rows held out for evaluation.
    NEVER randomly shuffle time series data, always split chronologically."""
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def naive_baseline_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Naive baseline: every future day = last known price. This is the bar
    any real forecasting model needs to beat to be worth using."""
    last_price = train["y"].iloc[-1]
    return np.full(horizon, last_price)


def run_prophet_forecast(symbol: str, test_days: int = 30):
    print(f"Fetching data for {symbol} ...")
    raw_df = fetch_stock_data(symbol, period="2y")  # more history helps Prophet
    prophet_df = prepare_prophet_data(raw_df)

    train, test = train_test_split_timeseries(prophet_df, test_days=test_days)
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

    # --- Train Prophet ---
    print("Training Prophet model ...")
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # controls trend flexibility
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=test_days, freq="B")  # business days
    forecast = model.predict(future)

    # Align forecast to test period
    forecast_test = forecast.set_index("ds").reindex(test["ds"])["yhat"].values

    # --- Naive baseline ---
    naive_preds = naive_baseline_forecast(train, len(test))

    # --- Evaluate both ---
    actual = test["y"].values

    # Some forecast values may be NaN if dates didn't align (holidays etc.)
    valid_mask = ~np.isnan(forecast_test)

    prophet_mae = mean_absolute_error(actual[valid_mask], forecast_test[valid_mask])
    prophet_rmse = np.sqrt(mean_squared_error(actual[valid_mask], forecast_test[valid_mask]))

    naive_mae = mean_absolute_error(actual, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(actual, naive_preds))

    print(f"\n=== FORECAST EVALUATION: {symbol} ===")
    print(f"{'Method':<20}{'MAE':>12}{'RMSE':>12}")
    print(f"{'Naive Baseline':<20}{naive_mae:>12.2f}{naive_rmse:>12.2f}")
    print(f"{'Prophet':<20}{prophet_mae:>12.2f}{prophet_rmse:>12.2f}")

    if prophet_mae < naive_mae:
        improvement = (1 - prophet_mae / naive_mae) * 100
        print(f"\nProphet beats naive baseline by {improvement:.1f}% (MAE)")
    else:
        decline = (prophet_mae / naive_mae - 1) * 100
        print(f"\nProphet performs {decline:.1f}% WORSE than naive baseline (MAE)")
        print("This is a common, honest finding for stock price forecasting --")
        print("worth discussing directly rather than hiding it.")

    return model, forecast, train, test


if __name__ == "__main__":
    symbol = "RELIANCE.NS"
    model, forecast, train, test = run_prophet_forecast(symbol, test_days=30)

    # Save forecast output for dashboard use later
    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(60)
    forecast_out.to_csv(f"data/processed/{symbol.replace('.NS','')}_prophet_forecast.csv", index=False)
    print(f"\nSaved forecast to data/processed/{symbol.replace('.NS','')}_prophet_forecast.csv")