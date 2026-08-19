"""
lstm_forecast.py
Uses an LSTM neural network to forecast future stock prices, using a
sliding window of past prices as input. Benchmarked against the naive
baseline and Prophet results from prophet_forecast.py.

Run from project root:
    python src/lstm_forecast.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_fetcher import fetch_stock_data


WINDOW_SIZE = 60   # use last 60 days to predict the next day
TEST_DAYS = 30     # same test window as Prophet, for fair comparison


def create_sequences(data: np.ndarray, window_size: int):
    """
    Converts a 1D price series into (X, y) sequences for supervised learning:
    X = [price[t-window : t]], y = price[t]
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(window_size: int):
    """A small, simple LSTM architecture -- deliberately kept small to
    avoid overfitting on a relatively small daily stock dataset (~500 rows)."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_lstm_forecast(symbol: str, window_size: int = WINDOW_SIZE, test_days: int = TEST_DAYS):
    print(f"Fetching data for {symbol} ...")
    raw_df = fetch_stock_data(symbol, period="3y")  # LSTMs benefit from more history
    prices = raw_df[["Close"]].values

    # Scale to [0,1] -- LSTMs train much better on normalized data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Chronological split: reserve last (test_days + window_size) for testing
    # so we have enough lookback history to build the first test window
    split_idx = len(scaled_prices) - test_days
    train_scaled = scaled_prices[:split_idx]
    # test needs window_size of history BEFORE the test period starts
    test_scaled = scaled_prices[split_idx - window_size:]

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")

    print("Training LSTM ...")
    model = build_lstm_model(window_size)
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predict and inverse-transform back to actual price scale
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    lstm_mae = mean_absolute_error(actual, preds)
    lstm_rmse = np.sqrt(mean_squared_error(actual, preds))

    # Naive baseline over the same test window, for direct comparison
    naive_preds = np.roll(actual, 1)
    naive_preds[0] = actual[0]  # first prediction has no prior, use actual as fallback
    naive_mae = mean_absolute_error(actual[1:], naive_preds[1:])
    naive_rmse = np.sqrt(mean_squared_error(actual[1:], naive_preds[1:]))

    print(f"\n=== FORECAST EVALUATION: {symbol} ===")
    print(f"{'Method':<20}{'MAE':>12}{'RMSE':>12}")
    print(f"{'Naive Baseline':<20}{naive_mae:>12.2f}{naive_rmse:>12.2f}")
    print(f"{'LSTM':<20}{lstm_mae:>12.2f}{lstm_rmse:>12.2f}")

    if lstm_mae < naive_mae:
        improvement = (1 - lstm_mae / naive_mae) * 100
        print(f"\nLSTM beats naive baseline by {improvement:.1f}% (MAE)")
    else:
        decline = (lstm_mae / naive_mae - 1) * 100
        print(f"\nLSTM performs {decline:.1f}% WORSE than naive baseline (MAE)")
        print("Also an honest, common finding -- report it rather than hide it.")

    # Save predictions for dashboard/visualization use
    results_df = pd.DataFrame({
        "actual": actual.flatten(),
        "lstm_prediction": preds.flatten(),
    })
    results_df.to_csv(f"data/processed/{symbol.replace('.NS','')}_lstm_forecast.csv", index=False)
    print(f"\nSaved to data/processed/{symbol.replace('.NS','')}_lstm_forecast.csv")

    return model, results_df


if __name__ == "__main__":
    symbol = "RELIANCE.NS"
    model, results_df = run_lstm_forecast(symbol)