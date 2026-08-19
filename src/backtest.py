"""
backtest.py
Simulates trading based on the short-term classifier's signals over a
historical period the model was NOT trained on, and compares the
resulting equity curve against a simple buy-and-hold baseline.

CRITICAL DESIGN CHOICE: the model is retrained using ONLY data before a
cutoff date, then tested purely on data after that cutoff. This avoids
lookahead bias -- using a model that was trained on the full dataset
(including the "future" test period) to backtest would give a falsely
inflated result, since the model would have implicitly "seen" the answers.

STRATEGY: each day, if the model predicts "Up" and no position is
currently held, ENTER and HOLD for the model's actual prediction horizon
(5 trading days) before re-evaluating -- rather than daily rebalancing
on every new signal. This version replaced an earlier daily-rebalance
approach that ignored the model's horizon and likely whipsawed in/out of
positions; that version underperformed buy-and-hold in ~77% of cases
across a 49-stock test, motivating this fix.

Run from project root:
    python src/backtest.py
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data_fetcher import fetch_stock_data
from feature_engineering import add_technical_indicators, label_signal

FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "BB_width", "ATR_14",
    "daily_return", "volatility_20", "price_vs_SMA20",
]

TRADING_DAYS_PER_YEAR = 252


def prepare_data(symbol: str, period: str = "3y"):
    raw_df = fetch_stock_data(symbol, period=period)
    featured_df = add_technical_indicators(raw_df)
    labeled_df = label_signal(featured_df, horizon_days=5, threshold=0.02)
    return labeled_df


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.25):
    """Splits by DATE, not randomly -- test period is strictly after
    train period, so nothing from the future leaks into training."""
    df = df.dropna(subset=FEATURE_COLUMNS + ["signal_label"]).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def train_backtest_model(train_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["signal_label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def run_backtest(test_df: pd.DataFrame, model, scaler, horizon_days: int = 5):
    """
    Simulates the strategy respecting the model's actual prediction
    horizon: when the model predicts "Up" and we're not already holding a
    position, ENTER and HOLD for `horizon_days` trading days, ignoring
    new signals during that hold (since the original prediction was about
    that whole window, not just the next single day). Only re-evaluate
    once the holding period ends.

    This replaces the earlier daily-rebalance version, which likely
    whipsawed in and out of positions based on a signal meant for a
    longer horizon -- a probable cause of its underperformance at scale.
    """
    X_test = test_df[FEATURE_COLUMNS]
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    test_df = test_df.reset_index(drop=True).copy()
    test_df["predicted_signal"] = predictions

    position = 0
    hold_days_remaining = 0
    positions = []

    for i in range(len(test_df)):
        if hold_days_remaining > 0:
            # Already holding from a prior "Up" signal -- stay in, don't
            # re-evaluate until the horizon completes
            position = 1
            hold_days_remaining -= 1
        elif test_df.loc[i, "predicted_signal"] == "Up":
            # New "Up" signal and we're flat -- enter and start the hold clock
            position = 1
            hold_days_remaining = horizon_days - 1  # this day counts as day 1
        else:
            position = 0

        positions.append(position)

    test_df["position"] = positions
    test_df["strategy_return"] = test_df["position"] * test_df["daily_return"]
    test_df["buyhold_return"] = test_df["daily_return"]

    test_df["strategy_equity"] = (1 + test_df["strategy_return"].fillna(0)).cumprod()
    test_df["buyhold_equity"] = (1 + test_df["buyhold_return"].fillna(0)).cumprod()

    return test_df


def compute_metrics(returns: pd.Series, equity_curve: pd.Series) -> dict:
    returns = returns.fillna(0)
    total_return = equity_curve.iloc[-1] - 1
    n_days = len(returns)
    annualized_return = (equity_curve.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / n_days) - 1 if n_days > 0 else np.nan

    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / volatility if volatility > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    winning_days = (returns > 0).sum()
    active_days = (returns != 0).sum()
    win_rate = winning_days / active_days if active_days > 0 else np.nan

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2) if not np.isnan(annualized_return) else None,
        "annualized_volatility_pct": round(volatility * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1) if not np.isnan(win_rate) else None,
        "active_trading_days": int(active_days),
        "total_days": n_days,
    }


def backtest_symbol(symbol: str, period: str = "3y", test_fraction: float = 0.25):
    print(f"\n{'='*60}\nBacktesting {symbol}\n{'='*60}")

    df = prepare_data(symbol, period=period)
    train_df, test_df = chronological_split(df, test_fraction=test_fraction)
    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} days)")
    print(f"Test period:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} days)")

    model, scaler = train_backtest_model(train_df)
    results_df = run_backtest(test_df, model, scaler)

    strategy_metrics = compute_metrics(results_df["strategy_return"], results_df["strategy_equity"])
    buyhold_metrics = compute_metrics(results_df["buyhold_return"], results_df["buyhold_equity"])

    print(f"\n{'Metric':<30}{'Strategy':>15}{'Buy & Hold':>15}")
    for key in strategy_metrics:
        s_val = strategy_metrics[key]
        b_val = buyhold_metrics[key]
        print(f"{key:<30}{str(s_val):>15}{str(b_val):>15}")

    return results_df, strategy_metrics, buyhold_metrics


if __name__ == "__main__":
    # Test across a few representative stocks -- large, mid, small cap
    test_symbols = ["RELIANCE.NS", "PERSISTENT.NS", "HAPPSTMNDS.NS"]

    all_results = []
    for symbol in test_symbols:
        try:
            results_df, strat_metrics, bh_metrics = backtest_symbol(symbol)
            strat_metrics["symbol"] = symbol
            strat_metrics["type"] = "Strategy"
            bh_metrics["symbol"] = symbol
            bh_metrics["type"] = "Buy & Hold"
            all_results.append(strat_metrics)
            all_results.append(bh_metrics)

            results_df.to_csv(f"data/processed/backtest_{symbol.replace('.NS','')}.csv", index=False)
        except Exception as e:
            print(f"FAILED for {symbol}: {e}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("data/processed/backtest_summary.csv", index=False)
    print(f"\n\n=== SUMMARY SAVED to data/processed/backtest_summary.csv ===")
    print(summary_df[["symbol", "type", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]])