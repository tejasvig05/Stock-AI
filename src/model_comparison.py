"""
model_comparison.py
Trains and benchmarks multiple ML algorithms on the short-term signal
classification task (Up / Down / Sideways). Compares accuracy, F1-score,
and per-class performance to decide the best model -- same benchmarking
approach used in the IPL project.

Run from project root:
    python src/model_comparison.py
"""

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_upper", "BB_lower", "BB_width", "ATR_14",
    "daily_return", "volatility_20", "price_vs_SMA20",
]

TARGET_COLUMN = "signal_label"


def load_and_prepare_data(path: str):
    """Load the training dataset and split into features (X) and target (y)."""
    df = pd.read_csv(path)

    # Drop any remaining NaNs in feature columns just in case
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y, df


def get_models():
    """Returns a dict of {model_name: model_instance} to benchmark.
    class_weight='balanced' is used where supported, to counter the
    model's tendency to over-predict the majority 'Sideways' class."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),  # no native class_weight support
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),  # no native class_weight support
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),  # no class_weight concept
        "Naive Bayes": GaussianNB(),  # no class_weight concept
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    }


def benchmark_models(X_train, X_test, y_train, y_test):
    """Trains each model, evaluates it, and returns a results DataFrame."""
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"Training: {name} ...")
        start = time.time()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        elapsed = time.time() - start
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        f1_weighted = f1_score(y_test, preds, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1 (macro)": round(f1_macro, 4),
            "F1 (weighted)": round(f1_weighted, 4),
            "Train time (s)": round(elapsed, 2),
        })

        print(f"  Accuracy: {acc:.4f} | F1 (macro): {f1_macro:.4f} | Time: {elapsed:.2f}s")

    return pd.DataFrame(results).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Loading dataset ...")
    X, y, full_df = load_and_prepare_data("data/processed/training_dataset.csv")
    print(f"Dataset shape: {X.shape}, Classes: {y.unique()}")

    # Train/test split -- stratify to keep class proportions consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Quick correlation check -- flag highly redundant features (e.g. SMA_20
    # vs EMA_20 tend to move almost identically)
    print("\n=== FEATURE CORRELATION CHECK (|corr| > 0.9) ===")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns for row in upper.index
        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.9
    ]
    if high_corr_pairs:
        for f1, f2, corr_val in high_corr_pairs:
            print(f"  {f1} <-> {f2}: {corr_val:.3f}")
    else:
        print("  None found above 0.9 threshold.")

    # Scale features -- important for Logistic Regression, KNN, SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n=== BENCHMARKING MODELS ===\n")
    results_df = benchmark_models(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n=== FINAL LEADERBOARD (sorted by F1 macro) ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest model: {best_model_name}")

    # Detailed report for the best model
    models = get_models()
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    best_preds = best_model.predict(X_test_scaled)

    print(f"\n=== DETAILED REPORT: {best_model_name} ===")
    print(classification_report(y_test, best_preds))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # Save leaderboard
    results_df.to_csv("data/processed/model_comparison_results.csv", index=False)
    print("\nSaved leaderboard to data/processed/model_comparison_results.csv")