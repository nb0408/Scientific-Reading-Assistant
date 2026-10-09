"""
Phase 6c: Train and Evaluate the Knowledge-Gap Prediction Model
----------------------------------------------------------------
Trains two classical ML models on the behavioural/text-complexity
features used by the project:

    1. Logistic Regression
    2. Random Forest

Outputs:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Confusion matrices
    - ROC curve comparison
    - Model-metric comparison chart
    - Random Forest feature-importance chart
    - metrics.csv

The trained model bundle is saved as:
    gap_model.joblib

Usage:
    python train_model.py
    python train_model.py real_data.csv

The input CSV must contain the columns listed in FEATURE_NAMES
plus a binary "label" column.
"""

import sys
import csv
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)

from features import FEATURE_NAMES


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
RESULTS_DIR = "results"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_csv(path: str):
    """Load features and binary labels from a CSV file."""
    X, y = [], []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        required_columns = set(FEATURE_NAMES + ["label"])
        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            X.append([float(row[name]) for name in FEATURE_NAMES])
            y.append(int(row["label"]))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    if len(X) == 0:
        raise ValueError("CSV contains no data rows.")

    unique_labels = np.unique(y)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"Labels must be binary 0/1, but found: {unique_labels}"
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "The dataset must contain both label classes (0 and 1)."
        )

    return X, y


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, X_test, y_test, model_name: str):
    """Evaluate a classifier and return metrics + prediction probabilities."""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 55}")
    print(f"{model_name}")
    print(f"{'=' * 55}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1-Score:   {f1:.3f}")
    print(f"ROC-AUC:    {auc:.3f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Struggle"],
            zero_division=0,
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def save_confusion_matrix(cm, model_name: str):
    """Save a publication-ready confusion matrix."""
    filename = model_name.lower().replace(" ", "_")
    path = os.path.join(
        RESULTS_DIR,
        f"confusion_matrix_{filename}.png"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Struggle"],
    )
    disp.plot(ax=ax)

    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_roc_curve(results):
    """Save ROC curves for both models."""
    path = os.path.join(RESULTS_DIR, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    for result in results:
        fpr, tpr, _ = roc_curve(
            # y_test is supplied later through the result dictionary
            result["y_test"],
            result["y_proba"],
        )

        ax.plot(
            fpr,
            tpr,
            label=f'{result["model"]} (AUC = {result["roc_auc"]:.3f})',
        )

    # Random-classifier reference line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Knowledge-Gap Prediction")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_model_comparison(results):
    """Save a bar chart comparing the main evaluation metrics."""
    path = os.path.join(RESULTS_DIR, "model_comparison.png")

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, result in enumerate(results):
        values = [result[m] for m in metric_names]

        offset = (i - (len(results) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=result["model"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Knowledge-Gap Prediction Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_feature_importance(model, feature_names):
    """Save Random Forest feature importance chart."""
    path = os.path.join(
        RESULTS_DIR,
        "random_forest_feature_importance.png"
    )

    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        np.array(feature_names)[order],
        importances[order],
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_metrics_csv(results):
    """Save all model metrics to a CSV file."""
    path = os.path.join(RESULTS_DIR, "metrics.csv")

    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "model": result["model"],
                "accuracy": f'{result["accuracy"]:.4f}',
                "precision": f'{result["precision"]:.4f}',
                "recall": f'{result["recall"]:.4f}',
                "f1": f'{result["f1"]:.4f}',
                "roc_auc": f'{result["roc_auc"]:.4f}',
            })

    return path


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------

def main():
    data_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "synthetic_training_data.csv"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("KNOWLEDGE-GAP PREDICTION MODEL TRAINING")
    print("=" * 70)

    print(f"\nLoading data from: {data_path}")

    X, y = load_csv(data_path)

    print(f"Loaded samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Positive (struggle) samples: {int(y.sum())}")
    print(f"Negative (normal) samples: {int((y == 0).sum())}")
    print(f"Struggle proportion: {y.mean():.1%}")

    # -------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # -------------------------------------------------------------
    # Standardisation
    # -------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Model 1: Logistic Regression
    # -------------------------------------------------------------

    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    log_reg.fit(X_train_scaled, y_train)

    lr_result = evaluate(
        log_reg,
        X_test_scaled,
        y_test,
        "Logistic Regression",
    )

    lr_result["y_test"] = y_test

    # -------------------------------------------------------------
    # Model 2: Random Forest
    # -------------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=RANDOM_STATE,
    )

    rf.fit(X_train, y_train)

    rf_result = evaluate(
        rf,
        X_test,
        y_test,
        "Random Forest",
    )

    rf_result["y_test"] = y_test

    results = [lr_result, rf_result]

    # -------------------------------------------------------------
    # Save evaluation figures
    # -------------------------------------------------------------

    print("\nGenerating evaluation figures...")

    lr_cm_path = save_confusion_matrix(
        lr_result["confusion_matrix"],
        "Logistic Regression",
    )

    rf_cm_path = save_confusion_matrix(
        rf_result["confusion_matrix"],
        "Random Forest",
    )

    roc_path = save_roc_curve(results)
    comparison_path = save_model_comparison(results)
    importance_path = save_feature_importance(
        rf,
        FEATURE_NAMES,
    )
    metrics_path = save_metrics_csv(results)

    # -------------------------------------------------------------
    # Select best model using ROC-AUC
    # -------------------------------------------------------------

    if rf_result["roc_auc"] >= lr_result["roc_auc"]:
        best_model = rf
        best_name = "Random Forest"
        needs_scaler = False
        best_auc = rf_result["roc_auc"]
    else:
        best_model = log_reg
        best_name = "Logistic Regression"
        needs_scaler = True
        best_auc = lr_result["roc_auc"]

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)
    print(f"Selected model: {best_name}")
    print(f"ROC-AUC:        {best_auc:.3f}")

    # -------------------------------------------------------------
    # Save trained model bundle
    # -------------------------------------------------------------

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler if needs_scaler else None,
            "feature_names": FEATURE_NAMES,
            "model_name": best_name,
        },
        "gap_model.joblib",
    )

    print("\nSaved trained model:")
    print("  gap_model.joblib")

    # -------------------------------------------------------------
    # Random Forest feature importance
    # -------------------------------------------------------------

    print("\nRandom Forest Feature Importances:")

    for name, importance in sorted(
        zip(FEATURE_NAMES, rf.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {name:<25} {importance:.3f}")

    # -------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED RESULTS")
    print("=" * 70)

    print(f"  {lr_cm_path}")
    print(f"  {rf_cm_path}")
    print(f"  {roc_path}")
    print(f"  {comparison_path}")
    print(f"  {importance_path}")
    print(f"  {metrics_path}")

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
