"""Train the census income model, save artifacts, and write slice metrics.

Run from the project root:
    python starter/train_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    format_slice_metrics,
    inference,
    save_artifact,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
SLICE_OUTPUT = ROOT / "slice_output.txt"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    data = pd.read_csv(DATA_PATH, skipinitialspace=True)

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)

    save_artifact(model, MODEL_DIR / "model.joblib")
    save_artifact(encoder, MODEL_DIR / "encoder.joblib")
    save_artifact(lb, MODEL_DIR / "lb.joblib")

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Test set | Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {fbeta:.4f}"
    )

    lines = [
        f"Overall test set | n: {len(test)} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {fbeta:.4f}",
        "",
    ]
    for feature in CAT_FEATURES:
        results = compute_slice_metrics(
            model, test, feature, CAT_FEATURES, encoder, lb
        )
        lines.append(f"===== Slice metrics for '{feature}' =====")
        lines.extend(format_slice_metrics(results))
        lines.append("")

    SLICE_OUTPUT.write_text("\n".join(lines))
    print(f"Slice metrics written to {SLICE_OUTPUT}")


if __name__ == "__main__":
    main()
