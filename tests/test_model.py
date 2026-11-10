"""Unit tests for the ML functions in starter/ml."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_artifact,
    save_artifact,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
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


@pytest.fixture(scope="module")
def sample():
    data = pd.read_csv(ROOT / "data" / "census.csv")
    return data.sample(1000, random_state=0)


@pytest.fixture(scope="module")
def processed(sample):
    return process_data(
        sample, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )


def test_process_data_shapes(sample, processed):
    X, y, encoder, lb = processed
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(sample) == y.shape[0]
    assert set(np.unique(y)) <= {0, 1}


def test_train_model_returns_random_forest(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_binary_array(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_values():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)


def test_save_and_load_artifact(tmp_path, processed):
    _, _, encoder, _ = processed
    path = tmp_path / "encoder.joblib"
    save_artifact(encoder, path)
    loaded = load_artifact(path)
    assert type(loaded) is type(encoder)
    assert len(loaded.categories_) == len(encoder.categories_)


def test_compute_slice_metrics_one_row_per_value(sample, processed):
    X, y, encoder, lb = processed
    model = train_model(X, y)
    results = compute_slice_metrics(
        model, sample, "sex", CAT_FEATURES, encoder, lb
    )
    assert len(results) == sample["sex"].nunique()
    for r in results:
        assert 0.0 <= r["fbeta"] <= 1.0
