"""Model training, persistence, inference, and evaluation helpers."""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifact(obj, path):
    """Save a model, encoder, or label binarizer to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_artifact(path):
    """Load a model, encoder, or label binarizer saved with joblib."""
    return joblib.load(Path(path))


def compute_slice_metrics(
    model, data, feature, categorical_features, encoder, lb, label="salary"
):
    """Compute metrics for each unique value of `feature`.

    Inputs
    ------
    model : trained classifier
    data : pd.DataFrame
        Raw (unprocessed) data that includes the label column.
    feature : str
        Name of the categorical feature to slice on.
    categorical_features : list[str]
    encoder : fitted OneHotEncoder
    lb : fitted LabelBinarizer
    label : str
        Name of the label column.

    Returns
    -------
    results : list[dict]
        One entry per unique value with its count, precision, recall, fbeta.
    """
    results = []
    for value in sorted(data[feature].unique()):
        subset = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            subset,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results.append({
            "feature": feature,
            "value": value,
            "n": int(len(subset)),
            "precision": float(precision),
            "recall": float(recall),
            "fbeta": float(fbeta),
        })
    return results


def format_slice_metrics(results):
    """Turn slice metric dictionaries into printable lines."""
    lines = []
    for r in results:
        lines.append(
            f"{r['feature']} = {r['value']} | n: {r['n']} | "
            f"Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | "
            f"F1: {r['fbeta']:.4f}"
        )
    return lines
