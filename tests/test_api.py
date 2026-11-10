"""Tests for the FastAPI endpoints."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }


def test_post_predict_above_50k():
    record = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=record)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_predict_below_50k():
    record = {
        "age": 22,
        "workclass": "Private",
        "fnlgt": 201490,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=record)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_missing_field():
    r = client.post("/predict", json={"age": 30})
    assert r.status_code == 422
