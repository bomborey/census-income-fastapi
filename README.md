# Census Income Prediction API

Repository platform: GitHub
Repository URL: https://github.com/<your-github-username>/<your-repo-name>

A Random Forest model trained on the UCI Census Income data predicts whether income is `>50K` or `<=50K`. The model is served with FastAPI, tested with pytest, linted with flake8 through GitHub Actions, and deployed to Render only after CI passes.

## Project layout

| Path | Purpose |
| --- | --- |
| `data/census.csv` | Cleaned census data (spaces after commas removed) |
| `starter/ml/data.py` | `process_data` (one-hot encoding and label binarizing) |
| `starter/ml/model.py` | Train, save, load, inference, metrics, slice metrics |
| `starter/train_model.py` | Trains the model, saves artifacts, writes `slice_output.txt` |
| `model/` | `model.joblib`, `encoder.joblib`, `lb.joblib` |
| `main.py` | FastAPI app (`GET /`, `POST /predict`) |
| `tests/` | Model unit tests and API tests |
| `live_post.py` | Sends a POST request to the live API |
| `model_card.md` | Model card |
| `slice_output.txt` | Metrics for every value of every categorical feature |
| `.github/workflows/ci.yml` | GitHub Actions CI (flake8 and pytest, Python 3.13) |
| `render.yaml` | Render service definition |
| `screenshots/` | Rubric screenshots |

## Run locally (Python 3.13)

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python starter/train_model.py        # optional: artifacts are already committed
flake8 .
pytest -v
uvicorn main:app --reload            # docs at http://127.0.0.1:8000/docs
```

## API

* `GET /` returns `{"message": "Welcome to the Census Income Prediction API!"}`
* `POST /predict` takes one census record (hyphenated field names such as `education-num`, mapped with Pydantic aliases) and returns `{"prediction": ">50K"}` or `{"prediction": "<=50K"}`.

## Live API

Live URL: https://<your-service>.onrender.com

```bash
python live_post.py https://<your-service>.onrender.com
```

## Model performance

Test set (20% stratified hold-out): precision 0.8009, recall 0.5874, F1 0.6777. See `model_card.md` and `slice_output.txt`.
