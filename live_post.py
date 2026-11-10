"""Send one POST request to the live API and print the result.

Usage:
    python live_post.py https://<your-service>.onrender.com
or set API_URL in the environment.
"""
import os
import sys

import requests

DEFAULT_URL = "https://census-income-api.onrender.com"

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


def main():
    base_url = (
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("API_URL", DEFAULT_URL)
    )
    url = base_url.rstrip("/") + "/predict"
    # Free Render instances sleep when idle, so allow time to wake up.
    response = requests.post(url, json=record, timeout=120)
    print(f"URL: {url}")
    print(f"Status code: {response.status_code}")
    print(f"Inference result: {response.json()}")


if __name__ == "__main__":
    main()
