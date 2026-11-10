# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by Rey Rosales in November 2026 as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project. It is a scikit-learn 1.7.2 `RandomForestClassifier` with 100 trees, a maximum depth of 15, a minimum of 2 samples per leaf, and `random_state=42`. Categorical features are one-hot encoded with a fitted `OneHotEncoder` (unknown categories are ignored), and the label is encoded with a fitted `LabelBinarizer`. The model, encoder, and label binarizer are saved as `model/model.joblib`, `model/encoder.joblib`, and `model/lb.joblib`.

## Intended Use

The model predicts whether a person's annual income is above or below $50,000 from census attributes. It is intended for learning and demonstration purposes, such as practicing model deployment, API design, and slice-based evaluation. It is not intended for real decisions about individuals, including hiring, lending, housing, insurance, or eligibility for services.

## Training Data

The data is the Census Income (Adult) dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income), extracted from the 1994 US Census database. It contains 32,561 rows with 14 features (6 numeric, 8 categorical) and a binary `salary` label (`<=50K` or `>50K`). About 24% of rows are labelled `>50K`. Leading spaces after commas were removed from the raw CSV; column names were not changed. Missing values appear as `?` and were kept as their own category. A stratified random 80% of the rows (26,048) were used for training.

## Evaluation Data

The remaining stratified 20% of the rows (6,513) were held out as the test set. The test set was processed with the encoder and label binarizer fitted on the training data only.

## Metrics

The model was evaluated with precision, recall, and F1 score, with `>50K` treated as the positive class. On the held-out test set the model achieved a precision of 0.8009, a recall of 0.5874, and an F1 score of 0.6777.

Performance was also measured on slices of every categorical feature, and the full output is in `slice_output.txt`. For example, the F1 score is 0.6830 for men and 0.6460 for women, and 0.6836 for White and 0.5946 for Black individuals. Slices with very few rows, such as `workclass = Never-worked` with one row, produce metrics that should not be treated as reliable.

## Ethical Considerations

The dataset contains sensitive attributes, including race, sex, marital status, and native country, and the model uses them as inputs. The data reflects income patterns and social inequalities from 1994, so the model can reproduce historical biases. Slice results show uneven performance across groups; for instance, recall is lower for women and for Black and Amer-Indian-Eskimo individuals, meaning high earners in those groups are missed more often. Any use beyond education would require a formal fairness review.

## Caveats and Recommendations

The data is more than thirty years old and the $50,000 threshold is not adjusted for inflation, so predictions do not describe today's population. Recall is moderate (about 0.59), so the model misses a large share of people who earn more than $50,000; the decision threshold could be tuned if recall matters more than precision. Small slices make some group metrics unstable. Recommended next steps are hyperparameter tuning with cross-validation, threshold calibration, fairness analysis with a tool such as Aequitas, and retraining on more recent data before any real-world use.
