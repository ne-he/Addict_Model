# Phone Addiction Level Predictor

Predicts how dependent someone is on their smartphone, on a scale of 1.0 to 10.0, from
their usage patterns, sleep, exercise, and self reported mental health.

This is the original coursework version. A production rewrite with a FastAPI service,
tests, CI, and a live deployment lives at
[Addictv2](https://github.com/ne-he/Addictv2).

## What it predicts

The model outputs a continuous `Addiction_Level` score, which maps to three bands:

| Score | Band | Reading |
| --- | --- | --- |
| 1.0 to 3.9 | Low | Healthy, balanced use |
| 4.0 to 6.9 | Moderate | Signs of overuse, worth limiting screen time |
| 7.0 to 10.0 | High | High dependency risk |

## Inputs

Nineteen features across four groups:

- **Profile:** age, gender, sleep hours
- **Phone activity:** daily usage hours, weekend usage hours, phone checks per day, apps used daily
- **Purpose and focus:** primary usage purpose, time on social media, time on gaming, time on education
- **Mental and physical:** anxiety level, depression level, self esteem, exercise hours, social interaction, family communication

## Model

CatBoost regressor, chosen after comparing candidates in the notebook and tuning
hyperparameters. The notebook also evaluates a stacking ensemble.

Preprocessing lives in one module, `src/preprocessing.py`, and is imported by both the
training script and the Streamlit app. That is deliberate: if training and serving each had
their own transform code, the two would eventually disagree and the app would quietly feed
the model differently shaped data than it was trained on.

## Repository layout

```
AOL_Machine_Learning.ipynb        EDA, feature engineering, model search, tuning, stacking
Phone_Addiction.csv               dataset
about.md                          full project write up
flow.md                           pipeline walkthrough
phone-addiction-predictor/
  app.py                          Streamlit interface
  train_and_save.py               trains CatBoost and writes artifacts to models/
  src/preprocessing.py            shared transform logic
  src/model.py
  models/                         catboost_model.cbm, encoders.pkl, scaler.pkl
  tests/                          pipeline and edge case tests
```

## Running it

```bash
cd phone-addiction-predictor
pip install -r requirements.txt
streamlit run app.py
```

To retrain from the dataset and regenerate the artifacts in `models/`:

```bash
python train_and_save.py
```

Run the tests with `pytest tests/`.

## A note on the metrics

The dataset is synthetic. The model scores well on it, but that reflects how the data was
generated, not clinical validity. Nothing here is a diagnostic tool.
