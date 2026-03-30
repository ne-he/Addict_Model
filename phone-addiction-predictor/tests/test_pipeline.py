"""
tests/test_pipeline.py
Automated tests for preprocessing pipeline and model inference.
Run with: python -m pytest phone-addiction-predictor/tests/ -v
Or directly: python phone-addiction-predictor/tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing import preprocess_pipeline
from src.model import predict

# ── load artifacts ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = CatBoostRegressor()
model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
bundle = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
ohe           = bundle["ohe"]
num_medians   = bundle["num_medians"]
cat_modes     = bundle["cat_modes"]
feature_order = bundle["feature_order"]


def make_input(**overrides):
    """Return a default input dict with optional overrides."""
    base = {
        "Age":                       18.0,
        "Gender":                    "Male",
        "Daily_Usage_Hours":         5.0,
        "Sleep_Hours":               7.0,
        "Interllectual_Performance": 70,
        "Social_Interactions":       5,
        "Exercise_Hours":            1.0,
        "Screen_Time_Before_Bed":    1.0,
        "Phone_Checks_Per_Day":      50,
        "Anxiety_Level":             5,
        "Depression_Level":          5,
        "Self_Esteem":               5,
        "Apps_Used_Daily":           10,
        "Time_on_Social_Media":      2.0,
        "Time_on_Gaming":            1.0,
        "Time_on_Education":         1.0,
        "Phone_Usage_Purpose":       "Browsing",
        "Family_Communication":      5,
        "Weekend_Usage_Hours":       6.0,
    }
    base.update(overrides)
    return base


def run_prediction(input_dict):
    processed = preprocess_pipeline(
        input_dict, ohe, scaler, num_medians, cat_modes, feature_order
    )
    return predict(model, processed)


# ── test 1: default input ─────────────────────────────────────────────────────
def test_default_input():
    pred = run_prediction(make_input())
    assert 1.0 <= pred <= 10.0, f"Prediction out of range: {pred}"
    print(f"[PASS] test_default_input: prediction = {pred:.2f}")


# ── test 2: output shape ──────────────────────────────────────────────────────
def test_output_shape():
    processed = preprocess_pipeline(
        make_input(), ohe, scaler, num_medians, cat_modes, feature_order
    )
    assert processed.shape[0] == 1, f"Expected 1 row, got {processed.shape[0]}"
    assert processed.shape[1] == len(feature_order), \
        f"Expected {len(feature_order)} cols, got {processed.shape[1]}"
    print(f"[PASS] test_output_shape: shape = {processed.shape}")


# ── test 3: no NaN in output ──────────────────────────────────────────────────
def test_no_nan_in_output():
    processed = preprocess_pipeline(
        make_input(), ohe, scaler, num_medians, cat_modes, feature_order
    )
    assert not processed.isnull().any().any(), "NaN found in preprocessed output"
    print("[PASS] test_no_nan_in_output")


# ── test 4: feature order consistency ────────────────────────────────────────
def test_feature_order():
    processed = preprocess_pipeline(
        make_input(), ohe, scaler, num_medians, cat_modes, feature_order
    )
    assert list(processed.columns) == feature_order, "Feature order mismatch"
    print(f"[PASS] test_feature_order: {len(feature_order)} features in correct order")


# ── test 5: pipeline idempotence ─────────────────────────────────────────────
def test_idempotence():
    inp = make_input()
    r1 = preprocess_pipeline(inp, ohe, scaler, num_medians, cat_modes, feature_order)
    r2 = preprocess_pipeline(inp, ohe, scaler, num_medians, cat_modes, feature_order)
    pd.testing.assert_frame_equal(r1, r2)
    print("[PASS] test_idempotence: same input → same output")


# ── test 6: prediction range clipping ────────────────────────────────────────
def test_prediction_range():
    for gender in ["Male", "Female", "Other"]:
        for purpose in ["Browsing", "Education", "Gaming", "Social Media", "Other"]:
            pred = run_prediction(make_input(Gender=gender, Phone_Usage_Purpose=purpose))
            assert 1.0 <= pred <= 10.0, f"Out of range: {pred} ({gender}, {purpose})"
    print("[PASS] test_prediction_range: all gender/purpose combos in [1, 10]")


if __name__ == "__main__":
    print("=" * 50)
    print("Running pipeline tests...")
    print("=" * 50)
    test_default_input()
    test_output_shape()
    test_no_nan_in_output()
    test_feature_order()
    test_idempotence()
    test_prediction_range()
    print("=" * 50)
    print("All tests passed!")
