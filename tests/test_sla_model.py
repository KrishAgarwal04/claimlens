"""Tests for Day 2 SLA Risk Model training, calibration, and inference.

Verifies zero target leakage, strict exclusion of open queue tickets from training,
reproducibility of predictions, probability calibration bounds, and model persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline.loader import DataLoader
from src.pipeline.features import FeatureEngineer, DataLeakageError
from src.models.sla_model import SLARiskModel
from src.config import MODELS_DIR


@pytest.fixture(scope="module")
def dataset_artifacts():
    loader = DataLoader()
    tickets = loader.load_tickets()
    merchants = loader.load_merchants()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    fe = FeatureEngineer()
    X_train, y_train = fe.prepare_training_dataset(hist_tickets, merchants)
    X_open = fe.prepare_open_queue_dataset(open_tickets, merchants)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_open": X_open,
        "hist_tickets": hist_tickets,
        "open_tickets": open_tickets,
    }


def test_leakage_prevention_in_model_inputs(dataset_artifacts):
    """Model must reject inputs containing forbidden post-intake columns."""
    X_train = dataset_artifacts["X_train"]
    model = SLARiskModel()

    # Inject forbidden resolution_hours
    leaked_X = X_train.copy()
    leaked_X["resolution_hours"] = 14.5
    with pytest.raises(DataLeakageError):
        model._validate_inputs(leaked_X)

    # Inject forbidden assigned_agent_id
    assigned_leak_X = X_train.copy()
    assigned_leak_X["assigned_agent_id"] = 201
    with pytest.raises(DataLeakageError):
        model._validate_inputs(assigned_leak_X)

    # Inject forbidden is_reopened (post-resolution outcome leakage)
    reopened_leak_X = X_train.copy()
    reopened_leak_X["is_reopened"] = 1
    with pytest.raises(DataLeakageError):
        model._validate_inputs(reopened_leak_X)


def test_leakage_is_reopened_not_in_model_features():
    """Verify is_reopened is not in SLARiskModel features and input features match expected."""
    model = SLARiskModel()
    assert "is_reopened" not in model.NUMERIC_FEATURES
    assert "is_reopened" not in model.feature_names
    assert len(model.NUMERIC_FEATURES) == 12
    assert len(model.CATEGORICAL_FEATURES) == 6
    assert len(model.feature_names) == 18


def test_model_rejects_open_tickets(dataset_artifacts):
    """Model must strictly refuse to train on tickets containing open queue rows."""
    open_tickets = dataset_artifacts["open_tickets"]
    model = SLARiskModel()
    with pytest.raises(ValueError, match="Open tickets have placeholder breach outcomes"):
        model._validate_inputs(open_tickets)


def test_model_reproducibility(dataset_artifacts):
    """Model fitted with fixed random_state must yield identical predictions."""
    X_train = dataset_artifacts["X_train"].iloc[:200]
    y_train = dataset_artifacts["y_train"].iloc[:200]
    X_eval = dataset_artifacts["X_train"].iloc[200:250]

    model1 = SLARiskModel(model_type="random_forest", calibrate=False, random_state=42)
    model1.fit(X_train, y_train)
    preds1 = model1.predict_risk(X_eval)

    model2 = SLARiskModel(model_type="random_forest", calibrate=False, random_state=42)
    model2.fit(X_train, y_train)
    preds2 = model2.predict_risk(X_eval)

    np.testing.assert_allclose(preds1, preds2, err_msg="Model predictions must be deterministic")


def test_probability_bounds_and_shapes(dataset_artifacts):
    """Probabilities must lie strictly in [0.0, 1.0] and sum to 1.0 across classes."""
    model = SLARiskModel.load()
    X_open = dataset_artifacts["X_open"]

    proba = model.predict_proba(X_open)
    risk = model.predict_risk(X_open)

    assert proba.shape == (len(X_open), 2)
    assert risk.shape == (len(X_open),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_open)), atol=1e-5)


def test_model_significantly_outperforms_baseline(dataset_artifacts):
    """Calibrated Random Forest must decisively beat the majority baseline."""
    model = SLARiskModel.load()
    meta = model.metadata
    holdout = meta["holdout_results"]

    baseline_roc = holdout["majority_baseline"]["roc_auc"]
    model_roc = holdout["calibrated_selected_model"]["roc_auc"]
    assert baseline_roc == 0.50
    assert model_roc > 0.85, f"Expected ROC-AUC > 0.85, got {model_roc}"

    baseline_brier = holdout["majority_baseline"]["brier_score"]
    model_brier = holdout["calibrated_selected_model"]["brier_score"]
    assert model_brier < baseline_brier, "Calibrated model must have substantially lower Brier score than baseline"


def test_model_persistence_and_loading(dataset_artifacts, tmp_path):
    """Saved model artifact must reload cleanly and yield identical risk predictions."""
    loaded_model = SLARiskModel.load()
    X_open = dataset_artifacts["X_open"]
    orig_risk = loaded_model.predict_risk(X_open)

    temp_model_path = tmp_path / "temp_sla_model.joblib"
    temp_meta_path = tmp_path / "temp_metadata.json"
    loaded_model.save(temp_model_path, temp_meta_path)

    reloaded = SLARiskModel.load(temp_model_path, temp_meta_path)
    reloaded_risk = reloaded.predict_risk(X_open)

    np.testing.assert_allclose(orig_risk, reloaded_risk, atol=1e-6)
    assert reloaded.metadata["selected_model"] == loaded_model.metadata["selected_model"]


def test_open_queue_inference_completeness(dataset_artifacts):
    """Inference on all 250 open queue tickets must be non-null and correctly formatted."""
    model = SLARiskModel.load()
    X_open = dataset_artifacts["X_open"]
    risks = model.predict_risk(X_open)

    assert len(risks) == 250
    assert not np.isnan(risks).any()
    # Check that risks produce varied predictions rather than constant output
    assert np.std(risks) > 0.10, "Risk distribution should reflect ticket priority and category variance"
