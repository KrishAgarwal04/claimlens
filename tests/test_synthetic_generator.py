"""Unit and integration tests for the Synthetic External Test-Data Generator."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    SUPPORT_CATEGORIES,
    CATEGORY_SUBCATEGORIES,
    CANONICAL_SUBCATEGORIES,
    MERCHANT_TIERS,
    MERCHANT_SECTORS,
    MERCHANT_REGIONS,
    FORBIDDEN_PRE_ASSIGNMENT_COLUMNS,
    RANDOM_STATE,
)
from src.models.sla_model import SLARiskModel
from external_test.generate_synthetic_test import SyntheticTestGenerator


@pytest.fixture
def generator():
    return SyntheticTestGenerator(seed=RANDOM_STATE)


@pytest.fixture
def sample_dataset(generator):
    return generator.generate(n_samples=200)


def test_generator_reproducibility():
    """Verify that identical random seeds produce byte-identical datasets."""
    gen1 = SyntheticTestGenerator(seed=42)
    gen2 = SyntheticTestGenerator(seed=42)
    df1 = gen1.generate(n_samples=50)
    df2 = gen2.generate(n_samples=50)
    pd.testing.assert_frame_equal(df1, df2)


def test_generator_schema_and_feature_counts(sample_dataset):
    """Verify exact 18 model features, plus ticket_id and actual_sla_breach."""
    df = sample_dataset
    assert len(df) == 200
    assert "ticket_id" in df.columns
    assert "actual_sla_breach" in df.columns

    # 18 expected model features
    model = SLARiskModel.load()
    for feat in model.feature_names:
        assert feat in df.columns, f"Missing required model feature: {feat}"

    assert len(model.feature_names) == 18
    assert df.isnull().sum().sum() == 0, "Generated dataset contains null values"


def test_canonical_vocabularies(sample_dataset):
    """Verify all categories and subcategories strictly match canonical config.py."""
    df = sample_dataset
    assert set(df["category"].unique()).issubset(set(SUPPORT_CATEGORIES))
    assert set(df["sub_category"].unique()).issubset(set(CANONICAL_SUBCATEGORIES))
    assert set(df["merchant_tier"].unique()).issubset(set(MERCHANT_TIERS))
    assert set(df["merchant_sector"].unique()).issubset(set(MERCHANT_SECTORS))
    assert set(df["merchant_region"].unique()).issubset(set(MERCHANT_REGIONS))

    # Parent-child consistency
    for _, row in df.iterrows():
        cat = row["category"]
        sub = row["sub_category"]
        assert sub in CATEGORY_SUBCATEGORIES[cat], f"Subcategory {sub} does not belong to category {cat}"


def test_temporal_coordinates_and_invariants(sample_dataset):
    """Verify mathematical exactness of reconstructed temporal coordinates."""
    df = sample_dataset
    assert df["intake_hour"].between(0, 23).all()
    assert df["intake_day_of_week"].between(0, 6).all()
    assert (df["intake_is_weekend"] == df["intake_day_of_week"].isin([5, 6]).astype(int)).all()
    assert df["intake_month"].between(1, 12).all()

    # Cyclical sin/cos consistency
    expected_hour_sin = np.sin(2.0 * np.pi * df["intake_hour"] / 24.0)
    expected_hour_cos = np.cos(2.0 * np.pi * df["intake_hour"] / 24.0)
    np.testing.assert_allclose(df["intake_hour_sin"], expected_hour_sin, atol=1e-6)
    np.testing.assert_allclose(df["intake_hour_cos"], expected_hour_cos, atol=1e-6)


def test_zero_target_leakage(sample_dataset):
    """Verify forbidden post-intake and outcome columns are absent."""
    df = sample_dataset
    for col in FORBIDDEN_PRE_ASSIGNMENT_COLUMNS:
        assert col not in df.columns, f"Forbidden leakage column detected: {col}"


def test_saved_model_inference_compatibility(sample_dataset):
    """Verify that the existing saved model can execute predictions without errors."""
    model = SLARiskModel.load()
    X_eval = sample_dataset[model.feature_names].copy()
    
    # Run inference
    risk_scores = model.predict_risk(X_eval)
    assert len(risk_scores) == len(sample_dataset)
    assert np.all((risk_scores >= 0.0) & (risk_scores <= 1.0))
    assert not np.isnan(risk_scores).any()


def test_plausible_risk_patterns():
    """Verify documented synthetic risk gradient (Payments/P2 > Notifications/P4)."""
    gen = SyntheticTestGenerator(seed=123)
    df = gen.generate(n_samples=1000)

    # Payments breach rate should be higher than Notifications
    payments_rate = df[df["category"] == "Payments & Checkout"]["actual_sla_breach"].mean()
    notif_rate = df[df["category"] == "Notifications"]["actual_sla_breach"].mean()
    assert payments_rate > notif_rate, f"Expected Payments ({payments_rate:.2f}) > Notifications ({notif_rate:.2f})"

    # P2 breach rate should be higher than P4
    p2_rate = df[df["priority"] == "P2"]["actual_sla_breach"].mean()
    p4_rate = df[df["priority"] == "P4"]["actual_sla_breach"].mean()
    assert p2_rate > p4_rate, f"Expected P2 ({p2_rate:.2f}) > P4 ({p4_rate:.2f})"
