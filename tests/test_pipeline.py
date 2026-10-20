"""Comprehensive tests for data ingestion, validation, and feature engineering.

Tests schema contracts, foreign key referential integrity, open queue invariant guarantees,
and strict target leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np

from src.pipeline.loader import DataLoader, DataValidationError
from src.pipeline.features import FeatureEngineer, DataLeakageError
from src.config import (
    EXPECTED_ROW_COUNTS,
    FORBIDDEN_PRE_ASSIGNMENT_COLUMNS,
    PRE_ASSIGNMENT_FEATURE_COLUMNS,
)


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture
def datasets(loader):
    tickets, agents, merchants = loader.get_datasets()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    return {
        "tickets": tickets,
        "agents": agents,
        "merchants": merchants,
        "hist_tickets": hist_tickets,
        "open_tickets": open_tickets,
    }


def test_merchants_loading_and_validation(loader):
    merchants = loader.load_merchants()
    assert len(merchants) == EXPECTED_ROW_COUNTS["merchants"]
    assert merchants["merchant_id"].nunique() == 112
    assert merchants.isnull().sum().sum() == 0
    assert set(merchants["tier"].unique()) == {"Foundation", "Advanced"}


def test_agents_loading_and_validation(loader):
    agents = loader.load_agents()
    assert len(agents) == EXPECTED_ROW_COUNTS["agents"]
    assert agents["agent_id"].nunique() == 20
    assert agents.isnull().sum().sum() == 0
    assert (agents["efficiency_multiplier"] > 0).all()
    assert set(agents["tier"].unique()) == {"L1", "L2", "L3"}


def test_tickets_referential_integrity(loader, datasets):
    tickets = datasets["tickets"]
    merchants = datasets["merchants"]
    agents = datasets["agents"]

    # All merchant_ids in tickets must exist in merchants.csv
    valid_merchants = set(merchants["merchant_id"])
    assert set(tickets["merchant_id"]).issubset(valid_merchants)

    # All assigned agents in tickets must exist in agents.csv
    assigned_agents = set(tickets["assigned_agent_id"].dropna().astype(int))
    valid_agents = set(agents["agent_id"])
    assert assigned_agents.issubset(valid_agents)


def test_temporal_sequence_in_historical_tickets(datasets):
    hist = datasets["hist_tickets"]
    # created_at <= first_response_at <= closed_at
    assert (hist["created_at"] <= hist["first_response_at"]).all()
    assert (hist["first_response_at"] <= hist["closed_at"]).all()


def test_open_queue_invariants(datasets):
    open_tickets = datasets["open_tickets"]
    assert len(open_tickets) == EXPECTED_ROW_COUNTS["open_queue_tickets"]

    # 7 post-intake columns MUST be 100% null
    forbidden_null_cols = [
        "assigned_agent_id",
        "category_mismatch",
        "first_response_at",
        "closed_at",
        "ttfr_hours",
        "resolution_hours",
        "csat_score",
    ]
    for col in forbidden_null_cols:
        assert open_tickets[col].isnull().all(), f"Expected all nulls for {col} in open queue"


def test_never_train_on_open_tickets(feature_engineer, datasets):
    merchants = datasets["merchants"]
    open_tickets = datasets["open_tickets"]

    # Attempting to train on open tickets must raise ValueError
    with pytest.raises(ValueError, match="Cannot train on open queue tickets"):
        feature_engineer.prepare_training_dataset(open_tickets, merchants)

    # Attempting to train on mixed dataset must also raise ValueError
    mixed_tickets = datasets["tickets"]
    with pytest.raises(ValueError, match="Cannot train on open queue tickets"):
        feature_engineer.prepare_training_dataset(mixed_tickets, merchants)


def test_pre_assignment_feature_matrices(feature_engineer, datasets):
    merchants = datasets["merchants"]
    hist_tickets = datasets["hist_tickets"]
    open_tickets = datasets["open_tickets"]

    X_train, y_train = feature_engineer.prepare_training_dataset(hist_tickets, merchants)
    X_open = feature_engineer.prepare_open_queue_dataset(open_tickets, merchants)

    assert len(X_train) == EXPECTED_ROW_COUNTS["historical_tickets"]
    assert len(y_train) == EXPECTED_ROW_COUNTS["historical_tickets"]
    assert len(X_open) == EXPECTED_ROW_COUNTS["open_queue_tickets"]

    # Schema equality between training and inference feature sets
    assert list(X_train.columns) == list(X_open.columns)
    assert list(X_train.columns) == PRE_ASSIGNMENT_FEATURE_COLUMNS

    # Target sanity check
    assert set(y_train.unique()) == {0, 1}
    assert y_train.isnull().sum() == 0


def test_strict_target_leakage_exclusion(feature_engineer, datasets):
    merchants = datasets["merchants"]
    hist_tickets = datasets["hist_tickets"]
    X_train, _ = feature_engineer.prepare_training_dataset(hist_tickets, merchants)

    # Check that none of the forbidden columns exist in X_train
    for col in FORBIDDEN_PRE_ASSIGNMENT_COLUMNS:
        assert col not in X_train.columns, f"Target leakage violation: '{col}' present in feature matrix!"

    # Explicit checks for agent assignment and outcome columns
    assert "assigned_agent_id" not in X_train.columns
    assert "category_mismatch" not in X_train.columns
    assert "resolution_hours" not in X_train.columns
    assert "ttfr_hours" not in X_train.columns
    assert "first_response_at" not in X_train.columns
    assert "closed_at" not in X_train.columns
    assert "csat_score" not in X_train.columns
    assert "resolution_breached" not in X_train.columns
    assert "response_breached" not in X_train.columns
    assert "is_reopened" not in X_train.columns


def test_leakage_is_reopened_forbidden_in_intake_pipeline(feature_engineer, datasets):
    """Verify is_reopened is flagged as forbidden post-resolution outcome and absent from features."""
    merchants = datasets["merchants"]
    hist_tickets = datasets["hist_tickets"]
    open_tickets = datasets["open_tickets"]

    X_train, _ = feature_engineer.prepare_training_dataset(hist_tickets, merchants)
    X_open = feature_engineer.prepare_open_queue_dataset(open_tickets, merchants)

    assert "is_reopened" not in X_train.columns
    assert "is_reopened" not in X_open.columns

    # Validate that an input containing is_reopened triggers DataLeakageError
    leaked_reopen_df = pd.DataFrame({"is_reopened": [1]})
    with pytest.raises(DataLeakageError, match="CRITICAL TARGET LEAKAGE"):
        feature_engineer.validate_no_leakage(leaked_reopen_df)


def test_data_leakage_validator_detects_leakage(feature_engineer):
    # Simulated leaked dataframe
    leaked_df = pd.DataFrame({
        "category": ["Payments & Checkout"],
        "resolution_hours": [12.5],  # Forbidden
    })
    with pytest.raises(DataLeakageError, match="CRITICAL TARGET LEAKAGE"):
        feature_engineer.validate_no_leakage(leaked_df)

    # Simulated assignment leakage dataframe
    assignment_leak_df = pd.DataFrame({
        "category": ["Payments & Checkout"],
        "assigned_agent_id": [201],  # Forbidden at intake
    })
    with pytest.raises(DataLeakageError, match="CRITICAL TARGET LEAKAGE"):
        feature_engineer.validate_no_leakage(assignment_leak_df)


def test_loader_diagnostics_report(loader):
    report = loader.run_diagnostics()
    assert report["status"] == "PASS"
    assert report["counts"]["merchants"] == 112
    assert report["counts"]["agents"] == 20
    assert report["counts"]["historical_tickets"] == 1807
    assert report["counts"]["open_queue_tickets"] == 250
    assert report["referential_integrity"]["unmatched_merchants"] == 0
    assert report["referential_integrity"]["unmatched_agents"] == 0
