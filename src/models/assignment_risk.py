"""SLA risk once a real or proposed agent is known, before work starts.

This is a separate prediction stage from the arrival-only SLA model. No response
timestamps, resolution durations or observed mismatch flags are model features.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models.sla_model import SLARiskModel
from src.pipeline.features import DataLeakageError, FeatureEngineer
from src.config import OUTCOME_COLUMNS, POST_RESPONSE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_NUMERIC = [
    "proposed_agent_efficiency", "proposed_agent_tier_rank", "domain_match",
    "region_match", "priority_capability_gap",
]
ASSIGNMENT_CATEGORICAL = ["proposed_agent_tier", "proposed_agent_specialty", "proposed_agent_region"]
FEATURES = SLARiskModel.NUMERIC_FEATURES + SLARiskModel.CATEGORICAL_FEATURES + ASSIGNMENT_NUMERIC + ASSIGNMENT_CATEGORICAL


def assignment_features(tickets, merchants, agents, proposed_agent_ids=None):
    """Join directory attributes for the specified assignment; do not read outcomes."""
    base = FeatureEngineer().build_pre_assignment_features(tickets, merchants).reset_index(drop=True)
    ids = tickets["assigned_agent_id"].to_numpy() if proposed_agent_ids is None else np.asarray(proposed_agent_ids)
    if len(ids) != len(base) or pd.isna(ids).any():
        raise ValueError("A known proposed agent is required for every ticket")
    directory = agents.set_index("agent_id").reindex(ids).reset_index(drop=True)
    if directory["primary_category"].isna().any():
        raise ValueError("Unknown proposed agent ID")
    out = base[SLARiskModel.NUMERIC_FEATURES + SLARiskModel.CATEGORICAL_FEATURES].copy()
    out["proposed_agent_tier"] = directory["tier"]
    out["proposed_agent_specialty"] = directory["primary_category"]
    out["proposed_agent_region"] = directory["shift_region"]
    out["proposed_agent_efficiency"] = directory["efficiency_multiplier"].astype(float)
    out["proposed_agent_tier_rank"] = directory["tier"].map({"L1": 1, "L2": 2, "L3": 3})
    out["domain_match"] = (out.category == directory.primary_category).astype(int)
    out["region_match"] = (out.merchant_region == directory.shift_region).astype(int)
    required = out.priority.map({"P1": 3, "P2": 2, "P3": 1, "P4": 1}).fillna(1)
    out["priority_capability_gap"] = (required-out.proposed_agent_tier_rank).clip(lower=0)
    return out[FEATURES]


class AssignmentRiskModel:
    PATH = ROOT / "models/assignment_risk_model.joblib"

    def __init__(self, pipeline, metadata):
        self.pipeline = pipeline
        self.metadata = metadata

    def predict_risk(self, features):
        forbidden = set(OUTCOME_COLUMNS + POST_RESPONSE_COLUMNS + ["category_mismatch", "assigned_agent_id", "ticket_id", "merchant_id"])
        if forbidden.intersection(features.columns):
            raise DataLeakageError("Assignment model accepts only intake and proposed-agent features")
        missing = set(FEATURES) - set(features.columns)
        if missing:
            raise ValueError(f"Missing assignment features: {sorted(missing)}")
        with threadpool_limits(limits=2):
            return self.pipeline.predict_proba(features[FEATURES])[:, 1]

    def predict(self, features):
        return (self.predict_risk(features) >= self.metadata["decision_threshold"]).astype(int)

    def save(self, path=None):
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata}, path or self.PATH)

    @classmethod
    def load(cls, path=None):
        payload = joblib.load(path or cls.PATH)
        return cls(payload["pipeline"], payload["metadata"])
