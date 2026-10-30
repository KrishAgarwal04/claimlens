"""Read-only accuracy verification for the saved assignment-time model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.assignment_risk import AssignmentRiskModel, assignment_features
from src.models.evaluate import score
from src.pipeline.loader import DataLoader


def main():
    root = Path(__file__).resolve().parents[2]
    model = AssignmentRiskModel.load()
    before = hashlib.sha256(model.PATH.read_bytes()).hexdigest()
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    merchants, agents = loader.load_merchants(), loader.load_agents()
    y = historical.resolution_breached.astype(int).reset_index(drop=True)
    train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    external = pd.read_csv(root / "evaluation_tickets.csv")
    external["created_at"] = pd.to_datetime(external.created_at)
    if external.closed_at.isna().any() or external.resolution_breached.isna().any():
        raise ValueError("Separate evaluation requires completed, labelled tickets")
    threshold = model.metadata["decision_threshold"]
    results = {
        "historical_holdout": score(y.iloc[test], model.predict_risk(assignment_features(historical.iloc[test], merchants, agents)), threshold),
        "separate_evaluation": score(external.resolution_breached.astype(int), model.predict_risk(assignment_features(external, merchants, agents)), threshold),
    }
    for dataset, values in results.items():
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]:
            if not np.isclose(values[metric], model.metadata["evaluation"][dataset][metric]):
                raise ValueError(f"Stored metric mismatch: {dataset}/{metric}")
    assert hashlib.sha256(model.PATH.read_bytes()).hexdigest() == before
    report = {
        **model.metadata, "model_sha256": before, "evaluation": results,
        "read_only_verification": True,
        "training_holdout_ticket_overlap": len(set(historical.iloc[train].ticket_id) & set(historical.iloc[test].ticket_id)),
        "separate_historical_ticket_overlap": int(external.ticket_id.isin(historical.ticket_id).sum()),
    }
    (root / "docs/assignment_accuracy_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model_sha256": before, "evaluation": results, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
