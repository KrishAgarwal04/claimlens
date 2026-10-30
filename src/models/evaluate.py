"""Evaluate the saved SLA model without retraining or changing its artifacts.

Run from the repository root: python -m src.models.evaluate
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.sla_model import SLARiskModel
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]


def score(y, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    accuracy = float(accuracy_score(y, predictions))
    n = len(y)
    z = 1.96
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1+z*z/n)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "samples": n,
        "decision_threshold": threshold,
        "accuracy": accuracy,
        "accuracy_95pct_wilson_interval": [center-margin, center+margin],
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "always_predict_breach_accuracy": float(np.mean(y)),
        "confusion_matrix": dict(zip(["tn", "fp", "fn", "tp"], map(int, [tn, fp, fn, tp]))),
    }


def main():
    model = SLARiskModel.load()
    threshold = model.metadata.get("decision_threshold", 0.5)
    artifact_hash = hashlib.sha256(model.DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    merchants = loader.load_merchants()
    engineer = FeatureEngineer()
    X, y = engineer.prepare_training_dataset(historical, merchants)
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,
        random_state=model.metadata["random_state"],
    )
    results = {"historical_holdout": score(y_test, model.predict_risk(X_test), threshold)}
    holdout = model.metadata["holdout_results"]
    expected = holdout.get("selected_model", holdout.get("calibrated_selected_model"))
    results["historical_holdout"]["matches_saved_metadata"] = all(
        np.isclose(results["historical_holdout"][metric], expected[metric])
        for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier_score"]
    )

    external = pd.read_csv(ROOT / "evaluation_tickets.csv")
    for column in ["created_at", "closed_at"]:
        external[column] = pd.to_datetime(external[column], errors="raise")
    if external["closed_at"].isna().any() or external["resolution_breached"].isna().any():
        raise ValueError("Evaluation tickets must have completed outcomes and labels")
    X_external = engineer.build_pre_assignment_features(external, merchants)
    results["evaluation_tickets"] = score(
        external["resolution_breached"].astype(int), model.predict_risk(X_external), threshold
    )
    features = model.feature_names
    training_hashes = set(pd.util.hash_pandas_object(X_train[features], index=False))
    results["evaluation_tickets"]["feature_rows_matching_training"] = int(
        pd.util.hash_pandas_object(X_external[features], index=False).isin(training_hashes).sum()
    )
    results["evaluation_tickets"]["ticket_ids_matching_historical"] = int(
        external.ticket_id.isin(historical.ticket_id).sum()
    )
    canonical = pd.read_csv(ROOT / "external_test/opspilot_external_test_canonical.csv")
    results["synthetic_canonical"] = score(
        canonical.actual_sla_breach.astype(int), model.predict_risk(canonical[features]), threshold
    )
    if hashlib.sha256(model.DEFAULT_MODEL_PATH.read_bytes()).hexdigest() != artifact_hash:
        raise RuntimeError("Model artifact changed during evaluation")
    report = {
        "threshold": threshold, "model_sha256": artifact_hash, "results": results,
        "limitations": [
            "Holdout is reconstructed from the documented split; metadata agreement checks reproducibility.",
            "All supplied datasets are synthetic; these scores do not establish real-world accuracy.",
            "External labels are used only for scoring; no model fitting or threshold tuning is performed.",
        ],
    }
    destination = ROOT / "docs/model_accuracy_results.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved results to {destination}")


if __name__ == "__main__":
    main()
