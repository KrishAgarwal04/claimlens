"""Evaluate both frozen SLA models on a supplied completed-ticket CSV.

Example: python -m src.models.evaluate_dataset --dataset PATH --name evaluation_v2
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from src.config import SLA_TARGETS
from src.models.assignment_risk import AssignmentRiskModel, assignment_features
from src.models.evaluate import score
from src.models.sla_model import SLARiskModel
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(dataset, name):
    root = Path(__file__).resolve().parents[2]
    dataset = Path(dataset)
    original_data_hash = digest(dataset)
    data = pd.read_csv(dataset)
    if data.empty or data.ticket_id.duplicated().any():
        raise ValueError("Evaluation requires nonempty data and unique ticket IDs")
    required = ["merchant_id", "assigned_agent_id", "created_at", "closed_at", "resolution_breached", "resolution_hours", "priority"]
    if data[required].isna().any().any():
        raise ValueError("Evaluation rows must have complete identifiers, labels and outcomes")
    for column in ["resolution_breached", "is_reopen_child", "is_incident_ticket"]:
        mapped = data[column].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})
        if mapped.isna().any():
            raise ValueError(f"Invalid binary values in {column}")
        data[column] = mapped.astype(bool)
    for column in ["created_at", "closed_at"]:
        data[column] = pd.to_datetime(data[column], errors="raise")
    if (data.closed_at < data.created_at).any():
        raise ValueError("Resolution timestamps precede creation")
    if not data.priority.isin(SLA_TARGETS).all():
        raise ValueError("Unknown ticket priorities")
    target = data.resolution_breached.astype(int)
    if target.nunique() != 2:
        raise ValueError("Both target classes are required to report ROC-AUC")
    thresholds = data.priority.map(lambda p: SLA_TARGETS[p]["resolution_hours"])
    inconsistent_labels = int((target != (data.resolution_hours > thresholds).astype(int)).sum())
    loader = DataLoader()
    merchants, agents = loader.load_merchants(), loader.load_agents()
    historical, _ = loader.split_tickets(loader.load_tickets())
    unknown_merchants = sorted(set(data.merchant_id)-set(merchants.merchant_id))
    unknown_agents = sorted(set(data.assigned_agent_id)-set(agents.agent_id))
    if unknown_merchants or unknown_agents:
        raise ValueError(f"Missing directory records: merchants={unknown_merchants}, agents={unknown_agents}")
    intake, assignment = SLARiskModel.load(), AssignmentRiskModel.load()
    paths = {"arrival_only": intake.DEFAULT_MODEL_PATH, "assignment_aware": assignment.PATH}
    before = {stage: digest(path) for stage, path in paths.items()}
    engineer = FeatureEngineer()
    matrices = {
        "arrival_only": engineer.build_pre_assignment_features(data, merchants),
        "assignment_aware": assignment_features(data, merchants, agents),
    }
    probabilities = {"arrival_only": intake.predict_risk(matrices["arrival_only"]),
                     "assignment_aware": assignment.predict_risk(matrices["assignment_aware"])}
    operating_thresholds = {"arrival_only": intake.decision_threshold,
                            "assignment_aware": assignment.metadata["decision_threshold"]}
    results = {stage: score(target, values, operating_thresholds[stage]) for stage, values in probabilities.items()}
    predictions = data[["ticket_id", "category", "priority", "assigned_agent_id"]].copy()
    predictions["actual_breach"] = target
    subgroups = {}
    for stage, values in probabilities.items():
        predicted = (values >= operating_thresholds[stage]).astype(int)
        predictions[f"{stage}_risk"] = values
        predictions[f"{stage}_predicted_breach"] = predicted
        predictions[f"{stage}_correct"] = predicted == target
        subgroups[stage] = {}
        for priority in ["P1", "P2", "P3", "P4"]:
            mask = data.priority == priority
            if not mask.any():
                continue
            tn, fp, fn, tp = confusion_matrix(target[mask], predicted[mask], labels=[0, 1]).ravel()
            subgroups[stage][priority] = {
                "samples": int(mask.sum()), "accuracy": float(accuracy_score(target[mask], predicted[mask])),
                "breach_recall": float(recall_score(target[mask], predicted[mask], zero_division=0)),
                "confusion_matrix": dict(zip(["tn", "fp", "fn", "tp"], map(int, [tn, fp, fn, tp]))),
            }
    _, historical_y = engineer.prepare_training_dataset(historical, merchants)
    train_ids, _ = train_test_split(np.arange(len(historical)), test_size=.2, stratify=historical_y, random_state=42)
    report = {
        "dataset": dataset.name, "dataset_sha256": original_data_hash, "rows": len(data),
        "date_range": [str(data.created_at.min()), str(data.created_at.max())],
        "actual_breaches": int(target.sum()), "actual_non_breaches": int((1-target).sum()),
        "audit": {
            "duplicate_ticket_ids": int(data.ticket_id.duplicated().sum()),
            "ticket_ids_matching_training": int(data.ticket_id.isin(historical.iloc[train_ids].ticket_id).sum()),
            "ticket_ids_matching_original_history": int(data.ticket_id.isin(historical.ticket_id).sum()),
            "ticket_ids_matching_previous_evaluation": int(data.ticket_id.isin(pd.read_csv(root / "evaluation_tickets.csv").ticket_id).sum()),
            "labels_inconsistent_with_recorded_resolution_and_sla": inconsistent_labels,
            "rows_after_original_historical_period": int((data.created_at > historical.created_at.max()).sum()),
            "unknown_merchant_ids": unknown_merchants, "unknown_agent_ids": unknown_agents,
            "rows_excluded": 0,
        },
        "model_sha256": before, "results": results, "by_priority": subgroups,
        "protocol": "Frozen saved models and saved thresholds. No fitting, calibration, threshold tuning or row exclusion. Outcome fields used for scoring and label audits only. Agent attributes describe the specified assignment, not arrival-only information.",
        "directory_assumption": "The repository merchant and agent directory is assumed to apply to this evaluation dataset.",
    }
    if before != {stage: digest(path) for stage, path in paths.items()} or digest(dataset) != original_data_hash:
        raise RuntimeError("A model or the input file changed during evaluation")
    report["model_artifacts_unchanged"] = True
    output = root / "docs"
    output.mkdir(exist_ok=True)
    (output / f"{name}_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(output / f"{name}_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", default="external_evaluation")
    arguments = parser.parse_args()
    if Path(arguments.name).name != arguments.name or any(c in arguments.name for c in ["/", "\\", ":"]):
        parser.error("name must be a simple output filename prefix")
    main(arguments.dataset, arguments.name)
