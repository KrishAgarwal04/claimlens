"""Assess a training-selected candidate and promote it only if accuracy improves.

Run ``python -m src.models.improvement`` before this command. Existing model and
metadata are backed up before promotion. Evaluation labels never fit the model.
"""

import json
import shutil
import hashlib
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.models.evaluate import score
from src.models.sla_model import SLARiskModel
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]


def display_name(name):
    for prefix, label in [
        ("catboost", "CatBoost"), ("gradient", "Gradient boosting"),
        ("hist", "Histogram gradient boosting"), ("logistic", "Regularized logistic regression"),
        ("forest", "Random forest"), ("soft_voting", "Soft voting ensemble"),
        ("duration", "Log-duration risk model"), ("original", "Calibrated random forest"),
    ]:
        if name.startswith(prefix):
            return label
    return name


def main():
    selection = json.loads((ROOT / ".model_experiments/selection.json").read_text())
    name = selection["selected"]
    threshold = selection["cv_results"][name]["decision_threshold"]
    candidate = joblib.load(ROOT / ".model_experiments/selected.joblib")
    baseline_dir = ROOT / "models/baseline"
    if (baseline_dir / "sla_risk_model.joblib").exists():
        baseline = SLARiskModel.load(
            baseline_dir / "sla_risk_model.joblib", baseline_dir / "model_metadata.json"
        )
    else:
        baseline = SLARiskModel.load()
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    merchants = loader.load_merchants()
    engineer = FeatureEngineer()
    X, y = engineer.prepare_training_dataset(historical, merchants)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    if selection.get("training_data_hash") != joblib.hash((X_train, y_train)):
        raise ValueError("Selection artifact does not match current training data; rerun model selection")
    external = pd.read_csv(ROOT / "evaluation_tickets.csv")
    external["created_at"] = pd.to_datetime(external.created_at)
    if external.closed_at.isna().any() or external.resolution_breached.isna().any():
        raise ValueError("Separate evaluation requires completed and labelled tickets")
    X_external = engineer.build_pre_assignment_features(external, merchants)
    canonical = pd.read_csv(ROOT / "external_test/opspilot_external_test_canonical.csv")
    datasets = {
        "historical_holdout": (X_test, y_test),
        "evaluation_tickets": (X_external, external.resolution_breached.astype(int)),
        "synthetic_canonical": (canonical[baseline.feature_names], canonical.actual_sla_breach.astype(int)),
    }
    comparison = {}
    baseline_threshold = baseline.metadata.get("decision_threshold", 0.5)
    with threadpool_limits(limits=2):
        for dataset, (features, target) in datasets.items():
            original_prob = baseline.predict_risk(features)
            selected_prob = candidate.predict_proba(features)[:, 1]
            original_correct = (original_prob >= baseline_threshold) == np.asarray(target)
            selected_correct = (selected_prob >= threshold) == np.asarray(target)
            fixes = int((~original_correct & selected_correct).sum())
            regressions = int((original_correct & ~selected_correct).sum())
            comparison[dataset] = {
                "baseline": score(target, original_prob, baseline_threshold),
                "candidate": score(target, selected_prob, threshold),
                "newly_correct": fixes, "newly_incorrect": regressions,
                "paired_mcnemar_exact_pvalue": float(binomtest(
                    fixes, fixes+regressions, p=0.5
                ).pvalue) if fixes+regressions else 1.0,
            }
    promote = all(
        comparison[dataset]["candidate"]["accuracy"] > comparison[dataset]["baseline"]["accuracy"]
        for dataset in ["historical_holdout", "evaluation_tickets"]
    )
    configurations = set()
    previous_selection = selection
    while previous_selection:
        configurations.update(previous_selection.get("cv_results", {}))
        previous_selection = previous_selection.get("previous_selection")
    report = {
        "selected_model": name, "decision_threshold": threshold,
        "configurations_tested": len(configurations),
        "active_model": name if promote else baseline.metadata["selected_model"],
        "baseline_model_sha256": hashlib.sha256((
            baseline_dir / "sla_risk_model.joblib" if (baseline_dir / "sla_risk_model.joblib").exists()
            else SLARiskModel.DEFAULT_MODEL_PATH
        ).read_bytes()).hexdigest(),
        "candidate_model_sha256": hashlib.sha256((ROOT / ".model_experiments/selected.joblib").read_bytes()).hexdigest(),
        "selection": selection, "comparison": comparison, "promoted": promote,
        "method": "Model and threshold selected using training OOF predictions only; fixed evaluation sets used for assessment and deployment gate.",
        "limitations": [
            "All datasets are synthetic; accuracy is not a real-world performance guarantee.",
            "Evaluation sets existed and earlier results were known before this assessment; use a new untouched dataset for independent confirmation.",
            "Selection CV scores have selection bias; holdout scores are reported separately.",
        ],
    }
    (ROOT / "docs/model_improvement_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    outcome = "Improved candidate deployed." if promote else (
        "Candidate rejected; the original model remains active at "
        f"{comparison['historical_holdout']['baseline']['accuracy']:.2%} holdout accuracy and "
        f"{comparison['evaluation_tickets']['baseline']['accuracy']:.2%} separate-evaluation accuracy."
    )
    lines = ["# Model accuracy comparison", "", outcome, "", report["method"], "",
             f"Selected candidate: `{name}`; decision threshold: {threshold:.3f}. Configurations tested: {len(configurations)}.", "",
             "| Dataset | Baseline accuracy | Candidate accuracy | Candidate precision | Candidate recall | Candidate ROC-AUC | Paired p-value |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, values in comparison.items():
        before, after = values["baseline"], values["candidate"]
        lines.append(f"| {dataset} (n={after['samples']}) | {before['accuracy']:.2%} | {after['accuracy']:.2%} | {after['precision']:.2%} | {after['recall']:.2%} | {after['roc_auc']:.4f} | {values['paired_mcnemar_exact_pvalue']:.4f} |")
    lines += ["", f"Candidate promoted: {promote}.", ""] + report["limitations"]
    lines += ["", "A paired p-value >= 0.05 does not establish a statistically significant accuracy difference on that dataset."]
    (ROOT / "docs/MODEL_IMPROVEMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": name, "threshold": threshold, "comparison": comparison, "promoted": promote}, indent=2), flush=True)
    if not promote:
        print("Current model retained: candidate did not improve accuracy on both required evaluation sets.")
        return

    # Explanations come from the deployed estimator, using training rows only.
    explanation_features = baseline.feature_names + selection.get("optional_feature_columns", [])
    with threadpool_limits(limits=2):
        importances = permutation_importance(
            candidate, X_train[explanation_features], y_train,
            scoring="roc_auc", n_repeats=4, random_state=42,
        ).importances_mean
    positive = np.maximum(importances, 0)
    positive /= positive.sum() if positive.sum() else 1
    top = sorted(zip(explanation_features, positive), key=lambda item: item[1], reverse=True)[:15]
    updated = SLARiskModel()
    updated.pipeline = candidate
    updated.model_type = name
    updated.calibrate = isinstance(candidate, CalibratedClassifierCV)
    updated.metadata = {
        **baseline.metadata,
        "selected_model": name,
        "model_display_name": display_name(name),
        "is_calibrated": updated.calibrate,
        "library_versions": {package: version(package) for package in ["numpy", "pandas", "scikit-learn", "catboost", "joblib"]},
        "decision_threshold": threshold,
        "optional_feature_columns": selection.get("optional_feature_columns", []),
        "selection_method": report["method"],
        "selection_cv_results": selection["cv_results"],
        "holdout_results": {
            **baseline.metadata["holdout_results"],
            "selected_model": {"model_name": name, **comparison["historical_holdout"]["candidate"]},
        },
        "external_evaluation_results": comparison,
        "feature_importance_method": "Normalized positive permutation importance on training data (ROC-AUC); descriptive, not causal.",
        "top_feature_importances": [{"feature": feature, "importance": float(value)} for feature, value in top],
        "limitations": report["limitations"] + baseline.metadata.get("limitations", []),
    }
    baseline_dir.mkdir(exist_ok=True)
    if not (baseline_dir / "sla_risk_model.joblib").exists():
        shutil.copy2(SLARiskModel.DEFAULT_MODEL_PATH, baseline_dir / "sla_risk_model.joblib")
        shutil.copy2(SLARiskModel.DEFAULT_METADATA_PATH, baseline_dir / "model_metadata.json")
    updated.save()
    print("Improved model saved; original artifact and metadata retained under models/baseline/.")


if __name__ == "__main__":
    main()
