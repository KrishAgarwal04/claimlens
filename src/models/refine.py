"""Repeated training-only CV to reduce sensitivity to a single fold assignment.

Run after ``python -m src.models.improvement``. No evaluation dataset is loaded.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.models.improvement import best_threshold, candidates, pipeline
from src.models.sla_model import SLARiskModel
from src.models.merchant_context import merchant_candidates
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


def main(merchant_only=False):
    root = Path(__file__).resolve().parents[2]
    destination = root / ".model_experiments"
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    X, y = FeatureEngineer().prepare_training_dataset(historical, loader.load_merchants())
    X, _, y, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    first = json.loads((destination / "selection.json").read_text())
    if first["training_data_hash"] != joblib.hash((X, y)):
        raise ValueError("Training data changed; rerun initial selection")
    models = candidates()
    # Retain leading candidates by training CV; the external assessment is not read.
    leading = sorted(
        (name for name in first["cv_results"] if name in models and not name.startswith("duration")),
        key=lambda name: (first["cv_results"][name]["cv_accuracy"], -first["cv_results"][name]["cv_brier_score"]),
        reverse=True,
    )[:5]
    models = {name: models[name] for name in leading}
    baseline_dir = root / "models/baseline"
    original = SLARiskModel.load(
        baseline_dir / "sla_risk_model.joblib", baseline_dir / "model_metadata.json"
    ) if (baseline_dir / "sla_risk_model.joblib").exists() else SLARiskModel.load()
    models["original_calibrated_forest"] = clone(original.pipeline)
    for c in [0.1, 1.0, 3.0, 10.0]:
        models[f"svm_full_C{c}"] = pipeline(SVC(C=c, probability=True, random_state=42), scale=True)
        preprocessor = ColumnTransformer([
            ("flags", "passthrough", ["is_reopen_child", "is_incident_ticket", "intake_is_weekend"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SLARiskModel.CATEGORICAL_FEATURES),
        ])
        models[f"svm_core_C{c}"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(C=c, probability=True, random_state=42)),
        ])
    for c in [0.01, 0.1, 1.0]:
        estimator = pipeline(LogisticRegression(C=c, max_iter=2000), scale=True)
        models[f"logistic_polynomial_C{c}"] = Pipeline([
            ("preprocessor", estimator.named_steps["preprocessor"]),
            ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("classifier", estimator.named_steps["classifier"]),
        ])
    models.update(merchant_candidates())
    if merchant_only:
        models = merchant_candidates()
    results = dict(first["cv_results"]) if merchant_only else {}
    seeds = [42, 17, 2026]
    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            predictions = [cross_val_predict(
                estimator, X, y, method="predict_proba",
                cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            )[:, 1] for seed in seeds]
            all_predictions = np.concatenate(predictions)
            all_labels = np.tile(y, len(seeds))
            threshold = best_threshold(all_labels, all_predictions)
            results[name] = {
                "cv_accuracy": float(accuracy_score(all_labels, all_predictions >= threshold)),
                "cv_accuracy_std_across_repeats": float(np.std([accuracy_score(y, p >= threshold) for p in predictions])),
                "cv_accuracy_at_0_5": float(accuracy_score(all_labels, all_predictions >= 0.5)),
                "cv_roc_auc": float(np.mean([roc_auc_score(y, p) for p in predictions])),
                "cv_brier_score": float(brier_score_loss(all_labels, all_predictions)),
                "cv_recall": float(recall_score(all_labels, all_predictions >= threshold)),
                "decision_threshold": threshold,
            }
            print(name, results[name], flush=True)
        winner = max(results, key=lambda name: (results[name]["cv_accuracy"], -results[name]["cv_brier_score"]))
        selected = models[winner].fit(X, y) if winner in models else joblib.load(destination / "selected.joblib")
    # Preserve the first assessment and its selection, including unsuccessful results.
    for filename in ["model_improvement_results.json", "MODEL_IMPROVEMENT.md"]:
        source = root / "docs" / filename
        if source.exists():
            target = source.with_name(source.stem + "_round1" + source.suffix)
            if not target.exists():
                target.write_bytes(source.read_bytes())
    report = {
        "selected": winner, "cv_results": results,
        "training_data_hash": joblib.hash((X, y)),
        "selection_data": "1445 historical training rows only; 5-fold stratified CV repeated with seeds 42, 17, 2026",
        "previous_selection": first,
        "evaluation_status": "Evaluation sets were assessed in round 1; a new untouched dataset is required for independent confirmation.",
        "optional_feature_columns": ["merchant_id"] if winner.startswith("merchant_") else [],
    }
    joblib.dump(selected, destination / "selected.joblib")
    (destination / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected from repeated training CV: {winner}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merchant-only", action="store_true", help="Extend a completed repeated-CV search with merchant history candidates only")
    main(merchant_only=parser.parse_args().merchant_only)
