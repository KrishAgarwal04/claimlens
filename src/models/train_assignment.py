"""Training-only selection and fixed-set evaluation for assignment-time SLA risk."""

import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.models.assignment_risk import (
    AssignmentRiskModel, assignment_features, ASSIGNMENT_NUMERIC, ASSIGNMENT_CATEGORICAL, FEATURES,
)
from src.models.evaluate import score
from src.models.improvement import best_threshold
from src.models.sla_model import SLARiskModel
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


def candidates(profile="compact"):
    classifiers = {}
    if profile == "compact":
        # Deliberately small, regularized candidates; selection never reads test scores.
        for c in [0.001, 0.003, 0.01, 0.03]:
            classifiers[f"compact_logistic_C{c}"] = LogisticRegression(C=c, max_iter=2000)
        for trees in [40, 80, 120]:
            classifiers[f"compact_stumps_{trees}"] = GradientBoostingClassifier(
                max_depth=1, n_estimators=trees, learning_rate=0.04,
                min_samples_leaf=40, random_state=42,
            )
    for leaves in ([4, 8, 15] if profile == "full" else []):
        for regularization in [3.0, 15.0]:
            classifiers[f"hist_leaves{leaves}_l2{regularization}"] = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.04, max_leaf_nodes=leaves,
                min_samples_leaf=20, l2_regularization=regularization,
                early_stopping=False, random_state=42,
            )
    for depth in ([2, 3] if profile == "full" else []):
        for trees in [120, 250]:
            classifiers[f"gradient_depth{depth}_trees{trees}"] = GradientBoostingClassifier(
                max_depth=depth, n_estimators=trees, learning_rate=0.04,
                min_samples_leaf=15, random_state=42,
            )
    if profile == "full":
        classifiers["random_forest"] = RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=2,
        )
    for c in ([0.1, 1.0, 10.0] if profile == "full" else []):
        classifiers[f"logistic_C{c}"] = LogisticRegression(C=c, max_iter=2000)
    return {name: Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", StandardScaler() if "logistic" in name else "passthrough", SLARiskModel.NUMERIC_FEATURES + ASSIGNMENT_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SLARiskModel.CATEGORICAL_FEATURES + ASSIGNMENT_CATEGORICAL),
        ])), ("classifier", estimator),
    ]) for name, estimator in classifiers.items()}


def main(profile="compact"):
    root = Path(__file__).resolve().parents[2]
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    merchants, agents = loader.load_merchants(), loader.load_agents()
    X = assignment_features(historical, merchants, agents)
    y = historical.resolution_breached.astype(int).reset_index(drop=True)
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    models = candidates(profile)
    trials = {}
    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            p = cross_val_predict(estimator, X.iloc[train_ids], y.iloc[train_ids], method="predict_proba",
                cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
            threshold = 0.5 if profile == "compact" else best_threshold(y.iloc[train_ids], p)
            trials[name] = {
                "accuracy": float(accuracy_score(y.iloc[train_ids], p >= threshold)),
                "roc_auc": float(roc_auc_score(y.iloc[train_ids], p)),
                "brier_score": float(brier_score_loss(y.iloc[train_ids], p)),
                "threshold": threshold,
            }
            print(name, trials[name], flush=True)
        if profile == "compact":
            eligible = [name for name in trials if 0.80 <= trials[name]["accuracy"] <= 0.90]
            if not eligible:
                raise RuntimeError("No compact candidate meets the requested training-CV accuracy band; active model unchanged")
            winner = min(eligible, key=lambda name: (abs(trials[name]["accuracy"]-0.85), trials[name]["brier_score"]))
        else:
            winner = max(trials, key=lambda name: (trials[name]["accuracy"], -trials[name]["brier_score"]))
        threshold = trials[winner]["threshold"]
        selected = models[winner].fit(X.iloc[train_ids], y.iloc[train_ids])
        holdout = score(y.iloc[test_ids], selected.predict_proba(X.iloc[test_ids])[:, 1], threshold)
        external = pd.read_csv(root / "evaluation_tickets.csv")
        external["created_at"] = pd.to_datetime(external.created_at)
        if external.closed_at.isna().any():
            raise ValueError("Evaluation outcomes must be completed")
        external_X = assignment_features(external, merchants, agents)
        separate = score(external.resolution_breached.astype(int), selected.predict_proba(external_X)[:, 1], threshold)
        # Global evidence is computed on training data only, for the actual selected model.
        importance = permutation_importance(selected, X.iloc[train_ids], y.iloc[train_ids],
            scoring="roc_auc", n_repeats=3, random_state=42).importances_mean
    metadata = {
        "stage": "assignment_time", "display_name": "Compact assignment-aware SLA risk" if profile == "compact" else "Assignment-aware SLA risk",
        "profile": profile,
        "selection_objective": "Closest to 85% training-CV accuracy within the requested 80-90% band, fixed threshold 0.5" if profile == "compact" else "Highest training-CV accuracy",
        "selected_model": winner, "decision_threshold": threshold,
        "train_samples": len(train_ids), "test_samples": len(test_ids),
        "feature_columns": FEATURES, "training_cv": trials,
        "evaluation": {"historical_holdout": holdout, "separate_evaluation": separate},
        "top_features": [{"feature": feature, "importance": float(value)} for feature, value in
                         sorted(zip(FEATURES, importance), key=lambda pair: pair[1], reverse=True)[:12]],
        "protocol": "Fixed original 80/20 stratified split, random_state=42. Model selected using five-fold CV on 1445 training rows. Compact profile uses a fixed 0.5 threshold. Test labels never fit models.",
        "limitations": [
            "Agent attributes are available only once an assignment is specified; not an arrival-only forecast.",
            "Scores for alternative assignments are observational scenario estimates, not proven causal improvements.",
            "All datasets are synthetic and have been assessed previously; confirm on new real operational data.",
        ],
    }
    archive = root / "models/baseline"
    archive.mkdir(exist_ok=True)
    if AssignmentRiskModel.PATH.exists() and not (archive / "assignment_risk_high_accuracy.joblib").exists():
        shutil.copy2(AssignmentRiskModel.PATH, archive / "assignment_risk_high_accuracy.joblib")
        shutil.copy2(root / "models/assignment_model_metadata.json", archive / "assignment_high_accuracy_metadata.json")
        for filename in ["assignment_accuracy_results.json", "evaluation_v2_results.json", "evaluation_v2_predictions.csv"]:
            source = root / "docs" / filename
            if source.exists():
                archived = source.with_name(source.stem + "_high_accuracy" + source.suffix)
                if not archived.exists():
                    shutil.copy2(source, archived)
    model = AssignmentRiskModel(selected, metadata)
    model.save()
    (root / "models/assignment_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "docs/assignment_accuracy_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selected": winner, "threshold": threshold, "evaluation": metadata["evaluation"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["compact", "full"], default="compact")
    main(profile=parser.parse_args().profile)
