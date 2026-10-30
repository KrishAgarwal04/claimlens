"""Training-only model selection and intake feature interactions.

The original 20% holdout and external datasets are excluded from selection.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.models.sla_model import SLARiskModel
from src.models.duration_risk import DurationRiskClassifier
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


class IntakeInteractions(TransformerMixin, BaseEstimator):
    """Represent SLA/context interactions using only the existing intake fields."""

    PAIRS = [
        ("category", "priority"), ("sub_category", "priority"),
        ("category", "merchant_tier"), ("category", "is_incident_ticket"),
        ("category", "is_reopen_child"), ("priority", "is_incident_ticket"),
    ]
    EXTRA_COLUMNS = [f"{left}__{right}" for left, right in PAIRS]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        for left, right in self.PAIRS:
            out[f"{left}__{right}"] = out[left].astype(str) + "|" + out[right].astype(str)
        return out


def pipeline(classifier, interactions=False, scale=False):
    categories = SLARiskModel.CATEGORICAL_FEATURES + (
        IntakeInteractions.EXTRA_COLUMNS if interactions else []
    )
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", SLARiskModel.NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
    ])
    steps = [("interactions", IntakeInteractions())] if interactions else []
    return Pipeline(steps + [("preprocessor", preprocessor), ("classifier", classifier)])


def candidates():
    models = {}
    for interactions in [False, True]:
        for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
            name = f"logistic_{'interactions' if interactions else 'plain'}_C{c}"
            models[name] = pipeline(LogisticRegression(C=c, max_iter=2000), interactions, True)
    for leaves in [4, 8, 15]:
        for regularization in [5.0, 20.0]:
            models[f"hist_leaves{leaves}_l2{regularization}"] = pipeline(
                HistGradientBoostingClassifier(
                    max_iter=180, learning_rate=0.04, max_leaf_nodes=leaves,
                    min_samples_leaf=30, l2_regularization=regularization,
                    early_stopping=False, random_state=42,
                )
            )
    for depth in [1, 2, 3]:
        for trees in [100, 250]:
            models[f"gradient_depth{depth}_trees{trees}"] = pipeline(
                GradientBoostingClassifier(
                    n_estimators=trees, max_depth=depth, learning_rate=0.04,
                    min_samples_leaf=25, random_state=42,
                )
            )
    for depth in [6, 12]:
        models[f"forest_depth{depth}"] = pipeline(RandomForestClassifier(
            n_estimators=250, max_depth=depth, min_samples_leaf=5,
            max_features=0.8, random_state=42, n_jobs=2,
        ))
    # Compare compact intake models to avoid learning noise from calendar fields.
    for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        core = ColumnTransformer([
            ("numeric", "passthrough", ["is_incident_ticket", "is_reopen_child", "intake_is_weekend"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["category", "sub_category", "priority", "merchant_tier", "merchant_region"]),
        ])
        models[f"logistic_core_C{c}"] = Pipeline([
            ("preprocessor", core), ("classifier", LogisticRegression(C=c, max_iter=2000)),
        ])
    from catboost import CatBoostClassifier

    for depth in [2, 4, 6]:
        for regularization in [3.0, 15.0]:
            models[f"catboost_depth{depth}_l2{regularization}"] = Pipeline([
                ("features", ColumnTransformer([
                    ("categorical", "passthrough", SLARiskModel.CATEGORICAL_FEATURES),
                    ("numeric", "passthrough", SLARiskModel.NUMERIC_FEATURES),
                ], verbose_feature_names_out=False).set_output(transform="pandas")),
                ("classifier", CatBoostClassifier(
                    iterations=500, depth=depth, learning_rate=0.03,
                    l2_leaf_reg=regularization, cat_features=tuple(SLARiskModel.CATEGORICAL_FEATURES),
                    random_seed=42, thread_count=2, verbose=False, allow_writing_files=False,
                )),
            ])
    for interactions in [False, True]:
        for alpha in [1.0, 10.0, 100.0]:
            models[f"duration_interactions{interactions}_alpha{alpha}"] = pipeline(
                DurationRiskClassifier(alpha=alpha), interactions=interactions,
            )
    return models


def best_threshold(y, probabilities):
    # Coarse grid limits adaptation to small-sample noise. Prefer 0.5 for ties.
    thresholds = np.round(np.arange(0.35, 0.651, 0.025), 3)
    return float(max(thresholds, key=lambda t: (
        accuracy_score(y, probabilities >= t), -abs(t-0.5)
    )))


def select_candidate():
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    X, y = FeatureEngineer().prepare_training_dataset(historical, loader.load_merchants())
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    durations = historical.resolution_hours.iloc[X_train.index]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = candidates()
    baseline_dir = Path(__file__).resolve().parents[2] / "models/baseline"
    original = SLARiskModel.load(
        baseline_dir / "sla_risk_model.joblib", baseline_dir / "model_metadata.json"
    ) if (baseline_dir / "sla_risk_model.joblib").exists() else SLARiskModel.load()
    models["original_calibrated_forest"] = clone(original.pipeline)
    probabilities = {}
    results = {}
    destination = Path(__file__).resolve().parents[2] / ".model_experiments"
    destination.mkdir(exist_ok=True)
    cache_path = destination / "cv_predictions.npz"
    signature_path = destination / "cache_signature.json"
    signature = {
        "training_data": joblib.hash((X_train, y_train)),
        "models": {name: joblib.hash(estimator) for name, estimator in models.items()},
        "folds": 5, "random_state": 42,
    }
    if cache_path.exists() and signature_path.exists() and json.loads(signature_path.read_text()) == signature:
        with np.load(cache_path) as cache:
            probabilities.update({name: cache[name] for name in cache.files})
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    def record(name, values):
        threshold = best_threshold(y_train, values)
        results[name] = {
            "cv_accuracy": float(accuracy_score(y_train, values >= threshold)),
            "cv_accuracy_at_0_5": float(accuracy_score(y_train, values >= 0.5)),
            "cv_recall": float(recall_score(y_train, values >= threshold)),
            "cv_roc_auc": float(roc_auc_score(y_train, values)),
            "cv_brier_score": float(brier_score_loss(y_train, values)),
            "decision_threshold": threshold,
        }
        print(f"{name}: {results[name]}", flush=True)

    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            values = probabilities.get(name)
            if values is None:
                if name.startswith("duration_"):
                    values = np.empty(len(y_train))
                    for train_indices, validation_indices in cv.split(X_train, y_train):
                        fold_model = clone(estimator).fit(
                            X_train.iloc[train_indices], y_train.iloc[train_indices],
                            classifier__resolution_hours=durations.iloc[train_indices],
                        )
                        values[validation_indices] = fold_model.predict_proba(X_train.iloc[validation_indices])[:, 1]
                else:
                    values = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            probabilities[name] = values
            np.savez(cache_path, **probabilities)
            record(name, values)
        ranked = sorted(results, key=lambda name: (
            results[name]["cv_accuracy"], -results[name]["cv_brier_score"]
        ), reverse=True)
        members = [name for name in ranked if not name.startswith("duration_")][:3]
        ensemble_name = "soft_voting_top3"
        models[ensemble_name] = VotingClassifier(
            [(f"m{i}", models[name]) for i, name in enumerate(members)], voting="soft"
        )
        record(ensemble_name, np.mean([probabilities[name] for name in members], axis=0))
        results[ensemble_name]["members"] = members
        winner = max(results, key=lambda name: (
            results[name]["cv_accuracy"], -results[name]["cv_brier_score"]
        ))
        print(f"Selected using training CV only: {winner}", flush=True)
        fit_params = {"classifier__resolution_hours": durations} if winner.startswith("duration_") else {}
        selected = models[winner].fit(X_train, y_train, **fit_params)
    return selected, winner, results, X_train, y_train


def main():
    estimator, name, results, training_features, training_labels = select_candidate()
    destination = Path(__file__).resolve().parents[2] / ".model_experiments"
    destination.mkdir(exist_ok=True)
    joblib.dump(estimator, destination / "selected.joblib")
    (destination / "selection.json").write_text(json.dumps({
        "selected": name, "cv_results": results,
        "training_data_hash": joblib.hash((training_features, training_labels)),
        "selection_data": "1445 historical training rows only; fixed 5-fold stratified CV",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Use the importable module name for transformers persisted by joblib.
    from src.models.improvement import main as run
    run()
