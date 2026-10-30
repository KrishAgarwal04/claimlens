"""SLA breach risk prediction model training, calibration, and evaluation.

Predicts the probability that an incoming ticket will breach its resolution SLA,
using STRICTLY pre-assignment features available at ticket creation time.
Excludes all outcome fields, response metrics, and assigned agent information.
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    PRE_ASSIGNMENT_FEATURE_COLUMNS,
    FORBIDDEN_PRE_ASSIGNMENT_COLUMNS,
)
from src.pipeline.features import FeatureEngineer, DataLeakageError


class SLARiskModel:
    """Predicts resolution breach risk using pre-assignment intake features."""

    CATEGORICAL_FEATURES = [
        "category",
        "sub_category",
        "priority",
        "merchant_tier",
        "merchant_sector",
        "merchant_region",
    ]

    NUMERIC_FEATURES = [
        "priority_rank",
        "sla_target_hours",
        "is_reopen_child",
        "is_incident_ticket",
        "intake_hour",
        "intake_day_of_week",
        "intake_is_weekend",
        "intake_month",
        "intake_hour_sin",
        "intake_hour_cos",
        "intake_dow_sin",
        "intake_dow_cos",
    ]

    EXCLUDED_IDENTIFIERS = ["ticket_id", "merchant_id"]

    DEFAULT_MODEL_PATH = MODELS_DIR / "sla_risk_model.joblib"
    DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"

    def __init__(
        self,
        model_type: str = "random_forest",
        calibrate: bool = True,
        random_state: int = RANDOM_STATE,
    ):
        self.model_type = model_type
        self.calibrate = calibrate
        self.random_state = random_state
        self.pipeline: Optional[BaseEstimator] = None
        self.feature_names: List[str] = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        self.metadata: Dict[str, Any] = {}

    def _build_preprocessor(self, scale_numeric: bool = False) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer for numerical and categorical features."""
        num_transformer = StandardScaler() if scale_numeric else "passthrough"
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        return ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.NUMERIC_FEATURES),
                ("cat", cat_transformer, self.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

    def build_pipeline(self, model_type: Optional[str] = None) -> Pipeline:
        """Construct leak-free pipeline with preprocessing and classifier."""
        m_type = model_type or self.model_type

        if m_type == "majority_baseline":
            pre = self._build_preprocessor(scale_numeric=False)
            clf = DummyClassifier(strategy="most_frequent")
        elif m_type == "logistic_regression":
            pre = self._build_preprocessor(scale_numeric=True)
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
        elif m_type == "random_forest":
            pre = self._build_preprocessor(scale_numeric=False)
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {m_type}. Must be 'majority_baseline', 'logistic_regression', or 'random_forest'.")

        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    def _validate_inputs(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Enforce strict zero-leakage and schema contracts on feature matrix."""
        # 1. Open tickets check (if raw tickets DataFrame with is_legacy flag passed)
        if "is_legacy" in X.columns:
            if not (X["is_legacy"] == True).all():
                open_cnt = (X["is_legacy"] == False).sum()
                raise ValueError(
                    f"CRITICAL: Found {open_cnt} open queue tickets. "
                    "Open tickets have placeholder breach outcomes and must NEVER be used for training or validation."
                )

        # 2. Leakage check
        fe = FeatureEngineer()
        fe.validate_no_leakage(X)

        # 3. Check required features are present
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows.")
            if y.isnull().any():
                raise ValueError("Target y contains null values.")

    def benchmark_candidate_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Run stratified k-fold cross-validation across baseline, logistic regression, and random forest."""
        self._validate_inputs(X_train, y_train)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        candidates = ["majority_baseline", "logistic_regression", "random_forest"]
        cv_results = {}

        for m_name in candidates:
            pipe = self.build_pipeline(m_name)
            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
                return_train_score=False,
            )
            cv_results[m_name] = {
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "cv_pr_auc_mean": float(np.mean(scores["test_average_precision"])),
                "cv_pr_auc_std": float(np.std(scores["test_average_precision"])),
                "cv_f1_mean": float(np.mean(scores["test_f1"])),
                "cv_f1_std": float(np.std(scores["test_f1"])),
                "cv_precision_mean": float(np.mean(scores["test_precision"])),
                "cv_recall_mean": float(np.mean(scores["test_recall"])),
            }

        return cv_results

    def fit_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Perform end-to-end stratified train/test split, model comparison, fitting, calibration, and evaluation.
        
        Args:
            X: Intake pre-assignment feature matrix (1,807 historical rows).
            y: Binary target resolution_breached.
            test_size: Holdout evaluation proportion (default: 0.20).
            cv_folds: Number of stratified cross-validation folds on training set.
            
        Returns:
            Dict[str, Any]: Full evaluation metrics, benchmark comparison, and model metadata.
        """
        self._validate_inputs(X, y)

        # 1. Stratified Split (80% Train, 20% Holdout Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # 2. Run Stratified Cross-Validation on Training Split
        cv_benchmarks = self.benchmark_candidate_models(X_train, y_train, cv_folds=cv_folds)

        # 3. Fit Candidate Models on Training Split and Evaluate on Holdout Test Set
        holdout_results = {}
        fitted_models = {}

        for m_name in ["majority_baseline", "logistic_regression", "random_forest"]:
            pipe = self.build_pipeline(m_name)
            pipe.fit(X_train, y_train)
            fitted_models[m_name] = pipe

            y_pred = pipe.predict(X_test)
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
                roc = float(roc_auc_score(y_test, y_proba))
                pr_auc = float(average_precision_score(y_test, y_proba))
                brier = float(brier_score_loss(y_test, y_proba))
            else:
                y_proba = y_pred.astype(float)
                roc = 0.5
                pr_auc = float(np.mean(y_test))
                brier = float(brier_score_loss(y_test, y_proba))

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]

            holdout_results[m_name] = {
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "brier_score": brier,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            }

        # 4. Fit Calibrated Model if Selected
        selected_pipe = fitted_models[self.model_type]
        if self.calibrate and self.model_type != "majority_baseline":
            # Calibrate using internal 5-fold CV strictly on the training set
            calibrated_pipe = CalibratedClassifierCV(
                estimator=self.build_pipeline(self.model_type),
                cv=5,
                method="sigmoid",
            )
            calibrated_pipe.fit(X_train, y_train)
            self.pipeline = calibrated_pipe

            cal_proba = self.pipeline.predict_proba(X_test)[:, 1]
            cal_pred = (cal_proba >= 0.5).astype(int)
            cal_cm = confusion_matrix(y_test, cal_pred)
            c_tn, c_fp, c_fn, c_tp = [int(v) for v in cal_cm.ravel()]

            holdout_results["calibrated_selected_model"] = {
                "model_name": f"Calibrated {self.model_type.replace('_', ' ').title()} (Sigmoid cv=5)",
                "roc_auc": float(roc_auc_score(y_test, cal_proba)),
                "pr_auc": float(average_precision_score(y_test, cal_proba)),
                "f1": float(f1_score(y_test, cal_pred, zero_division=0)),
                "precision": float(precision_score(y_test, cal_pred, zero_division=0)),
                "recall": float(recall_score(y_test, cal_pred, zero_division=0)),
                "brier_score": float(brier_score_loss(y_test, cal_proba)),
                "confusion_matrix": {"tn": c_tn, "fp": c_fp, "fn": c_fn, "tp": c_tp},
            }
        else:
            self.pipeline = selected_pipe

        # 5. Extract Feature Importances from Base Random Forest (for interpretability)
        rf_pipe = fitted_models["random_forest"]
        feature_names_out = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = rf_pipe.named_steps["classifier"].feature_importances_
        top_indices = np.argsort(importances)[::-1][:15]
        top_features = [
            {"feature": feature_names_out[i], "importance": float(importances[i])}
            for i in top_indices
        ]

        # 6. Assemble Full Metadata
        self.metadata = {
            "target": "resolution_breached (1 = SLA breached, 0 = resolved within target)",
            "selected_model": self.model_type,
            "is_calibrated": self.calibrate,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": self.random_state,
            "feature_columns_in": self.feature_names,
            "cv_benchmarks": cv_benchmarks,
            "holdout_results": holdout_results,
            "top_feature_importances": top_features,
            "class_distribution": {
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            },
            "limitations": [
                "Intake-time only features: cannot observe agent skill or queue congestion at prediction time.",
                "Synthetic ITSM distribution: priority and category have deterministic SLA thresholds shaping the breach boundary.",
                "Stratified split preserves class balance across priorities, but does not simulate non-stationary regime shifts over time.",
            ],
        }

        return self.metadata

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SLARiskModel":
        """Fit model directly on provided dataset."""
        self._validate_inputs(X, y)
        if self.calibrate and self.model_type != "majority_baseline":
            self.pipeline = CalibratedClassifierCV(
                estimator=self.build_pipeline(self.model_type),
                cv=5,
                method="sigmoid",
            )
        else:
            self.pipeline = self.build_pipeline(self.model_type)

        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(0), P(1)] for incoming feature matrix."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() or fit_and_evaluate() first.")
        self._validate_inputs(X)
        return self.pipeline.predict_proba(X)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict breach probability P(resolution_breached = 1) in [0.0, 1.0]."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    @property
    def decision_threshold(self) -> float:
        """Training-selected threshold; legacy artifacts use 0.5."""
        return float(self.metadata.get("decision_threshold", 0.5))

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Use the saved operating threshold unless an explicit override is given."""
        threshold = self.decision_threshold if threshold is None else threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Classification threshold must be between 0 and 1")
        risk = self.predict_risk(X)
        return (risk >= threshold).astype(int)

    def save(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Persist fitted model pipeline and metadata to disk."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted model.")

        m_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        meta_path = Path(metadata_path or self.DEFAULT_METADATA_PATH)

        m_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, m_path)
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> "SLARiskModel":
        """Load fitted pipeline and metadata from disk."""
        m_path = Path(model_path or cls.DEFAULT_MODEL_PATH)
        meta_path = Path(metadata_path or cls.DEFAULT_METADATA_PATH)

        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found at {m_path}")

        instance = cls()
        instance.pipeline = joblib.load(m_path)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                instance.metadata = json.load(f)
                instance.model_type = instance.metadata.get("selected_model", "random_forest")
                instance.calibrate = instance.metadata.get("is_calibrated", True)

        return instance
