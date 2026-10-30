"""Estimate breach probability from a supervised log-duration distribution.

Completed resolution times are auxiliary training targets, never input features.
At inference, only the usual pre-assignment feature matrix is needed.
"""

import numpy as np
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


class DurationRiskClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=10.0, sla_column=1):
        self.alpha = alpha
        self.sla_column = sla_column

    def fit(self, X, y, resolution_hours=None):
        if resolution_hours is None:
            raise ValueError("Completed durations are required as auxiliary training targets")
        durations = np.asarray(resolution_hours, dtype=float)
        if len(durations) != len(y) or not np.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError("Training durations must be positive, finite and aligned with y")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        log_duration = np.log(durations)
        self.regressor_ = Ridge(alpha=self.alpha).fit(X, log_duration)
        residuals = log_duration - self.regressor_.predict(X)
        self.scale_ = max(float(np.std(residuals, ddof=1)), 1e-6)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "regressor_")
        sla_hours = np.asarray(X[:, self.sla_column], dtype=float)
        if (sla_hours <= 0).any():
            raise ValueError("SLA targets must be positive")
        mean_log_duration = self.regressor_.predict(X)
        probability = ndtr((mean_log_duration - np.log(sla_hours)) / self.scale_)
        return np.column_stack([1-probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
