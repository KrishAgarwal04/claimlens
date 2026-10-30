"""Optional, cross-fitted merchant history context available before assignment."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.models.sla_model import SLARiskModel


class MerchantHistoryKey(TransformerMixin, BaseEstimator):
    """Unknown accounts fall back to the training prior rather than a made-up ID."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["merchant_history_key"] = (
            out["merchant_id"].fillna(-1).astype(str) if "merchant_id" in out else "unknown"
        )
        return out


def merchant_candidates():
    models = {}
    for smoothing in [10.0, 50.0, 150.0]:
        for c in [0.1, 1.0]:
            preprocessor = ColumnTransformer([
                ("numeric", StandardScaler(), SLARiskModel.NUMERIC_FEATURES),
                ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SLARiskModel.CATEGORICAL_FEATURES),
                # fit_transform cross-fits this target encoding inside each outer CV fold.
                ("history", TargetEncoder(smooth=smoothing, cv=5, shuffle=True, random_state=42), ["merchant_history_key"]),
            ])
            name = f"merchant_logistic_C{c}_smooth{smoothing}"
            models[name] = Pipeline([
                ("merchant_context", MerchantHistoryKey()),
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(C=c, max_iter=2000)),
            ])
    return models
