"""Safety and inference contracts for the extended model search."""

import numpy as np
import pytest
from sklearn.base import clone

from src.models.duration_risk import DurationRiskClassifier
from src.models.improvement import IntakeInteractions, pipeline
from src.pipeline.features import FeatureEngineer
from src.pipeline.loader import DataLoader


def test_duration_auxiliary_targets_are_not_required_at_inference():
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    historical = historical.iloc[:180]
    X, y = FeatureEngineer().prepare_training_dataset(historical, loader.load_merchants())
    estimator = pipeline(DurationRiskClassifier(alpha=10))
    estimator.fit(X.iloc[:140], y.iloc[:140], classifier__resolution_hours=historical.resolution_hours.iloc[:140])
    probabilities = estimator.predict_proba(X.iloc[140:])
    assert probabilities.shape == (40, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    # IDs are neither model inputs nor a lookup for completed outcomes.
    changed_ids = X.iloc[140:].copy()
    changed_ids["ticket_id"] = -1
    changed_ids["merchant_id"] = -1
    np.testing.assert_allclose(estimator.predict_proba(changed_ids), probabilities)


def test_duration_model_rejects_missing_or_invalid_training_targets():
    estimator = DurationRiskClassifier()
    with pytest.raises(ValueError, match="required"):
        estimator.fit(np.ones((4, 3)), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="positive"):
        estimator.fit(np.ones((4, 3)), [0, 1, 0, 1], resolution_hours=[1, 2, 0, 4])


def test_search_estimators_can_be_cloned_without_fitted_state():
    # Catches incompatible third-party estimator parameters before a long CV run.
    pytest.importorskip("catboost")
    from src.models.improvement import candidates
    for estimator in candidates().values():
        cloned = clone(estimator)
        classifier = cloned.named_steps["classifier"]
        if hasattr(classifier, "is_fitted"):
            assert not classifier.is_fitted()
        else:
            assert not hasattr(classifier, "classes_")


def test_interactions_do_not_change_the_source_features():
    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    X, _ = FeatureEngineer().prepare_training_dataset(historical.iloc[:10], loader.load_merchants())
    columns_before = list(X.columns)
    result = IntakeInteractions().fit_transform(X)
    assert list(X.columns) == columns_before
    assert len(result) == len(X)
    assert "resolution_hours" not in result


def test_merchant_encoding_cross_fits_and_handles_unknown_accounts():
    import pandas as pd
    from sklearn.preprocessing import TargetEncoder
    from src.models.merchant_context import merchant_candidates

    keys = pd.DataFrame({"merchant_history_key": [str(i) for i in range(20)]})
    labels = np.tile([0, 1], 10)
    encoder = TargetEncoder(smooth=10.0, cv=5, shuffle=True, random_state=42)
    cross_fitted = encoder.fit_transform(keys, labels)
    # Each unique account is unknown to its own inner training fold.
    np.testing.assert_allclose(cross_fitted, 0.5)
    assert not np.allclose(encoder.transform(keys), cross_fitted)

    loader = DataLoader()
    historical, _ = loader.split_tickets(loader.load_tickets())
    X, y = FeatureEngineer().prepare_training_dataset(historical.iloc[:120], loader.load_merchants())
    estimator = next(iter(merchant_candidates().values())).fit(X.iloc[:100], y.iloc[:100])
    unknown = X.iloc[100:].copy()
    unknown["merchant_id"] = -1
    probabilities = estimator.predict_proba(unknown.drop(columns="merchant_id"))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities, estimator.predict_proba(unknown))
