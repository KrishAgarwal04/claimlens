"""Assignment-stage boundary, constrained planning and forecast backtests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.analytics.workload import WorkloadForecaster
from src.models.assignment_risk import AssignmentRiskModel, assignment_features
from src.models.sla_model import SLARiskModel
from src.pipeline.features import DataLeakageError, FeatureEngineer
from src.pipeline.loader import DataLoader
from src.routing.decision_engine import DecisionEngine


@pytest.fixture(scope="module")
def data():
    loader = DataLoader()
    tickets = loader.load_tickets()
    historical, queue = loader.split_tickets(tickets)
    agents, merchants = loader.load_agents(), loader.load_merchants()
    model = AssignmentRiskModel.load()
    intake = SLARiskModel.load().predict_risk(FeatureEngineer().prepare_open_queue_dataset(queue, merchants))
    return tickets, historical, queue, agents, merchants, model, intake


def test_assignment_features_ignore_observed_outcomes_and_mismatch(data):
    _, historical, _, agents, merchants, model, _ = data
    raw = historical.iloc[:20].copy()
    original = assignment_features(raw, merchants, agents)
    raw["resolution_hours"] = 100000
    raw["resolution_breached"] = raw.resolution_breached.map(lambda value: not value)
    raw["category_mismatch"] = raw.category_mismatch.map(lambda value: not value)
    raw["first_response_at"] = pd.Timestamp("2099-01-01")
    raw["is_reopened"] = raw.is_reopened.map(lambda value: not value)
    pd.testing.assert_frame_equal(original, assignment_features(raw, merchants, agents))
    leaked = original.assign(resolution_hours=12)
    with pytest.raises(DataLeakageError):
        model.predict_risk(leaked)


def test_proposed_assignment_overrides_historical_agent_and_accepts_open_tickets(data):
    _, _, queue, agents, merchants, model, _ = data
    raw = queue.iloc[:5].copy()
    proposed_ids = [int(agents.agent_id.iloc[0])]*5
    original = assignment_features(raw, merchants, agents, proposed_ids)
    raw["assigned_agent_id"] = -999
    pd.testing.assert_frame_equal(original, assignment_features(raw, merchants, agents, proposed_ids))
    risk = model.predict_risk(original)
    assert len(risk) == 5 and np.isfinite(risk).all()
    assert ((risk >= 0) & (risk <= 1)).all()
    with pytest.raises(ValueError, match="Unknown"):
        assignment_features(raw, merchants, agents)


def test_saved_assignment_holdout_metrics_reproduce(data, tmp_path):
    _, historical, _, agents, merchants, model, _ = data
    y = historical.resolution_breached.astype(int).reset_index(drop=True)
    _, test_ids = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    features = assignment_features(historical, merchants, agents).iloc[test_ids]
    predicted = model.predict(features)
    assert accuracy_score(y.iloc[test_ids], predicted) == pytest.approx(model.metadata["evaluation"]["historical_holdout"]["accuracy"])
    assert accuracy_score(y.iloc[test_ids], predicted) > .85
    path = tmp_path / "assignment.joblib"
    model.save(path)
    np.testing.assert_allclose(AssignmentRiskModel.load(path).predict_risk(features), model.predict_risk(features))


def test_plan_respects_capacity_specialty_seniority_and_unique_tickets(data):
    _, _, queue, agents, merchants, model, intake = data
    capacities = {int(agent_id): 15 for agent_id in agents.agent_id}
    plan, summary = DecisionEngine(model, agents, merchants).plan(queue, intake, capacities)
    assigned = plan[plan.status == "Planned"]
    assert len(plan) == len(queue) and plan.ticket_id.is_unique
    counts = assigned.agent_id.value_counts()
    assert all(counts.get(agent_id, 0) <= capacity for agent_id, capacity in capacities.items())
    joined = assigned.merge(agents, left_on="agent_id", right_on="agent_id")
    assert (joined.category == joined.primary_category).all()
    assert joined.loc[joined.priority.isin(["P1", "P2"]), "tier"].isin(["L2", "L3"]).all()
    assert summary["planned"] + summary["waiting"] == len(queue)
    # This dataset's default scenario improves the stated weighted objective.
    assert summary["planned_weighted_cost"] <= summary["baseline_weighted_cost"]
    assert assigned.evidence.str.contains("estimated breach risk").all()


def test_zero_capacity_and_invalid_capacity(data):
    _, _, queue, agents, merchants, model, intake = data
    engine = DecisionEngine(model, agents, merchants)
    plan, summary = engine.plan(queue.iloc[:12], intake[:12], {})
    assert summary["planned"] == 0 and summary["waiting"] == 12
    assert plan.agent_id.isna().all()
    assert plan.assignment_risk_pct.isna().all()
    with pytest.raises(ValueError, match="whole numbers"):
        engine.plan(queue.iloc[:12], intake[:12], {int(agents.agent_id.iloc[0]): 1.5})


def test_known_overdue_tickets_are_distinguished_from_model_forecasts(data):
    _, _, queue, agents, merchants, model, _ = data
    sample = queue.iloc[:2].copy()
    sample["priority"] = "P1"
    sample["created_at"] = pd.to_datetime(["2026-06-30 08:00:00", "2026-06-30 12:00:00"])
    plan, summary = DecisionEngine(model, agents, merchants).plan(sample, [.2, .3], {int(i): 5 for i in agents.agent_id})
    assert summary["already_overdue"] == 1
    assert plan.iloc[0].sla_state == "Already overdue"
    assert plan.iloc[0].assignment_risk_pct == 100
    assert plan.iloc[0].intake_risk_pct == 20
    assert "cannot undo" in plan.iloc[0].evidence


def test_forecast_has_no_missing_calendar_days_and_keeps_backtest_separate(data):
    tickets = data[0]
    results = WorkloadForecaster(tickets).analyze(7)
    assert len(results["forecast"]) == 7
    assert results["forecast"].date.min() > results["origin"]
    assert len(results["backtest"]) == 28
    assert (results["forecast"].upper_80 >= results["forecast"].lower_80).all()
    assert results["backtest_mae"] >= 0
    assert results["daily"].date.diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_future_counts_cannot_change_prior_surge_evidence():
    dates = pd.date_range("2025-01-01", periods=120)
    original = pd.DataFrame({"created_at": dates})
    changed = pd.concat([original, pd.DataFrame({"created_at": [dates[80]]*40 + [dates[119]]*100})])
    first = WorkloadForecaster(changed).analyze()["anomalies"]
    changed_again = pd.concat([changed, pd.DataFrame({"created_at": [dates[119]]*500})])
    second = WorkloadForecaster(changed_again).analyze()["anomalies"]
    pd.testing.assert_frame_equal(first[first.date == dates[80]].reset_index(drop=True), second[second.date == dates[80]].reset_index(drop=True))
    assert len(first[first.date == dates[80]]) == 1
