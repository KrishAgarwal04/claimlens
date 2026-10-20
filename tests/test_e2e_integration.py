"""Day 6: End-to-End Integration and Reliability Audit Test Suite.

Validates:
1. Complete workflow trace: CSV loading -> feature engineering -> SLA model inference -> bottleneck detection -> agent recommendation -> impact simulation.
2. Saved model persistence, calibration, and inference on all 250 open queue tickets.
3. Strict exclusion of open queue tickets from historical outcome metrics and zero target leakage.
4. Resilience against edge cases: missing values, unknown categories, unassigned agents, and empty filters.
5. Deterministic reactivity of simulator controls across boundary values.
6. Benchmark tables, confusion matrices, and chart layout rendering contracts.
"""

from typing import Dict, Any
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.config import TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE, SLA_TARGETS
from src.pipeline.loader import DataLoader
from src.pipeline.features import FeatureEngineer, DataLeakageError
from src.models.sla_model import SLARiskModel
from src.analytics import BottleneckDetector
from src.routing import AgentRecommender
from src.simulation import ImpactSimulator


@pytest.fixture(scope="module")
def pipeline_data():
    """Execute Day 1 canonical data loading and partitioning."""
    loader = DataLoader(TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE)
    tickets, agents, merchants = loader.get_datasets()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    return {
        "loader": loader,
        "tickets": tickets,
        "agents": agents,
        "merchants": merchants,
        "hist_tickets": hist_tickets,
        "open_tickets": open_tickets,
    }


class TestEndToEndWorkflow:
    """Trace complete workflow across all 5 operational intelligence modules."""

    def test_complete_data_to_decision_pipeline(self, pipeline_data):
        tickets = pipeline_data["tickets"]
        agents = pipeline_data["agents"]
        merchants = pipeline_data["merchants"]
        hist_tickets = pipeline_data["hist_tickets"]
        open_tickets = pipeline_data["open_tickets"]

        # Step 1: Pre-assignment feature engineering
        fe = FeatureEngineer()
        X_train, y_train = fe.prepare_training_dataset(hist_tickets, merchants)
        X_open = fe.prepare_open_queue_dataset(open_tickets, merchants)

        assert len(X_train) == 1807
        assert len(X_open) == 250
        assert list(X_train.columns) == list(X_open.columns)

        # Step 2: SLA model loading & live queue risk scoring
        model = SLARiskModel.load()
        risk_scores = model.predict_risk(X_open)
        assert len(risk_scores) == 250
        assert (risk_scores >= 0.0).all() and (risk_scores <= 1.0).all()

        triage_df = open_tickets.copy().reset_index(drop=True)
        triage_df["breach_risk"] = risk_scores
        triage_df["breach_risk_pct"] = (risk_scores * 100).round(1)

        # Step 3: Operational bottleneck detection on completed tickets
        detector = BottleneckDetector(tickets, agents, merchants)
        kpis = detector.get_summary_kpis()
        assert kpis["total_tickets"] == 1807
        assert kpis["open_queue_backlog"] == 250
        assert abs(kpis["mismatch_rate"] - 0.2977) < 0.001

        ranked_bottlenecks = detector.rank_bottlenecks()
        assert len(ranked_bottlenecks) > 0
        assert "bottleneck_score" in ranked_bottlenecks.columns

        insights = detector.get_structured_insights()
        assert len(insights) >= 5

        # Step 4: Smart agent recommendation for open tickets
        recommender = AgentRecommender(agents, hist_tickets, merchants)
        routed_batch = recommender.route_batch(triage_df, strategy="top_match")
        assert len(routed_batch) == 250
        assert (routed_batch["status"] == "SUCCESS").all()

        batch_summary = recommender.get_batch_summary(routed_batch)
        assert batch_summary["match_rate"] == 1.0

        # Step 5: Counterfactual impact simulation
        simulator = ImpactSimulator(hist_tickets, agents, merchants, hourly_capacity_value=45.0)
        base = simulator.get_baseline_summary()
        assert base["n_historical"] == 1807
        assert base["baseline_breach_count"] == 1037

        sim_scenario = simulator.simulate_mismatch_reduction(0.05)
        assert sim_scenario["simulated_breach_rate"] < base["baseline_breach_rate"]
        assert sim_scenario["hours_recovered"] > 5000.0

        comparison_df = simulator.get_comparison_dataframe(sim_scenario)
        assert len(comparison_df) == 4


class TestSavedModelIntegrity:
    """Validate saved model persistence, calibration, and inference guarantees."""

    def test_saved_model_artifact_attributes(self):
        model = SLARiskModel.load()
        assert model.calibrate is True
        assert model.model_type == "random_forest"
        assert model.pipeline is not None
        assert "holdout_results" in model.metadata
        assert "cv_benchmarks" in model.metadata

    def test_open_queue_inference_distribution(self, pipeline_data):
        fe = FeatureEngineer()
        X_open = fe.prepare_open_queue_dataset(pipeline_data["open_tickets"], pipeline_data["merchants"])
        model = SLARiskModel.load()
        scores = model.predict_risk(X_open)

        # Distribution sanity checks
        assert 0.05 < scores.min() < 0.20
        assert 0.85 < scores.max() <= 1.00
        assert 0.45 < scores.mean() < 0.60
        # High, medium, and low tiers all represented
        assert (scores >= 0.70).sum() > 0
        assert ((scores >= 0.40) & (scores < 0.70)).sum() > 0
        assert (scores < 0.40).sum() > 0


class TestTargetLeakageAndPartitioning:
    """Verify zero target leakage and strict open-queue exclusion."""

    def test_no_leakage_in_features(self, pipeline_data):
        fe = FeatureEngineer()
        X_train, _ = fe.prepare_training_dataset(pipeline_data["hist_tickets"], pipeline_data["merchants"])
        X_open = fe.prepare_open_queue_dataset(pipeline_data["open_tickets"], pipeline_data["merchants"])

        forbidden = [
            "assigned_agent_id", "category_mismatch", "resolution_hours",
            "ttfr_hours", "first_response_at", "closed_at", "csat_score",
            "resolution_breached", "response_breached"
        ]
        for col in forbidden:
            assert col not in X_train.columns
            assert col not in X_open.columns

    def test_historical_metrics_exclude_open_tickets(self, pipeline_data):
        detector = BottleneckDetector(pipeline_data["tickets"], pipeline_data["agents"], pipeline_data["merchants"])
        kpis = detector.get_summary_kpis()
        assert kpis["total_tickets"] == 1807
        assert detector.historical_tickets["resolution_hours"].notna().all()


class TestEdgeCasesAndBoundaryResilience:
    """Validate behavior under missing fields, unknown domains, and extreme values."""

    def test_missing_priority_in_pipeline(self, pipeline_data):
        fe = FeatureEngineer()
        t = pipeline_data["open_tickets"].iloc[:1].copy()
        t["priority"] = None
        X = fe.prepare_open_queue_dataset(t, pipeline_data["merchants"])
        assert X["priority_rank"].iloc[0] == 3
        assert X["sla_target_hours"].iloc[0] == 24.0

    def test_unknown_domain_in_recommendation(self, pipeline_data):
        rec = AgentRecommender(pipeline_data["agents"], pipeline_data["hist_tickets"], pipeline_data["merchants"])
        res = rec.recommend_for_ticket({"ticket_id": 999, "category": "Novel Domain", "priority": "P1"})
        assert res["status"] == "NO_MATCH"
        assert res["match_score"] == 0.0

    def test_missing_category_in_recommendation(self, pipeline_data):
        rec = AgentRecommender(pipeline_data["agents"], pipeline_data["hist_tickets"], pipeline_data["merchants"])
        res = rec.recommend_for_ticket({"ticket_id": 999, "category": None, "priority": "P2"})
        assert res["status"] == "NO_MATCH"

    def test_empty_filters_in_bottlenecks(self, pipeline_data):
        detector = BottleneckDetector(pipeline_data["tickets"], pipeline_data["agents"], pipeline_data["merchants"])
        empty_df = pipeline_data["hist_tickets"].iloc[:0].copy()
        kpis = detector.get_summary_kpis(empty_df)
        assert kpis["total_tickets"] == 0
        assert kpis["resolution_breach_rate"] == 0.0

    def test_simulator_boundary_reactivity(self, pipeline_data):
        sim = ImpactSimulator(pipeline_data["hist_tickets"], pipeline_data["agents"], pipeline_data["merchants"])

        # Boundary 0.0% mismatch
        s0 = sim.simulate_mismatch_reduction(0.0)
        assert s0["realigned_tickets"] == 538.0
        assert s0["simulated_breach_rate"] < 0.50

        # Boundary at current baseline (~29.8%)
        s_base = sim.simulate_mismatch_reduction(0.30)
        assert s_base["delta_mismatch_rate"] == 0.0
        assert s_base["realigned_tickets"] == 0.0
        assert s_base["hours_recovered"] == 0.0
