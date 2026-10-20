"""Unit and integration tests for ImpactSimulator (Day 5).

Validates:
- Baseline metrics calculated strictly from 1,807 completed historical tickets (excluding open tickets).
- Exact mathematical formulation of mismatch penalty, breach reduction, recovered hours, and capacity value.
- Slider sensitivity consistency (e.g. 28% target mismatch producing exact traceable outputs).
- Boundary values (0% mismatch, >= 29.8% mismatch, negative/excess inputs).
- Strict immutability of underlying source datasets.
- Deterministic and reproducible outputs.
- Defensible modeling of staffing capacity enablers without ungrounded breach drop claims.
- Baseline-versus-scenario comparison dataframe structure.
"""

from typing import Dict, Any
import pytest
import pandas as pd
import numpy as np

from src.pipeline.loader import DataLoader
from src.simulation.simulator import ImpactSimulator
from src.config import TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE


@pytest.fixture(scope="module")
def data():
    """Load canonical datasets via DataLoader."""
    loader = DataLoader(TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE)
    tickets, agents, merchants = loader.get_datasets()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    return {
        "tickets": tickets,
        "agents": agents,
        "merchants": merchants,
        "hist_tickets": hist_tickets,
        "open_tickets": open_tickets,
    }


@pytest.fixture(scope="module")
def simulator(data):
    """Instantiate ImpactSimulator with canonical datasets."""
    return ImpactSimulator(
        historical_tickets=data["hist_tickets"],
        agents_df=data["agents"],
        merchants_df=data["merchants"],
        hourly_capacity_value=45.0,
    )


class TestImpactSimulatorBaselines:
    """Verify that baseline metrics are calculated strictly from 1,807 completed historical tickets."""

    def test_historical_partition_and_sample_size(self, simulator, data):
        """Historical outcomes must be computed exclusively over N=1,807 tickets, excluding open backlog (N=250)."""
        base = simulator.get_baseline_summary()
        assert base["n_historical"] == 1807
        assert len(data["open_tickets"]) == 250

        # Exact ground-truth historical constants
        assert base["baseline_breach_count"] == 1037
        assert abs(base["baseline_breach_rate"] - (1037 / 1807)) < 1e-4  # ~57.39%
        assert base["baseline_mismatch_count"] == 538
        assert abs(base["baseline_mismatch_rate"] - (538 / 1807)) < 1e-4  # ~29.77%

    def test_mismatch_penalties_accuracy(self, simulator):
        """Verify empirical matched vs mismatched outcomes and penalties."""
        base = simulator.get_baseline_summary()
        assert base["matched_count"] == 1269
        assert base["mismatched_count"] == 538
        assert base["matched_count"] + base["mismatched_count"] == 1807

        # Matched breach rate ~49.96%, mismatched ~74.91%
        assert abs(base["matched_breach_rate"] - (634 / 1269)) < 1e-4
        assert abs(base["mismatched_breach_rate"] - (403 / 538)) < 1e-4

        # Breach penalty ~24.95 pp
        expected_penalty = (403 / 538) - (634 / 1269)
        assert abs(base["breach_penalty_pp"] - (expected_penalty * 100.0)) < 0.05

        # Resolution delay penalty ~12.11 hours
        assert 12.0 <= base["hours_penalty"] <= 12.2

        # Aggregate excess drag ~6,513 hours
        expected_drag = simulator.baseline_mismatch_count * simulator.hours_penalty
        assert abs(base["total_excess_drag_hours"] - round(expected_drag, 1)) < 0.1


class TestSliderSensitivityAndScenarios:
    """Validate scenario calculation consistency across slider values."""

    def test_simulation_at_28_percent_mismatch(self, simulator):
        """A target mismatch rate of 28.0% must produce exact, traceable outputs."""
        sim = simulator.simulate_mismatch_reduction(target_mismatch_rate=0.28)

        # Baseline mismatch is ~29.77%. Delta = ~1.77 pp (0.0177)
        delta_m = simulator.baseline_mismatch_rate - 0.28
        assert abs(sim["delta_mismatch_rate"] - delta_m) < 1e-4

        # Realigned tickets ~32.0
        expected_realigned = delta_m * 1807
        assert abs(sim["realigned_tickets"] - expected_realigned) < 0.2

        # Breaches avoided ~8.0
        expected_breaches_avoided = expected_realigned * simulator.breach_penalty
        assert abs(sim["breaches_avoided"] - expected_breaches_avoided) < 0.2

        # Breach rate reduction ~0.44 pp (from ~57.39% down to ~56.95%)
        expected_drop_pp = delta_m * simulator.breach_penalty * 100.0
        assert abs(sim["breach_rate_reduction_pp"] - expected_drop_pp) < 0.05
        assert abs(sim["simulated_breach_rate"] - (simulator.baseline_breach_rate - expected_drop_pp / 100.0)) < 1e-4
        assert 0.569 <= sim["simulated_breach_rate"] <= 0.570

        # Hours recovered ~387.9 hours
        expected_hours = expected_realigned * simulator.hours_penalty
        assert abs(sim["hours_recovered"] - expected_hours) < 1.0
        assert 385.0 <= sim["hours_recovered"] <= 390.0

        # Capacity value (@ $45/hr) ~ $17,454
        expected_val = expected_hours * 45.0
        assert abs(sim["estimated_capacity_value"] - expected_val) < 20.0
        assert 17300.0 <= sim["estimated_capacity_value"] <= 17600.0

    def test_boundary_zero_mismatch(self, simulator):
        """At 0.0% mismatch, all 538 mismatched tickets are realigned to matched baseline."""
        sim0 = simulator.simulate_mismatch_reduction(target_mismatch_rate=0.0)

        assert sim0["realigned_tickets"] == 538.0
        # Breaches avoided ~134.2
        assert abs(sim0["breaches_avoided"] - 134.2) < 0.5
        # Max breach drop ~7.43 pp
        assert abs(sim0["breach_rate_reduction_pp"] - 7.43) < 0.05
        # Simulated breach rate should equal matched baseline (~49.96%)
        assert abs(sim0["simulated_breach_rate"] - simulator.matched_breach_rate) < 1e-4
        # Hours recovered ~6,513 hours
        assert abs(sim0["hours_recovered"] - simulator.total_excess_drag_hours) < 1.0
        # Capacity value ~ $293,083
        assert abs(sim0["estimated_capacity_value"] - (simulator.total_excess_drag_hours * 45.0)) < 20.0

    def test_boundary_above_baseline_mismatch(self, simulator):
        """Setting target mismatch rate >= baseline produces 0 delta and 0 impact."""
        for target in [simulator.baseline_mismatch_rate, 0.30, 0.50]:
            sim_null = simulator.simulate_mismatch_reduction(target_mismatch_rate=target)
            assert sim_null["delta_mismatch_rate"] == 0.0
            assert sim_null["realigned_tickets"] == 0.0
            assert sim_null["breaches_avoided"] == 0.0
            assert sim_null["breach_rate_reduction_pp"] == 0.0
            assert sim_null["hours_recovered"] == 0.0
            assert sim_null["estimated_capacity_value"] == 0.0

    def test_slider_sensitivity_monotonicity(self, simulator):
        """Decreasing target mismatch rate must strictly monotonically increase recovered hours and breach drop."""
        targets = [0.25, 0.20, 0.15, 0.10, 0.05, 0.0]
        results = [simulator.simulate_mismatch_reduction(t) for t in targets]

        for i in range(len(results) - 1):
            curr = results[i]
            next_res = results[i + 1]
            assert next_res["realigned_tickets"] > curr["realigned_tickets"]
            assert next_res["breaches_avoided"] > curr["breaches_avoided"]
            assert next_res["hours_recovered"] > curr["hours_recovered"]
            assert next_res["estimated_capacity_value"] > curr["estimated_capacity_value"]
            assert next_res["simulated_breach_rate"] < curr["simulated_breach_rate"]


class TestModelIntegrityAndContracts:
    """Validate data immutability, comparison tables, and defensible modeling."""

    def test_source_data_immutability(self, simulator, data):
        """Running simulations must never mutate historical tickets, agents, or merchants."""
        hist_before = data["hist_tickets"].copy(deep=True)
        agents_before = data["agents"].copy(deep=True)

        for _ in range(10):
            simulator.simulate_mismatch_reduction(0.05)
            simulator.simulate_staffing_capacity_enabler({"Payments & Checkout": 2, "API Integrations": 1})

        pd.testing.assert_frame_equal(data["hist_tickets"], hist_before)
        pd.testing.assert_frame_equal(data["agents"], agents_before)

    def test_deterministic_results(self, simulator):
        """Multiple identical calls produce bitwise identical results."""
        r1 = simulator.simulate_mismatch_reduction(0.08)
        r2 = simulator.simulate_mismatch_reduction(0.08)
        assert r1 == r2

    def test_custom_hourly_capacity_value(self, data):
        """Configuring custom loaded labor cost ($60/hr) linearly scales capacity value without altering breach metrics."""
        sim45 = ImpactSimulator(data["hist_tickets"], data["agents"], data["merchants"], hourly_capacity_value=45.0)
        sim60 = ImpactSimulator(data["hist_tickets"], data["agents"], data["merchants"], hourly_capacity_value=60.0)

        res45 = sim45.simulate_mismatch_reduction(0.10)
        res60 = sim60.simulate_mismatch_reduction(0.10)

        assert res45["hours_recovered"] == res60["hours_recovered"]
        assert res45["simulated_breach_rate"] == res60["simulated_breach_rate"]
        assert abs(res60["estimated_capacity_value"] - (res45["estimated_capacity_value"] * (60.0 / 45.0))) < 1.0

    def test_comparison_dataframe_structure(self, simulator):
        """Comparison table contains all 4 required operational metrics and analytical nature labels."""
        sim_res = simulator.simulate_mismatch_reduction(0.05)
        df_comp = simulator.get_comparison_dataframe(sim_res)

        assert len(df_comp) == 4
        expected_metrics = {
            "Category Mismatch Rate",
            "Resolution Breach Rate",
            "Excess Resolution Delay",
            "Operational Capacity Value (@ $45/h)",
        }
        assert set(df_comp["Operational Metric"]) == expected_metrics
        assert "Historical Baseline (Observed)" in df_comp.columns
        assert "Simulated Scenario (Estimate)" in df_comp.columns
        assert "Estimated Variance" in df_comp.columns
        assert "Analytical Nature" in df_comp.columns

    def test_defensible_staffing_capacity_enabler(self, simulator):
        """Staffing additions model domain ticket absorption capacity without claiming unmodeled breach drops."""
        staffing = simulator.simulate_staffing_capacity_enabler({
            "Payments & Checkout": 2,
            "API Integrations": 1,
        })
        assert staffing["total_added_specialists"] == 3
        assert staffing["total_enabled_ticket_capacity"] > 0
        assert "methodology_note" in staffing
        assert "queueing" in staffing["methodology_note"].lower()
