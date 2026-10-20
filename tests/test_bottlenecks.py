"""Unit and integration tests for Operational Bottleneck Detection & Business Insights (Day 3).

Validates:
- Strict exclusion of open queue tickets from historical outcome rates.
- Accuracy of metric calculations (resolution breach, response breach, reopen rate, mismatch penalty).
- Configurable sample-size-penalized ranking formula stability.
- Structured insights schema contract (title, metrics, sample size, comparison, action, caveat).
- Edge cases: zero denominators, empty dataframes, missing values, unassigned agents.
"""

from typing import Dict, Any, List
import pytest
import pandas as pd
import numpy as np

from src.pipeline.loader import DataLoader
from src.analytics.bottlenecks import BottleneckDetector
from src.config import TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE, SLA_TARGETS


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
def detector(data):
    """Instantiate BottleneckDetector with canonical operational data."""
    return BottleneckDetector(
        tickets_df=data["tickets"],
        agents_df=data["agents"],
        merchants_df=data["merchants"],
    )


class TestBottleneckDataPartitioning:
    """Validate strict separation between historical outcome analytics and open backlog."""

    def test_open_ticket_exclusion(self, detector, data):
        """Historical outcome calculations must strictly exclude open tickets (N=250)."""
        assert len(detector.historical_tickets) == 1807
        assert len(detector.open_tickets) == 250
        assert len(detector.raw_tickets) == 2057

        # Historical tickets must all have valid resolution metrics
        assert detector.historical_tickets["resolution_hours"].notna().all()
        assert detector.historical_tickets["is_legacy"].all()

        # Open tickets must not have closed resolution hours
        assert detector.open_tickets["resolution_hours"].isna().all()
        assert not detector.open_tickets["is_legacy"].any()

    def test_get_summary_kpis_excludes_open_tickets(self, detector):
        """Summary KPIs must compute over 1,807 tickets, not 2,057 total."""
        kpis = detector.get_summary_kpis()
        assert kpis["total_tickets"] == 1807
        assert kpis["open_queue_backlog"] == 250
        assert 0.50 <= kpis["resolution_breach_rate"] <= 0.65
        assert 0.15 <= kpis["response_breach_rate"] <= 0.25
        assert 0.10 <= kpis["reopen_rate"] <= 0.25


class TestMetricCalculations:
    """Validate empirical calculation accuracy across operational dimensions."""

    def test_category_mismatches_detection(self, detector):
        """Quantify dispatch mismatch volume and SLA penalty."""
        mismatch = detector.detect_category_mismatches()
        assert mismatch["total_tickets"] == 1807
        assert mismatch["mismatch_count"] == 538
        assert abs(mismatch["mismatch_rate"] - (538 / 1807)) < 1e-4

        # Mismatched tickets must have higher breach rate than matched tickets
        assert mismatch["mismatched_breach_rate"] > mismatch["matched_breach_rate"]
        assert mismatch["breach_penalty"] > 20.0  # ~24.95 pp penalty
        assert mismatch["hours_penalty"] > 10.0   # ~12.1h mean delay
        assert mismatch["excess_breaches"] > 0

    def test_category_hotspots(self, detector):
        """Validate category distribution and volume conservation."""
        cat_df = detector.detect_category_hotspots()
        assert len(cat_df) == 5
        assert cat_df["total_volume"].sum() == 1807
        assert set(cat_df["category"].tolist()) == {
            "Account Access", "API Integrations", "Fulfilment & Logistics",
            "Notifications", "Payments & Checkout"
        }
        # Payments and API Integrations have the highest breach rates
        top_breach_cats = cat_df.sort_values("breach_rate", ascending=False)["category"].tolist()
        assert "Payments & Checkout" in top_breach_cats[:2]
        assert "API Integrations" in top_breach_cats[:2]

    def test_priority_bottlenecks(self, detector):
        """Validate priority SLA overage against target windows."""
        prio_df = detector.detect_priority_bottlenecks()
        assert len(prio_df) == 4
        assert prio_df["priority"].tolist() == ["P1", "P2", "P3", "P4"]

        # Targets match SLA_TARGETS config
        for _, row in prio_df.iterrows():
            p = row["priority"]
            expected_target = SLA_TARGETS[p]["resolution_hours"]
            assert row["target_resolution_hours"] == expected_target

        # P1 and P2 breach rates exceed 70%
        p1_rate = prio_df[prio_df["priority"] == "P1"]["breach_rate"].values[0]
        p2_rate = prio_df[prio_df["priority"] == "P2"]["breach_rate"].values[0]
        assert p1_rate > 0.70
        assert p2_rate > 0.70

    def test_agent_and_tier_diagnostics(self, detector):
        """Validate agent load and tier performance gradient."""
        agent_df = detector.detect_agent_load_imbalance()
        assert len(agent_df) == 20
        assert agent_df["ticket_count"].sum() == 1807
        assert (agent_df["ticket_count"] > 0).all()

        tier_df = detector.detect_agent_tier_performance()
        assert len(tier_df) == 3
        assert tier_df["tier"].tolist() == ["L1", "L2", "L3"]
        # L1 agents have higher breach rate than L3 agents
        l1_br = tier_df[tier_df["tier"] == "L1"]["breach_rate"].values[0]
        l3_br = tier_df[tier_df["tier"] == "L3"]["breach_rate"].values[0]
        assert l1_br > l3_br

    def test_shift_region_imbalance(self, detector):
        """Validate regional shift coverage asymmetry."""
        shift_df = detector.detect_shift_region_imbalance()
        assert len(shift_df) == 3
        uk_row = shift_df[shift_df["region"] == "UK & Europe"].iloc[0]
        # UK merchants ~35.7%, tickets ~33.5%, but only 2 agents (10.0%)
        assert uk_row["agent_count"] == 2
        assert uk_row["merchant_pct"] > 30.0
        assert uk_row["ticket_pct"] > 30.0

    def test_merchant_segment_bottlenecks(self, detector):
        """Validate segmentation by merchant sector."""
        sector_df = detector.detect_merchant_segment_bottlenecks(segment_col="sector")
        assert len(sector_df) > 0
        assert sector_df["volume"].sum() == 1807

    def test_intake_time_patterns(self, detector):
        """Validate temporal intake patterns across DOW, hours, and weekends."""
        time_patterns = detector.detect_intake_time_patterns()
        assert "by_dow" in time_patterns
        assert "by_hour" in time_patterns
        assert "by_weekend" in time_patterns
        assert "by_hour_bin" in time_patterns
        assert len(time_patterns["by_dow"]) == 7
        assert time_patterns["by_dow"]["volume"].sum() == 1807


class TestRankingFormulaAndStability:
    """Validate explicit ranking formula and shrinkage behavior."""

    def test_ranking_formula_correctness(self, detector):
        """Validate explicit formula: Excess Breaches * (N / (N + k))."""
        ranked = detector.rank_bottlenecks(min_sample_size=10, shrinkage_k=10.0, top_n=5)
        assert len(ranked) == 5
        assert list(ranked["rank"]) == [1, 2, 3, 4, 5]

        for _, row in ranked.iterrows():
            n = row["sample_size"]
            breaches = row["breach_count"]
            baseline = row["baseline_breach_rate"]
            expected_excess = max(0.0, round(breaches - n * baseline, 1))
            expected_weight = round(n / (n + 10.0), 3)
            expected_score = round(expected_excess * expected_weight, 2)

            assert abs(row["excess_breaches"] - expected_excess) <= 0.15
            assert abs(row["credibility_weight"] - expected_weight) <= 0.01
            assert abs(row["bottleneck_score"] - expected_score) <= 0.15

    def test_shrinkage_penalizes_small_samples(self, detector):
        """Increasing shrinkage parameter k must disproportionately downweight small groups."""
        ranked_k5 = detector.rank_bottlenecks(min_sample_size=5, shrinkage_k=5.0, top_n=10)
        ranked_k50 = detector.rank_bottlenecks(min_sample_size=5, shrinkage_k=50.0, top_n=10)

        # Higher k yields strictly lower credibility weights for any finite N
        assert (ranked_k50["credibility_weight"] < ranked_k5["credibility_weight"]).all()

    def test_ranking_stability_and_determinism(self, detector):
        """Two identical calls must produce bitwise-identical ranking tables."""
        r1 = detector.rank_bottlenecks(min_sample_size=15, shrinkage_k=10.0, top_n=10)
        r2 = detector.rank_bottlenecks(min_sample_size=15, shrinkage_k=10.0, top_n=10)
        pd.testing.assert_frame_equal(r1, r2)

    def test_rank_bottlenecks_breach_rate_stored_as_proportion(self, detector):
        """Underlying breach_rate in rank_bottlenecks must be stored as proportion (0-1)."""
        ranked = detector.rank_bottlenecks(min_sample_size=10, shrinkage_k=10.0, top_n=10)
        assert len(ranked) >= 3

        # Values must be proportions (0-1), not percentages (> 1.0)
        assert (ranked["breach_rate"] >= 0.0).all()
        assert (ranked["breach_rate"] <= 1.0).all()

        # breach_rate must exactly equal breach_count / sample_size
        for _, row in ranked.iterrows():
            expected_rate = row["breach_count"] / row["sample_size"]
            assert abs(row["breach_rate"] - expected_rate) < 1e-6

        # Check confirmed bottleneck failure mode proportions
        payout = ranked[ranked["sub_category"] == "Payout Delays"].iloc[0]
        assert payout["sample_size"] == 119
        assert payout["breach_count"] == 108
        assert abs(payout["breach_rate"] - (108 / 119)) < 1e-6
        assert abs(payout["breach_rate"] - 0.90756) < 0.001

        card = ranked[ranked["sub_category"] == "Card Declines"].iloc[0]
        assert card["sample_size"] == 111
        assert card["breach_count"] == 98
        assert abs(card["breach_rate"] - (98 / 111)) < 1e-6
        assert abs(card["breach_rate"] - 0.88288) < 0.001

        auth = ranked[ranked["sub_category"] == "3DS Authentication"].iloc[0]
        assert auth["sample_size"] == 105
        assert auth["breach_count"] == 94
        assert abs(auth["breach_rate"] - (94 / 105)) < 1e-6
        assert abs(auth["breach_rate"] - 0.89524) < 0.001


class TestStructuredInsightsContract:
    """Validate structure and integrity of leadership insights."""

    def test_insight_schema_and_fields(self, detector):
        """Every insight must contain all 6 required fields with non-empty values."""
        insights = detector.get_structured_insights()
        assert len(insights) >= 5

        for item in insights:
            assert "title" in item and len(item["title"]) > 0
            assert "metric_values" in item and isinstance(item["metric_values"], dict)
            assert "sample_size" in item and isinstance(item["sample_size"], int) and item["sample_size"] > 0
            assert "comparison" in item and len(item["comparison"]) > 0
            assert "suggested_action" in item and len(item["suggested_action"]) > 0
            assert "caveat" in item and len(item["caveat"]) > 0

    def test_is_incident_ticket_invariance_note(self, detector):
        """Must explicitly document the zero-variation incident ticket invariant."""
        insights = detector.get_structured_insights()
        incident_insights = [i for i in insights if "Incident Ticket Invariance" in i["title"]]
        assert len(incident_insights) == 1
        item = incident_insights[0]
        assert item["metric_values"]["incident_ticket_count"] == 0
        assert "cannot explain" in item["suggested_action"].lower() or "zero variation" in item["suggested_action"].lower()


class TestRobustnessAndEdgeCases:
    """Validate detector resilience against empty dataframes, missing values, and zero denominators."""

    def test_empty_dataframe_handling(self, data):
        """Detector must gracefully handle empty ticket dataframes without crashing."""
        empty_tickets = data["tickets"].iloc[:0].copy()
        detector_empty = BottleneckDetector(empty_tickets, data["agents"], data["merchants"])

        kpis = detector_empty.get_summary_kpis()
        assert kpis["total_tickets"] == 0
        assert kpis["resolution_breach_rate"] == 0.0

        mismatches = detector_empty.detect_category_mismatches()
        assert mismatches["total_tickets"] == 0

        ranked = detector_empty.rank_bottlenecks()
        assert len(ranked) == 0

        insights = detector_empty.get_structured_insights()
        assert insights == []

    def test_missing_values_in_tickets(self, data):
        """Detector must handle unassigned tickets and missing merchant links gracefully."""
        dirty_tickets = data["tickets"].copy()
        # Set some agent IDs to NaN (unassigned tickets)
        dirty_tickets.loc[dirty_tickets.index[:50], "assigned_agent_id"] = np.nan
        # Set some merchant IDs to an unknown ID
        dirty_tickets.loc[dirty_tickets.index[50:100], "merchant_id"] = 999999

        detector_dirty = BottleneckDetector(dirty_tickets, data["agents"], data["merchants"])
        kpis = detector_dirty.get_summary_kpis()
        assert kpis["total_tickets"] == 1807

        agent_perf = detector_dirty.detect_agent_load_imbalance()
        assert len(agent_perf) > 0

        ranked = detector_dirty.rank_bottlenecks()
        assert len(ranked) > 0

    def test_zero_breach_edge_case(self, data):
        """Detector must handle scenario with 0 breaches without ZeroDivisionError."""
        no_breach_tickets = data["tickets"].copy()
        no_breach_tickets["resolution_breached"] = False
        no_breach_tickets["response_breached"] = False

        detector_nobreach = BottleneckDetector(no_breach_tickets, data["agents"], data["merchants"])
        kpis = detector_nobreach.get_summary_kpis()
        assert kpis["resolution_breach_rate"] == 0.0
        assert kpis["resolution_breach_count"] == 0

        ranked = detector_nobreach.rank_bottlenecks()
        assert (ranked["excess_breaches"] == 0.0).all()
        assert (ranked["bottleneck_score"] == 0.0).all()
