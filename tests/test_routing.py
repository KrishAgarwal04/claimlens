"""Unit and integration tests for Smart Agent Recommendation Engine (Day 4).

Validates:
- Strict primary category specialty matching (preventing the 24.9 pp mismatch penalty).
- Priority-to-tier alignment (P1/P2 routed to L2/L3; P4 to L1).
- Graceful handling of missing categories, unknown domains, and missing priorities.
- Deterministic ranking and reproducible tie-breaking.
- Batch routing across the open queue (250 tickets) under top_match and balanced strategies.
- Schema integrity of recommendation outputs, match reasons, alternatives, and limitations.
"""

from typing import Dict, Any
import pytest
import pandas as pd
import numpy as np

from src.pipeline.loader import DataLoader
from src.routing.recommender import AgentRecommender, DEFAULT_RECOMMENDER_WEIGHTS
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
def recommender(data):
    """Instantiate AgentRecommender with canonical dataset."""
    return AgentRecommender(
        agents_df=data["agents"],
        historical_tickets_df=data["hist_tickets"],
        merchants_df=data["merchants"],
    )


class TestRecommendationMatchingRules:
    """Validate core multi-criteria matching rules and scoring constraints."""

    def test_strict_category_matching(self, recommender, data):
        """Recommended agent must strictly match ticket category when specialists exist."""
        for cat in data["agents"]["primary_category"].unique():
            ticket = {"ticket_id": 101, "category": cat, "priority": "P2"}
            res = recommender.recommend_for_ticket(ticket)

            assert res["status"] == "SUCCESS"
            assert res["recommended_agent_category"] == cat
            assert res["match_score"] > 0.60
            assert any("Domain specialist" in r for r in res["reasons"])

    def test_priority_tier_handling_p1_p2(self, recommender):
        """Urgent P1/P2 tickets must favor senior L2/L3 specialists over L1."""
        # API Integrations has specialists across L1, L2, L3 (Alex Mercer L1, Priya Sharma L2, Julian Ross L3)
        ticket_p1 = {"ticket_id": 201, "category": "API Integrations", "priority": "P1"}
        res_p1 = recommender.recommend_for_ticket(ticket_p1, top_k=4)

        assert res_p1["status"] == "SUCCESS"
        assert res_p1["recommended_agent_tier"] in ("L2", "L3")

        # Find L1 candidate score among alternatives
        l1_scores = [
            alt["match_score"] for alt in res_p1["alternatives"]
            if alt["tier"] == "L1"
        ]
        if l1_scores:
            assert res_p1["match_score"] > max(l1_scores)

    def test_priority_tier_handling_p4(self, recommender):
        """Low-severity P4 tickets must prioritize L1 specialists to preserve senior capacity."""
        ticket_p4 = {"ticket_id": 202, "category": "Fulfilment & Logistics", "priority": "P4"}
        res_p4 = recommender.recommend_for_ticket(ticket_p4, top_k=4)

        assert res_p4["status"] == "SUCCESS"
        assert res_p4["recommended_agent_tier"] == "L1"

    def test_regional_shift_alignment(self, recommender, data):
        """Agent working in merchant region receives higher score than cross-region agent."""
        # Priya Sharma (214) is API Integrations in Asia-Pacific
        # Julian Ross (211) is API Integrations in North America
        ticket_apac = {
            "ticket_id": 301,
            "category": "API Integrations",
            "priority": "P2",
            "merchant_region": "Asia-Pacific",
        }
        res_apac = recommender.recommend_for_ticket(ticket_apac)
        assert res_apac["recommended_agent_shift"] == "Asia-Pacific"


class TestEdgeCasesAndFallbacks:
    """Validate graceful degradation and error handling on anomalous inputs."""

    def test_missing_category_handling(self, recommender):
        """Ticket with null/missing category returns transparent NO_MATCH fallback."""
        for null_val in [None, np.nan, ""]:
            ticket = {"ticket_id": 401, "category": null_val, "priority": "P1"}
            res = recommender.recommend_for_ticket(ticket)

            assert res["status"] == "NO_MATCH"
            assert res["recommended_agent_id"] is None
            assert res["match_score"] == 0.0
            assert res["alternatives"] == []
            assert any("missing or null" in r.lower() for r in res["reasons"])

    def test_unknown_category_handling(self, recommender):
        """Unrecognized domain category returns transparent NO_MATCH without throwing exceptions."""
        ticket = {"ticket_id": 402, "category": "Quantum Computing", "priority": "P1"}
        res = recommender.recommend_for_ticket(ticket)

        assert res["status"] == "NO_MATCH"
        assert res["recommended_agent_id"] is None
        assert res["match_score"] == 0.0
        assert any("No support specialists found" in r for r in res["reasons"])

    def test_missing_priority_fallback(self, recommender):
        """Ticket with missing priority defaults gracefully to neutral baseline."""
        ticket = {"ticket_id": 403, "category": "Account Access", "priority": None}
        res = recommender.recommend_for_ticket(ticket)

        assert res["status"] == "SUCCESS"
        assert res["recommended_agent_id"] is not None
        assert res["match_score"] > 0.50

    def test_invalid_agents_schema_raises_error(self):
        """Passing DataFrame missing essential agent columns raises ValueError."""
        invalid_agents = pd.DataFrame({"agent_id": [1, 2], "agent_name": ["A", "B"]})
        with pytest.raises(ValueError, match="missing required columns"):
            AgentRecommender(invalid_agents)

    def test_custom_weights_configuration(self, data):
        """Custom scoring weights are properly normalized and applied."""
        custom_weights = {"category": 0.80, "tier_priority": 0.10, "efficiency": 0.10, "region": 0.00}
        rec_custom = AgentRecommender(data["agents"], weights=custom_weights)

        assert abs(sum(rec_custom.weights.values()) - 1.0) < 1e-6
        assert rec_custom.weights["category"] == 0.80

    def test_deterministic_ranking(self, recommender, data):
        """Repeated recommendation calls on identical ticket must yield bitwise-identical outputs."""
        open_ticket = data["open_tickets"].iloc[0].to_dict()
        r1 = recommender.recommend_for_ticket(open_ticket)
        r2 = recommender.recommend_for_ticket(open_ticket)

        assert r1["recommended_agent_id"] == r2["recommended_agent_id"]
        assert r1["match_score"] == r2["match_score"]
        assert r1["reasons"] == r2["reasons"]
        assert r1["alternatives"] == r2["alternatives"]


class TestBatchRoutingOpenQueue:
    """Validate batch routing capabilities across the live open backlog (250 tickets)."""

    def test_route_batch_top_match(self, recommender, data):
        """Route all 250 open tickets under greedy top_match strategy."""
        open_queue = data["open_tickets"].copy()
        routed = recommender.route_batch(open_queue, strategy="top_match")

        assert len(routed) == 250
        assert "recommended_agent_id" in routed.columns
        assert "recommended_agent_name" in routed.columns
        assert "recommended_agent_tier" in routed.columns
        assert "match_score" in routed.columns
        assert "primary_reason" in routed.columns
        assert "alternatives" in routed.columns

        # All 250 open tickets must be successfully matched with zero mismatches
        assert (routed["status"] == "SUCCESS").all()
        assert (routed["category"] == routed["recommended_agent_category"]).all()
        assert (routed["match_score"] >= 0.70).all()

        # Check summary metrics
        summary = recommender.get_batch_summary(routed)
        assert summary["total_tickets"] == 250
        assert summary["matched_count"] == 250
        assert summary["unmatched_count"] == 0
        assert summary["match_rate"] == 1.0
        assert summary["mean_match_score"] >= 0.85
        assert summary["p1_p2_tier_coverage"] >= 0.90

    def test_route_batch_balanced(self, recommender, data):
        """Route all 250 open tickets under load-balanced category-preserving strategy."""
        open_queue = data["open_tickets"].copy()
        routed_bal = recommender.route_batch(open_queue, strategy="balanced")

        assert len(routed_bal) == 250
        assert (routed_bal["status"] == "SUCCESS").all()
        assert (routed_bal["category"] == routed_bal["recommended_agent_category"]).all()

        summary_bal = recommender.get_batch_summary(routed_bal)
        # In balanced routing, every active agent in the pool must receive tickets
        assigned_agents = set(routed_bal["recommended_agent_id"].unique())
        assert len(assigned_agents) == len(data["agents"])

    def test_structured_recommendation_and_limitations_contract(self, recommender, data):
        """Recommendations must adhere strictly to the schema and include explicit limitations."""
        ticket = data["open_tickets"].iloc[0].to_dict()
        res = recommender.recommend_for_ticket(ticket)

        # Mandatory keys per Day 4 specification
        assert "ticket_id" in res
        assert "recommended_agent_id" in res
        assert "recommended_agent_name" in res
        assert "recommended_agent_tier" in res
        assert "match_score" in res
        assert "reasons" in res and len(res["reasons"]) > 0
        assert "alternatives" in res
        assert "uncertainty_limitations" in res

        # Must explicitly acknowledge observational nature and unobserved live availability
        limitations = res["uncertainty_limitations"].lower()
        assert "observational" in limitations
        assert "availability" in limitations or "concurrency" in limitations
