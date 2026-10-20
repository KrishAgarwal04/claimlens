"""Contract and schema validation tests."""

import pytest
from src.pipeline.loader import DataLoader
from src.config import SLA_TARGETS, LEAKAGE_FEATURES


def test_data_loader_shapes():
    loader = DataLoader()
    tickets, agents, merchants = loader.get_datasets()
    
    assert len(merchants) == 112, f"Expected 112 merchants, got {len(merchants)}"
    assert len(agents) == 20, f"Expected 20 agents, got {len(agents)}"
    assert len(tickets) == 2057, f"Expected 2057 tickets, got {len(tickets)}"


def test_ticket_partitioning():
    loader = DataLoader()
    tickets = loader.load_tickets()
    historical, open_queue = loader.split_tickets(tickets)
    
    assert len(historical) == 1807, f"Expected 1807 historical tickets, got {len(historical)}"
    assert len(open_queue) == 250, f"Expected 250 open queue tickets, got {len(open_queue)}"
    assert open_queue["assigned_agent_id"].isnull().all(), "All open tickets should be unassigned"


def test_sla_targets_defined():
    for priority in ["P1", "P2", "P3", "P4"]:
        assert priority in SLA_TARGETS
        assert "ttfr_hours" in SLA_TARGETS[priority]
        assert "resolution_hours" in SLA_TARGETS[priority]
