"""Capacity-constrained assignment planning using predicted SLA risk.

Capacities are explicit operator-supplied scenario inputs, not inferred live state.
Alternative assignment risks are observational estimates, not causal guarantees.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.models.assignment_risk import assignment_features
from src.config import SLA_TARGETS


class DecisionEngine:
    def __init__(self, model, agents, merchants):
        self.model = model
        self.agents = agents.reset_index(drop=True).copy()
        self.merchants = merchants

    def plan(self, tickets, intake_risk, capacities):
        tickets = tickets.reset_index(drop=True).copy()
        n, m = len(tickets), len(self.agents)
        if not n:
            return pd.DataFrame(), {"planned": 0, "waiting": 0}
        intake_risk = np.asarray(intake_risk, dtype=float)
        if len(intake_risk) != n or not np.isfinite(intake_risk).all() or ((intake_risk < 0) | (intake_risk > 1)).any():
            raise ValueError("Every ticket requires an intake risk between 0 and 1")
        arrival_risk = intake_risk.copy()
        limits = []
        for agent_id in self.agents.agent_id:
            value = capacities.get(int(agent_id), 0)
            if not np.isfinite(value) or value < 0 or value != int(value):
                raise ValueError("Planning capacities must be nonnegative whole numbers")
            limits.append(min(int(value), n))
        limits = np.asarray(limits)
        if int(limits.sum()) > 20000:
            raise ValueError("Planning capacity is too large for this batch; reduce per-agent budgets")

        pairs = tickets.iloc[np.repeat(np.arange(n), m)].reset_index(drop=True)
        ids = np.tile(self.agents.agent_id.to_numpy(), n)
        features = assignment_features(pairs, self.merchants, self.agents, ids)
        risks = self.model.predict_risk(features).reshape(n, m)
        # Open tickets past their deadline are already breaches, not uncertain forecasts.
        snapshot = pd.to_datetime(tickets.created_at).max()
        elapsed = (snapshot-pd.to_datetime(tickets.created_at)).dt.total_seconds().to_numpy()/3600
        remaining = tickets.priority.map(lambda p: SLA_TARGETS.get(p, SLA_TARGETS["P3"])["resolution_hours"]).to_numpy()-elapsed
        overdue = remaining <= 0
        risks[overdue, :] = 1.0
        intake_risk = np.where(overdue, 1.0, intake_risk)
        domain = tickets.category.to_numpy()[:, None] == self.agents.primary_category.to_numpy()[None, :]
        senior = self.agents.tier.isin(["L2", "L3"]).to_numpy()
        urgent = tickets.priority.isin(["P1", "P2"]).to_numpy()
        eligible = domain & (~urgent[:, None] | senior[None, :])
        tier_lookup = self.merchants.set_index("merchant_id").tier
        impact = tickets.merchant_id.map(tier_lookup).map({"Advanced": 1.5, "Foundation": 1.0}).fillna(1).to_numpy()
        urgency = tickets.priority.map({"P1": 8, "P2": 5, "P3": 2, "P4": 1}).fillna(2).to_numpy()
        weights = urgency * impact
        # Deferring an urgent high-risk ticket carries a larger planning penalty.
        defer_cost = weights * (1+intake_risk)
        slots = np.repeat(np.arange(m), limits)
        slot_depth = np.concatenate([np.arange(limit) for limit in limits]) if len(slots) else np.array([])
        assigned_cost = weights[:, None]*risks[:, slots] + 0.015*slot_depth[None, :]
        assigned_cost = np.where(eligible[:, slots], assigned_cost, 1e6)
        # A private dummy slot per ticket guarantees feasible solutions when overloaded.
        costs = np.column_stack([assigned_cost, np.repeat(defer_cost[:, None], n, axis=1)])
        rows, columns = linear_sum_assignment(costs)
        allocation = np.full(n, -1, dtype=int)
        for row, column in zip(rows, columns):
            if column < len(slots) and assigned_cost[row, column] < 1e6:
                allocation[row] = slots[column]

        # Capacity- and eligibility-matched round-robin reference policy.
        baseline = np.full(n, -1, dtype=int)
        usage = np.zeros(m, dtype=int)
        for row in np.argsort(-defer_cost, kind="stable"):
            available = np.where(eligible[row] & (usage < limits))[0]
            if len(available):
                agent = min(available, key=lambda j: (usage[j], int(self.agents.agent_id.iloc[j])))
                baseline[row] = agent
                usage[agent] += 1

        output = tickets[["ticket_id", "merchant_id", "category", "priority"]].copy()
        output["intake_risk_pct"] = np.round(100*arrival_risk, 1)
        output["sla_remaining_hours"] = np.round(remaining, 2)
        output["sla_state"] = np.where(overdue, "Already overdue", "Within SLA window")
        output["agent_id"] = pd.array([int(self.agents.agent_id.iloc[j]) if j >= 0 else None for j in allocation], dtype="Int64")
        output["agent_name"] = [self.agents.agent_name.iloc[j] if j >= 0 else "Awaiting capacity" for j in allocation]
        planned_risk = np.array([risks[i, j] if j >= 0 else intake_risk[i] for i, j in enumerate(allocation)])
        baseline_risk = np.array([risks[i, j] if j >= 0 else intake_risk[i] for i, j in enumerate(baseline)])
        output["assignment_risk_pct"] = [round(100*risks[i, j], 1) if j >= 0 else np.nan for i, j in enumerate(allocation)]
        output["baseline_risk_pct"] = np.round(100*baseline_risk, 1)
        output["status"] = np.where(allocation >= 0, "Planned", "Waiting")
        explanations = []
        for i, j in enumerate(allocation):
            deadline_note = " Deadline already passed at the dataset snapshot; prioritize recovery. Reassignment cannot undo that breach." if overdue[i] else ""
            if j < 0:
                explanations.append("No remaining capacity among domain specialists with the required seniority; escalate staffing or revise the planning budget." + deadline_note)
            else:
                agent = self.agents.iloc[j]
                explanations.append(
                    f"{agent.primary_category} specialist; {agent.tier}; efficiency {agent.efficiency_multiplier:.3f}x; "
                    f"{agent.shift_region} shift; estimated breach risk {risks[i,j]:.1%}. "
                    "Selected within the full batch's capacity and urgency constraints." + deadline_note
                )
        output["evidence"] = explanations
        summary = {
            "planned": int((allocation >= 0).sum()), "waiting": int((allocation < 0).sum()),
            "baseline_planned": int((baseline >= 0).sum()),
            "planning_capacity": int(limits.sum()),
            "already_overdue": int(overdue.sum()), "snapshot": str(snapshot),
            "baseline_expected_breaches": float(baseline_risk.sum()),
            "planned_expected_breaches": float(planned_risk.sum()),
            "estimated_breaches_avoided": float((baseline_risk-planned_risk).sum()),
            "baseline_weighted_cost": float(np.where(baseline >= 0, weights*baseline_risk, defer_cost).sum()),
            "planned_weighted_cost": float(np.where(allocation >= 0, weights*planned_risk, defer_cost).sum()),
            "assumptions": "Past-deadline tickets count as certain breaches. Other waiting-ticket risks retain the intake estimate; assignment estimates use the proposed agent. Predicted differences are scenario estimates, not realized outcomes.",
        }
        return output, summary
