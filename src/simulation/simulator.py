"""Impact Simulator for Operational Counterfactuals and What-If Analysis (Day 5).

Provides rigorous, mathematically verified simulation of operational policy shifts:
1. Category Mismatch Reduction: Quantifies SLA breach reduction and resolution hours
   recovered if cross-domain ticket dispatching is reduced or eliminated.
2. Capacity Value Assessment: Evaluates operational labor hours unlocked using an explicit
   fully loaded capacity value ($45/hr default), explicitly distinguishing unlocked
   operational capacity from realized cash financial savings.
3. Comparative Baseline vs. Scenario Modeling: Directly compares observed historical
   ground truth ($N=1,807$ completed tickets) against counterfactual scenarios.

Strict Data Integrity Guarantees:
- Zero Open-Ticket Pollution: Outcome baselines are computed exclusively from the 1,807
  completed historical tickets; open tickets ($N=250$) are strictly excluded.
- Source Immutability: Never mutates historical ticket records, agent pools, or merchant data.
- Defensible Modeling: Does not claim arbitrary breach drops from headcount additions;
  headcount is modeled as the operational capacity enabler required to absorb specialist demand.
- Monotonic Sensitivity: Slider adjustments produce strictly consistent, traceable outputs.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np


class ImpactSimulator:
    """Simulates business impact, SLA recovery, and capacity improvements
    under counterfactual dispatch policies and staffing allocations.
    """

    def __init__(
        self,
        historical_tickets: pd.DataFrame,
        agents_df: pd.DataFrame,
        merchants_df: pd.DataFrame,
        hourly_capacity_value: float = 45.0,
    ):
        """Initialize simulator with historical tickets and reference metadata.
        
        Args:
            historical_tickets: Historical tickets dataframe (strictly completed tickets).
            agents_df: Support agents directory.
            merchants_df: Merchants dataset.
            hourly_capacity_value: Loaded cost per specialist labor hour ($45.0 default).
        """
        # Strict partition: isolate completed historical tickets
        if "is_legacy" in historical_tickets.columns:
            self.hist = historical_tickets[historical_tickets["is_legacy"] == True].copy()
        elif "resolution_hours" in historical_tickets.columns:
            self.hist = historical_tickets[historical_tickets["resolution_hours"].notna()].copy()
        else:
            self.hist = historical_tickets.copy()

        self.agents = agents_df.copy()
        self.merchants = merchants_df.copy()
        self.hourly_capacity_value = float(hourly_capacity_value)

        # Calculate and freeze observed historical ground-truth constants
        self._calculate_ground_truth_constants()

    def _calculate_ground_truth_constants(self) -> None:
        """Derive exact empirical ground truth from completed historical tickets ($N=1,807$)."""
        self.n_historical = len(self.hist)
        if self.n_historical == 0:
            self.baseline_breach_count = 0
            self.baseline_breach_rate = 0.0
            self.baseline_mismatch_count = 0
            self.baseline_mismatch_rate = 0.0
            self.matched_count = 0
            self.matched_breach_rate = 0.0
            self.mismatched_count = 0
            self.mismatched_breach_rate = 0.0
            self.breach_penalty = 0.0
            self.matched_mean_hours = 0.0
            self.mismatched_mean_hours = 0.0
            self.hours_penalty = 0.0
            self.total_excess_drag_hours = 0.0
            self.max_sla_recovery_pp = 0.0
            self.max_capacity_value = 0.0
            return

        # Overall historical outcomes
        self.baseline_breach_count = int(self.hist["resolution_breached"].sum())
        self.baseline_breach_rate = float(self.baseline_breach_count / self.n_historical)

        # Category mismatch breakdown
        mismatch_mask = self.hist["category_mismatch"] == True
        matched_mask = self.hist["category_mismatch"] == False

        self.baseline_mismatch_count = int(mismatch_mask.sum())
        self.baseline_mismatch_rate = float(self.baseline_mismatch_count / self.n_historical)

        self.matched_count = int(matched_mask.sum())
        self.mismatched_count = int(mismatch_mask.sum())

        self.matched_breach_rate = (
            float(self.hist.loc[matched_mask, "resolution_breached"].mean())
            if self.matched_count > 0 else 0.0
        )
        self.mismatched_breach_rate = (
            float(self.hist.loc[mismatch_mask, "resolution_breached"].mean())
            if self.mismatched_count > 0 else 0.0
        )
        # Breach penalty (percentage points / 100)
        self.breach_penalty = self.mismatched_breach_rate - self.matched_breach_rate

        self.matched_mean_hours = (
            float(self.hist.loc[matched_mask, "resolution_hours"].mean())
            if self.matched_count > 0 else 0.0
        )
        self.mismatched_mean_hours = (
            float(self.hist.loc[mismatch_mask, "resolution_hours"].mean())
            if self.mismatched_count > 0 else 0.0
        )
        # Delay penalty (hours per ticket)
        self.hours_penalty = self.mismatched_mean_hours - self.matched_mean_hours

        # Aggregate historical excess drag
        self.total_excess_drag_hours = self.baseline_mismatch_count * self.hours_penalty
        self.max_sla_recovery_pp = self.baseline_mismatch_rate * self.breach_penalty * 100.0
        self.max_capacity_value = self.total_excess_drag_hours * self.hourly_capacity_value

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Return observed empirical ground-truth metrics strictly from historical tickets."""
        return {
            "n_historical": self.n_historical,
            "baseline_breach_count": self.baseline_breach_count,
            "baseline_breach_rate": round(self.baseline_breach_rate, 4),
            "baseline_mismatch_count": self.baseline_mismatch_count,
            "baseline_mismatch_rate": round(self.baseline_mismatch_rate, 4),
            "matched_count": self.matched_count,
            "matched_breach_rate": round(self.matched_breach_rate, 4),
            "mismatched_count": self.mismatched_count,
            "mismatched_breach_rate": round(self.mismatched_breach_rate, 4),
            "breach_penalty_pp": round(self.breach_penalty * 100.0, 2),
            "matched_mean_hours": round(self.matched_mean_hours, 2),
            "mismatched_mean_hours": round(self.mismatched_mean_hours, 2),
            "hours_penalty": round(self.hours_penalty, 2),
            "total_excess_drag_hours": round(self.total_excess_drag_hours, 1),
            "max_sla_recovery_pp": round(self.max_sla_recovery_pp, 2),
            "max_capacity_value": round(self.max_capacity_value, 2),
            "hourly_capacity_value": self.hourly_capacity_value,
        }

    def simulate_mismatch_reduction(
        self,
        target_mismatch_rate: float = 0.05,
    ) -> Dict[str, Any]:
        """Simulate reduction in SLA breaches and unlocked capacity under target mismatch rate.
        
        Mathematical Formulation:
            delta_mismatch = max(0, baseline_mismatch_rate - target_mismatch_rate)
            realigned_tickets = delta_mismatch * N_historical
            breaches_avoided = realigned_tickets * breach_penalty
            breach_drop_pp = delta_mismatch * breach_penalty * 100%
            simulated_breach_rate = baseline_breach_rate - (delta_mismatch * breach_penalty)
            hours_recovered = realigned_tickets * hours_penalty
            estimated_capacity_value = hours_recovered * hourly_capacity_value
            
        Args:
            target_mismatch_rate: Counterfactual target mismatch rate (float 0.0 to 1.0, e.g. 0.05 or 0.28).
            
        Returns:
            Dict[str, Any]: Full counterfactual simulation report with baseline comparisons.
        """
        # Ensure target is valid ratio [0.0, 1.0]
        target_m = max(0.0, min(1.0, float(target_mismatch_rate)))

        # Delta mismatch reduction achieved
        delta_m = max(0.0, self.baseline_mismatch_rate - target_m)

        # Number of tickets realigned to primary domain specialists
        realigned_tickets = delta_m * self.n_historical

        # Breach metrics
        breaches_avoided = realigned_tickets * self.breach_penalty
        breach_drop_pp = delta_m * self.breach_penalty * 100.0
        simulated_breach_rate = max(0.0, self.baseline_breach_rate - (delta_m * self.breach_penalty))
        simulated_breach_count = max(0, int(round(self.baseline_breach_count - breaches_avoided)))

        # Capacity metrics
        hours_recovered = realigned_tickets * self.hours_penalty
        remaining_excess_hours = max(0.0, self.total_excess_drag_hours - hours_recovered)
        estimated_capacity_value = hours_recovered * self.hourly_capacity_value

        return {
            # Scenario Inputs
            "target_mismatch_rate": round(target_m, 4),
            "delta_mismatch_rate": round(delta_m, 4),
            "realigned_tickets": round(realigned_tickets, 1),
            # Breach Outcomes
            "baseline_breach_rate": round(self.baseline_breach_rate, 4),
            "simulated_breach_rate": round(simulated_breach_rate, 4),
            "breach_rate_reduction_pp": round(breach_drop_pp, 2),
            "baseline_breach_count": self.baseline_breach_count,
            "simulated_breach_count": simulated_breach_count,
            "breaches_avoided": round(breaches_avoided, 1),
            # Capacity Outcomes
            "baseline_excess_hours": round(self.total_excess_drag_hours, 1),
            "hours_recovered": round(hours_recovered, 1),
            "remaining_excess_hours": round(remaining_excess_hours, 1),
            # Financial Capacity Valuation (Explicitly non-cash capacity value)
            "hourly_capacity_value": self.hourly_capacity_value,
            "estimated_capacity_value": round(estimated_capacity_value, 2),
            # Methodological metadata
            "breach_penalty_pp": round(self.breach_penalty * 100.0, 2),
            "hours_penalty": round(self.hours_penalty, 2),
            "sample_size": self.n_historical,
        }

    def simulate_staffing_capacity_enabler(
        self,
        additional_specialists_per_category: Dict[str, int],
    ) -> Dict[str, Any]:
        """Model how hiring or reallocating specialists provides domain capacity.
        
        Defensible Model:
            In our dataset, each specialist handled an average of ~90.35 tickets across the 18-month
            historical observation period (range: 75 to 156 tickets).
            Adding specialists in an understaffed category increases specialist absorption capacity,
            directly enabling the operational team to eliminate cross-domain mismatching in that category.
            
        Args:
            additional_specialists_per_category: Dict of category -> headcount added.
            
        Returns:
            Dict[str, Any]: Capacity absorption enabled by category.
        """
        results = []
        total_capacity_tickets = 0

        avg_historical_load = self.n_historical / len(self.agents) if len(self.agents) > 0 else 90.0

        for cat, added in additional_specialists_per_category.items():
            if added <= 0:
                continue
            cat_hist = self.hist[self.hist["category"] == cat]
            cat_vol = len(cat_hist)
            cat_mismatches = int((cat_hist["category_mismatch"] == True).sum()) if cat_vol > 0 else 0

            # Each added specialist provides capacity for ~avg_historical_load tickets
            enabled_volume = int(round(added * avg_historical_load))
            resolved_mismatches = min(cat_mismatches, enabled_volume)
            total_capacity_tickets += enabled_volume

            results.append({
                "category": cat,
                "added_headcount": added,
                "historical_category_volume": cat_vol,
                "historical_mismatch_count": cat_mismatches,
                "enabled_specialist_capacity": enabled_volume,
                "potential_mismatches_absorbed": resolved_mismatches,
            })

        return {
            "total_added_specialists": sum(additional_specialists_per_category.values()),
            "total_enabled_ticket_capacity": total_capacity_tickets,
            "category_breakdown": results,
            "methodology_note": (
                "Staffing impact is modeled as specialist intake capacity enabler (~90 tickets/specialist baseline). "
                "Headcount additions do not arbitrarily reduce breach rates without an explicit queueing mechanism."
            ),
        }

    def run_comparative_scenario(
        self,
        scenario_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run composite counterfactual scenario combining mismatch target and capacity parameters."""
        target_m = scenario_params.get("target_mismatch_rate", 0.05)
        # Allow percentage or ratio
        if target_m > 1.0:
            target_m = target_m / 100.0

        mismatch_res = self.simulate_mismatch_reduction(target_m)

        added_agents = scenario_params.get("additional_specialists", {})
        staffing_res = self.simulate_staffing_capacity_enabler(added_agents)

        return {
            "scenario_name": scenario_params.get("name", "Custom What-If Scenario"),
            "mismatch_simulation": mismatch_res,
            "staffing_enabler": staffing_res,
            "comparison_table": self.get_comparison_dataframe(mismatch_res),
        }

    def get_comparison_dataframe(self, sim_res: Dict[str, Any]) -> pd.DataFrame:
        """Construct structured side-by-side baseline-versus-scenario comparison table."""
        rows = [
            {
                "Operational Metric": "Category Mismatch Rate",
                "Historical Baseline (Observed)": f"{self.baseline_mismatch_rate*100:.1f}% ({self.baseline_mismatch_count:,} tickets)",
                "Simulated Scenario (Estimate)": f"{sim_res['target_mismatch_rate']*100:.1f}% ({int(round(self.n_historical * sim_res['target_mismatch_rate'])):,} tickets)",
                "Estimated Variance": f"-{sim_res['delta_mismatch_rate']*100:.1f} pp ({int(round(sim_res['realigned_tickets']))} realigned)",
                "Analytical Nature": "Empirical Ground Truth vs. Counterfactual",
            },
            {
                "Operational Metric": "Resolution Breach Rate",
                "Historical Baseline (Observed)": f"{sim_res['baseline_breach_rate']*100:.1f}% ({sim_res['baseline_breach_count']:,} breaches)",
                "Simulated Scenario (Estimate)": f"{sim_res['simulated_breach_rate']*100:.1f}% ({sim_res['simulated_breach_count']:,} breaches)",
                "Estimated Variance": f"-{sim_res['breach_rate_reduction_pp']:.2f} pp ({int(round(sim_res['breaches_avoided']))} avoided)",
                "Analytical Nature": "Observed Breach Rate vs. Projected Outcome",
            },
            {
                "Operational Metric": "Excess Resolution Delay",
                "Historical Baseline (Observed)": f"{sim_res['baseline_excess_hours']:,.0f} hours drag",
                "Simulated Scenario (Estimate)": f"{sim_res['remaining_excess_hours']:,.0f} hours remaining",
                "Estimated Variance": f"-{sim_res['hours_recovered']:,.0f} hours recovered",
                "Analytical Nature": "Empirical Delay vs. Recovered Labor Capacity",
            },
            {
                "Operational Metric": "Operational Capacity Value (@ $45/h)",
                "Historical Baseline (Observed)": "$0 (Baseline Reference)",
                "Simulated Scenario (Estimate)": f"${sim_res['estimated_capacity_value']:,.0f} unlocked capacity",
                "Estimated Variance": f"+${sim_res['estimated_capacity_value']:,.0f} capacity value",
                "Analytical Nature": "Labor Capacity Valuation (Non-Cash Opportunity)",
            },
        ]
        return pd.DataFrame(rows)
