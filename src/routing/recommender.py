"""Intelligent Agent Recommendation Engine for OpsPilot AI (Day 4).

Matches incoming unassigned tickets to optimal support specialists based on:
1. Primary domain specialization (eliminating the ~24.9 pp category mismatch penalty).
2. Priority-to-tier capability fit (routing P1/P2 critical tickets to L2/L3 specialists).
3. Continuous agent efficiency rating from directory metadata.
4. Regional shift alignment with merchant account operating window.
5. Statistically supported historical performance context (with explicit sample size caveats).

Strict Data Integrity Guarantees:
- Zero Target Leakage: Never uses post-intake resolution outcomes (resolution_hours,
  resolution_breached, ttfr, csat) as recommendation features.
- No Fabricated Live State: Does not assume or invent unobserved real-time agent
  concurrency, live queue backlog, or active ticket availability.
- Explicit Non-Probabilistic Score: Match scores are structured multi-criteria alignment
  indices [0.0 - 1.0], not calibrated probabilities of SLA compliance.
- Transparent Fallbacks: Handles missing categories, unknown domains, and zero-match cases gracefully.
- Deterministic Sorting: Enforces reproducible tie-breaking on identical scores.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np


# Default multi-criteria scoring weights (must sum to 1.0)
DEFAULT_RECOMMENDER_WEIGHTS: Dict[str, float] = {
    "category": 0.50,       # Domain alignment: primary driver of quality and SLA integrity
    "tier_priority": 0.25,  # Seniority-to-urgency alignment (P1/P2 -> L2/L3)
    "efficiency": 0.15,     # Normalized agent velocity multiplier
    "region": 0.10,         # Regional operating shift alignment
}

# Empirical fit between ticket priority urgency and support specialist tier capability
# Derived from Day 3 findings: L3 agents have 35.6% breach rate vs 75.7% for L1 agents on high-impact tickets
TIER_PRIORITY_FIT: Dict[str, Dict[str, float]] = {
    "P1": {"L3": 1.00, "L2": 0.75, "L1": 0.35},  # P1 (2h window): requires senior escalation authority
    "P2": {"L3": 0.95, "L2": 1.00, "L1": 0.50},  # P2 (8h window): primary target for L2/L3 specialists
    "P3": {"L2": 1.00, "L1": 0.85, "L3": 0.70},  # P3 (24h window): bread-and-butter; preserve L3 capacity
    "P4": {"L1": 1.00, "L2": 0.70, "L3": 0.40},  # P4 (48h window): ideal for junior L1 specialist capacity
}


class AgentRecommender:
    """Recommends and ranks support agents for unassigned incoming tickets."""

    def __init__(
        self,
        agents_df: pd.DataFrame,
        historical_tickets_df: Optional[pd.DataFrame] = None,
        merchants_df: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize recommendation engine with support agent pool and reference metadata.
        
        Args:
            agents_df: Support agents directory with tier, specialty, and shift info.
            historical_tickets_df: Optional completed historical tickets to compute empirical baselines.
            merchants_df: Optional merchant directory to resolve merchant region from merchant_id.
            weights: Optional custom dictionary of multi-criteria weights.
        """
        self._validate_agents_schema(agents_df)
        self.agents = agents_df.copy().reset_index(drop=True)

        # Normalize and validate scoring weights
        self.weights = self._normalize_weights(weights or DEFAULT_RECOMMENDER_WEIGHTS)

        # Efficiency scaling boundaries derived from agents directory
        if "efficiency_multiplier" in self.agents.columns and len(self.agents) > 0:
            self.min_eff = float(self.agents["efficiency_multiplier"].min())
            self.max_eff = float(self.agents["efficiency_multiplier"].max())
        else:
            self.min_eff, self.max_eff = 0.65, 1.40

        # Merchant region lookup dictionary
        self.merchant_regions: Dict[Union[int, str], str] = {}
        if merchants_df is not None and "merchant_id" in merchants_df.columns and "region" in merchants_df.columns:
            self.merchant_regions = dict(zip(merchants_df["merchant_id"], merchants_df["region"]))

        # Empirical agent performance baselines on historical completed tickets
        self.agent_hist_stats: Dict[int, Dict[str, Any]] = {}
        if historical_tickets_df is not None and len(historical_tickets_df) > 0:
            self._compile_historical_agent_stats(historical_tickets_df)

    @staticmethod
    def _validate_agents_schema(df: pd.DataFrame) -> None:
        """Verify agents dataframe contains all required fields."""
        required_cols = {"agent_id", "agent_name", "tier", "primary_category", "shift_region", "efficiency_multiplier"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Agents dataframe missing required columns: {sorted(missing)}")

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure multi-criteria weights sum strictly to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return DEFAULT_RECOMMENDER_WEIGHTS.copy()
        return {k: v / total for k, v in weights.items()}

    def _compile_historical_agent_stats(self, hist_df: pd.DataFrame) -> None:
        """Compute empirical baseline metrics per agent strictly on completed historical tickets."""
        # Ensure only completed tickets are evaluated
        if "is_legacy" in hist_df.columns:
            completed = hist_df[hist_df["is_legacy"] == True].copy()
        elif "resolution_hours" in hist_df.columns:
            completed = hist_df[hist_df["resolution_hours"].notna()].copy()
        else:
            completed = hist_df.copy()

        if len(completed) == 0 or "assigned_agent_id" not in completed.columns:
            return

        grouped = completed.groupby("assigned_agent_id")
        for agent_id, group in grouped:
            n = len(group)
            br = float(group["resolution_breached"].mean()) if "resolution_breached" in group.columns else 0.0
            mr = float(group["category_mismatch"].mean()) if "category_mismatch" in group.columns else 0.0
            mean_h = float(group["resolution_hours"].mean()) if "resolution_hours" in group.columns else 0.0

            self.agent_hist_stats[int(agent_id)] = {
                "ticket_count": n,
                "breach_rate": round(br, 4),
                "mismatch_rate": round(mr, 4),
                "mean_resolution_hours": round(mean_h, 2),
            }

    def _get_ticket_region(self, ticket: Dict[str, Any]) -> Optional[str]:
        """Extract or resolve merchant region for an unassigned ticket."""
        # Direct key check
        if "merchant_region" in ticket and pd.notna(ticket["merchant_region"]):
            return str(ticket["merchant_region"])
        if "region" in ticket and pd.notna(ticket["region"]):
            return str(ticket["region"])

        # Lookup by merchant_id
        m_id = ticket.get("merchant_id")
        if m_id is not None and m_id in self.merchant_regions:
            return self.merchant_regions[m_id]

        return None

    def score_agent_for_ticket(
        self,
        agent: Union[pd.Series, Dict[str, Any]],
        ticket: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate detailed multi-component match score and justifications for an agent-ticket pair.
        
        Args:
            agent: Agent profile series or dictionary.
            ticket: Ticket attributes dictionary.
            
        Returns:
            Dict[str, Any]: Score breakdown, individual component values, and explanatory reasons.
        """
        agent_id = int(agent["agent_id"])
        agent_name = str(agent["agent_name"])
        agent_tier = str(agent["tier"])
        agent_cat = str(agent["primary_category"])
        agent_region = str(agent["shift_region"])
        agent_eff = float(agent["efficiency_multiplier"])

        ticket_cat = ticket.get("category")
        ticket_prio = ticket.get("priority", "P3")
        ticket_region = self._get_ticket_region(ticket)

        reasons: List[str] = []

        # 1. Category Specialty Match (0.0 to 1.0)
        if pd.isna(ticket_cat) or ticket_cat is None:
            cat_score = 0.0
            reasons.append("Ticket domain category is missing (neutral/mismatched).")
        elif agent_cat == ticket_cat:
            cat_score = 1.0
            reasons.append(f"Domain specialist in {agent_cat} (eliminates the +24.9 pp mismatch penalty).")
        else:
            cat_score = 0.0
            reasons.append(f"Cross-domain assignment: specialist in {agent_cat} vs ticket {ticket_cat}.")

        # 2. Priority-to-Tier Alignment (0.0 to 1.0)
        prio_str = str(ticket_prio).strip().upper() if pd.notna(ticket_prio) else "P3"
        tier_map = TIER_PRIORITY_FIT.get(prio_str, {"L1": 0.70, "L2": 0.85, "L3": 0.80})
        tier_score = tier_map.get(agent_tier, 0.50)

        if prio_str in ("P1", "P2") and agent_tier in ("L2", "L3"):
            reasons.append(f"{agent_tier} senior specialist capability suited for urgent {prio_str} SLA window.")
        elif prio_str in ("P3", "P4") and agent_tier == "L1":
            reasons.append(f"Optimal capacity utilization: L1 specialist matched to standard {prio_str} ticket.")
        elif prio_str in ("P1", "P2") and agent_tier == "L1":
            reasons.append(f"Caution: L1 specialist assigned to high-severity {prio_str} ticket.")

        # 3. Agent Efficiency Factor (0.0 to 1.0)
        eff_range = max(0.01, self.max_eff - self.min_eff)
        eff_score = float(np.clip((agent_eff - self.min_eff) / eff_range, 0.0, 1.0))
        if agent_eff >= 1.15:
            reasons.append(f"High historical velocity rating ({agent_eff:.3f}x efficiency multiplier).")
        elif agent_eff <= 0.85:
            reasons.append(f"Lower efficiency rating ({agent_eff:.3f}x multiplier).")

        # 4. Regional Shift Alignment (0.0 to 1.0)
        if ticket_region is None:
            region_score = 0.70  # Neutral when merchant region is unrecorded
        elif agent_region == ticket_region:
            region_score = 1.00
            reasons.append(f"Working shift ({agent_region}) matches merchant region ({ticket_region}).")
        else:
            region_score = 0.40
            reasons.append(f"Cross-regional shift: agent in {agent_region} vs merchant in {ticket_region}.")

        # 5. Composite Match Score
        w = self.weights
        composite_score = float(
            w["category"] * cat_score +
            w["tier_priority"] * tier_score +
            w["efficiency"] * eff_score +
            w["region"] * region_score
        )
        composite_score = round(composite_score, 4)

        # Historical statistical context (if available)
        hist_info = self.agent_hist_stats.get(agent_id)
        if hist_info and hist_info["ticket_count"] >= 30:
            reasons.append(
                f"Historical baseline: {hist_info['ticket_count']} completed tickets, "
                f"{hist_info['breach_rate']*100:.1f}% breach rate, {hist_info['mean_resolution_hours']:.1f}h avg."
            )

        return {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "tier": agent_tier,
            "primary_category": agent_cat,
            "shift_region": agent_region,
            "efficiency_multiplier": agent_eff,
            "cat_score": cat_score,
            "tier_score": tier_score,
            "eff_score": eff_score,
            "region_score": region_score,
            "composite_score": composite_score,
            "reasons": reasons,
            "hist_stats": hist_info,
        }

    def recommend_for_ticket(
        self,
        ticket: Union[Dict[str, Any], pd.Series],
        top_k: int = 3,
        allow_cross_category: bool = False,
        min_score_threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """Rank eligible agents and produce structured recommendation with reasons and alternatives.
        
        Args:
            ticket: Dict or Series with ticket fields (ticket_id, category, priority, merchant_id).
            top_k: Total number of top candidates to evaluate (1 recommended + alternatives).
            allow_cross_category: Whether to evaluate non-specialist agents if no specialist exists.
            min_score_threshold: Minimum match score required to declare a valid recommendation.
            
        Returns:
            Dict[str, Any]: Structured output adhering to Day 4 specification.
        """
        # Convert Series to dictionary if passed
        if isinstance(ticket, pd.Series):
            ticket_dict = ticket.to_dict()
        else:
            ticket_dict = dict(ticket)

        ticket_id = ticket_dict.get("ticket_id")
        ticket_cat = ticket_dict.get("category")
        ticket_prio = ticket_dict.get("priority", "P3")

        # Standard limitations caveat present across all recommendation queries
        uncertainty_note = (
            "Observational ranking grounded in domain specialty, tier alignment, and directory efficiency. "
            "Real-time agent queue concurrency and live shift availability are unobserved in the dataset and not assumed. "
            "Historical metrics reflect past ticket structures and do not guarantee future resolution outcomes."
        )

        # Edge Case 1: Missing, null, or completely unclassified ticket category
        if pd.isna(ticket_cat) or ticket_cat is None or str(ticket_cat).strip() == "":
            return {
                "ticket_id": ticket_id,
                "recommended_agent_id": None,
                "recommended_agent_name": "No Recommendation (Missing Domain)",
                "recommended_agent_tier": "N/A",
                "recommended_agent_category": "N/A",
                "recommended_agent_shift": "N/A",
                "match_score": 0.0,
                "reasons": [
                    "Ticket domain category is missing or null.",
                    "Cannot assign specialist without domain classification to prevent severe mismatch penalty.",
                ],
                "alternatives": [],
                "candidate_count": 0,
                "uncertainty_limitations": uncertainty_note,
                "status": "NO_MATCH",
            }

        # Filter eligible agent candidates
        specialist_pool = self.agents[self.agents["primary_category"] == str(ticket_cat)].copy()

        if len(specialist_pool) > 0:
            candidate_pool = specialist_pool
        elif allow_cross_category:
            candidate_pool = self.agents.copy()
        else:
            # Edge Case 2: No specialists exist in directory for this category
            return {
                "ticket_id": ticket_id,
                "recommended_agent_id": None,
                "recommended_agent_name": f"No Specialist Available ({ticket_cat})",
                "recommended_agent_tier": "N/A",
                "recommended_agent_category": "N/A",
                "recommended_agent_shift": "N/A",
                "match_score": 0.0,
                "reasons": [
                    f"No support specialists found with primary domain '{ticket_cat}' in agent roster.",
                    "Strict specialty matching enforced; cross-category offloading rejected.",
                ],
                "alternatives": [],
                "candidate_count": 0,
                "uncertainty_limitations": uncertainty_note,
                "status": "NO_MATCH",
            }

        # Score all eligible candidates
        scored_candidates: List[Dict[str, Any]] = []
        for _, agent_row in candidate_pool.iterrows():
            scored = self.score_agent_for_ticket(agent_row, ticket_dict)
            scored_candidates.append(scored)

        # Deterministic sorting:
        # Primary: composite_score (descending)
        # Tie-breaker 1: efficiency_multiplier (descending)
        # Tie-breaker 2: agent_id (ascending)
        scored_candidates.sort(
            key=lambda c: (-c["composite_score"], -c["efficiency_multiplier"], c["agent_id"])
        )

        top_candidate = scored_candidates[0]

        # Edge Case 3: Score fails to satisfy minimum threshold
        if top_candidate["composite_score"] < min_score_threshold:
            return {
                "ticket_id": ticket_id,
                "recommended_agent_id": None,
                "recommended_agent_name": "No Suitable Match",
                "recommended_agent_tier": "N/A",
                "recommended_agent_category": "N/A",
                "recommended_agent_shift": "N/A",
                "match_score": top_candidate["composite_score"],
                "reasons": [f"Highest candidate score ({top_candidate['composite_score']:.2f}) failed minimum threshold ({min_score_threshold:.2f})."],
                "alternatives": [],
                "candidate_count": len(scored_candidates),
                "uncertainty_limitations": uncertainty_note,
                "status": "NO_MATCH",
            }

        # Format alternative candidates
        alternatives: List[Dict[str, Any]] = []
        for alt in scored_candidates[1:top_k]:
            alternatives.append({
                "agent_id": alt["agent_id"],
                "agent_name": alt["agent_name"],
                "tier": alt["tier"],
                "primary_category": alt["primary_category"],
                "shift_region": alt["shift_region"],
                "match_score": alt["composite_score"],
                "primary_reason": alt["reasons"][0] if alt["reasons"] else "Eligible domain specialist",
            })

        return {
            "ticket_id": ticket_id,
            "recommended_agent_id": top_candidate["agent_id"],
            "recommended_agent_name": top_candidate["agent_name"],
            "recommended_agent_tier": top_candidate["tier"],
            "recommended_agent_category": top_candidate["primary_category"],
            "recommended_agent_shift": top_candidate["shift_region"],
            "match_score": top_candidate["composite_score"],
            "reasons": top_candidate["reasons"],
            "alternatives": alternatives,
            "candidate_count": len(scored_candidates),
            "uncertainty_limitations": uncertainty_note,
            "status": "SUCCESS",
        }

    def route_batch(
        self,
        open_tickets_df: pd.DataFrame,
        strategy: str = "top_match",
        top_k_alternatives: int = 3,
    ) -> pd.DataFrame:
        """Route an entire batch of incoming open queue tickets.
        
        Supported Strategies:
        - 'top_match': Greedy best-fit; assigns the single highest scoring candidate to each ticket.
        - 'balanced': Category-preserving round-robin; balances ticket distribution across
          eligible domain specialists so no single specialist receives disproportionate load.
          
        Args:
            open_tickets_df: DataFrame of unassigned tickets (typically is_legacy == False).
            strategy: Routing strategy ('top_match' or 'balanced').
            top_k_alternatives: Number of alternative agents to record.
            
        Returns:
            pd.DataFrame: Open queue augmented with recommended agent, tier, match score,
                          reasons, and alternatives.
        """
        if len(open_tickets_df) == 0:
            return pd.DataFrame(columns=[
                "ticket_id", "category", "sub_category", "priority", "merchant_id",
                "recommended_agent_id", "recommended_agent_name", "recommended_agent_tier",
                "recommended_agent_category", "match_score", "primary_reason", "alternatives", "status"
            ])

        records: List[Dict[str, Any]] = []

        if strategy == "balanced":
            # Track batch assignment counts to balance load among specialists
            batch_load: Dict[int, int] = {int(aid): 0 for aid in self.agents["agent_id"]}

            # Sort tickets by severity (P1 -> P2 -> P3 -> P4) to route high-impact tickets first
            p_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
            df_sorted = open_tickets_df.copy()
            df_sorted["_p_rank"] = df_sorted["priority"].map(p_order).fillna(5)
            df_sorted = df_sorted.sort_values("_p_rank").drop(columns=["_p_rank"])

            for _, row in df_sorted.iterrows():
                t_dict = row.to_dict()
                t_cat = t_dict.get("category")

                # Get eligible specialists
                specialists = self.agents[self.agents["primary_category"] == str(t_cat)].copy()
                if len(specialists) == 0:
                    res = self.recommend_for_ticket(t_dict, top_k=top_k_alternatives)
                else:
                    # Score specialists and adjust score by current batch assignment load
                    scored = []
                    for _, ag in specialists.iterrows():
                        base = self.score_agent_for_ticket(ag, t_dict)
                        current_cnt = batch_load.get(base["agent_id"], 0)
                        # Light penalty for each ticket already assigned in this batch (0.04 per ticket)
                        adjusted_score = max(0.10, base["composite_score"] - (current_cnt * 0.04))
                        base["balanced_score"] = round(adjusted_score, 4)
                        scored.append(base)

                    scored.sort(key=lambda c: (-c["balanced_score"], -c["composite_score"], c["agent_id"]))
                    top_ag = scored[0]
                    batch_load[top_ag["agent_id"]] += 1

                    alts = [{
                        "agent_id": a["agent_id"],
                        "agent_name": a["agent_name"],
                        "tier": a["tier"],
                        "match_score": a["composite_score"],
                    } for a in scored[1:top_k_alternatives]]

                    res = {
                        "ticket_id": t_dict.get("ticket_id"),
                        "recommended_agent_id": top_ag["agent_id"],
                        "recommended_agent_name": top_ag["agent_name"],
                        "recommended_agent_tier": top_ag["tier"],
                        "recommended_agent_category": top_ag["primary_category"],
                        "match_score": top_ag["composite_score"],
                        "reasons": top_ag["reasons"],
                        "alternatives": alts,
                        "status": "SUCCESS",
                        "uncertainty_limitations": "Balanced batch allocation adjusting for within-batch queue distribution.",
                    }

                alt_str = ", ".join([f"{a['agent_name']} ({a['tier']}, {a['match_score']:.2f})" for a in res["alternatives"]])
                records.append({
                    "ticket_id": res["ticket_id"],
                    "category": row.get("category"),
                    "sub_category": row.get("sub_category"),
                    "priority": row.get("priority"),
                    "merchant_id": row.get("merchant_id"),
                    "recommended_agent_id": res["recommended_agent_id"],
                    "recommended_agent_name": res["recommended_agent_name"],
                    "recommended_agent_tier": res["recommended_agent_tier"],
                    "recommended_agent_category": res["recommended_agent_category"],
                    "match_score": res["match_score"],
                    "primary_reason": res["reasons"][0] if res["reasons"] else "",
                    "alternatives": alt_str,
                    "status": res["status"],
                })

            result_df = pd.DataFrame(records)
            # Reorder to match original ticket order
            return open_tickets_df[["ticket_id"]].merge(result_df, on="ticket_id", how="left")

        else:
            # Standard greedy top_match
            for _, row in open_tickets_df.iterrows():
                res = self.recommend_for_ticket(row, top_k=top_k_alternatives)
                alt_str = ", ".join([f"{a['agent_name']} ({a['tier']}, {a['match_score']:.2f})" for a in res["alternatives"]])

                records.append({
                    "ticket_id": res["ticket_id"],
                    "category": row.get("category"),
                    "sub_category": row.get("sub_category"),
                    "priority": row.get("priority"),
                    "merchant_id": row.get("merchant_id"),
                    "recommended_agent_id": res["recommended_agent_id"],
                    "recommended_agent_name": res["recommended_agent_name"],
                    "recommended_agent_tier": res["recommended_agent_tier"],
                    "recommended_agent_category": res["recommended_agent_category"],
                    "match_score": res["match_score"],
                    "primary_reason": res["reasons"][0] if res["reasons"] else "",
                    "alternatives": alt_str,
                    "status": res["status"],
                })

            return pd.DataFrame(records)

    def get_batch_summary(self, routed_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute executive metrics on batch ticket assignments."""
        n_total = len(routed_df)
        if n_total == 0:
            return {
                "total_tickets": 0,
                "matched_count": 0,
                "unmatched_count": 0,
                "match_rate": 0.0,
                "mean_match_score": 0.0,
                "tier_distribution": {},
                "agent_distribution": {},
                "p1_p2_tier_coverage": 0.0,
            }

        matched = routed_df[routed_df["recommended_agent_id"].notna()]
        n_matched = len(matched)
        n_unmatched = n_total - n_matched

        mean_score = float(matched["match_score"].mean()) if n_matched > 0 else 0.0

        tier_counts = matched["recommended_agent_tier"].value_counts().to_dict()
        agent_counts = matched["recommended_agent_name"].value_counts().to_dict()

        # P1 / P2 coverage by L2 / L3 specialists
        p1_p2 = matched[matched["priority"].isin(["P1", "P2"])]
        if len(p1_p2) > 0:
            p1_p2_l2_l3 = p1_p2["recommended_agent_tier"].isin(["L2", "L3"]).sum()
            p1_p2_cov = float(p1_p2_l2_l3 / len(p1_p2))
        else:
            p1_p2_cov = 1.0

        return {
            "total_tickets": n_total,
            "matched_count": n_matched,
            "unmatched_count": n_unmatched,
            "match_rate": round(n_matched / n_total, 4),
            "mean_match_score": round(mean_score, 4),
            "tier_distribution": tier_counts,
            "agent_distribution": agent_counts,
            "p1_p2_tier_coverage": round(p1_p2_cov, 4),
        }
