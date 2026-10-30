"""Agent-first operational command center views for OpsPilot AI.

Clean, simple, action-first interface designed for IT support agents:
- My Workspace: Greeting, Work Summary, Most Urgent Ticket (Assigned vs Claimable),
  My Assigned Tickets, Recommended Opportunities, AI Guidance.
- My Tickets: Personal queue management, robust empty state, instant claim top match.
- Ticket Action Panel: Straightforward ticket management (status, dispatch, escalation, resolution notes).
- Agent Profile: Compact specialist identity switcher and domain capability overview.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

from src.config import SLA_TARGETS
from src.routing.recommender import AgentRecommender
from src.analytics.bottlenecks import BottleneckDetector
from app.components import render_kpi_card


def _compute_ticket_timings(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SLA deadlines, elapsed hours, and remaining hours based on dataset snapshot."""
    df = tickets_df.copy()
    created_dt = pd.to_datetime(df["created_at"])
    snapshot = created_dt.max()

    target_hours = df["priority"].map(
        lambda p: float(SLA_TARGETS.get(p, {}).get("resolution_hours", 24.0))
    )
    deadline_dt = created_dt + pd.to_timedelta(target_hours, unit="h")
    elapsed_hours = (snapshot - created_dt).dt.total_seconds() / 3600.0
    remaining_hours = target_hours - elapsed_hours

    df["sla_target_hours_num"] = target_hours
    df["sla_deadline_dt"] = deadline_dt
    df["sla_deadline_str"] = deadline_dt.dt.strftime("%d %b %H:%M")
    df["elapsed_hours"] = elapsed_hours.round(1)
    df["remaining_hours"] = remaining_hours.round(1)
    df["is_overdue"] = remaining_hours <= 0.0
    df["is_due_soon"] = (remaining_hours > 0.0) & (remaining_hours <= 4.0)

    def format_remaining(row):
        rem = row["remaining_hours"]
        if row["is_overdue"]:
            return f"🚨 Overdue ({abs(rem):.1f}h)"
        elif row["is_due_soon"]:
            return f"⏰ {rem:.1f}h left"
        return f"{rem:.1f}h left"

    df["remaining_display"] = df.apply(format_remaining, axis=1)
    return df


def _build_short_description(row: pd.Series) -> str:
    """Format concise, informative short description for a ticket."""
    cat = row.get("category", "General")
    sub = row.get("sub_category", "Standard")
    merchant = row.get("merchant_name", "Merchant")
    prefix = ""
    if row.get("is_incident_ticket", False):
        prefix = "[Incident] "
    elif row.get("is_reopened", False):
        prefix = "[Reopened] "
    return f"{prefix}{cat}: {sub} ({merchant})"


def explain_specialist_recommendation(
    row: pd.Series,
    active_agent: Dict[str, Any],
) -> Dict[str, Any]:
    """Distinguish active agent from recommended receiving specialist and explain routing rationale.
    
    Grounded in:
    - Primary domain match (eliminates mismatch penalty)
    - Seniority-to-urgency alignment (P1 critical outage escalation to Tier L3)
    - Regional shift coverage
    """
    agent_name = active_agent.get("agent_name", "")
    agent_cat = active_agent.get("primary_category", "")
    agent_tier = active_agent.get("tier", "L2")
    agent_shift = active_agent.get("shift_region", "")

    rec_agent = row.get("recommended_agent", "Unassigned")
    rec_tier = row.get("recommended_tier", "L1")
    rec_cat = row.get("recommended_agent_category", row.get("category", ""))
    rec_shift = row.get("recommended_agent_shift", "")

    prio = str(row.get("priority", "P3")).upper()
    t_cat = str(row.get("category", ""))
    t_region = row.get("region")

    is_active = (rec_agent == agent_name)
    is_escalation = False

    if is_active:
        role_label = f"Optimal Specialist: {agent_name} (You)"
        reasons = [f"Primary domain match in {agent_cat}"]
        if prio in ("P1", "P2") and agent_tier in ("L2", "L3"):
            reasons.append(f"{agent_tier} seniority fit for {prio} urgency")
        if t_region and t_region == agent_shift:
            reasons.append(f"Shift match ({agent_shift})")
        explanation = " • ".join(reasons)
    else:
        reasons = []
        if prio == "P1" and rec_tier == "L3" and agent_tier != "L3":
            role_label = f"Escalation Required: Senior Specialist {rec_agent} ({rec_tier})"
            reasons.append(f"P1 critical outage: requires Tier L3 escalation authority ({rec_agent})")
            is_escalation = True
        elif t_cat != agent_cat:
            role_label = f"Domain Specialist: {rec_agent} ({rec_tier})"
            reasons.append(f"Domain mismatch: Ticket is {t_cat}, requires {t_cat} specialist {rec_agent} (vs your {agent_cat} specialty)")
        elif rec_tier != agent_tier:
            role_label = f"Seniority Alignment: {rec_agent} ({rec_tier})"
            reasons.append(f"{prio} ticket routed to {rec_tier} specialist {rec_agent}")
        else:
            role_label = f"Recommended Specialist: {rec_agent} ({rec_tier})"
            reasons.append(f"Eligible domain specialist in {t_cat}")

        if t_region and rec_shift and t_region == rec_shift and t_region != agent_shift:
            reasons.append(f"Shift alignment ({rec_shift})")

        explanation = " • ".join(reasons)

    return {
        "role_label": role_label,
        "recommended_agent": rec_agent,
        "recommended_tier": rec_tier,
        "is_active_agent": is_active,
        "is_escalation": is_escalation,
        "explanation": explanation,
    }


def _build_action_recommendation(
    row: pd.Series,
    recommended_agent: str,
    recommended_tier: str,
    is_bottleneck: bool,
    active_agent_name: str = "",
) -> str:
    """Construct concise, grounded next-action recommendation without inventing capabilities."""
    prio = str(row.get("priority", "P3")).upper()
    is_overdue = row.get("is_overdue", False)
    risk = row.get("breach_risk", 0.0)

    if prio == "P1" and recommended_tier == "L3" and recommended_agent != active_agent_name:
        return f"🚨 Critical P1 outage: Escalate to Tier L3 specialist {recommended_agent} for senior intervention."
    elif is_overdue:
        return f"🚨 Overdue recovery: Expedite resolution with {recommended_agent} ({recommended_tier})."
    elif risk >= 0.70:
        return f"⚡ Critical breach risk ({row.get('breach_risk_pct', 0.0):.0f}%): Prioritize dispatch to {recommended_agent} ({recommended_tier})."
    elif row.get("is_due_soon", False):
        return f"⏰ Due in {row.get('remaining_hours', 0.0):.1f}h: Expedite resolution with {recommended_agent}."
    elif is_bottleneck:
        return f"⚠️ Known bottleneck ({row.get('sub_category', '')}): Route to domain specialist {recommended_agent}."
    return f"Assign to {recommended_agent} ({recommended_tier} specialist)."


def prepare_agent_workspace_data(
    triage_df: pd.DataFrame,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    agent_recommender: AgentRecommender,
    bottleneck_detector: BottleneckDetector,
) -> pd.DataFrame:
    """Prepare enriched ticket queue data for the agent workspace."""
    df = _compute_ticket_timings(triage_df)

    # Pre-route batch using greedy recommender
    routed_df = agent_recommender.route_batch(df, strategy="top_match")
    route_map = {}
    for _, r in routed_df.iterrows():
        route_map[r["ticket_id"]] = {
            "rec_agent_id": r.get("recommended_agent_id"),
            "rec_agent_name": r.get("recommended_agent_name", "Unassigned"),
            "rec_agent_tier": r.get("recommended_agent_tier", "L1"),
            "rec_agent_category": r.get("recommended_agent_category", ""),
            "rec_agent_shift": r.get("recommended_agent_shift", ""),
            "match_score": r.get("match_score", 0.0),
            "primary_reason": r.get("primary_reason", ""),
        }

    # Identify top bottleneck subcategories
    try:
        ranked_bottlenecks = bottleneck_detector.rank_bottlenecks(min_sample=15)
        bottleneck_subs = set(ranked_bottlenecks["sub_category"].tolist()) if len(ranked_bottlenecks) else set()
    except Exception:
        bottleneck_subs = set()

    descriptions = []
    rec_agents = []
    rec_tiers = []
    rec_cats = []
    rec_shifts = []
    rec_actions = []
    risk_labels = []
    is_bn_list = []

    for _, row in df.iterrows():
        tid = row["ticket_id"]
        desc = _build_short_description(row)
        descriptions.append(desc)

        rm = route_map.get(tid, {})
        ragent = rm.get("rec_agent_name", "Unassigned")
        rtier = rm.get("rec_agent_tier", "L1")
        rcat = rm.get("rec_agent_category", row.get("category", ""))
        rshift = rm.get("rec_agent_shift", "")

        rec_agents.append(ragent)
        rec_tiers.append(rtier)
        rec_cats.append(rcat)
        rec_shifts.append(rshift)

        is_bn = row.get("sub_category") in bottleneck_subs
        is_bn_list.append(is_bn)

        action = _build_action_recommendation(row, ragent, rtier, is_bn)
        rec_actions.append(action)

        risk = row.get("breach_risk", 0.0)
        if risk >= 0.70:
            risk_labels.append("Critical (≥70%)")
        elif risk >= 0.40:
            risk_labels.append("At Risk (40-70%)")
        else:
            risk_labels.append("Low (<40%)")

    df["short_description"] = descriptions
    df["recommended_agent"] = rec_agents
    df["recommended_tier"] = rec_tiers
    df["recommended_agent_category"] = rec_cats
    df["recommended_agent_shift"] = rec_shifts
    df["is_bottleneck"] = is_bn_list
    df["recommended_action"] = rec_actions
    df["risk_label"] = risk_labels

    return df


def get_active_agent(agents: pd.DataFrame) -> Dict[str, Any]:
    """Retrieve active agent dictionary from session state, defaulting to Priya Sharma or first agent."""
    if "active_agent_name" not in st.session_state:
        priya = agents[agents["agent_name"].str.contains("Priya", case=False)]
        if len(priya) > 0:
            st.session_state["active_agent_name"] = priya.iloc[0]["agent_name"]
        else:
            st.session_state["active_agent_name"] = agents.iloc[0]["agent_name"]

    name = st.session_state["active_agent_name"]
    matches = agents[agents["agent_name"] == name]
    if len(matches) > 0:
        return matches.iloc[0].to_dict()
    return agents.iloc[0].to_dict()


def get_most_urgent_ticket(
    queue_df: pd.DataFrame,
    active_agent: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Identify the single most urgent actionable ticket dynamically.
    
    CRITICAL WORKFLOW GUARANTEES:
    1. If the active agent HAS open assigned tickets:
       Select the most urgent actionable ticket from their assigned queue.
       Tagged with is_assigned_to_me = True, assignment_status = "Assigned to You".
    2. If the active agent has NO assigned tickets:
       Do NOT misleadingly label an unassigned ticket as already assigned.
       Find the most urgent eligible UNASSIGNED ticket available to claim.
       Tagged with is_assigned_to_me = False, assignment_status = "Unassigned (Available to Claim)".
    3. If no tickets exist, return None.
    """
    if len(queue_df) == 0:
        return None

    assigned = st.session_state.get("ticket_assignments", {})
    statuses = st.session_state.get("ticket_statuses", {})
    agent_name = active_agent.get("agent_name", "")
    agent_cat = active_agent.get("primary_category", "")

    prio_map = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    # 1. Check active agent's own open assigned/claimed tickets
    my_open_tids = [
        tid for tid, assigned_to in assigned.items()
        if assigned_to == agent_name and statuses.get(tid, "Open") != "Resolved"
    ]
    if my_open_tids:
        my_tickets_df = queue_df[queue_df["ticket_id"].isin(my_open_tids)].copy()
        if len(my_tickets_df) > 0:
            my_tickets_df["_prio_rank"] = my_tickets_df["priority"].map(prio_map)
            sorted_my = my_tickets_df.sort_values(
                by=["is_overdue", "_prio_rank", "breach_risk", "remaining_hours"],
                ascending=[False, True, False, True],
            )
            res = sorted_my.iloc[0].to_dict()
            res["is_assigned_to_me"] = True
            res["assignment_status"] = "Assigned to You"
            return res

    # 2. Agent has zero assigned tickets: find most urgent UNASSIGNED ticket available to claim
    claimed_tids = set(assigned.keys())
    unassigned_df = queue_df[~queue_df["ticket_id"].isin(claimed_tids)].copy()

    if len(unassigned_df) == 0:
        return None

    # Prefer unassigned tickets matching agent's primary domain specialty
    domain_unassigned = unassigned_df[unassigned_df["category"] == agent_cat].copy()
    if len(domain_unassigned) > 0:
        domain_unassigned["_prio_rank"] = domain_unassigned["priority"].map(prio_map)
        sorted_domain = domain_unassigned.sort_values(
            by=["is_overdue", "_prio_rank", "breach_risk", "remaining_hours"],
            ascending=[False, True, False, True],
        )
        res = sorted_domain.iloc[0].to_dict()
        res["is_assigned_to_me"] = False
        res["assignment_status"] = "Unassigned (Available to Claim)"
        return res

    # Fall back to queue-wide unassigned tickets
    unassigned_df["_prio_rank"] = unassigned_df["priority"].map(prio_map)
    sorted_all = unassigned_df.sort_values(
        by=["is_overdue", "_prio_rank", "breach_risk", "remaining_hours"],
        ascending=[False, True, False, True],
    )
    res = sorted_all.iloc[0].to_dict()
    res["is_assigned_to_me"] = False
    res["assignment_status"] = "Unassigned (Available to Claim)"
    return res


def get_recommended_tickets(
    queue_df: pd.DataFrame,
    active_agent: Dict[str, Any],
    limit: int = 3,
    exclude_ticket_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top matching eligible unassigned tickets for active agent.
    
    Strict guarantees:
    - Only unassigned tickets (excludes tickets assigned to active agent or claimed by any other agent).
    - Excludes exclude_ticket_id (prevents duplicate appearance with urgent ticket).
    - Ranks by domain match, priority urgency, and breach risk.
    """
    assigned = st.session_state.get("ticket_assignments", {})
    agent_cat = active_agent.get("primary_category", "")
    agent_tier = active_agent.get("tier", "L2")

    # Strictly unassigned tickets: exclude all claimed/assigned ticket IDs
    claimed_tids = set(assigned.keys())
    eligible_df = queue_df[~queue_df["ticket_id"].isin(claimed_tids)].copy()

    # Exclude the urgent ticket ID if provided
    if exclude_ticket_id is not None:
        eligible_df = eligible_df[eligible_df["ticket_id"] != exclude_ticket_id]

    if len(eligible_df) == 0:
        return []

    # Filter to agent's domain specialty first
    matching_df = eligible_df[eligible_df["category"] == agent_cat].copy()
    if len(matching_df) == 0:
        matching_df = eligible_df.copy()

    prio_map = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    matching_df["_prio_rank"] = matching_df["priority"].map(prio_map)
    sorted_matching = matching_df.sort_values(
        by=["_prio_rank", "breach_risk", "remaining_hours"],
        ascending=[True, False, True],
    ).head(limit)

    results = []
    for _, row in sorted_matching.iterrows():
        rec_reasons = []
        if row["category"] == agent_cat:
            rec_reasons.append(f"Domain match: {agent_cat} specialist")
        else:
            rec_reasons.append(f"Intake domain: {row['category']}")

        if row["priority"] in ("P1", "P2") and agent_tier in ("L2", "L3"):
            rec_reasons.append(f"Seniority fit: {agent_tier} for {row['priority']} urgency")
        elif row["priority"] in ("P3", "P4") and agent_tier == "L1":
            rec_reasons.append(f"Capacity fit: L1 for standard {row['priority']}")

        if row.get("region") == active_agent.get("shift_region"):
            rec_reasons.append(f"Shift match: {active_agent.get('shift_region')}")

        results.append({
            "ticket_id": row["ticket_id"],
            "category": row["category"],
            "sub_category": row["sub_category"],
            "short_description": row["short_description"],
            "priority": row["priority"],
            "remaining_display": row["remaining_display"],
            "remaining_hours": row["remaining_hours"],
            "breach_risk_pct": row["breach_risk_pct"],
            "risk_label": row["risk_label"],
            "breach_risk": row["breach_risk"],
            "recommended_agent": row.get("recommended_agent", "Unassigned"),
            "recommended_tier": row.get("recommended_tier", "L1"),
            "merchant_name": row.get("merchant_name", "Merchant"),
            "why_recommended": " • ".join(rec_reasons),
        })
    return results


def row_agent_rec(df: pd.DataFrame, tid: int) -> str:
    """Helper to get recommended agent name for a ticket."""
    m = df[df["ticket_id"] == tid]
    if len(m) > 0:
        return m.iloc[0].get("recommended_agent", "Unassigned")
    return "Unassigned"


@st.dialog("Ticket Details & Diagnostic", width="large")
def show_inspect_ticket_dialog(ticket_data: Dict[str, Any], active_agent: Dict[str, Any]):
    """Modal dialog displaying comprehensive, read-only ticket details without auto-claiming."""
    tid = int(ticket_data.get("ticket_id"))
    prio = str(ticket_data.get("priority", "P3"))
    cat = str(ticket_data.get("category", "General"))
    subcat = str(ticket_data.get("sub_category", "Standard"))
    m_name = str(ticket_data.get("merchant_name", "Merchant"))
    m_tier = str(ticket_data.get("tier", "Foundation"))
    risk_pct = float(ticket_data.get("breach_risk_pct", 0.0))
    risk_val = float(ticket_data.get("breach_risk", 0.0))
    rem_disp = str(ticket_data.get("remaining_display", ""))
    deadline_str = str(ticket_data.get("sla_deadline_str", ""))
    target_hours = float(ticket_data.get("sla_target_hours_num", 24.0))
    short_desc = str(ticket_data.get("short_description", f"{cat}: {subcat}"))
    rec_action = str(ticket_data.get("recommended_action", "Assign to domain specialist."))

    spec_explain = explain_specialist_recommendation(pd.Series(ticket_data), active_agent)

    assigned = st.session_state.get("ticket_assignments", {})
    curr_owner = assigned.get(tid)
    is_claimed = curr_owner is not None and curr_owner not in ("", "Unassigned")

    risk_class = "kpi-badge-red" if risk_val >= 0.70 else ("kpi-badge-amber" if risk_val >= 0.40 else "kpi-badge-green")

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E5DFE9; padding-bottom: 10px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.4rem; font-weight: 800; color: #120A18;">Ticket #{tid}</span>
                <span class="kpi-badge-blue">Priority {prio}</span>
                <span class="{risk_class}">{risk_pct:.1f}% Breach Risk</span>
            </div>
            <div>
                <span class="kpi-badge-{'blue' if is_claimed else 'amber'}">
                    {'Assigned: ' + curr_owner if is_claimed else 'Unassigned Intake'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_overdue = bool(ticket_data.get("is_overdue", False))
    is_due_soon = bool(ticket_data.get("is_due_soon", False))
    rem_hours = float(ticket_data.get("remaining_hours", 0.0))

    if is_overdue:
        banner_bg, banner_border, banner_color = "#FDF0F1", "#F5BFC3", "#A32832"
        banner_text = f"🚨 SLA Deadline Breached ({abs(rem_hours):.1f}h overdue)"
    elif is_due_soon:
        banner_bg, banner_border, banner_color = "#FEF6EB", "#F8D8A7", "#8A5310"
        banner_text = f"⏰ SLA Deadline Imminent ({rem_hours:.1f}h remaining)"
    else:
        banner_bg, banner_border, banner_color = "#EBF7F0", "#B0E2C7", "#16603E"
        banner_text = f"🟢 SLA On Schedule ({rem_hours:.1f}h remaining)"

    st.markdown(
        f"""
        <div style="background-color: {banner_bg}; border: 1px solid {banner_border}; border-radius: 9px; padding: 10px 14px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; font-weight: 800; color: {banner_color};">
            <div>{banner_text}</div>
            <div style="font-size: 0.825rem; font-weight: 700;">Deadline: <b>{deadline_str}</b> (Target: {target_hours:.0f}h)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 14px; margin-bottom: 14px;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #5A5262; text-transform: uppercase;">Ticket Description</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin: 2px 0 8px 0;">{short_desc}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                <div><b>Category:</b> {cat} &rsaquo; {subcat}</div>
                <div><b>Merchant:</b> {m_name} ({m_tier} Tier)</div>
            </div>
        </div>
        <div style="background-color: #FFFFFF; border: 1px solid #E5DFE9; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; font-size: 0.875rem;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #5A5262; text-transform: uppercase; margin-bottom: 4px;">Specialist Routing Diagnostics</div>
            <div style="color: #120A18; margin-bottom: 4px;"><b>Match:</b> {spec_explain['role_label']}</div>
            <div style="color: #5A5262; font-size: 0.825rem;">{spec_explain['explanation']}</div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #F0EAF5; color: #120A18;">
                <b>Recommended Next Action:</b> {rec_action}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if not is_claimed:
            if st.button("📥 Claim Ticket Now", key=f"dialog_claim_{tid}", use_container_width=True, type="primary"):
                st.session_state["ticket_assignments"][tid] = active_agent["agent_name"]
                st.session_state["ticket_statuses"][tid] = "In Progress"
                st.session_state["active_ticket_id"] = tid
                st.session_state["my_tickets_selector"] = tid
                st.toast(f"Claimed Ticket #{tid}!", icon="✅")
                st.rerun()
    with c2:
        if st.button("Close", key=f"dialog_close_{tid}", use_container_width=True):
            st.rerun()


def render_my_workspace(
    queue_df: pd.DataFrame,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    sla_model: Any,
    agent_recommender: AgentRecommender,
    bottleneck_detector: BottleneckDetector,
    assignment_model: Optional[Any] = None,
):
    """Render My Workspace (Main Landing Page) in strict accordance with the action-first hierarchy."""
    active_agent = get_active_agent(agents)
    agent_name = active_agent["agent_name"]
    agent_tier = active_agent["tier"]
    agent_cat = active_agent["primary_category"]
    agent_shift = active_agent["shift_region"]

    # Session State initializations
    st.session_state.setdefault("ticket_statuses", {})
    st.session_state.setdefault("ticket_assignments", {})
    st.session_state.setdefault("ticket_notes", {})
    st.session_state.setdefault("active_ticket_id", None)

    assigned = st.session_state["ticket_assignments"]
    statuses = st.session_state["ticket_statuses"]

    # Compute personal work summary
    my_assigned_tids = [
        tid for tid, assigned_to in assigned.items()
        if assigned_to == agent_name and statuses.get(tid, "Open") != "Resolved"
    ]
    my_assigned_count = len(my_assigned_tids)
    crit_mask = ((queue_df["breach_risk"] >= 0.70) | (queue_df["priority"] == "P1")) & (queue_df["category"] == agent_cat)
    critical_domain_count = int(crit_mask.sum())
    due_soon_count = int((queue_df["is_due_soon"] | queue_df["is_overdue"]).sum())

    # -------------------------------------------------------------------------
    # 1. GREETING & COMPACT WORK SUMMARY
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="agent-summary">
            <div class="agent-summary-content">
                <div>
                    <div class="agent-summary-title">
                        <span>👋</span> Welcome back, {agent_name}
                    </div>
                    <div class="agent-summary-subtitle">
                        <b>Tier {agent_tier} Specialist</b> • Domain: <b>{agent_cat}</b> • Shift: <b>{agent_shift}</b>
                    </div>
                </div>
                <div class="agent-summary-stats">
                    <span class="agent-summary-stat">
                        📋 My Assigned: <b>{my_assigned_count}</b>
                    </span>
                    <span class="agent-summary-stat agent-summary-stat-critical">
                        🚨 Critical in Domain: <b>{critical_domain_count}</b>
                    </span>
                    <span class="agent-summary-stat agent-summary-stat-due">
                        ⏰ Due Soon / Overdue: <b>{due_soon_count}</b>
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. MOST URGENT TICKET (ASSIGNED OR CLAIMABLE)
    # -------------------------------------------------------------------------
    urgent_ticket = get_most_urgent_ticket(queue_df, active_agent)
    urgent_tid = urgent_ticket["ticket_id"] if urgent_ticket is not None else None

    if urgent_ticket is not None:
        u_tid = urgent_ticket["ticket_id"]
        u_desc = urgent_ticket["short_description"]
        u_prio = urgent_ticket["priority"]
        u_rem_disp = urgent_ticket["remaining_display"]
        u_risk_pct = urgent_ticket["breach_risk_pct"]
        u_action = urgent_ticket["recommended_action"]
        u_merchant = urgent_ticket.get("merchant_name", "Merchant")
        is_assigned_to_me = urgent_ticket.get("is_assigned_to_me", False)

        # Build specialist explanation
        spec_explain = explain_specialist_recommendation(pd.Series(urgent_ticket), active_agent)

        if is_assigned_to_me:
            header_title = "⚡ YOUR MOST URGENT ASSIGNED TICKET"
            badge_html = f'<span class="kpi-badge-blue">📋 Assigned to You • {statuses.get(u_tid, "In Progress")}</span>'
            status_desc = f"<b>Assigned Specialist:</b> {agent_name} (You) &nbsp;|&nbsp; <b>SLA Status:</b> <span style='font-weight: 800; color: {'#A32832' if urgent_ticket['is_overdue'] else '#8A5310'};'>{u_rem_disp}</span>"
        else:
            header_title = "⚡ MOST URGENT AVAILABLE TICKET TO CLAIM"
            badge_html = '<span class="kpi-badge-amber">📥 Unassigned Intake • Available to Claim</span>'
            status_desc = f"<b>Status:</b> Unassigned Intake &nbsp;|&nbsp; <b>SLA Status:</b> <span style='font-weight: 800; color: {'#A32832' if urgent_ticket['is_overdue'] else '#8A5310'};'>{u_rem_disp}</span>"

        st.markdown(
            f"""
            <div class="agent-section-header">
                <div class="agent-section-title">
                    {header_title}
                </div>
                {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True, key="agent-urgent-ticket"):
            h_col1, h_col2 = st.columns([2.15, 1], gap="large", vertical_alignment="center")
            with h_col1:
                st.markdown(
                    f"""
                    <div class="urgent-ticket-meta">
                        <span class="urgent-ticket-id">Ticket #{u_tid}</span>
                        <span class="kpi-badge-blue">Priority {u_prio}</span>
                        <span class="kpi-badge-{'red' if u_risk_pct >= 70 else 'amber'}">{u_risk_pct:.1f}% Breach Risk</span>
                        <span style="font-size: 0.85rem; font-weight: 600; color: #4A4252; margin-left: 2px;">• {u_merchant}</span>
                    </div>
                    <div class="urgent-ticket-description">
                        {u_desc}
                    </div>
                    <div class="urgent-ticket-status">
                        {status_desc}
                    </div>
                    <div class="urgent-ticket-detail">
                        <b>Specialist Routing:</b> {spec_explain['role_label']} • {spec_explain['explanation']}
                    </div>
                    <div class="urgent-ticket-detail urgent-ticket-action">
                        <b>Recommended Next Action:</b> {u_action}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with h_col2:
                if is_assigned_to_me:
                    if st.button("👉 VIEW & TAKE ACTION", key="btn_take_action_urgent", use_container_width=True, type="primary"):
                        st.session_state["active_ticket_id"] = u_tid
                        st.session_state["my_tickets_selector"] = u_tid
                        st.session_state["scroll_to_action_panel"] = True
                        st.rerun()
                else:
                    # Unassigned ticket: provide enabled Claim and Inspect buttons
                    claim_urgent_clicked = st.button("📥 Claim Ticket", key=f"claim_urgent_{u_tid}", use_container_width=True, type="primary")
                    if claim_urgent_clicked:
                        # Collision check
                        curr_owner = st.session_state["ticket_assignments"].get(u_tid)
                        if curr_owner and curr_owner != agent_name:
                            st.error(f"Ticket #{u_tid} has already been claimed by {curr_owner}.")
                        else:
                            st.session_state["ticket_assignments"][u_tid] = agent_name
                            st.session_state["ticket_statuses"][u_tid] = "In Progress"
                            st.session_state["active_ticket_id"] = u_tid
                            st.session_state["my_tickets_selector"] = u_tid
                            st.toast(f"Claimed Ticket #{u_tid}! Added to My Assigned Tickets.", icon="✅")
                            st.rerun()

                    if st.button("🔍 Inspect Details", key=f"inspect_urgent_{u_tid}", use_container_width=True):
                        show_inspect_ticket_dialog(urgent_ticket, active_agent)
    else:
        st.info("🎉 All caught up! No urgent tickets pending in your queue.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. MY ASSIGNED TICKETS (ONLY when active agent has assigned tickets)
    # -------------------------------------------------------------------------
    if my_assigned_count > 0:
        st.markdown(
            f"""
            <div class="agent-section-header agent-recommendations-header">
                <div class="agent-section-title">
                    📋 MY ASSIGNED TICKETS ({my_assigned_count})
                </div>
                <div class="agent-section-note">
                    Active tickets assigned to you
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        my_active_df = queue_df[queue_df["ticket_id"].isin(my_assigned_tids)].copy()
        a_cols = st.columns(min(len(my_active_df), 3), gap="large", vertical_alignment="top")
        for i, (_, a_row) in enumerate(my_active_df.head(3).iterrows()):
            with a_cols[i]:
                with st.container(border=True, key=f"agent-assigned-{a_row['ticket_id']}"):
                    a_tid = a_row["ticket_id"]
                    a_risk = a_row["breach_risk_pct"]
                    a_curr_stat = statuses.get(a_tid, "In Progress")
                    st.markdown(
                        f"""
                        <div class="recommendation-meta">
                            <span class="recommendation-ticket-id">#{a_tid}</span>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span class="kpi-badge-blue">{a_row['priority']}</span>
                                <span class="kpi-badge-{'red' if a_risk >= 70 else ('amber' if a_risk >= 40 else 'green')}">
                                    {a_risk:.0f}% Risk
                                </span>
                            </div>
                        </div>
                        <div class="recommendation-description">
                            {a_row['short_description']}
                        </div>
                        <div class="recommendation-sla">
                            SLA: <b>{a_row['remaining_display']}</b>
                        </div>
                        <div style="font-size: 0.825rem; font-weight: 700; color: #120A18; background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 8px; padding: 7px 12px; margin-bottom: 0;">
                            Status: <b>{a_curr_stat}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("Manage Ticket", key=f"btn_manage_my_{a_tid}", use_container_width=True):
                        st.session_state["active_ticket_id"] = a_tid
                        st.session_state["my_tickets_selector"] = a_tid
                        st.session_state["scroll_to_action_panel"] = True
                        st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. RECOMMENDED FOR YOU (UNASSIGNED QUEUE - EXCLUDING URGENT TICKET)
    # -------------------------------------------------------------------------
    @st.fragment(run_every=30)
    def render_recommended_section_fragment():
        now_str = datetime.now().strftime("%H:%M:%S")
        f_active_agent = get_active_agent(agents)
        f_urgent = get_most_urgent_ticket(queue_df, f_active_agent)
        f_urgent_tid = f_urgent["ticket_id"] if f_urgent is not None else None

        f_rec_tickets = get_recommended_tickets(
            queue_df,
            f_active_agent,
            limit=3,
            exclude_ticket_id=f_urgent_tid,
        )

        r_hdr_col1, r_hdr_col2 = st.columns([2.5, 1.2], vertical_alignment="center")
        with r_hdr_col1:
            st.markdown(
                """
                <div class="agent-section-header" style="margin-top: 26px; margin-bottom: 8px;">
                    <div class="agent-section-title">
                        🎯 RECOMMENDED FOR YOU (Unassigned Opportunities)
                    </div>
                    <div class="agent-section-note">
                        Domain-matched opportunities ready to claim
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_hdr_col2:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin-top: 22px;">
                    <span style="font-size: 0.775rem; font-weight: 600; color: #817889;">
                        Updated: <b>{now_str}</b>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_space, c_ref_btn = st.columns([1.2, 1.0])
            with c_ref_btn:
                st.button("🔄 Refresh", key="btn_manual_refresh_recs", use_container_width=True, help="Immediately refresh recommendation queue")

        if f_rec_tickets:
            r_cols = st.columns(len(f_rec_tickets), gap="large", vertical_alignment="top")
            for i, r_item in enumerate(f_rec_tickets):
                with r_cols[i]:
                    with st.container(border=True, key=f"agent-recommendation-{r_item['ticket_id']}"):
                        r_tid = r_item["ticket_id"]
                        r_risk = r_item["breach_risk_pct"]
                        st.markdown(
                            f"""
                            <div class="recommendation-meta">
                                <span class="recommendation-ticket-id">#{r_tid}</span>
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <span class="kpi-badge-blue">{r_item['priority']}</span>
                                    <span class="kpi-badge-{'red' if r_risk >= 70 else ('amber' if r_risk >= 40 else 'green')}">
                                        {r_risk:.0f}% Risk
                                    </span>
                                </div>
                            </div>
                            <div class="recommendation-description">
                                {r_item['short_description']}
                            </div>
                            <div class="recommendation-sla">
                                SLA: <b>{r_item['remaining_display']}</b>
                            </div>
                            <div class="recommendation-reason">
                                💡 {r_item['why_recommended']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        c_btn1, c_btn2 = st.columns([1, 1], gap="small")
                        with c_btn1:
                            claim_clicked = st.button("Claim Ticket", key=f"claim_{r_tid}", use_container_width=True, type="primary")
                            if claim_clicked:
                                # Collision check
                                curr_owner = st.session_state["ticket_assignments"].get(r_tid)
                                if curr_owner and curr_owner != f_active_agent["agent_name"]:
                                    st.error(f"Ticket #{r_tid} has already been claimed by {curr_owner}.")
                                else:
                                    st.session_state["ticket_assignments"][r_tid] = f_active_agent["agent_name"]
                                    st.session_state["ticket_statuses"][r_tid] = "In Progress"
                                    st.session_state["active_ticket_id"] = r_tid
                                    st.session_state["my_tickets_selector"] = r_tid
                                    st.toast(f"Claimed Ticket #{r_tid}! Added to My Assigned Tickets.", icon="✅")
                                    st.rerun()
                        with c_btn2:
                            if st.button("Inspect", key=f"inspect_{r_tid}", use_container_width=True):
                                show_inspect_ticket_dialog(r_item, f_active_agent)
        else:
            st.info("No additional unassigned tickets currently in your domain.")

    render_recommended_section_fragment()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. AI GUIDANCE (Concise, grounded operational recommendation)
    # -------------------------------------------------------------------------
    if urgent_ticket is not None:
        u_prio = urgent_ticket["priority"]
        u_rem = urgent_ticket["remaining_hours"]
        u_risk = urgent_ticket["breach_risk_pct"]
        u_cat = urgent_ticket["category"]

        if urgent_ticket["is_overdue"]:
            guidance_msg = (
                f"🚨 <b>Immediate Recovery Action:</b> Ticket #{urgent_ticket['ticket_id']} ({u_prio}, {u_cat}) has passed its SLA deadline. "
                f"Expedite resolution workflow immediately."
            )
            g_bg, g_border, g_color = "#EFF6FF", "#93C5FD", "#1E40AF"
        elif u_risk >= 70.0:
            guidance_msg = (
                f"⚡ <b>Priority Action:</b> Ticket #{urgent_ticket['ticket_id']} ({u_prio}, {u_cat}) has a high predicted breach risk ({u_risk:.1f}%). "
                f"Prioritize resolution to prevent SLA breach."
            )
            g_bg, g_border, g_color = "#EFF6FF", "#93C5FD", "#1E40AF"
        else:
            guidance_msg = (
                f"⏰ <b>Monitor SLA Window:</b> Ticket #{urgent_ticket['ticket_id']} has {u_rem:.1f}h remaining before SLA expiration. "
                f"Review intake diagnostics and proceed with resolution."
            )
            g_bg, g_border, g_color = "#EFF6FF", "#93C5FD", "#1E40AF"

        st.markdown(
            f"""
            <div class="agent-guidance" style="--guidance-bg: {g_bg}; --guidance-border: {g_border}; --guidance-color: {g_color};">
                <span style="font-size: 1.1rem;">💡</span>
                <div>{guidance_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # 6. TICKET ACTION PANEL (When active ticket is selected)
    # -------------------------------------------------------------------------
    active_tid = st.session_state.get("active_ticket_id")
    if active_tid is not None and len(queue_df[queue_df["ticket_id"] == active_tid]) > 0:
        st.markdown("<div id='ticket-action-panel'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E5DFE9;'>", unsafe_allow_html=True)
        active_row = queue_df[queue_df["ticket_id"] == active_tid].iloc[0]
        render_ticket_action_panel(active_row, agents, merchants, agent_recommender)
        if st.session_state.get("scroll_to_action_panel"):
            st.session_state["scroll_to_action_panel"] = False
            st.html(
                """
                <script>
                (function() {
                    const scrollToAction = () => {
                        const el = document.getElementById('ticket-action-panel') || (window.parent && window.parent.document.getElementById('ticket-action-panel'));
                        if (el) {
                            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    };
                    scrollToAction();
                    requestAnimationFrame(scrollToAction);
                    setTimeout(scrollToAction, 100);
                })();
                </script>
                """
            )


def render_my_tickets(
    queue_df: pd.DataFrame,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    agent_recommender: AgentRecommender,
):
    """Render My Tickets: Personal queue management, robust empty state, instant claim top match."""
    active_agent = get_active_agent(agents)
    agent_name = active_agent["agent_name"]

    st.session_state.setdefault("ticket_statuses", {})
    st.session_state.setdefault("ticket_assignments", {})
    st.session_state.setdefault("ticket_notes", {})
    st.session_state.setdefault("active_ticket_id", None)

    assigned = st.session_state["ticket_assignments"]
    statuses = st.session_state["ticket_statuses"]

    st.markdown(
        f"""
        <div class="brand-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h2 class="brand-title" style="display: flex; align-items: center; gap: 8px;">
                    <span>📋</span> My Assigned Tickets
                </h2>
                <p class="brand-desc">Personal queue management for <b>{agent_name}</b> ({active_agent['tier']}, {active_agent['primary_category']}).</p>
            </div>
            <div>
                <span class="kpi-badge-blue">{active_agent['shift_region']} Shift</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check top eligible unassigned ticket for claim action
    top_recs = get_recommended_tickets(queue_df, active_agent, limit=1)
    has_claimable = len(top_recs) > 0

    # Top Toolbar
    with st.container(border=True):
        col_s1, col_s2, col_s3 = st.columns([2.0, 1.2, 1.0], vertical_alignment="center")
        with col_s1:
            my_search = st.text_input(
                "Search My Tickets",
                placeholder="Search ticket ID, description, merchant...",
                key="my_tickets_search_input",
                label_visibility="collapsed",
            )
        with col_s2:
            status_filter = st.selectbox(
                "Filter Status",
                ["All", "Critical", "At Risk", "Due Soon", "Resolved"],
                key="my_tickets_status_filter",
                label_visibility="collapsed",
            )
        with col_s3:
            if has_claimable:
                claim_top_clicked = st.button("📥 Claim Top Match", key="btn_quick_claim_top", use_container_width=True, type="primary")
                if claim_top_clicked:
                    claim_tid = top_recs[0]["ticket_id"]
                    st.session_state["ticket_assignments"][claim_tid] = agent_name
                    st.session_state["ticket_statuses"][claim_tid] = "In Progress"
                    st.session_state["active_ticket_id"] = claim_tid
                    st.toast(f"Claimed Ticket #{claim_tid}!", icon="✅")
                    st.rerun()
            else:
                st.button("📥 Claim Top Match", key="btn_quick_claim_top", use_container_width=True, disabled=True, help="No eligible unassigned tickets available to claim.")

    # Get tickets strictly assigned to this active agent
    my_tids = [
        tid for tid, assigned_to in assigned.items()
        if assigned_to == agent_name
    ]

    my_df = queue_df[queue_df["ticket_id"].isin(my_tids)].copy()

    # Synchronize live status & notes
    if len(my_df) > 0:
        my_df["current_status"] = my_df["ticket_id"].map(lambda tid: statuses.get(tid, "Open"))
    else:
        my_df["current_status"] = pd.Series(dtype=str)

    # If active agent has ZERO assigned tickets across all filters: render clean, actionable empty state
    if len(my_tids) == 0:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1.5px dashed #D8CEE2; border-radius: 14px; padding: 36px 24px; text-align: center; margin: 16px 0;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">📭</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #120A18; margin-bottom: 6px;">
                    No tickets assigned yet.
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #4A4252; max-width: 520px; margin: 0 auto 16px auto;">
                    Browse your recommended tickets and claim your next task to start resolving issues in your domain.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if has_claimable:
            rec_tid = top_recs[0]["ticket_id"]
            rec_prio = top_recs[0]["priority"]
            rec_cat = top_recs[0]["category"]
            c_empty_col1, c_empty_col2, c_empty_col3 = st.columns([1.0, 2.0, 1.0])
            with c_empty_col2:
                if st.button(f"📥 Claim Top Match: Ticket #{rec_tid} ({rec_prio}, {rec_cat})", key="empty_state_claim_btn", use_container_width=True, type="primary"):
                    st.session_state["ticket_assignments"][rec_tid] = agent_name
                    st.session_state["ticket_statuses"][rec_tid] = "In Progress"
                    st.session_state["active_ticket_id"] = rec_tid
                    st.toast(f"Claimed Ticket #{rec_tid}!", icon="✅")
                    st.rerun()
        return

    # Apply search filter
    if my_search.strip():
        q = my_search.strip().lower()
        my_df = my_df[
            my_df["ticket_id"].astype(str).str.contains(q, case=False)
            | my_df["category"].astype(str).str.contains(q, case=False)
            | my_df["sub_category"].astype(str).str.contains(q, case=False)
            | my_df["merchant_name"].astype(str).str.contains(q, case=False)
            | my_df["short_description"].astype(str).str.contains(q, case=False)
        ]

    # Apply status filter
    if status_filter == "Critical":
        my_df = my_df[(my_df["breach_risk"] >= 0.70) | (my_df["priority"] == "P1")]
    elif status_filter == "At Risk":
        my_df = my_df[my_df["breach_risk"] >= 0.40]
    elif status_filter == "Due Soon":
        my_df = my_df[my_df["is_due_soon"] | my_df["is_overdue"]]
    elif status_filter == "Resolved":
        my_df = my_df[my_df["current_status"] == "Resolved"]
    elif status_filter == "All":
        pass

    if len(my_df) == 0:
        st.info(f"No assigned tickets match the '{status_filter}' filter criteria.")
    else:
        st.markdown(
            f"<div style='font-size: 0.85rem; font-weight: 800; color: #4A4252; margin-bottom: 4px;'>"
            f"SHOWING {len(my_df)} TICKETS ASSIGNED TO {agent_name.upper()}"
            f"</div>",
            unsafe_allow_html=True,
        )

        display_df = my_df[[
            "ticket_id",
            "priority",
            "short_description",
            "remaining_display",
            "breach_risk_pct",
            "risk_label",
            "current_status",
            "recommended_action",
        ]].rename(columns={
            "ticket_id": "Ticket ID",
            "priority": "Priority",
            "short_description": "Description",
            "remaining_display": "SLA Remaining",
            "breach_risk_pct": "Breach Risk (%)",
            "risk_label": "Risk Tier",
            "current_status": "Status",
            "recommended_action": "Recommended Next Action",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticket ID": st.column_config.NumberColumn(format="%d", width="small"),
                "Priority": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="large"),
                "SLA Remaining": st.column_config.TextColumn(width="medium"),
                "Breach Risk (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Status": st.column_config.TextColumn(width="small"),
                "Recommended Next Action": st.column_config.TextColumn(width="large"),
            },
        )

        # Ticket selector for action panel
        sel_tids = my_df["ticket_id"].tolist()
        c_sel1, _ = st.columns([2.0, 1.0])
        with c_sel1:
            if "active_ticket_id" in st.session_state and st.session_state["active_ticket_id"] in sel_tids:
                st.session_state["my_tickets_selector"] = st.session_state["active_ticket_id"]
            chosen_tid = st.selectbox(
                "Select a Ticket to Open Action Panel",
                sel_tids,
                index=0 if st.session_state.get("active_ticket_id") not in sel_tids else sel_tids.index(st.session_state["active_ticket_id"]),
                format_func=lambda tid: f"Ticket #{tid} — {my_df[my_df['ticket_id'] == tid]['short_description'].iloc[0]} ({my_df[my_df['ticket_id'] == tid]['remaining_display'].iloc[0]})",
                key="my_tickets_selector",
            )
            st.session_state["active_ticket_id"] = chosen_tid

        active_row = my_df[my_df["ticket_id"] == chosen_tid].iloc[0]
        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E5DFE9;'>", unsafe_allow_html=True)
        render_ticket_action_panel(active_row, agents, merchants, agent_recommender)


def render_ticket_action_panel(
    ticket_row: pd.Series,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    agent_recommender: AgentRecommender,
):
    """Render straightforward ticket action panel: status, countdown, probability, dispatch, escalate, notes."""
    tid = int(ticket_row["ticket_id"])
    prio = ticket_row["priority"]
    risk_pct = ticket_row["breach_risk_pct"]
    risk_val = ticket_row["breach_risk"]

    active_agent = get_active_agent(agents)
    active_agent_name = active_agent["agent_name"]

    curr_status = st.session_state["ticket_statuses"].get(tid, "Open")
    curr_agent = st.session_state["ticket_assignments"].get(tid, "Unassigned")
    is_claimed = curr_agent not in ("", "Unassigned")

    st.markdown(
        f"""
        <div style="font-size: 1.25rem; font-weight: 800; color: #120A18; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
            <span>⚡</span> Ticket Action Panel — Ticket #{tid}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # 1. Header with Badges
        risk_class = "kpi-badge-red" if risk_val >= 0.70 else ("kpi-badge-amber" if risk_val >= 0.40 else "kpi-badge-green")
        risk_text = "Critical Risk" if risk_val >= 0.70 else ("At Risk" if risk_val >= 0.40 else "Low Risk")

        owner_badge_color = "#F8F7FB" if is_claimed else "#FEF6EB"
        owner_text_color = "#120A18" if is_claimed else "#8A5310"
        owner_border_color = "#E5DFE9" if is_claimed else "#F8D8A7"
        owner_label = f"Assigned: <b>{curr_agent}</b>" if is_claimed else "<b>Unassigned Intake</b>"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E5DFE9; padding-bottom: 8px; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 800; color: #120A18;">#{tid}</span>
                    <span class="kpi-badge-blue" style="margin-left: 6px;">Priority {prio}</span>
                    <span class="{risk_class}" style="margin-left: 4px;">{risk_text} ({risk_pct:.1f}%)</span>
                </div>
                <div>
                    <span style="font-size: 0.85rem; font-weight: 800; background-color: #F8F7FB; color: #120A18; border: 1px solid #E5DFE9; padding: 4px 10px; border-radius: 7px;">
                        Status: <b>{curr_status}</b>
                    </span>
                    <span style="font-size: 0.85rem; font-weight: 800; background-color: {owner_badge_color}; color: {owner_text_color}; border: 1px solid {owner_border_color}; padding: 4px 10px; border-radius: 7px; margin-left: 4px;">
                        {owner_label}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. SLA Countdown Banner
        remaining = ticket_row["remaining_hours"]
        is_overdue = ticket_row["is_overdue"]
        is_due_soon = ticket_row["is_due_soon"]
        deadline_str = ticket_row["sla_deadline_str"]
        target_hours = ticket_row["sla_target_hours_num"]

        if is_overdue:
            alert_bg, alert_border, alert_color = "#FDF0F1", "#F5BFC3", "#A32832"
            alert_msg = f"🚨 SLA Deadline Elapsed by {abs(remaining):.1f} hours! Ticket is breached."
        elif is_due_soon:
            alert_bg, alert_border, alert_color = "#FEF6EB", "#F8D8A7", "#8A5310"
            alert_msg = f"⏰ SLA Deadline Imminent: {remaining:.1f} hours remaining before breach."
        else:
            alert_bg, alert_border, alert_color = "#EBF7F0", "#B0E2C7", "#16603E"
            alert_msg = f"🟢 Within SLA Window: {remaining:.1f} hours remaining."

        st.markdown(
            f"""
            <div style="background-color: {alert_bg}; border: 1px solid {alert_border}; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.925rem; font-weight: 800; color: {alert_color}; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>{alert_msg}</div>
                <div style="font-size: 0.85rem; font-weight: 700;">Deadline: <b>{deadline_str}</b> (Target: {target_hours:.0f}h)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Context & Routing Guidance
        m_name = ticket_row.get("merchant_name", "Merchant")
        m_tier = ticket_row.get("tier", "Foundation")
        cat = ticket_row.get("category", "General")
        subcat = ticket_row.get("sub_category", "Standard")
        rec_action = ticket_row.get("recommended_action", "Assign to domain specialist.")
        rec_spec = ticket_row.get("recommended_agent", "Priya Sharma")

        spec_explain = explain_specialist_recommendation(ticket_row, active_agent)

        st.markdown(
            f"""
            <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-bottom: 2px;">
                    {cat} — {subcat}
                </div>
                <div style="font-size: 0.875rem; font-weight: 600; color: #4A4252; margin-bottom: 6px;">
                    Merchant: <b>{m_name}</b> ({m_tier} Tier)
                </div>
                <div style="font-size: 0.875rem; font-weight: 600; color: #120A18; background-color: #FFFFFF; border: 1px solid #E5DFE9; border-radius: 8px; padding: 7px 12px; margin-bottom: 6px;">
                    <b>Specialist Guidance:</b> {spec_explain['role_label']} • {spec_explain['explanation']}
                </div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #1A1322; background-color: #FFFFFF; border: 1px solid #E5DFE9; border-radius: 8px; padding: 7px 12px;">
                    <b>Recommended Next Action:</b> {rec_action}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # If ticket is unassigned: show prominent Claim Ticket button
        if not is_claimed:
            if st.button("📥 Claim Ticket Now (Add to My Assigned Tickets)", key=f"panel_claim_{tid}", use_container_width=True, type="primary"):
                st.session_state["ticket_assignments"][tid] = active_agent_name
                st.session_state["ticket_statuses"][tid] = "In Progress"
                st.toast(f"Claimed Ticket #{tid}!", icon="✅")
                st.rerun()
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        # 4. Action Controls: Status, Assign, Reassign, Escalate, Resolve
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            status_options = ["Open", "In Progress", "Waiting on Customer", "Escalated", "Resolved"]
            new_status = st.selectbox(
                "Update Status",
                status_options,
                index=status_options.index(curr_status) if curr_status in status_options else 0,
                key=f"status_select_action_{tid}",
            )
        with act_c2:
            all_agent_names = sorted(agents["agent_name"].unique().tolist())
            init_agent = curr_agent if curr_agent in all_agent_names else (rec_spec if rec_spec in all_agent_names else all_agent_names[0])
            new_agent = st.selectbox(
                "Assign / Reassign Specialist",
                all_agent_names,
                index=all_agent_names.index(init_agent),
                key=f"agent_select_action_{tid}",
            )

        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            if st.button("⚡ Quick Dispatch", key=f"btn_quick_dispatch_{tid}", use_container_width=True, help=f"Assign directly to recommended specialist {rec_spec}"):
                st.session_state["ticket_assignments"][tid] = rec_spec
                st.session_state["ticket_statuses"][tid] = "In Progress"
                st.success(f"Dispatched Ticket #{tid} to {rec_spec}!")
                st.rerun()

        with b_c2:
            if st.button("✅ Mark Resolved", key=f"btn_mark_resolved_{tid}", use_container_width=True):
                st.session_state["ticket_statuses"][tid] = "Resolved"
                st.success(f"Ticket #{tid} marked as Resolved!")
                st.rerun()

        with b_c3:
            if st.button("🚨 Escalate to Tier L3", key=f"btn_escalate_{tid}", use_container_width=True):
                st.session_state["ticket_statuses"][tid] = "Escalated"
                l3_agents = agents[(agents["tier"] == "L3") & (agents["primary_category"] == cat)]
                if len(l3_agents) > 0:
                    st.session_state["ticket_assignments"][tid] = l3_agents.iloc[0]["agent_name"]
                st.warning(f"Ticket #{tid} escalated to L3 specialist!")
                st.rerun()

        # 5. Resolution Notes
        st.markdown(
            """<div style="font-size: 0.85rem; font-weight: 800; color: #4A4252; text-transform: uppercase; margin: 10px 0 4px 0;">
                RESOLUTION NOTES
            </div>""",
            unsafe_allow_html=True,
        )
        saved_notes = st.session_state["ticket_notes"].get(tid, "")
        notes_input = st.text_area(
            "Resolution Notes",
            value=saved_notes,
            placeholder="Log customer correspondence, troubleshooting steps, or resolution details...",
            key=f"notes_area_{tid}",
            height=80,
            label_visibility="collapsed",
        )

        if st.button("💾 Save Notes & Update Ticket", key=f"btn_save_notes_{tid}", use_container_width=True):
            st.session_state["ticket_statuses"][tid] = new_status
            st.session_state["ticket_assignments"][tid] = new_agent
            st.session_state["ticket_notes"][tid] = notes_input
            st.toast(f"Ticket #{tid} updated successfully!", icon="✅")
            st.rerun()


def render_agent_profile(
    agents: pd.DataFrame,
    queue_df: pd.DataFrame,
):
    """Render Agent Profile view: Compact identity selector, capability stats, and workload intelligence."""
    st.session_state.setdefault("active_agent_name", "Priya Sharma")
    st.session_state.setdefault("ticket_assignments", {})
    st.session_state.setdefault("ticket_statuses", {})

    st.markdown(
        """
        <div class="brand-header" style="margin-bottom: 12px;">
            <h2 class="brand-title" style="display: flex; align-items: center; gap: 8px;">
                <span>👤</span> Support Agent Profile &amp; Identity Switcher
            </h2>
            <p class="brand-desc">Configure your active identity to tailor My Workspace and My Tickets to your domain specialization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_names = sorted(agents["agent_name"].unique().tolist())
    curr_name = st.session_state["active_agent_name"]
    curr_idx = all_names.index(curr_name) if curr_name in all_names else 0

    active_agent = get_active_agent(agents)
    assigned = st.session_state["ticket_assignments"]
    statuses = st.session_state["ticket_statuses"]

    my_tids = [
        tid for tid, assigned_to in assigned.items()
        if assigned_to == active_agent["agent_name"] and statuses.get(tid, "Open") != "Resolved"
    ]
    resolved_count = sum(
        1 for tid, assigned_to in assigned.items()
        if assigned_to == active_agent["agent_name"] and statuses.get(tid) == "Resolved"
    )
    domain_open_count = len(queue_df[queue_df["category"] == active_agent["primary_category"]])

    # 1. Single compact unified profile container (Preserved verbatim)
    with st.container(border=True):
        sel_col1, sel_col2 = st.columns([2.5, 1.2])
        with sel_col1:
            sel_name = st.selectbox(
                "Active Support Agent Profile",
                all_names,
                index=curr_idx,
                key="agent_profile_switcher_sb",
            )
            if sel_name != curr_name:
                st.session_state["active_agent_name"] = sel_name
                st.toast(f"Switched identity to {sel_name}!", icon="👤")
                st.rerun()
        with sel_col2:
            st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="kpi-badge-blue" style="font-size: 0.8rem; padding: 6px 12px; display: inline-block;">
                    Active Session: <b>{sel_name}</b>
                </span>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #E5DFE9;'>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seniority Tier", f"Tier {active_agent['tier']}")
        c2.metric("Domain Specialty", active_agent["primary_category"])
        c3.metric("Shift Region", active_agent["shift_region"])
        c4.metric("Efficiency Rating", f"{active_agent['efficiency_multiplier']:.2f}x")

        st.markdown(
            f"""
            <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 12px 16px; margin-top: 10px;">
                <div style="font-size: 0.85rem; font-weight: 800; color: #4A4252; text-transform: uppercase; margin-bottom: 4px;">
                    Workload Summary for {active_agent['agent_name']}
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.9rem; font-weight: 600; color: #1A1322;">
                    <div>📋 Currently Assigned: <b>{len(my_tids)}</b></div>
                    <div>✅ Resolved in Session: <b>{resolved_count}</b></div>
                    <div>⚡ Domain Queue Backlog (<b>{active_agent['primary_category']}</b>): <b>{domain_open_count}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. WORKLOAD OVERVIEW (4 Compact KPI Cards)
    agent_name = active_agent["agent_name"]
    agent_tier = str(active_agent.get("tier", "L2"))
    agent_cat = active_agent.get("primary_category", "")
    agent_shift = active_agent.get("shift_region", "")
    agent_eff = float(active_agent.get("efficiency_multiplier", 1.0))

    my_assigned_df = queue_df[queue_df["ticket_id"].isin(my_tids)].copy() if len(my_tids) > 0 else pd.DataFrame()
    active_assigned_count = len(my_assigned_df)
    critical_assigned_count = int(((my_assigned_df["breach_risk"] >= 0.70) | (my_assigned_df["priority"] == "P1")).sum()) if active_assigned_count > 0 else 0
    due_soon_overdue_count = int((my_assigned_df["is_due_soon"] | my_assigned_df["is_overdue"]).sum()) if active_assigned_count > 0 else 0

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            render_kpi_card(
                title="Active Assigned Tickets",
                value=f"{active_assigned_count}",
                subtext="In Personal Queue",
                badge_text="Active",
                badge_variant="purple",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_kpi_card(
                title="Critical Assigned Tickets",
                value=f"{critical_assigned_count}",
                subtext="P1 or Breach Risk ≥ 70%",
                badge_text="Urgent" if critical_assigned_count > 0 else "Clear",
                badge_variant="red" if critical_assigned_count > 0 else "green",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_kpi_card(
                title="Due Soon / Overdue Tickets",
                value=f"{due_soon_overdue_count}",
                subtext="SLA Window Tight / Breached",
                badge_text="Attention" if due_soon_overdue_count > 0 else "On Track",
                badge_variant="amber" if due_soon_overdue_count > 0 else "green",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_kpi_card(
                title="Resolved in Current Session",
                value=f"{resolved_count}",
                subtext="Closed this session",
                badge_text="Completed",
                badge_variant="green",
            ),
            unsafe_allow_html=True,
        )

    # 3 & 4. TWO-COLUMN LAYOUT: MY TICKET DISTRIBUTION & MY DOMAIN QUEUE
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    col_dist, col_queue = st.columns(2)

    with col_dist:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 0.925rem; font-weight: 800; color: #120A18; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                    <span>📊 MY TICKET DISTRIBUTION</span>
                    <span class="kpi-badge-purple" style="font-size: 0.75rem;">Assigned Breakdown</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if active_assigned_count == 0:
                st.markdown(
                    """
                    <div style="background-color: #F8F7FB; border: 1.5px dashed #D5CBDD; border-radius: 12px; padding: 28px 20px; text-align: center; margin: 4px 0;">
                        <div style="font-size: 1.8rem; margin-bottom: 8px;">📭</div>
                        <div style="font-size: 0.95rem; font-weight: 800; color: #120A18; margin-bottom: 4px;">No Active Tickets Assigned</div>
                        <div style="font-size: 0.85rem; color: #5A5262; line-height: 1.45; max-width: 320px; margin: 0 auto;">
                            Claim tickets from your domain queue or wait for supervisor routing to view your personal priority distribution.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                p1_count = int((my_assigned_df["priority"] == "P1").sum())
                p2_count = int((my_assigned_df["priority"] == "P2").sum())
                p3_count = int((my_assigned_df["priority"] == "P3").sum())
                p4_count = int((my_assigned_df["priority"] == "P4").sum())

                p1_pct = (p1_count / active_assigned_count * 100)
                p2_pct = (p2_count / active_assigned_count * 100)
                p3_pct = (p3_count / active_assigned_count * 100)
                p4_pct = (p4_count / active_assigned_count * 100)

                priorities_config = [
                    ("P1", "Critical", p1_count, p1_pct, "#A32832", "red"),
                    ("P2", "High", p2_count, p2_pct, "#8A5310", "amber"),
                    ("P3", "Medium", p3_count, p3_pct, "#6C5478", "purple"),
                    ("P4", "Low", p4_count, p4_pct, "#4A4252", "blue"),
                ]

                # Combined segmented distribution bar
                segments_html = []
                for p_code, _, count, pct, color, _ in priorities_config:
                    if count > 0:
                        segments_html.append(f'<div style="background-color: {color}; width: {pct:.1f}%; height: 100%;" title="{p_code}: {count} ({pct:.0f}%)"></div>')
                all_segments = "".join(segments_html)

                st.markdown(
                    f"""
                    <div style="background-color: #F0EAF5; border-radius: 6px; height: 10px; width: 100%; display: flex; overflow: hidden; margin-bottom: 16px;">
                        {all_segments}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for p_code, p_label, count, pct, bar_color, badge_var in priorities_config:
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 11px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-size: 0.85rem;">
                                <span style="font-weight: 700; color: #120A18; display: inline-flex; align-items: center; gap: 6px;">
                                    <span class="kpi-badge-{badge_var}" style="padding: 2px 7px; font-size: 0.725rem;">{p_code}</span>
                                    <span>{p_label}</span>
                                </span>
                                <span style="font-weight: 800; color: #120A18;">
                                    {count} ticket{'s' if count != 1 else ''}
                                    <span style="font-weight: 600; color: #5A5262; font-size: 0.8rem;">({pct:.0f}%)</span>
                                </span>
                            </div>
                            <div style="background-color: #F0EAF5; border-radius: 5px; height: 6px; width: 100%; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {pct:.1f}%; height: 100%; border-radius: 5px; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #E5DFE9; font-size: 0.8rem; font-weight: 700; color: #5A5262; display: flex; justify-content: space-between;">
                        <span>Total Active Queue</span>
                        <span style="color: #120A18; font-weight: 800;">{active_assigned_count} Assigned Tickets</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_queue:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="font-size: 0.925rem; font-weight: 800; color: #120A18; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                    <span>⚡ MY DOMAIN QUEUE</span>
                    <span class="kpi-badge-purple" style="font-size: 0.75rem;">{agent_cat}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            domain_df = queue_df[queue_df["category"] == agent_cat].copy()
            domain_total = len(domain_df)
            domain_critical = int(((domain_df["breach_risk"] >= 0.70) | (domain_df["priority"] == "P1")).sum()) if domain_total > 0 else 0
            claimed_tids = set(assigned.keys())
            domain_claimable = int((~domain_df["ticket_id"].isin(claimed_tids)).sum()) if domain_total > 0 else 0
            domain_overdue = int(domain_df["is_overdue"].sum()) if domain_total > 0 else 0
            domain_due_soon = int(domain_df["is_due_soon"].sum()) if domain_total > 0 else 0
            domain_on_track = max(0, domain_total - domain_overdue - domain_due_soon)

            sla_badge = "kpi-badge-green"
            sla_text = "Healthy"
            sla_sub = f"{domain_on_track} tickets on schedule"
            sla_bg = "#EBF7F0"
            sla_border = "#B0E2C7"
            sla_hdr_color = "#16603E"

            if domain_overdue > 0:
                sla_badge = "kpi-badge-red"
                sla_text = f"{domain_overdue} Overdue Breach"
                sla_sub = f"{domain_due_soon} due soon • {domain_on_track} on track"
                sla_bg = "#FDF0F1"
                sla_border = "#F5BFC3"
                sla_hdr_color = "#A32832"
            elif domain_due_soon > 0:
                sla_badge = "kpi-badge-amber"
                sla_text = f"{domain_due_soon} Due Soon"
                sla_sub = f"{domain_on_track} tickets on track"
                sla_bg = "#FEF6EB"
                sla_border = "#F8D8A7"
                sla_hdr_color = "#8A5310"

            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                    <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #4A4252; text-transform: uppercase; letter-spacing: 0.04em;">Total in Domain</div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #120A18; margin: 2px 0 1px 0; font-family: Manrope, sans-serif;">{domain_total}</div>
                        <div style="font-size: 0.775rem; font-weight: 600; color: #5A5262;">Intake queue volume</div>
                    </div>
                    <div style="background-color: #FDF0F1; border: 1px solid #F5BFC3; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #A32832; text-transform: uppercase; letter-spacing: 0.04em;">Critical in Domain</div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #A32832; margin: 2px 0 1px 0; font-family: Manrope, sans-serif;">{domain_critical}</div>
                        <div style="font-size: 0.775rem; font-weight: 600; color: #A32832;">P1 or Risk ≥ 70%</div>
                    </div>
                    <div style="background-color: #EDF3FC; border: 1px solid #C4D7F0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #2D4E7D; text-transform: uppercase; letter-spacing: 0.04em;">Claimable Tickets</div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #2D4E7D; margin: 2px 0 1px 0; font-family: Manrope, sans-serif;">{domain_claimable}</div>
                        <div style="font-size: 0.775rem; font-weight: 600; color: #2D4E7D;">Unassigned in domain</div>
                    </div>
                    <div style="background-color: {sla_bg}; border: 1px solid {sla_border}; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: {sla_hdr_color}; text-transform: uppercase; letter-spacing: 0.04em;">SLA Queue Health</div>
                        <div style="margin: 4px 0 2px 0;"><span class="{sla_badge}" style="font-size: 0.8rem; padding: 2px 8px;">{sla_text}</span></div>
                        <div style="font-size: 0.775rem; font-weight: 600; color: #5A5262;">{sla_sub}</div>
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #E5DFE9; font-size: 0.8rem; font-weight: 700; color: #5A5262; display: flex; justify-content: space-between;">
                    <span>Domain Specialty</span>
                    <span style="color: #765184; font-weight: 800;">{agent_cat} Intake</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. SPECIALIST PROFILE SUMMARY (Clean Two-Column Panel)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 0.925rem; font-weight: 800; color: #120A18; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between;">
                <span>📋 SPECIALIST PROFILE SUMMARY</span>
                <span class="kpi-badge-blue" style="font-size: 0.75rem;">Operational Authority</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tier_upper = agent_tier.upper()
        if tier_upper == "L1":
            supported_priorities_title = "P3 (Medium), P4 (Low)"
            supported_priorities_desc = "Routine inquiries, standard technical assistance, and general queue resolution."
        elif tier_upper == "L2":
            supported_priorities_title = "P2 (High), P3 (Medium), P4 (Low)"
            supported_priorities_desc = "Senior escalation triage, payment discrepancies, and complex merchant issues."
        elif tier_upper == "L3":
            supported_priorities_title = "P1 (Critical), P2 (High), P3 (Medium), P4 (Low)"
            supported_priorities_desc = "Full escalation authority, critical outage management, and SLA breach interception."
        else:
            supported_priorities_title = "All Priority Tiers (Standard Protocol)"
            supported_priorities_desc = "Universal routing authority across general queue triage."

        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.markdown(
                f"""
                <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 14px 16px; height: 100%;">
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Primary Domain Specialization</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px;">{agent_cat}</div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">Aligned domain routing eliminating category mismatch penalties.</div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Seniority Tier</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px; display: flex; align-items: center; gap: 8px;">
                            <span class="kpi-badge-purple" style="font-size: 0.8rem; padding: 3px 10px;">Tier {agent_tier}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">Specialist operational tier and incident response capability.</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Supported Priority Levels</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px;">{supported_priorities_title}</div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">{supported_priorities_desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sum_col2:
            st.markdown(
                f"""
                <div style="background-color: #F8F7FB; border: 1px solid #E5DFE9; border-radius: 10px; padding: 14px 16px; height: 100%;">
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Shift Region</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px; display: flex; align-items: center; gap: 8px;">
                            <span>🌐 {agent_shift}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">Synchronous regional operating coverage for merchant support.</div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Efficiency Rating</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px; display: flex; align-items: center; gap: 8px;">
                            <span class="kpi-badge-green" style="font-size: 0.8rem; padding: 3px 10px;">{agent_eff:.2f}x Throughput</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">Calibrated resolution velocity factor utilized by recommender engine.</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 800; color: #5A5262; text-transform: uppercase; letter-spacing: 0.04em;">Current Assigned Workload</div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #120A18; margin-top: 2px;">
                            {active_assigned_count} Active Ticket{'s' if active_assigned_count != 1 else ''} <span style="font-size: 0.85rem; font-weight: 600; color: #5A5262;">({resolved_count} resolved in session)</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #5A5262; margin-top: 1px;">Real-time capacity utilization across active support assignments.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Backward compatibility alias
render_agent_workspace = render_my_workspace
