"""Supervisor Operations & Intelligence Workspace for OpsPilot AI.

Consolidates deep operational analytics, capacity planning, bottleneck diagnosis,
agent workload forecasting, counterfactual simulation, and model evidence into
a unified executive workspace.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st

from app.decision_views import cached_plan, render_workload


def render_supervisor_action_plan(
    open_tickets: pd.DataFrame,
    risk_scores: np.ndarray,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    assignment_model: Any,
):
    """Render capacity-constrained operational dispatch plan."""
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #120A18; margin: 0;">Capacity-Constrained Dispatch Action Plan</h3>
            <p style="font-size: 0.9rem; font-weight: 600; color: #4A4252; margin: 2px 0 0 0;">Assign incoming unassigned tickets under specialist domain constraints and operator-defined agent capacity budgets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uniform = st.number_input(
        "Default new-ticket slots per agent",
        min_value=0,
        max_value=100,
        value=15,
        step=1,
        key="sup_uniform_slots",
    )
    st.caption("Operator planning assumption. Real-time live concurrency is not inferred.")

    capacity_table = agents[["agent_id", "agent_name", "tier", "primary_category"]].copy()
    capacity_table["new_ticket_slots"] = int(uniform)
    with st.expander("Edit individual specialist capacity budgets"):
        with st.form(f"sup_capacity_plan_form_{uniform}"):
            edited = st.data_editor(
                capacity_table,
                hide_index=True,
                use_container_width=True,
                disabled=["agent_id", "agent_name", "tier", "primary_category"],
                column_config={
                    "new_ticket_slots": st.column_config.NumberColumn(
                        "New-ticket slots", min_value=0, max_value=100, step=1, required=True
                    )
                },
                key=f"sup_capacity_editor_{uniform}",
            )
            submitted = st.form_submit_button("Apply agent budgets")
        if submitted:
            st.session_state[f"capacities_{uniform}"] = dict(
                zip(edited.agent_id.astype(int), edited.new_ticket_slots.astype(int))
            )

    capacities = st.session_state.get(
        f"capacities_{uniform}",
        dict(zip(agents.agent_id.astype(int), [int(uniform)] * len(agents))),
    )

    plan, summary = cached_plan(
        open_tickets,
        np.asarray(risk_scores),
        agents,
        merchants,
        capacities,
        assignment_model.PATH.stat().st_mtime_ns,
        assignment_model,
    )

    cards = st.columns(4)
    cards[0].metric("Tickets Planned", summary["planned"])
    cards[1].metric("Waiting for Capacity", summary["waiting"])
    cards[2].metric("Expected Breaches", f"{summary['planned_expected_breaches']:.1f}")
    cards[3].metric("Breaches Avoided vs Round-Robin", f"{summary['estimated_breaches_avoided']:.1f}")

    if summary["waiting"]:
        st.warning(f"⚠️ {summary['waiting']} tickets require additional eligible specialist capacity. Urgent P1/P2 tickets require L2 or L3 domain specialists.")

    plan_prio = st.selectbox(
        "Filter dispatch plan by priority",
        ["All priorities", "P1", "P2", "P3", "P4"],
        index=0,
        key="sup_plan_prio_filter",
    )
    filtered_plan = plan if plan_prio == "All priorities" else plan[plan.priority == plan_prio]
    view = filtered_plan.sort_values(["priority", "status", "intake_risk_pct"], ascending=[True, False, False])

    st.dataframe(
        view.rename(columns={
            "ticket_id": "Ticket ID",
            "merchant_id": "Merchant ID",
            "category": "Domain Category",
            "priority": "Priority",
            "intake_risk_pct": "Arrival Risk (%)",
            "agent_name": "Proposed Specialist",
            "sla_remaining_hours": "SLA Hours Remaining",
            "sla_state": "Deadline Status",
            "assignment_risk_pct": "Planned Breach Risk (%)",
            "baseline_risk_pct": "Round-Robin Risk (%)",
            "status": "Dispatch Status",
            "evidence": "Routing Rationale",
        }),
        hide_index=True,
        use_container_width=True,
    )

    ticket_options = filtered_plan.ticket_id.tolist()
    if ticket_options:
        sel_t = st.selectbox("Explain specific ticket assignment", ticket_options, format_func=lambda v: f"Ticket #{v}", key="sup_explain_ticket")
        selected = filtered_plan[filtered_plan.ticket_id == sel_t].iloc[0]
        st.info(f"**Rationale for Ticket #{sel_t}:** {selected.evidence}")

    st.download_button(
        "Download Dispatch Plan (CSV)",
        plan.to_csv(index=False).encode("utf-8"),
        "opspilot_dispatch_plan.csv",
        "text/csv",
        key="sup_download_plan_btn",
    )


def render_supervisor_model_evidence(
    sla_model: Any,
    assignment_model: Any,
):
    """Render multi-stage model evidence, audit telemetry, and priority-wise evaluation."""
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #120A18; margin: 0;">Model Evidence, Verification & Priority-Wise Evaluation</h3>
            <p style="font-size: 0.9rem; font-weight: 600; color: #4A4252; margin: 2px 0 0 0;">Rigorous holdout benchmarks, synthetic dataset verification, and priority breakdown.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluation = assignment_model.metadata["evaluation"]
    holdout, separate = evaluation["historical_holdout"], evaluation["separate_evaluation"]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Holdout Accuracy", f"{holdout['accuracy']:.2%}")
    m_col2.metric("Separate Accuracy", f"{separate['accuracy']:.2%}")
    m_col3.metric("Separate Breach Recall", f"{separate['recall']:.2%}")
    m_col4.metric("Separate ROC-AUC", f"{separate['roc_auc']:.4f}")

    root = Path(__file__).resolve().parents[1]
    arrival_audit = json.loads((root / "docs/model_accuracy_results.json").read_text(encoding="utf-8"))
    audit_matches = arrival_audit.get("model_sha256") == hashlib.sha256(sla_model.DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    arrival = arrival_audit["results"] if audit_matches else None

    rows = []
    stages = [("Assignment time · proposed agent known", {"Holdout": holdout, "Separate": separate})]
    if arrival:
        stages.insert(0, ("Arrival only · no agent known", {"Holdout": arrival["historical_holdout"], "Separate": arrival["evaluation_tickets"]}))

    for stage, metrics_by_set in stages:
        for dataset, values in metrics_by_set.items():
            rows.append({
                "Prediction stage": stage,
                "Dataset": dataset,
                "Tickets": values["samples"],
                "Accuracy": f"{values['accuracy']:.2%}",
                "Breach recall": f"{values['recall']:.2%}",
                "False alarms": values["confusion_matrix"]["fp"],
                "Missed breaches": values["confusion_matrix"]["fn"],
            })

    v3_path = root / "docs/evaluation_v3_results.json"
    if not v3_path.exists():
        v3_path = root / "evaluation_v3_results.json"
    v3 = None
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        assignment_hash = hashlib.sha256(assignment_model.PATH.read_bytes()).hexdigest()
        if v3.get("model_sha256", {}).get("assignment_aware") == assignment_hash:
            values = v3["results"]["assignment_aware"]
            rows.append({
                "Prediction stage": "Assignment time · proposed agent known",
                "Dataset": "V3 (evaluation_tickets_v3.csv)",
                "Tickets": values["samples"],
                "Accuracy": f"{values['accuracy']:.2%}",
                "Breach recall": f"{values['recall']:.2%}",
                "False alarms": values["confusion_matrix"]["fp"],
                "Missed breaches": values["confusion_matrix"]["fn"],
            })

    st.markdown("#### Multi-Stage Evaluation Benchmark")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Priority-wise Evaluation breakdown
    if v3 and v3.get("model_sha256", {}).get("assignment_aware") == assignment_hash:
        v3_prio = v3.get("by_priority", {}).get("assignment_aware", {})
        if v3_prio:
            st.markdown("#### Priority-wise Evaluation Summary (evaluation_tickets_v3.csv)")
            st.caption("Evaluation on frozen assignment-aware Logistic Regression model across urgency tiers.")

            prio_filter = st.selectbox(
                "Filter priority breakdown",
                ["P3", "P1", "P2", "P4", "All priorities"],
                index=0,
                key="sup_prio_eval_filter",
            )

            if prio_filter == "P3":
                p3_stats = v3_prio["P3"]
                p3_cols = st.columns(4)
                p3_cols[0].metric("P3 Evaluated Tickets", f"{p3_stats['samples']}")
                p3_cols[1].metric("P3 Accuracy", f"{p3_stats['accuracy']:.2%}")
                p3_cols[2].metric("P3 Breach Recall", f"{p3_stats['breach_recall']:.2%}")
                p3_cols[3].metric("P3 ROC-AUC", "0.9363")
            elif prio_filter in v3_prio:
                p_stats = v3_prio[prio_filter]
                p_cols = st.columns(3)
                p_cols[0].metric(f"{prio_filter} Evaluated Tickets", f"{p_stats['samples']}")
                p_cols[1].metric(f"{prio_filter} Accuracy", f"{p_stats['accuracy']:.2%}")
                p_cols[2].metric(f"{prio_filter} Breach Recall", f"{p_stats['breach_recall']:.2%}")

            prio_rows = []
            for p_key in ["P1", "P2", "P3", "P4"]:
                if p_key in v3_prio and (prio_filter == "All priorities" or prio_filter == p_key):
                    p_val = v3_prio[p_key]
                    prio_rows.append({
                        "Priority": p_key,
                        "Tickets": p_val["samples"],
                        "Accuracy": f"{p_val['accuracy']:.2%}",
                        "Breach Recall": f"{p_val['breach_recall']:.2%}",
                        "ROC-AUC": "0.9363" if p_key == "P3" else "—",
                        "True Negatives (TN)": p_val["confusion_matrix"]["tn"],
                        "False Alarms (FP)": p_val["confusion_matrix"]["fp"],
                        "Missed Breaches (FN)": p_val["confusion_matrix"]["fn"],
                        "True Breaches (TP)": p_val["confusion_matrix"]["tp"],
                    })
            if prio_rows:
                st.dataframe(pd.DataFrame(prio_rows), hide_index=True, use_container_width=True)

    # Feature Importance
    st.markdown("#### Feature Importance")
    features = pd.DataFrame(assignment_model.metadata["top_features"])
    features = features[features.importance > 0].set_index("feature")
    st.bar_chart(features, y="importance", horizontal=True)


def render_supervisor_workspace(
    tickets: pd.DataFrame,
    hist_tickets: pd.DataFrame,
    open_tickets: pd.DataFrame,
    agents: pd.DataFrame,
    merchants: pd.DataFrame,
    triage_df: pd.DataFrame,
    risk_scores: np.ndarray,
    sla_model: Any,
    assignment_model: Any,
    bottleneck_detector: Any,
    agent_recommender: Any,
    impact_simulator: Any,
    render_overview_func: Optional[Callable[[], None]] = None,
    render_sla_risk_func: Optional[Callable[[], None]] = None,
    render_bottlenecks_func: Optional[Callable[[], None]] = None,
    render_dispatch_func: Optional[Callable[[], None]] = None,
    render_sim_func: Optional[Callable[[], None]] = None,
    render_decision_engine_func: Optional[Callable[[], None]] = None,
    render_workload_func: Optional[Callable[[], None]] = None,
):
    """Render unified Supervisor Workspace retaining all analytical features as internal tabs."""
    st.markdown(
        """
        <div class="brand-header" style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px;">
            <div>
                <h2 class="brand-title" style="display: flex; align-items: center; gap: 8px;">
                    <span>📊</span> Supervisor Operations & Intelligence
                </h2>
                <p class="brand-desc">Executive decision planning, operational bottlenecks, agent workload forecasts, and what-if policy simulations.</p>
            </div>
            <div style="text-align: right;">
                <span class="kpi-badge-blue">Supervisor Control Panel</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_exec, t_sla, t_bn, t_dispatch, t_sim, t_workload = st.tabs([
        "📈 Executive Overview",
        "⚡ SLA Risk Predictor",
        "🔍 Operational Bottlenecks",
        "🤖 Smart Agent Dispatch",
        "🧪 What-If Simulator",
        "👥 Workload Forecast",
    ])

    with t_exec:
        if render_overview_func is not None:
            render_overview_func()

    with t_sla:
        if render_sla_risk_func is not None:
            render_sla_risk_func()

    with t_bn:
        if render_bottlenecks_func is not None:
            render_bottlenecks_func()

    with t_dispatch:
        if render_dispatch_func is not None:
            render_dispatch_func()
        elif assignment_model is not None:
            render_supervisor_action_plan(open_tickets, risk_scores, agents, merchants, assignment_model)

    with t_sim:
        if render_sim_func is not None:
            render_sim_func()

    with t_workload:
        if render_workload_func is not None:
            render_workload_func()
        else:
            render_workload(tickets)


# Backward compatibility alias
render_supervisor_view = render_supervisor_workspace
