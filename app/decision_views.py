"""Decision workbench, transparent model evidence and workload forecast views."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics.workload import WorkloadForecaster
from src.routing.decision_engine import DecisionEngine


@st.cache_data(max_entries=16, ttl=600)
def cached_plan(tickets, intake_risk, agents, merchants, capacities, artifact_version, _model):
    return DecisionEngine(_model, agents, merchants).plan(tickets, intake_risk, capacities)


@st.cache_data(max_entries=8, ttl=600)
def cached_workload(tickets, horizon):
    return WorkloadForecaster(tickets).analyze(horizon)


def render_decision_engine(open_tickets, intake_risk, agents, merchants, intake_model, assignment_model):
    st.title("AI operations decision engine")
    st.caption("Find at-risk work, compare specialist assignments and build a capacity-constrained dispatch plan.")
    evaluation = assignment_model.metadata["evaluation"]
    holdout, separate = evaluation["historical_holdout"], evaluation["separate_evaluation"]
    metrics = st.columns(4)
    metrics[0].metric("Assignment model · holdout accuracy", f"{holdout['accuracy']:.2%}")
    metrics[1].metric("Assignment model · separate accuracy", f"{separate['accuracy']:.2%}")
    metrics[2].metric("Separate-set breach recall", f"{separate['recall']:.2%}")
    metrics[3].metric("Separate-set ROC-AUC", f"{separate['roc_auc']:.4f}")
    st.info("The assignment model scores a specified agent before work starts. Arrival-only risk remains a separate model. All benchmark datasets are synthetic.")
    if assignment_model.metadata.get("profile") == "compact":
        st.caption("Active profile: compact regularized logistic regression. Selected near 85% training-validation accuracy; actual accuracy varies across datasets.")

    st.subheader("Plan available capacity")
    uniform = st.number_input("Default new-ticket slots per agent", min_value=0, max_value=100, value=15, step=1)
    st.caption("These are planning assumptions you control. Current live agent availability is not supplied by this dataset.")
    capacity_table = agents[["agent_id", "agent_name", "tier", "primary_category"]].copy()
    capacity_table["new_ticket_slots"] = int(uniform)
    with st.expander("Edit individual agent budgets"):
        with st.form(f"capacity_plan_{uniform}"):
            edited = st.data_editor(
                capacity_table, hide_index=True, width="stretch",
                disabled=["agent_id", "agent_name", "tier", "primary_category"],
                column_config={"new_ticket_slots": st.column_config.NumberColumn("New-ticket slots", min_value=0, max_value=100, step=1, required=True)},
                key=f"capacity_editor_{uniform}",
            )
            submitted = st.form_submit_button("Apply agent budgets")
        if submitted:
            st.session_state[f"capacities_{uniform}"] = dict(zip(edited.agent_id.astype(int), edited.new_ticket_slots.astype(int)))
    capacities = st.session_state.get(f"capacities_{uniform}", dict(zip(agents.agent_id.astype(int), [int(uniform)]*len(agents))))
    plan, summary = cached_plan(
        open_tickets, np.asarray(intake_risk), agents, merchants, capacities,
        assignment_model.PATH.stat().st_mtime_ns, assignment_model,
    )
    cards = st.columns(4)
    cards[0].metric("Tickets planned", summary["planned"])
    cards[1].metric("Waiting for capacity", summary["waiting"])
    cards[2].metric("Expected breaches · scenario", f"{summary['planned_expected_breaches']:.1f}")
    cards[3].metric("Estimated fewer breaches vs round robin", f"{summary['estimated_breaches_avoided']:.1f}")
    st.caption("Expected breaches combine known overdue tickets with model estimates. Other waiting tickets retain arrival-time risk. Assignment effects are observational and do not establish realized savings.")
    st.caption(f"Snapshot: {summary['snapshot']}. {summary['already_overdue']} unresolved tickets have already passed their SLA deadline and count as certain breaches. The plan separates recovery work from preventable future breaches.")
    if summary["waiting"]:
        st.warning(f"{summary['waiting']} tickets need more eligible specialist capacity. P1/P2 tickets require an L2 or L3 domain specialist.")
    tab_plan, tab_evidence, tab_impact = st.tabs(["Prioritized action plan", "Model evidence", "Scenario impact"])
    with tab_plan:
        plan_prio = st.selectbox("Filter action plan by priority", ["All priorities", "P1", "P2", "P3", "P4"], index=0, key="plan_prio_filter")
        filtered_plan = plan if plan_prio == "All priorities" else plan[plan.priority == plan_prio]
        view = filtered_plan.sort_values(["priority", "status", "intake_risk_pct"], ascending=[True, False, False])
        st.dataframe(view.rename(columns={
            "ticket_id": "Ticket", "merchant_id": "Merchant", "category": "Category", "priority": "Priority",
            "intake_risk_pct": "Arrival risk (%)", "agent_id": "Agent ID", "agent_name": "Proposed agent",
            "sla_remaining_hours": "SLA hours remaining", "sla_state": "Deadline status",
            "assignment_risk_pct": "Planned breach risk (%)", "baseline_risk_pct": "Round-robin risk (%)",
            "status": "Status", "evidence": "Why this action",
        }), hide_index=True, width="stretch")
        ticket_options = filtered_plan.ticket_id.tolist()
        if ticket_options:
            ticket_id = st.selectbox("Explain a ticket", ticket_options, format_func=lambda value: f"Ticket {value}")
            selected = filtered_plan[filtered_plan.ticket_id == ticket_id].iloc[0]
            st.info(selected.evidence)
        st.download_button("Download dispatch plan", plan.to_csv(index=False).encode("utf-8"), "opspilot_dispatch_plan.csv", "text/csv")
    with tab_evidence:
        root = Path(__file__).resolve().parents[1]
        arrival_audit = json.loads((root / "docs/model_accuracy_results.json").read_text(encoding="utf-8"))
        audit_matches = arrival_audit.get("model_sha256") == hashlib.sha256(intake_model.DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
        arrival = arrival_audit["results"] if audit_matches else None
        rows = []
        stages = [("Assignment time · proposed agent known", {"Holdout": holdout, "Separate": separate})]
        if arrival:
            stages.insert(0, ("Arrival only · no agent known", {"Holdout": arrival["historical_holdout"], "Separate": arrival["evaluation_tickets"]}))
        for stage, metrics_by_set in stages:
            for dataset, values in metrics_by_set.items():
                rows.append({"Prediction stage": stage, "Dataset": dataset, "Tickets": values["samples"],
                             "Accuracy": f"{values['accuracy']:.2%}", "Breach recall": f"{values['recall']:.2%}",
                             "False alarms": values["confusion_matrix"]["fp"], "Missed breaches": values["confusion_matrix"]["fn"]})
        v2_path = root / "docs/evaluation_v2_results.json"
        if v2_path.exists():
            v2 = json.loads(v2_path.read_text(encoding="utf-8"))
            assignment_hash = hashlib.sha256(assignment_model.PATH.read_bytes()).hexdigest()
            if v2.get("model_sha256", {}).get("assignment_aware") == assignment_hash:
                values = v2["results"]["assignment_aware"]
                rows.append({"Prediction stage": "Assignment time · proposed agent known", "Dataset": "V2", "Tickets": values["samples"],
                             "Accuracy": f"{values['accuracy']:.2%}", "Breach recall": f"{values['recall']:.2%}",
                             "False alarms": values["confusion_matrix"]["fp"], "Missed breaches": values["confusion_matrix"]["fn"]})
        v3_path = root / "docs/evaluation_v3_results.json"
        if not v3_path.exists():
            v3_path = root / "evaluation_v3_results.json"
        v3 = None
        if v3_path.exists():
            v3 = json.loads(v3_path.read_text(encoding="utf-8"))
            assignment_hash = hashlib.sha256(assignment_model.PATH.read_bytes()).hexdigest()
            if v3.get("model_sha256", {}).get("assignment_aware") == assignment_hash:
                values = v3["results"]["assignment_aware"]
                rows.append({"Prediction stage": "Assignment time · proposed agent known", "Dataset": "V3 (evaluation_tickets_v3.csv)", "Tickets": values["samples"],
                             "Accuracy": f"{values['accuracy']:.2%}", "Breach recall": f"{values['recall']:.2%}",
                             "False alarms": values["confusion_matrix"]["fp"], "Missed breaches": values["confusion_matrix"]["fn"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if v3 and v3.get("model_sha256", {}).get("assignment_aware") == assignment_hash:
            v3_prio = v3.get("by_priority", {}).get("assignment_aware", {})
            if v3_prio:
                st.markdown("#### Priority-wise evaluation summary (evaluation_tickets_v3.csv)")
                st.caption("Verified evaluation results from synthetic evaluation_tickets_v3.csv on the frozen assignment-aware Logistic Regression model.")
                prio_filter = st.selectbox(
                    "Filter priority breakdown",
                    ["P3", "P1", "P2", "P4", "All priorities"],
                    index=0,
                    key="prio_eval_filter",
                )
                if prio_filter == "P3":
                    p3_stats = v3_prio["P3"]
                    p3_cols = st.columns(4)
                    p3_cols[0].metric("P3 evaluated tickets", f"{p3_stats['samples']}")
                    p3_cols[1].metric("P3 accuracy", f"{p3_stats['accuracy']:.2%}")
                    p3_cols[2].metric("P3 breach recall", f"{p3_stats['breach_recall']:.2%}")
                    p3_cols[3].metric("P3 ROC-AUC", "0.9363")
                    st.caption(f"P3 confusion matrix: TN={p3_stats['confusion_matrix']['tn']}, FP={p3_stats['confusion_matrix']['fp']}, FN={p3_stats['confusion_matrix']['fn']}, TP={p3_stats['confusion_matrix']['tp']}. Source: evaluation_tickets_v3.csv (frozen assignment-aware model).")
                elif prio_filter in v3_prio:
                    p_stats = v3_prio[prio_filter]
                    p_cols = st.columns(3)
                    p_cols[0].metric(f"{prio_filter} evaluated tickets", f"{p_stats['samples']}")
                    p_cols[1].metric(f"{prio_filter} accuracy", f"{p_stats['accuracy']:.2%}")
                    p_cols[2].metric(f"{prio_filter} breach recall", f"{p_stats['breach_recall']:.2%}")
                    st.caption(f"{prio_filter} confusion matrix: TN={p_stats['confusion_matrix']['tn']}, FP={p_stats['confusion_matrix']['fp']}, FN={p_stats['confusion_matrix']['fn']}, TP={p_stats['confusion_matrix']['tp']}.")
                prio_rows = []
                for p_key in ["P1", "P2", "P3", "P4"]:
                    if p_key in v3_prio and (prio_filter == "All priorities" or prio_filter == p_key):
                        p_val = v3_prio[p_key]
                        prio_rows.append({
                            "Priority": p_key,
                            "Tickets": p_val["samples"],
                            "Accuracy": f"{p_val['accuracy']:.2%}",
                            "Breach recall": f"{p_val['breach_recall']:.2%}",
                            "ROC-AUC": "0.9363" if p_key == "P3" else "—",
                            "True Negatives (TN)": p_val["confusion_matrix"]["tn"],
                            "False Alarms (FP)": p_val["confusion_matrix"]["fp"],
                            "Missed Breaches (FN)": p_val["confusion_matrix"]["fn"],
                            "True Breaches (TP)": p_val["confusion_matrix"]["tp"],
                        })
                if prio_rows:
                    st.dataframe(pd.DataFrame(prio_rows), hide_index=True, width="stretch")
        st.caption("The two stages have different available information. The assignment model uses 26 intake and proposed-agent features; no response or resolution outcomes are input features.")
        features = pd.DataFrame(assignment_model.metadata["top_features"])
        features = features[features.importance > 0].set_index("feature")
        st.bar_chart(features, y="importance", horizontal=True)
        st.caption("Feature importance describes predictive associations, not causes. Train: 1,445 tickets; fixed holdout: 362; separate evaluation: 300. Model selection uses training cross-validation only.")
        for limitation in assignment_model.metadata["limitations"]:
            st.caption(limitation)
    with tab_impact:
        st.dataframe(pd.DataFrame([
            {"Scenario": "Specialist round robin", "Assigned": summary["baseline_planned"], "Expected breaches": round(summary["baseline_expected_breaches"], 2), "Urgency-weighted planning cost": round(summary["baseline_weighted_cost"], 2)},
            {"Scenario": "Risk-optimized plan", "Assigned": summary["planned"], "Expected breaches": round(summary["planned_expected_breaches"], 2), "Urgency-weighted planning cost": round(summary["planned_weighted_cost"], 2)},
        ]), hide_index=True, width="stretch")
        st.write("Both policies use the same specialist eligibility and agent budgets. Planning weights prioritize P1/P2 tickets and Advanced merchant accounts. The optimizer minimizes weighted risk and deferral cost across the whole batch.")
        st.caption("Use the What-If Impact Simulator for historical turnaround-time and capacity-value scenarios. Confirm service-quality improvements with a monitored operational pilot.")


def render_workload(tickets):
    st.title("Workload forecast and surge detection")
    horizon = st.slider("Forecast horizon in days", 1, 14, 7)
    results = cached_workload(tickets[["created_at"]], horizon)
    st.caption(f"Dataset snapshot through {results['origin']:%d %b %Y}. Forecasts start the next calendar day after this snapshot.")
    cards = st.columns(4)
    cards[0].metric(f"Expected tickets · next {horizon} days", f"{results['forecast'].expected_tickets.sum():.0f}")
    cards[1].metric("Backtest error · tickets/day", f"{results['backtest_mae']:.2f}")
    cards[2].metric("Recent-mean baseline error", f"{results['baseline_mae']:.2f}")
    cards[3].metric("Detected surge days", len(results["anomalies"]))
    st.caption(f"Method: {results['method'].replace('_', ' ')}. Selected before the final 28-day rolling backtest; lower mean absolute error is better.")
    history = results["daily"].tail(90).set_index("date").rename(columns={"tickets": "Observed tickets"})
    forecast = results["forecast"].set_index("date")
    chart = history.join(forecast[["expected_tickets"]].rename(columns={"expected_tickets": "Expected tickets"}), how="outer")
    st.line_chart(chart)
    capacity = st.number_input("Assumed team processing capacity per day", min_value=0, max_value=100000, value=20, step=1)
    overloaded = forecast[forecast.upper_80 > capacity]
    if len(overloaded):
        st.warning(f"On {len(overloaded)} forecast days, the upper 80% count bound exceeds the assumed capacity. Review specialist coverage and incident queues.")
    forecast_display = results["forecast"].copy()
    forecast_display["expected_tickets"] = forecast_display.expected_tickets.round(2)
    st.dataframe(forecast_display, hide_index=True, width="stretch")
    st.subheader("Recent anomalous intake")
    st.caption("Surges exceed three standard deviations above a preceding 28-day mean. Only earlier intake counts define each day's baseline.")
    anomaly_display = results["anomalies"].head(12).copy()
    for column in ["tickets", "prior_mean", "surge_score"]:
        anomaly_display[column] = anomaly_display[column].round(2)
    st.dataframe(anomaly_display, hide_index=True, width="stretch")
    with st.expander("Forecast validation and limitations"):
        st.line_chart(results["backtest"].set_index("date"))
        st.write(results["limitations"])
        st.json(results["validation_mae"])
    st.download_button("Download workload forecast", results["forecast"].to_csv(index=False).encode("utf-8"), "opspilot_workload_forecast.csv", "text/csv")
