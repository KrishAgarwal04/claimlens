"""Ticket classification and the workspace triage table."""

import numpy as np
import pandas as pd
import streamlit as st


RISK_LEVELS = ["High risk", "Medium risk", "Low risk", "Unknown"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
SLA_STATUSES = ["Overdue", "Due soon", "Within SLA"]


def prepare_ticket_table(queue_df, assignments, statuses):
    """Classify model risk independently of priority and synchronize live state."""
    df = queue_df.copy()
    risk = pd.to_numeric(df["breach_risk"], errors="coerce")
    df["Risk level"] = np.select(
        [risk.isna(), risk >= 0.70, risk >= 0.40],
        ["Unknown", "High risk", "Medium risk"],
        default="Low risk",
    )
    df["Breach risk (%)"] = (risk * 100).round(1)
    df["SLA status"] = np.select(
        [df["is_overdue"], df["is_due_soon"]],
        ["Overdue", "Due soon"],
        default="Within SLA",
    )
    df["Status"] = df["ticket_id"].map(statuses).astype("string").fillna("Open")
    df["Assigned agent"] = df["ticket_id"].map(assignments).astype("string").fillna("Unassigned")
    df["Ticket ID"] = df["ticket_id"].astype(str)
    return df.rename(columns={
        "priority": "Priority", "category": "Category",
        "short_description": "Description", "remaining_display": "SLA remaining",
        "recommended_agent": "Recommended agent", "recommended_action": "Recommended action",
    })


def render_ticket_triage_table(queue_df):
    """Render one filterable table grouped by risk, priority, or SLA urgency."""
    st.subheader("Ticket risk and priority overview")
    st.caption(
        "All queue tickets • High risk: ≥70% • Medium risk: 40–<70% • Low risk: <40%. "
        "Priority and SLA status are shown separately. Due soon means 4 hours or less remaining."
    )
    df = prepare_ticket_table(
        queue_df,
        st.session_state.get("ticket_assignments", {}),
        st.session_state.get("ticket_statuses", {}),
    )
    with st.container(border=True):
        controls = st.columns(4)
        group_by = controls[0].selectbox(
            "Group by", ["Risk level", "Priority", "SLA status"], key="triage_group_by",
        )
        risk_filter = controls[1].multiselect(
            "Risk level", RISK_LEVELS, key="triage_risk_filter", placeholder="All risk levels",
        )
        priority_filter = controls[2].multiselect(
            "Priority", PRIORITIES, key="triage_priority_filter", placeholder="All priorities",
        )
        sla_filter = controls[3].multiselect(
            "SLA status", SLA_STATUSES, key="triage_sla_filter", placeholder="All SLA statuses",
        )
        for column, values in [
            ("Risk level", risk_filter), ("Priority", priority_filter), ("SLA status", sla_filter),
        ]:
            if values:
                df = df[df[column].isin(values)]

        counts = df["Risk level"].value_counts()
        st.caption(
            f"Showing {len(df):,} of {len(queue_df):,} tickets · "
            f"High risk: {counts.get('High risk', 0):,} · "
            f"Medium risk: {counts.get('Medium risk', 0):,} · "
            f"Low risk: {counts.get('Low risk', 0):,} · "
            f"Unknown: {counts.get('Unknown', 0):,}"
        )
        if df.empty:
            st.info("No tickets match these filters. Clear a filter to see more tickets.")
            return

        orders = {"Risk level": RISK_LEVELS, "Priority": PRIORITIES, "SLA status": SLA_STATUSES}
        sort_columns = [group_by] + [c for c in orders if c != group_by]
        for column in sort_columns:
            df[f"_sort_{column}"] = df[column].map(
                {value: rank for rank, value in enumerate(orders[column])}
            ).fillna(len(orders[column]))
        df = df.sort_values(
            [f"_sort_{c}" for c in sort_columns] + ["Breach risk (%)"],
            ascending=[True] * len(sort_columns) + [False], kind="stable",
        )
        columns = [
            "Ticket ID", "Risk level", "Priority", "Breach risk (%)", "SLA status",
            "SLA remaining", "Category", "Description", "Status", "Assigned agent",
            "Recommended agent", "Recommended action",
        ]
        columns.remove(group_by)
        columns.insert(0, group_by)
        colors = {
            "High risk": "background-color: #FEE2E2; color: #991B1B",
            "Medium risk": "background-color: #FEF3C7; color: #92400E",
            "Low risk": "background-color: #DCFCE7; color: #166534",
            "Unknown": "background-color: #E2E8F0; color: #334155",
        }
        table = df[columns].reset_index(drop=True)
        st.dataframe(
            table.style.map(lambda value: colors.get(value, ""), subset=["Risk level"]),
            hide_index=True, width="stretch", height=440,
            column_config={
                "Breach risk (%)": st.column_config.ProgressColumn(
                    "Breach risk (%)", min_value=0, max_value=100, format="%.1f%%",
                ),
                "Description": st.column_config.TextColumn(width="large"),
                "Recommended action": st.column_config.TextColumn(width="large"),
            },
            alt="Tickets grouped by risk, priority, or SLA status with assignment and next actions",
        )
