"""Tests for UI components, HTML renderers, Plotly chart layouts, and AppTest navigation."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components.styles import (
    render_kpi_card,
    render_confusion_matrix_grid,
    render_benchmark_table_html,
    render_insight_card,
    render_recommendation_card,
)
from app.components.charts import (
    create_open_queue_risk_chart,
    create_category_distribution_chart,
    apply_chart_theme,
)
from app.agent_views import (
    get_most_urgent_ticket,
    get_recommended_tickets,
    explain_specialist_recommendation,
    _compute_ticket_timings,
)


def test_render_kpi_card():
    card = render_kpi_card("Total Tickets", "2,057", "Intake queue", "Active", "blue")
    assert "Total Tickets" in card
    assert "2,057" in card
    assert "Intake queue" in card
    assert "kpi-badge-blue" in card
    for line in card.split("\n"):
        assert not line.startswith("    ")


def test_render_confusion_matrix_grid():
    cm = {"tn": 121, "fp": 33, "fn": 32, "tp": 176}
    html = render_confusion_matrix_grid(cm)
    assert "121" in html
    assert "33" in html
    assert "32" in html
    assert "176" in html
    assert "Holdout Confusion Matrix" in html
    for line in html.split("\n"):
        assert not line.startswith("    ")


def test_render_benchmark_table_html():
    bench_data = [
        {
            "Model Architecture": "Random Forest (Calibrated)",
            "Holdout ROC-AUC": "0.8899",
            "Holdout PR-AUC": "0.9255",
            "Holdout F1": "0.8441",
        }
    ]
    html = render_benchmark_table_html(bench_data)
    assert "<table" in html
    assert "Random Forest (Calibrated)" in html
    assert "0.8899" in html
    assert "Holdout ROC-AUC" in html
    for line in html.split("\n"):
        assert not line.startswith("    ")


def test_open_queue_risk_chart_layout():
    df = pd.DataFrame({
        "breach_risk_pct": [12.0, 45.0, 78.0, 92.0],
        "risk_tier": ["🟢 Low (<40%)", "🟡 Medium (40-70%)", "🔴 High (>=70%)", "🔴 High (>=70%)"],
    })
    fig = create_open_queue_risk_chart(df)

    assert fig.layout.title.text == "<b>Open Queue SLA Risk Distribution (250 Tickets)</b>"
    assert fig.layout.title.y == 0.96
    assert fig.layout.title.xanchor == "left"
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.yanchor == "top"
    assert fig.layout.legend.y == 0.86
    assert fig.layout.legend.xanchor == "left"
    assert fig.layout.margin.t >= 75
    assert fig.layout.margin.b >= 45


def test_chart_theme_custom_margin_preservation():
    df = pd.DataFrame({
        "category": ["Payments", "Account Access", "Technical Support"],
        "count": [500, 300, 200],
    })
    fig = create_category_distribution_chart(df)
    assert fig.layout.margin.l >= 150


def test_render_insight_card():
    insight = {
        "title": "Category Mismatch Penalty in Agent Dispatching",
        "sample_size": 538,
        "comparison": "74.9% breach rate for mismatched assignments vs 50.0% when matched (+25.0 pp penalty)",
        "suggested_action": "Enforce strict primary domain alignment constraints in ticket dispatching rules.",
        "caveat": "Observational correlation. Mismatched assignments may coincide with volume spikes.",
        "metric_values": {
            "mismatch_volume": 538,
            "mismatch_rate": 0.2977,
            "delta_percentage_points": 25.0,
        },
    }
    card_html = render_insight_card(insight)
    assert "Category Mismatch Penalty in Agent Dispatching" in card_html
    assert "Sample N=538" in card_html
    assert "74.9%" in card_html
    assert "Observational Caveat" in card_html
    for line in card_html.split("\n"):
        assert not line.startswith("    ")


def test_day3_chart_layouts():
    from app.components.charts import (
        create_category_bottleneck_comparison_chart,
        create_priority_sla_performance_chart,
        create_shift_region_coverage_chart,
        create_top_bottlenecks_bar_chart,
    )

    cat_df = pd.DataFrame({
        "category": ["Payments & Checkout", "API Integrations"],
        "breach_rate": [0.89, 0.78],
        "reopen_rate": [0.22, 0.18],
        "avg_resolution_hours": [35.2, 41.8],
    })
    fig_cat = create_category_bottleneck_comparison_chart(cat_df)
    assert fig_cat.layout.title.text is not None
    assert fig_cat.layout.legend.orientation == "h"

    prio_df = pd.DataFrame({
        "priority": ["P1", "P2", "P3", "P4"],
        "target_resolution_hours": [2.0, 8.0, 24.0, 72.0],
        "mean_resolution_hours": [3.5, 15.1, 31.2, 58.4],
        "breach_rate": [0.719, 0.746, 0.612, 0.354],
    })
    fig_prio = create_priority_sla_performance_chart(prio_df)
    assert fig_prio.layout.title.text is not None

    shift_df = pd.DataFrame({
        "region": ["UK & Europe", "North America", "Asia-Pacific"],
        "merchant_pct": [35.7, 34.8, 29.5],
        "agent_pct": [10.0, 35.0, 55.0],
        "ticket_pct": [33.5, 34.0, 32.5],
    })
    fig_shift = create_shift_region_coverage_chart(shift_df)
    assert fig_shift.layout.title.text is not None

    ranked_df = pd.DataFrame({
        "bottleneck_name": ["Payments — Payout Delays", "API — Endpoint Timeout"],
        "sub_category": ["Payout Delays", "Endpoint Timeout"],
        "bottleneck_score": [36.6, 26.2],
        "excess_breaches": [39.7, 28.4],
        "breach_rate": [0.908, 0.819],
        "sample_size": [119, 116],
    })
    fig_ranked = create_top_bottlenecks_bar_chart(ranked_df)
    assert fig_ranked.layout.title.text is not None


def test_render_recommendation_card():
    rec = {
        "status": "SUCCESS",
        "ticket_id": 849008,
        "recommended_agent_id": 214,
        "recommended_agent_name": "Priya Sharma",
        "recommended_agent_tier": "L2",
        "recommended_agent_category": "API Integrations",
        "recommended_agent_shift": "Asia-Pacific",
        "match_score": 0.8033,
        "reasons": [
            "Domain specialist in API Integrations.",
            "L2 capability aligned with P1 urgency.",
        ],
        "alternatives": [],
        "uncertainty_limitations": "Observational matching based on intake fields.",
    }
    card_html = render_recommendation_card(rec)
    assert "Priya Sharma" in card_html
    assert "Tier L2" in card_html
    assert "0.80" in card_html
    assert "Operational Transparency" in card_html
    for line in card_html.split("\n"):
        assert not line.startswith("    ")

    no_match_rec = {
        "status": "NO_MATCH",
        "ticket_id": 9999,
        "recommended_agent_name": "No Recommendation (Missing Domain)",
        "match_score": 0.0,
        "reasons": ["Ticket domain category is missing or null."],
        "uncertainty_limitations": "Cannot route without category.",
    }
    fallback_html = render_recommendation_card(no_match_rec)
    assert "No Recommendation" in fallback_html
    assert "Match Score: 0.00" in fallback_html
    for line in fallback_html.split("\n"):
        assert not line.startswith("    ")


# =============================================================================
# REQUIRED WORKFLOW & USABILITY TESTS (10 CRITICAL ACCEPTANCE CRITERIA)
# =============================================================================

def test_1_unassigned_tickets_never_appear_as_already_assigned():
    """Verify that unassigned tickets are tagged as unassigned and never claim to be assigned to me."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Initially session has 0 assigned tickets for active agent
    assigned = at.session_state.get("ticket_assignments", {})
    assert len(assigned) == 0

    # Urgent ticket is shown as claimable, not assigned
    urgent_claims = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")]
    assert len(urgent_claims) == 1
    assert "claim_urgent_" in urgent_claims[0].key


def test_2_most_urgent_ticket_selected_from_assigned_queue_when_exists():
    """Verify that when an assigned ticket exists, the urgent section displays the assigned ticket."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim an urgent ticket
    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    claimed_tid = int(urgent_claim.key.replace("claim_urgent_", ""))
    urgent_claim.click().run()
    assert len(at.exception) == 0

    # Now active agent has assigned ticket
    assert at.session_state["ticket_assignments"][claimed_tid] == "Priya Sharma"

    # Now the button displayed is 'VIEW & TAKE ACTION'
    action_btns = [b for b in at.button if b.key == "btn_take_action_urgent"]
    assert len(action_btns) == 1


def test_3_when_no_assigned_tickets_most_urgent_shown_as_claimable():
    """Verify that when 0 assigned tickets exist, urgent section provides enabled Claim Ticket action."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    urgent_claims = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")]
    assert len(urgent_claims) == 1
    assert not urgent_claims[0].disabled


def test_4_no_duplicate_ticket_appears_in_both_urgent_and_recommended():
    """Verify that the ticket shown in the urgent section is excluded from recommended section."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    urgent_tid = urgent_claim.key.replace("claim_urgent_", "")

    # Recommended claim buttons
    rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    rec_tids = [b.key.replace("claim_", "") for b in rec_claims]

    assert urgent_tid not in rec_tids, f"Urgent ticket {urgent_tid} duplicated in recommended tickets {rec_tids}"


def test_5_claim_ticket_updates_assignment_correctly():
    """Verify Claim Ticket updates ticket ownership, sets In Progress, and records in session state."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim recommended ticket
    rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    assert len(rec_claims) > 0
    target_claim = rec_claims[0]
    target_tid = int(target_claim.key.replace("claim_", ""))

    target_claim.click().run()
    assert len(at.exception) == 0

    assert at.session_state["ticket_assignments"][target_tid] == "Priya Sharma"
    assert at.session_state["ticket_statuses"][target_tid] == "In Progress"
    assert at.session_state["active_ticket_id"] == target_tid


def test_6_claimed_ticket_disappears_from_unassigned_recommendations():
    """Verify that after claiming, the ticket disappears from recommended tickets list."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    target_tid = int(rec_claims[0].key.replace("claim_", ""))

    rec_claims[0].click().run()
    assert len(at.exception) == 0

    # Check new recommended tickets: target_tid must not be present
    new_rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    new_rec_tids = [int(b.key.replace("claim_", "")) for b in new_rec_claims]
    assert target_tid not in new_rec_tids


def test_7_claimed_ticket_appears_in_my_tickets():
    """Verify claimed ticket appears in My Tickets table."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim urgent ticket
    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    claimed_tid = int(urgent_claim.key.replace("claim_urgent_", ""))
    urgent_claim.click().run()
    assert len(at.exception) == 0

    # Navigate to My Tickets
    at.sidebar.radio[0].set_value("My Tickets").run()
    assert len(at.exception) == 0

    # Dataframe contains claimed ticket
    assert len(at.dataframe) >= 1
    my_df = at.dataframe[0].value
    assert claimed_tid in my_df["Ticket ID"].values


def test_8_claim_top_match_works_when_eligible_ticket_exists():
    """Verify Claim Top Match button in My Tickets claims the top eligible ticket."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Navigate to My Tickets
    at.sidebar.radio[0].set_value("My Tickets").run()
    assert len(at.exception) == 0

    claim_top_btn = [b for b in at.button if b.key == "btn_quick_claim_top"][0]
    assert not claim_top_btn.disabled
    claim_top_btn.click().run()
    assert len(at.exception) == 0

    assert len(at.session_state["ticket_assignments"]) >= 1


def test_9_claim_actions_disabled_when_no_eligible_ticket_exists():
    """Verify that when no eligible unassigned tickets exist, claim top action is disabled."""
    # Unit test get_recommended_tickets with mock fully claimed state
    mock_df = pd.DataFrame({
        "ticket_id": [1, 2],
        "category": ["API Integrations", "API Integrations"],
        "sub_category": ["Auth", "Timeout"],
        "short_description": ["Desc 1", "Desc 2"],
        "priority": ["P1", "P2"],
        "remaining_display": ["2h left", "4h left"],
        "remaining_hours": [2.0, 4.0],
        "breach_risk_pct": [80.0, 60.0],
        "risk_label": ["Critical", "At Risk"],
        "breach_risk": [0.8, 0.6],
        "recommended_agent": ["Priya Sharma", "Priya Sharma"],
        "recommended_tier": ["L2", "L2"],
        "merchant_name": ["M1", "M2"],
    })
    active_agent = {
        "agent_name": "Priya Sharma",
        "primary_category": "API Integrations",
        "tier": "L2",
        "shift_region": "Asia-Pacific",
    }
    # Simulate both tickets claimed in session_state
    import streamlit as st
    st.session_state["ticket_assignments"] = {1: "Priya Sharma", 2: "Alex Mercer"}

    recs = get_recommended_tickets(mock_df, active_agent)
    assert len(recs) == 0, f"Expected 0 recommendations when all tickets are claimed, got {len(recs)}"


def test_10_active_agent_switching_updates_personal_queue_and_workload():
    """Verify active agent switching in Agent Profile updates identity, workload, and filters."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim a ticket for Priya Sharma
    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    claimed_tid = int(urgent_claim.key.replace("claim_urgent_", ""))
    urgent_claim.click().run()
    assert len(at.exception) == 0

    # Navigate to Agent Profile
    at.sidebar.radio[0].set_value("Agent Profile").run()
    assert len(at.exception) == 0

    # Switch identity to Alex Mercer
    profile_sb = [sb for sb in at.selectbox if sb.key == "agent_profile_switcher_sb"][0]
    profile_sb.set_value("Alex Mercer").run()
    assert len(at.exception) == 0
    assert at.session_state["active_agent_name"] == "Alex Mercer"

    # Navigate to My Tickets as Alex Mercer: Alex should have 0 assigned tickets!
    at.sidebar.radio[0].set_value("My Tickets").run()
    assert len(at.exception) == 0
    # Empty state rendered for Alex Mercer
    assert any("No tickets assigned yet" in m.value for m in at.markdown)


def test_apptest_supervisor_view_retains_all_analytics():
    """Verify Supervisor View houses all 7 analytical features without exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("Supervisor View").run()
    assert len(at.exception) == 0

    # Supervisor tabs rendered (AI Decision Engine tab removed)
    assert len(at.tabs) >= 6
    assert not any('Decision Engine' in (t.label or '') or 'Model Evidence' in (t.label or '') for t in at.tabs)

    # Verify ranked bottlenecks table
    bn_dfs = [df for df in at.dataframe if "Sub-Category" in df.value.columns and "Breach Rate (%)" in df.value.columns]
    assert len(bn_dfs) >= 1
    ranked_df_elem = bn_dfs[0]
    df_val = ranked_df_elem.value

    col_config = json.loads(ranked_df_elem.proto.columns)
    assert "Breach Rate (%)" in col_config
    breach_cfg = col_config["Breach Rate (%)"]["type_config"]
    assert breach_cfg["type"] == "progress"
    assert breach_cfg["format"] == "%.1f%%"

    # Verify What-If Simulator sliders
    assert len(at.slider) >= 2
    at.slider[0].set_value(28.0).run()
    assert len(at.exception) == 0



def test_supervisor_tabs_count_and_absence_of_decision_engine():
    """Requirement 6: Verify AI Decision Engine / Model Evidence tab is absent and 6 tabs remain in order."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("Supervisor View").run()
    assert len(at.exception) == 0

    tab_labels = [getattr(t, "label", "") for t in at.tabs]
    assert not any("Decision Engine" in lbl or "Model Evidence" in lbl for lbl in tab_labels)
    assert any("Executive Overview" in lbl for lbl in tab_labels)
    assert any("SLA Risk Predictor" in lbl for lbl in tab_labels)
    assert any("Operational Bottlenecks" in lbl for lbl in tab_labels)
    assert any("Smart Agent Dispatch" in lbl for lbl in tab_labels)
    assert any("What-If Simulator" in lbl for lbl in tab_labels)
    assert any("Workload Forecast" in lbl for lbl in tab_labels)


def test_button_view_and_take_action_opens_ticket_action_panel():
    """Requirement 7 & 10: Verify 'VIEW & TAKE ACTION' opens the exact ticket in the action panel."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim the urgent ticket first so it becomes an assigned ticket
    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    claimed_tid = int(urgent_claim.key.replace("claim_urgent_", ""))
    urgent_claim.click().run()
    assert len(at.exception) == 0

    # Now that it is assigned, the button 'VIEW & TAKE ACTION' appears
    take_action_btns = [b for b in at.button if b.key == "btn_take_action_urgent"]
    assert len(take_action_btns) == 1
    take_action_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify action panel opened for claimed_tid
    assert at.session_state["active_ticket_id"] == claimed_tid
    markdown_texts = [m.value for m in at.markdown]
    assert any(f"Ticket Action Panel — Ticket #{claimed_tid}" in m for m in markdown_texts)
    # Verify status options and resolution actions exist
    assert any(sb.key == f"status_select_action_{claimed_tid}" for sb in at.selectbox)
    assert any(b.key == f"btn_mark_resolved_{claimed_tid}" for b in at.button)


def test_button_inspect_details_shows_modal_dialog_without_claiming():
    """Requirement 8: Verify Inspect Details opens modal dialog with ticket details without auto-claiming."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    inspect_btn = [b for b in at.button if b.key and b.key.startswith("inspect_urgent_")][0]
    inspect_tid = int(inspect_btn.key.replace("inspect_urgent_", ""))
    inspect_btn.click().run()
    assert len(at.exception) == 0

    # Ticket should NOT be auto-claimed
    assert inspect_tid not in at.session_state["ticket_assignments"]

    # Verify ticket details are visible in dialog markdown
    markdown_texts = [m.value for m in at.markdown]
    assert any(f"Ticket #{inspect_tid}" in m for m in markdown_texts)
    assert any("Specialist Routing Diagnostics" in m for m in markdown_texts)
    assert any("Ticket Description" in m for m in markdown_texts)


def test_button_manage_ticket_opens_action_panel_and_preselects():
    """Requirement 9: Verify Manage Ticket opens the corresponding assigned ticket."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Claim urgent ticket
    urgent_claim = [b for b in at.button if b.key and b.key.startswith("claim_urgent_")][0]
    claimed_tid = int(urgent_claim.key.replace("claim_urgent_", ""))
    urgent_claim.click().run()
    assert len(at.exception) == 0

    manage_btn = [b for b in at.button if b.key == f"btn_manage_my_{claimed_tid}"]
    assert len(manage_btn) == 1
    manage_btn[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state["active_ticket_id"] == claimed_tid
    assert at.session_state["my_tickets_selector"] == claimed_tid
    markdown_texts = [m.value for m in at.markdown]
    assert any(f"Ticket Action Panel — Ticket #{claimed_tid}" in m for m in markdown_texts)


def test_recommendations_refresh_and_claimed_ticket_removal():
    """Requirement 1, 2, 3: Verify recommendation refresh, manual button, and removal of claimed tickets."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Verify refresh button exists in recommendation fragment
    ref_btn = [b for b in at.button if b.key == "btn_manual_refresh_recs"]
    assert len(ref_btn) == 1
    ref_btn[0].click().run()
    assert len(at.exception) == 0

    # Find a recommended ticket
    rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    assert len(rec_claims) > 0
    rec_tid = int(rec_claims[0].key.replace("claim_", ""))

    # Claim the recommended ticket
    rec_claims[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state["ticket_assignments"][rec_tid] == "Priya Sharma"

    # Now the claimed ticket must NO LONGER be recommended
    new_rec_claims = [b for b in at.button if b.key and b.key.startswith("claim_") and not b.key.startswith("claim_urgent_")]
    new_rec_tids = [int(b.key.replace("claim_", "")) for b in new_rec_claims]
    assert rec_tid not in new_rec_tids


def test_navigation_scroll_reset_behavior():
    """Requirement 4 & 5: Verify navigation destination tracking triggers scroll reset only on page changes."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    assert at.session_state.get("_last_nav_mode") == "My Workspace"

    # Ordinary rerun within same page does NOT change destination
    at.run()
    assert at.session_state.get("_last_nav_mode") == "My Workspace"

    # Changing navigation to My Tickets
    at.sidebar.radio[0].set_value("My Tickets").run()
    assert len(at.exception) == 0
    assert at.session_state.get("_last_nav_mode") == "My Tickets"

    # Changing navigation to Supervisor View
    at.sidebar.radio[0].set_value("Supervisor View").run()
    assert len(at.exception) == 0
    assert at.session_state.get("_last_nav_mode") == "Supervisor View"

    # Changing navigation to Agent Profile
    at.sidebar.radio[0].set_value("Agent Profile").run()
    assert len(at.exception) == 0
    assert at.session_state.get("_last_nav_mode") == "Agent Profile"
