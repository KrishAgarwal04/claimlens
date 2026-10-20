"""Tests for UI components, HTML renderers, and Plotly chart layouts."""

import pandas as pd
import pytest
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


def test_render_kpi_card():
    card = render_kpi_card("Total Tickets", "2,057", "Intake queue", "Active", "blue")
    assert "Total Tickets" in card
    assert "2,057" in card
    assert "Intake queue" in card
    assert "kpi-badge-blue" in card
    # Ensure zero leading indentation on lines to prevent markdown code block bug
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
    # Ensure zero leading indentation
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
    # Ensure zero leading indentation
    for line in html.split("\n"):
        assert not line.startswith("    ")


def test_open_queue_risk_chart_layout():
    df = pd.DataFrame({
        "breach_risk_pct": [12.0, 45.0, 78.0, 92.0],
        "risk_tier": ["🟢 Low (<40%)", "🟡 Medium (40-70%)", "🔴 High (>=70%)", "🔴 High (>=70%)"],
    })
    fig = create_open_queue_risk_chart(df)

    # Check title configuration
    assert fig.layout.title.text == "<b>Open Queue SLA Risk Distribution (250 Tickets)</b>"
    assert fig.layout.title.y == 0.96
    assert fig.layout.title.xanchor == "left"

    # Check legend configuration: horizontal, placed cleanly below title outside plotting area
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.yanchor == "top"
    assert fig.layout.legend.y == 0.86
    assert fig.layout.legend.xanchor == "left"

    # Check top margin accommodates both title and legend outside plotting area
    assert fig.layout.margin.t >= 75
    assert fig.layout.margin.b >= 45


def test_chart_theme_custom_margin_preservation():
    df = pd.DataFrame({
        "category": ["Payments", "Account Access", "Technical Support"],
        "count": [500, 300, 200],
    })
    fig = create_category_distribution_chart(df)
    # create_category_distribution_chart sets left margin to 150 to avoid clipping category labels
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
    # Ensure zero leading indentation
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


def test_apptest_section_3_operational_bottlenecks():
    """Verify Section 3 renders without unhandled exceptions via Streamlit AppTest."""
    import json
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Switch to Section 3: Operational Bottlenecks
    at.sidebar.radio[0].set_value("3. Operational Bottlenecks").run()
    assert len(at.exception) == 0
    # Dataframes rendered for bottlenecks and breakdowns
    assert len(at.dataframe) >= 2

    # Verify Ranked Bottlenecks table (dataframe 0) breach rate percentage formatting
    ranked_df_elem = at.dataframe[0]
    df_val = ranked_df_elem.value
    assert "Breach Rate (%)" in df_val.columns

    # Verify column config: ProgressColumn with format %.1f%% and max_value 100
    col_config = json.loads(ranked_df_elem.proto.columns)
    assert "Breach Rate (%)" in col_config
    breach_cfg = col_config["Breach Rate (%)"]["type_config"]
    assert breach_cfg["type"] == "progress"
    assert breach_cfg["format"] == "%.1f%%"
    assert breach_cfg["min_value"] == 0
    assert breach_cfg["max_value"] == 100

    # Verify percentage values (0-100 scale, approximately 90.8%, 88.3%, 89.5%)
    payout_row = df_val[df_val["Sub-Category"] == "Payout Delays"].iloc[0]
    assert payout_row["Sample N"] == 119
    assert payout_row["Breaches"] == 108
    assert abs(payout_row["Breach Rate (%)"] - 90.8) <= 0.1

    card_row = df_val[df_val["Sub-Category"] == "Card Declines"].iloc[0]
    assert card_row["Sample N"] == 111
    assert card_row["Breaches"] == 98
    assert abs(card_row["Breach Rate (%)"] - 88.3) <= 0.1

    auth_row = df_val[df_val["Sub-Category"] == "3DS Authentication"].iloc[0]
    assert auth_row["Sample N"] == 105
    assert auth_row["Breaches"] == 94
    assert abs(auth_row["Breach Rate (%)"] - 89.5) <= 0.1

    # Verify values are on percentage scale (all > 1.0 for these bottlenecks, not 0-1 proportions)
    assert (df_val["Breach Rate (%)"] > 1.0).all()


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

    # Test fallback card on NO_MATCH
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


def test_apptest_section_4_smart_agent_recommendation():
    """Verify Section 4 renders without unhandled exceptions via Streamlit AppTest."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Switch to Section 4: Smart Agent Recommendation
    at.sidebar.radio[0].set_value("4. Smart Agent Recommendation").run()
    assert len(at.exception) == 0
    # Open queue tables and candidate alternatives rendered
    assert len(at.dataframe) >= 2


def test_apptest_section_1_executive_overview():
    """Verify Section 1: Executive Overview renders without unhandled exceptions via AppTest."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Section 1 is default; verify triage dataframe rendered
    assert len(at.dataframe) >= 1


def test_apptest_section_2_sla_risk_predictor():
    """Verify Section 2: SLA Risk Predictor renders evaluation and triage tabs via AppTest."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Switch to Section 2: SLA Risk Predictor (Pre-Assignment)
    at.sidebar.radio[0].set_value("2. SLA Risk Predictor (Pre-Assignment)").run()
    assert len(at.exception) == 0

    # Tabs for Model Evaluation & Live Open Queue Triage
    assert len(at.tabs) >= 2
    # Benchmark table dataframe
    assert len(at.dataframe) >= 1


def test_apptest_section_5_what_if_simulator():
    """Verify Section 5: What-If Impact Simulator renders and handles slider interaction via AppTest."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Switch to Section 5: What-If Impact Simulator
    at.sidebar.radio[0].set_value("5. What-If Impact Simulator").run()
    assert len(at.exception) == 0

    # Sliders for target mismatch and domain headcount additions
    assert len(at.slider) >= 2
    # Baseline-vs-scenario comparison table rendered
    assert len(at.dataframe) >= 1

    # Verify slider sensitivity at 28.0% target mismatch
    at.slider[0].set_value(28.0).run()
    assert len(at.exception) == 0


def test_apptest_all_five_sections_smoke_tour():
    """Tour all 5 sections sequentially verifying zero exceptions and clean rendering across full app."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    sections = [
        "1. Executive Overview",
        "2. SLA Risk Predictor (Pre-Assignment)",
        "3. Operational Bottlenecks",
        "4. Smart Agent Recommendation",
        "5. What-If Impact Simulator",
    ]

    for sec in sections:
        at.sidebar.radio[0].set_value(sec).run()
        assert len(at.exception) == 0, f"Exception encountered while rendering {sec}"
        assert len(at.dataframe) >= 1, f"Expected at least 1 dataframe in {sec}"


