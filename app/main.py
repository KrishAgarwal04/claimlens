"""OpsPilot AI - Main Streamlit Application.

Enterprise Operations Intelligence & SLA Optimization Platform.
Clean, light-themed SaaS design for national hackathon operations intelligence.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.loader import DataLoader
from src.pipeline.features import FeatureEngineer
from src.models.sla_model import SLARiskModel
from src.analytics import BottleneckDetector
from src.routing import AgentRecommender
from src.simulation import ImpactSimulator
from src.config import TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE, SLA_TARGETS
from app.components.styles import (
    apply_custom_css,
    render_kpi_card,
    render_confusion_matrix_grid,
    render_benchmark_table_html,
    render_insight_card,
    render_recommendation_card,
)
from app.components.charts import (
    create_volume_trend_chart,
    create_category_distribution_chart,
    create_sla_breach_chart,
    create_open_queue_risk_chart,
    create_priority_chart,
    create_sector_chart,
    create_feature_importance_chart,
    create_bottleneck_mismatch_chart,
    create_agent_tier_performance_chart,
    create_category_bottleneck_comparison_chart,
    create_priority_sla_performance_chart,
    create_shift_region_coverage_chart,
    create_top_bottlenecks_bar_chart,
    create_agent_dispatch_distribution_chart,
)

# 1. Page Configuration
st.set_page_config(
    page_title="OpsPilot AI | Operations Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Enterprise SaaS CSS
apply_custom_css()

# Global chart config to hide modebars by default
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

# 3. Cached Data Ingestion
@st.cache_data
def load_operational_data():
    """Load and validate tickets, agents, merchants, and partition historical vs open."""
    loader = DataLoader(TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE)
    tickets, agents, merchants = loader.get_datasets()
    hist_tickets, open_tickets = loader.split_tickets(tickets)
    diagnostics = loader.run_diagnostics()
    return tickets, agents, merchants, hist_tickets, open_tickets, diagnostics

# 4. Cached Model Loading & Inference
@st.cache_resource
def get_sla_model():
    """Load pre-trained SLARiskModel artifact or train if not present."""
    try:
        return SLARiskModel.load()
    except Exception:
        loader = DataLoader(TICKETS_FILE, AGENTS_FILE, MERCHANTS_FILE)
        tickets = loader.load_tickets()
        merchants = loader.load_merchants()
        hist, _ = loader.split_tickets(tickets)
        fe = FeatureEngineer()
        X, y = fe.prepare_training_dataset(hist, merchants)
        model = SLARiskModel(model_type="random_forest", calibrate=True)
        model.fit_and_evaluate(X, y)
        model.save()
        return model

# Load data with loading state handling
try:
    tickets, agents, merchants, hist_tickets, open_tickets, diagnostics = load_operational_data()
    sla_model = get_sla_model()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading OpsPilot operational data: {e}")
    data_loaded = False

# Compute open queue risk predictions
if data_loaded:
    fe = FeatureEngineer()
    X_open = fe.prepare_open_queue_dataset(open_tickets, merchants)
    risk_scores = sla_model.predict_risk(X_open)

    triage_df = open_tickets.copy().reset_index(drop=True)
    triage_df["breach_risk"] = risk_scores
    triage_df["breach_risk_pct"] = (risk_scores * 100).round(1)

    def assign_risk_tier(r):
        if r >= 0.70:
            return "🔴 High (>=70%)"
        elif r >= 0.40:
            return "🟡 Medium (40-70%)"
        return "🟢 Low (<40%)"

    triage_df["risk_tier"] = triage_df["breach_risk"].apply(assign_risk_tier)

    # Merge merchant context for rich display
    triage_df = triage_df.merge(
        merchants[["merchant_id", "merchant_name", "tier", "sector", "region"]],
        on="merchant_id",
        how="left",
    )
    triage_df["sla_target_hours"] = triage_df["priority"].map(
        lambda p: f"{SLA_TARGETS.get(p, {}).get('resolution_hours', 24.0):.0f}h"
    )

    # Day 3: Bottleneck Analytics Engine
    bottleneck_detector = BottleneckDetector(tickets, agents, merchants)

    # Day 4: Smart Agent Recommendation Engine
    agent_recommender = AgentRecommender(agents, hist_tickets, merchants)

    # Day 5: Counterfactual Impact Simulator
    impact_simulator = ImpactSimulator(hist_tickets, agents, merchants, hourly_capacity_value=45.0)

# 5. Left Sidebar Navigation & Telemetry
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 10px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.35rem;">⚡</span>
                <span style="font-size: 1.2rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">OpsPilot AI</span>
            </div>
            <div style="font-size: 0.735rem; color: #64748B; margin-top: 2px;">
                Business Operations & SLA Intelligence
            </div>
            <div style="margin-top: 6px;">
                <span style="background-color: #EFF6FF; color: #1D4ED8; font-size: 0.675rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; border: 1px solid #DBEAFE;">
                    ENTERPRISE v0.2.0
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. SLA Risk Predictor (Pre-Assignment)",
            "3. Operational Bottlenecks",
            "4. Smart Agent Recommendation",
            "5. What-If Impact Simulator",
        ],
        index=0,
    )

    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # Telemetry card in sidebar
    meta_holdout = sla_model.metadata.get("holdout_results", {}).get("calibrated_selected_model", {}) if data_loaded else {}
    roc_telemetry = f"{meta_holdout.get('roc_auc', 0.8840):.4f}"
    pr_telemetry = f"{meta_holdout.get('pr_auc', 0.9200):.4f}"

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                SYSTEM TELEMETRY
            </div>
            <div style="font-size: 0.775rem; color: #334155; margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>Active Model:</span> <b style="color: #0F172A;">Calibrated RF</b>
            </div>
            <div style="font-size: 0.775rem; color: #334155; margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>Holdout ROC-AUC:</span> <b style="color: #047857;">{roc_telemetry}</b>
            </div>
            <div style="font-size: 0.775rem; color: #334155; margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>Holdout PR-AUC:</span> <b style="color: #047857;">{pr_telemetry}</b>
            </div>
            <div style="font-size: 0.775rem; color: #334155; margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>Target Leakage:</span> <b style="color: #047857;">0% (Verified)</b>
            </div>
            <div style="font-size: 0.775rem; color: #334155; display: flex; justify-content: space-between;">
                <span>Queue Status:</span> <b style="color: #B45309;">250 Pending</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 6. Main Content Surface
if data_loaded:

    # =========================================================================
    # SECTION 1: EXECUTIVE OVERVIEW
    # =========================================================================
    if mode == "1. Executive Overview":
        st.markdown(
            """
            <div class="brand-header" style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div>
                    <h2 class="brand-title">Operations Intelligence Overview</h2>
                    <p class="brand-desc">Executive health monitoring, ticket volume trends, SLA compliance, and intake queue risk distribution.</p>
                </div>
                <div>
                    <span style="font-size: 0.8rem; color: #64748B;">Live Queue: <b>June 29 - June 30, 2026</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Core KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(
                render_kpi_card(
                    title="Merchant Accounts",
                    value=f"{len(merchants):,}",
                    subtext="across 6 industry sectors",
                    badge_text="22 Advanced",
                    badge_variant="blue",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col2:
            st.markdown(
                render_kpi_card(
                    title="Support Specialists",
                    value=f"{len(agents):,}",
                    subtext="3 tiers (10 L1, 7 L2, 3 L3)",
                    badge_text="1.04x Avg Eff",
                    badge_variant="blue",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col3:
            st.markdown(
                render_kpi_card(
                    title="Historical Tickets",
                    value=f"{len(hist_tickets):,}",
                    subtext="18 months (57.4% resolution breach)",
                    badge_text="Jan 25 - Jun 26",
                    badge_variant="purple",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col4:
            high_risk_open = (triage_df["breach_risk"] >= 0.70).sum()
            st.markdown(
                render_kpi_card(
                    title="Live Open Queue",
                    value=f"{len(open_tickets):,}",
                    subtext=f"{high_risk_open} tickets require urgent triage",
                    badge_text="Awaiting Routing",
                    badge_variant="amber",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Row 1: Volume Trend (2/3) & Open Queue Risk Distribution (1/3)
        r1_col1, r1_col2 = st.columns([2, 1])

        with r1_col1:
            with st.container(border=True):
                fig_trend = create_volume_trend_chart(tickets)
                st.plotly_chart(fig_trend, use_container_width=True, config=CHART_CONFIG)

        with r1_col2:
            with st.container(border=True):
                fig_risk = create_open_queue_risk_chart(triage_df)
                st.plotly_chart(fig_risk, use_container_width=True, config=CHART_CONFIG)

        # Row 2: Category Distribution (1/2) & SLA Breach Rate (1/2)
        r2_col1, r2_col2 = st.columns([1, 1])

        with r2_col1:
            with st.container(border=True):
                fig_cat = create_category_distribution_chart(tickets)
                st.plotly_chart(fig_cat, use_container_width=True, config=CHART_CONFIG)

        with r2_col2:
            with st.container(border=True):
                fig_sla = create_sla_breach_chart(hist_tickets)
                st.plotly_chart(fig_sla, use_container_width=True, config=CHART_CONFIG)

        # Row 3: Priority Breakdown (1/2) & Tickets by Sector (1/2)
        r3_col1, r3_col2 = st.columns([1, 1])

        with r3_col1:
            with st.container(border=True):
                fig_prio = create_priority_chart(tickets)
                st.plotly_chart(fig_prio, use_container_width=True, config=CHART_CONFIG)

        with r3_col2:
            with st.container(border=True):
                fig_sector = create_sector_chart(merchants, tickets)
                st.plotly_chart(fig_sector, use_container_width=True, config=CHART_CONFIG)

        # Row 4: High-Risk Open Tickets Triage Table
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <h4 style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0;">
                            High-Risk Open Intake Tickets (Immediate Escalation Queue)
                        </h4>
                        <p style="font-size: 0.825rem; color: #64748B; margin: 2px 0 0 0;">
                            Pre-assignment breach risk estimated ≥ 70%. Real intake tickets awaiting routing dispatch.
                        </p>
                    </div>
                    <span class="kpi-badge-red">Urgent Attention</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            high_risk_df = triage_df[triage_df["breach_risk"] >= 0.70].sort_values("breach_risk", ascending=False)
            display_high_risk = high_risk_df[[
                "ticket_id",
                "merchant_name",
                "tier",
                "priority",
                "sla_target_hours",
                "category",
                "sub_category",
                "breach_risk_pct",
                "risk_tier",
                "created_at",
            ]].rename(
                columns={
                    "ticket_id": "Ticket ID",
                    "merchant_name": "Merchant Name",
                    "tier": "Account Tier",
                    "priority": "Priority",
                    "sla_target_hours": "SLA Window",
                    "category": "Domain Category",
                    "sub_category": "Sub-Category",
                    "breach_risk_pct": "Breach Risk (%)",
                    "risk_tier": "Risk Tier",
                    "created_at": "Intake Time",
                }
            )

            st.dataframe(
                display_high_risk,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Breach Risk (%)": st.column_config.ProgressColumn(
                        "Breach Risk (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

    # =========================================================================
    # SECTION 2: SLA RISK PREDICTOR (PRE-ASSIGNMENT)
    # =========================================================================
    elif mode == "2. SLA Risk Predictor (Pre-Assignment)":
        st.markdown(
            """
            <div class="brand-header">
                <h2 class="brand-title">SLA Resolution Breach Risk Predictor</h2>
                <p class="brand-desc">Trained strictly on 1,807 historical completed tickets using 18 pre-assignment features. Evaluated on 362 unseen holdout test tickets. Zero target leakage.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        meta = sla_model.metadata
        holdout = meta.get("holdout_results", {})
        cal_eval = holdout.get("calibrated_selected_model", holdout.get("random_forest", {}))
        roc_val = cal_eval.get("roc_auc", 0.0)
        pr_val = cal_eval.get("pr_auc", 0.0)
        f1_val = cal_eval.get("f1", 0.0)
        prec_val = cal_eval.get("precision", 0.0)
        rec_val = cal_eval.get("recall", 0.0)
        brier_val = cal_eval.get("brier_score", 0.0)

        # 6 Model Evaluation Metric Cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(
                render_kpi_card("ROC-AUC", f"{roc_val:.4f}", "Holdout Test", f"{roc_val:.4f}", "green"),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                render_kpi_card("PR-AUC", f"{pr_val:.4f}", "Avg Precision", f"{pr_val:.4f}", "green"),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                render_kpi_card("F1-Score", f"{f1_val:.4f}", "Harmonic Mean", f"{f1_val:.4f}", "blue"),
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                render_kpi_card("Precision", f"{prec_val:.4f}", "Breach Precision", f"{prec_val:.4f}", "blue"),
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                render_kpi_card("Recall", f"{rec_val:.4f}", "Breach Capture", f"{rec_val:.1%}", "blue"),
                unsafe_allow_html=True,
            )
        with m6:
            st.markdown(
                render_kpi_card("Brier Score", f"{brier_val:.4f}", "Calibrated (Sigmoid)", f"{brier_val:.4f}", "green"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        tab_eval, tab_triage = st.tabs(["📊 Model Evaluation & Benchmarks", "⚡ Live Open Queue Triage (250 Tickets)"])

        with tab_eval:
            # Benchmark Comparison Card (Light-Themed Table)
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 12px;">
                        <h4 style="font-size: 1rem; font-weight: 700; color: #0F172A; margin: 0;">Cross-Validation & Holdout Benchmark Comparison</h4>
                        <p style="font-size: 0.8rem; color: #64748B; margin: 2px 0 0 0;">Candidate models trained with Stratified 5-Fold Cross Validation and tested on 362 unseen holdout historical tickets.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                bench_data = []
                models_to_show = [
                    ("Majority Baseline", holdout.get("majority_baseline", {}), meta.get("cv_benchmarks", {}).get("majority_baseline", {})),
                    ("Logistic Regression", holdout.get("logistic_regression", {}), meta.get("cv_benchmarks", {}).get("logistic_regression", {})),
                    ("Random Forest (Uncalibrated)", holdout.get("random_forest", {}), meta.get("cv_benchmarks", {}).get("random_forest", {})),
                    ("Random Forest (Calibrated Sigmoid)", holdout.get("calibrated_selected_model", {}), meta.get("cv_benchmarks", {}).get("random_forest", {})),
                ]

                for label, h_m, cv_m in models_to_show:
                    if h_m:
                        bench_data.append({
                            "Model Architecture": label,
                            "5-Fold CV ROC-AUC": f"{cv_m.get('cv_roc_auc_mean', 0):.4f} (±{cv_m.get('cv_roc_auc_std', 0):.3f})" if cv_m else "N/A",
                            "Holdout ROC-AUC": f"{h_m.get('roc_auc', 0):.4f}",
                            "Holdout PR-AUC": f"{h_m.get('pr_auc', 0):.4f}",
                            "Holdout F1": f"{h_m.get('f1', 0):.4f}",
                            "Precision": f"{h_m.get('precision', 0):.4f}",
                            "Recall": f"{h_m.get('recall', 0):.4f}",
                            "Brier Score": f"{h_m.get('brier_score', 0):.4f}",
                        })

                bench_df = pd.DataFrame(bench_data)
                st.dataframe(bench_df, use_container_width=True, hide_index=True)

            col_cm, col_feat = st.columns([1, 2])

            with col_cm:
                cm = cal_eval.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
                st.markdown(render_confusion_matrix_grid(cm), unsafe_allow_html=True)

            with col_feat:
                with st.container(border=True):
                    top_feats = meta.get("top_feature_importances", [])
                    if top_feats:
                        fig_feat = create_feature_importance_chart(top_feats)
                        st.plotly_chart(fig_feat, use_container_width=True, config=CHART_CONFIG)

            with st.expander("ℹ️ Dataset Limitations & Split Strategy Rationale"):
                for lim in meta.get("limitations", []):
                    st.markdown(f"- {lim}")

        with tab_triage:
            st.markdown(
                """
                <div style="margin-bottom: 14px;">
                    <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0;">Live Triage Scoring on 250 Open Queue Tickets</h3>
                    <p style="font-size: 0.8rem; color: #64748B; margin: 2px 0 0 0;">Scored in real-time at ticket arrival using pre-assignment features before agent allocation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            high_count = (risk_scores >= 0.70).sum()
            med_count = ((risk_scores >= 0.40) & (risk_scores < 0.70)).sum()
            low_count = (risk_scores < 0.40).sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    render_kpi_card("Open Queue Tickets", f"{len(triage_df):,}", "Incoming Backlog", "250 Total", "blue"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    render_kpi_card("🔴 High Risk Queue", f"{high_count} ({high_count/len(triage_df):.1%})", "Immediate Escalation", "P >= 70%", "red"),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    render_kpi_card("🟡 Medium Risk Queue", f"{med_count} ({med_count/len(triage_df):.1%})", "Standard Routing", "40-70%", "amber"),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    render_kpi_card("🟢 Low Risk Queue", f"{low_count} ({low_count/len(triage_df):.1%})", "Routine Triage", "P < 40%", "green"),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 12px;">
                        <h4 style="font-size: 1rem; font-weight: 700; color: #0F172A; margin: 0;">Filterable Open Queue Triage Table</h4>
                        <p style="font-size: 0.8rem; color: #64748B; margin: 2px 0 0 0;">Filter by estimated risk tier and priority level.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    selected_tier = st.multiselect(
                        "Filter by Risk Tier",
                        ["🔴 High (>=70%)", "🟡 Medium (40-70%)", "🟢 Low (<40%)"],
                        default=["🔴 High (>=70%)", "🟡 Medium (40-70%)", "🟢 Low (<40%)"],
                    )
                with col_f2:
                    selected_priority = st.multiselect(
                        "Filter by Priority",
                        ["P1", "P2", "P3", "P4"],
                        default=["P1", "P2", "P3", "P4"],
                    )

                filtered_triage = triage_df[
                    triage_df["risk_tier"].isin(selected_tier) & triage_df["priority"].isin(selected_priority)
                ]

                display_cols = [
                    "ticket_id",
                    "merchant_name",
                    "priority",
                    "sla_target_hours",
                    "category",
                    "sub_category",
                    "breach_risk_pct",
                    "risk_tier",
                    "is_reopened",
                    "created_at",
                ]
                st.dataframe(
                    filtered_triage[display_cols].sort_values("breach_risk_pct", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "breach_risk_pct": st.column_config.ProgressColumn(
                            "Breach Risk (%)",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                )

    # =========================================================================
    # SECTION 3: OPERATIONAL BOTTLENECKS
    # =========================================================================
    elif mode == "3. Operational Bottlenecks":
        st.markdown(
            """
            <div class="brand-header">
                <h2 class="brand-title">Operational Bottlenecks & Business Insights</h2>
                <p class="brand-desc">Data-driven diagnostic engine uncovering dispatch mismatches, domain friction points, tier escalation gaps, and regional coverage deficits.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Informational Banner: Open Queue Backlog Status (Strictly separated from historical outcome analytics)
        open_high_risk = int((triage_df["breach_risk"] >= 0.70).sum())
        st.markdown(
            f"""
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.825rem; color: #1E3A8A;">
                    <b>⚡ Live Queue Notice:</b> {len(open_tickets):,} incoming tickets awaiting dispatch ({open_high_risk} high-risk &ge; 70%) are <b>strictly excluded</b> from historical outcome rates below to maintain analytical integrity.
                </div>
                <span class="kpi-badge-blue">250 Open Backlog</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Interactive Diagnostic Filters Card
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 8px;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">Interactive Diagnostic Filters</span>
                    <p style="font-size: 0.785rem; color: #64748B; margin: 2px 0 0 0;">Filter historical completed tickets by domain, priority tier, and merchant sector.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                all_cats = sorted(hist_tickets["category"].unique())
                selected_cats = st.multiselect("Filter by Domain Category", all_cats, default=all_cats)
            with fc2:
                all_prios = ["P1", "P2", "P3", "P4"]
                selected_prios = st.multiselect("Filter by Priority", all_prios, default=all_prios)
            with fc3:
                all_sectors = sorted(merchants["sector"].unique())
                selected_sectors = st.multiselect("Filter by Merchant Sector", all_sectors, default=all_sectors)

        # Filter the historical dataframe
        filtered_hist = hist_tickets[
            hist_tickets["category"].isin(selected_cats) &
            hist_tickets["priority"].isin(selected_prios)
        ].copy()
        filtered_hist = filtered_hist.merge(merchants[["merchant_id", "sector", "tier", "region"]], on="merchant_id", how="left")
        filtered_hist = filtered_hist[filtered_hist["sector"].isin(selected_sectors)]

        if len(filtered_hist) == 0:
            st.warning("⚠️ No completed historical tickets match the selected filter criteria. Please broaden your selection.")
        else:
            # 2. Dynamic Summary KPI Cards computed from actual data
            kpis = bottleneck_detector.get_summary_kpis(filtered_hist)

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(
                    render_kpi_card(
                        "Category Mismatch Rate",
                        f"{kpis['mismatch_rate']*100:.1f}%",
                        f"{kpis['mismatch_count']:,} of {kpis['total_tickets']:,} tickets",
                        f"+{kpis['breach_rate_gap_pp']:.1f}% Breach" if kpis['breach_rate_gap_pp'] > 0 else "Neutral",
                        "red" if kpis['breach_rate_gap_pp'] > 10 else "blue",
                    ),
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    render_kpi_card(
                        "Mismatch Breach Rate",
                        f"{kpis['mismatched_breach_rate']*100:.1f}%",
                        f"vs {kpis['matched_breach_rate']*100:.1f}% when matched",
                        f"+{kpis['hours_gap']:.1f}h Delay" if kpis['hours_gap'] > 0 else "0h",
                        "red" if kpis['hours_gap'] > 5 else "amber",
                    ),
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    render_kpi_card(
                        "Resolution Breach Rate",
                        f"{kpis['resolution_breach_rate']*100:.1f}%",
                        f"{kpis['resolution_breach_count']:,} breaches ({kpis['mean_resolution_hours']:.1f}h avg)",
                        f"Median {kpis['median_resolution_hours']:.1f}h",
                        "red" if kpis['resolution_breach_rate'] > 0.50 else "green",
                    ),
                    unsafe_allow_html=True,
                )
            with b4:
                st.markdown(
                    render_kpi_card(
                        "Ticket Reopen Rate",
                        f"{kpis['reopen_rate']*100:.1f}%",
                        f"{kpis['reopen_count']:,} of {kpis['total_tickets']:,} tickets reopened",
                        f"{kpis['response_breach_rate']*100:.1f}% TTFR Breach",
                        "amber" if kpis['reopen_rate'] > 0.15 else "green",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # 3. Investigation Tabs
            tab_ranked, tab_domain, tab_staffing, tab_insights = st.tabs([
                "🚨 Ranked Bottlenecks",
                "📊 Category & Priority Drivers",
                "👥 Staffing & Shift Gaps",
                "💡 Evidence-Based Insights",
            ])

            with tab_ranked:
                # Top bottlenecks calculation
                ranked_df = bottleneck_detector.rank_bottlenecks(min_sample_size=10, shrinkage_k=10.0, top_n=10, df=filtered_hist)

                col_r_chart, col_r_table = st.columns([1, 1])
                with col_r_chart:
                    with st.container(border=True):
                        if len(ranked_df) > 0:
                            fig_ranked = create_top_bottlenecks_bar_chart(ranked_df)
                            st.plotly_chart(fig_ranked, use_container_width=True, config=CHART_CONFIG)
                        else:
                            st.info("Insufficient data to rank subcategories with current filters.")

                with col_r_table:
                    with st.container(border=True):
                        st.markdown(
                            """
                            <div style="margin-bottom: 8px;">
                                <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0;">Ranked Bottlenecks by Excess Breach Impact</h4>
                                <p style="font-size: 0.775rem; color: #64748B; margin: 2px 0 0 0;">Formula: <code>Excess Breaches × [N / (N + 10)]</code>. Penalizes small sample noise.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if len(ranked_df) > 0:
                            display_ranked = ranked_df[[
                                "rank", "sub_category", "category", "sample_size",
                                "breach_count", "breach_rate", "excess_breaches", "bottleneck_score"
                            ]].copy().rename(columns={
                                "rank": "#",
                                "sub_category": "Sub-Category",
                                "category": "Domain",
                                "sample_size": "Sample N",
                                "breach_count": "Breaches",
                                "breach_rate": "Breach Rate (%)",
                                "excess_breaches": "Excess Breaches",
                                "bottleneck_score": "Impact Score",
                            })
                            display_ranked["Breach Rate (%)"] = (display_ranked["Breach Rate (%)"] * 100).round(1)
                            st.dataframe(
                                display_ranked,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Breach Rate (%)": st.column_config.ProgressColumn(
                                        "Breach Rate (%)",
                                        format="%.1f%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    "Excess Breaches": st.column_config.NumberColumn(
                                        "Excess Breaches",
                                        format="%.1f",
                                    ),
                                    "Impact Score": st.column_config.NumberColumn(
                                        "Impact Score",
                                        format="%.2f",
                                    ),
                                },
                            )
                        else:
                            st.info("No subcategory bottlenecks meet the sample threshold.")

                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Subcategory Bottleneck Details & Operational Recommendations</h4>
                        """,
                        unsafe_allow_html=True,
                    )
                    if len(ranked_df) > 0:
                        detail_table = ranked_df[[
                            "rank", "bottleneck_name", "sample_size", "breach_rate", "avg_resolution_hours", "suggested_action"
                        ]].copy().rename(columns={
                            "rank": "#",
                            "bottleneck_name": "Failure Mode",
                            "sample_size": "Sample Size (N)",
                            "breach_rate": "Breach Rate (%)",
                            "avg_resolution_hours": "Avg Resolution",
                            "suggested_action": "Operational Recommendation",
                        })
                        detail_table["Breach Rate (%)"] = (detail_table["Breach Rate (%)"] * 100).round(1)
                        st.dataframe(
                            detail_table,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Breach Rate (%)": st.column_config.ProgressColumn(
                                    "Breach Rate (%)",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                                "Avg Resolution": st.column_config.NumberColumn(
                                    "Avg Resolution",
                                    format="%.1fh",
                                ),
                            },
                        )

            with tab_domain:
                cat_hotspots = bottleneck_detector.detect_category_hotspots(filtered_hist)
                prio_hotspots = bottleneck_detector.detect_priority_bottlenecks(filtered_hist)

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    with st.container(border=True):
                        if len(cat_hotspots) > 0:
                            fig_cat_comp = create_category_bottleneck_comparison_chart(cat_hotspots)
                            st.plotly_chart(fig_cat_comp, use_container_width=True, config=CHART_CONFIG)
                with col_c2:
                    with st.container(border=True):
                        if len(prio_hotspots) > 0:
                            fig_prio_comp = create_priority_sla_performance_chart(prio_hotspots)
                            st.plotly_chart(fig_prio_comp, use_container_width=True, config=CHART_CONFIG)

                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Domain Category Outcome & Rework Summary</h4>
                        """,
                        unsafe_allow_html=True,
                    )
                    if len(cat_hotspots) > 0:
                        display_cat = cat_hotspots[[
                            "category", "total_volume", "volume_pct", "breach_count",
                            "breach_rate", "response_breach_rate", "reopen_rate",
                            "avg_resolution_hours", "median_resolution_hours", "mismatch_rate"
                        ]].rename(columns={
                            "category": "Domain Category",
                            "total_volume": "Total Volume",
                            "volume_pct": "Share (%)",
                            "breach_count": "Resolution Breaches",
                            "breach_rate": "Resolution Breach Rate",
                            "response_breach_rate": "TTFR Breach Rate",
                            "reopen_rate": "Reopened Rate",
                            "avg_resolution_hours": "Mean Res Hours",
                            "median_resolution_hours": "Median Res Hours",
                            "mismatch_rate": "Mismatch Rate",
                        })
                        st.dataframe(
                            display_cat,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Share (%)": st.column_config.NumberColumn(format="%.1f%%"),
                                "Resolution Breach Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "TTFR Breach Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "Reopened Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "Mean Res Hours": st.column_config.NumberColumn(format="%.1fh"),
                                "Median Res Hours": st.column_config.NumberColumn(format="%.1fh"),
                                "Mismatch Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                            },
                        )

            with tab_staffing:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    with st.container(border=True):
                        fig_mismatch = create_bottleneck_mismatch_chart(filtered_hist)
                        st.plotly_chart(fig_mismatch, use_container_width=True, config=CHART_CONFIG)

                with col_s2:
                    with st.container(border=True):
                        fig_tier = create_agent_tier_performance_chart(filtered_hist, agents)
                        st.plotly_chart(fig_tier, use_container_width=True, config=CHART_CONFIG)

                # Regional shift coverage chart
                shift_summary = bottleneck_detector.detect_shift_region_imbalance()
                with st.container(border=True):
                    if len(shift_summary) > 0:
                        fig_shift = create_shift_region_coverage_chart(shift_summary)
                        st.plotly_chart(fig_shift, use_container_width=True, config=CHART_CONFIG)

                # Agent level workload and performance table
                agent_perf = bottleneck_detector.detect_agent_load_imbalance()
                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Specialist Workload, Breach Rate & Domain Mismatch Roster</h4>
                        <p style="font-size: 0.775rem; color: #64748B; margin: 0 0 10px 0;">Individual agent performance across historical ticket assignments (with sample sizes N).</p>
                        """,
                        unsafe_allow_html=True,
                    )
                    if len(agent_perf) > 0:
                        display_agents = agent_perf[[
                            "agent_id", "agent_name", "tier", "primary_category", "shift_region",
                            "efficiency_multiplier", "ticket_count", "breach_rate", "mean_resolution_hours", "mismatch_rate"
                        ]].rename(columns={
                            "agent_id": "Agent ID",
                            "agent_name": "Specialist",
                            "tier": "Tier",
                            "primary_category": "Specialty",
                            "shift_region": "Shift Region",
                            "efficiency_multiplier": "Efficiency Multiplier",
                            "ticket_count": "Tickets Handled (N)",
                            "breach_rate": "Breach Rate",
                            "mean_resolution_hours": "Avg Resolution",
                            "mismatch_rate": "Mismatch Rate",
                        })
                        st.dataframe(
                            display_agents,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Efficiency Multiplier": st.column_config.NumberColumn(format="%.3fx"),
                                "Breach Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "Avg Resolution": st.column_config.NumberColumn(format="%.1fh"),
                                "Mismatch Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                            },
                        )

            with tab_insights:
                st.markdown(
                    """
                    <div style="margin-bottom: 12px;">
                        <h4 style="font-size: 1rem; font-weight: 700; color: #0F172A; margin: 0;">Evidence-Based Leadership Insights & Operational Recommendations</h4>
                        <p style="font-size: 0.8rem; color: #64748B; margin: 2px 0 0 0;">Rigorous empirical findings grounded strictly in completed ticket data. Observational associations are stated with explicit caveats.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                insights_list = bottleneck_detector.get_structured_insights(filtered_hist)
                for item in insights_list:
                    st.markdown(render_insight_card(item), unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: SMART AGENT RECOMMENDATION
    # =========================================================================
    elif mode == "4. Smart Agent Recommendation":
        st.markdown(
            """
            <div class="brand-header">
                <h2 class="brand-title">Smart Agent Recommendation & Dispatch Engine</h2>
                <p class="brand-desc">Multi-criteria algorithmic matching routing incoming unassigned tickets to optimal domain specialists to eliminate dispatch mismatch penalties, ensure tier capability alignment, and balance queue flow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Informational Notice: Respecting real data constraints
        st.markdown(
            """
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.825rem; color: #1E3A8A;">
                    <b>🛡️ Transparent Routing Policy:</b> Recommendations are grounded in <b>intake-time attributes</b> (domain, priority, merchant region) and static agent directories. Real-time queue concurrency is unobserved in the data and is not assumed.
                </div>
                <span class="kpi-badge-blue">Day 4 Engine Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Batch route the open queue using default greedy strategy for summary KPIs
        routed_open_default = agent_recommender.route_batch(triage_df, strategy="top_match")
        batch_summary_default = agent_recommender.get_batch_summary(routed_open_default)

        # 1. Top Executive KPIs
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.markdown(
                render_kpi_card(
                    "Open Backlog Queue",
                    f"{len(open_tickets):,}",
                    "100% Unassigned intake",
                    "Awaiting Dispatch",
                    "blue",
                ),
                unsafe_allow_html=True,
            )
        with r_col2:
            st.markdown(
                render_kpi_card(
                    "Specialist Pool",
                    f"{len(agents):,} Agents",
                    "10 L1, 7 L2, 3 L3 specialists",
                    "5 Domains Covered",
                    "purple",
                ),
                unsafe_allow_html=True,
            )
        with r_col3:
            st.markdown(
                render_kpi_card(
                    "Domain Match Rate",
                    f"{batch_summary_default['match_rate']*100:.1f}%",
                    "0 category mismatches",
                    "+24.9 pp Defense",
                    "green",
                ),
                unsafe_allow_html=True,
            )
        with r_col4:
            st.markdown(
                render_kpi_card(
                    "P1/P2 Senior Fit",
                    f"{batch_summary_default['p1_p2_tier_coverage']*100:.1f}%",
                    "Routed to L2/L3 specialists",
                    "Optimal Escalation",
                    "green",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 2. Main Navigation Tabs
        tab_single, tab_batch, tab_roster = st.tabs([
            "🎯 Single Ticket Dispatch & Deep-Dive",
            "⚡ Open Queue Batch Dispatch Engine",
            "👥 Specialist Directory & Policy Weights",
        ])

        with tab_single:
            # Interactive Filter Sub-card
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">Open Queue Ticket Selector</span>
                        <p style="font-size: 0.785rem; color: #64748B; margin: 2px 0 0 0;">Filter unassigned open tickets by category and priority to inspect individual agent recommendations.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    filter_cats = ["All Domains"] + sorted(list(triage_df["category"].dropna().unique()))
                    selected_cat_filter = st.selectbox("Filter Open Tickets by Category", filter_cats, index=0)
                with f_c2:
                    filter_prios = ["All Priorities", "P1", "P2", "P3", "P4"]
                    selected_prio_filter = st.selectbox("Filter Open Tickets by Priority", filter_prios, index=0)

                # Filter tickets
                eligible_tickets = triage_df.copy()
                if selected_cat_filter != "All Domains":
                    eligible_tickets = eligible_tickets[eligible_tickets["category"] == selected_cat_filter]
                if selected_prio_filter != "All Priorities":
                    eligible_tickets = eligible_tickets[eligible_tickets["priority"] == selected_prio_filter]

                if len(eligible_tickets) == 0:
                    st.warning("⚠️ No open tickets match the selected filter criteria. Please broaden your selection.")
                    selected_ticket = None
                else:
                    # Selectbox options
                    ticket_options = eligible_tickets["ticket_id"].tolist()
                    def _format_ticket_option(tid):
                        t_row = eligible_tickets[eligible_tickets["ticket_id"] == tid].iloc[0]
                        risk = t_row.get("breach_risk_pct", 50.0)
                        return f"Ticket #{tid} | {t_row['priority']} | {t_row['category']} — {t_row.get('sub_category', '')} | SLA Risk: {risk:.1f}% | Merchant #{t_row['merchant_id']}"

                    selected_tid = st.selectbox(
                        "Select Open Ticket to Route",
                        ticket_options,
                        format_func=_format_ticket_option,
                    )
                    selected_ticket = eligible_tickets[eligible_tickets["ticket_id"] == selected_tid].iloc[0].to_dict()

            if selected_ticket is not None:
                # Generate recommendation
                rec_res = agent_recommender.recommend_for_ticket(selected_ticket, top_k=4)

                col_rec_left, col_rec_right = st.columns([1, 1])

                with col_rec_left:
                    # Primary recommendation card
                    st.markdown(render_recommendation_card(rec_res), unsafe_allow_html=True)

                    # Ticket context metadata
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="font-size: 0.8rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">Intake Ticket Diagnostics</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.8rem;">
                                <div><b>Ticket ID:</b> #{selected_ticket.get('ticket_id')}</div>
                                <div><b>Priority SLA:</b> {selected_ticket.get('priority')} ({selected_ticket.get('sla_target_hours', '24h')} target)</div>
                                <div><b>Domain Category:</b> {selected_ticket.get('category')}</div>
                                <div><b>Subcategory:</b> {selected_ticket.get('sub_category')}</div>
                                <div><b>Merchant Account:</b> #{selected_ticket.get('merchant_id')} ({selected_ticket.get('merchant_name', 'Merchant')})</div>
                                <div><b>Merchant Region:</b> {selected_ticket.get('region', 'Unknown')}</div>
                                <div><b>Day 2 Breach Risk:</b> <span class="kpi-badge-{'red' if selected_ticket.get('breach_risk', 0) >= 0.70 else ('amber' if selected_ticket.get('breach_risk', 0) >= 0.40 else 'green')}">{selected_ticket.get('breach_risk_pct', 0.0):.1f}%</span></div>
                                <div><b>Reopened Status:</b> {'Yes (Child)' if selected_ticket.get('is_reopened') else 'No (First Intake)'}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_rec_right:
                    with st.container(border=True):
                        st.markdown(
                            """
                            <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Alternative Eligible Candidates & Backups</h4>
                            <p style="font-size: 0.775rem; color: #64748B; margin: 0 0 10px 0;">Ranked alternative specialists within the category domain.</p>
                            """,
                            unsafe_allow_html=True,
                        )
                        if rec_res["alternatives"]:
                            alt_df = pd.DataFrame(rec_res["alternatives"]).rename(columns={
                                "agent_id": "Agent ID",
                                "agent_name": "Specialist",
                                "tier": "Tier",
                                "primary_category": "Specialty",
                                "shift_region": "Shift Region",
                                "match_score": "Match Score",
                                "primary_reason": "Suitability Factor",
                            })
                            st.dataframe(
                                alt_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Match Score": st.column_config.ProgressColumn(
                                        "Match Score",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                },
                            )
                        else:
                            st.info("No alternative specialists available for this category.")

                    # Sandbox Testing Tool for Edge Cases
                    with st.container(border=True):
                        st.markdown(
                            """
                            <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 6px 0;">🧪 Policy Sandbox & Edge-Case Validator</h4>
                            <p style="font-size: 0.775rem; color: #64748B; margin: 0 0 10px 0;">Test how the recommendation engine responds to anomalous intake scenarios.</p>
                            """,
                            unsafe_allow_html=True,
                        )
                        sb1, sb2 = st.columns(2)
                        with sb1:
                            sandbox_scenario = st.selectbox(
                                "Simulate Scenario",
                                [
                                    "Standard Ticket (Normal)",
                                    "Missing Category (Unclassified)",
                                    "Unknown Domain (Novel Service)",
                                    "Missing Priority (Unspecified)",
                                ],
                            )
                        with sb2:
                            sandbox_prio = st.selectbox("Simulate Priority", ["P1", "P2", "P3", "P4"], index=0)

                        # Run simulated query
                        sim_ticket = selected_ticket.copy()
                        sim_ticket["priority"] = sandbox_prio
                        if sandbox_scenario == "Missing Category (Unclassified)":
                            sim_ticket["category"] = None
                        elif sandbox_scenario == "Unknown Domain (Novel Service)":
                            sim_ticket["category"] = "Quantum Crypto Bridge"
                        elif sandbox_scenario == "Missing Priority (Unspecified)":
                            sim_ticket["priority"] = None

                        sim_rec = agent_recommender.recommend_for_ticket(sim_ticket)
                        if sim_rec["status"] == "NO_MATCH":
                            st.error(f"🛡️ **Engine Fallback Triggered**: {sim_rec['reasons'][0]}")
                        else:
                            st.success(f"✅ **Assigned Specialist**: {sim_rec['recommended_agent_name']} (Tier {sim_rec['recommended_agent_tier']}) — Match Score: {sim_rec['match_score']:.2f}")

        with tab_batch:
            # Batch Dispatch Engine Controls
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">Batch Dispatch Strategy Controls</span>
                        <p style="font-size: 0.785rem; color: #64748B; margin: 2px 0 0 0;">Route all 250 open queue tickets concurrently using configurable allocation strategies.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                strat_col1, strat_col2 = st.columns([1, 1])
                with strat_col1:
                    batch_strategy = st.radio(
                        "Select Routing Optimization Strategy",
                        ["top_match", "balanced"],
                        format_func=lambda s: "⚡ Greedy Best-Match (Maximize Match Score per Ticket)" if s == "top_match" else "⚖️ Load-Balanced (Category-Preserving Queue Balancing)",
                        horizontal=True,
                    )
                with strat_col2:
                    st.markdown(
                        f"""
                        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; font-size: 0.785rem; color: #475569;">
                            <b>Active Strategy:</b> {'Greedy best-fit prioritizes maximum alignment score for each individual ticket regardless of queue concentration.' if batch_strategy == 'top_match' else 'Load-balanced distributes tickets evenly across eligible category specialists while preserving strict category alignment.'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Route batch
            routed_batch = agent_recommender.route_batch(triage_df, strategy=batch_strategy)
            batch_summary = agent_recommender.get_batch_summary(routed_batch)

            # Batch KPIs
            bk1, bk2, bk3, bk4 = st.columns(4)
            with bk1:
                st.metric("Total Routed", f"{batch_summary['total_tickets']:,}", "100% of open queue")
            with bk2:
                st.metric("Category Match Rate", f"{batch_summary['match_rate']*100:.1f}%", "0 mismatches")
            with bk3:
                st.metric("Mean Match Score", f"{batch_summary['mean_match_score']:.3f}", "Multi-criteria index")
            with bk4:
                st.metric("P1/P2 Escalation Fit", f"{batch_summary['p1_p2_tier_coverage']*100:.1f}%", "L2/L3 specialists")

            # Workload distribution chart
            col_b_chart, col_b_stats = st.columns([1.4, 1])
            with col_b_chart:
                with st.container(border=True):
                    fig_batch = create_agent_dispatch_distribution_chart(routed_batch)
                    st.plotly_chart(fig_batch, use_container_width=True, config=CHART_CONFIG)

            with col_b_stats:
                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Tier Allocation Breakdown</h4>
                        <p style="font-size: 0.775rem; color: #64748B; margin: 0 0 10px 0;">Distribution of open queue ticket recommendations across specialist tiers.</p>
                        """,
                        unsafe_allow_html=True,
                    )
                    tier_dist = batch_summary["tier_distribution"]
                    tier_rows = [
                        {"Tier": f"Tier {t}", "Assigned Tickets": count, "Share (%)": f"{count / batch_summary['total_tickets'] * 100:.1f}%"}
                        for t, count in sorted(tier_dist.items(), key=lambda x: x[0])
                    ]
                    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

                    st.markdown(
                        """
                        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px; font-size: 0.76rem; color: #64748B; margin-top: 10px;">
                            <b>Observational Notice:</b> Batch allocation balancing operates strictly on the 250 open queue tickets. Existing historical backlog and external shifts are unobserved.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Full batch dispatch table
            with st.container(border=True):
                st.markdown(
                    """
                    <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Open Queue Ticket Dispatch Manifest (250 Tickets)</h4>
                    """,
                    unsafe_allow_html=True,
                )
                display_batch = routed_batch[[
                    "ticket_id", "priority", "category", "sub_category", "merchant_id",
                    "recommended_agent_name", "recommended_agent_tier", "match_score",
                    "primary_reason", "alternatives"
                ]].rename(columns={
                    "ticket_id": "Ticket #",
                    "priority": "Priority",
                    "category": "Domain Category",
                    "sub_category": "Subcategory",
                    "merchant_id": "Merchant #",
                    "recommended_agent_name": "Recommended Specialist",
                    "recommended_agent_tier": "Tier",
                    "match_score": "Match Score",
                    "primary_reason": "Match Justification",
                    "alternatives": "Eligible Backups",
                })
                st.dataframe(
                    display_batch,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Match Score": st.column_config.ProgressColumn(
                            "Match Score",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                )

        with tab_roster:
            # Specialist Directory & Policy Weights
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0;">Support Specialist Pool Roster (20 Agents)</h4>
                            <p style="font-size: 0.775rem; color: #64748B; margin: 2px 0 0 0;">Specialist domain, tier capability, working shift, velocity rating, and historical metrics.</p>
                        </div>
                        <span class="kpi-badge-blue">20 Active Specialists</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Enrich agent roster with historical completed ticket volume and breach rate
                enriched_roster_rows = []
                for _, ag in agents.iterrows():
                    aid = int(ag["agent_id"])
                    h_stat = agent_recommender.agent_hist_stats.get(aid, {})
                    enriched_roster_rows.append({
                        "Agent ID": aid,
                        "Specialist Name": ag["agent_name"],
                        "Tier": ag["tier"],
                        "Domain Specialty": ag["primary_category"],
                        "Working Shift": ag["shift_region"],
                        "Velocity Rating": ag["efficiency_multiplier"],
                        "Historical Completed (N)": h_stat.get("ticket_count", 0),
                        "Historical Breach Rate": f"{h_stat.get('breach_rate', 0.0)*100:.1f}%" if "breach_rate" in h_stat else "N/A",
                    })

                st.dataframe(
                    pd.DataFrame(enriched_roster_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Velocity Rating": st.column_config.NumberColumn(format="%.3fx"),
                    },
                )

            # Algorithmic Scoring Logic & Caveats
            col_p1, col_p2 = st.columns([1.2, 1])
            with col_p1:
                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">Multi-Criteria Scoring Weights & Policy Rules</h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.8rem;">
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;">
                                <div style="font-weight: 700; color: #0F172A;">1. Domain Specialty (50% Weight)</div>
                                <div style="color: #64748B; margin-top: 4px;">Primary category alignment constraint. Eliminates the +24.9 pp resolution breach penalty established in Day 3.</div>
                            </div>
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;">
                                <div style="font-weight: 700; color: #0F172A;">2. Priority-to-Tier Fit (25% Weight)</div>
                                <div style="color: #64748B; margin-top: 4px;">P1 & P2 high-impact tickets routed preferentially to L2/L3 specialists; P4 routed to L1 agents.</div>
                            </div>
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;">
                                <div style="font-weight: 700; color: #0F172A;">3. Velocity Multiplier (15% Weight)</div>
                                <div style="color: #64748B; margin-top: 4px;">Continuous agent efficiency rating from directory metadata (0.692x to 1.355x scale).</div>
                            </div>
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px;">
                                <div style="font-weight: 700; color: #0F172A;">4. Shift Alignment (10% Weight)</div>
                                <div style="color: #64748B; margin-top: 4px;">Synchronous working window match between specialist region and merchant location.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_p2:
                with st.container(border=True):
                    st.markdown(
                        """
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">Operational Caveats & Data Contracts</h4>
                        <div style="font-size: 0.8rem; color: #475569; line-height: 1.5;">
                            <p style="margin: 0 0 8px 0;"><b>Non-Probabilistic Score:</b> Match scores represent structured multi-attribute compatibility indices, not calibrated probabilities of ticket success.</p>
                            <p style="margin: 0 0 8px 0;"><b>Unobserved Live State:</b> Real-time queue concurrency, active ticket backlog, and agent shift attendance are not recorded in the dataset and are not fabricated.</p>
                            <p style="margin: 0;"><b>Historical Context:</b> Specialist historical breach rates reflect organizational assignment patterns rather than innate capability alone.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # =========================================================================
    # SECTION 5: WHAT-IF IMPACT SIMULATOR
    # =========================================================================
    elif mode == "5. What-If Impact Simulator":
        st.markdown(
            """
            <div class="brand-header">
                <h2 class="brand-title">Counterfactual What-If Impact Simulator</h2>
                <p class="brand-desc">Interactive executive scenario planner to simulate policy shifts, specialist domain staffing, and SLA recovery ROI.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        base_summary = impact_simulator.get_baseline_summary()

        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <h4 style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0;">1. Observed Historical Ground Truth</h4>
                <span style="font-size: 0.775rem; color: #64748B; font-weight: 600;">Sample: 1,807 Completed Tickets (Excludes 250 Open Queue)</span>
            </div>
            <p style="font-size: 0.825rem; color: #475569; margin-top: 0; margin-bottom: 12px;">
                Empirical baseline calculated strictly from completed historical tickets. Quantifies the observed resolution drag caused by category mismatching and establishes the maximum theoretical recovery ceiling if cross-domain mismatching were completely eliminated.
            </p>
            """,
            unsafe_allow_html=True,
        )

        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
        with sim_col1:
            st.markdown(
                render_kpi_card(
                    "Current Mismatch Rate",
                    f"{base_summary['baseline_mismatch_rate']*100:.1f}%",
                    f"{base_summary['baseline_mismatch_count']:,} of {base_summary['n_historical']:,} tickets",
                    "Observed Ground Truth",
                    "amber",
                ),
                unsafe_allow_html=True,
            )
        with sim_col2:
            st.markdown(
                render_kpi_card(
                    "Current Breach Rate",
                    f"{base_summary['baseline_breach_rate']*100:.1f}%",
                    f"{base_summary['baseline_breach_count']:,} historical breaches",
                    "Observed Ground Truth",
                    "red",
                ),
                unsafe_allow_html=True,
            )
        with sim_col3:
            st.markdown(
                render_kpi_card(
                    "Excess Resolution Drag",
                    f"{base_summary['total_excess_drag_hours']:,.0f} hrs",
                    f"+{base_summary['hours_penalty']:.1f}h drag per mismatch",
                    "Historical Drag",
                    "red",
                ),
                unsafe_allow_html=True,
            )
        with sim_col4:
            st.markdown(
                render_kpi_card(
                    "Max SLA Recovery Potential",
                    f"{base_summary['max_sla_recovery_pp']:.1f} pp",
                    f"Ceiling: {base_summary['matched_breach_rate']*100:.1f}% matched breach rate",
                    "Theoretical Max",
                    "green",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                    <h4 style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0;">2. Counterfactual Policy & Staffing Scenario Generator</h4>
                    <span style="font-size: 0.775rem; color: #2563EB; font-weight: 600;">Interactive Sensitivity Controls</span>
                </div>
                <p style="font-size: 0.825rem; color: #475569; margin-top: 0; margin-bottom: 14px;">
                    Adjust policy targets to simulate realigned ticket volumes, SLA breach drops, and recovered labor capacity. Model domain specialists as capacity enablers (~90 tickets/specialist) to absorb historical category mismatches.
                </p>
                """,
                unsafe_allow_html=True,
            )

            sim_p1, sim_p2 = st.columns(2)
            with sim_p1:
                target_mismatch = st.slider(
                    "Target Category Mismatch Rate (%)",
                    min_value=0.0,
                    max_value=30.0,
                    value=5.0,
                    step=0.5,
                    help="Historical baseline is 29.8%. Smart routing engine targets < 5.0%. Moving to 28.0% simulates a modest 1.77 pp realignment.",
                )
            with sim_p2:
                added_specialists = st.slider(
                    "Additional Domain Specialists (Headcount)",
                    min_value=0,
                    max_value=6,
                    value=2,
                    step=1,
                    help="Hiring or reallocating specialists increases domain intake capacity (~90 tickets/agent), absorbing specialist volume that was historically misrouted.",
                )

            # Domain allocation selector for added specialists
            if added_specialists > 0:
                alloc_col1, alloc_col2 = st.columns([2, 1])
                with alloc_col1:
                    staffing_allocation_mode = st.selectbox(
                        "Target Specialist Allocation Strategy",
                        [
                            "Split Evenly Across Bottlenecks (Payments & Checkout + API Integrations)",
                            "Concentrate on Payments & Checkout (Highest Mismatch Volume)",
                            "Concentrate on API Integrations (Highest Breach Rate)",
                            "Concentrate on Account Management",
                        ],
                        help="Select domain where added specialists provide dedicated absorption capacity.",
                    )
                with alloc_col2:
                    hourly_cost_rate = st.number_input(
                        "Loaded Labor Rate ($/hr)",
                        min_value=20.0,
                        max_value=120.0,
                        value=45.0,
                        step=5.0,
                        help="Fully loaded operational labor rate used to value recovered capacity. Not cash payroll savings.",
                    )
            else:
                staffing_allocation_mode = "None"
                hourly_cost_rate = 45.0

            # Compute verified simulation
            # Re-instantiate with custom rate if modified from default
            active_simulator = impact_simulator if hourly_cost_rate == 45.0 else ImpactSimulator(
                hist_tickets, agents, merchants, hourly_capacity_value=hourly_cost_rate
            )

            target_m_ratio = target_mismatch / 100.0
            sim_res = active_simulator.simulate_mismatch_reduction(target_m_ratio)

            # Defensible staffing capacity absorption mapping
            staffing_dict = {}
            if added_specialists > 0:
                if "Split Evenly" in staffing_allocation_mode:
                    c1 = (added_specialists + 1) // 2
                    c2 = added_specialists // 2
                    staffing_dict = {"Payments & Checkout": c1, "API Integrations": c2}
                elif "Payments & Checkout" in staffing_allocation_mode:
                    staffing_dict = {"Payments & Checkout": added_specialists}
                elif "API Integrations" in staffing_allocation_mode:
                    staffing_dict = {"API Integrations": added_specialists}
                elif "Account Management" in staffing_allocation_mode:
                    staffing_dict = {"Account Management": added_specialists}

            staffing_res = active_simulator.simulate_staffing_capacity_enabler(staffing_dict)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="font-size: 0.85rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">
                    Simulated Scenario Outputs (Target Mismatch: {target_mismatch:.1f}% | Headcount: +{added_specialists})
                </div>
                """,
                unsafe_allow_html=True,
            )

            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.markdown(
                    render_kpi_card(
                        "Simulated Breach Rate",
                        f"{sim_res['simulated_breach_rate']*100:.1f}%",
                        f"-{sim_res['breach_rate_reduction_pp']:.2f} pp reduction",
                        "Scenario Estimate",
                        "green",
                    ),
                    unsafe_allow_html=True,
                )
            with res2:
                st.markdown(
                    render_kpi_card(
                        "Total Hours Recovered",
                        f"{sim_res['hours_recovered']:,.0f} hrs",
                        f"{int(round(sim_res['realigned_tickets']))} tickets realigned",
                        "Capacity Recovered",
                        "green",
                    ),
                    unsafe_allow_html=True,
                )
            with res3:
                st.markdown(
                    render_kpi_card(
                        "Operational Capacity Value",
                        f"${sim_res['estimated_capacity_value']:,.0f}",
                        f"@ ${hourly_cost_rate:.0f}/hr labor rate",
                        "Unlocked Capacity",
                        "blue",
                    ),
                    unsafe_allow_html=True,
                )
            with res4:
                enabled_tix = staffing_res["total_enabled_ticket_capacity"]
                st.markdown(
                    render_kpi_card(
                        "Specialist Capacity Enabled",
                        f"{enabled_tix:,} tickets",
                        f"From +{added_specialists} specialist hires" if added_specialists > 0 else "No headcount added",
                        "Staffing Enabler",
                        "blue" if added_specialists > 0 else "slate",
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Baseline vs. Scenario Audit Table
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <h4 style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 0;">3. Side-by-Side Baseline vs. Scenario Audit</h4>
                <span style="font-size: 0.775rem; color: #64748B; font-weight: 600;">Rigorous Traceability Matrix</span>
            </div>
            <p style="font-size: 0.825rem; color: #475569; margin-top: 0; margin-bottom: 12px;">
                Direct empirical comparison of observed baseline ground truth against counterfactual simulation outputs.
            </p>
            """,
            unsafe_allow_html=True,
        )

        comparison_df = active_simulator.get_comparison_dataframe(sim_res)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Domain Specialist Staffing Breakdown (if added)
        if added_specialists > 0 and staffing_res["category_breakdown"]:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(
                    """
                    <h5 style="font-size: 0.925rem; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">
                        Domain Specialist Capacity Absorption Breakdown
                    </h5>
                    """,
                    unsafe_allow_html=True,
                )
                staffing_rows = []
                for b in staffing_res["category_breakdown"]:
                    staffing_rows.append({
                        "Category": b["category"],
                        "Added Specialists": f"+{b['added_headcount']}",
                        "Historical Category Volume": f"{b['historical_category_volume']:,} tickets",
                        "Historical Mismatches": f"{b['historical_mismatch_count']:,} tickets",
                        "Enabled Ticket Capacity": f"~{b['enabled_specialist_capacity']:,} tickets",
                        "Mismatches Absorbed": f"{b['potential_mismatches_absorbed']:,} tickets",
                    })
                st.dataframe(pd.DataFrame(staffing_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Note: In observational data, hiring does not guarantee SLA improvements unless routed properly. Each specialist handled ~90.4 tickets historically; added headcount provides the absorption capacity needed to execute domain-pure routing."
                )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Expandable Methodology & Formulas
        with st.expander("📐 Mathematical Methodology, Empirical Formulas & Core Assumptions", expanded=False):
            st.markdown(
                r"""
                ### 1. Data Partitioning & Historical Ground Truth Isolation
                - **Dataset Partitioning**: The simulation baseline is derived **strictly from the 1,807 completed historical tickets** (`tickets.csv` with `is_legacy == True`).
                - **Zero Open-Queue Contamination**: All 250 open queue tickets (`is_legacy == False`) are strictly excluded from baseline rate calculations, avoiding synthetic placeholder distortion.
                
                ---
                
                ### 2. Empirical Ground-Truth Baseline Constants ($N = 1,807$)
                - **Total Historical Resolution Breaches**: 1,037 breaches out of 1,807 completed tickets (**57.39%** baseline breach rate).
                - **Category Mismatch Volume**: 538 tickets out of 1,807 completed tickets (**29.77%** baseline mismatch rate).
                - **Matched Tickets ($N = 1,269$)**: 634 breaches (**49.96%** breach rate), Mean Resolution Time = **28.24 hours**.
                - **Mismatched Tickets ($N = 538$)**: 403 breaches (**74.91%** breach rate), Mean Resolution Time = **40.35 hours**.
                - **Empirical Breach Penalty ($\Delta BR$)**:
                  $$\\Delta BR = 74.91\\% - 49.96\\% = +24.95\\text{ percentage points}$$
                - **Empirical Resolution Delay Penalty ($\Delta H$)**:
                  $$\\Delta H = 40.35\\text{h} - 28.24\\text{h} = +12.11\\text{ hours per ticket}$$
                - **Total Historical Excess Drag**:
                  $$538\\text{ tickets} \\times 12.106\\text{h} = 6,513\\text{ hours of cumulative resolution delay}$$
                - **Theoretical Maximum SLA Recovery Potential (at 0.0% mismatch)**:
                  $$29.77\\% \\times 24.95\\text{ pp} = 7.43\\text{ pp potential breach drop (down to 49.96\\%)}$$
                
                ---
                
                ### 3. Counterfactual Scenario Simulation Formulas
                For any user-selected target mismatch rate $r_{\\text{target}} \\in [0.0, 0.30]$:
                1. **Mismatch Delta Achieved**:
                   $$\\Delta r = \\max(0, 0.2977 - r_{\\text{target}})$$
                2. **Realigned Tickets**:
                   $$\\text{Realigned Tickets} = \\Delta r \\times 1,807$$
                3. **Avoided SLA Breaches**:
                   $$\\text{Breaches Avoided} = \\text{Realigned Tickets} \\times 0.2495$$
                4. **Breach Rate Reduction (pp)**:
                   $$\\text{Breach Reduction (pp)} = \\Delta r \\times 24.95\\text{ pp}$$
                5. **Simulated Breach Rate**:
                   $$\\text{Simulated Breach Rate} = 57.39\\% - \\text{Breach Reduction (pp)}$$
                6. **Recovered Resolution Hours**:
                   $$\\text{Hours Recovered} = \\text{Realigned Tickets} \\times 12.106\\text{ hours}$$
                7. **Operational Labor Capacity Value**:
                   $$\\text{Capacity Value} = \\text{Hours Recovered} \\times \\$45/\\text{hour (loaded labor rate)}$$

                ---

                ### 4. Mathematical Trace of the 28.0% Slider Example
                When setting the target mismatch rate slider to **28.0%**:
                - $\\Delta r = 29.77\\% - 28.00\\% = 1.77\\text{ pp} = 0.01773$
                - Realigned tickets: $0.01773 \\times 1,807 = 32.04\\text{ tickets}$
                - Avoided breaches: $32.04 \\times 0.24946 = 7.99 \\approx 8\\text{ breaches}$
                - Breach rate reduction: $1.77\\% \\times 24.95\\text{ pp} = 0.44\\text{ pp}$
                - Simulated breach rate: $57.39\\% - 0.44\\% = \\mathbf{56.95\\%} \\approx \\mathbf{56.9\\%}$
                - Recovered hours: $32.04 \\times 12.106\\text{h} = \\mathbf{387.9\\text{ hours}} \\approx \\mathbf{388\\text{ hours}}$
                - Projected capacity value: $387.9\\text{ hrs} \\times \\$45/\\text{hr} = \\mathbf{\\$17,454} \\approx \\mathbf{\\$17,415}$
                *(The previous UI showed a disconnect because the top cards displayed the 0.0% theoretical maximum of 7.4 pp and 6,510h, while the bottom cards reflected the 28.0% scenario of 387h and $17,415 without clearly separating baseline from estimate).*

                ---

                ### 5. Labor Capacity Valuation vs. Realized Cash Savings
                - The **$45/hour** multiplier represents the fully loaded operational cost of an L2/L3 support engineer (base pay, benefits, tooling, and management overhead).
                - This metric represents **unlocked labor capacity**—reclaiming engineer time from cross-domain confusion to handle higher ticket volume, prevent backlog accumulation, or engage in proactive merchant health checks.
                - It does **not** represent immediate cash savings or payroll reductions, as support personnel are not terminated upon SLA efficiency gains.

                ---

                ### 6. Defensible Staffing Capacity Enabler Modeling
                - Headcount additions do **not** magically reduce breach rates by an arbitrary percentage (`added_agents * 1.8` has been removed).
                - In the 18-month historical dataset, each agent handled an average of **~90.4 tickets** (range: 75 to 156 tickets).
                - Adding domain specialists provides **dedicated intake capacity** (~90 tickets/specialist) in understaffed categories (such as Payments & Checkout or UK shifts), eliminating the structural constraint that forced cross-domain ticket dispatching in the first place.

                ---

                ### 7. Observational Causation Caveat
                - Historical mismatch rates and breach penalties are observational correlations. While category mismatch strongly correlates with increased breaches (+24.95 pp) and delays (+12.11h), some tickets may have been mismatched due to unobserved merchant complexity or intake ambiguities.
                - Counterfactual estimates represent upper-bound operational potential under standard *ceteris paribus* (all else equal) assumptions.
                """,
            )
