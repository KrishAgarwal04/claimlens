"""Enterprise SaaS CSS styles and HTML UI component helpers for OpsPilot AI.

Guarantees high-contrast typography, pure light backgrounds, accessible badges,
and zero markdown-escaping bugs across all screen sizes.
"""

import streamlit as st
from typing import Optional, Dict, Any, List

__all__ = [
    "apply_custom_css",
    "render_kpi_card",
    "render_confusion_matrix_grid",
    "render_benchmark_table_html",
    "render_insight_card",
    "render_recommendation_card",
]


def apply_custom_css():
    """Inject polished, enterprise light-themed CSS with high-contrast text and dense sidebar."""
    st.markdown(
        """<style>
        /* Modern Clean Typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #F8FAFC !important; /* slate-50 */
            color: #1E293B !important; /* slate-800 */
        }

        /* High-Contrast Headings & Text Overrides */
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important; /* slate-900 */
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }

        p, span, label, li, a {
            color: #334155 !important; /* slate-700 */
        }

        /* Stronger contrast on Streamlit Markdown */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #1E293B !important;
        }

        /* Main Container Spacing */
        .main .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Sidebar Styling & Compact Density */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 1px 0 3px rgba(15, 23, 42, 0.02) !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }

        [data-testid="stSidebar"] * {
            color: #334155 !important;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #0F172A !important;
            font-weight: 800 !important;
        }

        /* Radio Buttons in Sidebar: Ultra-compact spacing & clickable SaaS pills */
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-bottom: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            font-size: 0.7rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            color: #64748B !important;
            margin-bottom: 4px !important;
            padding-bottom: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            gap: 3px !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"],
        [data-testid="stSidebar"] .stRadio label {
            background-color: #F8FAFC !important;
            padding: 5px 10px !important;
            border-radius: 6px !important;
            font-size: 0.815rem !important;
            font-weight: 600 !important;
            color: #334155 !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 1px !important;
            line-height: 1.2 !important;
            width: 100% !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover,
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: #EFF6FF !important;
            color: #1D4ED8 !important;
            border-color: #DBEAFE !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background-color: #EFF6FF !important;
            color: #1D4ED8 !important;
            border-color: #93C5FD !important;
            font-weight: 700 !important;
        }

        /* Native Streamlit Container with border=True: Styled as Enterprise SaaS Card */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
            padding: 16px 18px !important;
            margin-bottom: 14px !important;
        }

        /* Streamlit Native Metric Card Overrides */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        }

        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.775rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-size: 1.75rem !important;
            font-weight: 800 !important;
        }

        /* Custom Responsive KPI Card */
        .kpi-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-title {
            font-size: 0.76rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }

        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.15;
            margin-bottom: 5px;
            letter-spacing: -0.02em;
        }

        .kpi-subtext {
            font-size: 0.785rem;
            color: #64748B;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }

        /* Status Badges */
        .kpi-badge-blue {
            background-color: #EFF6FF;
            color: #1D4ED8;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            border: 1px solid #DBEAFE;
        }

        .kpi-badge-green {
            background-color: #ECFDF5;
            color: #047857;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            border: 1px solid #A7F3D0;
        }

        .kpi-badge-amber {
            background-color: #FFFBEB;
            color: #B45309;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            border: 1px solid #FDE68A;
        }

        .kpi-badge-red {
            background-color: #FEF2F2;
            color: #B91C1C;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            border: 1px solid #FECACA;
        }

        .kpi-badge-purple {
            background-color: #F5F3FF;
            color: #6D28D9;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            border: 1px solid #DDD6FE;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 7px 16px;
            font-weight: 600;
            font-size: 0.865rem;
            color: #64748B !important;
            background-color: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #2563EB !important;
            background-color: #EFF6FF;
        }

        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
            background-color: #EFF6FF !important;
            border-bottom: 2px solid #2563EB !important;
            font-weight: 700 !important;
        }

        /* Dataframe light-theme overrides */
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }

        /* Section Brand Header */
        .brand-header {
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid #E2E8F0;
        }

        .brand-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .brand-desc {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 3px;
            margin-bottom: 0;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    badge_text: Optional[str] = None,
    badge_variant: str = "blue",
) -> str:
    """Return HTML string for an enterprise KPI card without leading indentation."""
    badge_html = ""
    if badge_text:
        badge_class = f"kpi-badge-{badge_variant}"
        badge_html = f'<span class="{badge_class}">{badge_text}</span>'

    # Zero indentation ensures markdown parser never treats this as a code block
    return f'<div class="kpi-container"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div><div class="kpi-subtext">{badge_html}<span>{subtext}</span></div></div>'


def render_confusion_matrix_grid(cm: Dict[str, int]) -> str:
    """Return HTML string for an enterprise 2x2 confusion matrix card with zero newline indentation."""
    tn = cm.get("tn", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)
    tp = cm.get("tp", 0)
    total = tn + fp + fn + tp
    accuracy = ((tn + tp) / total * 100) if total > 0 else 0.0

    return (
        f'<div class="kpi-container" style="padding: 16px 18px; height: 100%;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Holdout Confusion Matrix</div>'
        f'<div style="font-size: 0.8rem; color: #0F172A; font-weight: 700; margin-top: 2px;">{total:,} Unseen Test Tickets</div>'
        f'</div>'
        f'<span class="kpi-badge-green">Acc: {accuracy:.1f}%</span>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">'
        f'<div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px; text-align: center;">'
        f'<div style="font-size: 0.68rem; font-weight: 700; color: #166534; text-transform: uppercase;">True Negative (TN)</div>'
        f'<div style="font-size: 1.5rem; font-weight: 800; color: #166534; line-height: 1.2;">{tn:,}</div>'
        f'<div style="font-size: 0.675rem; color: #15803D;">Pred: Normal | Act: Normal</div>'
        f'</div>'
        f'<div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px; text-align: center;">'
        f'<div style="font-size: 0.68rem; font-weight: 700; color: #B45309; text-transform: uppercase;">False Positive (FP)</div>'
        f'<div style="font-size: 1.5rem; font-weight: 800; color: #B45309; line-height: 1.2;">{fp:,}</div>'
        f'<div style="font-size: 0.675rem; color: #B45309;">Pred: Breach | Act: Normal</div>'
        f'</div>'
        f'<div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 10px; text-align: center;">'
        f'<div style="font-size: 0.68rem; font-weight: 700; color: #DC2626; text-transform: uppercase;">False Negative (FN)</div>'
        f'<div style="font-size: 1.5rem; font-weight: 800; color: #DC2626; line-height: 1.2;">{fn:,}</div>'
        f'<div style="font-size: 0.675rem; color: #DC2626;">Pred: Normal | Act: Breach</div>'
        f'</div>'
        f'<div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px; text-align: center;">'
        f'<div style="font-size: 0.68rem; font-weight: 700; color: #1D4ED8; text-transform: uppercase;">True Positive (TP)</div>'
        f'<div style="font-size: 1.5rem; font-weight: 800; color: #1D4ED8; line-height: 1.2;">{tp:,}</div>'
        f'<div style="font-size: 0.675rem; color: #1D4ED8;">Pred: Breach | Act: Breach</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size: 0.72rem; color: #64748B; text-align: center; border-top: 1px solid #E2E8F0; padding-top: 6px;">'
        f'Evaluated on calibrated probability threshold <b>P &ge; 0.50</b>'
        f'</div>'
        f'</div>'
    )


def render_benchmark_table_html(bench_data: List[Dict[str, Any]]) -> str:
    """Return cleanly styled HTML table string for model benchmark comparison without indentation."""
    if not bench_data:
        return ""

    headers = list(bench_data[0].keys())
    th_html = "".join([f'<th style="padding: 10px 12px; text-align: left; font-size: 0.775rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 2px solid #E2E8F0; background-color: #F8FAFC;">{h}</th>' for h in headers])

    rows_html = []
    for row in bench_data:
        tds = "".join([f'<td style="padding: 10px 12px; font-size: 0.825rem; color: #1E293B; border-bottom: 1px solid #F1F5F9; font-weight: 500;">{row.get(h, "")}</td>' for h in headers])
        rows_html.append(f'<tr>{tds}</tr>')

    table_rows = "".join(rows_html)
    return f'<div style="overflow-x: auto; width: 100%; border: 1px solid #E2E8F0; border-radius: 8px;"><table style="width: 100%; border-collapse: collapse; text-align: left; background-color: #FFFFFF;"><thead><tr>{th_html}</tr></thead><tbody>{table_rows}</tbody></table></div>'


def render_insight_card(insight: Dict[str, Any]) -> str:
    """Return HTML card for structured evidence-based operational bottleneck insight without indentation."""
    title = insight.get("title", "Operational Insight")
    sample_size = insight.get("sample_size", 0)
    comparison = insight.get("comparison", "")
    action = insight.get("suggested_action", "")
    caveat = insight.get("caveat", "")
    metrics = insight.get("metric_values", {})

    badge_items = []
    for k, v in list(metrics.items())[:3]:
        label = k.replace("_", " ").title()
        val_str = f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else f"{v}"
        badge_items.append(f'<span style="background-color: #EFF6FF; color: #1D4ED8; font-size: 0.725rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; border: 1px solid #DBEAFE;">{label}: {val_str}</span>')
    badges_html = "".join(badge_items)

    return (
        f'<div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">'
        f'<div style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">{title}</div>'
        f'<span style="background-color: #F1F5F9; color: #475569; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">Sample N={sample_size:,}</span>'
        f'</div>'
        f'<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">{badges_html}</div>'
        f'<div style="font-size: 0.835rem; color: #334155; margin-bottom: 8px;"><b>Empirical Comparison:</b> {comparison}</div>'
        f'<div style="font-size: 0.835rem; color: #1E293B; margin-bottom: 8px;"><b>Operational Action:</b> {action}</div>'
        f'<div style="background-color: #F8FAFC; border-left: 3px solid #94A3B8; padding: 6px 10px; font-size: 0.785rem; color: #64748B; border-radius: 0 6px 6px 0;">'
        f'<b>Observational Caveat:</b> {caveat}'
        f'</div>'
        f'</div>'
    )


def render_recommendation_card(rec: Dict[str, Any]) -> str:
    """Return cleanly styled HTML card for an agent recommendation output without indentation."""
    status = rec.get("status", "SUCCESS")
    ticket_id = rec.get("ticket_id", "N/A")

    if status == "NO_MATCH":
        reasons_html = "".join([f'<li style="margin-bottom: 4px; color: #991B1B;">{r}</li>' for r in rec.get("reasons", [])])
        return (
            f'<div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #991B1B;">&#9888;&#65039; {rec.get("recommended_agent_name", "No Match")}</div>'
            f'<span style="background-color: #FEE2E2; color: #991B1B; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; border: 1px solid #FCA5A5;">Match Score: 0.00</span>'
            f'</div>'
            f'<div style="font-size: 0.825rem; color: #7F1D1D; margin-bottom: 8px;"><b>Triage Status:</b> Ticket #{ticket_id} could not be matched safely.</div>'
            f'<ul style="font-size: 0.8rem; margin: 0 0 10px 18px; padding: 0;">{reasons_html}</ul>'
            f'<div style="background-color: #FFF; border: 1px solid #FCA5A5; border-radius: 6px; padding: 8px 12px; font-size: 0.75rem; color: #7F1D1D;">'
            f'<b>Policy Limitation:</b> {rec.get("uncertainty_limitations", "")}'
            f'</div>'
            f'</div>'
        )

    agent_name = rec.get("recommended_agent_name", "Unknown Agent")
    agent_tier = rec.get("recommended_agent_tier", "L1")
    agent_cat = rec.get("recommended_agent_category", "General")
    agent_shift = rec.get("recommended_agent_shift", "Global")
    score = rec.get("match_score", 0.0)
    reasons = rec.get("reasons", [])
    limitations = rec.get("uncertainty_limitations", "")

    tier_badge_class = "kpi-badge-purple" if agent_tier == "L3" else ("kpi-badge-blue" if agent_tier == "L2" else "kpi-badge-green")
    score_pct = int(round(score * 100))

    reasons_list_html = "".join([f'<li style="margin-bottom: 4px; color: #334155; font-size: 0.815rem;">{r}</li>' for r in reasons])

    return (
        f'<div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">'
        f'<div>'
        f'<div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Primary Recommended Specialist</div>'
        f'<div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 2px;">{agent_name}</div>'
        f'</div>'
        f'<div style="text-align: right;">'
        f'<span class="{tier_badge_class}" style="font-size: 0.8rem; padding: 3px 10px; margin-right: 6px;">Tier {agent_tier}</span>'
        f'<span style="background-color: #EFF6FF; color: #1D4ED8; font-weight: 800; font-size: 0.85rem; padding: 3px 10px; border-radius: 6px; border: 1px solid #BFDBFE;">Score: {score:.2f} ({score_pct}%)</span>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">'
        f'<span style="background-color: #F8FAFC; border: 1px solid #E2E8F0; font-size: 0.75rem; color: #475569; font-weight: 600; padding: 3px 8px; border-radius: 6px;">Domain: <b>{agent_cat}</b></span>'
        f'<span style="background-color: #F8FAFC; border: 1px solid #E2E8F0; font-size: 0.75rem; color: #475569; font-weight: 600; padding: 3px 8px; border-radius: 6px;">Shift: <b>{agent_shift}</b></span>'
        f'</div>'
        f'<div style="font-size: 0.8rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em;">Match Rationale & Justifications:</div>'
        f'<ul style="margin: 0 0 12px 18px; padding: 0;">{reasons_list_html}</ul>'
        f'<div style="background-color: #F8FAFC; border-left: 3px solid #64748B; padding: 8px 12px; font-size: 0.75rem; color: #64748B; border-radius: 0 6px 6px 0;">'
        f'<b>Operational Transparency:</b> {limitations}'
        f'</div>'
        f'</div>'
    )



