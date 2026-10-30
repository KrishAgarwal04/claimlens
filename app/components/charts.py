"""Cohesive, enterprise light-themed Plotly charts for OpsPilot AI.

Guarantees high contrast, no label clipping, standardized margins,
and cleanly positioned legends outside plotting areas.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Enterprise SaaS Color Palette (Cohesive Light Theme)
CHART_COLORS = {
    "primary": "#6C5478",       # Soft Slate Plum
    "primary_light": "#F3EFF7", # Very Light Pastel Lavender
    "success": "#16603E",       # Dark Bold Green
    "warning": "#8A5310",       # Dark Bold Warm Amber
    "danger": "#A32832",        # Dark Bold Crimson
    "purple": "#6C5478",        # Soft Slate Plum
    "teal": "#3B6E65",          # Dark Sage Teal
    "navy": "#120A18",          # Deep Heading Black-Plum
    "text": "#1A1322",          # Deep Dark High-Contrast Text
    "muted": "#4A4252",         # High-Contrast Dark Slate Muted
    "grid": "#F0EAF5",          # Very Light Subtle Grid
    "border": "#E5DFE9",        # Soft Light Border
}


def apply_chart_theme(
    fig: go.Figure,
    height: int = 300,
    legend_position: str = "top_right",
    custom_margin: Optional[Dict[str, int]] = None,
) -> go.Figure:
    """Apply uniform enterprise light styling to a Plotly figure."""
    if legend_position == "bottom":
        legend_dict = dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=CHART_COLORS["muted"]),
            title=dict(text=""),
        )
        base_margin = dict(l=45, r=20, t=50, b=65)
    elif legend_position == "below_title":
        legend_dict = dict(
            orientation="h",
            yref="container",
            yanchor="top",
            y=0.86,
            xanchor="left",
            x=0.01,
            font=dict(size=10.5, color=CHART_COLORS["muted"]),
            title=dict(text=""),
        )
        base_margin = dict(l=45, r=20, t=75, b=45)
    elif legend_position == "none":
        legend_dict = dict(visible=False)
        base_margin = dict(l=45, r=20, t=45, b=35)
    else:  # top_right
        legend_dict = dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=11, color=CHART_COLORS["muted"]),
            title=dict(text=""),
        )
        base_margin = dict(l=45, r=20, t=50, b=40)

    # Preserve any custom margins explicitly configured on the figure
    if fig.layout.margin:
        for k in ["l", "r", "t", "b"]:
            val = getattr(fig.layout.margin, k, None)
            if val is not None and val != 80:  # 80 is Plotly default margin
                base_margin[k] = val

    if custom_margin:
        base_margin.update(custom_margin)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(
            family="Inter, Manrope, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            color=CHART_COLORS["text"],
            size=12,
        ),
        margin=base_margin,
        hoverlabel=dict(
            bgcolor=CHART_COLORS["navy"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor=CHART_COLORS["navy"],
        ),
        legend=legend_dict,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=CHART_COLORS["muted"], activecolor=CHART_COLORS["primary"]),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zeroline=False,
        linecolor=CHART_COLORS["border"],
        tickfont=dict(size=11, color=CHART_COLORS["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zeroline=False,
        linecolor=CHART_COLORS["border"],
        tickfont=dict(size=11, color=CHART_COLORS["muted"]),
    )
    return fig


def create_volume_trend_chart(tickets_df: pd.DataFrame) -> go.Figure:
    """Generate monthly ticket intake volume trend line with area fill."""
    df = tickets_df.copy()
    df["month_str"] = df["created_at"].dt.to_period("M").astype(str)
    trend = df.groupby(["month_str", "is_legacy"]).size().reset_index(name="count")

    piv = trend.pivot(index="month_str", columns="is_legacy", values="count").fillna(0)
    piv.columns = ["Open Queue (Jun 2026)", "Historical Intake"]
    piv = piv.reset_index()

    fig = go.Figure()

    # Historical Area Trace
    fig.add_trace(
        go.Scatter(
            x=piv["month_str"],
            y=piv["Historical Intake"],
            name="Historical Intake",
            mode="lines+markers",
            line=dict(color=CHART_COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(108, 84, 120, 0.08)",
            marker=dict(size=5, color=CHART_COLORS["primary"]),
            hovertemplate="<b>%{x}</b><br>Historical Tickets: %{y:,}<extra></extra>",
        )
    )

    # Open Queue Marker
    if "Open Queue (Jun 2026)" in piv.columns and (piv["Open Queue (Jun 2026)"] > 0).any():
        open_subset = piv[piv["Open Queue (Jun 2026)"] > 0]
        fig.add_trace(
            go.Scatter(
                x=open_subset["month_str"],
                y=open_subset["Open Queue (Jun 2026)"],
                name="Live Open Queue",
                mode="markers",
                marker=dict(size=9, color=CHART_COLORS["warning"], symbol="diamond"),
                hovertemplate="<b>%{x}</b><br>Live Open Queue: %{y:,} tickets<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Monthly Ticket Inflow & Intake Trend</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        margin=dict(l=45, r=20, t=55, b=40),
    )
    return apply_chart_theme(fig, height=320, legend_position="top_right")


def create_category_distribution_chart(tickets_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing ticket volume by category with wide label margin."""
    counts = tickets_df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)

    fig = go.Figure(
        go.Bar(
            x=counts["count"],
            y=counts["category"],
            orientation="h",
            marker=dict(
                color=counts["count"],
                colorscale=[[0, "#F3EFF7"], [1, "#6C5478"]],
                showscale=False,
            ),
            text=counts.apply(lambda r: f"{r['count']:,} ({r['pct']}%)", axis=1),
            textposition="auto",
            textfont=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Tickets: %{x:,} (%{text})<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Tickets by Primary Domain Category</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=150, r=25, t=45, b=30),  # Wide left margin prevents label clipping
    )
    return apply_chart_theme(fig, height=300, legend_position="none")


def create_sla_breach_chart(historical_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing historical SLA resolution breach rates across priority tiers."""
    grouped = (
        historical_df.groupby("priority")
        .agg(
            total=("ticket_id", "count"),
            breaches=("resolution_breached", "sum"),
        )
        .reset_index()
    )
    grouped["breach_rate_pct"] = (grouped["breaches"] / grouped["total"] * 100).round(1)

    p_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    grouped["rank"] = grouped["priority"].map(p_order)
    grouped = grouped.sort_values("rank")

    bar_colors = ["#A32832", "#8A5310", "#6C5478", "#4A4252"]

    fig = go.Figure(
        go.Bar(
            x=grouped["priority"],
            y=grouped["breach_rate_pct"],
            marker=dict(color=bar_colors),
            text=grouped["breach_rate_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=11, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>Priority %{x}</b><br>Breach Rate: %{y:.1f}%<extra></extra>",
        )
    )

    avg_breach = historical_df["resolution_breached"].mean() * 100
    fig.add_hline(
        y=avg_breach,
        line_dash="dash",
        line_color=CHART_COLORS["muted"],
        annotation_text=f"Historical Baseline: {avg_breach:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color=CHART_COLORS["muted"]),
    )

    fig.update_layout(
        title=dict(
            text="<b>Resolution SLA Breach Rate by Priority Tier</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(range=[0, 100], title="Breach Rate (%)"),
        margin=dict(l=45, r=20, t=45, b=35),
    )
    return apply_chart_theme(fig, height=300, legend_position="none")


def create_open_queue_risk_chart(triage_df: pd.DataFrame) -> go.Figure:
    """Histogram of predicted SLA breach risks with legend placed cleanly below title outside the plotting area."""
    fig = px.histogram(
        triage_df,
        x="breach_risk_pct",
        nbins=20,
        color="risk_tier",
        category_orders={
            "risk_tier": ["🔴 High (>=70%)", "🟡 Medium (40-70%)", "🟢 Low (<40%)"]
        },
        color_discrete_map={
            "🔴 High (>=70%)": "#A32832",
            "🟡 Medium (40-70%)": "#8A5310",
            "🟢 Low (<40%)": "#16603E",
        },
        labels={
            "breach_risk_pct": "Predicted SLA Breach Probability (%)",
            "count": "Ticket Count",
            "risk_tier": "",
        },
    )

    fig.update_layout(
        title=dict(
            text="<b>Open Queue SLA Risk Distribution (250 Tickets)</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.96,
            x=0.01,
            xanchor="left",
            yanchor="top",
            yref="container",
        ),
        bargap=0.08,
        margin=dict(l=45, r=20, t=75, b=45),
    )
    return apply_chart_theme(fig, height=320, legend_position="below_title")


def create_priority_chart(tickets_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing ticket breakdown by Priority (P1-P4) with bottom legend."""
    p_counts = tickets_df["priority"].value_counts().reset_index()
    p_counts.columns = ["priority", "count"]

    color_map = {
        "P1": "#A32832",
        "P2": "#8A5310",
        "P3": "#6C5478",
        "P4": "#4A4252",
    }

    fig = go.Figure(
        go.Pie(
            labels=p_counts["priority"],
            values=p_counts["count"],
            hole=0.55,
            marker=dict(colors=[color_map.get(p, "#E5DFE9") for p in p_counts["priority"]]),
            textinfo="label+percent",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Tickets: %{value:,} (%{percent})<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Ticket Volume by Priority</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        margin=dict(l=20, r=20, t=45, b=50),
    )
    return apply_chart_theme(fig, height=300, legend_position="bottom")


def create_sector_chart(merchants_df: pd.DataFrame, tickets_df: pd.DataFrame) -> go.Figure:
    """Bar chart of tickets by merchant industry sector."""
    merged = tickets_df.merge(
        merchants_df[["merchant_id", "sector"]], on="merchant_id", how="left"
    )
    s_counts = merged["sector"].value_counts().reset_index()
    s_counts.columns = ["sector", "count"]

    fig = go.Figure(
        go.Bar(
            x=s_counts["sector"],
            y=s_counts["count"],
            marker=dict(color="#6C5478"),
            text=s_counts["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Tickets: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Ticket Volume by Merchant Sector</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(title="Ticket Count"),
        margin=dict(l=45, r=20, t=45, b=55),
        xaxis=dict(tickangle=-20),
    )
    return apply_chart_theme(fig, height=300, legend_position="none")


def create_feature_importance_chart(top_features: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart for top ML model predictive features with wide label spacing."""
    df = pd.DataFrame(top_features[:10])
    df["display_name"] = (
        df["feature"]
        .str.replace("cat__", "")
        .str.replace("num__", "")
        .str.replace("_", " ")
        .str.title()
    )

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["display_name"],
            orientation="h",
            marker=dict(
                color=df["importance"],
                colorscale=[[0, "#F3EFF7"], [1, "#6C5478"]],
                showscale=False,
            ),
            text=df["importance"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="auto",
            textfont=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f} (%{text})<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Top 10 Pre-Assignment Feature Importances</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Gini Importance"),
        margin=dict(l=165, r=25, t=45, b=35),  # Wide left margin prevents label clipping
    )
    return apply_chart_theme(fig, height=340, legend_position="none")


def create_bottleneck_mismatch_chart(historical_df: pd.DataFrame) -> go.Figure:
    """Chart comparing SLA breach rates and resolution times for matched vs mismatched assignments."""
    comp = historical_df.groupby("category_mismatch").agg(
        breach_rate=("resolution_breached", "mean"),
        avg_res_hours=("resolution_hours", "mean"),
        count=("ticket_id", "count"),
    ).reset_index()

    comp["label"] = comp["category_mismatch"].map({False: "Specialist Matched", True: "Category Mismatch"})
    comp["breach_pct"] = (comp["breach_rate"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Resolution Breach Rate (%)",
            x=comp["label"],
            y=comp["breach_pct"],
            marker_color=["#16603E", "#A32832"],
            text=comp["breach_pct"].apply(lambda v: f"{v}%"),
            textposition="outside",
            textfont=dict(size=11, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Breach Rate: %{y}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Category Mismatch SLA Breach Penalty (+24.9 pp)</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(range=[0, 95], title="Breach Rate (%)"),
        margin=dict(l=45, r=20, t=45, b=35),
    )
    return apply_chart_theme(fig, height=300, legend_position="none")


def create_agent_tier_performance_chart(historical_df: pd.DataFrame, agents_df: pd.DataFrame) -> go.Figure:
    """Chart showing historical resolution breach rate by support agent tier."""
    df_copy = historical_df.copy()
    if "tier_agent" in df_copy.columns:
        merged = df_copy.rename(columns={"tier_agent": "_agent_tier"})
    else:
        agent_sub = agents_df[["agent_id", "tier"]].rename(columns={"tier": "_agent_tier"})
        merged = df_copy.merge(agent_sub, left_on="assigned_agent_id", right_on="agent_id", how="left")

    tier_perf = merged.groupby("_agent_tier").agg(
        breach_rate=("resolution_breached", "mean"),
        avg_hours=("resolution_hours", "mean"),
        count=("ticket_id", "count"),
    ).reset_index().rename(columns={"_agent_tier": "tier"})

    tier_perf["breach_pct"] = (tier_perf["breach_rate"] * 100).round(1)
    tier_order = {"L1": 1, "L2": 2, "L3": 3}
    tier_perf["rank"] = tier_perf["tier"].map(tier_order)
    tier_perf = tier_perf.sort_values("rank")

    fig = go.Figure(
        go.Bar(
            x=tier_perf["tier"],
            y=tier_perf["breach_pct"],
            marker_color=["#A32832", "#8A5310", "#16603E"],
            text=tier_perf["breach_pct"].apply(lambda v: f"{v}%"),
            textposition="outside",
            textfont=dict(size=11, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>Tier %{x}</b><br>Breach Rate: %{y}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Resolution Breach Rate by Agent Tier</b>",
            font=dict(size=13, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
        ),
        yaxis=dict(range=[0, 95], title="Breach Rate (%)"),
        margin=dict(l=45, r=20, t=45, b=35),
    )
    return apply_chart_theme(fig, height=300, legend_position="none")


def create_category_bottleneck_comparison_chart(cat_df: pd.DataFrame) -> go.Figure:
    """Grouped chart comparing Resolution Breach Rate (%) and Mean Resolution Time (Hours) by Category."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Breach Rate (%)",
            x=cat_df["category"],
            y=(cat_df["breach_rate"] * 100).round(1),
            marker_color=CHART_COLORS["danger"],
            text=(cat_df["breach_rate"] * 100).round(1).apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Breach Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Mean Resolution (Hours)",
            x=cat_df["category"],
            y=cat_df["avg_resolution_hours"].round(1),
            marker_color=CHART_COLORS["primary"],
            text=cat_df["avg_resolution_hours"].round(1).apply(lambda v: f"{v:.1f}h"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Avg Resolution: %{y:.1f} hrs<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Category SLA Breach Rate vs. Mean Resolution Time</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.96,
            x=0.01,
            xanchor="left",
            yanchor="top",
            yref="container",
        ),
        barmode="group",
        bargap=0.2,
        bargroupgap=0.1,
        yaxis=dict(title="Rate (%) / Hours (h)"),
        margin=dict(l=45, r=20, t=75, b=45),
    )
    return apply_chart_theme(fig, height=320, legend_position="below_title")


def create_priority_sla_performance_chart(prio_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing priority breach rates and actual vs target resolution hours."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Actual Resolution Hours",
            x=prio_df["priority"],
            y=prio_df["mean_resolution_hours"].round(1),
            marker_color=CHART_COLORS["primary"],
            text=prio_df["mean_resolution_hours"].round(1).apply(lambda v: f"{v:.1f}h"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Actual Mean: %{y:.1f} hrs<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="SLA Target Window",
            x=prio_df["priority"],
            y=prio_df["target_resolution_hours"],
            marker_color=CHART_COLORS["muted"],
            text=prio_df["target_resolution_hours"].apply(lambda v: f"{v:.0f}h"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Target SLA: %{y:.0f} hrs<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Priority Targets vs. Actual Resolution Time (Hours)</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.96,
            x=0.01,
            xanchor="left",
            yanchor="top",
            yref="container",
        ),
        barmode="group",
        yaxis=dict(title="Resolution Time (Hours)"),
        margin=dict(l=45, r=20, t=75, b=45),
    )
    return apply_chart_theme(fig, height=320, legend_position="below_title")


def create_shift_region_coverage_chart(shift_df: pd.DataFrame) -> go.Figure:
    """Compares Merchant Demand (%) vs Support Agent Capacity (%) across regional shifts."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Ticket Demand (% of volume)",
            x=shift_df["region"],
            y=shift_df["ticket_pct"],
            marker_color=CHART_COLORS["primary"],
            text=shift_df["ticket_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Ticket Share: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Agent Staffing (% of pool)",
            x=shift_df["region"],
            y=shift_df["agent_pct"],
            marker_color=CHART_COLORS["warning"],
            text=shift_df["agent_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(size=10.5, color=CHART_COLORS["navy"], family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Agent Share: %{y:.1f}%<extra></extra>",
        )
    )

    y_max = max(shift_df["ticket_pct"].max(), shift_df["agent_pct"].max()) * 1.25

    fig.update_layout(
        title=dict(
            text="<b>Regional Demand vs. Staffing Allocation Deficit</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.96,
            x=0.01,
            xanchor="left",
            yanchor="top",
            yref="container",
        ),
        barmode="group",
        yaxis=dict(title="Percentage (%)", range=[0, y_max]),
        margin=dict(l=45, r=20, t=75, b=45),
    )
    return apply_chart_theme(fig, height=320, legend_position="below_title")


def create_top_bottlenecks_bar_chart(ranked_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing the highest-impact subcategory bottlenecks."""
    top_df = ranked_df.head(8).copy()
    top_df = top_df.iloc[::-1]  # reverse for top-down display

    y_labels = top_df["sub_category"] if "sub_category" in top_df.columns else top_df["bottleneck_name"]
    fig = go.Figure(
        go.Bar(
            x=top_df["bottleneck_score"],
            y=y_labels,
            orientation="h",
            marker=dict(
                color=top_df["breach_rate"] * 100,
                colorscale=[[0, "#F3EFF7"], [0.5, "#8A5310"], [1, "#A32832"]],
                showscale=False,
            ),
            text=top_df.apply(lambda r: f"Score: {r['bottleneck_score']:.1f} | N={r['sample_size']} ({r['breach_rate']*100:.1f}%)", axis=1),
            textposition="auto",
            textfont=dict(size=11, color="#120A18", family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Bottleneck Score: %{x:.2f}<br>Sample Size: %{customdata[0]}<br>Breach Rate: %{customdata[1]:.1%}<extra></extra>",
            customdata=top_df[["sample_size", "breach_rate"]].values,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Top Subcategory Bottlenecks (Excess Breach Impact)</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.98,
            x=0.01,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(title="Bottleneck Score"),
        margin=dict(l=150, r=25, t=45, b=35),
    )
    return apply_chart_theme(fig, height=340, legend_position="none")


def create_agent_dispatch_distribution_chart(routed_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing tickets routed per agent, categorized by tier."""
    matched = routed_df[routed_df["recommended_agent_id"].notna()].copy()
    if len(matched) == 0:
        fig = go.Figure()
        fig.update_layout(title="No Routed Tickets to Display")
        return apply_chart_theme(fig, height=320, legend_position="none")

    counts = matched.groupby(["recommended_agent_name", "recommended_agent_tier"]).size().reset_index(name="ticket_count")
    counts = counts.sort_values("ticket_count", ascending=True)

    tier_colors = {
        "L1": "#EDE7F3",  # Very Light Lavender
        "L2": "#AEA0B6",  # Muted Purple
        "L3": "#6C5478",  # Soft Slate Plum
    }

    fig = go.Figure()
    for tier in ["L1", "L2", "L3"]:
        subset = counts[counts["recommended_agent_tier"] == tier]
        if len(subset) > 0:
            fig.add_trace(
                go.Bar(
                    x=subset["ticket_count"],
                    y=subset["recommended_agent_name"],
                    name=f"Tier {tier}",
                    orientation="h",
                    marker_color=tier_colors.get(tier, "#4A4252"),
                    text=subset["ticket_count"],
                    textposition="auto",
                    textfont=dict(size=10, family="Inter, sans-serif"),
                    hovertemplate="<b>%{y}</b> (Tier " + tier + ")<br>Assigned Tickets: %{x}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text="<b>Open Queue Batch Dispatch Distribution by Specialist</b>",
            font=dict(size=12.5, color=CHART_COLORS["navy"]),
            y=0.96,
            x=0.01,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(title="Recommended Open Tickets (N)"),
        yaxis=dict(title=""),
        barmode="stack",
        margin=dict(l=130, r=20, t=65, b=40),
    )
    return apply_chart_theme(fig, height=380, legend_position="below_title")


