"""UI components and chart builders for OpsPilot AI Streamlit application."""

from app.components.styles import (
    apply_custom_css,
    render_kpi_card,
    render_confusion_matrix_grid,
    render_benchmark_table_html,
    render_insight_card,
    render_recommendation_card,
)
from app.components.charts import (
    apply_chart_theme,
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

__all__ = [
    "apply_custom_css",
    "render_kpi_card",
    "render_confusion_matrix_grid",
    "render_benchmark_table_html",
    "render_insight_card",
    "render_recommendation_card",
    "apply_chart_theme",
    "create_volume_trend_chart",
    "create_category_distribution_chart",
    "create_sla_breach_chart",
    "create_open_queue_risk_chart",
    "create_priority_chart",
    "create_sector_chart",
    "create_feature_importance_chart",
    "create_bottleneck_mismatch_chart",
    "create_agent_tier_performance_chart",
    "create_category_bottleneck_comparison_chart",
    "create_priority_sla_performance_chart",
    "create_shift_region_coverage_chart",
    "create_top_bottlenecks_bar_chart",
    "create_agent_dispatch_distribution_chart",
]

