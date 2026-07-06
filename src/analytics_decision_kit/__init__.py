"""Analytics Decision Kit.

Simple notebook-first helpers for customer analytics.
"""

from analytics_decision_kit.customer_analysis import (
    assign_customer_segments,
    calculate_customer_metrics,
    calculate_pareto_summary,
    calculate_revenue_deciles,
    run_customer_analysis,
    write_summary_text,
)
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
    normalize_transactional_data,
)
from analytics_decision_kit.sample_data import create_demo_orders, make_synthetic_orders

__all__ = [
    "DEFAULT_SUPERSTORE_GITHUB_URL",
    "assign_customer_segments",
    "calculate_customer_metrics",
    "calculate_pareto_summary",
    "calculate_revenue_deciles",
    "create_demo_orders",
    "get_transactional_data",
    "make_synthetic_orders",
    "normalize_transactional_data",
    "run_customer_analysis",
    "write_summary_text",
]
