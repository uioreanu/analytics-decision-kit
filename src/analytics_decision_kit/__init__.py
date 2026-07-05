"""Analytics Decision Kit.

Simple notebook-first helpers for customer analytics.
"""

from analytics_decision_kit.sample_data import create_demo_orders, make_synthetic_orders
from analytics_decision_kit.customer_analysis import (
    calculate_customer_metrics,
    calculate_revenue_deciles,
    calculate_pareto_summary,
    assign_customer_segments,
    run_customer_analysis,
    write_summary_text,
)

__all__ = [
    "assign_customer_segments",
    "calculate_customer_metrics",
    "calculate_pareto_summary",
    "calculate_revenue_deciles",
    "create_demo_orders",
    "make_synthetic_orders",
    "run_customer_analysis",
    "write_summary_text",
]
