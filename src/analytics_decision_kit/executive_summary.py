"""Compatibility wrapper around customer_analysis."""

from analytics_decision_kit.customer_analysis import write_summary_text


def write_executive_summary(customer_metrics, deciles, pareto):
    """Old function name kept for the first archive API."""
    return write_summary_text(customer_metrics, deciles, pareto)


__all__ = ["write_executive_summary"]
