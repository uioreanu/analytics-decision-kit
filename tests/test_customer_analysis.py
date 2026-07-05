import pandas as pd

from analytics_decision_kit.customer_analysis import (
    assign_customer_segments,
    calculate_customer_metrics,
    calculate_pareto_summary,
    calculate_revenue_deciles,
    run_customer_analysis,
)
from analytics_decision_kit.sample_data import create_demo_orders


def test_customer_metrics_small_manual_data():
    orders = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "order_id": ["O1", "O2", "O3"],
            "order_date": ["2025-01-01", "2025-01-05", "2025-01-02"],
            "revenue": [100, 150, 50],
        }
    )

    customer_df = calculate_customer_metrics(orders)

    c1 = customer_df.loc[customer_df["customer_id"] == "C1"].iloc[0]
    c2 = customer_df.loc[customer_df["customer_id"] == "C2"].iloc[0]

    assert c1["orders"] == 2
    assert c1["revenue"] == 250
    assert c1["avg_order_value"] == 125
    assert c1["is_repeat_customer"] == True

    assert c2["orders"] == 1
    assert c2["revenue"] == 50


def test_deciles_have_10_rows():
    customer_df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(1, 101)],
            "revenue": list(range(100, 0, -1)),
        }
    )

    decile_df = calculate_revenue_deciles(customer_df)

    assert len(decile_df) == 10
    assert decile_df["customers"].sum() == 100
    assert round(decile_df["revenue_share"].sum(), 10) == 1.0


def test_pareto_summary_increases_with_customer_share():
    customer_df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(1, 101)],
            "revenue": list(range(100, 0, -1)),
        }
    )

    pareto_df = calculate_pareto_summary(customer_df, thresholds=(0.10, 0.20))

    assert pareto_df.loc[0, "customers"] == 10
    assert pareto_df.loc[1, "customers"] == 20
    assert pareto_df.loc[1, "revenue_share"] > pareto_df.loc[0, "revenue_share"]


def test_full_analysis_returns_expected_outputs():
    orders = create_demo_orders(n_customers=500, n_orders=1500, seed=44)
    results = run_customer_analysis(orders)

    assert "customer_metrics" in results
    assert "decile_summary" in results
    assert "pareto_summary" in results
    assert "summary_text" in results

    assert "Top 10% of customers generate" in results["summary_text"]
    assert "Repeat customers generate" in results["summary_text"]

    customer_df = results["customer_metrics"]
    segmented = assign_customer_segments(customer_df)
    assert "segment" in segmented.columns
    assert "High Value" in set(segmented["segment"])
