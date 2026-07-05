"""First customer concentration example.

Run:
python examples/01_customer_concentration.py
"""

import pandas as pd

from analytics_decision_kit.sample_data import create_demo_orders
from analytics_decision_kit.customer_analysis import run_customer_analysis


# create synthetic order data
orders = create_demo_orders(n_customers=10000, n_orders=50000, seed=666)

# quick inspect, like I would do in notebook
print("\n=== raw orders ===")
print(orders.head())
print(orders.shape)

# run complete analysis
results = run_customer_analysis(orders)

customer_df = results["customer_metrics"]
decile_df = results["decile_summary"]
pareto_df = results["pareto_summary"]

print("\n=== customer metrics ===")
print(customer_df.head())

print("\n=== revenue deciles ===")
print(
    decile_df.to_string(
        index=False,
        formatters={
            "revenue_share": "{:.1%}".format,
            "cumulative_revenue_share": "{:.1%}".format,
        },
    )
)

print("\n=== pareto summary ===")
print(
    pareto_df.to_string(
        index=False,
        formatters={
            "top_customer_share": "{:.0%}".format,
            "revenue_share": "{:.1%}".format,
        },
    )
)

print("\n=== executive summary ===")
print(results["summary_text"])

# optional save output for docs / review
# decile_df.to_csv("data/decile_summary.csv", index=False)
# customer_df.to_csv("data/customer_metrics.csv", index=False)
