"""Run the analysis on a public GitHub CSV dataset.

This uses a public Sample Superstore-style CSV and normalizes it to:

order_id, customer_id, order_date, revenue, category, brand

Run:
python examples/03_public_github_dataset.py
"""

from analytics_decision_kit.customer_analysis import run_customer_analysis
from analytics_decision_kit.data_loader import DEFAULT_SUPERSTORE_GITHUB_URL, get_transactional_data


orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)
results = run_customer_analysis(orders)

print("\n=== normalized public dataset ===")
print(orders.head())
print(orders.shape)

print("\n=== revenue deciles ===")
print(
    results["decile_summary"].to_string(
        index=False,
        formatters={
            "revenue_share": "{:.1%}".format,
            "cumulative_revenue_share": "{:.1%}".format,
        },
    )
)

print("\n=== executive summary ===")
print(results["summary_text"])
