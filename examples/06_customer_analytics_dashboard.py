"""Create a combined Plotly customer analytics dashboard.

This uses the public Sample Superstore-style dataset by default.

Run:
python examples/06_customer_analytics_dashboard.py
"""

from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)
from analytics_decision_kit.plotly_dashboards import create_customer_analytics_dashboard


orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)

output_path = create_customer_analytics_dashboard(
    orders=orders,
    output_path="docs/plotly/customer_analytics_dashboard.html",
    top_customer_share=0.20,
    transaction_sample_size=1000,
)

print("\nCreated dashboard:")
print(output_path)
