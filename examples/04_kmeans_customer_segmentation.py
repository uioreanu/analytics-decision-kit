"""KMeans customer segmentation example.

Run:
python examples/04_kmeans_customer_segmentation.py
"""

from analytics_decision_kit.customer_analysis import calculate_customer_metrics
from analytics_decision_kit.kmeans_segmentation import run_kmeans_segmentation
from analytics_decision_kit.sample_data import create_demo_orders


orders = create_demo_orders(n_customers=2500, n_orders=9000, seed=42)
customer_metrics = calculate_customer_metrics(orders)

result = run_kmeans_segmentation(customer_metrics, k_min=5, k_max=7)

print(f"\nSelected k: {result.selected_k}")

print("\n=== Diagnostics ===")
print(result.diagnostics.to_string(index=False))

print("\n=== Cluster profile ===")
print(
    result.cluster_profile.to_string(
        index=False,
        formatters={
            "customer_share": "{:.1%}".format,
            "revenue_share": "{:.1%}".format,
            "avg_revenue": "{:,.0f}".format,
            "median_revenue": "{:,.0f}".format,
            "avg_orders": "{:.1f}".format,
            "avg_order_value": "{:,.0f}".format,
            "avg_customer_age_days": "{:.0f}".format,
            "avg_revenue_per_day": "{:,.1f}".format,
        },
    )
)

print("\n=== Sample customers with cluster names ===")

sample_customers = (
    result.customers
    .sample(n=min(10, len(result.customers)), random_state=42)
    [["customer_id", "revenue", "orders", "cluster", "cluster_name"]]
    .sort_values(["cluster", "revenue"], ascending=[True, False])
)

print(sample_customers)