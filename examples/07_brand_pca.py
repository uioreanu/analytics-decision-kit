"""Brand PCA example.

Run:
python examples/07_brand_pca.py
"""

from analytics_decision_kit.brand_pca import create_brand_pca_dashboard, run_brand_pca
from analytics_decision_kit.sample_data import create_demo_orders


orders = create_demo_orders(n_customers=5000, n_orders=18000, seed=42)

result = run_brand_pca(orders)

print("\n=== brand metrics ===")
print(result.brand_metrics.head())

print("\n=== explained variance ===")
print(result.explained_variance.to_string(index=False))

print("\n=== top loadings ===")
print(
    result.loadings.head(10).to_string(
        index=False,
        formatters={
            "pc1_loading": "{:.3f}".format,
            "pc2_loading": "{:.3f}".format,
            "vector_length": "{:.3f}".format,
        },
    )
)

output_path = create_brand_pca_dashboard(
    result,
    output_path="docs/plotly/brand_pca_dashboard.html",
)

print("\nCreated brand PCA dashboard:")
print(output_path)
