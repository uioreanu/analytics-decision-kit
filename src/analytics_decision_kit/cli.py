"""Command line entry points for Analytics Decision Kit."""

from analytics_decision_kit.brand_pca import create_brand_pca_dashboard, run_brand_pca
from analytics_decision_kit.customer_analysis import (
    calculate_customer_metrics,
    run_customer_analysis,
)
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)
from analytics_decision_kit.kmeans_segmentation import run_kmeans_segmentation
from analytics_decision_kit.plotly_dashboards import (
    create_customer_analytics_dashboard,
    create_kmeans_plotly_dashboard,
)
from analytics_decision_kit.sample_data import create_demo_orders


def run_demo() -> None:
    """Run the basic customer concentration demo."""
    orders = create_demo_orders(n_customers=5000, n_orders=18000, seed=42)
    results = run_customer_analysis(orders)

    print("\n=== Revenue deciles ===")
    print(results["decile_summary"].to_string(index=False))

    print("\n=== Executive summary ===")
    print(results["summary_text"])


def create_dashboard() -> None:
    """Create the combined customer analytics Plotly dashboard."""
    orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)

    output_path = create_customer_analytics_dashboard(
        orders=orders,
        output_path="docs/plotly/customer_analytics_dashboard.html",
        top_customer_share=0.20,
        transaction_sample_size=1000,
    )

    print("\nCreated dashboard:")
    print(output_path)


def create_kmeans_dashboard() -> None:
    """Create the KMeans customer segmentation Plotly dashboard."""
    orders = create_demo_orders(n_customers=5000, n_orders=18000, seed=42)
    customer_metrics = calculate_customer_metrics(orders)

    result = run_kmeans_segmentation(customer_metrics, k_min=5, k_max=7)

    output_path = create_kmeans_plotly_dashboard(
        customers_with_clusters=result.customers,
        cluster_profile=result.cluster_profile,
        output_path="docs/plotly/kmeans_customer_segmentation_dashboard.html",
    )

    print(f"\nSelected k: {result.selected_k}")
    print("\nCreated KMeans dashboard:")
    print(output_path)