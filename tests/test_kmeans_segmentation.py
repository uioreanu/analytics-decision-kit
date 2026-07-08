from analytics_decision_kit.customer_analysis import calculate_customer_metrics
from analytics_decision_kit.kmeans_segmentation import run_kmeans_segmentation
from analytics_decision_kit.sample_data import create_demo_orders


def test_run_kmeans_segmentation_selects_cluster_count_between_5_and_7():
    orders = create_demo_orders(n_customers=300, n_orders=900, seed=123)
    customer_metrics = calculate_customer_metrics(orders)

    result = run_kmeans_segmentation(customer_metrics, k_min=5, k_max=7)

    assert 5 <= result.selected_k <= 7
    assert "cluster" in result.customers.columns
    assert "cluster_name" in result.customers.columns
    assert result.customers["cluster"].nunique() == result.selected_k
    assert len(result.cluster_profile) == result.selected_k
    assert {"silhouette_score", "inertia", "k"}.issubset(result.diagnostics.columns)
