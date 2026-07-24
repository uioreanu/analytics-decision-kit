from analytics_decision_kit.brand_pca import (
    create_brand_pca_dashboard,
    run_brand_pca,
)
from analytics_decision_kit.sample_data import create_demo_orders


def test_run_brand_pca_returns_expected_outputs():
    orders = create_demo_orders(n_customers=400, n_orders=1200, seed=321)

    result = run_brand_pca(orders)

    assert "pc1" in result.brand_scores.columns
    assert "pc2" in result.brand_scores.columns
    assert len(result.explained_variance) == 2
    assert result.explained_variance["explained_variance_ratio"].sum() > 0
    assert any(col.startswith("derived_") for col in result.brand_metrics.columns)
    assert {"feature", "pc1_loading", "pc2_loading"}.issubset(result.loadings.columns)


def test_create_brand_pca_dashboard_writes_file(tmp_path):
    orders = create_demo_orders(n_customers=400, n_orders=1200, seed=321)
    result = run_brand_pca(orders)

    output_path = tmp_path / "brand_pca_dashboard.html"
    created = create_brand_pca_dashboard(result, output_path=output_path)

    assert created == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
