import pandas as pd

from analytics_decision_kit.data_loader import get_transactional_data, normalize_transactional_data


def test_get_transactional_data_without_url_returns_synthetic_data():
    orders = get_transactional_data(n_customers=100, n_orders=300, seed=1)
    assert len(orders) == 300
    assert {"order_id", "customer_id", "order_date", "revenue", "category", "brand"}.issubset(orders.columns)


def test_normalize_superstore_style_data():
    raw = pd.DataFrame(
        {
            "Order ID": ["CA-1", "CA-1", "CA-2"],
            "Order Date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "Customer ID": ["C1", "C1", "C2"],
            "Category": ["Furniture", "Technology", "Office Supplies"],
            "Product Name": ["Bush Bookcase", "Canon Printer", "Xerox Paper"],
            "Sales": [100.0, 200.0, 50.0],
        }
    )
    orders = normalize_transactional_data(raw)
    assert list(orders.columns) == ["order_id", "customer_id", "order_date", "revenue", "category", "brand"]
    assert orders["revenue"].sum() == 350.0
    assert set(orders["brand"]) == {"Bush", "Canon", "Xerox"}


def test_get_transactional_data_from_local_csv(tmp_path):
    raw = pd.DataFrame(
        {
            "Order ID": ["CA-1"],
            "Order Date": ["2025-01-01"],
            "Customer ID": ["C1"],
            "Category": ["Furniture"],
            "Product Name": ["Bush Bookcase"],
            "Sales": [100.0],
        }
    )
    csv_path = tmp_path / "superstore_sample.csv"
    raw.to_csv(csv_path, index=False)
    orders = get_transactional_data(str(csv_path))
    assert len(orders) == 1
    assert orders.loc[0, "brand"] == "Bush"
