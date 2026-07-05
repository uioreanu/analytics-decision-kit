from analytics_decision_kit.sample_data import create_demo_orders, make_synthetic_orders


def test_create_demo_orders_basic_shape():
    orders = create_demo_orders(n_customers=100, n_orders=500, seed=1)

    assert len(orders) == 500
    assert {
        "order_id",
        "customer_id",
        "order_date",
        "category",
        "brand",
        "revenue",
    }.issubset(orders.columns)
    assert orders["revenue"].gt(0).all()


def test_old_sample_data_name_still_works():
    orders = make_synthetic_orders(n_customers=50, n_orders=100, seed=2)
    assert len(orders) == 100
