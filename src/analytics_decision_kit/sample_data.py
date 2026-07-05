"""Create fake order data for the examples.

boring on purpose: no real brands, no real customer data, no clever internal stuff. Just enough syntetic data to make the notebook function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_demo_orders(
    n_customers: int = 5000,
    n_orders: int = 18000,
    seed: int = 42,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Create a synthetic ecommerce orders table.

    I keep this in a normal function, but it still reads like a notebook cell.
    """
    if n_customers <= 0:
        raise ValueError("n_customers must be > 0")
    if n_orders <= 0:
        raise ValueError("n_orders must be > 0")

    rng = np.random.default_rng(seed)

    customers = np.array([f"C{idx:06d}" for idx in range(1, n_customers + 1)])

    # Some customers buy a lot, most buy little. Retail and inequality 
    customer_weight = rng.lognormal(mean=0.0, sigma=1.0, size=n_customers)
    customer_weight = customer_weight / customer_weight.sum()

    sampled_customer = rng.choice(customers, size=n_orders, replace=True, p=customer_weight)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days = (end - start).days + 1
    sampled_date = start + pd.to_timedelta(rng.integers(0, days, size=n_orders), unit="D")

    # Generic categories. 
    category_list = np.array(["Sportswear", "Ready-to-wear", "Shoes", "Bags", "Beauty"])
    category_prob = np.array([0.27, 0.31, 0.18, 0.16, 0.08])

    brand_list = np.array(["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F"])
    brand_prob = np.array([0.24, 0.21, 0.18, 0.15, 0.12, 0.10])

    sampled_category = rng.choice(category_list, size=n_orders, replace=True, p=category_prob)
    sampled_brand = rng.choice(brand_list, size=n_orders, replace=True, p=brand_prob)

    base_revenue = rng.lognormal(mean=5.6, sigma=0.8, size=n_orders)

    # Very rough multipliers. Fake, generic
    cat_factor = {
        "Sportswear": 1.1,
        "Ready-to-wear": 1.4,
        "Shoes": 1.2,
        "Bags": 2.2,
        "Beauty": 0.45,
    }
    factor = np.array([cat_factor[x] for x in sampled_category])
    revenue = np.round(base_revenue * factor, 2)

    orders = pd.DataFrame(
        {
            "order_id": [f"O{idx:08d}" for idx in range(1, n_orders + 1)],
            "customer_id": sampled_customer,
            "order_date": sampled_date,
            "category": sampled_category,
            "brand": sampled_brand,
            "revenue": revenue,
        }
    )

    # sort makes outputs easier to compare when debugging
    orders = orders.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    return orders


# old name kept
make_synthetic_orders = create_demo_orders
