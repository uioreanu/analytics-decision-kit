"""Same analysis, written more like a notebook.

This is intentionally less abstract. 
"""

import pandas as pd

from analytics_decision_kit.sample_data import create_demo_orders


orders = create_demo_orders(n_customers=3000, n_orders=100000, seed=99)

# check raw data
print(orders.head())
print(orders.shape)
print(orders["revenue"].describe())

# customer-level table
df = orders.copy()
df["order_date"] = pd.to_datetime(df["order_date"])
df["revenue"] = pd.to_numeric(df["revenue"])

customer_df = (
    df.groupby("customer_id")
    .agg(
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        orders=("order_id", "nunique"),
        revenue=("revenue", "sum"),
    )
    .reset_index()
)

customer_df["avg_order_value"] = customer_df["revenue"] / customer_df["orders"]
customer_df["is_repeat_customer"] = customer_df["orders"] >= 2
customer_df = customer_df.sort_values("revenue", ascending=False).reset_index(drop=True)

# deciles by revenue
customer_df["decile"] = (customer_df.index * 10 / len(customer_df)).astype(int) + 1
customer_df["decile"] = customer_df["decile"].clip(upper=10)

decile_summary = (
    customer_df.groupby("decile")
    .agg(
        customers=("customer_id", "nunique"),
        revenue=("revenue", "sum"),
        avg_revenue_per_customer=("revenue", "mean"),
    )
    .reset_index()
)

decile_summary["revenue_share"] = decile_summary["revenue"] / decile_summary["revenue"].sum()
decile_summary["cumulative_revenue_share"] = decile_summary["revenue_share"].cumsum()

print(decile_summary)
