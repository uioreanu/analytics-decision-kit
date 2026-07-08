"""Plotly visual examples using the public transactional dataset.

Run from repo root:

python examples/05_plotly_visuals_public_dataset.py

This creates dynamic HTML files in:

outputs/plotly/revenue_deciles_public_dataset.html
outputs/plotly/transactions_bubble_public_dataset.html

No Jupyter needed. Plotly writes standalone HTML files that can be opened in a browser.
"""

from pathlib import Path

import pandas as pd

try:
    import plotly.express as px
except ImportError as exc:
    raise SystemExit(
        "Plotly is not installed. Run:\n\n"
        "pip install plotly\n\n"
        "Then run this script again."
    ) from exc

from analytics_decision_kit.customer_analysis import (
    calculate_customer_metrics,
    calculate_revenue_deciles,
)
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)


OUTPUT_DIR = Path("outputs") / "plotly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_revenue_decile_chart(decile_df: pd.DataFrame) -> Path:
    """Create revenue share by customer decile chart."""
    plot_df = decile_df.copy()
    plot_df["decile_label"] = "Decile " + plot_df["decile"].astype(str)
    plot_df["revenue_share_pct"] = plot_df["revenue_share"] * 100
    plot_df["cumulative_revenue_share_pct"] = plot_df["cumulative_revenue_share"] * 100

    fig = px.bar(
        plot_df,
        x="decile_label",
        y="revenue_share_pct",
        text="revenue_share_pct",
        hover_data={
            "customers": ":,",
            "revenue": ":,.2f",
            "avg_revenue_per_customer": ":,.2f",
            "cumulative_revenue_share_pct": ":.1f",
            "revenue_share_pct": ":.1f",
            "decile_label": False,
        },
        title="Revenue distribution by customer decile",
        labels={
            "decile_label": "Customer revenue decile",
            "revenue_share_pct": "Revenue share (%)",
            "cumulative_revenue_share_pct": "Cumulative revenue share (%)",
            "customers": "Customers",
            "revenue": "Revenue",
            "avg_revenue_per_customer": "Avg revenue / customer",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title="Customer decile, 1 = highest revenue customers",
        yaxis_title="Revenue share (%)",
        yaxis_ticksuffix="%",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    output_path = OUTPUT_DIR / "revenue_deciles_public_dataset.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def create_transaction_bubble_chart(orders: pd.DataFrame, sample_size: int = 1000) -> Path:
    """Create transaction-level bubble chart.

    x-axis: order date
    y-axis: transaction revenue
    bubble size: transaction revenue
    color: category
    one bubble: one transaction row
    """
    plot_df = orders.copy()
    plot_df["order_date"] = pd.to_datetime(plot_df["order_date"])
    plot_df["revenue"] = pd.to_numeric(plot_df["revenue"], errors="coerce")
    plot_df = plot_df.dropna(subset=["order_date", "revenue"])
    plot_df = plot_df[plot_df["revenue"] > 0].copy()

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42).copy()

    plot_df = plot_df.sort_values("order_date")

    fig = px.scatter(
        plot_df,
        x="order_date",
        y="revenue",
        size="revenue",
        color="category",
        hover_data={
            "order_id": True,
            "customer_id": True,
            "brand": True,
            "category": True,
            "revenue": ":,.2f",
            "order_date": "|%Y-%m-%d",
        },
        title=f"Transaction revenue over time, sampled {len(plot_df):,} transactions",
        labels={
            "order_date": "Order date",
            "revenue": "Transaction revenue",
            "category": "Category",
            "brand": "Brand",
            "order_id": "Order ID",
            "customer_id": "Customer ID",
        },
    )

    fig.update_traces(
        marker=dict(
            sizemode="area",
            sizeref=max(plot_df["revenue"].max() / 60, 1),
            sizemin=4,
            opacity=0.65,
        )
    )
    fig.update_layout(
        xaxis_title="Order date",
        yaxis_title="Transaction revenue",
    )

    output_path = OUTPUT_DIR / "transactions_bubble_public_dataset.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main() -> None:
    orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)

    customer_metrics = calculate_customer_metrics(orders)
    decile_df = calculate_revenue_deciles(customer_metrics)

    decile_chart = create_revenue_decile_chart(decile_df)
    bubble_chart = create_transaction_bubble_chart(orders, sample_size=1000)

    print("\nCreated Plotly HTML outputs:")
    print(decile_chart)
    print(bubble_chart)


if __name__ == "__main__":
    main()
