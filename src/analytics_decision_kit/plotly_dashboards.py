"""Plotly dashboards for customer analytics.

No Jupyter needed. These helpers write standalone HTML dashboards that can be
opened in a browser or published through GitHub Pages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics_decision_kit.customer_analysis import (
    calculate_customer_metrics,
    calculate_revenue_deciles,
)
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)


def calculate_gini(values: pd.Series) -> float:
    """Calculate Gini index for a revenue distribution.

    0 = equal distribution
    1 = maximum concentration
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[x >= 0].sort_values()

    if len(x) == 0:
        return 0.0

    total = x.sum()
    if total == 0:
        return 0.0

    n = len(x)
    rank = pd.Series(range(1, n + 1), index=x.index)

    return float((2 * (rank * x).sum()) / (n * total) - (n + 1) / n)


def calculate_revenue_concentration_snapshot(
    customer_metrics: pd.DataFrame,
    top_customer_share: float = 0.20,
) -> dict[str, float]:
    """Calculate current top-x customer revenue concentration and Gini."""
    if not 0 < top_customer_share <= 1:
        raise ValueError("top_customer_share must be between 0 and 1")

    df = customer_metrics.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].sort_values("revenue", ascending=False)

    if df.empty:
        return {
            "customers": 0,
            "revenue": 0.0,
            "top_customer_share": top_customer_share,
            "top_revenue_share": 0.0,
            "gini_index": 0.0,
        }

    top_n = max(1, int(round(len(df) * top_customer_share)))
    total_revenue = df["revenue"].sum()
    top_revenue = df.head(top_n)["revenue"].sum()

    return {
        "customers": float(len(df)),
        "revenue": float(total_revenue),
        "top_customer_share": float(top_customer_share),
        "top_revenue_share": float(top_revenue / total_revenue if total_revenue else 0),
        "gini_index": calculate_gini(df["revenue"]),
    }


def calculate_concentration_evolution(
    orders: pd.DataFrame,
    period: str = "M",
    top_customer_share: float = 0.20,
) -> pd.DataFrame:
    """Calculate revenue concentration by period.

    For each period, this calculates:
    - total revenue
    - active customers
    - Gini index across customer revenue
    - top x% customer revenue share
    """
    if not 0 < top_customer_share <= 1:
        raise ValueError("top_customer_share must be between 0 and 1")

    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["order_date", "revenue", "customer_id"])
    df = df[df["revenue"] > 0].copy()

    df["period"] = df["order_date"].dt.to_period(period).dt.to_timestamp()

    rows = []
    for period_value, period_df in df.groupby("period"):
        by_customer = (
            period_df.groupby("customer_id", as_index=False)
            .agg(revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
        )

        total_customers = len(by_customer)
        total_revenue = by_customer["revenue"].sum()
        top_n = max(1, int(round(total_customers * top_customer_share)))
        top_revenue = by_customer.head(top_n)["revenue"].sum()

        rows.append(
            {
                "period": period_value,
                "customers": total_customers,
                "revenue": total_revenue,
                "top_customer_share": top_customer_share,
                "top_revenue_share": top_revenue / total_revenue if total_revenue else 0,
                "gini_index": calculate_gini(by_customer["revenue"]),
            }
        )

    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def calculate_rfm_scores(customer_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5 x 5 x 5 RFM scores from customer metrics.

    R = recency score, higher is better.
    F = frequency score, higher is better.
    M = monetary score, higher is better.
    """
    df = customer_metrics.copy()

    max_date = pd.to_datetime(df["last_order_date"]).max()
    df["recency_days"] = (max_date - pd.to_datetime(df["last_order_date"])).dt.days

    df["r_score"] = _safe_qcut_score(df["recency_days"], reverse=True)
    df["f_score"] = _safe_qcut_score(df["orders"], reverse=False)
    df["m_score"] = _safe_qcut_score(df["revenue"], reverse=False)

    df["rfm_cell"] = (
        "R" + df["r_score"].astype(str)
        + " F" + df["f_score"].astype(str)
        + " M" + df["m_score"].astype(str)
    )

    return df


def calculate_rfm_matrix(rfm_customers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 5 x 5 Recency/Frequency matrix."""
    total_revenue = rfm_customers["revenue"].sum()

    matrix = (
        rfm_customers.groupby(["r_score", "f_score"], as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
            avg_monetary=("revenue", "mean"),
            avg_orders=("orders", "mean"),
            avg_recency_days=("recency_days", "mean"),
        )
    )

    matrix["revenue_share"] = matrix["revenue"] / total_revenue if total_revenue else 0

    all_cells = pd.MultiIndex.from_product(
        [range(1, 6), range(1, 6)],
        names=["r_score", "f_score"],
    ).to_frame(index=False)

    matrix = all_cells.merge(matrix, on=["r_score", "f_score"], how="left")
    fill_cols = [
        "customers",
        "revenue",
        "avg_monetary",
        "avg_orders",
        "avg_recency_days",
        "revenue_share",
    ]
    matrix[fill_cols] = matrix[fill_cols].fillna(0)

    return matrix


def create_customer_analytics_dashboard(
    orders: pd.DataFrame | None = None,
    output_path: str | Path = "docs/plotly/customer_analytics_dashboard.html",
    top_customer_share: float = 0.20,
    transaction_sample_size: int = 1000,
) -> Path:
    """Create a combined Plotly customer analytics dashboard."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.io import to_html
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for dashboards. Install it with: pip install plotly"
        ) from exc

    if orders is None:
        orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)

    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["revenue"] = pd.to_numeric(orders["revenue"], errors="coerce")
    orders = orders.dropna(subset=["order_date", "revenue"])
    orders = orders[orders["revenue"] > 0].copy()

    customer_metrics = calculate_customer_metrics(orders)
    deciles = calculate_revenue_deciles(customer_metrics)
    snapshot = calculate_revenue_concentration_snapshot(
        customer_metrics,
        top_customer_share=top_customer_share,
    )
    evolution = calculate_concentration_evolution(
        orders,
        top_customer_share=top_customer_share,
    )
    rfm_customers = calculate_rfm_scores(customer_metrics)
    rfm_matrix = calculate_rfm_matrix(rfm_customers)

    top_pct = int(round(snapshot["top_customer_share"] * 100))
    top_rev_pct = snapshot["top_revenue_share"] * 100
    gini = snapshot["gini_index"]

    decile_plot = deciles.copy()
    decile_plot["decile_label"] = "Decile " + decile_plot["decile"].astype(str)
    decile_plot["revenue_share_pct"] = decile_plot["revenue_share"] * 100
    decile_plot["cumulative_revenue_share_pct"] = decile_plot["cumulative_revenue_share"] * 100

    fig_deciles = px.bar(
        decile_plot,
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
        title=(
            f"Revenue concentration: top {top_pct}% of customers cover "
            f"{top_rev_pct:.1f}% of revenue | Gini {gini:.2f}"
        ),
        labels={
            "decile_label": "Customer revenue decile",
            "revenue_share_pct": "Revenue share (%)",
            "cumulative_revenue_share_pct": "Cumulative revenue share (%)",
        },
    )
    fig_deciles.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_deciles.update_layout(yaxis_ticksuffix="%")

    evolution_plot = evolution.copy()
    evolution_plot["top_revenue_share_pct"] = evolution_plot["top_revenue_share"] * 100

    fig_evolution = make_subplots(specs=[[{"secondary_y": True}]])
    fig_evolution.add_trace(
        go.Scatter(
            x=evolution_plot["period"],
            y=evolution_plot["top_revenue_share_pct"],
            mode="lines+markers",
            name=f"Top {top_pct}% revenue share",
            hovertemplate="%{x|%Y-%m}<br>Top share: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_evolution.add_trace(
        go.Scatter(
            x=evolution_plot["period"],
            y=evolution_plot["gini_index"],
            mode="lines+markers",
            name="Gini index",
            hovertemplate="%{x|%Y-%m}<br>Gini: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_evolution.update_layout(
        title=(
            f"Gini evolution: latest top {top_pct}% customers cover "
            f"{top_rev_pct:.1f}% of revenue"
        )
    )
    fig_evolution.update_yaxes(title_text="Top customer revenue share (%)", secondary_y=False)
    fig_evolution.update_yaxes(title_text="Gini index", range=[0, 1], secondary_y=True)

    heatmap = rfm_matrix.pivot(index="r_score", columns="f_score", values="revenue_share")
    heatmap = heatmap.sort_index(ascending=False)
    heatmap_values = heatmap.values * 100

    fig_rfm_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_values,
            x=[f"F{x}" for x in heatmap.columns],
            y=[f"R{x}" for x in heatmap.index],
            colorscale="Viridis",
            text=[[f"{value:.1f}%" for value in row] for row in heatmap_values],
            texttemplate="%{text}",
            hovertemplate="Frequency: %{x}<br>Recency: %{y}<br>Revenue share: %{z:.1f}%<extra></extra>",
        )
    )
    fig_rfm_heatmap.update_layout(
        title="RFM 5 x 5 relief heatmap: revenue share by recency and frequency",
        xaxis_title="Frequency score, 5 = highest",
        yaxis_title="Recency score, 5 = most recent",
    )

    surface = rfm_matrix.pivot(index="r_score", columns="f_score", values="avg_monetary")
    surface = surface.sort_index(ascending=True)

    fig_rfm_surface = go.Figure(
        data=[
            go.Surface(
                z=surface.values,
                x=[f"F{x}" for x in surface.columns],
                y=[f"R{x}" for x in surface.index],
                colorscale="Viridis",
                hovertemplate="Frequency: %{x}<br>Recency: %{y}<br>Avg monetary: %{z:,.0f}<extra></extra>",
            )
        ]
    )
    fig_rfm_surface.update_layout(
        title="RFM relief surface: average monetary value by 5 x 5 RF groups",
        scene=dict(
            xaxis_title="Frequency score",
            yaxis_title="Recency score",
            zaxis_title="Avg monetary",
        ),
    )

    sample = orders.copy()
    if len(sample) > transaction_sample_size:
        sample = sample.sample(transaction_sample_size, random_state=42).copy()
    sample = sample.sort_values("order_date")

    hover_manufacturer = "manufacturer" if "manufacturer" in sample.columns else "brand"

    fig_transactions = px.scatter(
        sample,
        x="order_date",
        y="revenue",
        size="revenue",
        color="category",
        hover_data={
            "order_id": True,
            "customer_id": True,
            hover_manufacturer: True,
            "category": True,
            "revenue": ":,.2f",
            "order_date": "|%Y-%m-%d",
        },
        title=f"Transaction revenue over time, sampled {len(sample):,} transactions",
        labels={
            "order_date": "Order date",
            "revenue": "Transaction revenue",
            hover_manufacturer: "Manufacturer",
            "category": "Category",
        },
    )
    fig_transactions.update_traces(
        marker=dict(
            sizemode="area",
            sizeref=max(sample["revenue"].max() / 60, 1),
            sizemin=4,
            opacity=0.65,
        )
    )

    html = _build_html_dashboard(
        title="Analytics Decision Kit - Customer Analytics Dashboard",
        sections=[
            ("Revenue deciles", fig_deciles),
            ("Concentration evolution", fig_evolution),
            ("RFM 5 x 5 relief heatmap", fig_rfm_heatmap),
            ("RFM relief surface", fig_rfm_surface),
            ("Transaction bubble chart", fig_transactions),
        ],
        to_html=to_html,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def create_kmeans_plotly_dashboard(
    customers_with_clusters: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    output_path: str | Path = "docs/plotly/kmeans_customer_segmentation_dashboard.html",
) -> Path:
    """Create a Plotly dashboard for KMeans customer segmentation."""
    try:
        import plotly.express as px
        from plotly.io import to_html
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for dashboards. Install it with: pip install plotly"
        ) from exc

    customers = customers_with_clusters.copy()
    profile = cluster_profile.copy()

    profile["customer_share_pct"] = profile["customer_share"] * 100
    profile["revenue_share_pct"] = profile["revenue_share"] * 100

    fig_share = px.bar(
        profile.sort_values("revenue_share_pct"),
        x=["customer_share_pct", "revenue_share_pct"],
        y="cluster_name",
        orientation="h",
        barmode="group",
        title="KMeans cluster profile: customer share vs revenue share",
        labels={
            "value": "Share (%)",
            "cluster_name": "Cluster",
            "variable": "Metric",
        },
    )

    fig_scatter = px.scatter(
        customers.sample(min(2000, len(customers)), random_state=42),
        x="orders",
        y="revenue",
        size="avg_order_value",
        color="cluster_name",
        hover_data={
            "customer_id": True,
            "cluster": True,
            "orders": ":,.0f",
            "revenue": ":,.2f",
            "avg_order_value": ":,.2f",
            "customer_age_days": ":,.0f",
        },
        title="KMeans customer map: orders vs revenue",
        labels={
            "orders": "Orders",
            "revenue": "Revenue",
            "avg_order_value": "Average order value",
            "cluster_name": "Cluster",
        },
    )

    fig_avg = px.bar(
        profile.sort_values("avg_revenue"),
        x="avg_revenue",
        y="cluster_name",
        orientation="h",
        title="Average revenue per customer by KMeans cluster",
        labels={
            "avg_revenue": "Average revenue per customer",
            "cluster_name": "Cluster",
        },
    )

    html = _build_html_dashboard(
        title="Analytics Decision Kit - KMeans Segmentation Dashboard",
        sections=[
            ("Cluster share profile", fig_share),
            ("Customer map", fig_scatter),
            ("Average revenue by cluster", fig_avg),
        ],
        to_html=to_html,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def _safe_qcut_score(values: pd.Series, reverse: bool) -> pd.Series:
    """Return 1-5 quantile scores without failing on duplicate values."""
    numeric = pd.to_numeric(values, errors="coerce")
    rank = numeric.rank(method="first")

    if reverse:
        score = pd.qcut(rank, 5, labels=[5, 4, 3, 2, 1])
    else:
        score = pd.qcut(rank, 5, labels=[1, 2, 3, 4, 5])

    return score.astype(int)


def _build_html_dashboard(title: str, sections: list[tuple[str, object]], to_html) -> str:
    """Build a simple multi-chart HTML dashboard."""
    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        """
<style>
body {
    font-family: Arial, sans-serif;
    margin: 32px;
    background: #f6f8fa;
    color: #24292f;
}
h1 {
    margin-bottom: 8px;
}
.section {
    background: white;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 18px;
    margin: 22px 0;
}
.note {
    color: #57606a;
    margin-bottom: 24px;
}
</style>
""",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        '<div class="note">Generated with Plotly. No Jupyter required.</div>',
    ]

    for idx, (section_title, fig) in enumerate(sections):
        include_js = "cdn" if idx == 0 else False
        parts.append(f'<div class="section"><h2>{section_title}</h2>')
        parts.append(
            to_html(
                fig,
                include_plotlyjs=include_js,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
        )
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)
