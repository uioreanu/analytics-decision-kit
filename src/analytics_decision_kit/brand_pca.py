"""Brand-level PCA and biplot helpers.

This module stays separate from customer analysis on purpose:
- customer analysis builds customer KPIs
- KMeans builds customer clusters
- this module aggregates to brands and runs PCA on brand-level features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analytics_decision_kit.customer_analysis import calculate_customer_metrics
from analytics_decision_kit.kmeans_segmentation import (
    KMeansSegmentationResult,
    run_kmeans_segmentation,
)


BASE_BRAND_FEATURES = [
    "order_count",
    "customer_count",
    "total_revenue",
    "avg_order_value",
    "revenue_per_customer",
    "existing_customers",
    "existing_customer_share",
    "existing_customer_revenue_share",
    "brand_customer_share",
    "brand_revenue_share",
]


@dataclass(frozen=True)
class BrandPCAResult:
    """Container for brand PCA outputs."""

    brand_metrics: pd.DataFrame
    brand_scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    features: list[str]
    customer_metrics: pd.DataFrame
    kmeans_result: KMeansSegmentationResult
    pca_model: PCA


def calculate_brand_metrics(
    orders: pd.DataFrame,
    brand_col: str = "brand",
    customer_col: str = "customer_id",
    order_col: str = "order_id",
    revenue_col: str = "revenue",
    k_min: int = 5,
    k_max: int = 7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeansSegmentationResult]:
    """Aggregate transactions to brand-level metrics.

    The brand table includes base commercial metrics plus derived KMeans
    cluster-share features. Cluster-share columns are prefixed with ``derived_``
    so they are clearly distinguishable from the core brand KPIs.
    """
    _check_columns(orders, [brand_col, customer_col, order_col, revenue_col])

    if len(orders) == 0:
        raise ValueError("orders is empty")

    df = orders.copy()
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")
    df = df.dropna(subset=[brand_col, customer_col, order_col, revenue_col])
    df = df[df[revenue_col] >= 0].copy()

    customer_metrics = calculate_customer_metrics(
        df,
        customer_col=customer_col,
        order_col=order_col,
        revenue_col=revenue_col,
    )
    kmeans_result = run_kmeans_segmentation(
        customer_metrics,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

    customer_cluster_lookup = kmeans_result.customers[
        [customer_col, "cluster", "cluster_name", "is_repeat_customer"]
    ].drop_duplicates(subset=[customer_col])

    brand_orders = df.merge(customer_cluster_lookup, on=customer_col, how="left")
    brand_customers = customer_metrics[[customer_col, "is_repeat_customer"]].copy()

    brand_base = (
        brand_orders.groupby(brand_col, as_index=False)
        .agg(
            order_count=(order_col, "nunique"),
            customer_count=(customer_col, "nunique"),
            total_revenue=(revenue_col, "sum"),
        )
    )

    brand_aov = (
        brand_orders.groupby(brand_col, as_index=False)
        .agg(
            total_revenue=(revenue_col, "sum"),
            order_count=(order_col, "nunique"),
        )
    )
    brand_aov["avg_order_value"] = brand_aov["total_revenue"] / brand_aov["order_count"].clip(lower=1)
    brand_aov = brand_aov[[brand_col, "avg_order_value"]]

    brand_repeat = (
        brand_orders[[brand_col, customer_col]]
        .drop_duplicates()
        .merge(brand_customers, on=customer_col, how="left")
        .groupby(brand_col, as_index=False)
        .agg(
            existing_customers=("is_repeat_customer", "sum"),
        )
    )

    brand_repeat["existing_customers"] = brand_repeat["existing_customers"].fillna(0)

    brand_repeat_revenue = (
        brand_orders[[brand_col, customer_col, revenue_col]]
        .merge(brand_customers, on=customer_col, how="left")
    )
    brand_repeat_revenue = (
        brand_repeat_revenue[brand_repeat_revenue["is_repeat_customer"] == True]  # noqa: E712
        .groupby(brand_col, as_index=False)
        .agg(
            existing_customer_revenue=(revenue_col, "sum"),
        )
    )

    brand_metrics = (
        brand_base.merge(brand_aov, on=brand_col, how="left")
        .merge(brand_repeat, on=brand_col, how="left")
        .merge(brand_repeat_revenue, on=brand_col, how="left")
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    total_brand_customers = brand_metrics["customer_count"].sum()
    total_brand_revenue = brand_metrics["total_revenue"].sum()
    brand_metrics["revenue_per_customer"] = (
        brand_metrics["total_revenue"] / brand_metrics["customer_count"].clip(lower=1)
    )
    brand_metrics["existing_customer_share"] = (
        brand_metrics["existing_customers"] / brand_metrics["customer_count"].clip(lower=1)
    )
    brand_metrics["existing_customer_revenue_share"] = (
        brand_metrics["existing_customer_revenue"] / brand_metrics["total_revenue"].replace(0, np.nan)
    ).fillna(0)
    brand_metrics["brand_customer_share"] = (
        brand_metrics["customer_count"] / total_brand_customers if total_brand_customers else 0
    )
    brand_metrics["brand_revenue_share"] = (
        brand_metrics["total_revenue"] / total_brand_revenue if total_brand_revenue else 0
    )

    brand_metrics = _add_cluster_share_features(
        brand_metrics=brand_metrics,
        brand_orders=brand_orders,
        brand_col=brand_col,
        revenue_col=revenue_col,
    )

    return brand_metrics, customer_metrics, kmeans_result


def run_brand_pca(
    orders: pd.DataFrame,
    features: list[str] | None = None,
    brand_col: str = "brand",
    customer_col: str = "customer_id",
    order_col: str = "order_id",
    revenue_col: str = "revenue",
    k_min: int = 5,
    k_max: int = 7,
    random_state: int = 42,
) -> BrandPCAResult:
    """Run brand-level PCA using derived brand metrics."""
    brand_metrics, customer_metrics, kmeans_result = calculate_brand_metrics(
        orders,
        brand_col=brand_col,
        customer_col=customer_col,
        order_col=order_col,
        revenue_col=revenue_col,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

    if features is None:
        derived_share_features = [
            col for col in brand_metrics.columns if col.startswith("derived_")
        ]
        features = BASE_BRAND_FEATURES + derived_share_features

    _check_features(brand_metrics, features)

    if len(brand_metrics) < 3:
        raise ValueError("At least 3 brands are needed for a useful PCA view.")

    X = _prepare_feature_matrix(brand_metrics, features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    brand_scores = brand_metrics.copy()
    brand_scores["pc1"] = scores[:, 0]
    brand_scores["pc2"] = scores[:, 1]

    loadings = pd.DataFrame(
        {
            "feature": features,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
        }
    )
    loadings["vector_length"] = np.sqrt(
        loadings["pc1_loading"] ** 2 + loadings["pc2_loading"] ** 2
    )
    loadings["feature_type"] = np.where(
        loadings["feature"].str.startswith("derived_"),
        "derived",
        "base",
    )
    loadings = loadings.sort_values("vector_length", ascending=False).reset_index(drop=True)

    explained_variance = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    explained_variance["cumulative_explained_variance"] = explained_variance[
        "explained_variance_ratio"
    ].cumsum()

    return BrandPCAResult(
        brand_metrics=brand_metrics,
        brand_scores=brand_scores,
        loadings=loadings,
        explained_variance=explained_variance,
        features=features,
        customer_metrics=customer_metrics,
        kmeans_result=kmeans_result,
        pca_model=pca,
    )


def create_brand_pca_dashboard(
    result: BrandPCAResult,
    output_path: str | Path = "docs/plotly/brand_pca_dashboard.html",
    max_loadings: int = 12,
) -> Path:
    """Create a Plotly biplot dashboard for the brand PCA result."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for dashboards. Install it with: pip install plotly"
        ) from exc

    brands = result.brand_scores.copy()
    loadings = result.loadings.head(max_loadings).copy()
    ev = result.explained_variance

    x_span = max(brands["pc1"].abs().max(), 1e-9)
    y_span = max(brands["pc2"].abs().max(), 1e-9)
    loading_scale = 0.75 * min(x_span, y_span) / max(loadings[["pc1_loading", "pc2_loading"]].abs().to_numpy().max(), 1e-9)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=brands["pc1"],
            y=brands["pc2"],
            mode="markers+text",
            text=brands["brand"],
            textposition="top center",
            marker=dict(size=12, color="#1f77b4", opacity=0.85),
            hovertemplate=
                "Brand: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
                "<br>Total revenue: %{customdata[0]:,.0f}"
                "<br>Customer count: %{customdata[1]:,.0f}"
                "<extra></extra>",
            customdata=brands[["total_revenue", "customer_count"]].to_numpy(),
            name="Brands",
        )
    )

    for row in loadings.itertuples(index=False):
        x_end = float(row.pc1_loading) * loading_scale
        y_end = float(row.pc2_loading) * loading_scale
        fig.add_trace(
            go.Scatter(
                x=[0, x_end],
                y=[0, y_end],
                mode="lines+markers+text",
                text=["", row.feature],
                textposition="top center",
                line=dict(color="#d62728", width=2),
                marker=dict(size=[0, 7], color="#d62728"),
                hovertemplate=f"Feature: {row.feature}<extra></extra>",
                showlegend=False,
                name=row.feature,
            )
        )

    fig.add_hline(y=0, line_width=1, line_color="#9aa0a6")
    fig.add_vline(x=0, line_width=1, line_color="#9aa0a6")

    fig.update_layout(
        title=(
            f"Brand PCA biplot | PC1 {ev.loc[0, 'explained_variance_ratio'] * 100:.1f}%"
            f" | PC2 {ev.loc[1, 'explained_variance_ratio'] * 100:.1f}%"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=850,
        width=1150,
        legend_title_text="",
    )

    fig.add_annotation(
        x=0.99,
        y=0.01,
        xref="paper",
        yref="paper",
        text=(
            f"Features: {len(result.features)} | Brands: {len(brands)} | "
            f"Clusters: {result.kmeans_result.selected_k}"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=12, color="#57606a"),
    )

    html = to_html(fig, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _add_cluster_share_features(
    brand_metrics: pd.DataFrame,
    brand_orders: pd.DataFrame,
    brand_col: str,
    revenue_col: str,
) -> pd.DataFrame:
    """Add per-cluster customer and revenue share features to the brand table."""
    cluster_values = sorted(
        value for value in brand_orders["cluster"].dropna().unique().tolist()
    )

    if len(cluster_values) == 0:
        return brand_metrics

    rows = []
    brand_customer_counts = brand_orders.groupby(brand_col)["customer_id"].nunique()
    brand_revenue_totals = brand_orders.groupby(brand_col)[revenue_col].sum()

    unique_brand_customers = brand_orders[[brand_col, "customer_id", "cluster"]].drop_duplicates()
    brand_customer_cluster_counts = (
        unique_brand_customers.groupby([brand_col, "cluster"])["customer_id"]
        .nunique()
        .reset_index(name="customers")
    )

    brand_revenue_cluster = (
        brand_orders.groupby([brand_col, "cluster"], as_index=False)
        .agg(cluster_revenue=(revenue_col, "sum"))
    )

    for brand in brand_metrics[brand_col]:
        brand_customer_total = float(brand_customer_counts.get(brand, 0))
        brand_revenue_total = float(brand_revenue_totals.get(brand, 0))
        row = {brand_col: brand}

        for cluster_value in cluster_values:
            customer_count = brand_customer_cluster_counts.loc[
                (brand_customer_cluster_counts[brand_col] == brand)
                & (brand_customer_cluster_counts["cluster"] == cluster_value),
                "customers",
            ]
            revenue_total = brand_revenue_cluster.loc[
                (brand_revenue_cluster[brand_col] == brand)
                & (brand_revenue_cluster["cluster"] == cluster_value),
                "cluster_revenue",
            ]

            customer_count_value = float(customer_count.iloc[0]) if len(customer_count) > 0 else 0.0
            revenue_value = float(revenue_total.iloc[0]) if len(revenue_total) > 0 else 0.0

            row[f"derived_cluster_{int(cluster_value)}_customer_share"] = (
                customer_count_value / brand_customer_total if brand_customer_total else 0.0
            )
            row[f"derived_cluster_{int(cluster_value)}_revenue_share"] = (
                revenue_value / brand_revenue_total if brand_revenue_total else 0.0
            )

        rows.append(row)

    derived_df = pd.DataFrame(rows)
    return brand_metrics.merge(derived_df, on=brand_col, how="left")


def _prepare_feature_matrix(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Clean brand features before PCA."""
    X = data[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], pd.NA)
    X = X.fillna(X.median(numeric_only=True))

    log_features = {
        "order_count",
        "customer_count",
        "total_revenue",
        "avg_order_value",
        "revenue_per_customer",
        "existing_customers",
        "existing_customer_revenue_share",
    }
    for col in features:
        if col in log_features:
            X[col] = X[col].clip(lower=0)
            X[col] = np.log1p(X[col])

    return X


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when a required column is missing."""
    missing_cols = [x for x in columns if x not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns: {missing_cols}")


def _check_features(data: pd.DataFrame, features: list[str]) -> None:
    """Raise helpful errors before fitting."""
    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(f"Missing PCA feature columns: {missing}")

    if len(data) < 3:
        raise ValueError("At least 3 brands are needed for PCA.")
