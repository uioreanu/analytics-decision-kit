"""KMeans customer segmentation.

Simple unsupervised customer segmentation based on customer-level KPIs.

The module is intentionally generic:
- no employer thresholds
- no private labels
- no industry-specific logic
- no real customer data
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_KMEANS_FEATURES = [
    "orders",
    "revenue",
    "avg_order_value",
    "customer_age_days",
    "revenue_per_day",
]


@dataclass(frozen=True)
class KMeansSegmentationResult:
    """Container for KMeans segmentation outputs."""

    customers: pd.DataFrame
    cluster_profile: pd.DataFrame
    cluster_features: pd.DataFrame
    diagnostics: pd.DataFrame
    selected_k: int
    features: list[str]


def run_kmeans_segmentation(
    customer_metrics: pd.DataFrame,
    features: list[str] | None = None,
    k_min: int = 5,
    k_max: int = 7,
    random_state: int = 42,
) -> KMeansSegmentationResult:
    """Run KMeans segmentation and profile the resulting customer clusters.

    The default searches 5 to 7 clusters and selects the highest silhouette score.
    This is a practical starting point, not a universal truth.
    """
    if features is None:
        features = DEFAULT_KMEANS_FEATURES.copy()

    _check_features(customer_metrics, features)

    model_input = customer_metrics.copy()
    X = _prepare_feature_matrix(model_input, features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    diagnostics_rows = []
    fitted_models = {}

    for k in range(k_min, k_max + 1):
        if k >= len(model_input):
            continue

        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_id = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, cluster_id)

        diagnostics_rows.append(
            {
                "k": k,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )
        fitted_models[k] = model

    diagnostics = pd.DataFrame(diagnostics_rows)

    if diagnostics.empty:
        raise ValueError("No valid k found. Use fewer clusters or more customers.")

    selected_k = int(
        diagnostics.sort_values(
            ["silhouette_score", "k"],
            ascending=[False, True],
        )
        .iloc[0]["k"]
    )

    selected_model = fitted_models[selected_k]
    model_input["cluster"] = selected_model.predict(X_scaled)

    cluster_profile = profile_customer_clusters(model_input)
    cluster_features = profile_cluster_features(model_input, features)
    model_input = add_cluster_names(model_input, cluster_profile)

    return KMeansSegmentationResult(
        customers=model_input,
        cluster_profile=cluster_profile,
        cluster_features=cluster_features,
        diagnostics=diagnostics,
        selected_k=selected_k,
        features=features,
    )


def profile_customer_clusters(customers_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """Create a readable profile table by cluster."""
    required = [
        "cluster",
        "customer_id",
        "orders",
        "revenue",
        "avg_order_value",
        "customer_age_days",
        "revenue_per_day",
    ]
    missing = [col for col in required if col not in customers_with_clusters.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    total_revenue = customers_with_clusters["revenue"].sum()
    total_customers = len(customers_with_clusters)

    profile = (
        customers_with_clusters.groupby("cluster")
        .agg(
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            median_revenue=("revenue", "median"),
            avg_orders=("orders", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_customer_age_days=("customer_age_days", "mean"),
            avg_revenue_per_day=("revenue_per_day", "mean"),
        )
        .reset_index()
    )

    profile["customer_share"] = profile["customers"] / total_customers
    profile["revenue_share"] = profile["revenue"] / total_revenue if total_revenue else 0
    profile["cluster_name"] = profile.apply(_make_cluster_name, axis=1)

    return profile.sort_values("revenue_share", ascending=False).reset_index(drop=True)


def profile_cluster_features(
    customers_with_clusters: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Return feature means by cluster."""
    missing = [col for col in ["cluster", *features] if col not in customers_with_clusters.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return (
        customers_with_clusters.groupby("cluster")[features]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )


def add_cluster_names(customers: pd.DataFrame, cluster_profile: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive cluster names back to the customer table."""
    names = cluster_profile[["cluster", "cluster_name"]].drop_duplicates()
    return customers.merge(names, on="cluster", how="left")


def _prepare_feature_matrix(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Clean numeric features for clustering."""
    X = data[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True))

    # Log-transform heavy-tailed commercial metrics.
    for col in ["revenue", "avg_order_value", "revenue_per_day"]:
        if col in X.columns:
            X[col] = X[col].clip(lower=0)
            X[col] = X[col].map(math.log1p)

    return X


def _check_features(data: pd.DataFrame, features: list[str]) -> None:
    """Raise helpful errors before fitting."""
    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(f"Missing clustering feature columns: {missing}")

    if len(data) < 10:
        raise ValueError("At least 10 customers are needed for clustering.")


def _make_cluster_name(row: pd.Series) -> str:
    """Create a simple descriptive label for a cluster profile row."""
    revenue_share = row["revenue_share"]
    avg_orders = row["avg_orders"]
    avg_order_value = row["avg_order_value"]
    avg_revenue_per_day = row["avg_revenue_per_day"]

    if revenue_share >= 0.25 and avg_orders >= 4:
        return "High Revenue Loyalists"

    if avg_order_value >= 750 and avg_orders < 3:
        return "High AOV Occasional Buyers"

    if avg_orders >= 5 and avg_revenue_per_day < 20:
        return "Frequent Low Intensity Buyers"

    if avg_orders < 2:
        return "One-Time Low Engagement"

    if revenue_share >= 0.15:
        return "Core Repeat Buyers"

    return "Developing Customers"
