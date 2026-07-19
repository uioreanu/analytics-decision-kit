"""Main customer analysis helpers.

This is kept in one file. Later it can be split again.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Small helper for column checks."""
    missing_cols = [x for x in columns if x not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns: {missing_cols}")


def calculate_customer_metrics(
    orders: pd.DataFrame,
    customer_col: str = "customer_id",
    order_col: str = "order_id",
    date_col: str = "order_date",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """Aggregate order-level data to customer-level KPIs."""

    _check_columns(orders, [customer_col, order_col, date_col, revenue_col])

    if len(orders) == 0:
        raise ValueError("orders is empty")

    df = orders.copy()

    # clean basic data types; probably not enough for production but ok for this first version
    df[date_col] = pd.to_datetime(df[date_col])
    df[revenue_col] = pd.to_numeric(df[revenue_col])

    # Main customer aggregation. This is the table most later calculations use.
    customer_df = (
        df.groupby(customer_col)
        .agg(
            first_order_date=(date_col, "min"),
            last_order_date=(date_col, "max"),
            orders=(order_col, "nunique"),
            revenue=(revenue_col, "sum"),
        )
        .reset_index()
    )

    customer_df["avg_order_value"] = customer_df["revenue"] / customer_df["orders"]

    # +1 avoids zero days for single-day customers; ugly but practical.
    customer_df["customer_age_days"] = (
        customer_df["last_order_date"] - customer_df["first_order_date"]
    ).dt.days + 1

    customer_df["revenue_per_day"] = (
        customer_df["revenue"] / customer_df["customer_age_days"].clip(lower=1)
    )

    customer_df["is_repeat_customer"] = customer_df["orders"] >= 2

    # nice for reading the result immediatly
    customer_df = customer_df.sort_values("revenue", ascending=False).reset_index(drop=True)

    return customer_df


def calculate_revenue_deciles(
    customer_df: pd.DataFrame,
    customer_col: str = "customer_id",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """Create decile table based on customer revenue.

    Decile 1 = top revenue customers.
    """

    _check_columns(customer_df, [customer_col, revenue_col])

    if len(customer_df) == 0:
        raise ValueError("customer_df is empty")

    df = customer_df[[customer_col, revenue_col]].copy()
    df[revenue_col] = pd.to_numeric(df[revenue_col])
    df = df.sort_values(revenue_col, ascending=False).reset_index(drop=True)

    n_customers = len(df)

    # rank by position, not qcut, because qcut gets annoying with duplicated values
    df["decile"] = np.floor(np.arange(n_customers) * 10 / n_customers).astype(int) + 1
    df["decile"] = df["decile"].clip(upper=10)

    total_revenue = df[revenue_col].sum()

    decile_df = (
        df.groupby("decile")
        .agg(
            customers=(customer_col, "nunique"),
            revenue=(revenue_col, "sum"),
            avg_revenue_per_customer=(revenue_col, "mean"),
        )
        .reset_index()
        .sort_values("decile")
    )

    # calculate shares. consider denominator 0
    if total_revenue == 0:
        decile_df["revenue_share"] = 0
        decile_df["cumulative_revenue"] = decile_df["revenue"].cumsum()
        decile_df["cumulative_revenue_share"] = 0
    else:
        decile_df["revenue_share"] = decile_df["revenue"] / total_revenue
        decile_df["cumulative_revenue"] = decile_df["revenue"].cumsum()
        decile_df["cumulative_revenue_share"] = decile_df["cumulative_revenue"] / total_revenue

    return decile_df


def calculate_pareto_summary(
    customer_df: pd.DataFrame,
    customer_col: str = "customer_id",
    revenue_col: str = "revenue",
    thresholds: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Summarise how much revenue top customer groups generate."""

    _check_columns(customer_df, [customer_col, revenue_col])

    df = customer_df[[customer_col, revenue_col]].copy()
    df[revenue_col] = pd.to_numeric(df[revenue_col])
    df = df.sort_values(revenue_col, ascending=False).reset_index(drop=True)

    total_customers = len(df)
    total_revenue = df[revenue_col].sum()

    rows = []

    for top_share in thresholds:
        if top_share <= 0 or top_share > 1:
            raise ValueError("thresholds should be between 0 and 1")

        n_top_customers = max(1, math.ceil(total_customers * top_share))
        revenue_top = df.head(n_top_customers)[revenue_col].sum()

        rows.append(
            {
                "top_customer_share": top_share,
                "customers": n_top_customers,
                "revenue": revenue_top,
                "revenue_share": revenue_top / total_revenue if total_revenue != 0 else 0,
            }
        )

    pareto_df = pd.DataFrame(rows)

    return pareto_df


def calculate_order_ladder_summary(
    customer_df: pd.DataFrame,
    orders_col: str = "orders",
    revenue_col: str = "revenue",
    bucket_limit: int = 10,
) -> pd.DataFrame:
    """Summarise customer profiles by number of orders.

    Buckets are 1, 2, 3, ... up to bucket_limit - 1, and then bucket_limit+.
    """

    _check_columns(customer_df, [orders_col, revenue_col])

    if len(customer_df) == 0:
        raise ValueError("customer_df is empty")

    df = customer_df[[orders_col, revenue_col]].copy()
    df[orders_col] = pd.to_numeric(df[orders_col])
    df[revenue_col] = pd.to_numeric(df[revenue_col])

    rows = []
    total_customers = len(df)
    total_revenue = df[revenue_col].sum()

    for n_orders in range(1, bucket_limit):
        bucket_df = df[df[orders_col] == n_orders]
        rows.append(
            {
                "order_bucket": str(n_orders),
                "customers": len(bucket_df),
                "customer_share": len(bucket_df) / total_customers,
                "revenue": bucket_df[revenue_col].sum(),
                "revenue_share": bucket_df[revenue_col].sum() / total_revenue
                if total_revenue != 0
                else 0,
            }
        )

    bucket_df = df[df[orders_col] >= bucket_limit]
    rows.append(
        {
            "order_bucket": f"{bucket_limit}+",
            "customers": len(bucket_df),
            "customer_share": len(bucket_df) / total_customers,
            "revenue": bucket_df[revenue_col].sum(),
            "revenue_share": bucket_df[revenue_col].sum() / total_revenue
            if total_revenue != 0
            else 0,
        }
    )

    ladder_df = pd.DataFrame(rows)
    ladder_df["order_bucket"] = pd.Categorical(
        ladder_df["order_bucket"],
        categories=[str(x) for x in range(1, bucket_limit)] + [f"{bucket_limit}+"],
        ordered=True,
    )

    return ladder_df.sort_values("order_bucket").reset_index(drop=True)




def assign_customer_segments(
    customer_df: pd.DataFrame,
    revenue_col: str = "revenue",
    orders_col: str = "orders",
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Assign simple generic customer segments.

    Important: this is not meant to be a real employer segment logic. Keep it generic.
    """

    _check_columns(customer_df, [revenue_col, orders_col])

    df = customer_df.copy()

    # use dataset percentile, not a fixed business threshold
    high_value_limit = df[revenue_col].quantile(0.90)

    df[segment_col] = "Low Activity"
    df.loc[df[orders_col] == 1, segment_col] = "One-Time Buyer"
    df.loc[(df[orders_col] >= 2) & (df[revenue_col] < high_value_limit), segment_col] = (
        "Mid Value Repeat"
    )
    df.loc[df[revenue_col] >= high_value_limit, segment_col] = "High Value"

    return df


def write_summary_text(
    customer_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
) -> str:
    """Create short exec summary text.

    Rule based first. LLM later, once the base numbers are less nonsense.
    """

    _check_columns(customer_df, ["revenue", "orders", "is_repeat_customer"])
    _check_columns(decile_df, ["decile", "revenue_share"])
    _check_columns(pareto_df, ["top_customer_share", "revenue_share"])

    total_customers = len(customer_df)
    total_revenue = customer_df["revenue"].sum()

    # get top 10% revenue share from pareto if available
    top10 = pareto_df.loc[pareto_df["top_customer_share"].round(2) == 0.10, "revenue_share"]

    if len(top10) > 0:
        top10_share = float(top10.iloc[0])
    else:
        top10_share = float(decile_df.loc[decile_df["decile"] == 1, "revenue_share"].iloc[0])

    repeat_df = customer_df[customer_df["is_repeat_customer"] == True]  # noqa: E712
    one_time_df = customer_df[customer_df["is_repeat_customer"] == False]  # noqa: E712

    repeat_revenue_share = (
        repeat_df["revenue"].sum() / total_revenue if total_revenue != 0 else 0
    )

    repeat_rpc = repeat_df["revenue"].mean() if len(repeat_df) > 0 else 0
    one_time_rpc = one_time_df["revenue"].mean() if len(one_time_df) > 0 else 0
    rpc_ratio = repeat_rpc / one_time_rpc if one_time_rpc != 0 else 0

    if top10_share >= 0.50:
        concentration_msg = "Revenue concentration is high."
    else:
        concentration_msg = "Revenue concentration is moderate."

    if rpc_ratio >= 2:
        repeat_msg = "Repeat customers materially outperform one-time buyers."
    else:
        repeat_msg = "Repeat customer uplift exists, but is not the main story yet."

    summary_lines = [
        f"Customer base analyzed: {total_customers:,} customers.",
        f"Top 10% of customers generate {top10_share * 100:.1f}% of revenue.",
        (
            f"Repeat customers generate {repeat_revenue_share * 100:.1f}% of revenue "
            f"and have {rpc_ratio:.1f}x higher revenue per customer than one-time buyers."
        ),
        concentration_msg,
        repeat_msg,
        "Recommended next step: inspect activation, retention and category penetration by segment.",
    ]

    return "\n".join(summary_lines)


def run_customer_analysis(orders: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    """Run the full analysis in one function.

    This is mostly for demo speed. In real work I would still inspect every
    intermediate dataframe.
    """

    customer_df = calculate_customer_metrics(orders)
    customer_df = assign_customer_segments(customer_df)

    decile_df = calculate_revenue_deciles(customer_df)
    pareto_df = calculate_pareto_summary(customer_df)
    order_ladder_df = calculate_order_ladder_summary(customer_df)
    summary_text = write_summary_text(customer_df, decile_df, pareto_df)

    return {
        "customer_metrics": customer_df,
        "decile_summary": decile_df,
        "pareto_summary": pareto_df,
        "order_ladder_summary": order_ladder_df,
        "summary_text": summary_text,
    }
