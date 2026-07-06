"""Load transactional data for the examples.

The default is synthetic data. If a GitHub/raw CSV URL is passed, the loader
tries to normalize a public transactional dataset to the package schema:

- order_id
- customer_id
- order_date
- revenue
- category
- brand

The public-data path is intentionally generic. No employer data belongs here.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from analytics_decision_kit.sample_data import create_demo_orders


DEFAULT_SUPERSTORE_GITHUB_URL = (
    "https://gist.githubusercontent.com/nnbphuong/"
    "38db511db14542f3ba9ef16e69d3814c/raw/Superstore.csv"
)


def get_transactional_data(
    github_url: str | None = None,
    n_customers: int = 5000,
    n_orders: int = 18000,
    seed: int = 42,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Return transactional data in the package's standard schema.

    If github_url is None, synthetic data is generated.

    If github_url is provided, the function reads the CSV and normalizes known
    public schemas such as Sample Superstore or UCI Online Retail-style files.
    The name is github_url because that is the intended public use case, but a
    local CSV path also works for tests and experiments.
    """
    if github_url is None:
        return create_demo_orders(n_customers=n_customers, n_orders=n_orders, seed=seed)

    csv_location = _github_to_raw_url(github_url)
    raw = pd.read_csv(csv_location, low_memory=False)

    if limit_rows is not None:
        raw = raw.head(limit_rows).copy()

    return normalize_transactional_data(raw)


def normalize_transactional_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize known public transactional schemas to the package schema."""
    if raw.empty:
        raise ValueError("raw data is empty")

    columns = {_clean_col_name(col): col for col in raw.columns}

    if {"order_id", "customer_id", "order_date", "revenue"}.issubset(raw.columns):
        data = raw.copy()
        if "category" not in data.columns:
            data["category"] = "Unknown"
        if "brand" not in data.columns:
            data["brand"] = "Unknown"
        return _finalize_schema(data)

    if {"orderid", "orderdate", "customerid", "sales"}.issubset(columns):
        data = pd.DataFrame(
            {
                "order_id": raw[columns["orderid"]],
                "customer_id": raw[columns["customerid"]],
                "order_date": raw[columns["orderdate"]],
                "revenue": raw[columns["sales"]],
                "category": raw[columns["category"]] if "category" in columns else "Unknown",
            }
        )
        if "productname" in columns:
            data["brand"] = raw[columns["productname"]].map(_extract_brand_like_token)
        elif "productid" in columns:
            data["brand"] = raw[columns["productid"]].map(_stable_brand_bucket)
        else:
            data["brand"] = "Unknown"
        return _finalize_schema(data)

    if {"invoiceno", "invoicedate", "customerid", "quantity", "unitprice"}.issubset(columns):
        quantity = pd.to_numeric(raw[columns["quantity"]], errors="coerce")
        unit_price = pd.to_numeric(raw[columns["unitprice"]], errors="coerce")
        data = pd.DataFrame(
            {
                "order_id": raw[columns["invoiceno"]],
                "customer_id": raw[columns["customerid"]],
                "order_date": raw[columns["invoicedate"]],
                "revenue": quantity * unit_price,
                "category": "Retail Goods",
            }
        )
        if "description" in columns:
            data["brand"] = raw[columns["description"]].map(_extract_brand_like_token)
        else:
            data["brand"] = "Unknown"
        data = data.dropna(subset=["customer_id", "revenue"])
        data = data[data["revenue"] > 0].copy()
        return _finalize_schema(data)

    raise ValueError(
        "Unsupported dataset schema. Expected package columns, Sample Superstore columns, "
        "or UCI Online Retail-style columns."
    )


def _finalize_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Clean final output columns and types."""
    required = ["order_id", "customer_id", "order_date", "revenue", "category", "brand"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"normalized data is missing columns: {missing}")

    out = data[required].copy()
    out["order_id"] = out["order_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce")
    out["category"] = out["category"].fillna("Unknown").astype(str)
    out["brand"] = out["brand"].fillna("Unknown").astype(str)

    out = out.dropna(subset=["order_id", "customer_id", "order_date", "revenue"])
    out = out[out["revenue"] > 0].copy()
    out = out.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("normalized data is empty after cleaning")
    return out


def _github_to_raw_url(url_or_path: str) -> str:
    """Convert common GitHub blob URLs to raw URLs."""
    parsed = urlparse(url_or_path)
    if parsed.scheme == "" or Path(url_or_path).exists():
        return url_or_path
    if "raw.githubusercontent.com" in parsed.netloc or "gist.githubusercontent.com" in parsed.netloc:
        return url_or_path
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5:
            owner, repo, _, branch = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    return url_or_path


def _clean_col_name(col: object) -> str:
    return "".join(ch for ch in str(col).lower() if ch.isalnum())


def _extract_brand_like_token(value: object) -> str:
    """Extract a stable generic brand-like token from a product description."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    token = text.split()[0]
    token = "".join(ch for ch in token if ch.isalnum() or ch in {"-", "&"})
    return token[:40] if token else "Unknown"


def _stable_brand_bucket(value: object, buckets: int = 6) -> str:
    text = str(value)
    bucket_id = sum(ord(ch) for ch in text) % buckets
    return f"Brand {chr(65 + bucket_id)}"
