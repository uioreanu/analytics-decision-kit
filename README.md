# Analytics Decision Kit

Open-source analytics starter kit for customer KPIs, revenue concentration, segmentation, visual profiling and executive summaries.

The repo is built notebook-first.

## What is in here

- synthetic ecommerce order data as well as open dataset access
- customer level KPI calculation
- customer revenue deciles
- pareto / top customer concentration
- simple customer segmentation
- short executive summary text
- Plotly visual examples
- DQ tests

## Quick start

```bash
git clone https://github.com/uioreanu/analytics-decision-kit.git
cd analytics-decision-kit

python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"

python examples/01_customer_concentration.py
pytest
```

## Python example

```python
from analytics_decision_kit.sample_data import create_demo_orders
from analytics_decision_kit.customer_analysis import run_customer_analysis

orders = create_demo_orders(n_customers=5000, n_orders=18000, seed=42)
results = run_customer_analysis(orders)

print(results["summary_text"])
print(results["decile_summary"])
```

## Optional public GitHub dataset

```python
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)

# No URL -> synthetic data
synthetic_orders = get_transactional_data()

# GitHub/raw CSV URL -> public dataset normalized to the same schema
public_orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)
```

The output schema is the same in both cases:

```text
order_id, customer_id, order_date, revenue, category, manufacturer
```

## Plotly visual examples

Plotly is optional and does not require Jupyter.

Install Plotly only when you want to generate dynamic HTML charts:

```bash
pip install plotly
```

Run the public dataset visual example:

```bash
python examples/05_plotly_visuals_public_dataset.py
```

This creates dynamic HTML outputs in:

```text
outputs/plotly/revenue_deciles_public_dataset.html
outputs/plotly/transactions_bubble_public_dataset.html
```

## Example Plotly outputs

- [Revenue deciles](docs/plotly/revenue_deciles_public_dataset.html)
- [Transaction bubble chart](docs/plotly/transactions_bubble_public_dataset.html)

### Revenue distribution by customer decile

```python
from analytics_decision_kit.customer_analysis import (
    calculate_customer_metrics,
    calculate_revenue_deciles,
)
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)

orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)

customer_metrics = calculate_customer_metrics(orders)
decile_df = calculate_revenue_deciles(customer_metrics)
```

The Plotly example visualizes:

- revenue share by customer decile
- cumulative revenue share
- customers per decile
- average revenue per customer

### Transaction bubble chart

```python
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)

orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)
```

The Plotly example samples 1,000 transactions and visualizes:

- x-axis: order date
- y-axis: transaction revenue
- bubble size: transaction revenue
- color: category
- hover details: order id, customer id, manufacturer, category and revenue

## Data needed

The input dataframe should have:

| column | meaning |
|---|---|
| customer_id | customer id |
| order_id | order id |
| order_date | date of the order |
| revenue | revenue amount |
| category | optional |
| manufacturer | optional |

## Public / safe by design

Use synthetic or public data only. Alternatively using a publicly available dataset like "Sample Superstore CSV".

## Initial scope

The first version is deliberately small:

1. create demo data
2. calculate customer metrics
3. split customers into revenue deciles
4. calculate top customer revenue concentration
5. assign basic segments
6. write a short summary
