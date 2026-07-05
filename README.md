# Analytics Decision Kit

Small open-source analytics starter kit for customer KPIs, revenue concentration, segmentation and short executive summaries.

The repo is built notebook-first. 

## What is in here

- synthetic ecommerce order data
- customer level KPI calculation
- customer revenue deciles
- pareto / top customer concentration
- simple customer segmentation
- short executive summary text
- tests, becuase code without tests is basically hope in a hoodie

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

## Data needed

The input dataframe should have:

| column | meaning |
|---|---|
| customer_id | customer id |
| order_id | order id |
| order_date | date of the order |
| revenue | revenue amount |
| category | optional |
| brand | optional |

## Public / safe by design

Use synthetic or public data only.


## Initial scope

The first version is deliberately small:

1. create demo data
2. calculate customer metrics
3. split customers into revenue deciles
4. calculate top customer revenue concentration
5. assign basic segments
6. write a short summary
