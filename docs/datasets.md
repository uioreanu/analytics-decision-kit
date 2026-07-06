# Public datasets

The package can run on synthetic data or on public transactional CSV files.

## Default behavior

```python
from analytics_decision_kit.data_loader import get_transactional_data

orders = get_transactional_data()
```

If no URL is passed, synthetic data is generated.

## Public GitHub CSV

```python
from analytics_decision_kit.data_loader import (
    DEFAULT_SUPERSTORE_GITHUB_URL,
    get_transactional_data,
)

orders = get_transactional_data(DEFAULT_SUPERSTORE_GITHUB_URL)
```

The loader currently supports:

- package-native schema: `order_id`, `customer_id`, `order_date`, `revenue`, `category`, `brand`
- Sample Superstore-style schema: `Order ID`, `Order Date`, `Customer ID`, `Sales`, `Category`, `Product Name`
- UCI Online Retail-style schema: `InvoiceNo`, `InvoiceDate`, `CustomerID`, `Quantity`, `UnitPrice`, `Description`

The normalized output always uses:

```text
order_id
customer_id
order_date
revenue
category
brand
```

## Important

The public data loader is for demos and learning. Keep employer data, internal logic, and private datasets out of this repo.
