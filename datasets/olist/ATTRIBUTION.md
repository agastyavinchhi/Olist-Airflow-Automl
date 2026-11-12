# Olist Brazilian E-Commerce — attribution

**Source:** Brazilian E-Commerce Public Dataset by Olist (Kaggle)
**Slug:** `olistbr/brazilian-ecommerce`
**URL:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
**License:** CC BY-NC-SA 4.0
**Coverage:** ~99k orders placed 2016-09 → 2018-10 on the Olist marketplace (real, anonymized)

## Getting the data

All raw CSVs (~120 MB) are **committed** in `datasets/olist/raw/`, so the pipeline runs right
after cloning — no Kaggle login needed. They are the unmodified files from the Kaggle ZIP:

```
datasets/olist/raw/
  olist_orders_dataset.csv            # fact: one row per order (order_id, customer_id, timestamps)
  olist_customers_dataset.csv         # dim: customer_id → city / state
  olist_order_items_dataset.csv       # 1..21 items per order (price, freight, product_id, seller_id)
  olist_order_payments_dataset.csv    # 1..n payments per order (type, installments, value)
  olist_order_reviews_dataset.csv     # review_score 1–5 per order
  olist_products_dataset.csv          # dim: product category, weight, dimensions
  product_category_name_translation.csv
  olist_sellers_dataset.csv           # (unused)
  olist_geolocation_dataset.csv       # (unused, 61 MB)
```

## Join story

All tables join to `orders` on `order_id` (items, payments, reviews) or `customer_id` (customers);
products join through items on `product_id`. Items and payments are aggregated to one row per order
before the merge, so the final table has **one row per delivered order**.

## Data quality notes (from profiling)

| Table | Issue | Handling |
| --- | --- | --- |
| orders | 2,965 missing delivery dates; 8 statuses | keep `delivered` orders with a delivery date |
| reviews | 551 orders with >1 review | keep the latest review per order |
| products | 610 missing category / text lengths | fill category `unknown`, impute numerics |
| payments | 3 `not_defined` payment types | drop those rows |
| items | up to 21 items per order | aggregate per order |

**Prediction target:** `is_bad_review` = `review_score <= 2` (≈12.8% of delivered orders).
For context, ≈8.1% of delivered orders arrive after the estimated delivery date.
