# 旧版3客户演示数据字典

> 仅用于早期界面和飞书自动化测试，不代表企业主链的数据结构。企业版请以 `data_contract_v1.0.md` 为准。

所有表均含 `data_source=synthetic`，客户、产品、订单、库存、发票、回款通过编号关联。

## customer.csv

| 字段 | 含义 |
|---|---|
| customer_id | 客户唯一编号 |
| customer_name | 仿真客户名称 |
| region | 华东/华南 |
| channel_tier | 一级/二级/零售 |
| cooperation_months | 合作月数 |
| credit_limit | 信用额度 |
| credit_term_days | 账期天数 |
| data_source | 固定 synthetic |

## product.csv

`sku_id, sku_name, brand, category, launch_date, eol_date, unit_price, data_source`

## sales_order.csv

`order_id, order_date, customer_id, sku_id, quantity, revenue, gross_profit, return_flag, data_source`

## inventory_snapshot.csv

`snapshot_date, customer_id, sku_id, on_hand_qty, in_transit_qty, avg_inventory_age_days, data_source`

## invoice.csv

`invoice_id, order_id, customer_id, issue_date, due_date, invoice_amount, open_amount, dispute_flag, data_source`

## payment.csv

`payment_id, invoice_id, payment_date, payment_amount, data_source`

## intervention.csv

`intervention_id, risk_event_id, customer_id, action_type, owner, status, created_at, completed_at, result, data_source`

## 关联关系

- `customer.customer_id` 关联订单、库存、发票和干预记录。
- `product.sku_id` 关联订单与库存。
- `sales_order.order_id` 关联发票。
- `invoice.invoice_id` 关联回款。
- `intervention.risk_event_id` 关联飞书风险事件。
