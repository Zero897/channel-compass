# 旧版企业脱敏数据契约 v0.1

> 历史资料，仅用于说明数据申请前的假设；已由经全量数据验证的 `data_contract_v1.0.md` 替代。

状态：待企业真实字段确认。以下定义的是逻辑字段，不假设企业原始列名。

## A. 非项目类渠道分销（主线）

### 销售流水

粒度：一行代表一笔销售明细或一笔销售单据。

| 逻辑字段 | 必填 | 说明 |
|---|---|---|
| sale_id | 是 | 销售记录或单据唯一标识 |
| sale_date | 是 | 销售/出库确认日期 |
| customer_id | 是 | 下游经销商或集成商标识 |
| sales_amount | 是 | 销售金额，需确认含税口径 |
| sku_id | 否 | 产品标识 |
| quantity | 否 | 销售数量 |
| region | 否 | 区域 |
| channel_tier | 否 | 渠道层级 |

### 回款流水

粒度：一行代表一笔到账或核销记录。

| 逻辑字段 | 必填 | 说明 |
|---|---|---|
| payment_id | 是 | 回款记录唯一标识 |
| payment_date | 是 | 到账或核销日期，需企业确认 |
| customer_id | 是 | 客户标识 |
| payment_amount | 是 | 回款金额 |
| related_document_id | 否 | 被核销的销售、发票或应收单据 |

### 应收快照

粒度：一个快照日期、一个客户一行；若企业按单据或账龄分层提供，接入层先聚合到客户—快照。

| 逻辑字段 | 必填 | 说明 |
|---|---|---|
| snapshot_date | 是 | 应收状态观察日期 |
| customer_id | 是 | 客户标识 |
| receivable_balance | 是 | 应收余额 |
| overdue_balance | 否 | 已逾期余额 |
| max_overdue_days | 否 | 最大逾期天数 |
| open_document_count | 否 | 未清单据数 |

### 主线关联规则

- 三张表至少能通过 `customer_id` 关联；若存在单据级核销关系，再使用单据标识增强。
- 分销主线不默认存在库存字段，因此不基于这三张表宣称库存积压。
- 原始数据不直接进入飞书，只导出客户级特征和风险事件。

## B. 增值项目类（扩展）

四个逻辑实体为签约、销售出库、回款和应收。建议的主键为 `project_id/contract_id`，客户标识作为辅助关联。

| 表 | 最小逻辑字段 |
|---|---|
| 签约 | contract_id, project_id, customer_id, contract_date, contract_amount |
| 销售出库 | outbound_id, project_id, outbound_date, outbound_amount |
| 回款 | payment_id, project_id, payment_date, payment_amount |
| 应收 | snapshot_date, project_id, receivable_balance, overdue_balance |

项目类与非项目类不直接纵向拼接，也不共用未经校准的阈值。

## 时间与标签原则

- 观察时点为 `T`，特征只能使用 `T` 及以前的数据。
- 当时尚未冻结标签；当前企业主链已改为出库后120天固定观察期，详见v1.0。
- 付款日、最终清账结果等未来信息不得用于历史时点特征。
- 训练、验证和测试必须按时间先后划分。
