# 渠智罗盘：渠道经营风险决策支持原型

企业主链使用企业提供的AFFT脱敏模拟数据，完成数据审计、120天成熟标签、无泄漏特征、模型训练与校准、动态到期分层、五维健康诊断、SKU—仓库库存健康与4—8周需求基线、客户采购风险暴露、处置情景和流程事件链。系统不接管授信、停发或法务决策，飞书AI只负责解释结构化结果。

## 环境

```powershell
cd <解压后的channel-compass目录>
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
```

企业数据默认位于 `data/company/AFFT模拟数据集/`。路径可在 `config/company_pipeline.json` 修改，原始企业数据保持只读且不提交到代码仓库。

## 企业主链一键运行

```powershell
& ".\.venv\Scripts\python.exe" "src\run_company_pipeline.py"
```

一键入口依次完成：

1. 七表数据审计，并区分常规消费品分销与企业级项目业务；
2. 固定120天观察期和时间隔离带的训练数据构建；
3. 逻辑回归与LightGBM训练、Sigmoid概率校准、验证集选模；
4. 存量风险客户新增订单风险扩散监测、存量逾期事件和处置任务；
5. 带数据时效门禁的到期前5天及四档超期规则；当前提交物按2026-07-31快照做历史回放；
6. 营收质量、库存采购暴露、付款行为、信用暴露和合作稳定性五维诊断；
7. SKU—仓库库存风险与未来4/8周朴素需求基线；
8. 客户近180天采购高库存风险商品的暴露联动；
9. 透明What-if处置情景、完整任务卡和风险处置时间线。

## 关键输出

- 冻结模型：`models/primary_payment_risk.joblib`
- 模型报告：`docs/model_training_report.md`
- 模型冻结证据：`data/reports/model_freeze.json`
- 客户风险审计：`data/reports/customer_risk_aggregation_audit.json`
- 评委案例：`docs/judge_cases.md`
- 飞书企业表：`data/exports/feishu/企业*.csv`
- 反馈模板：`data/feedback/task_feedback_template.csv`
- 便携演示：`python src/run_portable_demo.py`。这是用于验证“造数—训练—评分—生成风险事件”的最小技术闭环，无需企业原始数据；它不复刻企业七表业务主链，正式业务结论以企业主链产物为准。

飞书建议导入十一张正式表，在原九表基础上增加企业动态回款监控和企业客户商品风险暴露。库存表没有客户字段，不得关联或归因到渠道客户；客户商品表表示采购暴露，不表示客户实际持有库存。

## 人工反馈回流

将飞书处置任务导出后，按反馈模板填写，再运行：

```powershell
& ".\.venv\Scripts\python.exe" "src\import_task_feedback.py" `
  "data\exports\feishu\企业处置任务.csv" `
  "data\feedback\task_feedback_template.csv" `
  "data\exports\feishu\企业处置任务_反馈合并.csv" `
  "data\reports\task_feedback_audit.json"
```

也可把反馈文件保存到 `data/feedback/task_feedback.csv`，企业主链会在重跑时按稳定任务编号自动合并。没有人工反馈时，项目不会虚构完成任务、处置收益或流程瓶颈。当前脚本也不宣称已经通过API连接飞书。

## 自动化验证

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

## 旧版演示

早期3客户硬编码演示仅用于飞书自动化回归，不再作为企业主展示：

```powershell
python src\run_pipeline.py
```

其输出被隔离到 `data/exports/feishu/legacy_demo/`。旧版健康度和客户库存规则不得与企业模型指标混用。

## 口径边界

- 数据为企业提供的脱敏模拟数据，金额和指标不能外推为真实经营水平。
- 主模型预测订单出库后120天内的超期风险，不是“未来30天坏账概率”。
- 当前5名预测风险客户均已有存量逾期，因此主展示定位为“新增订单风险扩散监测”，不宣称已实现客户首次逾期前预警。
- 界面优先展示风险分和风险分位；校准输出仅用于排序辅助和风险加权暴露，不解释为精确违约率。
- “风险加权应收暴露”是校准超期概率乘当前应收，不是预期坏账损失。
- 三演是参数化What-if，不证明干预具有因果效果。
- 最终授信、催收或暂停赊销均由业务人员审批。
- 120天模型负责未到期开放订单风险排序；动态规则依据最新应收快照的最终承诺还款日和超期天数确定处置阶段。
- 到期分层和履约修正规则是入围赛原型假设，后续需由企业校准。
- `historical_replay`模式以快照日为计算基准；切换到`business_current`时，数据超过配置的新鲜度阈值会停止实时分层，不继续生成临期结论。
- 未来4周和8周需求是最近90天日均销量外推基线，不宣称已完成客户—SKU机器学习需求预测。
- 五维健康度是透明诊断分，库存维度使用采购高库存风险商品的暴露，不能解释为客户库存。
