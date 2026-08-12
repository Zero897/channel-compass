from __future__ import annotations

from common import FEISHU_DIR, PROCESSED_DIR, read_csv, write_csv


TRIGGER_DATE = "2026-08-03"


def _pct(value: str) -> str:
    return f"{float(value):.1%}"


def _money(value: str) -> float:
    return round(float(value), 2)


def _feature_map() -> dict[str, dict[str, str]]:
    rows = read_csv(PROCESSED_DIR / "customer_features.csv")
    return {row["customer_id"]: row for row in rows}


def _make_event(
    event_id: str,
    customer: dict[str, str],
    risk_type: str,
    level: str,
    evidence: str,
    amount: float,
    confidence: str,
) -> dict[str, object]:
    return {
        "事件编号": event_id,
        "客户编号": customer["customer_id"],
        "客户名称": customer["customer_name"],
        "风险类型": risk_type,
        "风险等级": level,
        "触发时间": TRIGGER_DATE,
        "关键指标": evidence,
        "风险金额": amount,
        "置信说明": confidence,
        "客户历史摘要": f"合作{float(customer['cooperation_months']):.0f}个月，当前健康度{customer['health_score']}分",
        "规则解释": evidence,
        "AI风险摘要": "",
        "处理状态": "待处理",
        "data_source": "synthetic",
    }


def export() -> None:
    features = _feature_map()
    c1, c2, c3 = features["C001"], features["C002"], features["C003"]
    channel_rows = [
        {
            "客户编号": row["customer_id"],
            "客户名称": row["customer_name"],
            "区域": row["region"],
            "渠道层级": row["channel_tier"],
            "健康度": row["health_score"],
            "风险等级": row["overall_risk_level"],
            "库存风险": row["inventory_risk_level"],
            "回款风险": row["payment_risk_level"],
            "负责人": "成员B",
            "数据来源": "synthetic",
        }
        for row in features.values()
    ]
    events = [
        _make_event("R001", c1, "回款趋势关注", "黄色", c1["payment_evidence"], _money(c1["overdue_open_amount"]), "规则命中1项"),
        _make_event("R002", c1, "综合健康关注", "黄色", f"健康度{c1['health_score']}分；存在一次轻微迟付记录", _money(c1["open_amount"]), "需人工复核，不判定高违约"),
        _make_event("R003", c2, "库存积压", c2["inventory_risk_level"], c2["inventory_evidence"], _money(c2["latest_inventory_value"]), "库存规则命中3项"),
        _make_event("R004", c2, "销售下滑", "黄色", f"近4周销量下降{abs(float(c2['sales_growth_ratio'])):.1%}", 0, "趋势规则命中"),
        _make_event("R005", c2, "综合经营风险", c2["overall_risk_level"], f"健康度{c2['health_score']}分；{c2['inventory_evidence']}", _money(c2["open_amount"]), "库存证据主导"),
        _make_event("R006", c3, "库存积压", c3["inventory_risk_level"], c3["inventory_evidence"], _money(c3["latest_inventory_value"]), "库存规则命中3项"),
        _make_event("R007", c3, "回款逾期", c3["payment_risk_level"], c3["payment_evidence"], _money(c3["overdue_open_amount"]), "回款规则命中不少于3项"),
        _make_event("R008", c3, "信用暴露", "红色", f"信用额度使用率{_pct(c3['credit_utilization_ratio'])}", _money(c3["open_amount"]), "额度阈值命中"),
        _make_event("R009", c3, "销售下滑", "黄色", f"近4周销量下降{abs(float(c3['sales_growth_ratio'])):.1%}", 0, "单项趋势关注；综合风险另见R010"),
        _make_event("R010", c3, "库存与回款双风险", c3["overall_risk_level"], f"{c3['inventory_evidence']}；{c3['payment_evidence']}", _money(c3["open_amount"]), "库存和回款规则同时为红色"),
    ]
    tasks = [
        {
            "任务编号": "T001",
            "风险事件编号": "R010",
            "客户编号": "C003",
            "客户名称": "东启科技",
            "AI建议动作": "核查应收明细和近期回款计划，联系客户确认异常原因，并评估后续业务协同安排",
            "人工处置方案": "待选择",
            "负责人": "成员B",
            "截止日期": "2026-08-05",
            "审批状态": "待审批",
            "执行状态": "待处理",
            "执行结果": "",
            "预警有效性": "待确认",
            "data_source": "synthetic",
        }
    ]
    metrics = [
        {"模型名称": "规则基线", "版本": "v0.1", "时间区间": "演示样本", "指标名称": "预设案例识别通过数", "指标值": 3, "是否仿真数据": "是", "更新时间": TRIGGER_DATE},
        {"模型名称": "规则基线", "版本": "v0.1", "时间区间": "演示样本", "指标名称": "风险事件数量", "指标值": 10, "是否仿真数据": "是", "更新时间": TRIGGER_DATE},
        {"模型名称": "规则基线", "版本": "v0.1", "时间区间": "演示样本", "指标名称": "C003双风险识别", "指标值": 1, "是否仿真数据": "是", "更新时间": TRIGGER_DATE},
    ]
    write_csv(FEISHU_DIR / "渠道客户.csv", list(channel_rows[0]), channel_rows)
    write_csv(FEISHU_DIR / "风险事件.csv", list(events[0]), events)
    write_csv(FEISHU_DIR / "处置任务.csv", list(tasks[0]), tasks)
    write_csv(FEISHU_DIR / "模型指标.csv", list(metrics[0]), metrics)


if __name__ == "__main__":
    export()
