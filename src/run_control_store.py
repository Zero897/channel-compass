from __future__ import annotations

from typing import Any

from feishu_client import FeishuClient


RUN_CONTROL_FIELD_TYPES = {
    "运行编号": 1,
    "触发人": 1,
    "触发时间": 5,
    "运行状态": 3,
    "当前阶段": 1,
    "新增记录数": 2,
    "更新记录数": 2,
    "同步报告": 1,
    "错误信息": 1,
    "完成时间": 5,
}
RUN_CONTROL_FIELDS = set(RUN_CONTROL_FIELD_TYPES)


class FeishuRunControlStore:
    def __init__(self, client: FeishuClient, *, table_name: str = "运行控制") -> None:
        self._client = client
        self._table_name = table_name
        self._table_id = ""

    def _resolve_table(self) -> str:
        if self._table_id:
            return self._table_id
        matches = [
            item
            for item in self._client.list_tables()
            if str(item.get("name", "")).strip() == self._table_name
        ]
        if not matches:
            raise ValueError(f"飞书缺少运行控制表：{self._table_name}")
        if len(matches) > 1:
            raise ValueError(f"飞书存在重名运行控制表：{self._table_name}")
        table_id = str(matches[0].get("table_id", "")).strip()
        if not table_id:
            raise ValueError(f"飞书运行控制表缺少table_id：{self._table_name}")
        fields = {
            str(field.get("field_name", "")).strip(): int(field.get("type", 0))
            for field in self._client.list_fields(table_id)
        }
        missing = sorted(RUN_CONTROL_FIELDS - set(fields))
        if missing:
            raise ValueError(f"运行控制表缺少字段：{', '.join(missing)}")
        wrong_types = [
            f"{name}(实际{fields[name]}，需要{expected})"
            for name, expected in RUN_CONTROL_FIELD_TYPES.items()
            if fields[name] != expected
        ]
        if wrong_types:
            raise ValueError(f"运行控制表字段类型错误：{', '.join(wrong_types)}")
        self._table_id = table_id
        return table_id

    def create(self, fields: dict[str, object]) -> str:
        created = self._client.batch_create(
            self._resolve_table(), [{"fields": fields}]
        )
        record_id = str(created[0].get("record_id", "")).strip()
        if not record_id:
            raise ValueError("飞书创建运行控制记录后未返回record_id")
        return record_id

    def update(self, record_id: str, fields: dict[str, object]) -> None:
        if not record_id.strip():
            raise ValueError("运行控制record_id不能为空")
        self._client.batch_update(
            self._resolve_table(),
            [{"record_id": record_id.strip(), "fields": fields}],
        )
