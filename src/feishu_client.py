from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from typing import Any

import requests
from dotenv import load_dotenv


API_ROOT = "https://open.feishu.cn/open-apis"
RETRYABLE_CODES = {1254290, 1254291, 1254607, 1255001, 1255002, 1255040}
TOKEN_ERROR_CODES = {99991661, 99991663, 99991664}


class FeishuAPIError(RuntimeError):
    """飞书OpenAPI返回失败。"""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        status_code: int | None = None,
        log_id: str | None = None,
    ) -> None:
        details = []
        if code is not None:
            details.append(f"code={code}")
        if status_code is not None:
            details.append(f"http={status_code}")
        if log_id:
            details.append(f"log_id={log_id}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")
        self.code = code
        self.status_code = status_code
        self.log_id = log_id


class FeishuClient:
    """仅封装渠智罗盘需要的多维表格服务端API。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        app_token: str,
        *,
        timeout_seconds: int = 30,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not app_id.strip() or not app_secret.strip() or not app_token.strip():
            raise ValueError("飞书APP_ID、APP_SECRET和APP_TOKEN均不能为空")
        self.app_id = app_id.strip()
        self._app_secret = app_secret.strip()
        self.app_token = app_token.strip()
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._sleep = sleep
        self._tenant_access_token = ""
        self._token_expires_at = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        timeout_seconds: int = 30,
        env_path: str | None = None,
    ) -> "FeishuClient":
        load_dotenv(dotenv_path=env_path, override=False)
        return cls(
            os.getenv("FEISHU_APP_ID", ""),
            os.getenv("FEISHU_APP_SECRET", ""),
            os.getenv("FEISHU_APP_TOKEN", ""),
            timeout_seconds=timeout_seconds,
        )

    @property
    def masked_app_token(self) -> str:
        if len(self.app_token) <= 6:
            return "***"
        return f"{self.app_token[:3]}***{self.app_token[-3:]}"

    def _get_tenant_access_token(self, *, force_refresh: bool = False) -> str:
        now = time.monotonic()
        if (
            not force_refresh
            and self._tenant_access_token
            and now < self._token_expires_at
        ):
            return self._tenant_access_token

        try:
            response = self._session.post(
                f"{API_ROOT}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self._app_secret},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise FeishuAPIError(f"获取tenant_access_token失败：{exc}") from exc
        payload = self._decode_response(response, "获取tenant_access_token")
        token = str(payload.get("tenant_access_token", "")).strip()
        if not token:
            raise FeishuAPIError("飞书未返回tenant_access_token")
        expires_in = int(payload.get("expire", 7200))
        self._tenant_access_token = token
        self._token_expires_at = now + max(60, expires_in - 300)
        return token

    @staticmethod
    def _decode_response(response: Any, action: str) -> dict[str, Any]:
        log_id = response.headers.get("X-Tt-Logid") if response.headers else None
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuAPIError(
                f"{action}返回非JSON响应",
                status_code=getattr(response, "status_code", None),
                log_id=log_id,
            ) from exc
        if not isinstance(payload, dict):
            raise FeishuAPIError(f"{action}返回格式错误", log_id=log_id)
        status_code = int(getattr(response, "status_code", 0))
        code = int(payload.get("code", 0))
        if status_code >= 400 or code != 0:
            raise FeishuAPIError(
                str(payload.get("msg") or f"{action}失败"),
                code=code,
                status_code=status_code,
                log_id=log_id,
            )
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        delays = (0, 1, 2, 4, 8)
        token_refreshed = False
        last_error: FeishuAPIError | None = None
        for attempt, delay in enumerate(delays):
            if delay:
                self._sleep(delay)
            token = self._get_tenant_access_token(force_refresh=token_refreshed)
            token_refreshed = False
            try:
                response = self._session.request(
                    method,
                    f"{API_ROOT}{path}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                    params=params,
                    json=json_body,
                    timeout=self.timeout_seconds,
                )
                return self._decode_response(response, path)
            except requests.RequestException as exc:
                last_error = FeishuAPIError(f"请求飞书失败：{exc}")
            except FeishuAPIError as exc:
                last_error = exc
                if exc.code in TOKEN_ERROR_CODES and not token_refreshed:
                    self._tenant_access_token = ""
                    self._token_expires_at = 0.0
                    token_refreshed = True
                    continue
                retryable = (
                    exc.code in RETRYABLE_CODES
                    or exc.status_code == 429
                    or (exc.status_code is not None and exc.status_code >= 500)
                )
                if not retryable:
                    raise
            if attempt == len(delays) - 1:
                break
        if last_error is None:
            raise FeishuAPIError("飞书请求失败")
        raise last_error

    def list_tables(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            payload = self._request(
                "GET", f"/bitable/v1/apps/{self.app_token}/tables", params=params
            )
            data = payload.get("data") or {}
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                return items
            page_token = str(data.get("page_token", ""))
            if not page_token:
                raise FeishuAPIError("列出数据表分页缺少page_token")

    def list_fields(self, table_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            payload = self._request(
                "GET",
                f"/bitable/v1/apps/{self.app_token}/tables/{table_id}/fields",
                params=params,
            )
            data = payload.get("data") or {}
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                return items
            page_token = str(data.get("page_token", ""))
            if not page_token:
                raise FeishuAPIError("列出字段分页缺少page_token")

    def search_records(
        self,
        table_id: str,
        *,
        field_names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page_token = ""
        body: dict[str, Any] = {}
        if field_names:
            body["field_names"] = list(field_names)
        while True:
            params: dict[str, Any] = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            payload = self._request(
                "POST",
                f"/bitable/v1/apps/{self.app_token}/tables/{table_id}/records/search",
                params=params,
                json_body=body,
            )
            data = payload.get("data") or {}
            records.extend(data.get("items") or [])
            if not data.get("has_more"):
                return records
            page_token = str(data.get("page_token", ""))
            if not page_token:
                raise FeishuAPIError("查询记录分页缺少page_token")

    def batch_create(
        self, table_id: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        payload = self._request(
            "POST",
            f"/bitable/v1/apps/{self.app_token}/tables/{table_id}/records/batch_create",
            json_body={"records": records},
        )
        created = (payload.get("data") or {}).get("records") or []
        if len(created) != len(records):
            raise FeishuAPIError(
                f"批量新增返回{len(created)}条，预期{len(records)}条"
            )
        return created

    def batch_update(
        self, table_id: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        payload = self._request(
            "POST",
            f"/bitable/v1/apps/{self.app_token}/tables/{table_id}/records/batch_update",
            json_body={"records": records},
        )
        updated = (payload.get("data") or {}).get("records") or []
        if len(updated) != len(records):
            raise FeishuAPIError(
                f"批量更新返回{len(updated)}条，预期{len(records)}条"
            )
        return updated
