from __future__ import annotations

import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

from feishu_client import FeishuClient
from project_paths import PROJECT_ROOT
from run_control_store import FeishuRunControlStore
from run_online_job import build_online_runner
from run_service import RunBusyError, RunManager, RunNotFoundError


class RunRequest(BaseModel):
    requested_by: str = Field(default="飞书用户", max_length=100)
    control_record_id: str = Field(default="", max_length=100)


def create_app(
    manager: RunManager,
    *,
    trigger_token: str,
    write_enabled: bool,
) -> FastAPI:
    if not trigger_token.strip():
        raise ValueError("BACKEND_TRIGGER_TOKEN不能为空")
    expected_authorization = f"Bearer {trigger_token.strip()}"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="渠智罗盘运行服务", version="1.0", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if authorization is None or not hmac.compare_digest(
            authorization, expected_authorization
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="触发凭证无效",
            )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "write_enabled": write_enabled}

    @app.post(
        "/api/runs",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_run(
        request_body: dict[str, object] | str = Body(default={}),
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authorize(authorization)
        try:
            request_data = json.loads(request_body) if isinstance(request_body, str) else request_body
            request = RunRequest.model_validate(request_data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="请求体必须是包含requested_by和control_record_id的JSON对象",
            ) from exc
        if not write_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="BACKEND_ALLOW_FEISHU_WRITE尚未开启",
            )
        try:
            job = manager.submit(
                requested_by=request.requested_by,
                control_record_id=request.control_record_id,
            )
        except RunBusyError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authorize(authorization)
        try:
            return manager.get(run_id).to_dict()
        except RunNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="运行编号不存在") from exc

    return app


def app_factory() -> FastAPI:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    trigger_token = os.getenv("BACKEND_TRIGGER_TOKEN", "")
    write_enabled = os.getenv("BACKEND_ALLOW_FEISHU_WRITE", "false").strip().lower() == "true"
    client = FeishuClient.from_env(env_path=str(PROJECT_ROOT / ".env"))
    store = FeishuRunControlStore(client)
    runner = build_online_runner(
        PROJECT_ROOT / "config" / "company_pipeline.json",
        PROJECT_ROOT / "config" / "feishu_sync.json",
        client,
    )
    return create_app(
        RunManager(runner, store),
        trigger_token=trigger_token,
        write_enabled=write_enabled,
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    host = os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app_factory(), host=host, port=port)


if __name__ == "__main__":
    main()
