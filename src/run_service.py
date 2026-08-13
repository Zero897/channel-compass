from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol


RunResult = dict[str, object]
ProgressCallback = Callable[[str], None]
Runner = Callable[[ProgressCallback], RunResult]


class RunBusyError(RuntimeError):
    """已有分析任务正在执行。"""


class RunNotFoundError(KeyError):
    """运行编号不存在。"""


class RunStore(Protocol):
    def create(self, fields: dict[str, object]) -> str: ...

    def update(self, record_id: str, fields: dict[str, object]) -> None: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_milliseconds() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@dataclass
class RunJob:
    run_id: str
    requested_by: str
    status: str
    stage: str
    created_at: str
    started_at: str = ""
    finished_at: str = ""
    created: int = 0
    updated: int = 0
    report_path: str = ""
    error: str = ""
    control_record_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RunManager:
    def __init__(self, runner: Runner, store: RunStore) -> None:
        self._runner = runner
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="channel-run")
        self._lock = threading.Lock()
        self._jobs: dict[str, RunJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._active_run_id = ""

    def submit(self, *, requested_by: str, control_record_id: str = "") -> RunJob:
        with self._lock:
            if self._active_run_id:
                raise RunBusyError(f"运行{self._active_run_id}尚未结束")
            run_id = f"RUN-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
            job = RunJob(
                run_id=run_id,
                requested_by=requested_by.strip() or "飞书用户",
                status="待运行",
                stage="等待后台线程",
                created_at=_now_iso(),
                control_record_id=control_record_id.strip(),
            )
            initial_fields: dict[str, object] = {
                "运行编号": run_id,
                "触发人": job.requested_by,
                "触发时间": _now_milliseconds(),
                "运行状态": job.status,
                "当前阶段": job.stage,
                "新增记录数": 0,
                "更新记录数": 0,
                "错误信息": "",
            }
            if job.control_record_id:
                self._store.update(job.control_record_id, initial_fields)
            else:
                job.control_record_id = self._store.create(initial_fields)
            self._jobs[run_id] = job
            self._active_run_id = run_id
            self._futures[run_id] = self._executor.submit(self._execute, run_id)
            return job

    def _update_job(self, run_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[run_id]
            for name, value in changes.items():
                setattr(job, name, value)

    def _write_status(self, run_id: str, fields: dict[str, object]) -> None:
        record_id = self._jobs[run_id].control_record_id
        self._store.update(record_id, fields)

    def _execute(self, run_id: str) -> None:
        try:
            started_at = _now_iso()
            self._update_job(run_id, status="运行中", stage="初始化", started_at=started_at)
            self._write_status(
                run_id,
                {"运行状态": "运行中", "当前阶段": "初始化", "错误信息": ""},
            )

            def progress(stage: str) -> None:
                clean_stage = stage.strip()[:200] or "运行中"
                self._update_job(run_id, stage=clean_stage)
                self._write_status(run_id, {"当前阶段": clean_stage})

            result = self._runner(progress)
            created = int(result.get("created", 0))
            updated = int(result.get("updated", 0))
            report_path = str(result.get("report_path", ""))
            finished_at = _now_iso()
            self._update_job(
                run_id,
                status="成功",
                stage="完成",
                finished_at=finished_at,
                created=created,
                updated=updated,
                report_path=report_path,
            )
            self._write_status(
                run_id,
                {
                    "运行状态": "成功",
                    "当前阶段": "完成",
                    "新增记录数": created,
                    "更新记录数": updated,
                    "同步报告": report_path,
                    "完成时间": _now_milliseconds(),
                    "错误信息": "",
                },
            )
        except Exception as exc:
            message = str(exc).strip()[:1000] or type(exc).__name__
            self._update_job(
                run_id,
                status="失败",
                stage="失败",
                finished_at=_now_iso(),
                error=message,
            )
            try:
                self._write_status(
                    run_id,
                    {
                        "运行状态": "失败",
                        "当前阶段": "失败",
                        "错误信息": message,
                        "完成时间": _now_milliseconds(),
                    },
                )
            except Exception as write_exc:
                self._update_job(
                    run_id,
                    error=f"{message}；运行控制回写失败：{write_exc}"[:1000],
                )
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = ""

    def get(self, run_id: str) -> RunJob:
        with self._lock:
            if run_id not in self._jobs:
                raise RunNotFoundError(run_id)
            return RunJob(**self._jobs[run_id].to_dict())

    def wait(self, run_id: str, *, timeout: float | None = None) -> RunJob:
        with self._lock:
            future = self._futures.get(run_id)
        if future is None:
            raise RunNotFoundError(run_id)
        future.result(timeout=timeout)
        return self.get(run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
