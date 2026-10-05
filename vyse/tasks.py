"""Background task system: long operations run on worker threads so the conversation never blocks.

A task is any callable. It may accept a `cancel` argument (a threading.Event) and poll it to stop early. Finished tasks
call `on_event` so the CLI can tell the user; results stay queryable by id until pruned.
"""
from __future__ import annotations

import inspect
import itertools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

PENDING, RUNNING, DONE, FAILED, CANCELLED = "pending", "running", "done", "failed", "cancelled"


@dataclass
class Task:
    id: int
    name: str
    status: str = PENDING
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    result: Any = None
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None

    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return round((self.finished or time.time()) - self.started, 2)

    def brief(self) -> dict[str, Any]:
        d = {"id": self.id, "name": self.name, "status": self.status, "seconds": self.elapsed()}
        if self.error:
            d["error"] = self.error
        return d

    def summary(self) -> str:
        r = self.result
        if isinstance(r, dict):
            r = r.get("display") or r.get("error") or r
        return str(r)[:300] if r is not None else ""


class TaskManager:
    def __init__(self, workers: int = 4, on_event: Callable[[Task], None] | None = None, keep: int = 50) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vyse-task")
        self._tasks: dict[int, Task] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.on_event = on_event
        self.keep = keep

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Task:
        with self._lock:
            task = Task(next(self._ids), name)
            self._tasks[task.id] = task
            self._prune()
        takes_cancel = "cancel" in inspect.signature(fn).parameters if callable(fn) else False

        def run() -> None:
            if task.cancel.is_set():
                task.status, task.finished = CANCELLED, time.time()
                return
            task.status, task.started = RUNNING, time.time()
            try:
                task.result = fn(*args, cancel=task.cancel, **kwargs) if takes_cancel else fn(*args, **kwargs)
                task.status = CANCELLED if task.cancel.is_set() else DONE
            except Exception as e:
                task.status, task.error = FAILED, f"{type(e).__name__}: {e}"
            finally:
                task.finished = time.time()
                if self.on_event and task.status != CANCELLED:
                    try:
                        self.on_event(task)
                    except Exception:
                        pass

        task.future = self._pool.submit(run)
        return task

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(int(task_id))

    def list(self, active_only: bool = False) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        return [t for t in tasks if not active_only or t.status in (PENDING, RUNNING)]

    def cancel(self, task_id: int) -> str:
        """Returns the status after the request. Running tasks stop only if they poll their cancel event."""
        t = self.get(task_id)
        if t is None:
            return "unknown"
        if t.status in (DONE, FAILED, CANCELLED):
            return t.status
        t.cancel.set()
        if t.future and t.future.cancel():      # still queued: never starts
            t.status, t.finished = CANCELLED, time.time()
        return "cancelling" if t.status == RUNNING else t.status

    def wait(self, task_id: int, timeout: float = 30.0) -> Task | None:
        t = self.get(task_id)
        if t and t.future:
            try:
                t.future.result(timeout)
            except Exception:
                pass
        return t

    def _prune(self) -> None:
        done = [t for t in self._tasks.values() if t.status in (DONE, FAILED, CANCELLED)]
        for t in sorted(done, key=lambda x: x.finished)[:max(0, len(done) - self.keep)]:
            self._tasks.pop(t.id, None)

    def shutdown(self) -> None:
        for t in self.list(active_only=True):
            t.cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
