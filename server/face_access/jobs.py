"""One writer job at a time, with a bounded queue and bounded status history."""

from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

from .qt_bridge import BridgeError
from .store import StoreNotFound


class AccessJobs:
    def __init__(self, bridge):
        self.bridge = bridge
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qt-management")
        self._capacity = threading.BoundedSemaphore(4)
        self._lock = threading.Lock()
        self._jobs = {}

    def submit(self, operation, payload):
        if not self._capacity.acquire(blocking=False):
            raise BridgeError("管理操作队列已满，请稍后重试")
        job_id = str(uuid.uuid4())
        with self._lock:
            while len(self._jobs) >= 200:
                disposable = next((key for key, job in self._jobs.items()
                                   if job["status"] in ("completed", "failed")), None)
                if disposable is None:
                    self._capacity.release()
                    raise BridgeError("管理任务繁忙")
                del self._jobs[disposable]
            self._jobs[job_id] = dict(id=job_id, status="queued", result=None, error=None)
        self._executor.submit(self._run, job_id, operation, payload)
        return dict(job_id=job_id, status="queued")

    def _run(self, job_id, operation, payload):
        try:
            with self._lock:
                self._jobs[job_id]["status"] = "running"
            result = self.bridge.call(operation, payload, timeout_s=180)
            with self._lock:
                self._jobs[job_id].update(status="completed", result=result)
        except Exception as error:
            with self._lock:
                self._jobs[job_id].update(status="failed", error=dict(
                    code="board_operation_failed", message=str(error) if isinstance(error, BridgeError) else "板端操作失败"))
        finally:
            self._capacity.release()

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise StoreNotFound("任务不存在或后台已重启，请核对实际人脸库")
            return dict(self._jobs[job_id])

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
