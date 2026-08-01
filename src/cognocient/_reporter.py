"""
Async batching reporter — fires call metadata to Cognocient's ingestion
API without ever blocking or delaying the customer's real provider call,
and without ever raising back into their code.

Runs its own background daemon thread with a plain stdlib queue.Queue,
independent of whether the customer's application uses asyncio at all.
Most of the value of this wrapper is for the plain synchronous
openai.OpenAI() / anthropic.Anthropic() clients shown in the README,
which have no event loop to hang an asyncio task off of — a thread-based
worker is the one design that reports reliably regardless of whether the
calling application is sync or async.
"""

import atexit
import logging
import queue
import threading
from dataclasses import asdict, dataclass
from typing import Optional

import httpx

logger = logging.getLogger("cognocient")

DEFAULT_INGEST_URL = "https://api.cognocient.com/api/ingest/wrapper"
FLUSH_INTERVAL_SECONDS = 5.0
FLUSH_MAX_BATCH = 50
QUEUE_MAX_SIZE = 500


@dataclass
class CallReport:
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    status_code: int = 200
    cost_usd: Optional[float] = None
    tag_feature: Optional[str] = None
    tag_department: Optional[str] = None
    tag_user: Optional[str] = None
    tag_session: Optional[str] = None
    tag_tier: Optional[str] = None
    tag_project: Optional[str] = None
    tag_gl_account: Optional[str] = None
    tag_workload: Optional[str] = None
    tag_outcome: Optional[str] = None
    run_id: Optional[str] = None


class Reporter:
    """
    One instance per client. Every failure mode (network error, ingestion
    API unreachable, local queue full) is caught and logged locally at
    DEBUG level — never raised back into the caller. This is the actual
    reliability guarantee of the wrapper; it has to hold here, not just be
    claimed in the docs. See tests/test_reporter_failure_isolation.py.
    """

    def __init__(
        self,
        cognocient_key: str,
        ingest_url: str = DEFAULT_INGEST_URL,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        flush_max_batch: int = FLUSH_MAX_BATCH,
        queue_max_size: int = QUEUE_MAX_SIZE,
        http_timeout: float = 5.0,
    ):
        self._cognocient_key = cognocient_key
        self._ingest_url = ingest_url
        self._flush_interval = flush_interval
        self._flush_max_batch = flush_max_batch
        self._queue: "queue.Queue[CallReport]" = queue.Queue(maxsize=queue_max_size)
        self._client = httpx.Client(timeout=http_timeout)
        self._dropped_count = 0
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="cognocient-reporter", daemon=True)
        self._thread.start()
        atexit.register(self._shutdown)

    def report(self, call: CallReport) -> None:
        """Non-blocking, never raises. Call this right after the real provider call returns."""
        try:
            self._queue.put_nowait(call)
            if self._queue.qsize() >= self._flush_max_batch:
                self._wake.set()
        except queue.Full:
            self._dropped_count += 1
            logger.debug(
                "cognocient: reporting queue full, dropped a report (%d dropped total)",
                self._dropped_count,
            )
        except Exception:
            logger.debug("cognocient: failed to queue a report", exc_info=True)

    def flush(self) -> None:
        """Force an immediate flush — e.g. before a short-lived script exits."""
        try:
            self._flush()
        except Exception:
            logger.debug("cognocient: manual flush failed", exc_info=True)

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()
            self._flush()
        self._flush()  # final drain on shutdown

    def _flush(self) -> None:
        batch = []
        while len(batch) < self._flush_max_batch:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return
        try:
            self._client.post(
                self._ingest_url,
                json={"reports": [asdict(c) for c in batch]},
                headers={"Authorization": f"Bearer {self._cognocient_key}"},
            )
        except Exception:
            # Network error, timeout, Cognocient down — never propagate.
            # The batch is simply dropped; retrying would risk piling up
            # unbounded background work in the customer's process for
            # something that is not their application's job to guarantee.
            logger.debug("cognocient: failed to report %d call(s)", len(batch), exc_info=True)

    def _shutdown(self) -> None:
        try:
            self._stop = True
            self._wake.set()
            self._thread.join(timeout=2.0)
            self._client.close()
        except Exception:
            logger.debug("cognocient: error during reporter shutdown", exc_info=True)
