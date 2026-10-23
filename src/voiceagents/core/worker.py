"""Job queue with a single consumer thread: transcribe, then hand the job to its agent."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import numpy.typing as npt

from voiceagents.platform.winapi import WindowInfo

log = logging.getLogger(__name__)

Audio = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Job:
    mode: str  # which agent handles it, e.g. "dictation" or "command"
    audio: Audio
    context: WindowInfo  # foreground window captured at key-down
    text: str = ""  # the transcript, filled in by the worker before the agent runs
    created_at: float = field(default_factory=time.monotonic)


class Agent(Protocol):
    def handle(self, job: Job) -> None: ...


class Worker:
    """Runs jobs one at a time. An exception in a job is logged and reported, never fatal.

    `on_busy(True)` fires as each job starts and `on_busy(False)` when the queue drains.
    `on_error(job, exc)` fires for a failed job. `on_no_speech(job)` fires when the transcript is
    empty (the agent is not called).
    """

    def __init__(
        self,
        transcribe: Callable[[Audio], str],
        agents: dict[str, Agent],
        on_busy: Callable[[bool], None] = lambda busy: None,
        on_error: Callable[[Job, Exception], None] = lambda job, exc: None,
        on_no_speech: Callable[[Job], None] = lambda job: None,
    ):
        self._transcribe = transcribe
        self._agents = agents
        self._on_busy = on_busy
        self._on_error = on_error
        self._on_no_speech = on_no_speech
        self._jobs: queue.Queue[Job | None] = queue.Queue()
        self._pending = 0  # submitted and not yet finished
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        """Finish queued jobs, then stop. Gives up waiting after `timeout` seconds."""
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout)
            self._thread = None

    def submit(self, job: Job) -> None:
        with self._lock:
            self._pending += 1
        self._jobs.put(job)

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            self._safely(self._on_busy, True)
            self._process(job)
            with self._lock:
                self._pending -= 1
                drained = self._pending == 0
            if drained:
                self._safely(self._on_busy, False)

    def _process(self, job: Job) -> None:
        try:
            t0 = time.perf_counter()
            text = self._transcribe(job.audio).strip()
            t1 = time.perf_counter()
            if not text:
                log.info("%s: no speech recognised", job.mode)
                self._safely(self._on_no_speech, job)
                return
            log.info("%s transcript: %s", job.mode, text)
            agent = self._agents.get(job.mode)
            if agent is None:
                raise KeyError(f"no agent registered for mode {job.mode!r}")
            agent.handle(replace(job, text=text))
            log.debug(
                "%s: asr %.3f s, agent %.3f s, release-to-done %.3f s",
                job.mode,
                t1 - t0,
                time.perf_counter() - t1,
                time.monotonic() - job.created_at,
            )
        except Exception as exc:
            log.exception("%s job failed", job.mode)
            self._safely(self._on_error, job, exc)

    @staticmethod
    def _safely(callback: Callable[..., None], *args: object) -> None:
        try:
            callback(*args)
        except Exception:
            log.exception("worker callback failed")
