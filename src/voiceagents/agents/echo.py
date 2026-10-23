"""Placeholder agent for Phases 2 to 3: shows the transcript as a toast. Replaced by real agents."""

from __future__ import annotations

from collections.abc import Callable

from voiceagents.core.worker import Job


class EchoAgent:
    def __init__(self, toast: Callable[[str], None]):
        self._toast = toast

    def handle(self, job: Job) -> None:
        self._toast(f"{job.mode}: {job.text}")
