import threading

import numpy as np

from voiceagents.core.worker import Job, Worker
from voiceagents.platform.winapi import WindowInfo

AUDIO = np.zeros(1600, dtype=np.float32)


class RecordingAgent:
    def __init__(self) -> None:
        self.jobs: list[Job] = []

    def handle(self, job: Job) -> None:
        self.jobs.append(job)


def job(mode: str = "dictation") -> Job:
    return Job(mode, AUDIO, WindowInfo(hwnd=7, title="Notepad"))


def finish(worker: Worker) -> None:
    worker.stop(timeout=5)


def test_transcribes_then_calls_the_matching_agent() -> None:
    dictation, command = RecordingAgent(), RecordingAgent()
    worker = Worker(lambda audio: " hello world ", {"dictation": dictation, "command": command})
    worker.start()
    worker.submit(job("command"))
    finish(worker)
    assert dictation.jobs == []
    assert [(j.mode, j.text, j.context.hwnd) for j in command.jobs] == [
        ("command", "hello world", 7)
    ]


def test_jobs_run_in_submission_order() -> None:
    agent = RecordingAgent()
    texts = iter(["one", "two", "three"])
    worker = Worker(lambda audio: next(texts), {"dictation": agent})
    worker.start()
    for _ in range(3):
        worker.submit(job())
    finish(worker)
    assert [j.text for j in agent.jobs] == ["one", "two", "three"]


def test_failing_job_is_reported_and_the_next_one_still_runs() -> None:
    agent = RecordingAgent()
    errors: list[Exception] = []
    calls = iter([RuntimeError("deliberate"), "fine"])

    def transcribe(audio: object) -> str:
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    worker = Worker(transcribe, {"dictation": agent}, on_error=lambda j, e: errors.append(e))
    worker.start()
    worker.submit(job())
    worker.submit(job())
    finish(worker)
    assert [str(e) for e in errors] == ["deliberate"]
    assert [j.text for j in agent.jobs] == ["fine"]


def test_agent_exception_is_isolated() -> None:
    errors: list[Exception] = []

    class Broken:
        def handle(self, job: Job) -> None:
            raise ValueError("agent broke")

    worker = Worker(lambda a: "x", {"dictation": Broken()}, on_error=lambda j, e: errors.append(e))
    worker.start()
    worker.submit(job())
    worker.submit(job())
    finish(worker)
    assert len(errors) == 2


def test_unknown_mode_is_an_error_not_a_crash() -> None:
    errors: list[Exception] = []
    worker = Worker(lambda a: "x", {}, on_error=lambda j, e: errors.append(e))
    worker.start()
    worker.submit(job("nope"))
    finish(worker)
    assert len(errors) == 1


def test_empty_transcript_skips_the_agent() -> None:
    agent = RecordingAgent()
    quiet: list[Job] = []
    worker = Worker(lambda a: "  ", {"dictation": agent}, on_no_speech=quiet.append)
    worker.start()
    worker.submit(job())
    finish(worker)
    assert agent.jobs == []
    assert len(quiet) == 1


def test_busy_callback_brackets_a_burst_of_jobs() -> None:
    states: list[bool] = []
    gate = threading.Event()
    worker = Worker(
        lambda a: (gate.wait(2), "x")[1], {"dictation": RecordingAgent()}, on_busy=states.append
    )
    worker.start()
    worker.submit(job())
    worker.submit(job())
    gate.set()
    finish(worker)
    assert states[0] is True
    assert states[-1] is False
    assert states.count(False) == 1


def test_a_failing_callback_does_not_kill_the_worker() -> None:
    agent = RecordingAgent()

    def bad_busy(busy: bool) -> None:
        raise RuntimeError("ui exploded")

    worker = Worker(lambda a: "x", {"dictation": agent}, on_busy=bad_busy)
    worker.start()
    worker.submit(job())
    worker.submit(job())
    finish(worker)
    assert len(agent.jobs) == 2
