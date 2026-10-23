import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from voiceagents.app import VoiceApp, build_parser
from voiceagents.config import Config, FeedbackConfig
from voiceagents.core.audio import Audio
from voiceagents.core.feedback import Cue, Feedback, State
from voiceagents.core.hotkeys import HotkeyManager, KeyHandler
from voiceagents.core.worker import Job
from voiceagents.platform.winapi import WindowInfo

SPEECH = np.ones(16000, dtype=np.float32)
NOTEPAD = WindowInfo(hwnd=42, title="Untitled - Notepad")


def wait_for(condition: Callable[[], bool], timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.005)


class FakeBackend:
    handler: KeyHandler | None = None

    def start(self, handler: KeyHandler) -> None:
        self.handler = handler

    def stop(self) -> None:
        self.handler = None

    def tap(self, key: str, hold_s: float = 0) -> None:
        assert self.handler is not None
        self.handler(key, True)
        time.sleep(hold_s)
        self.handler(key, False)


class FakeTray:
    def __init__(self) -> None:
        self.states: list[State] = []
        self.toasts: list[str] = []
        self._quit = threading.Event()

    def set_state(self, state: State) -> None:
        self.states.append(state)

    def notify(self, message: str) -> None:
        self.toasts.append(message)

    def run(self, on_ready: Callable[[], None]) -> None:
        on_ready()
        self._quit.wait(5)

    def stop(self) -> None:
        self._quit.set()


class FakeBeeper:
    def __init__(self) -> None:
        self.cues: list[Cue] = []

    def play(self, cue: Cue) -> None:
        self.cues.append(cue)


class FakeRecorder:
    def __init__(self) -> None:
        self.audio: Audio = SPEECH
        self.fail_start = False
        self.opened = False
        self.closed = False
        self.recording = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def start(self) -> None:
        if self.fail_start:
            raise OSError("no microphone")
        self.recording = True

    def stop(self) -> Audio:
        self.recording = False
        return self.audio


class FakeASR:
    def __init__(self) -> None:
        self.ready = False
        self.fail_load = False
        self.release_load = threading.Event()
        self.release_load.set()
        self.transcribe_result: Callable[[], str] = lambda: "hello world"

    def load(self) -> None:
        self.release_load.wait(3)
        if self.fail_load:
            raise RuntimeError("model missing")
        self.ready = True

    def warm_up(self) -> None:
        pass

    def transcribe(self, audio: Audio) -> str:
        return self.transcribe_result()


class RecordingAgent:
    def __init__(self) -> None:
        self.jobs: list[Job] = []

    def handle(self, job: Job) -> None:
        self.jobs.append(job)


class Rig:
    def __init__(self) -> None:
        self.backend = FakeBackend()
        self.tray = FakeTray()
        self.beeper = FakeBeeper()
        self.recorder = FakeRecorder()
        self.asr = FakeASR()
        self.agent = RecordingAgent()
        self.feedback = Feedback(self.tray, self.beeper)
        config = Config(feedback=FeedbackConfig(error_flash_s=0.05))
        self.app = VoiceApp(
            config,
            self.recorder,
            self.asr,
            HotkeyManager(self.backend, debounce_ms=0),
            self.tray,
            self.feedback,
            {"dictation": self.agent, "command": self.agent},
            foreground=lambda: NOTEPAD,
        )

    def start(self, wait_ready: bool = True) -> None:
        self.app._startup()
        if wait_ready:
            wait_for(lambda: self.feedback.state is State.IDLE)

    def hold(self, key: str) -> None:
        assert self.backend.handler is not None
        self.backend.handler(key, True)

    def release(self, key: str) -> None:
        assert self.backend.handler is not None
        self.backend.handler(key, False)


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.app.shutdown()


def test_startup_loads_in_background_then_goes_idle(rig: Rig) -> None:
    rig.asr.release_load.clear()
    rig.start(wait_ready=False)
    wait_for(lambda: rig.feedback.state is State.LOADING)
    rig.asr.release_load.set()
    wait_for(lambda: rig.feedback.state is State.IDLE)
    assert rig.recorder.opened
    assert any("Ready" in t for t in rig.tray.toasts)


def test_press_before_the_model_is_ready_is_ignored_with_a_toast(rig: Rig) -> None:
    rig.asr.release_load.clear()
    rig.start(wait_ready=False)
    rig.backend.tap("f9")
    wait_for(lambda: any("loading" in t for t in rig.tray.toasts))
    assert not rig.recorder.recording
    assert rig.feedback.state is State.LOADING
    assert rig.beeper.cues == []


def test_hold_and_release_runs_the_job_and_walks_the_states(rig: Rig) -> None:
    rig.start()
    rig.hold("f9")
    wait_for(lambda: rig.feedback.state is State.RECORDING)
    assert rig.recorder.recording
    rig.release("f9")
    wait_for(lambda: len(rig.agent.jobs) == 1)
    wait_for(lambda: rig.feedback.state is State.IDLE)
    (job,) = rig.agent.jobs
    assert (job.mode, job.text, job.context) == ("dictation", "hello world", NOTEPAD)
    assert rig.tray.states[-4:] == [State.IDLE, State.RECORDING, State.PROCESSING, State.IDLE]
    assert rig.beeper.cues == [Cue.START, Cue.STOP]


def test_second_hotkey_selects_the_command_agent(rig: Rig) -> None:
    rig.start()
    rig.backend.tap("f10")
    wait_for(lambda: len(rig.agent.jobs) == 1)
    assert rig.agent.jobs[0].mode == "command"


def test_other_hotkey_is_ignored_while_recording(rig: Rig) -> None:
    rig.start()
    rig.hold("f9")
    wait_for(lambda: rig.feedback.state is State.RECORDING)
    rig.hold("f10")
    rig.release("f10")
    rig.release("f9")
    wait_for(lambda: len(rig.agent.jobs) == 1)
    assert rig.agent.jobs[0].mode == "dictation"
    assert rig.beeper.cues == [Cue.START, Cue.STOP]


def test_short_tap_enqueues_nothing(rig: Rig) -> None:
    rig.start()
    rig.recorder.audio = np.zeros(0, dtype=np.float32)
    rig.backend.tap("f9")
    wait_for(lambda: Cue.STOP in rig.beeper.cues)
    wait_for(lambda: rig.feedback.state is State.IDLE)
    assert rig.agent.jobs == []


def test_silence_toasts_and_skips_the_agent(rig: Rig) -> None:
    rig.start()
    rig.asr.transcribe_result = lambda: ""
    rig.backend.tap("f9")
    wait_for(lambda: "No speech detected" in rig.tray.toasts)
    assert rig.agent.jobs == []


def test_job_exception_shows_error_then_recovers_and_keeps_working(rig: Rig) -> None:
    rig.start()

    def explode() -> str:
        raise RuntimeError("deliberate")

    rig.asr.transcribe_result = explode
    rig.backend.tap("f9")
    wait_for(lambda: State.ERROR in rig.tray.states)
    assert any("deliberate" in t for t in rig.tray.toasts)
    assert Cue.ERROR in rig.beeper.cues
    wait_for(lambda: rig.feedback.state is State.IDLE)  # the flash expires

    rig.asr.transcribe_result = lambda: "still alive"
    rig.backend.tap("f9")
    wait_for(lambda: len(rig.agent.jobs) == 1)
    assert rig.agent.jobs[0].text == "still alive"


def test_microphone_failure_flashes_an_error_and_does_not_get_stuck(rig: Rig) -> None:
    rig.start()
    rig.recorder.fail_start = True
    rig.backend.tap("f9")
    wait_for(lambda: any("microphone" in t for t in rig.tray.toasts))
    wait_for(lambda: rig.feedback.state is State.IDLE)
    rig.recorder.fail_start = False
    rig.backend.tap("f9")
    wait_for(lambda: len(rig.agent.jobs) == 1)


def test_model_load_failure_is_reported_and_hotkeys_say_so(rig: Rig) -> None:
    rig.asr.fail_load = True
    rig.start(wait_ready=False)
    wait_for(lambda: rig.feedback.state is State.ERROR)
    rig.backend.tap("f9")
    wait_for(lambda: any("failed to load" in t for t in rig.tray.toasts))
    assert not rig.recorder.recording
    assert rig.feedback.state is State.ERROR  # does not expire


def test_run_blocks_until_quit_and_shuts_everything_down(rig: Rig) -> None:
    thread = threading.Thread(target=rig.app.run)
    thread.start()
    wait_for(lambda: rig.feedback.state is State.IDLE)
    rig.app.shutdown()
    thread.join(3)
    assert not thread.is_alive()
    assert rig.recorder.closed
    assert rig.backend.handler is None


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert str(args.config) == "config.toml"
    assert args.verbose is False
