"""Entry point: wires config, tray, recorder, ASR, hotkeys and worker; owns the lifecycle."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import threading
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Protocol

from voiceagents.agents.echo import EchoAgent
from voiceagents.config import Config, ConfigError, load_config
from voiceagents.core.asr import ParakeetASR
from voiceagents.core.audio import Audio, Recorder
from voiceagents.core.feedback import Cue, Feedback, PystrayTray, State, Tray, WinsoundBeeper
from voiceagents.core.hotkeys import HotkeyManager
from voiceagents.core.vad import make_detector
from voiceagents.core.worker import Agent, Job, Worker
from voiceagents.platform.winapi import WindowInfo, foreground_window

log = logging.getLogger(__name__)


MIC_ERROR = "Can't access the microphone. Check that it is connected and enabled."


class MicRecorder(Protocol):
    @property
    def healthy(self) -> bool: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> Audio: ...


class SpeechModel(Protocol):
    @property
    def ready(self) -> bool: ...
    def load(self) -> None: ...
    def warm_up(self) -> None: ...
    def transcribe(self, audio: Audio) -> str: ...


class VoiceApp:
    """Hotkey down starts recording, hotkey up enqueues a job; the tray and beeps follow along.

    Threads: the tray runs on the caller's thread, the hotkey dispatcher calls `_on_down` and
    `_on_up`, the worker thread runs ASR and agents, and one thread loads the model at startup.
    """

    def __init__(
        self,
        config: Config,
        recorder: MicRecorder,
        asr: SpeechModel,
        hotkeys: HotkeyManager,
        tray: Tray,
        feedback: Feedback,
        agents: dict[str, Agent],
        foreground: Callable[[], WindowInfo] = foreground_window,
    ):
        self._config = config
        self._recorder = recorder
        self._asr = asr
        self._hotkeys = hotkeys
        self._tray = tray
        self._feedback = feedback
        self._foreground = foreground
        self._worker = Worker(
            asr.transcribe,
            agents,
            on_busy=self._on_busy,
            on_error=self._on_job_error,
            on_no_speech=lambda job: feedback.toast("No speech detected"),
        )
        self._lock = threading.Lock()
        self._ready = False
        self._load_failed = False
        self._recording: tuple[str, WindowInfo] | None = None  # (mode, window at key-down)
        self._busy = False
        self._error = False
        self._error_timer: threading.Timer | None = None
        self._stopped = False
        bindings = {"dictation": config.hotkeys.dictation, "command": config.hotkeys.command}
        for mode, combo in bindings.items():
            hotkeys.register(mode, combo, partial(self._on_down, mode), partial(self._on_up, mode))

    def run(self) -> None:
        """Show the tray icon and block until quit."""
        try:
            self._tray.run(self._startup)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._error_timer is not None:
                self._error_timer.cancel()
        self._hotkeys.stop()
        self._worker.stop()
        self._recorder.close()
        self._tray.stop()

    def _startup(self) -> None:
        self._refresh()
        self._worker.start()
        self._hotkeys.start()  # before the model is ready, so early presses get a "loading" toast
        threading.Thread(target=self._load_model, name="asr-load", daemon=True).start()

    def _load_model(self) -> None:
        try:
            self._asr.load()
            self._asr.warm_up()
        except Exception:
            log.exception("failed to load the speech model")
            with self._lock:
                self._load_failed = True
            self._refresh()
            self._feedback.toast("Could not load the speech model. See the log.")
            return
        try:
            self._recorder.open()
        except Exception:
            log.warning("could not open the microphone yet; will retry on first use", exc_info=True)
        with self._lock:
            self._ready = True
        self._refresh()
        hk = self._config.hotkeys
        self._feedback.toast(f"Ready. Hold {hk.dictation} to dictate, {hk.command} for commands.")

    def _on_down(self, mode: str) -> None:
        with self._lock:
            if not self._ready:
                message = (
                    "The speech model failed to load. See the log."
                    if self._load_failed
                    else "Still loading the speech model."
                )
                ignored = True
            elif self._recording is not None:
                log.debug("%s key ignored: already recording %s", mode, self._recording[0])
                return
            else:
                ignored = False
        if ignored:
            self._feedback.toast(message)
            return
        window = self._foreground()
        try:
            self._recorder.start()
        except Exception:
            log.exception("could not start recording")
            self._flash_error(MIC_ERROR)
            return
        with self._lock:
            self._recording = (mode, window)
        self._feedback.cue(Cue.START)
        self._refresh()
        log.info("recording for %s in %r", mode, window.title)

    def _on_up(self, mode: str) -> None:
        with self._lock:
            if self._recording is None or self._recording[0] != mode:
                return
            _, window = self._recording
            self._recording = None
        self._feedback.cue(Cue.STOP)
        try:
            audio = self._recorder.stop()
        except Exception:
            log.exception("could not stop recording")
            self._flash_error("Recording failed.")
            return
        if audio.size == 0 and not self._recorder.healthy:
            log.error("%s: the microphone stopped delivering audio", mode)
            self._flash_error(MIC_ERROR)
            return
        if audio.size == 0:
            log.info("%s: recording too short, ignored", mode)
            self._refresh()
            return
        log.debug("%s: %.1f s of audio queued", mode, audio.size / self._config.audio.sample_rate)
        with self._lock:
            self._busy = True
        self._refresh()
        self._worker.submit(Job(mode, audio, window))

    def _on_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = busy
        self._refresh()

    def _on_job_error(self, job: Job, exc: Exception) -> None:
        self._flash_error(f"{job.mode} failed: {exc}")

    def _flash_error(self, message: str) -> None:
        """Show the error colour for `error_flash_s`, with a toast and a low beep."""
        ttl = self._config.feedback.error_flash_s
        with self._lock:
            self._error = True
            if self._error_timer is not None:
                self._error_timer.cancel()
            self._error_timer = threading.Timer(ttl, self._clear_error)
            self._error_timer.daemon = True
            if not self._stopped:
                self._error_timer.start()
        self._feedback.cue(Cue.ERROR)
        self._feedback.toast(message)
        self._refresh()

    def _clear_error(self) -> None:
        with self._lock:
            self._error = False
        self._refresh()

    def _refresh(self) -> None:
        """Pick the tray state from the current facts (recording beats everything)."""
        with self._lock:
            if self._recording is not None:
                state = State.RECORDING
            elif self._load_failed:
                state = State.ERROR
            elif not self._ready:
                state = State.LOADING
            elif self._error:
                state = State.ERROR
            elif self._busy:
                state = State.PROCESSING
            else:
                state = State.IDLE
            self._feedback.set_state(state)  # under the lock so updates cannot reorder


def build_app(config: Config) -> VoiceApp:
    def quit_app() -> None:
        app.shutdown()

    tray = PystrayTray(quit_app)
    feedback = Feedback(tray, WinsoundBeeper(), config.feedback.beeps)
    echo = EchoAgent(feedback.toast)  # placeholder until the real agents land in Phases 3 and 4
    app = VoiceApp(
        config,
        Recorder(config.audio),
        ParakeetASR(config.asr, make_detector(config.audio.vad)),
        HotkeyManager(debounce_ms=config.hotkeys.debounce_ms),
        tray,
        feedback,
        {"dictation": echo, "command": echo},
    )
    return app


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "voiceagents" / "logs"


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    handlers.append(
        logging.handlers.RotatingFileHandler(
            directory / "voiceagents.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    )
    if sys.stderr is not None:  # None under pythonw
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    logging.getLogger("numba").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceagents", description="Local voice agents.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"), help="config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        config = load_config(args.config)
        app = build_app(config)
    except (ConfigError, ValueError) as e:
        log.error("%s", e)
        return 2
    log.info("starting")
    app.run()
    return 0
