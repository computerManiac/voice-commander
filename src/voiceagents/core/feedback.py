"""User feedback: start/stop beeps, tray icon state, and toast notifications."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from enum import Enum
from typing import Protocol

log = logging.getLogger(__name__)


class State(Enum):
    LOADING = "Loading speech model"
    IDLE = "Ready"
    RECORDING = "Listening"
    PROCESSING = "Working"
    ERROR = "Error"


class Cue(Enum):
    START = (880, 60)  # (frequency Hz, duration ms)
    STOP = (587, 60)
    ERROR = (220, 250)


_ICON_COLOURS = {
    State.LOADING: (140, 140, 140),
    State.IDLE: (46, 160, 67),
    State.RECORDING: (220, 38, 38),
    State.PROCESSING: (234, 160, 12),
    State.ERROR: (127, 29, 29),
}


class Tray(Protocol):
    def set_state(self, state: State) -> None: ...
    def notify(self, message: str) -> None: ...
    def run(self, on_ready: Callable[[], None]) -> None: ...  # blocks until `stop()`
    def stop(self) -> None: ...


class Beeper(Protocol):
    def play(self, cue: Cue) -> None: ...  # must not block the caller


class WinsoundBeeper:
    """`winsound.Beep` blocks for its whole duration, so play cues from a small thread."""

    def __init__(self) -> None:
        self._cues: queue.Queue[Cue] = queue.Queue()
        threading.Thread(target=self._run, name="beeper", daemon=True).start()

    def play(self, cue: Cue) -> None:
        self._cues.put(cue)

    def _run(self) -> None:
        import winsound

        while True:
            freq, ms = self._cues.get().value
            try:
                winsound.Beep(freq, ms)
            except RuntimeError:
                log.debug("beep failed", exc_info=True)


def make_icon(state: State) -> object:
    """A 64x64 PIL image: a coloured disc with a white ring."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 60, 60), fill=(*_ICON_COLOURS[state], 255))
    draw.ellipse((22, 22, 42, 42), outline=(255, 255, 255, 255), width=5)
    return image


class PystrayTray:
    """Notification-area icon. `run()` blocks on the calling thread; call it from the main one."""

    TITLE = "Voice Agents"

    def __init__(self, on_quit: Callable[[], None]):
        import pystray

        self._state = State.LOADING
        menu = pystray.Menu(pystray.MenuItem("Quit", lambda icon, item: on_quit()))
        self._icon = pystray.Icon("voiceagents", make_icon(self._state), self._title(), menu)

    def set_state(self, state: State) -> None:
        self._state = state
        self._icon.icon = make_icon(state)
        self._icon.title = self._title()

    def notify(self, message: str) -> None:
        try:
            self._icon.notify(message, self.TITLE)
        except Exception:
            log.debug("toast failed", exc_info=True)

    def run(self, on_ready: Callable[[], None]) -> None:
        def setup(icon: object) -> None:
            self._icon.visible = True
            on_ready()

        self._icon.run(setup)

    def stop(self) -> None:
        self._icon.stop()

    def _title(self) -> str:
        return f"{self.TITLE}: {self._state.value}"


class Feedback:
    """What the app calls: `set_state`, `toast` and `cue`. Safe to call from any thread."""

    def __init__(self, tray: Tray, beeper: Beeper | None, beeps: bool = True):
        self._tray = tray
        self._beeper = beeper if beeps else None
        self._state: State | None = None

    @property
    def state(self) -> State | None:
        return self._state

    def set_state(self, state: State) -> None:
        if state is not self._state:
            self._state = state
            log.debug("state: %s", state.name)
            self._tray.set_state(state)

    def toast(self, message: str) -> None:
        log.info("toast: %s", message)
        self._tray.notify(message)

    def cue(self, cue: Cue) -> None:
        if self._beeper is not None:
            self._beeper.play(cue)
