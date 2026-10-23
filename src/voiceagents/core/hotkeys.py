"""Hold-to-talk hotkeys: key-down/key-up callbacks for named bindings, over a swappable backend.

The backend's hook runs on the OS keyboard-hook thread, which Windows drops if it is slow. So the
hook only updates a small state machine and decides whether to swallow the key; the user's
`on_down` / `on_up` callbacks run in order on a separate dispatcher thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

# (key name, is_down) -> True to pass the event on to the focused app, False to swallow it
KeyHandler = Callable[[str, bool], bool]

_ALIASES = {
    "control": "ctrl",
    "win": "windows",
    "cmd": "windows",
    "super": "windows",
    "esc": "escape",
    "return": "enter",
    "option": "alt",
}


class KeyboardBackend(Protocol):
    def start(self, handler: KeyHandler) -> None: ...
    def stop(self) -> None: ...


class KeyboardLibBackend:
    """Global low-level hook via the `keyboard` package (no admin rights needed on Windows)."""

    def __init__(self) -> None:
        self._remove: Callable[[], None] | None = None

    def start(self, handler: KeyHandler) -> None:
        import keyboard

        def on_event(event: keyboard.KeyboardEvent) -> bool:
            if not event.name:
                return True
            return handler(event.name, event.event_type == keyboard.KEY_DOWN)

        self._remove = keyboard.hook(on_event, suppress=True)

    def stop(self) -> None:
        import keyboard

        if self._remove is not None:
            keyboard.unhook(self._remove)
            self._remove = None


def normalize_key(name: str) -> str:
    """Lower-case, drop left/right, and map synonyms so config and hook names compare equal."""
    key = name.strip().lower()
    for prefix in ("left ", "right "):
        key = key.removeprefix(prefix)
    return _ALIASES.get(key, key)


def parse_combo(combo: str) -> frozenset[str]:
    keys = frozenset(normalize_key(part) for part in combo.split("+") if part.strip())
    if not keys:
        raise ValueError(f"empty hotkey: {combo!r}")
    return keys


@dataclass
class _Binding:
    name: str
    keys: frozenset[str]
    on_down: Callable[[], None]
    on_up: Callable[[], None]
    active: bool = False
    released_at: float = float("-inf")


class HotkeyManager:
    """Register bindings, `start()`, and get `on_down` once per hold and `on_up` on release.

    A binding is active while every key of its combo is held (chords work, in any press order).
    Auto-repeat down events are ignored, and a press within `debounce_ms` of the same binding's
    release is dropped. While a binding is active its non-modifier keys are swallowed, so F10 does
    not also open the app's menu bar.
    """

    _MODIFIERS = frozenset({"ctrl", "alt", "shift", "windows"})

    def __init__(
        self,
        backend: KeyboardBackend | None = None,
        debounce_ms: int = 50,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._backend = backend or KeyboardLibBackend()
        self._debounce_s = debounce_ms / 1000
        self._clock = clock
        self._bindings: dict[str, _Binding] = {}
        self._held: set[str] = set()
        self._events: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._dispatcher: threading.Thread | None = None

    def register(
        self, name: str, combo: str, on_down: Callable[[], None], on_up: Callable[[], None]
    ) -> None:
        if name in self._bindings:
            raise ValueError(f"hotkey {name!r} is already registered")
        keys = parse_combo(combo)
        for other in self._bindings.values():
            if other.keys == keys:
                raise ValueError(f"{combo!r} is already bound to {other.name!r}")
        self._bindings[name] = _Binding(name, keys, on_down, on_up)

    def start(self) -> None:
        if self._dispatcher is not None:
            return
        self._dispatcher = threading.Thread(target=self._dispatch, name="hotkey-dispatch")
        self._dispatcher.daemon = True
        self._dispatcher.start()
        self._backend.start(self._on_key)

    def stop(self) -> None:
        if self._dispatcher is None:
            return
        self._backend.stop()
        self._events.put(None)
        self._dispatcher.join(timeout=2)
        self._dispatcher = None
        self._held.clear()
        for binding in self._bindings.values():
            binding.active = False

    def _on_key(self, raw_name: str, is_down: bool) -> bool:
        """On the hook thread: update state, queue callbacks, say whether to pass the key on."""
        try:
            return self._update(normalize_key(raw_name), is_down)
        except Exception:
            log.exception("hotkey handler failed")
            return True

    def _update(self, key: str, is_down: bool) -> bool:
        if is_down:
            self._held.add(key)
        else:
            self._held.discard(key)
        swallow = False
        for b in self._bindings.values():
            if b.active:
                if key in b.keys:
                    swallow = swallow or key not in self._MODIFIERS
                    if not is_down:
                        self._release(b)
            elif is_down and key in b.keys and b.keys <= self._held:
                if self._clock() - b.released_at < self._debounce_s:
                    log.debug("hotkey %s: press ignored (debounce)", b.name)
                    continue
                b.active = True
                swallow = swallow or key not in self._MODIFIERS
                self._events.put(b.on_down)
        return not swallow

    def _release(self, b: _Binding) -> None:
        b.active = False
        b.released_at = self._clock()
        self._events.put(b.on_up)

    def _dispatch(self) -> None:
        while (callback := self._events.get()) is not None:
            try:
                callback()
            except Exception:
                log.exception("hotkey callback failed")
