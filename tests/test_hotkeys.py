import threading

import pytest

from voiceagents.core.hotkeys import HotkeyManager, KeyHandler, normalize_key, parse_combo


class FakeBackend:
    def __init__(self) -> None:
        self.handler: KeyHandler | None = None

    def start(self, handler: KeyHandler) -> None:
        self.handler = handler

    def stop(self) -> None:
        self.handler = None

    def press(self, key: str) -> bool:
        assert self.handler is not None
        return self.handler(key, True)

    def release(self, key: str) -> bool:
        assert self.handler is not None
        return self.handler(key, False)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Rig:
    """A manager with two bindings; `drain()` waits for the dispatcher thread to catch up."""

    def __init__(self, debounce_ms: int = 50) -> None:
        self.backend = FakeBackend()
        self.clock = Clock()
        self.events: list[str] = []
        self.manager = HotkeyManager(self.backend, debounce_ms, self.clock)
        for name, combo in (("dictation", "f9"), ("command", "f10")):
            self.manager.register(
                name,
                combo,
                lambda n=name: self.events.append(f"{n} down"),
                lambda n=name: self.events.append(f"{n} up"),
            )
        self.manager.start()

    def drain(self) -> list[str]:
        done = threading.Event()
        self.manager._events.put(done.set)
        assert done.wait(2)
        return self.events


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.manager.stop()


def test_normalize_and_parse() -> None:
    assert normalize_key("Right Ctrl") == "ctrl"
    assert normalize_key("Win") == "windows"
    assert parse_combo("Ctrl + Alt+D") == {"ctrl", "alt", "d"}
    with pytest.raises(ValueError):
        parse_combo(" + ")


def test_hold_fires_down_then_up(rig: Rig) -> None:
    rig.backend.press("f9")
    rig.backend.release("f9")
    assert rig.drain() == ["dictation down", "dictation up"]


def test_key_repeat_fires_down_once(rig: Rig) -> None:
    for _ in range(5):
        rig.backend.press("f9")
    rig.backend.release("f9")
    assert rig.drain() == ["dictation down", "dictation up"]


def test_two_bindings_are_independent(rig: Rig) -> None:
    rig.backend.press("f9")
    rig.backend.release("f9")
    rig.clock.now += 1
    rig.backend.press("f10")
    rig.backend.release("f10")
    assert rig.drain() == ["dictation down", "dictation up", "command down", "command up"]


def test_overlapping_holds_both_report(rig: Rig) -> None:
    rig.backend.press("f9")
    rig.backend.press("f10")
    rig.backend.release("f9")
    rig.backend.release("f10")
    assert rig.drain() == ["dictation down", "command down", "dictation up", "command up"]


def test_debounce_drops_a_quick_second_press(rig: Rig) -> None:
    rig.backend.press("f9")
    rig.backend.release("f9")
    rig.clock.now += 0.01
    rig.backend.press("f9")
    rig.backend.release("f9")
    rig.clock.now += 0.1
    rig.backend.press("f9")
    rig.backend.release("f9")
    assert rig.drain() == ["dictation down", "dictation up"] * 2


def test_bound_key_is_swallowed_and_others_pass(rig: Rig) -> None:
    assert rig.backend.press("f9") is False  # down
    assert rig.backend.press("f9") is False  # auto-repeat
    assert rig.backend.release("f9") is False
    assert rig.backend.press("a") is True
    assert rig.backend.release("a") is True


def test_chord_needs_all_keys_in_any_order() -> None:
    backend = FakeBackend()
    events: list[str] = []
    manager = HotkeyManager(backend, 0, Clock())
    manager.register("d", "ctrl+alt+d", lambda: events.append("down"), lambda: events.append("up"))
    manager.start()
    try:
        assert backend.press("left ctrl") is True  # modifiers are never swallowed
        backend.press("d")
        backend.press("right alt")  # now the full chord is held
        backend.release("d")
        backend.release("alt")
        backend.release("ctrl")
        done = threading.Event()
        manager._events.put(done.set)
        assert done.wait(2)
        assert events == ["down", "up"]
    finally:
        manager.stop()


def test_failing_callback_does_not_stop_dispatch() -> None:
    backend = FakeBackend()
    events: list[str] = []
    manager = HotkeyManager(backend, 0, Clock())

    def boom() -> None:
        raise RuntimeError("deliberate")

    manager.register("a", "f9", boom, lambda: events.append("up"))
    manager.start()
    try:
        backend.press("f9")
        backend.release("f9")
        done = threading.Event()
        manager._events.put(done.set)
        assert done.wait(2)
        assert events == ["up"]
    finally:
        manager.stop()


def test_duplicate_registration_is_rejected() -> None:
    manager = HotkeyManager(FakeBackend())
    manager.register("a", "f9", lambda: None, lambda: None)
    with pytest.raises(ValueError):
        manager.register("a", "f10", lambda: None, lambda: None)
    with pytest.raises(ValueError):
        manager.register("b", "F9", lambda: None, lambda: None)
