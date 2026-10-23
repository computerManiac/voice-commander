from voiceagents.core.feedback import Cue, Feedback, State, make_icon


class Tray:
    def __init__(self) -> None:
        self.states: list[State] = []
        self.toasts: list[str] = []

    def set_state(self, state: State) -> None:
        self.states.append(state)

    def notify(self, message: str) -> None:
        self.toasts.append(message)

    def run(self, on_ready) -> None:  # type: ignore[no-untyped-def]
        on_ready()

    def stop(self) -> None:
        pass


class Beeper:
    def __init__(self) -> None:
        self.cues: list[Cue] = []

    def play(self, cue: Cue) -> None:
        self.cues.append(cue)


def test_repeated_state_updates_the_tray_once() -> None:
    tray = Tray()
    feedback = Feedback(tray, Beeper())
    feedback.set_state(State.IDLE)
    feedback.set_state(State.IDLE)
    feedback.set_state(State.RECORDING)
    assert tray.states == [State.IDLE, State.RECORDING]
    assert feedback.state is State.RECORDING


def test_beeps_can_be_disabled() -> None:
    beeper = Beeper()
    Feedback(Tray(), beeper, beeps=False).cue(Cue.START)
    Feedback(Tray(), beeper, beeps=True).cue(Cue.STOP)
    assert beeper.cues == [Cue.STOP]


def test_toast_goes_to_the_tray() -> None:
    tray = Tray()
    Feedback(tray, None).toast("hi")
    assert tray.toasts == ["hi"]


def test_every_state_has_a_distinct_icon() -> None:
    images = {state: make_icon(state).tobytes() for state in State}  # type: ignore[attr-defined]
    assert len(set(images.values())) == len(State)
