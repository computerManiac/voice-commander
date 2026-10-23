from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from voiceagents.config import AudioConfig
from voiceagents.core.audio import Audio, Recorder, resample, resolve_device


class FakeStream:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeMic:
    """Stands in for sounddevice: remembers the callback so tests can push blocks into it."""

    def __init__(self, rate: int = 16000) -> None:
        self.rate = rate
        self.stream = FakeStream()
        self.device: Any = "unset"
        self.callback: Callable[..., None] | None = None

    def factory(self, device: Any, preferred: int, callback: Callable[..., None]) -> Any:
        self.device, self.callback = device, callback
        return self.stream, self.rate

    def push(self, samples: Audio, block: int = 160) -> None:
        assert self.callback is not None
        for i in range(0, samples.size, block):
            chunk = samples[i : i + block].reshape(-1, 1)
            self.callback(chunk, chunk.shape[0], None, None)


def tone(seconds: float, rate: int = 16000, freq: float = 440, amp: float = 0.5) -> Audio:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make(mic: FakeMic, **overrides: Any) -> Recorder:
    return Recorder(AudioConfig(**overrides), stream_factory=mic.factory)


def test_start_stop_returns_captured_audio() -> None:
    mic = FakeMic()
    rec = make(mic, preroll_ms=0)
    rec.start()
    audio = tone(1.0)
    mic.push(audio)
    out = rec.stop()
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, audio)


def test_preroll_is_prepended_and_bounded() -> None:
    mic = FakeMic()
    rec = make(mic, preroll_ms=200, min_duration_ms=0)
    rec.open()
    mic.push(np.full(16000, 0.1, dtype=np.float32))  # one second before the key press
    rec.start()
    mic.push(np.full(1600, 0.9, dtype=np.float32))
    out = rec.stop()
    pre = int((out < 0.5).sum())
    assert 3200 <= pre < 3200 + 160  # 200 ms, plus at most one block of overshoot
    assert int((out >= 0.5).sum()) == 1600


def test_short_tap_is_discarded() -> None:
    mic = FakeMic()
    rec = make(mic, min_duration_ms=300)
    rec.start()
    mic.push(tone(0.1))
    assert rec.stop().size == 0


def test_stop_without_start_is_empty() -> None:
    assert make(FakeMic()).stop().size == 0


def test_recording_is_truncated_at_max_duration() -> None:
    mic = FakeMic()
    rec = make(mic, max_duration_s=1.0, preroll_ms=0)
    rec.start()
    mic.push(tone(3.0))
    assert rec.stop().size == 16000


def test_native_rate_is_resampled_to_16k() -> None:
    mic = FakeMic(rate=48000)
    rec = make(mic, preroll_ms=0)
    rec.start()
    mic.push(tone(1.0, rate=48000), block=480)
    out = rec.stop()
    assert out.size == 16000
    np.testing.assert_allclose(out[200:-200], tone(1.0)[200:-200], atol=0.02)


def test_recorder_can_be_reused_and_closed() -> None:
    mic = FakeMic()
    rec = make(mic, preroll_ms=0)
    rec.start()
    mic.push(tone(0.5))
    assert rec.stop().size == 8000
    rec.start()
    mic.push(tone(0.4))
    assert rec.stop().size == 6400
    rec.close()
    assert mic.stream.closed


@pytest.mark.parametrize(("config", "expected"), [("", None), ("2", 2), ("USB Mic", "USB Mic")])
def test_resolve_device(config: str, expected: str | int | None) -> None:
    assert resolve_device(config) == expected
    mic = FakeMic()
    make(mic, device=config).open()
    assert mic.device == expected


@pytest.mark.parametrize("src", [8000, 44100, 48000])
def test_resample_preserves_tone(src: int) -> None:
    out = resample(tone(1.0, rate=src, freq=300), src, 16000)
    assert out.size == 16000
    np.testing.assert_allclose(out[300:-300], tone(1.0, freq=300)[300:-300], atol=0.02)


def test_resample_attenuates_content_above_new_nyquist() -> None:
    out = resample(tone(1.0, rate=48000, freq=12000), 48000, 16000)
    assert np.abs(out[300:-300]).max() < 0.05
