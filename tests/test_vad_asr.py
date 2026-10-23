import numpy as np
import pytest

from voiceagents.config import AsrConfig
from voiceagents.core.asr import ParakeetASR
from voiceagents.core.audio import Audio
from voiceagents.core.vad import (
    EnergyDetector,
    SegmentParams,
    SileroDetector,
    merge_segments,
)

RATE = 16000


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def recognize(self, waveform: Audio, *, sample_rate: int) -> str:
        assert sample_rate == RATE
        assert waveform.dtype == np.float32
        self.calls.append(waveform.size)
        return f" chunk{len(self.calls)} "


def fake_asr(detector: EnergyDetector | SileroDetector) -> tuple[ParakeetASR, FakeModel]:
    model = FakeModel()
    asr = ParakeetASR(AsrConfig(), detector, loader=lambda _: model)
    asr.load()
    return asr, model


def test_energy_detector_finds_speech_and_ignores_silence_and_noise(
    speech: Audio, silence: Audio, noise: Audio
) -> None:
    det = EnergyDetector()
    segments = det.segments(speech)
    assert segments
    assert all(0 <= s < e <= speech.size for s, e in segments)
    assert det.segments(silence) == []
    assert det.segments(noise) == []


def test_silence_and_noise_never_reach_the_model(silence: Audio, noise: Audio) -> None:
    asr, model = fake_asr(EnergyDetector())
    assert asr.transcribe(silence) == ""
    assert asr.transcribe(noise) == ""
    assert asr.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert model.calls == []


def test_speech_is_trimmed_and_transcribed(speech: Audio) -> None:
    asr, model = fake_asr(EnergyDetector())
    assert asr.transcribe(speech) == "chunk1"
    assert len(model.calls) == 1
    assert model.calls[0] <= speech.size


def test_long_recording_is_chunked_on_the_max_length() -> None:
    rng = np.random.default_rng(0)
    loud = rng.normal(0, 0.2, 70 * RATE).astype(np.float32)  # 70 s without a pause
    asr, model = fake_asr(EnergyDetector())
    text = asr.transcribe(loud)
    assert text == "chunk1 chunk2 chunk3"
    pad = 2 * SegmentParams().pad_ms * RATE // 1000
    assert all(n <= 30 * RATE + pad for n in model.calls)


def test_transcribe_requires_load() -> None:
    asr = ParakeetASR(AsrConfig(), EnergyDetector(), loader=lambda _: FakeModel())
    assert not asr.ready
    with pytest.raises(RuntimeError):
        asr.transcribe(np.zeros(RATE, dtype=np.float32))


def test_warm_up_calls_model_once() -> None:
    asr, model = fake_asr(EnergyDetector())
    asr.warm_up()
    assert len(model.calls) == 1


def test_merge_segments_joins_pauses_drops_blips_and_pads() -> None:
    p = SegmentParams(min_speech_ms=250, min_silence_ms=500, max_speech_s=30, pad_ms=100)
    raw = [(0, 8000), (12000, 20000), (60000, 60800), (100000, 120000)]
    total = 130000
    assert merge_segments(raw, total, p) == [(0, 21600), (98400, 121600)]
    assert merge_segments([(126000, 130000)], total, p) == [(124400, 130000)]  # clamped at the end


@pytest.mark.asr
def test_silero_detector(speech: Audio, silence: Audio, noise: Audio) -> None:
    det = SileroDetector()
    det.load()
    assert det.segments(speech)
    assert det.segments(silence) == []
    assert det.segments(noise) == []
    assert det.segments(speech[:100]) == []


@pytest.mark.asr
def test_parakeet_round_trip(speech: Audio, silence: Audio, noise: Audio) -> None:
    asr = ParakeetASR(AsrConfig(), SileroDetector())
    asr.load()
    asr.warm_up()
    text = asr.transcribe(speech).lower()
    assert "notepad" in text
    assert "hello world" in text
    assert asr.transcribe(silence) == ""
    assert asr.transcribe(noise) == ""
