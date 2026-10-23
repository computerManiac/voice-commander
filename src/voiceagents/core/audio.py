"""Microphone capture: an always-open input stream with a pre-roll buffer, plus resampling."""

from __future__ import annotations

import logging
import threading
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from voiceagents.config import AudioConfig

log = logging.getLogger(__name__)

Audio = npt.NDArray[np.float32]


class InputStream(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


# (device, preferred sample rate, callback) -> (stream, actual sample rate)
StreamFactory = Callable[[str | int | None, int, Callable[..., None]], tuple[InputStream, int]]


def open_input_stream(
    device: str | int | None, sample_rate: int, callback: Callable[..., None]
) -> tuple[InputStream, int]:
    """Open a mono float32 stream, falling back to the device's native rate if needed."""
    import sounddevice as sd

    def make(rate: int) -> Any:
        return sd.InputStream(
            device=device, channels=1, samplerate=rate, dtype="float32", callback=callback
        )

    try:
        return make(sample_rate), sample_rate
    except sd.PortAudioError:
        native = int(sd.query_devices(device, "input")["default_samplerate"])
        log.info("device rejected %d Hz, capturing at %d Hz and resampling", sample_rate, native)
        return make(native), native


def resample(audio: Audio, src_rate: int, dst_rate: int) -> Audio:
    """Low-pass (when downsampling) and linearly interpolate to `dst_rate`."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if dst_rate < src_rate:
        cutoff = 0.45 * dst_rate / src_rate  # cycles per source sample
        taps = int(32 * src_rate / dst_rate) | 1
        n = np.arange(taps) - (taps - 1) / 2
        kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        audio = np.convolve(audio, kernel / kernel.sum(), mode="same")
    positions = np.arange(int(audio.size * dst_rate / src_rate)) * (src_rate / dst_rate)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def read_wav(path: Path, sample_rate: int = 16000) -> Audio:
    """Read a 16-bit PCM WAV, downmixed to mono and resampled to `sample_rate`."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        rate, channels = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    mono = data.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return resample(mono, rate, sample_rate)


def resolve_device(device: str) -> str | int | None:
    """Map the config string to what sounddevice expects ("" = default, digits = index)."""
    if not device:
        return None
    return int(device) if device.isdigit() else device


class Recorder:
    """Keeps the mic open, remembers the last `preroll_ms`, and captures between start/stop.

    The stream stays open so speech just before the key press is not lost. The callback only
    copies the block into a list; resampling and trimming happen in `stop()`.
    """

    def __init__(self, config: AudioConfig, stream_factory: StreamFactory = open_input_stream):
        self._config = config
        self._stream_factory = stream_factory
        self._stream: InputStream | None = None
        self._rate = config.sample_rate
        self._lock = threading.Lock()
        self._preroll: deque[Audio] = deque()
        self._preroll_samples = 0
        self._chunks: list[Audio] = []
        self._recorded = 0  # samples captured since start(), excluding pre-roll
        self._recording = False
        self._truncated = False

    def open(self) -> None:
        if self._stream is not None:
            return
        device = resolve_device(self._config.device)
        stream, self._rate = self._stream_factory(device, self._config.sample_rate, self._callback)
        stream.start()
        self._stream = stream

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            self._recording = False
            self._preroll.clear()
            self._preroll_samples = 0

    def start(self) -> None:
        """Begin capturing, seeded with the pre-roll. Opens the stream if needed."""
        self.open()
        with self._lock:
            self._chunks = list(self._preroll)
            self._recorded = 0
            self._truncated = False
            self._recording = True

    def stop(self) -> Audio:
        """Stop capturing; return mono float32 at the configured rate (empty for a short tap)."""
        with self._lock:
            if not self._recording:
                return np.zeros(0, dtype=np.float32)
            self._recording = False
            chunks, recorded, truncated = self._chunks, self._recorded, self._truncated
            self._chunks = []
        if truncated:
            log.warning("recording truncated at %.0f s", self._config.max_duration_s)
        if recorded < self._config.min_duration_ms * self._rate // 1000:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return resample(audio, self._rate, self._config.sample_rate)

    def _callback(
        self, indata: npt.NDArray[np.float32], frames: int, time: Any, status: Any
    ) -> None:
        if status:
            log.debug("audio status: %s", status)
        block = indata[:, 0].copy()
        with self._lock:
            if self._recording:
                room = int(self._config.max_duration_s * self._rate) - self._recorded
                if room < block.size:
                    self._truncated = True
                    block = block[: max(room, 0)]
                if block.size:
                    self._chunks.append(block)
                    self._recorded += block.size
                return
            self._preroll.append(block)
            self._preroll_samples += block.size
            limit = self._config.preroll_ms * self._rate // 1000
            while self._preroll_samples - self._preroll[0].size >= limit:
                self._preroll_samples -= self._preroll.popleft().size
