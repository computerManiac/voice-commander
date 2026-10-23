"""Speech detection: find speech segments so silence never reaches the ASR model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]
Segment = tuple[int, int]  # [start, end) sample offsets

SAMPLE_RATE = 16000
SILERO_MIN_SAMPLES = 576  # one 512-sample hop plus 64 samples of context


@dataclass(frozen=True)
class SegmentParams:
    min_speech_ms: int = 250
    min_silence_ms: int = 500  # pauses shorter than this stay inside one segment
    max_speech_s: float = 30  # longer speech is split, Parakeet is slow on long audio
    pad_ms: int = 100


class SpeechDetector(Protocol):
    def load(self) -> None: ...
    def segments(self, audio: Audio) -> list[Segment]: ...


class SileroDetector:
    """Silero VAD through onnx-asr's onnxruntime wrapper (no torch)."""

    def __init__(self, params: SegmentParams | None = None):
        self._params = params or SegmentParams()
        self._vad: Any = None

    def load(self) -> None:
        import onnx_asr
        import onnxruntime as rt

        so = rt.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._vad = onnx_asr.load_vad("silero", sess_options=so)

    def segments(self, audio: Audio) -> list[Segment]:
        if self._vad is None:
            raise RuntimeError("SileroDetector.load() has not been called")
        if audio.size < SILERO_MIN_SAMPLES:
            return []
        p = self._params
        (found,) = self._vad.segment_batch(
            audio[None, :],
            np.array([audio.size], dtype=np.int64),
            SAMPLE_RATE,
            min_speech_duration_ms=p.min_speech_ms,
            min_silence_duration_ms=p.min_silence_ms,
            max_speech_duration_s=p.max_speech_s,
            speech_pad_ms=p.pad_ms,
        )
        return [(int(s), int(e)) for s, e in found]


class EnergyDetector:
    """Fixed RMS threshold. Cheap fallback for `audio.vad = false`; fooled by loud noise."""

    def __init__(
        self,
        params: SegmentParams | None = None,
        threshold: float = 0.01,
        frame_ms: int = 30,
    ):
        self._params = params or SegmentParams()
        self._threshold = threshold
        self._frame = SAMPLE_RATE * frame_ms // 1000

    def load(self) -> None:
        pass

    def segments(self, audio: Audio) -> list[Segment]:
        n = audio.size // self._frame
        if n == 0:
            return []
        frames = audio[: n * self._frame].reshape(n, self._frame)
        active = np.sqrt((frames**2).mean(axis=1)) > self._threshold
        edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(np.int8), [0]))))
        starts, ends = edges[0::2], edges[1::2]
        raw = [
            (int(s) * self._frame, int(e) * self._frame) for s, e in zip(starts, ends, strict=True)
        ]
        return merge_segments(raw, audio.size, self._params)


def merge_segments(raw: list[Segment], total: int, p: SegmentParams) -> list[Segment]:
    """Join close segments, drop blips, split overlong ones, pad and clamp to `total`."""
    ms = SAMPLE_RATE // 1000
    max_len = int(p.max_speech_s * SAMPLE_RATE)
    merged: list[Segment] = []
    for start, end in raw:
        joinable = merged and start - merged[-1][1] < p.min_silence_ms * ms
        if joinable and end - merged[-1][0] <= max_len:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    out: list[Segment] = []
    for start, end in merged:
        if end - start < p.min_speech_ms * ms:
            continue
        for chunk_start in range(start, end, max_len):
            chunk_end = min(chunk_start + max_len, end)
            out.append((max(chunk_start - p.pad_ms * ms, 0), min(chunk_end + p.pad_ms * ms, total)))
    return out


def make_detector(use_silero: bool) -> SpeechDetector:
    return cast(SpeechDetector, SileroDetector() if use_silero else EnergyDetector())
