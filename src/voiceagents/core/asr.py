"""Parakeet TDT 0.6B (ONNX int8, CPU) behind a VAD gate."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from voiceagents.config import AsrConfig
from voiceagents.core.vad import SpeechDetector

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
Audio = npt.NDArray[np.float32]


class Recognizer(Protocol):
    def recognize(self, waveform: Audio, *, sample_rate: int) -> Any: ...


def load_onnx_model(config: AsrConfig) -> Recognizer:
    import onnx_asr
    import onnxruntime as rt

    def options() -> rt.SessionOptions:
        so = rt.SessionOptions()
        so.intra_op_num_threads = config.threads
        so.inter_op_num_threads = 1
        return so

    model = onnx_asr.load_model(
        config.model,
        quantization=config.quantization,
        sess_options=options(),
        preprocessor_config={"sess_options": options(), "max_concurrent_workers": 1},
    )
    return cast(Recognizer, model)


class ParakeetASR:
    """One shared instance: `load()` and `warm_up()` on a background thread, then `transcribe()`."""

    def __init__(
        self,
        config: AsrConfig,
        detector: SpeechDetector,
        loader: Callable[[AsrConfig], Recognizer] = load_onnx_model,
    ):
        self._config = config
        self._detector = detector
        self._loader = loader
        self._model: Recognizer | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        t0 = time.perf_counter()
        self._detector.load()
        self._model = self._loader(self._config)
        log.debug("ASR loaded in %.2f s", time.perf_counter() - t0)

    def warm_up(self) -> None:
        """Run one inference on noise (bypassing the gate) so the first real call is fast."""
        t0 = time.perf_counter()
        noise = np.random.default_rng(0).normal(0, 0.05, 2 * SAMPLE_RATE).astype(np.float32)
        self._recognize(noise)
        log.debug("ASR warm-up in %.2f s", time.perf_counter() - t0)

    def transcribe(self, audio: Audio) -> str:
        """Transcribe 16 kHz mono float32 audio. Returns "" without calling the model if silent.

        Speech is split at pauses (at most `SegmentParams.max_speech_s` per chunk), so recordings
        of any length are handled without feeding the model long audio.
        """
        if self._model is None:
            raise RuntimeError("ParakeetASR.load() has not been called")
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        segments = self._detector.segments(audio)
        t1 = time.perf_counter()
        if not segments:
            log.debug(
                "no speech in %.1f s of audio (vad %.3f s)", audio.size / SAMPLE_RATE, t1 - t0
            )
            return ""
        texts = [self._recognize(audio[start:end]) for start, end in segments]
        log.debug(
            "%d segment(s): vad %.3f s, asr %.3f s",
            len(segments),
            t1 - t0,
            time.perf_counter() - t1,
        )
        return " ".join(t for t in texts if t)

    def _recognize(self, audio: Audio) -> str:
        assert self._model is not None
        return str(self._model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()
