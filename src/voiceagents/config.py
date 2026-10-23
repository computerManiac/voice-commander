"""Typed settings for the audio and ASR layers (defaults mirror the schema in CLAUDE.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    device: str = ""  # empty = default input device; otherwise a name substring or index
    sample_rate: int = 16000
    min_duration_ms: int = 300  # ignore shorter taps
    max_duration_s: float = 45
    preroll_ms: int = 200
    vad: bool = True  # False = cheap energy gate instead of Silero


@dataclass(frozen=True)
class AsrConfig:
    model: str = "nemo-parakeet-tdt-0.6b-v2"
    quantization: str = "int8"
    threads: int = 5
