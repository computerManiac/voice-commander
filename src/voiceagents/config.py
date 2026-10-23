"""Typed settings (defaults mirror the schema in CLAUDE.md) and `config.toml` loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """The config file is unreadable or has an unknown key or a value of the wrong type."""


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


@dataclass(frozen=True)
class HotkeysConfig:
    dictation: str = "f9"  # hold to talk
    command: str = "f10"  # hold to talk
    debounce_ms: int = 50  # ignore a press this soon after the same key was released


@dataclass(frozen=True)
class FeedbackConfig:
    beeps: bool = True
    error_flash_s: float = 3.0  # how long the tray icon stays red after an error


@dataclass(frozen=True)
class DictationConfig:
    inject: str = "sendinput"  # "sendinput" | "paste"
    paste_threshold_chars: int = 400
    restore_clipboard: bool = True
    spoken_tokens: bool = True


@dataclass(frozen=True)
class LocalLlmConfig:
    server_url: str = "http://127.0.0.1:8080"
    timeout_s: float = 20


@dataclass(frozen=True)
class ApiConfig:
    model: str = "claude-haiku-5-5"  # the API key comes from ANTHROPIC_API_KEY, never the file


@dataclass(frozen=True)
class CommandConfig:
    backend: str = "local"  # "local" | "api" | "rules_only"
    confirm_destructive: bool = True
    cancel_window_ms: int = 1500
    log_actions: bool = True
    local_llm: LocalLlmConfig = field(default_factory=LocalLlmConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


@dataclass(frozen=True)
class AliasesConfig:
    sites: dict[str, str] = field(default_factory=dict)
    apps: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    dictation: DictationConfig = field(default_factory=DictationConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    aliases: AliasesConfig = field(default_factory=AliasesConfig)


def load_config(path: Path | None) -> Config:
    """Read `path` (missing file or None = all defaults). Raises ConfigError on bad input."""
    if path is None or not path.exists():
        return Config()
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    return _build(Config, data, "")


def _build[T](cls: type[T], data: dict[str, Any], where: str) -> T:
    """Construct dataclass `cls` from a TOML table, recursing into nested dataclasses."""
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{where or 'root'}]: {', '.join(unknown)}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = getattr(defaults, name)
        here = f"{where}.{name}" if where else name
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ConfigError(f"[{here}] must be a table")
            kwargs[name] = _build(type(default), value, here)
        else:
            kwargs[name] = _check(default, value, here)
    return cls(**kwargs)


def _check(default: Any, value: Any, where: str) -> Any:
    """Accept `value` if it has the same type as `default` (ints are fine for floats)."""
    if isinstance(default, bool) or isinstance(value, bool):
        ok = isinstance(default, bool) and isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, int | float)
    else:
        ok = isinstance(value, type(default))
    if not ok:
        raise ConfigError(f"{where}: expected {type(default).__name__}, got {value!r}")
    return float(value) if isinstance(default, float) else value
