from pathlib import Path

import pytest

from voiceagents.config import Config, ConfigError, load_config

ROOT = Path(__file__).parent.parent


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "nope.toml") == Config()
    assert load_config(None) == Config()


def test_committed_config_matches_the_defaults() -> None:
    assert load_config(ROOT / "config.toml") == Config(
        aliases=load_config(ROOT / "config.toml").aliases
    )


def test_partial_file_overrides_only_what_it_names(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, '[hotkeys]\ndictation = "f8"\n[command.local_llm]\ntimeout_s = 5\n')
    )
    assert config.hotkeys.dictation == "f8"
    assert config.hotkeys.command == "f10"
    assert config.command.local_llm.timeout_s == 5.0
    assert config.audio == Config().audio


def test_aliases_are_read(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, '[aliases.sites]\nyt = "youtube.com"\n'))
    assert config.aliases.sites == {"yt": "youtube.com"}


@pytest.mark.parametrize(
    "text",
    [
        "[audio]\nsample_rat = 16000\n",  # typo
        "[nonsense]\nx = 1\n",
        '[asr]\nthreads = "five"\n',
        "[audio]\nvad = 1\n",  # an int is not a bool
        "audio = 3\n",
        "[audio\n",  # invalid TOML
    ],
)
def test_bad_config_raises(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))
