from pathlib import Path

import pytest

from voiceagents.core.audio import Audio, read_wav

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def speech() -> Audio:
    return read_wav(FIXTURES / "speech.wav")


@pytest.fixture(scope="session")
def silence() -> Audio:
    return read_wav(FIXTURES / "silence.wav")


@pytest.fixture(scope="session")
def noise() -> Audio:
    return read_wav(FIXTURES / "noise.wav")
