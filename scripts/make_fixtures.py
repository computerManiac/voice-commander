"""Regenerate the WAV fixtures in tests/fixtures (16 kHz mono 16-bit).

speech.wav   Windows SAPI TTS saying "open notepad and type hello world"
silence.wav  1.5 s of digital silence
noise.wav    3 s of quiet broadband hiss (about -50 dBFS RMS)

uv run python scripts\\make_fixtures.py
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 16000
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SPEECH_TEXT = "Open notepad and type hello world."


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())


def synthesize_speech(path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        f"{RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $f);"
        f"$s.Speak('{SPEECH_TEXT}');"
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthesize_speech(OUT / "speech.wav")
    write_wav(OUT / "silence.wav", np.zeros(int(1.5 * RATE)))
    rng = np.random.default_rng(1234)
    write_wav(OUT / "noise.wav", rng.normal(0, 0.003, 3 * RATE))
    for p in sorted(OUT.glob("*.wav")):
        print(f"{p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
