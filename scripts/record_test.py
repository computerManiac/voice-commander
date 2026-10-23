"""Mic sanity check: record a few seconds (or read a WAV), print the transcript and timings.

uv run python scripts\\record_test.py                  # 5 s from the default mic
uv run python scripts\\record_test.py --list-devices
uv run python scripts\\record_test.py --device "USB" --seconds 8 --save last.wav
uv run python scripts\\record_test.py --wav tests\\fixtures\\speech.wav
"""

from __future__ import annotations

import argparse
import logging
import time
import wave
from pathlib import Path

import numpy as np

from voiceagents.config import AsrConfig, AudioConfig
from voiceagents.core.asr import ParakeetASR
from voiceagents.core.audio import Audio, Recorder, read_wav
from voiceagents.core.vad import make_detector


def record(config: AudioConfig, seconds: float) -> Audio:
    recorder = Recorder(config)
    recorder.open()
    time.sleep(1.0)  # some devices (Bluetooth) take ~0.5 s to start delivering audio
    print(f"Recording {seconds:g} s... speak now")
    recorder.start()
    time.sleep(seconds)
    audio = recorder.stop()
    recorder.close()
    return audio


def save_wav(path: Path, audio: Audio, rate: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--device", default="", help="input device name substring or index")
    p.add_argument("--wav", type=Path, help="transcribe this file instead of recording")
    p.add_argument("--save", type=Path, help="write the recorded audio to this WAV")
    p.add_argument("--no-vad", action="store_true", help="use the energy gate instead of Silero")
    p.add_argument("--threads", type=int, default=AsrConfig.threads)
    p.add_argument("--model", default=AsrConfig.model)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)

    audio_config = AudioConfig(device=args.device, vad=not args.no_vad)
    asr = ParakeetASR(
        AsrConfig(model=args.model, threads=args.threads), make_detector(audio_config.vad)
    )
    t0 = time.perf_counter()
    asr.load()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    asr.warm_up()
    warm_s = time.perf_counter() - t0

    audio = read_wav(args.wav) if args.wav else record(audio_config, args.seconds)
    if args.save:
        save_wav(args.save, audio, audio_config.sample_rate)
    seconds = audio.size / audio_config.sample_rate
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"Captured {seconds:.2f} s, peak {peak:.3f}")

    t0 = time.perf_counter()
    text = asr.transcribe(audio)
    took = time.perf_counter() - t0

    print(f"Transcript: {text!r}")
    rtf = took / seconds if seconds else 0.0
    print(f"load {load_s:.2f} s | warm-up {warm_s:.2f} s | transcribe {took:.2f} s (RTF {rtf:.3f})")


if __name__ == "__main__":
    main()
