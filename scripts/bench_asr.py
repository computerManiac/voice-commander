"""Phase 0 ASR benchmark: Parakeet TDT int8 via onnx-asr on CPU.

For each thread count, a fresh subprocess loads the model, warms it up, then transcribes clips of
several lengths. A fresh process per config keeps load time and peak RAM honest (peak working set
never goes down within a process).

Test audio is a Windows SAPI text-to-speech WAV unless --audio is given. TTS is cleaner than a real
microphone, so speed numbers are representative but any accuracy impression is not.

    uv run python scripts\\bench_asr.py --model nemo-parakeet-tdt-0.6b-v2 --threads 6
    uv run python scripts\\bench_asr.py --threads 4 5 6 --runs 5 --audio my_speech.wav
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

SAMPLE_RATE: Final = 16000
CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "bench_audio"
TTS_TEXT = (
    "Open Chrome and visit youtube dot com. Then search for the weather forecast in Pune this "
    "weekend. Please remind me to send the quarterly report to the finance team before noon on "
    "Friday, and turn the volume down a little. The quick brown fox jumps over the lazy dog while "
    "the afternoon sun settles behind the hills."
)

Audio = npt.NDArray[np.float32]


def synthesize_tts(path: Path) -> None:
    """Write a 16 kHz mono 16-bit WAV of TTS_TEXT using Windows SAPI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $f);"
        f"$s.Speak('{TTS_TEXT}');"
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )


def read_wav(path: Path) -> Audio:
    """Read a 16-bit PCM WAV at 16 kHz, downmixed to mono float32."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: need 16-bit PCM at {SAMPLE_RATE} Hz")
        raw = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, channels).mean(axis=1).astype(np.float32)


def make_clip(source: Audio, seconds: float) -> Audio:
    """Tile source (with 0.3 s gaps) and crop to exactly `seconds`."""
    gap = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
    n = int(seconds * SAMPLE_RATE)
    reps = int(np.ceil(n / (len(source) + len(gap))))
    return np.tile(np.concatenate([source, gap]), reps)[:n].copy()


def worker(args: argparse.Namespace) -> None:
    """Benchmark one thread count in this process and print a JSON line."""
    import onnx_asr
    import onnxruntime as rt
    import psutil

    threads: int = args.single
    so = rt.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    pre_so = rt.SessionOptions()
    pre_so.intra_op_num_threads = threads
    pre_so.inter_op_num_threads = 1

    t0 = time.perf_counter()
    model = onnx_asr.load_model(
        args.model,
        quantization=args.quantization,
        sess_options=so,
        preprocessor_config={"sess_options": pre_so, "max_concurrent_workers": 1},
    )
    load_s = time.perf_counter() - t0

    source = read_wav(Path(args.audio))
    t0 = time.perf_counter()
    model.recognize(make_clip(source, 2.0), sample_rate=SAMPLE_RATE)
    warmup_s = time.perf_counter() - t0

    clips = []
    for seconds in args.durations:
        clip = make_clip(source, seconds)
        times: list[float] = []
        text = ""
        for _ in range(args.runs):
            t0 = time.perf_counter()
            text = model.recognize(clip, sample_rate=SAMPLE_RATE)
            times.append(time.perf_counter() - t0)
        median = statistics.median(times)
        clips.append(
            {
                "seconds": seconds,
                "median_s": median,
                "min_s": min(times),
                "max_s": max(times),
                "rtf": median / seconds,
                "text": text,
            }
        )

    proc = psutil.Process()
    peak_mb = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**20
    print(
        "RESULT "
        + json.dumps(
            {
                "model": args.model,
                "quantization": args.quantization,
                "threads": threads,
                "load_s": load_s,
                "warmup_s": warmup_s,
                "peak_ram_mb": peak_mb,
                "clips": clips,
            }
        )
    )


def run_config(args: argparse.Namespace, threads: int, audio: Path) -> dict:
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--single",
        str(threads),
        "--model",
        args.model,
        "--quantization",
        args.quantization,
        "--runs",
        str(args.runs),
        "--audio",
        str(audio),
        "--durations",
        *[str(d) for d in args.durations],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"benchmark worker failed (threads={threads}):\n{proc.stderr[-2000:]}")


def print_report(results: list[dict]) -> None:
    print(f"\nModel: {results[0]['model']}  quantization: {results[0]['quantization']}")
    header = "threads | load s | warmup s | peak RAM MB | " + " | ".join(
        f"{c['seconds']:g}s: latency / RTF" for c in results[0]["clips"]
    )
    print(header)
    print("-" * len(header))
    for r in results:
        cells = " | ".join(f"{c['median_s']:.2f}s / {c['rtf']:.3f}" for c in r["clips"])
        print(
            f"{r['threads']:>7} | {r['load_s']:>6.2f} | {r['warmup_s']:>8.2f} | "
            f"{r['peak_ram_mb']:>11.0f} | {cells}"
        )
    sample = results[0]["clips"][0]
    print(f"\nTranscript of {sample['seconds']:g}s clip: {sample['text']!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="nemo-parakeet-tdt-0.6b-v2")
    p.add_argument("--quantization", default="int8")
    p.add_argument("--threads", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--durations", type=float, nargs="+", default=[5, 15, 30])
    p.add_argument("--runs", type=int, default=3, help="timed runs per clip (median reported)")
    p.add_argument("--audio", help="16 kHz 16-bit WAV of speech; default: generated with SAPI TTS")
    p.add_argument("--json", help="also write raw results to this path")
    p.add_argument("--single", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.single is not None:
        worker(args)
        return

    audio = Path(args.audio) if args.audio else CACHE_DIR / "tts.wav"
    if not audio.exists():
        print(f"Generating test speech with SAPI -> {audio}")
        synthesize_tts(audio)
    print(
        f"Source audio: {len(read_wav(audio)) / SAMPLE_RATE:.1f}s of speech, tiled to each length"
    )

    results = []
    for t in args.threads:
        print(f"Running threads={t} ...", flush=True)
        results.append(run_config(args, t, audio))
    print_report(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
