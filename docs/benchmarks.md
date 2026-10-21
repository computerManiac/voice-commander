# ASR benchmarks (Phase 0)

Produced by `scripts/bench_asr.py`: Parakeet TDT 0.6B, ONNX int8 via `onnx-asr` 0.12.0 /
onnxruntime 1.31.0, CPU execution provider. Each thread count runs in a fresh process; latency is the
median of 5 runs after a warm-up. Test audio is Windows SAPI TTS (20.7 s of speech, tiled with gaps
and cropped to 5 / 15 / 30 s), so it is cleaner than a real mic.

## Run 1: development machine (NOT the target)

Ryzen AI 7 PRO 350 (8C/16T, Zen 5), 31 GB RAM, Windows 11. The target is a Ryzen 5 4500U
(6C/6T, Zen 2), which is considerably slower per core, so **these numbers are a best case, not the
acceptance measurement.**

### nemo-parakeet-tdt-0.6b-v2 (int8)

| threads | load s | warm-up s | peak RAM MB | 5 s clip | 15 s clip | 30 s clip |
|---|---|---|---|---|---|---|
| 4 | 3.09 | 0.10 | 1201 | 0.14 s / 0.028 | 0.38 s / 0.025 | 0.84 s / 0.028 |
| 5 | 2.38 | 0.11 | 1204 | 0.16 s / 0.031 | 0.42 s / 0.028 | 0.88 s / 0.029 |
| 6 | 2.39 | 0.13 | 1204 | 0.14 s / 0.029 | 0.37 s / 0.025 | 0.83 s / 0.028 |
| 8 | 2.42 | 0.17 | 1209 | 0.14 s / 0.029 | 0.33 s / 0.022 | 0.75 s / 0.025 |

### nemo-parakeet-tdt-0.6b-v3 (int8, multilingual)

| threads | load s | warm-up s | peak RAM MB | 5 s clip | 15 s clip | 30 s clip |
|---|---|---|---|---|---|---|
| 4 | 28.04* | 0.10 | 1232 | 0.14 s / 0.029 | 0.38 s / 0.025 | 0.82 s / 0.027 |
| 5 | 2.42 | 0.13 | 1215 | 0.15 s / 0.030 | 0.41 s / 0.027 | 0.89 s / 0.030 |
| 6 | 2.46 | 0.13 | 1216 | 0.15 s / 0.030 | 0.39 s / 0.026 | 0.82 s / 0.027 |

Cells are median latency / real-time factor (latency divided by audio length). *First load of a
model includes the Hugging Face download and cold disk cache; later loads are about 2.4 s.

### Observations

- RTF is about 0.03 for both models, so a 5 s utterance takes about 0.15 s. That is far inside the
  1.5 s budget, with roughly 10x headroom.
- Thread count barely matters between 4 and 8 on this machine. Differences are within run-to-run
  noise except at 30 s, where 8 threads is slightly faster.
- v2 and v3 are equivalent in speed and RAM (about 1.2 GB peak). The language choice can be made on
  need alone: v2 for English-only, v3 for European languages.
- Both transcribe the sample correctly ("Open Chrome and visit youtube.com. ..."), with casing and
  punctuation.

## Run 2: target machine (TODO)

Not yet measured. Run on the Ryzen 5 4500U mini PC and paste the output here:

```powershell
uv run python scripts\bench_asr.py --model nemo-parakeet-tdt-0.6b-v2 --threads 4 5 6 --runs 5
uv run python scripts\bench_asr.py --model nemo-parakeet-tdt-0.6b-v3 --threads 4 5 6 --runs 5
```

Then set the `asr.threads` default from the result. Pass `--audio your.wav` (16 kHz, 16-bit) to use
real speech instead of TTS. If the 5 s clip takes longer than about 2 s there, stop and discuss
fallbacks (Moonshine, whisper.cpp small) per CLAUDE.md.
