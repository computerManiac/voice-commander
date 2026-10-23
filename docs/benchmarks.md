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

## Run 2: target machine (Ryzen 5 4500U)

Ryzen 5 4500U (6C/6T, Zen 2), 15.4 GB RAM, Windows 11. This is the acceptance measurement. Same
script, TTS audio, 5 timed runs per clip. Reproduce with:

```powershell
uv run python scriptsench_asr.py --model nemo-parakeet-tdt-0.6b-v2 --threads 4 5 6 --runs 5
uv run python scriptsench_asr.py --model nemo-parakeet-tdt-0.6b-v3 --threads 4 5 6 --runs 5
```

### nemo-parakeet-tdt-0.6b-v2 (int8)

| threads | load s | warm-up s | peak RAM MB | 5 s clip | 15 s clip | 30 s clip |
|---|---|---|---|---|---|---|
| 4 | 49.78* | 0.24 | 1222 | 0.31 s / 0.062 | 0.94 s / 0.063 | 1.96 s / 0.065 |
| 5 | 5.34 | 0.31 | 1205 | 0.38 s / 0.077 | 0.96 s / 0.064 | 1.82 s / 0.061 |
| 6 | 5.48 | 0.40 | 1209 | 0.35 s / 0.070 | 0.85 s / 0.056 | 1.79 s / 0.060 |

### nemo-parakeet-tdt-0.6b-v3 (int8, multilingual)

| threads | load s | warm-up s | peak RAM MB | 5 s clip | 15 s clip | 30 s clip |
|---|---|---|---|---|---|---|
| 4 | 53.20* | 0.20 | 1232 | 0.33 s / 0.066 | 0.90 s / 0.060 | 1.96 s / 0.065 |
| 5 | 5.34 | 0.27 | 1215 | 0.33 s / 0.066 | 0.85 s / 0.056 | 1.81 s / 0.060 |
| 6 | 5.19 | 0.27 | 1214 | 0.34 s / 0.068 | 0.81 s / 0.054 | 1.73 s / 0.058 |

*First run of each model includes the Hugging Face download and cold disk cache.

### Observations

- RTF is about 0.06 to 0.07, roughly twice the dev machine. A 5 s utterance takes about 0.35 s,
  far under the 2 s stop threshold and the 1.5 s key-release budget (about 4x headroom for VAD,
  typing and overhead).
- A 30 s clip takes about 1.8 s, so `max_duration_s = 45` is fine (about 2.7 s worst case).
- Thread count 4, 5 and 6 differ by less than run-to-run noise. 6 is marginally fastest at 15 s and
  30 s, but uses every core while the UI, audio callback and hotkey threads also need time.
- v2 and v3 are equivalent in speed and RAM (about 1.2 GB peak). Load is about 5 s once cached.
- Scope: TTS audio, not a real microphone. Speed is representative; accuracy is not. Re-measure
  end-to-end key-release latency in Phase 7.

### Decision

- `asr.threads = 5` stays the default (leaves one core free; no measurable cost versus 6).
- Parakeet is fast enough on the target CPU, so no fallback (Moonshine, whisper.cpp) is needed.
- v2 vs v3: still a language-need decision. v2 is the default; switch `asr.model` to
  `nemo-parakeet-tdt-0.6b-v3` for European languages at no speed or memory cost.
