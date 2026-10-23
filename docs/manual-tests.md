# Manual test checklist

Tick these on the target machine when a feature lands. Automated coverage is in `tests/`.

## Phase 1: audio and ASR

Run from the repo root in PowerShell.

- [x] `uv run python scripts\record_test.py --list-devices` lists your microphone.
- [x] `uv run python scripts\record_test.py` then say "open notepad and type hello world": the transcript matches, and `transcribe` is well under 1.5 s.
- [x] Same command, stay silent: `Transcript: ''` and the log says "no speech".
- [x] Same command with music or a fan running but no speech: empty transcript.
- [ ] `--device "<part of a name>"` picks that microphone (try built-in, USB, Bluetooth).
- [ ] Bluetooth headset: the capture still works when the device only offers its native sample rate (the log says it is resampling).
- [ ] `--seconds 50`: the log warns "recording truncated at 45 s".
- [ ] `--save last.wav` plays back cleanly, with no clicks or speed change.
