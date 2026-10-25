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

## Phase 2: hotkeys, worker, feedback

Run `uv run python -m voiceagents -v` from the repo root in PowerShell (logs also go to `%LOCALAPPDATA%\voiceagents\logs`).

- [x] A grey tray icon appears at once; after the model loads it turns green and a "Ready" toast shows.
- [x] Press Ctrl+Alt+D before it turns green: a "Still loading" toast, no recording, no beeps.
- [ ] Hold Ctrl+Alt+D and speak: the icon turns red with a high beep; on release a lower beep, amber while working, then green and a toast `dictation: <your words>`.
- [x] Same with F10: the toast starts with `command:`.
- [x] Hold Ctrl+Alt+D and stay silent: "No speech detected" toast. A quick tap (under 0.3 s) does nothing.
- [x] Holding F10 in Chrome, Notepad and Explorer does not open a menu bar or move focus.
- [x] With `beeps = true`, the start beep does not show up as stray words in the transcript (if it does, set `beeps = false`).
- [x] Hold Ctrl+Alt+D, then press F10 as well: the F10 press is ignored; only the dictation job runs.
- [x] The default dictation chord Ctrl+Alt+D works, and bare `d` still types normally. Changing `dictation` in `config.toml` (e.g. to `f9`) takes effect on restart.
- [ ] Disable the microphone (Settings, Device Manager, or unplug it), hold Ctrl+Alt+D for a second: red icon, a "Can't access the microphone" toast and a low beep, and the app keeps running. Re-enable it: dictation works again without restarting.
- [x] Tray menu, Quit: the icon disappears and the process exits.
