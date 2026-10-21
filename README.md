# voiceagents

Local hold-to-talk voice agents for Windows 11 (dictation and commands) on Parakeet TDT, CPU-only.
See `CLAUDE.md` for the plan and architecture.

## Setup

```powershell
uv sync                  # creates .venv with Python 3.12 and installs deps
uv run pytest
uv run ruff check . ; uv run ruff format .
uv run mypy src
uv run python -m voiceagents
```
