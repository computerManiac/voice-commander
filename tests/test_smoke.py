from voiceagents.app import main


def test_main_runs() -> None:
    assert main() == 0
