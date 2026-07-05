"""Runs promptfidelity.examples.repair_loop end to end and checks it exits
cleanly -- it doubles as living documentation, so it has to actually work,
not just import."""

from promptfidelity.examples import repair_loop


def test_repair_loop_example_runs_cleanly(capsys):
    repair_loop.main()
    out = capsys.readouterr().out
    assert "UNHONORED (2):" in out
    assert "conjunction_honored: True" in out
    assert "--- executive ---" in out
