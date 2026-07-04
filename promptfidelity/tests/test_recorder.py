"""Tests for promptfidelity.recorder -- trace()/@instrument/Recorder."""

import pytest

import promptfidelity as pf
from promptfidelity.core import Constraint


@pf.instrument
def discover_movies(**kwargs):
    return {"called_with": kwargs}


def test_instrument_is_transparent_noop_outside_trace():
    # No active trace() -- the wrapped function still runs and returns
    # normally, it just isn't recorded anywhere.
    result = discover_movies(with_genres="878")
    assert result == {"called_with": {"with_genres": "878"}}


def test_instrument_records_kwargs_inside_trace():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    with pf.trace(constraints) as rec:
        discover_movies(with_genres="878")
    ledger = rec.ledger()
    assert ledger.entries[0].account == "verified"


def test_recorder_ledger_with_no_calls_books_dropped():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    with pf.trace(constraints) as rec:
        pass
    ledger = rec.ledger()
    assert ledger.entries[0].account == "dropped"


def test_recorder_best_booking_across_multiple_calls():
    constraints = [
        Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08),
        Constraint(id="c2", description="year", params={"primary_release_year": "1999"}, p=0.1),
    ]
    with pf.trace(constraints) as rec:
        # First call only satisfies c1; second call (a refinement) adds c2.
        discover_movies(with_genres="878")
        discover_movies(with_genres="878", primary_release_year="1999")
    ledger = rec.ledger()
    by_id = {e.id: e.account for e in ledger.entries}
    assert by_id["c1"] == "verified"
    assert by_id["c2"] == "verified"


def test_recorder_credits_constraint_satisfied_on_any_call_even_if_others_drop_it():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    with pf.trace(constraints) as rec:
        discover_movies(query="something else")  # drops c1
        discover_movies(with_genres="878")        # verifies c1
    ledger = rec.ledger()
    assert ledger.entries[0].account == "verified"


def test_nested_traces_get_independent_recorders():
    outer_constraints = [Constraint(id="outer", description="o", params={"a": "1"}, p=0.5)]
    inner_constraints = [Constraint(id="inner", description="i", params={"b": "2"}, p=0.5)]

    with pf.trace(outer_constraints) as outer_rec:
        discover_movies(a="1")
        with pf.trace(inner_constraints) as inner_rec:
            discover_movies(b="2")
        # After the inner trace exits, instrument() should record against
        # the outer recorder again.
        discover_movies(a="1")

    outer_ledger = outer_rec.ledger()
    inner_ledger = inner_rec.ledger()
    assert outer_ledger.entries[0].account == "verified"
    assert inner_ledger.entries[0].account == "verified"
    # The inner recorder never saw the outer-only calls.
    assert len(inner_rec.calls) == 1


def test_instrument_only_records_kwargs_not_positional_args():
    @pf.instrument
    def tool(positional_value, keyword_value=None):
        return positional_value, keyword_value

    constraints = [Constraint(id="c1", description="x", params={"keyword_value": "42"}, p=0.5)]
    with pf.trace(constraints) as rec:
        result = tool("ignored", keyword_value="42")
    assert result == ("ignored", "42")
    ledger = rec.ledger()
    assert ledger.entries[0].account == "verified"
