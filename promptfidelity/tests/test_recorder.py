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


def test_trace_raises_value_error_when_neither_constraints_nor_prompt_given():
    with pytest.raises(ValueError):
        with pf.trace():
            pass


def test_trace_with_prompt_extracts_rules_constraints():
    vocab = {
        "terms": {"sci-fi": {"with_genres": "878"}},
        "comparators": {"rating": "vote_average"},
        "date_param": "primary_release_date",
    }
    with pf.trace(prompt="a sci-fi movie rated above 7", vocab=vocab) as rec:
        discover_movies(with_genres="878", vote_average_gte="7")
    assert rec.constraints
    assert all(c.source == "rules" for c in rec.constraints)
    assert any(c.params == {"with_genres": "878"} for c in rec.constraints)


def test_trace_constraints_wins_over_prompt_when_both_given():
    constraints = [Constraint(id="declared1", description="x", params={"a": "1"}, p=0.5)]
    with pf.trace(constraints, prompt="a sci-fi movie") as rec:
        pass
    assert rec.constraints == constraints
    assert rec.constraints[0].source == "declared"


def test_trace_merges_extractor_output_as_llm_source():
    def fake_extractor(prompt):
        return [Constraint(id="llm1", description="mood: cozy", params={"query": "cozy"}, p=0.3)]

    with pf.trace(prompt="something cozy to watch", extractor=fake_extractor) as rec:
        pass
    sources = {c.source for c in rec.constraints}
    assert "llm" in sources
    llm_constraints = [c for c in rec.constraints if c.source == "llm"]
    assert llm_constraints[0].id == "llm1"
    assert llm_constraints[0].params == {"query": "cozy"}


def test_trace_extractor_source_is_force_set_even_if_caller_lied():
    def fake_extractor(prompt):
        # Caller-supplied extractor incorrectly claims source="declared" --
        # trace() must not trust it.
        return [Constraint(id="llm1", description="x", params={"a": "1"},
                            p=0.5, source="declared")]

    with pf.trace(prompt="anything", extractor=fake_extractor) as rec:
        pass
    llm_constraint = next(c for c in rec.constraints if c.id == "llm1")
    assert llm_constraint.source == "llm"


def test_trace_rules_wins_on_param_name_collision_with_llm():
    vocab = {"comparators": {"rating": "vote_average"}}

    def fake_extractor(prompt):
        # Disagrees with the rules layer on the same param name.
        return [Constraint(id="llm1", description="rating guess",
                            params={"vote_average.gte": "9"}, p=0.4)]

    with pf.trace(prompt="movies rated above 7", extractor=fake_extractor, vocab=vocab) as rec:
        pass
    matching = [c for c in rec.constraints if "vote_average.gte" in c.params]
    assert len(matching) == 1
    assert matching[0].source == "rules"
    assert matching[0].params["vote_average.gte"] == "7"


def test_trace_llm_residue_subsumes_rules_residue():
    def fake_extractor(prompt):
        return [Constraint(id="llm_residue", description="broader unmapped intent",
                            params={}, p=None)]

    with pf.trace(prompt="a movie that feels warm and nostalgic and slow",
                   extractor=fake_extractor) as rec:
        pass
    assert not any(c.id == "residue" and c.source == "rules" for c in rec.constraints)
    assert any(c.id == "llm_residue" for c in rec.constraints)


def test_trace_llm_adds_coverage_rules_missed_without_collision():
    def fake_extractor(prompt):
        return [Constraint(id="llm1", description="mood: wistful",
                            params={"mood": "wistful"}, p=0.3)]

    with pf.trace(prompt="a sci-fi movie that feels wistful",
                   extractor=fake_extractor,
                   vocab={"terms": {"sci-fi": {"with_genres": "878"}}}) as rec:
        pass
    assert any(c.params == {"with_genres": "878"} and c.source == "rules"
               for c in rec.constraints)
    assert any(c.params == {"mood": "wistful"} and c.source == "llm"
               for c in rec.constraints)


# ---------------------------------------------------------------------------
# check() -- mid-trace repair loop surface
# ---------------------------------------------------------------------------


def test_check_mid_trace_equals_ledger():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    with pf.trace(constraints) as rec:
        discover_movies(with_genres="878")
        mid = rec.check()
        assert mid.entries[0].account == "verified"
    assert mid.to_dict() == rec.ledger().to_dict()


def test_check_reflects_repair_across_calls():
    constraints = [
        Constraint(id="genre", description="genre", params={"with_genres": "878"}, p=0.08),
        Constraint(id="rating", description="rating", params={"vote_average.gte": "7.0"}, p=0.1),
    ]
    with pf.trace(constraints) as rec:
        discover_movies(with_genres="878")  # drops rating
        first = rec.check()
        assert first.unhonored
        assert any(e.id == "rating" for e in first.unhonored)

        # Scripted repair: re-issue ONE complete corrective call.
        discover_movies(**{"with_genres": "878", "vote_average.gte": "7.0"})
        second = rec.check()
    assert not second.unhonored


# ---------------------------------------------------------------------------
# ignore_params
# ---------------------------------------------------------------------------


def test_ignore_params_filters_imposed_on_recorder():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    rec = pf.Recorder(constraints, ignore_params={"page", "api_key"})
    rec.record_call("discover", {"with_genres": "878", "page": "2", "api_key": "secret"})
    ledger = rec.ledger()
    assert ledger.imposed == []


def test_ignore_params_only_filters_exact_names():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    rec = pf.Recorder(constraints, ignore_params={"page"})
    rec.record_call("discover", {"with_genres": "878", "page_size": "20"})
    ledger = rec.ledger()
    # "page_size" is not an exact match for "page" -- it must still book imposed.
    assert any(e.description == "page_size" for e in ledger.imposed)


def test_ignore_params_on_trace():
    constraints = [Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)]
    with pf.trace(constraints, ignore_params={"page"}) as rec:
        discover_movies(with_genres="878", page="3")
    ledger = rec.ledger()
    assert ledger.imposed == []


# ---------------------------------------------------------------------------
# result capture: @instrument records what tools RETURN, which feeds
# derivation chaining (ledger) and effect verification (effects()).


def test_instrument_records_results():
    import promptfidelity as pf

    @pf.instrument
    def list_zones():
        return {"zones": [{"name": "Front Garden",
                           "id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}]}

    c = pf.Constraint(id="c1", description="anything", params={}, p=0.5)
    with pf.trace([c]) as rec:
        list_zones()
    assert rec.results[0] is not None
    assert "Front Garden" in rec.results[0]


def test_instrument_records_exception_as_error_shaped_result():
    import promptfidelity as pf

    @pf.instrument
    def boom():
        raise RuntimeError("backend exploded")

    c = pf.Constraint(id="c1", description="anything", params={}, p=0.5)
    with pf.trace([c]) as rec:
        try:
            boom()
        except RuntimeError:
            pass
    assert "exception" in rec.results[0]
    report = rec.effects()
    assert report.entries[0].status == "failed"


def test_ledger_wires_prior_results_into_derivation():
    # call 1 looks the zone list up; call 2 sends the UUID. The constraint
    # naming "Front Garden" must book `derived` off call 2, with call 1's
    # result as the evidence surface -- no manual prior_results plumbing.
    import promptfidelity as pf

    @pf.instrument
    def list_zones():
        return {"zones": [{"name": "Front Garden",
                           "id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}]}

    @pf.instrument
    def start_zone(**kwargs):
        return ""

    c = pf.Constraint(id="c1", description="run the Front Garden zone",
                      params={}, p=0.5)
    with pf.trace([c]) as rec:
        list_zones()
        start_zone(**{"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"})
    ledger = rec.ledger()
    assert ledger.entries[0].account == "derived"


def test_a_calls_own_result_is_not_its_own_derivation_surface():
    # only ONE call, whose own result contains the mapping: prior_results
    # for call 0 is empty, so no derivation -- the agent never looked it up
    # before acting.
    import promptfidelity as pf

    @pf.instrument
    def start_zone(**kwargs):
        return {"started": {"name": "Front Garden",
                            "id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}}

    c = pf.Constraint(id="c1", description="run the Front Garden zone",
                      params={}, p=0.5)
    with pf.trace([c]) as rec:
        start_zone(**{"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"})
    assert rec.ledger().entries[0].account == "inferred"
