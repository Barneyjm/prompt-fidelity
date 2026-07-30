"""Tests for promptfidelity.hop2 -- attributing a hop-1 Ledger's booked
entries to hop-2 status (unaddressed / addressed_grounded /
addressed_ungrounded) against response text and tool-result blobs.

All fixtures here are hand-built Constraints/Ledgers and plain strings --
never model output -- these tests pin down the mechanical matching rules,
not any LLM's judgment. See hop2.py's module docstring for what "grounded"
does and doesn't mean (provenance, never truth).
"""

import pytest

from promptfidelity.core import Constraint, book
from promptfidelity.hop2 import Hop2Status, Hop2Report, attribute, render


# ---------------------------------------------------------------------------
# A realistic multi-account scenario: one entry per hop-1 account, hand
# constructed so each hop-2 outcome is unambiguous by inspection (see the
# per-entry commentary below).
# ---------------------------------------------------------------------------

_RESPONSE_TEXT = (
    "Here are some science fiction picks with a cozy feel, each one rated "
    "above 7.0 based on the average vote score, and each has more than 500 "
    "ratings behind it so the score is reliable."
)

_TOOL_RESULTS = [
    '{"title": "Deep Space", "genre": "science fiction", '
    '"vote_average": 6.5, "vote_count": 812}',
    '{"title": "Warm Orbit", "genre": "science fiction comedy", '
    '"vote_average": 6.7, "vote_count": 340}',
]


def _scenario_ledger():
    constraints = [
        # verified: with_genres sent and matches -> addressed via word
        # overlap ("science fiction" in both response and tool_results[0]),
        # grounded the same way.
        Constraint(id="genre", description="science fiction genre",
                   params={"with_genres": "878"}, p=0.08),
        # substituted: declared 7.0, actual call sent 6.5 -- the response
        # narrates the ORIGINAL declared threshold ("above 7.0"), which
        # matches no tool result -> addressed (param-value match on the
        # entry's own declared value), ungrounded.
        Constraint(id="rating", description="rated highly",
                   params={"vote_average.gte": "7.0"}, p=0.1),
        # transmitted (advisory "query"): mentioned in the response,
        # nothing in the tool results backs it -> addressed, ungrounded.
        Constraint(id="mood", description="cozy mood",
                   params={"query": "cozy"}, p=0.2),
        # inferred: no params at all, and the response never touches its
        # vocabulary -> unaddressed.
        Constraint(id="pacing", description="slow, meditative pacing", p=0.15),
        # dropped: never sent, but the response asserts the number anyway
        # (word "500") with no supporting tool result -> the
        # "knowledge_answered" case: dropped + addressed_ungrounded.
        Constraint(id="votes", description="minimum vote count of five hundred",
                   params={"vote_count.gte": "500"}, p=0.3),
        # dropped, and the response never mentions runtime at all -> the
        # "ignored" case: dropped + unaddressed.
        Constraint(id="runtime", description="under two hours long",
                   params={"with_runtime.lte": "120"}, p=0.1),
    ]
    args = {
        "with_genres": "878",
        "vote_average.gte": "6.5",  # declared 7.0 -> substituted
        "query": "cozy",
        # vote_count.gte and with_runtime.lte both absent -> dropped
    }
    return book(constraints, args, advisory_params={"query"})


def _scenario_report() -> Hop2Report:
    return attribute(_scenario_ledger(), _RESPONSE_TEXT, _TOOL_RESULTS)


def test_scenario_hop1_accounts_are_as_designed():
    # Pin down the hop-1 side of the fixture so a failure downstream can't
    # be blamed on a bad assumption about book()'s own behavior.
    ledger = _scenario_ledger()
    by_id = {e.id: e.account for e in ledger.entries}
    assert by_id == {
        "genre": "verified",
        "rating": "substituted",
        "mood": "transmitted",
        "pacing": "inferred",
        "votes": "dropped",
        "runtime": "dropped",
    }


def test_verified_entry_addressed_and_grounded_via_word_overlap():
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "genre")
    assert entry.status == Hop2Status.ADDRESSED_GROUNDED


def test_substituted_entry_addressed_via_declared_value_but_ungrounded():
    # The response narrates the ORIGINAL declared threshold (7.0), not the
    # actually-sent, lower one (6.5) -- addressed, but nothing in the tool
    # results supports "7.0", so it books ungrounded.
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "rating")
    assert entry.status == Hop2Status.ADDRESSED_UNGROUNDED
    assert "7.0" in entry.evidence


def test_transmitted_entry_addressed_but_ungrounded():
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "mood")
    assert entry.status == Hop2Status.ADDRESSED_UNGROUNDED


def test_inferred_entry_unaddressed():
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "pacing")
    assert entry.status == Hop2Status.UNADDRESSED


def test_dropped_entry_knowledge_answered():
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "votes")
    assert entry.status == Hop2Status.ADDRESSED_UNGROUNDED
    assert [e.id for e in report.knowledge_answered] == ["votes"]


def test_dropped_entry_ignored_when_never_mentioned():
    report = _scenario_report()
    entry = next(e for e in report.entries if e.id == "runtime")
    assert entry.status == Hop2Status.UNADDRESSED
    assert [e.id for e in report.ignored] == ["runtime"]


def test_narration_risk_covers_addressed_dropped_and_substituted_only():
    report = _scenario_report()
    # "rating" (substituted, addressed) and "votes" (dropped, addressed) --
    # NOT "mood" (transmitted, not a narration-risk account) and NOT
    # "runtime" (dropped, but never addressed).
    assert {e.id for e in report.narration_risk} == {"rating", "votes"}


def test_fate_matrix_counts_every_hop1_hop2_combination():
    report = _scenario_report()
    assert report.fate_matrix == {
        ("verified", "addressed_grounded"): 1,
        ("substituted", "addressed_ungrounded"): 1,
        ("transmitted", "addressed_ungrounded"): 1,
        ("inferred", "unaddressed"): 1,
        ("dropped", "addressed_ungrounded"): 1,
        ("dropped", "unaddressed"): 1,
    }
    assert sum(report.fate_matrix.values()) == len(report.entries) == 6


def test_to_dict_shape():
    report = _scenario_report()
    d = report.to_dict()
    assert set(d) == {"entries", "fate_matrix", "narration_risk", "knowledge_answered", "ignored"}
    assert {e["id"] for e in d["entries"]} == {
        "genre", "rating", "mood", "pacing", "votes", "runtime"
    }
    for e in d["entries"]:
        assert set(e) == {"id", "description", "hop1_account", "status", "evidence"}
    assert d["fate_matrix"]["verified/addressed_grounded"] == 1
    assert d["fate_matrix"]["dropped/addressed_ungrounded"] == 1
    assert d["fate_matrix"]["dropped/unaddressed"] == 1
    assert set(d["narration_risk"]) == {"rating", "votes"}
    assert d["knowledge_answered"] == ["votes"]
    assert d["ignored"] == ["runtime"]


def test_render_engineer_has_full_detail():
    report = _scenario_report()
    rendering = report.render("engineer")
    for combo in ("[verified/addressed_grounded]", "[substituted/addressed_ungrounded]",
                  "[transmitted/addressed_ungrounded]", "[inferred/unaddressed]",
                  "[dropped/addressed_ungrounded]", "[dropped/unaddressed]"):
        assert combo in rendering
    assert "fate_matrix" in rendering
    assert "evidence=" in rendering
    assert "narration_risk" in rendering
    assert "knowledge_answered" in rendering
    assert "ignored" in rendering
    assert "votes" in rendering
    assert "runtime" in rendering


def test_render_executive_is_short_and_covers_required_lines():
    report = _scenario_report()
    rendering = report.render("executive")
    lines = [line for line in rendering.splitlines() if line.strip()]
    assert len(lines) <= 5
    assert "addressed-ungrounded" in rendering
    assert "provenance risk" in rendering
    assert "narration-risk" in rendering
    assert "ignored" in rendering
    assert "provenance unknown" in rendering.lower() or "not necessarily false" in rendering.lower()


def test_render_unknown_audience_raises_value_error():
    report = _scenario_report()
    with pytest.raises(ValueError):
        render(report, "product")


def test_executive_closing_line_is_clean_when_everything_grounded():
    c = Constraint(id="c1", description="science fiction genre",
                    params={"with_genres": "878"}, p=0.5)
    ledger = book([c], {"with_genres": "878"})
    report = attribute(ledger, "a science fiction pick",
                        ['{"genre": "science fiction"}'])
    assert report.entries[0].status == Hop2Status.ADDRESSED_GROUNDED
    rendering = report.render("executive")
    assert "0 addressed-ungrounded" in rendering
    assert "traces to material the agent actually received" in rendering


# ---------------------------------------------------------------------------
# Isolated matching-rule tests
# ---------------------------------------------------------------------------


def test_grounded_via_param_value_match_in_tool_result():
    c = Constraint(id="c1", description="exact release id lookup",
                    params={"movie_id": "tt1234567"}, p=0.5)
    ledger = book([c], {"movie_id": "tt1234567"})  # verified at hop 1
    response = "I found the film with id tt1234567 in the catalog."
    tool_results = ['{"id": "tt1234567", "title": "Some Movie"}']
    report = attribute(ledger, response, tool_results)
    entry = report.entries[0]
    assert entry.status == Hop2Status.ADDRESSED_GROUNDED
    assert "tt1234567" in entry.evidence


def test_grounded_via_word_overlap_within_a_single_blob():
    c = Constraint(id="c1", description="quiet contemplative character study", p=0.5)
    ledger = book([c], {})  # inferred -- no params, irrelevant to hop 2
    response = "This is a quiet, contemplative character study."
    tool_results = [
        "A quiet, contemplative character study set in a small town.",
        "Unrelated content about cooking.",
    ]
    report = attribute(ledger, response, tool_results)
    assert report.entries[0].status == Hop2Status.ADDRESSED_GROUNDED


def test_words_scattered_across_blobs_stay_ungrounded():
    # Regression guard: pooling every blob's words together would find all
    # four content words somewhere in the combined tool_results and
    # wrongly call this grounded. Matching must stay per-blob.
    c = Constraint(id="c1", description="quiet contemplative character study", p=0.5)
    ledger = book([c], {})
    response = "This is a quiet, contemplative character study."
    tool_results = [
        "A quiet, contemplative introduction to the setting.",
        "This character study focuses on grief and loss.",
    ]
    report = attribute(ledger, response, tool_results)
    assert report.entries[0].status == Hop2Status.ADDRESSED_UNGROUNDED


def test_unaddressed_when_neither_value_nor_words_appear():
    c = Constraint(id="c1", description="under two hours long",
                    params={"with_runtime.lte": "120"}, p=0.1)
    ledger = book([c], {})  # dropped
    response = "Here are some great picks for you tonight."
    report = attribute(ledger, response, [])
    assert report.entries[0].status == Hop2Status.UNADDRESSED


def test_empty_response_text_leaves_everything_unaddressed():
    constraints = [
        Constraint(id="c1", description="cozy mood", params={"query": "cozy"}, p=0.2),
        Constraint(id="c2", description="slow, meditative pacing", p=0.1),
    ]
    ledger = book(constraints, {"query": "cozy"}, advisory_params={"query"})
    report = attribute(ledger, "", ["some tool result mentioning cozy"])
    assert all(e.status == Hop2Status.UNADDRESSED for e in report.entries)


def test_empty_tool_results_makes_every_addressed_entry_ungrounded():
    c = Constraint(id="c1", description="cozy mood", params={"query": "cozy"}, p=0.2)
    ledger = book([c], {"query": "cozy"}, advisory_params={"query"})
    report = attribute(ledger, "a cozy pick for tonight", [])
    entry = report.entries[0]
    assert entry.status == Hop2Status.ADDRESSED_UNGROUNDED
