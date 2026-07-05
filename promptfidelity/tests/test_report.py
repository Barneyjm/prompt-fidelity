"""Tests for promptfidelity.report -- deterministic text rendering at four
audiences. No LLM anywhere: every assertion here is a substring check
against a plain string template built from hand-built Constraint/Ledger
fixtures.
"""

import pytest

from promptfidelity.core import Constraint, Ledger, book, merge_ledgers
from promptfidelity.report import render


def _all_six_accounts_ledger() -> Ledger:
    """One constraint per user-side account, plus one imposed argument --
    six accounts total, exercised in every audience test below."""
    constraints = [
        Constraint(id="genre", description="science fiction genre",
                   params={"with_genres": "878"}, p=0.08),
        Constraint(id="mood", description="cozy mood",
                   params={"query": "cozy"}, p=0.2),
        Constraint(id="rating", description="rated above 7",
                   params={"vote_average.gte": "7.0"}, p=0.1),
        Constraint(id="pacing", description="slow, meditative pacing", p=0.15),
        Constraint(id="votes", description="min votes 50",
                   params={"vote_count.gte": "50"}, p=0.3),
    ]
    args = {
        "with_genres": "878",       # verified
        "query": "cozy",            # transmitted (advisory)
        "vote_average.gte": "6.5",  # substituted (declared 7.0)
        # "vote_count.gte" absent -> dropped
        "popularity.gte": "80",     # imposed -- no constraint claims it
    }
    return book(constraints, args, advisory_params={"query"})


def test_unknown_audience_raises_value_error():
    ledger = book([], {})
    with pytest.raises(ValueError):
        render(ledger, "marketing")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_model_render_empty_when_nothing_unhonored_and_conjunction_holds():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    ledger = book([c], {"g": "878"})
    assert ledger.render("model") == ""


def test_model_render_contains_declared_params_of_dropped_constraint():
    ledger = _all_six_accounts_ledger()
    rendering = ledger.render("model")
    assert "votes" in rendering
    assert "vote_count.gte" in rendering
    assert "50" in rendering
    assert "Re-issue ONE complete tool call" in rendering


def test_model_render_conjunction_warning_when_nothing_unhonored_but_piecewise():
    c1 = Constraint(id="genre", description="genre", params={"g": "878"}, p=0.5)
    c2 = Constraint(id="rating", description="rating", params={"r": "8"}, p=0.5)
    call_0 = book([c1, c2], {"g": "878"})
    call_1 = book([c1, c2], {"r": "8"})
    merged = merge_ledgers([c1, c2], [call_0, call_1])
    rendering = merged.render("model")
    assert "CONJUNCTION" in rendering
    assert "Re-issue" in rendering
    # Nothing unhonored, so the per-entry UNHONORED block must not appear.
    assert "UNHONORED" not in rendering


# ---------------------------------------------------------------------------
# engineer
# ---------------------------------------------------------------------------


def test_engineer_render_has_full_detail():
    ledger = _all_six_accounts_ledger()
    rendering = ledger.render("engineer")
    for account in ("verified", "transmitted", "substituted", "inferred", "dropped"):
        assert f"[{account}]" in rendering
    assert "bits=" in rendering
    assert "source=" in rendering
    assert "call=" in rendering
    assert "evidence=" in rendering
    assert "popularity.gte" in rendering  # imposed arg shows up
    assert "fidelity:" in rendering
    assert "conjunction_honored:" in rendering
    assert "calls credited:" in rendering


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------


def test_product_render_groups_plain_language_no_bits():
    ledger = _all_six_accounts_ledger()
    rendering = ledger.render("product")
    assert "Delivered as asked" in rendering
    assert "Delivered approximately" in rendering
    assert "Handled by model/agent judgment" in rendering
    assert "Not delivered" in rendering
    assert "unmet demand" in rendering
    assert "science fiction genre" in rendering
    assert "min votes 50" in rendering
    assert "bits" not in rendering.lower()


# ---------------------------------------------------------------------------
# executive
# ---------------------------------------------------------------------------


def test_executive_render_mentions_counts_caveat_when_basis_is_counts():
    # No entry carries a survival estimate -> fidelity_basis == "counts".
    constraints = [
        Constraint(id="c1", description="genre", params={"g": "878"}),
        Constraint(id="c2", description="rating", params={"r": "8"}),
    ]
    ledger = book(constraints, {"g": "878", "r": "7"})  # c2 substituted
    assert ledger.fidelity_basis == "counts"
    rendering = ledger.render("executive")
    assert "unweighted" in rendering
    assert "count ratio" in rendering


def test_executive_render_short_and_covers_required_lines():
    ledger = _all_six_accounts_ledger()
    rendering = ledger.render("executive")
    lines = [line for line in rendering.splitlines() if line.strip()]
    assert len(lines) <= 5
    assert "Fidelity:" in rendering
    assert "constraints altered or dropped without disclosure" in rendering
    assert "agent-imposed filter" in rendering
    assert "overclaim risk" in rendering.lower()


def test_executive_render_no_overclaim_when_everything_honored_and_no_imposed():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    ledger = book([c], {"g": "878"})
    rendering = ledger.render("executive")
    assert "execution matched stated intent" in rendering.lower()
    assert "overclaim" not in rendering.lower()
