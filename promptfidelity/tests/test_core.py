"""Tests for promptfidelity.core -- the pure book() diff and Ledger math.

All fixtures here are hand-built dicts/Constraints, never model output --
these tests exist to pin down the mechanical account-assignment rules, not
to exercise any LLM.
"""

import pytest

from promptfidelity.core import Constraint, Ledger, book, bits, merge_ledgers


def test_bits_none_when_no_probability():
    assert bits(None) is None


def test_bits_boundaries():
    assert bits(1.0) == 0.0
    assert bits(0.0) == 20.0
    assert bits(-1) == 20.0


def test_bits_matches_neg_log2():
    assert bits(0.25) == pytest.approx(2.0)
    assert bits(0.5) == pytest.approx(1.0)


def test_constraint_bits_property_delegates_to_bits():
    c = Constraint(id="c1", description="x", params={}, p=0.25)
    assert c.bits == pytest.approx(2.0)
    c_no_p = Constraint(id="c2", description="y", params={})
    assert c_no_p.bits is None


def test_verified_all_params_match():
    c = Constraint(id="c1", description="Science fiction genre",
                    params={"with_genres": "878"}, p=0.08)
    ledger = book([c], {"with_genres": "878"})
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.account == "verified"
    assert entry.bits == pytest.approx(bits(0.08), abs=0.01)
    assert ledger.imposed == []


def test_transmitted_when_matched_param_is_advisory():
    c = Constraint(id="c1", description="mood: cozy", params={"query": "cozy"}, p=0.2)
    ledger = book([c], {"query": "cozy"}, advisory_params={"query"})
    assert ledger.entries[0].account == "transmitted"


def test_substituted_when_value_altered():
    c = Constraint(id="c1", description="min rating 8.0",
                    params={"vote_average.gte": "8.0"}, p=0.1)
    ledger = book([c], {"vote_average.gte": "7.0"})
    assert ledger.entries[0].account == "substituted"
    assert "altered" in ledger.entries[0].evidence


def test_substituted_on_partial_multiparam_application():
    c = Constraint(id="c1", description="genre + year",
                    params={"with_genres": "878", "primary_release_year": "1999"}, p=0.05)
    ledger = book([c], {"with_genres": "878"})
    assert ledger.entries[0].account == "substituted"
    assert "missing" in ledger.entries[0].evidence


def test_dropped_when_declared_param_absent():
    c = Constraint(id="c1", description="min votes 50",
                    params={"vote_count.gte": "50"}, p=0.3)
    ledger = book([c], {"with_genres": "878"})
    assert ledger.entries[0].account == "dropped"


def test_inferred_when_no_params_declared():
    c = Constraint(id="c1", description="slow, meditative pacing", p=0.1)
    ledger = book([c], {"with_genres": "878"})
    assert ledger.entries[0].account == "inferred"


def test_imposed_for_unclaimed_call_argument():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)
    ledger = book([c], {"with_genres": "878", "vote_count.gte": "50"})
    assert len(ledger.imposed) == 1
    imp = ledger.imposed[0]
    assert imp.description == "vote_count.gte"
    assert imp.value == "50"
    assert imp.bits is None


def test_conservation_i_total_equals_sum_of_five_user_accounts():
    constraints = [
        Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08),
        Constraint(id="c2", description="rating", params={"vote_average.gte": "8.0"}, p=0.1),
        Constraint(id="c3", description="min votes", params={"vote_count.gte": "50"}, p=0.3),
        Constraint(id="c4", description="pacing", p=0.1),
        Constraint(id="c5", description="year+genre",
                   params={"primary_release_year": "1999", "with_genres": "27"}, p=0.05),
    ]
    args = {
        "with_genres": "878",       # verified for c1
        "vote_average.gte": "7.0",  # substituted for c2 (declared 8.0)
        # c3 (vote_count.gte) absent -> dropped
        "primary_release_year": "1999",  # partial for c5 -> substituted
    }
    ledger = book(constraints, args)
    accounts = {e.account for e in ledger.entries}
    assert accounts == {"verified", "substituted", "dropped", "inferred"}
    manual_total = sum(e.bits or 0.0 for e in ledger.entries)
    assert ledger.total_bits == pytest.approx(manual_total)
    assert ledger.total_bits == pytest.approx(
        ledger.verified_bits + ledger.transmitted_bits + ledger.substituted_bits
        + ledger.inferred_bits + ledger.dropped_bits
    )


def test_fidelity_is_one_when_no_intent():
    ledger = book([], {})
    assert ledger.fidelity == 1.0
    assert ledger.total_bits == 0.0


def test_fidelity_ratio():
    constraints = [
        Constraint(id="c1", description="a", params={"x": "1"}, p=0.5),   # verified, 1 bit
        Constraint(id="c2", description="b", params={"y": "2"}, p=0.25),  # dropped, 2 bits
    ]
    ledger = book(constraints, {"x": "1"})
    assert ledger.fidelity == pytest.approx(1.0 / 3.0, abs=0.01)


def test_to_dict_shape_matches_dev_promptfidelity_v1():
    constraints = [
        Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08),
        Constraint(id="c2", description="pacing", p=0.1),
    ]
    ledger = book(constraints, {"with_genres": "878", "vote_count.gte": "50"})
    d = ledger.to_dict()

    assert d["v"] == 1
    assert {e["id"] for e in d["entries"]} == {"c1", "c2"}
    assert d["imposed"][0]["description"] == "vote_count.gte"
    summary = d["summary"]
    assert set(summary) == {"fidelity", "verified_bits", "total_bits", "accounts"}
    assert set(summary["accounts"]) == {
        "verified", "transmitted", "substituted", "inferred", "dropped"
    }
    assert summary["verified_bits"] == summary["accounts"]["verified"]


def test_prompt_id_only_present_when_set():
    ledger = Ledger(entries=[], imposed=[])
    assert "prompt_id" not in ledger.to_dict()
    ledger.prompt_id = "abc123"
    assert ledger.to_dict()["prompt_id"] == "abc123"


def test_merge_ledgers_prefers_best_booking_across_calls():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)
    first_call = book([c], {})                          # dropped
    second_call = book([c], {"with_genres": "878"})      # verified
    merged = merge_ledgers([c], [first_call, second_call])
    assert merged.entries[0].account == "verified"


def test_merge_ledgers_rank_order_full():
    # verified > transmitted > substituted > dropped
    verified_c = Constraint(id="v", description="verified-able", params={"a": "1"}, p=0.5)
    transmitted_c = Constraint(id="t", description="transmitted-able", params={"q": "cozy"}, p=0.5)
    substituted_c = Constraint(id="s", description="substituted-able", params={"b": "2"}, p=0.5)
    dropped_c = Constraint(id="d", description="dropped-able", params={"c": "3"}, p=0.5)
    constraints = [verified_c, transmitted_c, substituted_c, dropped_c]

    call_a = book(constraints, {"a": "1", "q": "cozy", "b": "wrong"}, advisory_params={"q"})
    call_b = book(constraints, {"a": "1"})  # would demote nothing but adds no new info

    merged = merge_ledgers(constraints, [call_a, call_b])
    by_id = {e.id: e.account for e in merged.entries}
    assert by_id["v"] == "verified"
    assert by_id["t"] == "transmitted"
    assert by_id["s"] == "substituted"
    assert by_id["d"] == "dropped"


def test_merge_ledgers_empty_calls_books_against_empty_args():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)
    merged = merge_ledgers([c], [])
    assert merged.entries[0].account == "dropped"


def test_merge_ledgers_dedupes_imposed_across_calls():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"}, p=0.08)
    call_a = book([c], {"with_genres": "878", "vote_count.gte": "50"})
    call_b = book([c], {"with_genres": "878", "vote_count.gte": "50"})
    merged = merge_ledgers([c], [call_a, call_b])
    assert len(merged.imposed) == 1
