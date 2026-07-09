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
    assert set(summary) == {
        "fidelity", "fidelity_basis", "auditability", "verified_bits",
        "total_bits", "accounts", "constraints_source"
    }
    assert set(summary["accounts"]) == {
        "verified", "derived", "transmitted", "substituted", "inferred", "dropped"
    }
    assert summary["verified_bits"] == summary["accounts"]["verified"]
    assert summary["constraints_source"] == {"declared": 2}


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


def test_constraint_source_defaults_to_declared():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"})
    assert c.source == "declared"


def test_source_flows_from_constraint_to_ledger_entry_and_dict():
    c = Constraint(id="c1", description="genre", params={"with_genres": "878"},
                    p=0.08, source="rules")
    ledger = book([c], {"with_genres": "878"})
    entry = ledger.entries[0]
    assert entry.source == "rules"
    assert entry.to_dict()["source"] == "rules"


def test_constraints_source_counts_mixed_provenance():
    constraints = [
        Constraint(id="c1", description="a", params={"x": "1"}, source="declared"),
        Constraint(id="c2", description="b", params={"y": "2"}, source="rules"),
        Constraint(id="c3", description="c", params={"z": "3"}, source="rules"),
        Constraint(id="c4", description="d", params={"w": "4"}, source="llm"),
    ]
    ledger = book(constraints, {"x": "1", "y": "2", "z": "3", "w": "4"})
    assert ledger.constraints_source == {"declared": 1, "rules": 2, "llm": 1}
    assert ledger.to_dict()["summary"]["constraints_source"] == {
        "declared": 1, "rules": 2, "llm": 1
    }


def test_counts_basis_fidelity_when_no_entry_carries_bits():
    """All-p=None ledgers (e.g. pure rules extraction) must not report
    fidelity 1.0 off an empty bits denominator: a substituted entry has to
    show up in the headline number, on a counts basis."""
    constraints = [
        Constraint(id="c1", description="genre", params={"g": "878"}),
        Constraint(id="c2", description="rating 8+", params={"r": "8"}),
    ]
    ledger = book(constraints, {"g": "878", "r": "7"})
    accounts = {e.id: e.account for e in ledger.entries}
    assert accounts == {"c1": "verified", "c2": "substituted"}
    assert ledger.total_bits == 0
    assert ledger.fidelity_basis == "counts"
    assert ledger.fidelity == 0.5
    assert ledger.to_dict()["summary"]["fidelity_basis"] == "counts"


def test_bits_basis_used_whenever_any_entry_carries_bits():
    constraints = [
        Constraint(id="c1", description="genre", params={"g": "878"}, p=0.25),
        Constraint(id="c2", description="rating 8+", params={"r": "8"}),
    ]
    ledger = book(constraints, {"g": "878", "r": "7"})
    assert ledger.fidelity_basis == "bits"
    assert ledger.fidelity == 1.0  # the only weighted entry is verified


# ---------------------------------------------------------------------------
# unhonored / conjunction_honored / params+call on LedgerEntry
# ---------------------------------------------------------------------------


def test_unhonored_contains_only_substituted_and_dropped():
    constraints = [
        Constraint(id="verified_c", description="genre", params={"g": "878"}, p=0.5),
        Constraint(id="substituted_c", description="rating",
                   params={"r.gte": "8.0"}, p=0.5),
        Constraint(id="dropped_c", description="votes", params={"v.gte": "50"}, p=0.5),
        Constraint(id="inferred_c", description="pacing", p=0.5),
    ]
    ledger = book(constraints, {"g": "878", "r.gte": "7.0"})
    accounts = {e.id: e.account for e in ledger.entries}
    assert accounts == {
        "verified_c": "verified", "substituted_c": "substituted",
        "dropped_c": "dropped", "inferred_c": "inferred",
    }
    assert {e.id for e in ledger.unhonored} == {"substituted_c", "dropped_c"}


def test_book_leaves_call_none():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    ledger = book([c], {"g": "878"})
    assert ledger.entries[0].call is None


def test_entry_params_copied_through_book():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    ledger = book([c], {"g": "878"})
    assert ledger.entries[0].params == {"g": "878"}


def test_entry_params_empty_for_inferred():
    c = Constraint(id="c1", description="pacing", p=0.5)
    ledger = book([c], {})
    assert ledger.entries[0].params == {}


def test_to_dict_omits_params_when_empty_and_includes_when_present():
    constraints = [
        Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5),
        Constraint(id="c2", description="pacing", p=0.5),
    ]
    ledger = book(constraints, {"g": "878"})
    by_id = {e["id"]: e for e in ledger.to_dict()["entries"]}
    assert by_id["c1"]["params"] == {"g": "878"}
    assert "params" not in by_id["c2"]


def test_to_dict_omits_call_when_none_and_includes_when_set():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    plain = book([c], {"g": "878"})
    assert "call" not in plain.to_dict()["entries"][0]

    merged = merge_ledgers([c], [plain])
    assert merged.to_dict()["entries"][0]["call"] == 0


def test_conjunction_honored_true_for_bare_single_call_book():
    constraints = [
        Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5),
        Constraint(id="c2", description="rating", params={"r": "8"}, p=0.5),
    ]
    ledger = book(constraints, {"g": "878", "r": "8"})
    assert ledger.conjunction_honored is True


def test_conjunction_honored_false_when_repair_is_piecewise():
    # Two separate calls, each verifying a DIFFERENT constraint -- the
    # intent as a whole was honored, but never by any one call together.
    c1 = Constraint(id="genre", description="genre", params={"g": "878"}, p=0.5)
    c2 = Constraint(id="rating", description="rating", params={"r": "8"}, p=0.5)
    call_0 = book([c1, c2], {"g": "878"})       # verifies genre, drops rating
    call_1 = book([c1, c2], {"r": "8"})         # verifies rating, drops genre
    merged = merge_ledgers([c1, c2], [call_0, call_1])
    assert not merged.unhonored
    assert merged.conjunction_honored is False


def test_conjunction_honored_true_after_one_complete_repair_call():
    # First call honors NEITHER constraint (both dropped); the second,
    # complete call verifies both together in one shot.
    c1 = Constraint(id="genre", description="genre", params={"g": "878"}, p=0.5)
    c2 = Constraint(id="rating", description="rating", params={"r": "8"}, p=0.5)
    call_0 = book([c1, c2], {})                        # drops both
    call_1 = book([c1, c2], {"g": "878", "r": "8"})     # verifies both, together
    merged = merge_ledgers([c1, c2], [call_0, call_1])
    assert not merged.unhonored
    assert merged.conjunction_honored is True
    assert {e.call for e in merged.entries if e.account == "verified"} == {1}


def test_merge_ledgers_copies_entries_not_mutates_input_ledgers():
    c = Constraint(id="c1", description="genre", params={"g": "878"}, p=0.5)
    call_0 = book([c], {"g": "878"})
    assert call_0.entries[0].call is None
    merge_ledgers([c], [call_0])
    # The original ledger's own entry must be untouched by the merge.
    assert call_0.entries[0].call is None


def test_complete_repair_call_yields_conjunction_honored():
    """The recommended repair pattern -- re-issue ONE complete call
    including the already-honored params -- must read as a coherent
    conjunction. Regression: with first-wins tie-breaking, the earlier
    verified booking kept its old call index and conjunction_honored
    stayed False on a perfect repair."""
    a = Constraint(id="a", description="genre", params={"g": "878"}, p=0.25)
    b = Constraint(id="b", description="rating", params={"r": "8"}, p=0.25)
    first = book([a, b], {"g": "878"})              # a verified, b dropped
    repair = book([a, b], {"g": "878", "r": "8"})   # complete corrective call
    merged = merge_ledgers([a, b], [first, repair])
    assert {e.id: e.account for e in merged.entries} == {"a": "verified", "b": "verified"}
    assert {e.call for e in merged.entries} == {1}
    assert merged.conjunction_honored


def test_partial_patch_repair_stays_piecewise():
    """A repair that only patches the missing param leaves the earlier
    booking on its own call index -- no single call satisfied everything,
    and conjunction_honored must say so."""
    a = Constraint(id="a", description="genre", params={"g": "878"}, p=0.25)
    b = Constraint(id="b", description="rating", params={"r": "8"}, p=0.25)
    first = book([a, b], {"g": "878"})   # a verified, b dropped
    patch = book([a, b], {"r": "8"})     # b verified, a dropped
    merged = merge_ledgers([a, b], [first, patch])
    assert {e.id: e.account for e in merged.entries} == {"a": "verified", "b": "verified"}
    assert not merged.conjunction_honored


# ---------------------------------------------------------------------------
# paraphrase-aware transmission: agents rewrite intent into their own query
# vocabulary; exact matching alone systematically undercounts transmitted.


def test_paraphrased_advisory_value_books_transmitted():
    constraints = [Constraint(id="c1", description="budget under $500",
                              params={"query": "used bike under $500"}, p=0.5)]
    ledger = book(
        constraints,
        {"query": "used Trek FX hybrid commuter bike Charlotte NC for sale under $500"},
        advisory_params={"query"},
    )
    assert ledger.entries[0].account == "transmitted"
    assert "paraphrased" in ledger.entries[0].evidence


def test_paraphrase_below_fraction_stays_substituted():
    # only 2/4 declared content words survive -- below PARAPHRASE_WORD_FRACTION
    constraints = [Constraint(id="c1", description="lightweight bike",
                              params={"query": "lightweight city commuter bike"}, p=0.5)]
    ledger = book(
        constraints,
        {"query": "used Trek FX hybrid commuter bike Charlotte NC"},
        advisory_params={"query"},
    )
    assert ledger.entries[0].account == "substituted"


def test_paraphrase_never_applies_to_enforced_params():
    # identical containment, but the param is enforced: a different value is
    # a different filter, not a paraphrase.
    constraints = [Constraint(id="c1", description="summary",
                              params={"summary": "dentist appointment friday"}, p=0.5)]
    ledger = book(constraints, {"summary": "Dentist appointment on Friday at 2pm"})
    assert ledger.entries[0].account == "substituted"


def test_paraphrase_needs_min_declared_words():
    # one content word has no reliable fraction; exact match only.
    constraints = [Constraint(id="c1", description="zoo",
                              params={"query": "zoo"}, p=0.5)]
    ledger = book(constraints, {"query": "zoos in Washington DC"},
                  advisory_params={"query"})
    assert ledger.entries[0].account == "substituted"


# ---------------------------------------------------------------------------
# derivation chaining: an argument value looked up in a prior tool result,
# adjacent to the constraint's own words, is a documented translation of the
# user's vocabulary into the tool's.

_ZONES_BLOB = (
    '{"zones": [{"zoneNumber": 3, "name": "Mailbox Yard", '
    '"id": "aaaa1111-0000-0000-0000-000000000000", "enabled": true}, '
    '{"zoneNumber": 4, "name": "Front Garden", '
    '"id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a", "enabled": true}]}'
)


def test_inferred_constraint_with_prior_result_chain_books_derived():
    constraints = [Constraint(id="c1", description="run the Front Garden zone",
                              params={}, p=0.5)]
    ledger = book(
        constraints,
        {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
        prior_results=[_ZONES_BLOB],
    )
    assert ledger.entries[0].account == "derived"
    assert "prior_results[0]" in ledger.entries[0].evidence
    # the anchoring argument is claimed, not imposed
    assert ledger.imposed == []


def test_no_prior_results_means_no_derivation():
    constraints = [Constraint(id="c1", description="run the Front Garden zone",
                              params={}, p=0.5)]
    ledger = book(constraints,
                  {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"})
    assert ledger.entries[0].account == "inferred"
    assert [i.description for i in ledger.imposed] == ["zones[0].id"]


def test_derivation_requires_adjacency_not_mere_cooccurrence():
    # value present in a prior blob, but the constraint's words are nowhere
    # near it (outside DERIVATION_WINDOW) -- no chain.
    far_blob = ('{"id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}' + " " * 400
                + '{"name": "Front Garden zone"}')
    constraints = [Constraint(id="c1", description="run the Front Garden zone",
                              params={}, p=0.5)]
    ledger = book(constraints,
                  {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
                  prior_results=[far_blob])
    assert ledger.entries[0].account == "inferred"


def test_short_argument_values_never_anchor_derivation():
    # "1200" is under DERIVED_MIN_VALUE_LEN -- generic values chain everywhere.
    constraints = [Constraint(id="c1", description="run the Front Garden zone",
                              params={}, p=0.5)]
    ledger = book(constraints, {"duration_seconds": "1200"},
                  prior_results=['{"name": "Front Garden", "duration": 1200}'])
    assert ledger.entries[0].account == "inferred"


def test_derivation_never_shadows_argument_level_bookings():
    # a verified constraint stays verified even when a chain would also match
    constraints = [Constraint(id="c1", description="Front Garden zone id",
                              params={"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
                              p=0.5)]
    ledger = book(constraints,
                  {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
                  prior_results=[_ZONES_BLOB])
    assert ledger.entries[0].account == "verified"


def test_auditability_counts_verified_derived_transmitted():
    constraints = [
        Constraint(id="c1", description="genre filter", params={"with_genres": "878"}, p=0.5),
        Constraint(id="c2", description="run the Front Garden zone", params={}, p=0.5),
        Constraint(id="c3", description="melancholy tone", params={}, p=0.5),
    ]
    ledger = book(
        constraints,
        {"with_genres": "878", "zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
        prior_results=[_ZONES_BLOB],
    )
    accounts = {e.id: e.account for e in ledger.entries}
    assert accounts == {"c1": "verified", "c2": "derived", "c3": "inferred"}
    assert ledger.fidelity == pytest.approx(1.0 / 3.0)
    assert ledger.auditability == pytest.approx(2.0 / 3.0)


def test_merge_ranks_derived_above_transmitted_below_verified():
    constraints = [Constraint(id="c1", description="run the Front Garden zone",
                              params={}, p=0.5)]
    inferred = book(constraints, {})
    derived = book(constraints,
                   {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
                   prior_results=[_ZONES_BLOB])
    merged = merge_ledgers(constraints, [inferred, derived])
    assert merged.entries[0].account == "derived"
    assert merged.entries[0].call == 1


def test_derivation_phrase_match_survives_boilerplate_dilution():
    # LLM-proposed descriptions carry lead-ins ("User specified to...") that
    # push unigram overlap below DERIVED_WORD_FRACTION; the contiguous
    # phrase "Front Garden" next to the UUID qualifies on its own.
    constraints = [Constraint(
        id="c1", description="User specified to run only the 'Front Garden' zone",
        params={}, p=0.5)]
    ledger = book(
        constraints,
        {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
        prior_results=[_ZONES_BLOB],
    )
    assert ledger.entries[0].account == "derived"
    assert "phrase" in ledger.entries[0].evidence


def test_derivation_phrase_must_be_contiguous():
    # same words present but never adjacent in the constraint text -- the
    # phrase signal must not fire (and unigrams stay under the fraction).
    constraints = [Constraint(
        id="c1",
        description="User specified the front area with a garden watered nightly maybe",
        params={}, p=0.5)]
    blob = ('{"zoneNumber": 4, "name": "Front Garden", '
            '"id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}')
    ledger = book(constraints,
                  {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"},
                  prior_results=[blob])
    assert ledger.entries[0].account == "inferred"
