"""Tests for promptfidelity.extraction.extract_constraints() -- the
deterministic Tier-1 regex/vocab extractor. No LLM, no network: these tests
pin down transcription rules, never judgment/grading.
"""

from promptfidelity.extraction import extract_constraints

VOCAB = {
    "terms": {
        "sci-fi": {"with_genres": "878"},
        "comedy": {"with_genres": "35"},
    },
    "comparators": {
        "rating": "vote_average",
        "runtime": "with_runtime",
        "votes": "vote_count",
    },
    "date_param": "primary_release_date",
}


def test_every_extracted_constraint_is_tagged_rules_with_no_probability():
    constraints = extract_constraints("a sci-fi movie from 1994", VOCAB)
    assert constraints
    assert all(c.source == "rules" for c in constraints)
    assert all(c.p is None for c in constraints)


def test_comparator_above_maps_to_gte_with_rating_keyword():
    constraints = extract_constraints("movies rated above 7", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"vote_average.gte": "7"} in by_params


def test_comparator_at_least_maps_to_gte():
    constraints = extract_constraints("rating at least 8.0", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"vote_average.gte": "8.0"} in by_params


def test_comparator_under_hours_converts_to_minutes_lte():
    constraints = extract_constraints("runtime under 2 hours", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"with_runtime.lte": "120"} in by_params


def test_comparator_shorter_than_minutes_maps_to_lte_runtime():
    constraints = extract_constraints("something shorter than 90 minutes long", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"with_runtime.lte": "90"} in by_params


def test_comparator_plus_suffix_maps_to_gte():
    constraints = extract_constraints("votes 100+", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"vote_count.gte": "100"} in by_params


def test_comparator_without_resolvable_concept_is_not_silently_dropped():
    # "above 7" with no vocab concept keyword nearby (and no "rating" or
    # "runtime" vocab in this call) shouldn't manufacture a param -- it
    # should fold into the residue text instead of vanishing.
    constraints = extract_constraints("give me something above 7 out of 10 great", vocab=None)
    assert all(c.params == {} for c in constraints)
    assert any(c.id == "residue" for c in constraints)


def test_year_from_produces_gte_only():
    constraints = extract_constraints("movies from 1994", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"primary_release_date.gte": "1994-01-01"} in by_params


def test_year_before_produces_lte_only():
    constraints = extract_constraints("movies before 1994", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"primary_release_date.lte": "1993-12-31"} in by_params


def test_bare_year_produces_gte_and_lte_bounding_that_year():
    constraints = extract_constraints("a movie set in 1994", VOCAB)
    by_params = [c.params for c in constraints]
    assert {
        "primary_release_date.gte": "1994-01-01",
        "primary_release_date.lte": "1994-12-31",
    } in by_params


def test_decade_word_form_the_90s_is_one_constraint_with_gte_and_lte():
    constraints = extract_constraints("something from the 90s", VOCAB)
    decade_constraints = [
        c for c in constraints
        if c.params.get("primary_release_date.gte") == "1990-01-01"
    ]
    assert len(decade_constraints) == 1
    c = decade_constraints[0]
    assert c.params == {
        "primary_release_date.gte": "1990-01-01",
        "primary_release_date.lte": "1999-12-31",
    }


def test_decade_digit_form_1990s():
    constraints = extract_constraints("comedy from the 1990s", VOCAB)
    by_params = [c.params for c in constraints]
    assert {
        "primary_release_date.gte": "1990-01-01",
        "primary_release_date.lte": "1999-12-31",
    } in by_params


def test_between_years_range():
    constraints = extract_constraints("movies between 1970 and 1979", VOCAB)
    by_params = [c.params for c in constraints]
    assert {
        "primary_release_date.gte": "1970-01-01",
        "primary_release_date.lte": "1979-12-31",
    } in by_params


def test_date_extraction_emits_nothing_when_date_param_is_none():
    # Structural recognition still happens (the year text is consumed, not
    # left dangling as residue noise) but nothing is emitted -- there is no
    # param to express it against.
    constraints = extract_constraints("a movie from 1994", vocab=None)
    assert not any("gte" in k or "lte" in k for c in constraints for k in c.params)


def test_vocab_none_only_structural_extraction_applies():
    # No terms, no comparators, no date_param -> nothing but residue.
    constraints = extract_constraints("a sci-fi movie rated above 7 from 1994", vocab=None)
    assert len(constraints) == 1
    assert constraints[0].id == "residue"


def test_vocab_term_matching_is_case_insensitive():
    constraints = extract_constraints("I want a SCI-FI movie", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"with_genres": "878"} in by_params


def test_vocab_term_matching_multiple_terms():
    constraints = extract_constraints("a sci-fi comedy", VOCAB)
    by_params = [c.params for c in constraints]
    assert {"with_genres": "878"} in by_params
    assert {"with_genres": "35"} in by_params


def test_residue_emitted_when_meaningful_text_remains():
    constraints = extract_constraints("a sci-fi movie that feels warm and nostalgic", VOCAB)
    residue = [c for c in constraints if c.id == "residue"]
    assert len(residue) == 1
    assert residue[0].params == {}
    assert residue[0].p is None
    assert residue[0].source == "rules"


def test_no_residue_when_remaining_text_is_only_stopwords_or_short():
    # Everything meaningful is claimed by the sci-fi term; what's left is
    # short/stopword filler, below the residue threshold.
    constraints = extract_constraints("a sci-fi movie", VOCAB)
    assert not any(c.id == "residue" for c in constraints)


def test_empty_prompt_produces_no_constraints():
    assert extract_constraints("", VOCAB) == []


def test_extraction_is_deterministic():
    prompt = "a sci-fi movie rated above 7 from the 90s with votes over 50, feeling wistful"
    first = extract_constraints(prompt, VOCAB)
    second = extract_constraints(prompt, VOCAB)
    assert [(c.id, c.description, c.params, c.source) for c in first] == \
           [(c.id, c.description, c.params, c.source) for c in second]


def test_constraints_ordered_by_position_in_prompt():
    # sci-fi appears before 1994 which appears before "rated above 7" --
    # extract_constraints orders by where the match starts in the prompt,
    # not by vocab dict iteration order.
    constraints = extract_constraints("a sci-fi movie from 1994 rated above 7", VOCAB)
    non_residue = [c for c in constraints if c.id != "residue"]
    genre_idx = next(i for i, c in enumerate(non_residue) if c.params.get("with_genres") == "878")
    date_idx = next(i for i, c in enumerate(non_residue)
                     if "primary_release_date.gte" in c.params)
    rating_idx = next(i for i, c in enumerate(non_residue)
                       if "vote_average.gte" in c.params)
    assert genre_idx < date_idx < rating_idx
