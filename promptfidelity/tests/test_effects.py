"""Tests for promptfidelity.effects -- hop 1.5, effect verification.

All fixtures are hand-built calls/blobs modeled on the real-conversation
shapes that motivated the module (a permission-denied calendar write, a
silent irrigation start, an echoing cart build), never model output.
"""

import pytest

from promptfidelity.effects import EffectStatus, verify_effects


CAL_CALL = {"name": "create_event",
            "arguments": {"summary": "Eviction: 10-day appeal window ends",
                          "startTime": "2026-05-29T00:00:00Z"}}
ZONE_CALL = {"name": "start_multiple_zones",
             "arguments": {"zones[0].id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a",
                           "confirm": "True"}}


def test_failed_when_own_result_is_error_shaped():
    report = verify_effects([CAL_CALL], ["No approval received."])
    assert report.entries[0].status == EffectStatus.FAILED
    assert "no approval" in report.entries[0].evidence


def test_failed_disclosed_flag_reads_response_text():
    disclosed = verify_effects(
        [CAL_CALL], ["No approval received."],
        response_text="Looks like the calendar permission prompt didn't go through.")
    undisclosed = verify_effects(
        [CAL_CALL], ["No approval received."],
        response_text="All five events are on your calendar!")
    assert disclosed.entries[0].disclosed is True
    assert undisclosed.entries[0].disclosed is False
    assert undisclosed.undisclosed_failures == [undisclosed.entries[0]]


def test_unaudited_when_result_is_silent_acknowledgment():
    # the Rachio shape: the controller answers "" -- no error, no echo,
    # and nothing ever read the state back.
    report = verify_effects([ZONE_CALL], ['""'])
    assert report.entries[0].status == EffectStatus.UNAUDITED
    assert report.unaudited == report.entries


def test_confirmed_echo_when_own_result_echoes_argument_value():
    result = '{"items_queued": [{"query": "chicken thighs", "status": "added"}]}'
    call = {"name": "cart", "arguments": {"queries[0].query": "chicken thighs"}}
    report = verify_effects([call], [result])
    assert report.entries[0].status == EffectStatus.CONFIRMED_ECHO


def test_confirmed_readback_when_later_result_reflects_the_write():
    later = ('{"zones": [{"id": "78b0113a-ea50-4f1e-97a8-8c596b31f16a", '
             '"status": "WATERING"}]}')
    reader = {"name": "get_device_state", "arguments": {}}
    report = verify_effects([ZONE_CALL, reader], ['""', later])
    assert report.entries[0].status == EffectStatus.CONFIRMED_READBACK
    assert "later results[1]" in report.entries[0].evidence


def test_readback_outranks_echo():
    # both signals present: read-back is the stronger claim and must win.
    own = '{"accepted": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}'
    later = '{"running": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}'
    report = verify_effects([ZONE_CALL, {"name": "poll", "arguments": {}}],
                            [own, later])
    assert report.entries[0].status == EffectStatus.CONFIRMED_READBACK


def test_error_marker_outranks_echo():
    # an error blob that quotes the offending argument back is still a
    # failure, not a confirmation.
    err = ('{"error": "validation-error", '
           '"field": "summary", "value": "Eviction: 10-day appeal window ends"}')
    report = verify_effects([CAL_CALL], [err])
    assert report.entries[0].status == EffectStatus.FAILED


def test_short_argument_values_cannot_confirm():
    call = {"name": "set_flag", "arguments": {"confirm": "True", "n": "5"}}
    report = verify_effects([call], ['{"confirm": "True", "n": 5}'])
    assert report.entries[0].status == EffectStatus.UNAUDITED
    assert "no distinctive argument values" in report.entries[0].evidence


def test_none_result_books_unaudited_unless_readback():
    report = verify_effects([ZONE_CALL], [None])
    assert report.entries[0].status == EffectStatus.UNAUDITED

    later = '{"running": "78b0113a-ea50-4f1e-97a8-8c596b31f16a"}'
    report = verify_effects([ZONE_CALL, {"name": "poll", "arguments": {}}],
                            [None, later])
    assert report.entries[0].status == EffectStatus.CONFIRMED_READBACK


def test_deterministic_same_inputs_same_report():
    calls = [CAL_CALL, ZONE_CALL]
    results = ["No approval received.", '""']
    a = verify_effects(calls, results, "sorry, that failed")
    b = verify_effects(calls, results, "sorry, that failed")
    assert a.to_dict() == b.to_dict()


def test_render_audiences():
    report = verify_effects([CAL_CALL], ["No approval received."], "it failed")
    assert "failed" in report.render("engineer")
    assert "failed" in report.render("executive")
    with pytest.raises(ValueError):
        report.render("model")
