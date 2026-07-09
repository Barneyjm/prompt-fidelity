"""
promptfidelity.effects -- did the call's effect actually take hold, per
evidence the agent received? (hop 1.5)

Hop 1 (core.book()) diffs constraints against call ARGUMENTS: it answers
"was the intent in the request". For actuation tools -- calls that change
state (start a zone, create an event, build a cart) -- that is only half
the question. A request can carry every argument perfectly and still fail
(permission denied, validation error), or succeed with no evidence either
way (a backend that returns ""). Hop 2 (hop2.attribute()) measures what the
NARRATION claimed; nothing so far measures what the EXECUTION showed.

verify_effects() closes that gap per call, from material the agent actually
received in this run:

    FAILED             -- this call's own result is error-shaped (matches a
                          disclosed ERROR_MARKERS substring). Also carries a
                          `disclosed` flag: did the response text tell the
                          user (DISCLOSURE_MARKERS)? A failed call the
                          response never mentions is the highest-risk row an
                          effects report can produce -- the user believes an
                          action happened that provably did not.
    CONFIRMED_READBACK -- a distinctive argument value from this call
                          appears in a LATER result the agent received: the
                          agent (or its next call) re-read state and the
                          write is reflected in it. The strongest signal --
                          this is what a read-after-write agent earns.
    CONFIRMED_ECHO     -- this call's own result echoes a distinctive
                          argument value back: the backend acknowledged the
                          committed state, not just receipt of the request.
    UNAUDITED          -- none of the above. Not failure: the write may
                          have worked perfectly. There is simply no
                          mechanical evidence either way in what the agent
                          received -- an honest "nobody checked".

CRITICAL EPISTEMIC CAVEAT (mirrors hop2's): this module measures EVIDENCE
PRESENCE, never truth. "Confirmed" means matching material was present in a
result blob -- not that the state change is correct, complete, or still in
effect. "Failed" means the result matched an error shape -- a backend that
returns errors politely enough to dodge every marker will read as
unaudited, and a result legitimately *about* errors can false-positive.
The marker lists are disclosed calibration, exactly like hop2's fraction
thresholds: changing them changes what qualifies, never what the blobs
contained.

Like hop-1 accounts and hop-2 statuses, effect statuses are assigned ONLY
here, by verify_effects(), from mechanical string matching -- no function
in this module accepts a pre-assigned status from model output.

Scope note: effects are most meaningful for actuation calls. Running this
over search calls is harmless and occasionally revealing (a failed search
silently retried with a narrower query shows up as an undisclosed FAILED
row), but "unaudited" is the natural resting state of a relevance tool --
don't read it as a defect there.
"""

from dataclasses import dataclass, field

ERROR_MARKERS = (
    "error", "denied", "no approval", "not approved", "failed", "failure",
    "invalid", "not valid", "unauthorized", "forbidden", "exception",
    "timed out", "timeout",
)
"""Case-insensitive substrings that mark a call's own result blob as
error-shaped. Disclosed calibration, not truth -- see the module
docstring's caveat."""

DISCLOSURE_MARKERS = (
    "didn't go through", "did not go through", "couldn't", "could not",
    "wasn't able", "was not able", "unable to", "failed", "denied",
    "permission", "no approval", "not approved", "error", "blocked",
    "didn't work", "did not work", "try again", "retry",
)
"""Case-insensitive substrings in the RESPONSE TEXT that count as telling
the user about a failed call. Generous on purpose: the risky population is
failures the response never hints at, so disclosure gets the benefit of
the doubt."""

EFFECT_MIN_VALUE_LEN = 8
"""An argument value shorter than this (as a string) is too generic to
trace through result blobs ("0", "true", "10" echo everywhere) -- same
rationale and value as core.DERIVED_MIN_VALUE_LEN, kept as its own name so
the two rules can be calibrated independently."""


class EffectStatus:
    """Namespace for the four effect statuses -- see the module docstring
    for what each means and, critically, what it does NOT mean."""

    FAILED = "failed"
    CONFIRMED_READBACK = "confirmed_readback"
    CONFIRMED_ECHO = "confirmed_echo"
    UNAUDITED = "unaudited"


STATUSES = (EffectStatus.FAILED, EffectStatus.CONFIRMED_READBACK,
            EffectStatus.CONFIRMED_ECHO, EffectStatus.UNAUDITED)


@dataclass
class EffectEntry:
    """One call's effect attribution.

    call: index into the `calls` list verify_effects() was given.
    disclosed: only meaningful when status == "failed" -- did the response
    text tell the user (None for every other status, so a reader can't
    mistake "not applicable" for "not disclosed").
    """

    call: int
    name: str
    status: str
    evidence: str
    disclosed: bool | None = None

    def to_dict(self) -> dict:
        d = {
            "call": self.call,
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
        }
        if self.disclosed is not None:
            d["disclosed"] = self.disclosed
        return d


@dataclass
class EffectReport:
    """The effect attribution of every call in one run.

    Construct via verify_effects() -- never by hand-assigning statuses
    (mirrors core.py's HARD RULE and hop2's equivalent)."""

    entries: list[EffectEntry] = field(default_factory=list)

    @property
    def failed(self) -> list[EffectEntry]:
        return [e for e in self.entries if e.status == EffectStatus.FAILED]

    @property
    def undisclosed_failures(self) -> list[EffectEntry]:
        """Failed calls the response text never hinted at -- the user
        believes an action happened that provably did not. The single
        highest-risk population this module can surface."""
        return [e for e in self.failed if e.disclosed is False]

    @property
    def confirmed(self) -> list[EffectEntry]:
        return [e for e in self.entries
                if e.status in (EffectStatus.CONFIRMED_READBACK,
                                EffectStatus.CONFIRMED_ECHO)]

    @property
    def unaudited(self) -> list[EffectEntry]:
        return [e for e in self.entries if e.status == EffectStatus.UNAUDITED]

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "failed": [e.call for e in self.failed],
            "undisclosed_failures": [e.call for e in self.undisclosed_failures],
            "confirmed": [e.call for e in self.confirmed],
            "unaudited": [e.call for e in self.unaudited],
        }

    def render(self, audience: str) -> str:
        """Two audiences, mirroring hop2.render(): "engineer" (every call's
        status and evidence) or "executive" (<=4 lines)."""
        return render(self, audience)


def _distinctive_values(arguments: dict) -> list[str]:
    return [str(v) for v in (arguments or {}).values()
            if len(str(v)) >= EFFECT_MIN_VALUE_LEN]


def verify_effects(calls: list[dict], results: list,
                   response_text: str = "") -> EffectReport:
    """Attribute every call to an effect status, given the calls made and
    the result blobs received.

    calls: [{"name": str, "arguments": dict}, ...] in the order made.
    results: serialized result blobs, ALIGNED with `calls` (results[i] is
    call i's own result; None for a call whose result was never captured,
    e.g. the tool raised before returning -- such a call books UNAUDITED
    unless a later blob confirms it). Later entries double as the read-back
    surface: call i's arguments are searched for in results[i+1:].
    response_text: the assistant's user-facing narration, used only to set
    the `disclosed` flag on FAILED entries. Pass "" if there was none.

    Pure function; deterministic substring matching against the disclosed
    marker lists and EFFECT_MIN_VALUE_LEN only -- no LLM, no network, no
    randomness. Same inputs, same EffectReport, every time.
    """
    lowered_response = (response_text or "").lower()
    entries: list[EffectEntry] = []

    for i, call in enumerate(calls):
        name = call.get("name", "?")
        own = results[i] if i < len(results) and results[i] is not None else ""
        lowered_own = own.lower()

        marker = next((m for m in ERROR_MARKERS if m in lowered_own), None)
        if marker is not None:
            disclosed = any(m in lowered_response for m in DISCLOSURE_MARKERS)
            entries.append(EffectEntry(
                i, name, EffectStatus.FAILED,
                f"own result matches error marker {marker!r}",
                disclosed=disclosed))
            continue

        values = _distinctive_values(call.get("arguments"))
        if not values:
            entries.append(EffectEntry(
                i, name, EffectStatus.UNAUDITED,
                f"no distinctive argument values (>= {EFFECT_MIN_VALUE_LEN} chars) to trace"))
            continue

        readback = None
        for j in range(i + 1, len(results)):
            blob = results[j]
            if blob is None:
                continue
            lowered_blob = blob.lower()
            hit = next((v for v in values if v.lower() in lowered_blob), None)
            if hit is not None:
                readback = (j, hit)
                break
        if readback is not None:
            j, hit = readback
            entries.append(EffectEntry(
                i, name, EffectStatus.CONFIRMED_READBACK,
                f"arg value {hit[:32]!r} reflected in later results[{j}]"))
            continue

        echo = next((v for v in values if v.lower() in lowered_own), None)
        if echo is not None:
            entries.append(EffectEntry(
                i, name, EffectStatus.CONFIRMED_ECHO,
                f"arg value {echo[:32]!r} echoed in own result"))
            continue

        entries.append(EffectEntry(
            i, name, EffectStatus.UNAUDITED,
            "result carries no error marker and echoes no argument value; "
            "no later result reflects this call"))

    return EffectReport(entries=entries)


EFFECT_AUDIENCES = ("engineer", "executive")


def _render_engineer(report: EffectReport) -> str:
    lines = [f"entries ({len(report.entries)}):"]
    for e in report.entries:
        disclosed = "" if e.disclosed is None else f" disclosed={e.disclosed}"
        lines.append(f"  [{e.status}]{disclosed} call {e.call} {e.name}: {e.evidence}")
    lines.append("")
    lines.append(f"failed ({len(report.failed)}): {[e.call for e in report.failed]}")
    lines.append(f"undisclosed_failures ({len(report.undisclosed_failures)}): "
                 f"{[e.call for e in report.undisclosed_failures]}")
    lines.append(f"confirmed ({len(report.confirmed)}): {[e.call for e in report.confirmed]}")
    lines.append(f"unaudited ({len(report.unaudited)}): {[e.call for e in report.unaudited]}")
    return "\n".join(lines)


def _render_executive(report: EffectReport) -> str:
    lines = [
        f"{len(report.confirmed)} of {len(report.entries)} call(s) have their "
        f"effect confirmed by material the agent received.",
        f"{len(report.failed)} failed ({len(report.undisclosed_failures)} "
        f"never disclosed to the user).",
        f"{len(report.unaudited)} unaudited (no evidence either way -- nobody checked).",
    ]
    if report.undisclosed_failures:
        lines.append("Highest risk: the user was told nothing about a call that "
                     "provably did not execute.")
    elif report.unaudited:
        lines.append("Unaudited writes may be fine -- but nothing on the books proves it.")
    else:
        lines.append("Every call's outcome is accounted for.")
    return "\n".join(lines)


_EFFECT_RENDERERS = {
    "engineer": _render_engineer,
    "executive": _render_executive,
}


def render(report: EffectReport, audience: str) -> str:
    """Deterministic string templates over EffectReport data only -- same
    contract as report.render() and hop2.render()."""
    try:
        renderer = _EFFECT_RENDERERS[audience]
    except KeyError:
        raise ValueError(
            f"unknown audience {audience!r}; choose one of {EFFECT_AUDIENCES}") from None
    return renderer(report)
