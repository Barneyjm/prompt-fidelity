"""
promptfidelity.report -- one instrument, four altitudes.

render(ledger, audience) turns an already-booked Ledger into a deterministic
text block. There is no LLM anywhere in this module: every string is a plain
template filled in from Ledger/LedgerEntry/ImposedEntry fields that book()
already computed. Same ledger in, same string out, every time.

Four audiences, one instrument:
    "model"     -- short, actionable, meant to be INJECTED into an agent's
                   own context mid-run so it can repair itself. Tokens are
                   money here: this is the only audience that omits detail
                   on purpose. See recorder.Recorder.check() for the loop
                   this feeds.
    "engineer"  -- everything: every entry's account/bits/source/call
                   index/evidence, every imposed arg, fidelity + basis,
                   the conjunction flag, how many calls were credited.
    "product"   -- plain language, grouped by what a non-technical reader
                   cares about (delivered as asked / delivered
                   approximately / handled by model judgment / not
                   delivered), no bits, no account jargon in the body text.
    "executive" -- at most ~5 lines: the fidelity headline (with its basis
                   caveat spelled out when it matters), how much was
                   altered/dropped without disclosure, how many filters the
                   agent added unasked, one closing risk line.

No emoji, plain ASCII -- matches the rest of this package.
"""

from .core import Ledger

AUDIENCES = ("model", "engineer", "product", "executive")

_MODEL_REPAIR_INSTRUCTION = (
    "Re-issue ONE complete tool call including all previously-honored "
    "params plus these."
)

_MODEL_CONJUNCTION_WARNING = (
    "CONJUNCTION WARNING: every constraint has been honored, but not by "
    "one single call -- no single result set satisfied them all together. "
    "Re-issue one complete call carrying every previously-honored param at "
    "once."
)


def _distinct_calls(ledger: Ledger) -> int:
    """Number of distinct recorded calls credited with at least one
    verified/transmitted/substituted/dropped entry -- an approximation of
    "how many calls fed this ledger", not an exact count: a call that won
    NO entry (every one of its bookings was beaten by some other call)
    leaves no trace in `entry.call` and isn't counted. Good enough for the
    engineer audience's "how many calls got merged here" context; not a
    substitute for len(Recorder.calls) when the exact count matters."""
    return len({e.call for e in ledger.entries if e.call is not None})


def _render_model(ledger: Ledger) -> str:
    """Short and actionable: only what a repair loop needs. See this
    module's docstring and Ledger.unhonored / Ledger.conjunction_honored
    for the contract this implements.

    Empty string ("" -- nothing to inject) when there is nothing unhonored
    and conjunction holds: the caller's loop should treat "" as "stop,
    nothing left to repair."
    """
    unhonored = ledger.unhonored
    if not unhonored:
        if not ledger.conjunction_honored:
            return _MODEL_CONJUNCTION_WARNING
        return ""

    lines = [f"UNHONORED ({len(unhonored)}):"]
    for e in unhonored:
        send = f" Send: {e.params}" if e.params else ""
        lines.append(f"- {e.id} ({e.description}): {e.account} -- {e.evidence}.{send}")
    lines.append("")
    lines.append(_MODEL_REPAIR_INSTRUCTION)
    return "\n".join(lines)


def _render_engineer(ledger: Ledger) -> str:
    """Full detail: every entry, every imposed arg, the whole summary math."""
    lines = [
        f"fidelity: {ledger.fidelity:.3f} (basis={ledger.fidelity_basis})",
        f"conjunction_honored: {ledger.conjunction_honored}",
        f"calls credited: {_distinct_calls(ledger)}",
        f"constraints_source: {ledger.constraints_source}",
        "",
        f"entries ({len(ledger.entries)}):",
    ]
    for e in ledger.entries:
        lines.append(
            f"  [{e.account}] {e.id} \"{e.description}\" "
            f"bits={e.bits!r} source={e.source} call={e.call!r} "
            f"params={e.params!r} evidence={e.evidence!r}"
        )
    lines.append("")
    lines.append(f"imposed ({len(ledger.imposed)}):")
    if not ledger.imposed:
        lines.append("  (none)")
    for imp in ledger.imposed:
        lines.append(
            f"  {imp.description}={imp.value!r} bits={imp.bits!r} evidence={imp.evidence!r}"
        )
    return "\n".join(lines)


def _render_product(ledger: Ledger) -> str:
    """Plain language, no jargon, no bits: what a non-technical reader
    (product manager, support agent, the end user themselves) cares about.

    Grouped as:
        delivered as asked          -- verified
        delivered approximately     -- substituted (with what changed)
        handled by model judgment   -- inferred + transmitted (no
            mechanical way to pin down compliance, but not ignored either)
        not delivered                -- dropped

    Honesty caveat: "with what changed" quotes book()'s own evidence
    string for substituted entries, which was written for an engineer --
    it may still contain a raw parameter name. This module renders plain
    grouping and labeling, not a full natural-language rewrite of every
    evidence string.
    """
    verified = [e for e in ledger.entries if e.account == "verified"]
    substituted = [e for e in ledger.entries if e.account == "substituted"]
    judgment = [e for e in ledger.entries if e.account in ("inferred", "transmitted")]
    dropped = [e for e in ledger.entries if e.account == "dropped"]

    def _group(heading: str, items: list, describe) -> list[str]:
        block = ["", heading]
        if not items:
            block.append("  (none)")
        else:
            block.extend(f"  - {describe(e)}" for e in items)
        return block

    lines = [f"What the user asked for: {len(ledger.entries)} request(s)."]
    lines += _group("Delivered as asked:", verified, lambda e: e.description)
    lines += _group(
        "Delivered approximately (something changed):", substituted,
        lambda e: f"{e.description} -- {e.evidence}",
    )
    lines += _group(
        "Handled by model/agent judgment (not mechanically checkable):", judgment,
        lambda e: e.description,
    )
    lines += _group("Not delivered:", dropped, lambda e: e.description)

    lines.append("")
    lines.append(
        "Dropped and substituted items are unmet demand -- what users asked "
        "for that the product couldn't or wouldn't do."
    )
    return "\n".join(lines)


def _render_executive(ledger: Ledger) -> str:
    """At most ~5 lines: the headline number, the disclosure gap, the
    unasked-for filtering, one risk line."""
    total = len(ledger.entries)
    altered_or_dropped = sum(1 for e in ledger.entries if e.account in ("substituted", "dropped"))
    imposed_count = len(ledger.imposed)

    fidelity_line = f"Fidelity: {ledger.fidelity * 100:.1f}%"
    if ledger.fidelity_basis == "counts":
        fidelity_line += " (unweighted -- treat as a count ratio)"

    lines = [
        fidelity_line,
        f"{altered_or_dropped} of {total} constraints altered or dropped without disclosure.",
        f"{imposed_count} agent-imposed filter(s) the user never asked for.",
    ]
    if ledger.unhonored or ledger.imposed:
        lines.append("Overclaim risk: narration may promise more than execution delivered.")
    else:
        lines.append("Execution matched stated intent.")
    return "\n".join(lines)


_RENDERERS = {
    "model": _render_model,
    "engineer": _render_engineer,
    "product": _render_product,
    "executive": _render_executive,
}


def render(ledger: Ledger, audience: str) -> str:
    """Render `ledger` as plain text for one of four audiences.

    audience: one of "model", "engineer", "product", "executive" (see
    AUDIENCES and this module's docstring). Raises ValueError for anything
    else.

    Deterministic string templates over Ledger/LedgerEntry/ImposedEntry
    data only -- no LLM call, no randomness, no network access. Calling
    this twice on the same ledger always returns the same string.
    """
    try:
        renderer = _RENDERERS[audience]
    except KeyError:
        raise ValueError(f"unknown audience {audience!r}; choose one of {AUDIENCES}") from None
    return renderer(ledger)
