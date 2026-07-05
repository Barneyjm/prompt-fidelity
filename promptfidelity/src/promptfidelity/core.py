"""
promptfidelity.core -- mechanical booking of every declared constraint's fate.

Books each constraint into exactly one account by diffing its declared
params against the ACTUAL arguments a tool call was made with -- no LLM
self-reporting in the loop. This is the reference implementation of the
`dev.promptfidelity/v1` booking rules (see the interchange format spec).

Accounts (user-side):
    VERIFIED    -- every declared (name, value) pair is present in the
                   actual call arguments with a matching value, via
                   ENFORCED params only
    TRANSMITTED -- params present and matching, but via an advisory param
                   the backend does not enforce (e.g. a search engine's
                   free-text query string): faithfully delivered,
                   compliance not guaranteed
    SUBSTITUTED -- a declared param name is present but with an altered
                   value, or only some names of a multi-param constraint
                   made it into the call (partial application)
    INFERRED    -- the constraint declared no params (not expressible
                   against this tool)
    DROPPED     -- params were declared but none of their names appear in
                   the call arguments

Accounts (agent-side):
    IMPOSED     -- an argument present in the call that matches NO declared
                   constraint (agent-added filtering: popularity floors,
                   truncation, score cutoffs, ...)

Conservation: every constraint is booked exactly once, into one of the five
user-side accounts.
    I_total = I_verified + I_transmitted + I_substituted + I_inferred + I_dropped

HARD RULE: LLMs may propose ledger ENTRIES (Constraints -- a description, a
params dict, a survival probability), never ACCOUNTS. Account assignment is
always computed here, by book(), from a mechanical diff against the actual
tool-call arguments. No function anywhere in this package accepts a
pre-assigned account from model output; book() is the only place an account
is ever set.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any

MAX_BITS = 20.0

ACCOUNTS = ("verified", "transmitted", "substituted", "inferred", "dropped")
_ACCOUNT_RANK = {"verified": 4, "transmitted": 3, "substituted": 2, "dropped": 1, "inferred": 0}


def bits(p: float | None) -> float | None:
    """Convert a survival probability to information content, -log2(p).

    Returns None when `p` is None: the caller made no confidence estimate,
    so only constraint *counts* are reliable, not weighted bits.
    """
    if p is None:
        return None
    if p <= 0:
        return MAX_BITS
    if p >= 1:
        return 0.0
    return min(-math.log2(p), MAX_BITS)


@dataclass
class Constraint:
    """One piece of user intent, decomposed to the params it should produce.

    id: stable identifier for this constraint within a prompt/run.
    description: human-readable statement of the intent.
    params: the tool-call arguments this constraint expects to produce,
        {name: value}. Empty/omitted means the constraint is not
        expressible against this tool -- it can only be inferred or
        handled elsewhere (e.g. by a reranker).
    p: estimated survival probability in (0, 1]. None means "no estimate";
        bits book as None and only constraint counts are reliable.
    source: provenance of this constraint -- "declared" (a human/caller
        wrote it directly), "rules" (promptfidelity.extraction's
        deterministic Tier-1 regex/vocab extractor), or "llm" (a
        pluggable model-backed extractor). Every constraint that flows
        through book() carries this tag through to the Ledger's receipt
        (Ledger.to_dict()["summary"]["constraints_source"]), so a reader
        can see how much of a measurement rests on LLM transcription vs.
        deterministic rules vs. a human declaring intent directly. Source
        is metadata about *where the entry came from*, never a judgment
        about its correctness -- it plays no role in book()'s diff.

    A Constraint never carries an account -- see the HARD RULE in this
    module's docstring. Only book() decides where a constraint lands.
    """

    id: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    p: float | None = None
    source: str = "declared"

    @property
    def bits(self) -> float | None:
        """Information content of this constraint, -log2(p)."""
        return bits(self.p)


@dataclass
class LedgerEntry:
    """One booked constraint: exactly one of the five user-side accounts,
    mechanically assigned by book(), plus the evidence for the assignment.

    source carries forward the Constraint's provenance tag ("declared" |
    "rules" | "llm") -- see Constraint.source. book() only reads it through
    to the entry; it never affects account assignment.

    params: the constraint's own declared {name: value} pairs, copied
    through from Constraint.params by book(). This is what a repair loop
    needs to know WHAT to send to honor an unhonored entry -- see
    report.render()'s "model" audience, which reads this field directly.
    Empty for an `inferred` entry (the constraint declared no params).

    call: which recorded call (0-indexed into the list of per-call Ledgers)
    this booking came from. None for a bare, single-call book() -- there is
    only one call, so the index is meaningless -- and left None by book()
    itself. merge_ledgers() is the only place this is ever set, to the
    index of the ledger the winning booking came from: see
    Ledger.conjunction_honored for why that index matters.
    """

    id: str
    description: str
    account: str  # verified | transmitted | substituted | inferred | dropped
    bits: float | None
    evidence: str
    source: str = "declared"
    params: dict[str, Any] = field(default_factory=dict)
    call: int | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "description": self.description,
            "account": self.account,
            "bits": round(self.bits, 2) if self.bits is not None else None,
            "evidence": self.evidence,
            "source": self.source,
        }
        if self.params:
            d["params"] = dict(self.params)
        if self.call is not None:
            d["call"] = self.call
        return d


@dataclass
class ImposedEntry:
    """One agent-side filter applied that maps to no declared constraint."""

    description: str  # the unclaimed argument name
    value: Any
    bits: float | None
    evidence: str

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "value": self.value,
            "bits": round(self.bits, 2) if self.bits is not None else None,
            "evidence": self.evidence,
        }


@dataclass
class Ledger:
    """The booked fate of every constraint in one prompt/run.

    Construct via book() or merge_ledgers() -- never by hand-assigning
    accounts to entries.
    """

    entries: list[LedgerEntry] = field(default_factory=list)
    imposed: list[ImposedEntry] = field(default_factory=list)
    prompt_id: str | None = None
    v: int = 1

    def _sum(self, account: str) -> float:
        return sum(e.bits or 0.0 for e in self.entries if e.account == account)

    @property
    def verified_bits(self) -> float:
        return self._sum("verified")

    @property
    def transmitted_bits(self) -> float:
        return self._sum("transmitted")

    @property
    def substituted_bits(self) -> float:
        return self._sum("substituted")

    @property
    def inferred_bits(self) -> float:
        return self._sum("inferred")

    @property
    def dropped_bits(self) -> float:
        return self._sum("dropped")

    @property
    def imposed_bits(self) -> float:
        return sum(e.bits or 0.0 for e in self.imposed)

    @property
    def total_bits(self) -> float:
        """I_total: sum of the five user-side accounts (imposed excluded --
        it is agent-side and has no place in the user-intent denominator)."""
        return sum(self._sum(a) for a in ACCOUNTS)

    @property
    def fidelity(self) -> float:
        """Fraction of user intent booked verified.

        Bits-weighted when any entry carries bits; falls back to a plain
        entry-count ratio when every entry has bits=None (e.g. all
        constraints came from rules extraction, which never estimates p) --
        a run with substituted/dropped entries must not report 1.0 just
        because nothing was bit-weighted. to_dict() discloses which basis
        was used as summary["fidelity_basis"]. 1.0 when there are no
        entries at all ("nothing asked, nothing to fail")."""
        total = self.total_bits
        if total:
            return self.verified_bits / total
        return self.fidelity_by_count

    @property
    def fidelity_by_count(self) -> float:
        """Fraction of entries booked verified, ignoring bit weights.
        1.0 when there are no entries."""
        if not self.entries:
            return 1.0
        verified = sum(1 for e in self.entries if e.account == "verified")
        return verified / len(self.entries)

    @property
    def fidelity_basis(self) -> str:
        """Which denominator fidelity used: "bits" (some entry carried a
        survival estimate) or "counts" (no entry did)."""
        return "bits" if self.total_bits else "counts"

    @property
    def transmission_rate(self) -> float:
        """Fraction of user intent booked transmitted -- the metric that
        matters most for relevance-heavy tools (e.g. free-text search),
        where most intent legitimately lives in the transmitted account."""
        total = self.total_bits
        return self.transmitted_bits / total if total else 0.0

    @property
    def constraints_source(self) -> dict[str, int]:
        """Count of booked entries per provenance tag ("declared" | "rules"
        | "llm") -- the receipt discloses how much of this measurement
        rests on LLM transcription vs. deterministic rules vs. a human
        declaring intent directly. E.g. {"declared": 2, "rules": 3, "llm": 1}."""
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.source] = counts.get(e.source, 0) + 1
        return counts

    @property
    def unhonored(self) -> list[LedgerEntry]:
        """Entries booked `substituted` or `dropped` -- the mechanically
        REPAIRABLE ones.

        `inferred` is not repairable via a tool call at all: the constraint
        declared no params, so there is nothing to add to a call's
        arguments -- it's disclosable (say so in the response), not fixable
        by re-issuing a call. `imposed` isn't a constraint booking in the
        first place; it's an agent-side filter with no declared intent
        behind it. Only `substituted` and `dropped` name a concrete gap
        between what was declared and what a call actually carried, which a
        corrective call can close. This is the list a repair loop iterates
        over -- see recorder.Recorder.check() and report.render()'s "model"
        audience, which is built entirely from this property.
        """
        return [e for e in self.entries if e.account in ("substituted", "dropped")]

    @property
    def conjunction_honored(self) -> bool:
        """True iff every entry booked `verified` or `transmitted` came
        from the SAME call.

        This is a different question from "was each constraint ever
        honored" (which is what merge_ledgers()/book() answer, credit by
        credit): it's "did one single call honor them all TOGETHER". A
        `call` index is only ever set by merge_ledgers() (book() leaves it
        None -- a bare single-call ledger has nothing to disagree about).
        So:
          - a ledger booked by a bare book() call: nothing has a `call`
            index at all -> vacuously True.
          - a ledger produced by merge_ledgers() where every verified/
            transmitted entry's winning booking traces back to the same
            call index -> True: one real tool call, with one real result
            set, actually satisfied every constraint that ledger credits.
          - a ledger where those entries trace back to DIFFERENT call
            indices -> False: the intent was honored piecewise, across
            calls that never coexisted -- e.g. call 0 verified the genre
            filter and call 1 (separately) verified the rating filter, but
            no single call, and therefore no single result set, ever
            satisfied both together.

        Why this matters for a repair loop: a naive repair that patches
        just the missing param (a small, cheap follow-up call) can turn a
        `dropped` entry into `verified` -- but if that follow-up call is
        itself missing something an earlier call had, the ledger now shows
        every constraint verified-across-calls while conjunction_honored is
        False. That's a real gap: nothing ever received a request carrying
        the full, honored set of arguments, so no single result set can be
        trusted to reflect the whole intent. The honest repair is not "add
        the missing param to a fresh minimal call" -- it's "re-issue ONE
        complete call carrying every previously-honored param plus the
        missing one" (see report.render()'s "model" audience, which spells
        out that instruction).

        True (vacuously) when there are no verified/transmitted entries, or
        when none of them carry a `call` index (a plain book()-only run).
        """
        calls = {e.call for e in self.entries if e.account in ("verified", "transmitted")}
        calls.discard(None)
        return len(calls) <= 1

    def render(self, audience: str) -> str:
        """Delegates to report.render(self, audience) -- see that module
        for the four audiences and their contracts. Imported locally to
        avoid a circular import (report.py imports Ledger from this
        module)."""
        from .report import render
        return render(self, audience)

    def to_dict(self) -> dict:
        """Emit the dev.promptfidelity/v1 ledger shape."""
        accounts = {a: round(self._sum(a), 2) for a in ACCOUNTS}
        d = {
            "v": self.v,
            "entries": [e.to_dict() for e in self.entries],
            "imposed": [e.to_dict() for e in self.imposed],
            "summary": {
                "fidelity": round(self.fidelity, 3),
                "fidelity_basis": self.fidelity_basis,
                "verified_bits": accounts["verified"],
                "total_bits": round(self.total_bits, 2),
                "accounts": accounts,
                "constraints_source": self.constraints_source,
            },
        }
        if self.prompt_id:
            d["prompt_id"] = self.prompt_id
        return d


def book(
    constraints: list[Constraint],
    arguments: dict[str, Any],
    advisory_params: set | None = None,
) -> Ledger:
    """Book every constraint against actual tool-call arguments. Pure function.

    See this module's docstring for the account rules. In summary:
        - no params declared                              -> inferred
        - all declared params present & matching           -> verified
          (or transmitted, if any matched param is advisory)
        - some params present but a value differs, or only
          some of a multi-param constraint's names present -> substituted
        - params declared, none of the names present       -> dropped
        - any call argument matching no declared constraint -> imposed

    advisory_params: argument names transmitted to the backend but not
    enforced by it (e.g. a search engine's free-text query). A constraint
    matched only via advisory params books as `transmitted`, never
    `verified`.
    """
    advisory = advisory_params or set()
    args = {k: str(v) for k, v in (arguments or {}).items()}
    claimed_names: set = set()
    entries: list[LedgerEntry] = []

    for c in constraints:
        b = c.bits
        params = {k: str(v) for k, v in (c.params or {}).items()}

        if not params:
            entries.append(LedgerEntry(
                c.id, c.description, "inferred", b,
                "no params expressible against this tool", source=c.source))
            continue

        claimed_names |= params.keys()
        present = {k: v for k, v in params.items() if k in args}
        # Keep the constraint's ORIGINAL (unstringified) params on the entry
        # -- this is what a repair loop should send verbatim, not the
        # str()-coerced copy used only for the diff above.
        orig_params = dict(c.params or {})

        if len(present) == len(params) and all(args[k] == v for k, v in params.items()):
            if any(k in advisory for k in params):
                entries.append(LedgerEntry(
                    c.id, c.description, "transmitted", b,
                    f"faithfully passed via advisory param(s) "
                    f"{sorted(k for k in params if k in advisory)}; "
                    f"backend does not enforce compliance", source=c.source,
                    params=orig_params))
            else:
                entries.append(LedgerEntry(
                    c.id, c.description, "verified", b,
                    f"arguments match declared params {sorted(params)}", source=c.source,
                    params=orig_params))
        elif present and any(args[k] != v for k, v in present.items()):
            diffs = {k: {"declared": v, "actual": args[k]}
                     for k, v in present.items() if args[k] != v}
            entries.append(LedgerEntry(
                c.id, c.description, "substituted", b,
                f"param value(s) altered: {diffs}", source=c.source,
                params=orig_params))
        elif present:
            missing = sorted(set(params) - set(present))
            entries.append(LedgerEntry(
                c.id, c.description, "substituted", b,
                f"partial application; missing {missing}", source=c.source,
                params=orig_params))
        else:
            entries.append(LedgerEntry(
                c.id, c.description, "dropped", b,
                f"declared params {sorted(params)} absent from call", source=c.source,
                params=orig_params))

    imposed = [
        ImposedEntry(k, v, None, "argument present in call but matches no declared constraint")
        for k, v in args.items() if k not in claimed_names
    ]

    return Ledger(entries=entries, imposed=imposed)


def merge_ledgers(constraints: list[Constraint], ledgers: list[Ledger]) -> Ledger:
    """Merge several per-call Ledgers (all booked against the same
    `constraints`) into one: the best booking per constraint across calls.

    Rank: verified > transmitted > substituted > dropped > inferred. A
    constraint that a *later* call satisfies is not penalized for having
    been dropped or substituted by an earlier one -- the question this
    answers is "was this piece of intent ever honored", across the whole
    run. (`inferred` never varies across calls: it's assigned the moment a
    constraint declares no params, before any argument diff happens.)

    Used whenever one prompt's intent is spread across multiple tool calls:
    promptfidelity.recorder.Recorder.ledger() and
    promptfidelity.anthropic_ext.record().

    Sets each winning entry's `call` to the index (into `ledgers`) it was
    booked from -- via dataclasses.replace(), never by mutating the input
    ledgers' own entries, which callers may still hold references to. This
    is what Ledger.conjunction_honored reads to tell "honored, but only
    piecewise across separate calls" apart from "one call honored these
    together" -- see that property's docstring.
    """
    if not ledgers:
        return book(constraints, {})

    best: dict[str, LedgerEntry] = {}
    for i, ledger in enumerate(ledgers):
        for e in ledger.entries:
            cur = best.get(e.id)
            if cur is None or _ACCOUNT_RANK[e.account] > _ACCOUNT_RANK[cur.account]:
                best[e.id] = replace(e, call=i)

    order = [c.id for c in constraints]
    merged_entries = [best[cid] for cid in order if cid in best]

    seen = set()
    merged_imposed: list[ImposedEntry] = []
    for ledger in ledgers:
        for imp in ledger.imposed:
            key = (imp.description, imp.value)
            if key not in seen:
                seen.add(key)
                merged_imposed.append(imp)

    return Ledger(entries=merged_entries, imposed=merged_imposed)
