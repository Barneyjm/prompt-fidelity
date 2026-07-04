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
from dataclasses import dataclass, field
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
    """

    id: str
    description: str
    account: str  # verified | transmitted | substituted | inferred | dropped
    bits: float | None
    evidence: str
    source: str = "declared"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "account": self.account,
            "bits": round(self.bits, 2) if self.bits is not None else None,
            "evidence": self.evidence,
            "source": self.source,
        }


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
        """Fraction of user intent booked verified. 1.0 when there is no
        intent to measure (total_bits == 0), matching the "nothing asked,
        nothing to fail" convention used throughout this package."""
        total = self.total_bits
        return self.verified_bits / total if total else 1.0

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

    def to_dict(self) -> dict:
        """Emit the dev.promptfidelity/v1 ledger shape."""
        accounts = {a: round(self._sum(a), 2) for a in ACCOUNTS}
        d = {
            "v": self.v,
            "entries": [e.to_dict() for e in self.entries],
            "imposed": [e.to_dict() for e in self.imposed],
            "summary": {
                "fidelity": round(self.fidelity, 3),
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

        if len(present) == len(params) and all(args[k] == v for k, v in params.items()):
            if any(k in advisory for k in params):
                entries.append(LedgerEntry(
                    c.id, c.description, "transmitted", b,
                    f"faithfully passed via advisory param(s) "
                    f"{sorted(k for k in params if k in advisory)}; "
                    f"backend does not enforce compliance", source=c.source))
            else:
                entries.append(LedgerEntry(
                    c.id, c.description, "verified", b,
                    f"arguments match declared params {sorted(params)}", source=c.source))
        elif present and any(args[k] != v for k, v in present.items()):
            diffs = {k: {"declared": v, "actual": args[k]}
                     for k, v in present.items() if args[k] != v}
            entries.append(LedgerEntry(
                c.id, c.description, "substituted", b,
                f"param value(s) altered: {diffs}", source=c.source))
        elif present:
            missing = sorted(set(params) - set(present))
            entries.append(LedgerEntry(
                c.id, c.description, "substituted", b,
                f"partial application; missing {missing}", source=c.source))
        else:
            entries.append(LedgerEntry(
                c.id, c.description, "dropped", b,
                f"declared params {sorted(params)} absent from call", source=c.source))

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
    """
    if not ledgers:
        return book(constraints, {})

    best: dict[str, LedgerEntry] = {}
    for ledger in ledgers:
        for e in ledger.entries:
            cur = best.get(e.id)
            if cur is None or _ACCOUNT_RANK[e.account] > _ACCOUNT_RANK[cur.account]:
                best[e.id] = e

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
