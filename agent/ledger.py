"""
Intent Ledger: mechanical booking of every constraint's fate.

Books each decomposed constraint into exactly one account by diffing
the decomposition against ACTUAL tool-call parameters -- no LLM
self-reporting in the loop.

Accounts (user-side):
    VERIFIED    -- constraint's api_param(s) appear in the params actually
                   sent, with matching values, via ENFORCED params only
    TRANSMITTED -- params present and matching, but via an advisory param the
                   backend does not enforce (e.g. a search engine query
                   string): faithfully delivered, compliance not guaranteed
    SUBSTITUTED -- constraint was demoted/altered en route (failed ID lookup
                   with a resolution_note, a param value changed, or only
                   some of a multi-param constraint applied)
    INFERRED    -- constraint classified inferred AND handed to the reranker
    DROPPED     -- constraint appears in no tool call and no rerank criteria

Accounts (agent-side):
    IMPOSED     -- filtering the agent applied that maps to NO user constraint
                   (popularity floors, truncation, score cutoffs)

Conservation: every user constraint is booked exactly once.
    I_total = I_verified + I_transmitted + I_substituted + I_inferred + I_dropped
"""

import math
from dataclasses import dataclass, field


MAX_BITS = 20.0


def bits(survival_rate: float) -> float:
    if survival_rate <= 0:
        return MAX_BITS
    if survival_rate >= 1:
        return 0.0
    return min(-math.log2(survival_rate), MAX_BITS)


@dataclass
class LedgerEntry:
    description: str
    account: str            # verified | transmitted | substituted | inferred | dropped | imposed
    bits: float
    evidence: str           # the mechanical reason for the booking


@dataclass
class IntentLedger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def _sum(self, account: str) -> float:
        return sum(e.bits for e in self.entries if e.account == account)

    @property
    def user_total(self) -> float:
        return sum(e.bits for e in self.entries if e.account != "imposed")

    @property
    def fidelity(self) -> float:
        return self._sum("verified") / self.user_total if self.user_total else 1.0

    @property
    def transmitted_rate(self) -> float:
        return self._sum("transmitted") / self.user_total if self.user_total else 0.0

    @property
    def substitution_rate(self) -> float:
        return self._sum("substituted") / self.user_total if self.user_total else 0.0

    @property
    def drop_rate(self) -> float:
        return self._sum("dropped") / self.user_total if self.user_total else 0.0

    @property
    def imposed_bits(self) -> float:
        return self._sum("imposed")

    def honesty_gap(self, narrated_verified_bits: float) -> float:
        """Bits the narration claims as verified minus bits actually verified.
        Positive = the agent overstated its grounding."""
        return narrated_verified_bits - self._sum("verified")

    def to_dict(self) -> dict:
        return {
            "fidelity": round(self.fidelity, 3),
            "transmitted_rate": round(self.transmitted_rate, 3),
            "substitution_rate": round(self.substitution_rate, 3),
            "drop_rate": round(self.drop_rate, 3),
            "imposed_bits": round(self.imposed_bits, 2),
            "accounts": {
                a: round(self._sum(a), 2)
                for a in ("verified", "transmitted", "substituted", "inferred", "dropped", "imposed")
            },
            "entries": [
                {"description": e.description, "account": e.account,
                 "bits": round(e.bits, 2), "evidence": e.evidence}
                for e in self.entries
            ],
        }


def _constraint_params(c: dict) -> list[tuple[str, str]]:
    """All (param, value) pairs a constraint claims it will send."""
    pairs = []
    if "api_params" in c:
        for p in c["api_params"]:
            if p.get("param") is not None:
                pairs.append((p["param"], str(p.get("value"))))
    elif c.get("api_param"):
        pairs.append((c["api_param"], str(c.get("api_value"))))
    return pairs


def book_ledger(
    all_constraints: list[dict],
    actual_params: dict,
    rerank_criteria_descriptions: list[str],
    imposed_operations: list[dict] | None = None,
    advisory_params: set[str] | None = None,
) -> IntentLedger:
    """
    Book every constraint mechanically.

    Args:
        all_constraints: post-resolution constraints (verified + inferred),
            i.e. resolved_constraints + original inferred ones
        actual_params: the params dict ACTUALLY passed to the TMDb call
        rerank_criteria_descriptions: descriptions actually sent to reranker
        imposed_operations: agent-added filters, e.g.
            [{"description": "min_votes floor", "survival_rate": 0.35,
              "evidence": "discover_movies(min_votes=50)"}]
        advisory_params: param names the backend transmits but does not
            enforce (e.g. a free-text search query). Constraints applied
            via any advisory param book as `transmitted`, not `verified`.
    """
    ledger = IntentLedger()
    actual = {k: str(v) for k, v in actual_params.items()}
    rerank_set = set(rerank_criteria_descriptions)
    advisory = advisory_params or set()

    for c in all_constraints:
        b = bits(c.get("estimated_survival_rate", 1.0))
        desc = c.get("description", "?")

        # Substitution: demoted en route (failed lookup etc.)
        if c.get("resolution_note"):
            ledger.entries.append(LedgerEntry(
                desc, "substituted", b,
                f"demoted verified->inferred: {c['resolution_note']}"))
            continue

        pairs = _constraint_params(c)
        if c.get("type") == "verified" and pairs:
            # Verified iff EVERY claimed param made it into the actual call.
            # Comma-joined multi-values count if the value is a member.
            def matches(param, value):
                return param in actual and value in actual[param].split(",")
            sent = [(p, v) for p, v in pairs if p in actual]
            if all(matches(p, v) for p, v in pairs):
                if any(p in advisory for p, _ in pairs):
                    ledger.entries.append(LedgerEntry(
                        desc, "transmitted", b,
                        f"faithfully passed via advisory param(s) "
                        f"{sorted(p for p, _ in pairs if p in advisory)}; "
                        f"backend does not enforce compliance"))
                else:
                    ledger.entries.append(LedgerEntry(
                        desc, "verified", b,
                        f"params present in tool call: {pairs}"))
            elif sent and any(not matches(p, v) for p, v in sent):
                diffs = {p: {"declared": v, "actual": actual[p]}
                         for p, v in sent if not matches(p, v)}
                ledger.entries.append(LedgerEntry(
                    desc, "substituted", b,
                    f"param value(s) altered en route: {diffs}"))
            elif sent:
                missing = sorted(p for p, _ in pairs if p not in actual)
                ledger.entries.append(LedgerEntry(
                    desc, "substituted", b,
                    f"partial application; missing param(s): {missing}"))
            else:
                ledger.entries.append(LedgerEntry(
                    desc, "dropped", b,
                    f"classified verified but params absent from tool call: {pairs}"))
        elif c.get("type") == "inferred":
            if desc in rerank_set:
                ledger.entries.append(LedgerEntry(
                    desc, "inferred", b, "handed to reranker"))
            else:
                ledger.entries.append(LedgerEntry(
                    desc, "dropped", b, "never queried, never reranked"))
        else:
            ledger.entries.append(LedgerEntry(
                desc, "dropped", b, "verified-classified but no api_param produced"))

    for op in (imposed_operations or []):
        ledger.entries.append(LedgerEntry(
            op["description"], "imposed",
            bits(op.get("survival_rate", 1.0)),
            op.get("evidence", "agent-added filter")))

    return ledger
