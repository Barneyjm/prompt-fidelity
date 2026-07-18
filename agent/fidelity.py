"""
Fidelity calculation module for prompt fidelity framework.

Computes the fidelity score based on verified vs inferred constraint bits.
Fidelity = verified_bits / total_bits, representing the fraction of
information that can be reliably satisfied through structured API queries.
"""

import math
from dataclasses import dataclass
from typing import Literal


# Default maximum bits for any single constraint (prevents inf/NaN issues)
# 20 bits ≈ survival rate of ~0.000001 (one in a million, roughly the size
# of the TMDb catalog). For larger candidate pools, pass pool_size to
# compute_fidelity() so the cap scales to log2(pool_size): a constraint
# cannot carry more information than it takes to identify a single row.
MAX_CONSTRAINT_BITS = 20.0


def max_bits_for_pool(pool_size: float | None) -> float:
    """Per-constraint bit cap for a candidate pool of the given size."""
    if pool_size is None:
        return MAX_CONSTRAINT_BITS
    if pool_size <= 1:
        raise ValueError(f"pool_size must be greater than 1, got {pool_size}")
    return math.log2(pool_size)


@dataclass
class Constraint:
    """Represents a single constraint extracted from a user prompt.

    constraint_type:
        - "verified": mechanically checkable against the data source
        - "inferred": requires subjective LLM judgment
        - "injected": a filter the system applied that the user never
          requested (quality floors, top-N truncation, sampling, default
          sort). Excluded from the fidelity score but always reported.
    rate_source: how the survival rate was obtained —
        - "measured": an exact or system-returned count against the data
        - "approximated": system-derived but inexact (planner statistics,
          sampled counts; possibly stale)
        - "estimated": a guess
        None if unknown.
    """

    description: str
    constraint_type: Literal["verified", "inferred", "injected"]
    estimated_survival_rate: float | None = None
    api_param: str | None = None
    api_value: str | None = None
    max_bits: float = MAX_CONSTRAINT_BITS
    rate_source: Literal["measured", "approximated", "estimated"] | None = None

    @property
    def bits(self) -> float:
        """Calculate information content in bits using -log2(survival_rate)."""
        if self.estimated_survival_rate is None:
            return 0.0
        if self.estimated_survival_rate <= 0:
            return self.max_bits  # Cap at max to avoid inf/NaN
        if self.estimated_survival_rate >= 1:
            return 0.0
        return min(-math.log2(self.estimated_survival_rate), self.max_bits)

    def to_dict(self) -> dict:
        """Convert constraint to dictionary representation."""
        result = {
            "description": self.description,
            "type": self.constraint_type,
            "estimated_survival_rate": self.estimated_survival_rate,
            "bits": round(self.bits, 2)
        }
        if self.api_param:
            result["api_param"] = self.api_param
        if self.api_value:
            result["api_value"] = self.api_value
        if self.rate_source:
            result["rate_source"] = self.rate_source
        return result


@dataclass
class FidelityReport:
    """Complete fidelity analysis for a set of constraints.

    verified_joint_survival_rate: measured survival rate of ALL verified
        filters ANDed together. When set, the verified side of the score
        uses the exact joint information -log2(rate) instead of summing
        per-constraint bits (which assumes independence). Per-constraint
        bits remain as attribution.
    """

    constraints: list[Constraint]
    verified_joint_survival_rate: float | None = None

    @property
    def verified_constraints(self) -> list[Constraint]:
        """Return only verified constraints."""
        return [c for c in self.constraints if c.constraint_type == "verified"]

    @property
    def inferred_constraints(self) -> list[Constraint]:
        """Return only inferred constraints."""
        return [c for c in self.constraints if c.constraint_type == "inferred"]

    @property
    def injected_constraints(self) -> list[Constraint]:
        """System-applied filters the user never requested (score-excluded)."""
        return [c for c in self.constraints if c.constraint_type == "injected"]

    @property
    def verified_bits_summed(self) -> float:
        """Verified bits summed per-constraint (assumes independence)."""
        return sum(c.bits for c in self.verified_constraints)

    @property
    def verified_bits(self) -> float:
        """Verified bits used for the score: joint-measured when available."""
        if self.verified_joint_survival_rate is not None:
            rate = self.verified_joint_survival_rate
            if rate <= 0:
                raise ValueError(
                    f"verified_joint_survival_rate must be positive, got {rate}")
            return 0.0 if rate >= 1 else -math.log2(rate)
        return self.verified_bits_summed

    @property
    def inferred_bits(self) -> float:
        """Total bits from inferred constraints."""
        return sum(c.bits for c in self.inferred_constraints)

    @property
    def total_bits(self) -> float:
        """Total information content across scored (non-injected) constraints."""
        return self.verified_bits + self.inferred_bits

    @property
    def fidelity_score(self) -> float:
        """
        Compute fidelity as the ratio of verified bits to total bits.

        Returns 1.0 if there are no constraints (trivially satisfied).
        Returns value in [0, 1] otherwise.
        """
        if self.total_bits == 0:
            return 1.0
        return self.verified_bits / self.total_bits

    def to_dict(self) -> dict:
        """Convert report to dictionary representation."""
        return {
            "fidelity_score": round(self.fidelity_score, 3),
            "verified_bits": round(self.verified_bits, 2),
            "verified_bits_summed": round(self.verified_bits_summed, 2),
            "verified_rate_basis": (
                "joint-measured" if self.verified_joint_survival_rate is not None
                else "summed"),
            "inferred_bits": round(self.inferred_bits, 2),
            "total_bits": round(self.total_bits, 2),
            "num_verified_constraints": len(self.verified_constraints),
            "num_inferred_constraints": len(self.inferred_constraints),
            "num_injected_constraints": len(self.injected_constraints),
            "constraints": [c.to_dict() for c in self.constraints]
        }


def compute_fidelity(constraints: list[dict],
                     pool_size: float | None = None,
                     verified_joint_survival_rate: float | None = None
                     ) -> FidelityReport:
    """
    Compute fidelity score from a list of constraint dictionaries.

    Args:
        constraints: List of constraint dicts with keys:
            - description: str
            - type: "verified", "inferred", or "injected"
            - estimated_survival_rate: float (0, 1); optional for injected
            - rate_source: "measured", "approximated", or "estimated"
              (optional)
            - api_param: str (optional, for verified)
            - api_value: str (optional, for verified)
        pool_size: Number of rows in the candidate pool. Caps each
            constraint at log2(pool_size) bits. Defaults to the
            one-in-a-million cap (20 bits) when omitted.
        verified_joint_survival_rate: measured survival rate of all
            verified filters combined. When given, the score uses the
            exact joint information instead of summed per-constraint
            bits (correlation correction).

    Returns:
        FidelityReport with computed fidelity score and breakdown.
        Injected constraints are reported but excluded from the score.
    """
    max_bits = max_bits_for_pool(pool_size)
    parsed_constraints = []

    for c in constraints:
        constraint = Constraint(
            description=c["description"],
            constraint_type=c["type"],
            estimated_survival_rate=c.get("estimated_survival_rate"),
            api_param=c.get("api_param"),
            api_value=c.get("api_value"),
            max_bits=max_bits,
            rate_source=c.get("rate_source")
        )
        parsed_constraints.append(constraint)

    return FidelityReport(
        constraints=parsed_constraints,
        verified_joint_survival_rate=verified_joint_survival_rate)


def estimate_survival_rate_from_bits(bits: float) -> float:
    """Convert information bits back to survival rate."""
    if bits <= 0:
        return 1.0
    return 2 ** (-bits)


def combine_survival_rates(rates: list[float]) -> float:
    """
    Combine independent survival rates (assuming independence).

    The combined survival rate for independent constraints is the product
    of individual survival rates.
    """
    result = 1.0
    for rate in rates:
        result *= rate
    return result


# Theoretical maximum fidelity calculations

def compute_max_verified_bits(schema: dict) -> float:
    """
    Compute the theoretical maximum verified bits (I_max) for a schema.

    This represents the maximum information that can be verified through
    the API if all fields are used with maximum selectivity.
    """
    total_bits = 0.0

    for field_name, field_info in schema.get("fields", {}).items():
        # Use the minimum reasonable survival rate for each field
        # This represents maximum selectivity
        min_survival = field_info.get("default_survival_rate", 0.1)
        # For fields with ID lookup, they can be even more selective
        if field_info.get("requires_id_lookup"):
            min_survival = min(min_survival, 0.001)
        total_bits += -math.log2(min_survival)

    return total_bits


def compute_fidelity_frontier(verified_bits: float, inferred_bits: float,
                               i_max: float) -> dict:
    """
    Compute where a query falls on the fidelity frontier.

    Args:
        verified_bits: Bits from verified constraints
        inferred_bits: Bits from inferred constraints
        i_max: Maximum possible verified bits for the schema

    Returns:
        Dict with frontier analysis including:
        - observed_fidelity: actual fidelity score
        - max_achievable_fidelity: best possible given total specificity
        - frontier_efficiency: how close to the frontier we are
    """
    total_bits = verified_bits + inferred_bits

    if total_bits == 0:
        return {
            "observed_fidelity": 1.0,
            "max_achievable_fidelity": 1.0,
            "frontier_efficiency": 1.0,
            "analysis": "No constraints specified"
        }

    observed_fidelity = verified_bits / total_bits

    # Maximum achievable fidelity given the total specificity requested
    # is bounded by i_max / total_bits (capped at 1.0)
    max_achievable = min(1.0, i_max / total_bits)

    # How efficiently are we using the verified capacity?
    frontier_efficiency = observed_fidelity / max_achievable if max_achievable > 0 else 0

    return {
        "observed_fidelity": round(observed_fidelity, 3),
        "max_achievable_fidelity": round(max_achievable, 3),
        "frontier_efficiency": round(frontier_efficiency, 3),
        "total_bits_requested": round(total_bits, 2),
        "schema_max_bits": round(i_max, 2)
    }
