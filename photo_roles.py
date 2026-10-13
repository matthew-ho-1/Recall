
"""
Photo role classification for Recall v0.10.

Assigns candidate photos to likely collection roles using
the role-specific semantic scores gathered during retrieval.

These are preliminary relevance estimates, not verified
scene classifications.
"""

from dataclasses import dataclass

from collection_candidates import CollectionCandidate
from collection_planner import CollectionPlan


@dataclass(frozen=True)
class RoleMatch:
    role: str
    score: float
    rank: int


def classify_candidate_roles(
    candidate: CollectionCandidate,
    plan: CollectionPlan,
) -> list[RoleMatch]:
    """
    Return role matches for a candidate, sorted by semantic score.

    Only roles defined in the collection plan are considered.
    """

    matches = []

    for role in plan.roles:
        if role.name not in candidate.role_scores:
            continue

        score = candidate.role_scores[role.name]
        rank = candidate.role_ranks.get(role.name, 0)

        matches.append(
            RoleMatch(
                role=role.name,
                score=float(score),
                rank=int(rank),
            )
        )

    return sorted(
        matches,
        key=lambda match: (-match.score, match.rank, match.role),
    )


def best_candidate_role(
    candidate: CollectionCandidate,
    plan: CollectionPlan,
) -> RoleMatch | None:
    """Return the candidate's highest-scoring role, if any."""

    matches = classify_candidate_roles(candidate, plan)

    return matches[0] if matches else None
