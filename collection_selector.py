
"""
Balanced collection selection for Recall v0.10.

Finds a maximum-scoring assignment of distinct images to
collection roles using role-specific semantic scores.
"""

from dataclasses import dataclass
from math import isfinite

from collection_candidates import CollectionCandidate
from collection_planner import CollectionPlan


@dataclass(frozen=True)
class CollectionSelection:
    role: str
    image_id: int
    path: str
    score: float


def select_collection(
    plan: CollectionPlan,
    candidates: list[CollectionCandidate],
    *,
    excluded_ids: set[int] | None = None,
) -> list[CollectionSelection]:
    """
    Assign distinct photos to roles.

    Maximizes the number of filled roles first, then the total
    semantic score. Missing roles are left unfilled.
    """

    excluded = set() if excluded_ids is None else set(excluded_ids)
    roles = [role.name for role in plan.roles]

    # Merge duplicate candidate IDs and keep the best finite score
    # available for each (role, image) pair.
    image_paths = {}
    role_scores = {role: {} for role in roles}

    for candidate in candidates:
        if candidate.image_id in excluded:
            continue

        image_paths.setdefault(candidate.image_id, candidate.path)

        for role in roles:
            score = candidate.role_scores.get(role)

            if score is None or not isfinite(float(score)):
                continue

            existing = role_scores[role].get(candidate.image_id)

            if existing is None or score > existing:
                role_scores[role][candidate.image_id] = float(score)

    # Hungarian assignment with dummy columns for unfilled roles.
    # SciPy is already commonly installed alongside the ML stack.
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    image_ids = sorted(
        {
            image_id
            for scores in role_scores.values()
            for image_id in scores
        }
    )

    if not image_ids:
        return []

    role_count = len(roles)
    image_count = len(image_ids)

    # A large bonus prioritizes filling available roles over
    # maximizing scores for fewer roles.
    bonus = 1_000_000.0
    missing_cost = bonus * 2

    cost = np.full(
        (role_count, image_count + role_count),
        missing_cost,
        dtype=float,
    )

    for row, role in enumerate(roles):
        for col, image_id in enumerate(image_ids):
            score = role_scores[role].get(image_id)

            if score is not None:
                cost[row, col] = -(bonus + score)

        # Every role has its own "unfilled" dummy option.
        cost[row, image_count + row] = 0.0

    rows, cols = linear_sum_assignment(cost)

    selections = []

    for row, col in zip(rows, cols):
        if col >= image_count:
            continue

        role = roles[row]
        image_id = image_ids[col]
        score = role_scores[role].get(image_id)

        if score is None:
            continue

        selections.append(
            CollectionSelection(
                role=role,
                image_id=image_id,
                path=image_paths[image_id],
                score=score,
            )
        )

    # Present selections in the plan's intended role order.
    role_order = {role: index for index, role in enumerate(roles)}

    return sorted(
        selections,
        key=lambda selection: role_order[selection.role],
    )
