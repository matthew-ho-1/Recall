
"""
Candidate retrieval for Recall v0.10.

Searches the indexed photo library for each collection role,
then merges results by image ID.
"""

from dataclasses import dataclass, field

from collection_planner import CollectionPlan
from search_api import search_photos

from tqdm import tqdm



@dataclass
class CollectionCandidate:
    image_id: int
    path: str
    role_scores: dict[str, float] = field(default_factory=dict)
    role_ranks: dict[str, int] = field(default_factory=dict)
    search_results: dict[str, dict] = field(default_factory=dict)


def retrieve_collection_candidates(
    plan: CollectionPlan,
    *,
    per_role_limit: int = 100,
    identity_limit: int = 1000,
) -> list[CollectionCandidate]:
    """
    Retrieve and merge candidates across collection roles.

    Each role runs its own semantic search. The same image may
    match multiple roles, but appears only once in the returned list.
    """

    if per_role_limit <= 0:
        raise ValueError("per_role_limit must be positive.")

    if identity_limit <= 0:
        raise ValueError("identity_limit must be positive.")

    candidates: dict[int, CollectionCandidate] = {}

    with tqdm(
    plan.roles,
    desc="Finding photos",
    unit="role",
    dynamic_ncols=True,
    ) as progress:
        for role in progress:
            progress.set_postfix_str(role.name)

            results = search_photos(
                query=role.query,
                limit=per_role_limit,
                identity_limit=identity_limit,
                pool_size=max(30, per_role_limit),
                rerank=False,
                selection="none",
            )

            for rank, result in enumerate(results, start=1):
                image_id = result["image_id"]

                if image_id not in candidates:
                    candidates[image_id] = CollectionCandidate(
                        image_id=image_id,
                        path=str(result["path"]),
                    )

                candidate = candidates[image_id]

                candidate.role_scores[role.name] = float(
                    result["semantic_score"]
                )
                candidate.role_ranks[role.name] = rank
                candidate.search_results[role.name] = result

    return list(candidates.values())
