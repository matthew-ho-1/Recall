
"""
Collection planning for Recall v0.10.

Defines which photo roles a curated collection should contain.
No image processing or model inference happens here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoRole:
    name: str
    query: str
    description: str


@dataclass(frozen=True)
class CollectionPlan:
    goal: str
    roles: tuple[PhotoRole, ...]

    @property
    def count(self) -> int:
        return len(self.roles)


HINGE_ROLES = (
    PhotoRole(
        name="main_portrait",
        query="clear smiling portrait of a person",
        description="A clear, welcoming photo where you are easy to recognize",
    ),
    PhotoRole(
        name="full_body",
        query="full body photo of a person standing outdoors",
        description="A photo showing your overall appearance",
    ),
    PhotoRole(
        name="group",
        query="person with friends in a group photo",
        description="A social photo with friends",
    ),
    PhotoRole(
        name="activity",
        query="person doing an activity or hobby",
        description="A photo showing you doing something",
    ),
    PhotoRole(
        name="lifestyle",
        query="person traveling or exploring a place",
        description="A photo showing an interesting setting or experience",
    ),
    PhotoRole(
        name="personality",
        query="candid photo of a person laughing or having fun",
        description="A natural photo showing personality",
    ),
)


def create_collection_plan(
    goal: str,
    *,
    count: int = 6,
) -> CollectionPlan:
    """
    Create a deterministic collection plan for a supported goal.

    v0.10 initially supports six-photo Hinge collections.
    """

    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Collection goal cannot be empty.")

    if count != 6:
        raise ValueError(
            "The initial Hinge collection preset requires exactly 6 photos."
        )

    normalized_goal = goal.strip().lower()

    if "hinge" not in normalized_goal:
        raise ValueError(
            "Unsupported collection goal. Currently supported: Hinge."
        )

    return CollectionPlan(
        goal="hinge",
        roles=HINGE_ROLES,
    )
