
"""
Local curation session management for Recall v0.10.

Tracks previously shown photos, kept selections, and rejected
photos across recommendation batches.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from collection_candidates import CollectionCandidate
from collection_planner import CollectionPlan
from collection_selector import CollectionSelection, select_collection


DEFAULT_SESSION_PATH = Path(".recall/curation_session.json")


@dataclass
class CurationSession:
    goal: str
    shown_ids: set[int] = field(default_factory=set)
    rejected_ids: set[int] = field(default_factory=set)
    kept: dict[str, CollectionSelection] = field(default_factory=dict)
    current: list[CollectionSelection] = field(default_factory=list)
    batch_number: int = 0


def create_session(plan: CollectionPlan) -> CurationSession:
    """Start a new session for a collection plan."""
    return CurationSession(goal=plan.goal)


def another_six(
    session: CurationSession,
    plan: CollectionPlan,
    candidates: list[CollectionCandidate],
) -> list[CollectionSelection]:
    """
    Generate a new batch while preserving kept selections.

    Previously shown images cannot be recommended again unless
    they have been explicitly kept.
    """
    if session.goal != plan.goal:
        raise ValueError("Session goal does not match collection plan.")

    role_names = {role.name for role in plan.roles}

    if any(role not in role_names for role in session.kept):
        raise ValueError("Session contains an unknown kept role.")

    kept_ids = [item.image_id for item in session.kept.values()]

    if len(kept_ids) != len(set(kept_ids)):
        raise ValueError("The same image is kept in multiple roles.")

    if set(kept_ids) & session.rejected_ids:
        raise ValueError("A rejected image cannot also be kept.")

    remaining_roles = tuple(
        role
        for role in plan.roles
        if role.name not in session.kept
    )

    remaining_plan = CollectionPlan(
        goal=plan.goal,
        roles=remaining_roles,
    )

    excluded = (
        session.shown_ids
        | session.rejected_ids
        | set(kept_ids)
    )

    new_selections = select_collection(
        remaining_plan,
        candidates,
        excluded_ids=excluded,
    )

    by_role = {
        **session.kept,
        **{item.role: item for item in new_selections},
    }

    session.current = [
        by_role[role.name]
        for role in plan.roles
        if role.name in by_role
    ]

    session.shown_ids.update(
        item.image_id for item in session.current
    )
    session.batch_number += 1

    return list(session.current)


def keep_photo(session: CurationSession, image_id: int) -> None:
    """Keep a photo from the current batch in its assigned role."""
    selection = next(
        (item for item in session.current if item.image_id == image_id),
        None,
    )

    if selection is None:
        raise ValueError("Photo is not in the current collection.")

    if image_id in session.rejected_ids:
        raise ValueError("Cannot keep a rejected photo.")

    session.kept[selection.role] = selection


def reject_photo(session: CurationSession, image_id: int) -> None:
    """Reject a previously shown photo and release its role."""
    if image_id not in session.shown_ids:
        raise ValueError("Cannot reject a photo that has not been shown.")

    session.rejected_ids.add(image_id)

    session.kept = {
        role: item
        for role, item in session.kept.items()
        if item.image_id != image_id
    }

    session.current = [
        item
        for item in session.current
        if item.image_id != image_id
    ]


def _selection_to_dict(item: CollectionSelection) -> dict:
    return {
        "role": item.role,
        "image_id": item.image_id,
        "path": item.path,
        "score": item.score,
    }


def save_session(
    session: CurationSession,
    path: Path = DEFAULT_SESSION_PATH,
) -> None:
    """Atomically save session state to a local JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "goal": session.goal,
        "shown_ids": sorted(session.shown_ids),
        "rejected_ids": sorted(session.rejected_ids),
        "kept": {
            role: _selection_to_dict(item)
            for role, item in session.kept.items()
        },
        "current": [
            _selection_to_dict(item)
            for item in session.current
        ],
        "batch_number": session.batch_number,
    }

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")

        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_session(
    path: Path = DEFAULT_SESSION_PATH,
) -> CurationSession:
    """Load a previously saved local curation session."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("version") != 1:
        raise ValueError("Unsupported curation session version.")

    def parse_selection(data: dict) -> CollectionSelection:
        return CollectionSelection(
            role=str(data["role"]),
            image_id=int(data["image_id"]),
            path=str(data["path"]),
            score=float(data["score"]),
        )

    return CurationSession(
        goal=str(payload["goal"]),
        shown_ids=set(map(int, payload["shown_ids"])),
        rejected_ids=set(map(int, payload["rejected_ids"])),
        kept={
            role: parse_selection(item)
            for role, item in payload["kept"].items()
        },
        current=[
            parse_selection(item)
            for item in payload["current"]
        ],
        batch_number=int(payload["batch_number"]),
    )
