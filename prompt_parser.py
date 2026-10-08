
"""
Deterministic natural-language prompt parser for Recall.

Extracts simple search preferences without requiring
an LLM or network access.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchIntent:
    query: str
    limit: int = 10
    rerank: bool = False
    selection: str = "none"


def parse_search_prompt(prompt: str) -> SearchIntent:
    """Convert a natural-language request into search options."""

    if not prompt or not prompt.strip():
        raise ValueError("Search prompt cannot be empty")

    text = prompt.strip()
    limit = 10
    rerank = False
    selection = "none"

    # Extract a requested result count from common command phrases.
    count_match = re.search(
        r"\b(?:find|show(?: me)?|get)\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    if count_match:
        limit = int(count_match.group(1))
        if limit < 1:
            raise ValueError("Requested result count must be at least 1")

        text = (
            text[:count_match.start(1)]
            + text[count_match.end(1):]
        )

    # Quality-related preferences.
    if re.search(
        r"\b(best|high[- ]quality|good(?: quality)?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        rerank = True

        text = re.sub(
            r"\b(best|high[- ]quality|good(?: quality)?)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # Deduplication takes priority over general diversity.
    if re.search(
        r"\b(without duplicates|no duplicates|deduplicat(?:e|ed))\b",
        text,
        flags=re.IGNORECASE,
    ):
        selection = "deduplicate"

        text = re.sub(
            r"\b(without duplicates|no duplicates|deduplicat(?:e|ed))\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    elif re.search(
        r"\b(diverse|different|varied)\b",
        text,
        flags=re.IGNORECASE,
    ):
        selection = "mmr"

        text = re.sub(
            r"\b(diverse|different|varied)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # Remove leading command wording, not subject words.
    text = re.sub(
        r"^\s*(?:find|show(?: me)?|get)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove leading photo-related filler.
    text = re.sub(
        r"^\s*(?:(?:some|the|my)\s+)?"
        r"(?:photos?|pictures?|images?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove an optional leading "of".
    text = re.sub(
        r"^\s*of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove a trailing photo-related noun phrase.
    text = re.sub(
        r"\b(?:photos?|pictures?|images?)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    text = " ".join(text.split()).strip()

    # Preserve the original request if parsing stripped everything.
    if not text:
        text = prompt.strip()

    return SearchIntent(
        query=text,
        limit=limit,
        rerank=rerank,
        selection=selection,
    )
