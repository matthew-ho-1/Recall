
"""
Recall unified command-line interface.

Uses the reusable search API directly instead of
launching quality_search.py as a subprocess.
"""

import argparse

from search_api import search_photos
import json

from result_formatter import format_results
from prompt_parser import parse_search_prompt

from pathlib import Path
import webbrowser

from gallery import render_gallery




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall — local-first photo search"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search your local photo library",
    )

    # --------------------------------------------------
    # Search arguments
    # --------------------------------------------------

    search_parser.add_argument(
        "query",
        help="Photo search query or natural-language request",
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum results (default: 10; overrides interpreted count)",
    )

    search_parser.add_argument(
        "--identity-limit",
        type=int,
        default=100,
        help="Number of identity candidates (default: 100)",
    )

    search_parser.add_argument(
        "--pool-size",
        type=int,
        default=30,
        help="Candidate pool size for reranking or selection (default: 30)",
    )

    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable quality-aware reranking",
    )

    # --------------------------------------------------
    # Selection methods
    # --------------------------------------------------

    selection_group = search_parser.add_mutually_exclusive_group()

    selection_group.add_argument(
        "--diverse",
        action="store_true",
        help="Use OpenCLIP diversity selection",
    )

    selection_group.add_argument(
        "--deduplicate",
        action="store_true",
        help="Use perceptual-hash near-duplicate filtering",
    )

    selection_group.add_argument(
        "--mmr",
        action="store_true",
        help="Use quality-aware MMR selection",
    )

    search_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="OpenCLIP similarity threshold (default: 0.90)",
    )

    search_parser.add_argument(
        "--max-hash-distance",
        type=int,
        default=8,
        help="Maximum pHash distance for near-duplicates (default: 8)",
    )

    search_parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.4,
        help="MMR diversity weight between 0 and 1 (default: 0.4)",
    )

    # --------------------------------------------------
    # Output and interpretation
    # --------------------------------------------------

    output_group = search_parser.add_mutually_exclusive_group()

    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )

    output_group.add_argument(
        "--gallery",
        action="store_true",
        help="Generate and open a local HTML photo gallery",
    )

    search_parser.add_argument(
        "--interpret",
        action="store_true",
        help="Interpret natural-language instructions in the query",
    )

    # --------------------------------------------------
    # Parse arguments
    # --------------------------------------------------

    args = parser.parse_args()

    if args.command == "search":
        query = args.query
        limit = 10 if args.limit is None else args.limit
        rerank = args.rerank

        selection = "none"

        if args.mmr:
            selection = "mmr"
        elif args.deduplicate:
            selection = "deduplicate"
        elif args.diverse:
            selection = "diverse"

        # --------------------------------------------------
        # Interpret natural-language prompt (opt-in)
        # --------------------------------------------------

        if args.interpret:
            try:
                intent = parse_search_prompt(args.query)
            except ValueError as error:
                search_parser.error(str(error))

            query = intent.query

            # Explicit --limit overrides the count in the prompt.
            limit = (
                intent.limit
                if args.limit is None
                else args.limit
            )

            # Explicit --rerank can enable quality reranking.
            rerank = intent.rerank or args.rerank

            # Explicit selection flags override parsed intent.
            if args.mmr:
                selection = "mmr"
            elif args.deduplicate:
                selection = "deduplicate"
            elif args.diverse:
                selection = "diverse"
            else:
                selection = intent.selection

        # --------------------------------------------------
        # Execute search using the reusable API
        # --------------------------------------------------

        try:
            results = search_photos(
                query=query,
                limit=limit,
                identity_limit=args.identity_limit,
                pool_size=args.pool_size,
                rerank=rerank,
                selection=selection,
                similarity_threshold=args.similarity_threshold,
                max_hash_distance=args.max_hash_distance,
                diversity_weight=args.diversity_weight,
            )
        except (ValueError, FileNotFoundError) as error:
            search_parser.error(str(error))

        # --------------------------------------------------
        # JSON output
        # --------------------------------------------------

        if args.json:
            payload = format_results(
                results,
                query=query,
                rerank=rerank,
                selection=selection,
            )

            print(json.dumps(payload, indent=2))
            return

        # --------------------------------------------------
        # Local HTML gallery
        # --------------------------------------------------

        if args.gallery:
            payload = format_results(
                results,
                query=query,
                rerank=rerank,
                selection=selection,
            )

            gallery_path = render_gallery(
                payload,
                Path(".recall/gallery.html"),
            )

            gallery_url = gallery_path.resolve().as_uri()

            print(f"Gallery created: {gallery_path.resolve()}")
            webbrowser.open(gallery_url)
            return

        # --------------------------------------------------
        # Human-readable output
        # --------------------------------------------------

        mode = (
            "Quality-aware ranking"
            if rerank
            else "Semantic-only ranking"
        )

        if selection == "mmr":
            mode += " + MMR selection"
        elif selection == "deduplicate":
            mode += " + near-duplicate filtering"
        elif selection == "diverse":
            mode += " + diversity selection"

        print(f'\n{mode}: "{query}"')
        print("-" * 45)

        if not results:
            print("No matching images found.")
            return

        for rank, result in enumerate(results, start=1):
            evidence = result["quality_evidence"]

            defects = []

            if evidence.blur_evidence:
                defects.append("blur")

            if evidence.underexposure_evidence:
                defects.append("underexposed")

            if evidence.overexposure_evidence:
                defects.append("overexposed")

            defect_text = (
                ", ".join(defects)
                if defects
                else "none detected"
            )

            print(
                f"\n{rank}. "
                f"semantic={result['semantic_score']:.4f} "
                f"identity={result['identity_score']:.4f} "
                f"tier={result['quality_tier']}"
            )

            print(f"   {result['path']}")
            print(f"   Defects: {defect_text}")

if __name__ == "__main__":
    main()
