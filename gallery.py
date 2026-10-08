
"""
Local HTML gallery renderer for Recall.

Creates a static HTML page displaying search results.
No external scripts, stylesheets, or cloud services.
"""

from html import escape
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def get_gallery_image_url(image_path: Path) -> str:
    """Return a browser-compatible image URL."""

    image_path = image_path.resolve()

    if image_path.suffix.lower() not in {".heic", ".heif"}:
        return image_path.as_uri()

    import hashlib

    thumbnail_dir = Path(".recall/gallery_thumbnails")
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    image_hash = hashlib.sha256(
        str(image_path).encode("utf-8")
    ).hexdigest()[:16]

    thumbnail_path = thumbnail_dir / f"{image_hash}.jpg"

    if not thumbnail_path.exists():
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((1000, 1000))
            image.save(thumbnail_path, "JPEG", quality=85)

    return thumbnail_path.resolve().as_uri()

def render_gallery(payload: dict, output_path: Path) -> Path:
    """Write a standalone HTML gallery for a search response."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = escape(str(payload["query"]))
    cards = []

    for result in payload["results"]:
        image_path = Path(result["path"])
        image_url = get_gallery_image_url(image_path)

        rank = int(result["rank"])
        tier = int(result["quality_tier"])
        semantic = float(result["semantic_score"])
        identity = float(result["identity_score"])

        defects = result.get("defects", [])
        defect_text = ", ".join(defects) if defects else "none detected"

        cards.append(
            f"""
            <article class="card">
                <img
                    src="{escape(image_url, quote=True)}"
                    alt="Search result {rank}"
                    loading="lazy"
                >
                <div class="details">
                    <strong>#{rank} — {escape(image_path.name)}</strong>
                    <p>Semantic: {semantic:.4f}</p>
                    <p>Identity: {identity:.4f}</p>
                    <p>Quality tier: {tier}</p>
                    <p>Defects: {escape(defect_text)}</p>
                </div>
            </article>
            """
        )

    cards_html = "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Recall — {query}</title>
    <style>
        body {{
            margin: 0;
            padding: 32px;
            font-family: system-ui, sans-serif;
            background: #10151d;
            color: #f1f5f9;
        }}

        h1 {{
            margin-bottom: 8px;
        }}

        .summary {{
            color: #94a3b8;
            margin-bottom: 28px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 20px;
        }}

        .card {{
            overflow: hidden;
            border-radius: 14px;
            background: #1e293b;
            border: 1px solid #334155;
        }}

        .card img {{
            display: block;
            width: 100%;
            height: 220px;
            object-fit: cover;
        }}

        .details {{
            padding: 16px;
        }}

        .details p {{
            margin: 6px 0;
            color: #cbd5e1;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Recall</h1>
    <p class="summary">
        Query: {query} —
        {len(payload["results"])} results —
        {escape(str(payload["ranking"]))} —
        {escape(str(payload["selection"]))}
    </p>

    <main class="grid">
        {cards_html}
    </main>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
