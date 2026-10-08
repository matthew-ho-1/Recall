# Recall

> Find the moments you forgot you captured.

Recall is a **local-first photo retrieval engine** for rediscovering meaningful photos buried in a personal photo library.

Instead of generating new images, Recall makes the real photos you already have searchable using computer vision, face embeddings, and natural language. It can search for both **what is in a photo** and **whether you are in it**, then use technical quality signals and diversity-aware selection to help organize the results.

Example searches:

```text
person holding a camera
group of friends
city skyline
photos of me outside
photos of me wearing a suit
```

## Why Recall?

Some of our best photos are also the ones we've forgotten about.

A great portrait might be sitting in a folder from three years ago. A photo with an old friend might be buried among thousands of screenshots and duplicates. Manually searching an entire photo library does not scale well.

Recall started from a simple question:

> **What if I could search my entire photo library for the moments worth remembering?**

The goal is not just to search photos. It is to **remember what was there**.

## Current Status

Recall is under active development. The latest implemented and tested milestone is **v0.9 — Unified Search Experience**, with **39 passing automated tests** and manual confirmation of the local gallery, including HEIC display. The v0.9 release has **not yet been confirmed as merged, pushed, or tagged**.

The pipeline supports local photo discovery, incremental indexing, semantic search, identity-aware retrieval, technical quality analysis, optional quality-aware reranking, diversity-aware selection, a unified search CLI, deterministic natural-language prompt interpretation, JSON output, and a local HTML gallery. A polished v1.0 release remains planned.

### v0.1 — Image Discovery

- Recursive directory scanning
- `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, and `.webp` support
- File-extension statistics

### v0.2 — Persistent Image Index

- Local SQLite index
- New, modified, unchanged, and deleted image detection
- Incremental indexing
- Scan-root-aware deletion cleanup

### v0.3 — Image Processing

- Pillow-based image decoding and HEIC/HEIF decoding
- Image metadata extraction and thumbnail generation
- Incremental processing and failure tracking
- Cleanup of generated thumbnail artifacts

### v0.4 — Semantic Retrieval

- OpenCLIP image and text embeddings
- Normalized vectors and persistent local embedding cache
- Incremental embedding generation and invalidation on source changes
- Cleanup when source images are deleted
- Cosine-similarity ranking and natural-language photo search

At the current library scale, Recall intentionally uses brute-force similarity search rather than an approximate nearest-neighbor index. This provides a simple, measurable baseline before adding more complex vector indexing.

### v0.5 — Identity-Aware Retrieval

- Face detection and face embeddings using InsightFace
- Persistent face embedding cache
- Identity embedding built from reference photos
- Identity-only and identity-aware semantic retrieval
- Face artifact invalidation and deletion cleanup
- Precision@K evaluation and persisted relevance judgments

Identity-aware retrieval separates two signals:

```text
"me"                      "holding a camera"
  |                                |
  v                                v
Identity similarity         Semantic similarity
  |                                |
  +---------------+----------------+
                  |
                  v
          Filter by identity,
           rank by semantics
```

This allows Recall to move from searching for `person holding a camera` toward searching for `photos of me holding a camera`.

### v0.6 — Technical Photo Quality

- Image resolution and whole-image exposure measurements
- Face-region exposure, face sharpness, and face prominence measurements
- Rule-based technical defect evidence and quality tiers

The quality system detects **specific technical issues**, such as blur and problematic exposure. A photo with no detected defects is **not necessarily** a strong portrait or a good dating-profile photo; composition, expression, context, and personal preference still matter.

### v0.7 — Quality-Aware Retrieval

- Optional quality-aware reranking of identity-filtered, semantically relevant photos
- Configurable candidate pool for reranking
- Semantic similarity remains the primary retrieval signal
- Cached raw quality measurements in SQLite, keyed to the image and selected face
- Cache invalidation when source images or relevant face geometry change
- Quality tiers recomputed from cached measurements
- Semantic-only mode for comparison

Quality-aware reranking uses a heuristic semantic tolerance. It is intended to improve ordering, not guarantee an objectively best photo.

### v0.8 — Diversity-Aware Selection

- Optional OpenCLIP embedding similarity filtering to defer visually similar results
- Optional perceptual-hash (pHash) selection to defer near-duplicate photos
- Quality-aware Maximal Marginal Relevance (MMR) selection
- Configurable similarity threshold, maximum pHash distance, candidate pool, and MMR diversity weight
- Fallback to fill the requested result count when fewer distinct candidates are available
- Automated regression tests for the three selection methods

Selection happens after identity-aware semantic retrieval and optional quality reranking. The three diversity modes are mutually exclusive. MMR with `--diversity-weight 0.0` preserves upstream result order; `0.4` is a practical starting point, not a universally optimal setting. Diversity selection does not guarantee duplicate-free results.

### v0.9 — Unified Search Experience

- `recall.py search` provides one entry point for the retrieval workflow
- `search_api.py` exposes reusable search functionality
- `prompt_parser.py` interprets common natural-language instructions when `--interpret` is enabled
- Explicit CLI options can override interpreted result counts and selection preferences
- `result_formatter.py` provides consistent, JSON-serializable search results
- `--json` produces structured results for scripts and integrations
- `--gallery` generates and opens a **local, static HTML gallery** with scores, quality tiers, and defect evidence
- HEIC/HEIF gallery images are converted to cached JPEG previews for browser compatibility; original photos are not modified
- JSON and gallery output modes are mutually exclusive
- CLI integration tests cover gallery launch and output-mode validation

The prompt parser is deterministic and rule-based, not a generative AI assistant. The gallery uses local files and does not require a hosted photo service.

## How It Works

Recall is an incremental multimodal retrieval pipeline:

```text
Photo Library
     |
     v
Image Discovery --> SQLite Index
     |                 |
     +---------+-------+
               |
       +-------+--------+
       |                |
       v                v
Image Processing   Face Detection
       |                |
       v                v
  Thumbnails       Face Embeddings
       |                |
       v                v
 OpenCLIP Image    Identity Matching
  Embeddings            |
       +--------+-------+
                |
                v
     Identity-Filtered
       Semantic Ranking
                |
                v
      Technical Quality
         Measurement
                |
                v
       Optional Quality
          Reranking
                |
                v
     Optional Diversity
          Selection
                |
                v
        Search Results
          /     |    \
         v      v     v
       Text    JSON  HTML Gallery
```

The stages are deliberately separated:

```text
Discovery   -> What files exist?
Indexing    -> What changed?
Processing  -> Can the image be decoded?
Embedding   -> What does the image represent?
Identity    -> Is the target person in the image?
Quality     -> Are there detectable technical defects?
Retrieval   -> Which matching images best fit the query?
Selection   -> Which results form a useful, varied set?
Presentation-> How should the results be viewed or consumed?
Evaluation  -> How relevant are the ranked results?
```

Expensive work is cached and rerun when needed. Image similarity and face similarity scores are ranking signals, **not calibrated probabilities**.

## Runtime Data

Recall stores generated state separately from the original photo library:

```text
.recall/
├── recall.db
├── thumbnails/
│   └── <image_id>.jpg
├── embeddings/
│   └── <image_id>.npy
├── faces/
│   └── <face_id>.npy
├── identities/
│   └── me.npy
├── gallery.html
└── gallery_thumbnails/
    └── <image_path_hash>.jpg
```

The SQLite database stores image and face records, retrieval evaluation judgments, and cached quality measurements. The gallery's HEIC-compatible JPEG previews currently use a separate cache. The preview cache is created on demand; it does not currently check whether a source HEIC image changed after a preview was generated.

Original photos are treated as read-only. The indexing and processing pipeline tracks source changes and invalidates affected derived state. When a source photo is deleted, Recall cleans up its managed index and processing artifacts rather than modifying the source library. The separate gallery preview cache may require manual cleanup if source HEIC files change.

**Keep `.recall/` out of version control.** It contains local indexes and derived data from a personal photo library, including face and identity embeddings.

## Installation

### Requirements

- Python 3.10+ (subject to compatibility of installed computer-vision dependencies)
- Dependencies listed in `requirements.txt`
- `pillow-heif` for HEIC/HEIF decoding and gallery previews
- A local photo library

Clone the repository:

```bash
git clone https://github.com/matthew-ho-1/Recall.git
cd Recall
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If `pillow_heif` is not installed in the environment, install it with:

```bash
python -m pip install pillow-heif
```

Some computer-vision dependencies may require platform-specific setup. Recall is currently a development-stage CLI, not a packaged desktop application. Search requires a prepared local index and persisted identity embedding (for example, `.recall/identities/me.npy`).

## Usage

### Index a Photo Library

```bash
python scanner.py /path/to/photos
```

On Windows:

```bash
python scanner.py "E:\\Photos"
```

Recall recursively scans the directory and updates its index and generated processing artifacts. Subsequent runs avoid repeating unchanged work. Indexing, image processing, embedding generation, and identity setup are separate development workflows; a scan alone does not necessarily prepare every artifact required for search.

### Unified Search (v0.9)

Use `recall.py search` for the current user-facing search workflow:

```bash
python recall.py search "outside"
python recall.py search "outside" --limit 10 --rerank
python recall.py search "outside" --rerank --mmr --pool-size 30
```

The default ranking is semantic-only. Add `--rerank` to enable quality-aware reranking. The optional selection modes are mutually exclusive:

```bash
python recall.py search "outside" --diverse --similarity-threshold 0.90
python recall.py search "outside" --deduplicate --max-hash-distance 8
python recall.py search "outside" --rerank --mmr --diversity-weight 0.4
```

`--limit` controls the number of results; `--pool-size` controls the candidate pool; `--identity-limit` controls identity candidates. Selection can fill remaining result slots when fewer distinct photos are available.

### Natural-Language Prompt Interpretation

Add `--interpret` to extract a query and options from supported prompt patterns:

```bash
python recall.py search "Find 5 good photos of me outside" --interpret
python recall.py search "Show me 10 different beach photos" --interpret
python recall.py search "Show 8 photos of me hiking without duplicates" --interpret
```

Examples of interpreted preferences include a requested count, quality-aware reranking, varied results, and near-duplicate filtering. An explicit `--limit` overrides an interpreted count; explicit selection flags override interpreted selection preferences. Interpretation is opt-in and does not attempt unrestricted language understanding.

### JSON Output

```bash
python recall.py search "outside" --rerank --mmr --json
python recall.py search "Show me 10 different beach photos" --interpret --json
```

JSON includes the query, ranking mode, selection mode, result count, and per-photo fields such as rank, image ID, path, semantic score, identity score, quality tier, and detected defects.

### Local HTML Gallery

```bash
python recall.py search "outside" --gallery
python recall.py search "Find 5 good photos of me outside" --interpret --gallery
python recall.py search "outside" --rerank --mmr --gallery
```

Recall writes `.recall/gallery.html` and attempts to open it in the default browser. The gallery is a static local page showing photo previews and ranking evidence. HEIC/HEIF images use JPEG previews stored under `.recall/gallery_thumbnails/`; source photos remain unchanged. The gallery may expose local file paths in its HTML, so treat it as private data.

Use **either** `--gallery` **or** `--json` in a single command.

### Legacy Development Commands

The earlier component-specific commands remain useful for debugging and comparisons:

```bash
python search.py "person holding a camera"
python identity_search.py --limit 20
python combined_search.py "outside" --limit 10
python quality_search.py "outside" --limit 10
python quality_search.py "outside" --rerank --pool-size 30 --limit 10
python quality_search.py "outside" --rerank --diverse --similarity-threshold 0.90
python quality_search.py "outside" --rerank --deduplicate --max-hash-distance 8
python quality_search.py "outside" --rerank --mmr --diversity-weight 0.4
```

Quality tiers reflect rule-based evidence of technical defects, not attractiveness, photographic artistry, or suitability for a particular profile. A tier of `0` means **no technical defect was detected by the current rules**, not that a photo is perfect.

### Evaluate Retrieval

```bash
python evaluate.py
python evaluate_relevance.py
```

Relevance judgments are persisted in SQLite so previously labeled query/image pairs can be reused across evaluation runs.

### Run Tests

```bash
python -m unittest discover -p "test_*.py" -v
python -m py_compile recall.py gallery.py prompt_parser.py result_formatter.py search_api.py
```

The v0.9 development branch has been reported to pass **39 automated tests**, including CLI gallery integration and output-mode validation. This is not a claim of comprehensive end-to-end coverage across every image format or operating system.

## Technology

- Python and SQLite
- Pillow and pillow-heif
- PyTorch and OpenCLIP
- NumPy
- InsightFace and ONNX Runtime
- OpenCV
- Standard-library `argparse`, `json`, and `webbrowser` for CLI and gallery integration

## Design Decisions

### Local-first

Personal photo libraries contain sensitive data. Recall is designed so its index, thumbnails, embeddings, identity data, retrieval state, and gallery remain local wherever practical.

### Incremental processing

Image decoding, embedding generation, face detection, and quality measurement are expensive relative to filesystem scanning. Recall tracks source-file changes and caches reusable results. Gallery HEIC previews are a newer, simpler cache with more limited invalidation behavior.

### Simple retrieval baseline

For a library of a few thousand photos, brute-force cosine similarity is sufficient and easy to reason about. Approximate nearest-neighbor indexing can be introduced later if benchmarks show it is necessary.

### Separate identity, semantics, quality, and diversity

Recall does not treat identity as just another text-search concept. Face similarity helps find photos of the target person; OpenCLIP similarity ranks matches to the natural-language query; technical quality evidence can refine the ordering of nearby semantic matches; optional diversity selection can then reduce repetitive results.

### Interpretable quality signals

Quality scoring uses explicit measurements and heuristic thresholds. Thresholds are preliminary and require broader validation.

### Small, reusable user-interface components

The v0.9 CLI calls a reusable search API, while prompt parsing, result formatting, and gallery rendering remain separate modules. This makes the search pipeline usable beyond one command-line output format.

## Roadmap

### Implemented and tested — v0.1 to v0.9

- [x] Recursive image discovery and supported-format scanning
- [x] Persistent SQLite image index and incremental updates
- [x] Image processing, metadata, thumbnails, and artifact cleanup
- [x] OpenCLIP image/text embeddings and semantic search
- [x] Face detection, identity embeddings, and identity-aware retrieval
- [x] Retrieval evaluation and persisted relevance judgments
- [x] Technical image and face quality measurements
- [x] Rule-based quality tiers and defect evidence
- [x] Quality-aware reranking and SQLite measurement caching
- [x] OpenCLIP-based diversity filtering
- [x] pHash-based near-duplicate deferral
- [x] Quality-aware MMR selection
- [x] Unified search CLI and reusable search API
- [x] Opt-in natural-language prompt interpretation
- [x] JSON search output
- [x] Local HTML gallery with HEIC-compatible previews
- [x] 39 passing automated tests on the v0.9 development branch

### Release preparation — v0.9

- [ ] Review and commit updated documentation
- [ ] Merge the v0.9 feature branch into `main`
- [ ] Re-run tests on `main`
- [ ] Tag and push `v0.9.0`

### Planned — v1.0: Local Release

- [ ] Broader retrieval benchmarks and identity threshold calibration
- [ ] More robust end-to-end testing and error handling
- [ ] Performance and usability improvements
- [ ] Reuse the existing thumbnail pipeline for gallery previews, with reliable cache invalidation
- [ ] Installation and usage documentation for a reproducible release
- [ ] Privacy and packaging review

### Milestones

```text
v0.1  Image discovery                       Implemented
v0.2  Persistent incremental indexing      Implemented
v0.3  Image processing + thumbnails        Implemented
v0.4  Semantic retrieval                   Implemented
v0.5  Identity-aware retrieval            Implemented
v0.6  Technical photo quality             Implemented
v0.7  Quality-aware retrieval + caching   Implemented
v0.8  Diversity-aware photo selection     Implemented and tested
v0.9  Unified search UX + local gallery    Implemented and tested; release pending
v1.0  Polished local release              Planned
```

## Privacy

Recall never intentionally modifies or deletes an original photo.

Generated state lives under `.recall/` and can be rebuilt from the source library. Source-photo deletion is handled by removing Recall-owned database records and cached processing artifacts rather than touching unrelated files in the photo library. Gallery preview files may need separate cleanup.

Local storage does not make derived data non-sensitive: face embeddings, identity embeddings, thumbnails, generated galleries, and indexes should be protected like other personal data. **Do not commit `.recall/` or personal photo files to a public repository.**

If Recall is developed into a distributed or hosted product, its privacy model and third-party model licensing will need to be reviewed explicitly.

## Why the Name?

Photos are more than files—they are fragments of journeys we have already lived.

Recall is about finding moments that were captured but no longer remembered: the people, places, and ordinary experiences whose significance may only become clear later.

The goal is not just to search photos.

It is to **remember what was there**.
