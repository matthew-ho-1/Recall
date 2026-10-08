# Recall

> Find the moments you forgot you captured.

Recall is a local-first photo retrieval engine for rediscovering meaningful photos buried in a personal photo library.

Instead of generating new images, Recall makes the real photos you already have searchable using computer vision, face embeddings, and natural language. It can search for both **what is in a photo** and **whether you are in it**, then use technical quality signals to help rank the results.

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

Recall is under active development. The latest completed milestone is **v0.7.0 — Quality-Aware Retrieval**.

The current pipeline supports local photo discovery, incremental indexing, semantic search, identity-aware retrieval, technical quality analysis, and optional quality-aware reranking. **Diversity-aware selection, a unified user experience, and a polished v1.0 release are still planned.**

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

- Pillow-based image decoding
- HEIC/HEIF decoding
- Image metadata extraction
- Thumbnail generation
- Incremental processing
- Processing failure tracking
- Cleanup of generated thumbnail artifacts

### v0.4 — Semantic Retrieval

- OpenCLIP image and text embeddings
- Normalized embedding vectors
- Persistent local embedding cache
- Incremental embedding generation
- Invalidation when source images change
- Cleanup when source images are deleted
- Cosine-similarity ranking
- Natural-language photo search

At the current library scale, Recall intentionally uses brute-force similarity search rather than an approximate nearest-neighbor index. This provides a simple, measurable baseline before adding more complex vector indexing.

### v0.5 — Identity-Aware Retrieval

- Face detection and face embeddings using InsightFace
- Persistent face embedding cache
- Identity embedding built from reference photos
- Identity-only photo retrieval
- Identity-aware semantic retrieval
- Face artifact invalidation and deletion cleanup
- Precision@K evaluation
- Persisted relevance judgments

Identity-aware retrieval separates two signals:

```text
"me"                         "holding a camera"
  │                                    │
  ▼                                    ▼
Identity similarity             Semantic similarity
  │                                    │
  └────────────────┬───────────────────┘
                   ▼
            Filter by identity,
             rank by semantics
```

This allows Recall to move from searching for `person holding a camera` toward searching for `photos of me holding a camera`.

### v0.6 — Technical Photo Quality

- Image resolution measurement
- Whole-image exposure measurement
- Face-region exposure measurement
- Face sharpness measurement
- Face prominence measurement
- Rule-based technical defect evidence and quality tiers

The quality system detects **specific technical issues**, such as blur and problematic exposure. A photo with no detected defects is **not necessarily** a strong portrait or a good dating-profile photo; composition, expression, context, and personal preference still matter.

### v0.7 — Quality-Aware Retrieval

- Quality-aware reranking of identity-filtered, semantically relevant photos
- Configurable candidate pool for reranking
- Semantic similarity remains the primary retrieval signal; quality helps distinguish nearby candidates
- Cached raw quality measurements in SQLite, keyed to the image and selected face
- Cache invalidation when source images or relevant face geometry change
- Quality tiers recomputed from cached measurements
- Semantic-only mode for comparison with quality-aware results

Quality-aware reranking is **optional** and uses a heuristic semantic tolerance. It is intended to improve ordering, not to guarantee an objectively best photo.

## How It Works

Recall is an incremental multimodal retrieval pipeline:

```text
Photo Library
     │
     ▼
Image Discovery
     │
     ▼
SQLite Index
     │
     ├───────────────────┐
     ▼                   ▼
Image Processing     Face Detection
     │                   │
     ▼                   ▼
Thumbnails          Face Embeddings
     │                   │
     ▼                   ▼
OpenCLIP Image       Identity Matching
Embeddings               │
     │                   │
     └─────────┬─────────┘
               ▼
     Identity-Filtered
      Semantic Ranking
               │
               ▼
     Technical Quality
        Measurement
               │
               ▼
     Optional Quality
       Reranking
               │
               ▼
        Ranked Results
```

The stages are deliberately separated:

```text
Discovery   → What files exist?
Indexing    → What changed?
Processing  → Can the image be decoded?
Embedding   → What does the image represent?
Identity    → Is the target person in the image?
Quality     → Are there detectable technical defects?
Retrieval   → Which matching images best fit the query?
Evaluation  → How relevant are the ranked results?
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
└── identities/
    └── me.npy
```

The SQLite database stores image and face records, retrieval evaluation judgments, and cached quality measurements. Quality measurements do not require a separate directory.

Original photos are treated as read-only. When a source photo changes, Recall tracks changes and invalidates affected derived state. When a source photo is deleted, Recall cleans up its database records and generated artifacts rather than touching unrelated files.

**Keep `.recall/` out of version control.** It contains local indexes and derived data from a personal photo library, including face and identity embeddings.

## Installation

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- A local photo library

Clone the repository:

```bash
git clone https://github.com/matthew-ho-1/Recall.git
cd Recall
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Some computer-vision dependencies may require platform-specific setup. The current project is a development-stage CLI rather than a packaged desktop application.

## Usage

### Index a Photo Library

```bash
python scanner.py /path/to/photos
```

On Windows:

```bash
python scanner.py "E:\Photos"
```

Recall recursively scans the directory and updates its index and generated processing artifacts. Subsequent runs avoid repeating unchanged work.

### Semantic Search

Search the indexed library with natural language:

```bash
python search.py "person holding a camera"
python search.py "group of friends" --limit 10
```

Similarity scores represent relative closeness in the multimodal embedding space, not confidence percentages.

### Identity Search

Search for photos matching the persisted identity:

```bash
python identity_search.py --limit 20
```

Identity search requires an identity embedding created from reference photos.

### Identity-Aware Semantic Search

Filter to photos matching the identity, then rank candidates by a semantic query:

```bash
python combined_search.py "with people" --limit 10
python combined_search.py "outside" --limit 10
python combined_search.py "wearing a suit" --limit 10
```

### Quality-Aware Search

The default command retrieves identity-aware semantic results and reports their quality measurements **without changing semantic ranking**:

```bash
python quality_search.py "outside" --limit 10
```

Enable quality-aware reranking over a larger candidate pool:

```bash
python quality_search.py "outside" --limit 10 --pool-size 30 --rerank
python quality_search.py "with people" --limit 10 --pool-size 30 --rerank
```

`--limit` controls the number of returned photos; `--pool-size` controls how many candidates are considered when reranking. `--identity-limit` can also adjust the size of the identity candidate set.

Quality tiers reflect rule-based evidence of technical defects. They are not a measure of attractiveness, photographic artistry, or suitability for a particular profile.

### Evaluate Retrieval

Recall includes evaluation tooling for measuring retrieval behavior and Precision@K:

```bash
python evaluate.py
python evaluate_relevance.py
```

Relevance judgments are persisted in SQLite so previously labeled query/image pairs can be reused across evaluation runs.

## Technology

- Python
- SQLite
- Pillow and pillow-heif
- PyTorch
- OpenCLIP
- NumPy
- InsightFace
- ONNX Runtime
- OpenCV

## Design Decisions

### Local-first

Personal photo libraries contain sensitive data. Recall is designed so its index, thumbnails, embeddings, identity data, and retrieval state remain local wherever practical.

### Incremental processing

Image decoding, embedding generation, face detection, and quality measurement are expensive relative to filesystem scanning. Recall tracks source-file changes and caches reusable results.

### Simple retrieval baseline

For a library of a few thousand photos, brute-force cosine similarity is sufficient and easy to reason about. Approximate nearest-neighbor indexing can be introduced later if benchmarks show it is necessary.

### Separate identity, semantics, and quality

Recall does not treat identity as just another text-search concept. Face similarity helps find photos of the target person; OpenCLIP similarity ranks matches to the natural-language query; technical quality evidence can refine the ordering of nearby semantic matches.

### Interpretable quality signals

Quality scoring uses explicit measurements and heuristic thresholds. A tier of `0` means **no technical defect was detected by the current rules**, not that a photo is perfect. Thresholds are preliminary and require broader validation.

## Roadmap

### Completed — v0.1 to v0.7

- [x] Recursive image discovery and supported-format scanning
- [x] Persistent SQLite image index and incremental updates
- [x] Image processing, metadata, thumbnails, and artifact cleanup
- [x] OpenCLIP image/text embeddings and semantic search
- [x] Face detection, identity embeddings, and identity-aware retrieval
- [x] Retrieval evaluation and persisted relevance judgments
- [x] Technical image and face quality measurements
- [x] Rule-based quality tiers and defect evidence
- [x] Quality-aware reranking with a configurable candidate pool
- [x] SQLite quality measurement cache and invalidation

### Next — v0.8: Diversity-Aware Selection

- [ ] Compare candidate photos using existing image embeddings
- [ ] Reduce near-duplicate and repetitive results
- [ ] Select varied photos without losing too much relevance or quality
- [ ] Evaluate diversity against the existing ranking baseline

### Planned — v0.9: User Experience and Prompts

- [ ] More unified search workflow
- [ ] Natural-language prompt assistance
- [ ] Easier review of image results, potentially with a local gallery
- [ ] Improve usability beyond separate development CLI commands

### Planned — v1.0: Local Release

- [ ] Broader retrieval benchmarks and identity threshold calibration
- [ ] More robust end-to-end testing and error handling
- [ ] Performance and usability improvements
- [ ] Installation and usage documentation for a reproducible release
- [ ] Privacy and packaging review

### Milestones

```text
v0.1  Image discovery                         ✓
v0.2  Persistent incremental indexing        ✓
v0.3  Image processing + thumbnails          ✓
v0.4  Semantic retrieval                     ✓
v0.5  Identity-aware retrieval              ✓
v0.6  Technical photo quality               ✓
v0.7  Quality-aware retrieval + caching     ✓
v0.8  Diversity-aware photo selection       Planned
v0.9  User experience + prompts             Planned
v1.0  Polished local release                Planned
```

## Privacy

Recall never intentionally modifies or deletes an original photo.

Generated state lives under `.recall/` and can be rebuilt from the source library. Source-photo deletion is handled by removing Recall-owned database records and cached artifacts rather than touching any other files in the photo library.

Local storage does not make derived data non-sensitive: face embeddings, identity embeddings, thumbnails, and indexes should be protected like other personal data. Do not commit `.recall/` or personal photo files to a public repository.

If Recall is developed into a distributed or hosted product, its privacy model and third-party model licensing will need to be reviewed explicitly.

## Why the Name?

Photos are more than files—they are fragments of journeys we have already lived.

Recall is about finding moments that were captured but no longer remembered: the people, places, and ordinary experiences whose significance may only become clear later.

The goal is not just to search photos.

It is to **remember what was there**.
