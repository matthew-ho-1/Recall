# Recall

> Find the moments you forgot you captured.

Recall is a local-first photo retrieval engine for rediscovering
meaningful photos buried in a personal photo library.

Instead of generating new images, Recall makes the real photos you
already have searchable using computer vision, face embeddings, and
natural language. It can search for both **what is in a photo** and
**whether you are in it**.

``` text
person holding a camera
group of friends
city skyline
photos of me outside
photos of me wearing a suit
```

## Why Recall?

Some of our best photos are also the ones we've forgotten about.

A great portrait might be sitting in a folder from three years ago. A
photo with an old friend might be buried among thousands of screenshots
and duplicates. Manually searching an entire photo library does not
scale well.

Recall started from a simple question:

> **What if I could search my entire photo library for the moments worth
> remembering?**

The goal is not just to search photos. It is to **remember what was
there**.

## Current Status

Recall is under active development. The current milestone, **v0.5**,
supports identity-aware semantic retrieval.

### v0.1 --- Image Discovery

-   Recursive directory scanning
-   `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, and `.webp` support
-   File-extension statistics

### v0.2 --- Persistent Image Index

-   Local SQLite index
-   New, modified, unchanged, and deleted image detection
-   Incremental indexing
-   Scan-root-aware deletion cleanup

### v0.3 --- Image Processing

-   Pillow-based image decoding
-   HEIC/HEIF decoding
-   Image metadata extraction
-   Thumbnail generation
-   Incremental processing
-   Processing failure tracking
-   Cleanup of generated thumbnail artifacts

### v0.4 --- Semantic Retrieval

-   CLIP-style image and text embeddings
-   Normalized embedding vectors
-   Persistent local embedding cache
-   Incremental embedding generation
-   Invalidation when source images change
-   Cleanup when source images are deleted
-   Cosine-similarity ranking
-   Natural-language photo search

At the current library scale, Recall intentionally uses brute-force
similarity search rather than an approximate nearest-neighbor index.
This provides a simple, measurable baseline before adding more complex
vector indexing.

### v0.5 --- Identity-Aware Retrieval

-   Face detection
-   Face embeddings
-   Persistent face embedding cache
-   Identity embedding built from reference photos
-   Identity-only photo retrieval
-   Identity-aware semantic retrieval
-   Face artifact invalidation and deletion cleanup
-   Precision@K evaluation
-   Persisted relevance judgments

Identity-aware retrieval separates two signals:

``` text
"me"                         "holding a camera"
  │                                  │
  ▼                                  ▼
Identity similarity            Semantic similarity
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
       Filter by identity,
       rank by semantics
```

This allows Recall to move from:

``` text
person holding a camera
```

toward:

``` text
photos of me holding a camera
```

## How It Works

Recall is an incremental multimodal retrieval pipeline:

``` text
Photo Library
     │
     ▼
Image Discovery
     │
     ▼
SQLite Index
     │
     ├───────────────┐
     ▼               ▼
Thumbnails       Face Detection
     │               │
     ▼               ▼
CLIP Image       Face Embeddings
Embeddings           │
     │               ▼
     │          Identity Matching
     │               │
     └───────┬───────┘
             ▼
     Semantic Ranking
             │
             ▼
       Ranked Results
```

The stages are deliberately separated:

``` text
Discovery   → What files exist?
Indexing    → What changed?
Processing  → Can the image be decoded?
Embedding   → What does the image represent?
Identity    → Is the target person in the image?
Retrieval   → Which matching images best fit the query?
Evaluation  → How relevant are the ranked results?
```

Expensive work is cached and only rerun when necessary.

## Runtime Data

Recall stores generated state separately from the original photo
library:

``` text
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

Original photos are treated as read-only.

When a source photo changes, Recall invalidates and regenerates the
affected derived data. When a source photo is deleted, Recall removes
its database state and generated thumbnail, semantic embedding, and face
embeddings.

## Installation

### Requirements

-   Python 3.10+
-   Dependencies listed in `requirements.txt`

Clone the repository:

``` bash
git clone https://github.com/matthew-ho-1/Recall.git
cd Recall
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Usage

### Index a Photo Library

``` bash
python scanner.py /path/to/photos
```

On Windows:

``` bash
python scanner.py "E:\Photos"
```

Recall recursively scans the directory, updates its index, processes
changed images, generates semantic embeddings, and detects faces.
Subsequent scans skip unchanged work.

### Semantic Search

Search the indexed library with natural language:

``` bash
python search.py "person holding a camera"
```

Specify the number of results:

``` bash
python search.py "group of friends" --limit 10
```

Similarity scores represent relative closeness in the multimodal
embedding space, not probabilities or confidence percentages.

### Identity Search

Search for photos matching the persisted identity:

``` bash
python identity_search.py --limit 20
```

### Identity-Aware Semantic Search

Filter to photos matching the identity, then rank those candidates by a
semantic query:

``` bash
python combined_search.py "with people" --limit 10
python combined_search.py "outside" --limit 10
python combined_search.py "wearing a suit" --limit 10
```

### Evaluate Retrieval

Recall includes evaluation tooling for measuring retrieval behavior and
Precision@K:

``` bash
python evaluate.py
python evaluate_relevance.py
```

Relevance judgments are persisted in SQLite so previously labeled
query/image pairs can be reused across evaluation runs.

## Technology

Current technologies include:

-   Python
-   SQLite
-   Pillow
-   pillow-heif
-   PyTorch
-   OpenCLIP
-   NumPy
-   InsightFace
-   ONNX Runtime
-   OpenCV

## Design Decisions

### Local-first

Personal photo libraries contain sensitive data. Recall is designed so
its index, thumbnails, embeddings, identity data, and retrieval state
remain local wherever practical.

### Incremental processing

Image decoding, embedding generation, and face detection are expensive
relative to filesystem scanning. Recall tracks source-file changes and
avoids repeating those stages for unchanged images.

### Simple retrieval baseline

For a library of a few thousand photos, brute-force cosine similarity is
sufficient and easy to reason about. Approximate nearest-neighbor
indexing can be introduced later if benchmarks show it is necessary.

### Separate identity and semantic signals

Recall does not treat identity as just another text-search concept. Face
similarity determines whether the target identity appears in an image;
CLIP similarity determines how well that image matches the
natural-language query.

## Roadmap

### Foundation

-   [x] Recursive image discovery
-   [x] Persistent SQLite image index
-   [x] Incremental indexing
-   [x] Image metadata extraction
-   [x] HEIC/HEIF support
-   [x] Thumbnail generation
-   [x] Generated-artifact cleanup

### Semantic Retrieval

-   [x] Image embeddings
-   [x] Text embeddings
-   [x] Natural-language search
-   [x] Persistent embedding cache
-   [x] Incremental embedding generation
-   [x] Embedding invalidation and cleanup

### Identity

-   [x] Face detection
-   [x] Face embeddings
-   [x] Identity matching
-   [x] Identity-aware semantic retrieval
-   [x] Face embedding invalidation and cleanup

### Evaluation

-   [x] Search latency measurement
-   [x] Precision@K evaluation
-   [x] Persisted relevance judgments
-   [ ] Larger retrieval benchmark set
-   [ ] Identity threshold calibration

### Ranking

-   [ ] Duplicate and near-duplicate detection
-   [ ] Image quality signals
-   [ ] Context-aware ranking

### Application

-   [ ] Unified CLI
-   [ ] Search API
-   [ ] Web interface
-   [ ] Interactive result gallery

## Milestones

``` text
v0.1  Image discovery
  │
  ▼
v0.2  Persistent incremental indexing
  │
  ▼
v0.3  Image processing + thumbnails
  │
  ▼
v0.4  Semantic retrieval
  │
  ▼
v0.5  Identity-aware retrieval
  │
  ▼
v1.0  Full Recall experience
```

## Privacy

Recall never intentionally modifies or deletes an original photo.

Generated state lives under `.recall/` and can be rebuilt from the
source library. Source-photo deletion is handled by removing
Recall-owned database records and cached artifacts rather than touching
any other files in the photo library.

If Recall is developed into a distributed or hosted product, its privacy
model and third-party model licensing will need to be reviewed
explicitly.

## Why the Name?

Photos are more than files---they are fragments of journeys we have
already lived.

Recall is about finding moments that were captured but no longer
remembered: the people, places, and ordinary experiences whose
significance may only become clear later.

The goal is not just to search photos.

It is to **remember what was there**.
