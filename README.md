# Recall

> Find the moments you forgot you captured.

Recall is a privacy-focused photo retrieval engine for rediscovering meaningful photos buried in your personal photo library.

Modern photo collections can contain thousands of images spread across drives and folders. Recall makes those collections searchable using computer vision and natural language, helping you find photos based on **who is in them, what is happening, and what you're looking for**.

Instead of generating new images, Recall helps you rediscover the real ones you already have.

## Motivation

Some of our best photos are also the ones we've forgotten about.

A great portrait might be sitting in a folder from three years ago. A photo with an old friend might be buried among thousands of screenshots and duplicates. Manually searching through an entire photo library doesn't scale particularly well.

Recall started from a simple question:

> **What if I could search my entire photo library for the moments worth remembering?**

Recall now supports semantic searches such as:

```text
person holding a camera
group of friends
food at a restaurant
city skyline
person wearing a suit
```

The long-term goal is to support more personal and context-aware queries such as:

```text
photos of me filmmaking
photos where I'm genuinely smiling
me with friends in college
good photos of me for a profile picture
```

Recall retrieves and ranks relevant images from the user's existing photo library while keeping its generated data local.

## Current Status

Recall is currently under active development.

### v0.1 — Image Discovery

Recall began as a command-line tool for recursively scanning a directory and discovering supported image files.

Supported formats include:

- `.jpg`
- `.jpeg`
- `.png`
- `.heic`
- `.heif`
- `.webp`

The scanner also reports the number of discovered images by file type.

### v0.2 — Persistent Image Index

Recall maintains a persistent local SQLite index of discovered images.

The index tracks filesystem metadata and synchronizes with the photo library across scans.

Recall distinguishes between:

- New images
- Modified images
- Unchanged images
- Deleted images

Only new or modified files need to be re-indexed, allowing later image-processing and machine-learning stages to avoid unnecessary work.

Deleted files are removed from Recall's index and generated cache without modifying the user's original photo library.

### v0.3 — Image Processing Pipeline

Recall can open indexed images and build a local processing cache.

The processing pipeline supports:

- Image decoding with Pillow
- HEIC/HEIF decoding
- Width and height extraction
- Image format detection
- Incremental image processing
- Thumbnail generation
- Persistent thumbnail paths
- Processing failure tracking
- Thumbnail cleanup for deleted images

Original photos are treated as read-only.

### v0.4 — Semantic Retrieval

Recall can generate multimodal embeddings for indexed photos and search the library using natural-language queries.

The semantic retrieval pipeline supports:

- CLIP-style image embeddings
- CLIP-style text embeddings
- Normalized embedding vectors
- Persistent image embedding cache
- Incremental embedding generation
- Embedding invalidation for modified images
- Embedding cleanup for deleted images
- Cosine-similarity search
- Ranked natural-language search results

Image embeddings are generated once and persisted locally. Subsequent searches embed only the text query and compare it against the cached image vectors.

For the current scale of the project, Recall uses brute-force similarity search rather than an approximate nearest-neighbor index. This keeps the retrieval implementation simple while providing a baseline that can be benchmarked before introducing more complex vector indexing.

Runtime data is stored separately from the photo library:

```text
.recall/
├── recall.db
├── thumbnails/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── embeddings/
    ├── 1.npy
    ├── 2.npy
    └── ...
```

## Usage

### Requirements

- Python 3.10+

Clone the repository:

```bash
git clone https://github.com/matthew-ho-1/Recall.git
cd Recall
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Index a Photo Library

Run Recall against a photo directory:

```bash
python scanner.py /path/to/photos
```

On Windows:

```bash
python scanner.py "E:\Photos"
```

or using the Python launcher:

```bash
py scanner.py "E:\Photos"
```

Recall recursively scans all subdirectories under the provided path.

A scan may produce output similar to:

```text
Scanning E:\Photos...

Found 1847 images:

.heic     724
.jpeg      83
.jpg      912
.png      103
.webp      25

Indexing images...

Discovered: 1847
New:          12
Modified:      2
Unchanged:  1833
Deleted:       0

Processing images...

Processed:    14
Skipped:    1833
Failed:        0

Generating embeddings...

Embedded:     14
Skipped:    1833
Failed:        0
```

Subsequent scans avoid unnecessarily processing or embedding unchanged images.

### Search Your Photo Library

Once the library has been indexed, search it using natural language:

```bash
python search.py "person holding a camera"
```

Specify the number of results:

```bash
python search.py "group of friends" --limit 10
```

Example output:

```text
Results for "person holding a camera":

1. 0.312  E:\Photos\film-shoot.jpg
2. 0.287  E:\Photos\camera.jpg
3. 0.265  E:\Photos\friends.jpg
4. 0.251  E:\Photos\trip.jpg
5. 0.243  E:\Photos\portrait.jpg
```

Similarity scores represent relative closeness in the multimodal embedding space rather than probabilities or confidence percentages.

## Architecture

Recall is built as an incremental multimodal retrieval pipeline:

```text
Photo Library
      │
      ▼
┌──────────────────┐
│ Image Discovery  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Persistent       │
│ SQLite Index     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Image Processing │
│ + Thumbnails     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Image Embeddings │
│      (CLIP)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Persistent       │
│ Embedding Cache  │
└────────┬─────────┘
         │
         │              Natural-Language Query
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ Text Embedding   │
         │              │     (CLIP)       │
         │              └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌──────────────────┐
            │ Cosine           │
            │ Similarity       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Ranked Results   │
            └──────────────────┘
```

The current architecture deliberately separates:

```text
Discovery      → What files exist?
Indexing       → What changed?
Processing     → Can the image be decoded?
Embedding      → What does the image represent?
Retrieval      → Which images match the query?
```

This allows expensive stages such as image processing and embedding generation to run incrementally.

Current technologies include:

- Python
- SQLite
- Pillow
- pillow-heif
- PyTorch
- OpenCLIP
- NumPy
- CLIP-style multimodal embeddings

Potential future technologies include:

- Face detection and face embeddings
- Perceptual hashing
- Approximate nearest-neighbor search
- FastAPI
- Web-based result visualization

## Roadmap

### Foundation

- [x] Recursive image discovery
- [x] File-extension statistics
- [x] Persistent SQLite image index
- [x] Incremental indexing
- [x] New, modified, unchanged, and deleted image detection
- [x] Image metadata extraction
- [x] HEIC/HEIF image support
- [x] Thumbnail generation
- [x] Incremental image processing
- [x] Processing failure tracking
- [x] Thumbnail cleanup

### Retrieval

- [x] Multimodal image embeddings
- [x] Text embeddings
- [x] Natural-language photo search
- [x] Vector similarity search
- [x] Persistent embedding cache
- [x] Incremental embedding generation
- [x] Embedding invalidation and cleanup

### Identity

- [ ] Face detection
- [ ] Face embeddings
- [ ] Identity matching
- [ ] Identity-aware semantic queries

### Ranking

- [ ] Duplicate and near-duplicate detection
- [ ] Image quality signals
- [ ] Context-aware ranking
- [ ] Retrieval evaluation and benchmarks

### Application

- [ ] Search API
- [ ] Web interface
- [ ] Interactive result gallery

## Project Milestones

```text
v0.1    Image discovery
  │
  ▼
v0.2    Persistent incremental indexing
  │
  ▼
v0.3    Image processing + thumbnails
  │
  ▼
v0.4    Semantic retrieval
  │
  ▼
v0.5    Identity-aware retrieval
  │
  ▼
v1.0    Full Recall experience
```

The next major milestone is **v0.5 — identity-aware retrieval**.

Semantic retrieval can answer:

```text
person holding a camera
```

Identity-aware retrieval will begin addressing a different question:

```text
photos of me holding a camera
```

This requires separating two signals:

```text
"me"                  "holding a camera"
 │                           │
 ▼                           ▼
Identity similarity     Semantic similarity
 │                           │
 └────────────┬──────────────┘
              ▼
        Combined ranking
```

## Privacy

Personal photo libraries contain highly sensitive data.

Recall is designed around a **local-first** philosophy. Photo indexing, thumbnails, embeddings, metadata, and retrieval are intended to remain on the user's machine wherever practical.

Original photos are treated as read-only.

Recall's generated state is stored separately:

```text
.recall/
├── recall.db
├── thumbnails/
└── embeddings/
```

When an original photo is modified, Recall invalidates and regenerates derived data as needed.

When an original photo is deleted, Recall removes its corresponding database entry, thumbnail, and embedding.

Recall never deletes an original photo.

## Why "Recall"?

Photos are more than files—they're fragments of journeys we've already lived.

Recall is about finding the moments that were captured but no longer remembered: the people, places, and ordinary experiences whose significance may only become clear later.

The goal isn't just to search photos.

It's to **remember what was there**.

## License

This project is currently under development.