# Recall

> Find the moments you forgot you captured.

Recall is a privacy-focused photo retrieval engine for rediscovering meaningful photos buried in your personal photo library.

Modern photo collections can contain thousands of images spread across drives and folders. Recall aims to make those collections searchable using computer vision and natural language, helping you find photos based on **who is in them, what is happening, and what you're looking for**.

Instead of generating new images, Recall helps you rediscover the real ones you already have.

## Motivation

Some of our best photos are also the ones we've forgotten about.

A great portrait might be sitting in a folder from three years ago. A photo with an old friend might be buried among thousands of screenshots and duplicates. Manually searching through an entire photo library doesn't scale particularly well.

Recall started from a simple question:

> **What if I could search my entire photo library for the moments worth remembering?**

The long-term goal is to build a local-first multimodal retrieval system capable of queries such as:

```text
photos of me filmmaking

me wearing a suit

photos where I'm genuinely smiling

me with friends in college

good photos of me for a profile picture
```

Recall can then retrieve and rank relevant images from the user's existing photo library.

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

Recall can distinguish between:

- New images
- Modified images
- Unchanged images
- Deleted images

Only new or modified files need to be re-indexed, allowing later image-processing and machine-learning stages to avoid unnecessary work.

Deleted files are removed from the index without modifying the user's original photo library.

### v0.3 — Image Processing Pipeline

Recall can now open indexed images and build a local processing cache.

The processing pipeline currently supports:

- Image decoding with Pillow
- HEIC/HEIF decoding
- Width and height extraction
- Image format detection
- Incremental image processing
- Thumbnail generation
- Persistent thumbnail paths
- Processing failure tracking
- Cleanup of thumbnails for deleted images

Generated thumbnails are stored inside Recall's local cache rather than alongside the user's original photos.

Example runtime data:

```text
.recall/
├── recall.db
└── thumbnails/
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    └── ...
```

Original photos are never modified by the processing pipeline.

## Usage

### Requirements

- Python 3.10+

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Recall.git
cd Recall
```

Install dependencies:

```bash
pip install -r requirements.txt
```

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

.heic    724
.jpeg     83
.jpg     912
.png     103
.webp     25

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
```

Subsequent scans avoid unnecessarily processing unchanged images.

## Architecture

Recall is being built as a multi-stage retrieval pipeline:

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
         ├────────────────┐
         ▼                ▼
┌────────────────┐  ┌────────────────┐
│ Face / Identity│  │ Image / Text   │
│ Embeddings     │  │ Embeddings     │
└────────┬───────┘  └────────┬───────┘
         │                   │
         └─────────┬─────────┘
                   ▼
          ┌────────────────┐
          │ Vector Search  │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │ Ranking /      │
          │ Reranking      │
          └────────┬───────┘
                   ▼
              Best Matches
```

Current and potential technologies include:

- Python
- SQLite
- Pillow
- pillow-heif
- OpenCV
- CLIP-style multimodal embeddings
- Face embeddings
- Perceptual hashing
- FAISS
- FastAPI

The architecture is evolving as Recall moves from filesystem indexing toward multimodal retrieval.

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

- [ ] Multimodal image embeddings
- [ ] Natural-language photo search
- [ ] Vector similarity search
- [ ] Persistent vector index

### Identity

- [ ] Face detection
- [ ] Face embeddings
- [ ] Identity matching

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

The next major milestone is **v0.4**, where Recall will begin generating multimodal embeddings and retrieving photos using natural-language queries.

## Privacy

Personal photo libraries contain highly sensitive data.

Recall is designed around a **local-first** philosophy. Wherever practical, photo indexing, thumbnails, embeddings, metadata, and retrieval should remain on the user's machine.

Original photos are treated as read-only.

Recall's generated state is stored separately:

```text
.recall/
├── recall.db
└── thumbnails/
```

The indexing and processing pipeline should never modify, move, or delete original photos.

## Why "Recall"?

Photos are more than files—they're fragments of journeys we've already lived.

Recall is about finding the moments that were captured but no longer remembered: the people, places, and ordinary experiences whose significance may only become clear later.

The goal isn't just to search photos.

It's to **remember what was there**.

## License

This project is currently under development.