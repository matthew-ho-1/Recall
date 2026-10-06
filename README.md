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

Recall is currently under development.

### v0.1 — Image Discovery

The current CLI recursively scans a directory and discovers supported image files.

Supported formats currently include:

- `.jpg`
- `.jpeg`
- `.png`
- `.heic`
- `.heif`
- `.webp`

The scanner also reports the number of discovered images by file type.

Example:

```text
Scanning E:\Photos...

Found 1847 images:

.heic    724
.jpeg     83
.jpg     912
.png     103
.webp     25
```

## Usage

### Requirements

- Python 3.10+

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Recall.git
cd Recall
```

Run the scanner:

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

The scanner recursively searches all subdirectories under the provided path.

## Planned Architecture

Recall will use a multi-stage retrieval pipeline:

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
│ Metadata +       │
│ Thumbnail Index  │
└────────┬─────────┘
         │
         ├───────────────┐
         ▼               ▼
┌────────────────┐  ┌────────────────┐
│ Face / Identity│  │ Image/Text     │
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

Potential technologies include:

- Python
- OpenCV
- Pillow
- CLIP-style multimodal embeddings
- Face embeddings
- perceptual hashing
- FAISS
- SQLite
- FastAPI

The architecture is still evolving as the project develops.

## Roadmap

- [x] Recursive image discovery
- [x] File-extension statistics
- [ ] Image metadata extraction
- [ ] Thumbnail generation
- [ ] Persistent local image index
- [ ] Duplicate and near-duplicate detection
- [ ] Face detection
- [ ] Identity matching
- [ ] Multimodal image embeddings
- [ ] Natural-language photo search
- [ ] Vector similarity search
- [ ] Image quality ranking
- [ ] Retrieval evaluation and benchmarks
- [ ] Web interface

## Privacy

Personal photo libraries contain highly sensitive data.

Recall is being designed with a **local-first** philosophy. Wherever practical, photo indexing, embeddings, metadata, and retrieval should remain on the user's machine.

Original photos should be treated as read-only and should never be modified, moved, or deleted by the indexing pipeline.

## Why "Recall"?

Photos are more than files—they're fragments of journeys we've already lived.

Recall is about finding the moments that were captured but no longer remembered: the people, places, and ordinary experiences whose significance may only become clear later.

The goal isn't just to search photos.

It's to **remember what was there**.

## License

This project is currently under development.