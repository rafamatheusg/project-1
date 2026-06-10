"""
chunk.py — Milestone 3: Chunking pipeline.

Strategy: review-level chunking.
Each document contains numbered reviews ("Review 1:", "Review 2:", ...).
Each review is one chunk — reviews are self-contained opinion units that can
answer questions independently. We prepend a metadata header to each chunk so
the embedding model has context about who and what course is being described.

Chunk size: ~300–500 characters (natural review length)
Overlap: none — reviews are independent; overlap would duplicate content

Run: python chunk.py
"""

import os
import json
import re

CLEANED_DIR = "data/cleaned"
CHUNKS_DIR = "data/chunks"


def split_into_reviews(text: str) -> list[str]:
    """
    Split document text on 'Review N:' markers.
    Returns a list of review text strings (without the 'Review N:' prefix).
    """
    # Match "Review 1:", "Review 2:", etc. (possibly with extra whitespace)
    pattern = re.compile(r"\bReview\s+\d+:", re.IGNORECASE)
    parts = pattern.split(text)

    reviews = []
    for part in parts:
        stripped = part.strip()
        if stripped:  # skip empty parts before first review
            reviews.append(stripped)
    return reviews


def make_chunk(review_text: str, meta: dict, review_num: int) -> dict:
    """
    Build a chunk dict with the review text prepended by a metadata header.
    The header gives the embedding model context so semantic search works
    even for queries that mention professor name or course.
    """
    header = (
        f"[Professor: {meta['professor']} | "
        f"Course: {meta['course']} | "
        f"Source: {meta['source']}]\n\n"
    )
    full_text = header + review_text

    return {
        "chunk_id": f"{meta['source']}__review_{review_num}",
        "source": meta["source"],
        "professor": meta["professor"],
        "course": meta["course"],
        "review_num": review_num,
        "text": full_text,
        "char_count": len(full_text),
    }


def chunk_document(doc: dict) -> list[dict]:
    reviews = split_into_reviews(doc["text"])
    chunks = []
    for i, review_text in enumerate(reviews, start=1):
        if len(review_text.strip()) < 20:
            continue  # skip fragments
        chunk = make_chunk(review_text, doc, i)
        chunks.append(chunk)
    return chunks


def inspect_chunks(chunks: list[dict], n: int = 5) -> None:
    """Print n representative chunks for visual inspection."""
    import random
    sample = random.sample(chunks, min(n, len(chunks)))
    print("\n" + "=" * 60)
    print(f"CHUNK INSPECTION — {n} representative chunks")
    print("=" * 60)
    for chunk in sample:
        print(f"\nChunk ID : {chunk['chunk_id']}")
        print(f"Char count: {chunk['char_count']}")
        print(f"Text:\n{chunk['text'][:500]}")
        print("-" * 40)


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    corpus_path = os.path.join(CLEANED_DIR, "corpus.json")
    if not os.path.exists(corpus_path):
        print(f"corpus.json not found at {corpus_path} — run ingest.py first")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Chunking {len(corpus)} documents...\n")

    all_chunks = []
    for doc in corpus:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Chunk stats
    lengths = [c["char_count"] for c in all_chunks]
    print(f"Chunk size — min: {min(lengths)}, max: {max(lengths)}, "
          f"avg: {sum(lengths)//len(lengths)} chars")

    # Save
    out_path = os.path.join(CHUNKS_DIR, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_chunks)} chunks to {out_path}")

    # Inspect 5 random chunks
    inspect_chunks(all_chunks, n=5)

    # Verify no bad chunks
    empty = [c for c in all_chunks if len(c["text"].strip()) < 50]
    if empty:
        print(f"\n⚠ WARNING: {len(empty)} chunks seem too short — inspect them:")
        for c in empty:
            print(f"  {c['chunk_id']}: '{c['text'][:80]}'")
    else:
        print("\n✓ All chunks passed basic length check (≥50 chars)")


if __name__ == "__main__":
    main()
