"""
embed.py — Milestone 4: Embedding and vector store.

Loads chunks from data/chunks/chunks.json, embeds each using
all-MiniLM-L6-v2 (sentence-transformers), and stores them in a local
persistent ChromaDB collection with source metadata.

Run: python embed.py
"""

import os
import json

from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_PATH = "data/chunks/chunks.json"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "professor_reviews"

# Embedding model: all-MiniLM-L6-v2
# - Runs locally, no API key
# - 384-dim embeddings, max 256 tokens
# - Strong semantic similarity on short opinion text
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if re-running (avoids duplicates)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "professor": c["professor"],
            "course": c["course"],
            "review_num": c["review_num"],
        }
        for c in chunks
    ]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Storing in ChromaDB...")
    # Add in batches to avoid memory issues with large corpora
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"  Stored chunks {i+1}–{batch_end}")

    return collection


def test_retrieval(collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """
    Test retrieval with 3 of the 5 evaluation questions to verify
    chunks returned are relevant before adding generation.
    """
    test_queries = [
        "Does Dr. Johnson curve her exams?",
        "What are Professor Martinez's office hours?",
        "Are calculators allowed on Professor Thompson's exams?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL VERIFICATION — 3 eval queries")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        q_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=4,
            include=["documents", "metadatas", "distances"],
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"\n  Result {i+1} | distance: {dist:.4f} | source: {meta['source']}")
            print(f"  {doc[:200].replace(chr(10), ' ')}...")

        top_dist = results["distances"][0][0]
        if top_dist < 0.5:
            print(f"\n  ✓ Good match (distance {top_dist:.3f} < 0.5)")
        else:
            print(f"\n  ⚠ Weak match (distance {top_dist:.3f} ≥ 0.5) — consider adjusting chunks")


def main():
    if not os.path.exists(CHUNKS_PATH):
        print(f"chunks.json not found at {CHUNKS_PATH} — run chunk.py first")
        return

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    collection = build_vector_store(chunks)
    print(f"\n✓ Vector store built: {collection.count()} embeddings stored in {CHROMA_DIR}")

    # Reload model for testing (already downloaded)
    model = SentenceTransformer(MODEL_NAME)
    test_retrieval(collection, model)


if __name__ == "__main__":
    main()
