"""
query.py — Milestone 5: Retrieval + grounded generation.

Retrieves top-4 chunks from ChromaDB using semantic similarity,
then sends them as context to Groq's LLaMA-3.3-70b-versatile.
The system prompt enforces grounding: the LLM must answer only
from the retrieved context, not from general training knowledge.
Source attribution is appended programmatically.

Usage:
    from query import ask
    result = ask("Does Professor Johnson curve her exams?")
    print(result["answer"])
    print(result["sources"])

Or run directly: python query.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "professor_reviews"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

# Grounding system prompt — explicitly instructs the LLM to answer ONLY
# from the retrieved context, not from general knowledge.
SYSTEM_PROMPT = """You are a helpful assistant for students at State University.
You answer questions about professors and courses using ONLY the information
provided in the retrieved review excerpts below.

STRICT RULES:
1. Answer ONLY using information from the provided context. Do not use your general
   training knowledge about universities, professors, or academic conventions.
2. If the context does not contain enough information to answer the question,
   say exactly: "I don't have enough information in my sources to answer that."
3. If reviews in the context disagree with each other, acknowledge both perspectives.
4. Be specific — quote or closely paraphrase the review content when relevant.
5. Do NOT make up or infer details not present in the context.
6. Keep answers concise (2–5 sentences unless the question requires more detail)."""


# Singleton model and collection — loaded once per process
_model = None
_collection = None
_groq_client = None


def _get_resources():
    global _model, _collection, _groq_client

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. Run embed.py first."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _groq_client = Groq(api_key=api_key)

    return _model, _collection, _groq_client


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Embed the query and retrieve top-k chunks from ChromaDB.
    Returns list of dicts with 'text', 'source', 'professor', 'course', 'distance'.
    """
    model, collection, _ = _get_resources()
    q_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "professor": meta.get("professor", ""),
            "course": meta.get("course", ""),
            "distance": round(dist, 4),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Excerpt {i} (from {chunk['source']}, distance: {chunk['distance']}) ---\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> str:
    """Send context + query to Groq and return the generated answer."""
    _, _, groq_client = _get_resources()

    context = build_context(chunks)
    user_message = f"RETRIEVED CONTEXT:\n{context}\n\nQUESTION: {query}"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # low temperature for factual, grounded answers
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def ask(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → generate → return answer + sources.

    Returns:
        {
            "answer": str,
            "sources": list[str],      # unique source filenames
            "chunks": list[dict],      # raw retrieved chunks (for debugging)
        }
    """
    chunks = retrieve(query)
    answer = generate(query, chunks)

    # Programmatic source attribution — collect unique sources from retrieved chunks
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


def run_cli():
    """Simple CLI for testing queries interactively."""
    print("Professor Review RAG System")
    print("Ask questions about professors and courses at State University.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        print("\nRetrieving relevant reviews...")
        result = ask(query)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"\n(Retrieved {len(result['chunks'])} chunks, "
              f"top distance: {result['chunks'][0]['distance']})")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    run_cli()
