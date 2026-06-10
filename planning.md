# Planning

## Domain

Professor and course reviews at State University. Students write candid reviews of professors but this knowledge is scattered and hard to search. Official channels don't publish it. This system makes it queryable.

## Documents

10 text files, one per professor, each with 5 student reviews:

- `prof_johnson_cs101.txt` — Dr. Sarah Johnson, CS101
- `prof_martinez_cs201.txt` — Dr. Carlos Martinez, CS201
- `prof_lee_cs101.txt` — Dr. Emily Lee, CS101
- `prof_thompson_math201.txt` — Dr. Robert Thompson, MATH201
- `prof_chen_phys101.txt` — Dr. Wei Chen, PHYS101
- `prof_okafor_eng110.txt` — Dr. Amara Okafor, ENG110
- `prof_park_econ101.txt` — Dr. James Park, ECON101
- `prof_brennan_psyc101.txt` — Dr. Lisa Brennan, PSYC101
- `prof_torres_chem101.txt` — Dr. Michael Torres, CHEM101
- `prof_nguyen_hist110.txt` — Dr. Thanh Nguyen, HIST110

## Chunking Strategy

Each document is split on `Review N:` markers — one chunk per review. Reviews are self-contained opinion units (3–8 sentences) that can answer a question on their own. Splitting by fixed character count would break sentences mid-thought and lose meaning.

- Chunk size: ~300–500 characters (natural review length)
- Overlap: none — reviews are independent, overlap would just duplicate content
- Each chunk gets a metadata header: professor name, course, source filename

## Retrieval Approach

- Embedding model: `all-MiniLM-L6-v2` via sentence-transformers (local, no API key)
- Vector store: ChromaDB (local persistent)
- Top-k: 4 chunks per query

Tradeoffs: MiniLM is free and fast but caps at 256 tokens. For longer documents or multilingual text, `bge-large-en-v1.5` or OpenAI's embedding API would be better choices. For this corpus of short reviews, MiniLM is the right call.

## Evaluation Plan

| #   | Question                                               | Expected Answer                                           |
| --- | ------------------------------------------------------ | --------------------------------------------------------- |
| 1   | Does Dr. Johnson curve her exams?                      | Curves the midterm but NOT the final                      |
| 2   | What are Professor Martinez's office hours?            | MWF 10am–12pm, CS building room 215                       |
| 3   | What is the grade breakdown for Dr. Chen's PHYS101?    | HW 15%, clicker 5%, midterms 20% each, final 30%, lab 10% |
| 4   | Are calculators allowed on Professor Thompson's exams? | No — never                                                |
| 5   | Does Dr. Lee drop any homework grades?                 | Yes — the two lowest                                      |

## Anticipated Challenges

1. Both Johnson and Lee teach CS101. A vague query like "how hard is CS101?" may retrieve reviews from both, creating a confusing answer. The grounding prompt instructs the LLM to attribute answers to specific professors.
2. A question spanning multiple reviews ("what's the grading AND are exams hard?") may not retrieve all relevant chunks. Top-k=4 helps by returning multiple slots.

## AI Tool Plan

Used Claude to generate all pipeline scripts (`ingest.py`, `chunk.py`, `embed.py`, `query.py`, `app.py`). For each: provided the relevant planning section as context, reviewed the output, and corrected anything that didn't match the spec.

## Architecture

```
data/raw/*.txt
     │
     ▼  ingest.py — clean text, extract metadata
     │
     ▼  chunk.py — split on Review N: markers, prepend metadata header
     │
     ▼  embed.py — all-MiniLM-L6-v2 → ChromaDB
     │
     ▼  query.py — embed query → top-4 chunks → Groq LLaMA-3.3-70b
     │
     ▼  app.py — Gradio UI at localhost:7860
```
