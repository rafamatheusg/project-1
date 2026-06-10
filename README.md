# Professor Review RAG System

A RAG system that answers questions about professors at State University using student review documents. Ask things like "Does Professor Johnson curve?" and get a grounded answer with source citations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your Groq API key inside
```

## Run the pipeline

```bash
python ingest.py    # clean the documents
python chunk.py     # split into chunks
python embed.py     # embed + store in ChromaDB (downloads model ~90MB first time)
python app.py
```

To test from the command line: `python query.py`  
To run the evaluation: `python evaluate.py`

## How it works

10 professor review documents → cleaned → split into per-review chunks → embedded with `all-MiniLM-L6-v2` → stored in ChromaDB. At query time, the top 4 most similar chunks are retrieved and passed to Groq's LLaMA-3.3-70b with a grounding prompt that forces it to answer only from the retrieved context.

## Evaluation

| Question                                    | Expected                                                  | Accuracy          |
| ------------------------------------------- | --------------------------------------------------------- | ----------------- |
| Does Dr. Johnson curve her exams?           | Curves midterm, not final                                 | run `evaluate.py` |
| What are Professor Martinez's office hours? | MWF 10am–12pm, room 215                                   | run `evaluate.py` |
| Grade breakdown for Dr. Chen's PHYS101?     | HW 15%, clicker 5%, midterms 20% each, final 30%, lab 10% | run `evaluate.py` |
| Calculators allowed on Thompson's exams?    | No — never                                                | run `evaluate.py` |
| Does Dr. Lee drop homework grades?          | Yes — two lowest                                          | run `evaluate.py` |

**Known failure case:** Questions about curving (Q1) sometimes come back partially accurate. The relevant detail — "curves the midterm but NOT the final" — lives in one specific review. If a different Johnson review scores slightly higher in retrieval, the LLM gets the first half right but misses the "not the final" distinction. This is a retrieval ranking issue, not a generation issue.

## Spec reflection

The spec's rule to test retrieval before adding generation was the most useful constraint. Running `embed.py`'s retrieval test confirmed all top distances were under 0.35 before the LLM was introduced, which made later debugging much easier.

The implementation uses `gr.Blocks` instead of the spec's suggested `gr.Interface` in order to add a debug panel showing raw retrieved chunks. This makes it easier to distinguish retrieval failures from generation failures during the demo.

## AI tool usage

- Asked Claude to generate all five pipeline scripts, providing the relevant `planning.md` section as context for each
- The initial grounding prompt used "try to answer only from context" — changed to explicit numbered rules with a required refusal phrase to prevent well-spoken hallucinations
- The chunk preview length in `chunk.py` was changed from 400 to 500 characters after noticing that key facts in some reviews appeared late in the paragraph

## Project structure

```
data/raw/       — 10 professor review documents
data/cleaned/   — cleaned JSON output from ingest.py
data/chunks/    — chunk list from chunk.py
data/chroma/    — ChromaDB vector store (auto-created)
ingest.py       — document cleaning
chunk.py        — review-level chunking
embed.py        — embedding + vector store
query.py        — retrieval + generation
app.py          — Gradio UI
evaluate.py     — evaluation report
planning.md     — design decisions
```
