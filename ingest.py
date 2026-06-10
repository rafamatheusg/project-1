"""
ingest.py — Milestone 3: Document ingestion and cleaning pipeline.

Loads raw .txt professor review files, extracts metadata from the header,
cleans the text (strips extra whitespace, normalizes line endings), and saves
structured JSON to data/cleaned/.

Run: python ingest.py
"""

import os
import json
import re

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"


def load_raw(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_metadata(text: str) -> dict:
    """
    Pull Professor, Course, Department, University from the header lines.
    Returns a dict with those fields (empty string if not found).
    """
    meta = {"professor": "", "course": "", "department": "", "university": ""}
    patterns = {
        "professor": r"^Professor:\s*(.+)$",
        "course": r"^Course:\s*(.+)$",
        "department": r"^Department:\s*(.+)$",
        "university": r"^University:\s*(.+)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            meta[key] = match.group(1).strip()
    return meta


def clean_text(text: str) -> str:
    """
    Clean the document:
    - Normalize Windows line endings
    - Collapse 3+ blank lines to 2
    - Strip leading/trailing whitespace from each line
    - Remove the metadata header block (first 4 lines)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Skip the header block (Professor:, Course:, Department:, University:, blank line)
    content_lines = []
    header_done = False
    header_count = 0
    for line in lines:
        stripped = line.strip()
        if not header_done:
            if any(stripped.startswith(p) for p in ["Professor:", "Course:", "Department:", "University:"]):
                header_count += 1
                continue
            elif header_count > 0 and stripped == "":
                continue  # blank line right after header
            else:
                header_done = True
        content_lines.append(stripped)

    # Collapse multiple blank lines
    cleaned_lines = []
    blank_count = 0
    for line in content_lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def process_file(filename: str) -> dict:
    filepath = os.path.join(RAW_DIR, filename)
    raw = load_raw(filepath)
    meta = extract_metadata(raw)
    cleaned = clean_text(raw)
    return {
        "source": filename,
        "professor": meta["professor"],
        "course": meta["course"],
        "department": meta["department"],
        "university": meta["university"],
        "text": cleaned,
    }


def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
    if not raw_files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    print(f"Found {len(raw_files)} documents to ingest...\n")

    all_docs = []
    for filename in sorted(raw_files):
        doc = process_file(filename)
        all_docs.append(doc)

        # Save individual cleaned file
        out_path = os.path.join(CLEANED_DIR, filename.replace(".txt", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        print(f"  ✓ {filename}")
        print(f"    Professor : {doc['professor']}")
        print(f"    Course    : {doc['course']}")
        print(f"    Text len  : {len(doc['text'])} chars")

        # Verification: print first 200 chars of cleaned text
        preview = doc["text"][:200].replace("\n", " ")
        print(f"    Preview   : {preview}...")
        print()

    # Save combined corpus
    combined_path = os.path.join(CLEANED_DIR, "corpus.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_docs)} cleaned documents to {CLEANED_DIR}/")
    print(f"Combined corpus: {combined_path}")


if __name__ == "__main__":
    main()
