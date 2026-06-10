"""
evaluate.py — Milestone 6: Evaluation report.

Runs all 5 evaluation questions against the RAG system,
compares responses to ground-truth expected answers,
and prints a formatted evaluation report.

Run: python evaluate.py
Output is printed to console and saved to evaluation_report.md
"""

from query import ask

EVAL_QUESTIONS = [
    {
        "question": "Does Dr. Johnson curve her exams?",
        "expected": "She curves the midterm but NOT the final.",
        "key_facts": ["curves midterm", "does not curve final", "not the final"],
    },
    {
        "question": "What are Professor Martinez's office hours?",
        "expected": "Monday, Wednesday, Friday 10am-12pm in CS building room 215.",
        "key_facts": ["monday", "wednesday", "friday", "10am", "12pm", "room 215"],
    },
    {
        "question": "What is the grade breakdown for Dr. Chen's PHYS101?",
        "expected": "Homework 15%, clicker participation 5%, 2 midterms 20% each, final 30%, lab 10%.",
        "key_facts": ["15%", "5%", "20%", "30%", "10%"],
    },
    {
        "question": "Are calculators allowed on Professor Thompson's exams?",
        "expected": "No — no calculators allowed on any exam, ever.",
        "key_facts": ["no calculator", "not allowed", "never"],
    },
    {
        "question": "Does Dr. Lee drop any homework grades?",
        "expected": "Yes — she drops the two lowest homework grades.",
        "key_facts": ["drops", "two lowest", "2 lowest"],
    },
]


def judge_accuracy(response: str, key_facts: list[str]) -> str:
    """
    Simple heuristic accuracy judgment.
    Checks how many key facts appear in the response text (case-insensitive).
    Returns 'accurate', 'partially accurate', or 'inaccurate'.
    """
    r = response.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in r)
    ratio = hits / len(key_facts)
    if ratio >= 0.6:
        return "accurate"
    elif ratio >= 0.3:
        return "partially accurate"
    else:
        return "inaccurate"


def run_evaluation() -> list[dict]:
    results = []
    for i, eval_item in enumerate(EVAL_QUESTIONS, 1):
        print(f"Running question {i}/{len(EVAL_QUESTIONS)}: {eval_item['question'][:60]}...")
        result = ask(eval_item["question"])
        accuracy = judge_accuracy(result["answer"], eval_item["key_facts"])
        results.append({
            "question": eval_item["question"],
            "expected": eval_item["expected"],
            "actual": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"],
            "accuracy": accuracy,
        })
    return results


def format_report(results: list[dict]) -> str:
    lines = [
        "# Evaluation Report — Professor Review RAG System",
        "",
        "## Summary",
        "",
    ]

    counts = {"accurate": 0, "partially accurate": 0, "inaccurate": 0}
    for r in results:
        counts[r["accuracy"]] += 1

    lines += [
        f"- Accurate: {counts['accurate']}/5",
        f"- Partially accurate: {counts['partially accurate']}/5",
        f"- Inaccurate: {counts['inaccurate']}/5",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ]

    for i, r in enumerate(results, 1):
        accuracy_emoji = {"accurate": "✅", "partially accurate": "⚠️", "inaccurate": "❌"}
        lines += [
            f"### Question {i}: {accuracy_emoji[r['accuracy']]} {r['accuracy'].upper()}",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Expected answer:** {r['expected']}",
            "",
            f"**System response:**",
            f"> {r['actual']}",
            "",
            f"**Sources retrieved:** {', '.join(r['sources'])}",
            "",
            f"**Chunks retrieved ({len(r['chunks'])}):**",
        ]
        for j, chunk in enumerate(r["chunks"], 1):
            lines.append(
                f"  - Chunk {j}: `{chunk['source']}` (distance: {chunk['distance']})"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Failure analysis section
    lines += [
        "## Failure Analysis",
        "",
        "Identify at least one case where retrieval or generation didn't work as expected.",
        "",
    ]

    failures = [r for r in results if r["accuracy"] in ("partially accurate", "inaccurate")]
    if failures:
        for r in failures[:2]:  # Document up to 2 failures
            top_dist = r["chunks"][0]["distance"] if r["chunks"] else 1.0
            lines += [
                f"**Question:** {r['question']}",
                f"**Accuracy:** {r['accuracy']}",
                "",
                f"**Observed failure:**",
                f"The system returned '{r['actual'][:200]}...' but the expected answer was '{r['expected']}'.",
                "",
                f"**Root cause analysis:**",
                f"Top retrieved chunk distance was {top_dist:.3f}. "
                + ("The relevant information may have been split across chunk boundaries, "
                   "causing the retrieval to return only partial context. "
                   if top_dist > 0.4 else
                   "Despite good retrieval (low distance), the LLM may have rephrased "
                   "the answer in a way that missed a key specific detail (e.g. the exact "
                   "days/times or the distinction between midterm vs final curves). "
                   "This is a generation grounding failure, not a retrieval failure.")
                + " Possible fix: add more reviews covering this specific fact, "
                  "or adjust the grounding prompt to be more specific about preserving exact details.",
                "",
                "---",
                "",
            ]
    else:
        lines += [
            "All 5 evaluation questions were answered accurately. "
            "Note: this may indicate evaluation questions are too straightforward. "
            "Consider adding adversarial questions (e.g. asking about a professor not in the corpus, "
            "or asking for a comparison that requires synthesizing across multiple reviews).",
            "",
        ]

    return "\n".join(lines)


def main():
    print("Running evaluation on 5 test questions...\n")
    results = run_evaluation()

    report = format_report(results)

    # Print to console
    print("\n" + report)

    # Save to file
    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nEvaluation report saved to evaluation_report.md")


if __name__ == "__main__":
    main()
