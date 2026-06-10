"""
app.py — Milestone 5: Gradio query interface.

Provides a web UI for querying the professor review RAG system.
Run: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from query import ask


def handle_query(question: str):
    """
    Called by Gradio on submit. Returns (answer_text, sources_text, debug_text).
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        result = ask(question)
    except RuntimeError as e:
        return f"Error: {e}", "", ""

    answer = result["answer"]

    sources_lines = [f"• {s}" for s in result["sources"]]
    sources_text = "\n".join(sources_lines) if sources_lines else "No sources retrieved."

    # Debug panel: show chunks with distances
    debug_lines = []
    for i, chunk in enumerate(result["chunks"], 1):
        debug_lines.append(
            f"Chunk {i} | {chunk['source']} | distance: {chunk['distance']}\n"
            f"{chunk['text'][:300]}..."
        )
    debug_text = "\n\n---\n\n".join(debug_lines)

    return answer, sources_text, debug_text


example_questions = [
    "Does Dr. Johnson curve her exams?",
    "What are Professor Martinez's office hours?",
    "Is CS101 with Professor Lee easier than with Professor Johnson?",
    "Are calculators allowed on Professor Thompson's exams?",
    "What is the grade breakdown for Dr. Chen's physics course?",
    "Does Dr. Lee drop any homework grades?",
    "How hard is CS201 with Martinez?",
    "What do students say about Dr. Brennan's psychology class?",
]

with gr.Blocks(title="Professor Review RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 Professor Review Assistant
        Ask questions about professors and courses at State University.
        Answers are grounded in real student reviews and include source citations.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. Does Professor Martinez curve his exams?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Example questions:**")
            for example in example_questions[:4]:
                gr.Button(example, size="sm").click(
                    fn=lambda q=example: q,
                    outputs=question_box,
                )

    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Answer",
                lines=6,
                interactive=False,
            )
        with gr.Column(scale=1):
            sources_box = gr.Textbox(
                label="Retrieved from",
                lines=6,
                interactive=False,
            )

    with gr.Accordion("Debug: retrieved chunks", open=False):
        debug_box = gr.Textbox(
            label="Raw retrieved chunks (for inspection)",
            lines=15,
            interactive=False,
        )

    submit_btn.click(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )
    question_box.submit(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )

    gr.Markdown(
        """
        ---
        *Answers are grounded in the retrieved review excerpts only.
        If the system says it doesn't have enough information, that question
        is outside the coverage of the review corpus.*
        """
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)
