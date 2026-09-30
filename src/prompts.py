"""
All prompt templates live here. Each template takes one test example and
returns a chat (list of messages) for the model.

Phase 3 adds only the baseline. Phase 4 adds the engineered prompts.
"""


def baseline(example, **_):
    """Plain zero-shot prompt: the model gets the passage and the question,
    with no special instructions, no examples, and no mention that some
    questions have no answer. This is the 'model as-is' starting point."""
    return [{
        "role": "user",
        "content": (
            "Answer the question based on the context.\n\n"
            f"Context: {example['context']}\n\n"
            f"Question: {example['question']}"
        ),
    }]


PROMPTS = {
    "baseline": baseline,
}
