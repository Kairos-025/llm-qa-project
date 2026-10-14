"""
All prompt templates live here. Each template takes one test example and
returns a chat (list of messages) for the model.

Phase 3: baseline (plain zero-shot).
Phase 4: engineered prompts that test instructions, few-shot examples,
structured output and step-by-step (evidence-first) reasoning.
"""
import json
from functools import partial

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
INSTRUCTIONS = (
    "Read the context and answer the question.\n"
    "Rules:\n"
    "- Copy the answer exactly as it appears in the context, using as few words as possible.\n"
    "- Do not write a full sentence and do not explain.\n"
    "- If the context does not contain the answer, reply with exactly: unanswerable"
)


def _qa_block(ex):
    return f"Context: {ex['context']}\n\nQuestion: {ex['question']}"


def _gold(ex):
    """Answer used in few-shot demonstrations."""
    return ex["answers"][0] if ex["answers"] else "unanswerable"


_POOL = None


def _fewshot_examples(k, path="data/fewshot_pool.jsonl"):
    """Pick k demonstration examples from the training-split pool, alternating
    answerable and unanswerable (k=1: A, k=3: A U A, k=5: A U A U A).
    The same examples are used for every test question, so runs are comparable."""
    global _POOL
    if _POOL is None:
        with open(path, encoding="utf-8") as f:
            _POOL = [json.loads(line) for line in f if line.strip()]
    ans = [r for r in _POOL if not r["is_impossible"]]
    unans = [r for r in _POOL if r["is_impossible"]]
    chosen, i, j = [], 0, 0
    while len(chosen) < k:
        if len(chosen) % 2 == 0:
            chosen.append(ans[i]); i += 1
        else:
            chosen.append(unans[j]); j += 1
    return chosen


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
def baseline(example, **_):
    """Plain zero-shot prompt: no rules, no examples, and no mention that some
    questions have no answer. This is the 'model as-is' starting point."""
    return [{
        "role": "user",
        "content": (
            "Answer the question based on the context.\n\n"
            f"Context: {example['context']}\n\n"
            f"Question: {example['question']}"
        ),
    }]


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------
def instructed(example, **_):
    """Zero-shot with clear rules: short extractive answers, and permission
    to say 'unanswerable'. Tests the effect of instructions alone."""
    return [{"role": "user", "content": INSTRUCTIONS + "\n\n" + _qa_block(example)}]


def fewshot(example, k=3, **_):
    """Instructions plus k worked examples, given as earlier chat turns so the
    model sees exactly what a good answer looks like."""
    messages = []
    for n, demo in enumerate(_fewshot_examples(k)):
        content = (INSTRUCTIONS + "\n\n" if n == 0 else "") + _qa_block(demo)
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": _gold(demo)})
    messages.append({"role": "user", "content": _qa_block(example)})
    return messages


def structured(example, **_):
    """Instructions plus a fixed output format, which makes answers easier
    to extract and score."""
    return [{
        "role": "user",
        "content": (
            INSTRUCTIONS + "\n\n"
            "Reply in exactly this format and nothing else:\n"
            "Answer: <answer or unanswerable>\n\n"
            + _qa_block(example)
        ),
    }]


def cot(example, **_):
    """Step-by-step (evidence-first) prompt: the model first quotes the
    relevant sentence, then gives the answer. A short form of
    chain-of-thought suited to extractive QA. Produces longer outputs."""
    return [{
        "role": "user",
        "content": (
            INSTRUCTIONS + "\n\n"
            "Work in two steps and use exactly this format:\n"
            "Evidence: <copy the one sentence from the context that answers the question, or None>\n"
            "Answer: <shortest exact answer, or unanswerable>\n\n"
            + _qa_block(example)
        ),
    }]


PROMPTS = {
    "baseline": baseline,
    "instructed": instructed,
    "fewshot1": partial(fewshot, k=1),
    "fewshot3": partial(fewshot, k=3),
    "fewshot5": partial(fewshot, k=5),
    "structured": structured,
    "cot": cot,
}

# Most methods need only a short answer; the step-by-step prompt writes more.
MAX_NEW_TOKENS = {"cot": 150}
DEFAULT_MAX_NEW_TOKENS = 48
