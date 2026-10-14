"""
Shared evaluation script. EVERY method (baseline, prompting, RAG, fine-tuned)
is scored with this file so the comparison is fair.

Usage:
    python -m src.evaluate_qa --preds results/baseline_preds.jsonl \
                              --gold data/test.jsonl \
                              --out results/baseline_scores.json

Prediction file format (one JSON object per line):
    {"id": "<squad id>", "prediction": "<raw model output>"}

Metrics:
    exact_match, f1            official SQuAD 2.0 style, all questions
    has_ans_em, has_ans_f1     answerable questions only
    no_ans_accuracy            unanswerable questions: did the model abstain?
    lenient_accuracy           gold answer appears inside the prediction
    bleu, rouge_l              answerable questions only (undefined for empty answers);
                               BLEU is sentence-level, averaged (see bleu_rouge)
"""
import argparse
import collections
import json
import re
import string

# ---------------------------------------------------------------------------
# "No answer" rule (project decision): the model is told to reply exactly
# "unanswerable". Any of these phrasings also count as abstaining.
# ---------------------------------------------------------------------------
NO_ANSWER_PATTERNS = [
    r"\bunanswerable\b",
    r"\bnot answerable\b",
    r"\bno answer\b",
    r"^none$",
    r"^n/?a$",
    r"cannot be answered",
    r"can't be answered",
    r"not (mentioned|stated|provided|given|found) in the (context|passage|text)",
    r"(context|passage|text) does not (contain|mention|provide|say)",
    r"^i (do not|don't) know$",
]

# A line like "Answer: Paris" or "Final answer: Paris"
ANSWER_LINE = re.compile(r"^(final answer|answer)\s*[:\-]\s*", re.IGNORECASE)
ANSWER_PREFIXES = re.compile(r"^(final answer|answer|a)\s*[:\-]\s*", re.IGNORECASE)


def clean_prediction(raw: str) -> str:
    """Turn a raw LLM output into a short answer string ('' means no answer).

    If the output contains a line starting with "Answer:" (structured and
    step-by-step prompts ask for this), the last such line is used.
    Otherwise the first non-empty line is used, since LLMs often add
    explanations after the answer.
    """
    if raw is None:
        return ""
    lines = [l.strip().strip("*#").strip() for l in raw.strip().splitlines() if l.strip()]
    answer_lines = [l for l in lines if ANSWER_LINE.match(l)]
    text = answer_lines[-1] if answer_lines else (lines[0] if lines else "")
    text = ANSWER_PREFIXES.sub("", text).strip()
    text = text.strip(" \"'`*")
    if text.endswith("."):
        text = text[:-1]
    low = text.lower().strip()
    if low == "" or any(re.search(p, low) for p in NO_ANSWER_PATTERNS):
        return ""
    return text


# --- Official SQuAD 2.0 normalization and scoring --------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_tokens(s: str):
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_toks, pred_toks = get_tokens(gold), get_tokens(pred)
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def lenient_match(golds, pred: str) -> int:
    if not golds or golds == [""]:
        return int(pred == "")
    p = normalize_answer(pred)
    return int(bool(p) and any(normalize_answer(g) in p for g in golds if g))


# --- BLEU / ROUGE (answerable questions only) -------------------------------
def bleu_rouge(pairs):
    """pairs: list of (gold_answers, prediction). Returns (bleu, rouge_l) in 0-100.

    BLEU is computed per answer and averaged (sentence-level BLEU with
    effective n-gram order). Standard corpus BLEU needs matching 4-word
    sequences, so it scores a perfect one-word answer like "Paris" as 0,
    which would unfairly punish methods that give short, correct answers.
    """
    if not pairs:
        return None, None
    import sacrebleu
    from rouge_score import rouge_scorer

    preds = [p if p else "unanswerable" for _, p in pairs]
    bleu = sum(sacrebleu.sentence_bleu(p, golds).score
               for (golds, _), p in zip(pairs, preds)) / len(pairs)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = sum(
        max(scorer.score(g, p)["rougeL"].fmeasure for g in golds)
        for (golds, _), p in zip(pairs, preds)
    ) / len(pairs)
    return bleu, rouge * 100


# --- Main evaluation ---------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(gold_records, pred_records):
    preds = {r["id"]: clean_prediction(r.get("prediction", "")) for r in pred_records}
    missing = [g["id"] for g in gold_records if g["id"] not in preds]
    if missing:
        print(f"WARNING: {len(missing)} test questions have no prediction (scored as wrong answer).")

    em, f1, lenient = [], [], []
    has_em, has_f1, no_acc = [], [], []
    text_pairs = []
    per_example = []

    for g in gold_records:
        golds = [a for a in g["answers"] if normalize_answer(a)] or [""]
        pred = preds.get(g["id"], "<<missing>>")
        e = max(compute_exact(a, pred) for a in golds)
        f = max(compute_f1(a, pred) for a in golds)
        l = lenient_match(golds, pred)
        em.append(e); f1.append(f); lenient.append(l)
        if g["is_impossible"]:
            no_acc.append(int(pred == ""))
        else:
            has_em.append(e); has_f1.append(f)
            text_pairs.append((golds, pred))
        per_example.append({"id": g["id"], "question": g["question"], "gold": golds,
                            "prediction": pred, "em": e, "f1": round(f, 4)})

    avg = lambda xs: round(100 * sum(xs) / len(xs), 2) if xs else None
    bleu, rouge_l = bleu_rouge(text_pairs)
    scores = {
        "n_total": len(gold_records),
        "n_answerable": len(has_em),
        "n_unanswerable": len(no_acc),
        "exact_match": avg(em),
        "f1": avg(f1),
        "lenient_accuracy": avg(lenient),
        "has_ans_em": avg(has_em),
        "has_ans_f1": avg(has_f1),
        "no_ans_accuracy": avg(no_acc),
        "bleu": round(bleu, 2) if bleu is not None else None,
        "rouge_l": round(rouge_l, 2) if rouge_l is not None else None,
    }
    return scores, per_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", default="data/test.jsonl")
    ap.add_argument("--out", required=True, help="where to save the scores JSON")
    args = ap.parse_args()

    scores, per_example = evaluate(load_jsonl(args.gold), load_jsonl(args.preds))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    details = args.out.replace(".json", "_details.jsonl")
    with open(details, "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(scores, indent=2))
    print(f"Saved scores to {args.out} and per-question details to {details}")


if __name__ == "__main__":
    main()
