"""
Phase 2: build every data file the project uses. Run once, then commit the
outputs to Drive (not GitHub; they are large).

Usage:
    python -m src.data_prep --test_size 1000

Creates:
    data/test.jsonl          fixed test set, used by ALL four methods (from SQuAD 2.0 validation)
    data/train.jsonl         fine-tuning data (from SQuAD 2.0 train)
    data/fewshot_pool.jsonl  examples for few-shot prompts (from train, never from test)
    data/corpus.jsonl        passages the RAG system searches (all validation contexts)
"""
import argparse
import json
import os
import random

from datasets import load_dataset

SEED = 42


def to_record(ex, context_ids):
    answers = list(ex["answers"]["text"])
    return {
        "id": ex["id"],
        "title": ex["title"],
        "question": ex["question"].strip(),
        "context": ex["context"],
        "context_id": context_ids.get(ex["context"]),
        "answers": answers,
        "is_impossible": len(answers) == 0,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>6} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--fewshot_size", type=int, default=20)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("rajpurkar/squad_v2")
    train, val = ds["train"], ds["validation"]

    # RAG corpus: every unique passage in the validation split
    context_ids = {}
    for c in val["context"]:
        context_ids.setdefault(c, f"ctx_{len(context_ids):05d}")
    title_of = {c: t for c, t in zip(val["context"], val["title"])}
    corpus = [{"context_id": cid, "title": title_of[c], "text": c} for c, cid in context_ids.items()]

    # Fixed test set (same questions for every method)
    test = val.shuffle(seed=SEED).select(range(min(args.test_size, len(val))))
    test_rows = [to_record(ex, context_ids) for ex in test]

    # Training data for fine-tuning (full train split; subsample later if needed)
    train_rows = [to_record(ex, {}) for ex in train]

    # Few-shot pool: half answerable, half unanswerable, from train only
    rng = random.Random(SEED)
    ans = [r for r in train_rows if not r["is_impossible"] and len(r["context"]) < 800]
    unans = [r for r in train_rows if r["is_impossible"] and len(r["context"]) < 800]
    k = args.fewshot_size // 2
    fewshot = rng.sample(ans, k) + rng.sample(unans, k)
    rng.shuffle(fewshot)

    write_jsonl(f"{args.out_dir}/test.jsonl", test_rows)
    write_jsonl(f"{args.out_dir}/train.jsonl", train_rows)
    write_jsonl(f"{args.out_dir}/fewshot_pool.jsonl", fewshot)
    write_jsonl(f"{args.out_dir}/corpus.jsonl", corpus)

    n_imp = sum(r["is_impossible"] for r in test_rows)
    print(f"\nTest set: {len(test_rows)} questions "
          f"({len(test_rows) - n_imp} answerable, {n_imp} unanswerable)")
    print(f"RAG corpus: {len(corpus)} passages")


if __name__ == "__main__":
    main()
