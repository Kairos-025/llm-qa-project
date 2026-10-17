"""
Phase 5: runs RAG on the test set, saves predictions, then scores them with
the same scorer as every other method. Works like run_inference.py (same
model loader, batching, resume after disconnects, run log), but each question
gets its top-k RETRIEVED passages instead of the correct passage.

Run `python -m src.rag build` first (once per Colab session).

Usage:
    python -m src.run_rag --k 3 --limit 20   # quick test
    python -m src.run_rag --k 1              # full runs: k = 1, 3, 5
    python -m src.run_rag --k 3
    python -m src.run_rag --k 5

Saves results/rag_k<k>_preds.jsonl, _scores.json, _prompt_example.txt and a
line in results/run_log.jsonl, so compare_results.py picks RAG up automatically.
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from src.evaluate_qa import evaluate, load_jsonl
from src.model_utils import DEFAULT_MODEL, generate_batch, load_model
from src.rag import attach_retrieved, rag_prompt
from src.run_inference import log_run, run_totals

MAX_NEW_TOKENS = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="number of retrieved passages (1, 3 or 5)")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = ap.parse_args()

    method = f"rag_k{args.k}"
    tag = method + (f"_sample{args.limit}" if args.limit else "")
    os.makedirs("results", exist_ok=True)
    preds_path = f"results/{tag}_preds.jsonl"
    scores_path = f"results/{tag}_scores.json"

    test = load_jsonl(args.test)
    if args.limit:
        test = test[: args.limit]
    test = attach_retrieved(test)

    done = {}
    if os.path.exists(preds_path):
        done = {r["id"]: r for r in load_jsonl(preds_path)}
        print(f"Resuming: {len(done)} of {len(test)} already done")
    todo = [ex for ex in test if ex["id"] not in done]
    # similar-length prompts batch together, which wastes less GPU time on padding
    todo.sort(key=lambda ex: len(" ".join(ex["retrieved"][: args.k])))

    if todo:
        print(f"Loading {args.model} ...")
        model, tok = load_model(args.model)
        with open(f"results/{tag}_prompt_example.txt", "w", encoding="utf-8") as f:
            f.write(tok.apply_chat_template(rag_prompt(test[0], args.k), tokenize=False, add_generation_prompt=True))
        start = time.time()
        with open(preds_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=method):
                batch = todo[i : i + args.batch_size]
                answers = generate_batch(model, tok, [rag_prompt(ex, args.k) for ex in batch], MAX_NEW_TOKENS)
                for ex, ans in zip(batch, answers):
                    f.write(json.dumps({"id": ex["id"], "prediction": ans, "retrieved": ex["retrieved_ids"][: args.k]},
                                       ensure_ascii=False) + "\n")
                f.flush()
        minutes = (time.time() - start) / 60
        print(f"Generation took {minutes:.1f} min")
        log_run(tag, args, len(todo), minutes, MAX_NEW_TOKENS)

    scores, per_example = evaluate(test, load_jsonl(preds_path))
    scores["method"] = method
    scores["model"] = args.model
    scores["k"] = args.k
    scores.update(run_totals(tag))
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    with open(scores_path.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(scores, indent=2))
    print(f"\nSaved {preds_path} and {scores_path}")


if __name__ == "__main__":
    main()
