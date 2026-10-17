"""
Phase 6: scores the fine-tuned model on the same 1,000 test questions, with the
same structured prompt and scorer as every other method. Works like
run_inference.py (batching, resume after disconnects, run log), but loads the
LoRA adapter and saves under its own name, so no earlier result is overwritten.

Usage:
    python -m src.eval_finetuned --limit 20   # quick test
    python -m src.eval_finetuned              # full run

Saves results/finetuned_preds.jsonl, _scores.json, _prompt_example.txt and a
line in results/run_log.jsonl, so compare_results.py picks it up automatically.
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from src.evaluate_qa import evaluate, load_jsonl
from src.model_utils import DEFAULT_MODEL, generate_batch, load_model
from src.prompts import structured
from src.run_inference import log_run, run_totals
from src.train_lora import ADAPTER_DIR

MAX_NEW_TOKENS = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    ap.add_argument("--name", default="finetuned", help="results name (change it for a second adapter)")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        raise SystemExit(f"No adapter in {args.adapter}. Train first, or git pull the saved adapter.")

    tag = args.name + (f"_sample{args.limit}" if args.limit else "")
    os.makedirs("results", exist_ok=True)
    preds_path = f"results/{tag}_preds.jsonl"
    scores_path = f"results/{tag}_scores.json"

    test = load_jsonl(args.test)
    if args.limit:
        test = test[: args.limit]

    done = {}
    if os.path.exists(preds_path):
        done = {r["id"]: r for r in load_jsonl(preds_path)}
        print(f"Resuming: {len(done)} of {len(test)} already done")
    todo = [ex for ex in test if ex["id"] not in done]
    todo.sort(key=lambda ex: len(ex["context"]))   # less padding per batch

    if todo:
        print(f"Loading {args.model} + adapter {args.adapter} ...")
        model, tok = load_model(args.model, args.adapter)
        with open(f"results/{tag}_prompt_example.txt", "w", encoding="utf-8") as f:
            f.write(tok.apply_chat_template(structured(test[0]), tokenize=False, add_generation_prompt=True))
        start = time.time()
        with open(preds_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=args.name):
                batch = todo[i : i + args.batch_size]
                answers = generate_batch(model, tok, [structured(ex) for ex in batch], MAX_NEW_TOKENS)
                for ex, ans in zip(batch, answers):
                    f.write(json.dumps({"id": ex["id"], "prediction": ans}, ensure_ascii=False) + "\n")
                f.flush()
        minutes = (time.time() - start) / 60
        print(f"Generation took {minutes:.1f} min")
        log_run(tag, args, len(todo), minutes, MAX_NEW_TOKENS)

    scores, per_example = evaluate(test, load_jsonl(preds_path))
    scores["method"] = args.name
    scores["model"] = f"{args.model} + {args.adapter}"
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
