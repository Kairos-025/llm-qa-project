"""
Runs a method over the test set, saves predictions, then scores them.

Usage:
    python -m src.run_inference --method baseline --limit 20   # quick test
    python -m src.run_inference --method baseline              # full run

Saves:
    results/<method>_preds.jsonl           every answer
    results/<method>_scores.json           scores, total generation time, GPU
    results/<method>_prompt_example.txt    one full prompt, for the report
    results/run_log.jsonl                  one line per run (time, GPU, settings)

Resumes automatically if a run is interrupted (e.g. a Colab disconnect):
just run the same command again. If all answers already exist, it only
re-scores them (no model is loaded), which is useful after scoring changes.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

from tqdm import tqdm

from src.evaluate_qa import evaluate, load_jsonl
from src.model_utils import DEFAULT_MODEL, generate_batch, load_model
from src.prompts import DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS, PROMPTS


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "unknown"


def log_run(tag, args, n_generated, minutes, max_new):
    """Append one line per generation run. Resumed runs add extra lines,
    which run_totals() sums, so the total time stays correct."""
    entry = {
        "tag": tag,
        "n_generated": n_generated,
        "minutes": round(minutes, 2),
        "gpu": gpu_name(),
        "batch_size": args.batch_size,
        "max_new_tokens": max_new,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open("results/run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_totals(tag):
    path = "results/run_log.jsonl"
    if not os.path.exists(path):
        return {}
    runs = [r for r in load_jsonl(path) if r["tag"] == tag]
    if not runs:
        return {}
    return {
        "generation_minutes": round(sum(r["minutes"] for r in runs), 2),
        "gpu": runs[-1]["gpu"],
        "max_new_tokens": runs[-1]["max_new_tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder (Phase 6)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="defaults to the per-method value in prompts.py")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = ap.parse_args()

    max_new = args.max_new_tokens or MAX_NEW_TOKENS.get(args.method, DEFAULT_MAX_NEW_TOKENS)
    os.makedirs("results", exist_ok=True)
    tag = args.method + (f"_sample{args.limit}" if args.limit else "")
    preds_path = f"results/{tag}_preds.jsonl"
    scores_path = f"results/{tag}_scores.json"

    test = load_jsonl(args.test)
    if args.limit:
        test = test[: args.limit]

    # Resume: skip questions already answered in a previous (interrupted) run
    done = {}
    if os.path.exists(preds_path):
        done = {r["id"]: r for r in load_jsonl(preds_path)}
        print(f"Resuming: {len(done)} of {len(test)} already done")
    todo = [ex for ex in test if ex["id"] not in done]
    # Similar-length prompts batch together, which wastes less GPU time on padding
    todo.sort(key=lambda ex: len(ex["context"]))

    if todo:
        print(f"Loading {args.model} ...")
        model, tok = load_model(args.model, args.adapter)
        make_prompt = PROMPTS[args.method]
        with open(f"results/{tag}_prompt_example.txt", "w", encoding="utf-8") as f:
            f.write(tok.apply_chat_template(make_prompt(test[0]), tokenize=False, add_generation_prompt=True))
        start = time.time()
        with open(preds_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=args.method):
                batch = todo[i : i + args.batch_size]
                answers = generate_batch(
                    model, tok, [make_prompt(ex) for ex in batch], max_new
                )
                for ex, ans in zip(batch, answers):
                    f.write(json.dumps({"id": ex["id"], "prediction": ans}, ensure_ascii=False) + "\n")
                f.flush()  # saved after every batch, so a disconnect loses almost nothing
        minutes = (time.time() - start) / 60
        print(f"Generation took {minutes:.1f} min")
        log_run(tag, args, len(todo), minutes, max_new)

    scores, per_example = evaluate(test, load_jsonl(preds_path))
    scores["method"] = args.method
    scores["model"] = args.model + (f" + {args.adapter}" if args.adapter else "")
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
