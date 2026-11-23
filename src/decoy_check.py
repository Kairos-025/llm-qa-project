"""Compare made-up answers with the decoy answers of SQuAD 2.0 (report Section 6.2).

When crowdworkers wrote the unanswerable questions of SQuAD 2.0, they also marked a
plausible but wrong answer in the passage ("plausible_answers" in the official file).
For every saved run, this script takes the unanswerable test questions the model
answered (a made-up answer) and checks how often that answer is the decoy:
exactly (official exact match) or closely (word-overlap F1 of at least 0.5).

The Hugging Face copy of the dataset drops the decoys, so the official dev file is
downloaded once into data/ (about 4 MB).

    python -m src.decoy_check                      # every run in results/
    python -m src.decoy_check --methods finetuned raft_rag_k3
"""
import argparse
import glob
import json
import os
import urllib.request

from src.evaluate_qa import clean_prediction, compute_exact, compute_f1, load_jsonl

DEV_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"
DEV_FILE = "data/dev-v2.0.json"
OUT = "results/analysis/decoy_answers.csv"
CLOSE = 0.5   # word-overlap F1 at or above this counts as "close to the decoy"


def decoys():
    """question id -> list of decoy answers (empty for answerable questions)."""
    if not os.path.exists(DEV_FILE):
        os.makedirs("data", exist_ok=True)
        print(f"Downloading {DEV_URL} ...")
        urllib.request.urlretrieve(DEV_URL, DEV_FILE)
    with open(DEV_FILE, encoding="utf-8") as f:
        dev = json.load(f)
    out = {}
    for article in dev["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                out[qa["id"]] = [a["text"] for a in qa.get("plausible_answers", [])]
    return out


def check(test, preds, plausible):
    answered = exact = close = 0
    for ex in test:
        if not ex["is_impossible"]:
            continue
        answer = clean_prediction(preds.get(ex["id"], ""))
        if answer == "":
            continue                      # abstained: not a made-up answer
        answered += 1
        options = plausible.get(ex["id"], [])
        if options:
            exact += max(compute_exact(d, answer) for d in options)
            close += max(compute_f1(d, answer) for d in options) >= CLOSE
    return answered, exact, close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--methods", nargs="*", help="run names (default: every *_preds.jsonl in results/)")
    args = ap.parse_args()

    test = load_jsonl(args.test)
    plausible = decoys()
    n_unans = sum(ex["is_impossible"] for ex in test)
    n_with = sum(1 for ex in test if ex["is_impossible"] and plausible.get(ex["id"]))
    print(f"{n_unans} unanswerable test questions, {n_with} with a decoy answer\n")

    methods = args.methods or sorted(
        os.path.basename(p)[: -len("_preds.jsonl")] for p in glob.glob("results/*_preds.jsonl"))
    rows = []
    print(f"{'method':<26}{'made up':>8}{'decoy, exact':>16}{'decoy, F1>=0.5':>18}")
    for m in methods:
        path = f"results/{m}_preds.jsonl"
        if not os.path.exists(path):
            print(f"{m:<26} no predictions file")
            continue
        preds = {r["id"]: r["prediction"] for r in load_jsonl(path)}
        answered, exact, close = check(test, preds, plausible)
        pct = lambda x: round(100 * x / answered, 1) if answered else 0.0
        rows.append((m, answered, exact, pct(exact), close, pct(close)))
        print(f"{m:<26}{answered:>8}{exact:>9} ({pct(exact):>4}%){close:>11} ({pct(close):>4}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("method,made_up,decoy_exact,decoy_exact_pct,decoy_close,decoy_close_pct\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
