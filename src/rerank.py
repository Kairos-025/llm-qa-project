"""
Extension (RAG depth): re-rank the Phase 5 search results with a cross-encoder.

The Phase 5 retriever embeds the question and each passage separately and compares the
two vectors. A cross-encoder reads the question and one passage together and scores how
well that passage answers the question (passage re-ranking; Nogueira and Cho, 2019). It is
too slow to run on all 1,204 passages, so it only re-orders the 10 passages that the
Phase 5 search already returned for each question. Recall@10 therefore cannot change;
the goal is to move the correct passage to the top.

Purely additive: no earlier file is changed. The RAG prompt, model, decoding and scorer
are exactly those of Phases 5, 6 and 8; only the order of the passages differs.

Usage:
    python -m src.rag build                              # Phase 5 search (top 10 per question)
    python -m src.rerank build                           # re-rank them, measure recall@k
    python -m src.rerank run --method rag --k 1          # plain RAG        -> rag_rerank_k1
    python -m src.rerank run --method finetuned --k 1    # Phase 6 adapter  -> finetuned_rag_rerank_k1
    python -m src.rerank run --method raft --k 1         # RAFT adapter     -> raft_rag_rerank_k1
    python -m src.rerank run --method raft --k 3         #                  -> raft_rag_rerank_k3
    python -m src.rerank run --method raft_p90 --k 1     # RAFT 10% adapter -> raft_p90_rag_rerank_k1

Creates:
    data/retrieved_rerank.jsonl                  re-ranked top 10 per question (rebuilt each session)
    results/retrieval_recall_rerank.json         recall@k before and after re-ranking
    results/<name>_preds.jsonl, _scores.json, _retrieval_split.json, _prompt_example.txt
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from src.evaluate_qa import evaluate, load_jsonl
from src.model_utils import DEFAULT_MODEL
from src.rag import CORPUS, RETRIEVED, TEST, passage_text, rag_prompt, recall_table, write_jsonl
from src.run_inference import log_run, run_totals

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKED = "data/retrieved_rerank.jsonl"
MAX_NEW_TOKENS = 48
ADAPTERS = {"rag": None, "finetuned": "adapters/phi3-squad-lora", "raft": "adapters/raft", "raft_p90": "adapters/raft_p90"}
PREFIX = {"rag": "rag", "finetuned": "finetuned_rag", "raft": "raft_rag", "raft_p90": "raft_p90_rag"}


# ---------------------------------------------------------------------------
# build: re-score the top 10 of every test question with the cross-encoder
# ---------------------------------------------------------------------------
def build(args, scorer=None):
    if not os.path.exists(RETRIEVED):
        raise SystemExit("No Phase 5 search results. Run first:  python -m src.rag build")
    text = {c["context_id"]: passage_text(c) for c in load_jsonl(CORPUS)}
    test = load_jsonl(TEST)
    retrieved = {r["id"]: r["retrieved"] for r in load_jsonl(RETRIEVED)}

    if scorer is None:
        import torch
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANK_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        scorer = lambda pairs: model.predict(pairs, batch_size=64, show_progress_bar=True)

    pairs = [(t["question"], text[c]) for t in test for c in retrieved[t["id"]]]
    print(f"Re-ranking {len(pairs)} question-passage pairs with {RERANK_MODEL} ...")
    start = time.time()
    scores = [float(s) for s in scorer(pairs)]
    seconds = round(time.time() - start, 1)

    rows, i = [], 0
    for t in test:
        ids = retrieved[t["id"]]
        sc = scores[i: i + len(ids)]
        i += len(ids)
        order = sorted(range(len(ids)), key=lambda j: -sc[j])   # stable: ties keep the Phase 5 order
        rows.append({"id": t["id"], "retrieved": [ids[j] for j in order],
                     "scores": [round(sc[j], 4) for j in order]})
    write_jsonl(RERANKED, rows)

    before = recall_table(test, retrieved)
    after = recall_table(test, {r["id"]: r["retrieved"] for r in rows})
    out = {
        "rerank_model": RERANK_MODEL,
        "reranked": "the top 10 passages of the Phase 5 search (all-MiniLM-L6-v2 + FAISS) for each question",
        "pairs_scored": len(pairs),
        "seconds": seconds,
        "recall_before": before,
        "recall_after": after,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/retrieval_recall_rerank.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nScored {len(pairs)} pairs in {seconds}s")
    print(f"{'':<14}{'before':>30}{'after':>30}")
    for name in before:
        b = " / ".join(f"{before[name][k]}" for k in ("recall@1", "recall@3", "recall@5"))
        a = " / ".join(f"{after[name][k]}" for k in ("recall@1", "recall@3", "recall@5"))
        print(f"{name:<14}{b:>30}{a:>30}   (recall@1 / @3 / @5)")
    print(f"\nSaved {RERANKED} and results/retrieval_recall_rerank.json")


# ---------------------------------------------------------------------------
# run: answer from the re-ranked passages (same prompt and scorer as before)
# ---------------------------------------------------------------------------
def attach_reranked(examples):
    if not os.path.exists(RERANKED):
        raise SystemExit("No re-ranked results. Run first:  python -m src.rerank build")
    text = {c["context_id"]: passage_text(c) for c in load_jsonl(CORPUS)}
    order = {r["id"]: r["retrieved"] for r in load_jsonl(RERANKED)}
    out = []
    for ex in examples:
        ex = dict(ex)
        ex["retrieved_ids"] = order[ex["id"]]
        ex["retrieved"] = [text[c] for c in order[ex["id"]]]
        out.append(ex)
    return out


def retrieval_split(test, per_example, k):
    """Same split as `python -m src.rag analyze`: retrieval misses vs model errors."""
    d = {r["id"]: r for r in per_example}

    def summary(rows):
        if not rows:
            return {"n": 0}
        return {"n": len(rows),
                "exact_match": round(100 * sum(d[r["id"]]["em"] for r in rows) / len(rows), 2),
                "f1": round(100 * sum(d[r["id"]]["f1"] for r in rows) / len(rows), 2)}

    ans = [t for t in test if not t["is_impossible"]]
    hit = [t for t in ans if t["context_id"] in t["retrieved_ids"][:k]]
    miss = [t for t in ans if t["context_id"] not in t["retrieved_ids"][:k]]
    model_fail = [t for t in hit if d[t["id"]]["em"] == 0]
    return {
        "k": k,
        "answerable_with_correct_passage_retrieved": summary(hit),
        "answerable_without_correct_passage": summary(miss),
        "error_sources_on_answerable": {
            "retrieval_failures": len(miss),
            "model_failures_despite_correct_passage": len(model_fail),
            "correct": len(hit) - len(model_fail),
        },
    }


def run(args):
    from src.model_utils import generate_batch, load_model

    adapter = args.adapter or ADAPTERS[args.method]
    if adapter and not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        raise SystemExit(f"No adapter in {adapter}. git pull it, or train it first.")
    name = args.name or f"{PREFIX[args.method]}_rerank_k{args.k}"
    tag = name + (f"_sample{args.limit}" if args.limit else "")
    os.makedirs("results", exist_ok=True)
    preds_path = f"results/{tag}_preds.jsonl"
    scores_path = f"results/{tag}_scores.json"

    test = load_jsonl(args.test)
    if args.limit:
        test = test[: args.limit]
    test = attach_reranked(test)

    done = {}
    if os.path.exists(preds_path):
        done = {r["id"]: r for r in load_jsonl(preds_path)}
        print(f"Resuming: {len(done)} of {len(test)} already done")
    todo = [ex for ex in test if ex["id"] not in done]
    todo.sort(key=lambda ex: len(" ".join(ex["retrieved"][: args.k])))

    if todo:
        print(f"Loading {args.model}" + (f" + adapter {adapter}" if adapter else "") + " ...")
        model, tok = load_model(args.model, adapter)
        with open(f"results/{tag}_prompt_example.txt", "w", encoding="utf-8") as f:
            f.write(tok.apply_chat_template(rag_prompt(test[0], args.k), tokenize=False, add_generation_prompt=True))
        start = time.time()
        with open(preds_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), args.batch_size), desc=name):
                batch = todo[i: i + args.batch_size]
                answers = generate_batch(model, tok, [rag_prompt(ex, args.k) for ex in batch], MAX_NEW_TOKENS)
                for ex, ans in zip(batch, answers):
                    f.write(json.dumps({"id": ex["id"], "prediction": ans, "retrieved": ex["retrieved_ids"][: args.k]},
                                       ensure_ascii=False) + "\n")
                f.flush()
        minutes = (time.time() - start) / 60
        print(f"Generation took {minutes:.1f} min")
        log_run(tag, args, len(todo), minutes, MAX_NEW_TOKENS)

    scores, per_example = evaluate(test, load_jsonl(preds_path))
    scores["method"] = name
    scores["model"] = args.model + (f" + {adapter}" if adapter else "")
    scores["k"] = args.k
    scores["retrieval"] = f"Phase 5 top 10, re-ranked with {RERANK_MODEL}"
    scores.update(run_totals(tag))
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    with open(scores_path.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    split = dict(method=name, **retrieval_split(test, per_example, args.k))
    with open(f"results/{tag}_retrieval_split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(json.dumps(scores, indent=2))
    print(json.dumps(split["error_sources_on_answerable"], indent=2))
    print(f"\nSaved {preds_path}, {scores_path} and results/{tag}_retrieval_split.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--method", choices=sorted(ADAPTERS), required=True)
    r.add_argument("--k", type=int, required=True, help="number of re-ranked passages shown (1, 3 or 5)")
    r.add_argument("--adapter", default=None, help="override the adapter folder")
    r.add_argument("--name", default=None, help="override the results name")
    r.add_argument("--test", default=TEST)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--batch_size", type=int, default=8)
    r.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = ap.parse_args()
    {"build": build, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
