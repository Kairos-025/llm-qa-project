"""
Phase 5: Retrieval-Augmented Generation (RAG).

The baseline and prompt methods are GIVEN the correct passage. RAG is not:
for each question it must SEARCH all 1,204 passages, take the top k, and
answer from those. This file holds the search and the RAG prompt;
run_rag.py does the answering with the same model, settings and scorer as
every other method.

Phase 5 is purely additive: no Phase 2-4 file is changed.

Usage:
    python -m src.rag build                    # embed passages, build the FAISS index,
                                               # retrieve for every test question,
                                               # measure retrieval accuracy
    python -m src.run_rag --k 3                # answer from the top 3 passages (see run_rag.py)
    python -m src.rag analyze --method rag_k3  # after a RAG run: split its scores by
                                               # whether the correct passage was retrieved

Creates:
    data/rag.index                 FAISS vector index of the 1,204 passages (Colab only, rebuilt each session)
    data/retrieved.jsonl           top-10 passage ids per test question
    results/retrieval_recall.json  retrieval accuracy (recall@1/3/5/10), time, model
    results/<method>_retrieval_split.json   retrieval vs model failures (analyze)
"""
import argparse
import json
import os
import re
import time

from src.evaluate_qa import load_jsonl

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_N = 10                      # how many passages are stored per question
KS = (1, 3, 5, 10)              # recall@k values reported

CORPUS = "data/corpus.jsonl"
TEST = "data/test.jsonl"
INDEX = "data/rag.index"
RETRIEVED = "data/retrieved.jsonl"


def passage_text(row):
    """What gets embedded: the article title plus the passage. The title helps
    the search tell apart passages about different topics."""
    return f"{row['title'].replace('_', ' ')}. {row['text']}"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def recall_table(test, retrieved):
    """Share of questions whose correct passage is among the top k results."""
    def recall(rows, k):
        if not rows:
            return None
        hits = sum(r["context_id"] in retrieved[r["id"]][:k] for r in rows)
        return round(100 * hits / len(rows), 2)

    groups = {
        "all": test,
        "answerable": [r for r in test if not r["is_impossible"]],
        "unanswerable": [r for r in test if r["is_impossible"]],
    }
    return {name: {f"recall@{k}": recall(rows, k) for k in KS} for name, rows in groups.items()}


def build(args):
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    corpus = load_jsonl(CORPUS)
    test = load_jsonl(TEST)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(corpus)} passages with {EMBED_MODEL} on {device} ...")
    start = time.time()

    model = SentenceTransformer(EMBED_MODEL, device=device)
    # normalised embeddings + inner product = cosine similarity ("closeness in meaning")
    passages = model.encode([passage_text(c) for c in corpus], batch_size=64,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(passages.shape[1])
    index.add(passages)
    faiss.write_index(index, INDEX)

    questions = model.encode([t["question"] for t in test], batch_size=64,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True).astype("float32")
    scores, idx = index.search(questions, min(TOP_N, len(corpus)))
    seconds = round(time.time() - start, 1)

    ids = [c["context_id"] for c in corpus]
    rows = [{"id": t["id"],
             "retrieved": [ids[j] for j in idx[i]],
             "scores": [round(float(s), 4) for s in scores[i]]}
            for i, t in enumerate(test)]
    write_jsonl(RETRIEVED, rows)

    retrieved = {r["id"]: r["retrieved"] for r in rows}
    table = recall_table(test, retrieved)
    out = {
        "embed_model": EMBED_MODEL,
        "n_passages": len(corpus),
        "n_questions": len(test),
        "embedding_dim": int(passages.shape[1]),
        "index_type": "faiss.IndexFlatIP (exact cosine search)",
        "build_and_search_seconds": seconds,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "recall": table,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/retrieval_recall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nIndexed {len(corpus)} passages, searched {len(test)} questions in {seconds}s")
    print(f"{'':<14}" + "".join(f"recall@{k:<5}" for k in KS))
    for name, vals in table.items():
        print(f"{name:<14}" + "".join(f"{vals[f'recall@{k}']:<12}" for k in KS))
    print(f"\nSaved {INDEX}, {RETRIEVED} and results/retrieval_recall.json")


def attach_retrieved(examples):
    """Called by run_rag.py: adds the top-10 retrieved
    passage texts to each test example, in search order. The correct passage is
    only included if the search found it."""
    if not os.path.exists(RETRIEVED):
        raise SystemExit("No retrieval results found. Run first:  python -m src.rag build")
    text = {c["context_id"]: passage_text(c) for c in load_jsonl(CORPUS)}
    retrieved = {r["id"]: r["retrieved"] for r in load_jsonl(RETRIEVED)}
    out = []
    for ex in examples:
        ex = dict(ex)
        ex["retrieved_ids"] = retrieved[ex["id"]]
        ex["retrieved"] = [text[c] for c in retrieved[ex["id"]]]
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# RAG prompt: the Phase 4 structured prompt (tied best F1, fastest, shortest),
# adapted to several retrieved passages instead of the one correct passage.
# ---------------------------------------------------------------------------
RAG_INSTRUCTIONS = (
    "Read the context passages and answer the question.\n"
    "Rules:\n"
    "- The passages come from a search, so some may be irrelevant. Use only a passage that answers the question.\n"
    "- Copy the answer exactly as it appears in the passage, using as few words as possible.\n"
    "- Do not write a full sentence and do not explain.\n"
    "- If no passage contains the answer, reply with exactly: unanswerable"
)

# Phi-3-mini reads at most 4,000 tokens. Passages average 127 words but reach
# 629, so 5 long passages could overflow; retrieved text is capped at this many
# words (about 3,200 tokens), cutting from the LOWEST-ranked passage first.
MAX_CONTEXT_WORDS = 2400


def rag_prompt(example, k):
    parts, used = [], 0
    for i, passage in enumerate(example["retrieved"][:k], start=1):
        words = passage.split()
        room = MAX_CONTEXT_WORDS - used
        if room <= 0:
            break
        words = words[:room]
        used += len(words)
        parts.append(f"[{i}] " + " ".join(words))
    return [{
        "role": "user",
        "content": (
            RAG_INSTRUCTIONS + "\n\n"
            "Reply in exactly this format and nothing else:\n"
            "Answer: <answer or unanswerable>\n\n"
            "Context passages:\n" + "\n\n".join(parts)
            + f"\n\nQuestion: {example['question']}"
        ),
    }]


def analyze(args):
    """Splits a RAG run into: questions where retrieval found the correct
    passage vs. where it did not, so errors can be blamed on the right part."""
    k = int(re.search(r"rag_k(\d+)", args.method).group(1))
    details_path = f"results/{args.method}_scores_details.jsonl"
    if not os.path.exists(details_path):
        raise SystemExit(f"{details_path} not found. Run the method first.")
    details = {d["id"]: d for d in load_jsonl(details_path)}
    retrieved = {r["id"]: r["retrieved"] for r in load_jsonl(RETRIEVED)}
    test = [t for t in load_jsonl(TEST) if t["id"] in details]

    def summary(rows):
        if not rows:
            return {"n": 0}
        return {"n": len(rows),
                "exact_match": round(100 * sum(details[r["id"]]["em"] for r in rows) / len(rows), 2),
                "f1": round(100 * sum(details[r["id"]]["f1"] for r in rows) / len(rows), 2)}

    ans = [t for t in test if not t["is_impossible"]]
    hit = [t for t in ans if t["context_id"] in retrieved[t["id"]][:k]]
    miss = [t for t in ans if t["context_id"] not in retrieved[t["id"]][:k]]
    model_fail = [t for t in hit if details[t["id"]]["em"] == 0]
    out = {
        "method": args.method, "k": k,
        "answerable_with_correct_passage_retrieved": summary(hit),
        "answerable_without_correct_passage": summary(miss),
        "error_sources_on_answerable": {
            "retrieval_failures": len(miss),
            "model_failures_despite_correct_passage": len(model_fail),
            "correct": len(hit) - len(model_fail),
        },
    }
    path = f"results/{args.method}_retrieval_split.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("analyze")
    a.add_argument("--method", required=True, help="e.g. rag_k3")
    args = ap.parse_args()
    {"build": build, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
