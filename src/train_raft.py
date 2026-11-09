"""
Phase 8: Retrieval-Augmented Fine-Tuning (RAFT; Zhang et al., 2024).

Phase 7 found that the Phase 6 adapter, trained only on the ONE correct
passage, is easily distracted by retrieved passages: going from 1 to 3
passages cut unanswerable accuracy by 13.8 points. RAFT trains the model on
the same kind of input it sees at RAG test time: several retrieved passages,
some of them irrelevant ("distractors").

For each training question (the same 4,000 questions as Phase 6):
  - k passages are shown, k in {1, 3, 5}, in the exact RAG prompt from Phase 5.
  - Answerable questions: with probability 0.8 the correct passage is among
    them (target: the answer); otherwise only distractors are shown and the
    target is "unanswerable", since the answer cannot be read from them.
  - Unanswerable questions: their own (related but non-answering) passage is
    included with probability 0.8, plus distractors; target "unanswerable".
  - Distractors are REAL search results: the closest other passages found by
    the Phase 5 retriever (same embedding model), searched over the training
    split's passages only. The test set is never touched.

Phase 8 is purely additive: no earlier file is changed. Evaluation reuses
eval_finetuned.py, analysis reuses analysis.py.

Usage:
    python -m src.train_lora prepare              # Phase 6 questions (if not on this server yet)
    python -m src.train_raft prepare              # build the RAFT training data (Session 1)
    python -m src.train_raft train --tiny         # quick test + full-run time estimate (Session 1)
    python -m src.train_raft train                # full training (Session 2)
    python -m src.train_raft train --resume       # continue after a crash (same server only)

    python -m src.rag build                       # then evaluate (Session 3):
    python -m src.eval_finetuned --adapter adapters/raft --name raft_rag_k1 --rag_k 1
    python -m src.eval_finetuned --adapter adapters/raft --name raft_rag_k3 --rag_k 3
    python -m src.eval_finetuned --adapter adapters/raft --name raft_rag_k5 --rag_k 5

Creates:
    data/raft_train.jsonl, data/raft_val.jsonl   training / validation examples (rebuilt each session)
    results/raft_data_stats.json                 what the training data looks like
    adapters/raft/                               the trained adapter (committed to GitHub)
    results/raft_train_log.json                  settings, loss curve, time, GPU
    results/raft_tiny_log.json                   same, for the tiny test run
"""
import argparse
import glob
import json
import os
import random
import time
from datetime import datetime, timezone

from src.evaluate_qa import load_jsonl
from src.model_utils import DEFAULT_MODEL
from src.rag import EMBED_MODEL, passage_text, rag_prompt
from src.train_lora import LORA, gpu_name, load_for_training, save_adapter_fp16, target, write_jsonl

SEED = 42
K_CHOICES = (1, 3, 5)           # passages shown per example
K_WEIGHTS = (0.3, 0.4, 0.3)
P_ORACLE = 0.8                  # share of examples that include the question's own passage (RAFT paper's best)
MAX_DISTRACTOR_WORDS = 300      # skips very long distractors to keep examples short
SEARCH_DEPTH = 20               # distractors are drawn from the top 20 search results
MAX_TOKENS = 1536               # 5 passages need more room than Phase 6's 1,024

ADAPTER_DIR = "adapters/raft"
TINY_ADAPTER_DIR = "checkpoints/raft-tiny-adapter"


# ---------------------------------------------------------------------------
# prepare: search the training passages, then build mixed-passage examples
# ---------------------------------------------------------------------------
def train_corpus(rows):
    """Every unique passage of the SQuAD training split (never validation/test)."""
    seen, corpus = {}, []
    for r in rows:
        if r["context"] not in seen:
            seen[r["context"]] = len(corpus)
            corpus.append({"title": r["title"], "text": r["context"]})
    return corpus, seen


def search(questions, corpus, depth, embedder=None):
    """Top `depth` passages for each question, with the Phase 5 retriever
    (MiniLM embeddings, exact cosine search). Returns passage positions."""
    import faiss
    import numpy as np
    if embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        embedder = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=True)
    p = np.asarray(embedder([passage_text(c) for c in corpus]), dtype="float32")
    index = faiss.IndexFlatIP(p.shape[1])
    index.add(p)
    q = np.asarray(embedder(questions), dtype="float32")
    _, idx = index.search(q, min(depth, len(corpus)))
    return idx.tolist()


def make_example(ex, own, ranked, corpus, rng):
    """One RAFT example: k passages (own passage included or not) and its target."""
    k = rng.choices(K_CHOICES, K_WEIGHTS)[0]
    distractors = [j for j in ranked if j != own
                   and len(corpus[j]["text"].split()) <= MAX_DISTRACTOR_WORDS]
    with_own = rng.random() < P_ORACLE
    chosen = distractors[: k - 1] + [own] if with_own else distractors[:k]
    rng.shuffle(chosen)                         # the correct passage can be anywhere
    out = dict(ex)
    out["k"] = len(chosen)
    out["oracle_included"] = with_own
    out["retrieved"] = [passage_text(corpus[j]) for j in chosen]
    if not with_own:                            # the answer is not in what the model sees
        out["original_answers"] = ex["answers"]
        out["answers"] = []
    return out


def prepare(args, embedder=None):
    for f in ("data/train.jsonl", "data/ft_train.jsonl", "data/ft_val.jsonl"):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found. Run first:  python -m src.data_prep --test_size 1000"
                             "  and  python -m src.train_lora prepare")
    corpus, pos = train_corpus(load_jsonl("data/train.jsonl"))
    ft_train, ft_val = load_jsonl("data/ft_train.jsonl"), load_jsonl("data/ft_val.jsonl")
    rows = ft_train + ft_val
    print(f"Searching {len(corpus)} training passages for {len(rows)} questions ...")
    start = time.time()
    ranked = search([r["question"] for r in rows], corpus, SEARCH_DEPTH, embedder)
    seconds = round(time.time() - start, 1)

    rng = random.Random(SEED)
    built = [make_example(r, pos[r["context"]], ranked[i], corpus, rng) for i, r in enumerate(rows)]
    train, val = built[: len(ft_train)], built[len(ft_train):]
    own_found = sum(pos[r["context"]] in ranked[i][:5] for i, r in enumerate(rows))
    write_jsonl("data/raft_train.jsonl", train)
    write_jsonl("data/raft_val.jsonl", val)

    def stats(rs):
        n = len(rs)
        return {
            "n": n,
            "k_counts": {str(k): sum(r["k"] == k for r in rs) for k in K_CHOICES},
            "oracle_included_share": round(100 * sum(r["oracle_included"] for r in rs) / n, 1),
            "target_unanswerable_share": round(100 * sum(not r["answers"] for r in rs) / n, 1),
            "answerable_but_oracle_removed": sum("original_answers" in r for r in rs),
            "avg_words_shown": round(sum(len(" ".join(r["retrieved"]).split()) for r in rs) / n, 1),
        }

    out = {
        "method": "RAFT (Zhang et al., 2024): train on retrieved passages with distractors",
        "questions": "the same 4,000 train / 200 validation questions as Phase 6 (data/ft_*.jsonl)",
        "distractor_source": f"top {SEARCH_DEPTH} search results over {len(corpus)} SQuAD train passages, "
                             f"{EMBED_MODEL}, exact cosine (FAISS IndexFlatIP)",
        "own_passage_in_top5_search": round(100 * own_found / len(rows), 1),
        "search_seconds": seconds,
        "seed": SEED,
        "k_choices": dict(zip(map(str, K_CHOICES), K_WEIGHTS)),
        "p_oracle": P_ORACLE,
        "max_distractor_words": MAX_DISTRACTOR_WORDS,
        "prompt_format": "Phase 5 RAG prompt (src.rag.rag_prompt); target 'Answer: <answer or unanswerable>'",
        "train": stats(train),
        "validation": stats(val),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/raft_data_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nSaved data/raft_train.jsonl, data/raft_val.jsonl and results/raft_data_stats.json")


# ---------------------------------------------------------------------------
# train: same QLoRA setup as Phase 6, only the input format differs
# ---------------------------------------------------------------------------
def encode(tok, ex):
    """Token ids for one example; only the answer is graded (-100 = ignore)."""
    msgs = rag_prompt(ex, ex["k"])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": target(ex)}], tokenize=False)
    assert full.startswith(prompt), "chat template changed the prompt part"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def build_dataset(tok, rows):
    from datasets import Dataset
    encoded = [encode(tok, r) for r in rows]
    kept = [e for e in encoded if len(e["input_ids"]) <= MAX_TOKENS]
    lengths = [len(e["input_ids"]) for e in kept]
    return Dataset.from_list(kept), len(encoded) - len(kept), lengths


def sanity_check(model, tok, rows, n=4):
    from src.model_utils import generate_batch
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    batch = rows[:n]
    preds = generate_batch(model, tok, [rag_prompt(r, r["k"]) for r in batch], max_new_tokens=24)
    print("\nSanity check on held-out questions:")
    out = []
    for r, p in zip(batch, preds):
        gold = r["answers"][0] if r["answers"] else "unanswerable"
        print(f"  Q: {r['question'][:80]}  (k={r['k']}, own passage shown: {r['oracle_included']})"
              f"\n     gold: {gold}\n     model: {p}")
        out.append({"question": r["question"], "k": r["k"], "oracle_included": r["oracle_included"],
                    "gold": gold, "prediction": p})
    return out


def train(args):
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    if not os.path.exists("data/raft_train.jsonl"):
        raise SystemExit("No RAFT data. Run first:  python -m src.train_raft prepare")
    train_rows, val_rows = load_jsonl("data/raft_train.jsonl"), load_jsonl("data/raft_val.jsonl")
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    if args.tiny:
        train_rows, val_rows = train_rows[:100], val_rows[:20]
        out_dir, ckpt_dir, log_path = TINY_ADAPTER_DIR, "checkpoints/raft-tiny", "results/raft_tiny_log.json"
        max_steps, every = 15, 5
    else:
        name = args.name or "raft"
        out_dir = ADAPTER_DIR if name == "raft" else f"adapters/{name}"
        ckpt_dir, log_path = f"checkpoints/{name}", f"results/{name}_train_log.json"
        max_steps, every = -1, 50

    lora = dict(LORA, r=args.rank, lora_alpha=2 * args.rank)
    print(f"Loading {args.model} for training ...")
    model, tok = load_for_training(args.model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    train_ds, dropped_t, lengths = build_dataset(tok, train_rows)
    val_ds, dropped_v, _ = build_dataset(tok, val_rows)
    print(f"Training examples: {len(train_ds)}  validation: {len(val_ds)}  "
          f"(dropped as too long: {dropped_t + dropped_v})  "
          f"average length: {sum(lengths) / len(lengths):.0f} tokens")

    on_gpu = torch.cuda.is_available()
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * args.epochs
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),
        logging_steps=1 if args.tiny else 10,
        eval_strategy="steps",
        eval_steps=every,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        fp16=on_gpu,
        optim="paged_adamw_8bit" if on_gpu else "adamw_torch",
        report_to="none",
        seed=args.seed,
        use_cpu=not on_gpu,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    resume = args.resume and bool(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    if args.resume and not resume:
        print("No checkpoint found on this server; starting from the beginning.")
    start = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - start) / 60
    final_eval = trainer.evaluate()
    print(f"Training took {minutes:.1f} min")

    estimate = None
    if args.tiny:   # 15 steps on 100 examples -> time for the full run
        full_steps = round(4000 / (args.batch_size * args.grad_accum))
        estimate = round(minutes / trainer.state.global_step * full_steps * 1.1)
        print(f"Estimated full run: about {estimate} min ({full_steps} steps)")

    size_mb = save_adapter_fp16(model, tok, out_dir)
    samples = sanity_check(model, tok, val_rows)

    curve, seen = [], set()
    for h in trainer.state.log_history:
        point = {k: round(h[k], 4) if k != "step" else h[k] for k in ("step", "loss", "eval_loss") if k in h}
        key = tuple(sorted(point.items()))
        if len(point) > 1 and key not in seen:
            seen.add(key)
            curve.append(point)
    log = {
        "run": "tiny test" if args.tiny else "full",
        "base_model": args.model,
        "method": "RAFT: QLoRA (4-bit NF4 base + LoRA adapter) on retrieved passages with distractors",
        "lora": lora,
        "training": {
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "dropped_too_long": dropped_t + dropped_v,
            "avg_tokens": round(sum(lengths) / len(lengths), 1),
            "epochs": args.epochs if not args.tiny else None,
            "max_steps": max_steps if args.tiny else None,
            "optimizer_steps": trainer.state.global_step,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "scheduler": "cosine",
            "max_tokens": MAX_TOKENS,
            "resumed_from_checkpoint": resume,
            "seed": args.seed,
        },
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(100 * trainable / total, 3),
        "final_train_loss": round(result.training_loss, 4),
        "final_val_loss": round(final_eval["eval_loss"], 4),
        "train_minutes": round(minutes, 2),
        "estimated_full_run_minutes": estimate,
        "gpu": gpu_name(),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if on_gpu else None,
        "adapter_dir": out_dir,
        "adapter_size_mb": size_mb,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loss_curve": curve,
        "sanity_check": samples,
    }
    os.makedirs("results", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\nFinal train loss {log['final_train_loss']}, validation loss {log['final_val_loss']}")
    print(f"Saved adapter ({size_mb} MB) to {out_dir} and the log to {log_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--tiny", action="store_true", help="quick test on 100 examples, prints a time estimate")
    t.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    t.add_argument("--epochs", type=float, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch_size", type=int, default=2, help="2 (not 4): RAFT examples are ~3x longer")
    t.add_argument("--grad_accum", type=int, default=8, help="effective batch stays 16, as in Phase 6")
    t.add_argument("--max_train", type=int, default=None, help="train on fewer examples (if time is short)")
    t.add_argument("--rank", type=int, default=LORA["r"])
    t.add_argument("--seed", type=int, default=SEED)
    t.add_argument("--name", default=None, help="name for an extra run; saves to adapters/<name>")
    args = ap.parse_args()
    {"prepare": prepare, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()