"""
Phase 6: fine-tune Phi-3-mini for SQuAD 2.0 question answering with QLoRA.

The base model stays frozen in 4-bit (the same model every other method uses).
Only a small LoRA adapter is trained, so it fits on a free T4 GPU. The model
learns from training-split examples written in the Phase 4 "structured"
prompt format, with the short correct answer (or "unanswerable") as the target.

Usage:
    python -m src.train_lora prepare          # build the balanced training data (Session 1)
    python -m src.train_lora train --tiny     # 5-minute test run (Session 1)
    python -m src.train_lora train            # full training (Session 2)
    python -m src.train_lora train --resume   # continue after a crash (same Colab server only)

Creates:
    data/ft_train.jsonl, data/ft_val.jsonl     training / validation examples (rebuilt each session)
    results/finetune_data_stats.json           what the training data looks like
    adapters/phi3-squad-lora/                  the trained adapter (full run; committed to GitHub)
    results/finetune_train_log.json            settings, loss curve, time, GPU (full run)
    results/finetune_tiny_log.json             same, for the tiny test run
    checkpoints/                               in-progress saves (Colab disk only, gitignored)
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
from src.prompts import structured

SEED = 42
N_TRAIN = 4000              # half answerable, half unanswerable (test set is ~50/50)
N_VAL = 200                 # held out from training, used to watch for overfitting
MAX_CONTEXT_WORDS = 450     # skips the longest passages to save GPU memory (test max is 629)
MAX_TOKENS = 1024           # any example longer than this is dropped

ADAPTER_DIR = "adapters/phi3-squad-lora"
TINY_ADAPTER_DIR = "checkpoints/tiny-adapter"

LORA = dict(
    r=16,                   # adapter size: bigger learns more but uses more memory
    lora_alpha=32,          # how strongly the adapter changes the model (usually 2 x r)
    lora_dropout=0.05,
    # every attention and feed-forward layer of Phi-3
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def target(ex):
    """What the model should reply: the structured format with the short gold answer."""
    return "Answer: " + (ex["answers"][0] if ex["answers"] else "unanswerable")


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


# ---------------------------------------------------------------------------
# prepare: balanced training data from the TRAINING split only
# ---------------------------------------------------------------------------
def prepare(args):
    rows = load_jsonl("data/train.jsonl")
    # never train on the few-shot demonstration examples (keeps methods independent)
    if os.path.exists("data/fewshot_pool.jsonl"):
        pool = {r["id"] for r in load_jsonl("data/fewshot_pool.jsonl")}
        rows = [r for r in rows if r["id"] not in pool]
    rows = [r for r in rows if len(r["context"].split()) <= MAX_CONTEXT_WORDS]

    rng = random.Random(SEED)
    ans = [r for r in rows if not r["is_impossible"]]
    unans = [r for r in rows if r["is_impossible"]]
    rng.shuffle(ans)
    rng.shuffle(unans)

    half_t, half_v = args.n_train // 2, args.n_val // 2
    if min(len(ans), len(unans)) < half_t + half_v:
        raise SystemExit(f"Not enough examples: need {half_t + half_v} of each type, "
                         f"have {len(ans)} answerable and {len(unans)} unanswerable.")
    train = ans[:half_t] + unans[:half_t]
    val = ans[half_t:half_t + half_v] + unans[half_t:half_t + half_v]
    rng.shuffle(train)
    rng.shuffle(val)

    os.makedirs("data", exist_ok=True)
    write_jsonl("data/ft_train.jsonl", train)
    write_jsonl("data/ft_val.jsonl", val)

    def stats(rs):
        return {
            "n": len(rs),
            "unanswerable_share": round(100 * sum(r["is_impossible"] for r in rs) / len(rs), 1),
            "avg_context_words": round(sum(len(r["context"].split()) for r in rs) / len(rs), 1),
            "articles": len({r["title"] for r in rs}),
        }

    out = {
        "source": "SQuAD 2.0 train split (test questions come from validation, so no overlap)",
        "seed": SEED,
        "max_context_words": MAX_CONTEXT_WORDS,
        "excluded": "few-shot pool examples; passages longer than max_context_words",
        "prompt_format": "Phase 4 structured prompt; target 'Answer: <short gold answer or unanswerable>'",
        "train": stats(train),
        "validation": stats(val),
        "example_target_answerable": target(next(r for r in train if not r["is_impossible"])),
        "example_target_unanswerable": target(next(r for r in train if r["is_impossible"])),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/finetune_data_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nSaved data/ft_train.jsonl, data/ft_val.jsonl and results/finetune_data_stats.json")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def encode(tok, ex):
    """Turns one example into token ids. The model is only graded (labels) on
    the answer part, never on copying the prompt; -100 means 'ignore'."""
    msgs = structured(ex)
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
    return Dataset.from_list(kept), len(encoded) - len(kept)


def load_for_training(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"          # right padding for training
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        # identical 4-bit setup to model_utils.load_model (QLoRA)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", dtype=torch.float16)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        print("WARNING: no GPU found. CPU mode is only for testing the code with a tiny model.")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.config.use_cache = False      # required with gradient checkpointing
    model = get_peft_model(model, LoraConfig(**LORA))
    return model, tok


def save_adapter_fp16(model, tok, out_dir):
    """Saves only the adapter (not the 3.8B base model), in float16 so it stays
    small enough for GitHub (limit 100 MB per file)."""
    import torch
    from safetensors.torch import load_file, save_file
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    path = os.path.join(out_dir, "adapter_model.safetensors")
    weights = {k: v.to(torch.float16).contiguous() for k, v in load_file(path).items()}
    save_file(weights, path, metadata={"format": "pt"})
    return round(os.path.getsize(path) / 1e6, 1)


def sanity_check(model, tok, rows, n=4):
    """Answers a few held-out questions so you can see the model's new behaviour."""
    from src.model_utils import generate_batch
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    batch = rows[:n]
    preds = generate_batch(model, tok, [structured(r) for r in batch], max_new_tokens=24)
    print("\nSanity check on held-out questions:")
    out = []
    for r, p in zip(batch, preds):
        gold = r["answers"][0] if r["answers"] else "unanswerable"
        print(f"  Q: {r['question'][:80]}\n     gold: {gold}\n     model: {p}")
        out.append({"question": r["question"], "gold": gold, "prediction": p})
    return out


def train(args):
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(SEED)
    if not os.path.exists("data/ft_train.jsonl"):
        raise SystemExit("No training data. Run first:  python -m src.train_lora prepare")
    train_rows = load_jsonl("data/ft_train.jsonl")
    val_rows = load_jsonl("data/ft_val.jsonl")

    if args.tiny:   # quick test: does training start, does loss drop, does it save?
        train_rows, val_rows = train_rows[:100], val_rows[:20]
        out_dir, ckpt_dir, log_path = TINY_ADAPTER_DIR, "checkpoints/tiny", "results/finetune_tiny_log.json"
        max_steps, every = 15, 5
    else:
        out_dir, ckpt_dir, log_path = args.output_dir, "checkpoints/full", "results/finetune_train_log.json"
        max_steps, every = -1, 50

    print(f"Loading {args.model} for training ...")
    model, tok = load_for_training(args.model)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    train_ds, dropped_t = build_dataset(tok, train_rows)
    val_ds, dropped_v = build_dataset(tok, val_rows)
    print(f"Training examples: {len(train_ds)}  validation: {len(val_ds)}  "
          f"(dropped as too long: {dropped_t + dropped_v})")

    on_gpu = torch.cuda.is_available()
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * args.epochs
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,   # effective batch = batch_size x grad_accum
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
        fp16=on_gpu,                                   # T4 has no bfloat16
        optim="paged_adamw_8bit" if on_gpu else "adamw_torch",
        report_to="none",
        seed=SEED,
        use_cpu=not on_gpu,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
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

    size_mb = save_adapter_fp16(model, tok, out_dir)
    samples = sanity_check(model, tok, val_rows)

    curve, seen = [], set()
    for h in trainer.state.log_history:
        point = {k: round(h[k], 4) if k != "step" else h[k] for k in ("step", "loss", "eval_loss") if k in h}
        key = tuple(sorted(point.items()))
        if len(point) > 1 and key not in seen:   # skip duplicates and summary lines
            seen.add(key)
            curve.append(point)
    log = {
        "run": "tiny test" if args.tiny else "full",
        "base_model": args.model,
        "method": "QLoRA (4-bit NF4 base + LoRA adapter)",
        "lora": LORA,
        "training": {
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "dropped_too_long": dropped_t + dropped_v,
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
        },
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(100 * trainable / total, 3),
        "final_train_loss": round(result.training_loss, 4),
        "final_val_loss": round(final_eval["eval_loss"], 4),
        "train_minutes": round(minutes, 2),
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
    p = sub.add_parser("prepare")
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--n_val", type=int, default=N_VAL)
    t = sub.add_parser("train")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--tiny", action="store_true", help="5-minute test on 100 examples")
    t.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    t.add_argument("--epochs", type=float, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--grad_accum", type=int, default=4)
    t.add_argument("--output_dir", default=ADAPTER_DIR)
    args = ap.parse_args()
    {"prepare": prepare, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
