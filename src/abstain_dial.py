"""
Extension (fine-tuning depth): an answer/abstain dial for the fine-tuned models.

With greedy decoding a model makes one fixed choice between answering and writing
"unanswerable". This script measures two things for every question instead:

  p_unans   how likely the model finds the reply "Answer: unanswerable" (its full
            probability, read from the model's own output distribution), and
  answer    the best answer the model gives when it is not allowed to start with
            "unanswerable" (greedy decoding with that first token blocked).

A threshold t then decides: abstain if p_unans >= t, otherwise give the answer. Moving t
traces the whole trade-off between answering and abstaining without retraining, so we can
ask whether RAFT's over-caution can be dialled back. The threshold reported as the
result is chosen on the 200 held-out validation questions, never on the test set.

Purely additive: no earlier file is changed.

Usage:
    python -m src.train_lora prepare               # validation questions (data/ft_val.jsonl)
    python -m src.rag build                        # test retrieval (RAFT only)
    python -m src.abstain_dial --method finetuned          # Phase 6 adapter, gold passage
    python -m src.abstain_dial --method raft --k 3         # RAFT adapter, 3 retrieved passages
    python -m src.abstain_dial --method raft_p90 --k 3     # after the RAFT 10% run

  For RAFT, the validation questions get passages from the same search over the training
  split that RAFT training used (the test passages are never searched).

Creates:
    results/dial_<name>.json             the trade-off curve (validation and test), the chosen
                                         threshold, and test scores at it and with greedy decoding
    results/dial_<name>_questions.jsonl  per test question: p_unans, greedy answer, forced answer
    results/<name>_dial_preds.jsonl      the dial's answers at the chosen threshold, so that
                                         compare_results.py and analysis.py include it
"""
import argparse
import json
import math
import os
import time

from src.evaluate_qa import clean_prediction, evaluate, load_jsonl
from src.model_utils import DEFAULT_MODEL
from src.prompts import structured

PREFIX = "Answer:"
ABSTAIN = " unanswerable"
MAX_NEW_TOKENS = 48
# thresholds on a log-odds scale: dense near 0 and 1, where the decisions actually change
GRID = [round(1 / (1 + math.exp(-z / 4)), 6) for z in range(-48, 49)]
CONFIG = {
    "finetuned": {"adapter": "adapters/phi3-squad-lora", "greedy": "finetuned"},
    "raft": {"adapter": "adapters/raft", "greedy": "raft_rag_k{k}"},
    "raft_p90": {"adapter": "adapters/raft_p90", "greedy": "raft_p90_rag_k{k}"},
}


# ---------------------------------------------------------------------------
# model scores
# ---------------------------------------------------------------------------
def encode(tok, chat):
    """Token ids of '<prompt> Answer: unanswerable', where the reply starts (n) and the
    decision point (d): the first token of ' unanswerable' that is not just a space. Some
    tokenizers write the space as its own token, which answers share, so it is skipped."""
    prefix = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + PREFIX
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    f_ids = tok(prefix + ABSTAIN, add_special_tokens=False)["input_ids"]
    n = 0
    while n < min(len(p_ids), len(f_ids)) and p_ids[n] == f_ids[n]:
        n += 1
    d = n
    while d < len(f_ids) - 1 and tok.decode([f_ids[d]]).strip() == "":
        d += 1
    return f_ids, n, d


def p_unanswerable(model, tok, chats, batch_size):
    """Probability of the reply 'Answer: unanswerable' given each prompt (teacher forcing).
    Returns (probabilities, encoded rows, id of the token to block when forcing an answer)."""
    import torch
    rows = [encode(tok, c) for c in chats]
    ban_ids = {f[d] for f, n, d in rows}
    if len(ban_ids) != 1:
        raise SystemExit("The tokenizer splits ' unanswerable' differently across prompts; cannot block it.")

    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))   # similar lengths together
    probs = [0.0] * len(rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            width = max(len(rows[i][0]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for b, i in enumerate(idx):
                f = rows[i][0]
                ids[b, :len(f)] = torch.tensor(f)
                mask[b, :len(f)] = 1                    # right padding: positions stay 0, 1, 2, ...
            logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
            for b, i in enumerate(idx):
                f, n, _ = rows[i]
                pos = torch.arange(n - 1, len(f) - 1, device=logits.device)
                lp = torch.log_softmax(logits[b, pos].float(), dim=-1)
                tgt = torch.tensor(f[n:], device=logits.device)
                probs[i] = math.exp(float(lp.gather(1, tgt[:, None]).sum()))
    return probs, rows, ban_ids.pop()


def forced_answers(model, tok, rows, ban_id, batch_size, return_first=False):
    """Greedy answers that may not start with 'unanswerable': generation starts at the
    decision point and the blocked token is removed from the first step only."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class BlockFirst(LogitsProcessor):
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores):
            if input_ids.shape[-1] == self.start:
                scores[:, ban_id] = -float("inf")
            return scores

    starts = [f[:d] for f, n, d in rows]          # prompt + 'Answer:' (+ a bare space token, if any)
    lead = [f[n:d] for f, n, d in rows]
    order = sorted(range(len(rows)), key=lambda i: len(starts[i]))
    out, first = [""] * len(rows), [None] * len(rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            width = max(len(starts[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for b, i in enumerate(idx):                 # left padding, as for normal generation
                ids[b, width - len(starts[i]):] = torch.tensor(starts[i])
                mask[b, width - len(starts[i]):] = 1
            gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad,
                                 logits_processor=LogitsProcessorList([BlockFirst(width)]))
            for b, i in enumerate(idx):
                new = gen[b, width:].tolist()
                first[i] = new[0] if new else None
                out[i] = f"{PREFIX} {tok.decode(lead[i] + new, skip_special_tokens=True).strip()}"
    return (out, first) if return_first else out


# ---------------------------------------------------------------------------
# the dial
# ---------------------------------------------------------------------------
def option_scores(gold, answers):
    """Per question: (em, f1) if the model abstains, and (em, f1) if it gives `answers`."""
    ids = [g["id"] for g in gold]
    _, abst = evaluate(gold, [{"id": i, "prediction": "Answer: unanswerable"} for i in ids])
    _, ans = evaluate(gold, [{"id": i, "prediction": a} for i, a in zip(ids, answers)])
    return [(a["em"], a["f1"]) for a in abst], [(a["em"], a["f1"]) for a in ans]


def curve(gold, probs, abst, ans):
    unans = [g["is_impossible"] for g in gold]
    n, n_u = len(gold), sum(unans)
    pts = []
    for t in GRID:
        pick = [abst[i] if probs[i] >= t else ans[i] for i in range(n)]
        abstained = [probs[i] >= t for i in range(n)]
        f1 = sum(p[1] for p in pick) / n
        em = sum(p[0] for p in pick) / n
        ans_f1 = sum(pick[i][1] for i in range(n) if not unans[i]) / (n - n_u)
        un_acc = sum(1 for i in range(n) if unans[i] and abstained[i]) / n_u
        refused = sum(1 for i in range(n) if not unans[i] and abstained[i]) / (n - n_u)
        pts.append({"t": t, "f1": round(100 * f1, 2), "em": round(100 * em, 2), "ans_f1": round(100 * ans_f1, 2),
                    "unans_acc": round(100 * un_acc, 2), "refused_answerable": round(100 * refused, 2)})
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(CONFIG), required=True)
    ap.add_argument("--k", type=int, default=None, help="retrieved passages (RAFT); leave out for the gold passage")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--val", default="data/ft_val.jsonl")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N test and validation questions (quick test)")
    args = ap.parse_args()
    if args.method != "finetuned" and not args.k:
        raise SystemExit("RAFT methods need --k (1, 3 or 5).")

    cfg = CONFIG[args.method]
    adapter = args.adapter or cfg["adapter"]
    greedy_name = cfg["greedy"].format(k=args.k)
    name = greedy_name + ("_sample" + str(args.limit) if args.limit else "")
    for path in (args.test, args.val, f"results/{greedy_name}_preds.jsonl", os.path.join(adapter, "adapter_config.json")):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. See the usage notes at the top of this file.")

    test, val = load_jsonl(args.test), load_jsonl(args.val)
    if args.limit:
        test, val = test[: args.limit], val[: args.limit]

    if args.k:   # RAFT: test passages from the Phase 5 search; validation passages from a training-split search
        from src.rag import attach_retrieved, passage_text, rag_prompt
        from src.train_raft import search, train_corpus
        test = attach_retrieved(test)
        corpus, _ = train_corpus(load_jsonl("data/train.jsonl"))
        ranked = search([v["question"] for v in val], corpus, args.k)
        val = [dict(v, retrieved=[passage_text(corpus[j]) for j in ranked[i][: args.k]]) for i, v in enumerate(val)]
        make = lambda ex: rag_prompt(ex, args.k)
    else:
        make = structured

    from src.model_utils import load_model
    print(f"Loading {args.model} + adapter {adapter} ...")
    model, tok = load_model(args.model, adapter)
    start = time.time()

    greedy = {r["id"]: r["prediction"] for r in load_jsonl(f"results/{greedy_name}_preds.jsonl")}
    p_test, rows_test, ban = p_unanswerable(model, tok, [make(t) for t in test], args.batch_size)
    p_val, rows_val, _ = p_unanswerable(model, tok, [make(v) for v in val], args.batch_size)
    print(f"Scored p(unanswerable) for {len(test)} test and {len(val)} validation questions "
          f"(blocked token for forced answers: {tok.decode([ban])!r})")

    # A forced answer is needed only where greedy decoding abstained; elsewhere it is the greedy answer.
    need = [i for i, t in enumerate(test) if clean_prediction(greedy.get(t["id"], "")) == ""]
    forced_test = [greedy.get(t["id"], "") for t in test]
    for i, a in zip(need, forced_answers(model, tok, [rows_test[i] for i in need], ban, args.batch_size)):
        forced_test[i] = a
    forced_val = forced_answers(model, tok, rows_val, ban, args.batch_size)
    minutes = round((time.time() - start) / 60, 2)
    print(f"Generated {len(need)} test and {len(val)} validation answers without abstaining ({minutes} min in total)")

    a_val, n_val = option_scores(val, forced_val)
    a_test, n_test = option_scores(test, forced_test)
    curve_val = curve(val, p_val, a_val, n_val)
    curve_test = curve(test, p_test, a_test, n_test)
    best = max(curve_val, key=lambda p: (p["f1"], -abs(p["t"] - 0.5)))   # ties: the threshold nearest 0.5
    chosen = best["t"]
    at_chosen = next(p for p in curve_test if p["t"] == chosen)

    final = [{"id": t["id"], "prediction": "Answer: unanswerable" if p_test[i] >= chosen else forced_test[i]}
             for i, t in enumerate(test)]
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_dial_preds.jsonl", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scores, per_example = evaluate(test, final)
    scores.update({"method": f"{name}_dial", "model": f"{args.model} + {adapter}", "threshold": chosen,
                   "threshold_chosen_on": f"{len(val)} validation questions ({args.val})"})
    with open(f"results/{name}_dial_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    greedy_scores, _ = evaluate(test, [{"id": t["id"], "prediction": greedy.get(t["id"], "")} for t in test])

    report = {
        "method": name, "adapter": adapter, "k": args.k, "greedy_results": greedy_name,
        "rule": "abstain if p(reply 'Answer: unanswerable') >= threshold, otherwise answer with greedy "
                "decoding where the first token of 'unanswerable' is blocked",
        "threshold_chosen_on_validation": chosen,
        "validation_at_chosen": best,
        "test_at_chosen": at_chosen,
        "test_greedy": {k: greedy_scores[k] for k in ("exact_match", "f1", "has_ans_f1", "no_ans_accuracy")},
        "minutes": minutes,
        "curve_test": curve_test,
        "curve_validation": curve_val,
    }
    with open(f"results/dial_{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(f"results/dial_{name}_questions.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(test):
            f.write(json.dumps({"id": t["id"], "p_unans": round(p_test[i], 6), "greedy": greedy.get(t["id"], ""),
                                "forced": forced_test[i]}, ensure_ascii=False) + "\n")

    g = report["test_greedy"]
    print(f"\nThreshold chosen on validation: {chosen}")
    print(f"{'':<22}{'F1':>8}{'answerable F1':>16}{'unanswerable acc':>19}")
    print(f"{'greedy (as before)':<22}{g['f1']:>8.2f}{g['has_ans_f1']:>16.2f}{g['no_ans_accuracy']:>19.2f}")
    print(f"{'dial, chosen t':<22}{at_chosen['f1']:>8.2f}{at_chosen['ans_f1']:>16.2f}{at_chosen['unans_acc']:>19.2f}")
    top = max(curve_test, key=lambda p: p["f1"])
    print(f"{'best t on test (ref)':<22}{top['f1']:>8.2f}{top['ans_f1']:>16.2f}{top['unans_acc']:>19.2f}   (t = {top['t']})")
    print(f"\nSaved results/dial_{name}.json, results/dial_{name}_questions.jsonl and results/{name}_dial_preds.jsonl")


if __name__ == "__main__":
    main()
