"""
Phase 7: deeper analysis of every finished run, using the saved answers of
every run. Needs no GPU and no model.

What it adds beyond the comparison table:
  1. 95% confidence intervals (bootstrap) for every method, so we can say
     which differences are real and which are noise.
  2. Paired comparisons: does method A really differ from method B on the same
     questions, in F1, answerable F1 and unanswerable accuracy?
  3. Error types: exact / partial / wrong answer / wrongly said "unanswerable"
     on answerable questions, and correct abstain / made-up answer on
     unanswerable ones.
  4. Scores by question type (what, who, when, where, which, why, how, how many).
  5. Training-size curve, if the size experiments were run.

Reads results/<method>_preds.jsonl for every full run and the test set
(data/test.jsonl, rebuilt with `python -m src.data_prep --test_size 1000`).

Usage:
    python -m src.analysis
    python -m src.analysis --compare finetuned fewshot5 --compare finetuned finetuned_natural

Saves to results/analysis/: confidence_intervals.csv, paired_tests.csv,
error_types.csv, by_question_type.csv, and charts (*.png).
"""
import argparse
import csv
import glob
import json
import os
import random
import re

from src.evaluate_qa import load_jsonl

OUT = "results/analysis"
SEED = 42
N_BOOT = 2000
QTYPES = ["how many", "how much", "what", "which", "who", "when", "where", "why", "how", "other"]

# chart colours (validated categorical palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def family(method):
    if method.startswith("finetuned"):
        return "Fine-tuning", AQUA
    if method.startswith("rag"):
        return "RAG", ORANGE
    if method == "baseline":
        return "Baseline", GRAY
    return "Prompt engineering", BLUE


def qtype(question):
    q = question.lower()
    for t in QTYPES[:-1]:
        if re.search(rf"\b{t}\b", q):
            return t
    return "other"


def load_runs(test):
    """Every full run, keyed by method name. Re-scores each run from its saved
    predictions (results/<method>_preds.jsonl, which are on GitHub) with the
    shared scorer, so no per-question files need to be kept between sessions."""
    from src.evaluate_qa import evaluate
    runs = {}
    for path in sorted(glob.glob("results/*_preds.jsonl")):
        method = os.path.basename(path).replace("_preds.jsonl", "")
        if "_sample" in method:
            continue
        preds = load_jsonl(path)
        if len(preds) < len(test):
            print(f"Skipping {method}: only {len(preds)} of {len(test)} answers")
            continue
        _, per_example = evaluate(test, preds)
        runs[method] = {r["id"]: r for r in per_example}
    return runs


def per_question(test, details):
    """Per-question outcomes for one method, in test-set order."""
    rows = []
    for t in test:
        d = details.get(t["id"])
        if d is None:
            continue
        abstained = d["prediction"] == ""
        if t["is_impossible"]:
            kind = "correct abstain" if abstained else "made-up answer"
        elif abstained:
            kind = "wrongly abstained"
        elif d["em"]:
            kind = "exact"
        elif d["f1"] > 0:
            kind = "partial"
        else:
            kind = "wrong answer"
        rows.append({"id": t["id"], "unans": t["is_impossible"], "qtype": qtype(t["question"]),
                     "em": d["em"], "f1": d["f1"], "abstain_ok": int(abstained) if t["is_impossible"] else None,
                     "kind": kind})
    return rows


def metrics(rows):
    ans = [r for r in rows if not r["unans"]]
    un = [r for r in rows if r["unans"]]
    m = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")
    return {"f1": m([r["f1"] for r in rows]),
            "em": m([r["em"] for r in rows]),
            "ans_f1": m([r["f1"] for r in ans]),
            "unans_acc": m([r["abstain_ok"] for r in un])}


def bootstrap(rows, rng):
    """Resample questions with replacement; returns metric -> (low, high)."""
    n = len(rows)
    samples = {k: [] for k in ("f1", "em", "ans_f1", "unans_acc")}
    for _ in range(N_BOOT):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        for k, v in metrics(s).items():
            samples[k].append(v)
    ci = {}
    for k, vals in samples.items():
        vals = sorted(v for v in vals if v == v)
        ci[k] = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])
    return ci


def paired(rows_a, rows_b, rng):
    """Paired bootstrap: resample the SAME questions for both methods, so the
    comparison is fair. For each measure: the difference A - B, its 95%
    interval, and a two-sided p-value (how often the sign flips)."""
    b = {r["id"]: r for r in rows_b}
    pairs = [(r, b[r["id"]]) for r in rows_a if r["id"] in b]
    n = len(pairs)

    def diffs(sample):
        ans = [(x, y) for x, y in sample if not x["unans"]]
        un = [(x, y) for x, y in sample if x["unans"]]
        d = lambda ps, key: 100 * sum(x[key] - y[key] for x, y in ps) / len(ps) if ps else 0.0
        return {"f1": d(sample, "f1"), "ans_f1": d(ans, "f1"), "unans_acc": d(un, "abstain_ok")}

    point = diffs(pairs)
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        for k, v in diffs(s).items():
            boot[k].append(v)
    out = {"n": n}
    for k, vals in boot.items():
        vals.sort()
        below = sum(v <= 0 for v in vals) / N_BOOT
        above = sum(v >= 0 for v in vals) / N_BOOT
        out[f"{k}_diff"] = round(point[k], 2)
        out[f"{k}_ci_low"] = round(vals[int(0.025 * N_BOOT)], 2)
        out[f"{k}_ci_high"] = round(vals[int(0.975 * N_BOOT) - 1], 2)
        out[f"{k}_p"] = round(min(1.0, 2 * min(below, above)), 4)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def chart_ci(ci_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = sorted(ci_rows, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2), dpi=200)
    for i, r in enumerate(rows):
        c = family(r["method"])[1]
        ax.plot([r["f1_low"], r["f1_high"]], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r["f1"], i, "o", color=c, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(r["f1_high"] + 1, i, f"{r['f1']:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["method"] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 with 95% confidence interval (bootstrap, 2,000 resamples)", color=INK2)
    seen = {}
    for r in rows:
        name, c = family(r["method"])
        seen[name] = c
    ax.legend(handles=[Line2D([], [], color=c, marker="o", linewidth=2, label=n) for n, c in seen.items()],
              loc="lower right", frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/f1_confidence_intervals.png")
    plt.close(fig)


def chart_errors(err_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds_a = [("exact", AQUA), ("partial", BLUE), ("wrong answer", ORANGE), ("wrongly abstained", YELLOW)]
    kinds_u = [("correct abstain", AQUA), ("made-up answer", ORANGE)]
    rows = sorted(err_rows, key=lambda r: r["exact"] + r["correct abstain"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.6), dpi=200, sharey=True)
    for ax, kinds, title, total in ((axes[0], kinds_a, "Answerable questions", "n_answerable"),
                                    (axes[1], kinds_u, "Unanswerable questions", "n_unanswerable")):
        for i, r in enumerate(rows):
            left = 0
            for k, c in kinds:
                w = 100 * r[k] / r[total]
                ax.barh(i, w, left=left, color=c, height=0.62, edgecolor="white", linewidth=1.5)
                left += w
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("% of questions", color=INK2)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in kinds], labels=[k for k, _ in kinds],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.4 / max(1, len(rows))), ncol=len(kinds),
                  frameon=False, fontsize=8)
        _style(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r["method"] for r in rows])
    fig.tight_layout()
    fig.savefig(f"{OUT}/error_types.png", bbox_inches="tight")
    plt.close(fig)


def chart_size_curve(points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points.sort()
    xs = [p[0] for p in points]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
    ends = []
    for key, label, c in (("f1", "F1 (all questions)", AQUA), ("ans_f1", "F1 (answerable)", BLUE),
                          ("unans_acc", "Unanswerable accuracy", ORANGE)):
        ys = [p[1][key] for p in points]
        ax.plot(xs, ys, "-o", color=c, linewidth=2, markersize=6, markeredgecolor="white", label=label)
        ends.append([ys[-1], ys[-1]])
    ends.sort()                       # spread end labels so they never overlap
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 5)
    for y, ly in ends:
        ax.text(xs[-1] * 1.06, ly, f"{y:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(xs[0] / 1.2, xs[-1] * 1.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.set_xlabel("Training examples (50% unanswerable, log scale)", color=INK2)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Fine-tuning: score vs. amount of training data", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_size_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"), default=[],
                    help="paired test: is method A better than method B? (repeatable)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    test = load_jsonl(args.test)
    runs = load_runs(test)
    if not runs:
        raise SystemExit("No full runs found in results/.")
    per_q = {m: per_question(test, d) for m, d in runs.items()}

    # 1. confidence intervals
    ci_rows = []
    for m, rows in per_q.items():
        mt, ci = metrics(rows), bootstrap(rows, rng)
        ci_rows.append({"method": m, "n": len(rows),
                        "f1": round(mt["f1"], 2), "f1_low": round(ci["f1"][0], 2), "f1_high": round(ci["f1"][1], 2),
                        "em": round(mt["em"], 2), "em_low": round(ci["em"][0], 2), "em_high": round(ci["em"][1], 2),
                        "ans_f1": round(mt["ans_f1"], 2), "ans_f1_low": round(ci["ans_f1"][0], 2),
                        "ans_f1_high": round(ci["ans_f1"][1], 2),
                        "unans_acc": round(mt["unans_acc"], 2), "unans_low": round(ci["unans_acc"][0], 2),
                        "unans_high": round(ci["unans_acc"][1], 2)})
    ci_rows.sort(key=lambda r: -r["f1"])
    write_csv(f"{OUT}/confidence_intervals.csv", ci_rows)
    print(f"{'method':<22}{'F1':>7}  95% CI")
    for r in ci_rows:
        print(f"{r['method']:<22}{r['f1']:>7.2f}  [{r['f1_low']:.2f}, {r['f1_high']:.2f}]")

    # 2. paired tests (defaults: best vs runner-up, and fine-tuned vs best other method)
    pairs = [tuple(p) for p in args.compare]
    if not pairs and len(ci_rows) > 1:
        pairs.append((ci_rows[0]["method"], ci_rows[1]["method"]))
        ft = [r["method"] for r in ci_rows if r["method"] == "finetuned"]
        others = [r["method"] for r in ci_rows if not r["method"].startswith("finetuned")]
        if ft and others and (ft[0], others[0]) not in pairs:
            pairs.append((ft[0], others[0]))
    paired_rows = []
    for a, b in pairs:
        if a not in per_q or b not in per_q:
            print(f"Skipping {a} vs {b}: missing results")
            continue
        res = paired(per_q[a], per_q[b], rng)
        paired_rows.append({"a": a, "b": b, **res})
        print(f"\n{a} vs {b} (difference A - B, 95% interval, p):")
        for k, label in (("f1", "F1"), ("ans_f1", "Answerable F1"), ("unans_acc", "Unanswerable acc.")):
            sig = "real difference" if res[f"{k}_p"] < 0.05 else "could be chance"
            pv = "< 0.001" if res[f"{k}_p"] < 0.001 else f"{res[f'{k}_p']:.3f}"
            print(f"  {label:<18}{res[f'{k}_diff']:+7.2f}  [{res[f'{k}_ci_low']:+.2f}, {res[f'{k}_ci_high']:+.2f}]"
                  f"  p {pv:<8} {sig}")
    if paired_rows:
        write_csv(f"{OUT}/paired_tests.csv", paired_rows)

    # 3. error types
    kinds = ["exact", "partial", "wrong answer", "wrongly abstained", "correct abstain", "made-up answer"]
    err_rows = []
    for m, rows in per_q.items():
        row = {"method": m, "n_answerable": sum(not r["unans"] for r in rows),
               "n_unanswerable": sum(r["unans"] for r in rows)}
        for k in kinds:
            row[k] = sum(r["kind"] == k for r in rows)
        err_rows.append(row)
    write_csv(f"{OUT}/error_types.csv", err_rows)

    # 4. by question type (answerable F1 and unanswerable accuracy per type)
    qt_rows = []
    for m, rows in per_q.items():
        for t in QTYPES:
            sub = [r for r in rows if r["qtype"] == t]
            if not sub:
                continue
            mt = metrics(sub)
            qt_rows.append({"method": m, "qtype": t, "n": len(sub),
                            "n_answerable": sum(not r["unans"] for r in sub),
                            "ans_f1": round(mt["ans_f1"], 2) if mt["ans_f1"] == mt["ans_f1"] else "",
                            "unans_acc": round(mt["unans_acc"], 2) if mt["unans_acc"] == mt["unans_acc"] else ""})
    write_csv(f"{OUT}/by_question_type.csv", qt_rows)

    # 5. training-size curve (finetuned_size<N> runs + the main 4,000-example run)
    points = []
    for m, rows in per_q.items():
        hit = re.fullmatch(r"finetuned_size(\d+)", m)
        if hit:
            points.append((int(hit.group(1)), metrics(rows)))
    if points and "finetuned" in per_q:
        points.append((4000, metrics(per_q["finetuned"])))

    chart_ci(ci_rows)
    chart_errors(err_rows)
    if len(points) >= 2:
        chart_size_curve(points)
    print(f"\nSaved tables and charts to {OUT}/")


if __name__ == "__main__":
    main()