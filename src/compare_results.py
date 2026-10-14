"""
Prints one table comparing every method that has been scored so far, and
saves it as results/comparison.csv (for the report and Phase 7 charts).

Usage:
    python -m src.compare_results
"""
import csv
import glob
import json

COLUMNS = [
    ("method", "Method"),
    ("exact_match", "EM"),
    ("f1", "F1"),
    ("lenient_accuracy", "Lenient acc"),
    ("has_ans_f1", "Ans F1"),
    ("no_ans_accuracy", "Unans acc"),
    ("bleu", "BLEU"),
    ("rouge_l", "ROUGE-L"),
    ("generation_minutes", "Minutes"),
]


def main():
    rows = []
    for path in sorted(glob.glob("results/*_scores.json")):
        if "_sample" in path:
            continue
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        print("No full-run scores found in results/")
        return
    rows.sort(key=lambda r: r.get("f1") or 0, reverse=True)

    header = [label for _, label in COLUMNS]
    table = [[("" if r.get(k) is None else r.get(k)) for k, _ in COLUMNS] for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]
    line = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for t in table:
        print(line(t))

    with open("results/comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print("\nSaved results/comparison.csv (sorted by F1)")


if __name__ == "__main__":
    main()
