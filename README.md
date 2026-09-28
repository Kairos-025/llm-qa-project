# Fine-Tuning LLMs vs. Prompt Engineering and RAG (SQuAD 2.0)

COMP 3710 group project. Team: Aymen Elhamamy, Ekemini John. Supervisor: Dr. Lu.

Compares four ways of improving an LLM at question answering, all scored on the
same test set with the same script:

1. Baseline (zero-shot)
2. Prompt engineering (instructions, few-shot, structured output)
3. Retrieval-Augmented Generation (FAISS + sentence-transformers)
4. Fine-tuning (QLoRA / PEFT)

## Locked project decisions

| Decision | Choice |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct` (Mistral-7B-Instruct as a stretch goal) |
| Test set | 1,000 questions from SQuAD 2.0 validation, seed 42, used by every method |
| No-answer rule | Model must reply exactly `unanswerable`; common variants also count |
| Metrics | EM, F1, lenient accuracy, answerable/unanswerable breakdown, BLEU and ROUGE-L (answerable only) |
| RAG setting | Gold passage is hidden; system must retrieve from all 1,204 validation passages |
| Few-shot examples | Drawn from the train split only, never from the test set |

## Repo layout

```
src/evaluate_qa.py   shared scoring script (every method uses this)
src/data_prep.py     builds test, train, few-shot and RAG corpus files
data/                generated data (not committed, keep a copy on Drive)
results/             predictions and scores per method
notebooks/           Colab/Kaggle notebooks per phase
docs/                report drafts, figures, slides
```

## Setup on Colab or Kaggle

Run in the first notebook cell:

```python
!git clone https://github.com/<your-username>/<repo-name>.git
%cd <repo-name>
!pip install -q -r requirements.txt

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE, switch runtime to GPU")

from google.colab import drive      # Colab only
drive.mount('/content/drive')
```

Then build the data once:

```python
!python -m src.data_prep --test_size 1000
!cp -r data /content/drive/MyDrive/llm-qa-data   # back it up
```

## Scoring a method

Each method writes `results/<method>_preds.jsonl` with lines like
`{"id": "...", "prediction": "raw model output"}`, then:

```bash
python -m src.evaluate_qa --preds results/baseline_preds.jsonl --out results/baseline_scores.json
```

This saves the scores plus a per-question file for error analysis.
