# Fine-Tuning Large Language Models: Evaluating Prompt Engineering and Retrieval-Augmented Generation as Baseline Techniques

COMP 4990 group project
Team: Aymen Elhamamy, Ekemini John
Supervisor: Dr. Lu

This project investigates how fine-tuning improves a large language model on question
answering (SQuAD 2.0). Prompt engineering and retrieval-augmented generation (RAG) serve
as baseline techniques for comparison. All methods are scored on the same test set with
the same evaluation script:

1. Baseline model (zero-shot)
2. Prompt engineering (instructions, few-shot, structured prompts)
3. RAG (FAISS vector store + sentence-transformers)
4. Fine-tuning (parameter-efficient fine-tuning with QLoRA / PEFT), the core of the project

## Locked project decisions

| Decision | Choice |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct`, loaded in 4-bit for every method (Mistral-7B as a stretch goal) |
| Test set | 1,000 questions from SQuAD 2.0 validation (493 answerable, 507 unanswerable), seed 42 |
| No-answer rule | Model must reply `unanswerable`; common variants also count |
| Metrics | Accuracy, F1, BLEU, ROUGE (required), plus Exact Match and answerable/unanswerable breakdown |
| RAG setting | Correct passage hidden; system retrieves from all 1,204 validation passages |
| Few-shot examples | Drawn from the training split only, never from the test set |

## Repo layout

```
src/data_prep.py       builds test, train, few-shot and RAG corpus files
src/evaluate_qa.py     shared scoring script (every method uses this)
src/model_utils.py     loads the model and generates answers
src/prompts.py         prompt templates for each method
src/run_inference.py   runs a method on the test set and scores it
data/                  generated data (not committed; rebuilt in under a minute)
results/               predictions and scores per method
notebooks/             Colab notebooks per phase
docs/                  report drafts, figures, slides
```

## Starting a Colab session (VS Code Colab extension, T4 GPU)

Run this cell at the start of every session. It asks for a GitHub token in a popup:

```python
import os
from getpass import getpass
token = getpass("Paste GitHub token: ")

if not os.path.exists("/content/llm-qa-project"):
    !git clone https://{token}@github.com/Kairos-025/llm-qa-project.git
%cd /content/llm-qa-project
!git remote set-url origin https://{token}@github.com/Kairos-025/llm-qa-project.git
!git pull
!pip install -q -r requirements.txt

if not os.path.exists("data/test.jsonl"):
    !python -m src.data_prep --test_size 1000
```

## Running a method

```bash
python -m src.run_inference --method baseline --limit 20   # quick test
python -m src.run_inference --method baseline              # full run
```

This saves `results/<method>_preds.jsonl` and `results/<method>_scores.json`, and resumes
automatically if the session disconnects.
