# Safe Public Comment Rewrite Reviewer (PS 215)

An **evaluation prototype** (not a rewriter) that inspects an original toxic comment alongside a candidate rewrite and reports whether toxicity dropped and whether the original meaning was preserved—without ever repeating harmful language in its outputs.

## Tech Stack

- **Toxicity Classifier:** Toxic-BERT (`unitary/toxic-bert`) via `detoxify` (`Detoxify('original')`)
- **Meaning Similarity:** `sentence-transformers` (`all-MiniLM-L6-v2`, cosine similarity)
- **LLM Judge:** Google Gemini via `google-genai` (model and API key loaded from environment variables via `python-dotenv`)
- **Word Masking:** `better-profanity` + per-word Toxic-BERT check (`toxic > 0.5`)
- **UI:** Streamlit
- **Dataset:** Local ParaDetox file (`Dataset Paradetox.xlsx`, configured via `DATASET_PATH` in `config.py`)

## Project Structure

```text
app.py                  # Streamlit UI
config.py               # Thresholds, constants, and DATASET_PATH
data.py                 # Local ParaDetox dataset loader (load_pairs)
toxicity.py             # Toxic-BERT scoring and offensive-word masking
similarity.py           # MiniLM cosine similarity
judge.py                # Gemini meaning-preservation judge with disk cache
verdict.py              # Verdict decision logic
report.py               # Masked evaluation report builder
tests/test_masking.py   # Safety tests preventing unmasked offensive output
eval/build_eval_set.py  # Labelled evaluation set builder from ParaDetox
eval/run_eval.py        # Pipeline evaluator outputting CSV + accuracy metrics
requirements.txt
.env.example
.gitignore
```

## Setup

1. Create and activate a virtual environment (Python 3.12 recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY` and `GEMINI_MODEL` (for example `gemini-2.5-flash`). If Gemini credentials are omitted, the evaluator safely falls back to MiniLM similarity (`< 0.35` -> `Meaning lost`).

## Running the App

```bash
streamlit run app.py
```

## Running Tests & Evaluation

Run safety tests:
```bash
pytest
```

Build the evaluation dataset and run the evaluation pipeline:
```bash
python eval/build_eval_set.py
python eval/run_eval.py
```

