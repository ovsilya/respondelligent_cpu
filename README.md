# ReAdvisor — Automated Review-Response Generation (Fine-Tuned mBART)

Code base for **automatic customer-review response generation**, developed in the ReAdvisor project (for re:spondelligent — AI-assisted review responses for hospitality/gastronomy). It generates on-brand, contextually appropriate replies to hotel/restaurant reviews in **German and English** using a **fine-tuned mBART** sequence-to-sequence model, served behind a **FastAPI** API, with a full pre/post-processing and evaluation pipeline.

## Table of Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Model](#model)
- [Serving (FastAPI)](#serving-fastapi)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Run the Service](#run-the-service)
- [Model Inference (demo)](#model-inference-demo)
- [Dependencies of Note](#dependencies-of-note)
- [Tech Stack](#tech-stack)

## Overview

Given a review and its metadata (language, domain, rating, establishment, greetings/salutations, etc.), the system produces several candidate responses, reranks them by semantic similarity to the review, and post-processes them (entity handling, truecasing, placeholder substitution) to yield a polished, on-brand reply. It supports **per-establishment style** via company-specific label tags.

The code is organized as a single installable package, `readvisor`, with five subpackages: `data` (preparation), `model` (fine-tuning/inference), `serving` (FastAPI app), `evaluation`, and `utils` (shared helpers).

## Pipeline

1. **Data preparation** (`readvisor/data/`) — clean and convert raw exports (JSON → well-formed CSV), merge sources, **mask entities**, **mask/normalize greetings & salutations** (Flair-based), and generate mBART-ready training files with **company/label tags prepended** (e.g. `<est_74>`).
2. **Fine-tuning** (`readvisor/model/`) — fine-tune a **trimmed mBART** on the review→response corpus (`train.py`, `trim_mbart.py`, drivers in `scripts/`), tags included, max input/output length 512. PyTorch-Lightning training.
3. **Inference / generation** (`readvisor/model/inference.py`, `readvisor/serving/generation.py`) — beam search (beam size 4) + sampling (top-k 10, temperature 1.2), `num_return_sequences=4` → multiple candidates per review.
4. **Post-processing** (`readvisor/serving/postprocess.py`, spaCy, per language) — NER-based entity identification and replacement (e.g. `<NAME>` → author), and a **company-name gazetteer** injected via spaCy `PhraseMatcher`.
5. **Reranking** — **SentenceTransformers** rescore candidates by semantic textual similarity between the source review and each generated hypothesis.
6. **Evaluation** (`readvisor/evaluation/`) — classifier metrics, surface + semantic **repetition** metrics, semantic-similarity metric, hypothesis reranking, and **Prodigy** recipes for human-in-the-loop review.

## Model

- **mBART** (multilingual BART, encoder-decoder seq2seq), **fine-tuned** on hospitality review-response pairs.
- Variant `mBART_DER` uses **company-specific labels** (e.g. `<est_74>`) for per-establishment response style.
- Trained with **PyTorch-Lightning** (torch 1.6) using a **trimmed-mBART** transformers fork (`ZurichNLP/transformers@trim_mbart`) to reduce the multilingual vocabulary/model size.
- **Model files are not included** in this repo due to size — they are shipped separately; place them under a model directory and point `readvisor/serving/config.json` at it (symlinks recommended).

## Serving (FastAPI)

- Entry point: `readvisor/serving/main.py`, config-driven via `readvisor/serving/config.json`. The config path is resolved from the CLI argument, else the `READVISOR_CONFIG` env var, else the packaged default.
- On startup: loads the company-label mapping, the DE and EN spaCy pipelines (+ gazetteer matchers), and the mBART generation model from a checkpoint.
- Endpoints: `GET /` (info), **`POST /item/`** — accepts `{ review, meta }`, returns the item with generated `responses`; interactive Swagger UI at `/docs`.

## Repository Structure

```
readvisor/
├── data/         # preparation: clean/convert/merge, entity+greeting masking, mBART input generation
├── model/        # mBART fine-tuning, inference, vocab trimming (+ scripts/ drivers)
├── serving/      # FastAPI app: main.py, generation.py, postprocess.py, config.json, egs/, Dockerfile
├── evaluation/   # metrics, reranking, Prodigy recipes
└── utils/        # shared: spacy_utils, special_tokens
data/latest_training_files_mbart/   # tokenizer/config artifacts and label files (large files gitignored)
pyproject.toml    # packaging + dependency extras (serve / train / dev) + ruff/pytest config
```

## Setup

```bash
python3.8 -m venv .venv && source .venv/bin/activate
pip install .              # serving runtime
pip install ".[train]"     # + data-prep & fine-tuning
pip install ".[dev]"       # + notebooks, ruff, pytest
```

Dependencies (and exact 2021 version pins) live in `pyproject.toml`. Obtain the model files separately and point `readvisor/serving/config.json` at their location.

## Run the Service

```bash
# with Docker (build from the repo root so pyproject + readvisor are in context)
docker build -m 14g -f readvisor/serving/Dockerfile -t respogen .
docker run -p 9530:8000 respogen
# → open http://0.0.0.0:8000/docs

# or directly
python -m readvisor.serving.main readvisor/serving/config.json
```

## Model Inference (demo)

`POST /item/` with a review + metadata, e.g.:

```json
{
  "review": "Wunderschöne grosse Zimmer und ein reichhaltiges Frühstück ...",
  "meta": {
    "lang": "de", "domain": "hotel", "rating": 5, "author": "Hans Heissen",
    "company": "Hotel du Commerce",
    "greetings": ["Guten Tag NAMEPLACEHOLDER,"],
    "salutations": ["Mit freundlichen Grüssen, Hotel du Commerce."]
  }
}
```

The response object returns the input plus a list of generated `responses` (reranked, post-processed). An offline smoke test with sample reviews lives in `readvisor/serving/egs/demo_reviews.json`.

## Dependencies of Note

- **spaCy 2.3** with `de_core_news_md` / `en_core_web_md` (Prodigy-tuned NER); company gazetteer via `PhraseMatcher`.
- **SentenceTransformers** for hypothesis reranking (`xx_paraphrase_xlm_r_multilingual_v1`, `xx_distiluse_base_multilingual_cased`).
- **Transformers** — custom trimmed-mBART fork (`ZurichNLP/transformers@trim_mbart`).
- **PyTorch-Lightning 1.1.6**, torch 1.6, sentencepiece, sacrebleu/rouge (eval), Flair.

## Tech Stack

Python 3.8 · **mBART** (fine-tuned, trimmed) · **PyTorch-Lightning** · **FastAPI** + uvicorn · **spaCy** (DE/EN, PhraseMatcher gazetteer) · **SentenceTransformers** (reranking) · Flair · Docker · GCP Cloud Build.

---

*This code base was modernized in 2026 (packaging, structure, dead-code removal, import hygiene, typing/docstrings) while intentionally preserving the original 2021 frameworks and versions.*
