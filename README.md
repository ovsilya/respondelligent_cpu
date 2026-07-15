# ReAdvisor — Automated Review-Response Generation (Fine-Tuned mBART)

Code base for **automatic customer-review response generation**, developed in the ReAdvisor project (for re:spondelligent — AI-assisted review responses for hospitality/gastronomy). It generates on-brand, contextually appropriate replies to hotel/restaurant reviews in **German and English** using a **fine-tuned mBART** sequence-to-sequence model, served behind a **FastAPI** API, with a full pre/postprocessing and evaluation pipeline.

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

## Pipeline

1. **Data preparation** (`data_prep/`) — clean and convert raw exports (JSON → well-formed CSV), merge sources, **mask entities**, **remove/normalize greetings & salutations** (incl. a Flair-based variant), truecasing, and generate mBART-ready training files with **company/label tags prepended** (e.g. `<est_74>`).
2. **Fine-tuning** (`fastapi_app/app/src/mbart_hospo_respo/`) — fine-tune a **trimmed mBART** on the review→response corpus (`train.py`, `run_finetuning.sh`, `trim_mbart.py`), tags included, max input/output length 512. PyTorch Lightning training.
3. **Inference / generation** (`generate.py`, `inference.py`) — beam search (beam size 4) + sampling (top-k 10, temperature 1.2), `num_return_sequences=4` → multiple candidates per review.
4. **Post-processing** (spaCy, per language) — NER-based entity identification and replacement (e.g. `NAMEPLACEHOLDER` → author), truecasing, and a **company-name gazetteer** injected via spaCy `PhraseMatcher`.
5. **Reranking** — **SentenceTransformers** rescore candidates by semantic textual similarity between the source review and each generated hypothesis.
6. **Evaluation** (`evaluation/`) — classifier metrics, surface + semantic **repetition** metrics, semantic-similarity metric, hypothesis reranking, and **Prodigy** recipes for human-in-the-loop review.

## Model

- **mBART** (multilingual BART, encoder-decoder seq2seq), **fine-tuned** on hospitality review-response pairs.
- Variant `mBART_DER` uses **company-specific labels** (e.g. `<est_74>`) for per-establishment response style.
- Trained with **PyTorch Lightning** (torch 1.6) using a **trimmed-mBART** transformers fork (`ZurichNLP/transformers@trim_mbart`) to reduce the multilingual vocabulary/model size.
- **Model files are not included** in this repo due to size — they are shipped separately; place them under `./app/models/` and update paths in `config.json` (symlinks recommended).

## Serving (FastAPI)

- Entry point: `fastapi_app/app/main.py`, config-driven via `config.json` (omegaconf).
- On startup: loads the company-label mapping, the DE and EN spaCy pipelines (+ gazetteer matchers), and the mBART generation model from a checkpoint.
- Endpoints: `GET /` (info), **`POST /item/`** — accepts `{ review, meta }`, returns the item with generated `responses`; interactive Swagger UI at `/docs`.

## Repository Structure

- `data_prep/` — preprocessing scripts + `utils_pkg/` (cleaning, greetings, entity masking, truecasing, sentiment, spaCy utils).
- `evaluation/` — metrics (repetition, semantic similarity), reranking, and Prodigy recipes.
- `fastapi_app/` — Dockerized FastAPI service (`app/main.py`, `app/src/generate.py`, `app/src/mbart_hospo_respo/`), `config.json`, `cloudbuild.yaml`.
- `data/latest_training_files_mbart/` — tokenizer/config artifacts and label files.

## Setup

```bash
conda create --name respondelligent python=3.8.5
conda activate respondelligent
conda install cudatoolkit=<your-cuda-version>   # optional, for GPU
pip install -r requirements.txt
```

Obtain the model files separately and place them under `fastapi_app/app/models/`, then align the paths in `fastapi_app/app/config.json`.

## Run the Service

```bash
# with Docker (from fastapi_app/)
docker build -m 14g -t respogen .
docker run -p 9530:8000 respogen
# → open http://0.0.0.0:8000/docs

# or directly
python fastapi_app/app/main.py fastapi_app/app/config.json
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

The response object returns the input plus a list of generated `responses` (reranked, post-processed).

## Dependencies of Note

- **spaCy 2.3** with `de_core_news_md` / `en_core_web_md` (Prodigy-tuned NER); company gazetteer via `PhraseMatcher`.
- **SentenceTransformers** for hypothesis reranking (`xx_paraphrase_xlm_r_multilingual_v1`, `xx_distiluse_base_multilingual_cased`; downloaded at container start).
- **Transformers** — custom trimmed-mBART fork (`ZurichNLP/transformers@trim_mbart`).
- **PyTorch Lightning 1.1.6**, torch 1.6, sentencepiece, sacrebleu/rouge (eval), Flair.

## Tech Stack

Python 3.8.5 · **mBART** (fine-tuned, trimmed) · **PyTorch Lightning** · **FastAPI** + uvicorn · **spaCy** (DE/EN, PhraseMatcher gazetteer) · **SentenceTransformers** (reranking) · Flair · Docker · GCP Cloud Build.
