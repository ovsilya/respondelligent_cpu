#!/usr/bin/env python3

"""FastAPI serving app for mBART review-response generation.

Loads the spaCy pipelines and the mBART checkpoint at import time so the app is
ready to serve as soon as the module is imported (e.g. by ``uvicorn``).

Config resolution (highest priority first):

1. ``sys.argv[1]`` — a path passed on the command line, e.g.
   ``python -m readvisor.serving.main readvisor/serving/config.json``;
2. the ``READVISOR_CONFIG`` environment variable;
3. the packaged default ``readvisor/serving/config.json`` (next to this file).

Because the path is resolved from an env var / packaged default when no argv is
given, the module is importable without arguments, e.g.::

    uvicorn readvisor.serving.main:app

NOTE: for the relative paths inside the config file to resolve, run from the
directory the config paths are written relative to (see config.json).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from readvisor.model.inference import InferenceSimplifier
from readvisor.serving.generation import (
    batchify,
    generate,
    load_model_from_checkpoint,
    prepare_input_for_mbart,
)
from readvisor.serving.postprocess import (
    add_company_label_to_input_data,
    load_company_data_from_text,
    load_company_label_mapping,
    load_config,
    postprocess_object,
)
from readvisor.utils.spacy_utils import (
    add_gazetteer_to_nlp,
    add_special_tokens_to_tokenizer,
    load_spacy_pipe,
)

# Configure logging once, here, for the whole serving app.
logging.basicConfig(format="[INFO] %(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def resolve_config_path() -> str:
    """Resolve the config path from argv, then ``READVISOR_CONFIG``, then default."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return os.environ.get("READVISOR_CONFIG", str(DEFAULT_CONFIG_PATH))


app = FastAPI()

config_file_path = resolve_config_path()

cfg = load_config(config_file_path)

###########
# Load data
###########

if cfg.response_generator.type == "mBART_DER" and cfg.data.company_label_mapping:
    company_label_mapping = load_company_label_mapping(cfg.data.company_label_mapping)
else:
    company_label_mapping = None

###################
# Load spacy models
###################

DE_NLP = load_spacy_pipe(cfg.models.spacy_de)
add_special_tokens_to_tokenizer(DE_NLP)

EN_NLP = load_spacy_pipe(cfg.models.spacy_en)
add_special_tokens_to_tokenizer(EN_NLP)

if cfg.data.company_gazetteer:
    company_names = load_company_data_from_text(cfg.data.company_gazetteer)
    DE_MATCHER = add_gazetteer_to_nlp(DE_NLP, company_names)
    EN_MATCHER = add_gazetteer_to_nlp(EN_NLP, company_names)
else:
    DE_MATCHER = None
    EN_MATCHER = None

#######################
# Load generation model
#######################

model_arg_parser = argparse.ArgumentParser(description="simplification")
model_arg_parser = InferenceSimplifier.add_model_specific_args(model_arg_parser, os.getcwd())
MODEL_ARGS = model_arg_parser.parse_args(cfg.response_generator.model_args)
MODEL = load_model_from_checkpoint(MODEL_ARGS)


#####
# App
#####

class ReviewInput(BaseModel):
    review: str
    meta: dict


@app.get("/")
def read_root():
    return ["Hi there! To query the model with SWAGGER UI add `/docs` to the URL."]


@app.post("/item/")
def post_item(item: ReviewInput):
    """Generate and post-process responses for a single review payload."""

    # ensure inupt item is of type :dict:
    if not isinstance(item, dict):
        try:
            item = item.dict()
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert input item of type {type(item)} to type dict."
            ) from e

    # prepare review input for mbart:
    if company_label_mapping is not None:
        item = add_company_label_to_input_data(item, company_label_mapping)
    model_input = prepare_input_for_mbart(item, cfg)

    # convert input item to valid batch input and generate responses
    for batch in batchify(MODEL, [model_input]):
        responses = generate(MODEL, MODEL_ARGS, batch)

    # add generated responses to item
    item["responses"] = responses

    # apply postprocessing
    if item["meta"]["lang"] == "de":
        item = postprocess_object(item, DE_NLP, DE_MATCHER)
    elif item["meta"]["lang"] == "en":
        item = postprocess_object(item, EN_NLP, EN_MATCHER)

    return item


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
