#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

NOTE: for paths in config file to match, run from one-level
up in directory structure

Example call (from `../`):
    python app/main.py app/config.json --test

"""

import os
import sys
import json
from typing import List, Dict

from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# dependent on trained fairseq model
from src.spacy_utils import *
from src.IO_utils import *
from src.generate import *

pwd = os.path.dirname(os.path.realpath(__file__))

app = FastAPI()

config_file_path = sys.argv[1]

cfg = load_config(config_file_path)

###########
# Load data
###########

if cfg.response_generator.type == 'mBART_DER' and cfg.data.company_label_mapping:
    company_label_mapping = load_company_label_mapping(cfg.data.company_label_mapping)
else:
    company_label_mapping = None

###################
# Load spacy models
###################

# DE_NLP = load_spacy_pipe(cfg.models.spacy_de, sbert_model=cfg.models.sentence_transformers.dist)
DE_NLP = load_spacy_pipe(cfg.models.spacy_de)
add_special_tokens_to_tokenizer(DE_NLP)

# EN_NLP = load_spacy_pipe(cfg.models.spacy_en, sbert_model=cfg.models.sentence_transformers.dist)
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
# print(MODEL_ARGS)
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

    # ensure inupt item is of type :dict:
    if not isinstance(item, dict):
        try:
            item = item.dict()
        except:
            raise RuntimeError(f'Failed to convert input item of type {type(item)} to type dict.')
    
    # prepare review input for mbart:
    if company_label_mapping is not None:
        item = add_company_label_to_input_data(item, company_label_mapping)
    model_input = prepare_input_for_mbart(item, cfg)
    
    # convert input item to valid batch input and generate responses
    for batch in batchify(MODEL, [model_input]):
        responses = generate(MODEL, MODEL_ARGS, batch)

    # add generated responses to item
    item['responses'] = responses

    # apply postprocessing
    if item['meta']['lang'] == 'de':
        item = postprocess_object(item, DE_NLP, DE_MATCHER)
    elif item['meta']['lang'] == 'en':
        item = postprocess_object(item, EN_NLP, EN_MATCHER)

    return item

def test():

    # postprocessing pipeline
    print('running test with:', str(cfg.data.test_int_outputs))
    with open(cfg.data.test_int_outputs, 'r', encoding='utf8') as f:
        response_items = json.loads(f.read())
        for item in response_items:
            if item['meta']['lang'] == 'de':
                item = postprocess_object(item, DE_NLP, DE_MATCHER)
            elif item['meta']['lang'] == 'en':
                item = postprocess_object(item, EN_NLP, EN_MATCHER)
            print('*'*50)
            print(item)
            print('*'*50)   

    # generation model
    print('running test with:', str(cfg.data.test_inputs))
    with open(cfg.data.test_inputs, 'r', encoding='utf8') as f:
        input_items = json.loads(f.read())
        for item in input_items:
            result = post_item(item)
            print('*'*50)
            print(result)
            print('*'*50)   

    return

if __name__ == "__main__":
    if not sys.argv[-1] == '--test':
        uvicorn.run(app, port=8000, host='0.0.0.0')
    else:
        test()
