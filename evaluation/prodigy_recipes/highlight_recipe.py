import random
import re

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens

import spacy
from spacy.tokenizer import Tokenizer

# NOTE: we just use the en model here since its purpose is
# to simply tokenize on white-space. Therefore, a
# language-specific model is not necessary.
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = Tokenizer(nlp.vocab)

with open('src/eval_criteria.html', 'r', encoding='utf8') as f:
    eval_html = f.read()

with open('src/script.js', 'r', encoding='utf8') as f:
    javascript = f.read()

@prodigy.recipe('rrgen-human-eval-v1-highlight')
def multi_eval_rrgen(dataset, file_path):

    def get_stream(stream):
        while True:
            for task in stream:
                yield task

    stream = JSONL(file_path)
    # NOTE: converting from generator to list allows a progress bar for annotator
    stream = list(add_tokens(nlp, stream, skip=True))
    
    # HACK: to avoid running out of annotation task after refreshing 
    stream = get_stream(stream)
    
    blocks = [
        {"view_id": "html", "html_template": "<h5>{{src}}</h5>"},
        {"view_id": "ner_manual"}, # better formatting and allows for highlighting
        {"view_id": "html", "html_template": eval_html},
    ]

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": 'blocks',
        "config": {
            "labels": ["GOOD", "BAD"],
            "blocks": blocks, # add the blocks to the config
            "javascript": javascript,
        }
    }