import random
import re

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens

with open('src/eval_criteria.html', 'r', encoding='utf8') as f:
    eval_html = f.read()

with open('src/script.js', 'r', encoding='utf8') as f:
    javascript = f.read()

@prodigy.recipe('rrgen-human-eval-v3')
def correct_rrgen(dataset, file_path):

    blocks = [
        {"view_id": "html", "html_template": """<p style="text-align:left;font-family:verdana;font-size:80%;"><strong>{{src}}</strong></p>"""},
        {"view_id": "html", "html_template": """<p style="text-align:left;font-family:verdana;font-size:100%;">{{text}}</p>"""},
        {"view_id": "html", "html_template": eval_html},
    ]
    
    def get_stream():
        while True:
            stream = JSONL(file_path)
            for eg in stream:
                yield eg

    # stream = get_stream()
    stream = list(get_stream()) # NOTE: converting from generator to list allows a progress bar for annotator

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {
            "blocks": blocks,
            "javascript": javascript,
            },
        }
