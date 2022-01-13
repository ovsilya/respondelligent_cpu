import random
import re

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens

with open('src/eval_criteria.html', 'r', encoding='utf8') as f:
    eval_html = f.read()

with open('src/script.js', 'r', encoding='utf8') as f:
    javascript = f.read()

@prodigy.recipe('rrgen-human-eval-v2-edit')
def correct_rrgen(dataset, file_path):

    blocks = [
        {"view_id": "html", "html_template": """<p style="text-align:left;font-family:verdana;font-size:80%;"><strong>{{src}}</strong></p>"""},
        {"view_id": "html", "html_template": """<p style="text-align:left;font-family:verdana;font-size:100%;">{{text}}</p>"""},
        {"view_id": "html", "html_template": eval_html},
        {"view_id": "text_input", "field_id": "edit_text", "field_rows": 8, "field_autofocus": False},
    ]
    counts = {"edited": 0, "unchanged": 0}

    def get_stream_helper(stream):
        while True:
            for task in stream:
                yield task

    def get_stream():
        stream = JSONL(file_path)
        tasks = []
        for eg in stream:
            # print(eg.keys())
            eg["edit_text"] = eg['text']
            tasks.append(eg)            
        return tasks

    def update(answers):
        for eg in answers:
            if eg["edit_text"] != eg["text"]:
                counts["edited"] += 1
            else:
                counts["unchanged"] += 1

    def on_exit(ctrl):
        print("\nResults")
        print(counts["edited"], "edited")
        print(counts["unchanged"], "unchanged")

    stream = get_stream()
    stream = get_stream_helper(stream)

    return {
        "dataset": dataset,
        "stream": stream, # NOTE: converting from generator to list allows a progress bar for annotator
        "update": update,
        "on_exit": on_exit,
        "view_id": "blocks",
        "config": {
            "blocks": blocks,
            "javascript": javascript,
            },
        }
