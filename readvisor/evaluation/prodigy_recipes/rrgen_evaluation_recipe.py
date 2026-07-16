import prodigy
from prodigy.components.loaders import JSONL

# NOTE: reads src/eval_criteria.html + src/script.js relative to the CWD.
with open('src/eval_criteria.html', encoding='utf8') as f:
    eval_html = f.read()

with open('src/script.js', encoding='utf8') as f:
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
            yield from stream

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
