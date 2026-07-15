"""Single source of truth for the project's custom special tokens.

These tokens are (a) registered with the spaCy tokenizers so they are never split
apart, and (b) used as masks and structural markers throughout preprocessing,
generation, and post-processing. Previously this list was duplicated across
``spacy_utils`` (twice) and the serving post-processing code; keep it here only.
"""

# Structural markers.
SEP = "---SEP---"
END_TITLE = "<endtitle>"
GREETING = "<GREETING>"
SALUTATION = "<SALUTATION>"

# Entity masks.
URL = "<URL>"
DIGIT = "<DIGIT>"
EMAIL = "<EMAIL>"
NAME = "<NAME>"
LOC = "<LOC>"
GPE = "<GPE>"

# BERT-style markers retained for backward compatibility with older data.
BERT_SEP = "[sep]"
BERT_CLS = "[cls]"

#: All special tokens registered with the spaCy tokenizer. Order is preserved from
#: the original implementation for reproducibility.
SPECIAL_TOKENS = [
    SEP,
    END_TITLE,
    URL,
    DIGIT,
    EMAIL,
    NAME,
    LOC,
    GPE,
    BERT_SEP,
    BERT_CLS,
    GREETING,
    SALUTATION,
]
