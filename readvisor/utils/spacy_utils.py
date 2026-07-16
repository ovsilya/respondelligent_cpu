"""spaCy pipeline helpers shared by the data-prep and serving code paths.

Consolidated from the two previously-duplicated ``spacy_utils`` modules (one under
``data_prep/utils_pkg`` and one under the FastAPI app). The special-token list now
lives in :mod:`readvisor.utils.special_tokens`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token

from readvisor.utils.special_tokens import SPECIAL_TOKENS

logger = logging.getLogger(__name__)

# Custom token attribute used by the response post-processing masks. Guard against
# re-registration (spaCy raises if the extension is set twice, e.g. on module reload).
if not Token.has_extension("tmp_mask"):
    Token.set_extension("tmp_mask", default="")


def load_spacy_pipe(model_path: str, sbert_model: str | None = None) -> Language:
    """Load a spaCy model (tagger/parser disabled), optionally adding SBERT vectors.

    Args:
        model_path: Path to, or name of, the spaCy model to load.
        sbert_model: Optional sentence-BERT model name. When given, sentence-BERT
            vectors are attached via ``spacy_sentence_bert`` (imported lazily so the
            data-prep path does not pull the dependency).

    Returns:
        The loaded spaCy ``Language`` pipeline.
    """
    nlp = spacy.load(model_path, disable=["tagger", "parser"])
    logger.info("Loaded spaCy model from %s", model_path)

    if sbert_model is not None:
        import spacy_sentence_bert  # lazy: only needed when SBERT vectors are requested

        logger.info("Adding sentence-BERT embeddings (%s)...", sbert_model)
        nlp = spacy_sentence_bert.create_from(nlp, sbert_model)
        logger.info("Loaded sentence-transformers model %s", sbert_model)

    return nlp


class WhitespaceTokenizer:
    """Whitespace tokenizer for pre-tokenized input (e.g. sentiment-system output)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def add_special_tokens_to_tokenizer(nlp: Language) -> None:
    """Register the project's special tokens so the tokenizer never splits them."""
    logger.info("Adding %d special tokens to the spaCy tokenizer", len(SPECIAL_TOKENS))
    for token in SPECIAL_TOKENS:
        nlp.tokenizer.add_special_case(token, [{"ORTH": token}])


def add_gazetteer_to_nlp(nlp: Language, terms: Iterable[str]) -> PhraseMatcher:
    """Build a lower-cased ``PhraseMatcher`` over a gazetteer of company names."""
    terms = list(terms)
    logger.info("Creating PhraseMatcher from gazetteer with %d terms", len(terms))
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = list(nlp.tokenizer.pipe(terms))
    matcher.add("Companies", patterns)
    return matcher
