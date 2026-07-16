#!/usr/bin/env python3

"""Text-cleaning helpers: HTML stripping, translation-marker removal, language ID."""

import re
from typing import List, Optional

from bs4 import BeautifulSoup  # for cleaning HTML
from langdetect import detect


def reverse_tokenization(tokens: List[str], spaces: List[bool]) -> str:
    """Reconstruct a string from tokens and their trailing-whitespace flags.

    Args:
        tokens: A list of tokens.
        spaces: A list of boolean values indicating whether the corresponding
            token has a following whitespace.

    Returns:
        The detokenized string.
    """
    assert len(tokens) == len(spaces)
    detokenized = ''
    for token, space in zip(tokens, spaces):
        if space:
            detokenized += token + ' '
        else:
            detokenized += token
    return detokenized.strip()


def assign_lang(text: Optional[str]) -> Optional[str]:
    """Detect the language of ``text``, returning ``None`` on empty/undetectable input."""
    if not text:
        return None
    else:
        try:
            lang = detect(text)
            return lang
        except Exception:
            return None


def clean_translations(text: Optional[str]) -> Optional[str]:
    """Strip Google-translation markers, keeping the original-language portion."""
    if not text:
        return None
    else:
        # if text contains (Original) marker, split at marker
        # and return only the text following
        texts = re.split(r'\(Original\)', text)

        if len(texts) > 1:
            return texts[1].strip()

        else:
            # if not, text can still contain (Translated by
            # Google) marker, so remove this before returning
            # cleaned text
            text = re.sub(
                r'(\(Translated by Google\)|\(Übersetzt von Google\))', ' ', text)
            return text.strip()


def clean_html(text: Optional[str]) -> Optional[str]:
    """Strip HTML markup and normalise whitespace in ``text``."""
    if not text:
        return None
    else:
        # remove HTML markup
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        # add space between remaining ''>''
        text = re.sub('>', '> ', text)

        # normalise whitespace
        text = re.sub(r'[\s\t\n\r]+', ' ', text)

        return text.strip()
