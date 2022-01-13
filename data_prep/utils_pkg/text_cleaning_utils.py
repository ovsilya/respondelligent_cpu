#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from bs4 import BeautifulSoup  # for cleaning HTML
# import langid # language identification
from langdetect import detect

def reverse_tokenization(tokens, spaces):
    """
    takes a list of tokens and a list of boolean values
    indicating whether the corresponding token has a
    floowing whitespace.
    """
    assert len(tokens) == len(spaces)
    detokenized = ''
    for token, space in zip(tokens, spaces):
        if space:
            detokenized += token + ' '
        else:
            detokenized += token
    return detokenized.strip()

def assign_lang(text):
    if not text:
        return None
    else:
        try:
            lang = detect(text)
            return lang
        except:
            return None 
        
def clean_translations(text):

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


def clean_html(text):
    """
    Does some simple text clean-up steps, e.g. normalising whitespace, removing common HTML tags
    """
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

if __name__ == "__main__":
    pass
