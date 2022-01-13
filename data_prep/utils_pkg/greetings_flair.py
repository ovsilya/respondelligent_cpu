#!/usr/bin/env python3
# -*- encoding: utf8 -*-

from typing import List, Tuple
from flair.data import Sentence
from flair.models import SequenceTagger

from .text_cleaning_utils import reverse_tokenization

def convert_bio_labeled_seq_to_masked_string(seq: List[Tuple]) -> str:
    """
    Each tuple in list is expected to contain the following
    (token, label, whitespace)
    """
    masked_seq = []
    spaces = []
    for token, label, space in seq:
        if label == 'B-GRT':
            masked_seq.append('<GREETING>')
            spaces.append(True)
        elif label == 'B-SLT':
            masked_seq.append('<SALUTATION>')
            spaces.append(True)
        elif label == 'O':
            masked_seq.append(token)
            spaces.append(space)
        else: # don't worry about it!
            pass
    
    return reverse_tokenization(masked_seq, spaces)
    
def mask_greetings_and_salutations(text: str, tagger: SequenceTagger) -> str:
    """
    text is expected to be a tokenized text string, i.e.
    tokens separated by whitespace.
    """
    sentence = Sentence(text.split()) # assumes pretokenised!
    # predict tags with model
    spaces = [True for token in sentence]

    tagger.predict(sentence)

    sequence = [(token.text, token.labels[0].value, space) for token, space in zip(sentence, spaces)]
    
    return convert_bio_labeled_seq_to_masked_string(sequence)

def mask_greetings_and_salutations_in_spacy_doc(doc, tagger: SequenceTagger) -> str:
    """
    If original text is not tokenized, pass a spacy doc in
    order to recover detokinzed form after predicting labels.
    """
    tokens = [token.text for token in doc]
    spaces = [token.whitespace_ for token in doc]
    
    sentence = Sentence(tokens)
    # predict tags with model
    tagger.predict(sentence)

    sequence = [(token.text, token.labels[0].value, space) for token, space in zip(sentence, spaces)]
    
    return convert_bio_labeled_seq_to_masked_string(sequence)