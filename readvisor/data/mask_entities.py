#!/usr/bin/env python3

"""Mask named entities (persons, locations, URLs, emails, ...) in DataFrame texts.

Date: March 2021

Example call:

    python -m readvisor.data.mask_entities --input /mnt/storage/clwork/projects/readvisor/RESPONSE_GENERATION/intermediary/de_rrgen.pkl --spacy_model /srv/scratch2/kew/spacy_models/readvisor_in_domain_ner/de_core_news_md-2.3.0 --n_cores 52 --tokenized
"""

import argparse
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import spacy
from joblib import Parallel, delayed

from readvisor.data.text_cleaning import reverse_tokenization
from readvisor.utils.spacy_utils import (
    WhitespaceTokenizer,
    add_special_tokens_to_tokenizer,
)

logger = logging.getLogger(__name__)


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=str, help='dataframe for processing')
    ap.add_argument('--output', required=False, type=str, help='path to save output dataframe, if not given, updates input file in place')
    ap.add_argument('--spacy_model', required=True, type=str, help='shortcut or full path to spacy model, e.g. custom model such as /srv/scratch2/kew/spacy_models/readvisor_in_domain_ner/de_core_news_md...')
    ap.add_argument('--n_cores', required=False, type=int, default=12, help='number of cores to use for parallel processing')
    ap.add_argument('--tokenized', default=False, action='store_true', help='set if input text is pretokenized. This is expected after running alphasystems sentiment analysis.')
    ap.add_argument('--columns', nargs='*', required=False, help='names of df columns to apply preprocessing to, e.g. `review_clean` `response_clean`.')
    return ap.parse_args()


def mask_entity_tokens(doc, authors: Optional[List[str]] = None, detokenize: bool = False) -> str:
    """Apply entity masking to tokenised texts for review-response generation.

    Args:
        doc: spaCy doc object.
        authors: Author names associated with the text (e.g. username).
        detokenize: If True, return a detokenized string; otherwise join on spaces.

    Returns:
        Tokenized text string with masks.
    """
    if authors is None:
        authors = []

    message = []
    spaces = []

    for tok in doc:
        spaces.append(tok.whitespace_)

        if tok.ent_type_ in ['PERSON', 'PER']:
            message.append('<NAME>')

        elif tok.ent_type_ == 'GPE':
            message.append('<GPE>')

        elif tok.ent_type_ == 'LOC':
            message.append('<LOC>')

        elif tok.ent_type_ == 'ORG':
            message.append('<ORG>')

        # NOTE: removed April 26, 2021 - digits are deemed
        # to be more important for the task and less an
        # issue for privacy
        # elif tok.is_digit or tok.like_num:
        #     message.append('<DIGIT>')

        elif tok.like_url:
            message.append('<URL>')

        elif tok.like_email:
            message.append('<EMAIL>')

        elif authors and tok.lower_ in authors:  # todo: ensure author list names are lowercased
            message.append('<NAME>')

        else:
            message.append(tok.text)

    assert len(message) == len(spaces) == len(doc)

    if detokenize:
        return reverse_tokenization(message, spaces)
    else:
        return ' '.join(message)


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists):
    """Flatten a list of lists to a combined list."""
    return [item for sublist in list_of_lists for item in sublist]


def process_df(df, col):
    batch_gen = (
        (r[col], r['review_author'].split() + r['response_author'].split()) for _, r in df.iterrows())

    processed_texts = []

    for doc, auths in nlp.pipe(batch_gen, as_tuples=True, batch_size=1000):
        processed_texts.append(mask_entity_tokens(doc, auths))

    return processed_texts


def main() -> None:
    global nlp

    args = set_args()

    nlp = spacy.load(args.spacy_model, disable=["tagger", "parser"])
    logger.info('loaded spacy model with `tagger` and `parser` disabled for speed ups')

    if args.tokenized:
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    else:
        # these special tokens are used in response generation preprocessing steps
        add_special_tokens_to_tokenizer(nlp)

    df = pd.read_pickle(args.input)

    # replace NA fields with empty string
    df.review_author = df.review_author.fillna('').str.lower()
    df.response_author = df.response_author.fillna('').str.lower()

    logger.info('loaded dataframe from %s', args.input)

    # setup output file
    if not args.output:
        logger.info('[!] %s will be updated in place', args.input)
        outfile = args.input
    else:
        outfile = args.output

    assert set(args.columns) <= set(list(df.columns))

    chunksize = 1000
    for col in args.columns:
        logger.info('processing column `%s` in %s', col, args.input)

        # in parallel:
        # process texts
        result = Parallel(n_jobs=args.n_cores, backend='multiprocessing', prefer="processes")(
            delayed(process_df)(df_chunk, col) for df_chunk in chunker(df, len(df), chunksize=chunksize))
        # flatten list of lists and update column
        if col == 'response_clean':
            df['response'] = flatten(result)
        else:
            df[col] = flatten(result)

    # replace empty strings with NA
    df.review_author = df.review_author.replace(r'^\s*$', np.nan, regex=True)
    df.response_author = df.response_author.replace(r'^\s*$', np.nan, regex=True)

    df.to_pickle(outfile)
    logger.info('saved updated dataframe to %s', outfile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
