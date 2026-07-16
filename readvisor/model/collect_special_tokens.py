#!/usr/bin/env python3

"""Collect special tokens (``<...>`` markers) from corpus files into a vocab list.

Reads the given corpus files, extracts every whitespace token wrapped in angle
brackets, and writes the sorted set to an output file (one token per line).

Example call:

    python -m readvisor.model.collect_special_tokens \\
        $data/train.review $data/train.response $data/train.rating \\
        $data/train.domain $data/train.est_label --outfile $data/vocab.txt
"""

import argparse
import logging
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_files', nargs='+', help='list of corpus files to read and encode to find relevant sentencepiece pieces.')
    ap.add_argument('-o', '--outfile', help='path to output file.')
    return ap.parse_args()

def collect_special_tokens(infiles: List[str]):
    
    relevant_tokens = Counter()

    for infile in infiles:
        logger.info('reading pieces from file %s ...', infile)
        with open(infile, encoding='utf8') as inf:
            for line in inf:
                line = line.strip()
                tokens = [token for token in line.split() if token.startswith('<') and token.endswith('>')]
                relevant_tokens.update(tokens)

    logger.info('collected %s tokens', len(relevant_tokens))

    return relevant_tokens

def write_vocab_file(tokens, outfile):

    with open(outfile, 'w', encoding='utf8') as outf:
        for token in sorted(tokens.keys()):
            logger.info('special token: %s', token)
            outf.write(f'{token}\n')
    return


if __name__ == "__main__":
    args = set_args()

    special_tokens = collect_special_tokens(args.corpus_files)
    
    # write pieces to vocab list
    write_vocab_file(special_tokens, args.outfile)

