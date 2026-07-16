#!/usr/bin/env python3

"""Collect sentencepiece pieces shared between corpus files and an mBART spm model.

Encodes each corpus file with the provided sentencepiece model, counts the
resulting pieces, adds the mBART language tags, and writes the vocabulary to an
output file (one piece per line, ordered by frequency).

Example call:

    python -m readvisor.model.collect_spm_pieces \\
        $data/train.review $data/train.response \\
        --spm sentencepiece.bpe.model --outfile $data/vocab.txt
"""

import argparse
import logging
from collections import Counter
from typing import List

import sentencepiece as sp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_files', nargs='+', help='list of corpus files to read and encode to find relevant sentencepiece pieces.')
    ap.add_argument('--spm', help='path to original mBART sentencepiece model')
    ap.add_argument('-o', '--outfile', help='path to output file.')
    return ap.parse_args()

def collect_pieces(infiles: List[str], spm: sp.SentencePieceProcessor):
    
    relevant_pieces = Counter()

    for infile in infiles:
        logger.info('reading pieces from file %s ...', infile)
        with open(infile, encoding='utf8') as inf:
            for line in inf:
                line = line.strip()
                pieces = spm.encode_as_pieces(line)
                relevant_pieces.update(pieces)

    logger.info('collected %s pieces', len(relevant_pieces))

    return relevant_pieces

def write_vocab_file(pieces, outfile):

    with open(outfile, 'w', encoding='utf8') as outf:
        for piece, _ in pieces.most_common():
            outf.write(f'{piece}\n')
    return


if __name__ == "__main__":
    args = set_args()

    # load spm:
    spm = sp.SentencePieceProcessor(model_file=args.spm)
    logger.info('loaded sentencepiece model from %s', args.spm)
    # collect overlapping sentencepiece tokens from corpus/spm
    relevant_pieces = collect_pieces(args.corpus_files, spm)

    relevant_pieces.update(["ar_AR","cs_CZ","de_DE","en_XX","es_XX","et_EE","fi_FI","fr_XX","gu_IN","hi_IN","it_IT","ja_XX","kk_KZ","ko_KR","lt_LT","lv_LV","my_MM","ne_NP","nl_XX","ro_RO","ru_RU","si_LK","tr_TR","vi_VN","zh_CN"])

    # write pieces to vocab list
    write_vocab_file(relevant_pieces, args.outfile)

