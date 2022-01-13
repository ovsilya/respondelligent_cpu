#!/usr/bin/env python
# coding: utf-8

"""

Example call:
python evaluate_jsonl_format.py \
    /srv/scratch6/kew/mbart/hospo_respo/respo_final/mbart_model_2021-06-04/ft/2021-06-04_15-27-39/inference/translations.json \
    --domain_ref /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/test.domain \
    --rating_ref /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/test.rating \
    --source_ref /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/test.source
"""


import argparse
from typing import List, Dict, Optional, Tuple
import json

from nltk.tokenize import RegexpTokenizer

from rrgen_evaluations import *

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('hyp_file', type=str, help='path to JSONL translation output file for evaluating')
    ap.add_argument('--domain_ref', type=str, required=False, default=None, help='path to domain ground truth labels, e.g. re_test.domain')
    ap.add_argument('--rating_ref', type=str, required=False, default=None, help='path to rating ground review rating labels, e.g. re_test.rating')
    ap.add_argument('--source_ref', type=str, required=False, default=None, help='path to source ground review rating labels, e.g. re_test.source')
    ap.add_argument('--compute_sts', action='store_true', required=False, help='use if need to compute repetition metric with sbert')
    ap.add_argument('--tokenize', action='store_true', required=False, help='use if texts need to be tokenized (e.g. outputs from HuggingFace model)')
    
    return ap.parse_args()


def read_jsonlines(infile: str, nbest: int = 1) -> Tuple[List[str], List[str], List[str]]:
    """reads in lines from JSONL generation output file"""
    
    items = []
    src_texts = []
    ref_texts = []
    hyp_texts = []

    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            d = json.loads(line)
            src_texts.append(d['src'])
            ref_texts.append(d['ref'])
            hyp_texts.append(d['hyps'][nbest-1]['hyp'])
    
    return src_texts, ref_texts, hyp_texts

def read_lines(infile: str) -> List[str]:
    items = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            items.append(line.strip())
    return items

def inspect(i: int, srcs: List[str], refs: List[str], hyps: List[str]):
    print('ID\t', i)
    print('SRC:\t', srcs[i][:70])
    print('REF:\t', refs[i][:70])
    print('HYP:\t', hyps[i][:70])
    return

if __name__ == '__main__':
    args = set_args()

    srcs, refs, hyps = read_jsonlines(args.hyp_file)

    if args.tokenize:
        tokenizer = RegexpTokenizer('<?\w+>?|\S+')

        srcs = [' '.join(tokenizer.tokenize(text)) for text in srcs]
        refs = [' '.join(tokenizer.tokenize(text)) for text in refs]
        hyps = [' '.join(tokenizer.tokenize(text)) for text in hyps]

    domain_refs, rating_refs, source_refs = None, None, None
    if args.domain_ref:
        domain_refs = read_lines(args.domain_ref)
        assert len(domain_refs) == len(hyps)

    if args.rating_ref:
        rating_refs = read_lines(args.rating_ref)
        assert len(rating_refs) == len(hyps)

    if args.source_ref:
        source_refs = read_lines(args.source_ref)
    else:
        # assume all items are respondelligent
        source_refs = ['re'] * len(hyps)
    
    inspect(10, srcs, refs, hyps)

    # breakpoint()

    run_eval(srcs, refs, hyps, args.hyp_file, domain_refs, rating_refs, source_refs, compute_sts_metrics=args.compute_sts)

