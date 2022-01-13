#!/usr/bin/env python
# coding: utf-8

"""

Example call:

    python evaluate_mbart_validation_outputs.py \
            /srv/scratch6/kew/mbart/hospo_respo/respo_final/mbart_model_2021-06-04/ft/2021-06-04_15-27-39/_val_out_checkpoint_0 \
            --src_file /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/valid.review \
            --ref_file /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/valid.response \
            --domain_ref /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/valid.domain \
            --rating_ref /srv/scratch6/kew/mbart/hospo_respo/respo_final/data/valid.rating

"""


import argparse
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

from rrgen_evaluations import *

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('hyp_files', type=str, help='path to longmbart training dir (containing files _val_out_checkpoint_*')
    # ap.add_argument('--ref_dir', type=str, required=False, help='path to original input source file e.g. re_test.review(.sp)')
    ap.add_argument('--src_file', type=str, required=True, help='path to original input source file e.g. re_test.review(.sp)')
    ap.add_argument('--ref_file', type=str, required=True, help='path to original target file e.g. re_test.response(.sp)')
    ap.add_argument('--sp_model', type=str, required=False, help='path to spm model to use for decoding')
    ap.add_argument('--domain_ref', type=str, required=False, default=None, help='path to domain ground truth labels, e.g. re_test.domain')
    ap.add_argument('--rating_ref', type=str, required=False, default=None, help='path to rating ground review rating labels, e.g. re_test.rating')
    ap.add_argument('--source_ref', type=str, required=False, default=None, help='path to source ground review rating labels, e.g. re_test.source')
    ap.add_argument('--compute_sts', action='store_true', required=False, help='use if need to compute repetition metric with sbert')
    return ap.parse_args()

def read_lines(infile: str) -> List[str]:
    """expects OSPL format file without sentencepiece encoding"""
    with open(infile, 'r', encoding='utf8') as f:
        return [line.strip() for line in f]
    
if __name__ == '__main__':
    args = set_args()

    # if args.src_file:
    srcs = read_lines(args.src_file)
    refs = read_lines(args.ref_file)

    # print(len(srcs))

    domain_refs, rating_refs, source_refs = None, None, None

    if args.domain_ref:
        domain_refs = read_lines(args.domain_ref)
        assert len(domain_refs) == len(srcs)
    if args.rating_ref:
        rating_refs = read_lines(args.rating_ref)
        assert len(rating_refs) == len(srcs)
    if args.source_ref:
        source_refs = read_lines(args.source_ref)
        assert len(source_refs) == len(srcs)

    if Path(args.hyp_files).is_dir():
        all_scores = []
        for i in range(20):
            hyp_file = Path(args.hyp_files) / str('_val_out_checkpoint_'+str(i))
            hyps = read_lines(hyp_file)   
            scores = run_eval(srcs, refs, hyps, hyp_file, domain_refs, rating_refs, source_refs, compute_sts_metrics=args.compute_sts, verbose=False)
            all_scores.append(scores)
        df = pd.concat(all_scores)
        print(df.to_csv())
            
    else:
        hyp_file = args.hyp_files
        hyps = read_lines(hyp_file)   
        scores = run_eval(srcs, refs, hyps, hyp_file, domain_refs, rating_refs, source_refs, compute_sts_metrics=args.compute_sts)



