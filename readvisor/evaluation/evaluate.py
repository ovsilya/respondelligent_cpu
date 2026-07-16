#!/usr/bin/env python3

"""Evaluate mBART review-response generation outputs.

Two input formats are supported via ``--format``:

* ``jsonl`` -- a single JSONL translation-output file (as produced by the
  mBART inference script), one JSON object per line with ``src``/``ref``/``hyps``.
* ``text`` -- plain-text OSPL files: separate source and reference files plus
  either a single hypothesis file or a training directory containing per-epoch
  ``_val_out_checkpoint_*`` files (each scored in turn).

Example (jsonl):
    python -m readvisor.evaluation.evaluate --format jsonl translations.json \
        --domain_ref test.domain --rating_ref test.rating --source_ref test.source

Example (text):
    python -m readvisor.evaluation.evaluate --format text <training_dir_or_file> \
        --src_file valid.review --ref_file valid.response \
        --domain_ref valid.domain --rating_ref valid.rating
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from nltk.tokenize import RegexpTokenizer

from readvisor.evaluation.metrics import run_eval

logger = logging.getLogger(__name__)


def set_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--format', choices=['jsonl', 'text'], default='jsonl',
                    help='input format: "jsonl" translation-output file, or "text" OSPL files')
    ap.add_argument('hyp_path', type=str,
                    help='jsonl: path to JSONL output file; text: path to a hypothesis file '
                         'or a training dir containing _val_out_checkpoint_* files')
    # text-format inputs
    ap.add_argument('--src_file', type=str, required=False, default=None,
                    help='[text] path to original input source file, e.g. valid.review')
    ap.add_argument('--ref_file', type=str, required=False, default=None,
                    help='[text] path to original target file, e.g. valid.response')
    ap.add_argument('--sp_model', type=str, required=False, help='[text] path to spm model to use for decoding')
    # jsonl-format inputs
    ap.add_argument('--tokenize', action='store_true', required=False,
                    help='[jsonl] use if texts need to be tokenized (e.g. outputs from HuggingFace model)')
    # shared reference labels
    ap.add_argument('--domain_ref', type=str, required=False, default=None, help='path to domain ground truth labels, e.g. re_test.domain')
    ap.add_argument('--rating_ref', type=str, required=False, default=None, help='path to rating ground review rating labels, e.g. re_test.rating')
    ap.add_argument('--source_ref', type=str, required=False, default=None, help='path to source ground review rating labels, e.g. re_test.source')
    ap.add_argument('--compute_sts', action='store_true', required=False, help='use if need to compute repetition metric with sbert')
    return ap.parse_args()


def read_jsonlines(infile: str, nbest: int = 1) -> Tuple[List[str], List[str], List[str]]:
    """Read src/ref/hyp texts from a JSONL generation-output file."""
    src_texts = []
    ref_texts = []
    hyp_texts = []

    with open(infile, encoding='utf8') as f:
        for line in f:
            d = json.loads(line)
            src_texts.append(d['src'])
            ref_texts.append(d['ref'])
            hyp_texts.append(d['hyps'][nbest - 1]['hyp'])

    return src_texts, ref_texts, hyp_texts


def read_lines(infile: str) -> List[str]:
    """Read stripped lines from a plain-text (OSPL) file."""
    with open(infile, encoding='utf8') as f:
        return [line.strip() for line in f]


def inspect(i: int, srcs: List[str], refs: List[str], hyps: List[str]) -> None:
    """Log a truncated preview of the i-th (src, ref, hyp) triple."""
    logger.info('ID\t %s', i)
    logger.info('SRC:\t %s', srcs[i][:70])
    logger.info('REF:\t %s', refs[i][:70])
    logger.info('HYP:\t %s', hyps[i][:70])


def evaluate_jsonl(args) -> None:
    """Evaluate a single JSONL translation-output file."""
    srcs, refs, hyps = read_jsonlines(args.hyp_path)

    if args.tokenize:
        tokenizer = RegexpTokenizer(r'<?\w+>?|\S+')
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

    run_eval(srcs, refs, hyps, args.hyp_path, domain_refs, rating_refs, source_refs,
             compute_sts_metrics=args.compute_sts)


def evaluate_text(args) -> None:
    """Evaluate plain-text OSPL hypothesis file(s) against src/ref files."""
    if not args.src_file or not args.ref_file:
        raise SystemExit('[!] --src_file and --ref_file are required for --format text')

    srcs = read_lines(args.src_file)
    refs = read_lines(args.ref_file)

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

    hyp_path = Path(args.hyp_path)
    if hyp_path.is_dir():
        all_scores = []
        for hyp_file in sorted(hyp_path.glob('_val_out_checkpoint_*')):
            hyps = read_lines(hyp_file)
            scores = run_eval(srcs, refs, hyps, str(hyp_file), domain_refs, rating_refs,
                              source_refs, compute_sts_metrics=args.compute_sts, verbose=False)
            all_scores.append(scores)
        df = pd.concat(all_scores)
        print(df.to_csv())
    else:
        hyps = read_lines(str(hyp_path))
        run_eval(srcs, refs, hyps, str(hyp_path), domain_refs, rating_refs, source_refs,
                 compute_sts_metrics=args.compute_sts)


if __name__ == '__main__':
    args = set_args()

    if args.format == 'jsonl':
        evaluate_jsonl(args)
    else:
        evaluate_text(args)
