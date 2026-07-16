#!/usr/bin/env python3

"""Simple lexical reranking for nbest list hypotheses."""

from typing import List


def lexical_overlap_rerank(src_text: str, nbest_hyps: List[str]) -> str:
    """Return the hypothesis with the highest token overlap with the source.

    Args:
        src_text: source text (whitespace-tokenised).
        nbest_hyps: candidate hypotheses to rerank.

    Returns:
        The hypothesis sharing the most unique tokens with the source.
    """
    src_tokens = src_text.split()

    best_overlap = 0
    best_hyp_idx = 0

    for i, hyp in enumerate(nbest_hyps):
        hyp_tokens = hyp.split()
        overlap_ratio = len(set(src_tokens).intersection(set(hyp_tokens))) / len(set(src_tokens))
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_hyp_idx = i

    return nbest_hyps[best_hyp_idx]
