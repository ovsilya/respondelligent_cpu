#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional, Set
import pandas as pd
import numpy as np

import vizseq # NOTE: requires vizseq from https://github.com/tannonk/vizseq
from vizseq.scorers.bleu import BLEUScorer
from vizseq.scorers.rouge import RougeLScorer
from vizseq.scorers.distinct_n import Distinct1Scorer, Distinct2Scorer
from vizseq.scorers.self_bleu import SelfBLEUScorer

import surface_repetition_metrics as srfc # requires numpy
import classifier_metrics as clsfr # requires fasttext

# init scorers
bleu = BLEUScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
rouge = RougeLScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
dist1 = Distinct1Scorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
dist2 = Distinct2Scorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
selfbleu = SelfBLEUScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)

def calculate_hyp_lens(hyps: List[str]) -> float:
    """
    simple hypothesis length evaluation (average)
    """
    hyp_lens = [len(hyp.split()) for hyp in hyps]
    return sum(hyp_lens) / len(hyp_lens)

def load_data_from_src_file_with_line_ids(infile: str, line_ids: Set):
    lines = []
    skipped_lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i in line_ids:
                lines.append(line.strip())
            else:
                skipped_lines.append(i)
    if skipped_lines:
        print(f'[!] lines {",".join(map(str, skipped_lines))} in {infile} are not valid line ids and will be skipped...')

    return lines

def run_eval(
    srcs: Optional[List[str]],
    refs: List[str],
    hyps: List[str],
    generation_file: str,
    domain_ref: Optional[List[str]] = None,
    rating_ref: Optional[List[str]] = None,
    source_ref: Optional[List[str]] = None,
    compute_sts_metrics: bool = False,
    verbose: bool = True) -> pd.DataFrame:

    # breakpoint()
    score_dict = {}

    score_dict['test set size'] = len(hyps)

    #################
    # classic metrics
    #################
 
    bleu_scores = bleu.score(hyps, [refs])
    rouge_scores = rouge.score(hyps, [refs])
    dist1_scores = dist1.score(hyps)
    dist2_scores = dist2.score(hyps)
    self_bleu_scores = selfbleu.score(hyps)

    score_dict['BLEU'] = bleu_scores.corpus_score / 100
    score_dict['ROUGE-L'] = rouge_scores.corpus_score
    score_dict['DIST-1'] = dist1_scores.corpus_score
    score_dict['DIST-2'] = dist2_scores.corpus_score
    score_dict['Self-BLEU'] = self_bleu_scores.corpus_score / 100

    ####################
    # repetirion metrics
    ####################
    
    srfc_rep_scores = srfc.get_scores_corpus_average(hyps)
    for k, v in srfc_rep_scores.items():
        score_dict[k] = v

    if compute_sts_metrics:
        import semantic_repetition_metrics as smntc # requires nltk, scipy, sentence_transformers (distiluse-base-multilingual-cased)
        import semantic_similarity_metric as sts

        smtc_rep_scores = smntc.calculate_paraphrase_ratio_corpus_average(hyps)
        score_dict['paraphrase reps'] = f"{smtc_rep_scores['mean']}"
    
        src_tgt_sts_score, _ = sts.compute_sentence_similarities(srcs, hyps)
        score_dict['src-tgt sts'] = src_tgt_sts_score
    else:
        score_dict['paraphrase_reps'] = None
        score_dict['src-tgt sts'] = None

    # score_dict['paraphrase reps'] = f"{smtc_rep_scores['mean']:.3f}±{smtc_rep_scores['std']:.3f}"

    ###############################
    # custom classification metrics
    ###############################

    if domain_ref is not None:
        domain_score = clsfr.estimate_domain_accuracy(refs, hyps, domain_ref)
        score_dict['domain acc'] = f"{float(domain_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            print(f"DOMAIN CLASSIFIER ACC: {domain_score['accuracy_on_refs']}")
    else:
        score_dict['domain acc'] = None

    if rating_ref is not None:
        rating_score = clsfr.estimate_rating_accuracy(refs, hyps, rating_ref)
        score_dict['rating acc'] = f"{float(rating_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            print(f"RATING CLASSIFIER ACC: {rating_score['accuracy_on_refs']}")
    else:
        score_dict['rating acc'] = None

    if source_ref is not None:
        source_score = clsfr.estimate_source_accuracy(refs, hyps, source_ref)
        score_dict['source acc'] = f"{float(source_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            print(f"SOURCE CLASSIFIER ACC: {source_score['accuracy_on_refs']}")
    else:
        score_dict['source acc'] = None

    #############################
    # hyp lens (for good measure)
    #############################
    score_dict['hyp lens'] = calculate_hyp_lens(hyps)

    summary_dict = {generation_file: score_dict}
    df = pd.DataFrame.from_dict(summary_dict, orient='index')
    # df = df.round(decimals=3)
    # print to csv for easy copying into excel score sheet
    if verbose:
        print(df.to_csv())
        
    return df

if __name__ == '__main__':
    pass
