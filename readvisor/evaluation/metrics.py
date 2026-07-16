#!/usr/bin/env python3

"""Core metric hub for ReAdvisor review-response evaluation.

Bundles the classic n-gram metrics (BLEU, ROUGE-L, distinct-n, self-BLEU)
together with the project's custom repetition, semantic-similarity and
attribute-classifier metrics into a single :func:`run_eval` entry point.
"""

import logging
from typing import List, Optional

import pandas as pd
import vizseq  # noqa: F401  # NOTE: requires vizseq from https://github.com/tannonk/vizseq
from vizseq.scorers.bleu import BLEUScorer
from vizseq.scorers.distinct_n import Distinct1Scorer, Distinct2Scorer
from vizseq.scorers.rouge import RougeLScorer
from vizseq.scorers.self_bleu import SelfBLEUScorer

from readvisor.evaluation import classifier_metrics as classifiers
from readvisor.evaluation import lexical_repetition as lexical_rep

logger = logging.getLogger(__name__)

# init scorers
bleu = BLEUScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
rouge = RougeLScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
dist1 = Distinct1Scorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
dist2 = Distinct2Scorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)
selfbleu = SelfBLEUScorer(corpus_level=True, sent_level=True, n_workers=4, verbose=False, extra_args=None)


def calculate_hyp_lens(hyps: List[str]) -> float:
    """Return the average hypothesis length (in whitespace tokens)."""
    hyp_lens = [len(hyp.split()) for hyp in hyps]
    return sum(hyp_lens) / len(hyp_lens)


def run_eval(
    srcs: Optional[List[str]],
    refs: List[str],
    hyps: List[str],
    run_id: str,
    domain_ref: Optional[List[str]] = None,
    rating_ref: Optional[List[str]] = None,
    source_ref: Optional[List[str]] = None,
    compute_sts_metrics: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute the full suite of evaluation metrics for a set of hypotheses.

    Args:
        srcs: source review texts (only required if ``compute_sts_metrics``).
        refs: reference (ground-truth) response texts.
        hyps: system-generated hypothesis texts.
        run_id: identifier for this run, used as the DataFrame row index.
        domain_ref: optional ground-truth domain labels for domain accuracy.
        rating_ref: optional ground-truth rating labels for rating accuracy.
        source_ref: optional ground-truth source labels for source accuracy.
        compute_sts_metrics: if True, also compute paraphrase-repetition and
            semantic-similarity metrics (loads extra sentence-transformer models).
        verbose: if True, log classifier accuracies and print the CSV summary.

    Returns:
        A single-row :class:`pandas.DataFrame` indexed by ``run_id`` with one
        column per metric.
    """
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
    # repetition metrics
    ####################

    srfc_rep_scores = lexical_rep.get_scores_corpus_average(hyps)
    for k, v in srfc_rep_scores.items():
        score_dict[k] = v

    if compute_sts_metrics:
        # imported lazily: these load heavy sentence-transformer models
        from readvisor.evaluation import paraphrase_repetition as paraphrase_rep
        from readvisor.evaluation import semantic_similarity as semantic_sim

        smtc_rep_scores = paraphrase_rep.calculate_paraphrase_ratio_corpus_average(hyps)
        score_dict['paraphrase reps'] = f"{smtc_rep_scores['mean']}"

        src_tgt_sts_score, _ = semantic_sim.compute_sentence_similarities(srcs, hyps)
        score_dict['src-tgt sts'] = src_tgt_sts_score
    else:
        score_dict['paraphrase_reps'] = None
        score_dict['src-tgt sts'] = None

    ###############################
    # custom classification metrics
    ###############################

    if domain_ref is not None:
        domain_score = classifiers.estimate_domain_accuracy(refs, hyps, domain_ref)
        score_dict['domain acc'] = f"{float(domain_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            logger.info("DOMAIN CLASSIFIER ACC: %s", domain_score['accuracy_on_refs'])
    else:
        score_dict['domain acc'] = None

    if rating_ref is not None:
        rating_score = classifiers.estimate_rating_accuracy(refs, hyps, rating_ref)
        score_dict['rating acc'] = f"{float(rating_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            logger.info("RATING CLASSIFIER ACC: %s", rating_score['accuracy_on_refs'])
    else:
        score_dict['rating acc'] = None

    if source_ref is not None:
        source_score = classifiers.estimate_source_accuracy(refs, hyps, source_ref)
        score_dict['source acc'] = f"{float(source_score['accuracy_on_hyps'].split()[0])}"
        if verbose:
            logger.info("SOURCE CLASSIFIER ACC: %s", source_score['accuracy_on_refs'])
    else:
        score_dict['source acc'] = None

    #############################
    # hyp lens (for good measure)
    #############################
    score_dict['hyp lens'] = calculate_hyp_lens(hyps)

    summary_dict = {run_id: score_dict}
    df = pd.DataFrame.from_dict(summary_dict, orient='index')
    # print to csv for easy copying into excel score sheet
    if verbose:
        print(df.to_csv())

    return df


if __name__ == '__main__':
    pass
