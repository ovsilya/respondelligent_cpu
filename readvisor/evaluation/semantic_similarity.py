#!/usr/bin/env python3

"""Semantic text similarity between source reviews and generated responses."""

import sys
from typing import List, Tuple

from nltk import tokenize  # for sentence tokenization
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Define the model
model = SentenceTransformer('distiluse-base-multilingual-cased')

def compute_max_semantic_text_similarity(src: str, hyp: str, model=model) -> float:

    src_sents = tokenize.sent_tokenize(src)
    hyp_sents = tokenize.sent_tokenize(hyp)

    src_embeddings = model.encode(src_sents, convert_to_tensor=True)
    hyp_embeddings = model.encode(hyp_sents, convert_to_tensor=True)

    # compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(src_embeddings, hyp_embeddings)

    max_values, _ = cosine_scores.max(dim=1)

    return max_values.mean().item()

def compute_sentence_similarities(src_texts: List[str], hyp_texts: List[str]) -> Tuple[float, List[float]]:
    
    scores = []
    for src, hyp in tqdm(zip(src_texts, hyp_texts)):
        scores.append(compute_max_semantic_text_similarity(src, hyp))

    scores_mean = sum(scores) / len(scores)
    return scores_mean, scores

def read_lines(file: str) -> List[str]:
    lines = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines


if __name__ == '__main__':

    review_file = sys.argv[1]
    response_file = sys.argv[2]

    reviews = read_lines(review_file)
    responses = read_lines(response_file)

    assert len(reviews) == len(responses)

    corpus_score, _ = compute_sentence_similarities(reviews, responses)

    print(corpus_score)
