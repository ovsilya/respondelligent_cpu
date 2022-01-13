#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict
import numpy as np
# from tqdm import tqdm
from scipy import stats
from nltk import tokenize # for sentence tokenization
from sentence_transformers import SentenceTransformer, util
# note, model will be downloaded if not already installed;
# check in ~/.cache/torch/sentence_transformers/

# model = SentenceTransformer('distiluse-base-multilingual-cased')
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
# NOTE: default model location = ~/.cache/torch/sentence_transformers

def calculate_paraphrase_ratio(text: str, sim_threshold: float = 0.75, model: SentenceTransformer = model, tokenizer=tokenize) -> float:
    """
    calculates ratio of repeated sentence (based on semantic paraphrases) to number of sentences
    
    :params:
        text: multi-sentence generated text
        sim_threshold: hyperparameter determines what
        classifies as a `paraphrase`. inspecting results
        showed that paraphrase sentences have a similariy of
        > 0.90, but we set it lower to be conservative.
    """
    sentences = tokenize.sent_tokenize(text)

    # this is unlikely, but just incase
    if len(sentences) < 2:
        return 0.0

    # paraphrase only works on texts of more than two
    # sentences (not sure why...)
    elif len(sentences) == 2:
        sim_score = util.pytorch_cos_sim(model.encode(sentences[0]), model.encode(sentences[1]))
        if sim_score > sim_threshold:
            return 1.0 # if sent a and sent b are paraphrases of each other, ratio = 1.0
        else:
            return 0.0 # otherwise ratio = 0

    else:
        # calculate paraphrases: Given a list of sentences/texts, compare all sentences against all
        # other sentences and return a list with the pairs that have the highest cosine similarity score
        paraphrases = util.paraphrase_mining(model, sentences, show_progress_bar=False)
        # paraphrases returns a list of lists, e.g. [[0.9280351400375366, 0, 1], [0.9266923666000366, 2, 1]]
    
        # calculate repetition ratio
        labels = [0] * len(sentences)
        for score, i, j in paraphrases:
            if score > sim_threshold: # threshold hyperparam
                labels[i] = labels[j] = 1

        return sum(labels)/len(labels)

def calculate_paraphrase_ratio_corpus_average(texts: List[str]) -> Dict:
    ratios = np.array([calculate_paraphrase_ratio(text) for text in texts])
    
    return {
        'mean': ratios.mean(),
        'std': ratios.std(),
        'variance': ratios.var(),
        'min': min(ratios),
        'max': max(ratios),
        'median': np.median(ratios),
        'mode': stats.mode(ratios)
        }

if __name__ == '__main__':
    # test
    ex1 = """<greeting> thank you for your positive
        feedback . it 's great to read you enjoyed our pizza .
        like in a true <name> , we use the original <name> flour
        for pizza dough and fior di latte instead of normal
        mozzarella . this ensures an authentic taste appreciated
        by many of our guests . we 'd love to welcome you back
        to <name> for lunch or dinner . perhaps , next time you
        'd like to try <digit> of our homemade pasta dishes .
        <salutation>"""

    ex2 = """<greeting> thank you for the <digit> - star review .
        we are happy to read you enjoyed your meal at santa
        lucia niederdorf in the heart of <loc> 's old town . we
        are pleased you enjoyed your meal at santa lucia
        niederdorf in the heart of <loc> 's old town . we look
        forward to welcoming you back to our restaurant in the
        heart of <loc> . <salutation>"""

    ex3 = """<greeting> thank you for the <digit> - star review .
        we are glad you enjoyed your stay at our hotel . we are
        happy to hear that you enjoyed your stay at our hotel .
        we are open every day and serve warm meals throughout
        the day . we look forward to seeing you again when you
        are next in <loc> . <salutation>"""

    ex4 = "thank you . thank you very much . thanks so much !"
    
    ex5 = "thank you . thank you very much ."

    ex6 = "thank you for taking the time to write a review . we hope to see you again soon ."

    print('ex 1:', calculate_paraphrase_ratio(ex1))
    print('ex 2:', calculate_paraphrase_ratio(ex2))
    print('ex 3:', calculate_paraphrase_ratio(ex3))
    print('ex 4:', calculate_paraphrase_ratio(ex4))
    print('ex 5:', calculate_paraphrase_ratio(ex5))
    print('ex 6:', calculate_paraphrase_ratio(ex6))
    print(compute_corpus_average([ex1, ex2, ex3, ex4, ex5, ex6]))
