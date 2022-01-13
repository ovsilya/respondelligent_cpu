#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from typing import List, Set, Tuple, Dict
from tqdm import tqdm

from nltk import tokenize # for sentence tokenization
from sentence_transformers import SentenceTransformer, util

# Define the model
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('distiluse-base-multilingual-cased')

def compute_max_semantic_text_similarity(src: str, hyp: str, model=model) -> float:

    src_sents = tokenize.sent_tokenize(src)
    hyp_sents = tokenize.sent_tokenize(hyp)

    # encode corpus with multiple processes (run on Vigrid (52) or Idavoll (30))
    
    src_embeddings = model.encode(src_sents, convert_to_tensor=True)
    hyp_embeddings = model.encode(hyp_sents, convert_to_tensor=True)
    
    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(src_embeddings, hyp_embeddings)

    max_values, max_indices = cosine_scores.max(dim=1)

    return max_values.mean().item()

def compute_sentence_similarities(src_texts: List[str], hyp_texts: List[str]) -> Tuple[float, List[float]]:
    
    scores = []
    for src, hyp in tqdm(zip(src_texts, hyp_texts)):
        scores.append(compute_max_semantic_text_similarity(src, hyp))

    scores_mean = sum(scores) / len(scores)
    return scores_mean, scores

def read_lines(file: str) -> List[str]:
    lines = []
    with open(file, 'r', encoding='utf8') as f:
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

    # reviews = [
    #     """Fondue Dinner ---SEP--- Yummy fondue , good sides
    #     , and lively place ! Bepary our server took great
    #     care of us and served us quickly as we got started !
    #     Definitely will come back when in <GPE>""",
    #     """Good hotel in a great location ---SEP--- We
    #     stayed at this hotel in July <DIGIT> . I will list
    #     the pros and cons belowPROS : great location in Old
    #     Town <GPE> . We LOVED the location . All cobblestone
    #     streets , tons of restaurants and shops all in the
    #     area . It is <DIGIT> miles from the train station
    #     and we had <DIGIT> pieces of luggage between the
    #     <DIGIT> of us and it was a simple easy trip . The
    #     cabs were going to charge $ <DIGIT> ! crazy . Easy
    #     walk . We did it at <DIGIT> pm and felt totally safe
    #     . Check in was quick and easy . The free breakfast
    #     was great . It had meats , cheeses , cereals ,
    #     breads .. all kinds of stuff . Plenty of variety to
    #     get your day going . In room there was a water
    #     kettle for coffee . CONS : NO Air Conditioning ! wow
    #     , it was HOT in the room . They give you a stand up
    #     fan , which we had to prop on a chair to get air but
    #     it was just so hot . We were on the second floor
    #     overlooking the courtyard so could not open the
    #     window as EVERYONE below would have a full view
    #     inside our room to us . I would not recommend
    #     staying in a room facing the courtyard . The room
    #     really can not hold more than <DIGIT> people . We
    #     could only open the suitcase on the bed , so a
    #     little small but that is normal for <LOC> .""",
    #     """( + ) Close to the railway station and staff were
    #     very willing to help with room issues . Clean ( - )
    #     The room was stifling hot with <DIGIT> small fan
    #     that did nothing . This hotel is not equipped to
    #     deal with the heat with no air conditioning . The
    #     bathroom was tiny with minimal amenities ,
    #     disappointing considering the price ."""
    # ]
        
        
    # responses = [
    #     """<GREETING> Thanks for the <DIGIT> - star review .
    #     Swiss Chuchi is the place to come for an authentic
    #     Swiss fondue experience . We started as the first "
    #     fondue parlour " in <GPE> back in the 1950s , and
    #     have become an iconic venue , visited by locals and
    #     visitors looking to experience an authentic taste of
    #     Switzerland . Bepary was very pleased to read your
    #     comments , as were all the team . We all take pride
    #     in our friendly , professional service , and when a
    #     guest comments favourably on this , it means a lot
    #     to us . We hope you will be back again soon .
    #     <SALUTATION>""",
    #     """<GREETING> Thank you for staying at Hotel Adler
    #     Zurich and taking the time to write such a detailed
    #     review on TripAdvisor . We appreciate it very much
    #     when our guests share their experience with us and
    #     other travelers . It 's great to hear you could
    #     benefit from our central location in <GPE> 's car -
    #     free old town . Indeed , it 's very easy to walk
    #     from our hotel to the main train station , besides ,
    #     you can take a tram which stops right in front of
    #     our hotel . We are glad you enjoyed our breakfast
    #     buffet that we prepare fresh every day and serve at
    #     Restaurant <DIGIT> .""",
    #     """<GREETING> Thank you for staying with us and
    #     taking the time to write a review . First of all ,
    #     we are glad you could benefit from our central
    #     location and were content with our staff 's service
    #     . We are sorry for the inconvenience due to the warm
    #     temperature in the room . We do our best to provide
    #     our guests with a fan to cool the room , but we are
    #     sorry you did n't find the room equipped with the
    #     air conditioners . If you need anything , please do
    #     n't hesitate to contact our reception staff who is
    #     available <DIGIT> . We hope to be able to welcome
    #     you back in the future . <SALUTATION>"""      
    # ]

    

