#!/usr/bin/env python3
# coding: utf-8

import sys
import math
import re
import spacy

# load spacy model

en = '/srv/scratch2/kew/spacy_models/en_core_web_sm-2.3.1/en_core_web_sm/en_core_web_sm-2.3.1'
de = '/srv/scratch2/kew/spacy_models/de_core_news_sm-2.3.0/de_core_news_sm/de_core_news_sm-2.3.0'

try:
    EN_NLP = spacy.load(en)
    DE_NLP = spacy.load(de)
except Exception as err:
    print('Failed to load spaCy models! Check that model exists at specified path.')
    print(err)
    sys.exit()

EN_NLP.tokenizer.add_special_case("---SEP---", [{"ORTH": "---SEP---"}])
EN_NLP.tokenizer.add_special_case("<endtitle>", [{"ORTH": "<endtitle>"}])
EN_NLP.tokenizer.add_special_case("<URL>", [{"ORTH": "<URL>"}])
EN_NLP.tokenizer.add_special_case("<DIGIT>", [{"ORTH": "<DIGIT>"}])
EN_NLP.tokenizer.add_special_case("<EMAIL>", [{"ORTH": "<EMAIL>"}])
EN_NLP.tokenizer.add_special_case("<NAME>", [{"ORTH": "<NAME>"}])
EN_NLP.tokenizer.add_special_case("<LOC>", [{"ORTH": "<LOC>"}])

DE_NLP.tokenizer.add_special_case("---SEP---", [{"ORTH": "---SEP---"}])
DE_NLP.tokenizer.add_special_case("<endtitle>", [{"ORTH": "<endtitle>"}])
DE_NLP.tokenizer.add_special_case("<URL>", [{"ORTH": "<URL>"}])
DE_NLP.tokenizer.add_special_case("<DIGIT>", [{"ORTH": "<DIGIT>"}])
DE_NLP.tokenizer.add_special_case("<EMAIL>", [{"ORTH": "<EMAIL>"}])
DE_NLP.tokenizer.add_special_case("<NAME>", [{"ORTH": "<NAME>"}])
DE_NLP.tokenizer.add_special_case("<LOC>", [{"ORTH": "<LOC>"}])

bad_emojis = ['�', '🇦', '🇧', '🇨', '🇬', '🇭', '🇮', '🇲', '🇳', '🇸', '🇹']

NAMEPLACEHOLDER = 'NAMEPLACEHOLDER'

##############################################
# Functions for response gen dataset cleaning
##############################################


def assign_grpid_value(grpid, domain):
    """
    TripAdvisor data has blank fields for group domain.
    This function updates these fields to generic group ID values.
    0 is reserved for restaurants
    1 is reserved for hotels
    """
    
    try:  # if grpid is already an integer, keep existing
        return int(grpid)
    except:  # ValueError or TypeError
        if domain.lower() == 'restaurant' :
            return 0
        elif domain.lower() == 'hotel':
            return 1
        else:
            print('Not a valid domain value!')
            sys.exit()
    
def rescale_value(orig_val, orig_min=0, orig_max=1, new_min=1, new_max=10):
    """
    This function rescales a value between an old range to a new range.

    Vader assigns sentiment values in the range [0,1], 
    whereas Sentistrength uses a [1,5] scale.

    https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    """

    # check to see if we're dealing with a negative number (for compound score)
    if orig_val < 0:
        neg_val = True
    else:
        neg_val = False

    orig_range = (orig_max - orig_min)

    if orig_range == 0:
        output_val = new_min
    else:
        new_range = (new_max - new_min)
        output_val = (((abs(orig_val) - orig_min) *
                       new_range) / orig_range) + new_min

    # round up or down depending on whether value is +'ve or -'ve
    if neg_val:
        return -math.ceil(output_val)
    else:
        return math.ceil(output_val)


def average_sentence_sentiments(scores):
    """
    Parses sentiment scores assigned by Vader Sentiment analysis tool

    scores: list of dictionary objects containing per sentence sentiment scores assigned by VADER
    [{'pos': float, 'neut': float, 'neg': float, 'compound': float}, {...}, {...}]
    """

    if len(scores) == 0:
        return 0, 0

    else:
        # pos and neg values are within [0,1]
        pos = sum([i['pos'] for i in scores]) / len(scores)
        neg = sum([i['neg'] for i in scores]) / len(scores)

        # comp is within [-1,1]
        comp = sum([i['compound'] for i in scores]) / len(scores)

    return pos, neg, comp


def vader_score_text(text, lang='en', only_comp=False):
    """
    Calculates review-level sentiment with Vader

    Averages Vader's compound scores for each sentence
    """

    sents = get_sentences(text, lang)

    if lang == 'de':
        v_scores = [de_sentiment.polarity_scores(sent) for sent in sents]
    else:
        v_scores = [en_sentiment.polarity_scores(sent) for sent in sents]
    # for each sentence, assign sentiment score
    # v_scores = [analyser.polarity_scores(str(sent))
    #            for sent in nlp(text).sents]

    # average pos/neg/comp sentiment scores for review document
    v_pos, v_neg, v_comp = average_sentence_sentiments(v_scores)

    # rescale values
    v_pos, v_neg, v_comp = map(rescale_value, [v_pos, v_neg, v_comp])

    # ensure negative value is < 0!
    if v_neg > 0:
        v_neg = -v_neg

    if only_comp:
        return v_comp
    else:
        return '{} {} {}'.format(v_pos, v_neg, v_comp)


def simple_tokenize(text, lang='en'):
    """
    Simple function to get a tokenized version of a text.
    """
    if lang == 'de':
        return " ".join([token.text for token in DE_NLP(text)])
    else:
        return " ".join([token.text for token in EN_NLP(text)])


def get_sentences(text, lang='en'):
    """
    Simple function to get the sentences of a text.

    Returns:
        list of sentence strings
    """
    if lang == 'de':
        return [str(sent) for sent in DE_NLP(text).sents]
    else:
        return [str(sent) for sent in EN_NLP(text).sents]

def tokenise(text, lang='en'):
    """
    Simple function to tokenised sentences from text.

    i.e. [['sentence', 'one'], ['sentence', 'two']]

    Returns:
        list of sentence strings
    """
    if lang == 'de':
        sents = []
        for sent in DE_NLP(text).sents:
            sents.append([token.text for token in sent])
    else:
        sents = []
        for sent in EN_NLP(text).sents:
            sents.append([token.text for token in sent])
    return sents

def preprocess_text(text, lang='en', authors=[]):
    """  
    Applies preprocessing to English raw texts for the purpose of review response generation experiments

    Args:
        text (str): original text string, e.g. review, response, review title
        authors (list): author names associated with text (could be username, person name or restuarant name)

    Returns:
        preprocessed text string
    """

    NAMEPLACEHOLDER = 'NAMEPLACEHOLDER'

    # cover user names
    if isinstance(authors, list) and len(authors) > 0:
        for name in authors:

            if isinstance(name, str) and not name in ['', '_']:
                # print(name)
                try:  # re.error: bad character range w-C at position 13

                    # search through entire text for exact matches
                    text = re.sub(r'\b{}\b'.format(name),
                                  NAMEPLACEHOLDER, text, flags=re.IGNORECASE)
                    text = re.sub(r'\b{}\b'.format(name.replace(
                        '_', ' ')), NAMEPLACEHOLDER, text, flags=re.IGNORECASE)

                except re.error:
                    pass

    if lang == 'de':
        text = DE_NLP(text)
    else:
        text = EN_NLP(text)

    message = []

    for tok in text:
        if tok.text in bad_emojis:
            pass

        elif tok.is_digit or tok.like_num:
            message.append('<DIGIT>')

        elif tok.like_url:
            message.append('<URL>')

        elif tok.like_email:
            message.append('<EMAIL>')

        elif tok.ent_type_ == 'PERSON':
            # replace first token of name with placeholder
            # and ignore others
            if tok.ent_iob == 3:
                message.append('<NAME>')
            else:
                pass

        elif tok.ent_type_ == 'GPE':
            # replace first token of name with placeholder
            # and ignore others
            if tok.ent_iob == 3:
                message.append('<LOC>')
            else:
                pass

        elif tok.text == NAMEPLACEHOLDER:
            message.append('<NAME>')

        elif tok.text == '---SEP---':
            message.append(tok.text)

        else:

            # apply capitalisation to for German Nouns
            if lang == 'de':
                if tok.pos_ in ['NOUN', 'PROPN']:
                    token = tok.lower_.title().strip()
                # NB. we DON'T bother with formal pronouns,
                # e.g. Sie, Ihnen since they're ambig.
                # elif tok.tag_ == 'PPER' and tok.lower_ in ['Sie':
                    # token = tok.lower_.title().strip()
                else:
                    token = tok.lower_.strip()
            else:
                token = tok.lower_.strip()

            if token:
                message.append(token)

#     message = mask_signoffs(' '.join(message), lang)        
    
    return ' '.join(message)


def preprocess_knowledge_text(text, lang):
    """  
    Applies preprocessing to knowledge grounding raw texts

    Args:
        text (str): original text string

    Returns:
        preprocessed text string
    """
    
    if lang == 'de':
        text = DE_NLP(text)
    elif lang == 'en':
        text = EN_NLP(text)

    message = []

    for tok in text:
        # apply capitalisation to for German Nouns
        if lang == 'de':
            if tok.pos_ in ['NOUN', 'PROPN']:
                token = tok.lower_.title().strip()
            # NB. we DON'T bother with formal pronouns,
            # e.g. Sie, Ihnen since they're ambig.
            elif tok.tag_ == 'PPER':
                token = tok.text.strip()
            else:
                token = tok.lower_.strip()
        else:
            token = tok.lower_.strip()

        if token:
            message.append(token)

#     message = mask_signoffs(' '.join(message), lang)        
    
    return ' '.join(message)

if __name__ == "__main__":
    pass
