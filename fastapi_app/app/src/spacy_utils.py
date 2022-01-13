#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import logging

from memory_profiler import profile

import spacy
import spacy_sentence_bert
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler

spacy.tokens.token.Token.set_extension('tmp_mask', default='')

logging.basicConfig(format='[INFO] %(asctime)s - %(message)s', level=logging.INFO)

@profile
def load_spacy_pipe(model_path, sbert_model=None):
    nlp_model = model_path
    nlp = spacy.load(nlp_model, disable=["tagger", "parser"])
    logging.info("loaded spacy model")
    
    if sbert_model is not None:
        # load one of the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
        # nlp = spacy_sentence_bert.load_model(cfg.models.sentence_transformers.xlm)
        #this adds in the sentence-bert vectors
        # xx_paraphrase_xlm_r_multilingual_v1
        # sbert_model = cfg.models.sentence_transformers.dist
        logging.info("adding the sentence embeddings...")
        nlp = spacy_sentence_bert.create_from(nlp, sbert_model)
        logging.info(f"loaded sentence-transformers model from {sbert_model}")

    return nlp

class WhitespaceTokenizer:
    """simple whitespace tokenizer to be used when
    input is pretokenised (e.g. output of sentiment
    analysis system)"""
    
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

def add_special_tokens_to_tokenizer(nlp):
    logging.info("adding special tokens to spacy model")
    nlp.tokenizer.add_special_case("---SEP---", [{"ORTH": "---SEP---"}])
    nlp.tokenizer.add_special_case("<endtitle>", [{"ORTH": "<endtitle>"}])
    nlp.tokenizer.add_special_case("<URL>", [{"ORTH": "<URL>"}])
    nlp.tokenizer.add_special_case("<DIGIT>", [{"ORTH": "<DIGIT>"}])
    nlp.tokenizer.add_special_case("<EMAIL>", [{"ORTH": "<EMAIL>"}])
    nlp.tokenizer.add_special_case("<NAME>", [{"ORTH": "<NAME>"}])
    nlp.tokenizer.add_special_case("<LOC>", [{"ORTH": "<LOC>"}])
    nlp.tokenizer.add_special_case("<GPE>", [{"ORTH": "<GPE>"}])
    nlp.tokenizer.add_special_case("[sep]", [{"ORTH": "[sep]"}])
    nlp.tokenizer.add_special_case("[cls]", [{"ORTH": "[cls]"}])
    nlp.tokenizer.add_special_case("<GREETING>", [{"ORTH": "<GREETING>"}])
    nlp.tokenizer.add_special_case("<SALUTATION>", [{"ORTH": "<SALUTATION>"}])
    

def add_gazetteer_to_nlp(nlp, terms):
    logging.info("creating `phrasematcher` from gazetteer with {} terms".format(len(terms)))
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Only run nlp.make_doc to speed things up
    # patterns = [nlp.make_doc(text) for text in terms]
    patterns = list(nlp.tokenizer.pipe(terms))
    matcher.add("Companies", patterns)
    
    return matcher

def add_entity_ruler(nlp, terms):
    logging.info("creating `entity_ruler` from gazetteer with {} terms".format(len(terms)))
    ruler = nlp.create_pipe("entity_ruler")
    nlp.add_pipe(ruler, before="ner")
    # ruler = nlp.add_pipe("entity_ruler", before="ner")
    # ruler = nlp.add_pipe(nlp.create_pipe('entity_ruler'))
    patterns = []
    for term in terms:
        term_tokens = re.split(r'[\s-]', term)
        pattern_list = [{"LOWER": token} for token in term_tokens]
        
        term_pattern = {"label": "ORG", "pattern": pattern_list, "id": term}
        patterns.append(term_pattern)
        print(term_pattern)
    
    ruler.add_patterns(patterns)
    return nlp

if __name__ == "__main__":
    pass