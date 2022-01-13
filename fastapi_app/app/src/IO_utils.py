#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import random
import numpy as np
from types import SimpleNamespace # used to convert dict to namedTuple
from typing import List, Dict, Set, Tuple, Optional
import logging
from spacy.tokens import Doc as spacy_doc
from spacy.matcher import PhraseMatcher as spacy_phrasematcher
from spacy.language import Language as spacy_language
logging.basicConfig(format='[INFO] %(asctime)s - %(message)s', level=logging.INFO)


#####################
# simple file loaders
#####################

def load_config(config_file_path: str) -> SimpleNamespace:
    """
    Reads in JSON config file and returns a SimpleNamespace
    object for clean access to file paths.
    """      
    with open(config_file_path, 'r', encoding='utf8') as cfg_file:
    # Parse JSON into an object with attributes corresponding to dict keys.
        cfg = json.loads(cfg_file.read(), object_hook=lambda d: SimpleNamespace(**d))
    logging.info("loaded config file from {}".format(config_file_path))
    return cfg

def load_company_data_from_json(json_data: str) -> Set[str]:
    """
    extracts human-readable company names from json data
    (sf-guard-group)
    """
    with open(json_data, 'r', encoding='utf8') as f:
        group_data = json.loads(f.read())        
    companies = set([entry.get('company') for entry in group_data if entry.get('company') is not None])
    logging.info("loaded company data from JSON file {}".format(json_data))
    return companies

def load_company_data_from_text(text_file: str) -> Set[str]:
    """
    reads in company names from TripAdvsior +
    Respondelligent datasets
    # all_company_names = '/srv/scratch2/kew/company_names_de_en.txt'
    """
    with open(text_file, 'r', encoding='utf8') as f:
        companies = set(f.read().splitlines())
    logging.info("loaded company data from text file {}".format(text_file))
    return companies

def load_company_label_mapping(tsv_file: str) -> Dict:
    """
    reads in comany names and id labels used for training
    company-aware generation model (DER)
    
    Expected format of tsv file:
        col 1: company name from db dump
        col 2: occurence freq in training data
        col 3: human-readble label
        col 4: machine-readable label
    
    e.g.:
        swiss-chuchi-zuerich    1585    <swiss-chuchi-zuerich>  <est_1>
        leoneck-swiss-hotel-zurich  852 <leoneck-swiss-hotel-zurich>    <est_2>
        sternen-grill-zuerich   690 <sternen-grill-zuerich> <est_3>
        
    """
    company_label_mapping = {}
    with open(tsv_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            company_label_mapping[line[0]] = line[-1]
    logging.info("loaded company-label mapping data from tsv file {}".format(tsv_file))
    return company_label_mapping

def add_company_label_to_input_data(item: Dict, company_label_mapping: Dict) -> Dict:
    """
    Extends item meta data by adding a unique company label
    tag that the model has seen during training.
    NOTE: this only applicable for mBART_DER models, which are trained with a
    company-sepecific label.
    """

    company_name = item["meta"].get("name", None)
    if company_name not in company_label_mapping:
        item["meta"]["company_label"] = '<est_0>'
    else:
        item["meta"]["company_label"] = company_label_mapping[company_name]
    return item

###################
# Output Processing
###################

def mask_entity_tokens(
    doc: spacy_doc, 
    authors: List = [], 
    matcher: Optional[spacy_phrasematcher] = None
    ) -> spacy_doc:
    """
    Processes a given hypothesis text (spacy Doc object)
    checking for named entities.
    
    Any token belonging to a named entity is replaced with a
    mask token that is later replaced using straight-forward
    regex find/replace.
    """
    
    # run first pass using quick PhraseMatcher
    # NOTE: Items in PhraseMatcher are stored with the same
    # tokenization scheme as the doc being processed, so
    # tokenization mismatches should be avoided. See https://spacy.io/usage/rule-based-matching#phrasematcher 
    
    if matcher:
        matches = matcher(doc)
        for _, span_start, span_end in matches:
            matched_tokens = doc[span_start:span_end]
            for tok in matched_tokens:
                tok._.tmp_mask = "<ORG>"
        
    for tok in doc:
        
        if tok._.tmp_mask != '': # skip if already filled by PhraseMatcher
            pass

        elif tok.ent_type_ in ['PERSON', 'PER']:
            tok._.tmp_mask = '<NAME>'

        elif tok.ent_type_ == 'GPE':
            tok._.tmp_mask = '<GPE>'

        elif tok.ent_type_ == 'LOC':
            tok._.tmp_mask = '<LOC>'

        elif tok.ent_type_ == 'ORG':
            tok._.tmp_mask = '<ORG>'

        elif tok.like_url:
            tok._.tmp_mask = '<URL>'

        elif tok.like_email:
            tok._.tmp_mask = '<EMAIL>'

        elif authors and tok.lower_ in authors: # todo: ensure author list names are lowercased
            tok._.tmp_mask = '<AUTH>'

    return doc
    
def apply_masks(doc: spacy_doc) -> str:
    """
    Annotates tokens in a spacy Doc object with a custom
    attribute to make it easier to identify and fill relevant named
    entities placeholders.

    NOTE: This also ensures that original whitespaces are not lost.
    """
    text = '' # build up text as string, keeping whitespace as is
    for tok in doc:
        if not tok._.tmp_mask:
            text += tok.text_with_ws
        else:
            text += tok._.tmp_mask
            if tok.whitespace_:
                text += ' ' 

    return text

def parse_greetingtext(greetingtype: str, lang:str) -> str:
    """
    Selects an appropriate greeting from list of allowable
    greetings based on value of `greetingtype` in customer data table `SF-guard-group`.

    Possible values for `greetingtype` = {'formally', 'informally'}
    Possible values for `lang` = {'en', 'de'}

    """
    
    # NOTE: for more elaborate/customizable entries, save
    # these dictionaries as JSON file that are loaded with
    # the config.json file. Due to lack of data/information
    # in the database provided, we simply use the following
    # base examples. These could be further customized by
    # using the company name information available in meta,
    # to get distinct greetings for each customer. Ideally
    # though, it'd better to structure the database properly
    # and avoid free text entries that lead to ambiguity.
    
    # assume formal is not specified!
    if greetingtype == '':
        greetingtype = 'formally'

    formal_greetings = {
        'de': [
            'Guten Tag NAMEPLACEHOLDER,'
            ],
        'en': [
            'Dear NAMEPLACEHOLDER,',
        ]
    }

    informal_greetings = {
        'de': [
            'Hallo NAMEPLACEHOLDER,'
            ],
        'en': [
            'Hi NAMEPLACEHOLDER,',
        ]
    }

    if greetingtype == 'informally':
        return random.choice(informal_greetings[lang])
    elif greetingtype == 'formally':
        return random.choice(formal_greetings[lang])
    else:
        logging.warning(f'Could not get greeing for greetingtype value {greetingtype}')
        return ''

def parse_goodbyetext(string: str, lang: str) -> str:
    """
    Attempts to parse unstructured `goodbyttext` column
    from customer data table `SF-guard-group` to fill
    salutation mask in generated response.

    Note, entries are poorly structured so parsing is not
    perfect.

    Currently designed to successfully extract
    language-specific greetings from following types:
        
        DE: Liebe Grüsse Ihr Gifthüttli Team EN: Kind regards, Your Gifthüttli Team
        Herzliche Grüsse (DE) / Kind regards (EN), Team Restaurant Gartäbeiz Eymatt62
        Herzliche Grüsse Berggasthaus Mostelberg
        
    Args:
        string: value in `goodbyttext` column
        lang: language code from input meta data
    """

    if not string:
        return ''

    # attempt to parse: 
        # DE: Liebe Grüsse Ihr Gifthüttli Team EN: Kind regards, Your Gifthüttli Team
    pref_format = re.search(r'DE:(.*)? EN:(.*)', string)
    if pref_format:
        if lang == 'de':
            return pref_format.group(1).strip()
        elif lang == 'en':
            return pref_format.group(2).strip()
        else:
            logging.warning(f'Could not parse salutation for language {lang}')
            return ''

    # attempt to parse: 
        # Herzliche Grüsse (DE) / Kind regards (EN), Team Restaurant Gartäbeiz Eymatt62   
    affix_format = re.search(r'(\(DE\)|\(EN\))\s\/', string)
    if affix_format:
        salutations, auth = string.split(',')
        sal_list = salutations.split('/')
        for s in sal_list:
            if lang == 'de':
                m = re.search(r'(.*)?\(DE\)', s)
                if m:
                    # group(1) to match sting up to (DE)
                    return m.group(1).strip() + ', ' + auth.strip()
            elif lang == 'en':
                m = re.search(r'(.*)?\(EN\)', s)
                if m:
                    # group(1) to match sting up to (EN)
                    return m.group(1).strip() + ', ' + auth.strip()
        else:
            logging.warning(f'Could not parse salutation for language {lang}')
            return ''

    else:
        # assumes single option string, e.g. 'Herzliche Grüsse Berggasthaus Mostelberg'
        return string

def fill_masks(text: str, meta: Dict) -> str:
    """
    Attempts to fill known mask tokens in a model output
    hypothesis with relevant fields from the meta data
    associated with an input review.

    NOTE: text should be a regular whitespace tokenised string
    """
    # replace <org> tokens, these often appear in repeated sequences
    if meta.get('company') is not None:
        # text = re.sub(name, meta['company'], text)
        text = re.sub(r'(?:\s<ORG>)+', ' '+meta['company'], text)
    else:
        logging.warning('`company` attribute in metadata is N/A')

    if meta.get('gpe') is not None:
        text = re.sub(r'(?:\s<LOC>)+', ' '+meta['gpe'], text)
        text = re.sub(r'(?:\s<GPE>)+', ' '+meta['gpe'], text)
    else:
        logging.warning('`gpe` attribute in metadata is N/A')
        
    if meta.get('url') is not None:
        text = re.sub(r'(?:<URL>)+', meta['url'], text)
    else:
        logging.warning('`url` attribute in metadata is N/A')

    if meta.get('email') is not None:
        text = re.sub(r'(?:<EMAIL>)+', meta['email'], text)
    else:
        logging.warning('`email` attribute in metadata is N/A')

    if meta.get('greetings') is not None:
        if isinstance(meta['greetings'], list):
            # if meta['greetings'] is a list, we assume
            # that is contains allowable customized
            # greetings and select one at random. DB
            # structure needs to be improved to get full
            # customization here, e.g. for pizzeria XX, should be 'bonjourno'
            greeting_selected = random.choice(meta['greetings'])
        elif isinstance(meta['greetings'], str):
            # given the current db structure, the value of
            # meta['greetings'] is assumed to simply be the value
            # from column `greetingtype` in table
            # `SF-guard-group` (either {formally,
            # informally})
            greeting_selected = parse_greetingtext(meta['greetings'], meta['lang'])

        # add linebreak after greeting for visualisation
        text = re.sub(r'(?:<GREETING>)+', greeting_selected+'\n', text)
    
    else:
        logging.warning('`greetings` attribute in metadata is N/A')

    if meta.get('salutations') is not None:
        if isinstance(meta['salutations'], list):
            # if meta['salutations'] is a list, we assume
            # that is contains allowable customized
            # salutations and select one at random
            salutation_selected = random.choice(meta['salutations'])
        elif isinstance(meta['salutations'], str):
            # given the current db structure, the value of
            # meta['salutations'] is assumed to simply be the value
            # from column `goodbyetext` in table
            # `SF-guard-group` NOTE: these are unstructured
            # and often empty!
            salutation_selected = parse_goodbyetext(meta['salutations'], meta['lang'])
        
        # add linebreak before salutation for visualisation
        text = re.sub(r'(?:<SALUTATION>)+', '\n'+salutation_selected, text)
        
    else:
        logging.warning('`salutations` attribute in metadata is N/A')

    # fill NAMEPLACEHOLDER inserted with greetings from database
    text = re.sub(r'NAMEPLACEHOLDER', meta.get('author', '****'), text)

    return text

def apply_reordering(
    hyps: List, 
    model_scores: List, 
    overlap_scores: List, 
    sts_scores: List) -> List[str]:
    """
    Reranks an n-best list of hypotheses based on the
    output model scores and heuristic overlap/sts scores.
    """
    # apply softmax to model scores to get probabilities
    model_probs = np.exp(model_scores) / np.sum(np.exp(model_scores), axis=0)
    summed_scores = model_probs + np.array(sts_scores) + np.array(overlap_scores)
    idx = np.argsort(summed_scores)[::-1] # sort in descending order
    reordered_hyps = [hyps[i] for i in idx]
    return reordered_hyps

def add_greeting_salutation_masks(text: str) -> str:
    """
    Manually insert greeting and salutation mask tokens for
    replacement if mssing from a hypothesis text.
    
    NOTE: mBART models remove special tokens from output
    during detokenization, so we need to add these to the
    hypothesis in order to match and replace with
    personalised greetings and salutations from the DB.
    """

    contains_greeting_mask = re.match('<GREETING>', text)

    contains_salutation_mask = re.search('<SALUTATION>', text)

    if not contains_greeting_mask:
        text = '<GREETING> ' + text

    if not contains_salutation_mask:
        text = text + ' <SALUTATION>'
    
    return text

def postprocess_object(
    response_object: Dict, 
    nlp: spacy_language,
    matcher: Optional[spacy_phrasematcher]=None) -> Dict:
    """
    Outermost postprocessing function.
    """
    src_doc = nlp(response_object['review'])
    src_tokens = set([tok for tok in src_doc if not tok.is_stop and not tok.is_punct])

    model_scores, hyp_texts = zip(*response_object['responses']) 
    
    overlap_scores = []
    sts_scores = []
    processed_hyps = []
    for doc in nlp.pipe(map(add_greeting_salutation_masks, hyp_texts)):
        doc = mask_entity_tokens(doc, [response_object['meta']['author']], matcher)
        overlap_scores.append(compute_word_overlap(src_tokens, doc))
        # if 'sentencebert_add_model_to_doc' in nlp.pipe_names:
        sts_scores.append(src_doc.similarity(doc))
        text_str = apply_masks(doc)
        text_str = fill_masks(text_str, response_object['meta'])
        # text = self.try_to_replace_named_ents(doc, response_object['meta'])
        processed_hyps.append(text_str)

    response_object['responses'] = apply_reordering(processed_hyps, model_scores, overlap_scores, sts_scores)
    return response_object

def compute_word_overlap(src_tokens: str, hyp_doc: str) -> float:
    """
    Simple word overlap metric to compute similarity between
    a given src text and model hypothesis.
    """      
    hyp_tokens = set([tok for tok in hyp_doc if not tok.is_stop or not tok.is_punct])
    if len(src_tokens):
        return len(src_tokens.intersection(hyp_tokens)) / len(src_tokens)
    else:
        return 0.0

if __name__ == '__main__':
    pass