#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

All code written by Li Tang August 2020 (restructured by Tannon Kew)

TODO: simplify further by potentially combining domains 

NOTE: requires python 3.8!!!

"""

import re
import sys
from typing import List
import string

# check python version (use of walrus operator requires v3.8)
if sys.version_info.minor < 8:
    sys.exit(f'Certain functions in this script require python 3.8+. You are running {sys.version_info.major}.{sys.version_info.minor}!')

GREETING = '<GREETING> '
SALUTATION = ' <SALUTATION>'


de_b1 = re.compile(
    r'^\s*(\w*)\s*?([Dd]ear|[Ll]ieber?|[Gg]rüezi|[bB]uon\w+|[Gg]uten\s[Tt]ag)\s+<?[A-Z]\w+>?\s+<?[A-Z]\w+>?\s*[,.]?')

de_b2 = re.compile(
    r'^\s*(\w*)\s*?([Dd]ear|[Ll]ieber?|[gG]eschätzter?|[hH][ae]llo|[sS]ervus|[sS]awadee|[dD]ank|[bB]uona?\s?\w+|[bB]uon\w+|[Gg]rüezi|[Gg]uten\s?([Tt]ag|[Aa]bend))\s+<?[A-Za-z]\w+>?\s*[,.]?')

de_b3 = re.compile(
    r'^\s*[sS]ehr\s+(geehrter?|[gG]eschätzter?)\s+<?[A-Z]\w+>?\s+<?[A-Z]\w+>?\s*[,.]?')
# de_rst_bos_3 = re.compile(
#     r'^\s*[sS]ehr\s+geehrter?\s+<?[A-Z]\w+>?\s+<?[A-Z]\w+>?\s*')

de_b4 = re.compile(
    r'\s*[sS]ehr\s+(geehrter?|[gG]eschätzter?)\s<?[A-Z]\w+>?\s*[,.]?')
# de_rst_bos_4 = re.compile(r'\s*[sS]ehr\s+geehrter?\s<?[A-Z]\w+>?\s*')

de_b5 = re.compile(
    r'\s*[sS]ehr\s+geehrter?\s+\w+[\s\,]+[sS]ehr\s+geehrter?\s+\w+[,.]?')

de_b6 = re.compile(r'^\s*([gG]uten\s[tT]ag|[Gg]eschätzter?\s\w+)\s[,.]?')

de_b7 = re.compile(r'^(.{3,36}?),\s*[,.]?')

de_e1 = re.compile(r'([\.\!]).{3,60}?[Gg]rü[sß]{1,2}e.{3,75}$')
# de_e1 = re.compile(r'([\.\!]).{3,60}?[Gg]rü[sß]{1,2}e.{3,75}$')

de_e2 = re.compile(r'(([\.\!])?\s*([Bb]is|[Aa]uf)\s+[Bb]ald.{3,75})$')
# de_e2 = re.compile(r'(\.\s*([Bb]is|[Aa]uf)\s+[Bb]ald.{3,75})$')

de_e3 = re.compile(r'(([\.\!])?\s*([Ii]hre?)\s+.{3,25}[Tt]eam)\s*$')
# de_e3 = re.compile(r'(\s*([Ii]hre?)\s+.{3,25}[Tt]eam)\s*$')

de_e4 = re.compile(r'([\.\!\?]).{1,8}\s*[vV]ita.{1,8}[bB]ella.{3,39}\s*$')

en_b1 = re.compile(
    r'^\s*([dD]ear)\s+([sS]ir|[mM]adam)\s+[oO]r\s+([sS]ir|[mM]adam)\s*[,.]?')

en_b2 = re.compile(
    r'^\s*(\w*)\s*?[Dd]ear\s+<?[a-zA-Z]\w+\W>?\s+<?[A-Z]\w+>?\s*[,.]?')

en_b3 = re.compile(
    r'^\s*(\w*)\s*?([Dd]ear|[bB]uon\w+|[Hh]ello)\s+\W?\s*<?[A-Za-z]\w+>?\s*\W?\s*[,.]?')

en_b4 = re.compile(
    r'^\s*[dD]ear\s+mrs\W\s+<?[A-Z]\w+>?\s+<?[A-Z]\w+>?\s*[,.]?')

en_b5 = re.compile(r'^<[A-Z]\w+>\s+[,.]?')

en_b6 = re.compile(r'\s*[sS]ehr\s+geehrter?\s<?[A-Z]\w+>?\s*[,.]?')

en_b7 = re.compile(
    r'\s*[sS]ehr\s+geehrter?\s+\w+[\s\,]+[sS]ehr\s+geehrter?\s+\w+[,.]?')

en_b8 = re.compile(r'^\s*([gG]uten\s[tT]ag|[Gg]eschätzter?\s\w+)\s?[,.]?')

en_b9 = re.compile(r'^(.{3,36}?),\s*[,.]?')

# -------
en_e1 = re.compile(r'([\.\!])?\s+[rR]egards{1,2}$')

en_e2 = re.compile(r'([\.\!]).{0,20}\s*[Ss]ee\s+you\s+soon.{9,85}\s*$')

en_e3 = re.compile(
    r'([\.\!])?\s*(with kind|with best|[Kk]ind|best)?\s+(regards|wishes).{0,75}$')

en_e4 = re.compile(
    r'(([\.\!])?\s*([Ss]incerely|yours?|<NAME>)\s*.{3,25}(manage\w+|team|<NAME>))\s*$')

en_e5 = re.compile(
    r'([\.\!\?]\s+).{1,8}\s*[vV]ita.{1,8}[bB]ella.{3,39}\s*$')

multispace = re.compile(r'\s+')

def quick_clean(s):
    """
    removes problematic backslashes, e.g. '\wyn' --> 'wyn', '\year' --> 'year' 
    these appear randomly in some go the greetings/salutations
    """
    return s.replace('\\', '')

def add_bio(toks: List[str]):
    bio = []
    needs_fixing=False
    for i, t in enumerate(toks):
        if i == 0:
            if t.strip('_S') in string.punctuation:
                # print(t)
                bio.append(t.strip('_S'))  # don't add bio label to punctuatuon
                needs_fixing=True 
            else:
                bio.append(t + '-B')
        # relevant tokens should only be labeled B `beginning` or I `inside`!
        # elif i == len(toks) - 1 and i != 0:
        #     bio.append(t + '-O')
        
        else:
            bio.append(t + '-I')

    # fix up B
    if needs_fixing and len(bio) > 1:
        bio[1] = re.sub(r'(-O|-I)', '-B', bio[1])

    return bio

def add_greeting_label(toks: str):
    toks = quick_clean(toks).split()
    toks = list(map(lambda x: x + '_G', toks))
    toks = add_bio(toks)
    # toks = re.sub(r'\s+', r'\1_GR ', toks)
    return ' '.join(toks)+' '

def add_salutation_label(toks: str):
    toks = quick_clean(toks).split()
    toks = list(map(lambda x: x + '_S', toks))
    toks = add_bio(toks)
    # toks = re.sub(r'\s+', r'\1_GR ', toks)
    return ' '.join(toks) + ' '
    

def label_salutations_de(text):
    """
    adds a BIO label/tag to matched tokens
    lang: de
    domain: hotels/restaurants

    NOTE requires python 3.8+ (uses walrus operator)
    implementation is a bit hacky - result is intended only
    to bootstrap human annotations with Prodigy    
    """

    # data = ""
    orig_text = text  # before any changes

    # greetings

    if (m := re.search(de_b1, text)):
        # print("### 2-word name or <NAME> tag")
        text = re.sub(de_b1, GREETING, text)

    elif (m := re.search(de_b2, text)):
        # print("### 1-word name or <NAME> tag")
        text = re.sub(de_b2, add_greeting_label(m.group()), text)

    elif (m := re.search(de_b3, text)):
        # print("### geehrt 1x 2w")
        text = re.sub(de_b3, add_greeting_label(m.group()), text)

    elif (m := re.search(de_b4, text)):
        # print("### geehrt 1x 1w")
        text = re.sub(de_b4, add_greeting_label(m.group()), text)

    elif (m := re.search(de_b5, text)):
        # print("### geehrt 2x")
        text = re.sub(de_b5, add_greeting_label(m.group()), text)

    elif (m := re.search(de_b6, text)):
        # print("### no name greetings")
        text = re.sub(de_b6, add_greeting_label(m.group()), text)

    elif (m := re.search(de_b7, text)):
        # print("### try other patterns")

        text = re.sub(de_b7, add_greeting_label(m.group()), text)

    # salutations

    if (m := re.search(de_e1, text)):
        # print("### greetings 1 at end")
        text = re.sub(de_e1, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(de_e2, text)):
        # print("### greetings 2 at end")
        text = re.sub(de_e2, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(de_e3, text)):
        # print("### greetings 3 at end")
        text = re.sub(de_e3, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(de_e4, text)):
        # print("### greetings 4 at end")
        text = re.sub(de_e4, r'\1 {}'.format(add_salutation_label(m.group())), text)

    return text


def label_salutations_en(text):
    """
    adds a BIO label/tag to matched tokens

    lang: en
    domain: hotels/restaurants
    
    NOTE requires python 3.8+ (uses walrus operator)
    implementation is a bit hacky - result is intended only
    to bootstrap human annotations with Prodigy    

    """

    orig_text = text  # before any changes
    # greetings at start
    if (m := re.search(en_b1, text)):
        # print("### sir / madam")
        text = re.sub(en_b1, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b2, text)):
        # print("### 2-word name or <NAME> tag")
        text = re.sub(en_b2, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b3, text)):
        # print("### 1-word name or <NAME> tag")
        text = re.sub(en_b3, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b4, text)):
        # print("### geehrt 1x 2w")
        text = re.sub(en_b4, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b5, text)):
        text = re.sub(en_b5, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b6, text)):
        # print("### geehrt 1x 1w")
        text = re.sub(en_b6, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b7, text)):
        # print("### geehrt 2x")
        text = re.sub(en_b7, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b8, text)):
        # print("### no name greetings")
        text = re.sub(en_b8, add_greeting_label(m.group()), text)

    elif (m := re.search(en_b9, text)):
        # print("### try other patterns")
        text = re.sub(en_b9, add_greeting_label(m.group()), text)

    # greetings at end
    if (m := re.search(en_e1, text)):
        # print("### greetings 1 at end")
        text = re.sub(en_e1, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(en_e2, text)):
        # print("### greetings 2 at end")
        text = re.sub(en_e2, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(en_e3, text)):
        # print("### greetings 3 at end")
        text = re.sub(en_e3, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(en_e4, text)):
        # print("### greetings 4 at end")
        text = re.sub(en_e4, r'\1 {}'.format(add_salutation_label(m.group())), text)

    elif (m := re.search(en_e5, text)):
        # print("### greetings 5 at end")
        text = re.sub(en_e5, r'\1 {}'.format(add_salutation_label(m.group())), text)

    return text



def mask_salutations_de(text):
    """
    replaces greeting/salutation span with <GREETING> or
    <SALUTATION> tokens
    lang: de
    domain: hotels/restaurants
    """

    # data = ""
    orig_text = text  # before any changes

    # greetings

    if (re.search(de_b1, text)):
        # print("### 2-word name or <NAME> tag")
        text = re.sub(de_b1, GREETING, text)

    elif (re.search(de_b2, text)):
        # print("### 1-word name or <NAME> tag")
        text = re.sub(de_b2, GREETING, text)

    elif (re.search(de_b3, text)):
        # print("### geehrt 1x 2w")
        text = re.sub(de_b3, GREETING, text)

    elif (re.search(de_b4, text)):
        # print("### geehrt 1x 1w")
        text = re.sub(de_b4, GREETING, text)

    elif (re.search(de_b5, text)):
        # print("### geehrt 2x")
        text = re.sub(de_b5, GREETING, text)

    elif (re.search(de_b6, text)):
        # print("### no name greetings")
        text = re.sub(de_b6, GREETING, text)

    else:
        # print("### try other patterns")
        text = re.sub(de_b7, GREETING, text)

    # salutations

    if (re.search(de_e1, text)):
        # print("### greetings 1 at end")
        text = re.sub(de_e1, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(de_e2, text)):
        # print("### greetings 2 at end")
        text = re.sub(de_e2, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(de_e3, text)):
        # print("### greetings 3 at end")
        text = re.sub(de_e3, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(de_e4, text)):
        # print("### greetings 4 at end")
        text = re.sub(de_e4, r'\1 {}'.format(SALUTATION), text)

    return text


def mask_salutations_en(text):
    """
    replaces greeting/salutation span with <GREETING> or
    <SALUTATION> tokens
    lang: en
    domain: hotels/restaurants
    """

    orig_text = text  # before any changes
    # greetings at start
    if (re.search(en_b1, text)):
        # print("### sir / madam")
        text = re.sub(en_b1, GREETING, text)

    elif (re.search(en_b2, text)):
        # print("### 2-word name or <NAME> tag")
        text = re.sub(en_b2, GREETING, text)

    elif (re.search(en_b3, text)):
        # print("### 1-word name or <NAME> tag")
        text = re.sub(en_b3, GREETING, text)

    elif (re.search(en_b4, text)):
        # print("### geehrt 1x 2w")
        text = re.sub(en_b4, GREETING, text)

    elif (re.search(en_b5, text)):
        text = re.sub(en_b5, GREETING, text)

    elif (re.search(en_b6, text)):
        # print("### geehrt 1x 1w")
        text = re.sub(en_b6, GREETING, text)

    elif (re.search(en_b7, text)):
        # print("### geehrt 2x")
        text = re.sub(en_b7, GREETING, text)

    elif (re.search(en_b8, text)):
        # print("### no name greetings")
        text = re.sub(en_b8, GREETING, text)

    else:
        # print("### try other patterns")
        text = re.sub(en_b9, GREETING, text)

    # greetings at end
    if (re.search(en_e1, text)):
        # print("### greetings 1 at end")
        text = re.sub(en_e1, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(en_e2, text)):
        # print("### greetings 2 at end")
        text = re.sub(en_e2, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(en_e3, text)):
        # print("### greetings 3 at end")
        text = re.sub(en_e3, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(en_e4, text)):
        # print("### greetings 4 at end")
        text = re.sub(en_e4, r'\1 {}'.format(SALUTATION), text)

    elif (re.search(en_e5, text)):
        # print("### greetings 5 at end")
        text = re.sub(en_e5, r'\1 {}'.format(SALUTATION), text)

    return text


# ----------------
# wrapper function
# ----------------

def mask_signoffs(text, lang):

    if lang == 'de':
        text = mask_salutations_de(text)

    elif lang == 'en':
        text = mask_salutations_en(text)

    else:
        pass
        # print(f'[!] Could not remove salutations for {lang}')

    text = re.sub(multispace, ' ', text)

    return text


def label_signoffs(text, lang):

    if lang == 'de':
        text = label_salutations_de(text)

    elif lang == 'en':
        text = label_salutations_en(text)

    else:
        pass
        # print(f'[!] Could not remove salutations for {lang}')

    text = re.sub(multispace, ' ', text)

    return text

# --------------
# experimental functions for producing prodigy annotation
# format
# -------------

def get_token_boundaries(text):
    """
    assuming text is a pre-tokenized text string, we split
    on whitespace and get the token boundaries.

    These are required by Prodigy in order to avoid
    additional tokenization by spacy model.
    """
    words = text.split()
    w_lens = [len(w) for w in words]

    tokens = []

    for i, (w, w_len) in enumerate(zip(words, w_lens)):
        token = {}

        token['text'] = w
        token['id'] = i

        if i == 0:
            token['start'] = 0
        else:
            token['start'] = tokens[i - 1]['end'] + 2 # previous offset +1 for whitespace +1 for first char
            # w_offset = w_onset + w_len - 1 # -1 to account for 0-indexing
        token['end'] = token['start'] + w_len - 1 # -1 to account for 0-indexing    

        tokens.append(token)
        
    return tokens

# def get_token_spans(m, text_len, label="GRT"):
#     # m = re.search
#     words = m.group().strip().split()
#     print(words)
#     w_lens = [len(w) for w in words]
#     # print(w_lens)

#     spans = []
    
#     for i, (w, w_len) in enumerate(zip(words, w_lens)):
#         if i == 0:
#             w_onset = m.span()[0]
#         else:
#             w_onset = spans[i - 1]["end"] + 2 # previous offset +1 for whitespace +1 for first char
#             # w_offset = w_onset + w_len - 1 # -1 to account for 0-indexing
#         w_offset = w_onset + w_len - 1 # -1 to account for 0-indexing    
        
#         if label == "GRT":
#             t_start = 0
#             t_end = len(words) - 1 # -1 to account for 0-indexing
#         elif label == "SLT":
#             t_end = text_len -1 # -1 to account for 0-indexing
#             t_start = t_end - len(words) # work backwards from end of string
            
#         span = {"start": w_onset,
#             "end": w_offset,
#             "label": label,
#             "token_start": t_start,
#             "token_end": t_end,
#             }
        
#         spans.append(span)
        
#     return spans

def get_match_span(m, toks, label='GRT'):
    """
    hacky method of getting span annotations for prodigy
    annotation format

    relying soley on spans provided by regex results in
    mismatches relating to the tokenization annotations (see
    func `get_token_boundaries`)
    """

    substring = m.group().strip()
    tok_count = len(substring.split())

    span = {}
    span["label"] = label
    span["start"] = m.span()[0]
    span["end"] = span["start"] + len(substring) - 1 # -1 to account for 0-indexing
    
    if label == "GRT":
        span["token_start"] = 0
        span["token_end"] = tok_count - 1 # -1 to account for 0-indexing
    elif label == "SLT":
        span["token_end"] = len(toks) - 1 # -1 to account for 0-indexing
        span["token_start"] = span["token_end"] - tok_count + 1 # work backwards from end of list +1 to account for 0-indexing
        
        # correct for setence-final punctuation
        if toks[span["token_start"]]["text"] in string.punctuation:
            # print(toks[span["token_start"]]["text"])
            span["token_start"] = span["token_start"]+1

    return span

def span_anno_salutations_de(text):
    """
    lang: de
    domain: hotels/restaurants

    NOTE requires python 3.8+ (uses walrus operator)
    implementation is a bit hacky - result is intended only
    to bootstrap human annotations with Prodigy    
    """

    tokens = get_token_boundaries(text)
    anno = {"text": text, "tokens": tokens, "spans": []}

    if (m := re.search(de_b1, text)):
        # print("### 2-word name or <NAME> tag")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b2, text)):
        # print("### 1-word name or <NAME> tag")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b3, text)):
        # print("### geehrt 1x 2w")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b4, text)):
        # print("### geehrt 1x 1w")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b5, text)):
        # print("### geehrt 2x")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b6, text)):
        # print("### no name greetings")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(de_b7, text)):
        # print("### try other patterns")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))


    # salutations

    if (m := re.search(de_e1, text)):
        # print("### greetings 1 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(de_e2, text)):
        # print("### greetings 2 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(de_e3, text)):
        # print("### greetings 3 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(de_e4, text)):
        # print("### greetings 4 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    return anno


def span_anno_salutations_en(text):
    """
    lang: de
    domain: hotels/restaurants

    NOTE requires python 3.8+ (uses walrus operator)
    implementation is a bit hacky - result is intended only
    to bootstrap human annotations with Prodigy    
    """

    tokens = get_token_boundaries(text)
    anno = {"text": text, "tokens": tokens, "spans": []}

    if (m := re.search(en_b1, text)):
        # print("### 2-word name or <NAME> tag")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b2, text)):
        # print("### 1-word name or <NAME> tag")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b3, text)):
        # print("### geehrt 1x 2w")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b4, text)):
        # print("### geehrt 1x 1w")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b5, text)):
        # print("### geehrt 2x")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b6, text)):
        # print("### no name greetings")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b7, text)):
        # print("### try other patterns")
        # anno['spans'] += get_token_spans(m, text_len, label='GRT')
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b8, text)):
        # print("### no name greetings")
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))

    elif (m := re.search(en_b9, text)):
        # print("### try other patterns")
        anno['spans'].append(get_match_span(m, tokens, label='GRT'))
    # salutations

    if (m := re.search(en_e1, text)):
        # print("### greetings 1 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(en_e2, text)):
        # print("### greetings 2 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(en_e3, text)):
        # print("### greetings 3 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    elif (m := re.search(en_e4, text)):
        # print("### greetings 4 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))
    
    elif (m := re.search(en_e5, text)):
        # print("### greetings 4 at end")
        # anno['spans'] += get_token_spans(m, text_len, label='SLT')
        anno['spans'].append(get_match_span(m, tokens, label='SLT'))

    return anno



if __name__ == "__main__":
    pass
