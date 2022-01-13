#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import Counter
from typing import List, Dict, Optional, Tuple
import csv
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
# from sqlitedict import SqliteDict

import sentencepiece as sp

from utils_pkg import multiprocessing_utils as mp
from utils_pkg import sentiment_utils as svec


RANDOM_SEED=1247

#####################
# data prep functions
#####################


def encode_establishments_as_ids(df, lang, establ_ids=None):
    """
    Experimental encoding for establishment-specific responses
    """
    if not establ_ids:
        print('Warning: no existing id mapping provided. A new one will be created')
        s = df.establishment.value_counts()
        df['establishment_cat'] = np.where(df['establishment'].isin(s.index[s >= 30]), df['establishment'], '<unk>')
        print(len(df.establishment_cat.unique()))
        print(df.establishment_cat.value_counts())

        c = Counter()
        c.update(df['establishment_cat'].to_list())
        print(c.most_common(10))

        establ_ids = {}
        # with open(f'/srv/scratch6/kew/establ_name_ids_{lang}.txt', 'w', encoding='utf8') as outf:
        for i, (est, _) in enumerate(c.most_common()):
            # outf.write(f'{est}\t{i}\n')
            establ_ids[est] = i
            
        # print(establ_ids['<unk>'])

    df['establishment_cat'] = df['establishment_cat'].apply(lambda x: f'<est_{establ_ids[x]}>')
    print(len(df.establishment_cat.unique()))
    print(df.establishment_cat.value_counts())

    for line_num, (i, row) in tqdm(enumerate(df.iterrows())):
        if row.split_imrg_compat == 'train' and line_num % 10 == 0:    
            df.at[i,'establishment_cat'] = '<est_0>'

    print(df.establishment_cat.value_counts())

    return df

def compute_and_encode_ave_tgt_informedness(df):
    """
    experimental encoding for target informedness score
    (inspired by Filippova 2020)
    """
    df['score:avg_tgt_sent_informedness'] = df['score:tgt_sent_informedness'].progress_apply(lambda x: np.mean(x[1]))
    print(df['score:avg_tgt_sent_informedness'].describe())

    # bins = np.arange(0, 1, 0.15)
    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    bin_labels = [f'<tgtinf_{i}>' for i in range(len(bins)-1)] 

    # print(bins)
    # print(bin_labels)
    df['tgt_inf_label'] = pd.cut(df['score:avg_tgt_sent_informedness'], bins, labels=bin_labels)
    df['tgt_inf_label'] = df.tgt_inf_label.astype(str)
    df['tgt_inf_label'].value_counts()
    return df

def categorize_domain(df):
    """
    Applied for Gao (2019)-style inputs
    """
    # convert domain column to numerical values
    # e.g. [Hotel, Restuarant] --> [0, 1]
    domain_numeric_cat = pd.Categorical(df['domain'])
    df['domain_numeric'] = domain_numeric_cat.codes
    print(df['domain_numeric'].value_counts())
    print(df['domain'].value_counts())

    return df

def rescale_rating_scores(df):
    """
    Applied for Gao (2019)-style inputs
    """
    # rescale review rating from [1, 5] to [0, 1] 
    rating_values = np.array(df['rating'].unique())
    print('Unique rating values:', rating_values)
    rating_scaler = MinMaxScaler(feature_range=(0,1))
    rating_scaler.fit(rating_values.reshape(-1, 1))
    rating_scaled = rating_scaler.transform(df.loc[:,('rating')].to_numpy().reshape(-1, 1))
    df.loc[:,('rating_scaled')] = rating_scaled
    print(df['rating_scaled'].value_counts())
    print(df['rating'].value_counts())
    return df


def unify_sentiment_vectors(df):
    """
    applies reshape_vector function to ensure that both htl
    and rst vectors have same dimensions.
    """
    df['sent_vec'] = df.progress_apply(lambda x: svec.reshape_vector(x.sent_vec, x.domain), axis=1)
    return df

def convert_numerical_items_to_labels(df):
    df['domain'] = '<' + df['domain'].str.lower() + '>'
    # cast int to string in order to add < and >
    df['rating'] = df['rating'].astype("string")
    df['rating'] = '<' + df['rating'].str.lower() + '>'
    # df['response_lang'] = '<' + df['response_lang'].str.lower() + '>' # replace title boundary with more explicit special token
    return df

def lowercase_src_tgt_texts(df):
    df['review'] = df['review'].str.lower()
    df['response'] = df['response'].str.lower()
    return df

def replace_bad_sep_token(df):
    df['review'] = df['review'].str.replace('---sep---', '<endtitle>') # replace title boundary with more explicit special token
    return df

def add_prefixes_to_source_text(df, domain=False, rating=False, tgt_inf=False, est_id=False):
    # domain-specific data

    df['review_pref'] = df['review']
    df['sent_seq_pref'] = df['sent_seq']
    
    if rating:
        df['review_pref'] = df['rating'] + ' ' + df['review_pref']
        df['sent_seq_pref'] = 'RATING' + ' '+ df['sent_seq_pref']
    
    if domain:
        df['review_pref'] = df['domain'] + ' ' + df['review_pref']
        df['sent_seq_pref'] = 'DOMAIN' + ' ' + df['sent_seq_pref']
        
    if tgt_inf:
        
        df['review_pref'] = df['tgt_inf_label'] + ' ' + df['review_pref']
        df['sent_seq_pref'] = 'INFORM' + ' ' + df['sent_seq_pref']
    
    if est_id:
        df['review_pref'] = df['establishment_cat'] + ' ' + df['review_pref']
        df['sent_seq_pref'] = 'EST' + ' ' + df['sent_seq_pref']


    return df

def prepare_for_rrgen_2104_models(df, bpemb_model):
    try:
        df = encode_establishments_as_ids(df, lang)
    except:
        print('Failed to encode establishments as id labels!')
    try:
        df = compute_and_encode_ave_tgt_informedness(df)
    except:
        print('Failed to encode avg target informedness score (not available for English)!')
    df = categorize_domain(df)
    df = rescale_rating_scores(df)
    df = unify_sentiment_vectors(df)
    df = convert_numerical_items_to_labels(df)
    df = lowercase_src_tgt_texts(df)
    df = replace_bad_sep_token(df)
    df = add_prefixes_to_source_text(df, domain=True, rating=True, tgt_inf=False, est_id=False)

    spm = sp.SentencePieceProcessor(model_file=bpemb_model)
    df = apply_spm_model_on_df_texts(df, spm)

    return df

def prepare_for_mbart(df, lower=False):
    df['review'] = df['review'].str.replace('---SEP---', '<endtitle>') # replace title boundary with more explicit special token
    if lower:
        df['review'] = df['review'].str.lower()
        df['response'] = df['response'].str.lower()

    df = convert_numerical_items_to_labels(df)

    mbart_lang_tags = {
        'de': 'de_DE',
        'en': 'en_XX',
        '<de>': 'de_DE',
        '<en>': 'en_XX',
    }    

    df['review'] = df.progress_apply(lambda row: row['review'] + ' </s> ' + mbart_lang_tags[row['lang'].lower()], axis=1)
    df['response'] = df.progress_apply(lambda row: row['response'] + ' </s> ' + mbart_lang_tags[row['lang'].lower()], axis=1)

    return df
    

def flatten(lst):
    return [item for sublist in lst for item in sublist]
    
def apply_sentencepiece_on_labeled_token_sequence(tok_seq: str, label_seq: str, spm) -> Tuple[str]:    
    
    tok_seq = tok_seq.split()
    label_seq = label_seq.split()
    
    assert len(tok_seq) == len(label_seq), f"[!] Input token sequence has different length to sentiment sequence\n{tok_seq}\n{label_seq}"

    sp_tok_seq = [spm.encode_as_pieces(tok) for tok in tok_seq]
    sp_label_seq = []
    
    for label, sp_tok in zip(label_seq, sp_tok_seq):
        sp_label_seq.append([label] * len(sp_tok))

    sp_tok_seq = flatten(sp_tok_seq)
    sp_label_seq = flatten(sp_label_seq)    

    assert len(sp_tok_seq) == len(sp_label_seq), f"[!] Output token sequence has different length to sentiment sequence ({len(sp_tok_seq)}:{len(sp_label_seq)})"
    
    return ' '.join(sp_tok_seq), ' '.join(sp_label_seq)

def apply_sentencepiece(line: str, spm):
    return ' '.join(spm.encode_as_pieces(line))

def apply_spm_model_on_df_texts(df, spm):

    df['response'] = df.progress_apply(
        lambda x: apply_sentencepiece(x['response'], spm), axis=1)

    df[['review', 'sent_seq']] = df.progress_apply(
        lambda x: apply_sentencepiece_on_labeled_token_sequence(
            x['review'], x['sent_seq'], spm), axis=1, result_type="expand")

    df[['review_pref', 'sent_seq_pref']] = df.progress_apply(
        lambda x: apply_sentencepiece_on_labeled_token_sequence(
            x['review_pref'], x['sent_seq_pref'], spm), axis=1, result_type="expand")
    
    return df

def subset_by_domain(df, domain):
    print(dom)
    if domain == 'rst':
        df = df[df['domain'] == '<restaurant>']
    elif domain == 'htl':
        df = df[df['domain'] == '<hotel>']
    print(len(df))
    return df

################
# write function
################

def generate_fairseq_input_files(df,
                                 outdir: str,
                                 col_name_outfile_mapping: Dict,
                                 split_col: str,
                                 n: int = 0):
    """
    Generates multiple individual files (one per column).
    For each split (train/test/valid) lines in each output file must correspond with each other!
    """
    for split in df[split_col].unique():  
        
        split_df = df[df[split_col] == split]
        
        # shuffle train set - mainly required after upsampling!
        if split == 'train':
            split_df = split_df.sample(frac=1, random_state=RANDOM_SEED)
        
        if n: # just take a head of dataframe
            if split == 'train':
                split_df = split_df.head(n)
            else:
                split_df = split_df.head(int(n*0.1))

        print(f'{split} split has length: {len(split_df)}')

        
        for k, v in col_name_outfile_mapping.items():
            if k == 'src_len_cates':
                write_length_file(split_df[k], outdir / f'{split}.{v}')
            elif 'sent_vec' in k:
                write_np_arrays_file(split_df[k], outdir / f'{split}.{v}')
            else:
                write_file(split_df[k], outdir / f'{split}.{v}')
        
    print('Done!')
    return


######################
# inspection functions
######################

def check_mem(df):
    print(f'DF memory usage {df.memory_usage(deep=True).sum()/1e+6:.2f}MB')

def get_df_len(df):
    print(f'DF has {len(df)} entries')
    return

def get_detailed_info_on_df(df, split_col):
    print(f'DF has {len(df)} entries')
    print('DF COLS:', df.columns)
    print(df.groupby('source')[split_col].value_counts())
    print(df[split_col].value_counts()) 
    df.head()
    return

def token_count(string):
    tokens = string.split()
    return len(tokens)

def check_for_split_overlap(df, column_a, column_b, split_col):
    """
    Use this function to ensure no overlap between train / test / dev splits.
    
    Duplicates can appear after removing greetings/salutations and apply bpe
    """
    print('CHECKING FOR DUPLICATES IN COLS:', column_a, column_b)
    
    train_src = df[df[split_col] == 'train'][column_a].to_list()
    train_tgt = df[df[split_col] == 'train'][column_b].to_list()
    
    test_src = df[df[split_col] == 'test'][column_a].to_list()
    test_tgt = df[df[split_col] == 'test'][column_b].to_list()
    
    valid_src = df[df[split_col] == 'valid'][column_a].to_list()
    valid_tgt = df[df[split_col] == 'valid'][column_b].to_list()
    
    train = set(zip(train_src, train_tgt))
    test = set(zip(test_src, test_tgt))
    valid = set(zip(valid_src, valid_tgt))
    print('TRAIN', len(train))
    print('TEST', len(test))
    print('VALID', len(valid))
    print('-----------------')
    tt = train.intersection(test)
    tv = train.intersection(valid)
    testv = test.intersection(valid)
    if (len(tt) != 0) or (len(tv) != 0) or (len(testv) != 0):
        print('WARNING: FOUND OVERLAP IN')
        print('\tTRAIN / TEST:', len(tt))
        print('\tTRAIN / VALID:', len(tv))
        print('\tTEST / VALID:', len(testv))
        
        return False
#         # if overlap is found, drop duplicates!
#         print('DROPPING DUPLICATES FROM DF WITH LENGTH:', len(df))
#         df = df.drop_duplicates(subset=[column_a, column_b], keep='last')
#         print('DEDUPED DF LENGTH:', len(df))
    else:
        print('NO OVERLAP FOUND!')
        return True
    # return df

def ensure_no_empty_strings(df, column_a, column_b):
    print('REMOVING EMPTY STRING VALUES FROM DF WITH LENGTH:', len(df))
    df = df[(df[column_a] != '') & (df[column_b] != '')]
    print('REMOVED ITEMS DF LENGTH:', len(df))
    return df
    
def write_file(series, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for line in series.to_list():
            f.write(f'{line}\n')
    return

def write_length_file(series, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for x in series.to_list():
            f.write(f'{x:.2f}\n')
    return

def write_np_arrays_file(series, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for x in series.to_list():
            f.write(f'{" ".join(map(str, x))}\n')
    return

def get_column_stats(df, col):
    
    uniq_vals = df[col].unique()
    print()
    print(f'Column `{col}` has {len(uniq_vals)} unique values: e.g.:', list(uniq_vals[:10]))

    return