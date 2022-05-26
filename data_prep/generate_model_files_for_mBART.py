#!/usr/bin/env python
# coding: utf-8

# This script takes a pickled pandas DF and applies preprocessing transformations to prepare input files for training an mBART model.
# 
# ---
# UPDATED 03.06.2021

# In[1]:


from typing import List, Dict, Optional, Tuple
import csv
from collections import Counter
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import MinMaxScaler

from utils_pkg import multiprocessing_utils as mp
from utils_pkg import sentiment_utils as svec
from utils_pkg.greetings_flair import mask_greetings_and_salutations_in_spacy_doc, mask_greetings_and_salutations
from utils_pkg.mask_entities_in_df_texts import mask_entity_tokens
from utils_pkg.spacy_utils import add_special_tokens_to_tokenizer

pd.options.display.max_columns = 999

import spacy

from flair.models import SequenceTagger
import sentencepiece as sp


# ## Setup path variables
# 
# **NOTE** these should be changed to match your operating system

# In[5]:


respo_data = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart/respo_data_2022.pkl'
establ_labels = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart/est_labels_2022.txt'
outdir = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart'
flair_model = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/ml_grt_slt_flair_multi_fast/best-model.pt'
en_spacy_model = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/spacy/readvisor_in_domain_ner/en_core_web_md-2.3.1'
de_spacy_model = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/spacy/readvisor_in_domain_ner/de_core_news_md-2.3.0'


# ## Setup processing variables
# 
# Here we set processing options (e.g. whether or not to mask greetings (recommended!))

# In[6]:


# processing variables
RANDOM_SEED = 1247
lang = 'ml' # ml = multilingual (for mBART)
split_col = 'split_imrg_compat' # use this instead of old `split`!!!
n_cores = 32 # number of cores for parallel processing
do_mask_greetings = True # take approx. an hour to process ~20K items in df
apply_lowercase = False


# In[7]:


##################
# helper functions
##################

def assign_splits(df):
    """
    expects a shuffled dataframe so naïvely populated a new split column based on position
    top 10% = test
    mid 10% = valid
    end 80% = train
    """
    total = len(df)
    test = ['test'] * (int(total * .05))
    valid = ['valid'] * (int(total * .05))
    train = ['train'] * (total-(len(test)+len(valid)))
    split_labels = test + valid + train 
    assert len(df) == len(split_labels)
    df['split'] = split_labels
    return df
    

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

def ensure_no_split_overlap(df, column_a, column_b, split_col):
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
    else:
        print('NO OVERLAP FOUND!')
    return df

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

def mask_greetings_and_salutations_in_raw_string_EN(text):
    doc = en_nlp(text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    return mask_greetings_and_salutations_in_spacy_doc(doc, tagger)

def mask_greetings_and_salutations_in_raw_string_DE(text):
    doc = de_nlp(text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    return mask_greetings_and_salutations_in_spacy_doc(doc, tagger)


# # Data prep for mBART model inputs 
#  
# Below we apply the following steps:
#    - mask greetings and salutations
#    - add brackets to rating
#    - add brackets to domain
#    - replace the `---SEP---` token used to demarkate title/text boundaries with a more explicit and consistent label e.g. `<endtitle>`
#    - add establishment labels
#    - write train/test/valid split files required as input to single column tsv files

# In[8]:


# load in models for processing grettings and salutations
print('loading tagger model...')
tagger = SequenceTagger.load(flair_model) 
en_nlp = spacy.load(en_spacy_model)
de_nlp = spacy.load(de_spacy_model)


# In[9]:


# read in data
df = pd.read_pickle(respo_data)
print(len(df))
print(df.columns)

df['source'].value_counts()


# In[10]:


# select only respondelligent sources!
# NOTE: source=platform are not always written by respondelligent and introduce noise so leave them behind
df = df[df['source'] == 're']
print(f'valid respondelligent responses: {len(df)}')
# ensure no empty values
df = df[df['review_clean'] != '']
print(f'valid respondelligent responses: {len(df)}')
df = df[df['response_clean'] != '']    
print(f'valid respondelligent responses: {len(df)}')

# subset data by lang
df_en = df[df['lang'] == 'en']
print(f'valid English responses: {len(df_en)}')
df_de = df[df['lang'] == 'de']
print(f'valid German responses: {len(df_de)}')


# In[11]:


print('Review-response pair distribution for German:')
print(df[df.lang == 'de'].domain.value_counts())
print()
print('Review-response pair distribution for English:')
print(df[df.lang == 'en'].domain.value_counts())


# In[ ]:


# apply greeting masks
# NOTE: this takes approx 20 mins to do 10K examples, so go make a coffee...
if do_mask_greetings:
    en_responses = df_en['response_clean'].tolist()
    en_responses = mp.parallelise(mask_greetings_and_salutations_in_raw_string_EN, en_responses, n_cores)
    assert len(en_responses) == len(df_en)
    df_en['response_clean'] = en_responses

    de_responses = df_de['response_clean'].tolist()
    de_responses = mp.parallelise(mask_greetings_and_salutations_in_raw_string_DE, de_responses, n_cores)
    assert len(de_responses) == len(df_de)
    df_de['response_clean'] = de_responses

df = pd.concat([df_en, df_de])
# shuffle dataset
df = df.sample(frac=1, random_state=RANDOM_SEED)


# In[1]:


# inspect results
df['response_clean']


# In[14]:


# when processing data from re:spondelligent DBs,
# split information based on IDs is not available, 
# so here we simply create new splits. 
# NOTE: for better reproducibility, between data versions, 
# a dedicated test set should be developed based on reviewids in re:spondelligent's DB
if not split_col in df.columns:
    df = assign_splits(df)
    split_col = 'split'

# inspect DF
get_detailed_info_on_df(df, split_col)
print()
ensure_no_split_overlap(df, 'review_clean', 'response_clean', split_col)
print()
get_column_stats(df, 'rating')
get_column_stats(df, 'domain')
get_column_stats(df, 'source')
get_column_stats(df, 'establishment')


# In[15]:


# map all negative rating values to 1
print(df['rating'].value_counts())
df.loc[df['rating'] < 1, 'rating'] = 1
print(df['rating'].value_counts())


# In[16]:


# Here, we load the predefined mapping between establishments and their labels for the model.
# NOTE: to generate the labels for a particular model, 
# see collect_establishment_occurence_freq_counts_from_respondelligent_data.py,
# which labels establishments according to collected frequency counts
# and produces the establ_labels tsv file

estabs = {}
if establ_labels:
    with open(establ_labels, 'r', encoding='utf8') as inf:
        for line in inf:
            line = line.rstrip().split('\t')
            estabs[line[0]] = (int(line[1]), line[2], line[3])

def map_establishments_to_labels_based_on_freq_counts(name, estabs=estabs, threshold=10):
    """
    Fetches appropriate establishment label for a given restaurant/hotel name.
    If the occurence frequency of the restauarant/hotel in the training data is lower
    than the specifies threshold, we return a catch-all placeholder label. This ensure that
    the model can generalise to infrequent/new customers.
    """
    freq, hum_label, cat_label = estabs.get(name, (0, '<unk_est>', '<est_0>'))
    if freq >= threshold:
        return cat_label
    else:
        return '<est_0>'

df['establishment_cat'] = df['establishment'].apply(lambda x: map_establishments_to_labels_based_on_freq_counts(x))
df.establishment_cat.value_counts()


# In[17]:


# duplicate src and ttgt texts (as  a backup in case a mistake was made - save re-doing mask greetings!)
df['review'] = df['review_clean']
df['response'] = df['response_clean']

# convert raw categorical values to 'special token' labels
df['domain'] = '<' + df['domain'].str.lower() + '>'
# cast int to string in order to add < and >
df['rating'] = df['rating'].astype("string")
df['rating'] = '<' + df['rating'].str.lower() + '>'


# In[18]:


# replace title boundary with more explicit special token
df['review'] = df['review'].str.replace('---SEP---', '<endtitle>')


# In[19]:


# add language tags used by mBART
mbart_lang_tags = {
    'de': 'de_DE',
    'en': 'en_XX',
    '<de>': 'de_DE',
    '<en>': 'en_XX',
}    

df['mbart_lang_tags'] = df['lang'].apply(lambda x: mbart_lang_tags[x])


# In[20]:


# inspect DF
get_detailed_info_on_df(df, split_col)
print()
df = ensure_no_split_overlap(df, 'review', 'response', split_col)
print()


# In[21]:


df.head()


# In[22]:


# inspect texts

print(df.iloc[1]['review_clean'])
print(df.iloc[1]['review'])
print(df.iloc[1]['response'])
print(df.iloc[1]['response_clean'])


# # Write output datasets
# 
# given the names of columns in dict `col_name_outfile_mapping` (keys), produce line-aligned output files for each for the different splits.

# In[23]:



col_name_outfile_mapping = {
    'reviewid': 'id', 
    'review': 'review', # normal review 
    'establishment_cat': 'est_label',
    'response': 'response',  # normal response
    'domain': 'domain', # normal domain
    'rating': 'rating', # normal review rating
    'establishment': 'establishment', 
    'source': 'source',
    'mbart_lang_tags': 'lang_tags'
}

def generate_model_files(df,
                         outdir: str,
                         col_name_outfile_mapping: Dict = col_name_outfile_mapping,
                         split_col: str = split_col,
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

outdir=Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)
generate_model_files(df, outdir, col_name_outfile_mapping, split_col)


# In[ ]:




