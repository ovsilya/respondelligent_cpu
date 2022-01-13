#!/usr/bin/env python3
# coding: utf-8

"""

This script reads in 'raw data' csvs extracted from
re:spondelligent DB dump and performs cleaning and
deduplication on review response pairs. The output is an
aggregated pandas dataframe with each row consisting of a
single data point for training review-response generation
models.

NOTE: depending on how the data is extracted from the DB,
errors may be raised when trying to parse the raw CSV files.
Exporting from the DB with JSON is much safer!

Example Call:
    
    (2020_04 dump)
    
    python clean_respondelligent_data_from_extracted_csv.py /mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/csv /mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/csv/respo_data.pkl

    (2021_01 dump)

    python clean_respondelligent_data_from_extracted_csv.py /mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv /mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv/respo_data.pkl

"""


import sys
import warnings
import pandas as pd
import re
import numpy as np
from pathlib import Path

from utils_pkg import respondelligent_prep_tools as resp_tools
from utils_pkg import text_cleaning_utils as clean
from utils_pkg import text_preprocessing_utils as text_pp
from utils_pkg import multiprocessing_utils as mp


warnings.filterwarnings('ignore')


# Respondelligent customer data (2021_01 DB dump)
# -----------
# Global vars
# -----------

inpath = Path(sys.argv[1]) # e.g. /mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql
output_file = sys.argv[2] # e.g. /mnt/storage/clwork/projects/readvisor/RESPONSE_GENERATION/intermediary/respondelligent_2021_01.pkl

review_csv = inpath / 'reviews.raw_export.csv' # 'reviews.raw_export.csv'
response_csv = inpath / 'reviewanswers.raw_export.csv' #'reviewanswers.raw_export.csv'

additional_info = inpath / 'sf_guard_rechannel_merged_cleaned.csv'

# Prevent overwiting existing data!
if Path(output_file).exists():
    print(f'{output_file} already exists. Choose another output file name!')
    sys.exit()

n_cores = 32

col_names = {'author': 'review_author',
             'groupid': 'grpid',
             'reviewtext': 'review_raw',
             'answertext': 'response_raw',
             'answerauthor': 'response_author',
             'channelgroup': 'domain',
             'name': 'establishment',
             'status': 'source',
             }


def get_group_ids(df):
    """
    extracts dict mapping `id` to `channelgroup` from sf_guard_rechannel_merged_cleaned.csv
    """

    grpids = dict()
    for tup in df.itertuples():
        grpids[tup.id] = tup.channelgroup
    return grpids

# ---------
# functions
# ---------


# def get_domain(grpid_val):
#     """
#     Gets the domain of the relevant groupid from the 
#     dictionary read in from group_domain.txt file
#     """
#     domain_lookup = grpids.get(grpid_val)
#     if not domain_lookup:
#         print('no domain for', grpid_val)
#     return domain_lookup


def merge_reviewtext_reviewtitle(title, text):

    if title and text:
        merge = '{} ---SEP--- {}'.format(title, text)
        return merge
    elif text:
        return text
    elif title:
        return title
    else:
        return None


def clean_and_assign_lang(df):

    # add high-level domain info
    # df['domain'] = df.apply(lambda x: get_domain(x.grpid), axis=1)

    # clean up
    df['review_raw'] = df.apply(
        lambda x: merge_reviewtext_reviewtitle(x['reviewtitle'], x['review_raw']), axis=1)

    df['review_clean'] = df.apply(
        lambda x: clean.clean_translations(clean.clean_html(x['review_raw'])), axis=1)

    df['response_clean'] = df.apply(
        lambda x: clean.clean_translations(clean.clean_html(x['response_raw'])), axis=1)

    # lang detection
    df['review_lang'] = df.apply(
        lambda x: clean.assign_lang(x['review_clean']), axis=1)

    df['response_lang'] = df.apply(
        lambda x: clean.assign_lang(x['response_clean']), axis=1)

    return df


def reduce_to_chars(text):
    """Reduces text to string of chars with no spaces
    to make string comparison more accurate."""
    try:
        text = re.sub('\s+', '', text.lower())
        return text
    except AttributeError:
        return ''


def special_deduplication(df):
    """
    Performs deduplication based on string matches with
    NO whitespace chars.

    """

    print('Deduplicating DF based on string matches without whitespace characters...')

    df['review_chars'] = df['review_clean'].apply(
        lambda x: reduce_to_chars(x))
    df['answer_chars'] = df['response_clean'].apply(
        lambda x: reduce_to_chars(x))

    # get the length of the original text string
    # e.g. 'thank you,user' < 'thank you, user'
    df['review_length'] = df['review_raw'].str.len()
    df['answer_length'] = df['response_raw'].str.len()

    # sort DF by length of review and answer string sequence: longest first
    df = df.sort_values(['review_length', 'answer_length'], ascending=False)

    # remove duplicate string texts based on character sequence duplicates
    # keep='first' we keep the longest sequence only
    df = df.drop_duplicates(['review_chars', 'answer_chars'],
                            keep='first').reset_index(drop=True)

    return df


if __name__ == "__main__":

    review_df = pd.read_csv(review_csv, sep=';', header=0, index_col=False, names=list(resp_tools.review_dtypes.keys()), dtype=resp_tools.review_dtypes, parse_dates=resp_tools.review_dates)
    response_df = pd.read_csv(response_csv, sep=';', header=0, index_col=False, names=list(resp_tools.answer_dtypes.keys()), dtype=resp_tools.answer_dtypes, parse_dates=resp_tools.answer_dates)

    info_df = pd.read_csv(additional_info, sep=';', header=0, index_col=0)

    # breakpoint()

    ### backward compatibility for getting date-specific entries
    # drop any items which were updated BEFORE the previous db dump was created (i.e. 27.03.2020) 
    # date_threshold = '2020-03-28'
    # review_df = review_df[review_df['updated_at'] >= date_threshold]
    # response_df = response_df[response_df['updated_at'] >= date_threshold]
    # print(f'Dropped items older than {date_threshold} from dataframes. Review count: {len(review_df)}. Response count: {len(response_df)}.')
    
    # read in tables from respondelligent DB dumps and drop any
    # columns not of interest to us
    review_df.drop(["reviewdate", "created_at", "updated_at",
                    "reviewstatus", "deleted_at", "imported_at", "assigneduserid",
                    "reviewcreated_at", "reviewlang", "assigned_at", "reviewid"], axis=1, inplace=True)

    response_df.drop(["id", "groupid", "answerdate", "created_at"], axis=1, inplace=True)
    
    # merge the two dfs into a single df using 'id' from review df and 'reviewid' from response df
    df = pd.merge(review_df, response_df, left_on='id', right_on='reviewid',
                  suffixes=('_rev', '_ans'), how='outer', validate='m:m').reset_index()
    
    print(f'Paired df count: {len(df)}. Info DF count: {len(info_df)}.')
    df.dropna(subset=['groupid'], inplace=True)
    info_df.dropna(subset=['id'], inplace=True)
    print(f'Dropped items with nan in id columns. Paired df count: {len(df)}. Info DF count: {len(info_df)}.')

    df = pd.merge(df, info_df, left_on='groupid', right_on='id', suffixes=('', '_y'), how='left', validate='m:m').reset_index()
    # keep only rows where answer status is 're' or 'platform'
    # i.e. dropping 'feedback', 'feedbackre' and 'declined'
    df = df[(df.status == 're') | (df.status == 'platform')]

    # rename column names for easier processing
    df.rename(columns=col_names, inplace=True)

    # replacing all empty values, helps for processing string columns
    df.replace({np.nan: None}, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # clean review+tile, removing translations and normalising whitespace
    print('Cleaning text fields in DataFrame...')

    df = mp.parallelize_dataframe(df, clean_and_assign_lang, n_cores)
    # df = clean_and_assign_lang(df)

    # drop rows where domain is not restaurant or hotel
    df = df[(df.domain == 'Restaurant') | (df.domain == 'Hotel')]

    # remove all non en-en and de-de pairs
    de = df[(df['review_lang'] == 'de') & (df['response_lang'] == 'de')]
    en = df[(df['review_lang'] == 'en') & (df['response_lang'] == 'en')]

    # print('German rr pairs:', len(de))
    # print('English rr pairs:', len(en))
    df = pd.concat([de, en])
    df['lang'] = df['review_lang']

    df = special_deduplication(df)

    # print('Preprocessing text fields in DataFrame...')
    # apply text preprocessing
    # df = mp.parallelize_dataframe(df, preprocess_df_fields, n_cores)

    # reorder columns and drop those that are no longer necessary
    df = df[
        ["reviewid",
         "grpid",
         "domain",
         "platformid_rev",
         "rating",
         "url",
         "platformrating",
         "review_author",
         "response_author",
         "review_clean",
         "response_clean",
         "lang",
         "source",
         "establishment"
         ]
    ]

    df.replace(r'^(None|\s*)$', pd.NA, regex=True, inplace=True)

    df.fillna(value=pd.NA, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # save processed dataframe
    df.to_pickle(output_file)
    df.to_csv(output_file[:-3]+'csv')
       
    # de = df[(df['review_lang'] == 'de')]
    # en = df[(df['review_lang'] == 'en')]
    
    # de.to_pickle(output_file[:-4]+'_de.pkl')
    # en.to_pickle(output_file[:-4]+'_en.pkl')
    
    print(f'DataFrame with {len(df)} entries and columns {str(df.columns)} pickled to files {output_file}')
    
    print('done!')
