#!/usr/bin/env python3

"""Clean and deduplicate re:spondelligent review-response pairs from raw CSVs.

This script reads in 'raw data' CSVs extracted from a re:spondelligent DB dump and
performs cleaning and deduplication on review-response pairs. The output is an
aggregated pandas DataFrame with each row consisting of a single data point for
training review-response generation models.

NOTE: depending on how the data is extracted from the DB, errors may be raised when
trying to parse the raw CSV files. Exporting from the DB with JSON is much safer!

Example Call:

    (2021_01 dump)

    python -m readvisor.data.clean_reviews_csv /mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv /mnt/storage/clwork/projects/readvisor/RESPONSE_GENERATION/intermediary/respondelligent_2021_01.pkl
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from readvisor.data import parallel as mp
from readvisor.data import prep_config as resp_tools
from readvisor.data import text_cleaning as clean

logger = logging.getLogger(__name__)

# Respondelligent customer data (2021_01 DB dump)

col_names = {'author': 'review_author',
             'groupid': 'grpid',
             'reviewtext': 'review_raw',
             'answertext': 'response_raw',
             'answerauthor': 'response_author',
             'channelgroup': 'domain',
             'name': 'establishment',
             'status': 'source',
             }


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inpath', type=Path, help='directory containing the raw exported CSVs')
    ap.add_argument('output_file', type=str, help='path to write the pickled output DataFrame to')
    ap.add_argument('--n_cores', type=int, default=32, help='number of cores for parallel processing')
    return ap.parse_args()


def get_group_ids(df):
    """Extract a dict mapping `id` to `channelgroup` from sf_guard_rechannel_merged_cleaned.csv."""
    grpids = dict()
    for tup in df.itertuples():
        grpids[tup.id] = tup.channelgroup
    return grpids


def merge_reviewtext_reviewtitle(title, text):
    if title and text:
        return f'{title} ---SEP--- {text}'
    elif text:
        return text
    elif title:
        return title
    else:
        return None


def clean_and_assign_lang(df):
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
    """Reduce text to a string of chars with no spaces to make comparison more accurate."""
    try:
        text = re.sub(r'\s+', '', text.lower())
        return text
    except AttributeError:
        return ''


def special_deduplication(df):
    """Deduplicate the DataFrame based on string matches with NO whitespace chars."""
    logger.info('Deduplicating DF based on string matches without whitespace characters...')

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


def main() -> None:
    args = set_args()

    inpath = args.inpath
    output_file = args.output_file
    n_cores = args.n_cores

    review_csv = inpath / 'reviews.raw_export_2022.csv'
    response_csv = inpath / 'reviewanswers.raw_export_2022.csv'
    additional_info = inpath / 'sf_guard_rechannel_merged_cleaned_2022.csv'

    # Prevent overwriting existing data!
    if Path(output_file).exists():
        logger.error('%s already exists. Choose another output file name!', output_file)
        sys.exit()

    review_df = pd.read_csv(review_csv, sep=';', header=0, index_col=False, names=list(resp_tools.review_dtypes.keys()), dtype=resp_tools.review_dtypes, parse_dates=resp_tools.review_dates, on_bad_lines='skip', engine='c')
    response_df = pd.read_csv(response_csv, sep=';', header=0, index_col=False, names=list(resp_tools.answer_dtypes.keys()), dtype=resp_tools.answer_dtypes, parse_dates=resp_tools.answer_dates, on_bad_lines='skip', engine='c')
    info_df = pd.read_csv(additional_info, sep=';', header=0, index_col=0)

    # read in tables from respondelligent DB dumps and drop any
    # columns not of interest to us
    review_df.drop(["reviewdate", "created_at", "updated_at",
                    "reviewstatus", "deleted_at", "imported_at", "assigneduserid",
                    "reviewcreated_at", "reviewlang", "assigned_at", "reviewid"], axis=1, inplace=True)

    response_df.drop(["id", "groupid", "answerdate", "created_at"], axis=1, inplace=True)

    # merge the two dfs into a single df using 'id' from review df and 'reviewid' from response df
    df = pd.merge(review_df, response_df, left_on='id', right_on='reviewid',
                  suffixes=('_rev', '_ans'), how='outer', validate='m:m').reset_index()

    logger.info(f'Paired df count: {len(df)}. Info DF count: {len(info_df)}.')
    df.dropna(subset=['groupid'], inplace=True)
    info_df.dropna(subset=['id'], inplace=True)
    logger.info(f'Dropped items with nan in id columns. Paired df count: {len(df)}. Info DF count: {len(info_df)}.')

    df = pd.merge(df, info_df, left_on='groupid', right_on='id', suffixes=('', '_y'), how='left', validate='m:m').reset_index()
    # keep only rows where answer status is 're' or 'platform'
    # i.e. dropping 'feedback', 'feedbackre' and 'declined'
    df = df[(df.status == 're') | (df.status == 'platform')]

    # rename column names for easier processing
    df.rename(columns=col_names, inplace=True)

    # replacing all empty values, helps for processing string columns
    df.replace({np.nan: None}, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # clean review+title, removing translations and normalising whitespace
    logger.info('Cleaning text fields in DataFrame...')

    df = mp.parallelize_dataframe(df, clean_and_assign_lang, n_cores)

    # drop rows where domain is not restaurant or hotel
    df = df[(df.domain == 'Restaurant') | (df.domain == 'Hotel')]

    # remove all non en-en and de-de pairs
    de = df[(df['review_lang'] == 'de') & (df['response_lang'] == 'de')]
    en = df[(df['review_lang'] == 'en') & (df['response_lang'] == 'en')]

    df = pd.concat([de, en])
    df['lang'] = df['review_lang']

    df = special_deduplication(df)

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
    df.to_csv(Path(output_file).with_suffix('.csv'))

    logger.info(f'DataFrame with {len(df)} entries and columns {str(df.columns)} pickled to files {output_file}')

    logger.info('done!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
