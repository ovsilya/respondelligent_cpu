#!/usr/bin/env python

"""Generate line-aligned mBART training input files from a prepared DataFrame.

This script takes a pickled pandas DataFrame and applies preprocessing
transformations to prepare input files for training an mBART model:

    - mask greetings and salutations
    - add brackets to rating
    - add brackets to domain
    - replace the ``---SEP---`` title/text boundary token with ``<endtitle>``
    - add establishment labels
    - write train/test/valid split files as single-column TSV files

UPDATED 03.06.2021
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import spacy
from flair.models import SequenceTagger
from tqdm.auto import tqdm

from readvisor.data import parallel as mp
from readvisor.data.greetings import mask_greetings_and_salutations_in_spacy_doc

tqdm.pandas()

pd.options.display.max_columns = 999

logger = logging.getLogger(__name__)

# Default config paths (originally hardcoded). Override on the command line.
DEFAULT_RESPO_DATA = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart/respo_data_2022.pkl'
DEFAULT_ESTABL_LABELS = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart/est_labels_2022.txt'
DEFAULT_OUTDIR = '/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart'
DEFAULT_FLAIR_MODEL = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/ml_grt_slt_flair_multi_fast/best-model.pt'
DEFAULT_EN_SPACY_MODEL = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/spacy/readvisor_in_domain_ner/en_core_web_md-2.3.1'
DEFAULT_DE_SPACY_MODEL = '/home/ovsyannikovilyavl/respondelligent/rg/data_prep/models/spacy/readvisor_in_domain_ner/de_core_news_md-2.3.0'

# Processing variables (defaults)
RANDOM_SEED = 1247
lang = 'ml'  # ml = multilingual (for mBART)
split_col = 'split_imrg_compat'  # use this instead of old `split`!!!
n_cores = 32  # number of cores for parallel processing
do_mask_greetings = True  # take approx. an hour to process ~20K items in df
apply_lowercase = False

# Populated at runtime by ``main`` and read by the module-level worker functions
# (inherited by worker processes via ``fork``).
tagger = None
en_nlp = None
de_nlp = None
estabs = {}

col_name_outfile_mapping = {
    'reviewid': 'id',
    'review': 'review',  # normal review
    'establishment_cat': 'est_label',
    'response': 'response',  # normal response
    'domain': 'domain',  # normal domain
    'rating': 'rating',  # normal review rating
    'establishment': 'establishment',
    'source': 'source',
    'mbart_lang_tags': 'lang_tags'
}


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--respo_data', type=str, default=DEFAULT_RESPO_DATA, help='pickled input DataFrame')
    ap.add_argument('--establ_labels', type=str, default=DEFAULT_ESTABL_LABELS, help='establishment-label TSV file')
    ap.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR, help='output directory for the model files')
    ap.add_argument('--flair_model', type=str, default=DEFAULT_FLAIR_MODEL, help='Flair greeting/salutation tagger model')
    ap.add_argument('--en_spacy_model', type=str, default=DEFAULT_EN_SPACY_MODEL, help='English spaCy model')
    ap.add_argument('--de_spacy_model', type=str, default=DEFAULT_DE_SPACY_MODEL, help='German spaCy model')
    ap.add_argument('--n_cores', type=int, default=n_cores, help='number of cores for parallel processing')
    ap.add_argument('--no_mask_greetings', dest='do_mask_greetings', action='store_false', help='disable greeting/salutation masking')
    return ap.parse_args()


##################
# helper functions
##################

def assign_splits(df):
    """Populate a new ``split`` column by position on an already-shuffled DataFrame.

    top 5% = test, next 5% = valid, remaining 90% = train.
    """
    total = len(df)
    test = ['test'] * (int(total * .05))
    valid = ['valid'] * (int(total * .05))
    train = ['train'] * (total - (len(test) + len(valid)))
    split_labels = test + valid + train
    assert len(df) == len(split_labels)
    df['split'] = split_labels
    return df


def get_detailed_info_on_df(df, split_col):
    """Log basic size/distribution info about the DataFrame's splits."""
    logger.info(f'DF has {len(df)} entries')
    logger.info(f'DF COLS: {df.columns}')
    logger.info(f"{df.groupby('source')[split_col].value_counts()}")
    logger.info(f'{df[split_col].value_counts()}')
    return


def token_count(string):
    """Return the whitespace-token count of ``string``."""
    tokens = string.split()
    return len(tokens)


def ensure_no_split_overlap(df, column_a, column_b, split_col):
    """Check for and warn about overlap between train / test / dev splits.

    Duplicates can appear after removing greetings/salutations and applying BPE.
    """
    logger.info(f'CHECKING FOR DUPLICATES IN COLS: {column_a} {column_b}')

    train_src = df[df[split_col] == 'train'][column_a].to_list()
    train_tgt = df[df[split_col] == 'train'][column_b].to_list()

    test_src = df[df[split_col] == 'test'][column_a].to_list()
    test_tgt = df[df[split_col] == 'test'][column_b].to_list()

    valid_src = df[df[split_col] == 'valid'][column_a].to_list()
    valid_tgt = df[df[split_col] == 'valid'][column_b].to_list()

    train = set(zip(train_src, train_tgt))
    test = set(zip(test_src, test_tgt))
    valid = set(zip(valid_src, valid_tgt))
    logger.info(f'TRAIN {len(train)}')
    logger.info(f'TEST {len(test)}')
    logger.info(f'VALID {len(valid)}')
    logger.info('-----------------')
    tt = train.intersection(test)
    tv = train.intersection(valid)
    testv = test.intersection(valid)
    if (len(tt) != 0) or (len(tv) != 0) or (len(testv) != 0):
        logger.warning('FOUND OVERLAP IN')
        logger.warning(f'\tTRAIN / TEST: {len(tt)}')
        logger.warning(f'\tTRAIN / VALID: {len(tv)}')
        logger.warning(f'\tTEST / VALID: {len(testv)}')
    else:
        logger.info('NO OVERLAP FOUND!')
    return df


def ensure_no_empty_strings(df, column_a, column_b):
    """Drop rows where either ``column_a`` or ``column_b`` is an empty string."""
    logger.info(f'REMOVING EMPTY STRING VALUES FROM DF WITH LENGTH: {len(df)}')
    df = df[(df[column_a] != '') & (df[column_b] != '')]
    logger.info(f'REMOVED ITEMS DF LENGTH: {len(df)}')
    return df


def write_file(series, outfile):
    """Write each value of ``series`` as a line to ``outfile``."""
    with open(outfile, 'w', encoding='utf8') as f:
        for line in series.to_list():
            f.write(f'{line}\n')
    return


def write_length_file(series, outfile):
    """Write each value of ``series`` as a 2-decimal float line to ``outfile``."""
    with open(outfile, 'w', encoding='utf8') as f:
        for x in series.to_list():
            f.write(f'{x:.2f}\n')
    return


def write_np_arrays_file(series, outfile):
    """Write each array value of ``series`` as a space-joined line to ``outfile``."""
    with open(outfile, 'w', encoding='utf8') as f:
        for x in series.to_list():
            f.write(f'{" ".join(map(str, x))}\n')
    return


def get_column_stats(df, col):
    """Log the number and a sample of unique values in ``col``."""
    uniq_vals = df[col].unique()
    logger.info(f'Column `{col}` has {len(uniq_vals)} unique values: e.g.: {list(uniq_vals[:10])}')


def mask_greetings_and_salutations_in_raw_string_EN(text):
    doc = en_nlp(text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    return mask_greetings_and_salutations_in_spacy_doc(doc, tagger)


def mask_greetings_and_salutations_in_raw_string_DE(text):
    doc = de_nlp(text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    return mask_greetings_and_salutations_in_spacy_doc(doc, tagger)


def map_establishments_to_labels_based_on_freq_counts(name, threshold=10):
    """Fetch the appropriate establishment label for a given restaurant/hotel name.

    If the occurrence frequency of the restaurant/hotel in the training data is
    lower than the specified threshold, a catch-all placeholder label is returned.
    This ensures that the model can generalise to infrequent/new customers.
    """
    freq, hum_label, cat_label = estabs.get(name, (0, '<unk_est>', '<est_0>'))
    if freq >= threshold:
        return cat_label
    else:
        return '<est_0>'


def generate_model_files(df,
                         outdir: Path,
                         col_name_outfile_mapping: Dict = col_name_outfile_mapping,
                         split_col: str = split_col,
                         n: int = 0):
    """Generate multiple individual files (one per column).

    For each split (train/test/valid), lines in each output file must correspond
    with each other!
    """
    for split in df[split_col].unique():

        split_df = df[df[split_col] == split]

        # shuffle train set - mainly required after upsampling!
        if split == 'train':
            split_df = split_df.sample(frac=1, random_state=RANDOM_SEED)

        if n:  # just take a head of dataframe
            if split == 'train':
                split_df = split_df.head(n)
            else:
                split_df = split_df.head(int(n * 0.1))

        logger.info(f'{split} split has length: {len(split_df)}')

        for k, v in col_name_outfile_mapping.items():
            if k == 'src_len_cates':
                write_length_file(split_df[k], outdir / f'{split}.{v}')
            elif 'sent_vec' in k:
                write_np_arrays_file(split_df[k], outdir / f'{split}.{v}')
            else:
                write_file(split_df[k], outdir / f'{split}.{v}')

    logger.info('Done!')
    return


def main() -> None:
    global tagger, en_nlp, de_nlp, estabs, split_col

    args = set_args()

    # load in models for processing greetings and salutations
    logger.info('loading tagger model...')
    tagger = SequenceTagger.load(args.flair_model)
    en_nlp = spacy.load(args.en_spacy_model)
    de_nlp = spacy.load(args.de_spacy_model)

    # read in data
    df = pd.read_pickle(args.respo_data)
    logger.info(f'{len(df)}')
    logger.info(f'{df.columns}')

    # select only respondelligent sources!
    # NOTE: source=platform are not always written by respondelligent and introduce noise so leave them behind
    df = df[df['source'] == 're']
    logger.info(f'valid respondelligent responses: {len(df)}')
    # ensure no empty values
    df = df[df['review_clean'] != '']
    logger.info(f'valid respondelligent responses: {len(df)}')
    df = df[df['response_clean'] != '']
    logger.info(f'valid respondelligent responses: {len(df)}')

    # subset data by lang
    df_en = df[df['lang'] == 'en']
    logger.info(f'valid English responses: {len(df_en)}')
    df_de = df[df['lang'] == 'de']
    logger.info(f'valid German responses: {len(df_de)}')

    logger.info('Review-response pair distribution for German:')
    logger.info(f'{df[df.lang == "de"].domain.value_counts()}')
    logger.info('Review-response pair distribution for English:')
    logger.info(f'{df[df.lang == "en"].domain.value_counts()}')

    # apply greeting masks
    # NOTE: this takes approx 20 mins to do 10K examples, so go make a coffee...
    if args.do_mask_greetings:
        en_responses = df_en['response_clean'].tolist()
        en_responses = mp.parallelise(mask_greetings_and_salutations_in_raw_string_EN, en_responses, args.n_cores)
        assert len(en_responses) == len(df_en)
        df_en['response_clean'] = en_responses

        de_responses = df_de['response_clean'].tolist()
        de_responses = mp.parallelise(mask_greetings_and_salutations_in_raw_string_DE, de_responses, args.n_cores)
        assert len(de_responses) == len(df_de)
        df_de['response_clean'] = de_responses

    df = pd.concat([df_en, df_de])
    # shuffle dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED)

    # when processing data from re:spondelligent DBs,
    # split information based on IDs is not available,
    # so here we simply create new splits.
    # NOTE: for better reproducibility, between data versions,
    # a dedicated test set should be developed based on reviewids in re:spondelligent's DB
    if split_col not in df.columns:
        df = assign_splits(df)
        split_col = 'split'

    # inspect DF
    get_detailed_info_on_df(df, split_col)
    ensure_no_split_overlap(df, 'review_clean', 'response_clean', split_col)
    get_column_stats(df, 'rating')
    get_column_stats(df, 'domain')
    get_column_stats(df, 'source')
    get_column_stats(df, 'establishment')

    # map all negative rating values to 1
    logger.info(f'{df["rating"].value_counts()}')
    df.loc[df['rating'] < 1, 'rating'] = 1
    logger.info(f'{df["rating"].value_counts()}')

    # Here, we load the predefined mapping between establishments and their labels for the model.
    # NOTE: to generate the labels for a particular model,
    # see collect_establishment_counts.py,
    # which labels establishments according to collected frequency counts
    # and produces the establ_labels tsv file
    estabs = {}
    if args.establ_labels:
        with open(args.establ_labels, encoding='utf8') as inf:
            for line in inf:
                line = line.rstrip().split('\t')
                estabs[line[0]] = (int(line[1]), line[2], line[3])

    df['establishment_cat'] = df['establishment'].apply(lambda x: map_establishments_to_labels_based_on_freq_counts(x))

    # duplicate src and tgt texts (as a backup in case a mistake was made - saves re-doing mask greetings!)
    df['review'] = df['review_clean']
    df['response'] = df['response_clean']

    # convert raw categorical values to 'special token' labels
    df['domain'] = '<' + df['domain'].str.lower() + '>'
    # cast int to string in order to add < and >
    df['rating'] = df['rating'].astype("string")
    df['rating'] = '<' + df['rating'].str.lower() + '>'

    # replace title boundary with more explicit special token
    df['review'] = df['review'].str.replace('---SEP---', '<endtitle>')

    # add language tags used by mBART
    mbart_lang_tags = {
        'de': 'de_DE',
        'en': 'en_XX',
        '<de>': 'de_DE',
        '<en>': 'en_XX',
    }

    df['mbart_lang_tags'] = df['lang'].apply(lambda x: mbart_lang_tags[x])

    # inspect DF
    get_detailed_info_on_df(df, split_col)
    df = ensure_no_split_overlap(df, 'review', 'response', split_col)

    # inspect texts
    logger.info(f"{df.iloc[1]['review_clean']}")
    logger.info(f"{df.iloc[1]['review']}")
    logger.info(f"{df.iloc[1]['response']}")
    logger.info(f"{df.iloc[1]['response_clean']}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    generate_model_files(df, outdir, col_name_outfile_mapping, split_col)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
