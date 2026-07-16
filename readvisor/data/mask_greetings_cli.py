#!/usr/bin/env python3

"""CLI to mask greetings/salutations in a DataFrame's ``response`` column with Flair.

March 2021

Using 52 cores on Vigrid should take ~ 40 seconds to do 1000 items.

Due to the large size of the English dataset (3.3M),
parallelisation of the pandas df cannot be done on all 52
cores. Simplest workaround is to extract the response column
from the dataset and save it to a tmp file. Then run this
script, specifying the much smaller tmp dataset as input.

Expected run time for 3.3M items:

    ~ 30 secs / 1000 items
    3300 * 30 / 3600 = 28 hours


Example call:

    python -m readvisor.data.mask_greetings_cli \
        --input /mnt/storage/clwork/projects/readvisor/RESPONSE_GENERATION/intermediary/de_rrgen.sent_seq.pkl \
        --output /mnt/storage/clwork/projects/readvisor/RESPONSE_GENERATION/intermediary/de_rrgen.sent_seq.rg.pkl \
        --flair_model /srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flair_multi_fast/best-model.pt \
        --n_cores 52

Available Flair models:

- /srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flair_multi_fast/best-model.pt
    - acc ~ 80%
- /srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flair/best-model.pt
    - acc ~ 82%
- /srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flairbert/best-model.pt:
    - acc ~ 77%
"""

import argparse
import logging
import sys
import time

import pandas as pd
from flair.models import SequenceTagger

from readvisor.data import greetings as rg_utils
from readvisor.data import parallel as mp_utils

logger = logging.getLogger(__name__)

# Loaded in ``main`` and read by ``remove_greetings_with_flair`` in each worker
# process (inherited via ``fork``); kept module-global so the worker function stays
# top-level and picklable, matching the original implementation.
tagger = None


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=str, help='dataframe for processing')
    ap.add_argument('--output', required=False, type=str, help='path to save output dataframe, if not given, updates input file in place')
    ap.add_argument('--flair_model', required=True, default='/srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flair_multi_fast/best-model.pt', type=str, help='full path to flair sequence tagger model')
    ap.add_argument('--n_cores', required=False, type=int, default=16, help='number of cores to use for parallel processing')
    ap.add_argument('--test', required=False, action='store_true', help='if provided, only a portion of the dataset is processed as a testrun.')
    return ap.parse_args()


def remove_greetings_with_flair(df):
    """Mask greetings/salutations in the ``response`` column via the module ``tagger``."""
    df['response'] = df['response'].apply(lambda x: rg_utils.mask_greetings_and_salutations(x, tagger))
    return df


def main() -> None:
    global tagger

    args = set_args()

    logger.info('loading df from %s ...', args.input)
    df = pd.read_pickle(args.input)

    # if running test, just take the first 1000 items of dataframe
    if args.test:
        df = df.sample(1000)

    logger.info('loading tagger model...')
    # load tagger model
    tagger = SequenceTagger.load(args.flair_model)

    # setup output file
    if not args.output:
        logger.info('[!] %s will be updated in place', args.input)
        outfile = args.input
    else:
        outfile = args.output

    if not outfile:
        raise RuntimeError

    logger.info('processing %d items...', len(df))

    # start timer for logging
    start_time = time.time()

    # process df with multiprocessing for speed ups
    df = mp_utils.parallelize_dataframe(df, remove_greetings_with_flair, n_cores=args.n_cores)

    logger.info('processed %d in %.4f seconds', len(df), time.time() - start_time)

    if args.test:
        sys.exit()

    df.to_pickle(outfile)
    logger.info('saved dataframe to %s', outfile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
