#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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

    python remove_greetings_in_df_texts.py \
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
import sys
import argparse
import pandas as pd
import time

import greetings_flair as rg_utils
import multiprocessing_utils as mp_utils

from flair.models import SequenceTagger


ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True, type=str, help='dataframe for processing')
ap.add_argument('--output', required=False, type=str, help='path to save output dataframe, if not given, updates input file in place')
ap.add_argument('--flair_model', required=True, default='/srv/scratch2/kew/flair_resources/taggers/ml_grt_slt_flair_multi_fast/best-model.pt', type=str, help='full path to flair sequence tagger model')    
ap.add_argument('--n_cores', required=False, type=int, default=16, help='number of cores to use for parallel processing')
ap.add_argument('--test', required=False, action='store_true', help='if provided, only a portion of the dataset is processed as a testrun and debugger is opened for inspection of results.')

args = ap.parse_args()

print(f'loading df from {args.input} ...')

# if args.input.endswith('.pkl'):
df = pd.read_pickle(args.input)
# elif args.input.endswith('.csv'):
#     df = pd.read_csv(args.input)


# if running test, just take the first 1000 items of dataframe
if args.test:
    df = df.sample(1000)

print('loading tagger model...')
# load tagger model
tagger = SequenceTagger.load(args.flair_model) 

# setup output file
if not args.output:
    print(f'[!] {args.input} will be updated in place')
    outfile = args.input
else:
    outfile = args.output

if not outfile:
    raise RuntimeError

print(f'processing {len(df)} items...')

def remove_greetings_with_flair(df, tagger=tagger):
    """
    Uses pandas vectorized apply function, calling masking
    function on texts in column
    """
    df['response'] = df['response'].apply(lambda x: rg_utils.mask_greetings_and_salutations(x, tagger))
    return df

# start timer for logging
START_TIME = time.time()

# process df with multiprocessing for speed ups
df = mp_utils.parallelize_dataframe(df, remove_greetings_with_flair, n_cores=args.n_cores)
   
END_TIME = time.time()

print(f'processed {len(df)} in {END_TIME-START_TIME:.4f} seconds')

if args.test:
    import pdb
    pdb.set_trace()
    sys.exit()

df.to_pickle(outfile)
print(f'saved dataframe to {outfile}')
    