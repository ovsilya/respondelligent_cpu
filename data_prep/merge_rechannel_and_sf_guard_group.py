#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Merges information from two tables in re:spondelligent DB
(exported as JSON files) and combines information using
`rechannel_id` as key.

This ensures that review-response pairs are combined with
their relevant meta data for model training.

Example call:

    python merge_rechannel_and_sf_guard_group.py ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/json ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/csv

"""

import re
import sys
from pathlib import Path

import pandas as pd

in_dir = sys.argv[1]
out_dir = sys.argv[2]

sf_guard_group = Path(in_dir) / 'sf_guard_group.raw_export.json'
rechannel = Path(in_dir) / 'rechannel.raw_export.json'
outfile = Path(out_dir) / 'sf_guard_rechannel_merged_cleaned.csv'

sf_guard_group_df = pd.read_json(sf_guard_group)
rechannel_df = pd.read_json(rechannel)

def norm_whitespace(string):
    try:
        string = re.sub(r'[\n\r\t\s]+', ' ', string)
    except:
        pass
    return string

df = pd.merge(sf_guard_group_df, rechannel_df, left_on='rechannel_id', right_on='id', suffixes=['', '_rechannel'])

df = df.applymap(norm_whitespace)

df.to_csv(outfile, sep=';', header=True)

print('done')