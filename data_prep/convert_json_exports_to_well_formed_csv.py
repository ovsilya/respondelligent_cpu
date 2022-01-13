#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:

    python convert_json_exports_to_well_formed_csv.py ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/json ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/csv

"""

import sys
import pandas as pd
import csv
from pathlib import Path

indir = Path(sys.argv[1])
outdir = Path(sys.argv[2])

outdir.mkdir(exist_ok=True, parents=True)

reviews_in = indir / 'reviews.raw_export.json'
responses_in = indir / 'reviewanswers.raw_export.json'

reviews_out = outdir / 'reviews.raw_export.csv'
responses_out = outdir / 'reviewanswers.raw_export.csv'

df = pd.read_json(reviews_in)
df.to_csv(reviews_out, header=True, index=False, sep=';', quoting=csv.QUOTE_NONNUMERIC)

df = pd.read_json(responses_in)
df.to_csv(responses_out, header=True, index=False, sep=';', quoting=csv.QUOTE_NONNUMERIC)

print('done!')