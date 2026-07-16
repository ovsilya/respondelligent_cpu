#!/usr/bin/env python3

"""Convert re:spondelligent JSON exports (reviews + answers) to well-formed CSV.

Example call:

    python -m readvisor.data.convert_json_to_csv ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/json ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/csv
"""

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('indir', type=Path, help='directory containing the raw JSON exports')
    ap.add_argument('outdir', type=Path, help='directory to write the CSV files to')
    return ap.parse_args()


def main() -> None:
    args = set_args()

    indir = args.indir
    outdir = args.outdir

    outdir.mkdir(exist_ok=True, parents=True)

    reviews_in = indir / 'reviews.raw_export.json'
    responses_in = indir / 'reviewanswers.raw_export.json'

    reviews_out = outdir / 'reviews.raw_export.csv'
    responses_out = outdir / 'reviewanswers.raw_export.csv'

    df = pd.read_json(reviews_in)
    df.to_csv(reviews_out, header=True, index=False, sep=';', quoting=csv.QUOTE_NONNUMERIC)

    df = pd.read_json(responses_in)
    df.to_csv(responses_out, header=True, index=False, sep=';', quoting=csv.QUOTE_NONNUMERIC)

    logger.info('done!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
