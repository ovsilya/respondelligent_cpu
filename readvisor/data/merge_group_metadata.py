#!/usr/bin/env python3

"""Merge ``sf_guard_group`` and ``rechannel`` exports into a single metadata CSV.

Merges information from two tables in the re:spondelligent DB (exported as JSON
files), combining rows using ``rechannel_id`` as key. This ensures that
review-response pairs can later be joined with their relevant metadata for model
training.

Example call:

    python -m readvisor.data.merge_group_metadata ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/json ~/readvisor_proj/CLFILES_readvisor/respondelligent/2021_01/exported_from_mysql/csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('in_dir', type=Path, help='directory containing the raw JSON exports')
    ap.add_argument('out_dir', type=Path, help='directory to write the merged CSV to')
    return ap.parse_args()


def norm_whitespace(string: Optional[str]) -> Optional[str]:
    """Collapse runs of whitespace to a single space; pass non-strings through."""
    try:
        string = re.sub(r'[\n\r\t\s]+', ' ', string)
    except Exception:
        pass
    return string


def main() -> None:
    args = set_args()

    sf_guard_group = args.in_dir / 'sf_guard_group.raw_export.json'
    rechannel = args.in_dir / 'rechannel.raw_export.json'
    outfile = args.out_dir / 'sf_guard_rechannel_merged_cleaned.csv'

    sf_guard_group_df = pd.read_json(sf_guard_group)
    rechannel_df = pd.read_json(rechannel)

    df = pd.merge(sf_guard_group_df, rechannel_df, left_on='rechannel_id', right_on='id', suffixes=['', '_rechannel'])

    df = df.map(norm_whitespace)

    df.to_csv(outfile, sep=';', header=True)

    logger.info('done')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
