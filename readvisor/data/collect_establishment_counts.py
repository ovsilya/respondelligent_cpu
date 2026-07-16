#!/usr/bin/env python

"""Collect establishment occurrence frequency counts from a prepared DataFrame.

Reads a pickled DataFrame of review-response pairs and writes a TSV mapping each
establishment name to its frequency count, a human-readable label, and a
categorical ``<est_N>`` label used by the mBART model.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def set_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('in_data', type=Path, help='pickled DataFrame of review-response pairs')
    ap.add_argument('outfile', type=Path, help='path to write the establishment-label TSV to')
    return ap.parse_args()


def main() -> None:
    args = set_args()

    df = pd.read_pickle(args.in_data)

    establishment_counts = df.establishment.value_counts()

    with open(args.outfile, 'w', encoding='utf8') as outf:
        for i, (k, v) in enumerate(establishment_counts.items(), 1):
            logger.info('%s %s', k, v)
            outf.write(f"{k}\t{v}\t<{k.replace(' ', '_')}>\t<est_{str(i)}>\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
