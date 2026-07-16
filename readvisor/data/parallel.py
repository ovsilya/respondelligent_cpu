#!/usr/bin/env python3

"""Multiprocessing helpers for applying functions over iterables and DataFrames."""

import logging
import multiprocessing
import time
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parallelise(func: Callable, iterable: Iterable, n_cores: int) -> List:
    """Apply ``func`` to each item of ``iterable`` across ``n_cores`` processes.

    Args:
        func: Function to be applied to each element.
        iterable: List-type object for processing.
        n_cores: Number of worker processes to spawn.

    Returns:
        List of results, in input order.
    """
    logger.info("Running jobs on %d CPU(s)", n_cores)

    start_time = time.time()

    with multiprocessing.Pool(n_cores) as p:
        result = list(tqdm(p.imap(func, iterable), total=len(iterable)))
        p.close()
        p.join()

    logger.info("Time taken: %.2f seconds", time.time() - start_time)

    return result


def parallelize_dataframe(df: pd.DataFrame, func: Callable, n_cores: int = 10) -> pd.DataFrame:
    """Split ``df`` into chunks and apply ``func`` to each chunk in parallel.

    For large DataFrames, limit ``n_cores`` to avoid OOM errors.

    Source: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

    Args:
        df: Pandas DataFrame for processing.
        func: Function to be applied to each chunk.
        n_cores: Number of jobs to be spawned.

    Returns:
        The concatenated, processed DataFrame.
    """
    logger.info("Running jobs on %d CPU(s)", n_cores)

    start_time = time.time()

    df_splits = np.array_split(df, n_cores)
    with multiprocessing.Pool(n_cores) as p:
        result = list(tqdm(p.imap(func, df_splits), total=len(df_splits)))
        p.close()
        p.join()

    df = pd.concat(result)

    logger.info("Time taken: %.2f seconds", time.time() - start_time)

    return df
