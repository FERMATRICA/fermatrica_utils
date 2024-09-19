"""
Math operations, mostly returning primitives.
For time series operations see `fermatrica_utils.arrays.ts`
"""


import pandas as pd
import numpy as np


def weighted_mean_group(ds: pd.DataFrame
                        , values: str
                        , weights: str
                        , by: str | list | None = None):
    """
    Weighted mean on dataset ds columns with or without groupby. Main idea is to circumvent pandas `.transform()`
    restrictions (only one column could be passed to function inside `.transform()` at once, but the weighted
    mean requires two columns: values and weights

    :param ds: data set containing columns mentioned below
    :param values: values column name
    :param weights: weights column name
    :param by: groupby column or columns names
    :return:
    """

    if isinstance(by, str):
        by = [by]

    if by is not None:
        ds = ds[[values] + [weights] + by].copy()

        grouped = ds.groupby(by)
        ds['weighted_average'] = ds[values] / grouped[weights].transform('sum') * ds[weights]

        rtrn = grouped['weighted_average'].sum(min_count=1)  # min_count is required for Grouper objects

    else:
        ds = ds[[values] + [weights]].copy()

        rtrn = sum(ds[values] / ds[weights].sum() * ds[weights])

    return rtrn


def round_up_to_odd(x: float | int):
    """
    Rounds to the nearest odd integer

    :param x:
    :return:
    """

    if np.ceil(x) // 2 * 2 + 1 > x:
        return int(np.ceil(x) // 2 * 2 + 1)
    else:
        return int(np.ceil(x) // 2 * 2 + 3)


