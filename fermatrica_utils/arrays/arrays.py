"""
Utilities to work with arrays of all types from list to pandas DataFrame.
Unlike fermatrica_utils.arrays.data it covers most common tasks not specific to
data pipeline or something.
"""


import copy
import numpy as np
import pandas as pd
import re

from line_profiler_pycharm import profile


def step_generator(start: int | float
                   , stop: int | float
                   , step: int | float = 1):
    """
    Budget (or something) generator

    :param start:
    :param stop:
    :param step:
    :return:
    """

    i = start
    while i <= stop:
        yield i
        i += step


"""
Dictionary utilities
"""


class DotDict(dict):
    """
    Class extending `dict` to access items with '.'. Use only for one-level dictionaries (w/o nexted dicts)
    It is especially useful in dynamic code
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dict_multifill(dct: dict
                   , keys: list | np.ndarray
                   , values: list | np.ndarray) -> dict:
    """
    Fill existing keys of the dictionary with values

    :param dct:
    :param keys:
    :param values:
    :return:
    """

    for k, v in zip(keys, values):
        dct[k] = v

    return dct


def dict_to_df(dct: dict
               , key_name: str | None = None
               , value_name: str | None = None
               , save_pth: str | None = None):
    """
    Convert dictionary to pandas DataFrame and melt it using `key_name` as id_vars and `value_name` as value_name

    :param dct:
    :param key_name:
    :param value_name:
    :param save_pth:
    :return:
    """

    if key_name is None:
        key_name = 'key'
    if value_name is None:
        value_name = 'value'

    df = pd.DataFrame.from_dict(dct, orient='index').reset_index(names=key_name)
    df = df.melt(id_vars=key_name, value_name=value_name)
    df = df.drop(columns='variable').sort_values(by=key_name).dropna(axis=0)

    if save_pth is not None:
        df.to_excel(save_pth, index=False)

    return df


"""
More efficient versions of some Pandas operations (true at least for pandas 1.5.2)
"""


def groupby_eff(ds: pd.DataFrame
                , group_vars: list
                , other_vars: list
                , mask: pd.Series | None = None
                , as_index: bool = True
                , sort: bool = True):
    """
    More efficient version of pandas groupby (at least for pandas 1.5.2)

    :param ds:
    :param group_vars:
    :param other_vars:
    :param mask:
    :param as_index:
    :param sort:
    :return:
    """

    total_vars = copy.deepcopy(group_vars)
    total_vars.extend(other_vars)

    rtrn = [ds[x] for x in total_vars]
    rtrn = pd.concat(rtrn, axis=1)

    if mask is None:
        rtrn = rtrn.groupby(group_vars, sort=sort, as_index=as_index)
    else:
        rtrn = rtrn[mask].groupby(group_vars, sort=sort, as_index=as_index)

    return rtrn


def select_eff(ds: pd.DataFrame
               , select_vars: list):
    """
    More efficient version of pandas [[]] or .loc[, []] (at least for pandas 1.5.2).
    For some pandas selects list of columns much slower than one column

    :param ds:
    :param select_vars:
    :return:
    """

    rtrn = [ds[x] for x in select_vars]
    rtrn = pd.concat(rtrn, axis=1)

    return rtrn


@profile
def str_replace_eff(sr: pd.Series
                    , pattern=r' '
                    , repl=''):
    """
    Replace for pandas.Series() taking into account mixed type columns

    :param sr:
    :param pattern:
    :param repl:
    :return:
    """

    p = re.compile(pattern)
    sr_new = [p.sub(repl, x) if isinstance(x, str) else np.nan for x in sr.tolist()]

    return sr_new


"""
Other Pandas utilities
"""


@profile
def like_sr(sr: pd.Series
            , pattern=r' '):
    """
    Variant of standard `.contains()` with more friendly name and suitable for mixed type columns

    :param sr:
    :param pattern:
    :return:
    """

    p = re.compile(r'.*' + pattern + r'.*')
    sr_new = [bool(p.match(x)) if isinstance(x, str) else np.nan for x in sr.tolist()]

    return sr_new


def rm_1_item_groups(ds: pd.DataFrame
                     , group_var: str | list
                     , inplace: bool = True) -> pd.DataFrame:
    """
    Remove groups containing 1 item only from dataset `ds`. Use it to cleanse data
    before passing to function requiring 2 or more item (some stats function etc.)

    :param ds:
    :param group_var:
    :param inplace:
    :return:
    """

    if not inplace:
        ds = ds.copy()

    if isinstance(group_var, str):

        to_rm = ds.groupby(group_var).apply(lambda x: len(x))
        to_rm = to_rm[to_rm == 1]

        ds = ds[~(ds[group_var].isin(to_rm.index))]

    else:

        to_rm = ds.groupby(group_var).apply(lambda x: len(x))
        to_rm = to_rm[to_rm == 1].rename('count')

        if len(to_rm) > 0:
            ds = ds.join(to_rm, on=group_var, how='left', sort=False)
            ds = ds[ds['count'].isna()]

            del ds['count']

    return ds


def pandas_tree_final_child(ds: pd.DataFrame
                            , var: str
                            , var_ch: str) -> pd.DataFrame:
    """
    If tree is saved in pandas DataFrame such as:

    - `var` column contains current item name
    - `var_ch` column contains child item name

    It is assumed every item has only 1 child item or 0.

    Run through the whole chain from the outermost parent to outermost child
    and keep only outermost parent and outermost child for every chain

    :param ds:
    :param var:
    :param var_ch:
    :return:
    """

    ds['___var'] = ds[var]

    for ind, row in ds.iterrows():
        ds.loc[ds[var] == row[var_ch], var] = row[var]

    ds = ds[~ds[var_ch].isin(ds['___var'])][[var, var_ch]].copy()

    return ds


@profile
def multi_index_destroyer(ds: pd.DataFrame
                          , sep: str = '_'):
    """
    Convert Pandas multi-level column names to string names, e.g.:
    ('awareness', 'f25_64', 'affinity') -> 'awareness_f25_64_affinity'

    :param ds:
    :param sep:
    :return:
    """

    ds.columns = [sep.join(col) for col in ds.columns]
    ds.reset_index(inplace=True)

    return ds


def pandas_filter_regex(ds: pd.DataFrame,
                        col: str,
                        include_pattern: str | None = None,
                        exclude_pattern: str | None = None):
    """
    Extends pandas .filter() method with include and exclude regex patterns. Use it to filter columns by name
    (effectively pick columns with names matching one pattern and not matching another)
    
    :param ds: 
    :param col: 
    :param include_pattern: 
    :param exclude_pattern: 
    :return: 
    """

    tmp = copy.deepcopy(ds)

    # col contains include_filter OR NOT contains exclude filter
    if (include_pattern is not None) & (exclude_pattern is not None):
        mask = tmp[col].str.contains(include_pattern) | ~tmp[col].str.contains(exclude_pattern)
    elif not include_pattern:
        mask = ~tmp[col].str.contains(exclude_pattern)
    else:
        mask = tmp[col].str.contains(include_pattern)

    tmp = tmp[mask]

    return tmp


"""
List utilities: extend with some features common in NumPy, Pandas etc.
"""


def list_select(search: str | list
                , lst: list
                , match: bool = False
                , include: bool = True) -> list | pd.DataFrame:
    """
    Select elements from list matching string or list search pattern precisely or by regex

    :param search: search pattern, string or list of strings
    :param lst: input list
    :param match: if True, use `search` argument as regex pattern. If False, treat `search` as equal to
    :param include: if True, return matching elements. If False, return non-matching elements
    :return:
    """

    if isinstance(search, str):
        if match:
            if include:
                rtrn = [e for e in lst if e == search]
            else:
                rtrn = [e for e in lst if e != search]
        else:
            if include:
                rtrn = [e for e in lst if re.search(search, e)]
            else:
                rtrn = [e for e in lst if not re.search(search, e)]

        return rtrn

    if isinstance(search, list):
        rtrn = {}
        if match:
            if include:
                for sr in search:
                    rtrn[sr] = [e for e in lst if e == sr]
            else:
                for sr in search:
                    rtrn[sr] = [e for e in lst if e != sr]
        else:
            if include:
                for sr in search:
                    rtrn[sr] = [e for e in lst if re.search(sr, e)]
            else:
                for sr in search:
                    rtrn[sr] = [e for e in lst if not re.search(sr, e)]

        rtrn = pd.DataFrame.from_dict(rtrn, orient='index')

        return rtrn


def list_select_pat(search: str
                    , lst: list) -> list:
    """
    Short alias for search in list by regex pattern

    :param search:
    :param lst:
    :return:
    """

    return list_select(search, lst, match=False, include=True)


def list_unique(lst: list) -> list:
    """
    Analogue of list(set(lst)) preserving order of (first appearance) elements

    :param lst: list of elements
    :return: unique list of elements
    """
    uniq_lst = []
    [uniq_lst.append(e) for e in lst if e not in uniq_lst]

    return uniq_lst
