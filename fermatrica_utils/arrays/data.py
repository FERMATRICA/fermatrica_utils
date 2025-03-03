"""
Utilities to work with data. More specific operations than defined in fermatrica_utils.arrays.arrays
"""


import numpy as np
import pandas as pd
from line_profiler_pycharm import profile
import os
import re
import cyrtranslit
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype

import fermatrica_utils
import fermatrica_utils.arrays.arrays
from fermatrica_utils.arrays.arrays import list_unique, list_select
from fermatrica_utils.primitives.string import cyrillic_trans


"""
Cleanse data operations (mostly to be used in data pipeline)
"""


@profile
def decapitalise_df(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all object and string columns in dataset `ds` to lower case

    :param ds:
    :return:
    """

    for i in ds.columns:
        if is_object_dtype(ds[i]):
            ds[i] = ds[i].astype(str)
        if is_string_dtype(ds[i]):
            ds[i] = ds[i].str.lower()

    return ds


@profile
def extra_spaces_df(ds: pd.DataFrame):
    """
    Remove duplicating and trailing spacing from all object and string columns in dataset `ds`

    :param ds:
    :return:
    """

    for i in ds.columns:
        if is_object_dtype(ds[i]):
            ds[i] = ds[i].astype(str)
        if is_string_dtype(ds[i]):
            ds.loc[:, i] = ds.loc[:, i].str.replace(r'( +)|(_+)', '.', regex=True)
            ds.loc[:, i] = ds.loc[:, i].str.replace(r'\.+', '_', regex=True)
            ds.loc[:, i] = ds.loc[:, i].str.strip(r' \.\_')

    return ds.copy()


@profile
def columns_normalize_df(ds: pd.DataFrame):
    """
    Normalize column names of `ds` dataset

    :param ds:
    :return:
    """

    ds.columns = ds.columns.str.replace(r'( +)|(_+)', '_', regex=True)
    ds.columns = ds.columns.str.replace(r'[_\.]+', '_', regex=True)
    ds.columns = ds.columns.str.strip(r' \.\_').str.lower()
    ds.columns = cyrillic_trans(ds.columns, cyr_dict='cyr_dict_1')

    return ds


"""
"""


def split_dataframe(ds: pd.DataFrame
                    , group_col: str = 'splt') -> list:
    """
    Split dataframe `ds` into list of dataframe by values of `group_col` column

    :param ds: data
    :param group_col: column name to use as groupby column
    :return:
    """

    grouped_ds = ds.groupby(group_col)
    grouped_ds = [grouped_ds.get_group(group) for group in grouped_ds.groups]

    return grouped_ds


@profile
def convert_pyarrow_string(ds: pd.DataFrame
                           , ptrn_incl: str | None = None
                           , ptrn_excl: str | None = None) -> pd.DataFrame:
    """
    Convert standard Pandas strings to PyArrow strings in columns selected by include and exclude patterns.
    Use it PyArrow backend for Pandas is used (significantly faster for string operations at least)

    :param ds:
    :param ptrn_incl:
    :param ptrn_excl:
    :return:
    """

    cln = ds.columns.to_list()

    if ptrn_incl is not None:
        cln = [x for x in cln if re.search(ptrn_incl, x)]

    if ptrn_excl is not None:
        cln = [x for x in cln if not re.search(ptrn_excl, x)]

    for cl in cln:
        if is_string_dtype(ds[cl]):
            ds[cl] = ds[cl].astype('string[pyarrow]')

    return ds


"""
Excel helpers and utilities
"""


def excel_sheets_look(dr: str
                      , ptrn_incl: str | None = None
                      , ptrn_excl: str | None = None
                      , num: bool = False
                      , make_df: bool = False):
    """
    Extract Excel sheet names from all Excel files in directory matching include and exclude patterns

    :param dr: directory to look into
    :param ptrn_incl: pattern to include files
    :param ptrn_excl: pattern to exclude files
    :param num:
    :param make_df: return output as pandas DataFrame. Otherwise, return dictionary
    :return:
    """

    fls = os.listdir(dr)
    if ptrn_incl is not None:
        fls = list_select(ptrn_incl, fls, True, True)
    if ptrn_excl is not None:
        fls = list_select(ptrn_excl, fls, True, False)

    fl_shts = {}
    for fl in fls:

        df = pd.ExcelFile(os.path.join(dr, fl))
        shts = df.sheet_names
        if num:
            fl_shts[fl] = [str(i) + ': ' + shts[i] for i in range(len(shts))]
        else:
            fl_shts[fl] = shts

    if make_df:
        rtrn = pd.DataFrame.from_dict(fl_shts, orient='index')
    else:
        rtrn = fl_shts

    return rtrn


def excel_header_look(dr: str
                      , sht
                      , row_num: int):
    """
    Read top rows from all Excel files in the directory. Use it to have a fast look over the content of the Excel
    files, especially large ones

    :param dr: directory to look into
    :param sht:
    :param row_num:
    :return:
    """
    fls = os.listdir(dr)

    fl_hds = {}
    for fl in fls:
        name = re.sub('\\..*', '', fl)
        pth = os.path.join(dr, fl)
        df = pd.read_excel(pth, sht, header=None)
        df = df.iloc[0:row_num, ]
        fl_hds[name] = df

    return fl_hds


def excel_header_look_addit(dr: str
                            , shts: list | None
                            , row_num: int):
    """

    :param dr:
    :param shts:
    :param row_num:
    :return:
    """
    fls = os.listdir(dr)

    fl_hds = {}
    for fl in fls:
        name = re.sub('\\..*', '', fl)
        pth = os.path.join(dr, fl)
        dt = pd.ExcelFile(pth)

        sht_hds = {}
        shts_c = dt.sheet_names

        if shts is not None:
            shts_c = [x for x in shts_c if x in shts]

        for sht in shts_c:
            df = pd.read_excel(pth, sht, header=None)
            df = df.iloc[0:row_num, ]
            sht_hds[sht] = df

        fl_hds[name] = sht_hds

    return fl_hds


def excel_get_col_vals(dr: str
                       , var_pat: str
                       , skip_rows: int):
    """
    One more Excel helper

    :param dr: directory to look into
    :param var_pat: pattern by which columns are selected
    :param skip_rows: num of row to drop from df
    :return:
    """

    fl_values = {}

    fls = os.listdir(dr)
    for fl in fls:

        name = re.sub('\\.xlsx', '', fl)
        pth = os.path.join(dr, fl)
        dt = pd.ExcelFile(pth)
        shts = dt.sheet_names

        all_values = []

        for sht in shts:

            df = pd.read_excel(pth, sht, skiprows=skip_rows)
            cls = df.columns
            cls_nd = cls[cls.str.contains(var_pat)]

            if len(cls_nd) > 1:
                print(fl, cyrtranslit.to_latin(sht, 'ru'),
                      '\nThere is more than one column relevant to set pattern\n')
                [all_values.extend(df[cl].unique()) for cl in cls_nd]
            else:
                values = df[cls_nd[0]].unique()
                all_values.extend(values)

        fl_values[name] = all_values

    return fl_values


def excel_get_row_vals(dr: str
                       , start_col
                       , row: int | list
                       , origin=True
                       , join=False
                       , sort=False):
    """
    One more Excel helper

    :param dr:
    :param start_col:
    :param row:
    :param origin:
    :param join:
    :param sort:
    :return:
    """

    fl_vals = {}

    fls = os.listdir(dr)
    for fl in fls:
        name = re.sub('\\.xlsx', '', fl)
        pth = os.path.join(dr, fl)
        dt = pd.ExcelFile(pth)
        shts = dt.sheet_names

        vals = []
        for sht in shts:
            df = pd.read_excel(pth, sht, header=None)

            if isinstance(row, int):
                vals.extend(list(fermatrica_utils.arrays.arrays.unique()))
            if isinstance(row, list):
                tmp = df.iloc[row, start_col:]
                tmp.dropna(axis=1, how='all', inplace=True)
                tmp.fillna(method='ffill', axis=1, inplace=True)

                vls = []
                for key in tmp:
                    vls.append(tmp[key].str.cat(sep='_'))

                vls = list_unique(vls)

                vals.extend(vls)

        if not origin:
            vals = [cyrtranslit.to_latin(x, 'ru') for x in vals]
            vals = [str.lower(x) for x in vals]
            vals = [re.sub('[^a-z0-9_]', '', x) for x in vals]

        fl_vals[name] = vals

    if join:
        res = []
        for key in fl_vals:
            res.extend(fl_vals[key])

        res = list_unique(res)
        if sort:
            res.sort()
    else:
        res = fl_vals

    return res


def excel_get_cols(dr: str, shts_dct: dict | None = None
                   , skip_rows: int | None = None
                   , cl_pat: str | None = None
                   , get_vals: bool = False
                   , join_cl_vals: bool = False
                   , join_shts: bool = False):
    """
    One more Excel helper

    :param dr:
    :param shts_dct:
    :param head:
    :param skip_rows:
    :param cl_pat:
    :param get_vals:
    :param join_cl_vals:
    :param join_shts:
    :return:
    """

    fl_shts_cls = {}

    fls = os.listdir(dr)

    for fl in fls:
        name = re.sub('\\.xlsx', '', fl)
        pth = os.path.join(dr, fl)
        dt = pd.ExcelFile(pth)

        if shts_dct is None:
            shts = dt.sheet_names
        else:
            if bool(shts_dct[name]):
                shts = shts_dct[name]
            else:
                shts = dt.sheet_names

        sht_cls = {}
        for sht in shts:
            df = pd.read_excel(pth, sht, skiprows=skip_rows)
            all_cls = df.columns
            if cl_pat is None:
                cls = all_cls.tolist()
            else:
                cls = []
                try:
                    cls = all_cls[all_cls.str.contains(cl_pat)].to_list()
                except:
                    print('sheet ' + sht + ': The error in str.contains()!')

            if get_vals:
                cl_vals = {}

                for cl in cls:
                    vals = df[cl].unique().tolist()
                    cl_vals[cl] = vals

                if join_cl_vals:
                    vals_lst = []
                    for key in cl_vals:
                        vals_lst.extend(cl_vals[key])
                    cl_vals = vals_lst
                cls = cl_vals

            sht_cls[sht] = cls
            if join_shts:
                if get_vals and not join_cl_vals:
                    vals = {}
                    for key in sht_cls:
                        vals.update(sht_cls[key])
                else:
                    vals = []
                    for key in sht_cls:
                        vals.extend(sht_cls[key])

                sht_cls = vals

        fl_shts_cls[name] = sht_cls

    return fl_shts_cls


def excel_df_look(dr: str
                  , shts_dict: dict | None = None
                  , head: int | list | None = None
                  , row_num: int | None = None):
    """
    One more Excel helper

    :param dr:
    :param shts_dict:
    :param head:
    :param row_num:
    :return:
    """

    if shts_dict is None:
        fls = os.listdir(dr)
    else:
        fls = list(shts_dict.keys())

    fl_hds = {}
    for fl in fls:
        # name = re.sub('\\..*', '', fl)
        pth = os.path.join(dr, fl)
        dt = pd.ExcelFile(pth)

        sht_hds = {}

        if shts_dict is None:
            shts = dt.sheet_names
        else:
            if len(shts_dict[fl]) == 0:
                print('Warning!', 'File: ' + fl + ', - sheets not specified\nAll sheets are taken\n')
                shts = dt.sheet_names
            else:
                shts = shts_dict[fl]

        for sht in shts:
            df = pd.read_excel(pth, sht, header=head)
            if row_num is not None:
                df = df.iloc[0:row_num, ]

            sht_hds[sht] = df

        fl_hds[fl] = sht_hds

    return fl_hds


"""
Explore loaded data
"""


@profile
def header_row_detect(ds: pd.DataFrame):
    """
    Find first meaningful (non-empty) row in the loaded dataset. Use it to automatically detect start row
    if garbage records over the first real row not to be expected

    :param ds:
    :return:
    """

    start_rows = []

    for i in ds.columns:
        start = ds.loc[~pd.isna(ds[i]), i].index[0]
        start_rows.append(start)

    start_rows = np.array(start_rows)

    return start_rows
