import math
import yaml
import os

import pandas as pd
import numpy as np
import pytest
from contextlib import nullcontext

from fermatrica_utils.flow import FermatricaUError

def construct_pandas_index(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Index:
    return pd.RangeIndex(**loader.construct_mapping(node))

def construct_pandas_series(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Series:
    return pd.Series(**loader.construct_mapping(node, deep=True))

def construct_pandas_dataframe(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.DataFrame:
    return pd.DataFrame(**loader.construct_mapping(node, deep=True))

def construct_numpy_array(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> np.array:
    return np.array(**loader.construct_mapping(node, deep=True))

def construct_numpy_ndarray(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> np.ndarray:
    return np.ndarray(**loader.construct_mapping(node, deep=True))

def construct_pandas_timestamp(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> pd.Timestamp:
    return pd.Timestamp(loader.construct_scalar(node))

def construct_pandas_datetime_series(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> pd.Series:
    mapping = loader.construct_mapping(node, deep=True)
    return pd.Series(pd.to_datetime(mapping['data']))

class ListOfDf(list):
    """
    Thin list wrapper used as a dispatch key in parametrize_params for
    expected results that are a list of DataFrames (e.g. split_dataframe output).
    """
    pass


def construct_list_of_dataframes(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> ListOfDf:
    items = loader.construct_sequence(node, deep=True)
    return ListOfDf(items)


yaml.add_constructor('!pd.RangeIndex', construct_pandas_index)
yaml.add_constructor('!pd.Series', construct_pandas_series)
yaml.add_constructor('!pd.DataFrame', construct_pandas_dataframe)
yaml.add_constructor('!np.array', construct_numpy_array)
yaml.add_constructor('!np.ndarray', construct_numpy_ndarray)
yaml.add_constructor('!pd.Timestamp', construct_pandas_timestamp)
yaml.add_constructor('!pd.DatetimeSeries', construct_pandas_datetime_series)
yaml.add_constructor('!list_of_df', construct_list_of_dataframes)


def assertNone(a):
    assert a is None

def assertEq(a, b):
    assert a == b

def assertListNanEq(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(y, float) and math.isnan(y):
            assert isinstance(x, float) and math.isnan(x)
        else:
            assert x == y


def assertListDfEq(a: list, b: list):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(
            x.reset_index(drop=True),
            y.reset_index(drop=True),
        )


def _list_has_nan(lst: list) -> bool:
    return any(isinstance(v, float) and math.isnan(v) for v in lst)


def parametrize_params(caller_file: str, subfolder: str, file_name: str):
    """
    Load YAML test cases from `subfolder/file_name` relative to the calling test file's directory.

    :param caller_file: pass `__file__` from the calling test module
    :param subfolder: path relative to the test module, e.g. 'test_cases/arrays'
    :param file_name: YAML file name, e.g. 'ma_eff.yaml'
    :return: list of (input_params, expected_result, fun, exp_err) tuples for pytest.mark.parametrize
    """

    pth = os.path.join(os.path.abspath(os.path.join(caller_file, "..")), subfolder, file_name)
    params = []

    fun_dict = {
        pd.DataFrame: lambda result, expected_result: pd.testing.assert_frame_equal(result, expected_result),
        pd.Series: lambda result, expected_result: pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False),
        np.ndarray: lambda result, expected_result: np.testing.assert_array_equal(result, expected_result),
        pd.Timestamp: lambda result, expected_result: assertEq(result, expected_result),
        ListOfDf: lambda result, expected_result: assertListDfEq(result, expected_result),
    }

    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        for item in yaml_data:
            if 'testing case' in item['description']:
                input_params = item['params']
                expected_result = item['expected_result']

                if type(expected_result) in fun_dict:
                    fun = fun_dict[type(expected_result)]
                elif expected_result is None:
                    fun = lambda result, expected_result: assertNone(result)
                elif isinstance(expected_result, list) and _list_has_nan(expected_result):
                    fun = lambda result, expected_result: assertListNanEq(result, expected_result)
                else:
                    fun = lambda result, expected_result: assertEq(result, expected_result)

                if item['exp_err']:
                    exp_err = pytest.raises(eval(item['exp_err']))
                else:
                    exp_err = nullcontext()

                params.append((input_params, expected_result, fun, exp_err))

    return params
