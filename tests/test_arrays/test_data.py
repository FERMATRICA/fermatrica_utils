import re

import pandas as pd
import numpy as np
import pytest

import fermatrica_utils.arrays.data as data
from tests.basics.basics import parametrize_params, ListOfDf


class TestDecapitaliseDF:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'decapitalise_df.yaml'))
    def test_decapitalise_df(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.decapitalise_df(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestExtraSpacesDF:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'extra_spaces_df.yaml'))
    def test_extra_spaces_df(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.extra_spaces_df(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestColumnsNormalizeDF:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'columns_normalize_df.yaml'))
    def test_columns_normalize_df(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.columns_normalize_df(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestSplitDataframe:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'split_dataframe.yaml'))
    def test_split_dataframe(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ListOfDf(data.split_dataframe(**input_params))

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestConvertPyarrowString:

    def test_all_string_cols_converted(self):
        pytest.importorskip('pyarrow')
        ds = pd.DataFrame({'a': pd.array(['foo', 'bar'], dtype='string'), 'b': [1, 2]})
        result = data.convert_pyarrow_string(ds)
        assert result['a'].dtype.storage == 'pyarrow'
        assert result['b'].dtype == np.int64

    def test_ptrn_incl_limits_conversion(self):
        pytest.importorskip('pyarrow')
        ds = pd.DataFrame({
            'keep_col': pd.array(['a', 'b'], dtype='string'),
            'skip_col': pd.array(['c', 'd'], dtype='string'),
        })
        result = data.convert_pyarrow_string(ds, ptrn_incl='keep')
        assert result['keep_col'].dtype.storage == 'pyarrow'
        assert result['skip_col'].dtype.storage != 'pyarrow'

    def test_ptrn_excl_skips_matching_cols(self):
        pytest.importorskip('pyarrow')
        ds = pd.DataFrame({
            'col_a': pd.array(['x', 'y'], dtype='string'),
            'col_b': pd.array(['p', 'q'], dtype='string'),
        })
        result = data.convert_pyarrow_string(ds, ptrn_excl='col_b')
        assert result['col_a'].dtype.storage == 'pyarrow'
        assert result['col_b'].dtype.storage != 'pyarrow'


class TestPandasForceMaterialize:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'pandas_force_materialize.yaml'))
    def test_pandas_force_materialize(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.pandas_force_materialize(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_empty_dataframe_returned_as_is(self):
        ds = pd.DataFrame()
        result = data.pandas_force_materialize(ds)
        assert result.empty

    def test_none_returned_as_is(self):
        result = data.pandas_force_materialize(None)
        assert result is None


class TestPandasForceMaterializeSeries:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'pandas_force_materialize_series.yaml'))
    def test_pandas_force_materialize_series(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.pandas_force_materialize_series(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_empty_series_returned_as_is(self):
        s = pd.Series([], dtype=float)
        result = data.pandas_force_materialize_series(s)
        assert len(result) == 0


class TestHeaderRowDetect:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/data', 'header_row_detect.yaml'))
    def test_header_row_detect(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = data.header_row_detect(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestExcelSheetsLook:

    def test_returns_all_files_and_sheets(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir))
        assert set(result.keys()) == {'sample_a.xlsx', 'sample_b.xlsx'}
        assert result['sample_a.xlsx'] == ['Sheet1', 'Data']
        assert result['sample_b.xlsx'] == ['Sheet1', 'Report']

    def test_ptrn_incl_exact_match_filters_files(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), ptrn_incl='sample_a.xlsx')
        assert set(result.keys()) == {'sample_a.xlsx'}
        assert 'sample_b.xlsx' not in result

    def test_ptrn_excl_exact_match_filters_files(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), ptrn_excl='sample_b.xlsx')
        assert set(result.keys()) == {'sample_a.xlsx'}

    def test_num_prefixes_sheet_names_with_index(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), ptrn_incl='sample_a.xlsx', num=True)
        assert result['sample_a.xlsx'] == ['0: Sheet1', '1: Data']

    def test_make_df_returns_dataframe(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), make_df=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_if_regex_include_pattern(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), ptrn_incl=r'sample_a', if_regex=True)
        assert set(result.keys()) == {'sample_a.xlsx'}

    def test_if_regex_exclude_pattern(self, excel_dir):
        result = data.excel_sheets_look(str(excel_dir), ptrn_excl=r'sample_b', if_regex=True)
        assert set(result.keys()) == {'sample_a.xlsx'}


class TestExcelHeaderLook:

    def test_returns_stem_keyed_dict(self, excel_dir):
        result = data.excel_header_look(str(excel_dir), 'Sheet1', 2)
        assert 'sample_a' in result
        assert 'sample_b' in result

    def test_row_count_respected(self, excel_dir):
        result = data.excel_header_look(str(excel_dir), 'Sheet1', 2)
        assert result['sample_a'].shape[0] == 2
        assert result['sample_b'].shape[0] == 2

    def test_single_row(self, excel_dir):
        result = data.excel_header_look(str(excel_dir), 'Sheet1', 1)
        assert result['sample_a'].shape[0] == 1

    def test_result_is_dataframe(self, excel_dir):
        result = data.excel_header_look(str(excel_dir), 'Sheet1', 3)
        assert isinstance(result['sample_a'], pd.DataFrame)
        assert isinstance(result['sample_b'], pd.DataFrame)


class TestExcelHeaderLookAddit:

    def test_returns_nested_dict_all_sheets(self, excel_dir):
        result = data.excel_header_look_addit(str(excel_dir), None, 2)
        assert 'sample_a' in result
        assert 'Sheet1' in result['sample_a']
        assert 'Data' in result['sample_a']
        assert 'Sheet1' in result['sample_b']
        assert 'Report' in result['sample_b']

    def test_row_count_respected(self, excel_dir):
        result = data.excel_header_look_addit(str(excel_dir), None, 2)
        assert result['sample_a']['Sheet1'].shape[0] == 2
        assert result['sample_b']['Report'].shape[0] == 2

    def test_filters_to_specified_sheets(self, excel_dir):
        result = data.excel_header_look_addit(str(excel_dir), ['Sheet1'], 2)
        assert 'Data' not in result['sample_a']
        assert 'Report' not in result['sample_b']
        assert 'Sheet1' in result['sample_a']
        assert 'Sheet1' in result['sample_b']

    def test_result_sheets_are_dataframes(self, excel_dir):
        result = data.excel_header_look_addit(str(excel_dir), None, 3)
        assert isinstance(result['sample_a']['Sheet1'], pd.DataFrame)


class TestExcelGetColVals:

    def test_returns_file_stem_keys(self, excel_dir):
        result = data.excel_get_col_vals(str(excel_dir), 'alpha', 0)
        assert set(result.keys()) == {'sample_a', 'sample_b'}

    def test_collects_values_across_sheets(self, excel_dir):
        result = data.excel_get_col_vals(str(excel_dir), 'alpha', 0)
        assert set(result['sample_a']) == {1, 2, 3, 4, 5}
        assert set(result['sample_b']) == {6, 7, 8}

    def test_result_values_are_list(self, excel_dir):
        result = data.excel_get_col_vals(str(excel_dir), 'alpha', 0)
        assert isinstance(result['sample_a'], list)
        assert isinstance(result['sample_b'], list)


class TestExcelGetRowVals:

    def test_returns_dict_by_default(self, excel_dir):
        result = data.excel_get_row_vals(str(excel_dir), 0, 1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'sample_a', 'sample_b'}

    def test_integer_row_collects_values_across_sheets(self, excel_dir):
        result = data.excel_get_row_vals(str(excel_dir), 0, 1)
        # row 1 (header=None): Sheet1 → [1, 10], Data → [4, 100]
        assert set(result['sample_a']) == {1, 10, 4, 100}
        # Sheet1 → [6, 1000], Report → [8, 99]
        assert set(result['sample_b']) == {6, 1000, 8, 99}

    def test_join_returns_flat_deduplicated_list(self, excel_dir):
        result = data.excel_get_row_vals(str(excel_dir), 0, 1, join=True)
        assert isinstance(result, list)
        assert set(result) == {1, 10, 4, 100, 6, 1000, 8, 99}

    def test_join_sort_returns_sorted_list(self, excel_dir):
        result = data.excel_get_row_vals(str(excel_dir), 0, 1, join=True, sort=True)
        assert result == sorted(result)
        assert set(result) == {1, 10, 4, 100, 6, 1000, 8, 99}

    def test_origin_false_applies_latin_lowercase(self, excel_dir):
        # row 0 with header=None contains string column names: 'alpha', 'beta', etc.
        result = data.excel_get_row_vals(str(excel_dir), 0, 0, origin=False)
        for val in result['sample_a']:
            assert val == val.lower()
            assert re.sub('[^a-z0-9_]', '', val) == val


class TestExcelGetCols:

    def test_returns_columns_per_sheet_per_file(self, excel_dir):
        result = data.excel_get_cols(str(excel_dir))
        assert set(result.keys()) == {'sample_a', 'sample_b'}
        assert result['sample_a']['Sheet1'] == ['alpha', 'beta']
        assert result['sample_a']['Data'] == ['alpha', 'gamma']
        assert result['sample_b']['Report'] == ['alpha', 'epsilon']

    def test_cl_pat_filters_columns(self, excel_dir):
        result = data.excel_get_cols(str(excel_dir), cl_pat='alpha')
        assert result['sample_a']['Sheet1'] == ['alpha']
        assert result['sample_b']['Sheet1'] == ['alpha']
        assert result['sample_b']['Report'] == ['alpha']

    def test_get_vals_returns_dict_of_values(self, excel_dir):
        result = data.excel_get_cols(str(excel_dir), cl_pat='alpha', get_vals=True)
        assert isinstance(result['sample_a']['Sheet1'], dict)
        assert set(result['sample_a']['Sheet1']['alpha']) == {1, 2, 3}
        assert set(result['sample_b']['Report']['alpha']) == {8}

    def test_get_vals_join_cl_vals_flattens_per_sheet(self, excel_dir):
        result = data.excel_get_cols(str(excel_dir), cl_pat='alpha', get_vals=True, join_cl_vals=True)
        assert isinstance(result['sample_a']['Sheet1'], list)
        assert set(result['sample_a']['Sheet1']) == {1, 2, 3}

    def test_shts_dct_limits_sheets_per_file(self, excel_dir):
        shts_dct = {'sample_a': ['Sheet1'], 'sample_b': ['Report']}
        result = data.excel_get_cols(str(excel_dir), shts_dct=shts_dct)
        assert list(result['sample_a'].keys()) == ['Sheet1']
        assert list(result['sample_b'].keys()) == ['Report']


class TestExcelDfLook:

    def test_returns_dataframes_all_sheets(self, excel_dir):
        result = data.excel_df_look(str(excel_dir))
        assert set(result.keys()) == {'sample_a.xlsx', 'sample_b.xlsx'}
        assert isinstance(result['sample_a.xlsx']['Sheet1'], pd.DataFrame)
        assert isinstance(result['sample_b.xlsx']['Report'], pd.DataFrame)

    def test_sheet1_columns_correct(self, excel_dir):
        result = data.excel_df_look(str(excel_dir), head=0)
        assert list(result['sample_a.xlsx']['Sheet1'].columns) == ['alpha', 'beta']
        assert list(result['sample_a.xlsx']['Data'].columns) == ['alpha', 'gamma']

    def test_row_num_limits_rows(self, excel_dir):
        result = data.excel_df_look(str(excel_dir), row_num=1)
        assert result['sample_a.xlsx']['Sheet1'].shape[0] == 1
        assert result['sample_b.xlsx']['Report'].shape[0] == 1

    def test_shts_dict_limits_to_specified_sheets(self, excel_dir):
        shts_dict = {'sample_a.xlsx': ['Sheet1']}
        result = data.excel_df_look(str(excel_dir), shts_dict=shts_dict)
        assert set(result.keys()) == {'sample_a.xlsx'}
        assert set(result['sample_a.xlsx'].keys()) == {'Sheet1'}

    def test_data_values_correct(self, excel_dir):
        result = data.excel_df_look(str(excel_dir), head=0)
        df = result['sample_a.xlsx']['Data']
        assert list(df.columns) == ['alpha', 'gamma']
        assert len(df) == 2
        assert set(df['alpha']) == {4, 5}
