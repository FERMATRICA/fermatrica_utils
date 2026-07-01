import pytest

import fermatrica_utils.arrays.arrays as arr
from tests.basics.basics import parametrize_params



class TestStepGenerator:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'step_generator.yaml'))
    def test_step_generator(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = list(arr.step_generator(**input_params))

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestDotDict:

    def test_dot_access(self):
        d = arr.DotDict({'key': 'value', 'num': 42})
        assert d.key == 'value'
        assert d.num == 42

    def test_missing_key_returns_none(self):
        d = arr.DotDict({'key': 'value'})
        assert d.missing is None

    def test_set_via_dot(self):
        d = arr.DotDict()
        d.new_key = 'hello'
        assert d['new_key'] == 'hello'

    def test_delete_via_dot(self):
        d = arr.DotDict({'a': 1})
        del d.a
        assert 'a' not in d

    def test_empty_dict_missing_key_returns_none(self):
        d = arr.DotDict()
        assert d.anything is None

    def test_overwrite_existing_key_via_dot(self):
        d = arr.DotDict({'x': 1})
        d.x = 99
        assert d['x'] == 99

    def test_numeric_and_string_coexist(self):
        d = arr.DotDict({'a': 'hello', 'b': 42, 'c': 3.14})
        assert d.a == 'hello'
        assert d.b == 42
        assert d.c == 3.14

    def test_init_from_keyword_args(self):
        d = arr.DotDict(x=1, y=2)
        assert d.x == 1
        assert d.y == 2


class TestDictMultifill:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'dict_multifill.yaml'))
    def test_dict_multifill(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.dict_multifill(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestSelectEff:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'select_eff.yaml'))
    def test_select_eff(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.select_eff(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestStrReplaceEff:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'str_replace_eff.yaml'))
    def test_str_replace_eff(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.str_replace_eff(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestLikeSr:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'like_sr.yaml'))
    def test_like_sr(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.like_sr(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestListSelect:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'list_select.yaml'))
    def test_list_select(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.list_select(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestRm1ItemGroups:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/arrays', 'rm_1_item_groups.yaml'))
    def test_rm_1_item_groups(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.rm_1_item_groups(**input_params)
            result = result.reset_index(drop=True)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_rm_1_item_groups_inplace_false_does_not_mutate_original(self):
        import pandas as pd
        ds = pd.DataFrame({'brand': ['A', 'A', 'B'], 'value': [1, 2, 3]})
        original_len = len(ds)
        result = arr.rm_1_item_groups(ds, group_var='brand', inplace=False)
        assert len(ds) == original_len
        assert len(result.reset_index(drop=True)) == 2


class TestDictToDf:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/arrays', 'dict_to_df.yaml'))
    def test_dict_to_df(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.dict_to_df(**input_params)
            result = result.reset_index(drop=True)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestGroupbyEff:

    def test_single_group_sum_matches_standard_groupby(self):
        import pandas as pd
        ds = pd.DataFrame({'g': ['a', 'a', 'b'], 'v': [1.0, 2.0, 3.0]})
        result = arr.groupby_eff(ds, group_vars=['g'], other_vars=['v']).sum()
        expected = ds.groupby(['g'])[['v']].sum()
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_group_sum(self):
        import pandas as pd
        ds = pd.DataFrame({
            'g1': ['a', 'a', 'b', 'b'],
            'g2': ['x', 'x', 'y', 'y'],
            'v': [1.0, 2.0, 3.0, 4.0]
        })
        result = arr.groupby_eff(ds, group_vars=['g1', 'g2'], other_vars=['v']).sum()
        expected = ds.groupby(['g1', 'g2'])[['v']].sum()
        pd.testing.assert_frame_equal(result, expected)

    def test_with_boolean_mask_filters_rows(self):
        import pandas as pd
        ds = pd.DataFrame({'g': ['a', 'a', 'b', 'b'], 'v': [1.0, 2.0, 3.0, 4.0]})
        mask = pd.Series([True, True, False, True])
        result = arr.groupby_eff(ds, group_vars=['g'], other_vars=['v'], mask=mask).sum()
        expected = pd.DataFrame({'v': [3.0, 4.0]}, index=pd.Index(['a', 'b'], name='g'))
        pd.testing.assert_frame_equal(result, expected)


class TestMultiIndexDestroyer:

    def test_two_level_columns_joined_with_default_sep(self):
        import pandas as pd
        columns = pd.MultiIndex.from_tuples([('a', 'sum'), ('b', 'mean')])
        ds = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns)
        ds.index = pd.Index(['x', 'y'], name='group')
        result = arr.multi_index_destroyer(ds)
        assert list(result.columns) == ['group', 'a_sum', 'b_mean']
        assert list(result['group']) == ['x', 'y']
        assert list(result['a_sum']) == [1.0, 3.0]

    def test_custom_separator(self):
        import pandas as pd
        columns = pd.MultiIndex.from_tuples([('col', 'agg')])
        ds = pd.DataFrame([[5.0]], columns=columns)
        ds.index = pd.Index(['r'], name='idx')
        result = arr.multi_index_destroyer(ds, sep='|')
        assert 'col|agg' in result.columns


class TestListUnique:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/arrays', 'list_unique.yaml'))
    def test_list_unique(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = arr.list_unique(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
