import pandas as pd
import numpy as np
import pytest

import fermatrica_utils.arrays.ts as ts
from tests.basics.basics import parametrize_params



class TestMaEff:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'ma_eff.yaml'))
    def test_ma_eff(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.ma_eff(**input_params)
            if not isinstance(result, pd.Series):
                result = pd.Series(result)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestMarEff:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'mar_eff.yaml'))
    def test_mar_eff(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.mar_eff(**input_params)
            if not isinstance(result, pd.Series):
                result = pd.Series(result)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestMalEff:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'mal_eff.yaml'))
    def test_mal_eff(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.mal_eff(**input_params)
            if not isinstance(result, pd.Series):
                result = pd.Series(result)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestSrNa:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'sr_na.yaml'))
    def test_sr_na(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.sr_na(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestNaMa:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'na_ma.yaml'))
    def test_na_ma(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.na_ma(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestNaReplaceInner:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'na_replace_inner.yaml'))
    def test_na_replace_inner(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.na_replace_inner(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestMeanTrim:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'mean_trim.yaml'))
    def test_mean_trim(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.mean_trim(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(np.round(float(result), 5), expected_result)

    def test_mean_trim_series_input_matches_array_input(self, if_debug):
        import pandas as pd
        x_arr = np.array([100.0, 100.0, 100.0, 1.0, 1.0])
        x_sr = pd.Series([100.0, 100.0, 100.0, 1.0, 1.0])
        result_arr = ts.mean_trim(x_arr, threshold_rel=-0.1, window=3)
        result_sr = ts.mean_trim(x_sr, threshold_rel=-0.1, window=3)
        assert np.round(float(result_arr), 5) == np.round(float(result_sr), 5)


class TestTrimNumericMask:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/ts', 'trim_numeric_mask.yaml'))
    def test_trim_numeric_mask(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.trim_numeric_mask(**input_params)
            
            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestNa0:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/ts', 'na_0.yaml'))
    def test_na_0(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = ts.na_0(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
