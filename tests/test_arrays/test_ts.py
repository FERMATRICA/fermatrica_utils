import pandas as pd
import numpy as np
import pytest

import fermatrica_utils.arrays.ts as ts
from fermatrica_utils import FermatricaUError
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


class TestTsLongestInterval:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/ts', 'ts_longest_interval.yaml'))
    def test_ts_longest_interval(self, input_params, expected_result, fun, exp_err, if_debug):
        input_params = dict(input_params)
        df = input_params.pop('df').copy()
        df['date'] = pd.to_datetime(df['date'])

        with exp_err:
            result = ts.ts_longest_interval(df, **input_params)
            result = result.reset_index(drop=True)

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


class TestDateRangeToPeriods:

    def test_month_span_sums_to_one(self):
        res = ts.date_range_to_periods('2026-03-15', '2026-04-10', period='month')
        assert list(res['period']) == ['month'] * 2
        assert res['days'].tolist() == [17, 10]
        assert res['share'].sum() == pytest.approx(1.0)
        assert res['date'].tolist() == [pd.Timestamp('2026-03-01'), pd.Timestamp('2026-04-01')]

    def test_single_day(self):
        res = ts.date_range_to_periods('2026-03-15', '2026-03-15', period='month')
        assert len(res) == 1
        assert res['days'].iloc[0] == 1
        assert res['share'].iloc[0] == 1.0

    def test_multi_year_month(self):
        res = ts.date_range_to_periods('2025-12-15', '2026-02-10', period='month')
        assert res['date'].tolist() == [
            pd.Timestamp('2025-12-01'), pd.Timestamp('2026-01-01'), pd.Timestamp('2026-02-01')]
        assert res['days'].tolist() == [17, 31, 10]
        assert res['share'].sum() == pytest.approx(1.0)

    def test_full_year_month_has_twelve_rows(self):
        res = ts.date_range_to_periods('2026-01-01', '2026-12-31', period='month')
        assert len(res) == 12
        assert res['days'].sum() == 365
        assert res['share'].sum() == pytest.approx(1.0)

    def test_week_anchors_are_mondays(self):
        res = ts.date_range_to_periods('2026-01-15', '2026-02-05', period='week')
        assert all(d.dayofweek == 0 for d in res['date'])
        assert res['days'].sum() == 22
        assert res['share'].sum() == pytest.approx(1.0)

    def test_week_numbers_round_trip_through_percent_w(self):
        res = ts.date_range_to_periods('2025-12-29', '2026-01-31', period='week')
        for d in res['date']:
            w = int(d.strftime('%W'))
            parsed = pd.to_datetime(f'{d.year}-W{w}-1', format='%Y-W%W-%w')
            assert parsed == d, f"week {d} did not round-trip (parsed {parsed})"

    def test_day_granularity(self):
        res = ts.date_range_to_periods('2026-01-01', '2026-01-03', period='day')
        assert len(res) == 3
        assert res['days'].tolist() == [1, 1, 1]
        assert res['share'].tolist() == [1 / 3] * 3

    def test_inverted_range_raises(self):
        with pytest.raises(FermatricaUError):
            ts.date_range_to_periods('2026-02-01', '2026-01-01', period='month')

    def test_unknown_period_raises(self):
        with pytest.raises(FermatricaUError):
            ts.date_range_to_periods('2026-01-01', '2026-01-02', period='quarter')

    def test_leap_year_february(self):
        res = ts.date_range_to_periods('2024-02-01', '2024-02-29', period='month')
        assert len(res) == 1
        assert res['days'].iloc[0] == 29
        assert res['share'].iloc[0] == pytest.approx(1.0)
