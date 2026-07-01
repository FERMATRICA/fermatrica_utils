import pytest

from fermatrica_utils.primitives.date import date_range, date_to_period
from tests.basics.basics import parametrize_params


class TestDateRange:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/date', 'date_range.yaml'))
    def test_date_range(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = date_range(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestDateToPeriod:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/date', 'date_to_period.yaml'))
    def test_date_to_period(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = date_to_period(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
