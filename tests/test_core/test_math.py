import pandas as pd
import pytest

from fermatrica_utils.math import weighted_mean_group, round_up_to_odd
from tests.basics.basics import parametrize_params



class TestRoundUpToOdd:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/math', 'round_up_to_odd.yaml'))
    def test_round_up_to_odd(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = round_up_to_odd(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestWeightedMeanGroup:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/math', 'weighted_mean_group.yaml'))
    def test_weighted_mean_group(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = weighted_mean_group(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(float(result), expected_result)

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/math', 'weighted_mean_group_by.yaml'))
    def test_weighted_mean_group_by(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = weighted_mean_group(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)
