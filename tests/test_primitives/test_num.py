import pytest

from fermatrica_utils.primitives.num import int_to_roman
from tests.basics.basics import parametrize_params



class TestIntToRoman:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/num', 'int_to_roman.yaml'))
    def test_int_to_roman(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = int_to_roman(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_negative_returns_non_empty_string(self):
        # Python divmod(-5, 1000) = (-1, 995), so -5 produces 'CMXCV'
        result = int_to_roman(-5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_int_input_identical_to_equivalent_float(self):
        assert int_to_roman(10) == int_to_roman(10.0)
