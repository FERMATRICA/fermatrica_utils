import pytest

from fermatrica_utils.visual import hex_to_rgb
from tests.basics.basics import parametrize_params


def _load_cases(file_name):
    params = []
    for input_params, expected_result, fun, exp_err in parametrize_params(__file__, 'test_cases/visual', file_name):
        params.append((input_params, expected_result, exp_err))
    return params


class TestHexToRgb:

    @pytest.mark.parametrize("input_params,expected_result,exp_err", _load_cases('hex_to_rgb.yaml'))
    def test_hex_to_rgb(self, input_params, expected_result, exp_err, if_debug):
        with exp_err:
            result = hex_to_rgb(**input_params)

            if if_debug:
                try:
                    assert list(result) == expected_result
                except:
                    breakpoint()
            else:
                assert list(result) == expected_result
