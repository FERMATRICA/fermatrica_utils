import pytest
import numpy as np
import pandas as pd

from fermatrica_utils.primitives.string import like_str, cyrillic_trans, latru_detect
from tests.basics.basics import parametrize_params



class TestLikeStr:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/string', 'like_str.yaml'))
    def test_like_str(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = like_str(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestCyrillicTrans:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/string', 'cyrillic_trans.yaml'))
    def test_cyrillic_trans(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = cyrillic_trans(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_cyrillic_trans_list_input_returns_array(self):
        result = cyrillic_trans(['привет', 'мир'])
        assert list(result) == ['privet', 'mir']

    def test_cyrillic_trans_ndarray_input_returns_array(self):
        result = cyrillic_trans(np.array(['привет', 'мир']))
        assert list(result) == ['privet', 'mir']

    def test_cyrillic_trans_pd_index_input_returns_index(self):
        idx = pd.Index(['привет', 'мир'])
        result = cyrillic_trans(idx)
        assert isinstance(result, pd.Index)
        assert list(result) == ['privet', 'mir']


class TestLatruDetect:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err", parametrize_params(__file__, 'test_cases/string', 'latru_detect.yaml'))
    def test_latru_detect(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = latru_detect(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_latru_detect_full_return_returns_tuple_with_lists(self):
        lng, lat_letters, rus_letters = latru_detect('helloмир', full_return=True)
        assert lng == 'mixed'
        assert isinstance(lat_letters, list)
        assert isinstance(rus_letters, list)
        # latin letters includes all non-Cyrillic chars (letters + digits + symbols)
        assert all(c in lat_letters for c in list('hello'))

    def test_latru_detect_full_return_latin_only(self):
        lng, lat_letters, rus_letters = latru_detect('hello', full_return=True)
        assert lng == 'latin'
        # lat_letters = chars matching [^А-Яа-я] (non-Cyrillic) = all 5 chars for pure Latin
        assert len(lat_letters) == 5
        assert 'h' in lat_letters
        # rus_letters = chars matching [^A-Za-z] (non-Latin) = empty for pure Latin
        assert rus_letters == []

    def test_latru_detect_verbose_does_not_raise(self, capsys):
        latru_detect('hello', if_verbose=True)
        captured = capsys.readouterr()
        assert 'latin' in captured.out
