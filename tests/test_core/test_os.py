import os
import math
import numpy as np
from datetime import date, datetime
import pytest

import fermatrica_utils.os as fos


class TestSanitizeString:

    def test_valid_string_passthrough(self):
        assert fos.sanitize_string('hello') == 'hello'

    def test_numeric_input_converted_to_string(self):
        assert fos.sanitize_string(42) == '42'
        assert fos.sanitize_string(3.14) == '3.14'

    def test_none_returns_empty_string(self):
        assert fos.sanitize_string(None) == ''

    def test_list_returns_empty_string(self):
        assert fos.sanitize_string([1, 2, 3]) == ''

    def test_bool_converted_to_string(self):
        assert fos.sanitize_string(True) == 'True'


class TestSanitizeNumeric:

    def test_float_passthrough(self):
        assert fos.sanitize_numeric(3.14) == 3.14

    def test_int_converted_to_float(self):
        assert fos.sanitize_numeric(7) == 7.0

    def test_parseable_string_converted(self):
        assert fos.sanitize_numeric('2.5') == 2.5

    def test_none_returns_zero(self):
        assert fos.sanitize_numeric(None) == 0.0

    def test_nan_returns_zero(self):
        assert fos.sanitize_numeric(float('nan')) == 0.0

    def test_inf_returns_zero(self):
        assert fos.sanitize_numeric(np.inf) == 0.0

    def test_unparseable_string_returns_zero(self):
        assert fos.sanitize_numeric('abc') == 0.0


class TestSanitizeInt:

    def test_int_passthrough(self):
        assert fos.sanitize_int(5) == 5

    def test_float_truncated_to_int(self):
        assert fos.sanitize_int(3.9) == 3

    def test_parseable_string_converted(self):
        assert fos.sanitize_int('10') == 10

    def test_none_returns_zero(self):
        assert fos.sanitize_int(None) == 0

    def test_unparseable_string_returns_zero(self):
        assert fos.sanitize_int('abc') == 0


class TestSanitizeBool:

    def test_true_passthrough(self):
        assert fos.sanitize_bool(True) is True

    def test_false_passthrough(self):
        assert fos.sanitize_bool(False) is False

    def test_int_one_is_truthy(self):
        assert fos.sanitize_bool(1) is True

    def test_int_zero_is_falsy(self):
        assert fos.sanitize_bool(0) is False

    def test_none_returns_false(self):
        assert fos.sanitize_bool(None) is False

    def test_list_returns_false(self):
        assert fos.sanitize_bool([1, 2]) is False


class TestSanitizeDateString:

    def test_valid_date_string_parsed(self):
        result = fos.sanitize_date_string('2023-06-15T00:00:00')
        assert result == date(2023, 6, 15)

    def test_invalid_string_returns_epoch(self):
        result = fos.sanitize_date_string('not-a-date')
        assert isinstance(result, (date, datetime))

    def test_none_returns_epoch(self):
        result = fos.sanitize_date_string(None)
        assert isinstance(result, (date, datetime))

    def test_custom_format(self):
        result = fos.sanitize_date_string('15/06/2023', date_format='%d/%m/%Y')
        assert result == date(2023, 6, 15)


class TestSecurePath:

    def test_path_traversal_neutralized(self):
        result = fos.secure_path('../../etc/passwd')
        assert '..' not in result.split(os.sep)
        assert 'passwd' in result

    def test_normal_path_preserved(self):
        result = fos.secure_path('mydir/myfile.txt')
        assert 'mydir' in result
        assert 'myfile.txt' in result

    def test_remove_sep_flattens_path(self):
        result = fos.secure_path('../../etc/passwd', remove_sep=True)
        assert os.sep not in result
        assert '/' not in result
        assert '..' not in result


class TestListdirAbs:

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / 'a.txt').write_text('a')
        (tmp_path / 'b.txt').write_text('b')
        result = fos.listdir_abs(str(tmp_path))
        assert len(result) == 2
        assert all(os.path.isabs(p) for p in result)

    def test_includes_subdirectory_files(self, tmp_path):
        subdir = tmp_path / 'sub'
        subdir.mkdir()
        (subdir / 'c.txt').write_text('c')
        result = fos.listdir_abs(str(tmp_path))
        assert any('c.txt' in p for p in result)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = fos.listdir_abs(str(tmp_path))
        assert result == []
