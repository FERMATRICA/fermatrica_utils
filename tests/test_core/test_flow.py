import logging
import re

import pytest

import fermatrica_utils.flow as flow
from fermatrica_utils.flow import _import_module_from_string_worker, ColoredFormatter, fermatrica_warner
from tests.basics.basics import parametrize_params


class TestExecExecute:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/flow', 'exec_execute.yaml'))
    def test_exec_execute(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = flow.exec_execute(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)

    def test_exec_execute_invalid_expression_raises(self):
        import pandas as pd
        ds = pd.DataFrame({'a': [1, 2]})
        with pytest.raises((SyntaxError, NameError)):
            flow.exec_execute(ds, 'undefined_name_xyz')


class TestImportModuleFromString:

    def test_worker_loads_module_with_function(self):
        source = "def add(a, b):\n    return a + b\n"
        module = _import_module_from_string_worker('test_mod_add', source)
        assert hasattr(module, 'add')
        assert module.add(2, 3) == 5

    def test_worker_loads_module_with_constant(self):
        source = "VALUE = 42\n"
        module = _import_module_from_string_worker('test_mod_const', source)
        assert module.VALUE == 42

    def test_worker_accepts_list_of_source_lines(self):
        source = ["def greet(name):\n", "    return 'hello ' + name\n"]
        module = _import_module_from_string_worker('test_mod_greet', source)
        assert module.greet('world') == 'hello world'

    def test_import_module_from_string_nested_name(self):
        source = "RESULT = 'ok'\n"
        flow.import_module_from_string(
            'test_pkg.test_sub',
            source,
            parent_frame_name='tests.test_core.test_flow'
        )
        # The function injects the top-level name into the calling frame's globals
        g = globals()
        assert 'test_pkg' in g
        assert hasattr(g['test_pkg'], 'test_sub')
        assert g['test_pkg'].test_sub.RESULT == 'ok'


def _colored_formatter_format(levelno: int, msg: str) -> str:
    """Helper: create a LogRecord and format it with ColoredFormatter."""
    formatter = ColoredFormatter()
    record = logging.LogRecord(
        name='test', level=levelno, pathname='', lineno=0,
        msg=msg, args=(), exc_info=None
    )
    return formatter.format(record)


class TestColoredFormatter:

    @pytest.mark.parametrize("input_params,expected_result,fun,exp_err",
                             parametrize_params(__file__, 'test_cases/flow', 'colored_formatter.yaml'))
    def test_colored_formatter(self, input_params, expected_result, fun, exp_err, if_debug):
        with exp_err:
            result = _colored_formatter_format(**input_params)

            if if_debug:
                try:
                    fun(result, expected_result)
                except:
                    breakpoint()
            else:
                fun(result, expected_result)


class TestFermatricaWarner:

    def test_warner_is_logger(self):
        assert isinstance(fermatrica_warner, logging.Logger)

    def test_warner_name_pattern(self):
        assert re.match(r'^fermatrica\.[0-9a-f]{8}$', fermatrica_warner.name)

    def test_warner_propagate_false(self):
        assert fermatrica_warner.propagate is False

    def test_warner_has_handler(self):
        assert len(fermatrica_warner.handlers) >= 1

    def test_warner_handler_is_stream_handler(self):
        assert isinstance(fermatrica_warner.handlers[0], logging.StreamHandler)

    def test_warner_handler_formatter_is_colored(self):
        assert isinstance(fermatrica_warner.handlers[0].formatter, ColoredFormatter)
