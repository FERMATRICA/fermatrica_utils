"""
Utilities to control flow, mostly dynamic code utils, but also custom exceptions etc.
"""


import importlib.abc
import importlib.util
import inspect
import sys

import pandas as pd
from line_profiler_pycharm import profile


class FermatricaUError(Exception):
    """
    Exception class for errors specific for Fermatrica Utils
    """
    pass


def fermatrica_utils_error(msg):
    """
    Raise FermatrricaUError exception

    :param msg:
    :return:
    """

    # limit traceback to prevent all that exception stack from showing
    sys.tracebacklimit = 1
    raise FermatricaUError(msg)

    pass


@profile
def exec_execute(ds: pd.DataFrame
                 , x_str: str):
    """
    Execute some string as a command (mostly as function call) with dataset `ds` as part of the environment.
    Useful for dynamic code operations with dataset

    :param ds:
    :param x_str:
    :return:
    """

    rtrn = eval(compile(x_str, '<string>', 'eval'), globals(), locals())

    return rtrn


"""
Utilities to import code from string. The string may be some string loaded from disc as plain text file(s) or extracted
from the callable objects via `inspect.getsource()` or something like this.

Main goal is to load it with or within module hierarchy tree to match the original full name (e.g. `code_py.adhoc.model`).
Otherwise functions sensitive to the context they are defined to won't work or even loaded.   
"""


class _StringLoader(importlib.abc.SourceLoader):
    """
    Implements abstract / interface class of "loader" as loader from the source code saved as a string
    """
    def __init__(self, data):
        self.data = data

    def get_source(self, fullname):
        return self.data

    def get_data(self, path):
        return self.data.encode("utf-8")

    def get_filename(self, fullname):
        return "<not a real path>/" + fullname + ".py"


@profile
def _import_module_from_string_worker(name: str
                                      , source: str | list):
    """
    Load specific module and return it as an object

    :param name:
    :param source:
    :return:
    """

    if type(source) is list:
        source = ''.join(source)

    loader = _StringLoader(source)

    spec = importlib.util.spec_from_loader(name, loader, origin="global")
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


@profile
def import_module_from_string(name: str
                              , source: str | list
                              , parent_frame_name: str = '__main__'):
    """
    Load Python module from the code saved as a string including the whole module hierarchy tree.
    Upper level existing modules are preserved, non-existing are being created as empty modules

    :param name: module name including module path, e.g. 'code_py.adhoc.model'
    :param source: string or list or strings (code lines) containing module source code
    :param parent_frame_name: frame (environment, context) to append module hierarchy to be created
    :return:
    """

    name_split = name.split(".")

    # top module

    top_source = ''
    if len(name_split) == 1:
        top_source = source

    call_stack = inspect.stack(0)
    cs_cur = None

    for cs in call_stack:
        fr_name = cs.frame.f_globals['__name__']
        if fr_name == parent_frame_name:
            cs_cur = cs
            break

    top_name = name_split[0]

    if top_name in cs_cur.frame.f_globals.keys():
        top_module = cs_cur.frame.f_globals[top_name]
    else:
        top_module = _import_module_from_string_worker(top_name, top_source)
        cs_cur.frame.f_globals[top_name] = top_module

    if len(name_split) == 1:
        return

    # middle modules

    name_split_iter = name_split[1:-1]

    if len(name_split_iter) > 0:
        for i, nm in enumerate(name_split_iter):

            current_name = '.'.join(name_split[0:(i+2)])

            if current_name in cs_cur.frame.f_globals.keys():
                current_module = cs_cur.frame.f_globals[current_name]
            else:
                current_module = _import_module_from_string_worker(current_name, '')
                setattr(top_module, nm, current_module)

                cs_cur.frame.f_globals[current_name] = current_module

            top_module = current_module

    # bottom module

    main_module = _import_module_from_string_worker(name, source)
    setattr(top_module, name_split[-1], main_module)

    cs_cur.frame.f_globals[name] = main_module

    pass
