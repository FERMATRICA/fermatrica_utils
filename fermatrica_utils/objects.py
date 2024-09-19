"""
Utilities to work with objects / classes / callables.
For dynamic code utilities check `fermatrica_utils.flow` module

-----------------------------------------------

The function `get_size` has the specific license, different form the license of the other source code of the file
and of the whole project. The function `get_size` is adopted from `bosswissam/pysize` Github repository
and is redistributed under MIT License:

"
MIT License

Copyright (c) [2018] [Wissam Jarjoui]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"
"""


import inspect
import logging
import sys


class StableClass(object):
    """
    Prevent class instance from assigning non-predefined attributes after initialization.
    Call `._init_finish()` method at the end of `.__init__()` method of the derived class
    """

    __if_stable = False

    def __setattr__(self, key, value):
        if self.__if_stable and not hasattr(self, key):
            raise AttributeError("'" + type(self).__name__ + "' object has no attribute ' " + str(key) + "'")
        object.__setattr__(self, key, value)

    def _init_finish(self):
        self.__if_stable = True


def get_size(obj
             , seen: set | None = None):
    """
    Recursively finds object size. The problem is the standard Python function `sys.getsizeof()` doesn't take into account
    sizes of the sub-objects including non-primitive list elements etc. So the recursion or loop is required

    Source: https://github.com/bosswissam/pysize

    :param obj:
    :param seen:
    :return:
    """

    size = sys.getsizeof(obj)

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break

    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum((get_size(i, seen) for i in obj))
        except TypeError:
            logging.exception("Unable to get size of %r. This may lead to incorrect sizes. Please report this error.",
                              obj)

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


