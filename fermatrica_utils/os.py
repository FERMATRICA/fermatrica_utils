"""
I/O utilities.

-----------------------------------------------

The function `_secure_filename` has the specific license, different form the license of the other source code of the file
and of the whole project. The function `_secure_filename` is adopted with some modification from `werkzeug` library
and is redistributed under BSD-3-Clause License:

"
Copyright 2007 Pallets

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"
"""


from datetime import datetime
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import unicodedata


_windows_device_files = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(10)),
    *(f"LPT{i}" for i in range(10)),
}


def listdir_abs(dir_path: str) -> list:
    """
    Get full (absolute) paths of all files in the directory including subdirectories

    :param dir_path: path to the directory to be listed
    :return: the list of full (absolute) paths
    """

    lst = []
    for fld, flds, filenames in os.walk(dir_path):
        for file in filenames:
            lst.append(os.path.join(os.path.abspath(fld), file))

    return lst


def _secure_filename(filename: str
                     , allowed_pattern: str = r"[^A-Za-zа-яА-Я0-9_.-]") -> str:
    r"""Pass it a filename and it will return a secure version of it.  This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`.

    On Windows systems the function also makes sure that the file is not
    named after one of the special device files.

    >>> _secure_filename("My cool movie.mov")
    'My_cool_movie.mov'
    >>> _secure_filename("../../../etc/passwd")
    'etc_passwd'
    >>> _secure_filename('i contain cool \xfcml\xe4uts.txt')
    'i_contain_cool_umlauts.txt'

    The function might return an empty filename.  It's your responsibility
    to ensure that the filename is unique and that you abort or
    generate a random filename if the function returned an empty one.

    .. versionadded:: 0.5

    Adopted from https://github.com/pallets/werkzeug/blob/main/src/werkzeug/utils.py
    allowed_pattern added to allow cyrillic alphabet (or other symbols if neccessary)

    :param filename: the filename to secure
    :param allowed_pattern: regex pattern describing symbols to be accepted as safe
    """

    allowed_pattern = re.compile(allowed_pattern)

    filename = unicodedata.normalize("NFKD", filename).encode('utf-8', 'ignore').decode("utf8")
    # filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(allowed_pattern.sub("", "_".join(filename.split()))).strip(
        "._"
    )

    # on nt a couple of special files are present in each folder.  We
    # have to ensure that the target file is not such a filename.  In
    # this case we prepend an underline
    if (
        os.name == "nt"
        and filename
        and filename.split(".")[0].upper() in _windows_device_files
    ):
        filename = f"_{filename}"

    return filename


def secure_path(path: str
                , remove_sep: bool = False) -> str:
    """
    Sanitize path to the file or the directory. Use it when some path is recieved from the untrusted user,
    e.g. web user. For dashboards and other user apps mostly

    :param path:
    :param remove_sep: if separators between path elements to be removed (`/` and r`\`)
    :return:
    """

    if remove_sep:
        path = _secure_filename(path)
    else:
        path = Path(path).parts
        path = [_secure_filename(x) for x in path]
        path = os.path.join(*[x for x in path if x != ''])

    return path


def sanitize_string(val):

    if not isinstance(val, (str, int, float, bool)):
        val = ''
    else:
        val = str(val)
        val = val.encode('utf-8', 'ignore').decode("utf8")

    return val


def sanitize_numeric(val) -> float:
    """
    Sanitize input known to be numeric (if both float and integer are allowed). Returns float

    :param val:
    :return:
    """

    if not isinstance(val, (str, int, float, bool)):
        val = 0.0
    else:
        try:
            val = float(val)
        except ValueError:
            val = 0.0

    if pd.isna(val) or val == np.inf:
        val = 0.0

    return val


def sanitize_int(val) -> int:
    """
    Sanitize input known to be integer

    :param val:
    :return:
    """

    if not isinstance(val, (str, int, float, bool)):
        val = 0
    else:
        try:
            val = int(val)
        except ValueError:
            val = 0

    if pd.isna(val) or val == np.inf:
        val = 0

    return val


def sanitize_bool(val) -> bool:
    """
    Sanitize input known to be boolean

    :param val:
    :return:
    """

    if not isinstance(val, (str, int, float, bool)):
        val = False
    else:
        try:
            val = bool(val)
        except ValueError:
            val = False

    return val


def sanitize_date_string(val
                         , date_format: str = '%Y-%m-%dT%H:%M:%S'):
    """
    Sanitize input known to be date from string. Check date format codes here:
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

    :param val:
    :param date_format: string representing date format according to C standard
    :return:
    """

    if not isinstance(val, (str, int, float, bool)):
        val = datetime.strptime('1970-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
    else:
        val = str(val)
        val = sanitize_string(val)

        try:
            val = datetime.strptime(val, date_format).date()
        except ValueError:
            val = datetime.strptime('1970-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')

    return val


def sanitize_datetime_string(val
                             , date_format: str = '%Y-%m-%dT%H:%M:%S'):
    """
    Sanitize input known to be date from string. Check date format codes here:
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

    :param val:
    :param date_format: string representing date format according to C standard
    :return:
    """

    if not isinstance(val, (str, int, float, bool)):
        val = datetime.strptime('1970-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
    else:
        val = str(val)
        val = sanitize_string(val)

        try:
            val = datetime.strptime(val, date_format)
        except ValueError:
            val = datetime.strptime('1970-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')

    return val

