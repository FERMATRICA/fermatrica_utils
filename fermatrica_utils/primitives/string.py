"""
Utilities for string manipulation
"""


import numpy as np
import pandas as pd
import re

from line_profiler_pycharm import profile

from fermatrica_utils.arrays.arrays import list_select
from fermatrica_utils.flow import fermatrica_utils_error


cyr_dict_1 = {'а': 'a',
              'б': 'b',
              'в': 'v',
              'г': 'g',
              'д': 'd',
              'е': 'e',
              'ё': 'e',
              'ж': 'zh',
              'з': 'z',
              'и': 'i',
              'й': 'y',
              'к': 'k',
              'л': 'l',
              'м': 'm',
              'н': 'n',
              'о': 'o',
              'п': 'p',
              'р': 'r',
              'с': 's',
              'т': 't',
              'у': 'u',
              'ф': 'f',
              'х': 'h',
              'ц': 'ts',
              'ч': 'ch',
              'ш': 'sh',
              'щ': 'shch',
              'ъ': '',
              'ы': 'y',
              'ь': '',
              'э': 'e',
              'ю': 'iu',
              'я': 'ia'}

cyr_dict_2 = {'а': 'a',
              'б': 'b',
              'в': 'v',
              'г': 'g',
              'д': 'd',
              'е': 'e',
              'ё': 'yo',
              'ж': 'zh',
              'з': 'z',
              'и': 'i',
              'й': 'i',
              'к': 'k',
              'л': 'l',
              'м': 'm',
              'н': 'n',
              'о': 'o',
              'п': 'p',
              'р': 'r',
              'с': 's',
              'т': 't',
              'у': 'u',
              'ф': 'f',
              'х': 'h',
              'ц': 'c',
              'ч': 'ch',
              'ш': 'sh',
              'щ': 'sch',
              'ъ': '',
              'ы': 'y',
              'ь': '',
              'э': 'e',
              'ю': 'u',
              'я': 'ya'}


def like_str(pattern: str
             , string: str) -> bool:
    """
    Effectively alias for `re.search()` with strictly boolean output

    :param pattern:
    :param string:
    :return:
    """

    return bool(re.search(pattern, string))


def cyrillic_trans_str(string: str
                       , cyr_dict: str | dict | None = None):
    """
    Transcribes Cyrillic letters to Latin, string input (not vectorized)

    :param string:
    :param cyr_dict:
    :param inplace:
    :return:
    """

    if isinstance(cyr_dict, str):
        if cyr_dict == 'cyr_dict_1':
            cyr_dict = cyr_dict_1
        elif cyr_dict == 'cyr_dict_2':
            cyr_dict = cyr_dict_2
        else:
            fermatrica_utils_error('Latin-Cyrillic transcription dictionary "' + cyr_dict + '" not found. ' +
                                   'Check dictionary name spelling or pass the full dictionary as the param')

    elif cyr_dict is None or cyr_dict is False:
        cyr_dict = cyr_dict_2

    # run transcription

    string = string.translate(str.maketrans(cyr_dict))

    return string


def cyrillic_trans(sr: pd.Series | pd.Index | list | np.ndarray
                   , cyr_dict: str | dict | None = None) ->(
        np.ndarray | pd.api.extensions.ExtensionArray | pd.Series | pd.Index):
    """
    Transcribes Cyrillic letters to Latin, array input (vectorized)

    :param sr:
    :param cyr_dict:
    :return:
    """

    if isinstance(cyr_dict, str):
        if cyr_dict == 'cyr_dict_1':
            cyr_dict = cyr_dict_1
        elif cyr_dict == 'cyr_dict_2':
            cyr_dict = cyr_dict_2
        else:
            fermatrica_utils_error('Latin-Cyrillic transcription dictionary "' + cyr_dict + '" not found. ' +
                                   'Check dictionary name spelling or pass the full dictionary as the param')

    elif cyr_dict is None or cyr_dict is False:
        cyr_dict = cyr_dict_2

    if_series = True
    if not isinstance(sr, pd.Series) and not isinstance(sr, pd.Index):
        sr = pd.Series(sr)
        if_series = False

    # run transcription

    rtrn = str.maketrans(cyr_dict)
    rtrn = sr.str.translate(rtrn)

    # return

    if not if_series:
        rtrn = rtrn.array

    return rtrn


def latru_detect(string: str
                 , if_verbose: bool = True) -> tuple:
    """
    Detects if string is in Latin, Russian or mixed letters. Useful if Cyrillic-Latin transcription is not an option

    :param string:
    :param if_verbose:
    :return:
    """

    string_spl = list(string)

    string_len = len(string_spl)

    rus_letters = list_select('[^А-Яа-я]', string_spl, False)
    lat_letters = list_select('[^A-Za-z]', string_spl, False)

    if string_len == len(lat_letters):
        lng = 'latin'
    elif string_len == len(rus_letters):
        lng = 'russian'
    else:
        lng = 'mixed'

    if if_verbose:
        print(string + ': language = ' + lng)
        print('latin letters: ' + str(lat_letters))
        print('russian letters: ' + str(rus_letters))

    return lng, lat_letters, rus_letters


@profile
def cyrillic_detect(string: str):
    """
    Checks if string contains Cyrillic letters

    :param string:
    :return:
    """

    return bool(re.search('[а-яА-Я]', string))
