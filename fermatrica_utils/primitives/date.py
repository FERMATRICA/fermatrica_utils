"""
Utilities for date manipulation
"""


from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import tarfile
import urllib3

import pandas as pd
from line_profiler_pycharm import profile


@profile
def date_range(start_date: str
               , end_date: str
               , period: str):
    """
    Creates range of dates with `start_date` and `end_date` as starting and ending points respectively
    and `period` as step (could be day, week, month)

    :param start_date: 
    :param end_date: 
    :param period: 
    :return: 
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    period = period.lower()

    rtrn = []

    while start_date <= end_date:
        rtrn.append(start_date)
        if period in ['day', 'days', 'd']:
            start_date += relativedelta(days=1)
        elif period in ['week', 'weeks', 'w']:
            start_date += relativedelta(weeks=1)
        elif period in ['month', 'months', 'm']:
            start_date += relativedelta(months=1)

    return rtrn


def date_to_period(sr: pd.Series
                   , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                   ):
    """
    Converts date to period effectively replacing original date with start date of the period
    (1st of the month, date of the Monday in the week etc.)

    :param sr:
    :param period:
    :return:
    """

    period = period.lower()

    if period in ['day', 'days', 'd']:
        sr = sr.dt.floor(freq='D')
    elif period in ['week', 'weeks', 'w']:
        sr = sr.dt.to_period('W').dt.start_time
    elif period in ['month', 'months', 'm']:
        sr = sr.dt.to_period('M').dt.start_time
    elif period in ['quarter', 'quarters', 'q']:
        sr = sr.dt.to_period('Q').dt.start_time
    elif period in ['year', 'years', 'y']:
        sr = sr.dt.to_period('Y').dt.start_time

    return sr


def tzdata_windows(folder: str | None = None
                   , year: int | None = 2022
                   , name: str = "tzdata"
                   , if_overwrite: bool = False):
    """
    Downloads timezone data to the local desktop. Required by pyarrow library to work with dates
    on Windows.

    Use it only if timezone data is not downloaded and saved before to avoid ban for too frequent
    requests. `if_overwrite` param is to overwrite existing data

    :param folder:
    :param year: better point out specific year, e.i. 2022 works fine
    :param name:
    :param if_overwrite:
    :return:
    """

    # get year

    if year is None:
        year = datetime.now().year

    # get paths

    if folder is None:
        folder = os.path.join(os.path.expanduser('~'), "Downloads")

    path_load = os.path.join(folder, "tzdata.tar.gz")

    folder = os.path.join(folder, name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # check if data should be overwritten

    if os.path.exists(os.path.join(folder, "windowsZones.xml")) and not if_overwrite:
        return None

    # load main data

    http = urllib3.PoolManager()

    with open(path_load, "wb") as f:
        rtrn = http.request('GET', f'https://data.iana.org/time-zones/releases/tzdata{year}f.tar.gz').data
        f.write(rtrn)

    # unpack main data and load XML

    tarfile.open(path_load).extractall(folder)

    with open(os.path.join(folder, "windowsZones.xml"), "wb") as f:
        rtrn = http.request('GET', f'https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/windowsZones.xml').data
        f.write(rtrn)
