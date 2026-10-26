"""
Time series utilities: moving average, decomposition etc.
"""


import copy
import numpy as np
import pandas as pd
import pandas._libs.missing as libmissing
import re
from typing import List

from packaging import version
import statsmodels.tsa

import fermatrica_utils
from fermatrica_utils.flow import fermatrica_warner, fermatrica_utils_error
from fermatrica_utils.arrays.arrays import groupby_eff
from fermatrica_utils.primitives.date import date_to_period, date_range


"""
Time series decomposition in different ways
"""


def date_range_to_periods(date_start: str | pd.Timestamp,
                          date_end: str | pd.Timestamp,
                          period: str = 'month'  # 'month' | 'week' | 'day'
                          ) -> pd.DataFrame:
    """
    Split an inclusive date interval into per-period day counts and shares.

    Returns one row per calendar period overlapping ``[date_start, date_end]``:
    ``date`` — the period anchor (1st of month / Monday of the week / the day
    itself), ``period`` — the granularity string, ``days`` — the number of
    calendar days of the interval falling in that period, ``share`` — that
    count divided by the total interval length (rows sum to 1.0).

    Week anchors are Mondays and the emitted dates round-trip through the
    ``%Y-W%W-%w`` parsing used by ``fermatrica_rep.options.define``.

    :param date_start: first date of the interval (inclusive)
    :param date_end: last date of the interval (inclusive)
    :param period: 'month' (default), 'week' or 'day'
    :return: DataFrame with columns ``date``, ``period``, ``days``, ``share``
    :raises FermatricaUError: when ``date_end`` precedes ``date_start`` or the
        ``period`` is not one of 'month', 'week', 'day'
    """

    period = period.lower()

    if period not in ('month', 'week', 'day'):
        fermatrica_utils_error(
            f"date_range_to_periods: unknown period '{period}'. "
            "Use 'month', 'week' or 'day'"
        )

    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)

    if date_end < date_start:
        fermatrica_utils_error(
            f"date_range_to_periods: date_end {date_end.date()} "
            f"precedes date_start {date_start.date()}"
        )

    total_days = (date_end - date_start).days + 1

    if period == 'day':
        anchors = date_range(date_start, date_end, 'day')
        offset = pd.DateOffset(days=1)
    elif period == 'month':
        first = date_to_period(pd.Series([date_start]), 'month').iloc[0]
        anchors = date_range(first, date_end, 'month')
        offset = pd.DateOffset(months=1)
    else:  # week
        first = date_to_period(pd.Series([date_start]), 'week').iloc[0]
        anchors = date_range(first, date_end, 'week')
        offset = pd.DateOffset(weeks=1)

    rows = []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        period_end = anchor + offset - pd.Timedelta(days=1)
        ov_start = anchor if anchor > date_start else date_start
        ov_end = period_end if period_end < date_end else date_end
        days_in_period = (ov_end - ov_start).days + 1
        rows.append({
            'date': anchor,
            'period': period,
            'days': days_in_period,
            'share': days_in_period / total_days,
        })

    return pd.DataFrame(rows, columns=['date', 'period', 'days', 'share'])


def ts_decompose(ds: pd.DataFrame
                 , period_var: str
                 , group_vars: list | str
                 , value_var: str
                 , return_comp: str = 'season'
                 , season_type: str = 'multiplicative'
                 , n_fourier: int | None = None) -> pd.DataFrame:
    """
    Decompose time series defined as `period_var` and `value_var` in `ds` dataset in different ways
    and return specific part of the series. Grouping by `group_vars` column(s) is supported

    :param ds:
    :param period_var: 'week', 'month', 'quarter'
    :param group_vars:
    :param value_var:
    :param return_comp: what type of series to be returned: 'season', 'season_fourier' (different algo),
        'trend', 'trend_stl' (different algo), 'random', 'random_stl' (different algo)
    :param season_type: ''multiplicative' or 'additive' (not all algos take into account)
    :param n_fourier: number of fourier components (for 'season_fourier' `return_comp` only)
    :return:
    """

    if isinstance(group_vars, str):
        group_vars = [group_vars]

    # define frequency

    match period_var:
        case 'quarter':
            frq = 4
            season_stl_val=5
            trend_stl_val=7
            ds['period'] = ds.date.dt.quarter
        case 'week':
            frq = 52
            season_stl_val=55
            trend_stl_val=75
            ds['period'] = ds.date.dt.strftime("%V")
        case 'month':
            frq = 12
            season_stl_val=13
            trend_stl_val=17
            ds['period'] = ds.date.dt.month

    if n_fourier is None and return_comp == 'season_fourier':
        n_fourier = int(frq / 2) - 1

    if n_fourier is not None and return_comp == 'season_fourier':
        if n_fourier >= (int(frq / 2) - 1):
            n_fourier = int(frq / 2) - 1

    ds[value_var] = ds[value_var].round(5)
    ds = ds.loc[ds[value_var] != 0]

    clnd = fermatrica_utils.primitives.date.date_range(ds.date.min(), ds.date.max(), period_var)
    clnd = pd.DataFrame(columns=["date"], data=clnd)

    if clnd['date'].dtype != ds['date'].dtype:
        clnd['date'] = clnd['date'].astype(ds['date'].dtype)

    clnd = clnd.merge(ds, on="date", how='left')

    if group_vars[0] in clnd.columns:
        clnd[group_vars[0]] = clnd[group_vars[0]].shift()
        clnd.loc[clnd[group_vars[0]].isna(), "tmp"] = clnd.loc[clnd[group_vars[0]].isna(), "date"]
    else:
        clnd["_grp_marker"] = np.where(clnd[value_var].notna(), 1, np.nan)
        clnd["_grp_marker"] = clnd["_grp_marker"].shift()
        clnd.loc[clnd["_grp_marker"].isna(), "tmp"] = clnd.loc[clnd["_grp_marker"].isna(), "date"]
        del clnd["_grp_marker"]
    
    clnd["tmp"] = clnd["tmp"].ffill()
    gaps_grp = clnd.groupby("tmp")["tmp"].count()
    longest_interval = gaps_grp.loc[gaps_grp == gaps_grp.max()].index[0]
    longest_interval = clnd[(clnd['tmp'] == longest_interval) & ~(clnd[value_var].isna())]

    if len(longest_interval) >= frq * 2:
        ts_t = longest_interval[value_var]
        if return_comp == "season":
            longest_interval['ssn'] = statsmodels.tsa.seasonal.seasonal_decompose(ts_t, model=season_type,
                                                                                  period=frq).seasonal.values
            ds = pd.merge(ds, longest_interval.groupby(['period'], as_index=False)['ssn'].mean(),
                          on='period',
                          how='left')
        elif return_comp == "trend":
            longest_interval['ssn'] = statsmodels.tsa.seasonal.seasonal_decompose(ts_t, model=season_type,
                                                                                  period=frq).trend.values
            ds = pd.merge(ds, longest_interval[['date', 'ssn']], on=['date'], how='left')
            ds['ssn'] = na_ma(ds['ssn'])
        elif return_comp == "trend_stl":
            longest_interval['ssn'] = statsmodels.tsa.seasonal.STL(ts_t, seasonal=season_stl_val, trend=trend_stl_val,
                                                                   period=frq).fit().trend.values
            ds = pd.merge(ds, longest_interval[['date', 'ssn']], on=['date'], how='left')
            ds['ssn'] = na_ma(ds['ssn'])
        elif return_comp == "random_stl":
            longest_interval['ssn'] = statsmodels.tsa.seasonal.STL(ts_t, seasonal=season_stl_val, trend=trend_stl_val,
                                                                   period=frq).fit().resid.values
            ds = pd.merge(ds, longest_interval[['date', 'ssn']], on=['date'], how='left')
            ds['ssn'] = na_ma(ds['ssn'])
        elif return_comp == "random":
            longest_interval['ssn'] = statsmodels.tsa.seasonal.seasonal_decompose(ts_t, model=season_type,
                                                                                  period=frq).resid.values
            ds = pd.merge(ds, longest_interval[['date', 'ssn']], on=['date'], how='left')
            ds['ssn'] = na_ma(ds['ssn'])

        elif return_comp == "season_fourier":

            ts_t_mult = ts_t / ma_eff(ts_t, frq) - 1

            A, b = _create_fourier_matrix(frq, len(ts_t_mult), ts_t_mult, n_fourier)
            betta = np.linalg.solve(A, b)

            longest_interval["ssn"] = _create_fourier(n_fourier, frq, betta, len(ts_t_mult)) + 1

            ds = pd.merge(ds, longest_interval.groupby(['period'], as_index=False)['ssn'].mean(),
                          on='period',
                          how='left')

    return ds


def _create_fourier(k,
                    T1,
                    betta,
                    N) -> np.ndarray:
    """
    Fourier worker function

    :param k:
    :param T1:
    :param betta:
    :param N:
    :return:
    """

    ii = np.array([r for r in range(N)])
    x = np.zeros((N, 2 * k))

    for j in range(2 * k):
        if j % 2 == 0:
            x[:, j] = np.cos(2 * np.pi * (j / 2) * ii / T1)
        else:
            x[:, j] = np.sin(2 * np.pi * ((j + 1) / 2) * ii / T1)

    x = np.matmul(x, betta)

    return x


def _create_fourier_matrix(T1,
                           N,
                           y,
                           k) -> tuple:
    """
    Fourier worker function

    :param T1:
    :param N:
    :param y:
    :param k:
    :return:
    """

    ii = np.array([r for r in range(1, N + 1)])
    A = np.zeros((2 * k, 2 * k))
    b = np.zeros((2 * k, 1))

    for j in range(2 * k):
        if j % 2 != 0:
            for l in range(2 * k):
                if l % 2 != 0:
                    A[j, l] = sum(
                        np.cos(2 * np.pi * ii * ((l + 1) / 2) / T1) * np.cos(2 * np.pi * ii * ((j + 1) / 2) / T1))
                else:
                    A[j, l] = sum(
                        np.sin(2 * np.pi * ii * ((l + 2) / 2) / T1) * np.cos(2 * np.pi * ii * ((j + 1) / 2) / T1))
        else:
            for l in range(2 * k):
                if l % 2 != 0:
                    A[j, l] = sum(
                        np.cos(2 * np.pi * ii * ((l + 1) / 2) / T1) * np.sin(2 * np.pi * ii * ((j + 2) / 2) / T1))
                else:
                    A[j, l] = sum(
                        np.sin(2 * np.pi * ii * ((l + 2) / 2) / T1) * np.sin(2 * np.pi * ii * ((j + 2) / 2) / T1))

        if j % 2 != 0:
            b[j] = sum(y * np.cos(2 * np.pi * ii * (j / 2) / T1))
        else:
            b[j] = sum(y * np.sin(2 * np.pi * ii * ((j + 2) / 2) / T1))

    return A, b


"""
Moving averages with floating window: instead of keeping trailing NAs replace them with
continuously decreasing window at one or both ends of the series
"""


def ma_eff(sr: pd.Series | np.ndarray | list
           , n: int | float):
    """
    Right moving average with floating window, performance efficient

    :param sr:
    :param n:
    :return:
    """

    if_series = True
    if not isinstance(sr, pd.Series):
        sr = pd.Series(sr)
        if_series = False

    if isinstance(n, float):
        n = int(round(n))

    if n == 0:
        return sr.array

    rtrn = sr.rolling(window=n, min_periods=1, center=True).mean()

    if not if_series and isinstance(rtrn, pd.Series):
        rtrn = rtrn.array

    return rtrn


def mar_eff(sr: pd.Series
            , n: int | float) -> np.ndarray | pd.api.extensions.ExtensionArray:
    """
    Right moving average with floating window, performance efficient

    :param sr:
    :param n:
    :return:
    """

    if_series = True
    if not isinstance(sr, pd.Series):
        sr = pd.Series(sr)
        if_series = False

    if isinstance(n, float):
        n = int(round(n))

    if n == 0:
        if not if_series:
            return sr.array
        else:
            return sr

    rtrn = sr.rolling(window=n, min_periods=1, center=False).mean()

    if not if_series and isinstance(rtrn, pd.Series):
        rtrn = rtrn.array

    return rtrn


def mal_eff(sr: pd.Series
            , n: int | float) -> np.ndarray | pd.api.extensions.ExtensionArray:
    """
    Left moving average with floating window, performance efficient

    :param sr:
    :param n:
    :return:
    """

    if_series = True
    if not isinstance(sr, pd.Series):
        sr = pd.Series(sr)
        if_series = False

    if isinstance(n, float):
        n = int(round(n))

    if n == 0:
        if not if_series:
            return sr.array
        else:
            return sr

    rtrn = sr[::-1].rolling(window=n, min_periods=1, center=False).mean()[::-1]

    if not if_series and isinstance(rtrn, pd.Series):
        rtrn = rtrn.array

    return rtrn


"""
Manipulate with NA / missing / outlier values: fill, trim etc.
"""


def sr_na(sr: pd.Series
          , inf_as_na: bool = False):
    """
    Workaround for NA detection in pyarrow pd.Series. Switch to standard implementation
    when pandas fixes the issue (just change hard-coded version number below).

    Returns mask

    :param sr:
    :param inf_as_na:
    :return:
    """

    pd_version = pd.__version__

    if version.parse(pd_version) < version.parse('3.0'):

        if not isinstance(sr, np.ndarray):
            rtrn = sr.to_numpy()
        else:
            rtrn = copy.deepcopy(sr)

        rtrn = libmissing.isnaobj(rtrn, inf_as_na=inf_as_na)
        if isinstance(sr, pd.Series):
            rtrn = sr._constructor(rtrn, index=sr.index, name=sr.name, copy=False)
    else:
        rtrn = pd.isna(sr)

    return rtrn


def na_0(ds: pd.DataFrame
         , ptrn: str
         , replace_val: str | int | float = 0.0) -> pd.DataFrame:
    """
    Fill NA values with 0 or other replacement in all columns with column names filtered by ptrn

    :param ds: dataset
    :param ptrn: regex pattern to filter column names
    :param replace_val: replacement value (float 0.0 as default)
    :return:
    """

    cln = [x for x in ds.columns.tolist() if re.search(ptrn, x)]

    for cl in cln:
        if ds[cl].dtype == 'null[pyarrow]':
            ds[cl] = replace_val
        else:
            ds.loc[sr_na(ds[cl]), cl] = replace_val

    return ds


def na_ma(sr: pd.Series,
          window=4):
    """
    Fill NA values with Moving Average with floating window

    :param sr:
    :param window:
    :return:
    """

    if_series = True
    if not isinstance(sr, pd.Series):
        sr = pd.Series(sr)
        if_series = False

    if isinstance(window, float):
        window = int(round(window))

    if window == 0:
        if not if_series:
            return sr.array
        else:
            return sr

    # run

    tmp = sr.fillna(sr.rolling(window=window, min_periods=1, center=False).mean())
    tmp = tmp.loc[::-1].fillna(tmp.loc[::-1].rolling(window=window, min_periods=1, center=False).mean()).loc[::-1]
    tmp = tmp.bfill()
    tmp = tmp.ffill()

    return tmp


def na_group_mean(df: pd.DataFrame,
                  val_col: str,
                  group_cols: List[str]):
    """
    Fill NA values with mean by group

    :param df: df subset with val_col and group_cols
    :param val_col: col to fill na values
    :param group_cols: group to calculate mean
    :return: val_col pd.Series with filled na values
    """

    return df.groupby(group_cols)[val_col].transform(lambda x: x.fillna(x.mean()))


def na_replace_inner(sr: pd.Series | np.ndarray
                     , replace_val: 0):
    """
    Fill NA values inside Series (i.e. trailing NAs to be kept). Use to fill periods
    of temporary decline in sales etc.

    :param sr:
    :param replace_val:
    :return:
    """

    not_na_index = ~sr_na(sr)
    not_na_index = not_na_index[not_na_index].index

    if len(not_na_index) > 0:
        sr[(sr.index >= not_na_index[0]) & (sr.index <= not_na_index[-1]) & (sr_na(sr))] = replace_val

    return sr


def mean_trim(x: pd.Series | np.ndarray
              , threshold_rel: float | int = -.1
              , window: int = 3
              , if_remove_na: bool = True) -> pd.Series | np.ndarray:
    """
    Compute a robust mean of a series by excluding periods of significant decline.
    The series is first smoothed with a right-aligned moving average, then walked
    sequentially: a value is included in the mean only if its relative change from
    the last accepted value is greater than `threshold_rel`. This focuses the mean
    on the stable "active" part of the series, ignoring ramp-up and ramp-down phases.
    
    IMPORTANT: not designed for series with negative values.

    :param x: input 1-D series
    :param threshold_rel: minimum acceptable relative change from the previous accepted
        value. A value of -0.1 means drops larger than 10 % cause the current element
        to be excluded from the mean. More negative values are more permissive.
    :param window: window size for the right-aligned moving average smoothing step
        applied before the filtering walk
    :param if_remove_na: if True (default), NA elements are dropped before processing;
        if False, NAs are replaced with 0.0
    :return: scalar mean of the accepted (non-declining) values
    """

    # cleanse data

    if isinstance(x, pd.Series):
        x = x.to_numpy().astype(float)

    if if_remove_na:
        x = x[~sr_na(x)]
    else:
        x[sr_na(x)] = 0.0

    # smooth series

    if len(x) > (window + 1):
        x = mar_eff(x, window)

    # exclude decreases

    x_mean = [x[0]]

    for i in x[1:]:

        if x_mean[-1] != 0.0:
            x_mean_diff = (i - x_mean[-1]) / x_mean[-1]
        else:
            x_mean_diff = threshold_rel + 1.0

        if x_mean_diff > threshold_rel:
            x_mean = x_mean + [i]

    x_mean = np.array(x_mean).mean()

    return x_mean


def trim_numeric_mask(x: pd.Series | np.ndarray
                      , trim_threshold_rel: float | int = .1
                      , mean_threshold_rel: float | int = -.1
                      , window: int = 3
                      , if_remove_na: bool = True
                      , trim_middle: bool = False) -> pd.Series | np.ndarray:
    """
    Build a boolean mask that marks structurally low values as False. The absolute
    threshold is derived as `mean_trim(x) * trim_threshold_rel`, where `mean_trim`
    returns a robust mean focused on the stable part of the series (see `mean_trim`).

    By default only edge periods are trimmed: starting from index 0 the mask is set
    to False for every consecutive low value until the first value that exceeds the
    threshold; the same walk is repeated from the end of the series inward. This
    makes the function suitable for excluding early launch ramp-ups or end-of-life
    tail-offs (e.g. brand sales before full distribution is reached).

    When `trim_middle=True`, any remaining value below the threshold in the interior
    of the series is also marked False, covering gaps or dips anywhere in the series.

    :param x: input 1-D series
    :param trim_threshold_rel: fraction of the trimmed mean used as the low-value
        threshold. E.g. 0.1 means values below 10 % of the trimmed mean are
        considered low.
    :param mean_threshold_rel: passed to `mean_trim` as `threshold_rel`; controls
        the sensitivity of the reference mean to drops in the series
    :param window: smoothing window forwarded to `mean_trim`
    :param if_remove_na: forwarded to `mean_trim`; governs NA handling before the
        reference mean is computed
    :param trim_middle: if True, low values in positions 1 … len-2 (the interior)
        are also marked False after the edge passes; default False preserves the
        original behaviour
    :return: numpy bool array of the same length as `x`; True where the value is
        at or above the threshold (or in the interior when `trim_middle` is False)
    """

    # cleanse data

    if isinstance(x, pd.Series):
        x = x.to_numpy().astype(float)

    if len(x) <= window:
        fermatrica_warner.warning(f"There is `bs_key` with data series {x} of len = {len(x)}, which is shorter than the smoothing window {window}. Not to be trimmed")
        return np.full(len(x), True)

    x = copy.deepcopy(x)
    x[sr_na(x)] = 0.0

    # create threshold from sophisticated mean and relative threshold

    x_mean = mean_trim(x, threshold_rel=mean_threshold_rel, window=window, if_remove_na=if_remove_na)
    x_threshold = x_mean * trim_threshold_rel

    # create mask array

    rtrn = np.full(len(x), True)

    # fill forward
    # check if first value to be excluded and then walk forward until large enough value is not found

    if x[0] < x_threshold:
        rtrn[0] = False

    for i in range(1, len(rtrn)):
        if x[i] < x_threshold and rtrn[i-1] == False:  # use `==` cause it's numpy.bool_, not Python bool
            rtrn[i] = False

    # fill backward
    # check if last value to be excluded and then walk backward until large enough value is not found

    if x[-1] < x_threshold:
        rtrn[-1] = False

    for i in range(len(rtrn)-2, 1, -1):
        if x[i] < x_threshold and rtrn[i+1] == False:  # use `==` cause it's numpy.bool_, not Python bool
            rtrn[i] = False

    # mark remaining low values in the middle of the series

    if trim_middle:
        for i in range(1, len(rtrn) - 1):
            if x[i] < x_threshold:
                rtrn[i] = False

    return rtrn


def ts_longest_interval(ttl: pd.DataFrame,
                        group_vars: list,
                        value_var: str = "units_ret",
                        value_threshold: float | int = 50
                        ) -> pd.DataFrame:
    """
    Compute per-group length of the longest uninterrupted sequence of valid
    (above-threshold) values.

    The result is a DataFrame with ``[group_vars..., 'NN']`` columns where
    ``NN`` is the maximum number of consecutive time periods in which
    ``value_var`` stays above ``value_threshold``.

    :param ttl: source DataFrame with date, group and value columns
    :param group_vars: list of column names that define a group
    :param value_var: name of the value column that is checked against the threshold
    :param value_threshold: values strictly above this threshold are considered valid
    :return: DataFrame with ``group_vars`` and ``NN``
    """

    ds = groupby_eff(ttl, group_vars=["date"] + group_vars, other_vars=[value_var], as_index=False)[value_var].sum()
    ds = ds.sort_values(group_vars + ["date"])

    ds['_break'] = (ds[value_var] <= value_threshold).astype(int)
    ds['_run_id'] = ds.groupby(group_vars)['_break'].cumsum()
    ds = ds[ds[value_var] > value_threshold].groupby(group_vars + ['_run_id']).size().reset_index(name='NN')
    ds = ds.groupby(group_vars, as_index=False)['NN'].max()

    return ds
