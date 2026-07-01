import pandas as pd
import numpy as np
from datequarter import DateQuarter

from fermatrica_utils.primitives.date import date_range, date_to_period
from fermatrica_utils.decorators import spinner


def period_align(ds: pd.DataFrame,
                period_trg: str,
                vars_agg_dict: dict,
                dim_cols: list | tuple | None
                ) -> pd.DataFrame:
    """
    Merge and aggregate data based on specified periods.

    This function processes the input dataset based on the specified period information provided in the 'period_trg' argument.
    It aggregates the data according to the periods and variables specified in 'vars_agg_dict'.
    The 'dim_cols' parameter allows for including additional dimensions in the output.

    :param ds: pandas DataFrame containing the dataset to be processed.
    :param period_trg: Target period for aggregation.
    :param vars_agg_dict: A dictionary mapping column names to aggregation functions.
    :param dim_cols: Optional list or tuple of columns to include in the final output.

    :return: A pandas DataFrame with merged and aggregated data based on the specified periods.
    """

    if "period" not in ds.columns:
        raise "ds (attached dataset) should contain 'period' variable"

    if period_trg in ['month', 'm', 'Month', 'M']:
        period_trg = 'month'
    elif period_trg in ['week', 'w', 'Week', 'W']:
        period_trg = 'week'
    elif period_trg in ['day', 'd', 'Day', 'D']:
        period_trg = 'day'
    elif period_trg in ['quarter', 'q', 'Quarter', 'Q']:
        period_trg = 'quarter'
    elif period_trg in ['year', 'y', 'Year', 'Y']:
        period_trg = 'year'
    else:
        raise "Target period is not recognised"

    rtrn = pd.DataFrame()

    for prd in ds.period.unique():
        
        tmp = ds.loc[ds.period == prd].copy()
        del tmp['period']

        if prd in ['month', 'm', 'Month', 'M']:
            prd = 'month'
        elif prd in ['week', 'w', 'Week', 'W']:
            prd = 'week'
        elif prd in ['day', 'd', 'Day', 'D']:
            prd = 'day'
        elif prd in ['quarter', 'q', 'Quarter', 'Q']:
            prd = 'quarter'
        elif prd in ['year', 'y', 'Year', 'Y']:
            prd = 'year'
        else:
            raise "Source period is not recognised"

        if period_trg == prd:
            cln = ["date"] + list(vars_agg_dict.keys())

            if dim_cols is not None:
                dim_cols = [i for i in dim_cols if i not in ["date", "period"]]
                cln = cln + dim_cols

            tmp = tmp[cln]

            rtrn = pd.concat([rtrn, tmp])

        elif period_trg != 'week':
            if prd != 'week':

                start_date = tmp.date.min()
                end_date = tmp.date.max()

                if prd == 'quarter':
                    end_date = pd.to_datetime((DateQuarter.from_date(end_date) + 1).start_date()) - pd.Timedelta(
                        days=1)
                elif prd == 'month':
                    end_date = end_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                elif prd == 'year':
                    end_date = end_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)

                date_list = pd.DataFrame(date_range(start_date, end_date, period='day'), columns=['date'])

                clnd = pd.DataFrame()
                clnd.loc[:, 'date'] = date_to_period(date_list['date'], period_trg)
                clnd.loc[:, 'period'] = date_to_period(date_list['date'], prd)
                clnd = clnd.drop_duplicates()

                if dim_cols is not None:
                    dim_cols = [i for i in dim_cols if i not in ["date", "period"]]
                    clnd = pd.merge(clnd, tmp[dim_cols].drop_duplicates(), how='cross')

                tmp.rename(columns={"date": "period"}, inplace=True)

                if dim_cols is None:
                    dim_cols = []

                if tmp['period'].dtype == 'timestamp[us][pyarrow]':
                    tmp['period'] = tmp['period'].astype('datetime64[ns]')

                tmp = clnd.merge(tmp, how='left', on=['period'] + dim_cols, sort=True)
                tmp["N"] = tmp.groupby(['period'] + dim_cols)['period'].transform('count')

                keys = np.array(list(vars_agg_dict.keys()))
                mask = np.array([i == "sum" for i in vars_agg_dict.values()])

                for col in keys[mask]:
                    if tmp[col].dtype in ['int64[pyarrow]', 'int64', int]:
                        tmp[col] = tmp[col].astype('double[pyarrow]')

                tmp.loc[:, keys[mask]] = tmp.loc[:, keys[mask]].div(tmp["N"], axis=0)
                tmp = tmp.groupby(['date'] + dim_cols).agg(vars_agg_dict).reset_index()

                rtrn = pd.concat([rtrn, tmp])

            else:
                start_date = tmp.date.min()
                end_date = tmp.date.max() + pd.Timedelta(days=6)

                date_list = pd.DataFrame(date_range(start_date, end_date, period='day'), columns=['date'])

                clnd = pd.DataFrame()
                clnd.loc[:, 'date'] = date_to_period(date_list['date'], 'day')
                clnd.loc[:, 'period'] = date_to_period(date_list['date'], prd)
                clnd = clnd.drop_duplicates()

                if dim_cols is not None:
                    dim_cols = [i for i in dim_cols if i not in ["date", "period"]]
                    clnd = pd.merge(clnd, tmp[dim_cols].drop_duplicates(), how='cross')

                tmp.rename(columns={"date": "period"}, inplace=True)

                if dim_cols is None:
                    dim_cols = []

                if tmp['period'].dtype == 'timestamp[us][pyarrow]':
                    tmp['period'] = tmp['period'].astype('datetime64[ns]')

                tmp = clnd.merge(tmp, how='left', on=['period'] + dim_cols, sort=True)
                tmp["N"] = pd.Series(tmp.groupby(['period'] + dim_cols)['period'].transform('count'),
                                     dtype='double[pyarrow]')

                keys = np.array(list(vars_agg_dict.keys()))
                mask = np.array([i == "sum" for i in vars_agg_dict.values()])

                for col in keys[mask]:
                    if tmp[col].dtype in ['int64[pyarrow]', 'int64', int]:
                        tmp[col] = tmp[col].astype('double[pyarrow]')

                tmp.loc[:, keys[mask]] = tmp.loc[:, keys[mask]].div(tmp["N"], axis=0)

                tmp.loc[:, "date"] = date_to_period(tmp['date'], period_trg)
                tmp = tmp.groupby(['date'] + dim_cols).agg(vars_agg_dict).reset_index()

                rtrn = pd.concat([rtrn, tmp])

        else:
            start_date = tmp.date.min()
            end_date = tmp.date.max()
            
            if prd == 'quarter' or period_trg == 'quarter':
                end_date = pd.to_datetime((DateQuarter.from_date(end_date) + 1).start_date()) - pd.Timedelta(
                    days=1)
            elif prd == 'month' or period_trg == 'month':
                end_date = end_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            elif prd == 'year' or period_trg == 'year':
                end_date = end_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)

            date_list = pd.DataFrame(date_range(start_date, end_date, period='day'), columns=['date'])

            clnd = pd.DataFrame()
            clnd.loc[:, 'date'] = date_list['date']
            clnd.loc[:, 'period'] = date_to_period(date_list['date'], prd)
            clnd = clnd.drop_duplicates()

            if dim_cols is not None:
                dim_cols = [i for i in dim_cols if i not in ["date", "period"]]
                clnd = pd.merge(clnd, tmp[dim_cols].drop_duplicates(), how='cross')

            tmp.rename(columns={"date": "period"}, inplace=True)

            if dim_cols is None:
                dim_cols = []

            if tmp['period'].dtype == 'timestamp[us][pyarrow]':
                tmp['period'] = tmp['period'].astype('datetime64[ns]')

            tmp = clnd.merge(tmp, how='left', on=['period'] + dim_cols, sort=True)
            tmp["N"] = tmp.groupby(['period'] + dim_cols)['period'].transform('count')

            keys = np.array(list(vars_agg_dict.keys()))
            mask = np.array([i == "sum" for i in vars_agg_dict.values()])

            for col in keys[mask]:
                if tmp[col].dtype in ['int64[pyarrow]', 'int64', int]:
                    tmp[col] = tmp[col].astype('double[pyarrow]')

            tmp.loc[:, keys[mask]] = tmp.loc[:, keys[mask]].div(tmp["N"], axis=0)

            tmp.loc[:, "date"] = date_to_period(tmp['date'], "week")
            tmp = tmp.groupby(['date'] + dim_cols).agg(vars_agg_dict).reset_index()

            rtrn = pd.concat([rtrn, tmp])

    if rtrn['date'].dtype != 'timestamp[us][pyarrow]':
        rtrn['date'] = rtrn['date'].astype('timestamp[us][pyarrow]')

    return rtrn


def ds_merge(ttl: pd.DataFrame,
            ds: pd.DataFrame,
            period_trg: str,
            vars_agg_dict: dict,
            vars_group_list: list | tuple | None = None
            ) -> pd.DataFrame:
    """
    Most generalized version of merging main DS with additional DS
    Use for basic merge when specific transformations of any kind are not expected

    :param ttl: The main pandas DataFrame to merge the additional dataset with.
    :param ds: The additional pandas DataFrame to be merged with the main dataset.
    :param period_trg: Target period for aggregation.
    :param vars_agg_dict: A dictionary mapping column names to aggregation functions for the merge.
    :param vars_group_list: Optional list or tuple of variables to group by during the merge. Default is ['date'].

    :return: A pandas DataFrame resulting from the merge of 'ttl' and 'ds' based on the specified variables and rules.
    """

    if vars_group_list is None:
        vars_group_list = ['date']
    if 'date' not in vars_group_list:
        vars_group_list = vars_group_list + ['date']

    dim_cols = list(set(vars_group_list) - {'date'})
    if len(dim_cols) == 0:
        dim_cols = None

    tmp = period_align(ds, period_trg, vars_agg_dict, dim_cols=dim_cols)

    if ttl['date'].dtype == 'timestamp[us][pyarrow]' and tmp['date'].dtype != 'timestamp[us][pyarrow]':
        tmp['date'] = tmp['date'].astype('timestamp[us][pyarrow]')
    elif ttl['date'].dtype == 'datetime64[ns]' and tmp['date'].dtype != 'datetime64[ns]':
        tmp['date'] = tmp['date'].astype('datetime64[ns]')

    if 'period' in tmp.columns.to_list() and 'period' in ttl.columns.to_list():
        del tmp['period']

    ttl = ttl.merge(tmp, on=vars_group_list, how='left', sort=False)

    return ttl
