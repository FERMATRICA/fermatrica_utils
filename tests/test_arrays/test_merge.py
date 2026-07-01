import pandas as pd
import numpy as np
import pytest

import fermatrica_utils.arrays.merge as merge


class TestPeriodAlign:

    def test_same_period_month_passthrough(self):
        ds = pd.DataFrame({
            'period': ['month', 'month'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'value': [10.0, 20.0]
        })
        result = merge.period_align(ds, 'month', {'value': 'sum'}, dim_cols=None)
        result = result.copy()
        result['date'] = result['date'].astype('datetime64[ns]')
        result = result.sort_values('date').reset_index(drop=True)

        expected = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'value': [10.0, 20.0]
        })
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_month_to_day_distributes_evenly(self):
        # January has 31 days; 31.0 distributed evenly → 1.0 per day
        ds = pd.DataFrame({
            'period': ['month'],
            'date': pd.to_datetime(['2023-01-01']),
            'value': [31.0]
        })
        result = merge.period_align(ds, 'day', {'value': 'sum'}, dim_cols=None)
        assert len(result) == 31
        assert abs(result['value'].sum() - 31.0) < 1e-6

    def test_day_to_month_aggregates_sum(self):
        ds = pd.DataFrame({
            'period': ['day', 'day', 'day'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-01-31']),
            'value': [10.0, 20.0, 30.0]
        })
        result = merge.period_align(ds, 'month', {'value': 'sum'}, dim_cols=None)
        assert len(result) == 1
        assert abs(result['value'].iloc[0] - 60.0) < 1e-6

    def test_same_period_with_dim_cols(self):
        ds = pd.DataFrame({
            'period': ['month', 'month', 'month', 'month'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01']),
            'brand': ['A', 'B', 'A', 'B'],
            'value': [10.0, 20.0, 30.0, 40.0]
        })
        result = merge.period_align(ds, 'month', {'value': 'sum'}, dim_cols=['brand'])
        result = result.copy()
        result['date'] = result['date'].astype('datetime64[ns]')
        result = result.sort_values(['date', 'brand']).reset_index(drop=True)

        expected = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01']),
            'brand': ['A', 'B', 'A', 'B'],
            'value': [10.0, 20.0, 30.0, 40.0]
        })
        pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_like=True)

    def test_returns_pyarrow_timestamp_date_column(self):
        ds = pd.DataFrame({
            'period': ['month'],
            'date': pd.to_datetime(['2023-03-01']),
            'value': [5.0]
        })
        result = merge.period_align(ds, 'month', {'value': 'sum'}, dim_cols=None)
        assert str(result['date'].dtype) == 'timestamp[us][pyarrow]'


class TestDsMerge:

    def test_basic_merge_adds_column_to_ttl(self):
        ttl = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01']).astype('datetime64[ns]'),
            'brand': ['A', 'B', 'A'],
            'sales': [100.0, 200.0, 150.0]
        })
        ds = pd.DataFrame({
            'period': ['month', 'month'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'spend': [50.0, 60.0]
        })

        result = merge.ds_merge(ttl, ds, 'month', {'spend': 'sum'})

        assert 'spend' in result.columns
        assert len(result) == len(ttl)
        jan_spend = result.loc[result['date'] == pd.Timestamp('2023-01-01'), 'spend'].unique()
        assert len(jan_spend) == 1
        assert abs(jan_spend[0] - 50.0) < 1e-6

    def test_merge_with_dim_group_column(self):
        ttl = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01']).astype('datetime64[ns]'),
            'brand': ['A', 'B'],
            'sales': [100.0, 200.0]
        })
        ds = pd.DataFrame({
            'period': ['month', 'month'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01']),
            'brand': ['A', 'B'],
            'spend': [10.0, 20.0]
        })

        result = merge.ds_merge(ttl, ds, 'month', {'spend': 'sum'}, vars_group_list=['brand'])

        assert 'spend' in result.columns
        assert len(result) == 2
        spend_a = result.loc[result['brand'] == 'A', 'spend'].iloc[0]
        spend_b = result.loc[result['brand'] == 'B', 'spend'].iloc[0]
        assert abs(spend_a - 10.0) < 1e-6
        assert abs(spend_b - 20.0) < 1e-6

    def test_merge_preserves_ttl_row_count(self):
        ttl = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01']).astype('datetime64[ns]'),
            'value': [1.0, 2.0, 3.0]
        })
        ds = pd.DataFrame({
            'period': ['month', 'month'],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'extra': [100.0, 200.0]
        })

        result = merge.ds_merge(ttl, ds, 'month', {'extra': 'sum'})

        assert len(result) == len(ttl)
        assert 'extra' in result.columns
