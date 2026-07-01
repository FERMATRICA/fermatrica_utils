import pytest
import pandas as pd


@pytest.fixture(scope='session')
def excel_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('excel')

    with pd.ExcelWriter(str(d / 'sample_a.xlsx'), engine='openpyxl') as writer:
        pd.DataFrame({'alpha': [1, 2, 3], 'beta': [10, 20, 30]}).to_excel(
            writer, sheet_name='Sheet1', index=False)
        pd.DataFrame({'alpha': [4, 5], 'gamma': [100, 200]}).to_excel(
            writer, sheet_name='Data', index=False)

    with pd.ExcelWriter(str(d / 'sample_b.xlsx'), engine='openpyxl') as writer:
        pd.DataFrame({'alpha': [6, 7], 'delta': [1000, 2000]}).to_excel(
            writer, sheet_name='Sheet1', index=False)
        pd.DataFrame({'alpha': [8], 'epsilon': [99]}).to_excel(
            writer, sheet_name='Report', index=False)

    return d
