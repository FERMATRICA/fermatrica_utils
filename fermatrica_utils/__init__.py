"""
FERMATRICA_UTILS

Basic / common / non-specific operations and tasks used by FERMATRICA econometrics framework
"""


from fermatrica_utils.decorators import spinner
from fermatrica_utils.flow import fermatrica_utils_error, FermatricaUError, exec_execute, import_module_from_string
from fermatrica_utils.objects import StableClass, get_size
from fermatrica_utils.os import (listdir_abs, sanitize_string, sanitize_date_string, sanitize_datetime_string
    , sanitize_int, sanitize_bool, sanitize_numeric, secure_path)

from fermatrica_utils.primitives.num import int_to_roman
from fermatrica_utils.primitives.string import like_str, latru_detect, cyrillic_detect, cyrillic_trans, cyrillic_trans_str
from fermatrica_utils.primitives.date import date_to_period, date_range, tzdata_windows

from fermatrica_utils.math import round_up_to_odd, weighted_mean_group
from fermatrica_utils.visual import hex_to_rgb

from fermatrica_utils.arrays.arrays import DotDict, dict_to_df, dict_multifill, pandas_filter_regex, pandas_tree_final_child, \
    select_eff, str_replace_eff, groupby_eff, rm_1_item_groups, step_generator, like_sr, multi_index_destroyer, \
    list_select, list_select_pat, list_unique
from fermatrica_utils.arrays.ts import ma_eff, mar_eff, mal_eff, ts_decompose, na_ma, na_0, na_group_mean, \
    na_replace_inner, sr_na, mean_trim, trim_numeric_mask
from fermatrica_utils.arrays.data import decapitalise_df, columns_normalize_df, extra_spaces_df, excel_df_look, \
    excel_get_cols, excel_get_col_vals, excel_get_row_vals, excel_header_look, excel_sheets_look, excel_header_look_addit, \
    header_row_detect, convert_pyarrow_string, split_dataframe
