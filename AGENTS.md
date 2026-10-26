# FERMATRICA_UTILS — AI Instructions

> **⚠️ Maintenance reminder**: If you add new modules, rename directories, change conventions, or modify the project structure, update this file. The AI does not auto-detect staleness. Run `/fermatrica-utils-sync-instructions` to have the AI review and update this file against the current codebase.

## Framework Overview

FERMATRICA_UTILS is the **shared utility layer** of the FERMATRICA econometrics framework. It collects basic, cross-cutting functionality that is not specific to MMM or two-contour time series / panel modeling:

- Object and class utilities (`StableClass`, deep size calculation)
- Flow / dynamic code execution (import from string, custom exceptions)
- I/O and sanitization helpers (secure paths, string/date/bool sanitizers)
- Math helpers (weighted mean, rounding)
- Primitives manipulation (string transliteration, date period conversion, int-to-roman)
- Array operations on lists, pandas DataFrames, and Series (`DotDict`, efficient groupby/select, column cleansing, time series MA/NA/trim, period alignment and merge)
- Visual colour conversion (HEX→RGB)

The functionality is designed to be usable both inside and outside the FERMATRICA framework.

## Multi-Repo Architecture

FERMATRICA is developed across three repositories. This repo (`fermatrica_utils`) provides the **shared utilities** consumed by the other repos.

| Repo | Purpose | Python package |
|------|---------|---------------|
| `fermatrica` (core) | Model definition, transformation, optimization, evaluation | `fermatrica` |
| `fermatrica_utils` (this repo) | Utility functions (`StableClass`, `DotDict`, `groupby_eff`, `select_eff`, sanitizers, date ranges, string ops, time series helpers) | `fermatrica_utils` |
| `fermatrica_rep` | Reporting: decomposition, waterfall charts, PPTX export | `fermatrica_rep` |

## Code Organization (This Repo)

```
fermatrica_utils/
├── flow.py                  # Exceptions, dynamic code execution, coloured logger
├── decorators.py            # @spinner decorator
├── objects.py               # StableClass, get_size (recursive object size)
├── os.py                    # I/O: listdir_abs, sanitize_*, secure_path
├── math.py                  # weighted_mean_group, round_up_to_odd
├── visual.py                # hex_to_rgb
├── profiling.py             # lprof_to_html (.lprof → interactive HTML report)
├── __init__.py              # Public API exports
├── primitives/
│   ├── __init__.py
│   ├── date.py              # date_range, date_to_period, tzdata_windows
│   ├── num.py               # int_to_roman
│   └── string.py            # like_str, latru_detect, cyrillic_trans, cyrillic_detect
└── arrays/
    ├── __init__.py
    ├── arrays.py            # DotDict, groupby_eff, select_eff, list_select, str_replace_eff, etc.
    ├── data.py              # Column cleansing, Excel helpers, pickle compat, materialization
    ├── ts.py                # MA/AR, NA fill, decomposition, Fourier, mean_trim, trim_numeric_mask
    └── merge.py             # period_align, ds_merge
```

**Dependency note**: Unlike the core `fermatrica` repo, there is no tight dependency chain here. Modules are loosely coupled and can be imported independently. The main internal dependency is `arrays/data.py` importing from `arrays/arrays.py` and `primitives/string.py`.

## Local Configuration

Machine-specific settings are stored in `config.toml` (repo root, gitignored, created by each developer):

1. Copy `config_example.toml` to `config.toml`
2. Set `[paths].python` to the Python executable path in your Conda environment
3. Set `[python].env_name` to your Conda environment name

The `[paths].python` field points to the Python interpreter the AI should invoke. When running Python commands (`python ...`, `pytest ...`), always prefix with this path:

```
C:\Users\...\anaconda3\envs\<env_name>\python.exe -m pytest tests/
```

The `[python].env_name` field is informational — the environment name corresponding to the interpreter.

## Subpackage Details

### `fermatrica_utils.flow` — Flow and dynamic code

| Export | Description |
|--------|-------------|
| `FermatricaUError` | Custom exception class for fermatrica_utils errors |
| `fermatrica_utils_error(msg)` | Raise `FermatricaUError` with truncated traceback |
| `fermatrica_warner` | Logger `'fermatrica'` with yellow WARNING-level output |
| `fermatrica_hard_warner` | Logger `'fermatrica_hard'` with orange WARNING-level output |
| `ColoredFormatter` | Logging formatter that renders WARNING in yellow |
| `exec_execute(ds, x_str)` | Evaluate a string as code with dataset `ds` in scope |
| `import_module_from_string(name, source)` | Import a module hierarchy from source code strings |

### `fermatrica_utils.decorators` — Decorators

| Export | Description |
|--------|-------------|
| `@spinner` | Animated spinner for time-consuming functions (PyCharm-compatible) |

### `fermatrica_utils.objects` — Object / class utilities

| Export | Description |
|--------|-------------|
| `StableClass` | Base class — prevents adding new attributes after `_init_finish()` |
| `get_size(obj)` | Recursively compute the total memory size of an object (MIT-licensed, adapted from `bosswissam/pysize`) |

`StableClass` is the foundation of `Model` and `ModelRep` in sibling repos. Call `self._init_finish()` at the end of `__init__` in derived classes.

### `fermatrica_utils.os` — I/O and sanitization

| Export | Description |
|--------|-------------|
| `listdir_abs(dir_path)` | Recursively list absolute paths of all files in a directory |
| `sanitize_string(val)` | Normalise to UTF-8 string, empty on failure |
| `sanitize_numeric(val)` | Convert to float, 0.0 on failure |
| `sanitize_int(val)` | Convert to int, 0 on failure |
| `sanitize_bool(val)` | Convert to bool, `False` on failure |
| `sanitize_date_string(val)` | Parse date string, falls back to `1970-01-01` |
| `sanitize_datetime_string(val)` | Parse datetime string, falls back to epoch |
| `secure_path(path)` | Sanitize file/dir path, optionally remove separators (Web/werkzeug-inspired, BSD-3-Clause) |

### `fermatrica_utils.math` — Math operations

| Export | Description |
|--------|-------------|
| `weighted_mean_group(ds, values, weights, by)` | Grouped weighted mean on a DataFrame |
| `round_up_to_odd(x)` | Round up to the nearest odd integer |

### `fermatrica_utils.visual` — Visual/colour helpers

| Export | Description |
|--------|-------------|
| `hex_to_rgb(hex_code)` | Convert HEX colour string to `(r, g, b)` tuple |

### `fermatrica_utils.profiling` — Line-profiler report rendering

| Export | Description |
|--------|-------------|
| `lprof_to_html(lprof_path, out_path=None, open_browser=False)` | Render a kernprof `.lprof` file into a self-contained interactive HTML report (summary + per-line breakdown with source) |

`line_profiler` is imported lazily inside `lprof_to_html()`; the package does not depend on it unless the function is used.

### `fermatrica_utils.primitives.date` — Date manipulation

| Export | Description |
|--------|-------------|
| `date_range(start_date, end_date, period)` | Generate a list of dates between start and end with day/week/month step |
| `date_to_period(sr, period)` | Floor datetime Series to period start (day, week, month, quarter, year) |
| `tzdata_windows(folder, year)` | Download IANA timezone data for PyArrow on Windows |

### `fermatrica_utils.primitives.num` — Numeric utilities

| Export | Description |
|--------|-------------|
| `int_to_roman(number)` | Convert integer to Roman numerals |

### `fermatrica_utils.primitives.string` — String manipulation

| Export | Description |
|--------|-------------|
| `like_str(pattern, string)` | Regex match with boolean output (alias for `re.search`) |
| `latru_detect(string)` | Detect Latin / Cyrillic / mixed script |
| `cyrillic_trans(sr, cyr_dict)` | Vectorised Cyrillic→Latin transcription (pandas Series/Index/array) |
| `cyrillic_trans_str(string, cyr_dict)` | Scalar Cyrillic→Latin transcription |
| `cyrillic_detect(string)` | Check if string contains any Cyrillic letters |

Two built-in transcription dictionaries: `cyr_dict_1` (simplified) and `cyr_dict_2` (detailed with 'yo', 'iu', etc.)

### `fermatrica_utils.arrays.arrays` — General array/DataFrame operations

| Export | Description |
|--------|-------------|
| `DotDict` | `dict` subclass with attribute-style access (`d.key`) |
| `dict_to_df(dct, key_name, value_name)` | Convert dict to melted DataFrame |
| `dict_multifill(dct, keys, values)` | Bulk-fill dict keys |
| `groupby_eff(ds, group_vars, other_vars, mask)` | Efficient pandas groupby (bypasses slow column selection) |
| `select_eff(ds, select_vars)` | Efficient pandas column selection (concat per-column) |
| `str_replace_eff(sr, pattern, repl)` | String replace on mixed-type Series |
| `like_sr(sr, pattern)` | Regex filter on mixed-type Series |
| `rm_1_item_groups(ds, group_var)` | Remove groups with only 1 item |
| `pandas_tree_final_child(ds, var, var_ch)` | Collapse parent-child chains to final child only |
| `multi_index_destroyer(ds, sep)` | Flatten multi-level columns to single level |
| `pandas_filter_regex(ds, col, include, exclude)` | Filter rows by column regex |
| `list_select(search, lst, match)` | Select list elements by equality or regex |
| `list_select_pat(search, lst)` | Short alias for regex list selection |
| `list_unique(lst)` | Unique list preserving insertion order |
| `step_generator(start, stop, step)` | Budget/value generator |

### `fermatrica_utils.arrays.data` — Data pipeline operations

| Export | Description |
|--------|-------------|
| `decapitalise_df(ds)` | Lowercase all string columns |
| `extra_spaces_df(ds)` | Remove duplicate/trailing spaces |
| `columns_normalize_df(ds)` | Normalise column names (snake_case, transliterate) |
| `split_dataframe(ds, group_col)` | Split DataFrame by group column values |
| `convert_pyarrow_string(ds, ptrn_incl, ptrn_excl)` | Cast string columns to the pandas default `str` dtype |
| `read_pickle_compat(path)` | Read pickle with pandas StringDtype compatibility |
| `pandas_force_materialize(ds)` | Full DataFrame reconstruction (breaks all internal references) |
| `pandas_force_materialize_series(s)` | Full Series reconstruction (breaks all internal references) |
| `excel_sheets_look(dr, ...)` | List sheet names from Excel files matching patterns |
| `excel_header_look(dr, sht, row_num)` | Peek at top rows of Excel files |
| `excel_get_col_vals(dr, var_pat, skip_rows)` | Get unique column values from Excel files |
| `excel_get_row_vals(dr, start_col, row, ...)` | Get row values from Excel files |
| `excel_get_cols(dr, ...)` | Get column names/values from Excel files |
| `excel_df_look(dr, ...)` | Load head of Excel sheets into DataFrames |
| `header_row_detect(ds)` | Find the first non-empty row in a DataFrame |

### `fermatrica_utils.arrays.ts` — Time series utilities

| Export | Description |
|--------|-------------|
| `ma_eff(sr, n)` | Centred moving average with floating window (trailing NAs handled via min_periods) |
| `mar_eff(sr, n)` | Right-aligned moving average |
| `mal_eff(sr, n)` | Left-aligned moving average |
| `ts_decompose(ds, period_var, group_vars, value_var, ...)` | Decompose time series (seasonal, trend, residual via statsmodels or Fourier) |
| `na_0(ds, ptrn, replace_val)` | Fill NA with 0 (or other value) in columns matching pattern |
| `na_ma(sr, window)` | Fill NA with bidirectional moving average |
| `na_group_mean(df, val_col, group_cols)` | Fill NA with group mean |
| `na_replace_inner(sr, replace_val)` | Fill interior NAs only (preserve trailing NAs) |
| `sr_na(sr)` | NA mask compatible with PyArrow-backed Series |
| `mean_trim(x, ...)` | Robust trimmed mean — excludes periods of significant decline |
| `trim_numeric_mask(x, ...)` | Boolean mask marking structurally low values (launch ramp-up / end-of-life tail-off) |

### `fermatrica_utils.arrays.merge` — Period alignment and merge

| Export | Description |
|--------|-------------|
| `period_align(ds, period_trg, vars_agg_dict, dim_cols)` | Aggregate and crosswalk data between period granularities (day→week, month→quarter, etc.) |
| `ds_merge(ttl, ds, period_trg, vars_agg_dict, vars_group_list)` | General-purpose merge with automatic period alignment |

## Code Conventions

### Imports

```python
# Standard library
import copy, logging, os

# Third-party
import numpy as np
import pandas as pd
from line_profiler import profile

# fermatrica_utils internal
from fermatrica_utils.flow import fermatrica_utils_error, FermatricaUError
from fermatrica_utils.arrays.arrays import list_select, list_unique
```

Order: standard library → third-party → fermatrica_utils internal.

### Error handling

Use the module's error function, never raw `raise`:

```python
fermatrica_utils_error("Descriptive message about what went wrong")
```

Non-fatal issues: use the `fermatrica` logger at WARNING level:
```python
from fermatrica_utils.flow import fermatrica_warner
fermatrica_warner.warning("Descriptive message")
```

### Type annotations

Use Python 3.10+ union syntax: `str | None`, `pd.DataFrame | None`. Not `Optional[str]`.

### Performance profiling

Functions are decorated with `@profile` from `line_profiler`. Leave these in place — they are no-ops when profiling is not active.

### StableClass

`StableClass` prevents attribute creation after `__init__`. Call `self._init_finish()` as the last line in `__init__` of derived classes.

### Data utilities

- `pandas_force_materialize()` with `if_hard=True` performs full DataFrame reconstruction from values — nuclear-level reference breaking. Use when pandas memory fragmentation or shared references cause unexpected behaviour.
- `read_pickle_compat()` handles pandas StringDtype changes across versions via a custom unpickler.
- Excel helpers accept `ptrn_incl`/`ptrn_excl` patterns for filtering files and sheets.

### Time series helpers

- `ma_eff`/`mar_eff`/`mal_eff` use `min_periods=1` so trailing NAs shrink the window instead of expanding with NaNs.
- `na_ma` fills bidirectionally (forward then backward) so interior NAs get smoothed while edge NAs are ffill/bfilled.
- `mean_trim` and `trim_numeric_mask` are designed for removing brand launch ramp-up and end-of-life tail-off effects from mean calculations.
- `sr_na` is a pandas version-aware NA detector that works around PyArrow compatibility gaps in pandas < 3.0.

## Testing Conventions

### Running tests

```bash
# All tests
pytest tests/

# Specific test module
pytest tests/test_arrays/test_arrays.py -v

# Primitive tests
pytest tests/test_primitives/test_string.py -v

# With debug mode (breaks at assertion failures)
pytest tests/ --if-debug

# Verbose with shortened traceback
pytest tests/test_arrays/ -v --tb=short
```

### Test patterns

- **YAML-driven parametrization**: Test cases in `tests/**/test_cases/*.yaml` with custom constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`, `!pd.Timestamp`, `!list_of_df`). Use `parametrize_params()` from `tests/basics/basics.py`.
- **Expected error testing**: `exp_err` field — `pytest.raises(FermatricaUError)` or `nullcontext()`.
- **`if_debug` fixture**: When `True`, drops into `breakpoint()` on assertion failure. Set via `pytest --if-debug`.
- **Assertion helpers**: `assertNone`, `assertEq`, `assertListNanEq`, `assertListDfEq` in `tests/basics/basics.py`.

### What to test when changing code

| Change location | What tests to run |
|----------------|-------------------|
| `fermatrica_utils/arrays/arrays.py` | `tests/test_arrays/test_arrays.py`, `test_data.py` |
| `fermatrica_utils/arrays/data.py` | `tests/test_arrays/test_data.py` |
| `fermatrica_utils/arrays/ts.py` | `tests/test_arrays/test_ts.py` |
| `fermatrica_utils/arrays/merge.py` | `tests/test_arrays/test_merge.py` |
| `fermatrica_utils/primitives/date.py` | `tests/test_primitives/test_date.py` |
| `fermatrica_utils/primitives/num.py` | `tests/test_primitives/test_num.py` |
| `fermatrica_utils/primitives/string.py` | `tests/test_primitives/test_string.py` |
| `fermatrica_utils/flow.py` | `tests/basics/basics.py` |
| `fermatrica_utils/objects.py` | `tests/test_core/test_objects.py` |
| `fermatrica_utils/os.py` | `tests/test_core/test_os.py` |
| `fermatrica_utils/math.py` | `tests/test_core/test_math.py` |
| `fermatrica_utils/visual.py` | `tests/test_core/test_visual.py` |
| `fermatrica_utils/profiling.py` | `tests/test_core/test_profiling.py` |
| `fermatrica_utils/decorators.py` | Run spinner test manually |

### YAML constructors available

```yaml
expected_result: !pd.DataFrame
  data: [[1, 2], [3, 4]]
  columns: [a, b]

expected_result: !pd.Series
  data: [1, 2, 3]
  name: col_a

expected_result: !np.array
  object: [1, 2, 3]

expected_result: !pd.Timestamp
  value: "2024-01-15"

expected_result: !list_of_df
  - !pd.DataFrame
    data: [[1], [2]]
    columns: [x]
```

### Debug flag

Root `conftest.py` provides `if_debug` fixture:
```bash
pytest tests/ --if-debug
```

## Common Mistakes to Avoid

1. **Not using `fermatrica_utils_error`**: Always use `fermatrica_utils_error()` for raising errors, never raw `raise`. Use `fermatrica_warner.warning()` for non-fatal issues.

2. **Modifying StableClass after init**: `StableClass` subclasses (and `Model`/`ModelRep` in sibling repos) cannot accept new attributes after `_init_finish()`. Set all attributes during construction.

3. **Calling `groupby_eff` without a mask**: If you don't need a mask, pass `mask=None` explicitly — it is not a required keyword, but the function behaves correctly with or without it.

4. **Using `read_pickle_compat` with non-pickle files**: The function expects `.pickle`, `.pkl`, `.xz`, `.lzma`, or `.zst` files containing pickled pandas objects.

5. **Forgetting `period_align` constraints**: `period_align` requires a `period` column in the input dataset. The function expects exact period name strings (`'month'`, `'week'`, `'day'`, `'quarter'`, `'year'`).

6. **Breaking `sr_na` compatibility**: `sr_na` is a workaround for PyArrow-backed Series NA detection in older pandas. When pandas 3.0+ is standard, a future version may remove the workaround — keep the version check in place.

7. **Excel file locking**: Excel helper functions read files with `pd.read_excel` / `pd.ExcelFile`. Ensure no other process has the file open exclusively.
