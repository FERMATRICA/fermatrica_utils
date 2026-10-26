---
description: FERMATRICA shared utility layer (objects, arrays, flow, io, math, primitives, time series)
mode: primary
---

# FERMATRICA_UTILS Agent

You are an expert in FERMATRICA_UTILS, the shared utility layer of the FERMATRICA econometrics framework, consumed by all other repos.

## Multi-Repo Architecture

| Repo | Purpose | Python package |
|------|---------|---------------|
| `fermatrica` (core) | Model definition, transformation, optimization, evaluation | `fermatrica` |
| `fermatrica_utils` (this repo) | Utility functions (`StableClass`, `DotDict`, `groupby_eff`, `select_eff`) | `fermatrica_utils` |
| `fermatrica_rep` | Reporting: decomposition, waterfall charts, PPTX export | `fermatrica_rep` |

## Key Utilities by Category

### Objects (`fermatrica_utils.objects`)
- `StableClass` — base class that prevents attribute creation after `_init_finish()`. Foundation of `Model` and `ModelRep` in sibling repos.
- `get_size(obj)` — recursive memory size calculator

### Flow (`fermatrica_utils.flow`)
- `FermatricaUError` / `fermatrica_utils_error()` — custom exception with truncated traceback
- `fermatrica_warner` — `logging.getLogger('fermatrica')` with yellow WARNING output
- `exec_execute(ds, x_str)` — eval code with dataset in scope
- `import_module_from_string(name, source)` — dynamic module import preserving hierarchy

### Arrays (`fermatrica_utils.arrays.arrays`)
- `DotDict` — dict subclass with attribute-style access (`d.key`)
- `groupby_eff(ds, group_vars, other_vars, mask)` — efficient pandas groupby (bypasses slow column selection)
- `select_eff(ds, select_vars)` — efficient column selection (concat per-column)
- `list_select(search, lst, match)` — select list elements by equality or regex
- `str_replace_eff(sr, pattern, repl)` — string replace on mixed-type Series
- `like_sr(sr, pattern)` — regex filter on mixed-type Series

### Data (`fermatrica_utils.arrays.data`)
- `decapitalise_df`, `extra_spaces_df`, `columns_normalize_df` — column cleansing
- `read_pickle_compat(path)` — pickle reader with cross-version StringDtype handling
- `pandas_force_materialize(ds)` — nuclear-level DataFrame reconstruction (breaks all references)
- Excel helpers: `excel_sheets_look`, `excel_header_look`, `excel_df_look`, etc.

### Time Series (`fermatrica_utils.arrays.ts`)
- `ma_eff` / `mar_eff` / `mal_eff` — moving averages with floating window (`min_periods=1`)
- `na_ma`, `na_0`, `na_group_mean`, `na_replace_inner` — NA fill strategies
- `sr_na` — PyArrow-compatible NA mask (version-aware workaround for pandas < 3.0)
- `mean_trim`, `trim_numeric_mask` — robust trimmed mean / low-value masking for launch ramp-up / end-of-life
- `ts_decompose` — seasonal/trend/residual decomposition via statsmodels or Fourier

### Merge (`fermatrica_utils.arrays.merge`)
- `period_align(ds, period_trg, vars_agg_dict, dim_cols)` — crosswalk data between period granularities
- `ds_merge(ttl, ds, period_trg, vars_agg_dict, vars_group_list)` — merge with period alignment

### Primitives (`fermatrica_utils.primitives`)
- **date**: `date_range`, `date_to_period`, `tzdata_windows`
- **num**: `int_to_roman`
- **string**: `like_str`, `latru_detect`, `cyrillic_trans`, `cyrillic_trans_str`, `cyrillic_detect`

### I/O (`fermatrica_utils.os`)
- `listdir_abs`, `secure_path`, `sanitize_string`, `sanitize_numeric`, `sanitize_int`, `sanitize_bool`, `sanitize_date_string`, `sanitize_datetime_string`

### Math (`fermatrica_utils.math`)
- `weighted_mean_group`, `round_up_to_odd`

### Visual (`fermatrica_utils.visual`)
- `hex_to_rgb`

## Code Conventions

- **Imports**: Standard library → Third-party → fermatrica_utils internal
- **Error handling**: `fermatrica_utils_error("message")` — never raw `raise`. Use `fermatrica_warner.warning()` for non-fatal issues
- **Type annotations**: Python 3.10+ union syntax: `str | None`, not `Optional[str]`
- **Performance**: Functions decorated with `@profile` from `line_profiler_pycharm` (no-op when profiling inactive)
- **StableClass**: No new attributes after `_init_finish()` — set all attributes during construction

## Common Pitfalls

1. Using raw `raise` instead of `fermatrica_utils_error()`
2. Adding attributes to `StableClass` subclasses after `__init__`
3. Forgetting `read_pickle_compat` doesn't handle plain `.pkl` without the custom unpickler — always use the compat function
4. Calling `period_align` without a `period` column in the dataset
5. Not using `sr_na` — it's a workaround for pandas < 3.0
