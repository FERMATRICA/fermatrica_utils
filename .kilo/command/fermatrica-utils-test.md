---
description: Run FERMATRICA_UTILS tests with proper configuration
---

# Run FERMATRICA_UTILS Tests

You are a FERMATRICA_UTILS testing expert. Run the appropriate tests for the changes made and analyze failures.

## Test Commands

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_arrays/test_arrays.py -v

# Primitive tests
pytest tests/test_primitives/test_string.py -v

# With debug mode (drops into breakpoint() on assertion failure)
pytest tests/ --if-debug

# Verbose with shortened traceback
pytest tests/test_arrays/ -v --tb=short

# Run and stop at first failure
pytest tests/test_arrays/test_arrays.py -x
```

## What to test when changing code

| Change location | Tests to run |
|-----------------|-------------|
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
| `fermatrica_utils/decorators.py` | Run spinner test manually |

## Test Architecture

### YAML-driven parametrization
Test cases are defined in `tests/**/test_cases/*.yaml` with custom YAML constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`). Each test case entry has:
- `description`: test name
- `params`: input values
- `expected_result`: expected output
- `exp_err`: `pytest.raises(FermatricaUError)` or `nullcontext()`

Helper: `parametrize_params()` from `tests/basics/basics.py`.

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
Root `conftest.py` provides `if_debug` fixture. When `True`, drops into `breakpoint()` on failure:
```bash
pytest tests/ --if-debug
```

## When tests fail

1. **Read the assertion error carefully** — note the expected vs actual values
2. **Check if the test expectations need updating** (new behavior is correct) or code needs fixing
3. **YAML test failures** — check if `expected_result` in the YAML matches the new code behavior
4. **Use `--if-debug`** to get an interactive debugger on assertion failures
