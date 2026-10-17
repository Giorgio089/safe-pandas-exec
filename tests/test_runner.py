import sys

import pandas as pd
import pytest

from safe_pandas_exec import run_pandas_code

DF = pd.DataFrame({"a": [1, 2, 3], "c": ["x", "y", "x"]})


def test_success_and_df_not_mutated():
    r = run_pandas_code("df['a'] = 0\nresult = str(df['a'].sum())", DF)
    assert r.success and r.result == "0"
    assert DF["a"].sum() == 6


def test_groupby():
    r = run_pandas_code("result = df.groupby('c')['a'].sum().to_string()", DF)
    assert r.success and "x" in r.result and "4" in r.result


def test_missing_result():
    r = run_pandas_code("x = 1", DF)
    assert not r.success and "result" in r.error


def test_runtime_error():
    r = run_pandas_code("result = df['nope']", DF)
    assert not r.success and "KeyError" in r.error


def test_validation_blocks_before_running():
    r = run_pandas_code("import os\nresult = '1'", DF)
    assert not r.success and "validation failed" in r.error


def test_timeout():
    r = run_pandas_code("while True:\n    pass", DF, timeout=1)
    assert not r.success and "timeout" in r.error


def test_output_truncated():
    r = run_pandas_code("result = 'x' * 100000", DF, max_output_chars=100)
    assert r.success and r.result.endswith("[truncated]") and len(r.result) < 200


def test_print_captured_not_in_result():
    r = run_pandas_code("print('hi')\nresult = 'done'", DF)
    assert r.result == "done" and r.printed == "hi\n"


def test_no_builtin_escape_at_runtime():
    # bypass validator on purpose to prove the runtime builtins are restricted
    from safe_pandas_exec import runner
    orig = runner.validate_code
    runner.validate_code = lambda c: []
    try:
        r = run_pandas_code("result = str(open('/etc/passwd').read())", DF)
        assert not r.success and "NameError" in r.error
        r = run_pandas_code("result = str(__import__('os'))", DF)
        assert not r.success
    finally:
        runner.validate_code = orig


def test_cannot_write_files():
    from safe_pandas_exec import runner
    orig = runner.validate_code
    runner.validate_code = lambda c: []
    try:
        r = run_pandas_code("df.to_csv('leak.csv')\nresult = 'wrote'", DF)
        assert not r.success  # RLIMIT_FSIZE=0
    finally:
        runner.validate_code = orig


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS only enforced on Linux")
def test_memory_limit():
    r = run_pandas_code("x = [0] * (10 ** 10)\nresult = 'y'", DF, memory_mb=256)
    assert not r.success
