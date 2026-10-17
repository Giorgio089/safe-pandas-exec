"""Execute validated pandas code in a separate, resource-limited Python process."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import pandas as pd

from .validator import validate_code

MAX_OUTPUT_CHARS = 20_000

_CHILD = r'''
import io, json, pickle, sys, contextlib
try:
    import resource
    cpu = int(sys.argv[3])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    mem = int(sys.argv[4]) * 1024 * 1024
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
except Exception:
    pass
import numpy as np, pandas as pd
out = sys.stdout
with open(sys.argv[1], "rb") as f:
    df = pickle.load(f)
code = open(sys.argv[2]).read()
names = ("len range enumerate zip map filter sorted reversed list dict tuple set frozenset str int float "
         "bool round sum min max abs any all divmod pow print isinstance slice format repr "
         "ValueError TypeError KeyError IndexError ZeroDivisionError AttributeError Exception").split()
import builtins
safe = {n: getattr(builtins, n) for n in names}
scope = {"__builtins__": safe, "pd": pd, "np": np, "df": df}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<llm>", "exec"), scope)
    res = scope.get("result")
    payload = {"ok": res is not None, "result": "" if res is None else str(res),
               "printed": buf.getvalue(),
               "error": "" if res is not None else "code did not set a variable named 'result'"}
except BaseException as e:
    payload = {"ok": False, "result": "", "printed": buf.getvalue(), "error": f"{type(e).__name__}: {e}"}
out.write("\n@@RESULT@@" + json.dumps(payload))
'''


@dataclass
class ExecResult:
    success: bool
    result: str = ""
    error: str = ""
    code: str = ""
    printed: str = ""


def run_pandas_code(code: str, df: pd.DataFrame, timeout: float = 10.0, memory_mb: int = 1024,
                    max_output_chars: int = MAX_OUTPUT_CHARS) -> ExecResult:
    """Validate `code`, then run it against a copy of `df` in a child process.

    The code must assign its answer to a variable called `result`. `pd`, `np` and `df` are preloaded.
    """
    problems = validate_code(code)
    if problems:
        return ExecResult(False, error="validation failed: " + "; ".join(problems), code=code)

    with tempfile.TemporaryDirectory(prefix="spe_") as tmp:
        data_path, code_path = os.path.join(tmp, "df.pkl"), os.path.join(tmp, "code.py")
        with open(data_path, "wb") as f:
            pickle.dump(df, f)
        with open(code_path, "w") as f:
            f.write(code)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        cpu = max(1, int(timeout) + 1)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _CHILD, data_path, code_path, str(cpu), str(memory_mb)],
                capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, error=f"timeout after {timeout}s", code=code)

    marker = "@@RESULT@@"
    if marker not in proc.stdout:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        return ExecResult(False, error=f"execution crashed: {tail[0]}", code=code)
    data = json.loads(proc.stdout.rsplit(marker, 1)[1])
    result = data["result"]
    if len(result) > max_output_chars:
        result = result[:max_output_chars] + "\n... [truncated]"
    return ExecResult(data["ok"], result, data["error"], code, data["printed"][:max_output_chars])
