"""Static validation of LLM-generated pandas code (AST allowlist/denylist).

This is the first of two layers; the second is the restricted subprocess in `runner`.
"""

from __future__ import annotations

import ast

MAX_CODE_CHARS = 20_000

FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "input", "breakpoint", "help", "getattr", "setattr",
    "delattr", "hasattr", "globals", "locals", "vars", "dir", "memoryview", "type", "super",
    "object", "classmethod", "staticmethod", "property", "exit", "quit",
}

# Attribute names that reach files, processes, code evaluation or format-string attribute access.
FORBIDDEN_ATTRS = {
    "eval", "format", "format_map", "style", "system", "popen", "os", "sys", "io", "lib",
    "ctypeslib", "f2py", "testing", "load", "loads", "save", "savez", "savez_compressed",
    "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "fromregex",
    "ExcelFile", "ExcelWriter", "HDFStore", "read_clipboard", "to_clipboard",
    "subprocess", "builtins", "importlib", "shutil", "pathlib", "socket", "ctypes", "pickle",
    "tempfile", "urllib", "requests", "platform", "pty", "signal", "threading", "multiprocessing",
    "glob", "fnmatch", "compat", "util", "core", "_libs",
}

# `to_*` methods are denied unless they only return in-memory values.
ALLOWED_TO_METHODS = {
    "to_string", "to_dict", "to_list", "tolist", "to_numpy", "to_frame", "to_datetime",
    "to_numeric", "to_period", "to_timestamp", "to_timedelta", "to_markdown", "to_records",
}


def validate_code(code: str) -> list[str]:
    """Return a list of violations. Empty list means the code passed validation."""
    if len(code) > MAX_CODE_CHARS:
        return [f"code longer than {MAX_CODE_CHARS} characters"]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append("imports are not allowed (pd and np are preloaded)")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global,
                               ast.Nonlocal, ast.Yield, ast.YieldFrom)):
            problems.append(f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                problems.append(f"name '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                problems.append(f"private/dunder attribute '{attr}' is not allowed")
            elif attr in FORBIDDEN_ATTRS:
                problems.append(f"attribute '{attr}' is not allowed")
            elif attr.startswith("read_"):
                problems.append(f"I/O function '{attr}' is not allowed")
            elif attr.startswith("to_") and attr not in ALLOWED_TO_METHODS:
                problems.append(f"I/O method '{attr}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                problems.append("string literals containing '__' are not allowed")
    # de-duplicate, keep order
    return list(dict.fromkeys(problems))
