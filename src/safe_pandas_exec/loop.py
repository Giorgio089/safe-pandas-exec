"""Provider-agnostic generate -> run -> critic -> retry loop.

You supply `llm(system_prompt, user_prompt) -> str`; no SDK is imported here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .runner import ExecResult, run_pandas_code

LLM = Callable[[str, str], str]

CODE_SYSTEM_PROMPT = """You write pandas code that answers a question about a DataFrame.
Rules:
1. The DataFrame is named `df`; `pd` and `np` are preloaded. NEVER write import statements.
2. Store the final answer as a string in a variable named `result`.
3. Use vectorised pandas (groupby, merge, agg); avoid iterrows and Python loops.
4. Format numbers readably, e.g. f'{value:,.2f}'. Use f-strings, never str.format.
5. If `df` has a `_source_file` column, several tables were concatenated: split by it, then merge.
6. Reply with a single ```python code block and nothing else."""

CRITIC_SYSTEM_PROMPT = """You review the result of executed analysis code.
Check that the result answers the question, that magnitudes are plausible (e.g. CTR > 100% is impossible)
and that the code used the right columns. Reply ONLY with JSON:
{"valid": true|false, "confidence": 0.0-1.0, "issues": ["..."], "suggestion": "..."}"""


@dataclass
class CriticVerdict:
    valid: bool
    confidence: float = 1.0
    issues: list = field(default_factory=list)
    suggestion: str = ""


Critic = Callable[[str, str, str, str], CriticVerdict]  # question, code, result, schema


@dataclass
class AnalysisResult:
    success: bool
    result: str = ""
    code: str = ""
    error: str = ""
    attempts: int = 0
    verdict: Optional[CriticVerdict] = None


def build_schema(df: pd.DataFrame, max_unique: int = 10) -> str:
    """Compact schema description to put into the prompt."""
    lines = [f"rows: {len(df)}, columns: {len(df.columns)}"]
    for col in df.columns:
        s = df[col]
        info = f"- {col} ({s.dtype}), {int(s.notna().sum())} values"
        if s.isna().any():
            info += f", {int(s.isna().sum())} missing"
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info += f", min={s.min()}, max={s.max()}, mean={s.mean():.2f}"
        elif not pd.api.types.is_numeric_dtype(s):
            u = s.dropna().unique()
            info += f", values: {list(u)}" if len(u) <= max_unique else f", {len(u)} unique, e.g. {list(u[:5])}"
        lines.append(info)
    lines.append("first rows:\n" + df.head(3).to_string())
    return "\n".join(lines)


def extract_code(text: str) -> str:
    """Pull code out of an LLM reply (```python block, any block, or bare code mentioning `result`)."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text.strip() if "result" in text and "df" in text else ""


def llm_critic(llm: LLM) -> Critic:
    """Build a critic callable from any LLM function."""
    def critic(question: str, code: str, result: str, schema: str) -> CriticVerdict:
        raw = llm(CRITIC_SYSTEM_PROMPT,
                  f"Question: {question}\n\nSchema:\n{schema}\n\nCode:\n{code}\n\nResult:\n{result}")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return CriticVerdict(True, 0.0, ["critic reply contained no JSON"])  # fail open, flagged by 0.0
        try:
            d = json.loads(m.group(0))
            return CriticVerdict(bool(d.get("valid", True)), float(d.get("confidence", 0.5)),
                                 list(d.get("issues", [])), str(d.get("suggestion", "")))
        except (ValueError, TypeError):
            return CriticVerdict(True, 0.0, ["critic reply was not valid JSON"])  # fail open, flagged by 0.0
    return critic


def analyze(df: pd.DataFrame, question: str, llm: LLM, critic: Optional[Critic] = None,
            max_attempts: int = 3, timeout: float = 10.0) -> AnalysisResult:
    """Ask `llm` for code, run it safely, optionally have `critic` check it, retry with feedback."""
    schema = build_schema(df)
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
    last = AnalysisResult(False, error="no attempt made")
    for attempt in range(1, max_attempts + 1):
        code = extract_code(llm(CODE_SYSTEM_PROMPT, prompt))
        if not code:
            last = AnalysisResult(False, error="no code in LLM reply", attempts=attempt)
            prompt += "\n\nYour previous reply contained no code block. Reply with one ```python block."
            continue
        run: ExecResult = run_pandas_code(code, df, timeout=timeout)
        if not run.success:
            last = AnalysisResult(False, code=code, error=run.error, attempts=attempt)
            prompt += f"\n\nYour previous code failed.\nCode:\n{code}\nError: {run.error}\nFix it."
            continue
        verdict = critic(question, code, run.result, schema) if critic else None
        last = AnalysisResult(True, run.result, code, "", attempt, verdict)
        if verdict is None or verdict.valid:
            return last
        prompt += (f"\n\nA reviewer rejected your previous result.\nCode:\n{code}\nResult: {run.result}\n"
                   f"Issues: {verdict.issues}\nSuggestion: {verdict.suggestion}\nImprove the code.")
    return last
