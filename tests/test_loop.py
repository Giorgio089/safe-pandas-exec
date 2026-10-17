import pandas as pd

from safe_pandas_exec import CriticVerdict, analyze, build_schema, extract_code, llm_critic

DF = pd.DataFrame({"rev": [10.0, 20.0, 30.0], "c": ["a", "b", "a"]})


def scripted(*replies):
    it = iter(replies)
    calls = []
    def llm(system, user):
        calls.append(user)
        return next(it)
    llm.calls = calls
    return llm


def block(code):
    return f"```python\n{code}\n```"


def test_extract_code_variants():
    assert extract_code(block("result = 1")) == "result = 1"
    assert extract_code("```\nresult = df\n```") == "result = df"
    assert extract_code("result = df.sum()") == "result = df.sum()"
    assert extract_code("no code here") == ""


def test_build_schema_mentions_columns():
    s = build_schema(DF)
    assert "rev" in s and "min=10.0" in s and "values: ['a', 'b']" in s


def test_happy_path():
    llm = scripted(block("result = str(df['rev'].sum())"))
    r = analyze(DF, "total revenue?", llm)
    assert r.success and r.result == "60.0" and r.attempts == 1


def test_retry_after_error_gets_error_feedback():
    llm = scripted(block("result = df['missing']"), block("result = str(df['rev'].max())"))
    r = analyze(DF, "max?", llm)
    assert r.success and r.attempts == 2 and "KeyError" in llm.calls[1]


def test_gives_up_after_max_attempts():
    llm = scripted(*[block("import os")] * 3)
    r = analyze(DF, "q", llm, max_attempts=3)
    assert not r.success and r.attempts == 3 and "validation" in r.error


def test_critic_rejection_triggers_retry():
    llm = scripted(block("result = '999'"), block("result = '60'"))
    verdicts = iter([CriticVerdict(False, 0.9, ["too big"], "sum it"), CriticVerdict(True)])
    r = analyze(DF, "sum?", llm, critic=lambda q, c, res, s: next(verdicts))
    assert r.success and r.result == "60" and r.attempts == 2 and "too big" in llm.calls[1]


def test_llm_critic_parses_json_and_fails_open_flagged():
    good = llm_critic(lambda s, u: 'noise {"valid": false, "confidence": 0.8, "issues": ["x"], "suggestion": "y"}')
    v = good("q", "c", "r", "s")
    assert not v.valid and v.issues == ["x"]
    bad = llm_critic(lambda s, u: "not json")("q", "c", "r", "s")
    assert bad.valid and bad.confidence == 0.0
