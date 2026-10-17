# safe-pandas-exec

Run **LLM-generated pandas code** with several layers of containment, plus a provider-agnostic
**generate → run → critic → retry** loop. Extracted from the
[Mistral Data Detective](https://github.com/Giorgio089/mistral-data-detective) multi-agent project, where the original regex-based sandbox was replaced by the design below.

## What it does

1. **AST validation** (`validate_code`): rejects imports, class definitions, dunder/private attributes,
   `eval`/`exec`/`open`/`getattr`/`type`, file and process access (`read_*`, most `to_*`, `np.load`, …),
   `str.format` (format-string attribute access) and module hopping such as `pd.compat.os`.
2. **Subprocess execution** (`run_pandas_code`): fresh `python -I` process, cleared environment, temp working directory,
   restricted builtins, CPU limit, file-size limit 0 (no file writes), memory limit (Linux), wall-clock timeout, output truncation.
   Your DataFrame is passed as a copy and never mutated.
3. **Critic loop** (`analyze`): the LLM writes code, it is run, errors and critic objections are fed back for another try.

```python
from safe_pandas_exec import analyze, llm_critic

def llm(system: str, user: str) -> str:          # any provider: Mistral, Anthropic, OpenAI, local ...
    return my_client.chat(system=system, prompt=user)

res = analyze(df, "Which channel has the highest ROAS?", llm, critic=llm_critic(llm))
print(res.success, res.result, res.attempts)
print(res.code)
```

`run_pandas_code(code, df)` and `validate_code(code)` can be used on their own, e.g. inside an MCP tool.

## Security model - please read

This is **defence in depth, not a security boundary**. A denylist-based validator cannot prove arbitrary Python safe,
and the subprocess shares your kernel and network. Network access is blocked only by the validator, not by the OS.
If the code or the data can be influenced by untrusted people, additionally run the whole thing in a container, gVisor/Firecracker VM or similar with no network.
Memory limits via `RLIMIT_AS` are enforced on Linux only; on macOS rely on the timeout.

Known trade-offs: string literals containing `__` are rejected (even as column names), `str.format()` is rejected (use f-strings), and `df.style` / `df.eval` are unavailable.

## Install

```bash
pip install git+https://github.com/Giorgio089/safe-pandas-exec
```

MIT licensed.
