import pytest

from safe_pandas_exec import validate_code

OK = [
    "result = str(df['a'].sum())",
    "top = df.groupby('c')['a'].sum().sort_values(ascending=False).head(3)\nresult = f'{top.to_string()}'",
    "def f(x):\n    return x * 2\nresult = str(df['a'].map(f).sum())",
    "result = df.describe().to_string()",
    "result = pd.to_datetime(df['d']).dt.year.value_counts().to_dict().__len__().__str__()".replace(".__len__().__str__()", ""),
]

BAD = [
    "import os",
    "from os import system",
    "result = __import__('os').system('id')",
    "result = open('/etc/passwd').read()",
    "result = eval('1+1')",
    "exec('x=1')",
    "result = ().__class__.__bases__[0].__subclasses__()",
    "result = '{0.__class__}'.format(df)",
    "result = '{0.T}'.format_map({'0': df})",
    "result = getattr(df, 'values')",
    "pd.read_csv('/etc/passwd')",
    "df.to_csv('/tmp/x.csv')",
    "df.to_pickle('/tmp/x')",
    "np.load('/tmp/x.npy', allow_pickle=True)",
    "np.lib.npyio.os.system('id')",
    "pd.io.common.os.system('id')",
    "df.eval('a + 1')",
    "result = df.style.to_html()",
    "class A: pass",
    "result = type(df)",
    "x = df.__dict__",
    "result = f'{df.__class__}'",
    "result = df['__x__']",
    "while True: pass\nimport sys",
    "if True:\n  def f(:",
]


@pytest.mark.parametrize("code", OK)
def test_ok(code):
    assert validate_code(code) == []


@pytest.mark.parametrize("code", BAD)
def test_bad(code):
    assert validate_code(code), code


def test_too_long():
    assert validate_code("x=1\n" * 10_000)


@pytest.mark.parametrize("code", [
    "pd.compat.platform.subprocess", "pd.util.os", "np.core.multiarray", "pd.core.common.sys",
    "pd.compat.pickle.loads(b'')",
])
def test_module_hopping_blocked(code):
    assert validate_code(code)
