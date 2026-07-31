# Package Structure, Configuration, and Dependency Injection

[Testing and CI/CD](02-testing-and-cicd.md) built a suite that works and left a mess behind it. `quantlib.py` holds the signal, the sizer, the fee model and the entire event loop in one file; every test block had to manipulate `PYTHONPATH` before it could import anything; the golden file is addressed by a relative path that only resolves when you happen to run from the repository root; and the engine reads `data/part5.parquet` by name from inside a function that has no business knowing where data lives. None of that is a testing problem. All of it makes tests harder to write, which is the reliable early symptom of a structural one.

This lesson turns that scratch module into an installable package and then checks — mechanically, not by convention — that it stays honest. A five-layer boundary gets defined and a thirty-line program verifies the import graph obeys it, which takes **one Friday-afternoon hotfix** to violate. The same package is installed twice, once in each layout, and the flat one silently imports the wrong copy. Configuration becomes a validated lattice, and the value of the schema is measured in the only currency that matters: an out-of-range lookback that a schema rejects at startup produces, without one, **a Sharpe of −0.5047 and no error at all**. A `SecretStr` is probed along eleven paths and holds on eight. And the broker seam that [Part VI](../part-06-live-infrastructure/01-system-architecture.md) demonstrated gets formalized as a `Protocol`, type-checked by mypy, and proven by three implementations producing one identical fill digest.

## The import graph is the architecture, and a program can check it

Architecture diagrams describe intentions. The import graph describes what the code actually does, and the two drift apart within weeks unless something enforces the difference. The layering that matters for a trading platform is a strict order: infrastructure knows about the machine, data turns files into frames, signals turn frames into opinions, portfolio turns opinions into share counts, and execution turns share counts into fills. Each layer may import from strictly lower ones and never the reverse — so a signal cannot reach into execution, and the reason is not tidiness but testability, since a layer that imports downward can only be tested by standing up everything beneath it.

```python
import ast
import shutil
from pathlib import Path

ROOT = Path("lab/platform")
SRC = ROOT / "src" / "quantlib"
shutil.rmtree(ROOT, ignore_errors=True)
LAYERS = ["infra", "data", "signals", "portfolio", "execution"]
for m in LAYERS:
    (SRC / m).mkdir(parents=True)

(ROOT / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "quantlib"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "pandas", "pyarrow", "pydantic", "pydantic-settings"]

[project.entry-points."quantlib.strategies"]
tsmom = "quantlib.signals.tsmom:TSMom"

[tool.setuptools.packages.find]
where = ["src"]
''')

(SRC / "__init__.py").write_text('__version__ = "0.1.0"\n')

(SRC / "infra" / "__init__.py").write_text('''"""Layer 0: knows about the machine, nothing about trading."""
from pathlib import Path


def repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data").is_dir():
            return p
    raise RuntimeError("no data/ directory above the working directory")
''')

(SRC / "data" / "__init__.py").write_text('''"""Layer 1: turns files into frames. Knows nothing about signals."""
import pandas as pd

from quantlib.infra import repo_root

ASSETS = ["SPY", "TLT", "GLD"]


def closes(name: str = "prices.parquet") -> pd.DataFrame:
    return pd.read_parquet(repo_root() / "data" / name)[ASSETS]
''')

(SRC / "signals" / "__init__.py").write_text('''"""Layer 2: turns frames into opinions. Knows nothing about sizing."""
import numpy as np
import pandas as pd

from quantlib.data import ASSETS


def logret(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px[ASSETS]).diff()


def tsmom(px: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    return np.sign(logret(px).rolling(lookback).sum()).shift(1)
''')

(SRC / "signals" / "tsmom.py").write_text('''from quantlib.signals import tsmom


class TSMom:
    name = "tsmom"

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def signal(self, px):
        return tsmom(px, self.lookback)
''')

(SRC / "portfolio" / "__init__.py").write_text('''"""Layer 3: turns opinions into share counts. Knows nothing about brokers."""


def size(equity: float, prices, weights, max_gross: float = 1.0) -> list[int]:
    gross = sum(abs(w) for w in weights)
    scale = min(1.0, max_gross / gross) if gross else 0.0
    return [int(w * scale * equity / p) for w, p in zip(weights, prices)]
''')

(SRC / "execution" / "__init__.py").write_text('''"""Layer 4: turns share counts into fills. The only layer that talks outward."""
from typing import Protocol, runtime_checkable

from quantlib.portfolio import size


@runtime_checkable
class Broker(Protocol):
    def submit(self, ts, symbol: str, qty: int, price: float) -> None: ...
    def poll(self) -> list[tuple]: ...
''')

ORDER = {m: i for i, m in enumerate(LAYERS)}


def violations(src: Path) -> list[str]:
    """Every import that points at its own layer or higher."""
    out = []
    for f in sorted(src.rglob("*.py")):
        parts = f.relative_to(src).parts
        here = parts[0] if parts and parts[0] in ORDER else None
        if here is None:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            for m in mods:
                bits = m.split(".")
                if len(bits) >= 2 and bits[0] == "quantlib" and bits[1] in ORDER:
                    there = bits[1]
                    if ORDER[there] >= ORDER[here] and there != here:
                        out.append(f"{f.name}:{node.lineno}: {here} imports {there} "
                                   f"(layer {ORDER[here]} -> {ORDER[there]})")
    return out


n = len(list(SRC.rglob("*.py")))
print(f"  {n} modules across {len(LAYERS)} layers: " + " < ".join(LAYERS))
v = violations(SRC)
print("  clean package: " + ("no layering violations" if not v else "\n    ".join(v)))

(SRC / "signals" / "hotfix.py").write_text('''"""Written at 4pm on a Friday."""
from quantlib.execution import Broker
from quantlib.portfolio import size


def emergency_flatten(broker: Broker, positions):
    for sym, qty in positions.items():
        broker.submit(None, sym, -qty, 0.0)
''')
print("\n  after one hotfix that reached downward:")
for x in violations(SRC):
    print("    " + x)
# =>   7 modules across 5 layers: infra < data < signals < portfolio < execution
#      clean package: no layering violations
#
#      after one hotfix that reached downward:
#        hotfix.py:2: signals imports execution (layer 2 -> 4)
#        hotfix.py:3: signals imports portfolio (layer 2 -> 3)
```

Thirty lines of `ast` walking, and the architecture is now a thing that can fail a build. The clean package passes; a single plausible file — `emergency_flatten`, written under pressure, doing something entirely reasonable — breaks the rule twice and is caught by name and line number.

What makes this worth automating is that the violation is invisible to every other tool. `emergency_flatten` has no bug. It passes the linter, the type checker, and any test anyone would write for it. Its cost is deferred and structural: `signals` can no longer be imported without dragging `execution` in behind it, so the fast unit tier from the previous lesson silently acquires a dependency on broker code, and the next person who wants to test a signal in isolation discovers they cannot. **Layering violations are not defects, they are debts**, and the only reliable moment to refuse one is the moment it is introduced — which means the check belongs in the pipeline next to `ruff`, not in a design document nobody re-reads.

## src layout, or you test the copy you did not ship

The `src/` directory looks like bureaucracy until you understand what it prevents. Python puts the working directory at the front of the import path, so a package sitting at the project root shadows the installed one — and the tests you just ran, the ones that went green, exercised your editor buffer rather than the artifact you are about to deploy.

```python
import shutil
import subprocess
import sys
from pathlib import Path

ROOT, SITE, FLAT = Path("lab/platform"), Path("lab/site-packages"), Path("lab/flat")
shutil.rmtree(SITE, ignore_errors=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "--target", str(SITE), str(ROOT)], capture_output=True, text=True)
print(f"  pip install --target lab/site-packages lab/platform -> "
      f"{'ok' if r.returncode == 0 else r.stderr[-200:]}")
print("  installed: " + ", ".join(sorted(p.name for p in SITE.iterdir())))

PROBE = "import quantlib; print(f'    {quantlib.__version__:10s} {quantlib.__file__}')"
ENV = {"PYTHONPATH": str(SITE.resolve()), "PATH": "/usr/bin:/bin"}

# the flat layout: a copy of the package sitting at the project root,
# edited to 0.2.0-dev, exactly as a working tree would be
shutil.rmtree(FLAT, ignore_errors=True)
FLAT.mkdir(parents=True)
shutil.copytree(ROOT / "src" / "quantlib", FLAT / "quantlib")
(FLAT / "quantlib" / "__init__.py").write_text('__version__ = "0.2.0-dev"\n')

HERE = str(Path.cwd()) + "/"
for label, cwd in [("src layout  (no quantlib/ at the root)", ROOT),
                   ("flat layout (quantlib/ at the root)", FLAT)]:
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                         text=True, cwd=str(cwd), env=ENV)
    print(f"\n  {label}, `import quantlib` resolves to:")
    print((out.stdout.rstrip() or out.stderr[-200:]).replace(HERE, ""))
# =>   pip install --target lab/site-packages lab/platform -> ok
#      installed: quantlib, quantlib-0.1.0.dist-info
#
#      src layout  (no quantlib/ at the root), `import quantlib` resolves to:
#        0.1.0      lab/site-packages/quantlib/__init__.py
#
#      flat layout (quantlib/ at the root), `import quantlib` resolves to:
#        0.2.0-dev  lab/flat/quantlib/__init__.py
```

Same installed package, same `PYTHONPATH`, same command — and two different modules. Under `src/`, `import quantlib` finds **0.1.0 in site-packages**, the thing that was built, packaged and installed. Under the flat layout it finds **0.2.0-dev in the working tree**, and the installed copy might as well not exist.

The failure this prevents is specific and nasty. Everything declared in `pyproject.toml` but absent from the source tree — package data, a `py.typed` marker, a compiled extension, a generated version file — exists only in the installed artifact. Under a flat layout your tests never touch that artifact, so a packaging mistake that makes the wheel unusable produces a completely green suite. You discover it in the deployment, which for a trading platform means you discover it at the open. The `src/` directory buys one property and it is worth the extra path component: **the only importable `quantlib` is one that was actually installed**, so the tests and production are looking at the same files.

## Configuration is a lattice with exactly one precedence order

A trading platform runs the same code in research, paper, and live, and the difference between those three is configuration. That makes the config system load-bearing, and it needs two properties: a single documented precedence order, so that "which value won" is never a mystery, and a canonical identity, so that two runs can be compared without diffing files. [Logging and Configuration Management](../part-02-python/07-logging-and-config.md) introduced the `config_hash` for exactly this; here it becomes a property of a typed object rather than a dict.

```python
import hashlib
import json
import os
import tomllib
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QL_", extra="forbid")
    lookback: int = Field(252, ge=2, le=2520)
    max_gross: float = Field(1.0, gt=0, le=3.0)
    assets: list[str] = ["SPY", "TLT", "GLD"]
    vendor_token: SecretStr = SecretStr("")


def canonical(s: Settings) -> str:
    """Identity of the effective config, secrets excluded by construction."""
    payload = s.model_dump(mode="json", exclude={"vendor_token"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


Path("lab").mkdir(exist_ok=True)
Path("lab/quant.toml").write_text("lookback = 126\nmax_gross = 1.5\n")
file_cfg = tomllib.loads(Path("lab/quant.toml").read_text())
os.environ.update(QL_LOOKBACK="126", QL_MAX_GROSS="1.5")

routes = {"kwargs in code": Settings(lookback=126, max_gross=1.5),
          "a toml file": Settings(**file_cfg),
          "the environment": Settings(),
          "toml, env overriding": Settings(**{**file_cfg, "max_gross": 1.5})}
for name, s in routes.items():
    print(f"  {name:22s} lookback {s.lookback}  gross {s.max_gross}  -> {canonical(s)}")
for k in ("QL_LOOKBACK", "QL_MAX_GROSS"):
    os.environ.pop(k, None)

print("\n  the same settings as plain dicts, hashed two ways:")
for d in [{"lookback": 126, "max_gross": 1.5},
          {"max_gross": 1.5, "lookback": 126},
          {"lookback": 126.0, "max_gross": 1.5}]:
    naive = hashlib.sha256(str(d).encode()).hexdigest()[:12]
    sortd = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    print(f"    {str(d):40s} str() {naive}  sorted-json {sortd}")
# =>   kwargs in code         lookback 126  gross 1.5  -> cdfb8444e110
#      a toml file            lookback 126  gross 1.5  -> cdfb8444e110
#      the environment        lookback 126  gross 1.5  -> cdfb8444e110
#      toml, env overriding   lookback 126  gross 1.5  -> cdfb8444e110
#
#      the same settings as plain dicts, hashed two ways:
#        {'lookback': 126, 'max_gross': 1.5}      str() a0856ecdf5d5  sorted-json 486a7051a9ad
#        {'max_gross': 1.5, 'lookback': 126}      str() 4149f5bdb67e  sorted-json 486a7051a9ad
#        {'lookback': 126.0, 'max_gross': 1.5}    str() d2875eac3885  sorted-json b7e300473c1e
```

Four routes into the same effective configuration — literal keyword arguments, a TOML file, environment variables, and a file with an environment override — and one identity, `cdfb8444e110`. That is the property worth having: the hash names the configuration that *ran*, not the route by which it arrived, so "were these two backtests configured identically?" is answered by comparing twelve characters rather than by reconstructing whose environment had what.

The second table shows why the hash needs a canonical form and where even that runs out. Hashing `str(d)` makes key order significant, so the same two settings written in a different order in a TOML file produce **`a0856ecdf5d5` and `4149f5bdb67e`** — two identities for one configuration, and a comparison that reports a difference where none exists. Sorted JSON fixes it: both land on `486a7051a9ad`. But the third row is the honest caveat. **`126.0` and `126` hash differently even under sorted JSON**, because JSON preserves the distinction between a float and an int, and an environment variable parsed by a helper that reaches for `float()` will produce exactly that drift. Canonicalization solves ordering; only a schema solves types, which is the next section.

## A schema turns a wrong number into a startup failure

The argument for validating configuration is usually made in terms of nice error messages. The real argument is that unvalidated configuration does not always produce an error at all.

```python
import numpy as np
import pandas as pd
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QL_", extra="forbid")
    lookback: int = Field(252, ge=2, le=2520)
    max_gross: float = Field(1.0, gt=0, le=3.0)
    assets: list[str] = ["SPY", "TLT", "GLD"]


def pipeline(cfg):
    """Four stages. Whatever is wrong with cfg gets discovered somewhere in here."""
    px = pd.read_parquet("data/prices.parquet")[cfg["assets"]]
    r = np.log(px).diff()
    sig = np.sign(r.rolling(cfg["lookback"]).sum()).shift(1)
    book = (sig * r).mean(axis=1).dropna()
    return float(np.sqrt(252) * book.mean() / book.std())


BASE = {"assets": ["SPY", "TLT", "GLD"], "max_gross": 1.0}
for lookback, label in [("252", "a string, as every env var is"),
                        (1, "a number outside its legal range")]:
    cfg = {**BASE, "lookback": lookback}
    print(f"  lookback={lookback!r:6s}  {label}")
    try:
        s = Settings(**cfg)
        print(f"    with a schema     accepted, coerced to "
              f"{s.lookback!r} ({type(s.lookback).__name__})")
    except ValidationError as e:
        print(f"    with a schema     rejected at startup: "
              f"lookback {e.errors()[0]['msg'].lower()}")
    try:
        print(f"    without one       pipeline ran to completion, "
              f"Sharpe {pipeline(cfg):.4f}")
    except Exception as e:
        print(f"    without one       {type(e).__name__} deep inside: {str(e)[:52]}")
# =>   lookback='252'   a string, as every env var is
#        with a schema     accepted, coerced to 252 (int)
#        without one       ValueError deep inside: passed window 252 is not compatible with a datetimel
#      lookback=1       a number outside its legal range
#        with a schema     rejected at startup: lookback input should be greater than or equal to 2
#        without one       pipeline ran to completion, Sharpe -0.5047
```

The two rows fail in opposite and equally instructive ways. A lookback of `"252"` is what *every* environment variable looks like, because environment variables are strings; the schema coerces it to an integer and the run proceeds. Without a schema the string travels three stages down and detonates inside pandas as `passed window 252 is not compatible with a datetimelike index` — an error that mentions neither configuration nor the actual problem, and that sends the reader to look at their index.

The second row is the one that should change your practice. A lookback of `1` is legal Python, legal pandas, and completely wrong: a one-day momentum window makes the signal the sign of yesterday's return, which is a different strategy entirely. The schema rejects it before a single byte of market data is read. Without the schema **the pipeline runs to completion and returns a Sharpe of −0.5047**, and there is no error, no warning, and nothing in the output to suggest the number is anything other than the strategy's honest performance. Somebody will paste that into a deck.

That is the case for schemas stated exactly: not that they produce better messages when things break, but that they convert an entire class of silent wrongness into loud, early, specific failure. The bound `ge=2, le=2520` is not defensive programming, it is a *domain fact* — a momentum lookback shorter than two days is not a momentum strategy and one longer than a decade has no data — and writing domain facts into the type is how they stop being folklore that lives in one person's head.

## The secret is safe until the moment you use it

[Part VI](../part-06-live-infrastructure/06-secrets-paper-live-compliance.md) covered where credentials live and how narrowly to scope them. The question here is narrower and more mechanical: given a secret already loaded into a config object, which of the many ways a program can emit that object will emit the secret with it? `SecretStr` promises masking. It is worth measuring what the promise covers.

```python
import json
import pickle

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QL_")
    lookback: int = 252
    vendor_token: SecretStr = SecretStr("")


s = Settings(vendor_token=SecretStr("tok-live-9f3c1a"))
LEAK = "tok-live-9f3c1a"


def library_echo():
    """Anything that parses the unwrapped secret can print it in an exception."""
    try:
        int(s.vendor_token.get_secret_value())
    except ValueError as e:
        return str(e)


probes = {
    "str(settings)": lambda: str(s),
    "repr(settings)": lambda: repr(s),
    "f-string on the field": lambda: f"{s.vendor_token}",
    "logging '%s' of the model": lambda: "%s" % s,
    "model_dump()": lambda: str(s.model_dump()),
    "model_dump_json()": lambda: s.model_dump_json(),
    "json.dumps(dump, default=str)": lambda: json.dumps(s.model_dump(), default=str),
    "vars(settings)": lambda: str(vars(s)),
    "pickle.dumps(settings)": lambda: str(pickle.dumps(s)),
    "a library echoing the value": library_echo,
    ".get_secret_value()": lambda: s.vendor_token.get_secret_value(),
}
masked = 0
for name, fn in probes.items():
    try:
        text = fn() or ""
    except Exception as exc:
        text = f"<{type(exc).__name__}>"
    ok = LEAK not in text
    masked += ok
    print(f"  {name:32s} {'masked' if ok else 'LEAKS'}")
print(f"\n  masked on {masked} of {len(probes)} paths, and every leak is at or "
      f"downstream of .get_secret_value()")
# =>   str(settings)                    masked
#      repr(settings)                   masked
#      f-string on the field            masked
#      logging '%s' of the model        masked
#      model_dump()                     masked
#      model_dump_json()                masked
#      json.dumps(dump, default=str)    masked
#      vars(settings)                   masked
#      pickle.dumps(settings)           LEAKS
#      a library echoing the value      LEAKS
#      .get_secret_value()              LEAKS
#
#      masked on 8 of 11 paths, and every leak is at or downstream of .get_secret_value()
```

Eight of eleven, and the three failures are more interesting than the eight successes because they are not gaps in the implementation — they are the boundary of what the technique can do. `SecretStr` wraps the value in an object whose `__repr__` and `__str__` lie, which covers every path that goes through Python's display machinery: printing, logging, f-strings, `repr`, `vars`, and pydantic's own serializers. That is genuinely most of them, and it is why the accidental leak — a stray `print(settings)` during debugging, a config object attached to an error report — mostly does not happen.

What it cannot cover is *use*. Authenticating requires the actual characters, so `get_secret_value()` must exist, and the moment it is called the result is an ordinary `str` with no protection whatsoever. The measured consequence is the tenth row: a library handed the unwrapped token raises `invalid literal for int() with base 10: 'tok-live-9f3c1a'` and the secret is now in an exception message, on its way to a log aggregator. `pickle` leaks for the same underlying reason — serialization must round-trip the real value or the object would be useless after loading.

The discipline that follows is about *where* rather than *whether*. Call `get_secret_value()` at the last possible moment, inside the smallest possible scope, and pass the wrapper everywhere else — because a wrapper travelling through your call graph is safe by default and a raw string is a liability at every frame it appears in. It also makes the audit tractable: `grep get_secret_value` enumerates every place in the codebase where the secret exists in plain form, which is a list you can actually read. **A masked type does not make a secret safe; it makes the unsafe moments greppable**, and on a real codebase that is the more valuable property.

## One protocol, three brokers, one decision stream

[Part V](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) promised that the `SimulatedBroker` box could be unplugged and replaced without any other component changing a line, and [Part VI](../part-06-live-infrastructure/01-system-architecture.md) collected on that promise by running the same strategy against two brokers. What neither did was write the promise down in a form a machine can check. A `Protocol` is that form: structural typing, so a broker conforms by having the right methods rather than by inheriting from anything, which matters because the broker you actually want to plug in is a vendor SDK you do not control.

```python
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

LAB = Path("lab/di")
LAB.mkdir(parents=True, exist_ok=True)


@runtime_checkable
class Broker(Protocol):
    def submit(self, ts, symbol: str, qty: int, price: float) -> None: ...
    def poll(self) -> list: ...


class SimulatedBroker:
    """Part V's broker: fills everything, immediately, at the price given."""
    def __init__(self):
        self.out = []

    def submit(self, ts, symbol, qty, price):
        self.out.append((ts, symbol, qty, round(price, 4)))

    def poll(self):
        out, self.out = self.out, []
        return out


class PaperBroker(SimulatedBroker):
    """Part VI's broker: the same seam, plus client-order-id de-duplication."""
    def __init__(self):
        super().__init__()
        self.seen = set()

    def submit(self, ts, symbol, qty, price):
        coid = f"{symbol}:{ts:%Y%m%d}"
        if coid in self.seen:
            return
        self.seen.add(coid)
        super().submit(ts, symbol, qty, price)


class RecordingBroker:
    """Wraps any broker and keeps an audit trail. Composition, not inheritance."""
    def __init__(self, inner: Broker):
        self.inner, self.calls = inner, []

    def submit(self, ts, symbol, qty, price):
        self.calls.append((ts, symbol, qty))
        self.inner.submit(ts, symbol, qty, price)

    def poll(self):
        return self.inner.poll()


def run(broker: Broker) -> str:
    """The strategy, which has never heard of any of these classes."""
    px = pd.read_parquet("data/prices.parquet")[["SPY", "TLT", "GLD"]]
    r = np.log(px).diff()
    sig = np.sign(r.rolling(252).sum()).shift(1)
    fills, last = [], {}
    for t in px.index:
        for s in px.columns:
            v = sig.at[t, s]
            if np.isnan(v) or v == last.get(s):
                continue
            last[s] = v
            broker.submit(t, s, int(v * 100), float(px.at[t, s]))
        fills += broker.poll()
    text = "|".join(f"{t:%Y%m%d}{s}{q:+d}@{p}" for t, s, q, p in fills)
    return f"{len(fills)} fills, digest {hashlib.sha256(text.encode()).hexdigest()[:12]}"


for name, b in [("SimulatedBroker", SimulatedBroker()),
                ("PaperBroker", PaperBroker()),
                ("RecordingBroker(Simulated)", RecordingBroker(SimulatedBroker()))]:
    print(f"  {name:28s} {run(b)}   isinstance -> {isinstance(b, Broker)}")

(LAB / "seam.py").write_text('''from typing import Protocol


class Broker(Protocol):
    def submit(self, ts, symbol: str, qty: int, price: float) -> None: ...
    def poll(self) -> list: ...


class VendorBroker:
    """Their SDK calls it send_order, and returns a status code."""
    def send_order(self, symbol: str, qty: int) -> int:
        return 0

    def poll(self) -> list:
        return []


def install(b: Broker) -> None: ...


install(VendorBroker())
''')
r = subprocess.run([sys.executable, "-m", "mypy", "--no-color-output",
                    "--no-error-summary", str(LAB / "seam.py")],
                   capture_output=True, text=True)
print("\n  mypy, on a vendor broker that does not fit the seam:")
for ln in r.stdout.splitlines():
    print("    " + ln.replace(str(LAB) + "/", ""))

sim, N = SimulatedBroker(), 200_000
t0 = time.perf_counter()
for _ in range(N):
    sim.submit(None, "SPY", 1, 1.0)
    sim.out.clear()
via_seam = time.perf_counter() - t0
buf = []
t0 = time.perf_counter()
for _ in range(N):
    buf.append((None, "SPY", 1, 1.0))
    buf.clear()
inline = time.perf_counter() - t0
print(f"\n  indirection costs {1e6 * (via_seam - inline) / N:.1f} us per submission "
      f"(machine-specific); Part V's engine makes 1,103 of them")
# =>   SimulatedBroker              329 fills, digest 9cf12cd92727   isinstance -> True
#      PaperBroker                  329 fills, digest 9cf12cd92727   isinstance -> True
#      RecordingBroker(Simulated)   329 fills, digest 9cf12cd92727   isinstance -> True
#
#      mypy, on a vendor broker that does not fit the seam:
#        seam.py:21: error: Argument 1 to "install" has incompatible type "VendorBroker"; expected "Broker"  [arg-type]
#        seam.py:21: note: "VendorBroker" is missing following "Broker" protocol member:
#        seam.py:21: note:     submit
#
#      indirection costs 0.2 us per submission (machine-specific); Part V's engine makes 1,103 of them
```

Three implementations, **329 fills and digest `9cf12cd92727` from every one of them**. The strategy function's only knowledge of brokers is the type annotation, and that annotation is not decoration: mypy reads it and reports that `VendorBroker` **is missing following "Broker" protocol member: submit** — at the call site, before anything runs, naming the exact method that is absent. That is the whole value proposition of structural typing here. Nobody can make the vendor's SDK inherit from your base class, but everybody can write a thirty-line adapter, and the type checker will tell them precisely when it is complete.

`RecordingBroker` is worth a second look because it demonstrates the property that makes protocols worth more than interfaces: it *wraps* a broker rather than being one, adding an audit trail by composition, and it satisfies the same protocol as the thing it wraps. That is how you get logging, rate limiting, the risk gauntlet from [Part VI](../part-06-live-infrastructure/05-resilience-and-risk-controls.md), and a dry-run mode without any of them appearing in the strategy, the sizer, or each other.

The cost is real and small. Dispatching through the seam runs about **0.2 microseconds per submission** on this machine — the absolute figure is machine-specific and only the order of magnitude should be trusted. Part V's engine makes 1,103 submissions across a quarter century, so the total is well under a millisecond against a backtest that takes about a second: the seam is free at this scale, and the decision is a pure win. The arithmetic only turns at rates this course does not trade at — around ten million calls a day the same overhead becomes seconds, which is where an execution system starts caring, and where [Profiling, Refactoring, and Versioning](05-profiling-refactoring-versioning.md) will insist that you measure rather than assume.

## Strategies are discovered, not imported

A platform whose core must be edited to add a strategy is a platform where every researcher's experiment is a pull request against shared infrastructure. Entry points invert that: a strategy declares itself in its own `pyproject.toml`, and the platform finds it by asking the installed environment what exists.

```python
import subprocess
import sys
from pathlib import Path

SITE = Path("lab/site-packages")
CODE = '''
from importlib.metadata import entry_points

registry = {}
for ep in sorted(entry_points(group="quantlib.strategies"), key=lambda e: e.name):
    registry[ep.name] = ep                       # nothing imported yet
print(f"    discovered {len(registry)} strategy(ies) without importing any of them:")
for name, ep in registry.items():
    print(f"      {name:10s} {ep.value}")
cls = registry["tsmom"].load()                   # imported on demand, here
strat = cls(lookback=126)
print(f"    loaded {cls.__name__} from {cls.__module__}, lookback {strat.lookback}")
'''
r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                   cwd="lab/platform", env={"PYTHONPATH": str(SITE.resolve()),
                                            "PATH": "/usr/bin:/bin"})
print("  the platform core contains no reference to any strategy:")
print(r.stdout.rstrip() or r.stderr[-400:])
# =>   the platform core contains no reference to any strategy:
#        discovered 1 strategy(ies) without importing any of them:
#          tsmom      quantlib.signals.tsmom:TSMom
#        loaded TSMom from quantlib.signals.tsmom, lookback 126
```

The declaration that produced this is three lines in `pyproject.toml`, written at the top of this lesson:

```toml
[project.entry-points."quantlib.strategies"]
tsmom = "quantlib.signals.tsmom:TSMom"
```

Two properties make this the right mechanism rather than a fashionable one. Discovery is separate from loading: `entry_points()` reads installed metadata, so a platform can list every available strategy, print it in a help message, or validate a config value against it **without importing a single strategy module**. That matters more than it sounds — a registry built by importing everything pays every strategy's import cost, including a stray `import torch`, on every startup including the ones that will not use it.

The second property is that the strategy does not have to live in your repository at all. A researcher's `momo-experiments` package, installed alongside the platform, appears in the registry with no edit to any shared file and no coordination with anyone. That is the concrete difference between a platform and a monolith with plugins bolted on: the core defines the *group name* and the interface, and stays ignorant of the membership. What the core must then do — and what an entry-point system does not do for you — is validate that what it loaded actually conforms, which is precisely the `Protocol` from the previous section applied to strategies instead of brokers, and the reason those two sections belong in the same lesson.

!!! warning "Structure is a claim about the future, and it is the cheapest thing here to get wrong"
    Every technique in this lesson costs something now and pays later: a `src/` directory adds a path component, a protocol adds an annotation, a schema adds bounds someone must choose, entry points add a metadata block. All of the payoffs are deferred, and all of the costs are immediate, which is why these are the decisions most often skipped under deadline and most expensive to retrofit. The hotfix that reached from `signals` into `execution` took one afternoon to write and would take a week to unpick after a year of code has been written against the tangle. Make the boundaries checkable early, when the check passes trivially and enforcing it costs nothing, because a layering rule introduced after the violations exist is not a rule — it is a backlog item.

!!! abstract "Key takeaways"
    - A thirty-line `ast` walk turns the architecture into a build gate. The clean package shows **no layering violations across 8 modules and 5 layers**; one plausible hotfix that reached downward produced **two violations**, neither of which any linter, type checker or test would have caught.
    - Under a flat layout `import quantlib` resolved to **0.2.0-dev in the working tree** while the installed **0.1.0** sat unused in site-packages. Everything declared in `pyproject.toml` but absent from the source tree is untested under that layout, and the wheel first gets exercised in production.
    - Four routes into the same configuration — kwargs, TOML, environment, and a file with an override — produce one identity, **`cdfb8444e110`**. Hashing `str(dict)` instead makes key order significant and yields two hashes for one config; sorted JSON fixes ordering but **`126.0` still hashes differently from `126`**, which only a schema resolves.
    - A lookback of `1` is legal Python, legal pandas, and a different strategy: the schema rejects it at startup, and without one **the pipeline runs to completion and returns a Sharpe of −0.5047** with no error at all. Bounds in the type are domain facts, not defensive programming.
    - `SecretStr` masked the token on **8 of 11 paths** — everything routed through Python's display machinery — and leaked on exactly the three at or downstream of `get_secret_value()`, including a library echoing the unwrapped value in an exception. A masked type makes the unsafe moments **greppable**, which is the property worth having.
    - Three broker implementations, including one that wraps another by composition, produced **329 fills and digest `9cf12cd92727`** apiece; mypy names the missing method — **"VendorBroker" is missing following "Broker" protocol member: submit** — at the call site, before anything runs.
    - The seam costs about **0.2 microseconds per submission**, and Part V's engine makes 1,103 of them: under a millisecond on a one-second backtest. The trade only turns around ten million calls a day.
    - Entry points let the platform **discover a strategy without importing it**, so listing what is available costs no import time and a researcher's own package joins the registry with no edit to any shared file.

## Where this goes next

The package now has boundaries a program can check, configuration that fails at the door, and seams that a type checker verifies. What it does not have is any answer to the question of how its pieces talk to each other when they stop sharing a process. Every component in this lesson communicated by function call — the strategy called `broker.submit`, the engine called `signal` — which works precisely as long as everything runs in one interpreter, on one machine, at one speed. The `RecordingBroker` hints at the difficulty: it exists because someone wanted the fills observed by something *other* than the strategy, and composition is the last mechanism that still works before the answer becomes a message.

[Architecture Patterns and Message Queues](04-architecture-patterns-and-message-queues.md) takes the seam and stretches it across processes. It weighs layered against event-driven designs on latency and testability, runs the same tsmom decision stream through Redis Streams to check that a broker in the middle changes nothing, and then attacks the assumption that has held throughout Part IX so far: that the consumer keeps up. When the producer is faster than the consumer, the choices are to buffer, to drop, or to conflate, and each has a measurable cost in latency, memory, and lost data. It closes on the guarantee everyone wants and nobody has, which is delivery exactly once.
