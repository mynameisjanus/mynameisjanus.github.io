# Testing and CI/CD

[Git and Code Review](01-git-and-code-review.md) ended by handing this lesson a problem it could not solve with its own tools. A single bad mark in October 2008 cut the book's Sharpe from 0.3025 to 0.0803 while leaving the decision digest byte-identical, so the golden test that caught a deleted `.shift(1)` waved the data corruption straight through. Review operates on changes; bisect operates on changes; neither knows what the answer is supposed to be. This lesson builds the thing that does.

It builds it the way a desk actually would, in tiers, and then measures what each tier is worth. A real pytest suite gets written and run. Hypothesis is pointed at a position sizer that looks obviously correct and finds a counterexample in under a second. A statistical assertion is priced, and the price turns out to be brutal — at fifty stochastic assertions, a suite that is green 99% of the time when nothing is broken catches a genuine three-sigma regression **less than a quarter of the time**. The Part V engine is frozen into `data/part9golden.parquet` and then attacked with four refactors a reviewer would wave through; **all four produce an identical tearsheet**, and only the golden file can tell them apart. Coverage reaches **100% on a function that still contains a lookahead bug**. And this repository's own CI grows the gates the lesson argues for — including one gate whose very first act was to fail on the code that implements it.

## The pyramid inverts when the unit under test is a quarter century of bars

The testing pyramid says many fast unit tests, fewer integration tests, fewest end-to-end tests, and it is drawn that way because of *cost*. In a trading system the cost gradient is steeper than in most software: a pure function on three floats runs in microseconds, a signal over 6,411 bars in milliseconds, and a full event-driven backtest in about a second — six orders of magnitude between the cheapest and dearest assertion. That gap is not a nuisance to be optimized away; it is the constraint that determines which tests run on every save, which on every push, and which only before a release.

Here is the package and the first two tiers, written as real files and run by real pytest:

```python
import shutil
import subprocess
import sys
from pathlib import Path

SUITE = Path("lab/suite")
shutil.rmtree(SUITE, ignore_errors=True)
(SUITE / "tests").mkdir(parents=True)

(SUITE / "quantlib.py").write_text('''"""Signal, sizer, fees, and Part V's event loop -- the unit under test."""
import numpy as np
import pandas as pd

ASSETS = ["SPY", "TLT", "GLD"]
HS, COMM = {"SPY": 0.5, "TLT": 1.0, "GLD": 1.0}, 0.2


def logret(px):
    return np.log(px[ASSETS]).diff()


def signal(px, lookback=252):
    """+1 long, -1 short, formed at today's close and traded tomorrow."""
    return np.sign(logret(px).rolling(lookback).sum()).shift(1)


def size(equity, prices, weights, max_gross=1.0):
    """Integer share counts for target weights, capped at max_gross of equity."""
    gross = sum(abs(w) for w in weights)
    scale = min(1.0, max_gross / gross) if gross else 0.0
    return [int(round(w * scale * equity / p)) for w, p in zip(weights, prices)]


def fee(qty, price, half_spread_bp, commission_bp):
    return round(abs(qty) * price * (half_spread_bp + commission_bp) * 1e-4, 2)


def engine(bars):
    """Signals at the close, orders on flips and month-ends, fills at next open."""
    d = {s: bars.xs(s, axis=1, level=1).dropna() for s in ASSETS}
    close = pd.DataFrame({s: v["Close"] for s, v in d.items()})
    open_ = pd.DataFrame({s: v["Open"] for s, v in d.items()})
    sig = np.sign(np.log(close).diff().rolling(252).sum())
    idx = close.index
    cash, pos, last, pending = 1_000_000.0, dict.fromkeys(ASSETS, 0), {}, []
    eq_s, cash_s = pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
    pos_s = pd.DataFrame(0, index=idx, columns=ASSETS, dtype="int64")
    for i, t in enumerate(idx):
        for s, tgt in pending:
            dq, o = tgt - pos[s], open_.at[t, s]
            if dq and not np.isnan(o):
                cash = round(cash - dq * o - fee(dq, o, HS[s], COMM), 2)
                pos[s] += dq
        pending = []
        eq = cash + sum(pos[s] * close.at[t, s] for s in ASSETS if pos[s])
        eq_s[t], cash_s[t] = eq, cash
        pos_s.loc[t] = [pos[s] for s in ASSETS]
        if i == len(idx) - 1:
            break
        live = [s for s in ASSETS if not np.isnan(sig.at[t, s])]
        for s in live:
            if sig.at[t, s] != last.get(s, 0.0) or t.month != idx[i + 1].month:
                pending.append((s, int(sig.at[t, s] * eq / len(live) / close.at[t, s])))
            last[s] = sig.at[t, s]
    out = pd.DataFrame({"equity": eq_s, "cash": cash_s})
    for s in ASSETS:
        out[f"pos_{s}"] = pos_s[s]
    out.index.name = "Date"
    return out
''')

(SUITE / "tests" / "test_units.py").write_text('''"""Tier one: pure functions. No data, no clock, no network."""
import numpy as np
import pandas as pd
import pytest

import quantlib as q


def frame(vals):
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.DataFrame({a: vals for a in q.ASSETS}, index=idx)


def test_signal_is_long_when_the_trend_is_up():
    assert q.signal(frame(np.linspace(100, 200, 300)), 10).iloc[-1].eq(1.0).all()


def test_signal_is_short_when_the_trend_is_down():
    assert q.signal(frame(np.linspace(200, 100, 300)), 10).iloc[-1].eq(-1.0).all()


def test_signal_never_uses_the_bar_it_trades_on():
    """The load-bearing test of the whole course."""
    px = frame(np.linspace(100, 200, 300))
    before = q.signal(px, 10)
    px2 = px.copy()
    px2.iloc[-1] *= 10.0                     # rewrite only the final close
    assert before.equals(q.signal(px2, 10))


def test_fee_is_rounded_to_the_penny():
    assert q.fee(1000, 81.2537, 0.5, 0.2) == 5.69


@pytest.mark.parametrize("w", [0.0, 1.0, -1.0])
def test_sizer_respects_direction(w):
    qty, = q.size(1_000_000, [100.0], [w])
    assert np.sign(qty) == np.sign(w)
''')

(SUITE / "tests" / "test_regressions.py").write_text('''"""Tier two: one test per defect this course actually shipped and corrected."""
import numpy as np
import pandas as pd

import quantlib as q


def test_lookahead_regression():
    """With the shift, 0.30. Without it, 1.07. Pin the honest one forever."""
    px = pd.read_parquet("data/prices.parquet")
    book = (q.signal(px) * q.logret(px)).mean(axis=1).dropna()
    assert round(float(np.sqrt(252) * book.mean() / book.std()), 2) == 0.30


def test_window_is_counted_in_rows_not_calendar_days():
    """rolling('365D') is a different strategy that scores 0.23."""
    px = pd.read_parquet("data/prices.parquet")
    r = q.logret(px)
    rows = (np.sign(r.rolling(252).sum()).shift(1) * r).mean(axis=1).dropna()
    days = (np.sign(r.rolling("365D").sum()).shift(1) * r).mean(axis=1).dropna()
    assert len(rows) != len(days)
    assert abs(float(np.sqrt(252) * rows.mean() / rows.std()) - 0.3025) < 5e-4
''')

(SUITE / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")

env = {"PYTHONPATH": str(SUITE.resolve()), "PATH": "/usr/bin:/bin",
       "HOME": str(Path.home())}
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "-c", str(SUITE / "pytest.ini"), str(SUITE / "tests")],
                   capture_output=True, text=True, env=env)
# strip the wall-clock number: it is machine-specific and must not be pinned
summary = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
print("  " + summary[-1].split(" in ")[0])

sys.path.insert(0, str(SUITE.resolve()))
import pandas as pd                                   # noqa: E402
import quantlib as q                                  # noqa: E402
px = pd.read_parquet("data/prices.parquet")
print(f"\n  what each tier actually touches")
print(f"    unit tests      3 floats, no I/O")
print(f"    regression      {len(px):,} bars x {len(q.ASSETS)} assets")
print(f"    engine gate     {len(pd.read_parquet('data/part5.parquet')):,} bars, "
      f"one order at a time")
# =>   9 passed
#
#      what each tier actually touches
#        unit tests      3 floats, no I/O
#        regression      6,411 bars x 3 assets
#        engine gate     6,611 bars, one order at a time
```

Nine tests, all green, and the shape of the suite matters more than the count. The first tier asserts things about *pure functions* — a fee rounds to the penny, a sizer preserves direction — and it is the tier that can afford to be exhaustive because nothing in it touches a file. The second tier is where this course's own history lives: `test_lookahead_regression` pins the Sharpe at 0.30, which is precisely the assertion that would have turned the previous lesson's *Simplify the signal path* commit red on the developer's own machine, minutes after they wrote it, instead of two weeks later under bisect.

One test in the first tier deserves separate mention because it is the closest thing this course has to a load-bearing assertion. `test_signal_never_uses_the_bar_it_trades_on` rewrites *only the final close*, multiplies it by ten, and demands that the signal series be unchanged. It does not check a number; it checks a *property of the information flow*, and it is the only test in the suite that would survive a total rewrite of the strategy. Any signal implementation that passes it cannot be looking forward, whatever else it does wrong. Tests that assert values go stale the moment the strategy improves. Tests that assert causality never do.

## Hypothesis finds the input you would not have thought of

Every test above shares a weakness: a human chose the inputs. An up-trend, a down-trend, a thousand-dollar equity, three plausible prices — the cases that came to mind, which are by construction the cases the author already had in mind when writing the code. Property-based testing inverts the burden. You state an *invariant* that must hold for all inputs, and the library hunts for a violation, then shrinks whatever it finds to the smallest example that still breaks.

The sizer's invariant is the kind of thing a risk officer would write on a whiteboard: the gross notional it returns must never exceed the equity it was given.

```python
import shutil
import subprocess
import sys
from pathlib import Path

SUITE = Path("lab/suite")
(SUITE / "tests" / "test_properties.py").write_text(
    '''"""Tier three: invariants over all inputs, not the three you thought of."""
from hypothesis import HealthCheck, given, settings, strategies as st

import quantlib as q

PRICES = st.lists(st.floats(0.5, 5000, allow_nan=False, allow_infinity=False),
                  min_size=1, max_size=4)
WEIGHTS = st.lists(st.floats(-1, 1, allow_nan=False, allow_infinity=False),
                   min_size=1, max_size=4)


@settings(derandomize=True, database=None, max_examples=2000,
          suppress_health_check=[HealthCheck.filter_too_much])
@given(equity=st.floats(1e3, 1e8, allow_nan=False, allow_infinity=False),
       prices=PRICES, weights=WEIGHTS)
def test_gross_never_exceeds_equity(equity, prices, weights):
    n = min(len(prices), len(weights))
    prices, weights = prices[:n], weights[:n]
    qty = q.size(equity, prices, weights)
    gross = sum(abs(x * p) for x, p in zip(qty, prices))
    assert gross <= equity
''')

env = {"PYTHONPATH": str(SUITE.resolve()), "PATH": "/usr/bin:/bin",
       "HOME": str(Path.home())}
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "--no-header", "-c", str(SUITE / "pytest.ini"),
                    str(SUITE / "tests" / "test_properties.py")],
                   capture_output=True, text=True, env=env)
keep = False
for ln in r.stdout.splitlines():
    if ln.startswith("equity ="):
        keep = True
    if keep and (ln.startswith("E ") or ln.startswith("equity =")):
        print("  " + ln.rstrip())
    if " passed" in ln or " failed" in ln:
        print("  " + ln.split(" in ")[0])
# =>   equity = 1787.0, prices = [1.0, 1.0], weights = [1.0, 1.0]
#      E       assert 1788.0 <= 1787.0
#      E       Failing test case: test_gross_never_exceeds_equity(
#      E           equity=1787.0,
#      E           prices=[1.0, 1.0],
#      E           weights=[1.0, 1.0],
#      E       )
#      1 failed
```

Two assets, both priced at a dollar, both wanted at full weight, and $1,787 of equity. The sizer scales the weights to a half each, asks for $893.50 of a $1.00 stock twice, rounds each to 894 shares, and returns a book worth **$1,788 against $1,787 of equity**. It is a one-dollar breach — 5.6 basis points of unintended leverage — and it is exactly the sort of finding a human test author never produces, because nobody sits down and thinks to try $1,787.

The magnitude is not the point; the *class* is. `int(round(...))` rounds half away from the target on both legs simultaneously, so the error compounds with the number of positions rather than cancelling. On a three-asset book at $1,000 equity the breach stays around a dollar, but the same code sizing a two-hundred-name portfolio against a hard regulatory gross limit breaches it on essentially every rebalance, and the breach is invisible in any summary statistic because it is a rounding artifact rather than a position error. The fix is to round *toward* zero with `int()` — which is what [Part V's engine](../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) does, and now we know why that mattered rather than merely that it was conventional.

Two settings in that block are doing quiet work. `derandomize=True` makes Hypothesis explore the same sequence every run, and `database=None` stops it caching counterexamples between runs; without both, the test would be a different test on every invocation and the pinned output above could not exist. That is the general rule this section trades for the next one: **a test that hunts randomly must be told which random hunt to perform**, or it becomes the flakiness it was meant to detect.

## A statistical assertion is a coin the suite flips on every build

Some things genuinely cannot be asserted deterministically. A Monte Carlo estimate, a bootstrap interval, a stochastic optimizer's converged value, a strategy's Sharpe over resampled paths — all of these are random variables, and the honest way to test them is a statistical test: assert the observed value lies within some confidence band of the expected one. The difficulty is that a test suite runs this assertion over and over, and a confidence band is a false-alarm rate by construction.

```python
import numpy as np
from scipy import stats

K = 50                                    # stochastic assertions in the suite
rng = np.random.default_rng(42)

print("     alpha    P(suite red | nothing broken)     z    power vs 1 sigma  vs 3 sigma")
for a in (0.05, 0.01, 0.001, 1 - 0.99 ** (1 / K), 1e-6):
    p_red = 1 - (1 - a) ** K
    z = stats.norm.ppf(1 - a / 2)
    pw = [stats.norm.sf(z - d) + stats.norm.cdf(-z - d) for d in (1.0, 3.0)]
    print(f"  {a:8.6f}            {p_red:6.1%}              {z:4.2f}       "
          f"{pw[0]:6.1%}       {pw[1]:6.1%}")

runs = 20_000
red = (rng.random((runs, K)) < 0.05).any(axis=1).mean()
print(f"\n  {runs:,} simulated CI runs on unbroken code, alpha=0.05, K={K}: "
      f"{red:.1%} went red")
print(f"  analytic 1 - 0.95^{K} = {1 - 0.95 ** K:.1%}")
# =>      alpha    P(suite red | nothing broken)     z    power vs 1 sigma  vs 3 sigma
#      0.050000             92.3%              1.96        17.0%        85.1%
#      0.010000             39.5%              2.58         5.8%        66.4%
#      0.001000              4.9%              3.29         1.1%        38.6%
#      0.000201              1.0%              3.72         0.3%        23.6%
#      0.000001              0.0%              4.89         0.0%         2.9%
#
#      20,000 simulated CI runs on unbroken code, alpha=0.05, K=50: 92.5% went red
#      analytic 1 - 0.95^50 = 92.3%
```

Fifty assertions at the reflexive 95% confidence level make the suite fail **92.3% of the time on code where nothing whatsoever is wrong**, and the simulation confirms it at 92.5%. This is not a subtle effect and it is the mechanical origin of most flaky suites: not race conditions, not test pollution, just the multiple-comparisons arithmetic that [Part III](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) applied to strategy selection, arriving unannounced in the build pipeline. A team that meets this reaches for the obvious remedy — tighten the threshold — and the last three columns are the bill.

Read the row that a sane engineering policy would actually pick. To be green 99% of the time on unbroken code, α must fall to **0.000201**, a 3.72-sigma band. At that threshold the suite detects a one-sigma regression **0.3%** of the time and a genuine **three-sigma regression only 23.6%** of the time. Three sigma is not a subtle degradation — on a Sharpe estimate it is the difference between a strategy working and not — and the suite that never cries wolf will miss it three times in four. **There is no setting of α that is both non-flaky and sensitive**, because the two requirements are the same knob turned in opposite directions, and no amount of engineering effort moves the trade-off; only reducing K, or increasing the sample each assertion is computed from, changes the geometry at all.

The practical resolution is to stop writing statistical assertions wherever a deterministic one will do. Seed the generator and assert the exact value, as the previous section's `derandomize=True` did — then K falls, and the handful of genuinely stochastic assertions that remain can afford a wide band. Where a statistic must be tested, test it on the largest sample you can afford and against a threshold derived from a stated false-alarm budget rather than from habit. And when a stochastic test does go red, the correct first response is to *rerun it before investigating*, which is an admission that the assertion is weak evidence and should be treated as such.

## The golden file is the only test that knows what the backtest used to say

Unit tests check functions, property tests check invariants, and neither notices when a refactor quietly changes what a quarter century of trading produced. That requires a *golden file*: the full output of a known-good run, frozen, and compared against on every change. Part IX's frozen artifact is that file — the entire per-bar state of Part V's engine, equity, cash, and positions, 6,411 rows of it.

The freeze has to reconcile against what the course already published, or it is just a number in a new file:

```python
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("lab/suite").resolve()))
import quantlib as q

bars = pd.read_parquet("data/part5.parquet")
golden = q.engine(bars)
golden.to_parquet("data/part9golden.parquet")

close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                      for s in q.ASSETS})
lr = np.log(close).diff()
vec = (np.sign(lr.rolling(252).sum()).shift(1) * lr).mean(axis=1).dropna()
er = golden.equity.pct_change().loc[vec.index[0]:].dropna()
eq = golden.equity.loc[vec.index[0]:]
dd = eq / eq.cummax() - 1
last = golden.iloc[-1]
sha = hashlib.sha256(open("data/part9golden.parquet", "rb").read()).hexdigest()[:12]

print(f"  frozen data/part9golden.parquet  {golden.shape[0]:,} rows x "
      f"{golden.shape[1]} cols  sha256 {sha}")
print(f"  final equity ${eq.iloc[-1]:,.2f}   Part V published $2,522,514.08")
print(f"  Sharpe {np.sqrt(252) * er.mean() / er.std():.2f}   maxDD {dd.min():.1%}"
      f"        Part V published 0.38, -27.3%")
print(f"  final book SPY {int(last.pos_SPY):,} TLT {int(last.pos_TLT):,} "
      f"GLD {int(last.pos_GLD):,}   Part V published 1429, -10358, 2737")
# =>   frozen data/part9golden.parquet  6,411 rows x 5 cols  sha256 d15da35652e7
#      final equity $2,522,514.08   Part V published $2,522,514.08
#      Sharpe 0.38   maxDD -27.3%        Part V published 0.38, -27.3%
#      final book SPY 1,429 TLT -10,358 GLD 2,737   Part V published 1429, -10358, 2737
```

Every published figure reproduces: the equity to the cent, the Sharpe, the drawdown, and the final book share for share. That reconciliation is what makes the file admissible as a reference; a golden file that cannot be traced to a result someone already defended is a snapshot of a bug waiting to be canonized.

Now the interesting part. Four changes get applied to the engine, each of which a competent reviewer would approve without comment, and each is compared against the frozen file *and* against the tearsheet a human would actually look at:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("lab/suite").resolve()))
import quantlib as q

bars = pd.read_parquet("data/part5.parquet")
gold = pd.read_parquet("data/part9golden.parquet")
close = pd.DataFrame({s: bars.xs(s, axis=1, level=1).dropna()["Close"]
                      for s in q.ASSETS})
lr = np.log(close).diff()
first = (np.sign(lr.rolling(252).sum()).shift(1) * lr).mean(axis=1).dropna().index[0]

def variant(dtype=float, penny=True, sizer=int):
    """The same engine with one 'harmless' knob turned."""
    d = {s: bars.xs(s, axis=1, level=1).dropna() for s in q.ASSETS}
    op = pd.DataFrame({s: v["Open"] for s, v in d.items()})
    sig = np.sign(np.log(close).diff().rolling(252).sum())
    idx = close.index
    cash, pos, lastsig, pending = dtype(1_000_000.0), dict.fromkeys(q.ASSETS, 0), {}, []
    eq_s = pd.Series(np.nan, index=idx)
    fills = 0
    for i, t in enumerate(idx):
        for s, tgt in pending:
            dq, o = tgt - pos[s], dtype(op.at[t, s])
            if dq and not np.isnan(o):
                f = abs(dq) * o * (q.HS[s] + q.COMM) * 1e-4
                f = round(f, 2) if penny else f
                cash = cash - dq * o - f
                cash = round(cash, 2) if penny else cash
                pos[s] += dq
                fills += 1
        pending = []
        eq = cash + sum(pos[s] * dtype(close.at[t, s]) for s in q.ASSETS if pos[s])
        eq_s[t] = float(eq)
        if i == len(idx) - 1:
            break
        live = [s for s in q.ASSETS if not np.isnan(sig.at[t, s])]
        for s in live:
            if sig.at[t, s] != lastsig.get(s, 0.0) or t.month != idx[i + 1].month:
                pending.append((s, sizer(sig.at[t, s] * eq / len(live) / close.at[t, s])))
            lastsig[s] = sig.at[t, s]
    return eq_s, fills

print("  refactor                          fills   final equity    Sharpe   maxDD"
      "     golden file")
for name, kw in [("none -- the frozen run", {}),
                 ("extract helpers, no logic change", {}),
                 ("accumulate cash in float32", {"dtype": np.float32}),
                 ("drop the round(.., 2) calls", {"penny": False}),
                 ("size with round() not int()", {"sizer": lambda x: int(round(x))})]:
    e, fills = variant(**kw)
    er = e.pct_change().loc[first:].dropna()
    w = e.loc[first:]
    dd = (w / w.cummax() - 1).min()
    gap = (e - gold.equity).abs().max()
    verdict = "PASS" if gap == 0.0 else f"FAIL by ${gap:,.2f}"
    print(f"  {name:33s} {fills:5d} ${w.iloc[-1]:>13,.2f}    "
          f"{np.sqrt(252) * er.mean() / er.std():.2f}   {dd:.1%}   {verdict}")
# =>   refactor                          fills   final equity    Sharpe   maxDD     golden file
#      none -- the frozen run             1103 $ 2,522,514.08    0.38   -27.3%   PASS
#      extract helpers, no logic change   1103 $ 2,522,514.08    0.38   -27.3%   PASS
#      accumulate cash in float32         1103 $ 2,522,519.25    0.38   -27.3%   FAIL by $14.31
#      drop the round(.., 2) calls        1103 $ 2,522,514.06    0.38   -27.3%   FAIL by $0.12
#      size with round() not int()        1103 $ 2,522,816.06    0.38   -27.3%   FAIL by $337.12
```

Read the middle four columns first, because they are the ones a human would look at. **Every variant produces 1,103 fills, a Sharpe of 0.38, and a maximum drawdown of −27.3%.** A tearsheet cannot distinguish these runs. A code review cannot distinguish them either — one of them is a genuine no-op, one loses precision, one abandons the penny discipline [Part V's portfolio accounting](../part-05-backtesting-engine/02-portfolio-accounting.md) built deliberately, and one silently changes how many shares the strategy buys. Only the last column separates them, and it does so because it compares **every bar** rather than a summary.

The last variant is the one that should worry you most, and it is the one this lesson already met. Sizing with `round()` instead of `int()` is the same defect Hypothesis found two sections ago, arriving now as a P&L difference of **$337.12** — and it is a real behaviour change, not a rounding artifact: the strategy holds different share counts for twenty-four years. It shows up in the golden file on **2001-01-30**, the twentieth trading day, and it never shows up in the tearsheet at all.

Which raises the tolerance question, and there is no comfortable answer. A byte-exact comparison flags float32 accumulation, which is arguably a legitimate performance decision rather than a bug. Loosening the tolerance enough to admit it — **$14.31, or 0.06 basis points of the book** — leaves you within a factor of twenty of the sizing change at **1.26 basis points**, which is unambiguously a different strategy. **A tolerance wide enough to permit harmless numerical drift is nearly wide enough to hide a real change in what you trade**, and the resolution is not a cleverer tolerance but a different policy: compare exactly, expect the comparison to fail on deliberate changes, and require that every failure be explained and the golden file *re-frozen in the same commit* that causes it. The diff then becomes the review artifact — a reviewer sees "final equity moved $337.12 because the sizer now rounds" and can accept or reject that on its merits. Silent tolerance turns that conversation into a setting nobody remembers choosing.

## Coverage measures which lines ran, not which bugs are absent

Coverage is the metric most often mistaken for a measure of test quality, and trading code is where the mistake is most expensive. The demonstration is short. Take a signal function with the lookahead bug — the exact defect that cost Part IX's first lesson two weeks of history — and write the tests a careful engineer would write for it:

```python
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path("lab/cov")
shutil.rmtree(LAB, ignore_errors=True)
LAB.mkdir(parents=True)

(LAB / "signals.py").write_text('''"""Every line of this module is exercised by the suite beside it."""
import numpy as np


def logret(px):
    return np.log(px).diff()


def trend(px, lookback=252):
    r = logret(px)
    s = r.rolling(lookback).sum()
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    return np.sign(s)                 # <- the bug: no .shift(1)
''')

(LAB / "test_signals.py").write_text('''import numpy as np
import pandas as pd
import pytest

import signals


def px(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_up_trend_is_long():
    assert signals.trend(px(np.linspace(100, 200, 60)), 10).iloc[-1] == 1.0


def test_down_trend_is_short():
    assert signals.trend(px(np.linspace(200, 100, 60)), 10).iloc[-1] == -1.0


def test_logret_is_additive():
    p = px([100.0, 110.0, 121.0])
    assert abs(signals.logret(p).sum() - np.log(1.21)) < 1e-12


def test_short_lookback_is_rejected():
    with pytest.raises(ValueError):
        signals.trend(px([1.0, 2.0, 3.0]), 1)
''')

env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
       "PYTHONPATH": str(LAB.resolve())}
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "--cov=signals", "--cov-report=term-missing", "test_signals.py"],
                   capture_output=True, text=True, env=env, cwd=str(LAB))
for ln in r.stdout.splitlines():
    if ln.startswith(("signals.py", "TOTAL", "Name")) or " passed" in ln:
        print("  " + ln.split(" in ")[0])

sys.path.insert(0, str(LAB.resolve()))
import signals
p = pd.Series(np.linspace(100, 200, 60),
              index=pd.bdate_range("2020-01-01", periods=60))
p2 = p.copy()
p2.iloc[-1] *= 10.0
moved = int((signals.trend(p, 10) != signals.trend(p2, 10)).sum())
print(f"\n  rewriting only the final close moves {moved} signal values; "
      f"a leak-free signal moves 0")
# =>   Name         Stmts   Miss  Cover   Missing
#      signals.py       9      0   100%
#      TOTAL            9      0   100%
#      4 passed
#
#      rewriting only the final close moves 10 signal values; a leak-free signal moves 0
```

**100% line coverage, every test green, and the function looks forward.** Nine statements, nine executed, zero missed — and the last line of the block, four lines of code that no coverage tool would ever prompt you to write, exposes the defect immediately: rewriting a single future close changes ten of the signal's values.

The reason is structural rather than accidental. Coverage answers "did this line execute", and every line did. It cannot answer "was the output correct", because it has no notion of correctness; and it especially cannot answer "did information flow backwards in time", because that is a relationship *between* runs rather than a property of any single execution. The lookahead bug is invisible to coverage in principle, not by oversight — no threshold, however high, would have caught it. This generalizes past leakage: coverage is blind to every defect whose signature is a wrong value rather than an unexecuted branch, which in numerical code is most of them.

That is not an argument against measuring coverage. It is a good detector of *untested* code, and a module sitting at 30% is telling you something true. It is an argument against coverage as a *gate*, because a coverage target is satisfiable by tests that assert nothing, and a team held to 90% will reliably produce them. The useful discipline is the one this lesson has been building toward: assert properties that constrain the information flow, pin the outputs that a refactor must not change, and treat coverage as a map of where you have not looked rather than a certificate about where you have.

## The pipeline that gates the merge

A gate that runs on your machine is a habit; a gate that runs on every push is a policy. This repository now has one, and building it surfaced the honest constraint that shapes what any real pipeline can promise. The workflow runs three jobs in sequence, cheapest first:

```yaml
jobs:
  # Cheap gates first: nothing reaches the build until the tree lints, the
  # gate's own tests pass, and every lesson is structurally intact.
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements-dev.txt
      # scripts/migrate_math_pages.py is a one-off Jekyll converter kept for
      # provenance (see README) and is excluded rather than retro-fitted.
      - name: Lint the maintained Python
        run: ruff check hooks/ scripts/check_docs.py tests/
      - name: Test the structure gate
        run: pytest tests/ -q
      - name: Check course structure
        run: python scripts/check_docs.py docs

  build:
    needs: check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v5
        with:
          path: site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v5
```

The `needs:` edges are the whole design: `deploy` cannot run unless `build` succeeded, and `build` cannot start unless `check` passed. Ordering by cost is not an aesthetic preference — a lint error should cost seconds to discover, not the four minutes a full build and deploy would take. Running the three gates locally is exactly what the pipeline does:

```python
import subprocess
import sys

GATES = [("ruff", [sys.executable, "-m", "ruff", "check", "hooks/",
                   "scripts/check_docs.py", "tests/"]),
         ("pytest", [sys.executable, "-m", "pytest", "tests/", "-q",
                     "-p", "no:cacheprovider"]),
         ("structure", [sys.executable, "scripts/check_docs.py", "docs"])]

for name, cmd in GATES:
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
    print(f"  {name:10s} rc={r.returncode}  {tail.split(' in ')[0]}")
# =>   ruff       rc=0  All checks passed!
#      pytest     rc=0  11 passed
#      structure  rc=0  checked 255 markdown files, 0 problem(s)
```

Three details in that pipeline are worth stating plainly, because each was a decision rather than a default. The lint gate covers `hooks/`, `scripts/check_docs.py` and `tests/`, and **explicitly excludes `scripts/migrate_math_pages.py`** — a one-off Jekyll converter kept for provenance, which ruff flags for an unused variable and a missing executable bit. Retro-fitting a lint standard onto a historical artifact is work that buys nothing; excluding it with a comment naming the reason is honest, and an exclusion list that grows without comments is how lint gates die. The gate also earned its keep immediately and at its author's expense: the first thing it flagged was an unnecessary `# noqa: E402` in the test file written to test the gate itself.

Second, `scripts/check_docs.py` is a real program with real logic — it distinguishes the house's trailing `# =>` pinned-output blocks from inline `# =>` comments, and exempts the appendix, which is migrated mathematics with a different shape — so it gets its own eleven tests. A gate nobody tests is a gate that fails open, and a docs checker that silently stops finding problems is indistinguishable from a clean tree.

Third, and most important: **the backtest gate is not in this pipeline, and cannot be.** The golden-file test needs `data/part5.parquet` and `data/part9golden.parquet`, and `data/` is gitignored — the caches are large, they are vendor data, and every part of this course freezes them locally by design. CI therefore verifies that the tree lints, that the structure gate's own tests pass, that every lesson is structurally intact, and that the site builds under `--strict`. It does *not* verify that a single number on any page is still true. That gap is the exact shape of the compromise most teams make without noticing, and the honest way to hold it is to name it: the checks that need production data run where the data lives, on a schedule, and the pipeline that gates the merge makes a narrower promise than its green checkmark implies. A green build here means the site is well-formed. It does not mean the backtest still says 0.38.

!!! warning "A green suite is evidence about the tests, not about the code"
    Every result in this lesson is a way for a passing suite to be wrong. Nine tests went green on a sizer that breaches its gross limit. One hundred percent coverage went green on a function that looks forward in time. A tearsheet showing 1,103 fills, a Sharpe of 0.38 and a −27.3% drawdown was produced by four different engines, three of which had been silently altered. And a suite of fifty stochastic assertions goes red 92% of the time on code where nothing is wrong, which trains a team to ignore exactly the signal the suite exists to send. The question to ask of any test is not whether it passes but what would have to break for it to fail — and if the answer is "nothing that has ever actually gone wrong here", the test is decoration.

!!! abstract "Key takeaways"
    - The cost gradient across tiers spans roughly **six orders of magnitude** — microseconds for a pure function, about a second for a full backtest over 6,611 bars — and that gradient, not the pyramid diagram, is what decides which gate runs on every push.
    - The most durable test in the suite asserts causality rather than value: rewriting *only the final close* must leave the signal unchanged. Tests that pin numbers go stale when the strategy improves; **a test that pins the direction of information flow never does**.
    - Hypothesis broke the sizer's gross-exposure invariant in under a second with **equity $1,787 and two $1.00 assets, returning a book worth $1,788** — 5.6 bp of unintended leverage from `int(round(...))` rounding both legs away from target at once.
    - Fifty stochastic assertions at α = 0.05 turn the suite red **92.3% of the time on unbroken code** (simulated: 92.5%). Tightening α to 0.000201 buys a 99% green rate and costs almost all the sensitivity: a genuine **three-sigma regression is caught 23.6% of the time**. No α is both non-flaky and sensitive.
    - The golden file reconciles exactly to Part V — **$2,522,514.08, Sharpe 0.38, maxDD −27.3%, final book 1,429 / −10,358 / 2,737** — which is what makes it admissible as a reference rather than a snapshot of an unexamined run.
    - Four "harmless" refactors produced **identical tearsheets — 1,103 fills, 0.38, −27.3% for every one** — while the golden file separated them at **$0.00, $14.31, $0.12 and $337.12**. And a tolerance wide enough to admit float32 drift (**0.06 bp**) is within a factor of twenty of one that would hide a genuine sizing change (**1.26 bp**), so compare exactly and re-freeze the file in the commit that moves it.
    - **100% line coverage, four passing tests, and the signal still looks forward** — rewriting one future close moves ten of its values. Coverage cannot see leakage in principle, because information flow is a relationship between runs rather than a property of one execution.
    - The new pipeline gates lint, its own gate's tests, course structure and `mkdocs build --strict`, but **cannot run the backtest gate at all**, because `data/` is gitignored. A green check here means the site is well-formed, not that any number on it is still true.

## Where this goes next

The suite in `lab/suite` is now doing real work, and it is also becoming difficult to reason about for a reason that has nothing to do with testing. `quantlib.py` holds the signal, the sizer, the fee model and the entire event loop in one file; the tests reach it through a `sys.path` insertion; the golden file is addressed by a hard-coded relative path that only resolves if you happen to run from the repository root; and the engine reads `data/part5.parquet` by name from inside a function that ought not to know where data lives. None of this is a testing problem, but all of it makes tests harder to write — every one of the blocks above had to manipulate `PYTHONPATH` before it could import anything.

That is the subject of [Package Structure, Configuration, and Dependency Injection](03-package-structure-config-di.md). It turns the scratch module into an installable package with enforced boundaries, replaces the hard-coded paths with layered configuration that fails at startup instead of halfway through a resample, and puts the broker and the data feed behind interfaces so the engine can be tested against a fake without touching a parquet file at all. The payoff is measured in the same currency this lesson used: the fill digest must come out identical across three different broker implementations, and the import graph must be checkable by a program rather than by a convention nobody enforces.
