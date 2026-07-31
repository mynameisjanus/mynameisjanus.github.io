# Git and Code Review

[Drawdowns, Tail Risk, and Stress Testing](../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) closed Part VIII with a book that had been cut down rather than built up — two uncorrelated sleeves, sized under Kelly, allocated by risk, carrying a trend overlay against the slow crises. Every one of those decisions is a claim about a number, and every one of those numbers came out of code. That lesson's closing paragraph named the constraint that now binds: a risk report computed from a stale parquet file, a rebalancing rule whose thresholds were edited in production, an optimizer whose constraint set differs between research and live. None of those failures is detectable by any method in Part VIII, because none of them is a statistical error. They are engineering errors, and they are silent.

This lesson is about the two practices that catch them before money does: a history you can interrogate, and a second pair of eyes that knows what to look for. Both get measured rather than asserted. A notebook and a module receive the identical one-constant edit and their diffs are compared byte for byte; a fifteen-commit repository is built with one behaviour-changing commit hidden inside it and `git bisect` is asked to find it; six defects are planted in reviewable code and each is priced in Sharpe ratio; and the run manifest from [Logging and Configuration Management](../part-02-python/07-logging-and-config.md) is extended until it can actually settle an argument. Two of the results are uncomfortable. The defect that does the most damage per character changed arrives in a commit whose message reads like a cleanup, and the single most destructive data corruption in this lesson leaves the strategy's entire decision stream **byte-identical** — which means the obvious test for it passes.

!!! note "Versions"
    Part IX assumes Python 3.12+ with the Part III stack (NumPy 2.x, pandas 3.x, SciPy) plus pytest 9.1, Hypothesis 6.164, coverage 7.15, ruff 0.16, mypy 2.3, py-spy 0.4, and pydantic 2.13 with pydantic-settings 2.14; the examples were verified with those versions on Python 3.12.3 and git 2.43. Every block in this part writes into a `lab/` directory at the repository root — add it to `.gitignore` alongside `data/`, because nothing in it is a deliverable. Commit hashes reproduce exactly because every block pins `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`; if yours differ, that pinning is what you are missing.

## A notebook records that something changed, never what

The rule that notebooks are for looking and modules are for running has been stated more than once in this course without ever being priced. It is priceable. Take one research artifact in two forms — a notebook with its outputs stored, and a module holding the same constant — apply the identical edit to both, and read what version control has to show a reviewer afterwards:

```python
import base64
import io
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("lab/nbdiff")
ENV = {"GIT_AUTHOR_NAME": "Quant Lab", "GIT_AUTHOR_EMAIL": "lab@example.com",
       "GIT_COMMITTER_NAME": "Quant Lab", "GIT_COMMITTER_EMAIL": "lab@example.com",
       "GIT_AUTHOR_DATE": "2025-04-01T09:00:00+00:00",
       "GIT_COMMITTER_DATE": "2025-04-01T09:00:00+00:00", "PATH": "/usr/bin:/bin"}

px = pd.read_parquet("data/prices.parquet")
A = ["SPY", "TLT", "GLD"]
rets = np.log(px[A]).diff()

def cell_outputs(lookback):
    """Exactly what re-running the notebook would store: a number and a picture."""
    r = (np.sign(rets.rolling(lookback).sum()).shift(1) * rets).mean(axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot((1 + r).cumprod())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    return (f"Sharpe {np.sqrt(252) * r.mean() / r.std():.4f}, n {len(r)}\n",
            base64.b64encode(buf.getvalue()).decode())

def notebook(lookback, run_no):
    text, png = cell_outputs(lookback)
    return json.dumps({"cells": [
        {"cell_type": "code", "execution_count": run_no, "metadata": {},
         "source": [f"LOOKBACK = {lookback}\n", "sig = signal(px, LOOKBACK)\n"],
         "outputs": [{"name": "stdout", "output_type": "stream", "text": [text]}]},
        {"cell_type": "code", "execution_count": run_no + 1, "metadata": {},
         "source": ["plot_equity(sig)\n"],
         "outputs": [{"data": {"image/png": png}, "metadata": {},
                      "output_type": "display_data"}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n"

MODULE = 'LOOKBACK = {lookback}\n\n\ndef book(px):\n    return signal(px, LOOKBACK)\n'

def commit(msg):
    subprocess.run(["git", "add", "-A"], cwd=REPO, env=ENV, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO, env=ENV, check=True,
                   capture_output=True)

shutil.rmtree(REPO, ignore_errors=True)
REPO.mkdir(parents=True)
subprocess.run(["git", "init", "-q", "-b", "main"], cwd=REPO, env=ENV, check=True,
               capture_output=True)
for lb, run_no, msg in [(252, 1, "Research at a 252-day lookback"),
                        (126, 3, "Try a 126-day lookback"),
                        (126, 5, "Re-run, no code change")]:
    (REPO / "research.ipynb").write_text(notebook(lb, run_no))
    (REPO / "research.py").write_text(MODULE.format(lookback=lb))
    commit(msg)

def payload(rev_a, rev_b, path):
    """Lines touched, and the bytes a reviewer would have to read."""
    d = subprocess.run(["git", "diff", "-U0", rev_a, rev_b, "--", path], cwd=REPO,
                       env=ENV, capture_output=True, text=True).stdout.splitlines()
    add = sum(len(x) - 1 for x in d if x.startswith("+") and not x.startswith("+++"))
    rem = sum(len(x) - 1 for x in d if x.startswith("-") and not x.startswith("---"))
    ns = subprocess.run(["git", "diff", "--numstat", rev_a, rev_b, "--", path], cwd=REPO,
                        env=ENV, capture_output=True, text=True).stdout.strip()
    la, lr = (ns.split("\t")[:2] if ns else ("0", "0"))
    return int(la), int(lr), add, rem

nb, py = (REPO / "research.ipynb").stat().st_size, (REPO / "research.py").stat().st_size
png = len(cell_outputs(126)[1])
print(f"  research.ipynb {nb:,} bytes, of which {png:,} ({100 * png / nb:.0f}%) is one "
      f"base64 PNG\n  research.py    {py:,} bytes\n")
print("                            lines +/-      bytes a reviewer must read +/-")
for label, a, b, f in [("one constant, notebook", "HEAD~2", "HEAD~1", "research.ipynb"),
                       ("one constant, module  ", "HEAD~2", "HEAD~1", "research.py"),
                       ("pure re-run, notebook ", "HEAD~1", "HEAD", "research.ipynb"),
                       ("pure re-run, module   ", "HEAD~1", "HEAD", "research.py")]:
    la, lr, ab, rb = payload(a, b, f)
    print(f"  {label}    {la:3d}  {lr:3d}        {ab:8,}  {rb:8,}")
print("\n  the module's entire diff for the constant change:")
for line in subprocess.run(["git", "diff", "-U0", "HEAD~2", "HEAD~1", "--", "research.py"],
                           cwd=REPO, env=ENV, capture_output=True,
                           text=True).stdout.splitlines()[4:]:
    print("   " + line)
# =>   research.ipynb 19,131 bytes, of which 18,504 (97%) is one base64 PNG
#      research.py    63 bytes
#
#                                lines +/-      bytes a reviewer must read +/-
#      one constant, notebook      5    5          18,627    21,835
#      one constant, module        1    1              14        14
#      pure re-run, notebook       2    2              48        48
#      pure re-run, module         0    0               0         0
#
#      the module's entire diff for the constant change:
#       @@ -1 +1 @@
#       -LOOKBACK = 252
#       +LOOKBACK = 126
```

The line counts flatter the notebook, and that is the trap. Five lines against one looks like a factor of five; the bytes say **18,627 against 14**, a factor of **1,331**, because a notebook stores its rendered outputs and 97% of this one is a single base64 PNG on a single unreadable line. A reviewer cannot read that diff. A reviewer cannot even skim it, and the practical consequence is one every desk has lived: notebook diffs get approved unread, which means the notebook sits outside review whatever the branch-protection settings say.

The last two rows are the sharper finding, and they are why the rule is about *records* rather than readability. The third commit changes no code at all — same lookback, same source, the author simply pressed Run All — and the notebook still produces a **48-byte diff across two lines**, because execution counts advanced and the stored outputs were rewritten. The module produces **nothing**: zero lines, zero bytes, no commit worth making. So a notebook's history cannot distinguish "I changed the analysis" from "I re-ran the analysis", and neither can `git log`, `git blame`, or `git bisect`. That is a heavier charge than verbosity. It says the version-control operations this entire lesson depends on are *undefined* on a notebook, because every one of them assumes a diff encodes an intent. Stripping outputs before committing softens the problem; keeping the analysis in modules that notebooks merely import removes it, which is exactly the layout [Typing, Dataclasses, and Code Structure](../part-02-python/03-typing-dataclasses-structure.md) argued for on entirely different grounds.

## The commit that changed the number

Here is the situation the practice exists for. A backtest that produced a Sharpe of 0.30 now produces something else, nobody remembers touching the strategy, and there are two weeks of commits in between. `git bisect` turns that into a binary search, provided you can write a program that answers *good* or *bad* without a human in the loop. The predicate is a golden-file test in miniature: recompute the decision stream, hash it, compare against the hash the known-good code produced — the same digest idiom [One strategy, two harnesses](../part-06-live-infrastructure/01-system-architecture.md) used to prove that a backtest and a threaded live loop agreed timestamp for timestamp.

The repository below is fifteen commits of the kind of work that fills a real week: docstrings, type hints, an extracted helper, a rename finished across two commits, an `__all__`. Exactly one of them changes behaviour.

```python
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("lab/quantlab")
ENV = {"GIT_AUTHOR_NAME": "Quant Lab", "GIT_AUTHOR_EMAIL": "lab@example.com",
       "GIT_COMMITTER_NAME": "Quant Lab", "GIT_COMMITTER_EMAIL": "lab@example.com",
       "GIT_AUTHOR_DATE": "2025-03-01T09:00:00+00:00",
       "GIT_COMMITTER_DATE": "2025-03-01T09:00:00+00:00", "PATH": "/usr/bin:/bin"}

SRC = '''import numpy as np

LOOKBACK = 252
ASSETS = ["SPY", "TLT", "GLD"]


def signal(px):
    rets = np.log(px[ASSETS]).diff()
    return np.sign(rets.rolling(LOOKBACK).sum()).shift(1)
'''

# fourteen commits a reviewer would wave through; exactly one changes behaviour
EDITS = [
    ("Add a module docstring", "import numpy", '"""Time-series momentum."""\nimport numpy'),
    ("Name the annualization factor", "LOOKBACK = 252", "ANN = 252\nLOOKBACK = 252"),
    ("Type-hint signal()", "def signal(px):", "def signal(px: 'pd.DataFrame'):"),
    ("Docstring for signal()", "    rets =", '    """+1 long, -1 short."""\n    rets ='),
    ("Explain the shift", "    return np.sign",
     "    # trade tomorrow on a signal formed from today's close\n    return np.sign"),
    ("Extract the log-return helper", "    rets = np.log(px[ASSETS]).diff()",
     "    rets = logret(px)"),
    ("Define logret()", "def signal",
     "def logret(px):\n    return np.log(px[ASSETS]).diff()\n\n\ndef signal"),
    ("Rename rets to r", "    rets = logret(px)", "    r = logret(px)"),
    ("Finish the rename", "np.sign(rets.rolling", "np.sign(r.rolling"),
    ("Simplify the signal path",                              # <- the bug
     "    # trade tomorrow on a signal formed from today's close\n"
     "    return np.sign(r.rolling(LOOKBACK).sum()).shift(1)",
     "    return np.sign(r.rolling(LOOKBACK).sum())"),
    ("Add __all__", 'ASSETS = ["SPY", "TLT", "GLD"]',
     'ASSETS = ["SPY", "TLT", "GLD"]\n__all__ = ["signal", "logret"]'),
    ("Comment the helper", "def logret(px):",
     "def logret(px):\n    # log returns sum across time; simple returns do not"),
    ("Use ANN for the lookback", "LOOKBACK = 252", "LOOKBACK = ANN"),
    ("Tidy the constants block", "ANN = 252\n", "ANN = 252  # trading days\n"),
]

CHECK = '''import hashlib, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, ".")
import tsmom

px = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "prices.parquet")
d = hashlib.sha256(tsmom.signal(px).dropna(how="all").to_csv().encode()).hexdigest()[:12]
print(f"digest {d}")
sys.exit(0 if d == "GOLDEN" else 1)
'''

def git(*a, check=True):
    return subprocess.run(("git",) + a, cwd=REPO, env=ENV, check=check,
                          capture_output=True, text=True)

def commit(msg):
    git("add", "-A")
    git("commit", "-q", "-m", msg)

shutil.rmtree(REPO, ignore_errors=True)
REPO.mkdir(parents=True)
git("init", "-q", "-b", "main")
src = SRC
(REPO / "tsmom.py").write_text(src)
commit("Add the tsmom signal")
first = git("rev-parse", "HEAD").stdout.strip()
for msg, old, new in EDITS:
    src = src.replace(old, new, 1)
    (REPO / "tsmom.py").write_text(src)
    commit(msg)

# the golden digest is what the known-good code produces, computed here directly
px = pd.read_parquet("data/prices.parquet")
r = np.log(px[["SPY", "TLT", "GLD"]]).diff()
good = np.sign(r.rolling(252).sum()).shift(1).dropna(how="all")
golden = hashlib.sha256(good.to_csv().encode()).hexdigest()[:12]
Path("lab/check.py").write_text(CHECK.replace("GOLDEN", golden))

print(f"  {git('rev-list', '--count', 'HEAD').stdout.strip()} commits, "
      f"golden digest {golden}")
git("bisect", "start")
git("bisect", "bad", "main")
git("bisect", "good", first)
out = git("bisect", "run", sys.executable, "../check.py", check=False)
git("bisect", "reset", check=False)
for line in out.stdout.splitlines():
    if not line.startswith("running "):
        print(("  " + line).rstrip())
# =>   15 commits, golden digest 103cae9b6e35
#      digest 103cae9b6e35
#      Bisecting: 3 revisions left to test after this (roughly 2 steps)
#      [46e7077593a54ef0e7bc5d57ab9c71738d7d2ef9] Simplify the signal path
#      digest 817e5d3ef2ac
#      Bisecting: 0 revisions left to test after this (roughly 1 step)
#      [cc5b81690386c5095042169fb9c60bcbee635920] Finish the rename
#      digest 103cae9b6e35
#      46e7077593a54ef0e7bc5d57ab9c71738d7d2ef9 is the first bad commit
#      commit 46e7077593a54ef0e7bc5d57ab9c71738d7d2ef9
#      Author: Quant Lab <lab@example.com>
#      Date:   Sat Mar 1 09:00:00 2025 +0000
#
#          Simplify the signal path
#
#       tsmom.py | 3 +--
#       1 file changed, 1 insertion(+), 2 deletions(-)
#      bisect found first bad commit
```

Two probes. Fourteen candidate commits collapse to one in two automated tests, and the answer is exact — not a suspicion, not a shortlist, a commit hash and the three lines it touched. The hashes are stable, too: the block pins author and committer dates, so `46e7077593a5` is the same identifier on your machine as on mine. Reproducibility is not a property that applies only to numbers.

Now read what the commit actually says. **"Simplify the signal path"** — one file, one insertion, two deletions, and what it deleted was `.shift(1)` together with the comment explaining why `.shift(1)` was there. That is not a contrived example; it is the most common shape this failure takes. The comment justifying a line was removed in the same edit that removed the line, so the diff reads as internally consistent, and a reviewer scanning fourteen tidy-up commits has no reason to stop on the one whose message is the most reassuring of the set. The strategy went from a Sharpe of 0.30 to a Sharpe of 1.07 — a number so good it should itself have been the alarm, which is the subject of the next section. The narrower lesson from bisect is worth stating precisely: **bisect finds the commit, and the commit is only as informative as you made it.** Here it is three lines. Buried inside a four-hundred-line refactor, the same edit would have been located just as quickly and explained just as poorly.

## The checklist a test suite cannot replace

Reviewing quantitative code is not the same activity as reviewing code. The linter and the type checker have approved every line below; a test suite that asserts a Sharpe ratio comes back as a finite float passes on all of them. What review is for is the class of defect that produces a *plausible number for a wrong reason*, and the only way to build the instinct is to see the same defects priced side by side:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
A = ["SPY", "TLT", "GLD"]
rets = np.log(px[A]).diff()

def sharpe(r):
    r = r.dropna()
    return float(np.sqrt(252) * r.mean() / r.std()), len(r)

def book(sig, r=None):
    r = rets if r is None else r
    return (sig * r).mean(axis=1)

base = book(np.sign(rets.rolling(252).sum()).shift(1))

# 1. the reviewer's first question: does the signal use the bar it trades on
leak = book(np.sign(rets.rolling(252).sum()))

# 2. a regular business-day grid, forward-filled over holidays
rb = np.log(px[A].asfreq("B").ffill()).diff()
align = book(np.sign(rb.rolling(252).sum()).shift(1), rb).reindex(rets.index)

# 3. one basis point of turnover cost, typed as one percent
sig = np.sign(rets.rolling(252).sum()).shift(1)
turn = sig.diff().abs().sum(axis=1) / 2.0
costed = {bp: (book(sig) - turn * bp) for bp in (0.0001, 0.01)}

# 4. a window measured in calendar days rather than trading rows
window = book(np.sign(rets.rolling("365D").sum()).shift(1))

# 5. one row of warm-up history, gained or lost at the start
edge = base.loc["2001-01-04":]

rows = [("none -- the code as reviewed", base),
        ("1 lookahead: .shift(1) deleted", leak),
        ("2 alignment: asfreq('B').ffill()", align),
        ("3 units: 1 bp charged correctly", costed[0.0001]),
        ("3 units: 1 bp typed as 1 percent", costed[0.01]),
        ("4 window: rolling('365D'), not 252", window),
        ("5 edge date: one row of warm-up", edge)]
b0 = sharpe(base)[0]
print("  defect                               Sharpe    vs base       n")
for label, r in rows:
    s, n = sharpe(r)
    tag = "     --" if label.startswith("none") else f"{100 * (s / b0 - 1):+6.0f}%"
    print(f"  {label:36s} {s:+7.4f}   {tag}   {n:5d}")

# where does defect 2 actually come from? not where most people guess
sig_b = np.sign(rb.rolling(252).sum()).shift(1).reindex(rets.index)
sig_r = np.sign(rets.rolling(252).sum()).shift(1).reindex(rb.index)
print(f"\n  2 decomposed: 252 rows of a business-day grid span "
      f"{252 * len(rets) / len(rb):.0f} trading days")
print(f"      grid signal x real returns    {sharpe(book(sig_b, rets))[0]:+.4f}   the whole effect")
print(f"      real signal x grid returns    {sharpe(book(sig_r, rb))[0]:+.4f}   contributes nothing")
print(f"      real signal, rolling(246)     "
      f"{sharpe(book(np.sign(rets.rolling(246).sum()).shift(1)))[0]:+.4f}   below baseline")

# 6. the defect that moves no number at all until someone reruns it
def ci(r, rng, n=2000):
    v = r.dropna().values
    i = rng.integers(0, len(v), size=(n, len(v)))
    s = np.sqrt(252) * v[i].mean(axis=1) / v[i].std(axis=1)
    return np.percentile(s, [2.5, 97.5])

print(f"\n  6 seed: the Sharpe is unchanged at {b0:+.4f}; the confidence interval is not")
print("    an unseeded run draws a seed nobody recorded -- these three stand in for it")
for s in (0, 1, 2):
    lo, hi = ci(base, np.random.default_rng(s))
    print(f"      default_rng({s})   [{lo:+.4f}, {hi:+.4f}]")
print("    the same seed, three times, is the whole of the fix")
for _ in range(3):
    lo, hi = ci(base, np.random.default_rng(42))
    print(f"      default_rng(42)  [{lo:+.4f}, {hi:+.4f}]")
# =>   defect                               Sharpe    vs base       n
#      none -- the code as reviewed         +0.3025        --    6158
#      1 lookahead: .shift(1) deleted       +1.0656     +252%    6159
#      2 alignment: asfreq('B').ffill()     +0.3514      +16%    6165
#      3 units: 1 bp charged correctly      +0.2916       -4%    6158
#      3 units: 1 bp typed as 1 percent     -0.7530     -349%    6158
#      4 window: rolling('365D'), not 252   +0.2283      -25%    6409
#      5 edge date: one row of warm-up      +0.3192       +6%    6157
#
#      2 decomposed: 252 rows of a business-day grid span 243 trading days
#          grid signal x real returns    +0.3514   the whole effect
#          real signal x grid returns    +0.3025   contributes nothing
#          real signal, rolling(246)     +0.2806   below baseline
#
#      6 seed: the Sharpe is unchanged at +0.3025; the confidence interval is not
#        an unseeded run draws a seed nobody recorded -- these three stand in for it
#          default_rng(0)   [-0.0872, +0.6873]
#          default_rng(1)   [-0.0889, +0.6934]
#          default_rng(2)   [-0.1036, +0.7086]
#        the same seed, three times, is the whole of the fix
#          default_rng(42)  [-0.0730, +0.6954]
#          default_rng(42)  [-0.0730, +0.6954]
#          default_rng(42)  [-0.0730, +0.6954]
```

The baseline reconciles to the course: **0.3025 over 6,158 rows**, the number [Architecture and Event-Driven Design](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) printed, and the lookahead variant lands on **1.0656**, the same 1.07 that lesson measured when it ruled out look-ahead by construction. Defects one and three are the ones every checklist already lists, and they are the *easy* ones precisely because they are enormous: a Sharpe that triples, or a strategy that inverts to −0.7530 because someone typed `0.01` where they meant `0.0001`. Nobody ships those, because nobody believes them. A three-asset trend book does not have a Sharpe of 1.07, and the researcher who reports one has usually been caught by their own disbelief before anyone else sees it.

The dangerous rows are two, four, and five, and their danger is exactly their modesty. Putting prices on a regular business-day grid and forward-filling the holidays — a change that looks like *tidying*, and that a reviewer would read as an improvement — lifts the Sharpe to **0.3514, up 16%**. Measuring the momentum window in calendar days rather than trading rows drops it to **0.2283, down 25%**, and this one is worse than an overstatement: it makes a working strategy look mediocre, so its failure mode is a discarded edge that nobody ever revisits. And one row of warm-up history at the start moves the number **6%**, from 0.3025 to 0.3192, because the first trading session of 2001 was a surprise rate cut straight into a fresh short. Each of those three sits inside the range a researcher would accept as "roughly what I had before". None would fail a test asserting the Sharpe is positive and finite.

Defect two deserves its decomposition, because the obvious explanation for it is wrong and the lesson generalizes. The natural guess is that forward-filled holiday rows dilute the return series — and they contribute **exactly nothing**: pairing the original signal with the business-day returns reproduces the baseline **0.3025** to four decimals. The entire move comes from the *signal* side, because 252 rows of a business-day grid span about **243 trading days**, so a change presented as a calendar tidy-up quietly shortened the momentum lookback by nine sessions. Nor is the direction a rule: the same book at a 246-day lookback scores **0.2806**, *below* baseline, so the sensitivity curve is noisy and this defect happened to land on a favourable point of it. That is the honest shape of the finding — a reviewer who accepted "it just adds some zero-return rows" would have accepted a nine-day change to the strategy's single most important hyperparameter, and would have been lucky rather than right about the sign.

Defect six should reshape how you read a diff, because its effect on the reported Sharpe is **exactly zero**. The point estimate is 0.3025 whether or not the bootstrap is seeded. What moves is the confidence interval — across draws the lower bound wanders from −0.0730 to −0.1036, a **4% swing in the interval's width** on every rerun — and since that interval is what decides whether a sleeve gets capital, an unseeded generator makes the allocation partly a coin flip that nobody knows is being flipped. No test catches this, because every individual run is internally consistent and passes. Only a human reading the diff and asking *where does the randomness come from* catches it, which is the whole argument for review as a distinct activity: **tests check that the code does the same thing twice; review checks that what it does is what you meant.** The rule from [Logging and Configuration Management](../part-02-python/07-logging-and-config.md) belongs in the review checklist rather than the test suite — one `default_rng(cfg.seed)` created at the top of the run and passed explicitly, so the dependency is visible in every signature it touches.

## Reproducible means the data, not just the code

Part II's run manifest recorded the config, a hash of the config, the code version, and the outputs, and it was enough to turn a number into a claim rather than an anecdote. It is not enough to *settle* a dispute, because it is silent about the two inputs most likely to have moved: the data and the environment. Extend it, then attack it — change nothing but the bytes on disk, and watch which fields notice:

```python
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

A = ["SPY", "TLT", "GLD"]

def run(px):
    """The decision stream and the book it produces, from one price frame."""
    r = np.log(px[A]).diff()
    sig = np.sign(r.rolling(252).sum()).shift(1)
    book = (sig * r).mean(axis=1).dropna()
    d = hashlib.sha256(sig.dropna(how="all").to_csv().encode()).hexdigest()[:12]
    return d, float(np.sqrt(252) * book.mean() / book.std())

CFG = {"assets": A, "lookback": 252, "seed": 42}
px = pd.read_parquet("data/prices.parquet")
digest, sharpe = run(px)
manifest = {
    "config_hash": hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:12],
    "code_version": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()[:12],
    "env_hash": hashlib.sha256(subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True,
        text=True).stdout.encode()).hexdigest()[:12],
    "data": {"data/prices.parquet": sha("data/prices.parquet")},
    "outputs": {"sharpe": round(sharpe, 4), "decisions": digest},
}
VOLATILE = {"code_version": "<this checkout's HEAD>",
            "env_hash": "<this environment's pip freeze>"}
print("  run manifest")
for k, v in manifest.items():
    shown = VOLATILE.get(k, v if not isinstance(v, dict) else json.dumps(v))
    print(f"    {k:14s} {shown}")

# named rather than globbed: `data/` grows as the course does, and a listing
# that changes with it is exactly the drift this section is about
CACHES = ["prices.parquet", "part4.parquet", "part5.parquet",
          "part5trades.parquet", "part7.parquet", "part8.parquet"]
print("\n  the caches frozen by Parts III through VIII, and Part VII's pinned claim")
for name in CACHES:
    p = Path("data") / name
    print(f"    {p.name:20s} {p.stat().st_size:>9,} bytes  {sha(p)}")
pin = "b956966146cd"
print(f"    Part VII pinned data sha256 {pin} -> {sha('data/part7.parquet')} "
      f"({'HOLDS' if sha('data/part7.parquet') == pin else 'BROKEN'})")

print("\n  now change only the data; code, config and environment are untouched")
print("    perturbation                          Sharpe   decisions      data sha")
print(f"    {'none':36s} {sharpe:+7.4f}  {digest}  {sha('data/prices.parquet')}")
Path("lab").mkdir(exist_ok=True)
for label, col, when, mult in [
        ("vendor re-adjusts one dividend", "SPY", slice(None, "2024-12-19"), 1 - 0.0033),
        ("one bad mark: SPY / 100", "SPY", "2008-10-10", 0.01),
        ("one bad mark: SPY x 10", "SPY", "2008-10-10", 10.0),
        ("one bad mark: GLD / 10", "GLD", "2013-04-15", 0.1)]:
    q = px.copy()
    q.loc[when, col] *= mult
    out = Path("lab/perturbed.parquet")
    q.to_parquet(out)
    d, s = run(pd.read_parquet(out))
    flag = "  <- same decisions" if d == digest else ""
    print(f"    {label:36s} {s:+7.4f}  {d}  {sha(out)}{flag}")
# =>   run manifest
#        config_hash    b13da803562b
#        code_version   <this checkout's HEAD>
#        env_hash       <this environment's pip freeze>
#        data           {"data/prices.parquet": "bdeae092cc47"}
#        outputs        {"sharpe": 0.3025, "decisions": "103cae9b6e35"}
#
#      the caches frozen by Parts III through VIII, and Part VII's pinned claim
#        prices.parquet         218,897 bytes  bdeae092cc47
#        part4.parquet          621,993 bytes  d4cdb11d5072
#        part5.parquet          904,795 bytes  527bef3b66f5
#        part5trades.parquet     38,670 bytes  80cb021f363d
#        part7.parquet        3,260,866 bytes  b956966146cd
#        part8.parquet        1,290,329 bytes  32458dd465e4
#        Part VII pinned data sha256 b956966146cd -> b956966146cd (HOLDS)
#
#      now change only the data; code, config and environment are untouched
#        perturbation                          Sharpe   decisions      data sha
#        none                                 +0.3025  103cae9b6e35  bdeae092cc47
#        vendor re-adjusts one dividend       +0.3029  103cae9b6e35  cfe448e088cc  <- same decisions
#        one bad mark: SPY / 100              +0.0803  103cae9b6e35  28a683bae4c0  <- same decisions
#        one bad mark: SPY x 10               -0.0886  9725ec10f6ac  990cd55d45de
#        one bad mark: GLD / 10               +0.1432  f5116a3d5b17  dba948968d98
```

Two of the manifest's five fields are deliberately not printed, and the reason is the section's own subject. `code_version` is whatever commit you are standing on and `env_hash` digests your installed packages, so both are *supposed* to vary — which makes them useless as pinned output and essential inside the manifest, where their whole job is to record a context that differs. The block computes both and shows a placeholder instead. The cache listing is named rather than globbed for the same reason: `data/` gains a file in the next lesson, and a listing that silently grew would be precisely the undeclared drift this section exists to catch.

Everything that remains does reproduce, including the line that matters most for this course's own credibility — **Part VII's pinned `b956966146cd` still verifies against the file on disk**, which is the freeze doctrine of [Part III](../part-03-statistics/01-probability-and-random-variables.md) and [Part IV](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) holding across four parts and a year of downstream work. A pinned hash nobody ever checks is decoration; this one was checked, and it held.

The perturbation table is where the section earns its place. A vendor re-adjusting a single dividend moves the Sharpe by **+0.0004** — invisible, harmless, and still worth recording, because a colleague who reruns your notebook and gets 0.3029 where you published 0.3025 will otherwise spend an afternoon hunting a bug in the code. That is the everyday value of a data hash: it converts a mystery into a diff.

The third row is the one to sit with. A single bad print on 10 October 2008 — SPY marked at a hundredth of its true price, the cents-for-dollars error [Part VI's risk gate](../part-06-live-infrastructure/01-system-architecture.md) was built to refuse — collapses the Sharpe from **0.3025 to 0.0803**, destroying three quarters of the book's apparent risk-adjusted return. And the decision digest is **`103cae9b6e35`, byte-identical to the clean run**. Not one position changed. The reason is arithmetic rather than luck: a rolling sum of log returns telescopes to the log of the ratio of its endpoints, so a bad mark strictly inside a window cancels itself exactly and flips no signs; the damage lives entirely in the two enormous returns the bad mark manufactured, which inflate the standard deviation and crush the ratio. The consequence for testing is specific and uncomfortable — **a golden-file test on the decision stream, the exact test that caught the lookahead bug two sections ago, passes this bug without complaint.** Deciding what a golden file should actually contain is the first real problem [Testing and CI/CD](02-testing-and-cicd.md) has to solve.

## Every commit is a bisect resolution chosen in advance

Bisect's cost is logarithmic and its *resolution* is not. Eight probes will search a thousand commits as happily as ten, but what comes back is one commit, and if that commit is a four-hundred-line refactor then the search has narrowed the suspects from a thousand files to four hundred lines and stopped. The useful question is therefore not how fast bisect runs but what your commits contain, and this repository can answer that about itself:

```python
import math
import subprocess

import numpy as np

# pinned at Part VIII's last commit so this measurement does not drift as the
# course grows -- the same discipline the previous section argued for
REF = "740aa58"
out = subprocess.run(["git", "log", REF, "--numstat", "--pretty=format:%x00%s"],
                     capture_output=True, text=True).stdout

commits, lines, files, msg = [], 0, 0, None
for row in out.splitlines():
    if row.startswith("\x00"):
        if msg is not None:
            commits.append((lines, files, msg))
        lines, files, msg = 0, 0, row[1:]
    elif row.strip():
        a, r, _ = row.split("\t")
        if a != "-":
            lines += int(a) + int(r)
            files += 1
if msg is not None:
    commits.append((lines, files, msg))

L = np.array([c[0] for c in commits])
F = np.array([c[1] for c in commits])
n = len(L)
print(f"  {n} commits up to {REF}")
print("    quantile     lines changed    files touched")
for q in (50, 75, 90, 99):
    print(f"      p{q:<3d}         {np.percentile(L, q):8.0f}          {np.percentile(F, q):6.1f}")
big = commits[int(np.argmax(L))]
print(f"      max          {L.max():8d}          {F.max():6d}   ({big[2][:38]})")

probes = math.ceil(math.log2(n))
print(f"\n  bisect over {n} commits costs {probes} probes and returns one commit.")
print("  what that commit contains is the resolution of the answer:")
for lo, hi, span, label in [(0, 20, "under 20 lines", "read it in the diff"),
                            (20, 100, "20 to 100", "one sitting"),
                            (100, 400, "100 to 400", "a review"),
                            (400, 10 ** 9, "over 400", "an archaeology project")]:
    k = int(((L >= lo) & (L < hi)).sum())
    print(f"    {span:>14s}  {k:4d} commits  {100 * k / n:4.0f}%   {label}")
# =>   145 commits up to 740aa58
#        quantile     lines changed    files touched
#          p50                88             1.0
#          p75               272             4.0
#          p90               403            12.6
#          p99             19795           153.1
#          max             38330             168   (Initial upload of files)
#
#      bisect over 145 commits costs 8 probes and returns one commit.
#      what that commit contains is the resolution of the answer:
#        under 20 lines    32 commits    22%   read it in the diff
#             20 to 100    44 commits    30%   one sitting
#            100 to 400    52 commits    36%   a review
#              over 400    17 commits    12%   an archaeology project
```

The median commit in this repository changes **88 lines in a single file**, which is a good number: bisect lands on it and the diff is readable in one sitting. Better still, **22% of commits are under twenty lines** — the size at which a bisect result is not a lead but an *answer*, because the offending line is visible without leaving the terminal. The pinned `REF` is not decoration either; without it this block's output would drift every time the course gained a lesson, which is precisely the failure the previous section was about.

The tail is the warning. Twelve percent of commits change more than four hundred lines, and the largest changes **38,330 lines across 168 files** — a bulk import no review process could meaningfully have inspected, and one bisect can only ever point at, never into. Such commits are not mistakes in themselves; a migration is a migration. But each is a region of history where the search degrades from *which line* to *which week*, and they should be created deliberately and rarely, in full knowledge that you are trading future diagnosability for present convenience. The rule that falls out is unglamorous: **one intent per commit**, and when a refactor and a behaviour change genuinely must travel together, split them anyway and put the behaviour change last, so that bisect's answer is the small commit rather than the large one.

## History you can bisect is history you wrote on purpose

Everything so far assumed a property nobody guarantees: that every commit in the range actually runs. Real branches are not like that. There are commits pushed at the end of a day with a half-finished function, commits importing a module that does not exist yet, commits whose only purpose was to move work between machines. `git bisect` has a protocol for these — a predicate exiting with code **125** means *cannot test this one* — and it is worth knowing exactly how much that protocol buys, because the answer is less than you would hope:

```python
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("lab/quantlab_wip")
ENV = {"GIT_AUTHOR_NAME": "Quant Lab", "GIT_AUTHOR_EMAIL": "lab@example.com",
       "GIT_COMMITTER_NAME": "Quant Lab", "GIT_COMMITTER_EMAIL": "lab@example.com",
       "GIT_AUTHOR_DATE": "2025-03-01T09:00:00+00:00",
       "GIT_COMMITTER_DATE": "2025-03-01T09:00:00+00:00", "PATH": "/usr/bin:/bin"}

GOOD = "    return np.sign(r.rolling(LOOKBACK).sum()).shift(1)\n"
BAD = "    return np.sign(r.rolling(LOOKBACK).sum())\n"
HEAD = ('import numpy as np\n\nLOOKBACK = 252\nASSETS = ["SPY", "TLT", "GLD"]\n\n\n'
        'def signal(px):\n    r = np.log(px[ASSETS]).diff()\n')

# fifteen commits, the bug at index 10, and four left mid-refactor and unbuildable
BUG_AT, BROKEN = 10, {8, 9, 11, 12}

CHECK = '''import hashlib, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, ".")
try:
    import tsmom
except SyntaxError:
    print("unbuildable -- cannot answer")
    sys.exit(125)                      # 125 tells git bisect to skip this commit

px = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "prices.parquet")
d = hashlib.sha256(tsmom.signal(px).dropna(how="all").to_csv().encode()).hexdigest()[:12]
print(f"digest {d}")
sys.exit(0 if d == "103cae9b6e35" else 1)
'''

def git(*a, check=True):
    return subprocess.run(("git",) + a, cwd=REPO, env=ENV, check=check,
                          capture_output=True, text=True)

shutil.rmtree(REPO, ignore_errors=True)
REPO.mkdir(parents=True)
git("init", "-q", "-b", "main")
Path("lab/check_wip.py").write_text(CHECK)

for i in range(15):
    body = HEAD + (BAD if i >= BUG_AT else GOOD) + f"# refactor step {i}\n"
    if i in BROKEN:
        body += "def wip(:\n    pass\n"
    (REPO / "tsmom.py").write_text(body)
    git("add", "-A")
    git("commit", "-q", "-m", ("WIP: " if i in BROKEN else "") + f"Refactor step {i}")
    if i == 0:
        first = git("rev-parse", "HEAD").stdout.strip()

print(f"  15 commits, {len(BROKEN)} of them left unbuildable, bug at step {BUG_AT}")
git("bisect", "start")
git("bisect", "bad", "main")
git("bisect", "good", first)
out = git("bisect", "run", sys.executable, "../check_wip.py", check=False)
git("bisect", "reset", check=False)
probes = 0
for line in out.stdout.splitlines():
    if line.startswith("running "):
        probes += 1
        continue
    print("  " + line)
print(f"  ({probes} probes spent)")
if out.stderr.strip():
    print("  " + out.stderr.strip().splitlines()[-1])
# =>   15 commits, 4 of them left unbuildable, bug at step 10
#      digest 103cae9b6e35
#      Bisecting: 3 revisions left to test after this (roughly 2 steps)
#      [15574270e8f4f08f78634340f59329fd41ecb428] Refactor step 10
#      digest 817e5d3ef2ac
#      Bisecting: 0 revisions left to test after this (roughly 1 step)
#      [1beaef02936b70e0364d0aee7af6cb3f1b32c92b] WIP: Refactor step 9
#      unbuildable -- cannot answer
#      Bisecting: 0 revisions left to test after this (roughly 1 step)
#      [d468be0d60d6b401bb17a9db748edfbc4ad4fa20] WIP: Refactor step 8
#      unbuildable -- cannot answer
#      There are only 'skip'ped commits left to test.
#      The first bad commit could be any of:
#      1beaef02936b70e0364d0aee7af6cb3f1b32c92b
#      d468be0d60d6b401bb17a9db748edfbc4ad4fa20
#      15574270e8f4f08f78634340f59329fd41ecb428
#      We cannot bisect more!
#      (4 probes spent)
#      error: bisect run cannot continue any more
```

The same repository, the same bug, the same predicate — and the search **spends twice as many probes to return a worse answer**. Four probes instead of two, and instead of a commit hash the output is a list of three candidates and the sentence `We cannot bisect more!`. Four unbuildable commits out of fifteen, a rate that would not raise an eyebrow on a real feature branch, were enough to turn an exact answer into a shortlist. The arithmetic is unforgiving: the skipped commits sat directly adjacent to the guilty one, so the search converged on a region it could not probe, and no amount of further testing could break the tie.

The remedy is not a git setting, and that is the point of the section. Bisectability is a property you *maintain*, at some cost, on the bet that you will eventually need it. Concretely: every commit on a shared branch should at minimum import, which is cheap enough to enforce mechanically in the pipeline the next lesson builds; work in progress belongs on a local branch and gets squashed before it is pushed, so shared history contains intents rather than keystrokes; and a squash-merge policy, whatever else can be said against it, guarantees every commit on the main line was green at least once. The cost is a little ceremony on days when nothing is broken. The return is that on the day something is broken, "which change did this" has a two-probe answer instead of a week of reading. **You cannot decide to have a bisectable history on the day you need one** — by then the history is already written, and the commits you would most need to test are the ones you left unbuildable.

!!! warning "The commit that breaks the number will not say so in its message"
    Every defect in this lesson was introduced by someone competent doing something reasonable. `.shift(1)` was deleted in a commit called *Simplify the signal path*, alongside the very comment that explained why it was there. A business-day grid with forward-filled holidays is a tidy-up any reviewer would approve, and it moved the Sharpe 16%. A window measured in calendar days rather than rows is arguably the more natural reading of "one year", and it made a working strategy look mediocre. None of these is a typo, none is visible to a linter or a type checker, and all three of the subtle ones land inside the range a researcher would accept as unchanged. Review the diff for intent, because correctness is what the machines are for — and treat a result that improves without an explanation exactly as sceptically as one that degrades.

!!! abstract "Key takeaways"
    - The identical one-constant edit costs **18,627 bytes of diff in a notebook against 14 in a module**, a factor of **1,331**, because 97% of the notebook is one base64 PNG on one unreadable line — and a notebook merely *re-run*, with no code change at all, still produces a **48-byte diff** where the module produces nothing. `git log`, `git blame` and `git bisect` all assume a diff encodes an intent, so all three are undefined on a notebook.
    - `git bisect run` against a golden digest found the behaviour-changing commit among fourteen candidates in **two probes**, returning `46e7077593a5` — one file, one insertion, two deletions, and a commit message reading **"Simplify the signal path"**.
    - The five numeric defects price out as lookahead **+252%** (0.3025 to 1.0656), units **−349%**, alignment **+16%**, window semantics **−25%**, and one row of warm-up **+6%**. The two enormous ones are safe because nobody believes them; the three modest ones all fall inside "roughly what I had before".
    - The alignment defect's obvious explanation is wrong: the forward-filled rows contribute **exactly nothing** (**0.3025**, unchanged), and the whole 16% comes from 252 grid rows spanning **243 trading days** — a nine-session change to the lookback. At 246 days the same book scores **0.2806**, below baseline, so even the sign was luck.
    - An unseeded bootstrap leaves the Sharpe **exactly unchanged at 0.3025** while the interval's lower bound wanders from **−0.0730 to −0.1036** — the defect no test can see, because every individual run is self-consistent and passes.
    - A single bad mark on one day in 2008 cuts the Sharpe from **0.3025 to 0.0803** and leaves the decision digest **byte-identical at `103cae9b6e35`**: a rolling sum of log returns telescopes, so an interior bad mark flips no signs. A golden file on decisions passes this bug.
    - Part VII's pinned data hash **`b956966146cd` still verifies** against the file on disk four parts downstream — while a vendor re-adjusting one dividend moves the Sharpe **+0.0004**, small enough to waste an afternoon on if the manifest does not record the data.
    - Bisect's cost is logarithmic and its *resolution* is whatever you chose to put in a commit: the median commit here changes **88 lines in one file**, but **12% change more than 400** and the largest changes **38,330 across 168 files**. And four unbuildable commits out of fifteen made the search spend **twice the probes to return three candidates instead of one**, ending in `We cannot bisect more!` — bisectability is maintained in advance or it is not available at all.

## Where this goes next

The practices in this lesson share a boundary worth naming before crossing it. Review and bisect both operate on *changes*: they answer "what did you alter, and when". Neither answers "is this correct", and two results above show the gap directly — the unseeded generator that no rerun can detect, and the corrupted mark that leaves every decision identical while destroying the P&L. Both slipped past the check that caught the lookahead bug, and both would slip past a reviewer not specifically hunting them. What is missing is an assertion that runs unattended, on every change, and already knows what the answer is supposed to be.

That is the subject of [Testing and CI/CD](02-testing-and-cicd.md), and this lesson has handed it a concrete problem. The golden digest used as a bisect predicate was the right instrument for a behaviour change and the wrong one for a data corruption, so the next lesson has to decide what a golden file should actually contain, how much tolerance a floating-point pipeline needs before its own refactors start failing, and what a statistical assertion costs when a suite runs it two hundred times a day. It also inherits an inconvenient fact from the section on commit sizes: a test not fast enough to run on every commit will not be run on every commit, and a gate that gets skipped is not a gate.
