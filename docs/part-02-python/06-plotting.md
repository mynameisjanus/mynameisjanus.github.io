# Plotting for Research

A chart is an argument, and most research charts would not survive cross-examination: unlabeled axes, a linear scale that makes late gains look heroic, a sample window that starts conveniently after the bad year. The audience for every plot you make is a skeptical investment committee — even when that committee is only you, six months from now, trying to remember why this strategy seemed convincing. This lesson builds the small set of charts quantitative research actually runs on and the standards that make them trustworthy.

All the snippets run headless and end in `plt.show()`; in your own work the last line of a figure script is `fig.savefig("figures/equity.png", dpi=150)`, because a chart that exists only in a window you once looked at is not part of the research record.

## matplotlib for research, not decoration

matplotlib has two APIs, and the difference matters more than any styling decision. The implicit `pyplot` interface (`plt.plot(...)` onto whatever figure is "current") is fine for a throwaway look and hostile to scripts — hidden global state is exactly what the previous lessons taught you to avoid. The explicit interface names its objects, and every figure in this course starts the same way:

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 252)))

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(prices, lw=1.2)
ax.set_title("Synthetic daily closes, 252 trading days (seed 42)")
ax.set_xlabel("Trading day")
ax.set_ylabel("Price")
plt.show()
```

`fig` is the page, `ax` is the coordinate system drawn on it, and `layout="constrained"` stops labels colliding without manual margin surgery. The habit that matters most is not aesthetic: **every figure is produced by a script**, start to finish, data to pixels. A point-and-click chart cannot be regenerated when the data updates, cannot be diffed when a number changes, and cannot be trusted when it disagrees with a rerun — which connects this lesson directly to the reproducibility discipline of the next one.

## Equity curves

The equity curve — growth of one unit of capital — is the first chart anyone asks for and the easiest to make quietly dishonest. The dishonesty is the linear y-axis: under compounding, a move from 4 to 8 is the same *return* as a move from 1 to 2, but linear scaling draws it four times as tall, so every linearly scaled long-run equity curve looks like it discovered something in its final third. A log scale makes equal vertical distances equal percentage gains, which is the honest geometry for compounded wealth.

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0008, 0.012, 1260)       # 5 synthetic years, daily
bench = rng.normal(0.0004, 0.010, 1260)
curve = np.exp(np.cumsum(np.log1p(rets)))    # lesson 01's log-space habit
bcurve = np.exp(np.cumsum(np.log1p(bench)))

years = len(rets) / 252
cagr = curve[-1] ** (1 / years) - 1
vol = rets.std() * np.sqrt(252)
sharpe = rets.mean() / rets.std() * np.sqrt(252)
maxdd = (curve / np.maximum.accumulate(curve) - 1).min()
print(f"CAGR {cagr:.1%}  vol {vol:.1%}  Sharpe {sharpe:.2f}  maxDD {maxdd:.1%}")
# => CAGR 11.1%  vol 18.8%  Sharpe 0.65  maxDD -22.6%

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(curve, lw=1.2, label="Strategy")
ax.plot(bcurve, lw=1.0, label="Benchmark")
ax.set_yscale("log")
ax.set_title(f"Growth of 1 unit, 5y synthetic — CAGR {cagr:.1%}, "
             f"Sharpe {sharpe:.2f}, maxDD {maxdd:.1%}")
ax.set_xlabel("Trading day")
ax.set_ylabel("Growth of 1 (log scale)")
ax.legend(frameon=False)
plt.show()
```

The annualized statistics ride on the chart itself — CAGR as $W_T^{1/\text{years}} - 1$, volatility and Sharpe scaled by $\sqrt{252}$ — because a curve without its numbers invites the reader to eyeball what should be computed. And a strategy curve without a benchmark overlay is a rhetorical device: almost anything compounds impressively in isolation. What these statistics mean, and how little five years of them proves, is the business of [Part III](../part-03-statistics/index.md); here the job is displaying them honestly.

## Drawdowns

The equity curve says what you made; the **underwater plot** says what it cost to hold on. Drawdown at time $t$ is the loss from the running peak:

$$
dd_t \;=\; \frac{P_t}{\max_{s \le t} P_s} \;-\; 1 ,
$$

which vectorizes in one line with `np.maximum.accumulate` — the running-maximum ufunc pattern from [NumPy and Vectorization](01-numpy-and-vectorization.md).

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0008, 0.012, 1260)
curve = np.exp(np.cumsum(np.log1p(rets)))

dd = curve / np.maximum.accumulate(curve) - 1
print(f"max drawdown: {dd.min():.1%}")            # => -22.6%
print(f"days underwater: {(dd < 0).sum()} of {dd.size}")  # => 1177 of 1260

fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
ax.fill_between(np.arange(dd.size), dd, 0, alpha=0.45)
ax.set_title("Underwater plot — drawdown from running peak")
ax.set_xlabel("Trading day")
ax.set_ylabel("Drawdown")
plt.show()
```

Read the second number again: this strategy — a healthy 0.65 Sharpe — spends **93% of all days below a previous peak**. That is normal, it is what holding a real strategy feels like, and it is why the underwater plot belongs next to every equity curve you show. Depth, duration, and recovery are three distinct pains: a sharp 20% drawdown recovered in a month tests the risk budget, while a shallow one that grinds for two years tests whether anyone still believes the model. How long drawdowns *should* last for a given Sharpe is quantified in the appendix's [Drawdown Probabilities](../appendix/part-18-quant-finance-applications/03-drawdown-probabilities.md) — calibrating that expectation before going live is much cheaper than discovering it after.

## Return distributions

Summary statistics compress; distributions confess. The standard view is a density histogram of daily returns with a normal density — computed directly from the fitted mean and standard deviation, no fitting library required — drawn over it, shown twice: once on a linear axis, once with a log-scaled density axis, because the tails are where the risk lives and a linear axis renders them invisible.

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0008, 0.012, 1260)
mu, sd = rets.mean(), rets.std()
x = np.linspace(rets.min(), rets.max(), 200)
pdf = np.exp(-((x - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
for ax in axes:
    ax.hist(rets, bins=60, density=True, alpha=0.55)
    ax.plot(x, pdf, lw=1.5)
    ax.set_xlabel("Daily return")
axes[0].set_title("Daily returns vs fitted normal")
axes[1].set_yscale("log")
axes[1].set_title("Same data, log density — read the tails")
plt.show()
```

On this page the histogram hugs the curve even in the log view — necessarily, because the generator *is* Gaussian. That makes the pair a calibrated instrument: on real market returns the log-density view shows the empirical tails peeling above the normal curve, which is fat tails made visible, and the divergence between what you see there and what you see here is precisely what [Returns and Their Distributions](../part-03-statistics/02-returns-and-distributions.md) measures formally. A chart convention that only ever gets shown on data where nothing interesting happens teaches you nothing; know what the boring case looks like so the scary one registers.

## Heatmaps

Two heatmaps recur in research: parameter grids (a performance metric over a 2-D sweep) and correlation matrices. Both live or die by colormap choice, and the rules are short: **sequential** data (low to high, like volume) gets a perceptually uniform map such as `viridis`; **signed** data (Sharpe, correlation) gets a diverging map such as `RdBu_r`, centered so that zero is visually neutral; rainbow maps distort magnitude everywhere and are never the answer.

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
rets = rng.normal(0.0003, 0.011, 2520)        # 10 driftless-ish years

def strat_sharpe(lb: int, thr: float) -> float:
    c = np.concatenate(([0.0], np.cumsum(rets)))
    mom = (c[lb:-1] - c[:-lb - 1]) / lb        # mean of the PAST lb days
    sig = np.where(np.abs(mom) > thr * rets.std(), np.sign(mom), 0.0)
    pnl = sig * rets[lb:]
    return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() else 0.0

lbs = np.arange(5, 65, 5)
thrs = np.round(np.arange(0.0, 0.40, 0.05), 2)
grid = np.array([[strat_sharpe(int(lb), float(t)) for t in thrs] for lb in lbs])

i, j = np.unravel_index(np.abs(grid).argmax(), grid.shape)
print(f"best |Sharpe| {grid[i, j]:.2f} at lookback={lbs[i]}, thr={thrs[j]}")
# => best |Sharpe| 0.55 at lookback=50, thr=0.35

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
m = ax.pcolormesh(thrs, lbs, grid, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_title("Momentum Sharpe over parameter grid — synthetic noise")
ax.set_xlabel("Entry threshold (vol units)")
ax.set_ylabel("Lookback (days)")
fig.colorbar(m, label="Annualized Sharpe")
plt.show()
```

The punchline is in the printed line: these returns are pure noise around a negligible drift, and the sweep still "finds" a 0.55 Sharpe at one corner of the grid. This is what overfitting looks like on a heatmap — **a lone bright cell in an indifferent field**. The pattern you want to see is the opposite: a broad plateau of similar, modestly positive cells, evidence that performance survives parameter perturbation. When Part IV builds real strategies, this chart is the first robustness check every candidate faces. Correlation matrices follow the same recipe — `pcolormesh` of the matrix from lesson 02's aligned returns, `RdBu_r`, fixed limits at ±1.

## Plotly, and the IC-readiness checklist

Everything above is static, and static is usually right: matplotlib output drops into PDFs and papers, renders identically everywhere, and adds no dependencies to the record. Interactivity earns its complexity in specific situations — thousands of points where hover-to-identify beats a legend, tearsheets a non-programmer will explore, dashboards. For those, plotly express produces a self-contained HTML file that anyone with a browser can open:

```python
import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px

rng = np.random.default_rng(42)
curve = np.exp(np.cumsum(np.log1p(rng.normal(0.0008, 0.012, 1260))))

fig = px.line(y=curve, log_y=True, title="Growth of 1 unit (synthetic, 5y)",
              labels={"index": "Trading day", "y": "Growth of 1"})
out = Path(tempfile.mkdtemp()) / "equity.html"
fig.write_html(out)
print(out.stat().st_size > 100_000)  # => True — self-contained, shareable
```

The costs are real: a heavier dependency, HTML artifacts that are harder to version and diff than PNGs, and an invitation to zoom into a flattering window and screenshot it. Default to matplotlib; reach for plotly when the *audience*, not the author, needs the interaction.

Whichever library draws it, a chart headed for other people's decisions passes this checklist:

| Check | The failure it prevents |
|---|---|
| Title states what, which universe, which period | "Impressive — of what, exactly?" |
| Axes labeled, with units | Percent misread as basis points, and worse |
| Sample period and data source on the figure | Charts that outlive the context they came from |
| Log scale for compounded values; no truncated axes without a marker | Geometry that editorializes |
| Diverging colormap centered at zero for signed data; colorblind-safe | Red/green encodings a colorblind reader cannot parse |
| Regenerable from a script, from stored data | The chart that cannot be reproduced when questioned |

!!! warning "A chart you cannot regenerate is a rumor"
    If the data updates, the figure must update by rerunning a script — not by remembering what you clicked. Every figure in a research report should trace to code plus a dataset version, which is exactly the reproducibility contract the next lesson builds for the entire pipeline.

!!! abstract "Key takeaways"
    - Use the explicit `fig, ax` API and script every figure end to end — point-and-click charts are outside the research record.
    - Equity curves get log scales, benchmark overlays, and their summary statistics (CAGR, vol, Sharpe, max drawdown) printed on the figure.
    - The underwater plot is the honest companion to every equity curve — healthy strategies spend most days below a prior peak, and depth, duration, and recovery are different pains.
    - Show return distributions twice, linear and log-density; the log view is where fat tails become visible against the fitted normal.
    - Heatmaps: perceptually uniform maps for sequential data, zero-centered diverging maps for signed data — and a lone bright cell in a parameter grid is overfitting introducing itself.
    - Static matplotlib is the default; plotly's self-contained HTML is for audiences that need to explore. Either way, the IC checklist decides whether a chart ships.

## Where this goes next

Every chart in this lesson was regenerated from a seed, a script, and nothing else — which is the standard your whole research pipeline should meet. [Logging, Configuration, and Reproducibility](07-logging-and-config.md) closes Part II by making runs themselves auditable: what ran, with what parameters, on what data, producing what.
