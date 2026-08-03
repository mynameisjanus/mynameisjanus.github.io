# Validation and Overfitting

Every number this part has produced — the survivors' Sharpe ratios included — shares one unpaid debt: it was *selected*. Lookbacks came from grids, filters from menus, formations from a literature that is itself a survivor of selection. Selection is not a flaw in the research process; it *is* the research process — and it has a precise statistical price that almost no backtest pays. This closing lesson is the collections department. Its tools — walk-forward testing, purged cross-validation, White's Reality Check, the probability of backtest overfitting, the deflated Sharpe ratio — all answer the same question from different angles: *given everything that was tried, how surprised should anyone be by the best thing found?*

The machinery is the culmination this part promised in [its opening page](index.md), and it arrives with the receipts already gathered: [lesson one](01-momentum-and-trend-following.md) logged its 24 lookback trials the day it ran them, [lesson four](04-seasonality-and-calendar-effects.md) counted its nineteen calendar cells, and Part III's [multiple-testing lesson](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) built the fifty-variant momentum grid this lesson will now formally prosecute. Everything before this lesson is inadmissible until it runs.

## The null has teeth

Calibrate the enemy first. Fifty strategies with *no information whatsoever* — daily coin flips deciding long or short SPY — and the only question is what the best of them looks like:

```python
import numpy as np
import pandas as pd
from scipy import stats

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
rng = np.random.default_rng(42)

sharpes = [np.sqrt(252) * (s := pd.Series(rng.choice([-1.0, 1.0], len(r)),
           index=r.index) * r).mean() / s.std() for _ in range(50)]
se = 1 / np.sqrt(len(r) / 252)
emax = se * ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / 50) +
             np.euler_gamma * stats.norm.ppf(1 - 1 / (50 * np.e)))
print(f"50 coin-flip strategies on SPY: best Sharpe {max(sharpes):+.2f}, "
      f"worst {min(sharpes):+.2f}")
print(f"one strategy's standard error {se:.2f}; expected best of 50 nulls {emax:.2f}")
# => 50 coin-flip strategies on SPY: best Sharpe +0.43, worst -0.39
#    one strategy's standard error 0.20; expected best of 50 nulls 0.45
```

The best coin-flipper earned Sharpe 0.43 over a quarter century — and the expected-maximum formula says 0.45 was coming: with a single-strategy standard error of 0.20 and fifty draws, a "best" in the mid-0.4s is not luck to be marveled at, it is the *arithmetic default* — the sampling distribution of a maximum, and the fact that the winner's own interval covers the truth 28% of the time, is [Sampling Distributions](../appendix/part-10-statistics-foundations/03-sampling-distributions.md). Hold this number against the part's own record. The trend book's 0.30, the calendar strategy's 0.31, the sector book's 0.08 — every one of them lives *below* what a fifty-way search over pure noise is expected to produce. That is not proof they are noise; `tsmom` was one pre-committed trial, not fifty, and that distinction will be worth 0.29 of DSR by this lesson's end. But it fixes the burden of proof where it belongs: in this business the null hypothesis comes armed, and any evaluation that does not know the size of its own search cannot even state what the null predicts.

## Walk-forward: hindsight, priced

The honest simulation of parameter choice is to make the choice with only the past, repeatedly. Each year, pick the lookback that was best *up to that year*, then trade it for the following year — hindsight never enters:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
grid = {lb: (np.sign(rets.rolling(lb).sum()).shift(1) * rets).mean(axis=1).dropna()
        for lb in range(40, 501, 20)}
sh = lambda s: np.sqrt(252) * s.mean() / s.std() if len(s) > 50 else np.nan

oos = []
for year in range(2006, 2026):
    is_sh = {lb: sh(s.loc[:str(year - 1)]) for lb, s in grid.items()}
    oos.append(grid[max(is_sh, key=is_sh.get)].loc[str(year)])
wf = pd.concat(oos)
full = {lb: sh(s) for lb, s in grid.items()}
best_full = max(full, key=full.get)
print(f"picking the best lookback in hindsight: {best_full} days, "
      f"Sharpe {full[best_full]:.2f}")
print(f"walk-forward (best lookback known only up to each year): "
      f"Sharpe {sh(wf):.2f} over {wf.index[0].year}-2025")
# => picking the best lookback in hindsight: 480 days, Sharpe 0.56
#    walk-forward (best lookback known only up to each year): Sharpe 0.34 over 2006-2025
```

The gap between 0.56 and 0.34 is the price of hindsight, measured: about 40% of the "optimized" performance existed only in the optimizer's rear-view mirror. Two readings, both important. The pessimistic one: every backtest that reports its best parameter's full-sample performance is quoting the 0.56 — a number no implementable process could have earned. The genuinely encouraging one: 0.34 is not zero, and it sits close to the pre-committed 252-day rule's 0.30 — the walk-forward process, wandering the [lookback plateau](01-momentum-and-trend-following.md) year by year, ends up harvesting roughly what the plateau honestly offers. That is what a *real but parameter-insensitive* effect looks like under walk-forward: degradation toward the plateau's level, not collapse toward zero. Collapse is what a spike-shaped surface produces, and the degradation ratio — out-of-sample over in-sample — is therefore itself a diagnostic: near one, robust effect; near zero, the grid was the strategy.

## Cross-validation that respects time

Machine-learning workflows bring their own validation habit — k-fold cross-validation with *shuffled* folds — and applying it to overlapping financial labels is one of the fastest ways to manufacture skill from nothing. The mechanism: a 21-day forward return at date $t$ shares twenty of its days with the label at $t+1$, so shuffling scatters near-duplicate observations across the train/test boundary, and the model is graded on questions it has effectively seen. Three features that predict nothing — trailing month, trailing year, trailing volatility — against a 21-day forward return:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
H = 21
X = pd.concat([r.rolling(21).sum(), r.rolling(252).sum(),
               r.rolling(63).std()], axis=1).dropna()
y = r.rolling(H).sum().shift(-H).reindex(X.index)
ok = y.notna()
X, y = X[ok].values, y[ok].values
X = (X - X.mean(0)) / X.std(0)
n = len(y)

def cv_corr(fold_ids, purge=0):
    preds, acts = [], []
    for k in np.unique(fold_ids):
        test = np.where(fold_ids == k)[0]
        train = np.where(fold_ids != k)[0]
        if purge:
            near = (np.abs(np.arange(n)[:, None] - test[None, :]) <= purge).any(1)
            train = np.where((fold_ids != k) & ~near)[0]
        beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
        preds.append(X[test] @ beta); acts.append(y[test])
    return np.corrcoef(np.concatenate(preds), np.concatenate(acts))[0, 1]

rng = np.random.default_rng(42)
shuffled = rng.permutation(np.repeat(np.arange(5), np.ceil(n / 5)))[:n]
contig = np.repeat(np.arange(5), np.ceil(n / 5))[:n]
print(f"shuffled 5-fold (the library default habit): {cv_corr(shuffled):+.3f}")
print(f"contiguous 5-fold                          : {cv_corr(contig):+.3f}")
print(f"contiguous + 21-day purge                  : {cv_corr(contig, H):+.3f}")
# => shuffled 5-fold (the library default habit): +0.061
#    contiguous 5-fold                          : -0.087
#    contiguous + 21-day purge                  : -0.091
```

The truth, known by construction, is that these features contain nothing — and the honest folds say so, printing a small negative correlation. Shuffled folds print **+0.061**: a sign flip conjured entirely by label overlap, on a scale that would absolutely survive a pitch meeting ("our model's out-of-sample IC is six percent"). No parameter was tuned, no signal exists, and the standard library default manufactured skill anyway. The repair has two layers: *contiguous* folds remove the catastrophic leak by keeping time intact — that one change moves the answer from +0.061 to −0.087 — and *purging* deletes the training observations whose labels overlap the test block's edges, worth a further correction at the third decimal here ([Cross-Validation](../appendix/part-14-model-selection/02-cross-validation.md) covers the general machinery). The asymmetry of those two magnitudes is the lesson: leakage scales with how much boundary the split creates, and shuffling maximizes boundary — every observation borders test data, so the whole sample leaks.

## Embargoes and the geometry of leakage

Purging removes training labels that *overlap* the test window; an embargo goes further, deleting a guard band *beyond* the overlap, against serial dependence that outlives the label itself — volatility clustering, slow features, autocorrelated errors. Sweep the guard band's width and watch what it buys:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
H = 21
X = pd.concat([r.rolling(21).sum(), r.rolling(252).sum(),
               r.rolling(63).std()], axis=1).dropna()
y = r.rolling(H).sum().shift(-H).reindex(X.index)
ok = y.notna()
X, y = X[ok].values, y[ok].values
X = (X - X.mean(0)) / X.std(0)
n = len(y)
contig = np.repeat(np.arange(5), np.ceil(n / 5))[:n]

for emb in [0, 5, 10, 21, 42, 63]:
    preds, acts = [], []
    for k in range(5):
        test = np.where(contig == k)[0]
        near = (np.abs(np.arange(n)[:, None] - test[None, :]) <= emb).any(1)
        train = np.where((contig != k) & ~near)[0]
        beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
        preds.append(X[test] @ beta); acts.append(y[test])
    ic = np.corrcoef(np.concatenate(preds), np.concatenate(acts))[0, 1]
    print(f"embargo {emb:>2d} days: CV correlation {ic:+.3f}")
# => embargo  0 days: CV correlation -0.087
#    embargo  5 days: CV correlation -0.088
#    embargo 10 days: CV correlation -0.089
#    embargo 21 days: CV correlation -0.091
#    embargo 42 days: CV correlation -0.094
#    embargo 63 days: CV correlation -0.095
```

The corrections are real, monotone — every extra day of guard band nudges the estimate honestly downward — and *small*, third-decimal small, because five contiguous folds of twelve hundred observations have only eight boundaries, and 21-day labels can only leak across a boundary's width. That smallness is not a reason to skip the embargo; it is the *geometry lesson* of this whole topic, stated by counting: leakage is proportional to boundary-days over total days. Long contiguous folds make the ratio tiny; short folds multiply boundaries; shuffling — the previous section's villain — makes *every* day a boundary day, which is why its error was fifty times larger than anything in this table. Set the embargo to the label horizon plus the memory of the slowest feature, accept the tiny cost in training data, and spend your vigilance where the geometry says the danger lives: on the split design, not the guard band's exact width.

## White's Reality Check

Part III convicted the fifty-variant momentum grid informally; now the formal instrument. White's Reality Check asks exactly the right question: *under the null that no variant has any edge, how often does the best of fifty correlated variants look this good?* — answered by resampling the entire family's history with the [stationary bootstrap](../part-03-statistics/05-bootstrap-and-monte-carlo.md), preserving each variant's correlation with its siblings ([White's Reality Check](../appendix/part-15-multiple-testing/05-whites-reality-check.md); [Hansen's SPA test](../appendix/part-15-multiple-testing/06-hansens-spa-test.md) is its stricter descendant):

```python
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
fam = pd.DataFrame({lb: (np.sign(r.rolling(lb).sum()).shift(1) * r)
                    for lb in range(10, 501, 10)}).dropna()

obs = np.sqrt(252) * fam.mean() / fam.std()
print(f"family of {fam.shape[1]} lookback variants: best Sharpe {obs.max():+.2f} "
      f"(lookback {obs.idxmax()})")
demeaned = fam - fam.mean()
bs = StationaryBootstrap(63, demeaned, seed=42)
count, B = 0, 500
for data in bs.bootstrap(B):
    boot = data[0][0]
    best = (np.sqrt(252) * boot.mean() / boot.std()).max()
    count += best >= obs.max()
print(f"Reality Check p-value (stationary bootstrap, {B} draws): {count / B:.2f}")
# => family of 50 lookback variants: best Sharpe +0.46 (lookback 180)
#    Reality Check p-value (stationary bootstrap, 500 draws): 0.06
```

The family's champion prints Sharpe 0.46 — and after the search is charged for, its p-value is 0.06: in six of every hundred alternate histories where *no lookback has any edge at all*, the best of the family looks this good anyway. A hair from conventional significance, from a family built on the best-documented anomaly in finance, on twenty-four years of data. Sit with how sobering that is. The Reality Check's mechanics deserve one paragraph of respect: demeaning each variant imposes the null; resampling in random-length blocks (mean 63 days) preserves volatility clustering and the family's internal correlation; and taking the *max* over variants inside every bootstrap draw is what prices the selection — the same fifty-way maximum the naive backtest quietly takes and never reports. The difference between "my best variant has Sharpe 0.46" and "my search produced p = 0.06" is the difference between marketing and statistics. One caveat travels with that number and it is arithmetic rather than judgement: a resampled p-value is a sample proportion, so 500 draws put a standard error of about 0.011 on a p of 0.06, and [Bootstrap Methods](../appendix/part-09-monte-carlo-methods/07-bootstrap-methods.md) measures how often that misplaces a result across the 0.05 line.

## The probability of backtest overfitting

The Reality Check judges the family against an external null. Combinatorially symmetric cross-validation (CSCV) asks a nastier, internal question: *when this research process picks its in-sample winner, how often is that winner below the family's median out of sample?* Split history into 16 blocks; for every one of the $\binom{16}{8} = 12{,}870$ ways to call half of them "in-sample," find the IS winner and check its OOS rank:

```python
import numpy as np
import pandas as pd
from itertools import combinations

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
fam = pd.DataFrame({lb: (np.sign(r.rolling(lb).sum()).shift(1) * r)
                    for lb in range(10, 501, 10)}).dropna()

S = 16
blocks = np.array_split(np.arange(len(fam)), S)
bsum = np.array([fam.iloc[b].sum().values for b in blocks])
bsq = np.array([(fam.iloc[b] ** 2).sum().values for b in blocks])
bn = np.array([len(b) for b in blocks])

def sharpes_of(idx):
    n = bn[list(idx)].sum()
    mu = bsum[list(idx)].sum(0) / n
    return mu / np.sqrt(bsq[list(idx)].sum(0) / n - mu ** 2)

splits = list(combinations(range(S), S // 2))
below = 0
for ins in splits:
    oos = [b for b in range(S) if b not in ins]
    star = np.argmax(sharpes_of(ins))
    oos_sh = sharpes_of(oos)
    below += (oos_sh < oos_sh[star]).mean() < 0.5
print(f"CSCV over {len(splits):,} splits of {S} blocks: PBO = {below / len(splits):.2f}")
print("(probability the in-sample winner lands in the OOS bottom half)")
# => CSCV over 12,870 splits of 16 blocks: PBO = 0.60
#    (probability the in-sample winner lands in the OOS bottom half)
```

PBO = 0.60: the research process "pick the grid's best Sharpe" delivers a strategy in the out-of-sample *bottom half* of its own family sixty percent of the time — worse than picking a lookback at random. Selection on this grid is not merely uninformative; it is mildly *anti*-informative, because in a family where true differences are dwarfed by noise (the plateau, again), ranking by in-sample Sharpe ranks mostly by luck, and luck's OOS forecast is reversion. Note what PBO measures that nothing else in this lesson does: not whether the strategy family has edge, but whether the *selection procedure* adds value over ignorance — a question about the research process itself. A desk can run CSCV on any campaign that kept its trials (which is why [lesson five](05-feature-and-signal-engineering.md) made keeping them a ritual), and a PBO near one-half is the machine's way of saying: your grid search was an expensive random-number generator.

## The deflated Sharpe ratio

The final instrument compresses the whole lesson into one number. The deflated Sharpe ratio asks: what is the probability the true Sharpe exceeds zero, *after* charging for the number of trials, the non-normality of returns, and the track length?

$$
\mathrm{DSR} \;=\; \Phi\!\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
{\sqrt{1 - \gamma_3\,\widehat{SR} + \tfrac{\gamma_4 - 1}{4}\,\widehat{SR}^2}}\right),
\qquad
SR_0 = \sqrt{V[SR]}\;\mathbb{E}\big[\max_N Z\big],
$$

where $SR_0$ is the best Sharpe a search of $N$ dead variants was *expected* to find, and skew $\gamma_3$ and kurtosis $\gamma_4$ widen the error bars the way fat tails demand. Run it three ways:

```python
import numpy as np
import pandas as pd
from scipy import stats

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
fam = pd.DataFrame({lb: (np.sign(r.rolling(lb).sum()).shift(1) * r)
                    for lb in range(10, 501, 10)}).dropna()

def dsr(s, n_trials, trial_sr_var):
    T, sr = len(s), s.mean() / s.std()
    g3, g4 = stats.skew(s), stats.kurtosis(s, fisher=False)
    sr0 = 0.0 if n_trials <= 1 else np.sqrt(trial_sr_var) * (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) /
                          np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))

daily_sr = fam.mean() / fam.std()
best = daily_sr.idxmax()
print(f"grid best (lookback {best}, Sharpe {np.sqrt(252) * daily_sr[best]:.2f}), "
      f"as best of 50 trials: DSR = {dsr(fam[best], 50, daily_sr.var()):.2f}")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
print(f"tsmom (pre-committed 252 days), as 1 trial: "
      f"DSR = {dsr(tsmom, 1, daily_sr.var()):.2f}")
print(f"the same tsmom, if it had been picked from the grid: "
      f"DSR = {dsr(tsmom, 50, daily_sr.var()):.2f}")
# => grid best (lookback 180, Sharpe 0.46), as best of 50 trials: DSR = 0.86
#    tsmom (pre-committed 252 days), as 1 trial: DSR = 0.93
#    the same tsmom, if it had been picked from the grid: DSR = 0.64
```

The last two lines are this entire part's thesis, stated by a formula. The *identical* strategy — same rule, same returns, same 0.30 — scores DSR 0.93 as a pre-registered hypothesis and 0.64 as a grid selection, because the deflation term depends not on the returns but on the *history of the research process that produced them*. Pre-commitment is worth 0.29 of probability, and it was purchased in lesson one with a single sentence written before any backtest ran. Meanwhile the grid's best variant, its 0.46 Sharpe glittering above everything the honest process earned, deflates to 0.86 — a six-to-one bet in its favor, respectable and far from the certainty its raw number implies. And the honest reading cuts both ways: 0.93 is not 0.99 either — a quarter century of data leaves a one-in-fourteen chance the pre-committed trend book is a mirage, which is the correctly-sized humility for a Sharpe of 0.30, and the reason [lesson six](06-position-sizing-and-risk-budgeting.md) sized it like an estimate rather than a fact.

## The gauntlet, assembled

Every gate this part has built, applied to every strategy it built them for — the table [the index page](index.md) promised, with nothing airbrushed:

```python
rows = [
    ("sleeve", "hypothesis", "edge", "execution", "net Sh", "multiplicity", "verdict"),
    ("tsmom", "committed", "faint, real", "robust", "+0.29", "DSR 0.93 as 1 trial", "PASS"),
    ("pairs", "committed", "tether real", "collapses", "-0.07", "-", "FAIL"),
    ("xsmom", "committed", "IC 0.012", "robust", "+0.06", "-", "FAIL"),
    ("shortvol", "committed", "VRP real", "monthly", "+1.49", "1 filter, 1 history", "PASS*"),
    ("tom", "none", "t = 0.73", "robust", "+0.29", "0 of 19 pass FDR", "FAIL"),
]
for r in rows:
    print(f"{r[0]:9s} {r[1]:10s} {r[2]:12s} {r[3]:10s} {r[4]:>6s}  {r[5]:19s} {r[6]}")
# => sleeve    hypothesis edge         execution  net Sh  multiplicity        verdict
#    tsmom     committed  faint, real  robust      +0.29  DSR 0.93 as 1 trial PASS
#    pairs     committed  tether real  collapses   -0.07  -                   FAIL
#    xsmom     committed  IC 0.012     robust      +0.06  -                   FAIL
#    shortvol  committed  VRP real     monthly     +1.49  1 filter, 1 history PASS*
#    tom       none       t = 0.73     robust      +0.29  0 of 19 pass FDR    FAIL
```

Two of five survive, and each obituary names a different killer — which is the reason all the gates exist. `pairs` had a real hypothesis and a real tether and died at the execution gate, its Sharpe living entirely inside a fill assumption. `xsmom` had a real literature and died at the edge gate, its information coefficient four times too small for its breadth. `tom` sailed through execution and costs and died where it was always going to die — at multiplicity, a calendar cell with no mechanism, drawn from a family where nothing survives correction. The survivors carry asterisks honestly: `tsmom` passes as a *faint* edge whose every diagnostic — walk-forward at 0.34, DSR at 0.93 — says "real, small, handle with humility"; `shortvol` passes conditionally, its 1.49 net Sharpe leaning on a term-structure filter validated on exactly one history containing exactly one March 2020, and sized by its tail rather than its variance. Three failures kept on the books, two survivors kept on probation: that ratio is not this course being unlucky. It is what an honest research process *yields*, and any process reporting a better ratio is usually reporting a worse ledger.

!!! warning "Any validation gate you would not accept a 'fail' from is not a gate — it is a decoration"
    The test of a validation process is behavioral, not statistical: what happened the last time it said no? If every strategy that reaches the gauntlet passes the gauntlet, the selection is happening upstream, unrecorded, where no correction can reach it — and the gates are theater. Decide before running any gate what result kills the strategy, write it down, and let it kill. A research process is worth exactly as much as the most promising thing it has ever thrown away.

!!! abstract "Key takeaways"
    - The null comes armed: the best of fifty coin-flip strategies printed Sharpe 0.43 against an expected 0.45 — most of this part's own strategies earn less than an uninformed search is *expected* to find, which is why trial counting is not optional.
    - Walk-forward prices hindsight at 40%: the grid's retrospective 0.56 becomes an implementable 0.34 — degrading toward the plateau, not to zero, which is itself the signature of a real but parameter-insensitive effect.
    - Shuffled cross-validation manufactured a +0.061 "skill" from features known to contain nothing; contiguous folds restored the honest −0.087, and purging added a third-decimal correction — leakage scales with the boundary the split creates.
    - Embargoes buy real but small corrections on long contiguous folds (−0.087 to −0.095 across the sweep) because leakage is boundary-days over total days: vigilance belongs on split design, where the geometry is decided.
    - White's Reality Check charges the momentum family for its own search: best variant Sharpe 0.46, snooping-robust p = 0.06 — one coin toss from the null, on finance's best-documented anomaly.
    - CSCV reports PBO = 0.60: the in-sample winner lands in the out-of-sample bottom half more often than chance — this grid's selection step was an expensive random-number generator.
    - The DSR states the part's thesis as arithmetic: the same tsmom scores 0.93 pre-committed and 0.64 grid-selected — pre-registration was worth 0.29 of probability and cost one sentence written in advance.
    - The gauntlet passes two of five, each failure killed by a different gate — execution, edge, multiplicity — and each survivor annotated with the humility its diagnostics demand.

## Where this goes next

Part IV is complete, and its yield is deliberately unglamorous: one faint, robust trend book; one conditional insurance book; three documented corpses whose autopsies — breadth, execution, multiplicity — are worth more than most live strategies; and a validation gauntlet that now stands between every future idea and every future dollar. What the gauntlet cannot yet do is run at scale: every test in this part was hand-built against one cached history, evaluated at daily closes, with costs bolted on afterward — and when it does run at scale, the trial counting has to scale with it, which is the subject of [Distributed Backtesting](../advanced/09-distributed-backtesting.md). [Part V — Inside a Backtesting Engine](../part-05-backtesting-engine/index.md) industrializes exactly this machinery — event loops, fill models, portfolio accounting, and the discipline of point-in-time data — so that the questions this part asked by hand get asked automatically, of every strategy, every time.
