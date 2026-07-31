# Feature Engineering for ML

[Part VI](../part-06-live-infrastructure/06-secrets-paper-live-compliance.md) ended by pointing at the one component its checklists could not harden: the strategy itself. The sign of a 252-day sum — chosen in [Part IV](../part-04-strategy-development/01-momentum-and-trend-following.md) for its honesty, Sharpe 0.30 vectorized and 0.38 through the engine — now runs on infrastructure that restarts cleanly, alerts precisely, and reconciles to the penny. This part goes looking for something better, and the search begins in an unglamorous place: not with a model, but with a dataset. Machine learning's dirty secret in finance is that the model is rarely the decision that matters. The features, the labels, the weights, and the leaks decide the outcome before the first tree is grown.

The bar this part must clear was set long ago and is not negotiable: tsmom's 0.30, buy-and-hold's 0.38, both after the [cost model of Part IV](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md). The [index page](index.md) states the doctrine — every added complexity must *demonstrate* that it earns its keep — and this lesson supplies the evidence base for that demonstration: a feature matrix engineered under the same discipline Part V applied to bars and Part VI applied to processes. Everything is built from the frozen caches of earlier parts; nothing is downloaded.

The product is a single file. By the end of this lesson, `data/part7.parquet` holds nineteen features, two labelings, a sample weight, and the tsmom signal for SPY, TLT, and GLD — leak-audited, uniqueness-weighted, and frozen. Lessons two through five read that file the way Part V's engine read its bars: as the source of truth that no library upgrade or vendor revision can quietly rewrite.

!!! note "Versions"
    Part VII assumes Python 3.12+ with the Part III stack (NumPy 2.x, pandas 3.x, SciPy, statsmodels) plus scikit-learn 1.9, XGBoost 3.3, LightGBM 4.7, and PyTorch 2.13 — the CPU wheel (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) is deliberate: every model in this part trains on a laptop CPU in minutes, and the pinned numbers assume these versions with the seeds shown in each block.

## A feature is an opinion with a timestamp

A feature is a claim that something knowable at time $t$ says something about what happens after $t$. Prices contribute trend and reversal claims at several horizons; the high-low-open geometry of each bar contributes range and gap claims; volume contributes participation claims; and from daily OHLCV alone you can even construct *microstructure* claims — the Corwin–Schultz spread estimator reads transaction costs out of two-day high-low overlap, and Amihud's ratio reads illiquidity out of how much price a dollar of volume moves. These are estimates of quantities that properly live at tick resolution ([Alternative Data](../advanced/07-alternative-data.md) covers the real thing); daily proxies are what our frozen caches can honestly support. To the bar data of `data/part5.parquet` — which [Part V](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) verified against Part III's cache to within 0.01 bp — we add two market-state features from Part IV's file: the VIX level and its five-day change.

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()
mkt = pd.read_parquet("data/part4.parquet")
c, h, l, o, v = (bars[k] for k in ["Close", "High", "Low", "Open", "Volume"])
r, dv = np.log(c).diff(), c * v

hl2 = np.log(h / l) ** 2
beta = hl2 + hl2.shift(1)
gamma = np.log(h.rolling(2).max() / l.rolling(2).min()) ** 2
alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
f = pd.DataFrame({
    "f_ret_1": r, "f_ret_5": r.rolling(5).sum(), "f_ret_21": r.rolling(21).sum(),
    "f_ret_63": r.rolling(63).sum(), "f_ret_252": r.rolling(252).sum(),
    "f_dist_hi": np.log(c / c.rolling(252).max()),
    "f_vol_21": r.rolling(21).std(), "f_volratio": r.rolling(21).std() / r.rolling(63).std(),
    "f_park": np.sqrt((hl2 / (4 * np.log(2))).rolling(21).mean()),
    "f_range": np.log(h / l), "f_gap": np.log(o / c.shift(1)),
    "f_volz": (v - v.rolling(63).mean()) / v.rolling(63).std(),
    "f_dvolz": (np.log(dv) - np.log(dv).rolling(252).mean()) / np.log(dv).rolling(252).std(),
    "f_amihud": (r.abs() / dv).rolling(21).mean() * 1e10,
    "f_cs": (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(21).mean(),
    "f_acorr": r.rolling(21).apply(lambda x: pd.Series(x).autocorr(), raw=True),
    "f_vix": mkt["VIX"], "f_vix_chg": np.log(mkt["VIX"]).diff(5),
}).dropna()

corr = f.corr().abs()
pairs = [(a, b) for i, a in enumerate(f.columns) for b in f.columns[i + 1:] if corr.loc[a, b] > 0.8]
slope_start = (mkt["VIX"] - mkt["VIX3M"]).dropna().index[0]
print(f"panel: {f.shape[0]} rows x {f.shape[1]} features, {f.index[0].date()} -> {f.index[-1].date()}")
print(f"warmup consumed: {len(bars) - len(f)} of {len(bars)} SPY bars")
print(f"pairs with |rho| > 0.8: {len(pairs)}, tightest {max(pairs, key=lambda p: corr.loc[p])} "
      f"rho {corr.loc[max(pairs, key=lambda p: corr.loc[p])]:.2f}")
print(f"cost of a VIX3M slope feature: history starts {slope_start.date()}, "
      f"not {f.index[0].date()} - {len(f.loc[:slope_start])} rows surrendered")
# => panel: 6159 rows x 18 features, 2001-01-02 -> 2025-06-30
#    warmup consumed: 252 of 6411 SPY bars
#    pairs with |rho| > 0.8: 8, tightest ('f_vol_21', 'f_park') rho 0.97
#    cost of a VIX3M slope feature: history starts 2006-07-17, not 2001-01-02 - 1391 rows surrendered
```

Three numbers in that printout will matter for the rest of the part. First, the warmup: the 252-day lookbacks silently spend a year of history before the first usable row, so 6,411 bars become 6,159 — every feature you add is also a claim on your sample. Second, the correlation structure: eight pairs above $|\rho| = 0.8$, with 21-day close-to-close volatility and 21-day Parkinson volatility correlated at 0.97 — they are, after all, two estimators of the same latent quantity. Nothing is wrong with that redundancy *as input*; trees handle it gracefully. What it poisons is *interpretation*, and [Tree Ensembles](02-tree-ensembles.md) will show feature-importance rankings dissolving on exactly this pair. Third, the last line prices a temptation: the VIX term-structure slope is a genuinely informative feature, but its short leg only exists from mid-2006, so admitting it would surrender 1,391 rows — five and a half years containing an entire bear market. A feature buys information and sells history, and at 6,159 rows the exchange rate is brutal. The scoring of individual signals — information coefficients, decay, the fundamental law — was [Part IV's fifth lesson](../part-04-strategy-development/05-feature-and-signal-engineering.md) and is not repeated here; this part's concern is what happens when a model is allowed to combine them.

## Stationarity is a dial, not a switch

[Part III](../part-03-statistics/03-time-series.md) ran the ADF/KPSS pincer on this exact series and delivered the standard verdict: the log price level is nonstationary, returns are stationary. The standard response is to feed models returns and discard the level. But that discards something real — *where price sits* relative to its own past is information, and first-differencing erases all of it. Fractional differencing asks the question the binary transform never poses: how much differencing does stationarity actually require? The operator applies weights $w_0 = 1$, $w_k = -w_{k-1}\frac{d-k+1}{k}$ to lagged prices; $d=1$ recovers ordinary returns, $d=0$ the level, and the interesting territory is between.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
logp = np.log(c)

def ffd(x, d, width=252):                       # fixed-width fractional differencing
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    return pd.Series(np.convolve(x, w, "valid"), index=x.index[width - 1:])

def adf(s):                                     # one augmentation lag throughout
    return adfuller(s.dropna(), maxlag=1, autolag=None)[0]

print(f"ADF, 5% critical value -2.86: log price {adf(logp):+.2f}, "
      f"log returns {adf(logp.diff()):+.2f}")
ds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
stats = [adf(ffd(logp, d)) for d in ds]
corrs = [ffd(logp, d).corr(logp) for d in ds]
for d, s_, c_ in zip(ds, stats, corrs):
    print(f"  d={d:.1f}: ADF {s_:+7.2f}   corr with log price {c_:+.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(ds, stats, marker="o")
ax.axhline(-2.86, ls="--", color="gray")
ax.set_xlabel("differencing order d")
ax.set_ylabel("ADF statistic")
ax2 = ax.twinx()
ax2.plot(ds, corrs, marker="s", color="tab:orange")
ax2.set_ylabel("correlation with log price (orange)")
plt.show()
# => ADF, 5% critical value -2.86: log price +0.79, log returns -60.45
#      d=0.0: ADF   +0.72   corr with log price +1.000
#      d=0.2: ADF   -1.54   corr with log price +0.996
#      d=0.3: ADF   -3.38   corr with log price +0.985
#      d=0.4: ADF   -6.63   corr with log price +0.954
#      d=0.5: ADF  -12.54   corr with log price +0.871
#      d=0.7: ADF  -34.53   corr with log price +0.418
#      d=1.0: ADF  -58.95   corr with log price +0.025
```

Read the two columns against each other. The ADF statistic crosses its critical value between $d=0.2$ and $d=0.3$: a series that is 98.5% correlated with the raw log price already rejects a unit root. Full differencing, the industry default, sits at the far end of the table having destroyed nearly everything — returns correlate with the level at 0.025, which is precisely why every trend feature in section one had to *rebuild* memory by summing returns over windows. Fractional differencing keeps the memory and sheds the trend. We freeze $d=0.4$ rather than the knife-edge 0.3: the rejection at $-6.63$ is decisive rather than marginal, the correlation with the level is still 0.954, and a feature whose stationarity depends on which side of a critical value a test statistic lands is not a foundation to build on. One honesty note: the augmentation lag is pinned to one throughout — Part III's default-lag test printed +0.92 for the same level — because lag choice moves ADF's power substantially, and a sweep is only comparable to itself if the test underneath it holds still.

## Fixed-horizon labels grade the calendar, not the trade

Features are half the dataset; the label is the other half, and it is the half nobody audits. The default in every ML-for-finance tutorial is the fixed-horizon label: pick a horizon, compute the forward return, threshold at zero. Five days it is:

```python
import numpy as np
import pandas as pd

c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
r = np.log(c).diff()
fwd5 = np.log(c).diff(5).shift(-5)              # 5-day forward log return
y = (fwd5 > 0).astype(float)[fwd5.notna()]

vol = r.rolling(21).std()
tercile = pd.qcut(vol, 3, labels=["calm", "mid", "stressed"])
spread = fwd5.abs().groupby(tercile, observed=True).median()
print(f"base rate P(up): {y.mean():.3f}   label autocorrelation lag-1: {y.autocorr(1):+.2f}")
print(f"median |5d move| by vol regime: calm {spread['calm']:.2%}, "
      f"mid {spread['mid']:.2%}, stressed {spread['stressed']:.2%}")
calm_lo = fwd5[tercile == "calm"].abs().quantile(0.10)
big = fwd5[tercile == "stressed"].abs().max()
print(f"a label in calm markets can be a {calm_lo:.2%} drift; in stressed markets "
      f"the same class covers a {big:.1%} move - {big / calm_lo:,.0f}x the size")
# => base rate P(up): 0.580   label autocorrelation lag-1: +0.59
#    median |5d move| by vol regime: calm 0.87%, mid 1.27%, stressed 2.08%
#    a label in calm markets can be a 0.17% drift; in stressed markets the same class covers a 22.1% move - 134x the size
```

Every line is a pathology. The base rate of 0.580 means a model that always says "up" scores 58% accuracy before learning anything — every accuracy figure in this part must be read against that floor, not against 50%. The lag-one autocorrelation of +0.59 is not a property of markets but of arithmetic: consecutive 5-day labels share four of their five days, so the dataset holds far fewer independent facts than its row count advertises — section five quantifies exactly how few. And the third line is the deepest problem: the label is *indifferent to magnitude*. A 0.17% drift in a calm market and a 22.1% collapse in a stressed one — 134 times the size — receive labels of equal weight, so a model minimizing classification loss is being told, falsely, that these events matter equally. The fixed horizon grades the calendar. No trade you would actually run works that way: real positions exit on stops, targets, and time — which is precisely the structure the next labeling scheme imports.

## Triple-barrier labels respect the exit

The triple-barrier method labels each observation by the first of three events on the *path* of forward prices: an upper barrier (profit-taking), a lower barrier (stop-loss), or a vertical barrier (time's up). Scale the horizontal barriers by current volatility and the label finally means the same thing in 2017 and 2008: "a two-and-a-half-sigma move in your favor before one against you, within a month." The barrier multiple is a design choice with visible consequences, so we sweep it:

```python
import numpy as np
import pandas as pd

c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
logc, r = np.log(c), np.log(c).diff()
vol = r.rolling(21).std()

def triple_barrier(logc, vol, mult, horizon=21):
    out = {}
    for i in range(len(logc) - 1):
        if not np.isfinite(vol.iloc[i]) or vol.iloc[i] == 0:
            continue
        j = min(i + horizon, len(logc) - 1)
        path = logc.iloc[i + 1:j + 1].values - logc.iloc[i]
        up, dn = np.nonzero(path >= mult * vol.iloc[i])[0], np.nonzero(path <= -mult * vol.iloc[i])[0]
        fu, fd = (up[0] if len(up) else np.inf), (dn[0] if len(dn) else np.inf)
        if fu < fd:   k, hit, y = int(fu), "upper", 1
        elif fd < fu: k, hit, y = int(fd), "lower", -1
        else:         k, hit, y = j - i - 1, "vertical", (1 if path[-1] > 0 else -1)
        out[logc.index[i]] = (y, hit, logc.index[i + 1 + k], k + 1)
    return pd.DataFrame.from_dict(out, orient="index", columns=["y_tb", "barrier", "t1", "days"])

for m in [2.0, 2.5, 3.0]:
    lab = triple_barrier(logc, vol, m)
    pct = lab["barrier"].value_counts(normalize=True)
    print(f"mult {m}: upper {pct.get('upper', 0):.0%}  lower {pct.get('lower', 0):.0%}  "
          f"vertical {pct.get('vertical', 0):.0%}  median touch {lab['days'].median():.0f}d  "
          f"P(y=+1) {(lab['y_tb'] > 0).mean():.3f}")
# => mult 2.0: upper 54%  lower 42%  vertical 4%  median touch 5d  P(y=+1) 0.567
#    mult 2.5: upper 50%  lower 40%  vertical 10%  median touch 8d  P(y=+1) 0.573
#    mult 3.0: upper 45%  lower 37%  vertical 18%  median touch 10d  P(y=+1) 0.580
```

The sweep shows the dial's mechanics: tighter barriers make the label a coin-flip race decided in days (at 2.0, only 4% of paths survive to the time limit), wider ones let the vertical barrier — where the label degrades back into a fixed-horizon sign — reclaim nearly a fifth of the sample. We freeze the middle setting: at 2.5 sigmas, half the labels are genuine profit-target hits, 40% are stop-outs, only one in ten limps to the deadline, and the median resolution of eight days matches the holding period of the signals this part will actually trade. Two structural facts distinguish these labels from section three's. Each label now carries its own resolution time `t1` — the moment its outcome became knowable — which is exactly the metadata that purged cross-validation will need in [the next lesson](02-tree-ensembles.md), and which the fixed-horizon label kept implicit. And the class balance barely moved (0.573 against 0.580), a reminder that the barrier method does not manufacture signal; it changes what the label *means*, from "where was price on day five" to "which exit fired first" — the question a trade actually answers.

## Overlap is not evidence

Section three showed consecutive labels sharing their underlying days; the triple-barrier labels overlap too, since a label opened today lives until its barrier fires, and tomorrow opens another. The consequence is not cosmetic. Two overlapping labels are not two pieces of evidence about the market; they are closer to one piece of evidence counted twice. The standard correction weights each observation by its *average uniqueness* — one over the number of concurrently open labels, averaged over the label's lifetime:

```python
import numpy as np
import pandas as pd

c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
logc, r = np.log(c), np.log(c).diff()
vol = r.rolling(21).std()

# the labeler of the previous section, mult 2.5
def triple_barrier(logc, vol, mult=2.5, horizon=21):
    out = {}
    for i in range(len(logc) - 1):
        if not np.isfinite(vol.iloc[i]) or vol.iloc[i] == 0:
            continue
        j = min(i + horizon, len(logc) - 1)
        path = logc.iloc[i + 1:j + 1].values - logc.iloc[i]
        up, dn = np.nonzero(path >= mult * vol.iloc[i])[0], np.nonzero(path <= -mult * vol.iloc[i])[0]
        fu, fd = (up[0] if len(up) else np.inf), (dn[0] if len(dn) else np.inf)
        k = int(min(fu, fd)) if min(fu, fd) < np.inf else j - i - 1
        y = 1 if path[k] > 0 else -1
        out[logc.index[i]] = (y, logc.index[i + 1 + k])
    return pd.DataFrame.from_dict(out, orient="index", columns=["y_tb", "t1"])

lab = triple_barrier(logc, vol)
pos = pd.Series(np.arange(len(c)), index=c.index)
i0, i1 = pos.reindex(lab.index).astype(int).values, pos.reindex(lab["t1"]).astype(int).values
conc = np.zeros(len(c))
for a, b in zip(i0, i1):
    conc[a:b + 1] += 1
w = pd.Series([np.mean(1.0 / conc[a:b + 1]) for a, b in zip(i0, i1)], index=lab.index)
print(f"labels: {len(lab):,}   mean concurrency: {conc[conc > 0].mean():.1f}   "
      f"mean uniqueness: {w.mean():.3f}")
print(f"effective sample size: {w.sum():,.0f} of {len(lab):,} rows")

p = (lab["y_tb"] > 0).mean()
se_naive = np.sqrt(p * (1 - p) / len(lab))
se_eff = np.sqrt(p * (1 - p) / w.sum())
print(f"P(y=+1) = {p:.3f}: against a coin flip, t = {(p - .5) / se_naive:.1f} "
      f"at face-value n, t = {(p - .5) / se_eff:.1f} at effective n")
# => labels: 6,389   mean concurrency: 10.4   mean uniqueness: 0.106
#    effective sample size: 674 of 6,389 rows
#    P(y=+1) = 0.573: against a coin flip, t = 11.8 at face-value n, t = 3.8 at effective n
```

Sit with the second line. Twenty-four years of daily labels — 6,389 rows — carry the evidential weight of *674 independent observations*. At any moment, 10.4 labels are open simultaneously, each watching overlapping stretches of the same price path, and the uniqueness weight of 0.106 is the honest exchange rate between rows and facts. The third line shows what that exchange rate does to inference: the 57.3% base rate stands eleven standard errors from a coin flip if you believe the row count, and 3.8 if you believe the evidence. Still significant — the upward drift is real — but the gap between 11.8 and 3.8 is the size of the lie a naive sample count tells, and it is the same lie an unweighted classifier ingests when it treats each row as a fresh fact: it converges with false confidence on whatever regime the overlap happens to over-represent. The weight column `w` travels with the dataset from here on, and every model in this part passes it to `sample_weight`. This is [Part III's](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) effective-sample-size discipline resurfacing at the dataset layer — the row count is an implementation detail; the information count is what earns conclusions.

## Leakage is audited, not assumed

[Part IV's validation lesson](../part-04-strategy-development/08-validation-and-overfitting.md) demonstrated leakage at the *split* layer: shuffled cross-validation manufactured a +0.061 out-of-sample correlation from features known to contain nothing. Leakage also enters one layer earlier, inside the feature pipeline itself, and it does not announce itself — pipelines with future-peeking features run without error, produce plausible backtests, and die in production. So we do to the pipeline what Part V did to the backtest: make it prove itself. Plant three classic leaks among honest features, and then run the one audit that catches all three mechanically — recompute every feature at a date $t$ using *only data up to* $t$, and demand the value reproduce exactly.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
r = np.log(c).diff()
fwd5 = np.log(c).diff(5).shift(-5)

honest = pd.DataFrame({"mom21": r.rolling(21).sum(), "vol21": r.rolling(21).std(),
                       "mom252": r.rolling(252).sum()})
leaks = honest.assign(
    smooth=r.rolling(11, center=True).mean(),      # leak 1: centered window
    zmom=(honest["mom21"] - honest["mom21"].mean()) / honest["mom21"].std(),  # leak 2: full-sample z
    lagjoin=fwd5.shift(-1),                        # leak 3: the join shifted the wrong way
)

def oos_auc(F):
    df = F.assign(y=(fwd5 > 0).astype(int))[fwd5.notna()].dropna()
    tr, te = df.loc[:"2016"], df.loc["2017":]
    m = LogisticRegression(max_iter=1000).fit(tr.drop(columns="y"), tr["y"])
    return roc_auc_score(te["y"], m.predict_proba(te.drop(columns="y"))[:, 1])

print(f"honest features, 2017+ AUC: {oos_auc(honest):.3f}")
for col in ["smooth", "zmom", "lagjoin"]:
    print(f"  + {col:8s}: AUC {oos_auc(honest.assign(**{col: leaks[col]})):.3f}")

# the audit: recompute every feature from history truncated at t - a leak cannot survive it
t = pd.Timestamp("2015-06-30")
trunc_r = np.log(c[:t]).diff()
full = {"mom21": r.rolling(21).sum(), "vol21": r.rolling(21).std(),
        "mom252": r.rolling(252).sum(), "smooth": r.rolling(11, center=True).mean(),
        "zmom": (r.rolling(21).sum() - r.rolling(21).sum().mean()) / r.rolling(21).sum().std(),
        "lagjoin": fwd5.shift(-1)}
trunc = {"mom21": trunc_r.rolling(21).sum(), "vol21": trunc_r.rolling(21).std(),
         "mom252": trunc_r.rolling(252).sum(), "smooth": trunc_r.rolling(11, center=True).mean(),
         "zmom": (trunc_r.rolling(21).sum() - trunc_r.rolling(21).sum().mean()) / trunc_r.rolling(21).sum().std(),
         "lagjoin": np.log(c[:t]).diff(5).shift(-5).shift(-1)}
for k in full:
    diff = abs(full[k].loc[t] - trunc[k].loc[t]) if np.isfinite(trunc[k].get(t, np.nan)) else np.nan
    verdict = "REPRODUCES" if diff == 0 else ("DIVERGES" if np.isfinite(diff) else "NOT COMPUTABLE")
    print(f"  audit {k:8s}: value at t from truncated history -> {verdict}")
# => honest features, 2017+ AUC: 0.515
#      + smooth  : AUC 0.569
#      + zmom    : AUC 0.515
#      + lagjoin : AUC 0.871
#      audit mom21   : value at t from truncated history -> REPRODUCES
#      audit vol21   : value at t from truncated history -> REPRODUCES
#      audit mom252  : value at t from truncated history -> REPRODUCES
#      audit smooth  : value at t from truncated history -> NOT COMPUTABLE
#      audit zmom    : value at t from truncated history -> DIVERGES
#      audit lagjoin : value at t from truncated history -> NOT COMPUTABLE
```

The honest baseline is 0.515 — hold that number; it is what daily-bar predictability actually looks like, and the next two lessons will spend thousands of trees and parameters trying to improve on it. Now watch the leaks. The centered smoother — five future days hiding inside an innocent-looking `center=True` — buys five points of AUC. The wrong-way join buys thirty-five: 0.871 is the kind of number that ends up in pitch decks, produced by a one-character indexing error. And the full-sample z-score buys *nothing at all* — 0.515, unchanged, because rescaling one feature by constants is invisible to a linear model — yet it is still a leak: its mean and standard deviation are computed from data that did not exist yet, and in a live system the feature's values would silently differ from the backtest's. That is why the metric cannot be your leak detector: one leak inflates it wildly, one modestly, one not at all. The audit catches all three without knowing what it is looking for, and each failure mode is legible — honest features REPRODUCE; the z-score DIVERGES because its scaling constants shifted with the truncation; the smoother and the bad join are NOT COMPUTABLE at the boundary, which is the tell that they *require the future to exist*. This audit is cheap, mechanical, and it is the pipeline's equivalent of Part V's replay test.

## The freeze: one file for the whole part

Everything above now runs once, for all three symbols, and lands in a file. The pipeline is sections one through five condensed: the nineteen-feature panel (the fractional differencer joins at $d=0.4$), the 2.5-sigma triple-barrier labeler with its `t1` column, average-uniqueness weights, the 5-day forward return kept for later diagnostics, and the tsmom signal — computed per symbol on per-symbol frames, [Part V's calendar discipline](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) — because [lesson four](04-reinforcement-learning-and-meta-labeling.md) will need a primary signal to meta-label.

```python
import numpy as np
import pandas as pd

def features(sym, mkt):                          # the panel of section one, per symbol
    bars = pd.read_parquet("data/part5.parquet").xs(sym, axis=1, level=1).dropna()
    c, h, l, o, v = (bars[k] for k in ["Close", "High", "Low", "Open", "Volume"])
    r, dv = np.log(c).diff(), c * v
    w = [1.0]
    for k in range(1, 252):
        w.append(-w[-1] * (0.4 - k + 1) / k)     # d = 0.4, chosen in section two
    hl2 = np.log(h / l) ** 2
    beta, gamma = hl2 + hl2.shift(1), np.log(h.rolling(2).max() / l.rolling(2).min()) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    f = pd.DataFrame({
        "f_ret_1": r, "f_ret_5": r.rolling(5).sum(), "f_ret_21": r.rolling(21).sum(),
        "f_ret_63": r.rolling(63).sum(), "f_ret_252": r.rolling(252).sum(),
        "f_fracdiff": pd.Series(np.convolve(np.log(c), w, "valid"), index=c.index[251:]),
        "f_dist_hi": np.log(c / c.rolling(252).max()),
        "f_vol_21": r.rolling(21).std(), "f_volratio": r.rolling(21).std() / r.rolling(63).std(),
        "f_park": np.sqrt((hl2 / (4 * np.log(2))).rolling(21).mean()),
        "f_range": np.log(h / l), "f_gap": np.log(o / c.shift(1)),
        "f_volz": (v - v.rolling(63).mean()) / v.rolling(63).std(),
        "f_dvolz": (np.log(dv) - np.log(dv).rolling(252).mean()) / np.log(dv).rolling(252).std(),
        "f_amihud": (r.abs() / dv).rolling(21).mean() * 1e10,
        "f_cs": (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(21).mean(),
        "f_acorr": r.rolling(21).apply(lambda x: pd.Series(x).autocorr(), raw=True),
        "f_vix": mkt["VIX"], "f_vix_chg": np.log(mkt["VIX"]).diff(5),
    })
    return f, c, r

def label(c, r, mult=2.5, horizon=21):           # the labeler of section four
    logc, vol, out = np.log(c), r.rolling(21).std(), {}
    for i in range(len(logc) - 1):
        if not np.isfinite(vol.iloc[i]) or vol.iloc[i] == 0:
            continue
        j = min(i + horizon, len(logc) - 1)
        path = logc.iloc[i + 1:j + 1].values - logc.iloc[i]
        up, dn = np.nonzero(path >= mult * vol.iloc[i])[0], np.nonzero(path <= -mult * vol.iloc[i])[0]
        k = int(min(up[0] if len(up) else np.inf, dn[0] if len(dn) else np.inf)
                ) if len(up) + len(dn) else j - i - 1
        out[logc.index[i]] = (1 if path[k] > 0 else -1, logc.index[i + 1 + k])
    return pd.DataFrame.from_dict(out, orient="index", columns=["y_tb", "t1"])

mkt = pd.read_parquet("data/part4.parquet")
frames = []
for sym in ["SPY", "TLT", "GLD"]:
    f, c, r = features(sym, mkt)
    lab = label(c, r)
    pos = pd.Series(np.arange(len(c)), index=c.index)
    i0, i1 = pos.reindex(lab.index).astype(int).values, pos.reindex(lab["t1"]).astype(int).values
    conc = np.zeros(len(c))
    for a, b in zip(i0, i1):
        conc[a:b + 1] += 1
    f = f.join(lab)
    f["w"] = pd.Series([np.mean(1.0 / conc[a:b + 1]) for a, b in zip(i0, i1)], index=lab.index)
    f["ret_5d"] = np.log(c).diff(5).shift(-5)
    f["sig_tsmom"] = np.sign(r.rolling(252).sum())
    f["symbol"] = sym
    frames.append(f.dropna())

mat = pd.concat(frames).sort_index()
mat.to_parquet("data/part7.parquet")
feat = [k for k in mat.columns if k.startswith("f_")]
print(f"frozen: data/part7.parquet  {mat.shape[0]:,} rows x {mat.shape[1]} cols "
      f"({len(feat)} features), {mat.index[0].date()} -> {mat.index[-1].date()}")
for s in ["SPY", "TLT", "GLD"]:
    m = mat[mat.symbol == s]
    print(f"  {s}: {len(m):,} rows from {m.index[0].date()}, P(y=+1) {(m.y_tb > 0).mean():.3f}, "
          f"mean uniqueness {m.w.mean():.3f}")
import os
print(f"file size: {os.path.getsize('data/part7.parquet') / 1e6:.1f} MB")
# => frozen: data/part7.parquet  16,592 rows x 25 cols (19 features), 2001-01-02 -> 2025-06-23
#      SPY: 6,154 rows from 2001-01-02, P(y=+1) 0.578, mean uniqueness 0.106
#      TLT: 5,510 rows from 2003-07-30, P(y=+1) 0.533, mean uniqueness 0.105
#      GLD: 4,928 rows from 2005-11-17, P(y=+1) 0.563, mean uniqueness 0.108
#    file size: 3.3 MB
```

This block runs once; the file is the deliverable. Where Parts III through V froze *vendor bars* — because the vendor rewrites history through dividend adjustments — this part freezes a *derived* dataset, and the reasoning strengthens rather than changes: a feature matrix depends on every parameter above ($d=0.4$, barriers at 2.5 sigmas, 21-day windows), and re-deriving it after any future refactor would silently move every pinned number in the next four lessons. The file ends those arguments before they start. Note what the per-symbol lines already teach: TLT's base rate of 0.533 against SPY's 0.578 says the up-drift that inflates equity labels is much weaker in bonds — a model trained across all three symbols cannot simply learn "predict up" and coast. 16,592 rows, 3.3 megabytes, every value reproducible from the blocks above, and — by section five's arithmetic — the evidential content of roughly 1,750 independent observations across three assets. That is the budget this part's models must live within. Spend it remembering the honest AUC of 0.515.

!!! warning "A leak-free pipeline is a claim you prove, not a property you assume"
    Nobody believes their own pipeline leaks — the person who planted the centered window and the person auditing for it are the same person, five weeks apart. That is why the audit must be mechanical rather than vigilant: recompute every feature from truncated history and demand exact reproduction, the same way Part V demanded the backtest replay from its own log. The three verdicts mean different things — DIVERGES is a preprocessing constant that will shift in production, NOT COMPUTABLE is a feature that requires the future to exist — but both fail the same test, and the test does not care how plausible the feature looked. Run it when the pipeline is born, and run it again every time a feature is added, because leakage is not a stage of development you pass through; it is an entropy the pipeline drifts toward whenever someone touches it.

!!! abstract "Key takeaways"
    - Eighteen candidate features from frozen caches cost 252 bars of warmup and arrive pre-correlated: 8 pairs above $|\rho| = 0.8$, with close-to-close and Parkinson volatility at 0.97 — fine for prediction, fatal for naive importance rankings.
    - Admitting a VIX3M slope feature would surrender 1,391 rows (2001 to mid-2006); a feature buys information and sells history.
    - Fractional differencing at $d = 0.4$ rejects a unit root decisively (ADF $-6.63$ with one augmentation lag) while retaining 0.954 correlation with the log price level; full differencing retains 0.025.
    - Fixed-horizon labels autocorrelate at +0.59 by construction and put a 0.17% drift and a 22.1% crash in the same class; triple-barrier labels at 2.5 sigmas resolve by profit-target (50%), stop (40%), or deadline (10%) with a median touch of 8 days, and carry their resolution time `t1`.
    - Overlap deflates 6,389 SPY labels to an effective sample of 674; the 57.3% base rate is t = 11.8 from a coin flip by row count and t = 3.8 by evidence — the weight column `w` carries that correction to every model downstream.
    - The truncation audit caught all three planted leaks — including the full-sample z-score that moved AUC by nothing — while the honest features reproduced exactly; honest daily-bar AUC is 0.515, against a 0.580 base rate.
    - `data/part7.parquet`: 16,592 rows, 19 features, labels, weights, and the tsmom signal for SPY, TLT, GLD — frozen once; every number in lessons two through five derives from this file.

## Where this goes next

The dataset now exists and its honesty has been paid for: an AUC floor of 0.515, an effective sample a tenth of the row count, and a base rate that makes "always up" score 58%. [Tree Ensembles](02-tree-ensembles.md) sends the first real models against it — random forests and gradient boosting, the workhorses of tabular ML — under purged cross-validation that respects the `t1` column built here. The lesson is structured as a controlled experiment: the ensembles get every advantage first, and only then meet the baseline they must beat — a logistic regression that trains in under a second. The margin between them is the first honest measurement of what machine learning adds on daily bars.
