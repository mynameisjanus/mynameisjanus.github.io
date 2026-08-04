# Time Series Analysis

The most expensive silent assumption in quantitative research is computing a statistic on a series whose distribution was busy changing underneath the calculation. Every estimate in the last two lessons — every moment, correlation, and fitted density — implicitly assumed its inputs came from a process with stable properties. For prices that assumption is false by construction; for returns it is close enough to true to be useful, and the gap between those two statements is where time-series analysis lives.

This lesson tests stationarity instead of assuming it, locates where returns actually have memory, fits the models that exploit that memory — ARIMA for the mean, GARCH for the variance — and ends with cointegration, where the results on real data are messier and more instructive than any textbook example. The formal background is [Statistical Models](../appendix/part-10-statistics-foundations/04-statistical-models.md) in the appendix; everything here runs on the Part III cache.

## Stationarity is the license to estimate

A process is weakly stationary when its first two moments do not depend on when you look:

$$
\mathbb{E}[X_t] = \mu,
\qquad
\operatorname{Var}(X_t) = \sigma^2,
\qquad
\operatorname{Cov}(X_t, X_{t+h}) = \gamma(h) \quad \text{for all } t.
$$

A sample mean only estimates something real if that something exists — stationarity is what makes "the" mean a well-posed object. Four printed numbers show which of our two series qualifies:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
logp = np.log(px["SPY"])
r = logp.diff().dropna()

for a, b in [("2000", "2009"), ("2010", "2019"), ("2020", "2025")]:
    lp, x = logp.loc[a:b], r.loc[a:b]
    print(f"{a}-{b}  log price mean {lp.mean():.2f} std {lp.std():.2f}   "
          f"returns mean {x.mean():+.5f} std {x.std():.4f}")
# => 2000-2009  log price mean 4.37 std 0.18   returns mean -0.00004 std 0.0141
#    2010-2019  log price mean 5.04 std 0.37   returns mean +0.00050 std 0.0093
#    2020-2025  log price mean 6.01 std 0.22   returns mean +0.00053 std 0.0135
```

The log price's "mean" marches from 4.37 to 6.01 — it is not an estimate of anything, just a diary of where the random walk happened to wander (the null model for that wandering is the appendix's [Random Walks](../appendix/part-08-stochastic-processes/11-random-walks.md) page). Returns behave like a process with a home: means within a hair of zero every decade, standard deviations of the same order. Be honest about the fine print, though — those standard deviations differ by half (0.9% vs 1.4%), so strict stationarity fails for returns too. They are *locally* stationary with a slowly wandering variance, which is not a nuisance to be assumed away but the single most modelable structure in finance; the GARCH section below is that sentence turned into an estimator.

## Unit-root tests, and how to read their disagreement

Formal stationarity testing uses two instruments with opposite nulls. **ADF** (augmented Dickey-Fuller) takes the unit root as its null and asks whether the data can reject it; **KPSS** takes stationarity as its null and asks the reverse. Running only one is a common and quietly serious mistake, because failing to reject a null is not evidence for it — you need the pincer:

```python
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

px = pd.read_parquet("data/prices.parquet")
logp = np.log(px["SPY"])
r = logp.diff().dropna()

warnings.filterwarnings("ignore")   # KPSS p-values are table-clipped at 0.01 and 0.10
for name, s in [("log price", logp), ("returns", r)]:
    adf_s, adf_p = adfuller(s)[:2]
    kpss_s, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
    print(f"{name:9s}  ADF {adf_s:7.2f} (p={adf_p:.3f})   KPSS {kpss_s:6.2f} (p={kpss_p:.2f})")
# => log price  ADF    0.92 (p=0.993)   KPSS  11.41 (p=0.01)
#    returns    ADF  -19.99 (p=0.000)   KPSS   0.39 (p=0.08)
```

Both series land in the two consistent cells of the grid — prices are integrated, returns are stationary — but the grid has four cells and the other two occur constantly in practice:

| ADF says | KPSS says | Verdict |
|---|---|---|
| unit root not rejected | stationarity rejected | integrated — difference it |
| unit root rejected | stationarity not rejected | stationary — model it directly |
| unit root rejected | stationarity rejected | neither model fits: breaks, trends, or long memory |
| neither rejects | neither rejects | the sample is too short to say — collect data, not conclusions |

The suppressed warning is worth a sentence: statsmodels clips KPSS p-values at the edges of its lookup table, so "p=0.01" on the log price means *at most* 0.01. And the cointegration sections below will demonstrate that these tests' verdicts can hinge on their lag settings — a fragility to file away now.

## ACF and PACF: where the memory lives

The autocorrelation function is the memory map of a stationary series, with a $\pm 1.96/\sqrt{n}$ significance band separating structure from noise. The pattern that defines equity returns is which transformations of the series have memory:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

a, p, a2 = acf(r, nlags=10), pacf(r, nlags=10), acf(r**2, nlags=10)
print(f"95% band +/- {1.96 / np.sqrt(len(r)):.3f}")
for lag in [1, 2, 5, 10]:
    print(f"lag {lag:2d}: acf {a[lag]:+.3f}  pacf {p[lag]:+.3f}  acf(r^2) {a2[lag]:+.3f}")
# => 95% band +/- 0.024
#    lag  1: acf -0.086  pacf -0.086  acf(r^2) +0.263
#    lag  2: acf -0.017  pacf -0.024  acf(r^2) +0.435
#    lag  5: acf -0.011  pacf -0.016  acf(r^2) +0.291
#    lag 10: acf -0.010  pacf -0.008  acf(r^2) +0.240

roll = r.rolling(756).corr(r.shift(1))       # 3-year rolling lag-1 autocorrelation
for d in ["2009-12-31", "2015-12-31", "2020-12-31"]:
    print(d, f"{roll.asof(d):+.3f}")
# => 2009-12-31 -0.109
#    2015-12-31 +0.015
#    2020-12-31 -0.232
```

Returns clear the band only at lag one, and barely (−0.086) — while squared returns are ten to eighteen times outside it at every lag shown. **The memory is in the magnitude, not the direction.** The rolling window then undermines even that lag-one crumb: the "signal" was −0.11 through the financial crisis, zero mid-decade, −0.23 in the covid year. It is not a stable parameter but a crisis phenomenon — violent days cluster and partially reverse. A model of the conditional *mean* built on this foundation is building on sand; a model of the conditional *variance* is building on the strongest empirical regularity markets offer. The next two sections fit one of each, in that order, so the contrast is on the record.

## ARIMA: fit it, test it, respect the smallness

An ARMA(1,1) on returns is the canonical conditional-mean model: today's return regressed on yesterday's return and yesterday's surprise. Fitting it and — the step that separates modeling from curve-fitting — interrogating its residuals with the Ljung-Box test ([Residual Analysis](../appendix/part-13-regression/07-residual-analysis.md) is why the direction you test along decides what you find):

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

fit = ARIMA(r.values, order=(1, 0, 1)).fit()
const, ar1, ma1, sig2 = fit.params
print(f"ar.L1 {ar1:+.3f}  ma.L1 {ma1:+.3f}  (p = {fit.pvalues[1]:.1e}, {fit.pvalues[2]:.1e})")
# => ar.L1 +0.201  ma.L1 -0.289  (p = 2.4e-04, 5.5e-08)

print(f"Ljung-Box p, residuals:   {acorr_ljungbox(fit.resid, lags=[10])['lb_pvalue'].iloc[0]:.3f}")
print(f"Ljung-Box p, residuals^2: {acorr_ljungbox(fit.resid**2, lags=[10])['lb_pvalue'].iloc[0]:.2e}")
# => Ljung-Box p, residuals:   0.006
#    Ljung-Box p, residuals^2: 0.00e+00
```

The coefficients are statistically significant — with 6,400 observations, almost anything is — and economically feeble: they repackage a lag-one autocorrelation of −0.086, worth well under a basis point of daily predictability before costs. The diagnostics tell the story in stereo. The residuals still fail Ljung-Box at p = 0.006: even after the model, a whisper of linear structure remains, because the mean dynamics are unstable rather than ARMA-shaped. But the *squared* residuals fail at a p-value that underflows to zero — not a whisper but a siren. The mean model is quibbling over crumbs while the variance structure goes entirely unmodeled. Respect what daily index data is saying: the conditional mean is nearly unforecastable, and the interesting model lives one moment higher.

## GARCH: model the volatility instead

GARCH(1,1) makes the wandering variance explicit — today's variance is a blend of the long-run level, yesterday's surprise, and yesterday's variance:

$$
\sigma_t^2 \;=\; \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2 .
$$

```python
import numpy as np
import pandas as pd
from arch import arch_model

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

res = arch_model(100 * r, p=1, q=1, rescale=False).fit(disp="off")
om, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
print(f"omega {om:.4f}  alpha {al:.3f}  beta {be:.3f}  persistence {al + be:.3f}")
# => omega 0.0252  alpha 0.126  beta 0.856  persistence 0.982

nxt = res.forecast(horizon=1).variance.iloc[-1, 0]
print(f"next-day ann vol {np.sqrt(252 * nxt) / 100:.1%} vs unconditional {np.sqrt(252) * r.std():.1%}")
# => next-day ann vol 11.7% vs unconditional 19.5%

gjr = arch_model(100 * r, p=1, o=1, q=1, rescale=False).fit(disp="off")
print(f"GJR alpha {gjr.params['alpha[1]']:.3f}  gamma {gjr.params['gamma[1]']:.3f} "
      f"(p = {gjr.pvalues['gamma[1]']:.1e})")
# => GJR alpha 0.000  gamma 0.186 (p = 1.3e-10)
```

(Returns are scaled by 100 with `rescale=False` so the parameters live in interpretable percent² units.) Persistence of $\alpha + \beta = 0.982$ means volatility shocks decay with a half-life of about five weeks — the volatility clustering from [Returns and Their Distributions](02-returns-and-distributions.md), now a fitted parameter. The forecast line shows the model earning its keep: at the end of the sample it prices the next day at 11.7% annualized against the 19.5% full-sample constant a naive risk model would use — a 40% difference in every position size and risk limit downstream. The GJR fit adds the sharpest single result in this lesson: given a term that responds only to *negative* surprises, the symmetric ARCH coefficient collapses to zero. For equity indexes, volatility does not respond to surprise — it responds to *bad* surprise, entirely. The family tree in one table:

| Model | Extra term | Captures | Reach for it when |
|---|---|---|---|
| GARCH | — | clustering, persistence | default volatility model |
| GJR-GARCH | indicator on negative shocks | leverage effect | equities — here, it is the whole effect |
| EGARCH | log-variance, signed shocks | asymmetry without positivity constraints | when GJR's constraints bind |

## Cointegration by Engle-Granger

Two integrated series are cointegrated when some linear combination of them is stationary — prices wander, but the *spread* is tethered:

$$
y_t - \beta x_t = \varepsilon_t \sim I(0).
$$

The Engle-Granger procedure is the two-step everyone learns: regress $y$ on $x$ to estimate the hedge ratio, then unit-root-test the residual (with critical values adjusted for the estimated $\beta$, which `coint` handles — the adjustment is needed because a residual is not an error, as [Residual Analysis](../appendix/part-13-regression/07-residual-analysis.md) develops). The cache contains the most cointegrated pair in existence — SPY and IVV track the *same index* — so the test should be a formality:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

px = pd.read_parquet("data/prices.parquet")
pair = np.log(px[["SPY", "IVV"]].dropna())

beta = sm.OLS(pair["IVV"], sm.add_constant(pair["SPY"])).fit().params["SPY"]
stat, pval, _ = coint(pair["IVV"], pair["SPY"])
print(f"SPY-IVV: hedge ratio {beta:.3f}, EG stat {stat:.2f}, p = {pval:.2f}")
# => SPY-IVV: hedge ratio 1.002, EG stat -2.14, p = 0.46

gld = np.log(px[["SPY", "GLD"]].dropna())
print(f"SPY-GLD: EG p = {coint(gld['GLD'], gld['SPY'])[1]:.2f}")  # => SPY-GLD: EG p = 0.49

spread = pair["IVV"] - pair["SPY"]
rho = spread.autocorr(1)
print(f"spread std {1e4 * spread.std():.0f} bp, AR(1) rho {rho:.2f}, "
      f"half-life {np.log(0.5) / np.log(rho):.1f} days")
# => spread std 23 bp, AR(1) rho 0.82, half-life 3.4 days
```

SPY–GLD failing (p = 0.49) is the expected negative — no economic force ties gold to the S&P. But SPY–IVV *also fails*, at p = 0.46, statistically indistinguishable from the gold pair. Meanwhile the last line describes a spread that stays within a few dozen basis points for two decades and mean-reverts with a **3.4-day half-life** — as tradeable a spread as exists anywhere. That half-life is trustworthy at this sample size and not at the window a live system would re-estimate it on: the matched autocorrelation is biased down by a known function of $n$, and at sixty observations the same arithmetic returns 2.5 days instead of 3.5 ([Method of Moments](../appendix/part-11-parameter-estimation/04-method-of-moments.md)). The test and the trade are looking at the same data and reaching opposite conclusions. That is not a bug in statsmodels; it is a real property of this spread, and resolving it is the next section's job.

## Johansen, and the horizon you actually asked about

The Johansen framework tests cointegration for a whole system at once, estimating the *rank* — the number of independent stationary combinations — via trace statistics:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

px = pd.read_parquet("data/prices.parquet")

for cols in [["SPY", "IVV", "TLT"], ["SPY", "TLT", "GLD"]]:
    lp = np.log(px[cols].dropna())
    j = coint_johansen(lp, det_order=0, k_ar_diff=1)
    print(cols, f"trace {np.round(j.lr1, 1)} vs 95% crit {np.round(j.cvt[:, 1], 1)}")
# => ['SPY', 'IVV', 'TLT'] trace [348.5   5.5   1. ] vs 95% crit [29.8 15.5  3.8]
#    ['SPY', 'TLT', 'GLD'] trace [10.7  2.6  0. ] vs 95% crit [29.8 15.5  3.8]

pair = np.log(px[["SPY", "IVV"]].dropna())
for k in [1, 5, 20]:
    j = coint_johansen(pair, det_order=0, k_ar_diff=k)
    print(f"SPY-IVV, k_ar_diff={k:2d}: trace {j.lr1[0]:6.1f} (95% crit {j.cvt[0, 1]:.1f})")
# => SPY-IVV, k_ar_diff= 1: trace  345.3 (95% crit 15.5)
#    SPY-IVV, k_ar_diff= 5: trace   49.6 (95% crit 15.5)
#    SPY-IVV, k_ar_diff=20: trace   10.5 (95% crit 15.5)
```

The first sweep reads cleanly: the SPY/IVV/TLT system has rank one — exactly one stationary combination, and its eigenvector is (1, −1, 0), the SPY–IVV spread with no TLT in it — while SPY/TLT/GLD has rank zero, three assets wandering with no tether among them, an honest negative worth printing in full. But the second sweep is the real lesson of these two sections. The *same pair* produces a trace statistic of 345 with one lag of short-run dynamics absorbed, 50 with five, and 10.5 — below the critical value, no cointegration — with twenty. Engle-Granger's ADF step, whose automatic lag search settled on 31 lags, sits at the far end of the same dial; run that ADF on the spread with one lag instead and it rejects at p ≈ 10⁻²⁹.

Nobody is lying. The SPY–IVV spread is two superimposed processes: a violently mean-reverting component with a days-scale half-life — the ETF creation-redemption arbitrage at work — and a slow accrual from fee and dividend-timing differences that, inside any finite window, is statistically indistinguishable from a random walk. Short-lag tests see the first process; long-lag tests see the second. "Are these series cointegrated?" turns out not to be a well-posed question until you specify the horizon — what you can ask is whether the spread mean-reverts *at the horizon you intend to trade*, and the statistic that answers it is the half-life computation from the previous section, not any single p-value. This machinery — error correction, spread construction, the trade built on it — is exactly where [Mean Reversion and Pairs Trading](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) picks up in Part IV; the state-dependence of when the tether holds gets its own treatment in [Regime Detection](../appendix/part-18-quant-finance-applications/16-regime-detection.md).

!!! warning "If you scan a thousand pairs, cointegration will find you"
    This lesson ran cointegration tests on one pair and got p-values from 10⁻²⁹ to 0.46 by moving a lag setting. Now imagine scanning a thousand pairs, each with that much specification freedom, keeping whatever rejects at 5%. Cointegration is a property you predict from economics — same index, same issuer, same cash flows — and then confirm at your trading horizon; discovered by search, it is industrialized coincidence with a t-statistic. Counting how many tests you actually ran is the next lesson's entire subject.

!!! abstract "Key takeaways"
    - A statistic is only an estimate if the series is stationary: SPY's log-price "mean" wandered from 4.4 to 6.0 while returns kept a stable home near zero.
    - Run ADF and KPSS as a pincer — their nulls point in opposite directions, and only agreement is a verdict; disagreement is a diagnosis.
    - Returns clear the ACF significance band only at lag one and unstably; squared returns are massively autocorrelated at every lag — the memory is in the magnitude.
    - ARMA on daily returns yields significant, economically feeble coefficients, and its squared residuals fail Ljung-Box at machine zero: the conditional mean is a footnote, the variance is the story.
    - GARCH turns clustering into parameters — persistence 0.982, end-of-sample forecast 11.7% vs the 19.5% unconditional — and the GJR fit shows equity vol responds to bad news only.
    - The most cointegrated pair in existence fails Engle-Granger while mean-reverting with a 3.4-day half-life: tests see the horizon their lag structure selects, not "the" answer.
    - Johansen assigns SPY/IVV/TLT rank one and SPY/TLT/GLD rank zero — and its verdict on the same pair swings from 345 to 10.5 against a critical value of 15.5 as lags absorb the fast dynamics.

## Where this goes next

Every model in this lesson printed a p-value or a t-statistic, and each leaned on assumptions — independence, stationarity, one test run rather than many — that this lesson itself showed to be somewhere between fragile and false. Before any of those numbers is allowed near a trading decision, the tests themselves need auditing: what a t-statistic on autocorrelated strategy returns actually means, and what happens to significance when you tried fifty variants before finding it. That reckoning is [Hypothesis Testing and Multiple Testing](04-hypothesis-testing-and-multiple-testing.md).
