# Risk Parity, Diversification, and Factors

[Kelly, Volatility Targeting, and Leverage](02-kelly-vol-targeting-leverage.md) treated the book as given and asked how much of it to hold. It never asked whether the book was the right book — and [lesson one](01-risk-measurement.md) left a loud reason to doubt it, having measured `tsmom` and `tsmom_meta` at a correlation of 0.756 and a joint risk share of 50.7% in a five-sleeve portfolio. Two sleeves that are the same strategy, one of them filtered, are not two positions. They are one position, held twice, with a diversification ratio that flatters itself.

This lesson makes correlation the subject rather than a parameter. It measures how many independent bets a book actually contains, builds the allocation that equalizes risk instead of dollars and finds it wins on one universe and loses on another, watches the most respected diversification strategy in the industry meet the year its central assumption stopped holding, and then applies the only test that answers whether a sleeve deserves capital: regress it on everything else and see whether anything is left.

## The diversification math: correlation, not count

The arithmetic is short enough to state exactly. An equal-weight book of $N$ assets, each with volatility $\sigma$ and pairwise correlation $\rho$, has variance $\sigma^{2}\bigl(1 + (N-1)\rho\bigr)/N$, so the number of *effective* independent bets — the $N$ that would produce this variance at zero correlation — is

$$
N_{\text{eff}} \;=\; \frac{N}{1 + (N-1)\rho} \;\xrightarrow[N \to \infty]{}\; \frac{1}{\rho} .
$$

That limit is the whole lesson in one expression. Diversification is bounded by correlation and not by count, and the bound is reached fast:

```python
import numpy as np

print("  effective independent bets in an equal-weight book of N assets")
print("     N    rho=0.0  rho=0.1  rho=0.2  rho=0.3  rho=0.5")
for N in [2, 5, 10, 20, 50]:
    print(f"   {N:3d}    " + "  ".join(
        f"{N / (1 + (N - 1) * rho):7.2f}" for rho in [0.0, 0.1, 0.2, 0.3, 0.5]))
# =>   effective independent bets in an equal-weight book of N assets
#         N    rho=0.0  rho=0.1  rho=0.2  rho=0.3  rho=0.5
#         2       2.00     1.82     1.67     1.54     1.33
#         5       5.00     3.57     2.78     2.27     1.67
#        10      10.00     5.26     3.57     2.70     1.82
#        20      20.00     6.90     4.17     2.99     1.90
#        50      50.00     8.47     4.63     3.18     1.96
```

At a pairwise correlation of 0.2 — modest, and lower than most equity portfolios achieve — ten assets are 3.57 bets and fifty assets are **4.63**. Quadrupling the position count bought one additional bet. At $\rho = 0.5$, fifty assets are 1.96 bets: the book is a coin flip with extra commission. The practical corollary is that adding the fiftieth correlated position is nearly free of benefit and not free of cost, which is why the interesting question for a desk is never "how many strategies do we run" but "how correlated is the newest one to the ones we have". A single strategy at $\rho = 0$ to the existing book is worth more than twenty at $\rho = 0.3$.

## How many bets is a book of six strategies?

The formula above assumes one common correlation and equal weights. Real books have neither, so the count has to be estimated from the covariance matrix — and there are two defensible ways to do it that answer different questions. The **effective rank** takes the entropy of the correlation matrix's eigenvalue spectrum and asks how many independent directions the *universe* contains. Meucci's **effective number of bets** projects a specific weight vector onto the principal directions and asks how many of them a *particular portfolio* actually spends its risk on. A universe can be rich in independent directions while a portfolio concentrates in one of them:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
S6 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom", "s_pairs"]

def enb(rets, w):
    C = rets.corr().values
    ev = np.linalg.eigvalsh(C)[::-1]
    q = ev / ev.sum()
    erank = np.exp(-(q * np.log(q)).sum())            # entropy of the correlation spectrum
    cov = rets.cov().values
    lam, E = np.linalg.eigh(cov)
    v = E.T @ w                                        # weights in principal directions
    p = v ** 2 * lam / (w @ cov @ w)                   # Meucci: risk share per direction
    p = p[p > 1e-14]
    return erank, np.exp(-(p * np.log(p)).sum()), C[np.triu_indices_from(C, 1)].mean(), q[0]

sets = {"6 sleeves": S6, "5 sleeves (no pairs)": S6[:5],
        "2 surviving": ["s_tsmom", "s_shortvol"],
        "SPY/TLT/GLD": ["r_SPY", "r_TLT", "r_GLD"],
        "9 sectors": [f"r_{s}" for s in SECT],
        "9 sectors+TLT+GLD": [f"r_{s}" for s in SECT] + ["r_TLT", "r_GLD"]}
print("  universe                N   eff rank   ENB(eq wt)  ENB(eq vol)   mean rho    PC1")
for k, cols in sets.items():
    r = p8[cols].dropna()
    n = len(cols)
    er, e_eq, mr, pc1 = enb(r, np.ones(n) / n)
    iv = (1 / r.std()).values
    _, e_iv, _, _ = enb(r, iv / iv.sum())
    print(f"  {k:22s} {n:2d}   {er:8.2f}   {e_eq:10.2f}   {e_iv:10.2f}   {mr:+8.3f}  {pc1:5.1%}")
# =>   universe                N   eff rank   ENB(eq wt)  ENB(eq vol)   mean rho    PC1
#      6 sleeves               6       5.22         2.57         3.94     +0.102  31.9%
#      5 sleeves (no pairs)    5       4.24         2.54         3.00     +0.164  38.0%
#      2 surviving             2       2.00         1.37         1.41     +0.057  52.8%
#      SPY/TLT/GLD             3       2.87         2.23         2.06     -0.031  44.3%
#      9 sectors               9       3.62         1.03         1.10     +0.619  66.6%
#      9 sectors+TLT+GLD      11       4.53         1.19         1.35     +0.410  59.2%
```

Report both columns or mislead. By effective rank the six-sleeve book is **5.22 of a possible 6** — the strategies really are close to independent, mean pairwise correlation +0.102, and Part IV's insistence on hunting uncorrelated return streams is vindicated. By Meucci's portfolio-specific measure the *equal-weight* version of that same book is **2.57 bets**, because equal weighting piles risk into the trend direction that `tsmom` and `tsmom_meta` share. The universe offers five bets; the naive portfolio takes two and a half of them. Weighting by inverse volatility recovers most of the gap (3.94), which is the first quantitative argument for the risk-based allocation the next section builds.

The comparison that puts it in perspective is the bottom row but one. **Nine sector ETFs are 1.03 bets.** Not three, not two — one, with a mean pairwise correlation of +0.619 and a first principal component absorbing 66.6% of the variance. An equity sector rotation across the whole S&P sector complex is, in risk terms, a single position in the market with some tracking error attached, which is the structural reason `xsmom` was never going to survive Part IV's cost accounting: it was paying nine instruments' worth of spread to hold roughly one bet. Adding TLT and GLD to those nine lifts the count only to 1.19 at equal weight, because two low-correlation assets cannot offset nine that move together unless they are *sized* to matter — which is precisely what equal weighting refuses to do.

## Equal risk contribution, and where it earns its keep

Risk parity replaces the dollar allocation with a risk allocation: choose weights so every asset contributes the same share of portfolio volatility. There is a subtlety in the construction that bites in practice. The obvious formulation — minimize the squared dispersion of risk contributions — is **not convex**, and SLSQP will happily report success from a local minimum that pins assets against their bounds. The convex reformulation minimizes $\tfrac12 w^{\top}\Sigma w - \tfrac{1}{n}\sum_i \log w_i$, whose unique solution is the equal-risk portfolio up to scale, and it converges from any starting point:

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]

def erc(cov):
    """Convex log-barrier form: min 0.5 w'Sw - (1/n) sum log(w). The direct
    'equalize the risk contributions' least-squares objective is NOT convex, and
    SLSQP lands in local minima that pin assets at the bound."""
    n = len(cov)
    r = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, np.ones(n) / n,
                 method="SLSQP", jac=lambda w: cov @ w - 1 / (n * w),
                 bounds=[(1e-8, None)] * n, options={"maxiter": 800, "ftol": 1e-16})
    return r.x / r.x.sum(), r.success, r.nit

for lab, cols in [("SPY/TLT/GLD", ["r_SPY", "r_TLT", "r_GLD"]),
                  ("9 sectors + TLT + GLD", [f"r_{s}" for s in SECT] + ["r_TLT", "r_GLD"])]:
    R = p8[cols].dropna()
    cov = R.cov().values * 252
    w, ok, nit = erc(cov)
    rc = w * (cov @ w) / np.sqrt(w @ cov @ w)
    print(f"-- {lab}: SLSQP converged {ok} in {nit} iterations, "
          f"risk shares span {rc.min() / rc.sum():.1%} to {rc.max() / rc.sum():.1%}")
    print("   ERC weights: " + "  ".join(f"{c[2:]} {x:.1%}" for c, x in zip(cols, w)))
    print(f"   unlevered vol {np.sqrt(w @ cov @ w):.2%}  ->  leverage for a 10% target "
          f"{0.10 / np.sqrt(w @ cov @ w):.2f}x")
    iv = (1 / R.std()).values
    for nm, ww in [("ERC", w), ("equal-weight", np.ones(len(cols)) / len(cols)),
                   ("inverse-vol", iv / iv.sum())]:
        b = R @ ww
        e = np.exp(b.cumsum())
        print(f"     {nm:13s} Sharpe {np.sqrt(252) * b.mean() / b.std():+.3f}   "
              f"vol {np.sqrt(252) * b.std():6.2%}   maxDD {(e / e.cummax() - 1).min():+.1%}")
# => -- SPY/TLT/GLD: SLSQP converged True in 22 iterations, risk shares span 33.3% to 33.3%
#       ERC weights: SPY 32.4%  TLT 40.5%  GLD 27.0%
#       unlevered vol 9.31%  ->  leverage for a 10% target 1.07x
#         ERC           Sharpe +0.755   vol  9.31%   maxDD -25.1%
#         equal-weight  Sharpe +0.773   vol  9.67%   maxDD -23.5%
#         inverse-vol   Sharpe +0.753   vol  9.50%   maxDD -24.3%
#    -- 9 sectors + TLT + GLD: SLSQP converged True in 36 iterations, risk shares span 9.1% to 9.1%
#       ERC weights: XLK 5.6%  XLF 4.7%  XLE 4.7%  XLV 7.8%  XLY 5.6%  XLP 9.2%  XLI 5.7%  XLB 5.1%  XLU 7.0%  TLT 31.2%  GLD 13.5%
#       unlevered vol 10.05%  ->  leverage for a 10% target 0.99x
#         ERC           Sharpe +0.721   vol 10.05%   maxDD -28.2%
#         equal-weight  Sharpe +0.561   vol 15.06%   maxDD -45.6%
#         inverse-vol   Sharpe +0.621   vol 13.48%   maxDD -40.9%
```

The risk-share span confirms the solution — 33.3% to 33.3%, and 9.1% to 9.1%, equal to the digit — and the two universes disagree about whether it was worth computing. On **SPY/TLT/GLD**, risk parity *loses*: Sharpe 0.755 against equal weighting's 0.773, with a slightly deeper drawdown. Three assets that already have a mean pairwise correlation of −0.031 and 2.87 effective directions are close enough to balanced that the optimization has nothing to fix, and it spends its effort tilting toward TLT (40.5%), the lowest-volatility and lowest-returning leg. On **nine sectors plus TLT and GLD**, risk parity wins decisively: Sharpe **0.721 against 0.561**, and a maximum drawdown of −28.2% against −45.6%. Here the naive book was badly unbalanced — nine correlated equity sleeves drowning two diversifiers — and equalizing risk is exactly the repair, handing TLT 31.2% of the capital to buy it a ninth of the risk.

The rule that generalizes: **risk parity pays in proportion to how unbalanced the naive alternative is.** On a universe already close to equal-risk it is a rounding error with a solver attached; on a universe where one asset class dominates the count it is worth 0.16 of Sharpe. And one number deserves flagging before the next section, because the industry's version of this strategy does not look like the row above. This ERC book runs at 10.05% volatility unlevered, so hitting a 10% target requires **0.99×** — no leverage at all. Real risk-parity funds target equity-like returns from a bond-heavy book and get there with two to three turns of leverage, which multiplies everything that follows.

## 2022, and the correlation that stopped being negative

Risk parity's largest weight is its lowest-volatility asset, which in every implementation since the strategy was invented has meant long-duration bonds. That is defensible only while bonds diversify equities. The assumption is testable and it has a date on which it failed:

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
cols = [f"r_{s}" for s in SECT] + ["r_TLT", "r_GLD"]

def erc(cov):                                    # as in the previous section
    n = len(cov)
    r = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, np.ones(n) / n,
                 method="SLSQP", jac=lambda w: cov @ w - 1 / (n * w),
                 bounds=[(1e-8, None)] * n, options={"maxiter": 800, "ftol": 1e-16})
    return r.x / r.x.sum()

R = p8[cols].dropna()
tr = R[R.index < "2022-01-01"]                   # fitted before the year it is tested on
w = erc(tr.cov().values * 252)
lev = 0.10 / np.sqrt(w @ (tr.cov().values * 252) @ w)
y22 = p8.loc["2022"]
print(f"ERC fitted through {tr.index[-1].date()} (TLT is its largest weight at "
      f"{w[cols.index('r_TLT')]:.1%}), levered {lev:.2f}x to a 10% target")
for k, b in [("60/40", 0.6 * y22.r_SPY + 0.4 * y22.r_TLT),
             ("risk parity (ERC @ 10%)", lev * (y22[cols] @ w)),
             ("naive equal-weight", y22[cols].mean(axis=1))]:
    e = np.exp(b.cumsum())
    print(f"  {k:24s} 2022 {np.exp(b.sum()) - 1:+7.2%}   intra-year maxDD "
          f"{(e / e.cummax() - 1).min():+6.1%}   realized vol {np.sqrt(252) * b.std():5.1%}")
print(f"  {'SPY':24s} 2022 {np.exp(y22.r_SPY.sum()) - 1:+7.2%}")
print(f"  {'TLT (the diversifier)':24s} 2022 {np.exp(y22.r_TLT.sum()) - 1:+7.2%}")

d2 = p8.dropna(subset=["r_SPY", "r_TLT"])
yr = d2.index.year
cy = pd.Series({y: d2.r_SPY[yr == y].corr(d2.r_TLT[yr == y]) for y in sorted(set(yr))})
print("\n  corr(SPY, TLT), the assumption risk parity is built on:")
print("    " + "  ".join(f"{y}:{v:+.2f}" for y, v in cy.items() if 2008 <= y <= 2013))
print("    " + "  ".join(f"{y}:{v:+.2f}" for y, v in cy.items() if y >= 2020))
for a, b in [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]:
    m = cy[(cy.index >= a) & (cy.index <= b)].mean()
    print(f"    {a}-{b} mean {m:+.3f}")
roll = p8.r_SPY.rolling(252).corr(p8.r_TLT).dropna()
print(f"    rolling 252d: min {roll.min():+.3f} ({roll.idxmin():%Y-%m}), "
      f"max {roll.max():+.3f} ({roll.idxmax():%Y-%m})")
# => ERC fitted through 2021-12-31 (TLT is its largest weight at 35.4%), levered 1.08x to a 10% target
#      60/40                    2022 -23.67%   intra-year maxDD -27.0%   realized vol 17.3%
#      risk parity (ERC @ 10%)  2022 -16.91%   intra-year maxDD -22.7%   realized vol 15.5%
#      naive equal-weight       2022  -8.49%   intra-year maxDD -17.5%   realized vol 18.0%
#      SPY                      2022 -18.18%
#      TLT (the diversifier)    2022 -31.23%
#
#      corr(SPY, TLT), the assumption risk parity is built on:
#        2008:-0.49  2009:-0.32  2010:-0.55  2011:-0.71  2012:-0.65  2013:-0.21
#        2020:-0.47  2021:-0.14  2022:+0.08  2023:+0.13  2024:+0.06  2025:+0.14
#        2005-2009 mean -0.229
#        2010-2014 mean -0.517
#        2015-2019 mean -0.363
#        2020-2025 mean -0.034
#        rolling 252d: min -0.765 (2012-08), max +0.316 (2024-07)
```

A book targeting 10% volatility lost **16.91%** in a single calendar year — a −1.7σ outcome against its own promise — and realized 15.5% volatility while doing it, missing the target by half again. The diversifier is the culprit: TLT fell **31.23%**, worse than the equity leg it was there to offset, and risk parity had 35.4% of its capital in it. The comparison that stings is the last book in the table. Naive equal weighting, the allocation risk parity exists to improve on, lost **8.49%** — less than half as much — because it happened to hold only 9% in bonds and a full weight in the energy sector that 2022 rewarded. The sophisticated allocation lost twice as much as the unsophisticated one, and it lost for exactly the reason it was constructed.

The correlation table shows this was not a one-year accident. Through the decade risk parity was built and marketed, SPY–TLT correlation averaged **−0.517**; since 2020 it has averaged **−0.034**, and it has printed positive in every one of the last four years. The rolling 252-day series spans −0.765 to +0.316, a range of more than one full correlation unit, which means any allocation whose weights are a function of this number is a bet on the regime that produced it. That is the deepest critique of risk parity, and it is not "it uses leverage" or "it is crowded": it is that **an allocation derived from a covariance matrix inherits the stationarity assumption of that matrix**, and correlations are the least stationary thing in finance. Risk parity did not fail in 2022 because it was badly implemented. It failed because it was correctly implemented on an estimate that expired.

## Factor attribution: which sleeves are strategies

A strategy earns the name only if what it produces cannot be bought more cheaply as a passive exposure. The test is a regression of the sleeve on a panel of factors, where the intercept is the part no combination of factors explains. Six proxies are available from the frozen panel: the market, size (small minus large), international (developed minus US), duration, gold, and a defensive tilt. Standard errors are HAC-corrected at 21 lags because daily strategy returns are autocorrelated and naive standard errors would overstate every t-statistic:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

p8 = pd.read_parquet("data/part8.parquet")
S6 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom", "s_pairs"]
f = pd.DataFrame({"MKT": p8.r_SPY, "SIZE": p8.r_IWM - p8.r_SPY,
                  "INTL": p8.r_EFA - p8.r_SPY, "BOND": p8.r_TLT, "GOLD": p8.r_GLD,
                  "DEF": p8.r_XLP - p8.r_SPY}).dropna()

print("  sleeve         alpha/yr   t(a)     R2   resid vol  appraisal   strongest betas (t)")
for k in S6:
    d = pd.concat([p8[k].rename("y"), f], axis=1).dropna()
    m = sm.OLS(d.y, sm.add_constant(d[f.columns])).fit(cov_type="HAC",
                                                       cov_kwds={"maxlags": 21})
    rv = np.sqrt(252) * m.resid.std()
    top = sorted(f.columns, key=lambda c: -abs(m.tvalues[c]))[:2]
    print(f"  {k:13s} {m.params['const'] * 252:+8.2%} {m.tvalues['const']:+7.2f} "
          f"{m.rsquared:6.1%} {rv:10.2%} {m.params['const'] * 252 / rv:10.2f}   "
          + ", ".join(f"{c} {m.params[c]:+.3f} ({m.tvalues[c]:+.1f})" for c in top))

carry = ((p8.vix.shift(1) / 100) ** 2 / 252 - p8.r_SPY ** 2).rename("CARRY")
d = pd.concat([p8.s_tom.rename("y"), f, carry], axis=1).dropna()
m = sm.OLS(d.y, sm.add_constant(d[list(f.columns) + ["CARRY"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 21})
print(f"  s_tom, adding a variance-carry factor: alpha {m.params['const'] * 252:+.2%}/yr "
      f"(t {m.tvalues['const']:+.2f}), R2 {m.rsquared:.1%}")
# =>   sleeve         alpha/yr   t(a)     R2   resid vol  appraisal   strongest betas (t)
#      s_tsmom         +1.72%   +0.81   9.9%      9.81%       0.18   GOLD +0.162 (+4.8), SIZE +0.048 (+1.7)
#      s_tsmom_meta    +1.42%   +0.93  16.7%      6.84%       0.21   GOLD +0.124 (+5.6), BOND +0.109 (+5.0)
#      s_xsmom         -1.17%   -0.39   3.9%     15.06%      -0.08   GOLD +0.105 (+4.1), SIZE -0.148 (-2.8)
#      s_shortvol     +13.05%   +6.80  18.0%      9.06%       1.44   MKT +0.199 (+4.9), BOND -0.031 (-2.1)
#      s_tom           +0.17%   +0.11  17.3%      7.21%       0.02   MKT +0.166 (+8.0), BOND -0.016 (-1.5)
#      s_pairs         -0.09%   -0.60   4.2%      0.98%      -0.10   INTL +0.008 (+1.6), MKT +0.008 (+1.5)
#      s_tom, adding a variance-carry factor: alpha -0.19%/yr (t -0.11), R2 17.5%
```

`tom` is the clean kill. Its alpha is **+0.17% a year with a t-statistic of +0.11** — indistinguishable from zero by any standard — while its market beta is +0.166 at t = +8.0, overwhelmingly significant. The appraisal ratio, alpha over residual volatility, is **0.02**. Everything the turn-of-the-month strategy produced was market exposure held for a quarter of the days; buying and holding 16.6% of SPY would have done the same thing without the trading. Adding a variance-carry factor pushes the alpha to −0.19%, so there is nothing hiding in the residual either. Part IV called `tom` "the cadaver that trails buy-and-hold" on the strength of a return comparison; this is the same verdict with a t-statistic and a mechanism attached, and it retires the sleeve on evidence rather than on suspicion.

`shortvol` is the opposite result and the only unambiguous one in the table: alpha **+13.05% a year at t = +6.80**, an appraisal ratio of 1.44, and only 18.0% of its variance explained by six factors. Whatever it is doing, no passive combination of these exposures reproduces it. The two trend sleeves sit in the honest middle — alphas of +1.72% and +1.42% that are economically interesting and *statistically insignificant* (t = +0.81 and +0.93), which is the same conclusion Part IV reached from a different direction and the reason this course has never claimed trend following as a certainty. Their largest loading is worth noting on its own: **GOLD at +0.162 and +0.124**, both strongly significant. Over 2006–2025 a three-asset trend book is, to a substantial degree, a long-gold position — an exposure nobody chose and the strategy label conceals.

## The admission gate: spanning

Factor regressions ask whether a sleeve beats passive alternatives. The allocation question is narrower and more useful: does this sleeve beat *the sleeves already in the book*? Regressing each on all the others answers it directly — the intercept is the return that survives being replicated by the existing portfolio, and it is the only alpha that justifies adding a position rather than resizing one:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

p8 = pd.read_parquet("data/part8.parquet")
S5 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]
W = p8[S5].dropna()
print(f"  each sleeve regressed on the other four, {W.index[0].date()} to "
      f"{W.index[-1].date()}, n {len(W):,}")
print("  sleeve         alpha/yr   t(a)     R2   strongest loading (by t)")
for k in S5:
    others = [c for c in S5 if c != k]
    m = sm.OLS(W[k], sm.add_constant(W[others])).fit(cov_type="HAC",
                                                     cov_kwds={"maxlags": 21})
    big = max(others, key=lambda c: abs(m.tvalues[c]))
    print(f"  {k:13s} {m.params['const'] * 252:+8.2%} {m.tvalues['const']:+7.2f} "
          f"{m.rsquared:6.1%}   {big} {m.params[big]:+.2f} (t {m.tvalues[big]:+.1f})")
# =>   each sleeve regressed on the other four, 2006-08-01 to 2025-06-30, n 4,758
#      sleeve         alpha/yr   t(a)     R2   strongest loading (by t)
#      s_tsmom         -1.02%   -0.69  59.6%   s_tsmom_meta +1.01 (t +33.1)
#      s_tsmom_meta    +1.77%   +1.62  58.1%   s_tsmom +0.55 (t +17.8)
#      s_xsmom         -1.43%   -0.42   8.5%   s_tsmom +0.48 (t +4.6)
#      s_shortvol     +13.68%   +6.87   6.7%   s_tom +0.28 (t +4.4)
#      s_tom           -1.34%   -0.76   6.9%   s_shortvol +0.19 (t +6.0)
```

The first row settles the question [lesson one](01-risk-measurement.md) raised. `tsmom` has an alpha of **−1.02% a year (t = −0.69)** against the rest of the book, an $R^2$ of 59.6%, and a loading of **+1.01 on `tsmom_meta` at t = +33.1**. A unit of the raw trend book is, statistically, one unit of the meta-labeled trend book plus noise and a negative constant. It is *fully spanned* — holding it alongside its own filtered version adds no return and, as lesson one measured, consumes a quarter of the risk budget. The correct action is not to resize `tsmom` but to retire it, and the asymmetry in the second row confirms which one survives: `tsmom_meta` regressed on the others keeps a **positive** alpha of +1.77% and loads only +0.55 on `tsmom`. The filter is doing work in one direction and not the other. This is the payoff of [Part VII's meta-labeling](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) stated in allocation terms, and it is a stronger claim than the Sharpe comparison that motivated it.

The rest of the table is a clean admission gate. `shortvol` keeps **+13.68% at t = +6.87** with an $R^2$ of 6.7% — spanned by nothing, correlated with nothing, the only sleeve in this course that unambiguously earns its capital. `xsmom` (−1.43%, t = −0.42) and `tom` (−1.34%, t = −0.76) both fail, which is now the third independent method to reach that verdict after Part IV's cost accounting and the factor regression above. The gate itself generalizes into a rule worth writing into a research process: **a candidate strategy is admitted on its spanning alpha, not its standalone Sharpe.** A sleeve with a Sharpe of 1.5 that loads 0.9 on something already held adds nothing but fees; a sleeve with a Sharpe of 0.4 orthogonal to the book can be levered into a genuine contribution. Standalone performance is the wrong statistic, and it is the one every pitch deck leads with.

!!! warning "Every diversification number is a forecast of a correlation matrix, and correlation matrices expire"
    The six-sleeve book contains 5.22 effective directions, the diversification ratio is 1.74, and risk parity beats equal weighting by 0.16 of Sharpe — all measured on realized history, all of them statements about the future only to the extent that the correlation structure persists. It did not persist for SPY and TLT: a −0.517 decade average became −0.034, positive in four consecutive years, and a 10%-target risk-parity book lost 16.91% in 2022 doing exactly what it was designed to do. Treat every allocation derived from an estimated covariance matrix as a position in that matrix's stability, size it accordingly, and check whether the correlation you are relying on has a mechanism behind it or merely a track record.

!!! abstract "Key takeaways"
    - Diversification is bounded by correlation, not count: at ρ = 0.2 ten assets are 3.57 effective bets and fifty are **4.63**; at ρ = 0.5 fifty assets are 1.96. The limit is 1/ρ and it is reached fast.
    - The six-sleeve book holds **5.22 effective directions** but the equal-weight version spends its risk on only **2.57** of them; inverse-volatility weighting recovers 3.94. **Nine sector ETFs are 1.03 bets** (mean ρ +0.619, PC1 66.6%) — a sector rotation is one position wearing nine tickers.
    - The naive equal-risk objective is non-convex and SLSQP reports success from local minima that pin assets at the bound; the log-barrier form $\tfrac12 w'\Sigma w - \tfrac1n\sum\log w_i$ converges to risk shares equal to the digit.
    - Risk parity pays in proportion to how unbalanced the alternative is: it **loses** to equal weighting on SPY/TLT/GLD (0.755 vs 0.773) and **wins by 0.16 of Sharpe** on nine sectors plus TLT and GLD (0.721 vs 0.561, maxDD −28.2% vs −45.6%).
    - In 2022 a 10%-target risk-parity book lost **16.91%** — a −1.7σ year — while naive equal weighting lost 8.49%, because TLT fell 31.23% and held 35.4% of the capital. SPY–TLT correlation averaged −0.517 in 2010–14 and **−0.034** since 2020, positive in each of the last four years.
    - `tom` is a factor bet, not a strategy: alpha **+0.17%/yr at t = +0.11**, market beta +0.166 at t = +8.0, appraisal ratio 0.02, and adding a variance-carry factor drives the alpha negative. `shortvol` keeps alpha +13.05% at t = +6.80 with only 18% of its variance explained.
    - Spanning is the admission gate: `tsmom` has alpha **−1.02% (t = −0.69)** against the rest of the book and loads **+1.01 on `tsmom_meta` (t = +33.1)** — fully replicated by its own filtered version and due for retirement, while `tsmom_meta` keeps a positive alpha in the reverse regression.

## Where this goes next

Three allocations have now been compared — equal dollars, equal volatility, equal risk — and all three share a property that has not yet been questioned: none of them uses a return forecast. That is a deliberate abstention, and the obvious next move is to stop abstaining, feed expected returns into an optimizer alongside the covariance matrix, and let it find the efficient portfolio. [Portfolio Optimization and Correlation](04-portfolio-optimization-and-correlation.md) does exactly that, and the result is the most comprehensively negative in this part: mean-variance optimization loses to naive equal weighting in every configuration tested, the standard repair of shrinking the covariance matrix makes it *worse*, and the thing that actually rescues it turns out to be the crudest tool available. Along the way it measures what happens to the correlations this lesson relied on when markets are falling.
