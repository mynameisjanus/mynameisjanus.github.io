# Drawdowns, Tail Risk, and Stress Testing

[Portfolio Optimization and Correlation](04-portfolio-optimization-and-correlation.md) ended by measuring the thing that makes every other number in this part conditional: average pairwise sector correlation runs 0.314 in a calm year and 0.922 in March 2020, so a book the optimizer believed held nine positions held roughly one. That was the fourth time Part VIII has found its own summary statistics least reliable exactly where reliability matters — after a kurtosis of 29.2 on the surviving book, an expected shortfall 45% past its Gaussian value, and a Kelly fraction exceeding its own ruin bound.

This lesson goes to the left tail deliberately. It measures drawdowns and then asks the question that defuses most of what people conclude from them; it fits the tail rather than assuming it; it tests whether "correlations go to one" is true of daily tails as well as of crisis-period averages, and finds it is true of one asset class and false of another; it replays every crisis in the sample against the book; and it prices three ways of buying protection against what each actually delivers. The recurring finding is that the tail is both worse than the Gaussian says and less informative than intuition assumes — the distribution is fatter, and the single realized path is a much weaker piece of evidence than it feels like.

## Drawdowns, and whether the worst one meant anything

Drawdown is the statistic investors actually experience, and it has three dimensions that get collapsed into one number: depth, duration, and how much of the time is spent underwater at all. Measuring all three, and then simulating what a *random* strategy with the same Sharpe and length would have produced, separates the informative part from the inevitable:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
S5 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]

def book(cols, target=0.10):
    w = p8[cols].dropna()
    return (w / (np.sqrt(252) * w.std()) * target / len(cols)).sum(axis=1)

BK = {"surviving": book(["s_tsmom", "s_shortvol"]), "the book": book(S5),
      "tsmom": book(["s_tsmom"])}

print("  book       maxDD    on            episodes  longest  CDaR95   avg DD")
for k, b in BK.items():
    e = np.exp(b.cumsum())
    d = e / e.cummax() - 1
    runs, cur, eps = [], 0, 0
    for u in (d < -1e-12).values:
        if u:
            cur += 1
        else:
            if cur:
                runs.append(cur); eps += 1
            cur = 0
    if cur:
        runs.append(cur); eps += 1
    print(f"  {k:10s} {d.min():+.1%}  {d.idxmin():%Y-%m-%d} {eps:10d} {max(runs):8d}d "
          f"{d[d < 0].quantile(0.05):+7.1%} {d.mean():+8.2%}")

print("\n  is the worst drawdown unusual? 10k simulated paths at each book's own"
      " Sharpe, vol and length")
rng = np.random.default_rng(11)
for k, b in BK.items():
    n = len(b)
    real = (np.exp(b.cumsum()) / np.exp(b.cumsum()).cummax() - 1).min()
    sim = rng.normal(b.mean(), b.std(), (10000, n)).cumsum(axis=1)
    mdd = np.exp((sim - np.maximum.accumulate(sim, axis=1)).min(axis=1)) - 1
    blk = np.array([b.values[i:i + 21] for i in range(n - 21)])
    bs = blk[rng.integers(0, len(blk), (5000, n // 21))].reshape(5000, -1).cumsum(axis=1)
    bmdd = np.exp((bs - np.maximum.accumulate(bs, axis=1)).min(axis=1)) - 1
    print(f"  {k:10s} realized {real:+.1%}   iid-normal median {np.median(mdd):+.1%} "
          f"[5th {np.percentile(mdd, 5):+.1%}, 95th {np.percentile(mdd, 95):+.1%}] "
          f"-> {(mdd < real).mean():.0%}th pct   block-bootstrap median "
          f"{np.median(bmdd):+.1%} -> {(bmdd < real).mean():.0%}th pct")
# =>   book       maxDD    on            episodes  longest  CDaR95   avg DD
#      surviving  -12.0%  2008-11-04        298      598d   -9.4%   -2.33%
#      the book   -11.2%  2019-01-30        192     1123d   -8.6%   -2.81%
#      tsmom      -24.7%  2019-02-05         77     2623d  -20.3%  -10.45%
#
#      is the worst drawdown unusual? 10k simulated paths at each book's own Sharpe, vol and length
#      surviving  realized -12.0%   iid-normal median -12.2% [5th -18.6%, 95th -8.6%] -> 53%th pct   block-bootstrap median -12.3% -> 55%th pct
#      the book   realized -11.2%   iid-normal median -11.8% [5th -19.0%, 95th -8.0%] -> 58%th pct   block-bootstrap median -11.0% -> 47%th pct
#      tsmom      realized -24.7%   iid-normal median -30.6% [5th -49.6%, 95th -20.0%] -> 79%th pct   block-bootstrap median -28.0% -> 69%th pct
```

Take the second panel first, because it changes how the first should be read. Every realized maximum drawdown in this course sits **at the median of what pure chance produces** — the 53rd, 58th and 79th percentiles of ten thousand random walks matched to each book's own Sharpe, volatility and length. The surviving book's −12.0% has a simulated median of −12.2%. `tsmom`'s −24.7%, the worst number in the part, is *milder* than the −30.6% a random strategy with its Sharpe would typically deliver, and it sits comfortably inside a 5th-to-95th range that runs from −49.6% to −20.0%. A block bootstrap that preserves fat tails and volatility clustering agrees. Building a null distribution this way is the cheapest statistic a simulation produces, and the general treatment — including why the standard error on such a table is trustworthy only when the simulated statistic has a second moment — is [Monte Carlo Simulation](../appendix/part-09-monte-carlo-methods/03-monte-carlo-simulation.md).

The consequence is the most useful and least comfortable idea in this lesson. **A realized drawdown carries almost no information about whether a strategy is broken.** The sampling distribution is enormously wide — for `tsmom`, anywhere between −20% and −50% is an ordinary outcome — so observing a −24.7% drawdown is consistent with the strategy working exactly as designed, and would also be consistent with it having quietly stopped working. Practitioners routinely reason from a drawdown to a diagnosis ("the edge has decayed", "the regime changed"), and this table says that inference is unsupported at any conventional confidence level. The corollary matters for [lesson two's](02-kelly-vol-targeting-leverage.md) de-risking rules: a policy that cuts size at a drawdown threshold is reacting to a signal that is mostly noise, which is precisely why cutting to zero made the drawdowns *worse*.

Duration is the dimension that the headline number hides and the one that actually ends mandates. `tsmom` spent a single continuous stretch of **2,623 trading days — more than ten years — below a prior high**, and the five-sleeve book's longest underwater run is 1,123 days against a maximum depth of only 11.2%. No investor experiences "an 11% drawdown"; they experience four and a half years of not making money, and they redeem somewhere in the middle of it. That asymmetry between how drawdowns are measured and how they are lived is the mechanism behind the industry's oldest failure mode — books get cut at the bottom, by the people funding them rather than the people running them, because depth is reported quarterly and duration is felt continuously. The defense is the one [Part IV](../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) named for volatility breaches: agree the tolerance, the response and the re-entry rule in advance, when nobody is losing money.

## Fitting the tail instead of assuming it

[Lesson one](01-risk-measurement.md) established that the Gaussian understates this book's tail and measured by how much *inside the sample*. Extrapolating **beyond** the sample — to the 1-in-4-year and 1-in-40-year loss — needs a model of the tail itself. Extreme value theory supplies one: above a high enough threshold, exceedances of almost any distribution converge to a generalized Pareto, whose shape parameter $\xi$ says how heavy the tail is. Positive $\xi$ means power-law decay, and $\xi \geq 1/k$ means the $k$-th moment does not exist:

```python
import numpy as np
import pandas as pd
from scipy import stats

p8 = pd.read_parquet("data/part8.parquet")
S5 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]

def book(cols, target=0.10):
    w = p8[cols].dropna()
    return (w / (np.sqrt(252) * w.std()) * target / len(cols)).sum(axis=1)

BK = {"surviving": book(["s_tsmom", "s_shortvol"]), "the book": book(S5),
      "tsmom": book(["s_tsmom"])}

def gpd_var(b, q, p):
    u = -b.quantile(1 - q)
    exc = (-b[-b > u] - u).values
    xi, _, beta = stats.genpareto.fit(exc, floc=0)          # peaks over threshold
    return xi, u + beta / xi * (((len(b) / len(exc)) * (1 - p)) ** -xi - 1), len(exc)

print("  book        u=q97.5   n_exc    xi     EVT 99.9  Gauss 99.9  ratio   "
      "EVT 99.99  Gauss 99.99  ratio")
for k, b in BK.items():
    xi, v999, ne = gpd_var(b, 0.975, 0.999)
    _, v9999, _ = gpd_var(b, 0.975, 0.9999)
    g9 = -(b.mean() + b.std() * stats.norm.ppf(0.001))
    g99 = -(b.mean() + b.std() * stats.norm.ppf(0.0001))
    print(f"  {k:10s} {-b.quantile(0.025):7.3%} {ne:7d} {xi:+7.3f} {v999:10.2%} "
          f"{g9:11.2%} {v999 / g9:6.2f}x {v9999:10.2%} {g99:12.2%} {v9999 / g99:6.2f}x")

print("\n  threshold stability -- xi and the EVT 99.9% VaR at three thresholds")
for k, b in BK.items():
    out = [f"q{q:.3f}: xi {gpd_var(b, q, 0.999)[0]:+.3f}, VaR {gpd_var(b, q, 0.999)[1]:.2%}"
           for q in [0.95, 0.975, 0.99]]
    print(f"  {k:10s} " + "   ".join(out))
# =>   book        u=q97.5   n_exc    xi     EVT 99.9  Gauss 99.9  ratio   EVT 99.99  Gauss 99.99  ratio
#      surviving   0.945%     119  +0.327      3.67%       1.38%   2.65x      8.36%        1.67%   5.01x
#      the book    0.822%     119  +0.229      2.21%       1.10%   2.01x      4.06%        1.33%   3.06x
#      tsmom       1.331%     154  +0.266      3.75%       1.94%   1.94x      7.29%        2.33%   3.13x
#
#      threshold stability -- xi and the EVT 99.9% VaR at three thresholds
#      surviving  q0.950: xi +0.321, VaR 3.63%   q0.975: xi +0.327, VaR 3.67%   q0.990: xi +0.200, VaR 3.61%
#      the book   q0.950: xi +0.118, VaR 2.18%   q0.975: xi +0.229, VaR 2.21%   q0.990: xi +0.260, VaR 2.21%
#      tsmom      q0.950: xi +0.154, VaR 3.63%   q0.975: xi +0.266, VaR 3.75%   q0.990: xi -0.223, VaR 3.65%
```

The surviving book's shape parameter is **ξ = +0.327**, which is a specific and alarming statement: moments exist only up to order $1/0.327 = 3.1$, so the book's *kurtosis does not exist* in the limit. The sample kurtosis of 29.2 that [lesson one](01-risk-measurement.md) reported is not an estimate of a population quantity — it is a number that grows with sample size. Any risk method that requires a fourth moment, which includes most of the Cornish–Fisher machinery sold as a fat-tail correction, is undefined on this distribution.

The extrapolation is where it bites. At the 1-in-4-year loss the Gaussian says 1.38% and EVT says **3.67%** — an understatement of **2.65×**. At the 1-in-40-year loss the gap widens to **5.01×**: 8.36% against 1.67%. A desk sizing its capital buffer off a Gaussian tail has provisioned for a fifth of what the fitted tail implies, and the error grows with how rare the event is, which is the opposite of the direction anyone wants.

Two disciplines make this credible rather than decorative. First, EVT estimates are notoriously threshold-sensitive, so the second panel refits at three thresholds: the surviving book gives ξ of +0.321, +0.327 and +0.200 with 99.9% VaR estimates of 3.63%, 3.67% and 3.61% — stable to a hundredth, which is what a trustworthy fit looks like. Second, the counter-example is in the same panel and should not be hidden: `tsmom` at the 99th-percentile threshold produces **ξ = −0.223**, a *bounded* tail, flipping sign from the +0.266 estimated one threshold lower. That fit is running on 47 exceedances and is not to be believed. The rule is to fit at several thresholds and trust the result only where it is flat — and where it is not flat, say so rather than quoting the most convenient number.

## Do correlations go to one? Only within an asset class

"Correlations go to one in a crisis" is the most repeated claim in risk management, and [the previous lesson](04-portfolio-optimization-and-correlation.md) appeared to confirm it: nine sectors reached an average pairwise correlation of 0.922 in March 2020. But a *period-average* correlation conflates two things — the market falling and the tails arriving together. The cleaner measure is tail dependence: given one asset in its worst decile, how often is the other also in its worst decile? Independence gives 10%. That this is a *different* question from correlation, and not a refinement of it, is the point of [Marginal Distributions](../appendix/part-03-random-variables/06-marginal-distributions.md): two joint laws can match on every marginal and on their correlation to three decimals and still differ in the joint tail by a factor of five, so no amount of per-asset data settles the question below:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
print("  pair                  overall rho   worst-decile rho   P(both in worst 10%)   lambda")
for a, c in [("r_SPY", "r_TLT"), ("r_SPY", "r_GLD"), ("r_TLT", "r_GLD"),
             ("r_SPY", "r_EFA"), ("r_SPY", "r_EEM"),
             ("s_tsmom", "s_shortvol"), ("s_shortvol", "s_tom")]:
    d = p8[[a, c]].dropna()
    lo = d[d[a] <= d[a].quantile(0.10)]
    j = ((d[a] <= d[a].quantile(0.10)) & (d[c] <= d[c].quantile(0.10))).mean()
    print(f"  {a[2:]:9s}/{c[2:]:10s} {d[a].corr(d[c]):+11.3f} {lo[a].corr(lo[c]):+18.3f} "
          f"{j:21.2%} {j / 0.10:8.2f}")
print("  (independence gives lambda = 0.10; lambda = 1.00 would be perfect tail dependence)")
# =>   pair                  overall rho   worst-decile rho   P(both in worst 10%)   lambda
#      SPY      /TLT             -0.312             -0.264                 0.87%     0.09
#      SPY      /GLD             +0.055             +0.013                 1.72%     0.17
#      TLT      /GLD             +0.161             +0.080                 2.10%     0.21
#      SPY      /EFA             +0.872             +0.815                 6.62%     0.66
#      SPY      /EEM             +0.815             +0.753                 6.08%     0.61
#      tsmom    /shortvol        +0.057             +0.047                 2.73%     0.27
#      shortvol /tom             +0.240             +0.262                 2.80%     0.28
#      (independence gives lambda = 0.10; lambda = 1.00 would be perfect tail dependence)
```

The claim is true within an asset class and false across one, and the contrast is stark. **SPY and EFA have λ = 0.66** — when US equities are in their worst decile, developed international equities are there too two-thirds of the time, 6.6 times what independence would give, and their worst-decile correlation stays at +0.815. Emerging markets are the same story at 0.61. Holding US, developed and emerging equities is holding one asset in three wrappers, and the wrappers dissolve precisely on the days they were bought for.

**SPY and TLT go the other way: λ = 0.09, marginally *below* the 0.10 that independence implies**, with a worst-decile correlation of −0.264 that stays negative. On the days US equities fall hardest, long bonds are *less* likely than chance to be falling hardest too. This is the same asset pair that [lesson three](03-risk-parity-diversification-factors.md) showed has had a positive *annual* correlation for four consecutive years and cost risk parity 16.9% in 2022 — and both results are correct. Daily crash co-movement and annual co-movement are different quantities: a slow year-long repricing of the discount rate hurts stocks and bonds together, while a one-day equity panic still sends money into Treasuries. **A book can be exposed to the annual correlation and protected from the daily one, or the reverse, and only measuring both tells you which.** Quoting "correlations go to one" as a general law is how a desk ends up hedging the wrong horizon.

## Replaying the crises that happened

Fitted tails and simulated paths are models. The sample also contains the actual events, and running the book through each of them is the least model-dependent test available:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
S5 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]

def book(cols, target=0.10):
    w = p8[cols].dropna()
    return (w / (np.sqrt(252) * w.std()) * target / len(cols)).sum(axis=1)

surv, bk, tsm = book(["s_tsmom", "s_shortvol"]), book(S5), book(["s_tsmom"])
EV = {"2007-08 quant quake": ("2007-08-01", "2007-08-31"),
      "2008 GFC Sep-Nov": ("2008-09-01", "2008-11-30"),
      "2010 flash crash May": ("2010-05-01", "2010-05-31"),
      "2011 US downgrade": ("2011-07-25", "2011-10-10"),
      "2015-08 devaluation": ("2015-08-17", "2015-08-31"),
      "2018-02 volmageddon": ("2018-02-01", "2018-02-14"),
      "2018 Q4 selloff": ("2018-10-01", "2018-12-24"),
      "2020 COVID": ("2020-02-19", "2020-03-23"),
      "2022 rates bear": ("2022-01-01", "2022-12-31"),
      "2024-08 yen carry": ("2024-07-31", "2024-08-07"),
      "2025-04 tariffs": ("2025-04-02", "2025-04-08")}
print("  event                        SPY      TLT    60/40   surviving  the book    tsmom")
for k, (a, c) in EV.items():
    def tot(s):
        x = s.loc[a:c].dropna()
        return np.exp(x.sum()) - 1 if len(x) else float("nan")
    print(f"  {k:24s} {tot(p8.r_SPY):+8.1%} {tot(p8.r_TLT):+8.1%} "
          f"{tot(0.6 * p8.r_SPY + 0.4 * p8.r_TLT):+8.1%} {tot(surv):+10.1%} "
          f"{tot(bk):+9.1%} {tot(tsm):+8.1%}")
# =>   event                        SPY      TLT    60/40   surviving  the book    tsmom
#      2007-08 quant quake         +1.3%    +1.8%    +1.5%      +0.7%     +0.6%    +1.1%
#      2008 GFC Sep-Nov           -29.6%   +13.9%   -14.7%      -6.9%     -3.5%    +6.3%
#      2010 flash crash May        -7.9%    +5.1%    -2.9%      -4.4%     -3.3%    -2.6%
#      2011 US downgrade          -10.7%   +22.4%    +1.3%      +1.6%     -2.5%    +1.1%
#      2015-08 devaluation         -5.6%    -2.0%    -4.2%      -6.2%     -4.2%    -5.2%
#      2018-02 volmageddon         -4.4%    -3.9%    -4.2%      -4.5%     -3.7%    -1.8%
#      2018 Q4 selloff            -18.9%    +4.4%   -10.3%      -6.3%     -4.9%    -2.9%
#      2020 COVID                 -33.4%   +14.2%   -17.4%      -1.6%     +2.2%    -2.8%
#      2022 rates bear            -18.2%   -31.2%   -23.7%      -1.6%     +0.1%    -7.1%
#      2024-08 yen carry           -4.3%    +2.5%    -1.6%      -3.9%     -3.4%    -1.8%
#      2025-04 tariffs            -11.5%    -3.4%    -8.4%      -2.6%     -3.1%    -4.4%
```

The books look superb against the large events and that is the trap in the table. In the 2008 collapse the surviving book lost 6.9% against the market's 29.6% and `tsmom` *made* 6.3%; through the COVID crash the five-sleeve book **gained 2.2%** while equities fell a third; in the 2022 rates bear market it was flat against 60/40's −23.7%. A pitch deck would stop here.

The pattern in the remaining rows is the one to present to a risk committee. In the 2010 flash crash, the August 2015 devaluation, February 2018 and the 2024 yen-carry unwind, the surviving book lost **as much as or more than the market did on a risk-adjusted basis** — −6.2% against SPY's −5.6% in 2015, −4.5% against −4.4% in volmageddon, on a book running 7.3% volatility against the market's 19.3%. Scaled for risk, those are severe losses. The distinction between the two groups is duration: **the book survives slow crises and loses in fast volatility spikes**, and the mechanism is not mysterious — its edge is `shortvol`, which is a short position in exactly the thing that gapped in each of those episodes, and trend signals need weeks to reposition. A crisis that takes three months to develop lets the trend sleeve get short and pays the variance seller a widened premium afterwards; a crisis that takes three days does neither.

That reframes the stress-testing question. Replaying 2008 against this book is nearly uninformative because the book was built, tested and selected on a sample containing 2008. The events worth attention are the short sharp ones, and the honest summary for a committee is: *this book is well protected against a repeat of the crises that made it look good, and structurally exposed to a two-day volatility spike, which is the scenario least represented in its own history.*

## Shocking the correlations, not just the prices

Historical replay can only produce shocks the sample contains. Hypothetical stress fills the gap, and the mistake most stress tests make is shocking prices while leaving the *relationships* at their historical values — which assumes the diversifiers keep diversifying on the day everything else breaks:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
cols = ["r_SPY", "r_TLT", "r_GLD"] + [f"r_{s}" for s in SECT]
R = p8[cols].dropna()
sig, rho, SHOCK = R.std(), R.corr()["r_SPY"], -0.20

b_hist = rho * sig / sig["r_SPY"]
rho_str = rho.copy()
for c in cols:                                  # the stress: diversifiers stop diversifying
    if c in ("r_TLT", "r_GLD"):
        rho_str[c] = 0.5                        # flipped from negative/zero to positive
    elif c != "r_SPY":
        rho_str[c] = max(rho[c], 0.9)           # equities move as one
b_str = rho_str * sig / sig["r_SPY"]

print(f"  a {SHOCK:.0%} equity shock, betas re-derived under stressed correlations")
for c in ["r_TLT", "r_GLD", "r_XLE", "r_XLP"]:
    print(f"    beta({c[2:]}, SPY)  historical {b_hist[c]:+.2f}  ->  stressed {b_str[c]:+.2f}")
w12 = pd.Series(1 / len(cols), index=cols)
w6040 = pd.Series(0.0, index=cols)
w6040["r_SPY"], w6040["r_TLT"] = 0.6, 0.4
for lab, w in [("12-asset equal weight", w12), ("60/40", w6040)]:
    lh, ls = (w * b_hist * SHOCK).sum(), (w * b_str * SHOCK).sum()
    print(f"  {lab:22s} loss at historical betas {lh:+.1%}   under stress {ls:+.1%}   "
          f"({ls / lh - 1:+.0%} worse)")
# =>   a -20% equity shock, betas re-derived under stressed correlations
#        beta(TLT, SPY)  historical -0.24  ->  stressed +0.38
#        beta(GLD, SPY)  historical +0.05  ->  stressed +0.46
#        beta(XLE, SPY)  historical +1.15  ->  stressed +1.42
#        beta(XLP, SPY)  historical +0.56  ->  stressed +0.66
#      12-asset equal weight  loss at historical betas -15.5%   under stress -18.6%   (+20% worse)
#      60/40                  loss at historical betas -10.1%   under stress -15.1%   (+49% worse)
```

The correlation stress costs the diversified 12-asset book an extra 20% of its loss and costs **60/40 an extra 49%** — because 60/40 has 40% of its capital in the one position whose beta flips from −0.24 to +0.38, while the 12-asset book has only a twelfth. That ratio is the real output of the exercise: *the more a portfolio depends on a diversifying relationship, the more of its stress loss is correlation risk rather than price risk*, and a stress test that shocks only prices systematically flatters exactly the portfolios that most need testing.

Which converts stress-test results into policy. A number like "−18.6% under stress" is not actionable on its own; the decomposition is. Here it says the 12-asset book's loss is mostly directional and would be reduced by cutting gross, while 60/40's incremental loss is a bet on bond-equity correlation and would not be — de-grossing a 60/40 book proportionally leaves the correlation exposure untouched. **The stress-test review should end with which lever addresses which loss**, and a program that produces only headline numbers gives a committee no way to choose between reducing size, changing the mix, and buying protection. That last option has a price, and the price is measurable.

## What protection costs, and what it buys

Three ways to defend the left tail: buy index puts, overlay a trend rule that goes to cash, or simply hold less. All three cost return, so the comparison that matters is the exchange rate — points of compound return surrendered per point of maximum drawdown avoided. Options are priced with Black–Scholes at VIX-implied volatility and marked to market daily, which is not a detail: quarterly cash-flow accounting lands the payoff after the drawdown has already been recorded, and makes a hedged book look *worse* than an unhedged one:

```python
import numpy as np
import pandas as pd
from scipy import stats

FF = {2000: 6.24, 2001: 3.88, 2002: 1.67, 2003: 1.13, 2004: 1.35, 2005: 3.22,
      2006: 4.97, 2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18, 2011: 0.10,
      2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.39, 2017: 1.00,
      2018: 1.83, 2019: 2.16, 2020: 0.38, 2021: 0.08, 2022: 1.68, 2023: 5.02,
      2024: 5.15, 2025: 4.33}
p8 = pd.read_parquet("data/part8.parquet")
px = pd.read_parquet("data/prices.parquet")["SPY"].dropna()
vix = p8.vix.reindex(px.index).ffill().bfill()
spy = px.pct_change().fillna(0.0)

def bs_put(S, K, T, r, s):
    if T <= 1e-9:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T))
    return K * np.exp(-r * T) * stats.norm.cdf(s * np.sqrt(T) - d1) - S * stats.norm.cdf(-d1)

def put_program(life=63, moneyness=0.90, skew=1.20):
    """Rolling OTM puts, marked to market DAILY -- quarterly cash-flow accounting
    lands the payoff after the drawdown and makes maxDD look worse than unhedged."""
    pnl, prem, rolls, itm, i = pd.Series(0.0, index=px.index), 0.0, 0, 0, 0
    while i < len(px) - 1:
        j = min(i + life, len(px) - 1)
        S0, K = px.iloc[i], moneyness * px.iloc[i]
        prev = bs_put(S0, K, life / 252, FF[px.index[i].year] / 100,
                      skew * vix.iloc[i] / 100)
        prem, rolls = prem + prev / S0, rolls + 1
        for t in range(i + 1, j + 1):
            cur = bs_put(px.iloc[t], K, (life - (t - i)) / 252,
                         FF[px.index[t].year] / 100, skew * vix.iloc[t] / 100)
            pnl.iloc[t] = (cur - prev) / S0
            prev = cur
        itm += px.iloc[j] < K
        i = j
    return pnl, prem / (len(px) / 252), rolls, itm

def summarize(r, lab):
    r = r.dropna()
    e = (1 + r).cumprod()
    d = e / e.cummax() - 1
    w = lambda a, b: (1 + r.loc[a:b]).prod() - 1
    return dict(lab=lab, cagr=e.iloc[-1] ** (252 / len(r)) - 1,
                vol=np.sqrt(252) * r.std(), sharpe=np.sqrt(252) * r.mean() / r.std(),
                mdd=d.min(), y08=w("2008-01-01", "2008-12-31"),
                q20=w("2020-01-01", "2020-03-31"), y22=w("2022-01-01", "2022-12-31"),
                es=-r[r <= r.quantile(0.025)].mean())

START = "2001-01-01"
mom = np.sign(np.log(px).diff().rolling(252).sum())
rows = [summarize(spy.loc[START:], "SPY buy and hold")]
for lab, sk in [("rolling puts, IV = VIX", 1.00), ("rolling puts, IV = VIX x 1.2", 1.20)]:
    pnl, ann, rolls, itm = put_program(skew=sk)
    print(f"  {lab:30s} {rolls} quarterly rolls, {itm} finished in the money, "
          f"premium {ann:.2%}/yr")
    rows.append(summarize((spy + pnl).loc[START:], lab))
rows.append(summarize((spy * (mom > 0).shift(1)).loc[START:], "trend overlay (long/flat)"))
for g in [0.8, 0.6]:
    rows.append(summarize((g * spy).loc[START:], f"SPY at {g:.0%} gross"))

print("\n  program                          CAGR     vol   Sharpe    maxDD     2008"
      "   2020Q1     2022   ES97.5")
for r in rows:
    print(f"  {r['lab']:30s} {r['cagr']:+6.2%} {r['vol']:7.2%} {r['sharpe']:8.3f} "
          f"{r['mdd']:8.1%} {r['y08']:+8.1%} {r['q20']:+8.1%} {r['y22']:+8.1%} "
          f"{r['es']:8.2%}")
base = rows[0]
print("\n  the price of protection, per point of maximum drawdown saved")
for r in rows[1:]:
    saved, given = (r["mdd"] - base["mdd"]) * 100, (base["cagr"] - r["cagr"]) * 100
    print(f"  {r['lab']:30s} gave up {given:5.2f} pp of CAGR, saved {saved:5.1f} pp "
          f"of maxDD  ->  {given / saved:.3f}")
# =>   rolling puts, IV = VIX         102 quarterly rolls, 9 finished in the money, premium 3.24%/yr
#      rolling puts, IV = VIX x 1.2   102 quarterly rolls, 9 finished in the money, premium 5.05%/yr
#
#      program                          CAGR     vol   Sharpe    maxDD     2008   2020Q1     2022   ES97.5
#      SPY buy and hold               +8.51%  19.33%    0.519   -55.2%   -36.8%   -19.4%   -18.2%    3.71%
#      rolling puts, IV = VIX         +7.98%  13.69%    0.629   -44.1%   -29.8%    -6.7%   -19.4%    2.33%
#      rolling puts, IV = VIX x 1.2   +6.15%  12.97%    0.525   -46.8%   -32.4%    -6.4%   -21.4%    2.22%
#      trend overlay (long/flat)      +7.30%  12.90%    0.611   -31.2%    -6.4%   -22.9%   -18.5%    2.73%
#      SPY at 80% gross               +7.07%  15.46%    0.519   -46.5%   -29.8%   -15.4%   -14.4%    2.97%
#      SPY at 60% gross               +5.49%  11.60%    0.519   -36.7%   -22.5%   -11.4%   -10.7%    2.23%
#
#      the price of protection, per point of maximum drawdown saved
#      rolling puts, IV = VIX         gave up  0.53 pp of CAGR, saved  11.0 pp of maxDD  ->  0.048
#      rolling puts, IV = VIX x 1.2   gave up  2.35 pp of CAGR, saved   8.4 pp of maxDD  ->  0.281
#      trend overlay (long/flat)      gave up  1.21 pp of CAGR, saved  24.0 pp of maxDD  ->  0.050
#      SPY at 80% gross               gave up  1.44 pp of CAGR, saved   8.7 pp of maxDD  ->  0.166
#      SPY at 60% gross               gave up  3.01 pp of CAGR, saved  18.5 pp of maxDD  ->  0.163
```

**The single most important number in this section is the skew multiplier, and it is an assumption rather than a measurement.** Priced at flat at-the-money VIX, the rolling put program costs 3.24% a year, gives up only 0.53 points of CAGR, saves 11 points of drawdown at an exchange rate of **0.048**, and *raises* the Sharpe ratio from 0.519 to **0.629** — protection that appears to be free. Repriced at VIX × 1.2, a conservative allowance for the volatility skew that a 10%-delta put actually trades at, the same program costs 5.05% a year, the exchange rate deteriorates to **0.281**, and the Sharpe falls back to 0.525. Nothing about the strategy changed; one assumption did, and it moved the verdict from "best hedge available" to "worst". Real SPX puts trade well above ATM VIX, so the pessimistic column is closer to reality — but the course's data cannot settle it, and an analysis whose conclusion is set by an unmeasured parameter must say so in the sentence that states the conclusion.

The trend overlay is the robust answer precisely because it needs no such assumption. Long when the trailing year is positive and flat otherwise, it converts SPY's −55.2% drawdown into **−31.2%** for 1.21 points of CAGR, raises the Sharpe to 0.611, and is the only program that improves drawdown, volatility *and* risk-adjusted return simultaneously. Against the crude alternative it is dominant: holding 60% of the position saves 18.5 points of drawdown at an exchange rate of 0.163, more than three times the cost. That is the case for convexity through position sizing rather than through the options market — you are manufacturing the payoff yourself instead of paying a dealer's skew for it.

The final row of the crisis columns is where the recommendation gets its caveat, and it is a large one. In the first quarter of 2020 the put program held the loss to **−6.7%** against the market's −19.4%, while the trend overlay returned **−22.9% — worse than doing nothing at all** — because a 252-day signal cannot react to a 23-day crash and simply held the position all the way down. In 2008 the ranking inverts completely: the trend overlay lost 6.4% against buy-and-hold's 36.8%, while the puts still lost 29.8%. **Puts buy speed and trend buys cheapness, and neither buys both.** A serious tail program holds some of each, sized so that the cheap protection carries the slow crises and the expensive protection is reserved for the gaps — and it budgets the premium as an operating cost, not as an investment expected to pay for itself.

!!! warning "The tail you can measure is the one that already happened, and it is the smaller of the two"
    Every number in this lesson is fitted to 25 years containing exactly one banking crisis, one pandemic and one inflation shock. The EVT shape parameter of +0.327 is an estimate from 119 exceedances; the tail-dependence figures assume the future decile structure resembles the past; the crisis replay is a test the book was implicitly selected to pass. What the analysis does establish is directional and robust: the Gaussian understates the far tail by two to five times, the realized worst drawdown is statistically indistinguishable from an ordinary one, and diversification is weakest when it is needed. Size for the loss the model has never seen, treat every hedging verdict as conditional on its pricing assumption, and remember that a stress test's job is to make the failure imaginable, not to predict its magnitude.

!!! abstract "Key takeaways"
    - Every realized maximum drawdown in this course is a **coin-flip median**: the 53rd, 58th and 79th percentiles of 10,000 random walks matched to each book's own Sharpe and length. A drawdown carries almost no information about whether a strategy has broken.
    - Duration is what the headline hides: `tsmom` spent **2,623 consecutive trading days** underwater and the five-sleeve book 1,123 days against a maximum depth of only 11.2%. Books get cut for duration, not depth.
    - The surviving book's EVT shape parameter is **ξ = +0.327** — moments exist only to order 3.1, so its kurtosis does not exist in the limit and every fourth-moment correction is undefined on it.
    - The Gaussian understates the 1-in-4-year loss by **2.65×** (3.67% vs 1.38%) and the 1-in-40-year loss by **5.01×** (8.36% vs 1.67%). The fit is threshold-stable for the surviving book and *not* for `tsmom`, whose ξ flips to −0.223 at the 99th percentile — quote EVT only where it is flat.
    - Correlations go to one **within** an asset class and not across it: SPY/EFA tail dependence is λ = 0.66, 6.6× independence, while **SPY/TLT is λ = 0.09, below independence** — the same pair whose *annual* correlation has been positive for four years. Daily and annual co-movement are different quantities.
    - The books survive slow crises (−1.6% through COVID, +0.1% in 2022, +6.3% for `tsmom` in the 2008 collapse) and lose in fast volatility spikes (−6.2% in August 2015 against SPY's −5.6%, on a third of the market's volatility) — because the surviving edge is short volatility.
    - Stressing correlations as well as prices costs a 12-asset book 20% more and **60/40 49% more**, because 60/40 holds 40% in the one asset whose beta flips from −0.24 to +0.38 — the more a portfolio depends on a diversifying relationship, the more of its stress loss is correlation risk rather than price risk.
    - Protection priced per point of drawdown saved: the trend overlay costs **0.050** and raises Sharpe from 0.519 to 0.611; de-grossing costs 0.163; rolling puts cost **0.048 at flat VIX and 0.281 at VIX × 1.2** — the verdict is set by the skew assumption. In 2020Q1 the puts returned −6.7% and the trend overlay −22.9%, worse than unhedged; in 2008 the ranking inverted. Puts buy speed, trend buys cheapness.

## Where this goes next

Part VIII set out to turn a collection of strategies into a book, and it ends with one that is smaller than it started. `tom` was retired on a t-statistic of +0.11 against a market beta of +8.0, `xsmom` failed a spanning test for the third independent time, `tsmom` turned out to be fully replicated by its own meta-labeled version, and `pairs` never recovered from a bad vendor mark in October 2008. What survives is two genuinely uncorrelated sleeves, sized well under Kelly, allocated by risk rather than dollars, rebalanced on drift bands, and carrying a trend overlay against the slow crises with an options budget reserved for the fast ones. Every one of those decisions rests on code that must run correctly, reproducibly, and unchanged from the version that was tested.

That is now the binding constraint, and it is not a quantitative one. A risk report computed from a stale parquet file, a rebalancing rule whose thresholds were edited in production, an optimizer whose constraint set differs between research and live — each of these fails silently and expensively, and none of them is detectable by any method in this part. [Part IX — Professional Software Engineering](../part-09-software-engineering/index.md) takes the codebase built across Parts II through VI and makes it trustworthy: version control and review practices for research code, tests that catch lookahead bias and golden-file backtests that catch unintended P&L changes, configuration and dependency injection, message-queue architecture, and the profiling and refactoring discipline that keeps a platform fast without breaking what it computes.
