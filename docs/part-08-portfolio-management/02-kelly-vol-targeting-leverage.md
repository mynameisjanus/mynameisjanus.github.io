# Kelly, Volatility Targeting, and Leverage

[Risk Measurement](01-risk-measurement.md) produced the numbers a sizing decision needs — a 5.76% book volatility, a 1.12% daily VaR, an expected shortfall 1.46 times its own threshold — and stopped exactly where the interesting question starts. Knowing what a book risks does not say whether it is risking the right amount. This lesson answers *how much*, and it answers it three ways that constrain each other: growth theory sets a ceiling, financing costs set a price, and drawdown tolerance sets a floor that binds long before either.

Part IV settled some of this already, and the settled parts will not be re-derived. [Position Sizing and Risk Budgeting](../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) established that volatility targeting converts diversification into performance (Sharpe 0.30 to 0.57 on the three-asset trend book), that the Kelly fraction on that vol-targeted book was 5.5× with a −94% historical drawdown, and that real books live near a sixth of Kelly; [Portfolio Construction and Transaction Costs](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) showed that no-trade bands buy back turnover at no cost in Sharpe. What follows goes where those did not: the *shape* of the growth curve rather than one point on it, the reason fractional Kelly is right that turns out not to be the reason usually given, the financing bill that leverage actually generates, and whether cutting size in a drawdown protects a book or merely locks the loss in.

## The growth-optimal fraction, and the shape of the curve around it

For a strategy with arithmetic mean $\mu$ and variance $\sigma^2$, the fraction of capital that maximizes the long-run growth rate of wealth is $f^{*} = \mu/\sigma^{2}$ — equivalently $S/\sigma$, the Sharpe ratio over the volatility, which is why a doubling of Sharpe doubles the growth-optimal leverage ([Kelly Criterion](../appendix/part-18-quant-finance-applications/01-kelly-criterion.md) derives it). Part IV computed that number and read off two points. The whole curve is more informative, because the *asymmetry* around the optimum is where the practical advice lives:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
t = p8["s_tsmom"].dropna()
b = t / (np.sqrt(252) * t.std()) * 0.10          # the trend book at a 10% target
R = np.exp(b.values) - 1                          # simple returns: Kelly compounds
mu, var = 252 * R.mean(), 252 * R.var()
f_star = mu / var

print(f"tsmom at a 10% target, {b.index[0].date()} to {b.index[-1].date()}: "
      f"mu {mu:+.2%}/yr, sigma {np.sqrt(var):.2%} -> Kelly f* = mu/sigma^2 = {f_star:.2f}x "
      f"(a {f_star * np.sqrt(var):.0%}-vol book)")
grid = np.linspace(0.05, 2.5, 400) * f_star
g = np.array([np.log1p(f * R).sum() / (len(R) / 252) if (1 + f * R > 0).all() else np.nan
              for f in grid])
gmax = np.nanmax(g)
print(f"empirical log-optimal {grid[np.nanargmax(g)]:.2f}x at {gmax:+.2%}/yr")
print("   f/f*    lev   growth/yr   % of max   ann vol    maxDD   terminal")
for fr in [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    f = fr * f_star
    eq = np.cumprod(1 + f * R)
    gr = np.log(eq[-1]) / (len(R) / 252)
    print(f"   {fr:5.3f}  {f:4.2f}x    {gr:+7.2%}     {gr / gmax:6.1%}    {f * np.sqrt(var):6.1%}  "
          f"{(eq / np.maximum.accumulate(eq) - 1).min():+7.1%}   {eq[-1]:8.2f}")
# => tsmom at a 10% target, 2001-01-03 to 2025-06-30: mu +3.44%/yr, sigma 9.99% -> Kelly f* = mu/sigma^2 = 3.44x (a 34%-vol book)
#    empirical log-optimal 3.43x at +5.89%/yr
#       f/f*    lev   growth/yr   % of max   ann vol    maxDD   terminal
#       0.125  0.43x     +1.39%      23.5%      4.3%   -11.3%       1.40
#       0.250  0.86x     +2.59%      43.9%      8.6%   -21.6%       1.88
#       0.500  1.72x     +4.44%      75.3%     17.2%   -39.2%       2.96
#       0.750  2.58x     +5.54%      94.0%     25.8%   -54.9%       3.87
#       1.000  3.44x     +5.89%     100.0%     34.4%   -68.0%       4.22
#       1.250  4.30x     +5.49%      93.2%     43.0%   -78.1%       3.83
#       1.500  5.16x     +4.33%      73.5%     51.6%   -85.7%       2.88
#       2.000  6.89x     -0.31%      -5.3%     68.8%   -94.5%       0.93
```

The empirical optimum lands at 3.43× against the formula's 3.44×, which is the first useful result: the Gaussian arithmetic survives contact with a fat-tailed real series, at least where the optimum sits. The curve around it is a parabola in $f$, and the two halves of that parabola are priced completely differently *once drawdown is a column*. Half Kelly earns **75.3%** of the maximum growth with a −39.2% drawdown; one-and-a-half Kelly earns **73.5%** — statistically the same growth — with a −85.7% drawdown. Identical position on the growth curve, less than half the pain on one side of it. That is the entire argument for betting under the optimum rather than over it, and it does not require any uncertainty about $\mu$ to make: it follows from the geometry of a symmetric function evaluated against an asymmetric cost.

Two more readings before moving on. Double Kelly is not merely suboptimal but *destructive* — growth of −0.31% a year, a terminal wealth multiple below one, a −94.5% path — because the drag term $\tfrac{1}{2}f^{2}\sigma^{2}$ grows quadratically while the edge term grows linearly, and at $2f^{*}$ they exactly cancel. And the *level* of the whole exercise deserves suspicion: full Kelly on this book means running it at **34% annualized volatility** and accepting a −68% drawdown along the growth-optimal path. Nobody with external capital survives that path, which is why the practical question was never "what is $f^{*}$" but "what fraction of it".

## Full Kelly on the best book in this course cannot be placed

Applied to the surviving two-sleeve book — the highest-Sharpe construction Part IV produced — the formula returns an answer that is not merely aggressive but arithmetically unavailable:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
w = p8[["s_tsmom", "s_shortvol"]].dropna()
b = (w / (np.sqrt(252) * w.std()) * 0.10 / 2).sum(axis=1)
R = np.exp(b.values) - 1
mu, var = 252 * R.mean(), 252 * R.var()
f_star, ruin = mu / var, 1 / abs(R.min())

print(f"surviving book: Sharpe {np.sqrt(252) * b.mean() / b.std():.3f} at "
      f"{np.sqrt(var):.2%} vol, worst day {R.min():+.2%} on {b.idxmin():%Y-%m-%d}")
print(f"  Kelly f*                 {f_star:6.2f}x   (a {f_star * np.sqrt(var):.0%}-vol book)")
print(f"  ruin bound 1/|worst day| {ruin:6.2f}x   (above this, one day takes the book to zero)")
print(f"  Kelly exceeds the ruin bound by {f_star / ruin:.2f}x -- full Kelly is unplaceable")
grid = np.linspace(0.05, 1.2, 300) * f_star
g = np.array([np.log1p(f * R).sum() / (len(R) / 252) if (1 + f * R > 0).all() else np.nan
              for f in grid])
fg = grid[np.nanargmax(g)]
for lab, f in [("empirical log-optimal", fg), ("three-quarters of it", 0.75 * fg),
               ("half of it", 0.5 * fg)]:
    eq = np.cumprod(1 + f * R)
    print(f"  {lab:22s} {f:6.2f}x   growth {np.log(eq[-1]) / (len(R) / 252):+7.2%}/yr   "
          f"maxDD {(eq / np.maximum.accumulate(eq) - 1).min():+.1%}")
# => surviving book: Sharpe 1.174 at 7.23% vol, worst day -6.72% on 2018-02-05
#      Kelly f*                  16.84x   (a 122%-vol book)
#      ruin bound 1/|worst day|  14.88x   (above this, one day takes the book to zero)
#      Kelly exceeds the ruin bound by 1.13x -- full Kelly is unplaceable
#      empirical log-optimal   11.98x   growth +61.05%/yr   maxDD -94.0%
#      three-quarters of it     8.99x   growth +55.73%/yr   maxDD -80.0%
#      half of it               5.99x   growth +42.76%/yr   maxDD -61.3%
```

The Gaussian formula prescribes 16.84× leverage — a 122%-volatility book — and the worst day in the sample is −6.72%, so any leverage above **14.88×** converts that single day into total loss. The prescription exceeds the constraint by 13%. There is no interpretation under which the recommendation can be followed; the growth-optimal bet on the best book this course has built would have ended the book on 5 February 2018.

The failure is instructive because it is *structural*, not a rounding problem. $f^{*} = \mu/\sigma^{2}$ is derived from a continuous diffusion, in which arbitrarily large losses are arbitrarily unlikely and a position can always be adjusted before wealth reaches zero. Real returns arrive in daily jumps, and `shortvol` — the sleeve supplying most of this book's Sharpe — is precisely the kind of strategy whose jumps are one-sided. Volmageddon was a 6.72% loss on a 7.2%-volatility book, a nine-sigma day under the Gaussian the formula assumes, and the formula does not know it exists. The empirical optimum found by direct search is 11.98×, below the ruin bound because the search cannot cross it, and even *that* path draws down **94.0%**.

The general rule worth extracting: **Kelly is an upper bound derived under assumptions that fail exactly in the tail that binds**, so it is useful for saying "not more than this" and useless for saying "this much". A book whose Kelly fraction exceeds its ruin bound is telling you its return distribution is too jump-heavy for the growth-optimality framework to apply at all.

## Why fractional Kelly, really

The standard justification for betting a fraction of Kelly is estimation error: $\mu$ is measured with a wide standard error, Kelly overbets quadratically in an overestimated edge, so shade down. It is repeated everywhere, including in Part IV. It is also, as stated, mostly wrong — and separating the true reason from the popular one changes what fraction you choose. Two experiments, identical except for what the world is made of. In the first, returns are Gaussian and $\mu$ is estimated from $T$ days before being plugged into the formula. In the second, paths are block-bootstrapped from the *actual* book so that fat tails and volatility clustering survive, and ruin is counted rather than assumed away:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
w = p8[["s_tsmom", "s_shortvol"]].dropna()
b = (w / (np.sqrt(252) * w.std()) * 0.10 / 2).sum(axis=1)
R = np.exp(b.values) - 1
mu, var = 252 * R.mean(), 252 * R.var()
f_star = mu / var
rng = np.random.default_rng(7)
FR = [0.25, 0.5, 1.0, 1.5]

print(f"surviving book: true Kelly {f_star:.2f}x, full-information growth "
      f"{mu * f_star - 0.5 * f_star ** 2 * var:+.1%}/yr")
print("(a) Gaussian world -- mu estimated on T days, growth analytic, ruin impossible")
print("    f/f*    T=252 med/5th        T=1260 med/5th       T=4758 med/5th")
cell = {}
for T in [252, 1260, 4758]:
    for fr in FR:
        m_hat = rng.normal(mu / 252, np.sqrt(var / 252), (4000, T)).mean(axis=1) * 252
        f = fr * m_hat / var
        g = mu * f - 0.5 * f ** 2 * var
        cell[(T, fr)] = f"{np.median(g):+7.1%} /{np.percentile(g, 5):+8.1%}"
for fr in FR:
    print(f"    {fr:4.2f}   " + "  ".join(cell[(T, fr)] for T in [252, 1260, 4758]))

print("(b) block bootstrap of the real book -- 21-day blocks, fat tails, ruin counted")
print("    f/f*    T=252 med/mean/ruin        T=1260                     T=2520")
nb = 21
blocks = np.array([R[i:i + nb] for i in range(len(R) - nb)])
cell = {}
for T in [252, 1260, 2520]:
    for fr in FR:
        gs, ruined = [], 0
        for _ in range(2000):
            past = blocks[rng.integers(0, len(blocks), T // nb)].ravel()
            f = fr * (past.mean() * 252) / (past.var() * 252)
            fut = blocks[rng.integers(0, len(blocks), T // nb)].ravel()
            if (1 + f * fut <= 0).any():
                ruined += 1; gs.append(-1.0); continue
            gs.append(np.log1p(f * fut).sum() / (T / 252))
        cell[(T, fr)] = f"{np.median(gs):+7.1%} /{np.mean(gs):+7.1%} /{ruined / 2000:5.1%}"
for fr in FR:
    print(f"    {fr:4.2f}   " + "  ".join(cell[(T, fr)] for T in [252, 1260, 2520]))
# => surviving book: true Kelly 16.84x, full-information growth +74.1%/yr
#    (a) Gaussian world -- mu estimated on T days, growth analytic, ruin impossible
#        f/f*    T=252 med/5th        T=1260 med/5th       T=4758 med/5th
#        0.25    +32.7% /  -13.1%   +32.3% /  +13.9%   +32.2% /  +23.3%
#        0.50    +54.7% /  -27.2%   +55.5% /  +27.0%   +55.6% /  +42.8%
#        1.00    +51.9% / -115.0%   +69.6% /  +36.8%   +72.9% /  +63.9%
#        1.50    +11.9% / -419.8%   +52.7% /  -68.7%   +55.6% /   +6.7%
#    (b) block bootstrap of the real book -- 21-day blocks, fat tails, ruin counted
#        f/f*    T=252 med/mean/ruin        T=1260                     T=2520
#        0.25    +24.8% / +35.5% / 0.7%   +30.9% / +33.5% / 0.0%   +31.1% / +32.7% / 0.0%
#        0.50    +26.8% / +33.2% / 5.4%   +44.5% / +43.4% / 3.5%   +49.0% / +49.3% / 0.8%
#        1.00     +4.4% /  +4.0% /19.1%   +28.5% /  +7.7% /30.8%   +13.4% / -10.5% /43.4%
#        1.50    -52.6% / -11.1% /34.0%  -100.0% / -32.5% /54.6%  -100.0% / -60.4% /74.8%
```

Panel (a) does not support the folk argument. With one year of data, estimation error is genuinely punishing and full Kelly's fifth percentile is −115% — the overbetting risk is real at short samples. But by five years the ordering has already reversed, and at nineteen years full Kelly wins on **both** the median (+72.9% against half-Kelly's +55.6%) *and* the fifth percentile (+63.9% against +42.8%). In a Gaussian world with two decades of history, estimation error alone does not justify shading down at all. The standard argument, taken literally, expires with the sample size.

Panel (b) is the same experiment with the Gaussian assumption removed, and it reverses the conclusion permanently. **Half Kelly maximizes mean log-growth at every horizon** — +33.2%, +43.4%, +49.3% against full Kelly's +4.0%, +7.7% and **−10.5%** — and the mechanism is in the last column. Full Kelly ruins **19.1%, 30.8% and 43.4%** of paths, and more history makes it *worse* rather than better, because a longer path is more likely to contain the day that ends it. Note what happens to full Kelly's median and mean at T=2520: the median is still positive (+13.4%) while the mean is negative (−10.5%), the signature of a distribution where most paths do fine and a large minority go to zero. Reporting the median of a levered strategy is how that minority gets hidden.

So the correction is worth stating plainly: **fractional Kelly is justified by the left tail, not by the standard error of $\hat\mu$.** The distinction is not academic, because the two arguments prescribe different behavior. Estimation error is something you *learn away* — collect more data, shade less. Jump risk is a permanent property of the return distribution, and it does not shrink with sample size; on this book it gets more dangerous with it. A desk that shades to half Kelly because its estimates are noisy will re-lever as its track record lengthens, which is exactly backwards.

## Volatility targeting: not how often, but when

A volatility target is a promise to hold risk constant, and it is kept by trading — which makes rebalancing frequency the design decision. Part IV swept no-trade bands and found Sharpe almost completely insensitive to them, concluding that precision near the optimum is overpriced. That conclusion stands, and it is not the whole picture, because *Sharpe* is not what a volatility target promises. The promise is a volatility, so the metric is tracking error against it:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
w = p8[["s_tsmom", "s_shortvol"]].dropna()
raw = (w / (np.sqrt(252) * w.std()) * 0.10 / 2).sum(axis=1)
tgt = (0.10 / (np.sqrt(252) * raw.ewm(span=32).std())).shift(1).dropna()
base = raw.reindex(tgt.index)

def row(lab, held):
    b = (held * base).dropna()
    te = np.sqrt(((np.sqrt(252) * b.rolling(63).std() - 0.10) ** 2).mean())
    print(f"  {lab:12s} {held.diff().abs().sum() / (len(held) / 252):9.2f} "
          f"{(held.diff() != 0).mean() * 252:9.0f} {te:12.3%} {np.sqrt(252) * b.std():13.2%} "
          f"{np.sqrt(252) * b.mean() / b.std():8.3f}")

print("  rule         turnover/yr resets/yr   RMS vol TE  realized vol   Sharpe")
row("daily", tgt)
for lab, fr in [("weekly", "W"), ("monthly", "ME"), ("quarterly", "QE")]:
    row(lab, tgt.groupby(pd.Grouper(freq=fr)).transform("first"))
for band in [0.10, 0.25]:
    h, cur = tgt.to_numpy(copy=True), tgt.iloc[0]
    for i in range(len(h)):
        if abs(h[i] - cur) / cur > band:
            cur = h[i]
        h[i] = cur
    row(f"band {band:.0%}", pd.Series(h, index=tgt.index))
# =>   rule         turnover/yr resets/yr   RMS vol TE  realized vol   Sharpe
#      daily            17.72       252       2.943%        11.39%    1.312
#      weekly           11.57        52       3.719%        11.92%    1.280
#      monthly           7.26        12       7.301%        14.69%    1.116
#      quarterly         3.85         4      10.388%        17.16%    0.887
#      band 10%         11.04        37       2.922%        11.25%    1.293
#      band 25%          7.15        12       3.176%        11.37%    1.272
```

Compare the two rows that trade the same amount. Monthly rebalancing resets **12 times a year** at a turnover of 7.26; a 25% drift band resets **12 times a year** at a turnover of 7.15 — indistinguishable budgets. Their tracking errors are 7.301% and **3.176%**, and their realized volatilities are 14.69% and 11.37% against a 10% target. The calendar rule misses its target by 47%; the band misses it by 14%, for the same trading bill. **Calendar rebalancing does not fail because it trades too little. It fails because it trades on dates chosen by the calendar rather than by the portfolio**, holding stale leverage through the regime shifts that occur mid-month and then dutifully rebalancing on the 31st when nothing has changed. The band spends its identical budget only when the position has actually drifted, which is precisely when spending it does something.

One honest limit on all six rows: even *daily* rebalancing realizes 11.39% against a 10% target. No rule does better, because every one of them is driven by an EWMA estimate that lags, and the previous lesson measured that lag as a log-log slope of 0.737 — today's estimate systematically under-predicts the next volatility spike. A volatility target is a thermostat with a delay, and the delay, not the rebalancing rule, sets the floor on how well it can be held.

## What leverage actually costs

Everything above 1× is borrowed, and borrowing has a price that backtests routinely omit. The mechanics come first because they determine the price. Under **Regulation T**, a US margin account can borrow up to 50% of the value of a long equity position, so 2× is the retail ceiling; **portfolio margining** replaces that fixed haircut with a risk-based calculation that stresses the whole book, and a genuinely hedged portfolio can reach 6–8× under the same rules that cap a concentrated one at 2×. Short positions add a second cost entirely: the borrow fee on locating the shares, negligible for an S&P ETF and punitive for anything crowded. And the financing rate itself is a benchmark plus a broker spread, which for institutional accounts is tens of basis points and for retail accounts can exceed a full point.

No rate series exists in this course's caches and none is derivable from them, so financing is a stated assumption in the manner of Part IV's half-spread table — effective fed funds by calendar year, plus a 75 bp spread:

```python
import numpy as np
import pandas as pd

# effective fed funds, calendar-year averages (%). A stated assumption, like Part IV's
# half-spread table -- no rate series exists in the caches and none is derivable.
FF = {2000: 6.24, 2001: 3.88, 2002: 1.67, 2003: 1.13, 2004: 1.35, 2005: 3.22,
      2006: 4.97, 2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18, 2011: 0.10,
      2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.39, 2017: 1.00,
      2018: 1.83, 2019: 2.16, 2020: 0.38, 2021: 0.08, 2022: 1.68, 2023: 5.02,
      2024: 5.15, 2025: 4.33}
SPREAD = 0.75                                    # broker spread over the benchmark

p8 = pd.read_parquet("data/part8.parquet")
w = p8[["s_tsmom", "s_shortvol"]].dropna()
b = (w / (np.sqrt(252) * w.std()) * 0.10 / 2).sum(axis=1)
rate = pd.Series([FF[y] for y in b.index.year], index=b.index) / 100 + SPREAD / 100
print(f"financing = effective fed funds + {SPREAD:.2f}%: sample mean {rate.mean():.2%}, "
      f"range {rate.min():.2%} to {rate.max():.2%}")
print("  leverage   drag/yr   net Sharpe   at a flat 5% assumption   error")
for L in [1, 2, 3, 5]:
    net = L * b - (L - 1) * rate / 252
    flat = L * b - (L - 1) * 0.05 / 252
    s_net = np.sqrt(252) * net.mean() / net.std()
    s_flat = np.sqrt(252) * flat.mean() / flat.std()
    print(f"  {L}x        {(L - 1) * rate.mean():7.2%}      {s_net:7.3f}   "
          f"{s_flat:20.3f}   {s_flat - s_net:+7.3f}")
# => financing = effective fed funds + 0.75%: sample mean 2.32%, range 0.83% to 5.90%
#      leverage   drag/yr   net Sharpe   at a flat 5% assumption   error
#      1x          0.00%        1.174                  1.174    +0.000
#      2x          2.32%        1.014                  0.830    -0.184
#      3x          4.64%        0.961                  0.715    -0.246
#      5x          9.28%        0.918                  0.624    -0.295
```

Financing is not a rounding error and it is not catastrophic: 3× leverage costs 4.64% a year and **0.21 of Sharpe**, taking 1.174 to 0.961. The shape of the decay is the useful part — most of the damage happens between 1× and 2×, and the marginal cost of the third and fifth turns is small, because the drag scales linearly in $L$ while the numerator does too. Leverage does not degrade a Sharpe ratio the way costs degrade a fast strategy; it shifts it down by an amount set by the financing rate divided by the book's volatility.

The last column is the methodological point. Assuming a flat 5% financing rate — a plausible-sounding constant, and roughly today's level — puts 3× leverage at Sharpe **0.715** instead of 0.961, an error of **0.246** that is larger than most of the edges this course has certified. The sample average was 2.32% because it spans a decade at the zero bound, and a strategy backtested on 2009–2021 financing that goes live in 2024 will find that a third of its levered edge belonged to an interest-rate regime rather than to the strategy. Rate assumptions belong in the same category as spread assumptions: stated, dated, and stress-tested.

What the table cannot show is the risk that makes leverage qualitatively different from a cost. Financing is charged continuously; **margin is called discontinuously**. A book at 3× that loses a third of its equity is mechanically forced to sell into the market that just fell, converting a mark-to-market loss into a realized one at the worst available price, and the sale is not optional or negotiable. That is the gap-risk asymmetry: leverage multiplies returns linearly and multiplies the probability of being liquidated at the bottom far faster than linearly. Which leads directly to the question of whether cutting size *voluntarily* — before the broker does it involuntarily — is a good idea.

## Cutting size in a drawdown

The intuition is powerful: reduce risk when losing, restore it when recovered, and the deep drawdowns get truncated. It is the basis of every stop-loss overlay and most institutional de-risking policies. It is also testable. One implementation detail decides whether the test is meaningful — the trigger must read the **unmanaged** equity curve, because a rule that cuts to zero freezes its own curve, so its own drawdown never recovers and the rule deadlocks permanently after one trigger:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")

def scaled(cols, target=0.10):
    w = p8[cols].dropna()
    return (w / (np.sqrt(252) * w.std()) * target / len(cols)).sum(axis=1)

def control(b, cut, re, size):
    eq = np.exp(b.cumsum())
    dd = (eq / eq.cummax() - 1).values        # the UNMANAGED curve: a managed one deadlocks
    mult, on, sw = np.ones(len(b)), True, 0
    for i in range(1, len(b)):
        if on and dd[i - 1] < -cut:
            on, sw = False, sw + 1
        elif not on and dd[i - 1] > -re:
            on, sw = True, sw + 1
        mult[i] = 1.0 if on else size
    return pd.Series(mult, index=b.index) * b, sw

def line(b, lab, sw=None):
    eq = np.exp(b.cumsum())
    d = eq / eq.cummax() - 1
    runs, cur = [], 0
    for u in (d < -1e-12):
        cur = cur + 1 if u else 0
        runs.append(cur)
    after = d.loc[d.idxmin():]
    rec = after[after >= -1e-12]
    ttr = f"{(rec.index[0] - d.idxmin()).days}d" if len(rec) else "never"
    print(f"   {lab:24s} Sharpe {np.sqrt(252) * b.mean() / b.std():+.3f}  "
          f"CAGR {eq.iloc[-1] ** (252 / len(eq)) - 1:+.2%}  maxDD {d.min():+.1%}  "
          f"longest {max(runs):4d}d  recovery {ttr:>6s}"
          + ("" if sw is None else f"  switches {sw:2d}"))

for lab, b in [("the surviving book", scaled(["s_tsmom", "s_shortvol"])),
               ("tsmom alone", scaled(["s_tsmom"]))]:
    print(f"-- {lab}")
    line(b, "no control")
    for cut, re, size in [(0.10, 0.05, 0.0), (0.10, 0.05, 0.5),
                          (0.05, 0.025, 0.0), (0.05, 0.025, 0.5)]:
        m, sw = control(b, cut, re, size)
        line(m, f"cut {cut:.0%} / re {re:.1%} / {size:.0%}", sw)
# => -- the surviving book
#       no control               Sharpe +1.174  CAGR +8.91%  maxDD -12.0%  longest  598d  recovery   311d
#       cut 10% / re 5.0% / 0%   Sharpe +1.125  CAGR +7.89%  maxDD -13.2%  longest 1249d  recovery   396d  switches  6
#       cut 10% / re 5.0% / 50%  Sharpe +1.172  CAGR +8.40%  maxDD -11.5%  longest 1124d  recovery   342d  switches  6
#       cut 5% / re 2.5% / 0%    Sharpe +1.030  CAGR +6.32%  maxDD -16.0%  longest 1383d  recovery   222d  switches 22
#       cut 5% / re 2.5% / 50%   Sharpe +1.163  CAGR +7.60%  maxDD -11.5%  longest  622d  recovery   386d  switches 22
#    -- tsmom alone
#       no control               Sharpe +0.294  CAGR +2.98%  maxDD -24.7%  longest 2623d  recovery  never
#       cut 10% / re 5.0% / 0%   Sharpe +0.072  CAGR +0.50%  maxDD -27.4%  longest 2623d  recovery   849d  switches 19
#       cut 10% / re 5.0% / 50%  Sharpe +0.221  CAGR +1.73%  maxDD -22.8%  longest 2623d  recovery  1503d  switches 19
#       cut 5% / re 2.5% / 0%    Sharpe +0.166  CAGR +0.87%  maxDD -17.0%  longest 4349d  recovery  never  switches 29
#       cut 5% / re 2.5% / 50%   Sharpe +0.282  CAGR +1.92%  maxDD -17.0%  longest 2623d  recovery  1428d  switches 29
```

The headline is that **cutting to zero makes the maximum drawdown worse on both books** — from −12.0% to −13.2% on the surviving book, from −24.7% to −27.4% on `tsmom` — which is the opposite of the rule's entire purpose. The mechanism is whipsaw, and it is worth being precise about it. The rule sells after a decline and buys back after a partial recovery, so every round trip that does not precede a *continued* decline buys high and sells low by construction. On the surviving book six such trips over nineteen years were enough to cost 1.02 points of CAGR and to more than double the longest drawdown, from 598 days to 1,249. De-risking does not shorten drawdowns; it extends them, because a book held at reduced size recovers at reduced speed. That is the least intuitive number in this lesson and the one most worth internalizing: **a drawdown control converts depth into duration**, and duration is what actually ends careers and mandates.

Cutting to *half* rather than zero changes the sign of the verdict, and the two books disagree about how much. On the surviving book it is roughly a wash — Sharpe 1.172 against 1.174, maxDD −11.5% against −12.0%, at the cost of half a point of CAGR — which is a fair price for a policy whose real value is behavioral rather than statistical. On `tsmom` the tight version genuinely pays: cut at 5%, re-enter at 2.5%, hold half size, and the maximum drawdown falls from **−24.7% to −17.0%** while Sharpe barely moves (0.294 to 0.282). Better still, the uncontrolled `tsmom` book **never recovers** from its worst drawdown within the sample, while the controlled version recovers in 1,428 days — the control did not just make the hole shallower, it made it climbable. The price is a third of the compound return, 2.98% down to 1.92%.

The pattern across both books: drawdown control helps a weak, slow, trend-following book and does essentially nothing for a strong, fast-recovering one, and cutting to zero is destructive in every configuration tested. That is the sizing policy this part will carry — Kelly as a ceiling never approached, a volatility target held with drift bands rather than a calendar, leverage priced at a stated and dated financing curve, and a partial de-risking rule that is adopted for the discipline it imposes on humans rather than for a Sharpe improvement it does not deliver.

!!! warning "Every sizing rule is a bet that the past distribution's tail is the future distribution's tail"
    Kelly on the best book in this course prescribed 16.84× against a ruin bound of 14.88×, and the only reason the ruin bound is 14.88 rather than something smaller is that the sample happens to stop at a −6.72% day. One worse day in the next decade moves the bound, retroactively invalidating every leverage decision derived from it. The same is true of the volatility target, the drawdown thresholds, and the financing assumption. This is not an argument for paralysis; it is an argument for sizing off the tail you have *not* yet seen — stress the worst day, not the worst day observed — and for treating every number in this lesson as an upper bound with an unknown error rather than a setting.

!!! abstract "Key takeaways"
    - The growth curve is symmetric in $f$ and the drawdown column is not: half Kelly earns 75.3% of maximum growth at a −39.2% drawdown, one-and-a-half Kelly earns the same 73.5% at −85.7%, and double Kelly earns **−0.31% a year** while drawing down 94.5%.
    - Kelly on the surviving book is **16.84× against a ruin bound of 14.88×** — the growth-optimal bet is arithmetically unplaceable, because the formula assumes a diffusion and `shortvol` delivered −6.72% in one day.
    - Fractional Kelly is justified by the left tail, **not** by the standard error of $\hat\mu$: in a Gaussian world full Kelly wins on median *and* fifth percentile by five years of data, but block-bootstrapping the real book leaves half Kelly maximizing mean growth at every horizon while full Kelly ruins 19% to **43%** of paths — and gets worse with more history, not better.
    - Monthly rebalancing and a 25% drift band both reset 12 times a year at the same turnover; the band tracks the volatility target at 3.176% RMS error against monthly's 7.301%, and realizes 11.37% against 14.69%. Calendar rules fail on timing, not on frequency.
    - Financing at fed funds plus 75 bp costs 3× leverage 4.64% a year and 0.21 of Sharpe; assuming a flat 5% instead misprices it by **0.246 of Sharpe**, more than most edges in this course are worth.
    - Drawdown control that cuts to zero made maximum drawdown **worse on both books** (−12.0% → −13.2%, −24.7% → −27.4%) and more than doubled the longest drawdown from 598 to 1,249 days: de-risking converts depth into duration.
    - Cutting to half size at a 5% threshold is the only configuration that pays, and only on the weak book — `tsmom`'s maxDD falls from −24.7% to −17.0% and its worst drawdown becomes recoverable at all, for a third of the compound return.

## Where this goes next

Sizing has been treated so far as a question about one book taken as given, and the given was never examined: the surviving book holds two sleeves because Part IV's accounting killed three, and the five-sleeve book holds `tsmom` and `tsmom_meta` side by side despite [lesson one](01-risk-measurement.md) measuring their correlation at 0.756 and their joint risk share at half the total. Deciding *what goes in the book* is a different discipline from deciding how much of it to hold, and it turns on the only quantity that has been treated as a constant so far. [Risk Parity, Diversification, and Factors](03-risk-parity-diversification-factors.md) makes correlation the subject — how many independent bets a book of six strategies actually contains, why equal-risk allocation wins on wide universes and loses on narrow ones, what 2022 did to the most respected diversification strategy in the industry, and which of this course's sleeves turn out to be factor exposures wearing a strategy's name.
