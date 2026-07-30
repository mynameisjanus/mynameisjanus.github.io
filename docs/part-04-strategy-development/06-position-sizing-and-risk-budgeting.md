# Position Sizing and Risk Budgeting

Nothing in the previous five lessons decided how *much* of anything to hold. That was deliberate: sizing is a separate discipline with a separate failure mode — it cannot create edge, but it fully controls how edge is experienced, and how fast the absence of edge becomes fatal. This lesson runs the sizing stack from bottom to top: the function that maps a signal's value to a position; the volatility targeting that holds a book's risk constant; the budgets that divide risk across sleeves; the monitoring that checks whether the budget is being obeyed; the Kelly criterion that bounds how aggressive any of it may ever be; and the capacity arithmetic that says how much money the whole construction can absorb before it eats its own returns.

It also settles an account left open since [lesson one](01-momentum-and-trend-following.md). The `tsmom` book banked its diversification benefit as reduced volatility — 12.2% instead of 19.3% — while its Sharpe sat unmoved at 0.30. This lesson converts that stored benefit into performance, and the conversion is worth roughly a doubling.

## From signal to position

The trend signal inside `tsmom` is a number — the trailing year's return in volatility units — and lesson one kept only its *sign*. That is one choice from a menu: trade the sign, trade the magnitude linearly, or trade the magnitude up to a saturation point. Same signal, three books, everything else identical:

```python
import numpy as np
import pandas as pd

r = np.log(pd.read_parquet("data/prices.parquet")["SPY"]).diff().dropna()
score = r.rolling(252).sum() / (r.rolling(252).std() * np.sqrt(252))

for name, m in [("sign", np.sign(score)), ("linear", score),
                ("clipped +/-1", score.clip(-1, 1))]:
    pos = (m / m.abs().mean()).shift(1)          # same average gross for all
    s = (pos * r).dropna()
    turn = pos.diff().abs().sum() / (len(pos) / 252)
    print(f"{name:12s}: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"position turnover {turn:.1f}x gross/yr")
# => sign        : Sharpe 0.30, position turnover 7.0x gross/yr
#    linear      : Sharpe 0.23, position turnover 17.1x gross/yr
#    clipped +/-1: Sharpe 0.24, position turnover 11.8x gross/yr
```

The crudest mapping wins on both columns: the sign rule earns the best Sharpe *and* trades less than half as much as the linear rule. That is not a general law — it is a verdict about *this signal*, and it is exactly the verdict [lesson five's](05-feature-and-signal-engineering.md) machinery would have predicted: the trend signal's information lives in its sign; its magnitude is mostly noise, so a linear map faithfully converts noise into positions and then pays turnover to keep the noise current. The design rule that generalizes: position granularity must be justified by measured information in the signal's magnitude — an IC computed on magnitudes, not just direction — and absent that evidence, the coarse map is the conservative choice. Every unit of turnover is a real cost pre-committed; every unit of granularity must buy something.

## Volatility targeting

A constant-dollar book has whatever risk the market assigns it — quiet years at 8%, crises at 25% — which makes every risk number in its history an average over regimes. The alternative is to fix risk and let dollars float: scale the position by $\sigma^{*} / \hat\sigma_t$, a volatility target over a volatility estimate. The choice of estimator is the design decision, because the estimate is always late ([Time Series Analysis](../part-03-statistics/03-time-series.md) fitted the lag structure of volatility precisely):

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()

for name, est in [("63-day rolling", tsmom.rolling(63).std()),
                  ("EWMA span 32", tsmom.ewm(span=32).std())]:
    lever = (0.10 / (np.sqrt(252) * est)).shift(1)
    vt = (lever * tsmom).dropna()
    print(f"{name}: realized vol {np.sqrt(252) * vt.std():.1%}, "
          f"leverage {lever.min():.1f}x-{lever.max():.1f}x, "
          f"Sharpe {np.sqrt(252) * vt.mean() / vt.std():.2f}  "
          f"(raw {np.sqrt(252) * tsmom.mean() / tsmom.std():.2f})")
# => 63-day rolling: realized vol 10.5%, leverage 0.3x-2.9x, Sharpe 0.49  (raw 0.30)
#    EWMA span 32: realized vol 10.5%, leverage 0.2x-3.1x, Sharpe 0.57  (raw 0.30)
```

There is the conversion this part has owed since lesson one: Sharpe 0.30 becomes 0.49 with a slow estimator and 0.57 with a fast one, at a realized volatility pinned within half a point of the 10% target. Two distinct mechanisms deliver the improvement, and separating them matters. The first is bookkeeping: vol targeting spends the diversification dividend — the book borrows up to 3.1x when its sleeves are quiet and offsetting, exactly the stored benefit equal-dollar weighting left idle. The second is genuinely dynamic: volatility spikes cluster in the regimes where trend signals are being whipsawed ([Part III's HMM](../part-03-statistics/06-bayesian-methods-and-hmms.md) put numbers on that overlap), so scaling down in high vol systematically avoids the strategy's own worst conditions. The estimator comparison prices responsiveness: the EWMA's faster reaction to regime shifts is worth 0.08 of Sharpe against the rolling window's extra lag. And one number deserves respect before moving on: 3.1x is real leverage, and everything from here on assumes it can be financed and held — an assumption [lesson seven](07-portfolio-construction-and-transaction-costs.md) will stress.

## Risk budgets across sleeves

Equal dollars is not equal risk. The three trend sleeves have different volatilities, so the dollar-equal book quietly lets its riskiest sleeve spend the most risk — and "how much risk is each position spending" is precisely what a risk budget makes explicit. Each sleeve's risk contribution is

$$
RC_i \;=\; \frac{w_i\,(\Sigma w)_i}{\sqrt{w^{\top} \Sigma w}} ,
$$

its weight times its marginal contribution to portfolio volatility — contributions that sum exactly to the portfolio's total. Equal-risk-contribution (ERC) weights make them equal by construction:

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
sleeves = (np.sign(rets.rolling(252).sum()).shift(1) * rets).dropna()
cov = sleeves.cov().values * 252

def risk_contrib(w):
    return w * (cov @ w) / np.sqrt(w @ cov @ w)

eq3 = np.ones(3) / 3
rc = risk_contrib(eq3)
print("equal-dollar risk contributions:",
      "  ".join(f"{a} {c:.0%}" for a, c in zip(sleeves.columns, rc / rc.sum())))
res = minimize(lambda w: ((risk_contrib(w) - risk_contrib(w).mean()) ** 2).sum(),
               eq3, bounds=[(0.01, 1)] * 3,
               constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
w = res.x
print("ERC weights:              ",
      "  ".join(f"{a} {wi:.0%}" for a, wi in zip(sleeves.columns, w)))
eq = sleeves.mean(axis=1)
erc = sleeves @ w
for name, s in [("equal-dollar", eq), ("ERC", erc)]:
    print(f"{name}: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"ann vol {np.sqrt(252) * s.std():.1%}")
# => equal-dollar risk contributions: SPY 40%  TLT 24%  GLD 37%
#    ERC weights:               SPY 30%  TLT 39%  GLD 31%
#    equal-dollar: Sharpe 0.28, ann vol 10.5%
#    ERC: Sharpe 0.26, ann vol 10.3%
```

The diagnosis first: under equal dollars, the SPY sleeve spends 40% of the book's risk and TLT only 24% — the "equal-weight" book was never equal where it counts. ERC repairs that, handing each sleeve a third of the risk. Then the honest surprise: the repaired book's Sharpe is *lower*, 0.26 against 0.28, because ERC moved budget toward TLT — the sleeve lesson one measured at Sharpe 0.05 — and away from SPY, the one sleeve with an edge. This is the correct result, properly understood: risk budgeting is a *governance* technology, not a return forecast. ERC is the right allocation exactly when you decline to predict which sleeve will perform — insurance against your own confidence, priced here at 0.02 of Sharpe. When you *do* hold edge estimates, the budget should tilt toward edge — but then the estimates' error bars, which [Part III](../part-03-statistics/04-hypothesis-testing-and-multiple-testing.md) measured at ±0.2 of Sharpe apiece, come with them. Budgeting risk equally is what humility looks like as an allocation.

## Monitoring the budget

A target is a promise, and promises get audited. The vol-targeted book claims 10%; the audit asks what it actually ran:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
lever = (0.10 / (np.sqrt(252) * tsmom.ewm(span=32).std())).shift(1)
vt = (lever * tsmom).dropna()

realized = np.sqrt(252) * vt.rolling(63).std()
print(f"vol-targeted book, 63d realized vol: median {realized.median():.1%}, "
      f"above 12% on {(realized > 0.12).mean():.0%} of days, max {realized.max():.1%}")
print(f"worst breach window: {realized.idxmax():%Y-%m}")
# => vol-targeted book, 63d realized vol: median 10.4%, above 12% on 8% of days, max 15.9%
#    worst breach window: 2020-03
```

The median says the thermostat works — 10.4% against a 10% setting. The tail says what thermostats cannot do: 8% of days ran more than a fifth over budget, peaking at 15.9%, and the peak is dated March 2020, because every volatility estimator is a rear-view instrument and regime breaks happen in front of the car. The operational conclusion is a *trigger discipline*, decided before the breach: a standing threshold (say, realized 20% over target), a pre-agreed response (cut leverage mechanically, no meeting required), and a pre-agreed re-entry rule. The alternative — deciding what to do about a breach during the breach — reliably produces the worst of both worlds, de-risking at the bottom and re-risking after the recovery. Sizing rules exist precisely so that no one has to be smart on the worst day of the year.

## Kelly, and why half is the ceiling

How much leverage is too much is not a matter of taste — there is a growth-optimal answer. For a strategy with edge $\mu$ and variance $\sigma^2$, log-wealth growth is maximized at

$$
f^{*} \;=\; \frac{\mu}{\sigma^{2}} ,
$$

the Kelly fraction ([Kelly Criterion](../appendix/part-18-quant-finance-applications/01-kelly-criterion.md) derives it). Feed it the vol-targeted book and watch what "optimal" means operationally:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
rets = np.log(px[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(rets.rolling(252).sum()).shift(1) * rets).mean(axis=1).dropna()
lever = (0.10 / (np.sqrt(252) * tsmom.ewm(span=32).std())).shift(1)
vt = (lever * tsmom).dropna()

mu, var = 252 * vt.mean(), 252 * vt.var()
f_star = mu / var
print(f"vol-targeted book: mu {mu:+.1%}/yr at {np.sqrt(var):.1%} vol -> "
      f"full Kelly f* = {f_star:.1f}x (a {f_star * np.sqrt(var):.0%}-vol book)")
for k, frac in [("full Kelly", 1.0), ("half Kelly", 0.5)]:
    eq = (1 + frac * f_star * vt).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    print(f"{k}: growth {(eq.iloc[-1]) ** (252 / len(eq)) - 1:+.1%}/yr, "
          f"maxDD {mdd:.0%}")
# => vol-targeted book: mu +6.0%/yr at 10.5% vol -> full Kelly f* = 5.5x (a 57%-vol book)
#    full Kelly: growth +17.7%/yr, maxDD -94%
#    half Kelly: growth +13.1%/yr, maxDD -70%
```

Full Kelly instructs a 5.5x levered, 57%-volatility book, and its historical path — the *growth-optimal* path, remember — includes a 94% drawdown. Half Kelly surrenders 4.6 points of growth to shrink the ruin to 70%, which is still a number no investor, employer, or spouse survives ([Probability of Ruin](../appendix/part-18-quant-finance-applications/02-probability-of-ruin.md) and [Drawdown Probabilities](../appendix/part-18-quant-finance-applications/03-drawdown-probabilities.md) formalize why these paths are unlivable long before they are unprofitable). Three compounding reasons make even half Kelly a ceiling rather than a target: $\mu$ is estimated with the error bars this course keeps measuring, and Kelly overbets *quadratically* in an overestimated edge; returns have the fat tails of [Part III's second lesson](../part-03-statistics/02-returns-and-distributions.md), which the formula's Gaussian arithmetic ignores; and drawdown tolerance binds decades before growth-optimality does. The practical hierarchy: Kelly bounds the feasible from above, drawdown tolerance bounds it from below, and real books live near the *bottom* — the 10% target in use here is roughly a sixth of Kelly, which is not timidity but the standard price of estimated edges and institutional pain thresholds.

## Capacity

Every strategy has a size at which it stops working, because trading moves prices against the trader. The standard model says the impact of executing $Q$ dollars against average daily volume $ADV$ scales as $\sigma_{d}\sqrt{Q/ADV}$ — the square-root law — so annual impact drag is trade frequency times that, and capacity is the $Q$ where drag consumes some agreed share of the edge (here: half). Applied with stated, order-of-magnitude assumptions:

```python
import numpy as np

# stated assumptions: ADV $bn, daily vol, gross edge/yr, round trips/yr
assume = {
    "tsmom (SPY sleeve)": (30.0, 0.012, 0.037, 6),
    "xsmom (9 sectors)":  (1.5, 0.013, 0.030, 10.6),
    "pairs (SPY-IVV)":    (2.5, 0.0015, 0.0154, 13.2),
}
for name, (adv, sig, edge, rt) in assume.items():
    q = adv * 1e9 * (edge / (4 * rt * sig)) ** 2
    print(f"{name:18s}: capacity ~ ${q / 1e6:,.0f}M  "
          f"(ADV ${adv:.1f}bn, {rt:.0f} round trips/yr)")
# => tsmom (SPY sleeve): capacity ~ $495M  (ADV $30.0bn, 6 round trips/yr)
#    xsmom (9 sectors) : capacity ~ $4M  (ADV $1.5bn, 11 round trips/yr)
#    pairs (SPY-IVV)   : capacity ~ $95M  (ADV $2.5bn, 13 round trips/yr)
```

The numbers are rough by construction — the ADVs are hardcoded approximations, the impact coefficient is set to one, and ETF creation-redemption gives effective liquidity beyond on-screen volume — but the *structure* is what matters, because capacity scales with the **square** of edge-per-unit-trading: $Q^{*} = ADV \cdot (\text{edge}/4\,RT\,\sigma_d)^2$. That square is merciless to fast strategies. The slow SPY trend sleeve, six round trips a year in the deepest instrument on earth, supports about half a billion dollars; the sector rotation book — even granting it the 3% edge it does not have — supports *four million*, because monthly rebalancing across thin sector funds multiplies trips and the square does the rest. Two habits follow. Capacity is computed at design time, from the signal's turnover and the instrument's depth, before capital is raised — it is a property of the strategy's blueprint. And when a strategy must scale, the lever that works is the one inside the square: halve the turnover and capacity quadruples, which is why serious money is managed slowly.

## Sizing the course book

The framework, applied: put every sleeve on the common 10% budget and read off what each demands:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
p4 = pd.read_parquet("data/part4.parquet")
sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
r3 = np.log(px[["SPY", "TLT", "GLD"]]).diff()
rs = np.log(px["SPY"]).diff().dropna()

book = {}
book["tsmom"] = ((np.sign(r3.rolling(252).sum()).shift(1) * r3).mean(axis=1), 252)
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
raw = pd.Series(np.nan, index=z.index)
raw[z > 2], raw[z < -2] = -1.0, 1.0
raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
book["pairs"] = (raw.ffill().fillna(0.0).shift(1) * spread.diff(), 252)
mp = p4[sectors].resample("ME").last()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w9 = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
book["xsmom"] = ((w9.shift(1) * mp.pct_change()).sum(axis=1).loc["2001-02":], 12)
pim = rs.groupby(rs.index.to_period("M")).cumcount() + 1
rv = rs[::-1].groupby(rs[::-1].index.to_period("M")).cumcount()[::-1] + 1
book["tom"] = (rs.where((pim <= 3) | (rv == 1), 0.0), 252)

for name, (s, ann) in book.items():
    s = s.dropna()
    vol = np.sqrt(ann) * s.std()
    print(f"{name:6s}: ann vol {vol:5.1%}  ->  leverage for a 10% book: {0.10 / vol:4.1f}x")
# => tsmom : ann vol 12.2%  ->  leverage for a 10% book:  0.8x
#    pairs : ann vol  1.3%  ->  leverage for a 10% book:  8.0x
#    xsmom : ann vol 13.3%  ->  leverage for a 10% book:  0.8x
#    tom   : ann vol  8.1%  ->  leverage for a 10% book:  1.2x
```

Three sleeves size innocuously — deleverage the trend and sector books slightly, lever the calendar book a touch — and one line detonates, as promised in [lesson two](02-mean-reversion-and-pairs-trading.md): making `pairs` matter at book level requires **8x leverage**, because its raw volatility is 1.3% a year. Everything about that trade now scales by eight — its 154 bp of gross annual return, yes, but also its 168 round trips' worth of costs, its financing spread on 8x notional, and its exposure to the one scenario where a "riskless" arbitrage gaps: leverage is the mechanism by which small edges *and small flaws* both become book-sized. The fifth sleeve is deliberately absent from the loop: `shortvol`'s March 2020 print was 10.4 of its own monthly standard deviations, so a volatility-based size is an active misstatement of its risk — skew −7.6 means the budget that binds is the *tail* budget: size to survive the worst month on record plus margin, which lands far below what a 10%-vol rule would allocate. One framework, two currencies: sleeves with symmetric risk are sized in volatility; sleeves that sell insurance are sized in catastrophe.

!!! warning "Position sizing cannot create edge — it can only decide how fast you find out whether you have one"
    Every technique in this lesson is a multiplier on a number the earlier lessons estimated with wide error bars. Multipliers amplify in both directions: vol targeting doubled a real Sharpe honestly, and it would double an overfit one just as smoothly, right up until the out-of-sample bill arrives at 3.1x leverage. Size as if your edge estimate is too high — because after five lessons of watching estimates meet fresh data, you know it probably is.

!!! abstract "Key takeaways"
    - The mapping from signal to position is a measurable design choice: for the trend signal, the sign map beats linear (Sharpe 0.30 vs 0.23) while trading 60% less — magnitude granularity must be justified by measured information in the magnitude.
    - Volatility targeting converts the diversification dividend into performance: Sharpe 0.30 → 0.49 with a 63-day estimator, 0.57 with an EWMA, at 10.5% realized against a 10% target — the estimator's responsiveness alone is worth 0.08.
    - Equal dollars is not equal risk: the SPY sleeve was spending 40% of the budget. ERC (30/39/31) equalizes contributions and *costs* 0.02 of Sharpe by funding the edgeless sleeve — risk budgeting is governance and humility, not a return forecast.
    - The 10% thermostat held a 10.4% median but breached to 15.9% in March 2020 on an estimator that only sees backward — so triggers, responses, and re-entry rules are decided before the breach, never during it.
    - Full Kelly on the vol-targeted book prescribes 5.5x leverage and delivers a −94% drawdown along its growth-optimal path; half Kelly still hits −70%. Kelly is the ceiling of the feasible; drawdown tolerance sets the floor, and real books live near the floor.
    - Capacity scales with the square of edge-per-unit-trading: ~$495M for the slow SPY trend sleeve, ~$4M for monthly sector rotation even granting it an edge — halving turnover quadruples capacity, which is why serious money trades slowly.
    - On a common 10% budget, `pairs` demands 8x leverage — scaling its edge, its costs, and its flaws by eight — while `shortvol` refuses volatility sizing entirely: at skew −7.6, the binding budget is the tail, not the variance.

## Where this goes next

Each sleeve is now sized in isolation, and isolation is the last unrealistic assumption standing. A real book holds all five at once, and the moment they share a balance sheet, new arithmetic appears: correlations that make the whole cheaper than its parts, netting that cancels offsetting trades before they cost anything, and — the reckoning this part has promised since lesson two — transaction costs, modeled explicitly and charged against every gross number the last six lessons have printed. [Portfolio Construction and Transaction Costs](07-portfolio-construction-and-transaction-costs.md) assembles the book and presents the bill; two of the five sleeves will not survive the presentation.
