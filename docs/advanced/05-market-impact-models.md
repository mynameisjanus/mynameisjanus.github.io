# Market Impact Models

[The execution module](04-optimal-execution-almgren-chriss.md) took the impact coefficients $\gamma$ and $\eta$ as given and then demonstrated that estimating them from a single desk's fills would require 82,551 metaorders. This module is about where those numbers come from, why impact has the functional form it does, and what a desk can honestly say about its own footprint. The distinction that organizes everything is between **permanent** impact — the part of the price move that persists because the market inferred something from your trading — and **temporary** impact, the concession paid for demanding liquidity faster than it replenishes, which decays once you stop. Permanent impact is information; temporary impact is congestion, and confusing them produces both bad schedules and bad capacity estimates.

Two results anchor the module. Kyle's model derives permanent impact from first principles as the market maker's rational Bayesian updating, and its equilibrium is verified here by simulation to three decimal places. The square-root law, by contrast, is not derived from anything — it is an embarrassingly robust empirical regularity that [Part IV](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) and [Part V](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) both already used, and this module explains what makes it credible and where it breaks. The honest-failure set-piece is the course turning its instruments on itself: measuring impact from the **1,103 real fills** in `data/part5trades.parquet` returns confidence intervals that swallow every plausible answer, exactly as the noise-floor arithmetic predicts. The constructive payoff arrives at the end, where the same law that resists measurement nonetheless delivers a capacity number — and it independently reproduces [Part IV's ~$495M estimate](../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) to within 2%.

## Kyle's model: permanent impact is the market learning

Why should trading move prices *permanently* at all? If a large order were known to be uninformed — an index fund rebalancing, a retiree liquidating — a rational market would supply liquidity and the price would snap back. Permanent impact exists because the market cannot tell. Kyle's 1985 model makes this precise with three players in one period.

An **informed trader** observes the asset's true value $v \sim \mathcal N(\mu, \sigma_v^2)$ and submits a market order $x$. **Noise traders** submit a net order $u \sim \mathcal N(0, \sigma_u^2)$, independent of $v$, for reasons unrelated to value. A competitive **market maker** sees only the *combined* flow $y = x + u$ — never its decomposition — and, being competitive, prices at the conditional expectation:

$$
P \;=\; \mathbb{E}[\,v \mid y\,].
$$

Look for a linear equilibrium: $x = \beta(v - \mu)$ and $P = \mu + \lambda y$. The informed trader maximizes expected profit knowing their own order will move the price against them:

$$
\max_x\ \mathbb{E}\bigl[(v - P)\,x\bigr] \;=\; \max_x\ \bigl(v - \mu - \lambda x\bigr)x
\quad\Longrightarrow\quad
x \;=\; \frac{v - \mu}{2\lambda},
$$

so $\beta = 1/2\lambda$. The factor of two is the model's first lesson: the insider trades *half* as aggressively as their edge would suggest, because impact is a cost they impose on themselves. Now close the loop with the market maker's pricing rule. Since $(v, y)$ are jointly Gaussian, the conditional expectation is the linear projection $\mathbb{E}[v \mid y] = \mu + \frac{\operatorname{Cov}(v, y)}{\operatorname{Var}(y)}y$, and with $\operatorname{Cov}(v,y) = \beta\sigma_v^2$ and $\operatorname{Var}(y) = \beta^2\sigma_v^2 + \sigma_u^2$,

$$
\lambda \;=\; \frac{\beta\,\sigma_v^2}{\beta^2\sigma_v^2 + \sigma_u^2}.
$$

Substituting $\beta = 1/2\lambda$ and solving the resulting pair gives the equilibrium in closed form:

$$
\lambda \;=\; \frac{1}{2}\,\frac{\sigma_v}{\sigma_u},
\qquad
\beta \;=\; \frac{\sigma_u}{\sigma_v},
\qquad
\mathbb{E}[\text{insider profit}] \;=\; \frac{\sigma_v\sigma_u}{2}.
$$

Three readings follow immediately. **Impact is linear** in order flow, with slope $\lambda$ — "Kyle's lambda," still the standard measure of illiquidity. **Impact is permanent**, because $P$ is the market's posterior mean and posteriors do not revert absent new information. And **$\lambda$ scales as $\sigma_v/\sigma_u$**: markets are illiquid when there is much to learn (high $\sigma_v$) and liquid when there is much noise to hide in (high $\sigma_u$), which is the formal version of the intuition that uninformed volume is what makes a market deep. The insider's profit $\sigma_v\sigma_u/2$ is paid entirely by the noise traders, and the market maker breaks even by construction:

```python
import numpy as np

rng = np.random.default_rng(0)
sig_v, sig_u, mu_v, M = 20.0, 500_000.0, 100.0, 500_000
lam, beta = 0.5 * sig_v / sig_u, sig_u / sig_v

v = mu_v + sig_v * rng.standard_normal(M)          # true value, seen only by the insider
u = sig_u * rng.standard_normal(M)                 # noise flow
x = beta * (v - mu_v)                              # insider's order
y = x + u                                          # what the market maker sees
P = mu_v + lam * y                                 # competitive pricing

print(f"equilibrium: lambda {lam:.3e} $/share, beta {beta:,.0f} shares per $ of edge")
print(f"insider profit: theory ${sig_v * sig_u / 2 / 1e6:.2f}M, "
      f"simulated ${np.mean((v - P) * x) / 1e6:.2f}M")
print(f"market maker profit: ${np.mean((P - v) * y) / 1e6:+.3f}M (zero by competition)")
print(f"residual value uncertainty Var(v|y)/Var(v) = "
      f"{1 - np.corrcoef(v, y)[0, 1] ** 2:.3f} (theory 0.500)")
# => equilibrium: lambda 2.000e-05 $/share, beta 25,000 shares per $ of edge
#    insider profit: theory $5.00M, simulated $5.02M
#    market maker profit: $-0.010M (zero by competition)
#    residual value uncertainty Var(v|y)/Var(v) = 0.500 (theory 0.500)
```

The last line is the model's most quoted result: after observing the order flow, exactly **half** the insider's information has been incorporated into the price. Trading is how private information becomes public, and $\lambda$ is the exchange rate.

## The square-root law is embarrassingly universal

Kyle predicts linear impact. Four decades of measurement say otherwise: the price move caused by executing a metaorder of $Q$ shares in an instrument trading $V$ per day, over a horizon of roughly a day, is far better described by

$$
\Delta P \;=\; Y\,\sigma\,\sqrt{\frac{Q}{V}},
$$

with $\sigma$ the daily volatility and $Y$ a dimensionless constant near 1. This is the form [Part V's fill model III](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) used and [Part IV](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) used to price the sector book's capacity, and it is remarkable for two reasons. First, its *universality*: the same exponent near one-half, and roughly the same $Y$, has been measured in equities across dozens of markets, in futures, in FX, in options, and in crypto — across four decades and many orders of magnitude of size. Few empirical regularities in finance survive that kind of replication. Second, the form is essentially forced once you accept two premises. If impact depends only on $\sigma$, $Q$, and $V$, and if it is scale-invariant — doubling a market's typical size and volume leaves relative impact unchanged — then dimensional analysis leaves only $\sigma \cdot f(Q/V)$, and a power law $f(z) = Yz^{\delta}$ with $\delta \approx 1/2$ is what the data selects.

The tension with Kyle is real and instructive. Kyle describes a *single trade* revealing information; the square-root law describes a *metaorder* consuming a queue of latent liquidity that refills as it is depleted. The standard reconciliation — the latent-liquidity or "locally linear order book" argument of Tóth and coauthors — is that the volume available near the current price is much smaller than the volume that *would* appear if the price moved, so a large order walks into an effectively concave supply curve. Both models are right about different questions: Kyle about why impact persists, the square-root law about how it scales.

Reproducing the course's own ladder confirms the internal consistency of the numbers used since Part IV:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
spy = bars.xs("SPY", axis=1, level=1).dropna()
sigma = np.log(spy["Close"]).diff().std()
adv = (spy["Volume"] * spy["Close"]).tail(252).mean()      # recent dollar ADV
print(f"SPY: daily vol {sigma:.4%}, recent dollar ADV ${adv / 1e9:.1f}bn")
for q in [1e6, 1e8, 1e9]:
    print(f"  ${q / 1e6:>6,.0f}M order = {q / adv:7.4%} of ADV -> "
          f"{sigma * np.sqrt(q / adv) * 1e4:5.1f} bp")
sizes = np.array([1e6, 1e7, 1e8, 1e9])
print(f"log-log slope of the ladder: "
      f"{np.polyfit(np.log(sizes), np.log(sigma * np.sqrt(sizes / adv)), 1)[0]:.3f}")
# => SPY: daily vol 1.2304%, recent dollar ADV $33.1bn
#      $     1M order = 0.0030% of ADV ->   0.7 bp
#      $   100M order = 0.3023% of ADV ->   6.8 bp
#      $ 1,000M order = 3.0231% of ADV ->  21.4 bp
#    log-log slope of the ladder: 0.500
```

The ladder reproduces Part V's published 0.8 / 7.5 / 23.8 bp closely — the small differences are the ADV window, since that lesson used a $28.9bn figure against the $33.1bn measured over the most recent year here — and the fitted slope is 0.500 by construction, which is the point of running it: it confirms that the cost model the engine has been charging since Part V really is the square-root law and not something else wearing its name.

## Persistent order flow plus naive impact would break the market

One more empirical fact makes the modeling harder and more interesting. Order flow is *strongly autocorrelated*: the sign of successive market orders exhibits long memory, with autocorrelation $C(\ell) \sim \ell^{-\gamma_c}$ and $\gamma_c$ measured around 0.5 in equity markets, persisting over thousands of trades. This is not mysterious — it is metaorder splitting, exactly the behavior [the execution module](04-optimal-execution-almgren-chriss.md) prescribes.

But now there is a paradox. If each trade permanently moves the price by a fixed amount in its own direction, and trade signs are persistently autocorrelated, then prices would inherit that persistence and become strongly trending — wildly at odds with the near-martingale prices actually observed. The resolution is the **propagator model**: impact is neither purely permanent nor purely temporary, but *decays* according to a kernel $G$,

$$
m_t \;=\; \sum_{s < t} G(t - s)\,\varepsilon_s\,\lvert v_s\rvert^{\delta} \;+\; \text{noise},
$$

where $\varepsilon_s$ is the sign of trade $s$. Bouchaud and coauthors showed that requiring price variance to grow linearly in time — the diffusivity condition, i.e. no exploitable trending — *forces* the decay exponent to balance the flow's persistence: $G(\ell) \sim \ell^{-\beta_d}$ with

$$
\beta_d \;\approx\; \frac{1 - \gamma_c}{2}.
$$

The market's microstructure is thus finely tuned: impact decays at exactly the rate that converts persistent, predictable order flow into unpredictable prices. This is efficiency as an emergent property rather than an assumption, and it explains why the permanent/temporary split is a modeling convenience rather than a fact of nature — real impact is a continuum of decay rates, and "permanent" means "decays slower than the horizon you care about."

## Your own fills are a small, noisy, biased sample

Now the measurement. The standard instrument is the **markout**: for each fill, compare the price some time later against the fill price, signed by trade direction,

$$
m(\tau) \;=\; \operatorname{side}\cdot\left(\frac{P_{t+\tau}}{P_{\text{fill}}} - 1\right),
$$

so a buy that is followed by a price rise scores positive. The expected shape is diagnostic: temporary impact means the market moves *against* you at the moment of the fill and then relaxes, so markouts should start negative and decay toward the permanent level. Averaged over enough fills, the plateau estimates permanent impact and the initial gap estimates the temporary component.

"Enough fills" is doing enormous work in that sentence. The standard error of a mean markout over $N$ fills is $\sigma_\tau/\sqrt N$, where $\sigma_\tau \approx \sigma_{\text{daily}}\sqrt{\tau}$ is ordinary price noise over the markout horizon. With SPY's daily volatility around 123 basis points and an impact effect measured in fractions of a basis point at retail scale, the arithmetic is brutal before any data is touched:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
sigma_bp = np.log(bars["Close"]["SPY"]).diff().std() * 1e4
print(f"SPY daily volatility: {sigma_bp:.0f} bp")
for n in [1_103, 10_000, 100_000, 1_000_000]:
    print(f"  n = {n:>9,} fills: standard error of the mean 1-day markout = "
          f"{sigma_bp / np.sqrt(n):5.2f} bp")
print(f"fills needed to resolve a 1 bp effect at t = 2: {(2 * sigma_bp / 1.0) ** 2:,.0f}")
# => SPY daily volatility: 123 bp
#      n =     1,103 fills: standard error of the mean 1-day markout =  3.72 bp
#      n =    10,000 fills: standard error of the mean 1-day markout =  1.23 bp
#      n =   100,000 fills: standard error of the mean 1-day markout =  0.39 bp
#      n = 1,000,000 fills: standard error of the mean 1-day markout =  0.12 bp
#    fills needed to resolve a 1 bp effect at t = 2: 60,917
```

Sixty thousand fills to see one basis point. Hold that number against the next section.

## One thousand one hundred three fills, zero significant basis points

The course has a real trade log. [Part V's engine](../part-05-backtesting-engine/05-trade-logs-and-visualization.md) recorded **1,103 fills** across SPY, TLT, and GLD from 2001 to 2025, with timestamps, signed quantities, and prices. Point the markout machinery at it:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
trades = pd.read_parquet("data/part5trades.parquet")
close = bars["Close"]
print(f"trade log: {len(trades)} fills across {trades.symbol.nunique()} symbols, "
      f"{trades.ts.min().date()} to {trades.ts.max().date()}")

for horizon in [1, 5, 21]:
    parts = []
    for sym, g in trades.groupby("symbol"):
        s = close[sym].dropna()
        mk = []
        for ts, qty, p in zip(g.ts, g.qty, g.px):
            pos = s.index.searchsorted(ts)
            if pos + horizon < len(s):
                mk.append(np.sign(qty) * (s.iloc[pos + horizon] / p - 1) * 1e4)
        mk = np.array(mk)
        parts.append(f"{sym} {mk.mean():+6.1f} +/-{1.96 * mk.std(ddof=1) / np.sqrt(len(mk)):5.1f}")
    print(f"markout t+{horizon:>2}d (bp, 95% CI): " + " | ".join(parts))

pooled = []
for sym, g in trades.groupby("symbol"):
    s = close[sym].dropna()
    for ts, qty, p in zip(g.ts, g.qty, g.px):
        pos = s.index.searchsorted(ts)
        if pos + 1 < len(s):
            pooled.append(np.sign(qty) * (s.iloc[pos + 1] / p - 1) * 1e4)
pooled = np.array(pooled)
print(f"pooled t+1d: mean {pooled.mean():+.2f} bp, "
      f"95% CI +/- {1.96 * pooled.std(ddof=1) / np.sqrt(len(pooled)):.2f} bp, n = {len(pooled)}")
# => trade log: 1103 fills across 3 symbols, 2001-01-03 to 2025-06-04
#    markout t+ 1d (bp, 95% CI): GLD   -9.8 +/- 14.2 | SPY   +9.6 +/- 16.2 | TLT  +11.1 +/- 11.3
#    markout t+ 5d (bp, 95% CI): GLD   +5.8 +/- 25.2 | SPY  +17.6 +/- 31.4 | TLT   +0.0 +/- 21.7
#    markout t+21d (bp, 95% CI): GLD  -13.0 +/- 52.0 | SPY   -2.1 +/- 49.0 | TLT   +2.3 +/- 42.4
#    pooled t+1d: mean +4.11 bp, 95% CI +/- 8.12 bp, n = 1103
```

Nine estimates, and **not one of them is distinguishable from zero**. The signs disagree across instruments at every horizon, the confidence intervals run to ±14, ±31, ±52 basis points, and the pooled one-day estimate is +4.11 ± 8.12. Meanwhile the effect being hunted — the impact of a book trading a few million dollars in instruments with billion-dollar daily volume — is a *fraction* of one basis point by the square-root law. The previous section predicted a standard error of 3.72 bp on 1,103 fills, and the realized pooled interval of ±8.12 bp is that number scaled by 1.96 and inflated by the mixture of three instruments. **The experiment failed exactly as designed, and the design knew it would.**

Two further caveats prevent over-reading even the sign. These are *strategy* fills, not impact experiments: they were placed because a momentum signal fired, so any genuine forecasting power in that signal shows up in the markout as if it were impact. A positive markout after a buy is what alpha looks like *and* what impact looks like, and 1,103 observations cannot separate them. And the fills are the output of a simulator that charges a spread but assumes the order did not move the price — measuring impact in data generated by a model that excludes impact can only ever recover the model's own assumptions plus noise. The honest conclusion is the one this section exists to deliver: **"estimate your impact from your own fills" is advice that requires two orders of magnitude more fills than a retail-scale book will ever produce.** Impact coefficients come from published broker studies with millions of orders, and a desk's own data is for *monitoring* a calibration, not deriving one.

## Replacing the flat fee with a footprint

The constructive use of the law is not measuring your past but pricing your future. Every backtest in this course charged a size-independent cost — half-spread plus commission — which is correct at small size and increasingly fictional as capital grows. Replacing the constant with $c_0 + \sigma\sqrt{Q/V}$ turns a single Sharpe ratio into a curve against assets under management, and the curve is where a strategy's business case lives:

```python
import numpy as np
import pandas as pd

prices = pd.read_parquet("data/prices.parquet")
bars = pd.read_parquet("data/part5.parquet")
r3 = np.log(prices[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(r3.rolling(252).sum()).shift(1) * r3).mean(axis=1).dropna()
weights = (np.sign(r3.rolling(252).sum()).shift(1) / 3).dropna()
turn = weights.diff().abs().reindex(tsmom.index).fillna(0)

HS = {"SPY": 0.5, "TLT": 1.0, "GLD": 1.0}                  # Part IV half-spreads, bp
adv, vol = {}, {}
for s in HS:
    d = bars.xs(s, axis=1, level=1).dropna()
    adv[s] = (d["Volume"] * d["Close"]).mean()
    vol[s] = np.log(d["Close"]).diff().std()
print(f"tsmom gross Sharpe {np.sqrt(252) * tsmom.mean() / tsmom.std():.2f}, "
      f"{252 * tsmom.mean() * 1e4:.0f} bp/yr, one-way turnover "
      f"{turn.sum().sum() / (len(turn) / 252):.1f}x/yr")
print("dollar ADV: " + ", ".join(f"{s} ${adv[s] / 1e9:.2f}bn" for s in adv))

for aum in [1e6, 1e7, 1e8, 5e8, 1e9, 5e9]:
    cost = pd.Series(0.0, index=tsmom.index)
    for s in HS:
        bps = HS[s] + 0.2 + vol[s] * np.sqrt(turn[s] * aum / adv[s]) * 1e4
        cost += turn[s] * bps / 1e4
    net = (tsmom - cost).dropna()
    print(f"AUM ${aum / 1e6:>6,.0f}M: net Sharpe {np.sqrt(252) * net.mean() / net.std():+.2f}, "
          f"cost drag {252 * cost.mean() * 1e4:5.0f} bp/yr")
# => tsmom gross Sharpe 0.30, 370 bp/yr, one-way turnover 7.9x/yr
#    dollar ADV: SPY $17.23bn, TLT $0.92bn, GLD $1.28bn
#    AUM $     1M: net Sharpe +0.28, cost drag    25 bp/yr
#    AUM $    10M: net Sharpe +0.25, cost drag    60 bp/yr
#    AUM $   100M: net Sharpe +0.16, cost drag   172 bp/yr
#    AUM $   500M: net Sharpe -0.00, cost drag   375 bp/yr
#    AUM $ 1,000M: net Sharpe -0.13, cost drag   526 bp/yr
#    AUM $ 5,000M: net Sharpe -0.62, cost drag  1166 bp/yr
```

The gross Sharpe of 0.30 and 370 bp per year reconcile with [Part IV's published trend book](../part-04-strategy-development/01-momentum-and-trend-following.md), confirming the setup before the new column arrives. That column is the story: the strategy that looked identical at every size in Parts IV through VIII is worth **+0.28 at a million dollars and exactly nothing at five hundred million**, with the cost drag climbing from 25 to 375 basis points a year purely because the same trades are larger. Note also which instrument binds. SPY's $17bn of daily volume absorbs almost anything; TLT and GLD, at under $1.3bn, are where the drag accumulates — the book's capacity is set by its thinnest leg, not its average.

## Capacity is a number, so compute it

The curve above has a zero, and finding it converts an impact model into the single figure every allocator asks for. Setting net return to zero and solving for AUM,

$$
\alpha_{\text{gross}} \;=\; \mathcal{T}\left(c_0 + \sigma\sqrt{\frac{f \cdot \text{AUM}}{V}}\right)
\qquad\Longrightarrow\qquad
\text{AUM}^{*} \;=\; \frac{V}{f}\left(\frac{\alpha_{\text{gross}}/\mathcal{T} - c_0}{\sigma}\right)^{2},
$$

with $\mathcal T$ annual turnover and $f$ the per-trade fraction of capital. The quadratic is worth pausing on: capacity scales with the *square* of the edge and with the *inverse square* of volatility, so a strategy with twice the gross alpha has four times the capacity, and one trading instruments twice as volatile has a quarter of it. Solve the actual book numerically rather than through the simplified formula, since the three sleeves have different depths:

```python
import numpy as np
import pandas as pd
from scipy.optimize import brentq

prices = pd.read_parquet("data/prices.parquet")
bars = pd.read_parquet("data/part5.parquet")
r3 = np.log(prices[["SPY", "TLT", "GLD"]]).diff()
tsmom = (np.sign(r3.rolling(252).sum()).shift(1) * r3).mean(axis=1).dropna()
weights = (np.sign(r3.rolling(252).sum()).shift(1) / 3).dropna()
turn = weights.diff().abs().reindex(tsmom.index).fillna(0)
HS = {"SPY": 0.5, "TLT": 1.0, "GLD": 1.0}
adv, vol = {}, {}
for s in HS:
    d = bars.xs(s, axis=1, level=1).dropna()
    adv[s] = (d["Volume"] * d["Close"]).mean()
    vol[s] = np.log(d["Close"]).diff().std()

def net_bp(aum, Y=1.0):
    cost = pd.Series(0.0, index=tsmom.index)
    for s in HS:
        bps = HS[s] + 0.2 + Y * vol[s] * np.sqrt(turn[s] * aum / adv[s]) * 1e4
        cost += turn[s] * bps / 1e4
    return 252 * (tsmom - cost).mean() * 1e4

for Y in [0.5, 1.0, 2.0]:
    a = brentq(lambda A: net_bp(A, Y), 1e6, 1e13)
    print(f"impact coefficient Y = {Y:.1f}: breakeven capacity ${a / 1e6:,.0f}M")
half = brentq(lambda A: net_bp(A) - 0.5 * net_bp(1e6), 1e6, 1e13)
print(f"capacity retaining half the small-size edge (Y = 1.0): ${half / 1e6:,.0f}M")
# => impact coefficient Y = 0.5: breakeven capacity $1,949M
#    impact coefficient Y = 1.0: breakeven capacity $487M
#    impact coefficient Y = 2.0: breakeven capacity $122M
#    capacity retaining half the small-size edge (Y = 1.0): $133M
```

At the standard coefficient the trend book breaks even at **$487M** — which is worth comparing against [Part IV's independent estimate](../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) of roughly $495M for the same strategy, derived from different arithmetic in a different lesson. Two routes to within 2% of each other is the kind of agreement that earns a capacity number some trust.

The sensitivity rows are the honest fine print, and the spread is not small: halving $Y$ quadruples capacity to $1.9bn, doubling it cuts capacity to $122M. That quadratic sensitivity is the same exponent as in the formula, now working against you — and since $Y$ is precisely the parameter [the previous module](04-optimal-execution-almgren-chriss.md) showed to be nearly unmeasurable from a single desk's data, any capacity figure is really a *range* spanning an order of magnitude, which is how it should be published. The final line makes the planning point: breakeven capacity is not deployable capacity. Running at $487M means handing the market the entire edge for the privilege of trading, so a serious operator sizes to keep most of it — here **$133M**, the point where half the small-size edge survives — and treats breakeven as the wall rather than the target.

!!! warning "The law that prices your capacity cannot be estimated from your own fills"
    The square-root law is one of the most replicated regularities in finance, and it converts an impact coefficient into a capacity number that two independent routes in this course agree on within 2%. It is also, at any realistic desk's sample size, unmeasurable: 1,103 real fills produced nine markout estimates whose confidence intervals ran to ±52 bp and whose signs disagreed across instruments, because resolving one basis point needs 60,917 fills. Take the coefficient from the published literature, publish capacity as a range across plausible values of $Y$, and never let a monitoring statistic masquerade as a calibration.

!!! abstract "Key takeaways"
    - Kyle's equilibrium — $\lambda = \tfrac12\sigma_v/\sigma_u$, $\beta = \sigma_u/\sigma_v$, insider profit $\sigma_v\sigma_u/2$ — was verified by simulation at $5.02M against a theoretical $5.00M, with market-maker profit −$0.010M and residual value uncertainty 0.500 against a theoretical 0.500.
    - The insider trades half as aggressively as their raw edge implies, because impact is a cost they impose on themselves, and exactly half their information reaches the price.
    - The square-root law $\Delta P = Y\sigma\sqrt{Q/V}$ is forced by scale invariance and dimensional analysis; the course's own SPY ladder (0.7 / 6.8 / 21.4 bp from $1M to $1bn) fits a log-log slope of 0.500.
    - Long-memory order flow would make prices trend under naive permanent impact; the diffusivity condition forces the propagator to decay as $\ell^{-(1-\gamma_c)/2}$, so market efficiency is an emergent consequence of impact decay rather than an assumption.
    - Markouts on the course's **1,103 real fills** produced nine estimates, none distinguishable from zero, with intervals to ±52 bp and signs disagreeing across instruments — a predicted failure, since 60,917 fills are needed to resolve 1 bp against 123 bp of daily volatility.
    - Strategy fills confound impact with alpha, and fills generated by a simulator that assumes no impact can only return that assumption plus noise; own-fill markouts are for monitoring, never for calibration.
    - Impact-aware re-costing turns tsmom's single 0.30 into a curve: net Sharpe +0.28 at $1M, +0.16 at $100M, and 0.00 at $500M, with the binding constraint being TLT and GLD's sub-$1.3bn volume rather than SPY's $17bn.
    - Capacity scales with the square of gross edge and the inverse square of volatility; the trend book breaks even at **$487M** against Part IV's independently derived ~$495M, but spans $122M to $1.9bn as $Y$ ranges over 2.0 to 0.5.

## Where this goes next

The scheduling problem that this module's coefficients feed is [Optimal Execution](04-optimal-execution-almgren-chriss.md), which takes $\eta$ and $\gamma$ as inputs and produces the trajectory that trades impact against timing risk. The other side of the transaction — the liquidity provider deciding how wide to quote against flow that may be informed, which is Kyle's $\lambda$ seen from the market maker's chair — is [Market Making](12-market-making.md), where the same adverse-selection logic determines the spread rather than the price impact. For a market where the impact curve is not statistical at all but an exact closed-form function of pool reserves, [Crypto Market Microstructure](13-crypto-microstructure.md) works through automated market makers, whose slippage can be differentiated rather than estimated. And the capacity arithmetic here is the quantitative form of the constraint [Part X](../part-10-trading-business/01-capital-fund-structures-fees.md) describes commercially: a strategy's maximum size, not its Sharpe, is what determines whether it can carry a business.
