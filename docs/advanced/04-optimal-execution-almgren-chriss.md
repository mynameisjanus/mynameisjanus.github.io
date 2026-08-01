# Optimal Execution: Almgren–Chriss

[Part V's fill simulator](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) worked a two-million-share GLD order through a 10% participation cap, took three days to finish it, and reported an average fill 16 basis points above the day-one open — about $140,000 on an $89M order that a naive backtest would have awarded as free money. That lesson ended the section by noting that impact is sublinear in size, "which is why execution schedules exist." This module builds the schedule. Almgren–Chriss is the framework that turns a parent order into a trajectory by pricing the one trade-off execution cannot escape: trade fast and pay impact, trade slow and pay volatility, and there is no third option.

The model is developed in discrete time, matching Almgren and Chriss's own 2000 presentation and keeping every step verifiable in NumPy; [the stochastic calculus module](03-stochastic-calculus.md) is cited only for the $\sigma\sqrt{\tau}$ scaling that makes timing risk grow with the square root of elapsed time. Two results are worth previewing. First, the optimal trajectory turns out to be a hyperbolic sine — a formula that looks exotic and falls out of a second-order difference equation any reader of this course can solve. Second, and less comfortably, the same model whose *shape* is this clean has parameters that are nearly impossible to estimate: recovering the temporary-impact coefficient to a t-statistic of 2 would take **82,551 metaorders**, which is why the deliverable of an execution research desk is a frontier of trade-offs rather than a point estimate of a cost.

## Implementation shortfall is the benchmark that cannot be gamed

Before any model, a measurement convention. Perold's implementation shortfall compares the portfolio you actually got against a *paper portfolio* that transacted instantly at the price when the decision was made — the arrival price. Everything that went wrong between decision and completion lands in one number, and it decomposes:

$$
\text{IS} \;=\; \underbrace{Q\,(\bar{P}_{\text{exec}} - P_{\text{arrival}})}_{\text{execution: impact + timing}}
\;+\; \underbrace{Q_{\text{unfilled}}\,(P_{\text{final}} - P_{\text{arrival}})}_{\text{opportunity cost}} .
$$

The virtue of this benchmark is that it is not gameable. A VWAP benchmark rewards a trader who simply matches the day's volume profile — including on days when the right answer was to trade immediately and go home — and a trader graded against the closing price can improve their score by trading at the close regardless of whether that serves the fund. Implementation shortfall has no such loophole, because its reference price was fixed the moment the decision was made and no subsequent action can move it. Its cost, paid honestly, is high variance: a schedule can be perfect and still show a terrible shortfall because the stock moved. That variance is not noise to be averaged away. It is the timing risk this entire module exists to manage.

## Two impacts, one dial

Set up the problem. A position of $X$ shares must be liquidated over horizon $T$, split into $N$ intervals of length $\tau = T/N$. Write $x_k$ for the shares still held after interval $k$ (so $x_0 = X$, $x_N = 0$) and $n_k = x_{k-1} - x_k$ for the shares sold in interval $k$. The price follows an arithmetic random walk contaminated by the trading itself:

$$
S_k \;=\; S_{k-1} + \sigma\sqrt{\tau}\,\xi_k \;-\; \gamma\,n_k ,
\qquad \xi_k \sim \mathcal N(0,1)\ \text{i.i.d.},
$$

where $\gamma$ is **permanent impact** — the part of the price move that persists, because the market has updated its beliefs about value. Meanwhile the price actually received on each slice is worse than the prevailing price by a **temporary impact** that depends on how aggressively that slice was worked:

$$
\tilde S_k \;=\; S_{k-1} \;-\; \left(\epsilon\,\operatorname{sgn}(n_k) + \frac{\eta}{\tau}\,n_k\right),
$$

with $\epsilon$ a fixed cost per share (half-spread plus fees) and $\eta$ the temporary-impact coefficient, multiplying the *trading rate* $n_k/\tau$. Temporary impact is the cost of demanding liquidity faster than it replenishes; it disappears once you stop.

Total cost is $X S_0 - \sum_k n_k \tilde S_k$. Substituting and taking expectations, the permanent-impact terms telescope into a perfect square — and this is the model's first genuinely useful lesson:

$$
\mathbb{E}[C] \;=\; \underbrace{\tfrac12\gamma X^2}_{\text{schedule-independent}} \;+\; \epsilon X \;+\; \frac{\tilde\eta}{\tau}\sum_{k=1}^{N} n_k^2 ,
\qquad \tilde\eta = \eta - \tfrac12\gamma\tau .
$$

**Permanent impact does not depend on the schedule at all.** Whether the order is worked over an hour or a week, $\tfrac12\gamma X^2$ is paid in full — it is the cost of the *decision* to trade $X$ shares, not of the manner of trading. Only the $\sum n_k^2$ term is controllable, and being a sum of squares under a fixed sum constraint, it is minimized by spreading trades evenly. If cost were all that mattered, TWAP would be optimal and this module would end here.

Variance is what prevents that. Holding $x_k$ shares through interval $k+1$ exposes them to $\sigma\sqrt{\tau}$ of price movement, and the intervals are independent:

$$
\operatorname{Var}[C] \;=\; \sigma^2\tau\sum_{k=1}^{N-1} x_k^2 .
$$

Now the tension is explicit and algebraic. Expected cost penalizes $\sum n_k^2$ — trade evenly. Variance penalizes $\sum x_k^2$ — get flat early. Calibrate both to the actual GLD order Part V worked:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
gld = bars.xs("GLD", axis=1, level=1).dropna()
X = 2_000_000                                          # Part V's parent order
px0 = round(gld["Open"].iloc[0], 2)
adv = gld["Volume"].head(60).mean()
sigma = np.log(gld["Close"]).diff().head(252).std() * px0        # $ per share per day
eta = (10.0 / 1e4) * px0 / adv                          # 10 bp at one ADV per day
gamma = 0.5 * (2.5 / 1e4) * px0 / adv                   # permanent, a quarter the scale
eps = 0.5 * (1.0 / 1e4) * px0                           # half-spread, $ per share

print(f"GLD: arrival {px0:.2f}, 60-day ADV {adv:,.0f} shares, "
      f"daily vol {sigma / px0:.4%} (${sigma:.4f}/share/day)")
print(f"order {X:,} shares = ${X * px0 / 1e6:.1f}M = {X / adv:.2f}x ADV")
print(f"eta ${eta:.3e} per (share/day), gamma ${gamma:.3e} per share, eps ${eps:.4f}/share")
# => GLD: arrival 44.43, 60-day ADV 2,847,107 shares, daily vol 0.7603% ($0.3378/share/day)
#    order 2,000,000 shares = $88.9M = 0.70x ADV
#    eta $1.561e-08 per (share/day), gamma $1.951e-09 per share, eps $0.0022/share
```

## The optimal schedule is a hyperbolic sine, and the derivation is elementary

Minimize $\mathbb{E}[C] + \lambda\operatorname{Var}[C]$ over the interior holdings $x_1,\dots,x_{N-1}$, where $\lambda > 0$ is risk aversion in units of inverse dollars. Only two terms involve the decision variables, so differentiate with respect to $x_k$. It appears in exactly two of the squared trades — $n_k = x_{k-1}-x_k$ and $n_{k+1} = x_k - x_{k+1}$ — and in one variance term:

$$
\frac{\partial}{\partial x_k}\left[\frac{\tilde\eta}{\tau}\Bigl((x_{k-1}-x_k)^2 + (x_k-x_{k+1})^2\Bigr) + \lambda\sigma^2\tau\,x_k^2\right]
\;=\; 0 .
$$

Carrying out the differentiation gives $-2(x_{k-1}-x_k) + 2(x_k - x_{k+1}) + \frac{\lambda\sigma^2\tau^2}{\tilde\eta}x_k = 0$, which rearranges into a linear second-order difference equation — the discrete analogue of $x'' = \kappa^2 x$:

$$
\frac{x_{k-1} - 2x_k + x_{k+1}}{\tau^2} \;=\; \tilde\kappa^2\,x_k,
\qquad
\tilde\kappa^2 \;=\; \frac{\lambda\sigma^2}{\tilde\eta} .
$$

Difference equations of this form are solved by exponentials $x_k = A e^{\kappa t_k} + B e^{-\kappa t_k}$. Imposing the boundary conditions $x_0 = X$ and $x_N = 0$ and combining the exponentials into a hyperbolic sine gives the Almgren–Chriss trajectory:

$$
x_k \;=\; X\,\frac{\sinh\bigl(\kappa(T - t_k)\bigr)}{\sinh(\kappa T)},
\qquad
\kappa \;=\; \sqrt{\frac{\lambda\sigma^2}{\eta}}\ \ \text{(continuous limit)} .
$$

The single parameter $\kappa$ has units of inverse time and is the model's whole vocabulary for urgency. Two limits make it concrete. As $\kappa \to 0$, expand $\sinh(z) \approx z$ and the ratio collapses to $(T - t_k)/T$ — **TWAP is the risk-neutral solution**, a straight line to zero. As $\kappa T$ grows large, $\sinh(\kappa(T-t))/\sinh(\kappa T) \to e^{-\kappa t}$ and the trajectory becomes an exponential decay with time constant $1/\kappa$, so the position has a **half-life of $\ln 2/\kappa$** and the horizon $T$ becomes irrelevant — an urgent trader liquidates on their own clock, not the one they were given:

```python
import numpy as np

X, T, N = 2_000_000, 5.0, 5

def traj(kappa):
    t = np.linspace(0, T, N + 1)
    if kappa < 1e-9:
        return X * (1 - t / T)                          # TWAP
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

for kappa in [1e-9, 0.2, 0.5, 1.0, 2.0]:
    print(f"kappa {kappa:>7.1e}/day: holdings " + " ".join(f"{v / 1e6:5.2f}M" for v in traj(kappa)))
print(f"max |AC(kappa -> 0) - TWAP| = "
      f"{np.abs(traj(1e-9) - X * (1 - np.linspace(0, T, N + 1) / T)).max():.2e} shares")
# => kappa 1.0e-09/day: holdings  2.00M  1.60M  1.20M  0.80M  0.40M  0.00M
#    kappa 2.0e-01/day: holdings  2.00M  1.51M  1.08M  0.70M  0.34M  0.00M
#    kappa 5.0e-01/day: holdings  2.00M  1.20M  0.70M  0.39M  0.17M  0.00M
#    kappa 1.0e+00/day: holdings  2.00M  0.74M  0.27M  0.10M  0.03M  0.00M
#    kappa 2.0e+00/day: holdings  2.00M  0.27M  0.04M  0.00M  0.00M  0.00M
#    max |AC(kappa -> 0) - TWAP| = 2.33e-10 shares
```

The first row is TWAP to ten decimal places, confirming the limit numerically. The last row shows an urgent trader dumping 86% of the order on day one and finishing early in all but name — the five-day horizon was a constraint that stopped binding once $\kappa T = 10$.

## The efficient frontier is the deliverable, not the schedule

Sweeping $\lambda$ traces a curve in (expected cost, risk) space, and every point on it is optimal for *somebody*. Compute both moments in closed form and check them against simulation:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
gld = bars.xs("GLD", axis=1, level=1).dropna()
X, T, N = 2_000_000, 5.0, 5
px0 = round(gld["Open"].iloc[0], 2)
adv = gld["Volume"].head(60).mean()
sigma = np.log(gld["Close"]).diff().head(252).std() * px0
eta, gamma, eps = (10.0 / 1e4) * px0 / adv, 0.5 * (2.5 / 1e4) * px0 / adv, 0.5 * (1.0 / 1e4) * px0
tau = T / N

def traj(kappa):
    t = np.linspace(0, T, N + 1)
    return X * (1 - t / T) if kappa < 1e-9 else X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

def cost_risk(x):
    n = -np.diff(x)
    E = 0.5 * gamma * X ** 2 + eps * n.sum() + ((eta - 0.5 * gamma * tau) / tau) * np.sum(n ** 2)
    V = sigma ** 2 * tau * np.sum(x[1:-1] ** 2)
    return E, np.sqrt(V)

for lam in [1e-9, 1e-8, 5e-8, 2e-7, 1e-6]:
    kappa = np.sqrt(lam * sigma ** 2 / eta)
    E, sd = cost_risk(traj(kappa))
    print(f"lambda {lam:.0e}: kappa {kappa:5.3f}/day (half-life {np.log(2) / kappa:4.2f} d), "
          f"E[cost] ${E / 1e3:5.1f}k ({E / (X * px0) * 1e4:4.1f} bp), "
          f"sd ${sd / 1e3:6.1f}k ({sd / (X * px0) * 1e4:5.1f} bp)")
# => lambda 1e-09: kappa 0.086/day (half-life 8.11 d), E[cost] $ 20.1k ( 2.3 bp), sd $ 728.4k ( 82.0 bp)
#    lambda 1e-08: kappa 0.270/day (half-life 2.56 d), E[cost] $ 20.6k ( 2.3 bp), sd $ 639.8k ( 72.0 bp)
#    lambda 5e-08: kappa 0.605/day (half-life 1.15 d), E[cost] $ 26.1k ( 2.9 bp), sd $ 430.4k ( 48.4 bp)
#    lambda 2e-07: kappa 1.209/day (half-life 0.57 d), E[cost] $ 40.0k ( 4.5 bp), sd $ 211.2k ( 23.8 bp)
#    lambda 1e-06: kappa 2.704/day (half-life 0.26 d), E[cost] $ 59.5k ( 6.7 bp), sd $  45.3k (  5.1 bp)
```

Read the frontier as an exchange rate. Moving from the patient end to the urgent end costs **3.9 additional basis points of expected cost** (2.3 → 6.7) and removes **77 basis points of risk** (82.0 → 5.1). At the patient end the trade is a bargain — the first several basis points of urgency buy enormous risk reduction — and by the aggressive end it has turned expensive, each further basis point of cost buying progressively less. That curvature is the entire practical content of the model, and it explains why real desks cluster in the middle: nobody sensible sits at either corner.

Before trusting closed forms, simulate the whole thing — twenty thousand random price paths per schedule, each one accumulating impact and noise trade by trade:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
gld = bars.xs("GLD", axis=1, level=1).dropna()
X, T, N = 2_000_000, 5.0, 5
px0 = round(gld["Open"].iloc[0], 2)
adv = gld["Volume"].head(60).mean()
sigma = np.log(gld["Close"]).diff().head(252).std() * px0
eta, gamma, eps = (10.0 / 1e4) * px0 / adv, 0.5 * (2.5 / 1e4) * px0 / adv, 0.5 * (1.0 / 1e4) * px0
tau = T / N

def traj(kappa):
    t = np.linspace(0, T, N + 1)
    return X * (1 - t / T) if kappa < 1e-9 else X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

def closed_form(x):
    n = -np.diff(x)
    E = 0.5 * gamma * X ** 2 + eps * n.sum() + ((eta - 0.5 * gamma * tau) / tau) * np.sum(n ** 2)
    return E, np.sqrt(sigma ** 2 * tau * np.sum(x[1:-1] ** 2))

def simulate(x, M=200_000, seed=0):
    n = -np.diff(x)
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((M, N))
    S = np.zeros((M, N + 1))
    for k in range(N):
        S[:, k + 1] = S[:, k] + sigma * np.sqrt(tau) * xi[:, k] - gamma * n[k]
    exec_px = S[:, :-1] - (eps + (eta / tau) * n)       # price received on each slice
    return -(exec_px * n).sum(axis=1)                   # cost vs arrival price

for kappa in [1e-9, 0.5, 2.0]:
    x = traj(kappa)
    E, sd = closed_form(x)
    c = simulate(x)
    print(f"kappa {kappa:>7.1e}: formula E ${E / 1e3:6.1f}k sd ${sd / 1e3:6.1f}k | "
          f"200k paths E ${c.mean() / 1e3:6.1f}k sd ${c.std() / 1e3:6.1f}k")
# => kappa 1.0e-09: formula E $  20.0k sd $ 740.1k | 200k paths E $  19.1k sd $ 741.2k
#    kappa 5.0e-01: formula E $  23.9k sd $ 491.1k | 200k paths E $  23.2k sd $ 492.0k
#    kappa 2.0e+00: formula E $  52.9k sd $  92.3k | 200k paths E $  52.7k sd $  92.6k
```

Standard deviations agree to within a tenth of a percent and means to within Monte Carlo error, which validates both the algebra and the code.

## TWAP, VWAP, and shortfall answer different questions

Three benchmarks dominate practice, and choosing among them is choosing what you are willing to be judged on. **TWAP** slices evenly in clock time; it is the $\kappa \to 0$ solution above, it is trivially auditable, and it is the honest default when you have no view on intraday volume. **VWAP** slices in proportion to expected volume, $n_k \propto v_k$, which is TWAP in *volume time* rather than clock time — the same trajectory measured on a different clock, and the right benchmark when your goal is to be indistinguishable from the day's flow. **Implementation shortfall** schedules are the Almgren–Chriss family: they front-load, because they charge for the risk that TWAP and VWAP ignore entirely.

The distinction that matters operationally is what each benchmark does to the *agent's* incentives. VWAP is popular partly because it is easy to beat when you have discretion over participation, and a broker graded on VWAP can look excellent while systematically trading late in adverse markets. Shortfall benchmarks are unpopular for the mirror-image reason: they hold the agent responsible for volatility nobody controls. The resolution used by most serious desks is to grade *schedules* on shortfall and *fills within a schedule* on VWAP — the strategic decision against the benchmark that captures its economics, the tactical one against the benchmark that captures its craft.

## Part V's $140,000 was mostly weather, not footprint

Now return to the order that motivated the module and decompose it with the model in hand. Part V filled 2,000,000 GLD shares across three days under a 10% participation cap and paid an average of 44.5026 against a 44.43 arrival. How much of that was the order's own footprint, and how much was GLD simply drifting while the order worked?

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
gld = bars.xs("GLD", axis=1, level=1).dropna()
X, px0 = 2_000_000, round(gld["Open"].iloc[0], 2)
adv = gld["Volume"].head(60).mean()
sigma = np.log(gld["Close"]).diff().head(252).std() * px0
eta, gamma, eps = (10.0 / 1e4) * px0 / adv, 0.5 * (2.5 / 1e4) * px0 / adv, 0.5 * (1.0 / 1e4) * px0

fills = np.array([[599_200, 44.43], [1_165_530, 44.49], [235_270, 44.75]])   # Part V's trace
qty, pxs = fills[:, 0], fills[:, 1]
avg = (qty * pxs).sum() / qty.sum()
print(f"realized: average fill {avg:.4f} vs arrival {px0:.2f} = "
      f"{(avg / px0 - 1) * 1e4:+.1f} bp = ${(avg - px0) * X / 1e3:+.0f}k")

tau = 1.0
E_impact = (0.5 * gamma * X ** 2 + eps * qty.sum()
            + ((eta - 0.5 * gamma * tau) / tau) * np.sum(qty ** 2))
x_path = X - np.concatenate([[0], np.cumsum(qty)])
sd_timing = np.sqrt(sigma ** 2 * tau * np.sum(x_path[1:-1] ** 2))
print(f"model, same schedule: expected impact ${E_impact / 1e3:.0f}k "
      f"({E_impact / (X * px0) * 1e4:.1f} bp), timing-risk sd ${sd_timing / 1e3:.0f}k "
      f"({sd_timing / (X * px0) * 1e4:.1f} bp)")
resid = (avg - px0) * X - E_impact
print(f"residual ${resid / 1e3:+.0f}k = {resid / sd_timing:+.2f} standard deviations of timing risk")
# => realized: average fill 44.5026 vs arrival 44.43 = +16.3 bp = $+145k
#    model, same schedule: expected impact $34k (3.9 bp), timing-risk sd $480k (54.0 bp)
#    residual $+111k = +0.23 standard deviations of timing risk
```

The decomposition reframes Part V's number. Of the $145,000, roughly **$34,000 was the order's own footprint** and the remaining $111,000 was GLD drifting upward — a draw just **0.23 standard deviations** into a timing distribution whose standard deviation is $480,000. In other words, the headline slippage was mostly weather, and an identical order worked identically the following week could as easily have come in $300,000 *better* than arrival. This is not a correction to Part V, which was measuring realized slippage and said so; it is the refinement the model exists to provide. The practical consequence is sharp: judging an execution algorithm by a handful of realized shortfalls is measuring a $34,000 signal through $480,000 of noise, and the next section shows exactly how badly that goes.

## Calibration is where the model meets its epistemology

The trajectory formula needs $\eta$, $\gamma$, and $\sigma$. Volatility is easy. The impact coefficients are supposed to come from your own execution history: regress realized cost per share on trading rate and read off the slope. In a world where the model is exactly true and the only obstacle is sample size, how many metaorders does that take?

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet")
gld = bars.xs("GLD", axis=1, level=1).dropna()
px0 = round(gld["Open"].iloc[0], 2)
adv = gld["Volume"].head(60).mean()
sigma = np.log(gld["Close"]).diff().head(252).std() * px0
eta = (10.0 / 1e4) * px0 / adv

rng = np.random.default_rng(1)
for n_orders in [50, 500, 5_000, 50_000]:
    size = rng.uniform(0.05, 0.5, n_orders) * adv
    dur = rng.uniform(1, 5, n_orders)
    rate = size / dur
    observed = eta * rate + sigma * np.sqrt(dur) * rng.standard_normal(n_orders)
    eta_hat = np.sum(rate * observed) / np.sum(rate ** 2)          # OLS through the origin
    s2 = np.sum((observed - eta_hat * rate) ** 2) / (n_orders - 1)
    se = np.sqrt(s2 / np.sum(rate ** 2))
    print(f"{n_orders:>6,} metaorders: eta_hat {eta_hat:.3e} (true {eta:.3e}), "
          f"t = {eta_hat / se:5.2f}, 95% CI spans {2 * 1.96 * se / eta:5.1f}x the true value")

typ_rate = 0.275 * adv / 3
print(f"typical order: impact signal ${eta * typ_rate:.4f}/share against "
      f"${sigma * np.sqrt(3):.4f}/share of three-day price noise (ratio {eta * typ_rate / (sigma * np.sqrt(3)):.5f})")
print(f"metaorders needed for t = 2: {(2 * sigma * np.sqrt(3) / (eta * typ_rate)) ** 2:,.0f}")
# =>     50 metaorders: eta_hat -7.576e-08 (true 1.561e-08), t = -0.40, 95% CI spans  47.3x the true value
#       500 metaorders: eta_hat 4.089e-09 (true 1.561e-08), t =  0.05, 95% CI spans  19.8x the true value
#     5,000 metaorders: eta_hat 1.275e-08 (true 1.561e-08), t =  0.60, 95% CI spans   5.3x the true value
#    50,000 metaorders: eta_hat 1.512e-08 (true 1.561e-08), t =  2.24, 95% CI spans   1.7x the true value
#    typical order: impact signal $0.0041/share against $0.5851/share of three-day price noise (ratio 0.00696)
#    metaorders needed for t = 2: 82,551
```

This is the module's uncomfortable result, and it is worth stating without softening. In a simulation where the model is **exactly correct** by construction, fifty metaorders produce an estimate of the *wrong sign* whose confidence interval spans forty-seven times the true value. Five hundred produce a t-statistic of 0.05. Five thousand — more parent orders than most funds execute in a year — still leave a confidence interval five times wider than the quantity being estimated. The last two lines explain why, and the arithmetic is unforgiving: the impact signal is four-tenths of a cent per share against fifty-nine cents of three-day price noise, a ratio of 0.007, so a t-statistic of 2 requires **82,551 metaorders**.

The honest conclusion is that your own fills will not tell you your impact coefficient, and the published estimates practitioners rely on (Almgren et al. 2005 and its successors) come from broker datasets with millions of orders, not from any single fund's history. What survives this is the model's *shape*, which is worth more than its parameters. The trajectory is a hyperbolic sine whatever $\eta$ turns out to be; the frontier curves the same way; the ordering of strategies by urgency is stable. Calibrate with published coefficients, express the answer as a frontier, and treat any point estimate of your own impact with the suspicion the standard errors demand. [The impact module](05-market-impact-models.md) takes up this estimation problem directly — including what happens when the same measurement is attempted on this course's own 1,103 backtest fills.

## The static model's edges

Three assumptions deserve explicit examination. **Linear temporary impact** is the least defensible: the empirical literature overwhelmingly favors a square-root law, $\Delta P \propto \sqrt{Q/V}$, which [Part IV](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) and [Part V](../part-05-backtesting-engine/03-order-management-and-fill-simulation.md) both used and [module 05](05-market-impact-models.md) derives. Replacing $\eta v$ with $\eta v^{1/2}$ destroys the closed form but changes the answer surprisingly little — concave impact penalizes fast trading less at the margin, so optimal schedules flatten slightly toward TWAP, and the frontier shifts without changing shape.

**No drift** is the assumption that the price is a martingale over the execution horizon. If you have a short-term forecast, it belongs in the objective, and the solution acquires a term that trades faster when the forecast is adverse and slower when it is favorable — the natural home of execution alpha, and the first place a real desk extends the model.

**Static commitment** is the sharpest limitation: the entire trajectory is fixed at $t=0$ and never responds to what happens. Adaptive schedules that condition on realized price and liquidity are strictly better in principle, and Almgren–Chriss remains the benchmark they must beat — which is precisely the framing [Part VII's reinforcement-learning lesson](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) used when it pointed forward to [RL for execution](06-rl-for-execution.md). That module takes up the challenge and finds, among other things, that the risk-neutral version of this problem was solved exactly by dynamic programming in 1998, so a learning agent's real opportunity lies in the assumptions listed above rather than in the schedule itself.

!!! warning "The frontier is the deliverable; the point estimate is a rounding error with opinions"
    Almgren–Chriss gives a clean closed form whose *shape* is robust and whose *parameters* are nearly unmeasurable from any single desk's history — 82,551 metaorders for a t-statistic of 2, in a simulation where the model is exactly true. Report execution research as a curve of cost against risk, with published impact coefficients and stated sensitivity, and never as a single number claiming to know what your footprint costs. A model whose form you trust and whose parameters you doubt is an ordinary and workable situation; the failure mode is forgetting which half is which.

!!! abstract "Key takeaways"
    - Implementation shortfall is the only major benchmark with no gaming loophole, because its reference price was fixed when the decision was made — at the cost of variance so large it swamps the effect being measured.
    - Permanent impact contributes $\tfrac12\gamma X^2$ to expected cost regardless of schedule: it is the price of deciding to trade $X$ shares, and no amount of clever slicing avoids it.
    - The mean–variance objective yields the difference equation $(x_{k-1}-2x_k+x_{k+1})/\tau^2 = \tilde\kappa^2 x_k$, solved by $x_k = X\sinh(\kappa(T-t_k))/\sinh(\kappa T)$ with $\kappa = \sqrt{\lambda\sigma^2/\eta}$.
    - TWAP is the $\kappa \to 0$ limit, reproduced numerically to $2.3\times10^{-10}$ shares; at $\kappa T \gg 1$ the trajectory becomes exponential decay with half-life $\ln 2/\kappa$ and the assigned horizon stops mattering.
    - On Part V's GLD order the frontier ran from 2.3 bp expected cost with 82.0 bp of risk to 6.7 bp with 5.1 bp of risk — 3.9 bp of extra cost removing 77 bp of risk, with the bargain concentrated at the patient end.
    - Closed-form moments matched 200,000 simulated executions to within a tenth of a percent on standard deviation across three urgency settings.
    - Part V's +16.3 bp ($145k) decomposes into roughly $34k of the order's own footprint and $111k of drift — a mere **0.23 standard deviations** of a timing distribution with a $480k standard deviation.
    - Calibration on 50 metaorders returned the wrong sign with a CI spanning 47× the true value; 5,000 still spanned 5.3×; reaching t = 2 requires **82,551** metaorders, because the signal is $0.0041/share against $0.5851/share of noise.

## Where this goes next

The impact coefficients this module took as given — and then showed to be nearly unmeasurable from a single desk's fills — are the subject of [Market Impact Models](05-market-impact-models.md), which derives the square-root law, builds Kyle's model of why permanent impact exists at all, and attempts the estimation on this course's own 1,103 backtest fills with a result that the noise-floor arithmetic here predicts. The adaptive extensions listed above are taken up in [Reinforcement Learning for Execution](06-rl-for-execution.md), where the schedule derived here becomes the baseline a learning agent must beat. For the microstructure that makes temporary impact a real phenomenon rather than a modeling convenience, [Part I's order book lesson](../part-01-foundations/03-market-microstructure.md) is the foundation, and [Market Making](12-market-making.md) is the view from the other side of the trade — the liquidity provider whose inventory problem is this module's cost, inverted.
