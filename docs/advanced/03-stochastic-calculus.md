# Stochastic Calculus

Part IV fitted an Ornstein–Uhlenbeck process to the SPY–IVV spread by running an autoregression and converting the coefficient to a half-life, and the lesson was careful to present that as a *mapping* rather than a derivation — $\theta = -\ln\rho/\Delta$, stated and used. [The filtering module](02-particle-and-kalman-filters.md) wrote state equations in discrete time for the same reason: the course has been doing continuous-time modeling in discrete-time clothing for eight parts, because for daily-bar strategy work the clothing is usually enough. This module removes it. Brownian motion, the Itô integral, Itô's lemma, and the two SDEs that appear everywhere in this field get derived properly, and the payoff is not aesthetic — it is that the hedging argument which produces the Black–Scholes equation *only works in continuous time*, and that argument is the foundation of [options pricing](11-options-pricing.md), [optimal execution](04-optimal-execution-almgren-chriss.md), and the inventory control in [market making](12-market-making.md).

The scope is deliberately narrow. There is no measure theory here — no filtrations as $\sigma$-algebras, no martingale representation theorem — and Girsanov and Feynman–Kac are stated where needed and cited rather than proved. What is derived, in full, is the machinery a practitioner computes with: quadratic variation, the Itô integral and its isometry, Itô's lemma, the closed-form solutions of geometric Brownian motion and the OU process, the exact discretization that connects OU to AR(1), and the delta-hedging argument that forces the Black–Scholes PDE. Everything is verified numerically. The uncomfortable result, saved for the last section, concerns the very mapping Part IV used: it turns out the "crude" autoregression was *exactly* right, and the sophisticated-looking Euler discretization that a newcomer to SDEs would write instead is the one that is wrong — by 5.5% on the course's own spread.

## Brownian motion is the only honest limit of a random walk

Standard Brownian motion $W_t$ is defined by four properties: $W_0 = 0$; increments over disjoint intervals are independent; $W_t - W_s \sim \mathcal N(0, t-s)$ for $s < t$; and paths are continuous. The reason this specific object appears everywhere is that it is essentially the *only* possible limit — sum $n$ independent mean-zero shocks with finite variance, scale time by $1/n$ and space by $1/\sqrt n$, and Donsker's theorem says the rescaled walk converges to $W$ regardless of what the individual shocks looked like. Fat-tailed daily returns do not escape this; they merely converge slowly.

The variance-linear-in-time property forces the scaling that makes the whole subject work. Since $W_{ct} \overset{d}{=} \sqrt c\,W_t$, an increment over a small interval $\Delta$ has size on the order of $\sqrt\Delta$, not $\Delta$ — which is already enough to see that Brownian paths cannot be differentiated. The difference quotient over $\Delta$ has standard deviation

$$
\operatorname{sd}\!\left(\frac{W_{t+\Delta} - W_t}{\Delta}\right) \;=\; \frac{\sqrt{\Delta}}{\Delta} \;=\; \frac{1}{\sqrt{\Delta}} \;\longrightarrow\; \infty
\quad\text{as } \Delta \to 0,
$$

so the derivative fails to exist at every point, with probability one. This is not a pathology to be regretted; it is the mathematical content of "prices move unpredictably at every timescale," and it is precisely why an ordinary calculus of $dW$ cannot exist and something else is needed.

## Quadratic variation is where ordinary calculus dies

Take a partition of $[0,T]$ into $n$ equal steps and sum the *squared* increments. Each $(\Delta W_i)^2$ has mean $\Delta = T/n$, so the sum has mean $T$ exactly, for every $n$. The variance is where the magic happens: for a Gaussian, $\operatorname{Var}[(\Delta W_i)^2] = 2\Delta^2$, and the increments are independent, so

$$
\mathbb{E}\left[\sum_{i=1}^{n} (\Delta W_i)^2\right] = T,
\qquad
\operatorname{Var}\left[\sum_{i=1}^{n} (\Delta W_i)^2\right] = 2n\Delta^2 = \frac{2T^2}{n} \;\longrightarrow\; 0 .
$$

A random variable whose mean is $T$ and whose variance vanishes is the constant $T$. So the quadratic variation $[W]_T = T$ — **the sum of squared increments of a Brownian path is deterministic**, even though every individual increment is random. That single fact is the engine of the entire subject, and it is what the shorthand $(dW)^2 = dt$ abbreviates.

Contrast a differentiable path. If $f$ is $C^1$ then $\Delta f_i \approx f'\Delta$, so squared increments are $O(\Delta^2)$ and their sum is $O(n\Delta^2) = O(T^2/n) \to 0$. Ordinary functions have zero quadratic variation; Brownian motion does not, and that is the whole difference. Meanwhile the *first* variation $\sum\lvert\Delta W_i\rvert$, which for a $C^1$ function converges to the finite arc length, diverges like $\sqrt n$ for Brownian motion — which is why the Riemann–Stieltjes integral $\int H\,dW$ cannot be defined pathwise in the classical way, and why the Itô construction is needed at all:

```python
import numpy as np

T = 1.0
for n in [100, 1000, 10_000, 100_000, 1_000_000]:
    dt = T / n
    dW = np.sqrt(dt) * np.random.default_rng(0).standard_normal(n)
    smooth = np.linspace(0, 1, n + 1) ** 2                 # a C^1 path, for contrast
    print(f"n = {n:>9,}: QV(W) {np.sum(dW ** 2):.4f}, sum|dW| {np.sum(np.abs(dW)):9.2f}, "
          f"QV(smooth) {np.sum(np.diff(smooth) ** 2):.2e}")
# => n =       100: QV(W) 0.9323, sum|dW|      7.96, QV(smooth) 1.33e-02
#    n =     1,000: QV(W) 0.9564, sum|dW|     24.58, QV(smooth) 1.33e-03
#    n =    10,000: QV(W) 0.9962, sum|dW|     79.96, QV(smooth) 1.33e-04
#    n =   100,000: QV(W) 1.0003, sum|dW|    252.34, QV(smooth) 1.33e-05
#    n = 1,000,000: QV(W) 1.0013, sum|dW|    798.42, QV(smooth) 1.33e-06
```

Three columns, three theorems. Quadratic variation converges to $T = 1$ and stays there. Total variation grows without bound — 7.96, then 79.96, then 798.42, a factor of ten for each hundredfold refinement, exactly the $\sqrt n$ the theory predicts. And the smooth path's quadratic variation falls to zero like $1/n$. A Brownian path has infinite length and finite squared length; ordinary functions have the reverse.

## The Itô integral is a trading strategy, not an area

To define $\int_0^T H_t\,dW_t$, approximate by sums $\sum_i H_{\tau_i}(W_{t_{i+1}} - W_{t_i})$ and let the mesh vanish. For a Riemann–Stieltjes integral the evaluation point $\tau_i \in [t_i, t_{i+1}]$ does not matter. Here it decides the answer, because the integrand and the increment are correlated, and the correlation does not wash out — a direct consequence of the previous section's non-vanishing quadratic variation.

Itô's choice is the **left endpoint**, $\tau_i = t_i$, and for this course the justification needs no mathematics. Read $H_t$ as the position held in an asset and $dW_t$ as the price increment. Then $H_{t_i}(W_{t_{i+1}} - W_{t_i})$ is the P&L of a position *chosen before* the move it profits from. Any other evaluation point — the midpoint, the right endpoint — lets the position depend on the very increment it is about to earn, which is not a modeling convention but a lookahead bug. The Itô integral is the mathematics of non-anticipating strategies, and every other choice describes a strategy that cannot be traded.

The left-endpoint choice buys a second property: since $H_{t_i}$ is known before the mean-zero increment arrives, each term has zero conditional expectation, so the integral is a martingale — a strategy that trades a fair game is itself a fair game. And it makes the integral's variance computable. Squaring the sum, cross terms vanish (independent increments, non-anticipating integrand), and each square contributes $\mathbb{E}[H_{t_i}^2]\Delta$, giving the **Itô isometry**:

$$
\mathbb{E}\left[\left(\int_0^T H_t\,dW_t\right)^{\!2}\right] \;=\; \mathbb{E}\left[\int_0^T H_t^2\,dt\right].
$$

The cleanest demonstration that the evaluation point matters is $\int_0^T W\,dW$, which ordinary calculus would call $\tfrac12 W_T^2$. Telescoping the identity $W_{i+1}^2 - W_i^2 = 2W_i(W_{i+1}-W_i) + (W_{i+1}-W_i)^2$ over the partition gives $W_T^2 = 2\sum W_i\Delta W_i + \sum(\Delta W_i)^2$, and the last sum is the quadratic variation $T$. Therefore

$$
\int_0^T W_t\,dW_t \;=\; \tfrac12 W_T^2 \;-\; \tfrac12 T .
$$

The correction term $-T/2$ is quadratic variation making its first appearance in an answer. Take the midpoint instead and the same telescoping leaves the correction out, recovering the classical $\tfrac12 W_T^2$ — that is the Stratonovich integral, which obeys ordinary calculus and describes a strategy that peeks:

```python
import numpy as np

T = 1.0
for n in [1_000, 100_000]:
    dt = T / n
    dW = np.sqrt(dt) * np.random.default_rng(1).standard_normal(n)
    W = np.concatenate([[0], np.cumsum(dW)])
    ito = np.sum(W[:-1] * dW)                              # position set before the move
    strat = np.sum(0.5 * (W[:-1] + W[1:]) * dW)            # position peeks at the move
    print(f"n = {n:>7,}: Ito {ito:+.4f}, closed form W_T^2/2 - T/2 "
          f"{0.5 * W[-1] ** 2 - 0.5 * T:+.4f}, Stratonovich {strat:+.4f}, "
          f"gap {strat - ito:+.4f} (T/2 = {T / 2:.4f})")
# => n =   1,000: Ito +0.9839, closed form W_T^2/2 - T/2 +0.9717, Stratonovich +1.4717, gap +0.4878 (T/2 = 0.5000)
#    n = 100,000: Ito +0.5571, closed form W_T^2/2 - T/2 +0.5537, Stratonovich +1.0537, gap +0.4965 (T/2 = 0.5000)
```

The two rows use different random paths, so their integrals differ — that is expected, since $\int W\,dW$ is a random variable. What is *not* random is the relationship between the columns: the left-endpoint sum tracks the closed form to within discretization error in both rows, and the midpoint sum exceeds it by 0.4878 and then 0.4965, converging to $T/2 = 0.5$. The gap between trading on information you have and information you do not is a deterministic half-unit of time, and in this course's terms it is the difference between a backtest and a fantasy.

## Itô's lemma is the chain rule paying a volatility tax

Everything above is preparation for the one tool practitioners actually use. Let $X$ satisfy $dX_t = a\,dt + b\,dW_t$ and let $f(t, x)$ be smooth. Ordinary calculus would Taylor-expand to first order. Here second order cannot be dropped, because $(dX)^2$ contains $b^2 (dW)^2 = b^2\,dt$, which is first order in time. Expanding and applying the multiplication rules $dt^2 = 0$, $dt\,dW = 0$, $(dW)^2 = dt$ gives **Itô's lemma**:

$$
df \;=\; \left(\frac{\partial f}{\partial t} + a\,\frac{\partial f}{\partial x} + \tfrac12 b^2\,\frac{\partial^2 f}{\partial x^2}\right)dt \;+\; b\,\frac{\partial f}{\partial x}\,dW_t .
$$

The extra term $\tfrac12 b^2 f_{xx}$ is the volatility tax, and its sign is decided by convexity: convex functions of a diffusion drift upward relative to the function of the drift, concave ones drift downward. Every result in the rest of this module, and most of [options pricing](11-options-pricing.md), is an application of this one equation.

**Geometric Brownian motion** is the first application. Model prices as $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ — proportional returns, so the model cannot go negative — and apply the lemma to $f = \ln S$, where $f_x = 1/S$ and $f_{xx} = -1/S^2$:

$$
d(\ln S_t) \;=\; \left(\mu - \tfrac{\sigma^2}{2}\right)dt \;+\; \sigma\,dW_t,
\qquad\text{so}\qquad
S_T \;=\; S_0\exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)T + \sigma W_T\right].
$$

The $-\sigma^2/2$ is the most consequential correction in applied finance, and this course has met it twice already without deriving it. It is why [Part III](../part-03-statistics/01-probability-and-random-variables.md) insisted log returns and simple returns are different animals, and it is the volatility drag that makes [Part VIII's Kelly and volatility-targeting arithmetic](../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) work the way it does: the *median* outcome of a levered position grows at $\mu - \sigma^2/2$ while the *mean* grows at $\mu$, and leverage multiplies $\sigma^2$ faster than it multiplies $\mu$. Itô's lemma is where that asymmetry comes from.

## The Ornstein–Uhlenbeck process is Part IV's AR(1) in continuous clothing

The second canonical SDE adds a restoring force:

$$
dX_t \;=\; \theta\,(\mu - X_t)\,dt \;+\; \sigma\,dW_t ,
$$

with $\theta > 0$ the speed of mean reversion. Solve it by the integrating factor $e^{\theta t}$: applying Itô's lemma to $Y_t = e^{\theta t}X_t$ kills the drift, since $dY = \theta e^{\theta t}X\,dt + e^{\theta t}dX = \theta\mu e^{\theta t}dt + \sigma e^{\theta t}dW$ (the second-derivative term vanishes because $Y$ is linear in $X$). Integrating from $0$ to $t$ and multiplying back by $e^{-\theta t}$,

$$
X_t \;=\; \mu + (X_0 - \mu)\,e^{-\theta t} \;+\; \sigma\!\int_0^t e^{-\theta(t-s)}\,dW_s .
$$

The three terms are readable directly: the equilibrium level, an initial displacement decaying exponentially, and an integral of past shocks weighted by how recently they arrived. The isometry gives the variance of that integral, $\sigma^2(1-e^{-2\theta t})/2\theta$, so as $t \to \infty$ the process settles into a stationary distribution with

$$
\operatorname{Var}[X_\infty] \;=\; \frac{\sigma^2}{2\theta},
\qquad
t_{1/2} \;=\; \frac{\ln 2}{\theta} .
$$

Now the payoff. Evaluate the solution over one sampling interval $\Delta$ rather than from zero, and the result is *exact* — no approximation anywhere:

$$
X_{t+\Delta} \;=\; \mu + (X_t - \mu)\,e^{-\theta\Delta} + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal N\!\left(0,\; \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta\Delta}\bigr)\right).
$$

That is an AR(1) with $\phi = e^{-\theta\Delta}$ and Gaussian innovations. So the discrete autoregression Part IV ran was not an approximation to the OU process — **it is the OU process, sampled**, and the mapping $\theta = -\ln\rho/\Delta$ that the lesson stated is exact rather than asymptotic. Verify the round trip, then run it on the course's own data:

```python
import numpy as np
import pandas as pd

dt = 1 / 252
theta_true, sigma_ou = 51.0, 0.35
phi = np.exp(-theta_true * dt)
sd_eps = np.sqrt(sigma_ou ** 2 / (2 * theta_true) * (1 - np.exp(-2 * theta_true * dt)))

rng = np.random.default_rng(3)
x = np.zeros(6000)
for t in range(1, len(x)):
    x[t] = x[t - 1] * phi + sd_eps * rng.standard_normal()

rho = pd.Series(x).autocorr(1)
theta_hat = -np.log(rho) / dt
print(f"simulated: true theta {theta_true:.1f}/yr (half-life "
      f"{np.log(2) / theta_true * 252:.2f} d) -> fitted rho {rho:.4f}, "
      f"theta {theta_hat:.1f}/yr (half-life {np.log(2) / theta_hat * 252:.2f} d)")
print(f"stationary sd: theory {np.sqrt(sigma_ou ** 2 / (2 * theta_true)) * 1e4:.1f} bp, "
      f"sample {pd.Series(x).std() * 1e4:.1f} bp")

px = pd.read_parquet("data/prices.parquet")
for a, b in [("SPY", "IVV"), ("TLT", "GLD")]:
    s = (np.log(px[a]) - np.log(px[b])).dropna()
    r = s.autocorr(1)
    th = -np.log(r) / dt
    print(f"{a}-{b}: rho {r:.4f} -> theta {th:6.2f}/yr, half-life "
          f"{np.log(2) / th * 252:6.1f} days, equilibrium sd {s.std() * 1e4:.1f} bp")
# => simulated: true theta 51.0/yr (half-life 3.42 d) -> fitted rho 0.8207, theta 49.8/yr (half-life 3.51 d)
#    stationary sd: theory 346.6 bp, sample 351.6 bp
#    SPY-IVV: rho 0.8166 -> theta  51.05/yr, half-life    3.4 days, equilibrium sd 23.3 bp
#    TLT-GLD: rho 0.9989 -> theta   0.28/yr, half-life  631.0 days, equilibrium sd 2834.6 bp
```

The round trip recovers 49.8 against a true 51.0 — the 2% shortfall is the small-sample downward bias of autoregressive coefficient estimates, worth knowing about and not worth correcting here. The third line is the reconciliation this module owes Part IV: **θ = 51.05 per year, half-life 3.4 days, equilibrium standard deviation 23.3 bp**, reproducing that lesson's published 51/yr, 3.4 days, and 23 bp from an independent derivation. And the fourth line reproduces its warning: TLT–GLD's $\rho = 0.9989$ maps to a 631-day half-life and an "equilibrium" standard deviation of 2,834 bp — a fitted resting level that the spread wanders 28% away from, which is not mean reversion by any useful definition. This is a unit root in costume: as $\theta \to 0$ the OU process degenerates into Brownian motion and its stationary variance $\sigma^2/2\theta$ diverges, yet the formulas keep returning finite numbers for a process that has no equilibrium at all. The arithmetic stays well-defined long after the model stops applying, which is exactly why Part IV insisted on testing stationarity before fitting.

## Delta hedging forces the Black–Scholes equation

Here is what continuous time buys that discrete time cannot. Let $S$ follow geometric Brownian motion and let $V(t, S)$ be a derivative's value. Form a portfolio long one derivative and short $\Delta$ units of stock, $\Pi = V - \Delta S$. Over $dt$, using Itô's lemma on $V$,

$$
d\Pi \;=\; \left(\frac{\partial V}{\partial t} + \tfrac12\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + \mu S\frac{\partial V}{\partial S} - \Delta\,\mu S\right)dt
\;+\; \sigma S\left(\frac{\partial V}{\partial S} - \Delta\right)dW .
$$

Choose $\Delta = \partial V/\partial S$. The $dW$ term vanishes identically, and so does $\mu$ — the portfolio's value over the next instant is *deterministic*, regardless of which way the stock moves and regardless of its expected return. A riskless portfolio must earn the riskless rate or an arbitrage exists, so setting $d\Pi = r\Pi\,dt$ and substituting gives the **Black–Scholes PDE**:

$$
\frac{\partial V}{\partial t} + \tfrac12\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV \;=\; 0 .
$$

Two features deserve emphasis. First, $\mu$ is absent: the expected return of the underlying does not appear in the price of its derivative, because hedging removes the exposure that would have made it matter. Second, and this is the point of the whole module — **the argument requires continuous rebalancing**. Hedge discretely and the $dW$ term does not vanish, it merely becomes small, leaving a residual that [the options module](11-options-pricing.md) quantifies exactly (it is proportional to gamma and to realized-minus-implied variance). Continuous time is not a convenient idealization here; it is the only regime in which the replication is exact, and every practical options desk is managing the error term that discreteness reintroduces.

The PDE has a probabilistic twin. Feynman–Kac says a solution can be written as a discounted expectation, and Girsanov's theorem identifies the measure: under the risk-neutral measure $\mathbb{Q}$, the drift of $S$ is replaced by $r$, and

$$
V(0, S_0) \;=\; e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\text{payoff}(S_T)\right].
$$

This is not a statement that investors are risk-neutral; it is a change of probability measure that reprices the drift while leaving volatility — and therefore the hedging argument — untouched. It is also immediately computable, which makes it the easiest check that everything above is consistent. The closed-form solution of this expectation is [module 11's](11-options-pricing.md) business; here, simulate it:

```python
import numpy as np
from scipy import stats

S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.20, 1.0
d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
bs = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d1 - sigma * np.sqrt(T))

M = 2_000_000
Z = np.random.default_rng(5).standard_normal(M)
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)   # Q-measure drift is r
pay = np.exp(-r * T) * np.maximum(ST - K, 0)
se = pay.std() / np.sqrt(M)
print(f"Black-Scholes formula {bs:.4f}; risk-neutral Monte Carlo {pay.mean():.4f} "
      f"+/- {1.96 * se:.4f} (95%)")
print(f"difference {pay.mean() - bs:+.4f}, which is {(pay.mean() - bs) / se:+.2f} standard errors")
# => Black-Scholes formula 7.1281; risk-neutral Monte Carlo 7.1341 +/- 0.0174 (95%)
#    difference +0.0060, which is +0.68 standard errors
```

Two million paths simulated under a drift of $r$ — not $\mu$, which never enters — price the option at 7.1341 against the analytic 7.1281, a discrepancy of 0.68 standard errors. The PDE and the expectation are the same object seen from two directions. Note what two million paths bought: an interval of ±0.0174, because the error falls like one over the square root of the path count and nothing faster ([Monte Carlo Simulation](../appendix/part-09-monte-carlo-methods/03-monte-carlo-simulation.md) runs this same option across seven sample sizes to show it, and [Variance Reduction](../appendix/part-09-monte-carlo-methods/06-variance-reduction.md) buys back a factor of 1,345 on a related contract by regressing against a payoff whose answer is known).

## Simulating an SDE means choosing your error

Most SDEs have no closed-form solution and must be discretized. The **Euler–Maruyama** scheme is the obvious translation, $X_{n+1} = X_n + a(X_n)\Delta + b(X_n)\sqrt{\Delta}\,Z_n$, and it converges — but at a rate that surprises people who know ordinary ODE solvers. The relevant notion here is *strong* convergence, which measures pathwise accuracy $\mathbb{E}\lvert X_N - X_T\rvert \sim C\Delta^{\gamma}$ (as opposed to *weak* convergence, which only requires distributions to match and is what matters for pricing an expectation). Euler–Maruyama is strong order $\gamma = 1/2$: halving the step buys only a factor of $\sqrt 2$ in pathwise accuracy, because the leading error term involves $\int b'b\,dW\,dW$, which the scheme ignores. Adding exactly that term via Itô's lemma gives the **Milstein** scheme,

$$
X_{n+1} \;=\; X_n + a\,\Delta + b\,\sqrt{\Delta}\,Z_n + \tfrac12\,b\,b'\left(\Delta W_n^2 - \Delta\right),
$$

which is strong order 1. Note what the correction is: $(\Delta W^2 - \Delta)$ is the deviation of the realized squared increment from its expectation — quadratic variation again, now as a numerical correction. Measure both against GBM's exact solution on identical Brownian paths:

```python
import numpy as np

mu, sig, S0, T, M = 0.05, 0.20, 100.0, 1.0, 2000
ns, em_err, mil_err = [8, 32, 128, 512, 2048], [], []
for n in ns:
    dt = T / n
    dW = np.sqrt(dt) * np.random.default_rng(7).standard_normal((M, n))
    exact = S0 * np.exp((mu - 0.5 * sig ** 2) * T + sig * dW.sum(axis=1))
    em = np.full(M, S0)
    mil = np.full(M, S0)
    for i in range(n):
        em = em + mu * em * dt + sig * em * dW[:, i]
        mil = (mil + mu * mil * dt + sig * mil * dW[:, i]
               + 0.5 * sig ** 2 * mil * (dW[:, i] ** 2 - dt))
    em_err.append(np.mean(np.abs(em - exact)))
    mil_err.append(np.mean(np.abs(mil - exact)))
    print(f"n = {n:>5}: Euler-Maruyama {em_err[-1]:.5f}, Milstein {mil_err[-1]:.5f}")
print(f"log-log slopes: Euler-Maruyama {np.polyfit(np.log(ns), np.log(em_err), 1)[0]:.3f}, "
      f"Milstein {np.polyfit(np.log(ns), np.log(mil_err), 1)[0]:.3f}")
# => n =     8: Euler-Maruyama 0.82627, Milstein 0.10910
#    n =    32: Euler-Maruyama 0.40728, Milstein 0.02728
#    n =   128: Euler-Maruyama 0.21046, Milstein 0.00701
#    n =   512: Euler-Maruyama 0.10596, Milstein 0.00172
#    n =  2048: Euler-Maruyama 0.05114, Milstein 0.00041
#    log-log slopes: Euler-Maruyama -0.499, Milstein -1.005
```

The measured slopes are −0.499 and −1.005 against theoretical −0.5 and −1.0, to three decimal places. In absolute terms, Milstein at eight steps (0.109) beats Euler–Maruyama at 2,048 steps (0.051) — no, it does not quite, but Milstein at *32* steps (0.027) beats Euler at 2,048, which is a 64-fold reduction in work for better accuracy, bought by one extra term.

## Where continuous time earns its complexity, and where it does not

The honest scoping question is whether any of this changes a daily-bar strategy result, and the answer is mostly no — the core course covered nine parts without it. But "mostly" hides a specific, checkable exception, and it lands on the one piece of continuous-time machinery the core course actually used. Compare two ways of simulating the same OU process at a daily step: the exact discretization derived earlier, and the Euler–Maruyama scheme that a fresh reader of the SDE would naturally write.

```python
import numpy as np

dt, sigma_ou = 1 / 252, 0.35
for theta in [51.0, 5.0]:
    phi = np.exp(-theta * dt)
    sd_eps = np.sqrt(sigma_ou ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * dt)))
    z = np.random.default_rng(11).standard_normal(200_000)
    xe = np.zeros(len(z))                                  # exact discretization
    xu = np.zeros(len(z))                                  # Euler-Maruyama
    for t in range(1, len(z)):
        xe[t] = xe[t - 1] * phi + sd_eps * z[t]
        xu[t] = xu[t - 1] + theta * (0 - xu[t - 1]) * dt + sigma_ou * np.sqrt(dt) * z[t]
    print(f"theta {theta:5.1f}/yr (theta*dt = {theta * dt:.3f}): exact sd {xe.std() * 1e4:7.2f} bp, "
          f"Euler sd {xu.std() * 1e4:7.2f} bp, relative gap {abs(xu.std() / xe.std() - 1):.2%}")
# => theta  51.0/yr (theta*dt = 0.202): exact sd  344.47 bp, Euler sd  363.51 bp, relative gap 5.53%
#    theta   5.0/yr (theta*dt = 0.020): exact sd 1093.79 bp, Euler sd 1099.21 bp, relative gap 0.50%
```

For a slowly-reverting spread the two agree to half a percent and the choice is irrelevant. For the SPY–IVV spread's actual speed — $\theta\Delta = 0.202$, because a 3.4-day half-life sampled daily is *not* a fine discretization — Euler–Maruyama overstates the stationary standard deviation by **5.5%**, which propagates directly into every z-score threshold built on it. The resolution is the sentence this module has been building toward: Part IV's autoregression was not a shortcut that a continuous-time treatment would improve on. It *was* the exact solution, because the AR(1) mapping is the exact discretization, and the more sophisticated-looking Euler scheme is the one that introduces error. The general rule is that discretization error scales with $\theta\Delta$, so fast processes on coarse grids are exactly where naive schemes fail — and when an exact discretization exists, as it does for OU and GBM, use it and the question never arises.

That is the honest verdict on this module's relevance to the systematic-equities track: it changed no strategy result in Parts III through VIII, and it explained two of them ($-\sigma^2/2$ as volatility drag, AR(1) as sampled OU) while catching one error a newcomer would have introduced. Its real payoff is elsewhere. The PDE derived above cannot be reached in discrete time at all, and it is the entry point to [options pricing](11-options-pricing.md); the same Itô machinery drives the cost-versus-risk optimization in [optimal execution](04-optimal-execution-almgren-chriss.md) and the inventory control in [market making](12-market-making.md). Continuous time earns its complexity when you need a *hedging* argument or a *control* argument — not when you need a signal.

!!! warning "The exact discretization is free, and the sophisticated-looking one is wrong"
    Euler–Maruyama is what an SDE looks like when transcribed literally, and on the course's own SPY–IVV spread it inflates the equilibrium standard deviation by 5.5% — because a 3.4-day half-life sampled once a day gives $\theta\Delta = 0.202$, nowhere near the small-step regime the scheme assumes. Ornstein–Uhlenbeck and geometric Brownian motion both have exact discretizations that cost the same arithmetic. Before simulating any SDE, check whether its solution is known in closed form; discretization error is a choice, not a fact of life.

!!! abstract "Key takeaways"
    - Quadratic variation converged to $T = 1$ (0.9323 → 1.0013 across four refinements) while total variation diverged like $\sqrt n$ (7.96 → 798.42) and a smooth path's quadratic variation fell like $1/n$ — Brownian paths have infinite length and finite squared length.
    - The left-endpoint convention is the non-anticipation condition: the position is chosen before the increment it earns. The midpoint (Stratonovich) sum exceeded the Itô sum by 0.4878 and 0.4965 across two paths, converging to the deterministic $T/2 = 0.5$.
    - Itô's lemma adds $\tfrac12 b^2 f_{xx}$ to the chain rule, and applying it to $\ln S$ produces the $-\sigma^2/2$ that Part III called the log-versus-simple-return gap and Part VIII called volatility drag.
    - The OU solution gives an *exact* AR(1) with $\phi = e^{-\theta\Delta}$, so Part IV's autoregression was the sampled process rather than an approximation of it; the round trip recovered $\theta = 49.8$ from a true 51.0.
    - Fitting the real SPY–IVV spread reproduced Part IV independently: $\rho = 0.8166$, $\theta = 51.06$/yr, half-life **3.4 days**, equilibrium sd **23.3 bp** — and TLT–GLD's 631-day half-life remains a unit root in costume, since $\sigma^2/2\theta$ diverges as $\theta \to 0$.
    - Choosing $\Delta = \partial V/\partial S$ annihilates both the $dW$ term and $\mu$, forcing the Black–Scholes PDE — an argument that is exact only under continuous rebalancing, which is why discrete hedging leaves a gamma-proportional residual.
    - Risk-neutral Monte Carlo (drift $r$, two million paths) priced a call at 7.1341 against the analytic 7.1281 — a 0.68-standard-error agreement between the PDE and the expectation.
    - Measured strong-convergence slopes were −0.499 for Euler–Maruyama and −1.005 for Milstein against theoretical −0.5 and −1, and the exact-versus-Euler gap for OU was 5.5% at $\theta\Delta = 0.202$ against 0.5% at $\theta\Delta = 0.020$.

## Where this goes next

The Black–Scholes PDE derived here is solved in [Options Pricing](11-options-pricing.md), which carries the risk-neutral expectation through to the closed form, differentiates it into the Greeks, and quantifies precisely what discrete hedging costs — the error term this module's continuous-rebalancing assumption swept aside. The same Itô toolkit reappears as a control problem in [Optimal Execution](04-optimal-execution-almgren-chriss.md), where the trade-off is between market impact and the timing risk that $\sigma\sqrt{T}$ measures, and again in [Market Making](12-market-making.md), whose reservation price is a certainty-equivalent computed against exactly the diffusion written down here. Back in the core course, the OU process fitted in this module is the engine of [Part IV's pairs trade](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md), and the volatility drag it derives is the reason [Part VIII's leverage arithmetic](../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) punishes size the way it does.
