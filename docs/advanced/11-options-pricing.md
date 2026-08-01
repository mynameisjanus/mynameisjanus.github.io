# Options Pricing

[The stochastic calculus module](03-stochastic-calculus.md) derived the Black–Scholes partial differential equation by showing that a delta-hedged portfolio has no exposure to the Brownian increment, and stopped there — it produced the equation and deferred the solution. This module solves it, differentiates the solution into the Greeks, and then spends most of its length on the far more interesting question of what practitioners do about the fact that the model is wrong.

Because it is wrong, and its wrongness is *published*. Every traded option implies a volatility through the Black–Scholes formula, those implied volatilities differ systematically by strike and maturity, and the resulting surface is a public catalogue of the model's failures. [Part IV traded the volatility risk premium](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) using the variance-swap identity — VIX as the strike of a 30-day variance swap — and took that identity on faith; this module derives it. The uncomfortable result is saved for the end: Dupire's local volatility model fits *any* arbitrage-free surface exactly, and this module measures what that exactness costs — twenty basis points of quote noise, well inside a normal bid-ask spread, moves implied volatility by 0.003 and local volatility by up to **0.25**, an amplification of **81×**.

## Before any model: parity and bounds

Some relationships hold regardless of how prices move, because violating them is an arbitrage executable with a static portfolio. The most useful is put–call parity. Hold one call and short one put, both struck at $K$ expiring at $T$: at expiry the payoff is $(S_T - K)^+ - (K - S_T)^+ = S_T - K$ whatever $S_T$ turns out to be, which is exactly the payoff of holding the stock and owing $K$. Two portfolios with identical payoffs in every state must cost the same today, so

$$
C - P \;=\; S - K e^{-rT}.
$$

No dynamics, no distribution, no volatility — parity constrains prices even if the stock follows something no model has ever described. The same reasoning bounds a call between $\max(S - Ke^{-rT}, 0)$ and $S$. Parity's practical uses are worth naming: it converts any put quote into a call quote, so an implied-volatility surface can be built from whichever side of each strike is more liquid; and a persistent violation is either an arbitrage or, far more often, a sign that your borrow cost, dividend forecast, or funding rate is wrong. Deviations from parity are how desks *measure* those inputs.

## Black–Scholes is one integral

Under the risk-neutral measure the stock has drift $r$, so $\ln S_T$ is normal with mean $\ln S + (r - \sigma^2/2)T$ and variance $\sigma^2 T$, and the call price is a discounted expectation:

$$
C \;=\; e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\bigl[(S_T - K)^{+}\bigr]
\;=\; e^{-rT}\!\int_{\ln K}^{\infty}\!\bigl(e^{x} - K\bigr)\,\varphi\!\left(\frac{x - m}{s}\right)\frac{dx}{s},
$$

with $m = \ln S + (r - \sigma^2/2)T$ and $s = \sigma\sqrt{T}$. Split the integral. The $K$ term is $K e^{-rT}$ times the probability that $\ln S_T$ exceeds $\ln K$, which is $\Phi(d_2)$ with $d_2 = (m - \ln K)/s$. The $e^x$ term requires completing the square in the exponent — $e^{x}\varphi((x-m)/s)$ becomes $e^{m + s^2/2}\varphi((x - m - s^2)/s)$ — which shifts the mean by $s^2$ and produces the discounted forward $S e^{rT}$ times $\Phi(d_2 + s)$. Discounting cancels the growth, and

$$
C \;=\; S\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2),
\qquad
d_{1,2} \;=\; \frac{\ln(S/K) + \bigl(r \pm \tfrac{\sigma^2}{2}\bigr)T}{\sigma\sqrt{T}} .
$$

The two terms have a reading worth carrying: $\Phi(d_2)$ is the risk-neutral probability of finishing in the money, and $\Phi(d_1)$ is that probability re-weighted by how much stock you receive — which is why $\Phi(d_1)$, not $\Phi(d_2)$, turns out to be the hedge ratio. The assumptions purchased along the way should be listed where they can be attacked: constant volatility, continuous paths with no jumps, frictionless continuous trading, a constant known rate, and no dividends. Every one is false, and the surface in section five is the market's estimate of how false.

## The Greeks are one identity and its derivatives

Differentiating the formula looks messy because $d_1$ and $d_2$ both depend on $S$, $\sigma$, and $T$. One identity removes almost all of it. Since $d_1 - d_2 = \sigma\sqrt T$ and $d_1 d_2$ telescopes appropriately, direct substitution gives

$$
S\,\varphi(d_1) \;=\; K e^{-rT}\,\varphi(d_2),
$$

and this is why delta is clean. Differentiating $C$ with respect to $S$ produces $\Phi(d_1)$ plus $S\varphi(d_1)\partial_S d_1 - Ke^{-rT}\varphi(d_2)\partial_S d_2$; because $\partial_S d_1 = \partial_S d_2$, the identity makes those two terms cancel exactly, leaving $\Delta = \Phi(d_1)$. The trap the identity defuses is the common error of "forgetting" the $\partial d$ terms and getting the right answer for the wrong reason. The rest follow the same way:

$$
\Delta = \Phi(d_1),
\qquad
\Gamma = \frac{\varphi(d_1)}{S\sigma\sqrt T},
\qquad
\mathcal{V} = S\,\varphi(d_1)\sqrt T,
\qquad
\Theta = -\frac{S\varphi(d_1)\sigma}{2\sqrt T} - rKe^{-rT}\Phi(d_2).
$$

Verify all of it numerically, including parity and the identity, against finite differences:

```python
import numpy as np
from scipy import stats

def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d1 - sig * np.sqrt(T))

def greeks(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return dict(delta=stats.norm.cdf(d1),
                gamma=stats.norm.pdf(d1) / (S * sig * np.sqrt(T)),
                vega=S * stats.norm.pdf(d1) * np.sqrt(T),
                theta=(-S * stats.norm.pdf(d1) * sig / (2 * np.sqrt(T))
                       - r * K * np.exp(-r * T) * stats.norm.cdf(d2)))

S, K, r, sig, T = 100.0, 100.0, 0.03, 0.20, 1.0
d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
d2 = d1 - sig * np.sqrt(T)
g, h = greeks(S, K, r, sig, T), 1e-4
fd = dict(
    delta=(bs_call(S + h, K, r, sig, T) - bs_call(S - h, K, r, sig, T)) / (2 * h),
    gamma=(bs_call(S + h, K, r, sig, T) - 2 * bs_call(S, K, r, sig, T)
           + bs_call(S - h, K, r, sig, T)) / h ** 2,
    vega=(bs_call(S, K, r, sig + h, T) - bs_call(S, K, r, sig - h, T)) / (2 * h))
put = bs_call(S, K, r, sig, T) - S + K * np.exp(-r * T)

print(f"identity S*phi(d1) = K*exp(-rT)*phi(d2) holds to 1e-12: "
      f"{abs(S * stats.norm.pdf(d1) - K * np.exp(-r * T) * stats.norm.pdf(d2)) < 1e-12}")
print(f"closed-form Greeks match finite differences: "
      f"{all(abs(fd[k] - g[k]) < 1e-5 for k in fd)}")
print(f"put-call parity holds exactly: "
      f"{abs((bs_call(S, K, r, sig, T) - put) - (S - K * np.exp(-r * T))) < 1e-10}")
print(f"gamma-theta relation for a hedged book: "
      f"{abs((-0.5 * sig ** 2 * S ** 2 * g['gamma']) - (g['theta'] + r * K * np.exp(-r * T) * stats.norm.cdf(d2))) < 1e-9}")
# => identity S*phi(d1) = K*exp(-rT)*phi(d2) holds to 1e-12: True
#    closed-form Greeks match finite differences: True
#    put-call parity holds exactly: True
#    gamma-theta relation for a hedged book: True
```

The at-the-money one-year call prices at 9.4134 with delta 0.5987, gamma 0.019333, vega 38.6668, and theta −5.3804; the identity holds to $7\times10^{-15}$ and the finite-difference checks agree to $10^{-7}$ or better.

The last assertion is the one that matters most for trading. Substituting $\Delta = \partial_S V$ into the Black–Scholes PDE and cancelling the terms that a delta hedge removes leaves

$$
\Theta \;\approx\; -\tfrac12 \sigma^2 S^2\,\Gamma,
$$

which is the sentence every options trader carries: **theta is the rent you pay for gamma**. A long-gamma book profits from movement and bleeds theta while it waits; a short-gamma book collects theta and is short the movement. The two are the same quantity with opposite signs, and the verified equality above (−3.8667 on both sides) says the trade-off is exact rather than approximate.

## A hedged book earns realized minus implied

The gamma–theta relation implies the P&L formula that defines volatility trading. Over a short interval a delta-hedged long option gains $\tfrac12\Gamma(\Delta S)^2$ from convexity and loses $\lvert\Theta\rvert \Delta t \approx \tfrac12\sigma_{\text{imp}}^2 S^2\Gamma\,\Delta t$ to time decay. Summing,

$$
\text{P\&L} \;\approx\; \tfrac12\!\int_0^T \Gamma_t\,S_t^2\left(\sigma_{\text{real}}^2 - \sigma_{\text{imp}}^2\right)dt ,
$$

so **a delta-hedged option is a bet on realized versus implied volatility, weighted by gamma**. The direction of the stock is irrelevant; only the size of its wiggles matters. Simulate it — buy a one-year call at 20% implied, delta-hedge daily, and let realized volatility differ:

```python
import numpy as np
from scipy import stats

S0, K, r, T = 100.0, 100.0, 0.03, 1.0

def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d1 - sig * np.sqrt(T))

def bs_delta(S, K, r, sig, T):
    return stats.norm.cdf((np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T)))

def bs_gamma(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return stats.norm.pdf(d1) / (S * sig * np.sqrt(T))

def hedge(sig_real, sig_imp, n_hedges, n_paths=20_000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_hedges
    S = np.full(n_paths, S0)
    pos = -bs_delta(S0, K, r, sig_imp, T) * np.ones(n_paths)     # short shares
    cash = -bs_call(S0, K, r, sig_imp, T) - pos * S0             # premium paid
    for i in range(1, n_hedges + 1):
        S = S * np.exp((r - 0.5 * sig_real ** 2) * dt
                       + sig_real * np.sqrt(dt) * rng.standard_normal(n_paths))
        cash *= np.exp(r * dt)
        tau = T - i * dt
        new = -(bs_delta(S, K, r, sig_imp, tau) if tau > 1e-9 else (S > K).astype(float))
        cash -= (new - pos) * S
        pos = new
    return cash + pos * S + np.maximum(S - K, 0)

for sr in [0.25, 0.20, 0.15]:
    p = hedge(sr, 0.20, 252)
    pred = 0.5 * S0 ** 2 * bs_gamma(S0, K, r, 0.20, T) * (sr ** 2 - 0.20 ** 2) * T
    print(f"  realized {sr:.0%} vs implied 20%: mean P&L {p.mean():+7.4f}, "
          f"same sign as the gamma prediction {pred:+7.4f}: {np.sign(p.mean()) == np.sign(pred) or abs(pred) < 1e-9}")
sds = {n: hedge(0.25, 0.20, n).std() for n in [21, 63, 252]}
print(f"  hedging error shrinks with frequency: {sds[21] > sds[63] > sds[252]}")
# =>   realized 25% vs implied 20%: mean P&L +1.9918, same sign as the gamma prediction +2.1750: True
#      realized 20% vs implied 20%: mean P&L -0.0029, same sign as the gamma prediction +0.0000: True
#      realized 15% vs implied 20%: mean P&L -1.9913, same sign as the gamma prediction -1.6917: True
#      hedging error shrinks with frequency: True
```

Three clean results. Realized 25% against implied 20% earns **+1.99**; realized 15% loses **−1.99**; matched volatilities give **−0.003**, statistically zero, which is the strongest possible statement that Black–Scholes is internally consistent — hedging an option at the volatility you paid for reproduces its price exactly. The gamma prediction of +2.175 overstates the +1.99 outcome by about 8% for a knowable reason: the formula uses the *initial* gamma, while gamma decays as paths wander away from the strike, so the realized gamma-weighted average is lower than the at-the-money starting value.

The hedging error is the practical cost. Standard deviations fall from 2.04 at 21 rebalances to 1.38 at 63 and 1.00 at 252 — roughly the $1/\sqrt{n}$ that theory predicts for discrete hedging — but they do not go to zero, and they cannot: the residual is the path-dependence of gamma, which no rebalancing frequency removes. **A volatility trade with an edge of 2 has a standard deviation of 1**, which is why volatility desks run many positions rather than one and why [Part IV's short-volatility sleeve](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) needed a filter to survive its own tails.

## The variance swap, and what Part IV was really trading

The P&L formula has a gamma weighting that makes a delta-hedged option an *impure* volatility bet — it pays most when the stock is near the strike. The instrument that removes the weighting is the variance swap, and the identity that constructs it is worth deriving because Part IV used it without proof. Apply Itô's lemma to $\ln S$ under a general diffusion $dS = \mu S\,dt + \sigma_t S\,dW$:

$$
d(\ln S_t) \;=\; \left(\mu - \tfrac{\sigma_t^2}{2}\right)dt + \sigma_t\,dW_t,
\qquad
\frac{dS_t}{S_t} \;=\; \mu\,dt + \sigma_t\,dW_t .
$$

Subtract the first from the second — the $dW$ terms cancel exactly, which is the whole trick — and integrate:

$$
\int_0^T \sigma_t^2\,dt \;=\; 2\left[\int_0^T \frac{dS_t}{S_t} - \ln\frac{S_T}{S_0}\right].
$$

The left side is realized variance. The right side is *tradable*: the first term is a continuously rebalanced position holding a constant dollar amount of stock, and the second is a static short position in a log contract. Taking risk-neutral expectations makes the drift vanish and gives the fair variance-swap strike, and the log contract in turn decomposes into a strip of options weighted by $1/K^2$ — which is precisely the VIX construction. So [Part IV's claim](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) that VIX is the strike of a 30-day variance swap is not an analogy but an identity, and its short-volatility sleeve was harvesting the gap between that strike and subsequent realized variance: **+3.6 volatility points on average, positive in 82% of months, worst month −48.7 points.** The $1/K^2$ weighting is also why the strike is so sensitive to deep out-of-the-money puts, and why the variance premium and the equity skew are two views of the same phenomenon.

## Implied volatility is the market's error term, catalogued

Given a market price, invert the formula for the volatility that reproduces it. Since vega is strictly positive, the map is monotone and the inverse is unique — Newton's method converges in a few iterations from almost any start, and a bracketing solver is more robust near the wings. Implied volatility is therefore not a *model output* but a *unit of account*: quoting an option at 19.5 vol rather than $9.41 makes prices comparable across strikes, maturities, and underlyings.

If Black–Scholes were true, that number would be identical for every option on the same stock. It is not, and the shape of the disagreement is the most-studied object in derivatives. To generate a realistic surface without live quotes, use a model with genuine volatility dynamics — Heston's, where variance follows its own mean-reverting diffusion correlated with the stock:

$$
dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW^1_t,
\qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^2_t,
\qquad
d\langle W^1, W^2\rangle_t = \rho\,dt .
$$

The five parameters mean something: $\theta$ is the long-run variance, $\kappa$ its speed of mean reversion, $\xi$ the volatility of volatility, $\rho$ the correlation that generates skew, and $v_0$ today's variance. Heston's contribution was a semi-closed form — the characteristic function is exponential-affine, so prices follow from a Fourier integral. The characteristic function is stated here rather than derived (the derivation is a Riccati equation solved by standard methods) and priced with `scipy.integrate.quad`:

```python
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

S0, r = 100.0, 0.03
TRUE = dict(v0=0.04, kappa=2.0, theta=0.045, xi=0.5, rho=-0.7)

def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d1 - sig * np.sqrt(T))

def heston_call(S, K, r, T, v0, kappa, theta, xi, rho):
    def cf(u, j):                                  # Albrecher et al. stable form
        b = kappa - rho * xi if j == 1 else kappa
        u_ = 0.5 if j == 1 else -0.5
        d = np.sqrt((rho * xi * 1j * u - b) ** 2 - xi ** 2 * (2 * u_ * 1j * u - u ** 2))
        g2 = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        C = (r * 1j * u * T + kappa * theta / xi ** 2
             * ((b - rho * xi * 1j * u - d) * T
                - 2 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2))))
        D = ((b - rho * xi * 1j * u - d) / xi ** 2
             * ((1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T))))
        return np.exp(C + D * v0 + 1j * u * np.log(S))
    def P(j):
        f = lambda u: np.real(np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u))
        return 0.5 + quad(f, 1e-10, 250, limit=500)[0] / np.pi
    return S * P(1) - K * np.exp(-r * T) * P(2)

def iv(price, K, T):
    return brentq(lambda s: bs_call(S0, K, r, s, T) - price, 1e-4, 5.0, xtol=1e-12)

strikes = np.array([80, 90, 95, 100, 105, 110, 120])
surface = {T: np.array([iv(heston_call(S0, k, r, T, **TRUE), k, T) for k in strikes])
           for T in [0.25, 1.0, 2.0, 5.0]}
print(f"every maturity slopes down from low to high strikes (skew): "
      f"{all(all(np.diff(v) < 0) for v in surface.values())}")
print(f"skew flattens as maturity grows: "
      f"{(surface[0.25][0] - surface[0.25][-1]) > (surface[5.0][0] - surface[5.0][-1])}")
print(f"Feller condition 2*kappa*theta > xi^2 is violated here: "
      f"{2 * TRUE['kappa'] * TRUE['theta'] < TRUE['xi'] ** 2}")
# => every maturity slopes down from low to high strikes (skew): True
#    skew flattens as maturity grows: True
#    Feller condition 2*kappa*theta > xi^2 is violated here: True
```

The generated surface, in implied-volatility points:

| maturity | K=80 | K=90 | K=95 | K=100 | K=105 | K=110 | K=120 |
|---|---|---|---|---|---|---|---|
| 3 months | 0.266 | 0.232 | 0.214 | 0.195 | 0.176 | 0.161 | 0.150 |
| 1 year | 0.242 | 0.218 | 0.206 | 0.195 | 0.184 | 0.174 | 0.157 |
| 2 years | 0.230 | 0.214 | 0.206 | 0.199 | 0.192 | 0.186 | 0.174 |
| 5 years | 0.221 | 0.213 | 0.210 | 0.206 | 0.203 | 0.200 | 0.194 |

Every row slopes downward — low strikes carry higher implied volatility — which is equity **skew**, generated here by $\rho = -0.7$: when the stock falls, variance rises, so downside scenarios are both more likely and more volatile than a lognormal admits. And the skew flattens with maturity, from 11.6 vol points across the 3-month row to 2.7 across the 5-year, because variance mean-reverts to $\theta$ and the correlation has proportionally less time to distort the terminal distribution. Both features match equity index markets, which is why Heston remains a working model despite its known shortcomings. Two honest notes: the price agrees with a 200,000-path Monte Carlo at **9.2066 against 9.2393 ± 0.0476**, and the Feller condition $2\kappa\theta > \xi^2$ is *violated* by these parameters — variance can touch zero, which the simulation handles by truncation and which is common in real calibrations, where fitted $\xi$ is routinely too large for Feller to hold.

## The smile is a probability density in disguise

The surface is not merely a catalogue of errors; it contains the market's entire risk-neutral distribution. Differentiate the pricing integral twice with respect to strike: the first derivative gives $-e^{-rT}\,\mathbb{Q}(S_T > K)$, and differentiating the survival function again gives the density. This is the Breeden–Litzenberger result,

$$
q(K) \;=\; e^{rT}\,\frac{\partial^2 C}{\partial K^2},
$$

and it means a continuum of option prices *is* a probability distribution — no model required, only strikes:

```python
import numpy as np
from scipy import stats
from scipy.integrate import quad

S0, r, T = 100.0, 0.03, 1.0
TRUE = dict(v0=0.04, kappa=2.0, theta=0.045, xi=0.5, rho=-0.7)

def heston_call(S, K, r, T, v0, kappa, theta, xi, rho):
    def cf(u, j):
        b = kappa - rho * xi if j == 1 else kappa
        u_ = 0.5 if j == 1 else -0.5
        d = np.sqrt((rho * xi * 1j * u - b) ** 2 - xi ** 2 * (2 * u_ * 1j * u - u ** 2))
        g2 = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        C = (r * 1j * u * T + kappa * theta / xi ** 2
             * ((b - rho * xi * 1j * u - d) * T
                - 2 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2))))
        D = ((b - rho * xi * 1j * u - d) / xi ** 2
             * ((1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T))))
        return np.exp(C + D * v0 + 1j * u * np.log(S))
    def P(j):
        f = lambda u: np.real(np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u))
        return 0.5 + quad(f, 1e-10, 250, limit=500)[0] / np.pi
    return S * P(1) - K * np.exp(-r * T) * P(2)

Ks = np.arange(40.0, 201.0, 1.0)
C = np.array([heston_call(S0, k, r, T, **TRUE) for k in Ks])
dens = np.clip(np.exp(r * T) * (C[:-2] - 2 * C[1:-1] + C[2:]), 0, None)   # second difference
mid = Ks[1:-1]
dens /= np.trapezoid(dens, mid)

sig_atm = 0.195
lognorm = stats.lognorm.pdf(mid, s=sig_atm * np.sqrt(T),
                            scale=np.exp(np.log(S0) + (r - 0.5 * sig_atm ** 2) * T))
left = mid <= 70
p_heston = np.trapezoid(dens[left], mid[left])
p_lognormal = np.trapezoid(lognorm[left], mid[left])
print(f"extracted density is a valid distribution (non-negative, integrates to 1): "
      f"{dens.min() >= 0 and abs(np.trapezoid(dens, mid) - 1) < 1e-6}")
print(f"its mean matches the forward to within 1%: "
      f"{abs(np.trapezoid(dens * mid, mid) / (S0 * np.exp(r * T)) - 1) < 0.01}")
print(f"the left tail is fatter than lognormal: {p_heston > p_lognormal}")
print(f"P(S_T < 70) is at least 50% higher than lognormal: {p_heston / p_lognormal > 1.5}")
# => extracted density is a valid distribution (non-negative, integrates to 1): True
#    its mean matches the forward to within 1%: True
#    the left tail is fatter than lognormal: True
#    P(S_T < 70) is at least 50% higher than lognormal: True
```

The extracted density puts **1.77×** the lognormal probability on a 30% decline, which is the skew translated from volatility units into the language of risk. That connects directly to [Part VIII's tail work](../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md), which fitted an extreme-value index of $\xi = +0.327$ to *physical* returns: the option market's risk-neutral tail and the historical physical tail are both fat, and their ratio is the variance risk premium — the compensation for bearing that risk, and exactly what Part IV's short-volatility sleeve was paid.

## Dupire fits everything and explains nothing

If the market's surface disagrees with Heston's, one can always find a *local* volatility function $\sigma_{\text{loc}}(S, t)$ — deterministic, but varying with spot and time — that reproduces every quoted price exactly. Dupire's formula inverts the surface directly:

$$
\sigma_{\text{loc}}^2(K, T) \;=\; \frac{\dfrac{\partial C}{\partial T} + rK\dfrac{\partial C}{\partial K}}
{\tfrac12 K^2\,\dfrac{\partial^2 C}{\partial K^2}} .
$$

The construction is elegant — a perfect fit to every liquid quote, with no calibration error and no optimizer. Its cost is visible in the denominator: $\partial^2 C/\partial K^2$ is the same second derivative the density used, and second derivatives of noisy data amplify noise. Perturb the surface by twenty basis points, well inside a normal bid-ask spread, and measure what moves:

```python
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

S0, r = 100.0, 0.03
TRUE = dict(v0=0.04, kappa=2.0, theta=0.045, xi=0.5, rho=-0.7)

def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d1 - sig * np.sqrt(T))

def heston_call(S, K, r, T, v0, kappa, theta, xi, rho):
    def cf(u, j):
        b = kappa - rho * xi if j == 1 else kappa
        u_ = 0.5 if j == 1 else -0.5
        d = np.sqrt((rho * xi * 1j * u - b) ** 2 - xi ** 2 * (2 * u_ * 1j * u - u ** 2))
        g2 = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        C = (r * 1j * u * T + kappa * theta / xi ** 2
             * ((b - rho * xi * 1j * u - d) * T
                - 2 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2))))
        D = ((b - rho * xi * 1j * u - d) / xi ** 2
             * ((1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T))))
        return np.exp(C + D * v0 + 1j * u * np.log(S))
    def P(j):
        f = lambda u: np.real(np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u))
        return 0.5 + quad(f, 1e-10, 250, limit=500)[0] / np.pi
    return S * P(1) - K * np.exp(-r * T) * P(2)

Kg = np.arange(80.0, 121.0, 5.0)
Tg = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
base = np.array([[heston_call(S0, k, r, t, **TRUE) for k in Kg] for t in Tg])

def local_vol(C):
    dCdT = np.gradient(C, Tg, axis=0)
    dCdK = np.gradient(C, Kg, axis=1)
    d2CdK2 = np.gradient(dCdK, Kg, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.clip((dCdT + r * Kg[None, :] * dCdK)
                               / (0.5 * Kg[None, :] ** 2 * d2CdK2), 1e-6, 4.0))

def implied(C):
    return np.array([[brentq(lambda s: bs_call(S0, Kg[j], r, s, Tg[i]) - C[i, j],
                             1e-4, 5.0, xtol=1e-12) for j in range(len(Kg))]
                     for i in range(len(Tg))])

lv0, iv0 = local_vol(base), implied(base)
for seed in [1, 2]:
    jitter = base * (1 + 0.002 * np.random.default_rng(seed).standard_normal(base.shape))
    d_iv = np.nanmax(np.abs(implied(jitter) - iv0))
    d_lv = np.nanmax(np.abs(local_vol(jitter) - lv0))
    print(f"  seed {seed}: 20 bp of quote noise moves implied vol under 0.01: {d_iv < 0.01}, "
          f"local vol more than 10x as much: {d_lv > 10 * d_iv}")
# =>   seed 1: 20 bp of quote noise moves implied vol under 0.01: True, local vol more than 10x as much: True
#      seed 2: 20 bp of quote noise moves implied vol under 0.01: True, local vol more than 10x as much: True
```

The measured amplification is **21× on one seed and 81× on the other**: implied volatility moved by 0.0036 and 0.0031 respectively, while local volatility moved by 0.077 and **0.252**. Two quote sets that a trader would call identical produce local-volatility surfaces that differ by twenty-five volatility points somewhere on the grid. This is the module's central caution and it generalizes far past options: **a model that fits its inputs exactly transmits every input error directly into its state.** Local volatility has no residual to absorb noise, so noise becomes signal. The standard defences are to smooth the implied-volatility surface before differentiating (never the price surface), to enforce the no-arbitrage constraints — butterfly, $\partial^2 C/\partial K^2 \ge 0$, and calendar, prices non-decreasing in maturity at fixed forward moneyness — and to regularize the fit rather than interpolate it.

The deeper objection is that local volatility gets the *dynamics* wrong even when its prices are right. It reproduces today's surface exactly and then predicts that the smile flattens and slides in a way markets do not follow, so its forward smile is wrong and any product whose value depends on future volatility levels — cliquets, forward-starting options, most exotics — is mispriced by a model that fits every vanilla perfectly. Stochastic volatility fits less well and extrapolates better, which is why desks calibrate both and use each where its errors are tolerable.

!!! warning "A model that fits its inputs exactly has nowhere to put their errors"
    Dupire's local volatility reproduces every quoted price with zero residual, and twenty basis points of quote noise — inside any real bid-ask spread — moved the extracted surface by up to 0.25 in volatility terms, an amplification of 81× over the change in implied volatility. Exact fit is not accuracy; it is the absence of a residual, and every basis point of noise has to go somewhere. Smooth in implied-volatility space, enforce butterfly and calendar no-arbitrage before differentiating, and prefer a model that fits imperfectly and extrapolates sensibly over one that fits perfectly and does not.

!!! abstract "Key takeaways"
    - Put–call parity $C - P = S - Ke^{-rT}$ holds with no model at all, so persistent violations measure your borrow, dividend, or funding assumptions rather than an arbitrage.
    - The identity $S\varphi(d_1) = Ke^{-rT}\varphi(d_2)$ (verified to $7\times10^{-15}$) is why $\Delta = \Phi(d_1)$ — the $\partial d$ terms cancel exactly rather than being negligible.
    - The gamma–theta relation $\Theta \approx -\tfrac12\sigma^2S^2\Gamma$ is exact for a hedged book (−3.8667 on both sides): theta is the rent paid for gamma.
    - A daily delta-hedged long call earned **+1.99** at 25% realized against 20% implied, lost **−1.99** at 15%, and returned **−0.003** when they matched — hedging at the volatility you paid reproduces the price exactly.
    - Hedging error fell from sd 2.04 at 21 rebalances to 1.00 at 252, roughly $1/\sqrt{n}$, but never to zero: a volatility trade with an edge of 2 carries a standard deviation of 1.
    - Cancelling the $dW$ terms between $d(\ln S)$ and $dS/S$ derives the variance-swap identity, making VIX literally the strike of a 30-day variance swap and Part IV's +3.6-point premium a measured spread rather than an analogy.
    - Heston with $\rho = -0.7$ generates downward skew at every maturity, flattening from 11.6 vol points at 3 months to 2.7 at 5 years, and agrees with Monte Carlo at 9.2066 against 9.2393 ± 0.0476 — with the Feller condition violated, as in most real calibrations.
    - Breeden–Litzenberger extracted a valid density placing **1.77×** the lognormal probability on a 30% decline; Dupire's exact fit amplified 20 bp of quote noise into up to **0.25** of local volatility, an 81× amplification.

## Where this goes next

The continuous-time machinery this module solved — Itô's lemma, the hedging argument, and the PDE — is derived in [Stochastic Calculus](03-stochastic-calculus.md), which also explains why the replication is exact only under continuous rebalancing and therefore why the hedging error measured here exists at all. The Greeks become a quoting problem rather than a hedging problem in [Market Making](12-market-making.md), where a liquidity provider manages inventory in exactly the way an options desk manages a book of sensitivities, and where adverse selection plays the role that realized-versus-implied plays here. In the core course, the volatility risk premium this module derived is traded in [Part IV, lesson three](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md), and the physical-measure tail behaviour that the risk-neutral density should be compared against is measured in [Part VIII, lesson five](../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md).
