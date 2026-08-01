# Particle and Kalman Filters

Almost every estimate in this course adapts through time, and almost every one of them adapts by a constant somebody picked. Part III's EWMA volatility had a decay factor. Part IV's pairs trade standardized its spread against a 60-day rolling window — chosen, in that lesson's own words, because 3.4-day half-lives want "a few dozen days." [Part III's regime model](../part-03-statistics/06-bayesian-methods-and-hmms.md) ended by confessing that its most seductive table used *smoothed* probabilities, which read the future, and left the filtered version as an explicit unpaid debt. All three are the same problem wearing three costumes: an unobserved state — a fair hedge ratio, a current volatility level, a market regime — must be estimated from noisy observations, in real time, using only the past. That problem has a name and a theory, and this module is about both.

The Kalman filter solves it exactly when the system is linear and Gaussian, and its solution turns out to *derive* the smoothing constants that were previously chosen: a signal-to-noise ratio in, a decay rate out. Particle filters solve it approximately when the system is neither, which is the case for stochastic volatility. The module builds both from scratch — thirty lines each, no `filterpy`, no `pykalman` — validates them against `statsmodels` and `arch`, and then points them at the course's own open questions. The uncomfortable results are pre-announced, because there are two. A Kalman-filtered dynamic hedge ratio, the textbook application, makes [Part IV's dead pair](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) **deader** — gross Sharpe 1.23 falls to 0.88 — and the diagnostics will explain exactly why in a way that indicts the whole idea of adaptive estimation for that trade. And Part III's 1.10 regime Sharpe, recomputed honestly with filtered probabilities and annual refits, is **0.53** — less than half the headline, and still twice the unconditional baseline, which is the rare case in this course where an oversold result turns out to have something real underneath.

## Every rolling estimate is a filter that refuses to say so

A state-space model has two equations. The first says how a hidden state evolves; the second says how observations are generated from it:

$$
x_t \;=\; F\,x_{t-1} + w_t, \qquad w_t \sim \mathcal N(0, Q),
$$

$$
y_t \;=\; H\,x_t + v_t, \qquad v_t \sim \mathcal N(0, R).
$$

That is the entire vocabulary, and its expressiveness comes from what you are willing to call a state. Set $F = 1$, $H = 1$ and the state is a slowly-drifting mean — a "local level" model, and the honest version of a rolling average. Set the state to $(\alpha_t, \beta_t)$ and the observation row to $H_t = (1, x_t)$ and you have time-varying regression: a hedge ratio that moves. Let the state be log-variance and the model becomes stochastic volatility. In every case the two covariances $Q$ and $R$ do one job — they set the exchange rate between "the world changed" and "the measurement was noisy," and every rolling-window length you have ever chosen was an implicit, cruder answer to the same question.

The distinction that organizes the rest of this module is about *which* information a given estimate consumed. Write $\mathcal F_t$ for everything observable up to time $t$. Then $p(x_t \mid \mathcal F_t)$ is the **filtered** state — available in real time, the only kind a trading system can act on — while $p(x_t \mid \mathcal F_T)$, conditioning on the whole sample through the final date $T$, is the **smoothed** state. Smoothing is strictly more accurate and strictly unavailable, and the gap between them is not a technicality: it is where a Sharpe of 1.10 goes when you make it honest.

## The Kalman filter is Bayes' rule with linear algebra

The recursion has two half-steps. **Predict** pushes last period's belief through the state equation. If $x_{t-1} \mid \mathcal F_{t-1} \sim \mathcal N(\hat x_{t-1|t-1}, P_{t-1|t-1})$, then applying $F$ and adding independent noise $Q$ gives

$$
\hat x_{t|t-1} \;=\; F\,\hat x_{t-1|t-1},
\qquad
P_{t|t-1} \;=\; F\,P_{t-1|t-1}\,F^\top + Q,
$$

which is just the mean and variance of a linear map of a Gaussian. **Update** conditions on the new observation. Before seeing $y_t$ the model expects $H\hat x_{t|t-1}$, so the surprise — the *innovation* — and its covariance are

$$
\nu_t \;=\; y_t - H\,\hat x_{t|t-1},
\qquad
S_t \;=\; H\,P_{t|t-1}\,H^\top + R .
$$

Now $(x_t, y_t)$ are jointly Gaussian given $\mathcal F_{t-1}$, with cross-covariance $P_{t|t-1}H^\top$, and conditioning a Gaussian on a Gaussian is the same completion of the square that produced the [GP posterior in the previous module](01-bayesian-optimization.md):

$$
\hat x_{t|t} \;=\; \hat x_{t|t-1} + K_t\,\nu_t,
\qquad
K_t \;=\; P_{t|t-1}H^\top S_t^{-1},
\qquad
P_{t|t} \;=\; (I - K_tH)\,P_{t|t-1}.
$$

The gain $K_t$ is the whole story in one matrix: it is the ratio of what you do not know about the state to the total uncertainty in the measurement, so a filter that trusts its model ($Q$ small) has a small gain and moves slowly, while a filter that distrusts its sensor ($R$ large) does the same. Two practical notes that separate working filters from broken ones. The covariance update above is algebraically correct and numerically fragile — it subtracts two positive-definite matrices and, in finite precision, can produce a covariance that is not. The Joseph form,

$$
P_{t|t} \;=\; (I - K_tH)\,P_{t|t-1}\,(I - K_tH)^\top + K_t R\, K_t^\top,
$$

is a sum of two positive-semidefinite terms and stays symmetric positive-definite by construction; it costs a few flops and buys the difference between a filter that degrades gracefully and one that produces NaNs on year seven. And the filter needs a prior $(\hat x_0, P_0)$: setting $P_0$ enormous — a *diffuse* prior — says "I know nothing," and the first observations then dominate, which is almost always what you want. Here is the whole thing, thirty lines, on a simulated local-level series:

```python
import numpy as np

rng = np.random.default_rng(42)
n, Q, R = 500, 0.01, 1.0
x = np.cumsum(np.sqrt(Q) * rng.standard_normal(n))     # hidden state
y = x + np.sqrt(R) * rng.standard_normal(n)            # noisy observation

def kalman(y, Q, R, x0=0.0, P0=1e6):
    xf, Pf, ll, out = x0, P0, 0.0, []
    for yt in y:
        xp, Pp = xf, Pf + Q                             # predict
        nu, S = yt - xp, Pp + R                         # innovation
        K = Pp / S                                      # gain
        xf = xp + K * nu                                # update
        Pf = (1 - K) * Pp * (1 - K) + K * R * K         # Joseph form
        ll += -0.5 * (np.log(2 * np.pi * S) + nu ** 2 / S)
        out.append(xf)
    return np.array(out), Pf, ll

xf, Pf, ll = kalman(y, Q, R)
print(f"RMSE: filtered state {np.sqrt(np.mean((xf - x) ** 2)):.4f}, "
      f"raw observations {np.sqrt(np.mean((y - x) ** 2)):.4f}")
print(f"terminal P {Pf:.4f} (implied sd {np.sqrt(Pf):.4f}), log-likelihood {ll:.4f}")
# => RMSE: filtered state 0.3112, raw observations 1.0183
#    terminal P 0.0951 (implied sd 0.3084), log-likelihood -742.1022
```

The filter tracks a state it never observes to an RMSE of 0.31 using measurements whose own error is 1.02 — a threefold noise reduction bought entirely by the assumption that the state moves slowly. And the number it reports for its own accuracy, $\sqrt{P} = 0.3084$, matches the error it actually achieved, 0.3112, to two decimal places. A filter that is honest about its own uncertainty is the feature; everything downstream in this module depends on it.

## The likelihood lives in the innovations

Nothing above told you where $Q$ and $R$ come from, and in practice they are estimated, not known. The route is one of the most elegant results in time-series analysis. Factor the joint density of the observations into one-step-ahead conditionals; each of those is Gaussian with mean $H\hat x_{t|t-1}$ and variance $S_t$, both of which the filter already computes. So the log-likelihood of the *parameters* falls out of a single filtering pass as a by-product:

$$
\ell(Q, R) \;=\; -\tfrac12 \sum_{t=1}^{T}\Bigl(\log \lvert S_t\rvert + \nu_t^\top S_t^{-1} \nu_t\Bigr) + \text{const},
$$

the *prediction-error decomposition*. Maximize it over $(Q, R)$ and you have maximum-likelihood estimates of the noise levels — which is to say, the smoothing constant is no longer chosen, it is fitted. The same quantities give the diagnostic that matters: if the model is correct, the standardized innovations $\nu_t/\sqrt{S_t}$ are independent standard normals, so any autocorrelation in them ([Part III's Ljung–Box test](../part-03-statistics/03-time-series.md) is the instrument) is the filter telling you its state equation is missing something. Before trusting the implementation, check it against a mature one:

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n, Q, R = 500, 0.01, 1.0
x = np.cumsum(np.sqrt(Q) * rng.standard_normal(n))
y = x + np.sqrt(R) * rng.standard_normal(n)

def kalman(y, Q, R, x0=0.0, P0=1e6):
    xf, Pf, ll, out = x0, P0, 0.0, []
    for yt in y:
        xp, Pp = xf, Pf + Q
        nu, S = yt - xp, Pp + R
        K = Pp / S
        xf = xp + K * nu
        Pf = (1 - K) * Pp * (1 - K) + K * R * K
        ll += -0.5 * (np.log(2 * np.pi * S) + nu ** 2 / S)
        out.append(xf)
    return np.array(out), Pf, ll

mine, _, ll = kalman(y, Q, R)
mod = sm.tsa.UnobservedComponents(y, "local level", initialization="known",
                                  initial_state=np.array([0.0]),
                                  initial_state_cov=np.array([[1e6]]))
res = mod.filter([R, Q])                                # [obs var, level var]
print(f"max |state difference| vs statsmodels: {np.abs(mine - res.filtered_state[0]).max():.2e}")
print(f"log-likelihood: mine {ll:.4f}, statsmodels reports {res.llf:.4f}, "
      f"sum of its per-observation terms {np.sum(res.llf_obs):.4f}")
print(f"statsmodels discards the first {res.loglikelihood_burn} observation(s) under a diffuse prior")
# => max |state difference| vs statsmodels: 1.03e-08
#    log-likelihood: mine -742.1022, statsmodels reports -734.2755, sum of its per-observation terms -742.1022
#    statsmodels discards the first 1 observation(s) under a diffuse prior
```

The states agree to $10^{-8}$, which validates the recursion. The likelihoods appear to disagree by 7.83 — and the third line resolves it: under a diffuse prior the first observation contributes an essentially arbitrary term (the prior variance was made up), so `statsmodels` discards it by convention, while the naive sum keeps it. Summing its per-observation terms reproduces the hand-rolled number exactly. This is worth more than a footnote: two correct implementations can report different likelihoods for the same model, and comparing across libraries without checking the burn-in convention is a reliable way to "discover" a bug that is not there.

## Kalman is EWMA with a justified dial

The local-level filter has a fixed point. As $t$ grows, $P_{t|t}$ stops changing and the gain converges to a constant $K^\ast$, which turns the recursion into $\hat x_t = (1-K^\ast)\hat x_{t-1} + K^\ast y_t$ — an exponentially weighted moving average with decay $\lambda = 1 - K^\ast$. So EWMA is not an approximation to the Kalman filter; it *is* the Kalman filter, at steady state, for the local-level model. The steady-state condition $P = (1-K)(P+Q)$ with $K = (P+Q)/(P+Q+R)$ reduces, writing $q = Q/R$ for the signal-to-noise ratio, to the quadratic $K^2 + qK - q = 0$, whence

$$
K^\ast \;=\; \tfrac{1}{2}\left(\sqrt{q^2 + 4q} \;-\; q\right).
$$

This equation is the reason the module exists. Every EWMA in this course took its decay as an input; here the decay is an *output*, derived from a ratio of variances that can be estimated by maximum likelihood from the data. Choosing $\lambda = 0.94$ because RiskMetrics did is a tradition; deriving $\lambda = 0.905$ because the state moves with variance 0.01 against measurement noise 1.0 is a model:

```python
import numpy as np

for q in [0.001, 0.01, 0.1, 1.0]:
    K = 0.5 * (np.sqrt(q ** 2 + 4 * q) - q)
    print(f"signal-to-noise q = Q/R = {q:6.3f}: steady-state gain {K:.4f}, "
          f"EWMA lambda {1 - K:.4f}, half-life {np.log(0.5) / np.log(1 - K):5.1f} periods")
# => signal-to-noise q = Q/R =  0.001: steady-state gain 0.0311, EWMA lambda 0.9689, half-life  21.9 periods
#    signal-to-noise q = Q/R =  0.010: steady-state gain 0.0951, EWMA lambda 0.9049, half-life   6.9 periods
#    signal-to-noise q = Q/R =  0.100: steady-state gain 0.2702, EWMA lambda 0.7298, half-life   2.2 periods
#    signal-to-noise q = Q/R =  1.000: steady-state gain 0.6180, EWMA lambda 0.3820, half-life   0.7 periods
```

Read the table as a conversion chart between a modeling belief and a window length. A state that moves a thousandth as fast as the noise justifies a 22-period memory; a state as volatile as its measurement error justifies essentially none. The last row's gain, 0.618, is the golden ratio's reciprocal — an accident of $q = 1$, and a good mnemonic for the fact that when signal and noise are equal, the optimal filter still throws away a third of each new observation.

## A dynamic hedge ratio does not resurrect a dead pair

Now the textbook application, and the module's first uncomfortable result. [Part IV's pairs trade](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) imposed a one-for-one hedge between SPY and IVV — defensible, since the two funds hold the same index — and standardized the resulting spread on a 60-day window. The natural objection is that the correct hedge ratio is not exactly one and does not stay put: expense ratios differ, the funds' dividend calendars differ, and their relative pricing drifts. A Kalman filter is the canonical repair. Model the hedge as a random walk in $(\alpha_t, \beta_t)$ and let the observation equation be the regression itself,

$$
\ln P^{\text{IVV}}_t \;=\; \alpha_t + \beta_t \ln P^{\text{SPY}}_t + \varepsilon_t,
\qquad
\begin{pmatrix}\alpha_t\\\beta_t\end{pmatrix}
=
\begin{pmatrix}\alpha_{t-1}\\\beta_{t-1}\end{pmatrix} + w_t,
$$

with $Q = \tfrac{\delta}{1-\delta}I$ parameterized by a single "how fast may the hedge drift" knob $\delta$. The spread becomes the filter's own innovation, and the strategy trades that instead. Everything else — the rule, the 60-day window, the entry and exit thresholds, [the cost model](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md), and the next-close execution gate that Part IV used to grade this trade honestly — is held fixed, so the only thing that changes is the hedge:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")[["SPY", "IVV"]].dropna()
ls, li = np.log(px["SPY"]), np.log(px["IVV"])
LEG = (0.5 + 0.2 + 1.0 + 0.2) / 1e4              # both legs: half-spread + commission, Part IV

def zraw(spread):                                 # Part IV lesson two's rule, unshifted
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    raw = pd.Series(np.nan, index=z.index)
    raw[z > 2], raw[z < -2] = -1.0, 1.0
    raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
    return raw.ffill().fillna(0.0)

def grade(spread, label):
    base, res = zraw(spread), []
    for lag in (1, 2):                            # 1 = fill at signal close, 2 = next close
        pos = base.shift(lag)
        pnl = (pos * spread.diff()).dropna()
        trips = int(((pos != 0) & (pos.shift(1) == 0)).sum())
        cost = pos.diff().abs().fillna(0).reindex(pnl.index).fillna(0) * LEG
        net = (pnl - cost).dropna()
        res.append((np.sqrt(252) * pnl.mean() / pnl.std(), pnl.sum() * 1e4 / trips,
                    np.sqrt(252) * net.mean() / net.std(), trips))
    (g1, b1, _, t1), (g2, b2, n2, _) = res
    print(f"{label:15s} signal close {g1:+.2f} ({b1:+.1f} bp/trip, {t1} trips) | "
          f"next close {g2:+.2f} ({b2:+.1f} bp/trip) | net of costs {n2:+.2f}")

def kalman_beta(y, x, delta, R=1e-4):
    out, P, b = np.zeros((len(y), 2)), np.eye(2), np.array([0.0, 1.0])
    Q = delta / (1 - delta) * np.eye(2)
    for t in range(len(y)):
        H = np.array([1.0, x[t]])
        Pp = P + Q
        S = H @ Pp @ H + R
        K = Pp @ H / S
        b = b + K * (y[t] - H @ b)
        P = Pp - np.outer(K, H) @ Pp
        out[t] = b
    return out

grade((ls - li).dropna(), "static 1:1")
for delta in [1e-6, 1e-5, 1e-4]:
    bt = kalman_beta(li.values, ls.values, delta)
    grade(pd.Series(li.values - (bt[:, 0] + bt[:, 1] * ls.values), index=px.index),
          f"Kalman d={delta:.0e}")
# => static 1:1      signal close +1.23 (+23.0 bp/trip, 168 trips) | next close +0.19 (+2.8 bp/trip) | net of costs -0.07
#    Kalman d=1e-06  signal close +1.18 (+19.9 bp/trip, 127 trips) | next close -0.09 (-0.7 bp/trip) | net of costs -0.53
#    Kalman d=1e-05  signal close +0.88 (+9.4 bp/trip, 89 trips) | next close -0.41 (-2.6 bp/trip) | net of costs -0.95
#    Kalman d=1e-04  signal close +0.79 (+2.0 bp/trip, 71 trips) | next close -0.48 (-0.8 bp/trip) | net of costs -1.83
```

The first row is a reconciliation, and it is exact: **+1.23 Sharpe, 23.0 bp per round trip, 168 trips** at signal-close fills, **+0.19** and 2.8 bp at next-close fills, **−0.07** net of costs — the same three numbers [lesson two](../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md), [lesson seven](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md), and [the Part IV gauntlet](../part-04-strategy-development/08-validation-and-overfitting.md) published, reproduced here by independent code. That matters because it means the rows below are measuring the filter and nothing else.

And every one of them is worse. The most conservative filter, barely allowed to move, gives back 0.05 of gross Sharpe; the middle setting gives back 0.35; the fastest gives back 0.44. Net of costs and honest execution, the static pair's −0.07 becomes −0.53, −0.95, −1.83. The technique that was supposed to sharpen the trade *monotonically damages it*, and the more adaptive the filter, the worse the damage. Two diagnostics say why:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")[["SPY", "IVV"]].dropna()
ls, li = np.log(px["SPY"]), np.log(px["IVV"])

def kalman_beta(y, x, delta, R=1e-4):
    out, P, b = np.zeros((len(y), 2)), np.eye(2), np.array([0.0, 1.0])
    Q = delta / (1 - delta) * np.eye(2)
    for t in range(len(y)):
        H = np.array([1.0, x[t]])
        Pp = P + Q
        S = H @ Pp @ H + R
        K = Pp @ H / S
        b = b + K * (y[t] - H @ b)
        P = Pp - np.outer(K, H) @ Pp
        out[t] = b
    return out

bt = kalman_beta(li.values, ls.values, 1e-5)
static = (ls - li).dropna()
dyn = pd.Series(li.values - (bt[:, 0] + bt[:, 1] * ls.values), index=px.index)
print(f"Kalman beta: mean {bt[:, 1].mean():.4f}, sd {bt[:, 1].std():.4f}, "
      f"range [{bt[:, 1].min():.3f}, {bt[:, 1].max():.3f}]")
print(f"spread sd:        static {static.std() * 1e4:5.1f} bp, Kalman {dyn.std() * 1e4:5.1f} bp")
print(f"spread autocorr:  static {static.autocorr(1):+.4f}, Kalman {dyn.autocorr(1):+.4f}")
# => Kalman beta: mean 0.9830, sd 0.0098, range [0.937, 0.999]
#    spread sd:        static  23.3 bp, Kalman   3.3 bp
#    spread autocorr:  static +0.8166, Kalman -0.2702
```

There is the mechanism, and it is not subtle. The filter shrinks the spread's standard deviation from 23.3 basis points to **3.3** — it has absorbed seven-eighths of the very displacement the strategy exists to trade — and flips its first-order autocorrelation from +0.82 to **−0.27**. The static spread is a persistent, slowly-reverting quantity, exactly the object a z-score entry rule is built for. The Kalman residual is an overshooting, negatively-autocorrelated innovation series, because *that is a filter's job*: a well-specified filter produces white innovations, and white innovations have no exploitable mean reversion left in them by construction. Applying a mean-reversion rule to a filter's residual is asking the filter to leave behind precisely what it was designed to remove.

The deeper reading indicts the framing rather than the tool. The tempting story — "the pair failed because the hedge ratio was stale" — was never supported by evidence. The measured hedge ratio has a standard deviation of 0.0098 and never leaves $[0.937, 0.999]$; it is, to a very good approximation, one. Part IV diagnosed the actual cause and quantified it: a spread with a 3.4-day half-life does most of its reverting in the first day, so the trade's entire margin lives inside the fill assumption, and one day of execution lag takes it from 23.0 bp per trip to 2.8. Estimation error was never the binding constraint. **Adaptivity solved a problem the trade did not have, at the cost of the signal it did have** — and that failure mode is general enough to name: before deploying a more sophisticated estimator, confirm that estimation error, not economics, is what stands between you and the edge.

## Linearize and pray: extended and unscented filters

The Kalman filter assumed linearity. Most interesting states are not linear in their observations — an option's implied volatility, a position's liquidation risk, any state observed through a squaring or an absolute value. Two standard repairs exist, and their difference is instructive. The **extended** Kalman filter linearizes the observation function around the current estimate, replacing $H$ with the Jacobian $H_t = \partial h/\partial x\rvert_{\hat x_{t|t-1}}$ and otherwise running unchanged. The **unscented** Kalman filter refuses to linearize at all: it picks a small set of deterministic sigma points that reproduce the predicted mean and covariance, pushes each one through the exact nonlinear function, and recomputes moments from the transformed set. Linearizing the *function* versus transforming the *distribution* sounds like a distinction without a difference until the function has curvature the state uncertainty can feel. Test both on $y_t = x_t^2 + v_t$, where the Jacobian $2\hat x$ vanishes at the origin and takes the wrong sign whenever the estimate lands on the wrong side of zero:

```python
import numpy as np

def sim(seed, n=200):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + 0.3 * r.standard_normal()
    return x, x ** 2 + 0.5 * r.standard_normal(n)

def ekf(y, F=0.95, Q=0.09, R=0.25):
    xf, P, out = 0.5, 1.0, []
    for yt in y:
        xp, Pp = F * xf, F * P * F + Q
        H = 2 * xp                                    # Jacobian of x^2
        S = H * Pp * H + R
        K = Pp * H / S
        xf = xp + K * (yt - xp ** 2)
        P = (1 - K * H) * Pp
        out.append(xf)
    return np.array(out)

def ukf(y, F=0.95, Q=0.09, R=0.25, kappa=2.0):
    xf, P, out = 0.5, 1.0, []
    for yt in y:
        xp, Pp = F * xf, F * P * F + Q
        c = 1.0 + kappa
        sig = np.array([xp, xp + np.sqrt(c * Pp), xp - np.sqrt(c * Pp)])
        w = np.array([kappa / c, 0.5 / c, 0.5 / c])
        ys = sig ** 2                                 # exact nonlinearity, three times
        ybar = w @ ys
        Pyy = w @ (ys - ybar) ** 2 + R
        Pxy = w @ ((sig - xp) * (ys - ybar))
        K = Pxy / Pyy
        xf = xp + K * (yt - ybar)
        P = Pp - K * Pyy * K
        out.append(xf)
    return np.array(out)

e = np.array([np.sqrt(np.mean((ekf(sim(s)[1]) - sim(s)[0]) ** 2)) for s in range(20)])
u = np.array([np.sqrt(np.mean((ukf(sim(s)[1]) - sim(s)[0]) ** 2)) for s in range(20)])
print(f"EKF: median RMSE {np.median(e):.3f}, worst {e.max():.3f}, "
      f"failed (RMSE > 1) on {int((e > 1).sum())} of 20 seeds")
print(f"UKF: median RMSE {np.median(u):.3f}, worst {u.max():.3f}, "
      f"failed (RMSE > 1) on {int((u > 1).sum())} of 20 seeds")
# => EKF: median RMSE 1.375, worst 3.197, failed (RMSE > 1) on 15 of 20 seeds
#    UKF: median RMSE 0.862, worst 1.623, failed (RMSE > 1) on 3 of 20 seeds
```

The EKF fails on three-quarters of the seeds; the unscented filter fails on three of twenty and halves the median error. Neither is *good* here — the problem is genuinely hard, because $y = x^2$ destroys the sign of the state and no filter can recover information the observation never carried — but the ranking is the lesson, and it generalizes: the EKF's error grows with the product of the function's curvature and the state's uncertainty, so it is fine for gentle nonlinearities and dangerous exactly when the filter is already unsure. A practical tell, cheap to monitor: EKF divergence usually announces itself as standardized innovations that stop looking standard, so the whiteness diagnostic from the likelihood section is also the early-warning system.

## Particle filters trade the Gaussian assumption for Monte Carlo error

When the state is not Gaussian at all — bimodal, bounded, skewed — sigma points will not save you either, and the last resort is to represent the posterior by samples. Sequential importance sampling maintains $N$ weighted particles $\{x_t^{(i)}, w_t^{(i)}\}$ approximating $p(x_t \mid \mathcal F_t)$. Propagate each particle through the state equation, then reweight by how well it explains the new observation. With the state transition as the proposal — the *bootstrap* filter — the general weight update collapses to something you can implement in one line:

$$
w_t^{(i)} \;\propto\; w_{t-1}^{(i)}\; p\bigl(y_t \mid x_t^{(i)}\bigr).
$$

The method has one characteristic failure. After a few steps, one particle accumulates nearly all the weight and the other $N-1$ contribute nothing — *weight degeneracy* — and the effective sample size

$$
\mathrm{ESS} \;=\; \frac{1}{\sum_{i=1}^{N} \bigl(\tilde w_t^{(i)}\bigr)^2}
$$

measures how many particles are really doing work (it equals $N$ when weights are uniform and 1 when one particle owns everything). The repair is to resample when ESS falls below a threshold, drawing a new equally-weighted population in proportion to the old weights; systematic resampling, which spreads a single uniform draw across the cumulative distribution, is the low-variance standard. Resampling cures weight degeneracy and causes *path* degeneracy — repeated resampling means all surviving particles share a common ancestor, so the filter's memory of the distant past collapses to a single trajectory. Filtering the present is fine; reconstructing history from a particle filter's ancestry is not.

The natural application is stochastic volatility, where the log-variance follows an autoregression and the return is scaled by its exponential:

$$
h_t \;=\; \mu + \phi\,(h_{t-1} - \mu) + \sigma_\eta\,\eta_t,
\qquad
y_t \;=\; \exp(h_t/2)\;\varepsilon_t .
$$

No Kalman filter applies: the observation is nonlinear in the state and, once you condition on $h_t$, the return is Gaussian only in scale, not in location. [Part III fitted GARCH](../part-03-statistics/03-time-series.md) to the same data and got persistence $\alpha + \beta = 0.982$ with a strong asymmetry term — a natural benchmark, since GARCH makes volatility a deterministic function of past returns while stochastic volatility gives it its own shock. Run the particle filter on SPY and grade both against future realized volatility with the QLIKE loss, which is the standard proper scoring rule for variance forecasts:

```python
import numpy as np
import pandas as pd
from arch import arch_model

px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()
y = r.values

def bootstrap_pf(y, mu=-9.5, phi=0.98, seta=0.15, N=2000, seed=0):
    rng = np.random.default_rng(seed)
    h = mu + np.sqrt(seta ** 2 / (1 - phi ** 2)) * rng.standard_normal(N)
    vol, ess_hist, resamples = np.zeros(len(y)), [], 0
    for t, yt in enumerate(y):
        h = mu + phi * (h - mu) + seta * rng.standard_normal(N)       # propagate
        logw = -0.5 * (h + yt ** 2 * np.exp(-h))                      # log p(y_t | h_t)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        vol[t] = np.exp(0.5 * (w @ h))
        ess = 1.0 / np.sum(w ** 2)
        ess_hist.append(ess)
        if ess < N / 2:                                               # systematic resampling
            u = (rng.random() + np.arange(N)) / N
            h = h[np.searchsorted(np.cumsum(w), u)]
            resamples += 1
    return vol, np.array(ess_hist), resamples

vol_pf, ess, nres = bootstrap_pf(y)
print(f"particle filter (N = 2000): mean ESS {ess.mean():.0f}, "
      f"resampled on {nres} of {len(y)} days ({nres / len(y):.0%})")

vol_g = arch_model(y * 100, vol="GARCH", p=1, o=1, q=1).fit(disp="off").conditional_volatility / 100
print(f"correlation with GJR-GARCH conditional vol: {np.corrcoef(vol_pf, vol_g)[0, 1]:.3f}")

fwd = pd.Series(y, index=r.index).rolling(21).std().shift(-21)
roll = pd.Series(y, index=r.index).rolling(21).std()
ok = fwd.notna() & roll.notna()
def qlike(f, a):
    return np.mean(np.log(np.asarray(f) ** 2) + np.asarray(a) ** 2 / np.asarray(f) ** 2)
print(f"QLIKE against forward 21-day realized (lower is better): "
      f"particle filter {qlike(vol_pf[ok.values], fwd[ok]):.4f}, "
      f"GJR-GARCH {qlike(vol_g[ok.values], fwd[ok]):.4f}, "
      f"trailing 21-day {qlike(roll[ok], fwd[ok]):.4f}")
# => particle filter (N = 2000): mean ESS 1815, resampled on 91 of 6410 days (1%)
#    correlation with GJR-GARCH conditional vol: 0.869
#    QLIKE against forward 21-day realized (lower is better): particle filter -8.0610, GJR-GARCH -8.0429, trailing 21-day -7.9008
```

The filter is healthy: 1,815 of 2,000 particles carrying real weight on average, resampling needed on only 1% of days. Its volatility estimate correlates 0.869 with GARCH's — the two models genuinely disagree about the remaining 13%, since one treats volatility as observable-given-the-past and the other as a hidden state with its own noise. On forecast quality, both model-based estimators clearly beat the trailing 21-day window (−8.06 and −8.04 against −7.90), and the particle filter edges GARCH by 0.018 of QLIKE, which is a real but modest difference that a fair reader should discount: the SV parameters here were fixed at plausible values rather than fitted, while GARCH's were estimated by maximum likelihood on this exact sample, so the comparison is if anything tilted *against* the filter. The honest summary is that they are close, and the reason to reach for the particle filter is not a decisive edge in accuracy but access to the whole posterior — the distribution of current volatility, not just its mean, which is what a risk system actually wants.

## Filtered beats smoothed only in the sense that it is real

Now the debt. [Part III's regime lesson](../part-03-statistics/06-bayesian-methods-and-hmms.md) fitted a two-state Gaussian hidden Markov model to SPY returns, sized momentum by the probability of the calm state, and reported Sharpe rising from 0.30 to **1.10** with maximum drawdown shrinking from −43% to −16% — then immediately disowned the table, on the grounds that `predict_proba` returns *smoothed* probabilities computed from the entire sample. A hidden Markov model is a filtering problem with a discrete state, and its forward recursion is the exact analogue of the Kalman predict-update pair:

$$
a_t(j) \;\propto\; p\bigl(y_t \mid s_t = j\bigr)\,\sum_{i} a_{t-1}(i)\,A_{ij},
$$

where the sum is the predict step (push last period's belief through the transition matrix $A$) and the multiplication is the update (reweight by how well each state explains today). Normalizing gives $P(s_t = j \mid \mathcal F_t)$ — filtered, causal, and available in real time. Twenty lines, and it settles the question:

```python
import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

def forward_filter(model, x, target):             # P(state_t = target | y_1..y_t)
    m, v = model.means_.ravel(), model.covars_.ravel()
    B = np.exp(-0.5 * (x[:, None] - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    A, out = model.transmat_, np.zeros(len(x))
    a = model.startprob_ * B[0]
    a /= a.sum()
    out[0] = a[target]
    for t in range(1, len(x)):
        a = (a @ A) * B[t]                        # predict, then update
        a /= a.sum()
        out[t] = a[target]
    return out

hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=500,
                  random_state=42).fit(r.values.reshape(-1, 1))
calm = np.argsort(hmm.covars_.ravel())[0]
p_smooth = pd.Series(hmm.predict_proba(r.values.reshape(-1, 1))[:, calm], index=r.index)
p_filt = pd.Series(forward_filter(hmm, r.values, calm), index=r.index)
print(f"smoothed vs filtered P(calm): correlation {np.corrcoef(p_smooth, p_filt)[0, 1]:.3f}, "
      f"mean gap {np.abs(p_smooth - p_filt).mean():.3f}, max gap {np.abs(p_smooth - p_filt).max():.3f}")

pos = np.sign(r.rolling(252).sum()).shift(1)
for name, s in [("unconditional", (pos * r).dropna()),
                ("P(calm) smoothed", (pos * p_smooth.shift(1) * r).dropna()),
                ("P(calm) filtered", (pos * p_filt.shift(1) * r).dropna())]:
    eq = np.exp(s.cumsum())
    print(f"{name:18s}: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"maxDD {(eq / eq.cummax() - 1).min():.0%}")
# => smoothed vs filtered P(calm): correlation 0.907, mean gap 0.088, max gap 0.809
#    unconditional     : Sharpe 0.30, maxDD -43%
#    P(calm) smoothed  : Sharpe 1.10, maxDD -16%
#    P(calm) filtered  : Sharpe 0.68, maxDD -21%
```

The two probability series correlate 0.907 and differ by 0.088 on an average day — which sounds harmless until you notice the maximum gap of 0.809, because the disagreements are not spread evenly. They cluster precisely at regime transitions, where the smoother, having already seen the crash, marks the state as stressed days before any causal estimator could. That concentrated hindsight is worth **0.42 of Sharpe**: 1.10 collapses to 0.68 with no change but the direction information flows.

One hindsight remains. Those probabilities came from a model whose parameters were estimated on the full sample, so 2008 still informs how the filter reads 2003. Refit each January on data available at the time, filter forward with that vintage's parameters, and the result is the number a trading desk could actually have earned:

```python
import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
px = pd.read_parquet("data/prices.parquet")
r = np.log(px["SPY"]).diff().dropna()

def forward_filter(model, x, target):
    m, v = model.means_.ravel(), model.covars_.ravel()
    B = np.exp(-0.5 * (x[:, None] - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    A, out = model.transmat_, np.zeros(len(x))
    a = model.startprob_ * B[0]
    a /= a.sum()
    out[0] = a[target]
    for t in range(1, len(x)):
        a = (a @ A) * B[t]
        a /= a.sum()
        out[t] = a[target]
    return out

wf = pd.Series(np.nan, index=r.index)
for year in range(2005, 2026):
    train = r[r.index < f"{year}-01-01"]
    test = r[(r.index >= f"{year}-01-01") & (r.index < f"{year + 1}-01-01")]
    if len(test) == 0:
        continue
    m = GaussianHMM(n_components=2, covariance_type="full", n_iter=500,
                    random_state=42).fit(train.values.reshape(-1, 1))
    hist = r[r.index < f"{year + 1}-01-01"]       # filter forward, act only in `test`
    wf.loc[test.index] = pd.Series(forward_filter(m, hist.values,
                                                  np.argsort(m.covars_.ravel())[0]),
                                   index=hist.index).reindex(test.index)

pos = np.sign(r.rolling(252).sum()).shift(1)
overlay = (pos * wf.shift(1) * r).dropna()
base = (pos * r).dropna()
base = base[base.index >= overlay.index[0]]
for name, s in [("walk-forward filtered", overlay), ("unconditional, same window", base)]:
    eq = np.exp(s.cumsum())
    print(f"{name:27s}: Sharpe {np.sqrt(252) * s.mean() / s.std():.2f}, "
          f"maxDD {(eq / eq.cummax() - 1).min():.0%}")
# => walk-forward filtered      : Sharpe 0.53, maxDD -21%
#    unconditional, same window : Sharpe 0.26, maxDD -43%
```

**0.53**, against the 1.10 that Part III refused to bless. Slightly more than half the headline was hindsight — some from smoothing, the rest from full-sample parameters — and stripping it out is what a filtered, walk-forward implementation costs. But read the second line before filing this as another debunking: over the same window the unconditional strategy earned 0.26, so the honest overlay **doubled the Sharpe and halved the drawdown** (−21% against −43%) using nothing but information available on the day. That is the unusual outcome in this course — a seductive result that shrinks by half under scrutiny and is still worth having. The regime structure was real; only its magnitude was inflated, and the instrument that separated the two was the distinction between $\mathcal F_t$ and $\mathcal F_T$.

## The filter fails politely unless you make it fail loudly

Filters break quietly, which is their most dangerous property: a diverged Kalman filter still returns numbers, formatted identically to good ones. Five practices, each of which has already appeared above, are the difference between an estimator and a liability. **Use the Joseph form** for the covariance update and symmetrize with $P \leftarrow \tfrac12(P + P^\top)$ each step; the standard form loses positive-definiteness in finite precision and the failure appears months into a backtest as an impossible negative variance. **Initialize diffusely** ($P_0$ large) rather than confidently, and discard the burn-in period from any likelihood you compare across implementations — the `statsmodels` reconciliation earlier was a 7.83-unit demonstration of what that convention is worth. **Monitor the standardized innovations** $\nu_t/\sqrt{S_t}$: they should be white with unit variance, and when they are not, the state equation is misspecified — this single diagnostic catches EKF divergence, wrong $Q/R$ ratios, and structural breaks alike. **Tune $Q$ and $R$ by maximum likelihood rather than by eye**, but respect the identification problem: only the ratio $q = Q/R$ is well-determined from data in the local-level model, which is precisely why the steady-state formula depends on $q$ alone. And **check the effective sample size** in any particle filter — an ESS that collapses toward one means the filter has silently become a single trajectory, and its confidence intervals are fiction.

The last practice is the one this module argues hardest for, and it is not numerical. Before adopting an adaptive estimator, establish that estimation error is the binding constraint. The hedge-ratio experiment is the cautionary case: a technically flawless filter, correctly implemented and validated against `statsmodels`, applied to a problem whose difficulty lay entirely in execution latency — and it made things worse in exact proportion to how hard it worked.

!!! warning "A filter's job is to remove exactly what a mean-reversion rule wants to trade"
    Well-specified filters produce white innovations. That is the definition of working, and it means the residual of a good filter has no exploitable autocorrelation left in it — the SPY–IVV spread went from a standard deviation of 23.3 bp with autocorrelation +0.82 to 3.3 bp with autocorrelation −0.27, and the strategy built on it went from −0.07 to −0.95. Whenever a filter sits upstream of a signal, ask which of the two you actually want: the state estimate, or the thing the state estimate is designed to throw away.

!!! abstract "Key takeaways"
    - A Kalman filter tracked a hidden random walk to RMSE 0.311 from observations with error 1.018, and its self-reported uncertainty $\sqrt{P} = 0.308$ matched the error it actually achieved — the honesty is the feature.
    - The steady-state gain $K^\ast = \tfrac12(\sqrt{q^2+4q} - q)$ makes EWMA a *derived* estimator: signal-to-noise $q = 0.01$ implies $\lambda = 0.905$ and a 6.9-period half-life, replacing a chosen constant with a fitted one.
    - Hand-rolled and `statsmodels` filters agreed on states to $10^{-8}$ while reporting log-likelihoods 7.83 apart — the entire gap was the diffuse-prior burn-in convention, not a bug.
    - Kalman dynamic hedging made Part IV's pair strictly worse at every setting (gross 1.23 → 1.18 → 0.88 → 0.79; net −0.07 → −1.83), because the filter shrank the tradable spread from 23.3 bp to 3.3 bp and flipped its autocorrelation from +0.82 to −0.27.
    - The fitted hedge ratio never left [0.937, 0.999] with sd 0.0098 — the pair's problem was one-day execution lag (23.0 → 2.8 bp per trip), never a stale hedge, so adaptivity solved a problem that did not exist.
    - The EKF failed on 15 of 20 seeds where the unscented filter failed on 3, with median RMSE 1.375 against 0.862 — linearizing the function is strictly worse than transforming the distribution when curvature meets uncertainty.
    - A 2,000-particle bootstrap filter held mean ESS 1,815 and resampled on 1% of days; its stochastic-volatility estimate correlated 0.869 with GJR-GARCH and scored QLIKE −8.0610 against GARCH's −8.0429 and a trailing window's −7.9008.
    - Part III's 1.10 regime Sharpe was 0.68 filtered and **0.53** under annual refits — but the same honest overlay still doubled the unconditional 0.26 and halved drawdown from −43% to −21%, so the structure was real and only its magnitude was hindsight.

## Where this goes next

The continuous-time limit of the state equations used throughout this module — where $x_t = Fx_{t-1} + w_t$ becomes a stochastic differential equation and the Ornstein–Uhlenbeck process that Part IV fitted by autoregression gets solved properly — is [Stochastic Calculus](03-stochastic-calculus.md), which also derives the exact discretization connecting the two. Inventory in a market-making book is a filtering problem in disguise, and [Market Making](12-market-making.md) treats the quoting decision as control of a state whose evolution the maker only partly observes. Closer to home, the regime overlay rebuilt here belongs in [Part IV's walk-forward machinery](../part-04-strategy-development/08-validation-and-overfitting.md) before it is trusted with capital, and [Part VII's meta-labeling](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) is the machine-learning version of the same conditional-sizing idea — a second model deciding when the first one deserves to be believed.
