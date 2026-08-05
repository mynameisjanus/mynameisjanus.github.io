# Hitting and First-Passage Times

[Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) settled the driftless one-sided case and found the strangest possible answer: a stop is hit with probability one, in infinite expected time. This page adds the three ingredients a trade actually has — a drift, a second barrier, and mean reversion — and each one changes the answer qualitatively rather than by a constant. With drift the passage time becomes inverse Gaussian and its mean is finite exactly when the drift points at the barrier. With mean reversion the half-life stops being the waiting time: on the published SPY–IVV fit, a half-life of $3.4$ days goes with an expected first passage from two standard deviations of $7.0$ days, matching the exact integral's $7.1$, and a ninetieth percentile of $13.6$. And with two barriers the race has a closed form under which the reward-to-risk ratio buys nothing at all: at zero drift the probability of reaching the target first is exactly $\mathrm{stop}/(\mathrm{stop}+\mathrm{target})$, measured at $0.5014$, $0.3344$ and $0.2501$ against $0.5000$, $0.3333$ and $0.2500$, with expected profits of $0.0109\%$, $0.0131\%$ and $0.0016\%$. The failure that follows is a clock nobody thinks of as a barrier: a time stop at one and a half half-lives removes $28.61\%$ of the profit of a converging trade.

This page covers the first-passage law under drift and the condition for its mean to exist, the mean first-passage time of an Ornstein–Uhlenbeck process and its separation from the half-life, the two-barrier exit problem for a diffusion with its exit probability and expected duration, and what a deadline does to a right-skewed passage time. It does not derive the reflection principle, the running-maximum law, or the driftless first-passage density whose mean is infinite, all of which are [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md); it solves no discrete gambler's-ruin difference equation, which is [Random Walks](../part-08-stochastic-processes/11-random-walks.md); it computes no probability of ever reaching a level over an unbounded horizon, which is [Probability of Ruin](02-probability-of-ruin.md); it derives no stationary drawdown law, which is [Drawdown Probabilities](03-drawdown-probabilities.md); it fits no Ornstein–Uhlenbeck process to data and tests no spread for stationarity, which is [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md); it develops no Itô calculus, which is [Stochastic Calculus](../../advanced/03-stochastic-calculus.md); it optimizes no execution schedule, which is [Optimal Execution](../../advanced/04-optimal-execution-almgren-chriss.md); and it never treats a decay constant as a waiting time.

The trading stake is a promise a course lesson makes at the moment it introduces the half-life. [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) writes that an Ornstein–Uhlenbeck displacement "decays to half its size in $t_{1/2}=\ln 2/\theta$ — and that single number dictates the trade's entire tempo: how long positions are held, how fast the z-score window may be, how quickly a loss must be recognized as regime change rather than opportunity ([Hitting Times] makes the waiting-time math exact)." It then prints `SPY-IVV: AR(1) rho 0.817, half-life 3.4 days` against impostors at $631.0$ and $1{,}058.4$ days. Section 2 makes that math exact and finds the tempo is not the half-life: at the entry threshold the lesson trades, the expected wait is twice it and the ninetieth percentile is four times it.

## First Passage Under Drift Is Inverse Gaussian, and Its Mean Is Finite Exactly When the Drift Points at the Barrier

Adding a drift to the process of [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) repairs the pathology that page ends on, but only in one direction, and the asymmetry is the whole content of the result.

??? note "Proof that the first-passage time of a drifted Brownian motion has an inverse Gaussian law, that its mean is $a/m$ when $m>0$, and that the process reaches the barrier with probability $e^{2ma/v^{2}}<1$ when $m<0$"

    Let $Y_t=mt+vB_t$ and $\tau_a=\inf\{t:Y_t=a\}$ with $a>0$. For any $\lambda$, the process $\exp(\lambda Y_t-(\lambda m+\tfrac{1}{2}\lambda^{2}v^{2})t)$ is a martingale. Fix $s>0$ and choose $\lambda$ so that $\lambda m+\tfrac{1}{2}\lambda^{2}v^{2}=s$, taking the positive root
    $$\lambda=\frac{-m+\sqrt{m^{2}+2sv^{2}}}{v^{2}}.$$
    Optional stopping at $\tau_a$, where $Y_{\tau_a}=a$ exactly by continuity, gives the Laplace transform
    $$\mathbb{E}\!\left[e^{-s\tau_a}\right]=\exp\!\left(\frac{a\left(m-\sqrt{m^{2}+2sv^{2}}\right)}{v^{2}}\right),$$
    which is the transform of the inverse Gaussian law with density
    $$f_{\tau_a}(t)=\frac{a}{v\sqrt{2\pi t^{3}}}\exp\!\left(-\frac{(a-mt)^{2}}{2v^{2}t}\right),\qquad t>0.$$
    Letting $s\downarrow0$ gives $\mathbf{P}(\tau_a<\infty)=\exp\left(a(m-\lvert m\rvert)/v^{2}\right)$, which is $1$ for $m>0$ and $e^{2ma/v^{2}}<1$ for $m<0$. Differentiating at $s=0$ for $m>0$ gives $\mathbb{E}[\tau_a]=a/m$ and $\mathrm{Var}(\tau_a)=av^{2}/m^{3}$.

    Three regimes, and only the first is benign. With $m>0$ the barrier is reached almost surely in finite mean time $a/m$, exactly as a deterministic drift would suggest, but with standard deviation $\sqrt{av^{2}/m^{3}}$, so the coefficient of variation is $v/\sqrt{am}$ — the nearer the barrier, the less predictable the wait, in relative terms. With $m=0$ the barrier is reached almost surely and $\mathbb{E}[\tau_a]=\infty$, the density decaying like $t^{-3/2}$, which is the result [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) establishes. With $m<0$ the barrier is often never reached at all, and the mean is infinite because of the paths that do reach it late rather than because of the ones that do not.

    **The load-bearing hypothesis is the sign of the drift relative to the barrier, and it is not a matter of degree. A profit target on a book with a positive edge has a finite expected wait; the same target on the same book after the edge decays to zero has an infinite one, with no intermediate regime and no continuity in the answer as $m\downarrow0$.**

## A Half-Life Is the Decay of an Expectation, Not the Time to Arrive, and at Two Standard Deviations It Understates the Wait by a Factor of Two

For a mean-reverting spread the natural quantity to quote is $\ln 2/\theta$, and it is quoted everywhere. It answers a question about a conditional expectation. The question a trade asks is about a first passage, and the two answers differ by a factor that grows with the entry threshold.

??? note "Proof that the mean first-passage time of an Ornstein–Uhlenbeck process solves a second-order ordinary differential equation whose solution depends on the entry point in stationary units, while the half-life does not depend on it at all"

    For the process $dX_t=-\theta X_t\,dt+\sigma\,dW_t$, the conditional mean obeys $\mathbb{E}[X_t\mid X_0=x]=xe^{-\theta t}$, which halves at $t_{1/2}=\ln 2/\theta$ for every starting value $x$ and every $\sigma$. That is the half-life, and both facts about it — independence of $x$ and independence of $\sigma$ — should already be suspicious, because a first passage cannot be independent of how far away the target is.

    The mean first-passage time $T(x)$ to the level $0$ from $x>0$ satisfies the backward Kolmogorov equation
    $$-\theta x\,T'(x)+\frac{\sigma^{2}}{2}T''(x)=-1,\qquad T(0)=0,$$
    with $T'$ bounded at infinity. Measuring $x$ in units of the stationary standard deviation $s=\sigma/\sqrt{2\theta}$ reduces this to $T''-xT'=-1/\theta$, whose solution is
    $$T(z)=\frac{1}{\theta}\int_{0}^{z}e^{y^{2}/2}\sqrt{2\pi}\,\Phi(-y)\,dy,$$
    a quantity that depends on the entry point $z$ and on $\theta$ only through the overall factor $1/\theta$. Since $\ln 2/\theta$ carries the same factor, the *ratio* $T(z)/t_{1/2}$ is a pure function of $z$: it is about $1.3$ at one standard deviation, $2.0$ at two and $2.6$ at three, and it grows without bound as the entry threshold does.

    **The load-bearing distinction is between a decay constant and a passage time. The half-life is a property of the drift coefficient alone and contains no information about the noise, so it cannot answer a question whose answer depends on how far the process must travel; the ratio above is the correction, and it is a number the entry rule determines rather than the fit.**

```python
import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(18041)
RHO, PATHS, SUB = 0.817, 200_000, 8                     # SPY-IVV daily AR(1) from Part IV
THETA = -np.log(RHO)                                    # per day
HALF = np.log(2) / THETA
dt = 1 / SUB
VAR = 2 * THETA * dt                                    # local variance of one step


def mfpt(z):
    """Mean first passage to the mean from z stationary sds, solving
    T'' - x T' = -1 with T(0) = 0; answer in units of 1/theta."""
    return integrate.quad(
        lambda y: np.exp(y ** 2 / 2) * np.sqrt(2 * np.pi) * stats.norm.sf(y), 0.0, z)[0]


def simulate(z, horizon):
    x, t = np.full(PATHS, float(z)), np.full(PATHS, np.inf)
    live = np.ones(PATHS, bool)
    for k in range(int(horizon * SUB)):
        nxt = x + -THETA * x * dt + np.sqrt(VAR) * rng.standard_normal(PATHS)
        # exact bridge probability that the path touched zero between the two endpoints
        u = rng.random(PATHS)
        crossed = (nxt <= 0) | (u < np.exp(-2 * np.maximum(x, 0) * np.maximum(nxt, 0) / VAR))
        done = live & crossed
        t = np.where(done, (k + 1) * dt, t)
        live &= ~done
        x = nxt
    return t


print(f"  an Ornstein-Uhlenbeck spread with the published SPY-IVV daily AR(1) of {RHO}, so"
      f" theta = {THETA:.4f} per day and the half-life is ln2/theta = {HALF:.1f} days. Entering at"
      f" z stationary sds, how long until the spread first touches its mean? {PATHS:,} paths,"
      f" {SUB} steps per day with exact bridge crossings")
print("     entry z   half-life, days   E[first passage]: predicted   measured   ratio to half-life"
      "   median   90th pct   P(open at 3 half-lives)")
for z in (1.0, 1.5, 2.0, 2.5, 3.0):
    t = simulate(z, 40 * HALF)
    fin = t[np.isfinite(t)]
    pred = mfpt(z) / THETA
    print(f"    {z:7.1f}   {HALF:15.1f}   {pred:27.1f}   {fin.mean():8.1f}"
          f"   {fin.mean() / HALF:19.2f}   {np.median(fin):6.1f}"
          f"   {np.percentile(fin, 90):9.1f}   {np.mean(t > 3 * HALF):24.4f}")
# =>   an Ornstein-Uhlenbeck spread with the published SPY-IVV daily AR(1) of 0.817, so theta = 0.2021 per day and the half-life is ln2/theta = 3.4 days. Entering at z stationary sds, how long until the spread first touches its mean? 200,000 paths, 8 steps per day with exact bridge crossings
#         entry z   half-life, days   E[first passage]: predicted   measured   ratio to half-life   median   90th pct   P(open at 3 half-lives)
#            1.0               3.4                           4.5        4.5                  1.30      2.9        10.2                     0.0978
#            1.5               3.4                           5.9        5.9                  1.71      4.4        12.1                     0.1459
#            2.0               3.4                           7.1        7.0                  2.05      5.6        13.6                     0.1957
#            2.5               3.4                           8.0        8.0                  2.32      6.6        14.8                     0.2422
#            3.0               3.4                           8.8        8.8                  2.55      7.5        15.6                     0.2883
```

The integral and the simulation agree at every entry level — $4.5$ against $4.5$, $5.9$ against $5.9$, $7.1$ against $7.0$, $8.0$ against $8.0$ and $8.8$ against $8.8$ days — which is worth stating because the agreement required the discrete simulation to account for crossings that happen *between* sampled points. Without the bridge correction the simulated means come out about $0.8$ days too long at every threshold, and the resulting mismatch with the exact answer would have looked like a defect in the theory rather than in the measurement.

The trading content is the ratio column. The published fit gives a half-life of $3.4$ days, and a desk that reads that number as the holding period is understating the wait by $1.30$ at a one-sigma entry, $2.05$ at two and $2.55$ at three. Worse, the distribution is heavily right-skewed: at a two-sigma entry the median is $5.6$ days while the mean is $7.0$ and the ninetieth percentile is $13.6$, four times the half-life. Nearly one trade in five is still open after three half-lives. **The half-life is independent of the entry threshold and the waiting time is not, so a single fitted number cannot set the tempo of a rule whose entry level is a free parameter.**

## The Race Between Two Barriers Has a Closed Form, and Under a Fair Game the Reward-to-Risk Ratio Buys Exactly Nothing

A live trade has a stop and a target, so the relevant object is not a first passage to one level but the race between two. The diffusion version of the calculation [Random Walks](../part-08-stochastic-processes/11-random-walks.md) performs on a lattice answers a folklore question definitively.

??? note "Proof that the probability of reaching the upper barrier first is a ratio of exponentials, that it degenerates to $L/(L+U)$ at zero drift, and that the expected exit time follows from Wald's identity"

    Let $Y_t=mt+vB_t$ start at $0$ with absorbing barriers at $-L$ and $+U$, both positive, and let $\tau$ be the exit time and $p=\mathbf{P}(Y_\tau=U)$. The process $e^{-\theta Y_t}$ with $\theta=2m/v^{2}$ is the martingale of [Probability of Ruin](02-probability-of-ruin.md); optional stopping at $\tau$, legitimate because $Y$ is bounded on $[-L,U]$ until then, gives $1=p\,e^{-\theta U}+(1-p)e^{\theta L}$ and hence
    $$p=\frac{1-e^{\theta L}}{e^{-\theta U}-e^{\theta L}}.$$
    Wald's identity applied to $Y$ itself gives $\mathbb{E}[Y_\tau]=m\,\mathbb{E}[\tau]$, so for $m\neq0$
    $$\mathbb{E}[\tau]=\frac{Up-L(1-p)}{m}.$$

    At $m=0$ both expressions are indeterminate and both limits are elementary. Expanding to first order in $\theta$ gives $p\to L/(L+U)$, and the second Wald identity $\mathbb{E}[Y_\tau^{2}]=v^{2}\mathbb{E}[\tau]$ gives $\mathbb{E}[\tau]=LU/v^{2}$. The first of these is the whole of the matter: under a fair game the probability of reaching the target first is the stop distance over the total distance, so the expected profit is
    $$U\cdot\frac{L}{L+U}-L\cdot\frac{U}{L+U}=0$$
    at *every* reward-to-risk ratio. A rule promising three-to-one wins one time in four, a rule promising one-to-one wins one time in two, and both have expectation zero.

    **The load-bearing consequence is that a reward-to-risk ratio is not a property of a strategy but a reparameterization of its win rate. The two numbers are algebraically linked through the barrier distances, so quoting both as though they were independent evidence of an edge is double-counting one arbitrary choice, and the only quantity that moves the expectation is $m$.**

```python
import numpy as np

rng = np.random.default_rng(18043)
VOL, PATHS, D, SUB, STOP = 0.16, 120_000, 252, 4, 0.04
dt = 1 / (D * SUB)


def race(S, upside):
    """Barriers at -STOP and +upside on a drifted log-price, with exact bridge crossings.
    Ties inside a step go to whichever barrier the step ended nearer."""
    mu, v2 = S * VOL, VOL ** 2 * dt
    y = np.zeros(PATHS)
    won, t = np.zeros(PATHS, bool), np.full(PATHS, np.inf)
    live = np.ones(PATHS, bool)
    for k in range(D * SUB * 2):
        nxt = y + mu * dt + VOL * np.sqrt(dt) * rng.standard_normal(PATHS)
        u1, u2 = rng.random(PATHS), rng.random(PATHS)
        up = (nxt >= upside) | (u1 < np.exp(-2 * (upside - y) * (upside - nxt) / v2))
        dn = (nxt <= -STOP) | (u2 < np.exp(-2 * (y + STOP) * (nxt + STOP) / v2))
        both = up & dn
        up = np.where(both, nxt - (-STOP) > upside - nxt, up)         # nearer barrier wins
        dn = np.where(both, ~up, dn)
        won |= live & up
        t = np.where(live & (up | dn), (k + 1) * dt, t)
        live &= ~(up | dn)
        y = nxt
    closed = np.isfinite(t)
    return won.mean(), (closed & ~won).mean(), t[closed].mean() * D


print(f"  a trade with a {STOP:.0%} stop and a target at a reward-to-risk multiple of it, on a"
      f" {VOL:.0%}-volatility log-price with Sharpe S. Under a fair game the probability of reaching"
      f" the target first is exactly stop/(stop+target), so the expectation is zero at every"
      f" multiple. {PATHS:,} paths, {SUB} steps per day, exact bridge crossings")
print("     Sharpe   reward:risk   target   P(target first): predicted   measured   P(stop first)"
      "   E[exit], days: predicted   measured   expected P&L")
for S in (0.00, 0.50, 1.00):
    for mult in (1.0, 2.0, 3.0):
        up = mult * STOP
        theta = 2 * (S * VOL) / VOL ** 2
        if S == 0:
            pred_p, pred_t = STOP / (STOP + up), STOP * up / VOL ** 2 * D
        else:
            pred_p = (1 - np.exp(theta * STOP)) / (np.exp(-theta * up) - np.exp(theta * STOP))
            pred_t = (up * pred_p - STOP * (1 - pred_p)) / (S * VOL) * D
        p_up, p_dn, t_days = race(S, up)
        print(f"    {S:6.2f}   {mult:9.0f}:1   {up:6.0%}   {pred_p:26.4f}   {p_up:8.4f}"
              f"   {p_dn:13.4f}   {pred_t:26.1f}   {t_days:8.1f}"
              f"   {up * p_up - STOP * p_dn:12.4%}")
# =>   a trade with a 4% stop and a target at a reward-to-risk multiple of it, on a 16%-volatility log-price with Sharpe S. Under a fair game the probability of reaching the target first is exactly stop/(stop+target), so the expectation is zero at every multiple. 120,000 paths, 4 steps per day, exact bridge crossings
#         Sharpe   reward:risk   target   P(target first): predicted   measured   P(stop first)   E[exit], days: predicted   measured   expected P&L
#          0.00           1:1       4%                       0.5000     0.5014          0.4986                         15.8       15.9        0.0109%
#          0.00           2:1       8%                       0.3333     0.3344          0.6656                         31.5       31.5        0.0131%
#          0.00           3:1      12%                       0.2500     0.2501          0.7499                         47.2       47.4        0.0016%
#          0.50           1:1       4%                       0.5622     0.5626          0.4374                         15.7       15.9        0.5012%
#          0.50           2:1       8%                       0.4192     0.4207          0.5794                         32.5       32.6        1.0478%
#          0.50           3:1      12%                       0.3499     0.3509          0.6491                         50.4       50.4        1.6143%
#          1.00           1:1       4%                       0.6225     0.6227          0.3773                         15.4       15.6        0.9819%
#          1.00           2:1       8%                       0.5065     0.5074          0.4926                         32.7       32.8        2.0884%
#          1.00           3:1      12%                       0.4551     0.4545          0.5454                         51.7       51.8        3.2726%
```

Nine configurations, nine matches to three decimals on the probability and to a fraction of a day on the duration, including the zero-drift expected exit time $LU/v^{2}$ at $15.8$, $31.5$ and $47.2$ days against $15.9$, $31.5$ and $47.4$. The three zero-drift rows are the result worth extracting: win rates of $0.5014$, $0.3344$ and $0.2501$ produce expected profits of $0.0109\%$, $0.0131\%$ and $0.0016\%$, which is zero to within the Monte Carlo error, at reward-to-risk ratios of one, two and three to one. Nothing about the ratio created an edge, and nothing about the low win rate destroyed one.

What does move the expectation is in the other six rows, and it moves the win rate rather than the payoff. At a Sharpe of $1.00$ the three-to-one rule wins $0.4545$ of the time instead of $0.2500$, which is where its $3.27\%$ expected profit comes from. Meanwhile the expected duration barely moves with the edge at all — $15.9$, $15.9$ and $15.6$ days across the three Sharpe ratios at one-to-one — because the exit time is set by how far the barriers are, and the drift contributes almost nothing over the horizon it takes to travel four percent. **Barrier placement determines the clock and the win rate simultaneously, and the edge determines only which side of the race is favoured.**

!!! note "A half-life, a mean first-passage time, a median holding period and a time stop are four different clocks, and only the last is chosen deliberately"
    **A half-life** $\ln2/\theta$ is the decay constant of a conditional expectation. It is a property of the fitted process, is independent of the entry threshold and of the noise, and is the number every mean-reversion fit reports. **A mean first-passage time** is the expected wait until the spread actually touches its target from a stated starting displacement; it depends on that displacement and exceeds the half-life by the factor section 2 tabulates. **A median holding period** is what a backtest's trade log reports, and because the passage distribution is right-skewed it sits *below* the mean — $5.6$ days against $7.0$ at a two-sigma entry — so the two most commonly quoted numbers, the half-life and the median hold, bracket the true expected wait from below on both sides. **A time stop** is a barrier in the time coordinate, chosen by a risk committee rather than estimated, and it is the only one of the four that changes the distribution of the other three rather than describing it.

## A Time Stop at One and a Half Half-Lives Removes Twenty-Nine Percent of the Profit

Sections 2 and 3 established that passage times are right-skewed and that barriers set the clock. A time stop is a barrier placed on the clock itself, and because it truncates a right-skewed distribution it removes disproportionately much of what it touches.

```python
import numpy as np

rng = np.random.default_rng(18045)
RHO, PATHS, SUB, ENTRY, STOP_Z = 0.817, 200_000, 8, 2.0, 3.0
THETA = -np.log(RHO)
HALF = np.log(2) / THETA
dt = 1 / SUB
VAR = 2 * THETA * dt


def trade(max_days):
    """Enter a spread at ENTRY sds, take profit at the mean, stop at STOP_Z, and
    abandon the position after max_days. P&L is measured in sds of the spread."""
    x = np.full(PATHS, ENTRY)
    out = np.full(PATHS, np.nan)                       # signed P&L in sds, positive = converged
    live = np.ones(PATHS, bool)
    for k in range(int(max_days * SUB)):
        nxt = x - THETA * x * dt + np.sqrt(VAR) * rng.standard_normal(PATHS)
        u1, u2 = rng.random(PATHS), rng.random(PATHS)
        win = (nxt <= 0) | (u1 < np.exp(-2 * np.maximum(x, 0) * np.maximum(nxt, 0) / VAR))
        lose = (nxt >= STOP_Z) | (u2 < np.exp(-2 * (STOP_Z - x) * (STOP_Z - nxt) / VAR))
        win &= ~lose
        out = np.where(live & win, ENTRY, out)
        out = np.where(live & lose, ENTRY - STOP_Z, out)
        live &= ~(win | lose)
        x = nxt
    out = np.where(live, ENTRY - x, out)               # timed out: closed at the prevailing level
    return out, live


print(f"  a spread entered at {ENTRY:.0f} sds with the target at the mean and the stop at"
      f" {STOP_Z:.0f} sds, on the published SPY-IVV half-life of {HALF:.1f} days. A time stop closes"
      f" whatever is open at the deadline. P&L in sds. {PATHS:,} paths, {SUB} steps per day")
print("     time stop   in half-lives   P(target)   P(stop)   P(timed out)"
      "   E[P&L], sds   E[P&L | timed out]   share of total P&L lost to the deadline")
rows = [(d, *trade(d)) for d in (5, 10, 20, 40, 400)]
base = rows[-1][1].mean()                              # the 400-day row has no effective deadline
for days, out, live in rows:
    lab, hl = ("none", "     --") if days == 400 else (f"{days}d", f"{days / HALF:7.1f}")
    print(f"    {lab:>9}   {hl:>13}   {np.mean(out == ENTRY):9.4f}"
          f"   {np.mean(out == ENTRY - STOP_Z):7.4f}   {live.mean():12.4f}   {out.mean():12.4f}"
          f"   {(out[live].mean() if live.any() else float('nan')):19.4f}"
          f"   {1 - out.mean() / base:39.4f}")
# =>   a spread entered at 2 sds with the target at the mean and the stop at 3 sds, on the published SPY-IVV half-life of 3.4 days. A time stop closes whatever is open at the deadline. P&L in sds. 200,000 paths, 8 steps per day
#         time stop   in half-lives   P(target)   P(stop)   P(timed out)   E[P&L], sds   E[P&L | timed out]   share of total P&L lost to the deadline
#               5d             1.5      0.4324    0.1069         0.4607         1.1345                0.8174                                    0.2861
#              10d             2.9      0.7346    0.1272         0.1382         1.4582                0.8408                                    0.0824
#              20d             5.8      0.8504    0.1370         0.0126         1.5746                0.8616                                    0.0092
#              40d            11.7      0.8620    0.1379         0.0001         1.5862                0.8954                                    0.0018
#             none              --      0.8630    0.1370         0.0000         1.5891                   nan                                    0.0000
```

Without a deadline the trade is good: it converges on $0.8630$ of entries, stops out on $0.1370$, and earns $1.5891$ standard deviations of spread. A five-day time stop — which reads as generous, being nearly one and a half half-lives — cuts the convergence rate to $0.4324$, leaves $0.4607$ of positions open at the deadline, and surrenders $28.61\%$ of the expected profit. Ten days, nearly three half-lives, still costs $8.24\%$. Only at twenty days, close to six half-lives and above the ninetieth percentile of the passage distribution, does the cost fall below one percent.

The mechanism is in the second-to-last column and it is not that the abandoned trades were losers. Positions closed at the five-day deadline carry $+0.8174$ standard deviations of profit on average — they were converging, in the money, and simply had not arrived. A time stop does not cut losses; it cuts *duration*, and because the passage time is right-skewed while the payoff is not, the trades it removes are drawn disproportionately from the slow winners rather than from the fast losers. **The deadline is the only barrier on the page set without reference to the process it constrains, and it is calibrated against the half-life, which section 2 showed is the wrong number by a factor of two before any deadline is chosen at all.**

## Every Repair Is a Wider Barrier, a Longer Deadline, or an Exit Rule That Is Not a Clock

The three failures separate cleanly by which coordinate they live in. Section 2's is an error of translation: the fitted process is correct and the number carried forward from it answers a different question, so the repair is arithmetic — multiply the half-life by the ratio the entry threshold implies, which section 2 tabulates and which requires no new data. Section 3's is not a failure at all but a widely-held illusion, and the repair is to stop reporting the reward-to-risk ratio and the win rate as separate facts, since the barrier distances determine both. Section 4's is a genuine trade-off with no free side: shortening the deadline reduces capital tied up and removes profit, and the exchange rate between the two is the shape of the passage distribution rather than anything a risk committee has access to.

The one repair that costs nothing is to place the deadline in the units the process supplies. A time stop set at the ninetieth percentile of the first-passage distribution — $13.6$ days at a two-sigma entry on this fit, against the $3.4$-day half-life that would naively anchor it — costs under two percent of expected profit by construction, because it is defined to touch one trade in ten.

!!! warning "A time stop is a barrier that is never simulated, because it lives in the coordinate the backtest treats as free"
    Price barriers get tested. A stop-loss is swept across a range, a target is optimized, and both appear in every parameter table. The deadline is typically inherited from an operational constraint — a rebalance schedule, a month-end, a risk-system convention — and enters the backtest as a fact about the calendar rather than as a parameter with a cost. **The free diagnostic is the passage distribution's own quantiles: run the entry rule with no deadline, record the time to resolution, and read the fraction of trades and the fraction of profit that a candidate deadline would truncate.** On this fit the answer is $0.4607$ of trades and $28.61\%$ of profit at five days, against $0.0126$ and $0.92\%$ at twenty. The comparison needs no new model and no new data, only the trade log the backtest already produced with its exit times left in, and it is skipped almost universally because a deadline does not look like a parameter.

## Two Barriers, Two Clocks, and the One Nobody Sets Deliberately

This page established that first passage under drift is inverse Gaussian with mean $a/m$ and variance $av^{2}/m^{3}$, that the barrier is reached almost surely when the drift points at it and with probability $e^{2ma/v^{2}}$ when it does not, and that the driftless case sits between the two with probability one and infinite mean; that the Ornstein–Uhlenbeck mean first-passage time solves a backward equation whose solution depends on the entry threshold while the half-life does not, so on the published SPY–IVV fit a $3.4$-day half-life goes with expected waits of $4.5$, $5.9$, $7.1$, $8.0$ and $8.8$ days at one through three standard deviations, confirmed by simulation at $4.5$, $5.9$, $7.0$, $8.0$ and $8.8$, with a two-sigma median of $5.6$ and a ninetieth percentile of $13.6$; that the two-barrier race has exit probability $(1-e^{\theta L})/(e^{-\theta U}-e^{\theta L})$ and expected duration $(Up-L(1-p))/m$, verified in nine configurations, degenerating at zero drift to $L/(L+U)$ and $LU/v^{2}$ so that win rates of $0.5014$, $0.3344$ and $0.2501$ all produce expected profits indistinguishable from zero; and that a deadline at $1.5$ half-lives removes $28.61\%$ of the expected profit while the positions it closes are ahead by $0.8174$ standard deviations on average.

The relationship to the three pages before it is that this one supplies their missing coordinate. [Kelly Criterion](01-kelly-criterion.md) chose a size, [Probability of Ruin](02-probability-of-ruin.md) computed whether a level is ever reached and [Drawdown Probabilities](03-drawdown-probabilities.md) computed how far down the path goes, and all three are statements about levels with the clock integrated out. Here the level is fixed and the clock is the answer, which is why the driftless case that looked merely curious in [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md) — certain arrival, infinite expected wait — turns into a practical constraint the moment a deadline exists. The next two pages keep the clock and drop the price: an arrival is no longer a level being crossed but an event occurring, and the object of study becomes the times themselves.

**A decay constant, a passage time and a deadline are three numbers in units of days, and the trade is priced by the two that nobody fits.**
