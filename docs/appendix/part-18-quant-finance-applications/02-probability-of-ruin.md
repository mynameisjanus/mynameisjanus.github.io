# Probability of Ruin

[Kelly Criterion](01-kelly-criterion.md) produced a fraction and said nothing about the path taken to collect it. This page produces the path statistic that matters most and finds it depends on one number and nothing else: at a leverage of $c$ times the growth-optimal fraction, the probability that wealth ever falls to a fraction $x$ of where it started is exactly $x^{2/c-1}$ — no Sharpe ratio, no volatility, no horizon. Measured against simulation the identity holds to three decimals, $0.5000$ against $0.5015$ at full Kelly and $0.1250$ against $0.1278$ at half, so full Kelly has an even chance of halving the account and half Kelly cuts the chance of a ninety-percent loss from $0.1000$ to $0.0010$. Two things then break it. The identity assumes a path that cannot skip its barrier, and on a law with an identical $8.49\%$ mean and $7.23\%$ volatility carrying one $6\%$ jump a year, full Kelly wipes the book out entirely on $0.9811$ of twenty-year paths against a diffusion probability of $0.0000$. And the exponent's only input is a drift: a desk sizing at half of its *estimated* Kelly on ten years of history quotes $0.1250$ while the interval it is actually running spans $0.0138$ to $0.2957$.

This page covers ruin as a first-passage probability, the exponential martingale that settles it, the exact power law in the Kelly multiple, the Lundberg exponent that generalizes it and the overshoot that makes the diffusion answer an underestimate, absolute ruin as an event the diffusion assigns probability zero, and the estimation error the exponent inherits from the drift. It solves no discrete two-point boundary value problem and treats no gambler's stake, which is [Random Walks](../part-08-stochastic-processes/11-random-walks.md); it proves no optional stopping theorem and constructs no filtration, which is [Martingales](../part-08-stochastic-processes/10-martingales.md); it derives no reflection principle and no running-maximum law, which is [Brownian Motion](../part-08-stochastic-processes/08-brownian-motion.md); it computes no distribution for the deepest loss below a running peak, which is [Drawdown Probabilities](03-drawdown-probabilities.md); it derives no law for *when* the barrier is reached, which is [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md); it maximizes no growth rate, which is [Kelly Criterion](01-kelly-criterion.md); it fits no tail to the jump it invokes, which is [Extreme Value Theory](13-extreme-value-theory.md); and it never reports a ruin probability without the interval its drift estimate implies.

The trading stake is a sentence in a course lesson that names this page and asks it to justify a claim. [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) prints `full Kelly: growth +17.7%/yr, maxDD -94%` and `half Kelly: growth +13.1%/yr, maxDD -70%` on a vol-targeted book, then writes that "half Kelly surrenders 4.6 points of growth to shrink the ruin to 70%, which is still a number no investor, employer, or spouse survives," and defers the reason to "[Probability of Ruin] and [Drawdown Probabilities] formalize why these paths are unlivable long before they are unprofitable." Section 2 supplies the first half of that formalization, and the answer is sharper than the lesson needed: the ratio of the two loss probabilities is not a matter of degree but a change of exponent, from $1$ to $3$.

## Ruin Is a First-Passage Probability, and an Exponential Martingale Settles It in Two Lines

The gambler's-ruin calculation of [Random Walks](../part-08-stochastic-processes/11-random-walks.md) solved a difference equation on a discrete lattice with an absorbing state at zero. A levered book has neither: wealth is multiplicative, so it never reaches zero by drifting, and the quantity a desk cares about is not extinction but a threshold — the loss past which the business ends. The continuous version has a shorter derivation than the discrete one.

??? note "Proof that log wealth with positive drift falls by $a$ at some finite time with probability $e^{-2ma/v^{2}}$, via an exponential martingale and optional stopping"

    Let $Y_t=mt+vB_t$ be log wealth, with $m>0$ and $v>0$, and let $\tau_a=\inf\{t:Y_t=-a\}$ for $a>0$. Consider
    $$Z_t=\exp\!\left(-\frac{2m}{v^{2}}Y_t\right)=\exp\!\left(-\frac{2m^{2}}{v^{2}}t-\frac{2m}{v}B_t\right).$$
    Since $\mathbb{E}[e^{\lambda B_t}]=e^{\lambda^{2}t/2}$, taking $\lambda=-2m/v$ gives $\mathbb{E}[Z_t]=e^{-2m^{2}t/v^{2}}e^{2m^{2}t/v^{2}}=1$, and the same computation applied conditionally makes $Z$ a positive martingale with $Z_0=1$.

    Before $\tau_a$ the process satisfies $Y_t>-a$, so $Z_t<e^{2ma/v^{2}}$ and $Z_{t\wedge\tau_a}$ is a bounded martingale. Optional stopping — the theorem of [Martingales](../part-08-stochastic-processes/10-martingales.md), with boundedness supplying the hypothesis it needs — gives $\mathbb{E}[Z_{t\wedge\tau_a}]=1$ for every $t$. Let $t\to\infty$. On $\{\tau_a<\infty\}$ the stopped value is $Z_{\tau_a}=e^{2ma/v^{2}}$, attained *exactly*, because a continuous path reaching level $-a$ is at level $-a$. On $\{\tau_a=\infty\}$ the strong law forces $Y_t/t\to m>0$, so $Y_t\to\infty$ and $Z_t\to0$. Therefore
    $$1=\mathbf{P}(\tau_a<\infty)\,e^{2ma/v^{2}},\qquad \mathbf{P}(\tau_a<\infty)=e^{-2ma/v^{2}},$$
    and equivalently $-\inf_t Y_t$ is exponentially distributed with rate $2m/v^{2}$.

    **The load-bearing hypothesis is path continuity, and it enters at exactly one point: the claim that $Z_{\tau_a}$ equals $e^{2ma/v^{2}}$ rather than exceeding it. A process that can cross the barrier by jumping lands strictly below $-a$, the equality becomes an inequality in the direction that makes ruin more likely, and section 3 measures how much more.**

## At c Times Kelly the Probability of Ever Losing a Given Fraction Is That Fraction Raised to Two Over c Minus One

The exponent $2m/v^{2}$ looks as though it depends on the strategy's drift and volatility separately. Written in terms of the Kelly multiple it does not depend on either.

??? note "Proof that at leverage $c f^{*}$ the probability of ever reaching a fraction $x$ of starting wealth is $x^{2/c-1}$, a function of $c$ alone"

    At leverage $f$ on a diffusion with drift $\mu$ and volatility $\sigma$, log wealth has drift $m=f\mu-\tfrac{1}{2}f^{2}\sigma^{2}$ and volatility $v=f\sigma$, both from [Kelly Criterion](01-kelly-criterion.md). Substituting $f=cf^{*}=c\mu/\sigma^{2}$,
    $$m=\frac{\mu^{2}}{\sigma^{2}}\left(c-\frac{c^{2}}{2}\right),\qquad v^{2}=\frac{c^{2}\mu^{2}}{\sigma^{2}},\qquad \frac{2m}{v^{2}}=\frac{2c-c^{2}}{c^{2}}=\frac{2}{c}-1.$$
    Reaching a fraction $x<1$ of starting wealth means $Y$ falling by $a=\log(1/x)$, so
    $$\mathbf{P}\!\left(\inf_t W_t\le xW_0\right)=e^{-(2/c-1)\log(1/x)}=x^{\,2/c-1}.$$
    Both $\mu$ and $\sigma$ have cancelled. So has the horizon, since the statement is about the infimum over all time.

    Four readings follow. At $c=1$ the exponent is $1$ and the probability of ever reaching a fraction $x$ *is* $x$ — full Kelly halves the account with probability one half and loses ninety percent of it with probability one tenth. At $c=1/2$ the exponent is $3$, so the same two probabilities become $1/8$ and $1/1000$: cubing is what buys the safety, and [Kelly Criterion](01-kelly-criterion.md) already priced it at one quarter of the growth rate. At $c=2$ the exponent is $0$ and the probability is $1$ at every level, which is the same $c=2$ that grows at zero. Beyond $c=2$ the log drift is negative and ruin at every level is certain for a second reason.

    **The load-bearing quantity is the ratio $c$, and it is the one thing on this page that is not observable. Leverage is chosen and known exactly; $f^{*}$ contains $\mu$; so the exponent is a random variable whose distribution is the drift's, which is section 4.**

```python
import numpy as np

rng = np.random.default_rng(18021)
SHARPE, VOL, YEARS, PATHS, STEPS = 1.174, 0.0723, 200, 40_000, 200
LEVELS = (0.50, 0.25, 0.10, 0.01)
mu, sd = SHARPE * VOL, VOL
f_star = mu / sd ** 2
dt = 1 / STEPS


def running_min(m, v):
    """Exact all-time minimum of log wealth over [0, YEARS], using the
    Brownian-bridge minimum between sampled points so nothing is missed."""
    lo = np.zeros(PATHS)
    y = np.zeros(PATHS)
    for _ in range(YEARS * STEPS):
        step = m * dt + v * np.sqrt(dt) * rng.standard_normal(PATHS)
        y1 = y + step
        u = rng.random(PATHS)
        bridge = 0.5 * ((y + y1) - np.sqrt((y1 - y) ** 2 - 2 * v ** 2 * dt * np.log(u)))
        lo = np.minimum(lo, bridge)
        y = y1
    return lo


print(f"  a book at Sharpe {SHARPE} run at c times the Kelly fraction {f_star:.2f}x. P(wealth ever"
      f" falls to a fraction x of its start) = x^(2/c - 1), a function of c alone -- neither the"
      f" Sharpe ratio nor the volatility appears. {PATHS:,} paths x {YEARS} years, exact bridge minima")
print("     c      leverage   exponent 2/c-1   " + "".join(
    f"P(ever <= {x:.0%}) pred / meas   ".replace("<= 1%", "<=  1%") for x in LEVELS))
for c in (0.25, 0.50, 1.00, 1.50, 2.00):
    f = c * f_star
    m, v = f * mu - 0.5 * f ** 2 * sd ** 2, f * sd
    lo = running_min(m, v)
    cells = []
    for x in LEVELS:
        pred = x ** (2 / c - 1)
        cells.append(f"{pred:11.4f} / {np.mean(lo <= np.log(x)):.4f}   ")
    print(f"    {c:4.2f}   {f:7.2f}x   {2 / c - 1:14.2f}   " + "".join(cells))
# =>   a book at Sharpe 1.174 run at c times the Kelly fraction 16.24x. P(wealth ever falls to a fraction x of its start) = x^(2/c - 1), a function of c alone -- neither the Sharpe ratio nor the volatility appears. 40,000 paths x 200 years, exact bridge minima
#         c      leverage   exponent 2/c-1   P(ever <= 50%) pred / meas   P(ever <= 25%) pred / meas   P(ever <= 10%) pred / meas   P(ever <=  1%) pred / meas   
#        0.25      4.06x             7.00        0.0078 / 0.0077        0.0001 / 0.0001        0.0000 / 0.0000        0.0000 / 0.0000   
#        0.50      8.12x             3.00        0.1250 / 0.1278        0.0156 / 0.0162        0.0010 / 0.0011        0.0000 / 0.0000   
#        1.00     16.24x             1.00        0.5000 / 0.5015        0.2500 / 0.2504        0.1000 / 0.0979        0.0100 / 0.0094   
#        1.50     24.36x             0.33        0.7937 / 0.7910        0.6300 / 0.6252        0.4642 / 0.4598        0.2154 / 0.2142   
#        2.00     32.48x             0.00        1.0000 / 0.9838        1.0000 / 0.9674        1.0000 / 0.9451        1.0000 / 0.8916   
```

The law is confirmed in every cell where the horizon is long enough to stand in for "ever": $0.5000$ against $0.5015$, $0.2500$ against $0.2504$, $0.1000$ against $0.0979$ and $0.0100$ against $0.0094$ at full Kelly, and $0.1250$ against $0.1278$, $0.0156$ against $0.0162$ and $0.0010$ against $0.0011$ at half. The last row is the exception and is informative for it: at $c=2$ the predicted probability is $1.0000$ at every level and two hundred years of simulation returns $0.9838$, $0.9674$, $0.9451$ and $0.8916$, because certainty here is an eventual property and two centuries is not eventually.

What the table settles is the shape of the trade-off the course lesson stated without one. Moving from full to half Kelly costs a quarter of the growth rate and divides the probability of a ninety-percent loss by one hundred, from $0.1000$ to $0.0010$. Moving from full to a quarter of Kelly costs $56\%$ of the growth and divides that probability by more than ten thousand. **The growth rate is quadratic in the Kelly multiple and the ruin probability is exponential in its reciprocal, which is the entire argument for fractional Kelly stated in two sentences and without reference to estimation error.**

## The Identity Assumes a Path That Cannot Skip Its Barrier, and One Jump a Year Wipes Out Ninety-Eight Percent of Books

The proof in section 1 used continuity exactly once, and a return series does not have it. The general result names the price.

??? note "Proof that the exponent solves a cumulant equation, that $2m/v^{2}$ is its value when the third and higher cumulants vanish, and that negative skew lowers it while absolute ruin becomes possible at all"

    Let $L$ be the log-wealth increment per unit time, with cumulant generating function $\kappa(\theta)=\log\mathbb{E}[e^{\theta L}]$. The **Lundberg exponent** $R>0$ is the positive root of $\kappa(-R)=0$, which makes $e^{-RY_t}$ a martingale by the same computation as before, and the optional-stopping argument then gives
    $$\mathbf{P}(\tau_a<\infty)\le e^{-Ra},$$
    an inequality rather than an identity because a path that jumps past the barrier stops at $Y_{\tau_a}<-a$, so $\mathbb{E}[e^{-RY_{\tau_a}}]>e^{Ra}$. The gap is the overshoot.

    For a Gaussian increment, $\kappa(\theta)=m\theta+\tfrac{1}{2}v^{2}\theta^{2}$ and $\kappa(-R)=0$ gives $R=2m/v^{2}$ exactly, recovering section 1 with no overshoot, since the path is continuous. In general $\kappa(-R)=-mR+\tfrac{1}{2}v^{2}R^{2}-\tfrac{1}{6}\gamma_3R^{3}+\cdots$ with $\gamma_3$ the third cumulant. A negatively skewed increment has $\gamma_3<0$, which makes $\kappa(-R)$ larger at every $R>0$ and therefore moves its positive root *down*. A smaller $R$ means a larger bound: the same first two moments with a left tail attached give a strictly higher probability of ruin at every barrier.

    Absolute ruin is a separate event that the two frameworks disagree about categorically. At leverage $f$, one period with return $r\le-1/f$ takes wealth to zero or below in a single step, and no subsequent recovery exists. Under a continuous path $\mathbf{P}(W_t\le0)=0$ for all $t$ and for every $f$, so the diffusion model does not merely underestimate this event, it excludes it. Under any law whose support reaches $-1/f$ the probability is positive and grows with $f$, which is the bound [Kelly Criterion](01-kelly-criterion.md) computes as $1/\lvert x_{\min}\rvert$ arriving as a statement about time rather than about growth.

    **The load-bearing distinction is between crossing a barrier and skipping it. Every quantity on this page is defined by where a path first goes below a level; the diffusion says it arrives there exactly, and a jump process says it arrives somewhere strictly worse, so the diffusion's answer is not an approximation from an unknown direction but an underestimate from a known one.**

```python
import numpy as np

rng = np.random.default_rng(18023)
SHARPE, VOL, YEARS, PATHS, D = 1.174, 0.0723, 20, 40_000, 252
JUMP, P_JUMP = 0.06, 1 / D
mu_d, sd_d = SHARPE * VOL / D, VOL / np.sqrt(D)
f_star = mu_d / sd_d ** 2
M = mu_d + P_JUMP * JUMP                                   # ordinary-day mean and vol chosen so
S = np.sqrt(sd_d ** 2 - JUMP ** 2 * P_JUMP * (1 - P_JUMP))  # the mixture matches both moments


def paths(f, jumps):
    """Minimum of wealth relative to its start, and whether the book was wiped out."""
    lo, w, dead = np.ones(PATHS), np.ones(PATHS), np.zeros(PATHS, bool)
    for _ in range(YEARS * D):
        if jumps:
            r = M + S * rng.standard_normal(PATHS) - JUMP * (rng.random(PATHS) < P_JUMP)
        else:
            r = mu_d + sd_d * rng.standard_normal(PATHS)
        gross = 1 + f * r
        dead |= gross <= 0
        w = np.where(dead, 0.0, w * np.maximum(gross, 0.0))
        lo = np.minimum(lo, w)
    return lo, dead


print(f"  {PATHS:,} paths x {YEARS} years of daily returns at c times Kelly. Both laws carry an"
      f" annualized mean of {SHARPE * VOL:.2%} and volatility of {VOL:.2%}; the second moves"
      f" variance into a {JUMP:.0%} jump arriving once a year, so 1/|jump| = {1 / JUMP:.1f}x")
print("     c      leverage   Gaussian: P(<=50%)   P(<=10%)   P(wiped out)"
      "   jump law: P(<=50%)   P(<=10%)   P(wiped out)")
for c in (0.25, 0.50, 0.75, 1.00, 1.25):
    f = c * f_star
    g_lo, g_dead = paths(f, False)
    j_lo, j_dead = paths(f, True)
    print(f"    {c:4.2f}   {f:7.2f}x   {np.mean(g_lo <= 0.50):18.4f}   {np.mean(g_lo <= 0.10):8.4f}"
          f"   {g_dead.mean():12.4f}   {np.mean(j_lo <= 0.50):18.4f}   {np.mean(j_lo <= 0.10):8.4f}"
          f"   {j_dead.mean():12.4f}")
# =>   40,000 paths x 20 years of daily returns at c times Kelly. Both laws carry an annualized mean of 8.49% and volatility of 7.23%; the second moves variance into a 6% jump arriving once a year, so 1/|jump| = 16.7x
#         c      leverage   Gaussian: P(<=50%)   P(<=10%)   P(wiped out)   jump law: P(<=50%)   P(<=10%)   P(wiped out)
#        0.25      4.06x               0.0065     0.0000         0.0000               0.0406     0.0001         0.0000
#        0.50      8.12x               0.1197     0.0009         0.0000               0.3158     0.0392         0.0000
#        0.75     12.18x               0.2990     0.0213         0.0000               0.6892     0.3942         0.0000
#        1.00     16.24x               0.4827     0.0988         0.0000               0.9991     0.9986         0.9811
#        1.25     20.30x               0.6378     0.2442         0.0000               1.0000     1.0000         1.0000
```

The two halves of the table are the same book. Identical annualized mean, identical annualized volatility, identical leverage in each row; the only difference is that one law delivers its variance smoothly and the other saves a little of it for one day a year. At three-quarters of Kelly the probability of ever losing ninety percent goes from $0.0213$ to $0.3942$, a factor of $18.5$. At full Kelly it goes from $0.0988$ to $0.9986$, and the column the diffusion cannot populate at all reads $0.9811$: within twenty years, ninety-eight percent of these books are not down, they are gone, because $16.24\times$ leverage against a $6\%$ move is a loss of $97\%$ in an afternoon and the next one finishes it.

The discontinuity between $c=0.75$ and $c=1.00$ is the mechanism, not noise. The jump is $6\%$ and $1/0.06=16.7\times$, so leverage of $12.18\times$ survives a jump with a bad day attached while $16.24\times$ does not. **The ruin probability as a function of leverage is smooth under the diffusion and has a cliff under any law with a floor, and the cliff sits where the first two moments say nothing is happening.**

!!! note "Gambler's ruin, absolute ruin, the ruin bound and a drawdown limit are four different events, and the one a desk means is almost always the fourth"
    **Gambler's ruin** is the discrete two-point boundary problem of [Random Walks](../part-08-stochastic-processes/11-random-walks.md), where wealth is an integer count of stakes and zero is an absorbing lattice point. **Absolute ruin** is the event $W_t\le0$, which requires a single return past $-1/f$ and which a diffusion assigns probability zero at every leverage. **The ruin bound** is $1/\lvert x_{\min}\rvert$ from [Kelly Criterion](01-kelly-criterion.md), a leverage rather than a probability — the largest bet at which absolute ruin remains impossible against the worst observed outcome. **A drawdown limit** is a threshold on wealth relative to its starting value or its running maximum, and the event of touching it is what this page computes and what ends a real business, since a book down seventy percent has already lost its investors whether or not any subsequent path exists. The four are routinely collapsed into the word "ruin," and the collapse hides the fact that only the second is a statement about arithmetic while the fourth is a statement about people.

## The Formula's Only Input Is a Ratio Nobody Can Estimate

Section 2 proved that the exponent depends on $c=f/f^{*}$ and on nothing else, which sounds like a simplification and is the opposite of one. Leverage is chosen and known to machine precision; $f^{*}=\mu/\sigma^{2}$ is not. The whole formula therefore inherits the drift's standard error, and [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md) proves that this error depends only on the calendar span, so it cannot be reduced by sampling more often.

```python
import numpy as np

rng = np.random.default_rng(18025)
SHARPE, VOL, REPS, D, LEVEL, TARGET = 1.174, 0.0723, 200_000, 252, 0.50, 0.50
mu, sd = SHARPE * VOL, VOL
f_star = mu / sd ** 2
quoted = LEVEL ** (2 / TARGET - 1)

print(f"  a desk with T years of history sizes at {TARGET:.0%} of its *estimated* Kelly fraction and"
      f" quotes P(ever losing half) = {LEVEL}^(2/{TARGET} - 1) = {quoted:.4f}. The true law is"
      f" Sharpe {SHARPE} at {VOL:.2%} vol; below, what it was actually running. {REPS:,} replications")
print("     years   SE(Sharpe)   P(mu-hat <= 0)   true c: median    5th   95th"
      "   true P(ever lose half): median     5th    95th   P(true c >= 1)")
for T in (1, 3, 5, 10, 25):
    n = T * D
    x = mu / D + sd / np.sqrt(D) * rng.standard_normal((REPS, n))
    f = TARGET * x.mean(1) / x.var(1)
    c = f / f_star
    ok = c > 0
    p = np.where(ok, LEVEL ** (2 / np.where(ok, c, 1) - 1), np.nan)
    q = lambda a, z: np.nanpercentile(a, z)
    print(f"    {T:5d}   {1 / np.sqrt(T):10.3f}   {np.mean(~ok):14.4f}   {q(c, 50):15.3f}"
          f"   {q(c, 5):6.3f} {q(c, 95):6.3f}   {q(p, 50):30.4f}   {q(p, 5):5.4f}  {q(p, 95):6.4f}"
          f"   {np.mean(ok & (c >= 1)):14.4f}")
# =>   a desk with T years of history sizes at 50% of its *estimated* Kelly fraction and quotes P(ever losing half) = 0.5^(2/0.5 - 1) = 0.1250. The true law is Sharpe 1.174 at 7.23% vol; below, what it was actually running. 200,000 replications
#         years   SE(Sharpe)   P(mu-hat <= 0)   true c: median    5th   95th   true P(ever lose half): median     5th    95th   P(true c >= 1)
#            1        1.000           0.1193             0.503   -0.201  1.230                           0.1741   0.0000  0.6646           0.1281
#            3        0.577           0.0216             0.501    0.094  0.914                           0.1306   0.0001  0.4406           0.0235
#            5        0.447           0.0044             0.501    0.187  0.820                           0.1262   0.0016  0.3692           0.0053
#           10        0.316           0.0001             0.501    0.279  0.725                           0.1256   0.0138  0.2957           0.0001
#           25        0.200           0.0000             0.500    0.360  0.642                           0.1253   0.0424  0.2305           0.0000
```

The median column is the trap. At every history length the desk's realized Kelly multiple has a median of $0.500$ or $0.501$ and its ruin probability a median of $0.1253$ to $0.1741$ against a quoted $0.1250$, so a procedure audited on averages passes cleanly. The percentiles say what is actually being run. With ten years of daily data — more than most strategies have — the book described as half Kelly sits between $0.279$ and $0.725$ of Kelly with $90\%$ probability, and its true probability of ever losing half the account lies between $0.0138$ and $0.2957$, a spread of a factor of $21$ around a number quoted to four decimals. Twenty-five years narrows it only to $0.0424$ against $0.2305$, a factor of $5.4$.

At short histories the failure changes character. With one year of data, $11.93\%$ of desks estimate a non-positive drift, so the "half Kelly" prescription is a short position in a book with a positive edge; and $12.81\%$ are running at or above *full* Kelly while believing they are at half. **The exponent $2/c-1$ is exact, and the number substituted into it is a Sharpe ratio divided by a Sharpe ratio, so a ruin probability is exactly as knowable as a drift and no more — which, at ten years and a standard error of $0.316$, is not very.**

## Every Repair Is a Lower Fraction, a Wider Barrier, or an Admission That the Exponent Is Unknown

The three sections fail in three different places and admit three different repairs, none of which is a better estimator. Section 2's exponent is exact and cheap, so the honest use of it is inverse: fix the loss level the business cannot survive and the probability it will tolerate, and solve for the largest $c$, which requires no drift estimate at the tolerance end and puts the whole error in one place. Section 3's cliff is not repairable by lowering $c$ smoothly, because its position is set by the support rather than the moments; the only defence is the leverage cap $1/\lvert x_{\min}\rvert$, and [Extreme Value Theory](13-extreme-value-theory.md) shows that the observed minimum understates the true one. Section 4's spread is irreducible on any history a strategy has, so the repair is to quote the interval rather than the point.

The published figures show the first repair working in reverse. [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) reports that the course book runs at roughly a sixth of Kelly, and calls that "not timidity but the standard price of estimated edges and institutional pain thresholds." At $c=1/6$ the exponent is $11$, and the probability of ever halving the account is $0.5^{11}$, about one in two thousand. That is what a sixth of Kelly buys, and it is a far larger purchase than the roughly $31\%$ of the growth rate it costs.

!!! warning "The ruin probability and the growth rate are computed from the same two numbers, and only one of them is ever printed with an error bar"
    A Sharpe ratio arrives with a standard error because [Part XI](../part-11-parameter-estimation/index.md) insists on it and because every backtest reports one. The Kelly multiple it implies, and therefore the ruin exponent, almost never does — the fraction is quoted as $0.5\times$ or $0.25\times$ as though the denominator were known. **The free diagnostic is the ruin exponent recomputed at the ends of the drift's own confidence interval: substitute $\hat\mu\pm1.96\,\hat\sigma/\sqrt{T}$ into $c=f\hat\sigma^{2}/\hat\mu$ and read $x^{2/c-1}$ three times instead of once.** It requires no new data and no new model, it uses the interval the backtest already computed, and on ten years of a Sharpe-$1.17$ book it turns "an eighth" into "between one in seventy and one in three." The interval is the answer; the point estimate is the part of it that happens to be printed.

## A Probability That Depends on One Number, and the Number Is a Drift

This page established that ruin is a first-passage probability settled by an exponential martingale, with $\mathbf{P}(\tau_a<\infty)=e^{-2ma/v^{2}}$ and path continuity entering at exactly one step; that at $c$ times Kelly the probability of ever reaching a fraction $x$ of starting wealth is exactly $x^{2/c-1}$ with $\mu$, $\sigma$ and the horizon all cancelling, verified at $0.5000$ against $0.5015$, $0.1000$ against $0.0979$, $0.1250$ against $0.1278$ and $0.0010$ against $0.0011$, and returning $0.9838$ rather than $1.0000$ at $c=2$ only because two hundred years is not forever; that continuity is what fails first, so a law with an identical $8.49\%$ mean and $7.23\%$ volatility carrying one $6\%$ jump a year takes the ninety-percent-loss probability from $0.0213$ to $0.3942$ at three-quarters of Kelly and from $0.0988$ to $0.9986$ at full Kelly, while wiping the book out entirely on $0.9811$ of paths against a diffusion probability of exactly zero; and that the exponent's only input is unobservable, so a desk sizing at half of its estimated Kelly on ten years of history quotes $0.1250$ while running a true value between $0.0138$ and $0.2957$, and on one year is above full Kelly $12.81\%$ of the time and short $11.93\%$ of the time.

The symmetry with the previous page is exact and worth stating plainly. [Kelly Criterion](01-kelly-criterion.md) found growth quadratic in $c$, maximized in the interior, with a flat top that makes the optimum undecidable for a century. This page finds ruin exponential in $1/c$, monotone in $c$, with no interior optimum and no ambiguity at all: every reduction in $c$ improves it, immediately and by a measurable factor. So the two curves are not competing estimates of one quantity but a genuinely one-sided trade, and that asymmetry — not estimation error, and not risk aversion — is why nobody sits at the maximum. What neither page has computed is the loss measured from the top rather than from the start, which is the number a tearsheet actually prints and the one an investor actually reacts to, and it is [Drawdown Probabilities](03-drawdown-probabilities.md).

**Ruin is exponential in the reciprocal of a ratio whose numerator is chosen and whose denominator is estimated, so the probability is exact, the input is not, and the printed answer is a statement about the drift wearing a probability's clothes.**
