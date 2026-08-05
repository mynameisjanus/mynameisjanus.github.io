# Kelly Criterion

The growth-optimal fraction is the most quoted formula in position sizing and the least often read as what it is: the maximizer of a limit that nobody reaches, computed from two moments on a question the third decides. [Exponentials, Logarithms and Growth](../part-01-mathematical-foundations/07-exponentials-logarithms-growth.md) already derived $f^{*}=\mu/\sigma^{2}$ in one line from volatility drag and named this page. What that derivation cannot say, and what follows below, is the price of being wrong in either direction — a fraction $c$ of Kelly keeps exactly $2c-c^{2}$ of the growth rate, measured at $0.4375$, $0.7500$, $0.9375$ and $0.0000$ against predictions that are those numbers — and the two things the formula does not know. It does not know the support: on six laws with an identical annualized mean of $8.49\%$ and an identical volatility of $7.23\%$, so that $\mu/\sigma^{2}$ returns $16.24\times$ in every one, the largest fraction that can actually be placed falls from $64.87\times$ to $15.78\times$, and at a jump of $8\%$ the formula's own answer is placeable and delivers a growth rate of $-4.18\%$. And it does not know the horizon: three-quarters of Kelly surrenders $4.31$ percentage points of a $68.91\%$ growth rate and needs $125.6$ years of track record before full Kelly is $95\%$ likely to have been the better choice.

This page covers the growth-optimality theorem and the almost-sure dominance it delivers, log utility as its conclusion rather than its premise, the exact parabola in the leverage fraction, the discrete-outcome criterion and the bound the support puts on it, and the horizon at which the theorem's promise becomes decidable. It derives no compound-growth identity and no drag term, which is [Exponentials, Logarithms and Growth](../part-01-mathematical-foundations/07-exponentials-logarithms-growth.md); it solves no stochastic differential equation and develops no lognormal marginal, which is [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md); it proves no law of large numbers, which is [The Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md); it computes no probability of ever losing a given fraction, which is [Probability of Ruin](02-probability-of-ruin.md); it derives no law for the deepest loss along the way, which is [Drawdown Probabilities](03-drawdown-probabilities.md); it puts no error bar on the drift it consumes, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it allocates across no sleeves and sizes no actual book, which is [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md); and it never treats the growth-optimal fraction as a target rather than a ceiling.

The trading stake is a course lesson that computed this formula on the best book in the course and found the answer unplaceable. [Kelly, Volatility Targeting, and Leverage](../../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) prints `Kelly f*  16.84x`, `ruin bound 1/|worst day|  14.88x` and `Kelly exceeds the ruin bound by 1.13x -- full Kelly is unplaceable`, on a surviving book at Sharpe $1.174$ and $7.23\%$ volatility whose worst day was $-6.72\%$. The lesson calls the failure structural rather than a rounding problem and says the formula "does not know it exists." Section 3 is that sentence with the mechanism attached, on a family of laws built so that the first two moments cannot be the explanation.

## Growth Optimality Is a Statement About a Limit, and Log Utility Is Its Conclusion Rather Than Its Premise

The usual objection to Kelly is that it assumes logarithmic utility, and that nobody has logarithmic utility. The objection is misplaced, because the logarithm is not assumed anywhere. It arrives as the only function under which a product of independent factors becomes a sum the law of large numbers can act on, and the criterion that results is about wealth rather than about preference.

??? note "Proof that the log-optimal fixed fraction maximizes the almost-sure growth rate, that its wealth dominates every other fixed fraction, and that the diffusion case is a parabola"

    Let $X_1,X_2,\dots$ be i.i.d. returns and let a fixed fraction $f$ of wealth be committed each period, so that $W_n=W_0\prod_{i=1}^{n}(1+fX_i)$ with $1+fX_i>0$ almost surely. Taking logarithms turns the product into a sum,
    $$\frac{1}{n}\log\frac{W_n}{W_0}=\frac{1}{n}\sum_{i=1}^{n}\log(1+fX_i)\;\xrightarrow{\text{a.s.}}\;G(f):=\mathbb{E}\!\left[\log(1+fX)\right],$$
    by [The Strong Law of Large Numbers](../part-07-asymptotic-theory/02-strong-law-of-large-numbers.md). No utility was chosen; the logarithm is forced by the multiplicative structure, and $G$ is the *growth rate*, not a satisfaction index.

    $G$ is strictly concave, since $G''(f)=-\mathbb{E}[X^{2}/(1+fX)^{2}]<0$ wherever the expectation exists, so it has at most one interior maximizer $f^{*}$, characterized by the first-order condition $G'(f^{*})=\mathbb{E}[X/(1+f^{*}X)]=0$. For any other fixed fraction $f$, strict concavity gives $\delta:=G(f^{*})-G(f)>0$ and
    $$\frac{1}{n}\log\frac{W_n(f^{*})}{W_n(f)}\;\xrightarrow{\text{a.s.}}\;\delta>0,$$
    so $W_n(f^{*})/W_n(f)\to\infty$ almost surely. The dominance is not a matter of expected wealth — expected wealth is maximized at unbounded leverage — but of almost every path eventually and permanently overtaking.

    For a continuously rebalanced diffusion with drift $\mu$ and volatility $\sigma$, the wealth process satisfies $\log W_T=f(\mu T+\sigma B_T)-\tfrac{1}{2}f^{2}\sigma^{2}T$, so $G(f)=f\mu-\tfrac{1}{2}f^{2}\sigma^{2}$ exactly. This is a downward parabola with maximizer $f^{*}=\mu/\sigma^{2}$ and maximum $G(f^{*})=\mu^{2}/2\sigma^{2}=S^{2}/2$, where $S$ is the Sharpe ratio: the growth-optimal book grows at half the square of its Sharpe, whatever its volatility.

    **The load-bearing hypothesis is the limit. Every clause above quantifies over $n\to\infty$ at a fixed fraction, so the theorem transfers no information about any finite horizon, and the rate at which the almost-sure statement becomes visible is not part of the conclusion — it is measured in section 4 and it is slow.**

## The Growth Curve Is a Parabola, So Betting Half of Kelly Keeps Three-Quarters of the Growth and Betting Double Keeps None

Because the diffusion growth rate is exactly quadratic, the cost of missing the optimum has a closed form that depends on nothing but the ratio of the bet to the optimal bet — not on the Sharpe ratio, not on the volatility, not on the horizon.

??? note "Proof that a fraction $c$ of the Kelly bet grows at exactly $(2c-c^{2})$ times the optimal rate, and that overbetting and underbetting by the same multiplicative distance are not symmetric in anything but growth"

    Substituting $f=cf^{*}=c\mu/\sigma^{2}$ into $G(f)=f\mu-\tfrac{1}{2}f^{2}\sigma^{2}$,
    $$G(cf^{*})=c\frac{\mu^{2}}{\sigma^{2}}-\frac{c^{2}}{2}\frac{\mu^{2}}{\sigma^{2}}=\left(c-\frac{c^{2}}{2}\right)\frac{\mu^{2}}{\sigma^{2}}=(2c-c^{2})\,G(f^{*}).$$
    So the fraction of the maximum growth retained is $2c-c^{2}$, a function of $c$ alone. It equals $0.4375$ at $c=1/4$, $0.75$ at $c=1/2$, $0.9375$ at $c=3/4$, $1$ at $c=1$, $0$ at $c=2$, and is negative beyond. Since $2c-c^{2}=1-(1-c)^{2}$, the loss is quadratic in the *distance* from the optimum, which is why the top of the curve is nearly flat: three-quarters of Kelly forfeits one-sixteenth of the growth.

    The symmetry $G(cf^{*})=G((2-c)f^{*})$ says that $1.5\times$ Kelly and $0.5\times$ Kelly grow at the same rate. Nothing else about them is symmetric. The book's volatility is $cf^{*}\sigma$, linear in $c$, so the overbet runs three times the volatility of the underbet for identical growth, and the drawdown statistic below separates them by more than that. The parabola is symmetric; the two ways of being wrong are not.

    **The load-bearing hypothesis is continuous rebalancing, which is what makes $G$ exactly quadratic rather than quadratic to second order. Once the position is adjusted at discrete times against returns of finite size, $2c-c^{2}$ is an approximation whose error is a statement about the third moment and beyond — which is section 3.**

```python
import numpy as np

rng = np.random.default_rng(18011)
SHARPE, VOL, YEARS, PATHS, D = 1.174, 0.0723, 20, 4_000, 252
mu, sd = SHARPE * VOL, VOL                       # annualized arithmetic drift and vol
f_star = mu / sd ** 2

dt = 1 / D
dX = mu * dt + sd * np.sqrt(dt) * rng.standard_normal((PATHS, YEARS * D))
X = np.cumsum(dX, axis=1)                        # one driving path set, shared by every fraction
t = np.arange(1, YEARS * D + 1) * dt

print(f"  a book at Sharpe {SHARPE} and {VOL:.2%} volatility, so the growth-optimal leverage is"
      f" f* = mu/sigma^2 = {f_star:.2f}x. {PATHS:,} continuously rebalanced paths x {YEARS} years,"
      f" every fraction driven by the same Brownian increments")
print("     f/f*   leverage   growth kept, 2c-c^2   predicted growth   realized growth"
      "   book vol   median maxDD   P(beats f*)")
base = f_star * X[:, -1] - 0.5 * f_star ** 2 * sd ** 2 * YEARS
for c in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
    f = c * f_star
    logW = f * X - 0.5 * f ** 2 * sd ** 2 * t
    g = f * mu - 0.5 * f ** 2 * sd ** 2
    run_max = np.maximum.accumulate(logW, axis=1)
    dd = np.expm1(logW - run_max).min(axis=1)
    print(f"    {c:5.2f}   {f:7.2f}x   {2 * c - c ** 2:19.4f}   {g:16.2%}"
          f"   {np.median(logW[:, -1]) / YEARS:15.2%}   {f * sd:8.1%}"
          f"   {np.median(dd):12.1%}   {np.mean(logW[:, -1] > base):11.4f}")
# =>   a book at Sharpe 1.174 and 7.23% volatility, so the growth-optimal leverage is f* = mu/sigma^2 = 16.24x. 4,000 continuously rebalanced paths x 20 years, every fraction driven by the same Brownian increments
#         f/f*   leverage   growth kept, 2c-c^2   predicted growth   realized growth   book vol   median maxDD   P(beats f*)
#         0.25      4.06x                0.4375             30.15%            30.12%      29.3%         -43.9%        0.0250
#         0.50      8.12x                0.7500             51.69%            51.63%      58.7%         -71.6%        0.0970
#         0.75     12.18x                0.9375             64.61%            64.53%      88.0%         -87.5%        0.2587
#         1.00     16.24x                1.0000             68.91%            68.81%     117.4%         -95.5%        0.0000
#         1.25     20.30x                0.9375             64.61%            64.47%     146.8%         -98.8%        0.2465
#         1.50     24.36x                0.7500             51.69%            51.53%     176.1%         -99.8%        0.0835
#         2.00     32.48x                0.0000              0.00%            -0.21%     234.8%        -100.0%        0.0043
```

The identity holds to the second decimal in every row: predicted growth against realized growth is $30.15\%$ against $30.12\%$, $51.69\%$ against $51.63\%$, $64.61\%$ against $64.53\%$ and $68.91\%$ against $68.81\%$, and the ratio of each row's growth to the optimum reproduces $2c-c^{2}$ exactly. At $c=2$ the predicted growth rate is $0.00\%$ and the realized one is $-0.21\%$: a book running twice the growth-optimal leverage compounds at the rate of holding cash, in exchange for $234.8\%$ annualized volatility.

The last two columns are the reason no desk sits at $c=1$. Full Kelly's median maximum drawdown over twenty years is $-95.5\%$, and half Kelly's is $-71.6\%$ — which are, within a point, the $94\%$ and $70\%$ that [Position Sizing and Risk Budgeting](../../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) measured on a real book by direct search. Meanwhile the growth ordering is barely detectable: three-quarters of Kelly is behind full Kelly on only $74\%$ of paths, so a quarter of the time twenty years of history says the *smaller* bet was better. **The parabola's flatness at the top and the drawdown column's steepness are the same fact seen twice, and only one of them appears in the formula.**

## The Formula Is a Diffusion Result, and One Jump a Year Moves Its Answer Past the Placeable Maximum

$\mu/\sigma^{2}$ is the second-order truncation of a criterion that has an exact form. The truncation discards every moment above the second, and one of the things it discards is whether the return distribution has a floor at all.

??? note "Proof that the exact fraction solves $\mathbb{E}[X/(1+fX)]=0$, is strictly below $1/\lvert x_{\min}\rvert$, and that $\mu/\sigma^{2}$ is its second-order approximation and therefore blind to the support"

    With returns bounded below by $x_{\min}<0$ attained with positive probability, $G(f)=\mathbb{E}[\log(1+fX)]$ is finite exactly on $f\in[0,1/\lvert x_{\min}\rvert)$ and $G(f)\to-\infty$ as $f\uparrow1/\lvert x_{\min}\rvert$, since the worst outcome drives $\log(1+fx_{\min})$ to $-\infty$ on an event of positive probability. $G$ is continuous and strictly concave with $G'(0)=\mathbb{E}[X]>0$, so the maximizer is interior and solves
    $$G'(f)=\mathbb{E}\!\left[\frac{X}{1+fX}\right]=0,\qquad 0<f_{\text{exact}}<\frac{1}{\lvert x_{\min}\rvert}.$$
    The upper bound is strict and is a property of the *support* alone: it is the leverage at which one occurrence of the worst outcome takes wealth to zero, and no criterion that is finite can recommend it.

    Expanding the integrand for small $fX$ gives $\log(1+fX)=fX-\tfrac{1}{2}f^{2}X^{2}+O(f^{3}X^{3})$, hence $G(f)\approx f\mu-\tfrac{1}{2}f^{2}\mathbb{E}[X^{2}]$ and a maximizer $\mu/\mathbb{E}[X^{2}]$, which is $\mu/\sigma^{2}$ once $\mu^{2}$ is dropped against $\sigma^{2}$. Every moment above the second has been discarded, and $x_{\min}$ enters none of the two that survive.

    **The load-bearing consequence is a mismatch of information. The bound $1/\lvert x_{\min}\rvert$ is determined entirely by the support and not at all by the first two moments; the formula is determined entirely by the first two moments and not at all by the support. So the formula cannot detect the one condition that decides whether its own answer can be placed, and a family of laws sharing $\mu$ and $\sigma$ can carry any bound whatsoever.**

That family is constructible, and it is the exhibit. Six laws are standardized to the same annualized mean and volatility, with variance progressively moved out of ordinary days and into a single jump arriving once a year:

```python
import numpy as np
from scipy import optimize

rng = np.random.default_rng(18013)
SHARPE, VOL, N, D = 1.174, 0.0723, 5_000, 252
mu_d, sd_d = SHARPE * VOL / D, VOL / np.sqrt(D)
K = round(N / D)                                        # one jump day per year, exactly
Z = rng.standard_normal(N)
WHERE = rng.permutation(N)[:K]

print(f"  {N:,} days, standardized so the sample mean and variance are identical in every row"
      f" -- {SHARPE * VOL:.2%} and {VOL:.2%} annualized -- while {K} of the days carry a jump"
      f" and the rest are correspondingly quieter. f* = mu/sigma^2 therefore reads {mu_d / sd_d ** 2:.2f}x"
      f" every row")
print("     jump   ann. mean   ann. vol   worst day   ruin bound 1/|worst|   exact f   f*/bound"
      "   growth at exact f   growth at f*   growth at half of exact f")
for J in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10):
    raw = Z.copy()
    raw[WHERE] -= J / sd_d                              # jump measured in ordinary-day sigmas
    x = mu_d + sd_d * (raw - raw.mean()) / raw.std()    # fix both moments by construction

    f_diff = x.mean() / x.var()
    bound = -1 / x.min()
    f_exact = optimize.brentq(lambda f: (x / (1 + f * x)).sum(), 0.0, bound * (1 - 1e-12))
    g = lambda f: np.log1p(f * x).mean() * D
    print(f"    {J:5.1%}   {x.mean() * D:9.2%}   {x.std() * np.sqrt(D):8.2%}   {x.min():9.2%}"
          f"   {bound:20.2f}x   {f_exact:7.2f}x   {f_diff / bound:8.2f}   {g(f_exact):17.2%}"
          f"   {g(f_diff) if f_diff < bound else float('nan'):12.2%}   {g(f_exact / 2):24.2%}")
# =>   5,000 days, standardized so the sample mean and variance are identical in every row -- 8.49% and 7.23% annualized -- while 20 of the days carry a jump and the rest are correspondingly quieter. f* = mu/sigma^2 therefore reads 16.24x every row
#         jump   ann. mean   ann. vol   worst day   ruin bound 1/|worst|   exact f   f*/bound   growth at exact f   growth at f*   growth at half of exact f
#         0.0%       8.49%      7.23%      -1.54%                  64.87x     16.13x       0.25              68.67%         68.67%                     51.42%
#         2.0%       8.49%      7.23%      -2.69%                  37.18x     15.80x       0.44              67.81%         67.76%                     50.62%
#         4.0%       8.49%      7.23%      -4.20%                  23.83x     13.72x       0.68              62.23%         59.52%                     45.42%
#         6.0%       8.49%      7.23%      -5.24%                  19.09x     11.51x       0.85              55.41%         37.46%                     39.46%
#         8.0%       8.49%      7.23%      -5.91%                  16.92x     10.14x       0.96              50.57%         -4.18%                     35.52%
#        10.0%       8.49%      7.23%      -6.34%                  15.78x      9.34x       1.03              47.49%           nan%                     33.09%
```

Every row reports an annualized mean of $8.49\%$ and an annualized volatility of $7.23\%$, by construction, so $\mu/\sigma^{2}$ returns $16.24\times$ six times. Everything else moves. The worst day runs from $-1.54\%$ to $-6.34\%$, the placeable maximum from $64.87\times$ to $15.78\times$, and the exact fraction from $16.13\times$ — indistinguishable from the formula when there is no jump — down to $9.34\times$, a shortfall of $42\%$ that the first two moments do not report.

The last two columns are where it becomes expensive rather than merely inaccurate. At a jump of $6\%$ the formula's $16.24\times$ is still placeable, sitting at $0.85$ of the bound, and it delivers $37.46\%$ growth against the exact fraction's $55.41\%$ — so obeying the formula costs a third of the growth it was invoked to maximize. At $8\%$ it is still placeable, at $0.96$ of the bound, and it delivers $-4.18\%$: a book that loses money compounding, recommended by a criterion whose entire purpose is to maximize compounding. At $10\%$ the formula's answer exceeds the bound at a ratio of $1.03$ and there is nothing to report, which is the published $1.13$ with a different jump attached.

!!! note "The Kelly fraction, the growth-optimal leverage, the log-optimal portfolio and the Merton fraction at unit risk aversion are four names for $\mu/\sigma^{2}$, and the fourth name is the one that explains the other three"
    The **Kelly fraction** is the name from the information-theoretic derivation, where the quantity maximized is a doubling rate. The **growth-optimal leverage** is the name from the almost-sure statement proved in section 1. The **log-optimal portfolio** is the name from the utility-maximization literature, where $\mathbb{E}[\log W]$ is maximized subject to a budget constraint. The **Merton fraction** is $\mu/(\gamma\sigma^{2})$ for constant relative risk aversion $\gamma$, and reduces to the other three exactly at $\gamma=1$, which is the case where the utility is logarithmic. All four are the same number, and the fourth is the useful one to hold in mind, because it makes explicit that fractional Kelly at $c$ is not a hedge or a fudge: it is the Merton solution at $\gamma=1/c$, a fully coherent objective that happens not to be growth. A desk running half Kelly is not underbetting the right objective; it is optimizing a different one.

## The Limit the Theorem Promises Is Decidable After One Hundred and Twenty-Six Years

Section 1 proved that full Kelly's wealth eventually dominates. Nothing in the proof says when, and the flatness of the parabola that makes overbetting cheap near the top also makes the dominance slow to appear. Both facts have the same closed form, and it is worth reading before choosing $c$.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18015)
SHARPE, VOL, PATHS = 1.174, 0.0723, 200_000
HORIZONS = (1, 5, 20, 50)
mu, sd = SHARPE * VOL, VOL
g_star = 0.5 * mu ** 2 / sd ** 2

B = np.cumsum(rng.standard_normal((PATHS, HORIZONS[-1])), axis=1)          # annual steps

print(f"  the growth-optimality theorem is a statement about a limit. At Sharpe {SHARPE} the"
      f" full-Kelly book grows at {g_star:.2%} a year; below, how long a track record must run"
      f" before full Kelly is 95% likely to be ahead of a fraction c of it. {PATHS:,} paths")
print("     c      growth kept   growth surrendered   " + "".join(
    f"P(full ahead), {h}y   " for h in HORIZONS) + "years to 95% confidence")
for c in (0.25, 0.50, 0.75, 0.90):
    kept = 2 * c - c ** 2
    row = []
    for h in HORIZONS:
        # the log-wealth gap on a shared path is (1-c)^2 mu^2 T / 2 sigma^2 + (1-c)(mu/sigma) B_T
        d = (mu ** 2 * h / sd ** 2) * (1 - c) ** 2 / 2 + (1 - c) * (mu / sd) * B[:, h - 1]
        row.append(f"{(d > 0).mean():17.4f}   ")
    years = (2 * stats.norm.ppf(0.95) / ((1 - c) * SHARPE)) ** 2
    print(f"    {c:4.2f}   {kept:11.4f}   {(1 - kept) * g_star:18.2%}   "
          + "".join(row) + f"{years:22.1f}")

print("\n     closed form for the same probabilities, Phi((1-c) S sqrt(T) / 2):")
for c in (0.25, 0.50, 0.75, 0.90):
    p = [f"{stats.norm.cdf((1 - c) * SHARPE * np.sqrt(h) / 2):.4f}" for h in HORIZONS]
    print(f"    c = {c:4.2f}   " + "   ".join(p))
# =>   the growth-optimality theorem is a statement about a limit. At Sharpe 1.174 the full-Kelly book grows at 68.91% a year; below, how long a track record must run before full Kelly is 95% likely to be ahead of a fraction c of it. 200,000 paths
#         c      growth kept   growth surrendered   P(full ahead), 1y   P(full ahead), 5y   P(full ahead), 20y   P(full ahead), 50y   years to 95% confidence
#        0.25        0.4375               38.76%              0.6682              0.8368              0.9754              0.9991                     14.0
#        0.50        0.7500               17.23%              0.6137              0.7440              0.9055              0.9809                     31.4
#        0.75        0.9375                4.31%              0.5580              0.6283              0.7432              0.8493                    125.6
#        0.90        0.9900                0.69%              0.5228              0.5528              0.6021              0.6606                    785.2
#
#         closed form for the same probabilities, Phi((1-c) S sqrt(T) / 2):
#        c = 0.25   0.6701   0.8375   0.9755   0.9991
#        c = 0.50   0.6154   0.7442   0.9053   0.9810
#        c = 0.75   0.5583   0.6286   0.7442   0.8503
#        c = 0.90   0.5234   0.5522   0.6035   0.6610
```

The gap in log wealth between full Kelly and a fraction $c$ of it, on a shared path, is $(1-c)^{2}\mu^{2}T/2\sigma^{2}+(1-c)(\mu/\sigma)B_T$, so full Kelly is ahead with probability $\Phi\!\left((1-c)S\sqrt{T}/2\right)$ — and the sixteen simulated cells match that expression to the third decimal, $0.6682$ against $0.6701$, $0.9055$ against $0.9053$, $0.7432$ against $0.7442$ and $0.6021$ against $0.6035$. The consequence is in the last column. Against half Kelly, full Kelly needs $31.4$ years of track record to be $95\%$ likely to have been the right choice. Against three-quarters Kelly it needs $125.6$ years. Against nine-tenths Kelly it needs $785.2$ years, during which it surrenders $0.69$ of $68.91$ percentage points of growth.

This is the honest failure of the whole apparatus, and it is not a failure of the mathematics. The theorem is true and dominance is almost sure. It is a failure of the theorem to be about anything a desk can act on, because the quantity it optimizes converges at $\sqrt{T}$ while the quantity it sacrifices — the drawdown column of section 2, $-95.5\%$ against $-71.6\%$ — is realized immediately and in full. **A criterion whose advantage takes a century to become statistically visible and whose cost is paid in the first bad month is a ceiling, and the published practice of running roughly a sixth of Kelly is not timidity about the theorem but a correct reading of it.**

## Every Repair Is a Smaller Fraction, a Bounded Support, or a Different Objective

Three sections have produced three separate reasons to bet less than $f^{*}$, and they are worth separating because they justify different amounts. The horizon argument of section 4 says the top of the parabola is flat and undecidable, which licenses any $c$ in a broad band and prefers none of them. The support argument of section 3 says the formula is computing on information that excludes the binding constraint, which licenses a hard cap at $1/\lvert x_{\min}\rvert$ and says nothing about where below it to sit. Estimation error is a third and genuinely distinct reason, treated where it belongs, in the drift's standard error at [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) and [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md), which proves that the error does not shrink with sampling frequency.

The published empirical numbers show all three acting at once. [Kelly, Volatility Targeting, and Leverage](../../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) reports growth of $+61.05\%$, $+55.73\%$ and $+42.76\%$ at $11.98\times$, $8.99\times$ and $5.99\times$ — ratios of $0.9129$ and $0.7004$ against the parabola's $0.9375$ and $0.7500$. The fractional books keep slightly *less* than the diffusion identity predicts, in both rows, which is section 3's jump term showing up in the direction section 3 says it must.

!!! warning "The formula consumes two moments and is decided by a third, and the number that settles it is one line of code longer than the number everyone prints"
    The mean and the variance determine $f^{*}$; the support determines whether $f^{*}$ can be placed; and the two calculations are run on the same array of returns, seconds apart. **The free diagnostic is $1/\lvert\min_i x_i\rvert$, the leverage at which the worst observed day takes the book to zero — one line beside the two moments the formula already consumes, requiring no new data, no model and no assumption.** When it comes in below $f^{*}$, as it did at $14.88\times$ against $16.84\times$ on the best book in the course, the correct conclusion is not to bet the smaller of the two: it is that the return distribution is too jump-heavy for the growth-optimality framework to apply at all, because the framework's own derivation assumed a floor that this distribution does not have. The sample minimum is itself a downward-biased estimate of the true worst case, which is [Extreme Value Theory](13-extreme-value-theory.md), so the diagnostic is generous even where it binds.

## An Upper Bound Computed From Two Moments

This page established that the log-optimal fixed fraction maximizes the almost-sure growth rate and dominates every other fixed fraction in the limit, with the logarithm arriving from the multiplicative structure rather than from any assumption about preferences; that the diffusion growth curve is exactly quadratic, so a fraction $c$ of Kelly keeps exactly $2c-c^{2}$ of the growth, verified at $0.4375$, $0.7500$, $0.9375$ and $0.0000$ with predicted and realized growth agreeing at $30.15\%$ against $30.12\%$ and $68.91\%$ against $68.81\%$, while median maximum drawdown runs $-43.9\%$, $-71.6\%$, $-87.5\%$ and $-95.5\%$; that the formula is a second-order truncation blind to the support, so on six laws with an identical $8.49\%$ mean and $7.23\%$ volatility all returning $16.24\times$, the placeable maximum falls from $64.87\times$ to $15.78\times$, the exact fraction from $16.13\times$ to $9.34\times$, and the formula's own answer delivers $37.46\%$, then $-4.18\%$, then nothing at all; and that the dominance the theorem promises needs $31.4$, $125.6$ and $785.2$ years of track record to become $95\%$ decidable against half, three-quarters and nine-tenths of Kelly, against growth surrendered of $17.23$, $4.31$ and $0.69$ percentage points.

The symmetry worth carrying forward is with the two pages that follow. Kelly asks what fraction maximizes a limit and answers with a single number that says nothing about the path; [Probability of Ruin](02-probability-of-ruin.md) and [Drawdown Probabilities](03-drawdown-probabilities.md) ask what the path does at that fraction and answer with distributions that say nothing about growth. The three are the same question asked of the same two parameters, and only the first has a formula short enough to be quoted without its assumptions. Section 2's drawdown column is the first of the other two answers arriving early, and the exact law behind it — the probability that a book at $c$ times Kelly ever loses a given fraction, which turns out to depend on $c$ and on nothing else — is the next page.

**The growth-optimal fraction is computed from the mean and the variance, is capped by the minimum, and is decided by neither — so it belongs in the sentence "not more than this" and in no other.**
