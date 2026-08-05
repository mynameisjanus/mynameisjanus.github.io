# Bayesian Signal Updating

[Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) closes by saying that "the rule is exact and its inputs are estimates, so the posterior inherits every error in all three," and that this page is therefore "about where the inputs come from, and not a page about algebra." Of the three inputs the prior has an entire part arguing about it and the evidence is whatever arrived; the likelihood is chosen silently, usually by whoever typed the first line, and it decides two things nobody attributes to it. It sets the *rate* at which belief can move: log posterior odds accumulate as a random walk whose drift is a Kullback–Leibler divergence, so a Sharpe-$0.30$ strategy needs $16{,}379$ trading days — $65.0$ years — to reach $95\%$ posterior confidence, against a prediction of $16{,}489$, while a strategy with no edge at all reaches the same confidence anyway on $0.0500$ of runs against an exact bound of $0.0526$. And it sets how much *one day* is allowed to matter: a Gaussian likelihood makes the posterior mean linear in the observation and therefore unbounded, so a twenty-sigma day moves the posterior Sharpe by $0.3147$ where a Student-$t$ likelihood of identical variance moves it by $0.0039$, a factor of $80.3$. The consequence on real return distributions is that under a Gaussian likelihood the single largest day of twenty years carries $10.12\%$ of all the accumulated evidence, worth $26.0$ median days, against $0.79\%$ and $1.4$ under a likelihood that fits the tails.

This page covers the accumulation of log posterior odds as a random walk, the Kullback–Leibler divergence as its drift and the horizon that follows, the martingale bound that limits how often a flat strategy can look convincing, the influence of a single observation as a property of the likelihood's score, and what tail thickness does to the concentration of evidence. It does not prove that sequential updating equals batch updating, establish order invariance, treat overlapping windows, or develop exponential forgetting, all of which are [Bayesian Updating](../part-16-bayesian-statistics/05-bayesian-updating.md); it does not argue for any prior or build a hierarchical one, which is [Prior Distributions](../part-16-bayesian-statistics/02-prior-distributions.md); it does not shrink a cross-section, derive James–Stein or treat the winner's curse, which is [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it does not derive the odds form of Bayes' rule or the screening arithmetic, which is [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); it does not establish posterior asymptotics under misspecification, which is [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md); it fits nothing to real returns, which is [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md); it corrects for no multiplicity, which is [Part XV](../part-15-multiple-testing/index.md); and it never treats a likelihood as a description of the data rather than as a choice about which observations count.

The trading stake is a course lesson measuring exactly this rate in a different currency and finding it dismal. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) reports that a posterior interval on the equity premium "narrows like $1/\sqrt n$, exactly as theory promises — and after twenty-four years it is still ±7.7% around a +5.8% mean," concluding that "no philosophy of statistics rescues you from the information content of the data." Section 2 is that sentence with the constant supplied: the information content of one day is the Kullback–Leibler divergence $S^{2}/504$, which fixes the horizon to conviction at $504\log(19)/S^{2}$ days and makes twenty-four years a short record rather than a long one.

## Evidence Accumulates as a Random Walk Whose Drift Is a Kullback–Leibler Divergence

Sequential updating is a sum, because the logarithm turns the product of likelihood ratios into one. What that sum converges to, and how fast, is a question with an exact answer that does not depend on the prior at all.

??? note "Proof that log posterior odds are a random walk with drift equal to the Kullback–Leibler divergence, that the expected time to a target is $2L/d^{2}$, and that a false conviction has probability at most $e^{-L}$ whatever the horizon"

    Let $H_1$ and $H_0$ be two simple hypotheses with densities $p_1,p_0$, and let $\Lambda_n=\sum_{i\le n}\log\frac{p_1(x_i)}{p_0(x_i)}$ be the accumulated log-likelihood ratio, so that the posterior odds are the prior odds times $e^{\Lambda_n}$. Under $H_1$ the increments are i.i.d. with mean
    $$\mathbb{E}_1\!\left[\log\frac{p_1(X)}{p_0(X)}\right]=D_{\mathrm{KL}}(p_1\,\|\,p_0),$$
    so $\Lambda_n$ is a random walk whose drift *is* the divergence between the hypotheses — the information content of one observation, in the exact sense that it is what one observation contributes to the log-odds on average.

    For daily returns under "Sharpe $S$" against "Sharpe $0$" with common volatility, the standardized daily drift is $d=S/\sqrt{252}$ and the two densities are unit normals separated by $d$, giving $D_{\mathrm{KL}}=d^{2}/2=S^{2}/504$ and increment variance $d^{2}$. Reaching a log-odds target $L$ is then the first passage of a drifted walk, and [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md) gives its mean as $L$ over the drift,
    $$\mathbb{E}[n]=\frac{2L}{d^{2}}=\frac{504\,L}{S^{2}},$$
    with the inverse-Gaussian right skew that page establishes, so the median sits well below the mean.

    Under $H_0$ the same object is a different animal. The likelihood ratio $e^{\Lambda_n}$ is a non-negative martingale with $\mathbb{E}_0[e^{\Lambda_n}]=1$, because $\int p_0\cdot(p_1/p_0)=1$. Ville's maximal inequality for non-negative martingales then gives, for any horizon and any stopping rule whatsoever,
    $$\mathbf{P}_0\!\left(\sup_n\Lambda_n\ge L\right)\le e^{-L}.$$
    At $L=\log 19$, the level corresponding to $95\%$ posterior confidence from even prior odds, the bound is $1/19=0.0526$.

    **The load-bearing asymmetry is that the drift depends on the alternative and the bound does not. How long conviction takes is a property of the effect size, so a weak edge is slow; how often a flat strategy manufactures conviction is capped at $e^{-L}$ no matter how long anyone looks, which is the one guarantee on this page that survives optional stopping — and it survives it only while the likelihood is correct.**

## A Sharpe-0.30 Strategy Needs Sixty-Five Years, and a Flat One Reaches the Same Confidence Anyway One Time in Twenty

Both halves of the proof are measurable, and the second is the more surprising: a bound that holds for *any* stopping rule is exactly the thing a desk watching a running statistic does not otherwise have.

```python
import numpy as np

rng = np.random.default_rng(18071)
D, PATHS, YEARS, CHUNK = 252, 8_000, 400, 2_520
TARGET = np.log(0.95 / 0.05)                            # even prior odds to 95% posterior


def first_passage(d, drift_sign):
    """Days until the log-odds walk first reaches TARGET, streamed in chunks."""
    pos = np.zeros(PATHS)
    hit = np.full(PATHS, np.nan)
    for start in range(0, YEARS * D, CHUNK):
        step = rng.standard_normal((PATHS, CHUNK)) * d + drift_sign * d ** 2 / 2
        run = pos[:, None] + np.cumsum(step, axis=1)
        crossed = run >= TARGET
        fresh = crossed.any(axis=1) & np.isnan(hit)
        hit[fresh] = start + np.argmax(crossed[fresh], axis=1) + 1
        pos = run[:, -1]
    return hit


print(f"  a strategy is either running at Sharpe S or at zero, with even prior odds. Each day"
      f" contributes x*d - d^2/2 to the log posterior odds, where d = S/sqrt({D}) is the daily"
      f" standardized drift, so the evidence is a random walk with drift equal to the"
      f" Kullback-Leibler divergence d^2/2. Reaching {TARGET:.3f} means 95% posterior."
      f" {PATHS:,} paths x {YEARS} years")
print("     Sharpe   KL per day   E[days to 95%]: predicted   measured   in years   median"
      "   90th pct   reached within horizon   if truly flat: bound exp(-L)   measured")
for S in (0.30, 0.50, 0.80, 1.20, 2.00):
    d = S / np.sqrt(D)
    hit = first_passage(d, +1)
    flat = first_passage(d, -1)
    ok = ~np.isnan(hit)
    print(f"    {S:6.2f}   {d ** 2 / 2:10.6f}   {2 * TARGET / d ** 2:27,.0f}"
          f"   {np.nanmean(hit):8,.0f}   {np.nanmean(hit) / D:8.1f}   {np.nanmedian(hit):6,.0f}"
          f"   {np.nanpercentile(hit, 90):8,.0f}   {ok.mean():22.4f}"
          f"   {np.exp(-TARGET):27.4f}   {np.mean(~np.isnan(flat)):8.4f}")
# =>   a strategy is either running at Sharpe S or at zero, with even prior odds. Each day contributes x*d - d^2/2 to the log posterior odds, where d = S/sqrt(252) is the daily standardized drift, so the evidence is a random walk with drift equal to the Kullback-Leibler divergence d^2/2. Reaching 2.944 means 95% posterior. 8,000 paths x 400 years
#         Sharpe   KL per day   E[days to 95%]: predicted   measured   in years   median   90th pct   reached within horizon   if truly flat: bound exp(-L)   measured
#          0.30     0.000179                        16,489     16,379       65.0   12,287     32,935                   0.9991                        0.0526     0.0500
#          0.50     0.000496                         5,936      5,955       23.6    4,568     11,978                   1.0000                        0.0526     0.0506
#          0.80     0.001270                         2,319      2,333        9.3    1,743      4,708                   1.0000                        0.0526     0.0527
#          1.20     0.002857                         1,031      1,055        4.2      795      2,129                   1.0000                        0.0526     0.0510
#          2.00     0.007937                           371        379        1.5      284        761                   1.0000                        0.0526     0.0493
```

The drift prediction lands in every row — $16{,}489$ against $16{,}379$, $5{,}936$ against $5{,}955$, $2{,}319$ against $2{,}333$, $1{,}031$ against $1{,}055$ and $371$ against $379$ — and so does the bound, at $0.0500$, $0.0506$, $0.0527$, $0.0510$ and $0.0493$ against $0.0526$. Two readings follow and they point in opposite directions.

The first is that conviction is slow in a way that scales as the inverse square of the edge. A Sharpe-$0.80$ strategy needs $9.3$ years on average, which is already longer than most strategies survive; at $0.30$ it is $65.0$ years, with a ninetieth percentile of $32{,}935$ days, or $131$ years. Since the horizon enters through $S^{2}T$, this is the same quantity that set the drawdown crossover in [Drawdown Probabilities](03-drawdown-probabilities.md) and the decision horizon in [Kelly Criterion](01-kelly-criterion.md), arriving for the third time as the thing that governs what a record can settle.

The second is more cheerful and is the only unconditional guarantee in this part. However long a flat strategy is watched, and whenever the watcher chooses to stop, the probability that its evidence ever reaches $95\%$ confidence is at most $0.0526$. That is a genuine protection against optional stopping — the failure [Martingales](../part-08-stochastic-processes/10-martingales.md) diagnoses in repeated looks at an accumulating backtest — and it is bought entirely by the likelihood ratio being a martingale under the null. **Everything that follows on this page is about what happens to that guarantee when the density in the ratio is not the density the data came from.**

## The Likelihood Decides How Much One Day Is Allowed to Matter, and a Gaussian One Sets No Limit

The rate at which belief moves is one property of the likelihood. The other is its response to a single extreme observation, and it is fixed entirely by the shape of the log-density's derivative.

??? note "Proof that the posterior's sensitivity to one observation is the likelihood's score, unbounded and linear for a Gaussian and bounded and redescending for a Student-$t$, with a crossover at about $1.7$ standard deviations"

    For a prior $\pi$ and one observation $x$ with likelihood $f(x-\theta)$, the posterior mean is $\bar\theta(x)=\int\theta\pi(\theta)f(x-\theta)\,d\theta\big/\int\pi(\theta)f(x-\theta)\,d\theta$, and differentiating shows its response to $x$ is governed by the **score** $\psi=-(\log f)'$, since the likelihood enters only through $f$ and its derivative.

    For a Gaussian, $\psi(r)=r$: unbounded and linear. One observation therefore moves the posterior mean by an amount proportional to how extreme it is, with no ceiling, so a value at twenty standard deviations moves the belief twenty times as far as a value at one. In the conjugate case the posterior mean is exactly linear in $x$, which is usually presented as a convenience and is the whole problem.

    For a Student-$t$ with $\nu$ degrees of freedom rescaled to unit variance with $k=\sqrt{\nu/(\nu-2)}$,
    $$\psi(r)=\frac{(\nu+1)k^{2}r}{\nu+k^{2}r^{2}},$$
    which rises, peaks, and then *redescends* to zero as $\lvert r\rvert\to\infty$: a sufficiently extreme observation is read as a draw from the tail rather than as information about the location, and is discounted. At $\nu=4$ the score is $5r/(2+r^{2})$, which exceeds the Gaussian's $r$ for $\lvert r\rvert<\sqrt3\approx1.73$ and falls below it thereafter.

    That crossover is the part usually missed. A heavy-tailed likelihood is not uniformly less responsive; it is *more* responsive to ordinary observations, because under a law that expects occasional extremes an unremarkable value is stronger evidence about the centre. Robustness here is a reallocation of sensitivity from the tail to the body, not a blanket reduction of it.

    **The load-bearing quantity is the score's behaviour at infinity, and it is a choice made when the likelihood is written rather than a property of the data. The same observations, the same prior and the same arithmetic produce beliefs that differ by two orders of magnitude on exactly the days that matter most.**

```python
import numpy as np
from scipy import integrate, stats

D, TAU = 252, 0.5                                       # prior sd on the annualized Sharpe


def posterior_mean(x, nu):
    """Posterior mean of the annualized Sharpe after one standardized daily return x,
    under a N(0, TAU^2) prior and a likelihood with nu degrees of freedom (None = Gaussian)."""
    like = (lambda r: stats.norm.pdf(r)) if nu is None else (
        lambda r: stats.t.pdf(r * np.sqrt(nu / (nu - 2)), nu))
    num = integrate.quad(lambda s: s * stats.norm.pdf(s, 0, TAU)
                         * like(x - s / np.sqrt(D)), -8 * TAU, 8 * TAU)[0]
    den = integrate.quad(lambda s: stats.norm.pdf(s, 0, TAU)
                         * like(x - s / np.sqrt(D)), -8 * TAU, 8 * TAU)[0]
    return num / den


print(f"  one standardized daily return arrives. The prior on the annualized Sharpe is N(0,"
      f" {TAU}^2); the likelihood is the modelling choice. A Gaussian likelihood makes the"
      f" posterior mean linear in the observation and therefore unbounded; a Student-t likelihood"
      f" of the same variance makes it redescend, so an extreme value is read as a tail draw"
      f" rather than as evidence")
print("     |x|, sigmas   P(|X| >= x) if Gaussian   posterior Sharpe: Gaussian    t(6)     t(4)"
      "   Gaussian / t(4)")
for x in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0):
    g, t6, t4 = posterior_mean(x, None), posterior_mean(x, 6), posterior_mean(x, 4)
    print(f"    {x:12.1f}   {2 * stats.norm.sf(x):23.2e}   {g:25.4f}   {t6:6.4f}   {t4:6.4f}"
          f"   {g / t4:16.1f}")
# =>   one standardized daily return arrives. The prior on the annualized Sharpe is N(0, 0.5^2); the likelihood is the modelling choice. A Gaussian likelihood makes the posterior mean linear in the observation and therefore unbounded; a Student-t likelihood of the same variance makes it redescend, so an extreme value is read as a tail draw rather than as evidence
#         |x|, sigmas   P(|X| >= x) if Gaussian   posterior Sharpe: Gaussian    t(6)     t(4)   Gaussian / t(4)
#                 1.0                  3.17e-01                      0.0157   0.0220   0.0262                0.6
#                 2.0                  4.55e-02                      0.0315   0.0276   0.0263                1.2
#                 3.0                  2.70e-03                      0.0472   0.0254   0.0215                2.2
#                 4.0                  6.33e-05                      0.0629   0.0221   0.0175                3.6
#                 6.0                  1.97e-09                      0.0944   0.0165   0.0124                7.6
#                10.0                  1.52e-23                      0.1573   0.0106   0.0077               20.4
#                20.0                  5.51e-89                      0.3147   0.0055   0.0039               80.3
```

The Gaussian column is a straight line by construction — $0.0157$, $0.0315$, $0.0472$, $0.0629$, $0.0944$, $0.1573$, $0.3147$ — and it never stops. The $t(4)$ column rises to $0.0263$ at two standard deviations, then turns over and falls to $0.0039$ at twenty, so the ratio between the two runs $0.6$, $1.2$, $2.2$, $3.6$, $7.6$, $20.4$ and $80.3$. The crossover predicted at $\sqrt3\approx1.73$ standard deviations is visible in the first two rows, where the heavy-tailed likelihood is the *more* responsive of the two.

The second column is why the last row is not a curiosity. A twenty-sigma daily move has Gaussian probability $5.51\times10^{-89}$, which is to say the Gaussian likelihood has never contemplated it; the arithmetic nonetheless proceeds, and it proceeds by treating the observation as eighty times more informative than a model that expects such days would. This is the concrete form of the warning [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) issues — that a likelihood computed under a Gaussian assumption "assigns an absurdly small probability to a six-sigma move, so observing one produces an enormous likelihood ratio in favour of whatever hypothesis was allowed to have fat tails." **The likelihood ratio is not measuring the world; on that day it is measuring the modelling assumption, and the more confident the assumption the larger the measurement.**

!!! note "A prior, a likelihood, an evidence threshold and a stopping rule are four choices behind one posterior probability, and the argument is almost always about the first"
    **The prior** is what gets debated, has an entire part devoted to it, and — by section 1 — affects the *level* of the log-odds but not their drift, so it shifts the finish line rather than the speed. **The likelihood** is chosen once, rarely revisited, and sets both the drift and the per-observation sensitivity, which is to say almost everything this page measures. **The evidence threshold** converts a posterior into a decision and appears in the bound $e^{-L}$, so moving from $95\%$ to $99\%$ confidence cuts false conviction from $0.0526$ to $0.0101$ and multiplies the honest waiting time by $1.56$. **The stopping rule** is the one input usually not written down at all, and it is the reason the martingale bound matters: that bound is uniform over stopping rules, so it is the only quantity here that a desk cannot damage by choosing when to look. Three of the four are modelling decisions and one of them is a habit.

## On Fat-Tailed Returns One Day in Five Thousand Carries a Tenth of the Evidence

Sections 2 and 3 measured the two properties separately on hypothetical observations. Running them together on a realistic return distribution shows what the choice costs over a full record, and the answer is not that the posterior ends up in the wrong place.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18075)
D, PATHS, YEARS, S1 = 252, 20_000, 20, 0.80
TARGET = np.log(0.95 / 0.05)
d = S1 / np.sqrt(D)


def evidence(nu_true, nu_model):
    """A real edge of Sharpe S1 on t(nu_true) returns. Accumulate log odds for
    'Sharpe S1' against 'Sharpe 0' under a likelihood with nu_model degrees of freedom."""
    c = 1.0 if nu_true is None else np.sqrt(nu_true / (nu_true - 2))
    z = (rng.standard_normal((PATHS, YEARS * D)) if nu_true is None
         else rng.standard_t(nu_true, (PATHS, YEARS * D)) / c)
    x = z + d                                                    # the edge is genuinely there
    if nu_model is None:
        step = x * d - d ** 2 / 2
    else:
        k = np.sqrt(nu_model / (nu_model - 2))
        step = stats.t.logpdf((x - d) * k, nu_model) - stats.t.logpdf(x * k, nu_model)
    total = step.sum(axis=1)
    biggest = step.max(axis=1)
    top = np.sort(step, axis=1)[:, -step.shape[1] // 100:].sum(axis=1)   # largest 1% of days
    return (np.median(biggest / np.abs(total)), np.median(biggest),
            np.median(np.abs(step)), np.median(top / np.abs(total)))


print(f"  a genuine edge of Sharpe {S1} observed for {YEARS} years on t-distributed returns, with"
      f" the desk accumulating log odds for it against no edge. The likelihood is the only thing"
      f" that differs between the two halves, and the question is how concentrated the"
      f" evidence is in a handful of days. {PATHS:,} paths")
print("     true returns   Gaussian likelihood: largest day's share   its log odds   x median day"
      "   top 1% of days   correctly specified: largest day's share   its log odds   x median day"
      "   top 1% of days")
for nu in (None, 8, 6, 4, 3):
    lab = "Gaussian" if nu is None else f"t({nu})"
    sg, bg, mg, cg = evidence(nu, None)
    st, bt, mt, ct = evidence(nu, nu)
    print(f"    {lab:>12}   {sg:38.4f}   {bg:13.4f}   {bg / mg:13.1f}   {cg:17.2f}"
          f"   {st:38.4f}   {bt:13.4f}   {bt / mt:13.1f}   {ct:17.2f}")
# =>   a genuine edge of Sharpe 0.8 observed for 20 years on t-distributed returns, with the desk accumulating log odds for it against no edge. The likelihood is the only thing that differs between the two halves, and the question is how concentrated the evidence is in a handful of days. 20,000 paths
#         true returns   Gaussian likelihood: largest day's share   its log odds   x median day   top 1% of days   correctly specified: largest day's share   its log odds   x median day   top 1% of days
#            Gaussian                                   0.0293          0.1846             5.4                1.08                                   0.0292          0.1845             5.4                1.08
#                t(8)                                   0.0439          0.2688             8.7                1.26                                   0.0133          0.0926             2.1                0.68
#                t(6)                                   0.0516          0.3126            10.6                1.33                                   0.0118          0.0882             1.9                0.60
#                t(4)                                   0.0720          0.4312            16.3                1.48                                   0.0098          0.0891             1.6                0.50
#                t(3)                                   0.1012          0.5792            26.0                1.62                                   0.0079          0.1008             1.4                0.40
```

The first row is the control and it matters: on genuinely Gaussian returns the two likelihoods agree to four decimals — $0.0293$ against $0.0292$, $0.1846$ against $0.1845$, $5.4$ against $5.4$, $1.08$ against $1.08$. **Robustness costs nothing when it is not needed**, which removes the usual objection to using it.

As the tails thicken the two diverge completely, and the divergence is in concentration rather than in the answer. Under a Gaussian likelihood the single largest day of a twenty-year record grows from $2.93\%$ to $10.12\%$ of the total accumulated evidence, and that one day is worth $26.0$ median days at $\nu=3$. Under a likelihood that fits the tails the same statistic *falls* to $0.79\%$ and $1.4$ median days: the correctly specified model becomes more evenly weighted as tails thicken, because it expects the extremes and stops treating them as news. The last column says the same thing at less extreme quantiles — the largest $1\%$ of days carry $1.62$ times the total evidence under the Gaussian likelihood, meaning the other $99\%$ of days are net *negative* evidence being cancelled by fifty of them, against $0.40$ under the correct one.

This is the honest failure, and it is not a bias. A desk running the Gaussian likelihood on $t(3)$ returns will, over twenty years, usually arrive at approximately the right posterior; it will arrive there by a route on which one day in five thousand supplied a tenth of the reasoning, and its belief on any given morning is largely a statement about whether that day has happened yet. The martingale bound of section 1 is the casualty: it required $\mathbb{E}_0[e^{\Lambda}]=1$, which holds under the density the data actually came from and not under the one that was typed. **A guarantee that is uniform over stopping rules and conditional on the likelihood has traded a failure mode everyone worries about for one nobody logs.**

## Every Repair Is a Fatter Likelihood, a Longer Record, or a Threshold That Prices Its Own Rate

The three findings admit three repairs with very different prices. Section 3's is nearly free and is the one to take: replace the Gaussian likelihood with a Student-$t$ of matched variance, which costs four decimal places of agreement when the Gaussian was right and bounds the influence of a single day when it was not. The degrees of freedom need not be pinned down precisely — $t(6)$ and $t(4)$ differ from each other far less than either differs from the Gaussian.

Section 2's slow clock cannot be repaired at all, only respected. The horizon $504L/S^{2}$ is a property of the data-generating process, so the only levers are a bigger edge or a lower threshold, and lowering the threshold has a stated price: the bound $e^{-L}$ rises in exactly the same parameter. That trade is worth making explicitly rather than by default — a desk that would accept one false conviction in ten can reach $L=\log 9$ in $70\%$ of the days that $L=\log 19$ requires. What cannot be done is to watch the running posterior and stop when it looks good, because the bound that makes that safe is the one section 4 breaks.

!!! warning "The posterior is reported and the route it took is not, so a belief resting on one day looks exactly like a belief resting on five thousand"
    A posterior probability is a scalar, and two records producing the same number can differ entirely in how they got there. Section 4 measures records where the largest single day supplies $10.12\%$ of the evidence and the top $1\%$ of days supply $162\%$ of it, and none of that is visible in the posterior itself. **The free diagnostic is the share of total accumulated log-evidence contributed by the largest single observation, and by the largest one percent: both are one line over the per-observation log-likelihood-ratio series the update already computes, and a well-specified model on twenty years of data puts them near $0.03$ and $1.1$.** Values far above that mean the belief is a statement about a handful of days, which is worth knowing before it becomes a statement about a position size. The same series answers the complementary question for free — recomputing the posterior with the largest day deleted is a one-line sensitivity that no amount of argument about the prior substitutes for.

## A Rate, a Bound, and the One Input That Sets Both

This page established that log posterior odds accumulate as a random walk whose drift is the Kullback–Leibler divergence between the hypotheses, so the expected time to $95\%$ confidence is $504\log(19)/S^{2}$ days — verified at $16{,}489$ against $16{,}379$, $5{,}936$ against $5{,}955$, $2{,}319$ against $2{,}333$, $1{,}031$ against $1{,}055$ and $371$ against $379$, which is $65.0$ years at Sharpe $0.30$ with a ninetieth percentile of $131$ years; that the likelihood ratio is a martingale under the null, bounding false conviction at $e^{-L}=0.0526$ uniformly over every horizon and every stopping rule, measured at $0.0500$, $0.0506$, $0.0527$, $0.0510$ and $0.0493$; that a single observation's influence is the likelihood's score, unbounded and linear for a Gaussian and redescending for a Student-$t$, so a twenty-sigma day moves the posterior Sharpe by $0.3147$ against $0.0039$, a factor of $80.3$, with the heavy-tailed likelihood *more* responsive below its $1.73$-sigma crossover; and that on $t(3)$ returns the largest single day of twenty years carries $10.12\%$ of the accumulated evidence under a Gaussian likelihood against $0.79\%$ under a correct one, with the top $1\%$ of days carrying $1.62$ against $0.40$, while on genuinely Gaussian returns the two agree at $0.0293$ against $0.0292$.

The relationship to the microstructure pages before it is that all three have now found the same shape in different material. [Queue Models](05-queue-models.md) found a delay governed by a second moment nobody reports; [Order Arrival Processes](06-order-arrival-processes.md) found a tail governed by a branching ratio nobody estimates; and this page finds a belief governed by a likelihood nobody argues about — in each case a first-moment summary that is correct, a decision-relevant quantity that is not, and a diagnostic costing one line. What separates this page from those is that its failure is not in the answer: the posterior converges to the right place regardless, and what the likelihood decides is which observations were allowed to put it there. The pages that follow make that concrete in the only place where the distinction is priced directly, which is a tail — a risk number is a functional of exactly the observations this page has been deciding whether to trust.

**A prior sets where a belief starts and a likelihood sets what can move it, so the input that gets argued about is the one that matters least.**
