# The Bayesian Framework

The framework is usually introduced as a rival arithmetic, and it is not one. Bayes' rule is a theorem inside the probability everybody already accepts, and the disagreement it is famous for is entirely about which quantities are permitted to carry a distribution — not about how conditioning works once they do. Below, the four textbook cases of the rule, taught as four formulas, are computed by one line of code that never learns which case it is in, and the two-point and continuous versions of the same question agree throughout. What genuinely separates the paradigms shows up somewhere else, and it is sharp: the same twelve trades produce a posterior mean of $0.7142857143$ and a probability of $0.9538556763$ under two different stopping rules that disagree about the p-value, $0.0730$ against $0.0327$, one side of the conventional threshold and the other. Look at an honest coin after every trade and the p-value falls below $0.05$ on $0.4670$ of runs, while the Bayes factor exceeds $19$ on $0.0410$ of them against a bound of $0.0526$ that is a theorem rather than a correction. And the choice of framework is not available in the first place: exchangeability alone forces a prior to exist, recovered here as $0.2009$ against a true $0.2000$ from data that never named it.

This page covers what the framework asserts and what it does not, the single statement of Bayes' rule that the four discrete and continuous cases instantiate, the posterior as the complete output of an inference, the likelihood principle and the stopping rules that make it bite, the protection a marginal likelihood ratio inherits from being a martingale, and de Finetti's theorem as the reason a prior is discovered rather than adopted. It takes no posterior mode and minimizes no expected loss, which are [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md) and [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it constructs no prior and argues for none, which is [Prior Distributions](02-prior-distributions.md); it normalizes nothing it cannot integrate and derives no closed form, which are [Posterior Distributions](03-posterior-distributions.md) and [Conjugate Priors](04-conjugate-priors.md); it updates nothing sequentially, which is [Bayesian Updating](05-bayesian-updating.md); it computes no marginal likelihood for a model, which is [Bayesian Model Comparison](06-bayesian-model-comparison.md); it forecasts nothing, which is [Bayesian Prediction](07-bayesian-prediction.md); it quotes no credible interval and compares none to a confidence interval, which is [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md); it samples no posterior it cannot integrate, which is [Markov Chain Monte Carlo](../part-17-statistical-computing/04-markov-chain-monte-carlo.md); it inverts no conditional at the level of events, which is [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); and it never suggests that a probability statement about a parameter is more honest than one about a procedure merely for being easier to say out loud.

The trading stake is a course lesson finding the two frameworks agreeing to the third decimal and being careful to say why. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) puts a flat prior on a momentum strategy's hit rate, observes `hit rate 0.5406  (3329 of 6158)`, and reports `95% credible [0.5281, 0.5530], P(hit rate > 0.5) = 1.000` beside `Wald 95% CI  [0.5282, 0.5530]`. Its reading is that "the data has drowned the prior, as it should, and the Bayesian machinery here buys interpretability … not different numbers." That is the correct verdict at six thousand observations, and section 2 shows the same agreement arriving from four directions at once. Section 3 is where the agreement ends, and it ends over a question the lesson never had to ask: why the analyst stopped collecting.

## What May Be Given a Distribution Is the Entire Disagreement, and the Arithmetic of Conditioning Is Not in Dispute

A frequentist and a Bayesian handed the same model disagree about exactly one thing, and it is not a formula. Both accept that $f(x\mid\theta)$ describes how data arise. Both accept Bayes' rule as a theorem about conditional probability — it is proved in [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) from the axioms and nobody disputes it. The disagreement is whether $\theta$ is the sort of object that has a distribution at all. If it does, the rule applies to it and returns a posterior; if it does not, the rule has nothing to act on and inference must be conducted through the sampling behaviour of statistics instead. [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) named this as a modelling stance rather than a mathematical one, and it is worth being exact about how little of the mathematics it touches.

??? note "Proof that the four combinations of discrete and continuous $\Theta$ and $X$ are one statement about densities against a dominating measure, so the four displayed formulas are notation rather than content"

    Let $\Theta$ carry a $\sigma$-finite measure $\mu$ and the sample space carry a $\sigma$-finite measure $\nu$. Suppose the prior has density $\pi$ against $\mu$ and the model has density $f(x\mid\theta)$ against $\nu$ for each $\theta$. The joint law of $(\Theta,X)$ then has density $\pi(\theta)f(x\mid\theta)$ against the product measure $\mu\otimes\nu$, and the marginal density of $X$ is
    $$m(x)=\int_{\Theta}\pi(\theta')f(x\mid\theta')\,\mathrm{d}\mu(\theta').$$
    Wherever $m(x)>0$, the conditional density of $\Theta$ given $X=x$ against $\mu$ is
    $$\pi(\theta\mid x)=\frac{\pi(\theta)f(x\mid\theta)}{m(x)},$$
    which is the Radon–Nikodym derivative of the conditional law and is determined $\mu$-almost everywhere. Nothing in the derivation asked what $\mu$ or $\nu$ were.

    Now specialize. Taking $\mu$ to be counting measure on a finite or countable $\Theta$ turns $\pi$ into a pmf and the integral into $\sum_{\theta'}$; taking $\mu$ to be Lebesgue measure turns it into a density and the integral into $\int\mathrm{d}\theta'$. Independently, taking $\nu$ to be counting measure makes $f(x\mid\theta)$ a pmf and taking it Lebesgue makes it a density. The four choices give the four formulas usually displayed separately — discrete $\Theta$ with discrete $X$, discrete $\Theta$ with continuous $X$, continuous $\Theta$ with discrete $X$, continuous $\Theta$ with continuous $X$ — and they are the same equation with two symbols reinterpreted. The mixed cases are not hybrids requiring care; they are the generic case, and the pure ones are the special ones.

    The normalizer $m(x)$ is whatever makes the posterior integrate to one against $\mu$, which is why it can be ignored while a posterior is being identified and must be recovered before it is summarized — the subject of [Posterior Distributions](03-posterior-distributions.md). **The four cases are one theorem, the theorem is the definition of a conditional density, and every disagreement about Bayesian inference is therefore a disagreement about the model rather than about the derivation.**

    The load-bearing point is what the derivation quietly required: a joint law over $(\Theta,X)$. A frequentist declines to posit one, and that refusal is the whole of the objection. It is not an objection to the algebra above, which is a triviality, but to the claim that $\Theta$ is a random variable in the first place — and section 4 shows that for exchangeable data the claim is not optional.

Both parties compute with $f(x\mid\theta)$; the Bayesian additionally supplies $\pi(\theta)$ and reports a distribution over $\theta$, while the frequentist reports the behaviour of a procedure over hypothetical repetitions. When data are plentiful the two reports converge numerically — that is the lesson's `[0.5281, 0.5530]` against `[0.5282, 0.5530]`, and it is a theorem, taken up as Bernstein–von Mises in [Posterior Distributions](03-posterior-distributions.md). **What survives the convergence is that the two numbers are averages over different things, and no quantity of data turns an average over datasets into an average over parameter values.**

## The Posterior Is the Output, and the Four Textbook Cases Are One Computation Run Against Different Dominating Measures

The claim that the four cases are one line is checkable, and checking it also measures something the formulas conceal — that the choice of what to condition on costs information in a quantity with an exact value:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(16011)
mu_t, sig, pi_t, reps = 0.0004, 0.010, 0.30, 5_000
grid = np.linspace(-0.0060, 0.0060, 3001)
two = np.array([0.0, mu_t])


def posterior(logw):
    """Normalise a log prior-times-likelihood; the same line for a pmf and for a density."""
    e = np.exp(logw - logw.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


lp_d, lp_c = np.log([1 - pi_t, pi_t]), np.full(grid.size, -np.log(grid.size))
print(f"  true edge {mu_t * 1e4:.0f}bp against {sig * 1e4:.0f}bp daily noise, prior P(trend) ="
      f" {pi_t}; two-point theta reports P(trend | data), grid theta reports P(edge > 0 | data)")
print(f"  {reps:,} repetitions; the sign of a return keeps 2/pi = {2 / np.pi:.4f} of the"
      f" information a return carries about its mean, so the variance ratio should hold at"
      f" pi/2 = {np.pi / 2:.4f}")
print("        n   t-stat   theta in {0, 4bp}      theta on a grid       posterior sd, bp"
      "      variance")
print("                       count     series      count     series      count     series"
      "         ratio")
for n in (250, 1000, 4000, 16000):
    r = rng.standard_normal((reps, n)) * sig + mu_t
    k, S = (r > 0).sum(1), r.sum(1)
    o, sd = [], []
    for lp, ll, col in ((lp_d, stats.binom.logpmf(k[:, None], n, stats.norm.cdf(two / sig)), 1),
                        (lp_d, (S[:, None] * two - 0.5 * n * two ** 2) / sig ** 2, 1),
                        (lp_c, stats.binom.logpmf(k[:, None], n, stats.norm.cdf(grid / sig)), None),
                        (lp_c, (S[:, None] * grid - 0.5 * n * grid ** 2) / sig ** 2, None)):
        p = posterior(lp + ll)
        if col is not None:
            o.append(p[:, col].mean())
        else:
            o.append(p[:, grid > 0].sum(1).mean())
            m = p @ grid
            sd.append(np.sqrt(p @ grid ** 2 - m ** 2).mean())
    print(f"    {n:5d}   {mu_t * np.sqrt(n) / sig:6.3f}   {o[0]:8.4f}   {o[1]:8.4f}"
          f"   {o[2]:8.4f}   {o[3]:8.4f}   {sd[0] * 1e4:8.4f}   {sd[1] * 1e4:8.4f}"
          f"   {(sd[0] / sd[1]) ** 2:11.4f}")
# =>   true edge 4bp against 100bp daily noise, prior P(trend) = 0.3; two-point theta reports P(trend | data), grid theta reports P(edge > 0 | data)
#      5,000 repetitions; the sign of a return keeps 2/pi = 0.6366 of the information a return carries about its mean, so the variance ratio should hold at pi/2 = 1.5708
#            n   t-stat   theta in {0, 4bp}      theta on a grid       posterior sd, bp      variance
#                           count     series      count     series      count     series         ratio
#          250    0.632     0.3350     0.3534     0.6368     0.6667     7.9402     6.3246        1.5762
#         1000    1.265     0.4298     0.4897     0.7628     0.8132     3.9659     3.1623        1.5728
#         4000    2.530     0.6751     0.7803     0.9244     0.9637     1.9824     1.5811        1.5720
#        16000    5.060     0.9517     0.9860     0.9978     0.9998     0.9911     0.7906        1.5718
```

The function `posterior` is called four times and is never told which of the four cases it is executing. It receives a vector of log prior weights and a vector of log likelihoods, adds them, exponentiates and divides by the total — and that total is a sum over two regimes in two of the calls and a sum over three thousand grid points standing in for an integral in the other two. The four columns are the four textbook formulas, and the code that produced them does not distinguish among them.

The two-point columns report $P(\text{trend}\mid\text{data})$ against a prior of $0.30$, and they behave as a probability of a hypothesis should: $0.3534$ at two hundred and fifty days, $0.4897$ at a thousand, $0.7803$ at four thousand and $0.9860$ at sixteen thousand. The grid columns ask the continuous version of the same question, $P(\text{edge}>0\mid\text{data})$, and rise in step — $0.6667$, $0.8132$, $0.9637$, $0.9998$. They are not equal to one another and should not be: a two-point prior that already knows the edge is either zero or exactly four basis points is a far stronger assertion than a flat prior over a range, and it converges more slowly precisely because it has committed to a specific alternative the data must confirm rather than locate.

The information cost of conditioning on the wrong statistic is the exact part. The `count` columns see only how many days finished up; the `series` columns see the returns. The count is a coarsening, and for a Gaussian observation the sign retains exactly $2/\pi=0.6366$ of the Fisher information the observation carries about its mean, so a posterior built from counts should have $\pi/2=1.5708$ times the variance of one built from returns. The measured ratio is $1.5762$, $1.5728$, $1.5720$ and $1.5718$, converging on the constant from above as the grid's discreteness washes out. In posterior standard deviations that is $7.9402$ against $6.3246$ basis points at two hundred and fifty days, and it is why the hit rate the lesson computes is, in its own words, "trivia" beside the mean: a win-rate analysis of a return series has discarded a third of the evidence before it begins.

**The posterior is the entire output of the inference, and everything anybody reports — a probability of a hypothesis, a mean, a standard deviation, a position size — is a functional of it.** Which functional to report is a decision problem rather than an inferential one, and it belongs to [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md), which shows each summary is the Bayes rule under a particular loss. Nothing on this page needs that machinery; what matters here is that all four routes deliver an object of the same kind, and that the object is a distribution rather than a number.

## Two Experiments That Stop for Different Reasons Deliver the Same Posterior and Different P-Values

The convergence of section 2 makes the frameworks look like dialects. The following construction separates them completely, and it does so on a dataset small enough to check by hand. Consider a trader who reports nine winners out of twelve trades. Two accounts of how that came about are equally plausible: a fixed programme of twelve trades that happened to produce nine wins, or a rule to keep trading until the third loss, which happened to take twelve trades. The observed data are identical.

??? note "Proof that a stopping rule contributes a factor free of $\theta$, so the posterior is a function of the likelihood alone while a tail probability is not, and that the resulting evidence ratio is a martingale under the null"

    Let $X_1,X_2,\dots$ be conditionally independent given $\theta$ with density $f(x\mid\theta)$, and let $\tau$ be a stopping time — the decision to stop after $n$ observations depends on $x_1,\dots,x_n$ and on nothing else. The density of the observed path $(n,x_{1:n})$ is
    $$g(n,x_{1:n}\mid\theta)=\Big(\prod_{i=1}^{n}f(x_i\mid\theta)\Big)\,\mathbb{1}\{\tau(x_{1:n})=n\},$$
    and the indicator is a function of the data alone. Writing $c(x_{1:n})$ for that indicator, the posterior is
    $$\pi(\theta\mid n,x_{1:n})=\frac{\pi(\theta)c(x_{1:n})\prod_i f(x_i\mid\theta)}{\int\pi(\theta')c(x_{1:n})\prod_i f(x_i\mid\theta')\,\mathrm{d}\theta'}=\frac{\pi(\theta)\prod_i f(x_i\mid\theta)}{\int\pi(\theta')\prod_i f(x_i\mid\theta')\,\mathrm{d}\theta'},$$
    since $c$ cancels between numerator and denominator. Two experiments whose likelihood functions are proportional in $\theta$ therefore have identical posteriors, which is the likelihood principle. Concretely, twelve trials with nine wins gives $\binom{12}{9}\theta^{9}(1-\theta)^{3}$ and stopping at the third loss gives $\binom{11}{2}\theta^{9}(1-\theta)^{3}$; the binomial coefficients differ and are absorbed by the normalizer.

    A p-value is not a function of the likelihood at the observed data. It is $P(T\ge t_{\text{obs}}\mid H_0)$, an integral over outcomes that did not occur, and which outcomes those are is decided by the stopping rule. Under the fixed design the extreme outcomes are $\{K\ge9\}$ with $n=12$; under the inverse design they are $\{N\ge12\}$ with three losses required. These are different events with different probabilities, so the two analyses differ even though the data and the model agree.

    The martingale half is what makes this more than a curiosity. Let $m_0$ and $m_1$ be the marginal densities of the observed sequence under a simple null and under a proper prior on the alternative, and set $B_n=m_1(x_{1:n})/m_0(x_{1:n})$. Under the null, $\mathbb{E}_0[B_{n+1}\mid x_{1:n}]=B_n$, because integrating $m_1(x_{1:n+1})/m_0(x_{1:n})$ over $x_{n+1}$ against $m_0(x_{n+1}\mid x_{1:n})$ returns $m_1(x_{1:n})/m_0(x_{1:n})$; so $B_n$ is a non-negative martingale with $B_0=1$. Ville's inequality then gives
    $$P_0\big(\textstyle\sup_{n}B_n\ge k\big)\le 1/k$$
    for every $k>0$, **so the probability that a Bayes factor ever reaches $19$ against a true null is at most $1/19=0.0526$ however long and however often the data are examined, and no correction was applied to obtain that.**

    The load-bearing asymmetry is that one quantity is defined at the observed data and the other is defined over a sample space the analyst's intentions determine. That is why optional stopping is a catastrophe for one and a non-event for the other, and why the protection cannot be transferred: a p-value has no martingale to inherit from.

Both halves are measurable, the first exactly and the second by simulation:

```python
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(16013)
n0, k0, reps, nmax = 12, 9, 20_000, 2000
grid = np.linspace(1e-6, 1 - 1e-6, 200_001)

lik = grid ** k0 * (1 - grid) ** (n0 - k0)                  # bare likelihood, no constant
fix = stats.binom.pmf(k0, n0, grid)                         # twelve trades, nine wins
neg = stats.nbinom.pmf(k0, n0 - k0, 1 - grid)               # trade until the third loss
for nm, w in (("proportional", lik), ("fixed n = 12", fix), ("stop at 3 losses", neg)):
    p = w / w.sum()
    print(f"    {nm:18s}   mean {p @ grid:.10f}   P(theta > 1/2) {p[grid > 0.5].sum():.10f}")
print(f"    one-sided p-value, fixed n = 12      {stats.binom.sf(k0 - 1, n0, 0.5):.4f}")
print(f"    one-sided p-value, stop at 3 losses  {stats.nbinom.sf(k0 - 1, n0 - k0, 0.5):.4f}")

x = rng.random((reps, nmax)) < 0.5                          # honest coins, looked at always
k = x.cumsum(1)
n = np.arange(1, nmax + 1)
pv = stats.norm.sf((k - n / 2) / np.sqrt(n / 4))
logbf = special.betaln(k + 1, n - k + 1) + n * np.log(2.0)  # flat alternative against theta = 1/2

print(f"  {reps:,} honest coins, looked at after every trial up to n = {nmax:,}")
print("     look-up-to      p ever < 0.05   BF10 ever > 19   Ville bound   BF10 ever > 99"
      "   Ville bound")
for cap in (100, 500, 1000, 2000):
    m = n <= cap
    print(f"    {cap:10,d}   {(pv[:, m] < 0.05).any(1).mean():14.4f}"
          f"   {(logbf[:, m] > np.log(19)).any(1).mean():14.4f}   {1 / 19:11.4f}"
          f"   {(logbf[:, m] > np.log(99)).any(1).mean():14.4f}   {1 / 99:11.4f}")
# =>     proportional         mean 0.7142857143   P(theta > 1/2) 0.9538556763
#        fixed n = 12         mean 0.7142857143   P(theta > 1/2) 0.9538556763
#        stop at 3 losses     mean 0.7142857143   P(theta > 1/2) 0.9538556763
#        one-sided p-value, fixed n = 12      0.0730
#        one-sided p-value, stop at 3 losses  0.0327
#      20,000 honest coins, looked at after every trial up to n = 2,000
#         look-up-to      p ever < 0.05   BF10 ever > 19   Ville bound   BF10 ever > 99   Ville bound
#               100           0.3144           0.0303        0.0526           0.0054        0.0101
#               500           0.3996           0.0376        0.0526           0.0066        0.0101
#             1,000           0.4361           0.0398        0.0526           0.0071        0.0101
#             2,000           0.4670           0.0410        0.0526           0.0075        0.0101
```

The first three lines are the likelihood principle with the constants stripped. Feeding the bare $\theta^{9}(1-\theta)^{3}$, the binomial pmf and the negative binomial pmf into the same normalizer returns a posterior mean of $0.7142857143$ and a probability $P(\theta>1/2)=0.9538556763$ in all three cases, agreeing to every digit printed. The two designs are different experiments with different sample spaces, different sufficient statistics and different unbiased estimators, and they induce the same posterior because their likelihoods differ by a factor the normalizer removes.

The next two lines are the same data analysed by tail probability, and they do not agree: $0.0730$ under the fixed design and $0.0327$ under the inverse one. One is above the conventional threshold and the other is below it. Nothing about the trades changed — only the answer to "what would you have done had the ninth win come later", which is a question about the analyst rather than about the market.

The second table is the practical form of that gap. Twenty thousand honest coins are examined after every single trial. The p-value criterion fires on $0.3144$ of runs within a hundred trials, $0.4361$ within a thousand and $0.4670$ within two thousand, and it is a standard result that this tends to $1$ as the horizon grows: sampling to a foregone conclusion always succeeds eventually. The Bayes factor criterion, examined just as often, exceeds $19$ on $0.0303$, $0.0376$, $0.0398$ and $0.0410$ of runs against Ville's bound of $0.0526$, and exceeds $99$ on $0.0054$ to $0.0075$ against a bound of $0.0101$. The bound is not approached and cannot be exceeded at any horizon.

**This is what [Part XV](../part-15-multiple-testing/index.md) closed by pointing at: a treatment that assigns a probability to the hypothesis itself is not merely a different vocabulary for the same correction, because the quantity it reports is a martingale under the null and therefore carries an anytime-valid guarantee that no amount of looking can degrade.** The price is visible in the same table and is paid in the column that is not there: the Bayes factor's protection is against a *simple* null, and it is bought by committing to a proper prior on the alternative — a commitment the p-value never makes and never has to defend. [Bayesian Model Comparison](06-bayesian-model-comparison.md) prices that commitment and finds it expensive.

## Exchangeability Forces a Prior to Exist, So the Only Choice Is Whether to Write It Down

The standard objection to the framework is that the prior is an arbitrary addition, and that an analyst who declines to supply one is being appropriately modest. For any dataset whose ordering carries no information — which is the assumption behind every pooled estimate in quantitative finance — that position is unavailable.

??? note "Proof that an infinite exchangeable binary sequence is a mixture of independent Bernoulli sequences over a unique mixing measure, which is de Finetti's theorem"

    Call $X_1,X_2,\dots$ exchangeable if the law of $(X_{\sigma(1)},\dots,X_{\sigma(n)})$ equals that of $(X_1,\dots,X_n)$ for every $n$ and every permutation $\sigma$. Let $\bar X_n=n^{-1}\sum_{i\le n}X_i$. The sequence $\bar X_n$ is a reverse martingale with respect to the exchangeable $\sigma$-fields $\mathcal{E}_n=\sigma(\bar X_n,\bar X_{n+1},\dots)$, since exchangeability gives $\mathbb{E}[X_i\mid\mathcal{E}_n]=\bar X_n$ for every $i\le n$; reverse martingale convergence then yields $\bar X_n\to\Theta$ almost surely for some $\mathcal{E}_\infty$-measurable $\Theta$ taking values in $[0,1]$. Write $F$ for the law of $\Theta$.

    Conditionally on $\mathcal{E}_\infty$ the variables are independent Bernoulli$(\Theta)$. To see it, exchangeability makes $P(X_1=1,\dots,X_k=1\mid \bar X_n)$ equal to the probability of drawing $k$ ones without replacement from an urn holding $n\bar X_n$ ones, namely $\prod_{j<k}(n\bar X_n-j)/(n-j)$, which tends to $\Theta^{k}$. The same argument on any finite pattern with $k$ ones and $l$ zeros gives $\Theta^{k}(1-\Theta)^{l}$, so
    $$P(X_1=x_1,\dots,X_n=x_n)=\int_0^1\theta^{k}(1-\theta)^{n-k}\,\mathrm{d}F(\theta),\qquad k=\textstyle\sum_i x_i.$$
    Uniqueness follows because the moments $\int\theta^{k}\,\mathrm{d}F=P(X_1=\dots=X_k=1)$ are determined by the joint law and determine $F$ on $[0,1]$.

    Read the display in the direction that matters. The left-hand side is a modelling assumption an analyst is willing to make — trade order is uninformative. The right-hand side contains a prior $F$ and a likelihood, and it is not an alternative formulation offered for consideration; it is a *consequence*, unique, and already implied. **An analyst who asserts exchangeability has asserted a prior, and the only remaining question is whether they know which one.** The two visible defaults are both extreme: assuming the observations are iid at a single unknown $\theta$ sets $F$ to a point mass, and treating every subgroup as unrelated sets $F$ to something maximally diffuse.

    The load-bearing consequence is quantitative rather than philosophical. Under the mixture, $\mathrm{Cov}(X_i,X_j)=\mathbb{E}[\Theta^2]-\mathbb{E}[\Theta]^2=\mathrm{Var}(\Theta)$ for $i\ne j$, so exchangeable-but-not-independent data carry a positive pairwise correlation equal to the variance of the mixing measure, and the variance of a count is inflated by $n(n-1)\mathrm{Var}(\Theta)$ over the binomial value. An iid analysis of exchangeable data does not merely adopt an implicit prior; it reports a standard error wrong by a factor that grows with the sample.

The mixing measure is not a metaphysical object. It is estimable, and the inflation it causes is large:

```python
import numpy as np

rng = np.random.default_rng(16015)
mean, n, m = 0.40, 200, 40_000

print(f"  {m:,} strategies, {n} trades each, every strategy's hit rate drawn from a"
      f" Beta with mean {mean}; an analyst who assumes the trades are iid sees only the pool")
print("     a + b   prior sd   recovered   pair corr   predicted    Var(k)   iid Var(k)"
      "    ratio   next-win gap")
for tot in (5.0, 20.0, 100.0, 1000.0, 100000.0):
    a, b = mean * tot, (1 - mean) * tot
    th = rng.beta(a, b, m)
    x = rng.random((m, n)) < th[:, None]
    k = x.sum(1)
    vt = a * b / ((a + b) ** 2 * (a + b + 1))               # variance of the mixing measure
    iid = n * mean * (1 - mean)
    corr = (k.var() - iid) / (n * (n - 1) * mean * (1 - mean))
    gap = np.abs((a + k) / (a + b + n) - k / n).mean()
    print(f"    {tot:6.0f}   {np.sqrt(vt):8.4f}   {th.std():9.4f}   {corr:9.4f}"
          f"   {vt / (mean * (1 - mean)):9.4f}   {k.var():7.1f}   {iid:10.1f}"
          f"   {k.var() / iid:6.2f}   {gap:12.5f}")
# =>   40,000 strategies, 200 trades each, every strategy's hit rate drawn from a Beta with mean 0.4; an analyst who assumes the trades are iid sees only the pool
#         a + b   prior sd   recovered   pair corr   predicted    Var(k)   iid Var(k)    ratio   next-win gap
#             5     0.2000      0.2009      0.1686      0.1667    1658.3         48.0    34.55        0.00411
#            20     0.1069      0.1071      0.0477      0.0476     504.0         48.0    10.50        0.00825
#           100     0.0487      0.0488      0.0099      0.0099     142.7         48.0     2.97        0.01592
#          1000     0.0155      0.0156      0.0010      0.0010      57.8         48.0     1.20        0.02526
#        100000     0.0015      0.0015     -0.0000      0.0000      47.6         48.0     0.99        0.02744
```

The `recovered` column is de Finetti's $F$ read back out of data that never mentioned it: $0.2009$ against a true $0.2000$, $0.1071$ against $0.1069$, $0.0488$ against $0.0487$, $0.0156$ against $0.0155$ and $0.0015$ against $0.0015$. The mixing measure is a feature of the joint law and is identified by it, which is exactly what the uniqueness clause of the theorem claims. An analyst is not free to decline it; they are free only to estimate it badly.

The cost of declining is the `ratio` column. At a diffuse mixing measure the variance of the trade count is $1658.3$ where a binomial calculation predicts $48.0$ — a factor of $34.55$, so a standard error computed on the iid assumption is understated by a factor of nearly six. That factor falls to $10.50$, $2.97$, $1.20$ and finally $0.99$ as the mixing measure concentrates, and the last row is the point: **independence is not the neutral assumption sitting beside exchangeability, it is the degenerate special case in which the prior is a point mass, and asserting it is the most confident prior available rather than the least.** The measured pairwise correlation tracks the predicted $\mathrm{Var}(\Theta)/\bar p(1-\bar p)$ throughout — $0.1686$ against $0.1667$, then $0.0477$, $0.0099$, $0.0010$ and zero.

The final column shows where the prior surfaces in something anyone would actually report. The next-trade forecast under the mixture is $(a+k)/(a+b+n)$ against the sample frequency $k/n$, and the gap runs $0.00411$, $0.00825$, $0.01592$, $0.02526$ and $0.02744$ — growing as the mixing measure tightens, because a confident prior pulls harder. A desk that reports $k/n$ has not avoided the prior; it has adopted the improper one the arithmetic assigns when $a$ and $b$ are set to zero.

## The Framework Buys Coherence, Which Is Weaker Than It Sounds and Is Not the Same as Being Right

What has been established is narrow and worth stating without inflation. Bayes' rule applied to a parameter is the definition of a conditional density, the four textbook cases are one equation, the posterior is the whole output, the stopping rule cancels, the resulting evidence measure is a martingale, and exchangeability implies a prior. None of that establishes that the answers are correct, and the classical arguments for the framework are careful to claim something much weaker.

The central one is coherence. De Finetti's Dutch book argument shows that an agent quoting betting odds not obeying the probability axioms can be handed a set of bets they accept individually and which together lose money in every state; Savage's axioms reach the same conclusion from preferences over acts, yielding a subjective probability and a utility such that the agent behaves as if maximizing expected utility. The complete class theorems point the same way from the frequentist side: under mild conditions every admissible decision rule is a Bayes rule for some prior, so the question is never whether one is being Bayesian but which prior one is being Bayesian with — a result [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md) uses when it measures a Bayes rule's frequentist risk. What all three arguments deliver is internal consistency. An agent certain that a strategy has an edge, updating coherently, remains coherent and remains wrong, and no theorem in this section prevents that.

The second is calibration, which is empirical rather than axiomatic: a sequence of probability statements is calibrated if events assigned probability $p$ occur about a $p$ fraction of the time. It is checkable against data in a way coherence is not, and it is not implied by coherence — the martingale property of section 3 guarantees beliefs cannot be driven anywhere by selective looking, and says nothing about whether they are aimed at the right target.

The third is asymptotic and does most of the practical work. Under regularity conditions the posterior concentrates on the parameter minimizing Kullback–Leibler divergence to the truth at rate $\sqrt n$, with the prior's influence vanishing at rate $1/n$ — the reason the lesson's credible and Wald intervals agree at six thousand observations. That is a licence to be careless about the prior when data are plentiful and the model is right, and it is regularly cited where neither holds. A model class excluding the truth produces coherent, well-behaved, confidently wrong posteriors, which is the failure [Posterior Distributions](03-posterior-distributions.md) measures directly.

!!! note "The likelihood, the normalized likelihood, the posterior, the marginal likelihood and the predictive are five objects assembled from one integrand, and only two of them are distributions over the parameter"
    All five are built from $\pi(\theta)f(x\mid\theta)$ and they are routinely confused. The **likelihood** $f(x\mid\theta)$ is a function of $\theta$ for fixed data and is not a density in $\theta$ — it need not integrate to anything finite, which is why a flat prior can be improper while the posterior is proper. The **normalized likelihood** divides it by $\int f(x\mid\theta')\mathrm{d}\theta'$ when that integral converges, and it equals the posterior under a flat prior in the given coordinates, which is a coincidence of parameterization rather than a neutral summary — [Prior Distributions](02-prior-distributions.md) shows the coordinate dependence directly. The **posterior** $\pi(\theta\mid x)$ is the object this page is about, a distribution over $\theta$ given the data. The **marginal likelihood** $m(x)=\int\pi(\theta)f(x\mid\theta)\mathrm{d}\theta$ is the same integrand integrated over $\theta$ rather than normalized by, making it a density over *datasets* and not over parameters at all; it is invisible in every calculation on this page because it cancels, and it is the entire content of [Bayesian Model Comparison](06-bayesian-model-comparison.md). The **predictive** $\int f(\tilde y\mid\theta)\pi(\theta\mid x)\mathrm{d}\theta$ is a distribution over future data, which is [Bayesian Prediction](07-bayesian-prediction.md). Reporting a marginal likelihood as though it scored a parameter, or a normalized likelihood as though it were prior-free, are the two errors this list exists to prevent.

!!! warning "A posterior carries no record of where its prior came from, so a belief that was earned and a belief that was convenient produce output identical in form and indistinguishable on inspection"
    Every number on this page is conditional on inputs the output does not display. The two-point prior of $0.30$ in section 2 drives $P(\text{trend}\mid\text{data})$ to $0.9860$ at sixteen thousand days; a different prior gives a different number from the same returns, and the printed posterior looks equally authoritative either way. The Bayes factor's anytime-valid bound of $0.0526$ in section 3 holds only against the proper alternative that was specified, and specifying a different one moves the measured $0.0410$ with no indication in the output that anything was chosen. The mixing measure of section 4 is recovered to four decimals when forty thousand strategies are available and is pure assertion when one is. **The free diagnostic is to recompute every reported posterior quantity under two further priors — one an order of magnitude tighter and one an order of magnitude looser than the one you used — and to publish the three numbers together whenever they differ in the third significant figure, because a posterior that moves under that treatment is reporting the prior, and one that does not move has earned the right to be quoted alone.** The whole of [Prior Distributions](02-prior-distributions.md) is an argument that the looser of those two is not the safer one.

## What the Two Paradigms Actually Disagree About

This page established that Bayes' rule applied to a parameter is the definition of a conditional density against a dominating measure, so the four textbook cases are one equation and one line of code computed all four; that a posterior is the complete output and every reported quantity is a functional of it, with the two-point and grid formulations of one question rising together from a prior of $0.30$ to $0.9860$ and $0.9998$ as the sample grew from two hundred and fifty days to sixteen thousand; that conditioning on a coarser statistic costs an exactly known amount, a win count retaining $2/\pi=0.6366$ of a return's information about its mean and inflating the posterior variance by a measured $1.5762$, $1.5728$, $1.5720$ and $1.5718$ against a predicted $\pi/2=1.5708$; that a stopping rule contributes a factor free of $\theta$, so nine wins in twelve trades gives a posterior mean of $0.7142857143$ and $P(\theta>1/2)=0.9538556763$ under both a fixed and an inverse design while the p-value reads $0.0730$ under one and $0.0327$ under the other; that a marginal likelihood ratio is a non-negative martingale under the null, so looking after every one of two thousand trials drove the p-value below $0.05$ on $0.4670$ of honest runs while the Bayes factor exceeded $19$ on $0.0410$ against Ville's bound of $0.0526$ and exceeded $99$ on $0.0075$ against $0.0101$; and that exchangeability implies a unique mixing measure, recovered here at $0.2009$, $0.1071$, $0.0488$, $0.0156$ and $0.0015$ against truths of $0.2000$, $0.1069$, $0.0487$, $0.0155$ and $0.0015$, whose neglect inflates the variance of a trade count by a factor of $34.55$.

The shape shared by all three exhibits is that the framework's advantages are guarantees about *procedure* and its exposures are assumptions about *inputs*. The likelihood principle and the martingale bound are theorems holding whatever the analyst does next, which is a genuinely stronger position than the corrections of [Part XV](../part-15-multiple-testing/index.md) can occupy, since those require a search width [Data Snooping Bias](../part-15-multiple-testing/04-data-snooping-bias.md) showed is unavailable. But every one of those theorems is conditional on a prior and a model class, and neither appears in the output. The corrections of Part XV fail loudly, by demanding a number nobody has; the framework here fails quietly, by accepting whatever number is offered.

That exposure is the subject of the rest of this part. The immediate question is the one section 4 raised and did not answer: if a prior necessarily exists, what determines which one, and is there a choice that avoids committing to anything. The answer is that the usual candidates for a neutral prior are neutral only in a coordinate system nobody declared, and that is [Prior Distributions](02-prior-distributions.md).

**The two frameworks agree about the arithmetic and disagree about what may carry a distribution, and the practical consequence is not that they produce different numbers but that one of them will produce a number no matter what it is given.**
