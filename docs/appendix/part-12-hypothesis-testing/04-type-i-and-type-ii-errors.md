# Type I and Type II Errors

The two ways a test can be wrong are usually drawn as a two-by-two table, which makes them look like two entries in one accounting system. They are not. They are probabilities computed under different hypotheses, so they cannot be added, averaged, or traded against each other without supplying something the test never contained — a prior over which hypothesis holds and a price for each mistake. They are also not equally repairable. The false-positive rate can be measured and fixed by anyone willing to simulate their own null; the false-negative rate cannot be measured at all without naming an alternative, and naming it is the step the entire literature of reported results omits.

This page covers the two errors as conditional probabilities under different hypotheses, the frontier they trade along at fixed sample size, the choice of $\alpha$ as a decision problem with a cost ratio and prior odds behind it, the different currencies a desk pays for each error, and the asymmetry that makes one calibratable and the other not. It does not define the level, the size or the rejection region, which is [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md); it does not characterize the p-value, which is [p-values](03-p-values.md); it does not compute a power function, invert one for a sample size, or measure what conditioning on significance does to an estimate, which are [Statistical Power](05-statistical-power.md); it does not convert error rates into the probability that a hypothesis is true, which needs the base rates worked in [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md); it does not measure the size distortion that dependence produces, which is [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md); it corrects nothing for the number of tests run, which is [Part XV](../part-15-multiple-testing/index.md); and it never reports an accuracy.

The trading stake is a pair of risk models the course grades and finds each broken in the other's direction. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) reports `the book: n 4,758 ... expected breaches at 99% = 47.6`, then `historical VaR 1.121% breaches 48 (1.01x) Kupiec p 9.51e-01 indep p 0.0013` and `EWMA conditional breaches 117 (2.46x) indep p 0.5208`. One model gets the count right and the clustering wrong; the other gets the clustering right and the count wrong. The lesson's verdict is that "fixing the timing and fixing the level are two separate repairs, and the standard tool performs one of them." Sections 1 and 4 explain why no single number could have graded either model.

## The Two Errors Are Probabilities Under Two Different Hypotheses, So No Single Number Summarizes Both

A **Type I error** is rejecting when the null holds, with probability $\alpha=\mathbf{P}_{\theta_0}(\varphi=1)$. A **Type II error** is failing to reject when the alternative holds, with probability $\beta(\theta_1)=\mathbf{P}_{\theta_1}(\varphi=0)$. The subscripts are the entire content of this section. The two numbers are expectations under two different probability measures, and there is no measure under which both are simultaneously defined unless someone supplies a distribution over which hypothesis is true. Nothing in the data provides that distribution, and nothing in the test computes it.

This is why an "accuracy" is not available. Accuracy would be $\pi_0(1-\alpha)+\pi_1(1-\beta)$, which requires the prior weights $\pi_0,\pi_1$ on the two hypotheses; supply them and you are doing the base-rate calculation of [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md), which is a different exercise with different inputs. Refuse to supply them, as classical testing does, and the two rates simply do not combine. The course's two VaR models illustrate the consequence exactly: `Kupiec p 9.51e-01` says the historical model's breach *count* is indistinguishable from the promised rate, `indep p 0.0013` says its breach *timing* is not, and there is no weighted average of the two that constitutes a grade, because they are answers to different questions asked of different features of the same data.

The second asymmetry is that $\beta$ is not a number at all. It is a function on the alternative set, and quoting "the Type II error rate" has silently fixed a point in that set. Where $\Theta_1$ contains parameter values arbitrarily close to the null — as it almost always does — $\beta$ approaches $1-\alpha$ somewhere in $\Theta_1$, so the supremum of the Type II error is uninformative for every test. That function is the subject of [Statistical Power](05-statistical-power.md); what matters here is that it exists, so any single reported figure is a choice of alternative that the report did not disclose.

## At a Fixed Sample Size the Two Rates Trade Off Along One Curve, and Nothing Moves the Curve Except More Data

Holding $n$ fixed, lowering $\alpha$ shrinks the rejection region and therefore raises $\beta$; the two cannot be reduced together by any manipulation of the threshold. What can be improved is the *choice of region* at a given $\alpha$, and there is a best one.

??? note "Proof that the achievable pairs $(\alpha,\beta)$ at fixed sample size trace one decreasing convex curve determined by the likelihood ratio, so no rule improves both coordinates"

    Fix a simple null $P_0$ and simple alternative $P_1$ with densities $p_0,p_1$, and let a test be any measurable $\varphi:\mathcal{X}\to[0,1]$ giving the probability of rejection. Its coordinates are $\alpha(\varphi)=\mathbb{E}_0[\varphi]$ and $1-\beta(\varphi)=\mathbb{E}_1[\varphi]$. The set of achievable pairs $\{(\alpha(\varphi),1-\beta(\varphi))\}$ over all such $\varphi$ is **convex**, because for any $\varphi_1,\varphi_2$ and $\lambda\in[0,1]$ the randomized test $\lambda\varphi_1+(1-\lambda)\varphi_2$ is itself a test and its coordinates are the corresponding convex combination — both maps are linear in $\varphi$.

    Its upper boundary is traced by likelihood-ratio tests. By the Neyman–Pearson lemma, proved in [Likelihood Ratio Tests](06-likelihood-ratio-tests.md), the test maximizing $\mathbb{E}_1[\varphi]$ subject to $\mathbb{E}_0[\varphi]\le\alpha$ rejects when $p_1/p_0>k_\alpha$ and randomizes on the boundary, with $k_\alpha$ decreasing in $\alpha$. So the frontier is $\alpha\mapsto\sup\{1-\beta\}$, a concave non-decreasing function of $\alpha$ — equivalently $\beta$ is convex and decreasing in $\alpha$ — and it is indexed by the single number $k$. Every admissible test sits on it, and the entire family is generated by sliding one threshold along one statistic.

    Two consequences. No procedure attains a point above the curve, so at fixed $n$ there is no rule reducing both errors: the trade-off is a geometric fact about the achievable set, not a limitation of any particular test. And the curve's *position* depends only on how distinguishable $P_0$ and $P_1$ are — through the law of $p_1/p_0$ — so the only way to improve both coordinates at once is to change that law, which means more data or a better-separated alternative.

    The load-bearing quantity is the likelihood ratio: it is simultaneously the statistic generating every frontier point and the measure of how far apart the two hypotheses are. **Choosing $\alpha$ moves you along the curve; only more data moves the curve.**

The curve is worth pricing on the course's own strategy, taking the Sharpe of $0.30$ as *true* rather than estimated, so that every failure to reject is a genuine miss. The block reports, for each track-record length, the power available at the conventional level and the level that would be required to reach $50\%$ and $80\%$ power:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12041)
sr = 0.30                                          # the course's momentum Sharpe, assumed true

def se(T):                                         # Lo's standard error for an annual Sharpe
    return np.sqrt((1 + sr**2 / (2 * 252)) / T)   # Lo's factor is per-period

print("  one-sided test of a TRUE annualized Sharpe of 0.30, by track-record length")
print("    years   SE(SR)   SR/SE   power at 5%   alpha for 50%   alpha for 80%")
for T in (3, 5, 10, 24, 50, 100):
    d = sr / se(T)
    pw = stats.norm.sf(stats.norm.isf(0.05) - d)
    a50 = stats.norm.sf(d - stats.norm.isf(0.50))
    a80 = stats.norm.sf(d - stats.norm.isf(0.20))
    print(f"    {T:5d}   {se(T):6.4f}   {d:5.3f}   {pw:11.4f}   {a50:13.4f}   {a80:13.4f}")

T, reps = 24, 200_000
d = sr / se(T)
hits = (rng.normal(d, 1.0, reps) > stats.norm.isf(0.05)).mean()
print(f"  simulated check at 24 years: power {hits:.4f} at the 5% level")
# =>   one-sided test of a TRUE annualized Sharpe of 0.30, by track-record length
#        years   SE(SR)   SR/SE   power at 5%   alpha for 50%   alpha for 80%
#            3   0.5774   0.520        0.1302          0.3017          0.6263
#            5   0.4473   0.671        0.1650          0.2512          0.5678
#           10   0.3163   0.949        0.2431          0.1714          0.4574
#           24   0.2041   1.470        0.4304          0.0708          0.2650
#           50   0.1414   2.121        0.6831          0.0170          0.1004
#          100   0.1000   3.000        0.9123          0.0014          0.0155
#      simulated check at 24 years: power 0.4298 at the 5% level
```

Read the twenty-four-year row first, since it is the course's own sample. A real Sharpe of $0.30$ sits $1.470$ standard errors from zero, and a one-sided $5\%$ test detects it $43.04\%$ of the time — the simulated check returns $0.4298$ and confirms the closed form. That figure is already the frontier's verdict on the conventional threshold: on a genuine edge, with the longest history most desks will ever see, the test is close to a coin flip. Merely reaching an even chance requires $\alpha=0.0708$, and reaching $80\%$ requires $\alpha=0.2650$ — a false-positive rate of more than one in four, accepted deliberately, as the price of an eighty-percent chance of noticing something that is really there.

The rows above are the situations research actually runs in. At three years the same true edge yields $13.02\%$ power at the conventional level, and buying $80\%$ power would mean accepting $\alpha=0.6263$: a test that fires on nearly two-thirds of worthless strategies. At the other end, a hundred years of daily data reaches $91.23\%$ power at $5\%$ and needs only $\alpha=0.0155$ for $80\%$ — the same edge, the same statistic, the same threshold arithmetic, with the curve simply moved. Nothing in these rows involves a mistake, an assumption violation, or a bad estimator. It is the frontier, priced.

**The conventional $5\%$ was not chosen to sit anywhere in particular on this curve, and on a twenty-four-year record of a real edge it buys a coin flip.**

## Choosing Alpha Is a Decision Problem, and the Conventional Five Percent Solves It Almost Nowhere

If the two errors have prices and the hypotheses have prior odds, then $\alpha$ stops being a convention and becomes the solution to a minimization. Nothing about that minimization is exotic, and its answer is almost never $0.05$.

??? note "Proof that the cost-minimizing rule is a likelihood-ratio threshold set by the cost ratio and the prior odds, so a conventional $\alpha$ is an implicit claim about both"

    Let $\pi_1$ be the prior probability that the alternative holds, $\pi_0=1-\pi_1$, and let $c_{\mathrm{FP}}$ and $c_{\mathrm{FN}}$ be the costs of the two errors. The expected cost of a test $\varphi$ is
    $$C(\varphi)=c_{\mathrm{FP}}\,\pi_0\,\mathbb{E}_0[\varphi]+c_{\mathrm{FN}}\,\pi_1\,\mathbb{E}_1[1-\varphi].$$
    Writing the expectations as integrals against the densities and collecting the terms multiplying $\varphi(x)$,
    $$C(\varphi)=c_{\mathrm{FN}}\pi_1+\int \varphi(x)\big[c_{\mathrm{FP}}\pi_0\,p_0(x)-c_{\mathrm{FN}}\pi_1\,p_1(x)\big]\mathrm{d}x .$$
    The integral is minimized pointwise by setting $\varphi(x)=1$ exactly where the bracket is negative, since $\varphi$ is free to be $0$ or $1$ at each $x$ independently. That condition is
    $$\frac{p_1(x)}{p_0(x)}>\frac{c_{\mathrm{FP}}\,\pi_0}{c_{\mathrm{FN}}\,\pi_1},$$
    a likelihood-ratio test whose threshold is the product of the cost ratio and the prior odds and involves no conventional constant anywhere.

    Reading it backwards is the point. Fixing $\alpha=0.05$ fixes the threshold $k$, and since the optimal $k$ equals $(c_{\mathrm{FP}}\pi_0)/(c_{\mathrm{FN}}\pi_1)$, choosing the level is choosing a cost ratio *given* prior odds, or prior odds given a cost ratio. The choice is made either way; the convention merely leaves it unstated and identical across every problem it is applied to.

    The load-bearing step is that $\varphi$ may be optimized pointwise, which is what turns a constrained problem over functions into an inequality at each $x$. **A significance level is a price list, and the conventional one was written for no particular market.**

The optimal level can therefore be computed rather than assumed, given the same true Sharpe and the same twenty-four-year record:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12043)
sr, T = 0.30, 24
d = sr / np.sqrt((1 + sr**2 / (2 * 252)) / T)   # Lo's factor is per-period
grid = np.arange(0.001, 0.601, 0.001)
power = stats.norm.sf(stats.norm.isf(grid) - d)
cases = [("1:10", 0.1), ("1:3", 1 / 3), ("1:1", 1.0), ("3:1", 3.0), ("10:1", 10.0)]

print("  expected cost = c_FP * pi0 * alpha + c_FN * pi1 * beta(alpha), 24-year record")
print("  the alpha that minimises it, over cost ratios and prior odds of a real edge")
print("    c_FN:c_FP    pi1=0.02   pi1=0.10   pi1=0.30")
best = []
for lab, ratio in cases:
    row = [grid[int(((1 - pi1) * grid + ratio * pi1 * (1 - power)).argmin())]
           for pi1 in (0.02, 0.10, 0.30)]
    best.append(row)
    print(f"    {lab:9s}   {row[0]:8.3f}   {row[1]:8.3f}   {row[2]:8.3f}")

near = sum(1 for row in best for a in row if abs(a - 0.05) <= 0.005)
print(f"  cells of 15 where the cost-optimal alpha is within 0.005 of the conventional 0.05: {near}")
# =>   expected cost = c_FP * pi0 * alpha + c_FN * pi1 * beta(alpha), 24-year record
#      the alpha that minimises it, over cost ratios and prior odds of a real edge
#        c_FN:c_FP    pi1=0.02   pi1=0.10   pi1=0.30
#        1:10           0.001      0.001      0.002
#        1:3            0.001      0.001      0.020
#        1:1            0.001      0.013      0.095
#        3:1            0.004      0.069      0.286
#        10:1           0.035      0.254      0.600
#      cells of 15 where the cost-optimal alpha is within 0.005 of the conventional 0.05: 0
```

The optimal level ranges over nearly three orders of magnitude, from $0.001$ to $0.600$, and the final line is the finding: in none of the fifteen cells is the cost-optimal $\alpha$ within $0.005$ of the conventional $0.05$. That is not a claim that $5\%$ is a bad choice. It is the observation that $5\%$ is a choice, and that the grid of situations in which it would be the *right* choice does not include any of the fifteen plausible combinations of cost asymmetry and prior plausibility examined here.

The structure of the grid is worth reading across and down. Moving down a column raises the cost of a missed edge relative to a false one, and the optimal $\alpha$ rises with it — from $0.001$ to $0.035$ at $\pi_1=0.02$, and from $0.002$ to $0.600$ at $\pi_1=0.30$. Moving across a row raises the prior plausibility that the edge is real, and the optimal $\alpha$ rises again. The cells where a conventional threshold is nearly right are the middle ones, and even there the $1{:}1$ row reads $0.001$, $0.013$ and $0.095$: with symmetric costs and a one-in-ten prior, the cost-minimizing level is $0.013$, nearly four times stricter than convention, while at a three-in-ten prior it is $0.095$, twice as loose.

**A test run at $5\%$ has asserted a cost ratio and a prior, and the assertion is the same one whether the trade risks a thousand dollars or the fund.**

!!! note "Accuracy, false-positive rate, sensitivity, specificity and confidence are five names for two conditional probabilities taken in opposite directions, and machine-learning usage inverts the statistical one"
    **Sensitivity** is $1-\beta$ at a named alternative, which statisticians call power; **specificity** is $1-\alpha$. Both condition on the truth, as $\alpha$ and $\beta$ do. **Precision**, by contrast, conditions on the *decision* — it is the share of rejections that are correct — and therefore depends on the prior $\pi_1$, which is why it belongs to the base-rate calculation in [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md) and cannot be read off a test. "Confidence" names $1-\alpha$ for a test and the coverage of a procedure for an interval, and [Confidence Intervals](../part-11-parameter-estimation/07-confidence-intervals.md) shows those are the same guarantee seen through the duality of page 01. The trap is that a classifier reporting $95\%$ accuracy and a test reporting a $5\%$ level are describing unrelated quantities, and only the second is a promise about anything.

## A Desk Pays Different Currencies for the Two Errors, and Only One of Them Appears in the Backtest

The costs in the previous section were symbols, and on a trading desk they are not commensurable. A false positive — deploying a strategy with no edge — is paid in capital, financing, fees and the slow bleed of costs against a zero gross return, and it is *observable*: the equity curve records it, the attribution names it, and the review meeting has a number to discuss. A false negative — declining a real edge — is paid in opportunity, and it is invisible by construction. The strategy was never run, so no series exists, no attribution is produced, and nothing enters the record. There is no dataset of rejected ideas that later worked.

That asymmetry in *observability*, rather than any asymmetry in magnitude, is what makes the conventional threshold feel safe. A process tuned to a $5\%$ level generates false negatives at a rate the previous section priced at roughly six in ten on a real Sharpe-$0.30$ edge with twenty-four years of data, and none of them will ever appear on a report. The course's two VaR models are the same structure in miniature: `breaches 119 (2.50x)` is a visible failure that a risk committee will act on, while the power a repaired model gives up is not measured anywhere in the lesson's tables, or in any risk report.

## The False-Positive Rate Can Always Be Calibrated Against a Simulated Null, and the False-Negative Rate Never Can

Here is the practical asymmetry that organizes the rest of this part. The null, when it is a single law, can be generated: write it down, simulate it, run the testing code, count rejections. Nothing about the data is needed, and the previous pages did exactly this repeatedly. The alternative cannot be generated, because it is a set with no distinguished member; to simulate it, the analyst must pick one point, and that pick is a research assumption that no procedure supplies.

??? note "Proof that a size can be estimated to arbitrary accuracy without data while a power cannot, because the null is a point and the alternative is a set"

    Suppose the null is simple, $\Theta_0=\{\theta_0\}$. Then $\alpha=\mathbf{P}_{\theta_0}(\varphi=1)$ is the expectation of a bounded function under a *known* distribution, so drawing $B$ independent samples from $P_{\theta_0}$ and averaging $\varphi$ gives an estimate with standard error $\sqrt{\alpha(1-\alpha)/B}$, which is $O(B^{-1/2})$ and involves no observed data at all. Accuracy is bought with computation. If the null is composite the same works pointwise, and the size is the supremum over $\Theta_0$, which is the harder problem page 01 measured but still a computation and not an inference.

    For the alternative there is no corresponding object. Power is $\beta_\varphi(\theta_1)=\mathbf{P}_{\theta_1}(\varphi=1)$, a *function* on $\Theta_1$, and simulating it requires choosing $\theta_1$. Since $\Theta_1$ typically has $\theta_0$ in its closure, $\inf_{\theta_1\in\Theta_1}\beta_\varphi(\theta_1)=\alpha$ for any reasonable test: the infimum of power over the alternative equals the size, so the worst-case power carries no information and cannot be improved by any test. Any reported single figure is therefore $\beta_\varphi$ evaluated at a point somebody chose, and the choice is not identified by the data, the model, or the procedure.

    The load-bearing distinction is between a distribution and a set of them. **A size is a computation and a power is a claim, which is why every paper reports the first and almost none report the second.**

The consequence is that calibration is never free, and its price is charged to the column nobody prints. Below, a nominal $5\%$ $t$-test is run on serially correlated returns, first as-is and then with its critical value taken from a simulated null carrying the same dependence — the honest repair. A real edge worth an annualized Sharpe of $0.50$ is planted in the alternative runs:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12047)
n, reps, sd = 1260, 20_000, 0.012
mu = 0.50 / np.sqrt(252) * sd                      # a planted edge worth an annual Sharpe of 0.50

def tstats(phi, mean):
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), (reps, n))
    x = np.empty((reps, n))
    x[:, 0] = e[:, 0]
    for j in range(1, n):
        x[:, j] = phi * x[:, j - 1] + e[:, j]
    x = x + mean
    return np.abs(x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n)))

print("  a nominal 5% t-test on serially correlated returns, before and after calibration")
print("   phi   naive size   calib size   naive power   honest power   power kept")
for phi in (0.00, 0.20, 0.40):
    null = tstats(phi, 0.0)
    alt = tstats(phi, mu)
    c_naive = stats.t.isf(0.025, n - 1)
    c_calib = np.quantile(null, 0.95)
    ns, cs = (null > c_naive).mean(), (null > c_calib).mean()
    np_, hp = (alt > c_naive).mean(), (alt > c_calib).mean()
    print(f"  {phi:.2f}   {ns:10.4f}   {cs:10.4f}   {np_:11.4f}   {hp:12.4f}   {hp / np_:10.1%}")
# =>   a nominal 5% t-test on serially correlated returns, before and after calibration
#       phi   naive size   calib size   naive power   honest power   power kept
#      0.00       0.0522       0.0500        0.1994         0.1939        97.2%
#      0.20       0.1057       0.0500        0.2563         0.1555        60.7%
#      0.40       0.1985       0.0500        0.3099         0.1123        36.2%
```

The first row is the control and behaves: with no dependence the naive size is $0.0522$, calibration moves it to $0.0500$, and the power is essentially untouched at $0.1994$ against $0.1939$, so $97.2\%$ of it survives. Calibration costs nothing when there was nothing to calibrate.

The lower rows are the bill. At $\phi=0.40$ the naive test's size is $0.1985$ — four times its nominal level, the failure [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) diagnoses in detail for overlapping returns — and the naive test appears to have power $0.3099$ against the planted edge. Calibrating the critical value against a simulated null with the same $\phi$ restores the size to exactly $0.0500$, and the power falls to $0.1123$: only $36.2\%$ of the apparent power survives. At $\phi=0.20$ the same trade is $0.2563$ down to $0.1555$, keeping $60.7\%$.

This is the section's thesis in one table. The false positives and the apparent power were the same phenomenon — an inflated statistic — and removing one removes the other. What makes it dangerous is the reporting asymmetry: the size column can be computed by anyone, before any data arrives, and the honest-power column requires naming the planted edge, which no convention obliges anyone to do. So the repair is adopted, the size is fixed, the paper says so, and the two-thirds of power that left with the false positives is never mentioned, because measuring it would have required stating what the strategy was supposed to be earning.

!!! warning "A test whose size was repaired to nominal has bought that repair with power it did not report, and the second row is free to compute"
    Every honest correction on the following pages — HAC standard errors, block permutation, the stationary bootstrap, the calibrated critical values above — widens the null distribution, and widening the null lowers the power against every alternative simultaneously. That is not a defect of the corrections; an uncalibrated test's extra power was never real, since it was bought by rejecting true nulls at four times the advertised rate. The defect is in the reporting convention, which asks for the size and not the power, so the correction looks free. **The free diagnostic is to run the repair twice: once on your own simulated null to confirm the size lands near $\alpha$, and once on the same null with an effect of the size you would actually trade added to it, and read the second rejection rate — the drop from the naive test's apparent power is the true price of the fix, and if the calibrated power comes back near $\alpha$, the corrected test cannot detect the edge you are looking for and the study is finished before it starts.**

## One Error Is Chosen and the Other Is Inherited, and the Report Names Only the Chosen One

This page established that the two errors are expectations under different measures and therefore do not combine into an accuracy without a prior nobody supplies; that at fixed sample size the achievable pairs form one convex decreasing frontier generated by the likelihood ratio, so a true Sharpe of $0.30$ on twenty-four years yields $43.04\%$ power at the conventional level and would need $\alpha=0.2650$ for $80\%$; that the cost-minimizing level is a likelihood-ratio threshold set by the cost ratio and prior odds, running from $0.001$ to $0.600$ across fifteen plausible cells, none of which puts it within $0.005$ of $0.05$; that a desk pays for the two errors in currencies of which only one is recorded; and that a size can be simulated to arbitrary precision without data while a power cannot be computed at all without naming an alternative, so repairing a $\phi=0.40$ series' size from $0.1985$ to $0.0500$ silently discards $63.8\%$ of the apparent power.

The through-line is a division of labour that nobody agreed to. One error is chosen, in advance, by convention, and reported; the other is inherited from the sample size, the statistic, the dependence and the effect, and is reported by no one. The course's two VaR models are the clearest case: both were graded, each failed a different test, and the lesson had to say in words that these were "two separate repairs" because no number in either row could express it. The historical model's `Kupiec p 9.51e-01` and `indep p 0.0013` are not in tension and do not average — they are two measurements of two errors, and reading either alone grades a different model than the one being run.

What follows is the half this page kept deferring. The Type II rate is a function on the alternative, that function can be computed in advance, inverted for a sample size, and audited before a single observation is collected — and when it is computed for the effects trading actually hunts, the answers are severe enough to determine which questions are worth asking at all. That is [Statistical Power](05-statistical-power.md).

**The level is the error you chose and the power is the error you got, and only the first one has a convention, a name in the abstract, and a place on the report.**
