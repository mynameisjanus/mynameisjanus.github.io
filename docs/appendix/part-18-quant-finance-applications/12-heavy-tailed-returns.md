# Heavy-Tailed Returns

The appendix has said repeatedly that returns are not Gaussian and has named the consequences one at a time — a mean that will not converge, a kurtosis that grows with the sample, a Cornish–Fisher correction that is undefined. This page supplies the property underneath all of them. **Regular variation** says the tail is a power law, which means it has no scale: rescaling the threshold rescales the probability by a fixed factor whatever the threshold was. Two things follow that no amount of extra moment-counting delivers. The first is max-sum equivalence — a large total is one large term rather than an accumulation, measured here as the largest of forty independent losses accounting for $0.9331$ of the total at $\alpha=1.5$ and $0.0960$ at $\alpha=6.0$, conditional on the total being extreme. The second is that the tail index is estimable, and estimable well: the Hill estimator settles onto $2.502$, $3.993$ and $6.014$ against truths of $2.5$, $4.0$ and $6.0$, with a standard error falling like $1/\sqrt k$. The failure is what it does when there is no tail index to find. On a Student $t(3)$ it drifts from $2.920$ to $2.135$ as more of the sample is used, and on a lognormal — which has no power-law tail at all — it returns $3.005$, $2.764$, $2.401$, $2.119$ and $1.795$, each with a standard error small enough to exclude the next, and the $1$-in-$10{,}000$ loss it extrapolates moves from $48.09$ to $161.78$ against a truth of $41.22$.

This page covers regular variation as the definition of a heavy tail, the tail index and what it indexes, max-sum equivalence and the sense in which diversification fails in the far tail, the Hill estimator with its consistency and its second-order bias, and what the estimator reports on laws that have no tail index. It does not catalogue which moments exist, derive the standardized moments, or establish that a sample kurtosis is not an estimate of anything, all of which are [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md); it does not construct the Student $t$ as a scale mixture or develop its moment boundary, which is [Student's t-Distribution](../part-05-common-distributions/16-students-t-distribution.md); it does not prove the laws of large numbers or the Central Limit Theorem, or re-derive why a Cauchy mean does not converge, which are [Part VII](../part-07-asymptotic-theory/index.md); it does not develop the limit laws for maxima, the generalized Pareto, or threshold choice as a bias–variance problem, which is [Extreme Value Theory](13-extreme-value-theory.md); it does not fit competing families to real returns or rank them by information criteria, which is [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md); it computes no risk measure, which is [Value at Risk](10-value-at-risk.md); and it never reports a tail index without the range of thresholds over which it held still.

The trading stake is a fitted number the course treats as its fat-tail headline. [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) fits four families to SPY daily log returns by maximum likelihood and prints `fitted t df = 2.65`, calling the gap to the normal "nearly a thousand log-likelihood points, a magnitude of evidence that closes the question," while noting that a fitted stable's $\alpha=1.53$ "overshoots what data supports." Both numbers are tail indices, they disagree by a factor of $1.7$, and the lesson's judgement that one is credible and the other is not is exactly the judgement sections 3 and 4 make computable: the question is not which family fits best in the body but whether the estimate holds still as the tail is entered.

## Regular Variation Is the Statement That a Tail Has No Scale

Heavy-tailedness is usually introduced as a list of symptoms. It has a definition, and every symptom is a corollary of it.

??? note "Proof that regular variation makes the tail scale-free, that moments exist exactly below the index, and that a sum's tail is the largest term's tail"

    A survival function $S(x)=\mathbf{P}(X>x)$ is **regularly varying** with index $\alpha>0$ if for every $t>0$
    $$\lim_{x\to\infty}\frac{S(tx)}{S(x)}=t^{-\alpha}.$$
    The content is that the ratio depends on $t$ and not on $x$: doubling a threshold divides the exceedance probability by $2^{\alpha}$ whether the threshold was two standard deviations or twenty. A Gaussian has no such property, its ratio collapsing to zero, which is the precise sense in which it has a scale and a power law does not.

    Moments follow immediately. Writing $\mathbb{E}[X^{k}]=k\int_0^\infty x^{k-1}S(x)\,dx$ and substituting $S(x)\approx cx^{-\alpha}$, the integrand behaves like $x^{k-1-\alpha}$, which is integrable at infinity exactly when $k<\alpha$. So the index *is* the moment boundary — the result [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md) establishes for its own purposes, here recovered as a consequence rather than a definition.

    The deeper consequence is about sums. For $X_1,\dots,X_n$ independent and regularly varying,
    $$\mathbf{P}(X_1+\dots+X_n>x)\;\sim\;n\,\mathbf{P}(X>x)\;\sim\;\mathbf{P}\!\left(\max_i X_i>x\right)\qquad(x\to\infty),$$
    which is **max-sum equivalence**. The reasoning is that the cheapest way for a sum to be enormous is for one term to be enormous: any route requiring two terms to be large simultaneously costs a product of small probabilities, and under a power law that product is negligible against the $n$ single-term routes. Light-tailed laws behave oppositely — an exponential sum reaches a high level by many terms being moderately large, because the single-term route is exponentially penalized.

    **The load-bearing feature is that these are all statements about $x\to\infty$. Regular variation constrains the tail's shape in the limit and says nothing about any finite threshold, so every consequence here is asymptotic and the practical question is always whether the data has reached the regime — which is section 4's failure, arriving as an estimator that cannot tell.**

## A Large Total Loss Is One Large Loss, and Only Once the Total Is Genuinely Extreme

Max-sum equivalence is often paraphrased as "diversification fails under heavy tails," which overstates it. Measuring where it holds and where it does not is the useful version.

```python
import numpy as np

rng = np.random.default_rng(18121)
REPS, N = 200_000, 40

print(f"  a sum of {N} independent losses from a regularly varying law with tail index alpha."
      f" Max-sum equivalence says P(sum > x) ~ n P(X > x): one term dominates the whole sum, so"
      f" a large total loss is one large loss rather than many moderate ones. {REPS:,} draws")
print("     alpha   moments that exist   P(largest > 50% of the sum)   > 80%   > 95%"
      "   mean share of the largest   ratio at the 99.9th percentile of the sum")
for alpha in (1.5, 2.5, 3.5, 6.0):
    x = (1 - rng.random((REPS, N))) ** (-1 / alpha)     # Pareto(alpha), support [1, inf)
    tot, mx = x.sum(axis=1), x.max(axis=1)
    share = mx / tot
    hi = tot >= np.quantile(tot, 0.999)
    print(f"    {alpha:6.1f}   {int(np.floor(alpha)):18d}   {np.mean(share > 0.50):28.4f}"
          f"   {np.mean(share > 0.80):6.4f}   {np.mean(share > 0.95):6.4f}"
          f"   {share.mean():27.4f}   {share[hi].mean():40.4f}")

print("\n     the same statistic for a light-tailed law of matched mean, where no term dominates")
for name, draw in (("exponential", lambda s: rng.exponential(1.0, s)),
                   ("lognormal", lambda s: rng.lognormal(-0.5, 1.0, s))):
    x = draw((REPS, N))
    tot, mx = x.sum(axis=1), x.max(axis=1)
    share = mx / tot
    hi = tot >= np.quantile(tot, 0.999)
    print(f"    {name:11s}   {'all':>18}   {np.mean(share > 0.50):28.4f}"
          f"   {np.mean(share > 0.80):6.4f}   {np.mean(share > 0.95):6.4f}"
          f"   {share.mean():27.4f}   {share[hi].mean():40.4f}")
# =>   a sum of 40 independent losses from a regularly varying law with tail index alpha. Max-sum equivalence says P(sum > x) ~ n P(X > x): one term dominates the whole sum, so a large total loss is one large loss rather than many moderate ones. 200,000 draws
#         alpha   moments that exist   P(largest > 50% of the sum)   > 80%   > 95%   mean share of the largest   ratio at the 99.9th percentile of the sum
#           1.5                    1                         0.0393   0.0048   0.0004                        0.1884                                     0.9331
#           2.5                    2                         0.0012   0.0001   0.0000                        0.0926                                     0.6064
#           3.5                    3                         0.0000   0.0000   0.0000                        0.0643                                     0.3018
#           6.0                    6                         0.0000   0.0000   0.0000                        0.0434                                     0.0960
#
#         the same statistic for a light-tailed law of matched mean, where no term dominates
#        exponential                  all                         0.0000   0.0000   0.0000                        0.1070                                     0.1068
#        lognormal                    all                         0.0011   0.0000   0.0000                        0.1437                                     0.4113
```

The unconditional columns are almost featureless — even at $\alpha=1.5$, the largest of forty losses exceeds half the total on only $0.0393$ of draws, and its average share is $0.1884$ against the $0.0250$ that equal contributions would give. On a typical day a heavy-tailed book looks diversified, because it is.

The last column is where the theorem lives. Conditioning on the total being in its worst thousandth, the largest single term accounts for $0.9331$ of it at $\alpha=1.5$, $0.6064$ at $\alpha=2.5$, $0.3018$ at $\alpha=3.5$ and $0.0960$ at $\alpha=6.0$ — a monotone collapse toward the light-tailed behaviour as the index rises. The exponential control sits at $0.1068$ conditionally against $0.1070$ unconditionally, which is the signature of a law where conditioning on a large total tells you nothing about how it was assembled.

The lognormal row is worth its own sentence, because it reads $0.4113$ conditionally against $0.1437$ unconditionally and is *not* regularly varying. It is subexponential, a strictly larger class in which max-sum equivalence still holds without any power-law index existing — which is why it will pass every diagnostic in section 3 and break the extrapolation in section 4. **Max-sum equivalence is a statement about the far tail only, invisible in the body of the distribution and inaudible to any diagnostic run at a typical quantile.**

## The Hill Estimator Converges Wherever a Power Law Actually Exists

If the tail is a power law, its index is a single number and there is a natural estimator for it, whose properties are as good as anything in this part.

??? note "Proof that the Hill estimator is the maximum-likelihood estimator of the tail index above a threshold, is consistent, has asymptotic standard error $\alpha/\sqrt k$, and carries a bias set by a second-order parameter"

    Condition on the $k$ largest observations $X_{(n)}\ge\dots\ge X_{(n-k+1)}$ and let $u=X_{(n-k)}$ be the next one down. If the tail is exactly Pareto above $u$, then $\log(X_{(i)}/u)$ is exponential with rate $\alpha$, and the maximum-likelihood estimate of that rate is the reciprocal of the sample mean:
    $$\hat\alpha_{\mathrm{Hill}}=\left[\frac{1}{k}\sum_{i=1}^{k}\log\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}.$$
    Being a maximum-likelihood estimator for an exponential rate on $k$ observations, it is consistent as $k\to\infty$ and asymptotically normal with standard error $\alpha/\sqrt k$: the precision depends on how many order statistics are used and not at all on the sample size behind them.

    The bias is the whole difficulty and it comes from the tail being only *approximately* Pareto. Regular variation says $S(x)=cx^{-\alpha}(1+o(1))$, and the rate at which that $o(1)$ vanishes is governed by a **second-order parameter** $\rho<0$, with $S(x)=cx^{-\alpha}(1+dx^{\rho}+\dots)$. Including observations further from the tail adds data — reducing variance like $1/\sqrt k$ — while admitting more of the $dx^{\rho}$ contamination, so the bias grows in $k$. The mean squared error therefore has an interior optimum in $k$, and its location depends on $\rho$ and $d$, neither of which is known and both of which are harder to estimate than $\alpha$ itself.

    **The load-bearing consequence is that the estimator has one tuning parameter, no data-free rule for setting it, and a diagnostic that is not a number but a shape: the estimate must be plotted against $k$ and read where it is flat. A single value of $\hat\alpha$ reported without that plot has had its most important input chosen silently.**

```python
import numpy as np

rng = np.random.default_rng(18123)
N, REPS = 5_000, 400
KS = (25, 50, 100, 250, 500, 1_000)


def hill(x, k):
    """Hill estimator of the tail index from the k largest order statistics."""
    top = np.sort(x)[-(k + 1):]
    return 1.0 / (np.log(top[1:]).mean() - np.log(top[0]))


print(f"  the Hill estimator reads the k largest of {N:,} observations and returns an estimate of"
      f" the tail index alpha. On a law that really is a power law the estimate settles onto the"
      f" truth and its standard error falls like one over the square root of k. {REPS} replications")
print("     law                  true alpha   " + "".join(f"k={k}: mean   sd   " for k in KS))
for name, alpha, draw in (
        ("Pareto(2.5)", 2.5, lambda s: (1 - rng.random(s)) ** (-1 / 2.5)),
        ("Pareto(4.0)", 4.0, lambda s: (1 - rng.random(s)) ** (-1 / 4.0)),
        ("Pareto(6.0)", 6.0, lambda s: (1 - rng.random(s)) ** (-1 / 6.0))):
    x = draw((REPS, N))
    cells = ""
    for k in KS:
        est = np.array([hill(xr, k) for xr in x])
        cells += f"{est.mean():10.3f}   {est.std():4.3f}   "
    print(f"    {name:20s} {alpha:11.1f}   " + cells)
# =>   the Hill estimator reads the k largest of 5,000 observations and returns an estimate of the tail index alpha. On a law that really is a power law the estimate settles onto the truth and its standard error falls like one over the square root of k. 400 replications
#         law                  true alpha   k=25: mean   sd   k=50: mean   sd   k=100: mean   sd   k=250: mean   sd   k=500: mean   sd   k=1000: mean   sd   
#        Pareto(2.5)                  2.5        2.652   0.561        2.571   0.389        2.523   0.246        2.509   0.155        2.503   0.106        2.502   0.075   
#        Pareto(4.0)                  4.0        4.172   0.829        4.072   0.653        4.037   0.435        3.995   0.243        3.996   0.183        3.993   0.123   
#        Pareto(6.0)                  6.0        6.292   1.339        6.114   0.899        6.026   0.616        6.007   0.389        6.024   0.273        6.014   0.184   
```

On data that is genuinely Pareto the estimator behaves exactly as advertised. Each row settles onto its truth — $2.652$, $2.571$, $2.523$, $2.509$, $2.503$, $2.502$ against $2.5$; $4.172$ down to $3.993$ against $4.0$; $6.292$ down to $6.014$ against $6.0$ — with the small-$k$ upward bias that a maximum-likelihood rate estimate on few observations always carries, and with standard errors of $0.561$, $0.389$, $0.246$, $0.155$, $0.106$ and $0.075$ that track $\alpha/\sqrt k$ across a fortyfold change in $k$.

The plateau is what a correct fit looks like: the last four columns of the first row agree to within $0.021$, which is well inside a single standard error, so any threshold in that range gives the same answer. **The flatness is not a convenience of this example. It is the only available evidence that a power law exists at all, and section 4 is what the same picture looks like when it does not.**

## On Two Laws That Are Not Power Laws It Returns a Confident Number That Moves by a Factor of Four

The estimator asks "if the tail is a power law, what is its index." It is not able to decline the premise, and the number it returns when the premise is false is what an extrapolation gets built on.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18125)
N, REPS, P_FAR = 5_000, 400, 1 - 1e-4
KS = (50, 100, 250, 500, 1_000)


def hill_quantile(x, k, p):
    """Hill's tail index and the Weissman extrapolation to a quantile beyond the sample."""
    s = np.sort(x)
    top = s[-(k + 1):]
    a = 1.0 / (np.log(top[1:]).mean() - np.log(top[0]))
    return a, top[0] * (k / (len(x) * (1 - p))) ** (1 / a)


print(f"  the same estimator used for what it is for: extrapolating to the 1-in-{1 / (1 - P_FAR):,.0f}"
      f" loss from {N:,} observations, which is twice as far out as the sample reaches. On a true"
      f" power law the answer is stable in k; on the two laws below it is not, and neither is a"
      f" power law. {REPS} replications")
print("     law                true quantile   " + "".join(f"k={k}: alpha   quantile   " for k in KS))
for name, truth, draw in (
        ("Pareto(2.5)", (1 - P_FAR) ** (-1 / 2.5), lambda s: (1 - rng.random(s)) ** (-1 / 2.5)),
        ("Student t(3)", stats.t.ppf(P_FAR, 3.0), lambda s: np.abs(rng.standard_t(3.0, s))),
        ("lognormal(0, 1)", stats.lognorm.ppf(P_FAR, 1.0), lambda s: rng.lognormal(0.0, 1.0, s))):
    x = draw((REPS, N))
    cells = ""
    for k in KS:
        out = np.array([hill_quantile(xr, k, P_FAR) for xr in x])
        cells += f"{out[:, 0].mean():10.3f}   {np.median(out[:, 1]):8.2f}   "
    print(f"    {name:18s} {truth:13.2f}   " + cells)
# =>   the same estimator used for what it is for: extrapolating to the 1-in-10,000 loss from 5,000 observations, which is twice as far out as the sample reaches. On a true power law the answer is stable in k; on the two laws below it is not, and neither is a power law. 400 replications
#         law                true quantile   k=50: alpha   quantile   k=100: alpha   quantile   k=250: alpha   quantile   k=500: alpha   quantile   k=1000: alpha   quantile   
#        Pareto(2.5)                39.81        2.530      39.44        2.518      39.60        2.511      39.57        2.509      39.36        2.504      39.59   
#        Student t(3)               22.20        2.920      28.52        2.829      29.79        2.658      32.96        2.462      38.90        2.135      57.57   
#        lognormal(0, 1)            41.22        3.005      48.09        2.764      53.17        2.401      68.58        2.119      93.60        1.795     161.78   
```

The Pareto row is the control and it works: the extrapolated $1$-in-$10{,}000$ loss reads $39.44$, $39.60$, $39.57$, $39.36$ and $39.59$ against a truth of $39.81$, stable across a twentyfold change in $k$ and accurate despite being computed twice as far out as the sample reaches.

The Student $t(3)$ is regularly varying with index exactly $3$, and the estimator still fails, which is the subtler half of the finding. Its second-order parameter is such that convergence is slow, so $\hat\alpha$ drifts from $2.920$ down to $2.135$ as more of the sample is admitted, and the extrapolated quantile inflates from $28.52$ to $57.57$ against a truth of $22.20$ — a factor of $2.6$ across the range of $k$ a practitioner might reasonably choose, and an overstatement of the truth by $2.6\times$ at the far end. The premise was correct and the asymptotics had not arrived.

The lognormal is the honest failure. It has no tail index at all: no $\alpha$ exists, so every number in that row is an artefact. The estimator nonetheless returns $3.005$, $2.764$, $2.401$, $2.119$ and $1.795$, and section 3's table says each of those carries a standard error of about $0.4/\sqrt k$ — so the $k=1000$ estimate of $1.795$ excludes the $k=500$ estimate of $2.119$ at many standard errors, and both are answers to a question with no answer. The extrapolated loss runs $48.09$, $53.17$, $68.58$, $93.60$, $161.78$ against a true $41.22$, ending a factor of $3.9$ too high. **The estimator has no way to report "there is no tail index here," and the only symptom is that its answer will not sit still — which is precisely the symptom that a practitioner selecting a single $k$ never sees.**

!!! note "Heavy-tailed, fat-tailed, subexponential and regularly varying are four descriptions of a tail, and only the last one supports an extrapolation"
    **Heavy-tailed** in the strict sense means the moment generating function is infinite for every positive argument, which is true of the lognormal and of every law on this page. **Fat-tailed** is informal and usually means excess kurtosis, which is a statement about the fourth moment and therefore silent about laws that have none — the trap [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md) documents. **Subexponential** is the class where max-sum equivalence holds, containing both the lognormal and every regularly varying law, and it is the right class for the statement "a large total is one large term." **Regularly varying** is strictly smaller, adds the power-law index, and is the only one of the four that licenses extrapolating past the sample, because it is the only one that fixes the ratio $S(tx)/S(x)$. Section 4's lognormal is subexponential and not regularly varying, which is exactly why it satisfies the section 2 diagnostic and fails the section 4 extrapolation.

## Every Repair Is a Plot Rather Than a Number, or an Admission That the Class Is Wrong

The three findings converge on one procedural change. Section 3 shows the estimator is excellent when its premise holds; section 4 shows it is silent when the premise fails; and the difference between the two situations is visible only by varying $k$. So the repair is to stop reporting $\hat\alpha$ and start reporting $\hat\alpha(k)$ — a curve, with the estimate read off a flat region and the absence of a flat region treated as a finding rather than as an inconvenience.

That reframes the course's own judgement. [Returns and Distributions](../../part-03-statistics/02-returns-and-distributions.md) prefers a fitted $t$ with $\nu=2.65$ over a fitted stable with $\alpha=1.53$ on the grounds that the latter "overshoots what data supports," and both are tail-index claims made by maximum likelihood over the *whole* distribution, where the body carries almost all the observations and therefore almost all the influence. A likelihood fitted to five thousand days is answering a question about the middle; section 4's $t(3)$ row shows that even the correct family, fitted correctly in the tail, needs more data than that to stop drifting.

!!! warning "A tail-index estimator cannot decline the question, so it answers even where the answer does not exist"
    Every estimator on this page returns a finite number for every input. The lognormal has no tail index, and the estimator reports $1.795$ with a standard error near $0.06$ — a figure precise enough to publish and describing nothing. **The free diagnostic is the estimate as a function of the threshold: compute $\hat\alpha(k)$ across at least a decade of $k$, and treat a variation exceeding a couple of standard errors as evidence that the power-law premise fails rather than as noise to be averaged away.** The Pareto rows move by $0.021$ across their last four columns and the lognormal moves by $1.210$ across the same range; the standard errors are the same in both cases, so the comparison needs no new theory and no new data, only the loop that was already written to produce one number. Where the curve refuses to flatten, the honest report is that the sample has not reached the tail — which is the threshold-choice problem in its general form, and the subject of the next page.

## An Index That Exists, and an Estimator That Cannot Say Whether It Does

This page established that regular variation is the definition of a heavy tail and makes the tail scale-free, with moments existing exactly below the index and a sum's tail equal to its largest term's; that max-sum equivalence is a far-tail statement and nearly invisible elsewhere, the largest of forty losses taking $0.1884$ of the total unconditionally at $\alpha=1.5$ and $0.9331$ of it conditional on the total being in its worst thousandth, against $0.0960$ at $\alpha=6.0$ and $0.1068$ for an exponential; that the Hill estimator is a maximum-likelihood rate estimate with standard error $\alpha/\sqrt k$, settling onto $2.502$, $3.993$ and $6.014$ against truths of $2.5$, $4.0$ and $6.0$ and flattening to within $0.021$ across its last four thresholds; and that it returns an equally confident number where no index exists, drifting $3.005$, $2.764$, $2.401$, $2.119$, $1.795$ on a lognormal and turning a true $1$-in-$10{,}000$ loss of $41.22$ into $161.78$, while even the correctly specified $t(3)$ drifts from $2.920$ to $2.135$ and overstates its truth of $22.20$ by $2.6\times$.

The relationship to the two risk pages before it is one of supply. [Value at Risk](10-value-at-risk.md) and [Expected Shortfall](11-expected-shortfall.md) both found their estimators limited by how few observations sit past the threshold, and both stopped at the edge of the sample; this page is what can be said about the region beyond it, and the answer is that a power law can be extrapolated and nothing else can. What it has not supplied is the threshold. Every result here conditions on $k$ order statistics or on a level $u$ chosen by hand, the choice governs both the bias and the variance, and section 4 showed the consequences of getting it wrong without offering a way to get it right. Turning that into a procedure — with a limit theorem that says what the exceedances converge to, and a criterion for where to cut — is [Extreme Value Theory](13-extreme-value-theory.md).

**A power-law tail is the only kind that can be extrapolated, an estimator will report one whether or not it is there, and the difference between those two situations is a shape rather than a number.**
