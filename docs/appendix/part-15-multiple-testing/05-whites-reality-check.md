# White's Reality Check

The three preceding pages all needed a number the data does not contain. The Reality Check stops asking for it: instead of counting candidates and dividing a level, it resamples the whole family jointly and reads the null distribution of the maximum straight off the candidates' own joint behaviour, so the family's correlation does the work Bonferroni bounds away. That buys a great deal — below, correct size at $0.0500$ where Bonferroni delivers $0.0300$, and detection of a true Sharpe of $0.60$ on $0.2667$ of occasions against Bonferroni's $0.1733$. It is bought by placing the null at the least favourable configuration, and the bill arrives in a form that is easy to miss: a champion with a true Sharpe of $0.80$ that is rejected on $1.0000$ of occasions when tested alone is rejected on $0.7333$ once ninety-nine candidates that cannot possibly win are added beside it — and the damage is the same whether those candidates are merely worthless or lose money outright, $0.0529$ against $0.0882$ in mean p-value, because the procedure erases their means before it looks at them.

This page covers the composite null over a whole family and the recentring that imposes it, the resampling scheme the maximum inherits and what destroying the serial dependence costs, the power that using a family's correlation buys over bounding it, and the way a hopeless candidate still sets the bar its siblings are judged against. It does not construct the bootstrap from first principles or prove the plug-in principle, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it does not develop the modification a resampling scheme needs before it can test rather than estimate, which is [Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md); it divides no level by a count, which is [Bonferroni Correction](02-bonferroni-correction.md); it controls no proportion of discoveries, which is [False Discovery Rate](03-false-discovery-rate.md); it studentizes nothing and recentres nothing conditionally, which is [Hansen's SPA Test](06-hansens-spa-test.md); it recovers no unreported search, which is [Data Snooping Bias](04-data-snooping-bias.md); and it never claims that resampling a family supplies information the family did not contain.

The trading stake is the course deploying this test and reporting the one number the whole part has been building toward. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) takes the fifty-variant momentum grid, resamples its entire history with the stationary bootstrap, and reports `family of 50 lookback variants: best Sharpe +0.46 (lookback 180)` with `Reality Check p-value (stationary bootstrap, 500 draws): 0.06`. The lesson's reading — "the difference between 'my best variant has Sharpe 0.46' and 'my search produced p = 0.06' is the difference between marketing and statistics" — is section 3's result on real data, and its caveat that $500$ draws put a standard error of about $0.011$ on that p-value is section 2's arithmetic.

## The Null Is Composite Over the Whole Family, and Recentring Each Candidate on Its Own Mean Places It at the Configuration Where Rejection Is Hardest

Every test on the preceding pages was a collection of individual hypotheses with a correction bolted on afterwards. The Reality Check is a single test of a single hypothesis about the family as a whole, and the difference is what lets it use the joint distribution.

??? note "Proof that the null $\max_k\mathbb{E}[f_k]\le0$ is composite, that its least favourable configuration is $\mathbb{E}[f_k]=0$ for every $k$, and that subtracting each candidate's sample mean imposes exactly that configuration on the resamples"

    Let $f_k$ be the performance measure of candidate $k$ relative to a benchmark, $k=1,\dots,K$, observed over $n$ periods with sample means $\bar f_k$. The hypothesis that the search found nothing is
    $$H_0:\ \max_{k}\mathbb{E}[f_k]\le0,$$
    and the natural statistic is $\bar V=\max_k\sqrt n\,\bar f_k$. This null is *composite*: it is satisfied by every parameter vector $\boldsymbol{\mu}=(\mathbb{E}[f_1],\dots,\mathbb{E}[f_K])$ in the negative orthant, and the distribution of $\bar V$ differs across them. A valid test must control size at the worst of them.

    That worst case is $\boldsymbol{\mu}=\mathbf{0}$. Write $\sqrt n\,\bar f_k=\sqrt n\,\mu_k+Z_k$ with $Z_k$ the centred sampling error. Then $\bar V=\max_k(\sqrt n\mu_k+Z_k)$ is non-decreasing in every $\mu_k$, so $P_{\boldsymbol{\mu}}(\bar V>c)$ is maximised over the null region at $\boldsymbol{\mu}=\mathbf{0}$, and any $c$ controlling size there controls it everywhere in $H_0$. This is the least favourable configuration, and it is why the procedure is valid for families containing outright loss-makers.

    The bootstrap must therefore generate draws under $\boldsymbol{\mu}=\mathbf{0}$ rather than under whatever the data suggests. Resampling the raw $f_k$ reproduces the *observed* means, which is the estimation problem rather than the testing one — [Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md)'s central point that a resampling scheme must be modified before it can test. White's modification is to recentre each candidate on its own sample mean:
    $$\bar V^{*}_{b}=\max_k\sqrt n\,\big(\bar f^{*}_{b,k}-\bar f_k\big),\qquad p=\frac{1}{B}\sum_{b=1}^{B}\mathbb{1}\{\bar V^{*}_{b}\ge\bar V\},$$
    so every candidate enters the resampled maximum with mean zero while retaining its own variance and its covariance with the rest. That last clause is the whole gain over Bonferroni: the resamples inherit the family's joint dependence structure without anyone having to estimate, name or assume it.

    The load-bearing consequence is that recentring is applied to *every* candidate, including ones whose observed mean is far below zero. **The least favourable configuration is imposed rather than tested for, so a candidate that lost money throughout the sample enters the null distribution of the maximum as though it had broken even, and contributes to the bar its siblings must clear.**

## The Maximum Inherits Whatever Dependence the Resampling Scheme Preserves, and Discarding the Serial Dependence Costs Twelve Times the Nominal Size

The recentring settles what is resampled. How it is resampled is a separate choice, and on financial data it is the difference between a test and a random number generator:

```python
import numpy as np
from scipy import signal

rng = np.random.default_rng(15051)
n, k, B, reps = 1260, 20, 499, 400
row = np.repeat(np.arange(B), n) * n
ar = np.arange(n)


def resample(p):
    """Stationary-bootstrap indices, B by n; geometric blocks of mean 1/p (p=1 is iid)."""
    new = rng.random((B, n)) < p
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    return (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n


def reality_check(f, p):
    """White's p-value for max_k E[f_k] <= 0, recentring each column on its own mean."""
    idx = resample(p)
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    boot = (C @ f) / n - f.mean(0)
    return (np.sqrt(n) * boot.max(1) >= np.sqrt(n) * f.mean(0).max()).mean()


print(f"  White's Reality Check under the global null: {k} worthless strategies,"
      f" n = {n:,}, B = {B}, {reps} replications")
print("     phi   iid bootstrap   stationary mb=10   stationary mb=50")
for phi in (0.0, 0.3, 0.5):
    hit = np.zeros(3)
    for _ in range(reps):
        e = rng.standard_normal((n, k))
        f = signal.lfilter([1.0], [1.0, -phi], e, axis=0) if phi else e
        hit += [reality_check(f, q) < 0.05 for q in (1.0, 0.1, 0.02)]
    print(f"    {phi:4.2f}   {hit[0] / reps:13.4f}   {hit[1] / reps:16.4f}"
          f"   {hit[2] / reps:16.4f}")
# =>   White's Reality Check under the global null: 20 worthless strategies, n = 1,260, B = 499, 400 replications
#         phi   iid bootstrap   stationary mb=10   stationary mb=50
#        0.00          0.0525             0.0525             0.0550
#        0.30          0.3550             0.0725             0.0700
#        0.50          0.6225             0.0875             0.0625
```

The $\phi=0$ row is the control and everything passes it: with serially independent data all three schemes deliver $0.0525$, $0.0525$ and $0.0550$ against a nominal $0.05$. Whatever goes wrong below is caused by the dependence rather than by the recentring, the maximum, or the implementation.

The iid column is what happens when the scheme discards that dependence. At $\phi=0.3$ — mild persistence, far below what overlapping-window strategies generate — a nominal $5\%$ test rejects the true null $35.50\%$ of the time. At $\phi=0.5$ it rejects $62.25\%$ of the time. The mechanism is that resampling observations one at a time destroys the serial correlation, so the bootstrap means are far less variable than the real ones; the null distribution of the maximum comes out too narrow, and the observed maximum clears it on most samples. A twelvefold over-rejection is not a conservative approximation. It is a procedure that finds a superior strategy in almost two families out of three when none exists.

The stationary columns are the repair working. Resampling in blocks of random geometric length — mean ten days and mean fifty — keeps size at $0.0725$ and $0.0700$ at $\phi=0.3$ and $0.0875$ and $0.0625$ at $\phi=0.5$. The residual excess above $0.05$ is real and small, the usual finite-sample cost of a block scheme, and it moves in the safe direction relative to the alternative.

[Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md) measures the same repair on a *single* mean under AR(1) and finds an iid size of $0.1520$ against a stationary $0.0520$ — a threefold error. Here the statistic is a maximum over twenty candidates and the error is twelvefold. **The damage a wrong resampling scheme does grows with the number of candidates in the maximum, because every candidate's variance is understated and the maximum compounds twenty understatements rather than suffering one.** Block length is the one free parameter and the table shows it is forgiving: a factor of five in mean block length moves size by less than three points, so the choice worth caring about is blocks against no blocks, not ten against fifty.

## Resampling the Family Jointly Recovers the Power That Bonferroni Spends Bounding a Dependence It Refuses to Look At

Section 2 established the scheme; what the joint resampling actually buys is a separate question, and the comparison is against the procedure of [Bonferroni Correction](02-bonferroni-correction.md) with the size question already settled:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(15053)
n, k, B, reps, rho, mb = 2520, 20, 499, 300, 0.7, 10
row = np.repeat(np.arange(B), n) * n
ar = np.arange(n)
step = 0.05 / (k - np.arange(k))


def reality_check(f):
    """White's p-value, stationary bootstrap with mean block length mb."""
    new = rng.random((B, n)) < 1 / mb
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    idx = (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    boot = (C @ f) / n - f.mean(0)
    return (boot.max(1) >= f.mean(0).max()).mean()


print(f"  {k} strategies correlated at {rho}, n = {n:,} ({n // 252} years), one of them"
      f" carries the edge")
print(f"  serially independent, so section 2's scheme question does not arise;"
      f" B = {B}, {reps} replications")
print("    true SR   Reality Check   Bonferroni   Holm   uncorrected")
for sr in (0.0, 0.4, 0.6, 0.8, 1.0):
    hit = np.zeros(4)
    for _ in range(reps):
        f = (np.sqrt(rho) * rng.standard_normal((n, 1))
             + np.sqrt(1 - rho) * rng.standard_normal((n, k)))
        f[:, 0] += sr / np.sqrt(252)
        t = np.sqrt(n) * f.mean(0) / f.std(0, ddof=1)
        p = stats.norm.sf(t)
        ps = np.sort(p)
        over = ps > step
        nh = np.argmax(over) if over.any() else k
        hit += [reality_check(f) < 0.05, (p < 0.05 / k).any(), nh > 0, (p < 0.05).any()]
    print(f"    {sr:7.2f}   {hit[0] / reps:13.4f}   {hit[1] / reps:10.4f}"
          f"   {hit[2] / reps:4.4f}   {hit[3] / reps:11.4f}")
# =>   20 strategies correlated at 0.7, n = 2,520 (10 years), one of them carries the edge
#      serially independent, so section 2's scheme question does not arise; B = 499, 300 replications
#        true SR   Reality Check   Bonferroni   Holm   uncorrected
#           0.00          0.0500       0.0300   0.0300        0.2667
#           0.40          0.1100       0.0567   0.0567        0.3833
#           0.60          0.2667       0.1733   0.1733        0.6367
#           0.80          0.5233       0.3967   0.3967        0.8533
#           1.00          0.7633       0.6367   0.6367        0.9133
```

The first row is a size comparison and it is the reason for everything below it. The Reality Check delivers $0.0500$ — the nominal level, exactly — while Bonferroni and Holm deliver $0.0300$. That is [Bonferroni Correction](02-bonferroni-correction.md)'s conservatism appearing again on a family correlated at $0.7$, and it is being spent rather than banked: a procedure sitting at $0.03$ when it was allowed $0.05$ has given away the difference. The uncorrected column at $0.2667$ is a further internal check, since twenty tests correlated at $0.7$ behaving like far fewer independent ones is exactly the effective-test-count result of [Multiple Comparisons](01-multiple-comparisons.md).

The remaining rows are what that recovered size is worth. Against a true annualized Sharpe of $0.40$ the Reality Check rejects $0.1100$ of the time against Bonferroni's $0.0567$; at $0.60$ it is $0.2667$ against $0.1733$; at $0.80$, $0.5233$ against $0.3967$; at $1.00$, $0.7633$ against $0.6367$. The relative gain is largest exactly where research operates — a factor of nearly two at the low Sharpes markets actually leave, narrowing to a fifth at Sharpes large enough that any procedure would find them.

Holm's column is identical to Bonferroni's in every row. [Bonferroni Correction](02-bonferroni-correction.md) proved Holm dominates Bonferroni pathwise and measured the gain at half a percentage point; here, with one real effect among twenty candidates, the step-down almost never gets a second comparison and the two procedures coincide. **The step-down refinement and the bootstrap are repairs to the same defect at completely different scales: one loosens the thresholds after the first rejection and buys nothing here, the other replaces the union bound with the family's measured joint distribution and buys a factor of two.**

## A Candidate That Cannot Possibly Win Still Sets the Bar Its Siblings Must Clear, and How Badly It Loses Makes No Difference

The least favourable configuration of section 1 is what makes the test valid without assumptions about the candidates' true means. It is also the source of the one behaviour that surprises practitioners, and the surprise is sharper than the usual telling:

??? note "Proof that after recentring, a candidate's contribution to the null distribution of the maximum depends on its variance and its covariance with the family but not at all on its mean"

    Fix candidate $k$ and write its recentred bootstrap mean as $\sqrt n(\bar f^{*}_{b,k}-\bar f_k)$. The resampling draws indices from the observed sample, so conditional on the data $\mathbb{E}^{*}[\bar f^{*}_{b,k}]=\bar f_k$ exactly and the recentred quantity has conditional mean zero *whatever* $\bar f_k$ was. Its conditional variance is the bootstrap variance of the sample mean, which for a block scheme estimates the long-run variance $\omega_k^{2}$, and its conditional covariance with candidate $j$ estimates the corresponding long-run covariance. Neither depends on $\bar f_k$ or $\bar f_j$.

    Therefore the joint law of $\big(\sqrt n(\bar f^{*}_{b,1}-\bar f_1),\dots,\sqrt n(\bar f^{*}_{b,K}-\bar f_K)\big)$ — and hence the null distribution of its maximum, and hence the critical value — is a function of the family's second moments alone. Adding a candidate shifts that distribution upward by the amount a $(K{+}1)$-th mean-zero coordinate raises a maximum, which is positive whenever the new coordinate is not perfectly correlated with the existing ones, and is *identical* whether the new candidate earned $+0$ or $-100$ over the sample.

    Two consequences follow. The test is valid for any family, since section 1's monotonicity argument covers every negative mean vector — nothing here is a defect in size. And the power against a fixed genuine alternative is a decreasing function of $K$ that does not depend on the alternatives' quality, so a researcher can lower their own p-value by declining to test ideas they already believe are hopeless, which is a perverse incentive built into a correctly specified procedure.

    The load-bearing observation is that recentring is unconditional. **The Reality Check charges for the number of candidates examined and not for the amount of evidence they provided, so a family of a hundred obvious failures costs exactly what a family of a hundred plausible rivals costs.**

The prediction is that adding hopeless candidates degrades the test, and that it makes no difference how hopeless they are:

```python
import numpy as np

rng = np.random.default_rng(15055)
n, B, reps, mb, champ = 6300, 299, 150, 10, 0.80
row = np.repeat(np.arange(B), n) * n
ar = np.arange(n)


def family(k, sib):
    """One champion at true Sharpe champ plus k-1 siblings at true Sharpe sib."""
    f = rng.standard_normal((n, k))
    f[:, 0] += champ / np.sqrt(252)
    f[:, 1:] += sib / np.sqrt(252)
    return f


def reality_check(f):
    """White's p-value, recentring every candidate on its own sample mean."""
    new = rng.random((B, n)) < 1 / mb
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    idx = (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    boot = (C @ f) / n - f.mean(0)
    return (boot.max(1) >= f.mean(0).max()).mean()


print(f"  a champion at a TRUE annualized Sharpe of {champ}, n = {n:,}"
      f" ({n // 252} years), B = {B}, {reps} replications")
print(f"  the siblings cannot win: they are worthless or they lose money outright")
print("    siblings   sibling true SR   mean RC p   P(reject at 5%)   mean champ SR")
for sib in (0.0, -1.0):
    for k in (1, 10, 50, 100):
        if k == 1 and sib != 0.0:
            continue
        pv, rj, sh = 0.0, 0, 0.0
        for _ in range(reps):
            f = family(k, sib)
            p = reality_check(f)
            pv += p
            rj += p < 0.05
            sh += np.sqrt(252) * f[:, 0].mean() / f[:, 0].std(ddof=1)
        lab = "none" if k == 1 else f"{k - 1}"
        print(f"    {lab:>8s}   {sib:15.2f}   {pv / reps:9.4f}   {rj / reps:15.4f}"
              f"   {sh / reps:13.4f}")
# =>   a champion at a TRUE annualized Sharpe of 0.8, n = 6,300 (25 years), B = 299, 150 replications
#      the siblings cannot win: they are worthless or they lose money outright
#        siblings   sibling true SR   mean RC p   P(reject at 5%)   mean champ SR
#            none              0.00      0.0023            1.0000          0.7972
#               9              0.00      0.0157            0.9333          0.8155
#              49              0.00      0.0513            0.8067          0.7794
#              99              0.00      0.0529            0.7333          0.7859
#               9             -1.00      0.0206            0.9200          0.8142
#              49             -1.00      0.0545            0.8400          0.8144
#              99             -1.00      0.0882            0.7333          0.7908
```

The last column is the control and it never moves: the champion's own measured Sharpe is $0.7972$, $0.8155$, $0.7794$, $0.7859$, $0.8142$, $0.8144$, $0.7908$ across every row, because nothing whatever is being done to the champion. All the variation below is the *test's* opinion of an unchanged strategy.

Tested alone, the champion is conclusive: mean p-value $0.0023$, rejected on $1.0000$ of occasions. Nine worthless siblings raise the mean p-value to $0.0157$ and drop the rejection rate to $0.9333$. Forty-nine take it to $0.0513$ and $0.8067$. Ninety-nine take it to $0.0529$ and $0.7333$. A genuinely good strategy, over twenty-five years of data, loses its rejection a quarter of the time because ninety-nine candidates that never had a chance were included in the family it was tested within.

The two blocks together are the proof's second half. Siblings with a true Sharpe of $-1.0$ — losing money steadily and unmistakably for twenty-five years — do essentially the same damage as siblings with a true Sharpe of $0.0$: rejection rates of $0.9200$, $0.8400$, $0.7333$ against $0.9333$, $0.8067$, $0.7333$, with mean p-values in the same neighbourhood. The recentring subtracted their means before the maximum was taken, so the null distribution never learned that they were terrible. **The Reality Check cannot distinguish a family of a hundred serious rivals from a family of ninety-nine catastrophes and one good idea, because the only thing it takes from a candidate is its second moments.**

This is a defect in power and never in size, and the fix cannot be to drop candidates by eye, since that is [Data Snooping Bias](04-data-snooping-bias.md) reintroduced in its purest form. A principled version — dropping candidates whose evidence places them so far below zero that no reasonable null could have generated them — is what [Hansen's SPA Test](06-hansens-spa-test.md) constructs, and section 4's two blocks are precisely the comparison on which it succeeds or fails.

## The Bootstrap Is Inconsistent for a Maximum and This Maximum Is Not One of the Cases

[Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) devotes a section to the bootstrap's failure on extremes, and a reader arriving here with that in mind should be told why it does not apply, because the two situations look identical and are not.

The inconsistency result concerns a maximum over the *sample*: estimating $\theta=\max_i X_i$ or the distribution of a sample extreme, where the bootstrap fails because a resample can never exceed the largest observed value and the resampled maximum is degenerate at it. The set being maximised over grows with $n$, and the limit involves extreme-value rather than Gaussian behaviour.

The Reality Check maximises over the $K$ *candidates*, with $K$ fixed and $n\to\infty$. Each $\sqrt n\,\bar f_k$ obeys a central limit theorem, the vector of them converges jointly to a $K$-dimensional Gaussian, and the maximum is a continuous function of that vector — so the continuous mapping theorem of [The Continuous Mapping Theorem](../part-07-asymptotic-theory/06-continuous-mapping-theorem.md) transfers the convergence to the statistic, and the bootstrap is consistent for it exactly as it is for any smooth functional of a mean. Nothing is being maximised over a growing index set.

The distinction is worth holding because it identifies precisely when the Reality Check *would* break: when $K$ grows with $n$ fast enough that the Gaussian approximation to the maximum fails, which is the regime of a machine-generated search over millions of specifications on a few thousand observations. At $K=50$ and $n=6{,}048$, the course's configuration, the asymptotics are comfortable. At $K=10^{6}$ they are not, and no amount of resampling repairs it.

One further limitation is arithmetic rather than asymptotic and is the course's own caveat. The p-value is a sample proportion over $B$ draws, so it carries a standard error of $\sqrt{p(1-p)/B}$: at $p=0.06$ and $B=500$ that is $0.011$, which straddles the conventional threshold. A Reality Check p-value near $0.05$ reported without $B$ is missing the quantity that decides whether it means anything, and the repair is free — raise $B$ until the standard error is small against the distance from the threshold.

!!! note "The iid bootstrap, the moving-block bootstrap, the circular-block bootstrap and the stationary bootstrap are four resampling schemes for one series, and only the last returns a resample that is stationary"
    **The iid bootstrap** draws observations one at a time with replacement, which is correct for independent data and, section 2 measures, rejects a true null $62.25\%$ of the time at nominal $5\%$ once the data is AR(1) at $\phi=0.5$. **The moving-block bootstrap** draws fixed-length blocks from all $n-\ell+1$ overlapping positions, preserving dependence within a block and destroying it across block boundaries; observations near the ends of the series appear in fewer blocks than those in the middle, so the resample is not stationary and the block-boundary effects bias the variance estimate. **The circular-block bootstrap** wraps the series into a ring before drawing blocks, which equalises the number of blocks each observation can appear in and removes the end effects, at the cost of joining the end of the sample to its beginning as though they were adjacent. **The stationary bootstrap** of Politis and Romano draws blocks whose lengths are geometric with mean $1/p$, and the randomisation of the length is exactly what makes the resampled series stationary — a fixed block length imposes a periodicity the original series does not have. The distinction that matters operationally is that all three block schemes are consistent for a mean under weak dependence and differ mainly in finite-sample bias, while the choice between blocks and no blocks changes the size of the test by an order of magnitude.

!!! warning "A Reality Check p-value is a statement about the family submitted to it, and a researcher can lower it by testing fewer of the ideas they already doubt"
    Section 4 measured a champion at a true Sharpe of $0.80$ rejected on $1.0000$ of occasions alone, $0.9333$ with nine worthless siblings, $0.8067$ with forty-nine and $0.7333$ with ninety-nine — and rejected on $0.7333$ of occasions with ninety-nine siblings that lost money at a rate of $-1.0$ Sharpe for twenty-five years, which is to say the quality of the siblings did not matter at all. Nothing in the output reveals this. The p-value is reported as a property of the champion, and it is a property of the champion *and the list it was submitted with*, a list the reader cannot see and the procedure cannot audit. The incentive this creates runs opposite to every other result in this part: [Data Snooping Bias](04-data-snooping-bias.md) shows a researcher is rewarded for *under*-reporting the search, and here they are rewarded for it again by a procedure specifically designed to charge for search — because the charge is levied per candidate rather than per unit of evidence. **The free diagnostic is to run the test twice, once on the family as submitted and once with the champion removed: if the null distribution of the maximum barely moves, the bar the champion had to clear was being set by its siblings rather than by its own sampling variability, and the p-value is measuring the shape of the candidate list.** Report the family's size and composition beside the p-value as a matter of course, since a Reality Check p-value without them is a number whose denominator has been withheld.

## A Null Distribution Read From the Family Instead of Assumed About It

This page established that the Reality Check tests the composite null $\max_k\mathbb{E}[f_k]\le0$ whose least favourable configuration is the origin, and that recentring each candidate on its own sample mean imposes that configuration while preserving the family's variances and covariances; that the resampling scheme decides whether the test works at all, an iid bootstrap rejecting a true null $35.50\%$ and $62.25\%$ of the time on AR(1) data at $\phi=0.3$ and $0.5$ where stationary block schemes held size at $0.0725$, $0.0700$, $0.0875$ and $0.0625$, with all three agreeing at $0.0525$, $0.0525$, $0.0550$ on independent data — a twelvefold error against the threefold one [Bootstrap Tests](../part-12-hypothesis-testing/10-bootstrap-tests.md) measures for a single mean, because a maximum compounds twenty understated variances; that resampling the family jointly delivers exact size at $0.0500$ where Bonferroni and Holm deliver $0.0300$, and converts that recovered budget into detection rates of $0.1100$, $0.2667$, $0.5233$ and $0.7633$ against $0.0567$, $0.1733$, $0.3967$ and $0.6367$; and that a candidate's contribution to the null distribution depends only on its second moments, so a champion at a true Sharpe of $0.80$ conclusive alone at a mean p-value of $0.0023$ degrades to $0.0157$, $0.0513$ and $0.0529$ as nine, forty-nine and ninety-nine worthless siblings are added, and to $0.0206$, $0.0545$ and $0.0882$ when those siblings lose a full Sharpe point a year — the same damage, from candidates that could not conceivably have won.

The change of strategy this page represents is worth stating plainly against the three before it. Pages 2 and 3 took a count and computed a threshold, and [Data Snooping Bias](04-data-snooping-bias.md) showed the count is the one thing nobody has. This page takes no count. It takes the candidates themselves and reads the null distribution of their maximum off their own joint behaviour, which disposes at a stroke of the correlated-family error that made the deflated Sharpe ratio demand double the honest hurdle, and of the independence assumption that every formula on pages 2 and 3 rests on. What it does not dispose of is the file drawer: the family it resamples is the family it was handed, and candidates that were tried and never submitted are exactly as invisible to a bootstrap as they were to a Bonferroni denominator.

What remains is the cost of the least favourable configuration. Section 4's ninety-nine catastrophes are charged for at full price because the recentring erased the evidence that they were catastrophes, and that evidence is sitting in the sample, unused. A procedure that consulted it — dropping candidates whose sample means are so far below zero that no plausible null generated them, while keeping every candidate that might genuinely have broken even — would recover the power section 4 loses without giving up section 1's validity. Constructing it, and measuring how much of section 4 it actually repairs, is [Hansen's SPA Test](06-hansens-spa-test.md).

**The Reality Check replaces an assumption about how many tests were run with a measurement of how the tests actually move together, which is the single most useful trade in this part and leaves untouched the one quantity that was missing to begin with.**
