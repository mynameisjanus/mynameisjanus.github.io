# Hansen's SPA Test

[White's Reality Check](05-whites-reality-check.md) ended on a measured defect: a champion with a true Sharpe of $0.80$, conclusive when tested alone, lost its rejection a quarter of the time once ninety-nine candidates that could not possibly win were placed beside it, and it made no difference whether those candidates were worthless or lost a full Sharpe point a year. Hansen's test repairs that by studentizing the maximum and declining to recentre candidates whose evidence puts them implausibly far below zero, and on exactly that family it works completely — the rejection rate goes from $0.7333$ back to $1.0000$ while the mean p-value falls from $0.0916$ to $0.0017$. The repair also has a boundary, and the boundary is the page's point. Where the ninety-nine siblings are merely *worthless* rather than actively losing, the recentring discards $0.0199$ of them and the two tests become the same test: rejection $0.7533$ against White's $0.7667$, a difference of $-0.0133$. The threshold that does the work is not identified from the data either, which is why the procedure returns three p-values rather than one — spanning $0.2835$ to $0.6817$ at a five-year horizon.

This page covers studentization and what it changes about which candidate wins a maximum, the recentring rule and the reason its threshold cannot be estimated, the power it recovers on a family of loss-makers, and the family of variants on which it recovers nothing. It does not construct the composite null, the least favourable configuration or the stationary bootstrap, all of which are [White's Reality Check](05-whites-reality-check.md); it does not build a resampling scheme from first principles, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); it divides no level and ranks no p-values, which are [Bonferroni Correction](02-bonferroni-correction.md) and [False Discovery Rate](03-false-discovery-rate.md); it recovers no unreported search, which is [Data Snooping Bias](04-data-snooping-bias.md); it identifies no subset of superior candidates and constructs no confidence set over models, which is section 5's boundary; and it never treats a family's composition as though the test were indifferent to it.

The trading stake is a promise the course makes in one clause and this page prices. [Validation and Overfitting](../../part-04-strategy-development/08-validation-and-overfitting.md) reaches for the Reality Check on the fifty-variant momentum grid and notes in passing that Hansen's SPA test "is its stricter descendant". Stricter is the wrong axis: sections 3 and 4 measure SPA as uniformly *more* powerful than White and never less, and what varies is by how much — everything on a family of loss-makers, nothing at all on a family of variants that merely do not work, which is what a lookback grid over one signal actually is.

## Studentizing Stops the Maximum From Being Won on Variance Alone, and the Recentring That Follows Depends on a Threshold No Sample Can Estimate

Two changes separate this test from its predecessor, and only the first is a matter of taste.

??? note "Proof that the unstudentized maximum is won by whichever candidate has the largest variance rather than the strongest evidence, and that the recentring threshold must vanish more slowly than the sampling error and faster than any fixed mean"

    White's statistic is $\bar V=\max_k\sqrt n\,\bar f_k$. Under the least favourable configuration every $\sqrt n\,\bar f_k$ is asymptotically $\mathcal{N}(0,\omega_k^{2})$ with $\omega_k^{2}$ the long-run variance of candidate $k$, so the maximum is a maximum of *unequally scaled* normals. If candidate $j$ has twice the volatility of candidate $i$ and both have a true mean of zero, $j$ contributes a distribution twice as wide and dominates the maximum roughly twice as often, without carrying any more evidence of superiority. A candidate can therefore raise the bar for the entire family by being noisy. Dividing by $\hat\omega_k$ gives Hansen's
    $$T^{\mathrm{SPA}}=\max_k\max\!\left(\frac{\sqrt n\,\bar f_k}{\hat\omega_k},\,0\right),$$
    in which every candidate enters on the same scale and the maximum is taken over evidence rather than over volatility. The outer $\max(\cdot,0)$ makes the statistic zero when no candidate beats the benchmark, which is the correct boundary since the null is one-sided.

    The second change is that recentring need not be applied to every candidate. [White's Reality Check](05-whites-reality-check.md) recentres unconditionally, so a candidate that lost money throughout enters the null distribution as though it had broken even. A candidate whose sample mean is far enough below zero is implausible under *any* null in the region, and could be excluded — but "far enough" has to be defined, and this is where the construction becomes delicate. Let the recentred mean be $\hat\mu_k=\bar f_k\,\mathbb{1}\{\sqrt n\,\bar f_k/\hat\omega_k\ge -A_n\}$ for some sequence $A_n$. Two requirements pull in opposite directions:

    - *Validity* requires that a candidate with true mean exactly zero is retained with probability tending to one, since such a candidate is in the null and must be allowed to contribute to the maximum. Its studentized mean is $O_p(1)$, so this needs $A_n\to\infty$.
    - *Power* requires that a candidate with a fixed true mean $\mu_k<0$ is eventually discarded. Its studentized mean is $\sqrt n\mu_k/\omega_k\to-\infty$ at rate $\sqrt n$, so this needs $A_n=o(\sqrt n)$.

    Any $A_n$ with $A_n\to\infty$ and $A_n/\sqrt n\to0$ satisfies both, and Hansen takes $A_n=\sqrt{2\log\log n}$ by analogy with the law of the iterated logarithm, which is the slowest such rate that still separates. In sample-mean units the rule discards candidate $k$ when
    $$\bar f_k<-\hat\omega_k\sqrt{\frac{2\log\log n}{n}}.$$

    The load-bearing fact is that *every* admissible $A_n$ gives an asymptotically valid and consistent test, and they disagree at every finite $n$. **The threshold is pinned by two limits and by nothing in the data, so a sample cannot say which member of an infinite admissible family is right, and the honest output of the procedure is an interval of p-values rather than a p-value.** Hansen therefore reports three: a *lower* one that discards every candidate with a negative sample mean, an *upper* one that discards none and is exactly the studentized Reality Check, and a *consistent* one using the $\sqrt{2\log\log n}$ rule between them.

## The Three P-Values Bracket the Answer, and the Consistent One Migrates Across the Bracket as the Horizon Grows

The bracket is not a technicality to be reported and ignored: its width and the position of the consistent estimate inside it are both functions of the sample length, which means the same family answered at two horizons gets qualitatively different verdicts:

```python
import numpy as np

rng = np.random.default_rng(15061)
B, reps, mb, k, champ, sib = 299, 100, 10, 50, 0.80, -1.00


def spa_and_rc(f, n, row, ar):
    """Hansen's three p-values plus White's, from one set of bootstrap resamples."""
    new = rng.random((B, n)) < 1 / mb
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    idx = (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    bm = (C @ f) / n
    fb = f.mean(0)
    om = np.sqrt(n * bm.var(0))                     # Hansen: variance across the resamples
    t = max((np.sqrt(n) * fb / om).max(), 0.0)
    thr = -om * np.sqrt(2 * np.log(np.log(n))) / np.sqrt(n)
    out = []
    for g in (np.maximum(fb, 0.0), np.where(fb >= thr, fb, 0.0), fb):
        z = np.sqrt(n) * (bm - g) / om
        out.append((np.maximum(z, 0).max(1) > t).mean())
    out.append((np.sqrt(n) * (bm - fb).max(1) >= np.sqrt(n) * fb.max()).mean())
    return out + [float((fb < thr).mean())]


print(f"  {k} candidates: one champion at a TRUE Sharpe of {champ}, {k - 1} siblings"
      f" at {sib}")
print(f"  the recentring threshold is -omega*sqrt(2 log log n / n), so it moves with"
      f" the horizon; B = {B}, {reps} replications")
print("     years        n   threshold in SR   frac dropped    p_lower   p_consistent"
      "   p_upper   White")
for yr in (5, 10, 25, 50):
    n = yr * 252
    row, ar = np.repeat(np.arange(B), n) * n, np.arange(n)
    acc = np.zeros(5)
    for _ in range(reps):
        f = rng.standard_normal((n, k))
        f[:, 0] += champ / np.sqrt(252)
        f[:, 1:] += sib / np.sqrt(252)
        acc += spa_and_rc(f, n, row, ar)
    acc /= reps
    thr_sr = -np.sqrt(252) * np.sqrt(2 * np.log(np.log(n)) / n)
    print(f"    {yr:6d}   {n:6d}   {thr_sr:15.4f}   {acc[4]:12.4f}   {acc[0]:8.4f}"
          f"   {acc[1]:12.4f}   {acc[2]:7.4f}   {acc[3]:5.4f}")
# =>   50 candidates: one champion at a TRUE Sharpe of 0.8, 49 siblings at -1.0
#      the recentring threshold is -omega*sqrt(2 log log n / n), so it moves with the horizon; B = 299, 100 replications
#         years        n   threshold in SR   frac dropped    p_lower   p_consistent   p_upper   White
#             5     1260           -0.8867         0.5966     0.2835         0.5511    0.6817   0.6967
#            10     2520           -0.6416         0.8640     0.0427         0.1462    0.3740   0.3759
#            25     6300           -0.4165         0.9778     0.0008         0.0008    0.0426   0.0466
#            50    12600           -0.2997         0.9800     0.0000         0.0000    0.0026   0.0027
```

The `p_upper` and `White` columns are the first thing to check and they confirm the construction: $0.6817$ against $0.6967$, $0.3740$ against $0.3759$, $0.0426$ against $0.0466$, $0.0026$ against $0.0027$. Recentring every candidate — discarding none — reproduces the Reality Check to within the bootstrap's own noise, which is what the proof says it must, since the only remaining difference is the studentization. **The upper end of Hansen's bracket is White's test, so the SPA test cannot be less powerful than the Reality Check and the entire question is how far below that ceiling the other two p-values sit.**

The `threshold in SR` column is the sequence $A_n$ expressed in units a practitioner can read. It moves $-0.8867$, $-0.6416$, $-0.4165$, $-0.2997$ as the horizon runs five to fifty years — shrinking toward zero, but at the $\sqrt{\log\log n/n}$ rate, which is slow. Even at fifty years of daily data the rule only discards candidates whose measured Sharpe is below $-0.30$.

The consequence is the `p_consistent` column, and it does not sit anywhere stable. At five years it reads $0.5511$ against a bracket of $[0.2835,0.6817]$ — nearer the upper end, behaving almost like White. At ten years it is $0.1462$ in a bracket of $[0.0427,0.3740]$, now roughly central. At twenty-five and fifty years it reads $0.0008$ and $0.0000$, exactly equal to $p_{\text{lower}}$, because the fraction of candidates discarded has reached $0.9778$ and $0.9800$ and there is nothing left for the threshold to be ambiguous about. The consistent p-value migrates from one end of the bracket to the other as data accumulates, and at every finite horizon its position is a consequence of a constant chosen for its asymptotics.

The bracket's width is the practical warning. At five years a researcher reports either $0.28$ or $0.68$ depending on a choice with no empirical content — a factor of two and a half, straddling every threshold anyone uses. At twenty-five years the bracket is $[0.0008,0.0426]$, which is a factor of fifty but lands on the same side of $0.05$ at both ends, so the verdict is robust even though the number is not. **A single SPA p-value is a point estimate of a quantity the data does not determine, and reporting it without its bracket discards the one piece of information the procedure supplies about how much the choice mattered.**

## On a Family of Loss-Makers the Recentring Recovers Everything the Reality Check Lost

Section 4 of the preceding page is the test case this construction exists for, and it can be rerun with the two procedures side by side on the identical family:

```python
import numpy as np

rng = np.random.default_rng(15063)
n, B, reps, mb, champ, sib = 6300, 299, 150, 10, 0.80, -1.00
row = np.repeat(np.arange(B), n) * n
ar = np.arange(n)


def family(k, s):
    """The identical construction White's Reality Check section 4 uses."""
    f = rng.standard_normal((n, k))
    f[:, 0] += champ / np.sqrt(252)
    f[:, 1:] += s / np.sqrt(252)
    return f


def both(f):
    """White's p-value and Hansen's consistent p-value from one set of resamples."""
    new = rng.random((B, n)) < 1 / mb
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    idx = (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    bm = (C @ f) / n
    fb = f.mean(0)
    om = np.sqrt(n * bm.var(0))
    t = max((np.sqrt(n) * fb / om).max(), 0.0)
    thr = -om * np.sqrt(2 * np.log(np.log(n))) / np.sqrt(n)
    g = np.where(fb >= thr, fb, 0.0)
    spa = (np.maximum(np.sqrt(n) * (bm - g) / om, 0).max(1) > t).mean()
    rc = ((bm - fb).max(1) >= fb.max()).mean()
    return rc, spa, float((fb < thr).mean())


print(f"  the same family as White's Reality Check section 4: a champion at a TRUE")
print(f"  Sharpe of {champ} among siblings at {sib}, n = {n:,}, B = {B},"
      f" {reps} replications")
print("    siblings   frac dropped   mean White p   White rejects   mean SPA p"
      "   SPA rejects")
for k in (1, 10, 50, 100):
    acc = np.zeros(5)
    for _ in range(reps):
        rc, spa, d = both(family(k, sib))
        acc += [d, rc, rc < 0.05, spa, spa < 0.05]
    acc /= reps
    lab = "none" if k == 1 else f"{k - 1}"
    print(f"    {lab:>8s}   {acc[0]:12.4f}   {acc[1]:12.4f}   {acc[2]:13.4f}"
          f"   {acc[3]:10.4f}   {acc[4]:11.4f}")
# =>   the same family as White's Reality Check section 4: a champion at a TRUE
#      Sharpe of 0.8 among siblings at -1.0, n = 6,300, B = 299, 150 replications
#        siblings   frac dropped   mean White p   White rejects   mean SPA p   SPA rejects
#            none         0.0000         0.0015          1.0000       0.0015        1.0000
#               9         0.8973         0.0110          0.9400       0.0012        0.9933
#              49         0.9785         0.0368          0.8467       0.0016        0.9933
#              99         0.9879         0.0916          0.7333       0.0017        1.0000
```

The White columns reproduce the preceding page on a fresh draw of the same construction — mean p-value climbing $0.0015$, $0.0110$, $0.0368$, $0.0916$ and rejection falling $1.0000$, $0.9400$, $0.8467$, $0.7333$ as the hopeless siblings accumulate.

The SPA columns do not move at all. Mean p-value $0.0015$, $0.0012$, $0.0016$, $0.0017$; rejection $1.0000$, $0.9933$, $0.9933$, $1.0000$. Adding ninety-nine candidates that lose a Sharpe point a year for twenty-five years costs the test nothing whatsoever, where it cost the Reality Check a quarter of its rejections.

The `frac dropped` column is the entire mechanism, and it is worth reading as a diagnostic rather than as a parameter. It runs $0.0000$, $0.8973$, $0.9785$, $0.9879$: the recentring is discarding essentially every sibling, because a true Sharpe of $-1.0$ measured over twenty-five years lands around $-1.0\pm0.2$ and the threshold sits at $-0.42$. Having discarded them, the null distribution of the maximum is built from the champion and almost nothing else, which is the distribution it would have had if the researcher had never tested the siblings — and that is precisely the object the Reality Check was unable to construct. **The gain is not that SPA is a sharper test but that it declines to charge for candidates the sample has already ruled out, so the correction is levied on the number of hypotheses that were still live rather than the number that were typed.**

## The Recentring Only Discards Candidates That Lose Money, and a Parameter Grid Is Made of Candidates That Merely Do Not Work

Section 3's family was chosen to make the repair visible. The question that decides whether the repair matters in practice is what happens when the siblings are not catastrophes but simply nothing — which is what a grid of lookback lengths over one signal actually contains.

??? note "Proof that a candidate with a true mean of exactly zero is retained by the recentring with probability approaching one, so the discarded set is asymptotically the strictly-negative candidates and no others"

    Take candidate $k$ with true mean $\mu_k$ and long-run variance $\omega_k^{2}$, and write the retention event as $\sqrt n\,\bar f_k/\hat\omega_k\ge-A_n$ with $A_n=\sqrt{2\log\log n}$.

    If $\mu_k=0$ then $\sqrt n\,\bar f_k/\hat\omega_k\Rightarrow\mathcal{N}(0,1)$, so the retention probability is $\Phi(A_n)\to1$: a genuinely worthless candidate is kept, and must be, since it belongs to the null and excluding it would understate the maximum's null distribution and break the test's size. If $\mu_k<0$ then $\sqrt n\,\bar f_k/\hat\omega_k=\sqrt n\mu_k/\omega_k+O_p(1)\to-\infty$ at rate $\sqrt n$, which outruns $A_n$, so the retention probability tends to zero.

    The dividing line is therefore $\mu_k=0$ exactly, and the finite-sample version of it is the Sharpe threshold $-\hat\omega_k\sqrt{2\log\log n/n}$, which section 2 measured at $-0.42$ for twenty-five years of daily data. A candidate is discarded only if its *measured* performance is worse than that, so the fraction discarded is $\Phi\big((-0.42-\mu_k)/\sigma_{SR}\big)$ with $\sigma_{SR}\approx0.20$ at that horizon: about $0.02$ for a candidate whose true Sharpe is zero, and about $1.00$ for one whose true Sharpe is $-1$.

    The load-bearing consequence is that the repair is targeted at a set that has to be non-empty for it to do anything. **Hansen's recentring separates candidates that lose money from candidates that break even, and it is exactly the second group that a variant grid is made of, so the correction with the strongest theoretical motivation in this part is aimed at a population that strategy research does not generate.**

The prediction is that the gain over White should be a function of how badly the siblings perform, vanishing as their true mean approaches zero from below:

```python
import numpy as np

rng = np.random.default_rng(15065)
n, k, B, reps, mb, champ = 6300, 100, 299, 150, 10, 0.80
row = np.repeat(np.arange(B), n) * n
ar = np.arange(n)


def both(f):
    """White's p-value and Hansen's consistent p-value from one set of resamples."""
    new = rng.random((B, n)) < 1 / mb
    new[:, 0] = True
    st = rng.integers(0, n, size=(B, n))
    pos = np.maximum.accumulate(np.where(new, ar, 0), axis=1)
    idx = (np.take_along_axis(st, pos, axis=1) + (ar - pos)) % n
    C = np.bincount(row + idx.ravel(), minlength=B * n).reshape(B, n).astype(float)
    bm = (C @ f) / n
    fb = f.mean(0)
    om = np.sqrt(n * bm.var(0))
    t = max((np.sqrt(n) * fb / om).max(), 0.0)
    thr = -om * np.sqrt(2 * np.log(np.log(n))) / np.sqrt(n)
    g = np.where(fb >= thr, fb, 0.0)
    spa = (np.maximum(np.sqrt(n) * (bm - g) / om, 0).max(1) > t).mean()
    rc = ((bm - fb).max(1) >= fb.max()).mean()
    return rc, spa, float((fb < thr).mean())


thr_sr = -np.sqrt(252) * np.sqrt(2 * np.log(np.log(n)) / n)
print(f"  a champion at a TRUE Sharpe of {champ} among {k - 1} siblings, n = {n:,}"
      f" ({n // 252} years)")
print(f"  the recentring only discards candidates below a Sharpe of about"
      f" {thr_sr:.2f}, so how BAD the siblings are is what decides")
print("    sibling true SR   frac dropped   mean White p   White rejects   mean SPA p"
      "   SPA rejects   gain")
for sib in (0.00, -0.10, -0.25, -0.50, -1.00):
    acc = np.zeros(5)
    for _ in range(reps):
        f = rng.standard_normal((n, k))
        f[:, 0] += champ / np.sqrt(252)
        f[:, 1:] += sib / np.sqrt(252)
        rc, spa, d = both(f)
        acc += [d, rc, rc < 0.05, spa, spa < 0.05]
    acc /= reps
    print(f"    {sib:15.2f}   {acc[0]:12.4f}   {acc[1]:12.4f}   {acc[2]:13.4f}"
          f"   {acc[3]:10.4f}   {acc[4]:11.4f}   {acc[4] - acc[2]:+5.4f}")
# =>   a champion at a TRUE Sharpe of 0.8 among 99 siblings, n = 6,300 (25 years)
#      the recentring only discards candidates below a Sharpe of about -0.42, so how BAD the siblings are is what decides
#        sibling true SR   frac dropped   mean White p   White rejects   mean SPA p   SPA rejects   gain
#                   0.00         0.0199         0.0590          0.7667       0.0519        0.7533   -0.0133
#                  -0.10         0.0586         0.0647          0.7533       0.0618        0.7867   +0.0333
#                  -0.25         0.2012         0.0921          0.7533       0.0782        0.7867   +0.0333
#                  -0.50         0.6566         0.0928          0.7467       0.0493        0.8400   +0.0933
#                  -1.00         0.9885         0.0584          0.8533       0.0015        0.9867   +0.1333
```

The `frac dropped` column tracks the proof exactly. Siblings with a true Sharpe of $0.00$ are discarded $0.0199$ of the time — the $2\%$ tail below $-0.42$ that a $0.20$ standard error produces — and the fraction climbs $0.0586$, $0.2012$, $0.6566$, $0.9885$ as the true Sharpe falls to $-1.00$. The recentring is a filter on measured performance and it fires only when there is something to filter.

The `gain` column is the consequence and it is the page's finding. Against siblings with a true Sharpe of zero the gain is $-0.0133$: SPA rejects $0.7533$ of the time and White rejects $0.7667$, a difference indistinguishable from zero at a hundred and fifty replications and certainly not an improvement. The gain then rises monotonically with the siblings' badness — $+0.0333$, $+0.0333$, $+0.0933$, $+0.1333$ — reaching an eighth of all trials only when every sibling is losing a full Sharpe point a year.

The top row is the case that matters. Fifty lookback lengths applied to one momentum signal produce variants that are correlated with each other, close to zero in expectation, and almost never *negative* in expectation — a lookback that captures no signal earns approximately nothing rather than reliably losing. On such a family the recentring discards two percent of the candidates, the null distribution of the maximum is essentially unchanged, and Hansen's consistent p-value is White's p-value with the volatilities equalised. **SPA's advertised advantage over the Reality Check is purchased entirely from candidates the sample can prove are bad, and a parameter grid over a single idea contains almost none of those, so the test most often recommended for grids is the one whose repair does not engage on them.**

Two things follow that are worth separating. Using SPA in place of White is still correct and free — section 2 established that its upper p-value *is* White's, so the procedure cannot lose, and the studentization is a genuine improvement whenever the candidates differ in volatility. But expecting it to solve the problem [White's Reality Check](05-whites-reality-check.md) section 4 identified is expecting it to charge less for candidates that the data cannot distinguish from break-even, and no valid test can do that: those candidates are in the null, and a procedure that discarded them would be discarding exactly the configuration the size guarantee is defined at.

## Naming Which Candidates Are Superior Is a Different Procedure, and All of Them Take the Reported Family on Trust

Both tests in these two pages answer one question — whether *any* member of the family beats the benchmark — and return one p-value for the whole family. That is often not the question. A desk that has established some strategy in a grid is real still has to say which, and neither procedure offers an answer: the champion is simply the sample maximum, with all the selection problems of [Data Snooping Bias](04-data-snooping-bias.md) attached.

Two constructions extend the machinery to that question and both reuse everything already built. Romano and Wolf's **StepM** is [Bonferroni Correction](02-bonferroni-correction.md)'s step-down logic with a bootstrapped critical value in place of $\alpha/m$: compute the bootstrap distribution of the studentized maximum over the full family, reject every candidate exceeding its $1-\alpha$ quantile, then *remove* those candidates and recompute the maximum's distribution over what remains, repeating until nothing more is rejected. It controls the family-wise error rate, it names the rejected set rather than merely asserting that one exists, and because the critical value is bootstrapped it uses the family's dependence exactly as the Reality Check does. That fills the cell the rest of this part leaves empty: Bonferroni and Holm name *which* and ignore dependence, White and Hansen use dependence and name only *whether any*, StepM does both.

Hansen, Lunde and Nason's **Model Confidence Set** inverts the same idea into a set rather than a rejection: it repeatedly tests the hypothesis that all surviving candidates are equally good, eliminates the worst when that is rejected, and returns the collection that survives — a confidence set over models with the property that it contains the best one with a stated probability. It is the natural output for a research process that wants a shortlist rather than a winner, and it degrades gracefully, returning a large set when the data cannot discriminate rather than a spurious champion.

What none of the four does is recover the family. StepM steps down through the candidates it was given, the Model Confidence Set eliminates from the set it was handed, and both bootstrap the joint distribution of exactly the columns submitted to them. A variant deleted in week one because its equity curve looked wrong is as invisible to a step-down bootstrap as it was to a Bonferroni denominator, and [Data Snooping Bias](04-data-snooping-bias.md) measured what that invisibility costs: a family-wise rate of $0.6326$ against a nominal $0.05$ when a write-up admits to fifty variants out of a thousand. The resampling pages of this part removed the need to *count* the candidates and left untouched the need to *report* them.

!!! note "White's Reality Check, Hansen's SPA test, Romano–Wolf StepM and the Model Confidence Set are four bootstrap procedures on one family, and they differ in what they studentize, what they recentre and what they name"
    **The Reality Check** takes the maximum of unstudentized recentred means, recentres every candidate unconditionally, and returns one p-value for the hypothesis that no candidate beats the benchmark; section 4 of the preceding page measures its power falling to $0.7333$ on hopeless siblings. **The SPA test** studentizes the maximum so a volatile candidate cannot win it on scale alone, recentres conditionally on a $\sqrt{2\log\log n}$ threshold, and returns three p-values because that threshold is not identified — its upper one being the studentized Reality Check, so it can never be less powerful. **StepM** studentizes and bootstraps as SPA does but iterates: reject, remove, recompute, repeat, which converts the family-level answer into a named set of superior candidates with family-wise error control. **The Model Confidence Set** runs the elimination the other way, repeatedly testing equal predictive ability and discarding the worst survivor, and returns a *set* containing the best candidate with a stated probability rather than a verdict about any one of them. The distinction that matters operationally is between the first two, which answer "did the search find anything", and the last two, which answer "what did it find" — and every one of the four bootstraps only the columns it was handed.

!!! warning "SPA's repair is aimed at candidates the sample can prove are bad, and a family of variants that merely do not work supplies none of them"
    Section 4 measured Hansen's consistent p-value beating White's by $0.1333$ in rejection rate when the ninety-nine siblings had a true Sharpe of $-1.00$ and by $-0.0133$ — nothing, and slightly the wrong way — when they had a true Sharpe of $0.00$, with the fraction of candidates discarded by the recentring running $0.9885$ and $0.0199$ respectively. The test is not misbehaving in the second case; it is correctly declining to discard candidates that belong to the null. But the two cases are indistinguishable in the output. Both print a p-value, neither prints the fraction of the family the recentring removed, and a practitioner who adopts SPA on the strength of section 3's result and applies it to a lookback grid gets section 4's top row without being told which of the two situations they are in. The other invisible quantity is section 2's bracket: a single consistent p-value of $0.5511$ conceals that the admissible range at that horizon was $[0.2835,0.6817]$, and one of $0.0008$ conceals that the range was $[0.0008,0.0426]$ — the first ambiguous at every threshold, the second robust at all of them, and identical in appearance. **The free diagnostic is to report the fraction of candidates the recentring discarded alongside all three p-values: if that fraction is near zero, SPA has returned the studentized Reality Check and the family's size is being charged for in full, so the correct reading of the result is [White's Reality Check](05-whites-reality-check.md) section 4 rather than this page's section 3 — and if the three p-values straddle a decision threshold, the honest report is that the data does not determine the answer.**

## A Repair That Works Exactly Where the Sample Has Already Done the Work

This page established that studentizing the maximum stops a candidate from dominating it through volatility rather than evidence, and that the conditional recentring which follows requires a threshold $A_n\to\infty$ with $A_n/\sqrt n\to0$ — two limits that admit infinitely many sequences and no data, so the procedure's honest output is a bracket rather than a number; that the bracket's upper end reproduces the Reality Check, measured at $0.6817$, $0.3740$, $0.0426$ and $0.0026$ against White's $0.6967$, $0.3759$, $0.0466$ and $0.0027$, so SPA can never be the weaker test; that the consistent p-value migrates across that bracket as the horizon grows, sitting at $0.5511$ in $[0.2835,0.6817]$ at five years and at $0.0008$ in $[0.0008,0.0426]$ at twenty-five, while the discard threshold shrinks only from a Sharpe of $-0.89$ to $-0.42$; that on the family of loss-makers the preceding page used, the repair is total — White's mean p-value climbing $0.0015$, $0.0110$, $0.0368$, $0.0916$ and its rejection rate falling to $0.7333$ while SPA holds at $0.0015$, $0.0012$, $0.0016$, $0.0017$ and $1.0000$, discarding $0.9879$ of the siblings; and that the repair vanishes as the siblings approach break-even, the gain in rejection rate running $+0.1333$, $+0.0933$, $+0.0333$, $+0.0333$ and $-0.0133$ as their true Sharpe rises from $-1.00$ to $0.00$ and the discarded fraction falls from $0.9885$ to $0.0199$.

The part ends on a symmetry worth naming. Its first three pages corrected a statistic using a number the researcher supplied, and failed because the number was unavailable. Its last two corrected a statistic using the family's own joint behaviour, and needed nothing from the researcher except the family — which is a real advance, and which relocates the entire remaining problem into a single question of what was submitted. Every result on these two pages is a statement about the columns handed to the procedure. Both are honest about the candidates they can see and blind in exactly the same way to the ones they cannot, and no resampling scheme can be otherwise, because a bootstrap draws from what exists.

What has been assumed throughout, and never examined, is that a hypothesis is a thing one rejects or fails to reject. Every page in this part takes a null, computes a tail probability under it, and compares that probability to a threshold; the corrections differ only in which tail and which threshold. A treatment that instead assigns a probability to the hypothesis itself — that starts from what was believed before the search and reports what should be believed after it, with the number of candidates entering as a prior over how many could plausibly be real rather than as a divisor — answers the same question with none of this machinery. That is [Part XVI](../part-16-bayesian-statistics/index.md).

**Hansen's test is the best-motivated correction in this part and its motivation is a family that trading research does not produce: it declines to charge for strategies the data has already convicted, and a grid of variants over one idea contains almost nothing but strategies the data cannot convict of anything at all.**
