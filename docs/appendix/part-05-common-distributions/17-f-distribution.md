# F Distribution

The F is the law of a ratio of two variance estimates, which makes it the distribution behind every question of the form *is this volatility different from that one*. Its most useful lesson is negative. A variance ratio is estimated so imprecisely that at ordinary sample sizes the test cannot distinguish differences a trader would consider enormous, and the reason is visible in the density rather than hidden in the data.

This page covers the ratio construction, the reciprocal symmetry that halves the table of critical values, the moments and the conditions they need, the identity connecting the square of a $t$ to an F, and the power a comparison of two volatilities actually has. It does not build the chi-square in the numerator and denominator, which is [Chi-Square Distribution](15-chi-square-distribution.md); it does not cover the single-sample interval, which is also there; and it does not develop regression or analysis of variance, which is [Part XIII](../part-13-regression/index.md).

The trading stake is that a desk comparing two sleeves, two venues, or two periods routinely asks whether one is more volatile than the other, and the honest answer at a year of daily data is almost always *cannot tell*. The last section computes the threshold: on $252$ observations each, two volatilities must differ by more than $13\%$ before a two-sided test at the usual level will call them apart, and a genuine $10\%$ difference — enough to move a position size by a tenth — is caught only a third of the time.

## The Ratio of Two Chi-Squares

Let $Q_1\sim\chi^{2}_{d_1}$ and $Q_2\sim\chi^{2}_{d_2}$ be independent. Then

$$F=\frac{Q_1/d_1}{Q_2/d_2}\ \sim\ F_{d_1,d_2}.$$

Each chi-square is divided by its degrees of freedom, so each has mean one and the ratio is centred near one — which is what makes it a natural test statistic for the hypothesis that two variances agree. Substituting the sampling result of [Chi-Square Distribution](15-chi-square-distribution.md), two independent samples of sizes $n_1$ and $n_2$ from normal populations give

$$\frac{s_1^{2}/\sigma_1^{2}}{s_2^{2}/\sigma_2^{2}}\sim F_{n_1-1,\,n_2-1},$$

so under the null $\sigma_1=\sigma_2$ the observable ratio $s_1^{2}/s_2^{2}$ is itself $F$, with the unknown scale cancelling. That cancellation is what makes the test possible at all, and it traces back to the independence of a gamma ratio from its total proved on [Beta Distribution](14-beta-distribution.md).

??? note "Proof that the F is a rescaled beta, and that this is where the density comes from"
    Write $B=Q_1/(Q_1+Q_2)$. By the argument on [Beta Distribution](14-beta-distribution.md) — with $Q_1$ and $Q_2$ read as gammas of shapes $d_1/2$ and $d_2/2$ sharing scale $2$ — the ratio $B$ is $\mathrm{Beta}(d_1/2,d_2/2)$ and is independent of the total. Now

    $$F=\frac{d_2}{d_1}\cdot\frac{Q_1}{Q_2}=\frac{d_2}{d_1}\cdot\frac{B}{1-B},$$

    so the F is a fixed multiple of the odds transform of a beta. Changing variables with $b=d_1f/(d_1f+d_2)$ and Jacobian $d_1d_2/(d_1f+d_2)^{2}$ gives

    $$f_F(x)=\frac{1}{B(d_1/2,d_2/2)}\Big(\frac{d_1}{d_2}\Big)^{d_1/2}\frac{x^{d_1/2-1}}{\big(1+\tfrac{d_1}{d_2}x\big)^{(d_1+d_2)/2}},\qquad x>0.$$

    Two things are worth noting. The scale of the underlying data has vanished — it cancelled in the ratio before the beta was formed — which is why the null distribution is free of $\sigma$ and the test needs no nuisance-parameter estimate. And the tail decays as $x^{-(d_2/2+1)}$, a power law governed by the *denominator's* degrees of freedom alone, which is the fact the moment conditions below and the whole power analysis at the end depend on.

## Reciprocal Symmetry and the Two Degrees of Freedom

Inverting the ratio swaps the roles of the two samples, so

$$\frac{1}{F_{d_1,d_2}}\sim F_{d_2,d_1},\qquad\text{and hence}\qquad F_{d_1,d_2,\,\alpha}=\frac{1}{F_{d_2,d_1,\,1-\alpha}}.$$

This is not a curiosity; it is why printed F tables only ever give the upper tail. The lower critical value of one orientation is the reciprocal of the upper critical value of the other, so half the table is redundant. It also means the two degrees of freedom are not interchangeable — $F_{5,100}$ and $F_{100,5}$ are quite different laws, one concentrated and one wildly dispersed — and getting them the wrong way round produces a test that looks plausible and is badly wrong.

## Moments and Their Conditions

$$\mathbb{E}[F]=\frac{d_2}{d_2-2}\ (d_2>2),\qquad \mathrm{var}(F)=\frac{2d_2^{2}(d_1+d_2-2)}{d_1(d_2-2)^{2}(d_2-4)}\ (d_2>4).$$

Both conditions involve only $d_2$, for the reason the proof gave: the tail is set by the denominator. A small denominator sample makes the ratio heavy-tailed regardless of how much data went into the numerator, which is the F's version of the lesson [Student's t Distribution](16-students-t-distribution.md) drew — dividing by a noisy estimate is what manufactures a heavy tail.

Note also that the mean is $d_2/(d_2-2)$ rather than $1$. Under the null of equal variances the *expected* ratio exceeds one, by $2/(d_2-2)$, even though the two variances are identical. That bias is small at large $d_2$ and is a genuine trap at small $d_2$: comparing two twenty-observation samples, the expected variance ratio under the null is $1.12$.

```python
import numpy as np
from scipy.stats import f as fdist

rng = np.random.default_rng(127)
print("  F as a ratio of two scaled chi-squares, and its moment conditions")
print("      d1    d2      sample mean    exact    sample var     exact       tail")
for d1, d2 in ((251, 251), (251, 20), (20, 251), (10, 3), (10, 5)):
    q1 = rng.chisquare(d1, 400_000) / d1
    q2 = rng.chisquare(d2, 400_000) / d2
    x = q1 / q2
    em = f"{d2 / (d2 - 2):8.4f}" if d2 > 2 else "     inf"
    ev = (f"{2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4)):9.4f}"
          if d2 > 4 else "      inf")
    print(f"    {d1:4d} {d2:5d} {x.mean():14.4f} {em} {x.var():13.4f} {ev}"
          f"   x^-{d2 / 2 + 1:.1f}")
print("  reciprocal symmetry: the lower tail of one is the upper tail of the other")
for d1, d2 in ((251, 20), (10, 5)):
    print(f"    F({d1},{d2}) 2.5th pct {fdist.ppf(0.025, d1, d2):9.5f}"
          f"   1 / F({d2},{d1}) 97.5th pct {1 / fdist.ppf(0.975, d2, d1):9.5f}")
# =>   F as a ratio of two scaled chi-squares, and its moment conditions
#          d1    d2      sample mean    exact    sample var     exact       tail
#         251   251         1.0080   1.0080        0.0164    0.0164   x^-126.5
#         251    20         1.1111   1.1111        0.1655    0.1654   x^-11.0
#          20   251         1.0078   1.0080        0.1105    0.1107   x^-126.5
#          10     3         2.9534   3.0000      740.5129       inf   x^-2.5
#          10     5         1.6612   1.6667        6.2793    7.2222   x^-3.5
#      reciprocal symmetry: the lower tail of one is the upper tail of the other
#        F(251,20) 2.5th pct   0.56696   1 / F(20,251) 97.5th pct   0.56696
#        F(10,5) 2.5th pct   0.23607   1 / F(5,10) 97.5th pct   0.23607
```

The moment columns match wherever the conditions hold, and the $F_{10,3}$ row shows exactly where they stop. Its mean is fine — $2.95$ against an exact $3.00$, since $d_2=3$ clears the $d_2>2$ bar — while its sample variance reads $740$ against a population value that is infinite. That is the failure mode to recognise: not a number that looks wrong, but a number that looks merely large and will look differently large on the next sample.

The middle two rows isolate which degree of freedom matters. $F_{251,20}$ and $F_{20,251}$ use the same total information and have tail exponents of $11.0$ and $126.5$ respectively — the denominator alone sets the tail, so twenty observations in the denominator make the ratio heavy-tailed no matter how well-estimated the numerator is. The last two lines confirm the reciprocal identity to five decimals, which is why one tail of a table suffices.

## The Square of a t

$$T\sim t_{\nu}\ \Longrightarrow\ T^{2}\sim F_{1,\nu}.$$

??? note "Proof that the square of a t is an F with one numerator degree of freedom"
    Write $T=Z/\sqrt{Q/\nu}$ with $Z$ standard normal and $Q\sim\chi^{2}_{\nu}$ independent, which is the definition on [Student's t Distribution](16-students-t-distribution.md). Squaring,

    $$T^{2}=\frac{Z^{2}}{Q/\nu}=\frac{Z^{2}/1}{Q/\nu}.$$

    By [Chi-Square Distribution](15-chi-square-distribution.md), $Z^{2}\sim\chi^{2}_{1}$, and it is independent of $Q$. So $T^{2}$ is a ratio of two independent chi-squares each divided by its degrees of freedom, with $d_1=1$ and $d_2=\nu$ — the definition of $F_{1,\nu}$.

    The identity is the reason a two-sided $t$-test and an F-test of the same hypothesis give identical $p$-values, and it explains why: squaring discards the sign, which is exactly what a two-sided test does when it takes an absolute value. The F is therefore the natural statistic when direction is not part of the question, and the $t$ when it is — they are the same evidence, differently packaged.

    It also transfers the moment conditions. $T$ has moments below $\nu$ and $T^{2}$ has moments below $\nu/2$, which is consistent with the F's condition $d_2>4$ for a variance: $\mathrm{var}(T^{2})$ requires $\mathbb{E}[T^{4}]$, and that requires $\nu>4$.

## Comparing Two Volatilities Barely Works

Now the trading stake, which is a power calculation rather than a derivation.

```python
import numpy as np
from scipy.stats import f as fdist

print("  smallest volatility ratio a two-sided 5% F-test can call different")
print("      n per sample      variance ratio      volatility ratio")
for n in (60, 252, 756, 2520, 6410):
    crit = fdist.ppf(0.975, n - 1, n - 1)
    print(f"    {n:14d} {crit:19.4f} {np.sqrt(crit):22.4f}")
print("  power to detect a true volatility ratio, at 5% two-sided")
print("      true vol ratio     n=252      n=756     n=2520     n=6410")
for vr in (1.05, 1.10, 1.20, 1.50):
    row = []
    for n in (252, 756, 2520, 6410):
        hi = fdist.ppf(0.975, n - 1, n - 1)
        lo = fdist.ppf(0.025, n - 1, n - 1)
        # under the alternative, s1^2/s2^2 divided by vr^2 is central F
        row.append(fdist.sf(hi / vr ** 2, n - 1, n - 1)
                   + fdist.cdf(lo / vr ** 2, n - 1, n - 1))
    print(f"    {vr:14.2f}" + "".join(f"{p:11.3f}" for p in row))
# =>   smallest volatility ratio a two-sided 5% F-test can call different
#          n per sample      variance ratio      volatility ratio
#                    60              1.6741                 1.2939
#                   252              1.2814                 1.1320
#                   756              1.1535                 1.0740
#                  2520              1.0813                 1.0398
#                  6410              1.0502                 1.0248
#      power to detect a true volatility ratio, at 5% two-sided
#          true vol ratio     n=252      n=756     n=2520     n=6410
#                  1.05      0.120      0.268      0.687      0.974
#                  1.10      0.325      0.744      0.998      1.000
#                  1.20      0.822      0.999      1.000      1.000
#                  1.50      1.000      1.000      1.000      1.000
```

The first table is the detection threshold. On a year of daily data two volatilities must differ by more than $13\%$ — say $18\%$ against $20.4\%$ — before the test will call them apart. Ten years narrows that to $4\%$ and twenty-five years to $2.5\%$, so the blind spot shrinks at the $n^{-1/2}$ rate and never closes.

The second table is the one that matters, because it prices what is missed rather than what is caught. A $20\%$ volatility gap is detected $82\%$ of the time on a year of data, which is about the conventional standard, so that case is fine. A $10\%$ gap — still large enough to move a position size, a value-at-risk and a Sharpe ratio by a tenth — is caught only a third of the time on a year and three quarters of the time on three years. And a $5\%$ gap needs a full twenty-five years before the test finds it reliably.

!!! warning "Two volatilities that a test cannot distinguish may still differ by enough to matter, and the test's silence is not evidence that they agree"
    This is the standard confusion between a failure to reject and a demonstration of equality, and the F distribution makes the gap unusually wide. A $10\%$ volatility difference goes unnoticed two times in three on a year of data, and it is not a small difference. So the correct report from an insignificant F-test is the interval on the ratio rather than the $p$-value: at $n=252$ the two-sided $95\%$ interval on the volatility ratio runs from $0.88$ to $1.13$, and stating that is informative where "not significant" is not.

    The asymmetry with [Chi-Square Distribution](15-chi-square-distribution.md) is worth noticing. A single volatility is measured to $4.5\%$ on a year of data, which is quite precise; a *ratio* of two volatilities from the same amount of data is only good to $13\%$, because both numerator and denominator carry their own error and the errors compound. Comparing is much harder than measuring, and the factor is close to $\sqrt{2}$ on the variance scale, exactly as one would expect from two independent errors.

So the F's honest role in a trading context is as a source of intervals rather than verdicts. It is the exact null for a variance comparison, which is genuinely useful, and what it mostly reveals is how little a variance comparison can settle: the same heavy denominator tail that gives the family its shape is the reason the answer is imprecise, and no amount of care in constructing the test will change that.

The practical rule is to stop asking whether two volatilities are different and start asking how different they could be. The first question has an answer that is almost always no and almost always uninformative. The second has an interval, computable from the same two numbers, that says exactly what the data support — and on a year of daily observations, what the data support is a range wide enough that most operational decisions about position sizing should not be resting on the distinction in the first place.
