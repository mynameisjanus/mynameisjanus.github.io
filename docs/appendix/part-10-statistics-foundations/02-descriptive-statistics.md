# Descriptive Statistics

The previous page said a sample is one output of a process, which leaves open what to do with it. The universal first move is to compress: replace $n$ numbers with five or six and reason about those. Every such summary is the same kind of object — a functional of a distribution, evaluated at the empirical distribution instead of the true one — and that framing earns its keep immediately, because it says exactly what each summary is estimating and exactly what was discarded to get there. What it does not say, and what this page is about, is that some of these functionals estimate a parameter, some estimate a parameter that does not exist, and some estimate a property of the window they were computed over.

This page covers descriptive statistics as plug-in functionals of the empirical distribution, location summaries and the breakdown point that separates them, the algebraic ceiling a sample size puts on the third and fourth moments, the analysis-of-variance identity as the sample form of the law of total variance together with the explained-variance floor a narrow cross-section manufactures out of nothing, and rolling statistics as summaries whose window is part of their definition. It attaches a standard error to nothing, which is [Sampling Distributions](03-sampling-distributions.md); it corrects no summary for bias and derives no $n-1$, which is [Bias and Variance](07-bias-and-variance.md); it establishes which moments of a heavy-tailed law exist, which is [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md); it proves no population decomposition, which is [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md); it tests no between-group difference, which is [Part XII](../part-12-hypothesis-testing/index.md); it fits no regression and builds no design matrix, which is [Multiple Linear Regression](../part-13-regression/02-multiple-linear-regression.md); it resamples nothing, which is [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md); and it draws no charts.

The trading stake is a decomposition the course performs and does not name. [Cross-Sectional and Volatility Strategies](../../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) partitions nine sectors' monthly returns and reports that "Sectors co-move at an average correlation of 0.58, and 63% of a typical sector's monthly variance is simply the market's move wearing a sector costume", pinning `share of monthly variance explained by the common move: 63%`. That number is the sample version of the ratio [Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md) derives in the population, computed by exactly the arithmetic of the fourth section — and that section also prices what it costs to read, because with nine series and no common movement whatever the same calculation returns $11.4\%$. The floor of the statistic is one over the width of the cross-section, and nobody prints the floor.

## Every Descriptive Statistic Is One Functional Evaluated at the Wrong Distribution

Let $\theta=T(F)$ be a functional of the unknown law — the mean is $T(F)=\int x\,dF(x)$, the median is $T(F)=F^{-1}(1/2)$, the variance is $T(F)=\int(x-\int y\,dF)^2dF$. The **empirical distribution function** $\hat F_n$ places mass $1/n$ at each observation, and the **plug-in estimator** is $\hat\theta=T(\hat F_n)$: run the same formula on the sample's own law. Nearly every summary in routine use is of this form, and the ones that are not are worth spotting.

The framing pays three dividends at once. It names the estimand, so "what is the sample skewness an estimate of" has an answer rather than a shrug. It explains why some summaries are stable and others are not, since a functional that depends on $F$ only through a bounded region of it inherits that stability. And it makes the exceptions visible: the sample variance with divisor $n-1$ is *not* the plug-in, because the plug-in divides by $n$, and the gap between the two is a bias correction that [Bias and Variance](07-bias-and-variance.md) derives.

What the framing does not supply is a guarantee that $T(F)$ exists. $T(\hat F_n)$ always does — a finite sample has a fourth moment no matter what law produced it — and the fact that a formula returns a number is not evidence that the number is estimating anything. Two of the three failures on this page are of that kind, and the third is a case where $T$ was quietly redefined by the window it was computed over.

## Location Splits Into Two Answers Exactly Where the Tail Begins

The mean and the median are both location summaries and they answer different questions. The mean is $\int x\,dF$ and it is the only summary that adds: the mean of a sum is the sum of the means, which is why profit and loss, portfolio returns and total cost are all mean-shaped quantities. The median is $F^{-1}(1/2)$ and it describes a typical case, which is what a person planning for a normal day wants.

On a symmetric light-tailed law the distinction is academic. On the laws finance produces it is not. The course measures the overnight gap distribution and reports that "On a median night the open lands 29 bp from the prior close; the mean of 44.6 bp is dragged up by a fat tail", with `mean |gap| 44.6 bp, median 29.0 bp, 95th pct 134 bp, worst 1045 bp`. The two answers differ by fifty-four percent and both are correct: a trader estimating the cost of a typical overnight hold wants $29$, and a desk budgeting the annual cost of three hundred such holds wants $44.6$, because costs add.

The **breakdown point** — the fraction of the sample that must be corrupted to move a statistic arbitrarily far — orders these summaries and explains the pattern. The mean's is $0$, the median's is $1/2$, an $\alpha$-trimmed mean's is $\alpha$. That ordering is usually presented as a robustness ranking, which quietly assumes all three are estimating the same thing. They are not, and the sharpest case in the course is a levered book where "the median is still positive (+13.4%) while the mean is negative (−10.5%)" — the two summaries disagree about whether money was made. The lesson's verdict is the right one and it is a statement about estimands rather than about robustness: "Reporting the median of a levered strategy is how that minority gets hidden."

## The Sample Kurtosis Has a Ceiling of $n-3$, Which Is Why It Grows Instead of Converging

The course reports SPY's daily excess kurtosis as `skewness -0.20, excess kurtosis 11.41` and then, unusually, distrusts its own number: at the tail index fitted later, "the population fourth moment does not exist at all, so the sample kurtosis is not converging on eleven — it grows with the sample, and the honest statement is that the tail is heavy enough to break the statistic being used to describe it." Whether the moment exists is settled in [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md). What this section adds is the mechanism, which is purely algebraic and holds for every sample ever taken.

??? note "Proof that the sample excess kurtosis of $n$ observations cannot exceed $n-3$, and that the skewness is bounded in the same way"
    Write $z_i=(x_i-\bar x)/\sqrt{m_2}$ for the standardized residuals, where $m_2=\frac1n\sum_i(x_i-\bar x)^2$, so that $\sum_i z_i^2=n$ by construction. The sample kurtosis is $b_2=\frac1n\sum_i z_i^4$. By Cauchy–Schwarz applied to the vectors $(z_i^2)$ and $(1)$,

    $$\Big(\sum_i z_i^{2}\Big)^{2}\le n\sum_i z_i^{4},\qquad\text{so}\qquad n^{2}\le n\cdot n\,b_2\;\Longrightarrow\;b_2\ge1,$$

    which is the lower bound. For the upper bound note that $\sum_i z_i^4\le\big(\max_i z_i^2\big)\sum_i z_i^2=n\max_i z_i^2$, and $\max_i z_i^2\le n$ because a single standardized residual cannot exceed the total $\sum z_i^2=n$. Hence $b_2\le n$ and the excess kurtosis satisfies

    $$b_2-3\;\le\;n-3,$$

    with equality approached only when one observation carries the entire sum of squares and the rest are identical. The same argument on $(z_i)$ and $(z_i^2)$ gives $|b_1|\le(n-2)/\sqrt{n-1}$ for the sample skewness.

    The load-bearing hypothesis is the existence of the population fourth moment, and when it fails nothing in the algebra above changes — the bound is a fact about $n$ numbers and has no probabilistic content at all. What changes is the interpretation: with a finite population kurtosis the sample statistic converges to it and the ceiling is slack; without one the statistic has nowhere to converge and drifts upward with the only quantity that is growing, which is the ceiling. **A sample kurtosis is an estimate of a population kurtosis only when the latter exists, and otherwise it is a slowly growing function of the sample size wearing the name of a parameter.**

```python
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(10021)
reps, nu = 4_000, 3.4                                          # no fourth moment exists at all

print(f"  sample excess kurtosis of {reps} samples from a law whose population value is infinite")
print("        n     ceiling n-3    mean b2-3    median b2-3    share > 11.41    max b2-3")
for n in (60, 252, 1_260, 6_300, 25_200):
    x = tdist.rvs(nu, size=(reps, n), random_state=rng)
    z = x - x.mean(axis=1, keepdims=True)
    k = (z ** 4).mean(axis=1) / ((z ** 2).mean(axis=1)) ** 2 - 3
    print(f"  {n:9d} {n - 3:13d} {k.mean():12.2f} {np.median(k):14.2f}"
          f" {np.mean(k > 11.41):16.3f} {k.max():11.1f}")
# =>   sample excess kurtosis of 4000 samples from a law whose population value is infinite
#            n     ceiling n-3    mean b2-3    median b2-3    share > 11.41    max b2-3
#             60            57         3.93           2.05            0.074        49.1
#            252           249         8.99           4.64            0.180       204.7
#           1260          1257        18.75           8.72            0.365      1137.0
#           6300          6297        33.19          14.08            0.640      3311.8
#          25200         25197        68.38          21.14            0.920     14747.5
```

The mean column is the headline and it does not converge. It reads $3.93$, $8.99$, $18.75$, $33.19$, $68.38$ across sample sizes spanning a factor of four hundred and twenty, roughly doubling each time the sample quadruples. There is no plateau, no sign of one, and no sample size at which the sequence would settle, because the quantity it would settle at is infinite.

The median column shows what a single practitioner actually sees, and it is the more useful number: $2.05$, $4.64$, $8.72$, $14.08$, $21.14$. At $n=6{,}300$ — twenty-five years of daily data, the course's sample — the median sample excess kurtosis of this law is $14.08$, and the share of samples returning more than $11.41$ is $0.640$. **The course's measured $11.41$ is an entirely ordinary draw from a law with no fourth moment**, which is a far stronger statement than "SPY has fat tails": it means the number would have been different on a different quarter-century and larger on a longer one, by construction rather than by accident.

The ceiling column explains the mechanism and the maximum column shows it biting. At $n=60$ no sample can report more than $57$ and the worst observed is $49.1$; at $n=25{,}200$ the ceiling is $25{,}197$ and the worst observed is $14{,}747.5$. The statistic is not measuring the tail so much as reporting how much room the sample size gave it, and the two are indistinguishable from a single number.

## Analysis of Variance Is the Law of Total Variance With Hats On

[Law of Total Variance](../part-04-expectation-and-moments/08-law-of-total-variance.md) proves that $\mathrm{var}(X)=\mathrm{var}(\mathbb{E}[X\mid G])+\mathbb{E}[\mathrm{var}(X\mid G)]$ in the population and says the sample version "is analysis of variance, in Descriptive Statistics". This is that section, and the sample version is an identity rather than a theorem — it holds for any numbers whatever, with no model, no randomness and no assumption.

??? note "Proof that the total sum of squares splits exactly into between and within, and that the ratio it forms is biased upward by construction"
    Let $x_{gi}$ be observation $i$ of group $g$, with group means $\bar x_g$, grand mean $\bar x$ and group sizes $n_g$. Write each deviation from the grand mean as a deviation from its group mean plus a group effect, $x_{gi}-\bar x=(x_{gi}-\bar x_g)+(\bar x_g-\bar x)$, and square:

    $$\sum_g\sum_i(x_{gi}-\bar x)^2=\sum_g\sum_i(x_{gi}-\bar x_g)^2+\sum_g n_g(\bar x_g-\bar x)^2+2\sum_g(\bar x_g-\bar x)\sum_i(x_{gi}-\bar x_g).$$

    The inner sum of the cross term is $\sum_i(x_{gi}-\bar x_g)=0$ for every $g$, because a group mean is by definition the point about which its own deviations cancel. So the cross term vanishes identically and

    $$\underbrace{\text{SST}}_{\text{total}}=\underbrace{\text{SSW}}_{\text{within}}+\underbrace{\text{SSB}}_{\text{between}},$$

    which is the sample form of the population decomposition, with SSW playing $\mathbb{E}[\mathrm{var}(X\mid G)]$ and SSB playing $\mathrm{var}(\mathbb{E}[X\mid G])$. The **explained-variance ratio** is $\hat\eta^2=\text{SSB}/\text{SST}$. Under a null in which all group means are equal, $\mathbb{E}[\text{SSB}]=(G-1)\sigma^2$ and $\mathbb{E}[\text{SSW}]=(n-G)\sigma^2$, so $\hat\eta^2$ has expectation approximately $(G-1)/(n-1)$ rather than zero.

    The load-bearing step is the orthogonality, and it is doing two jobs that are usually credited separately. It makes the identity exact, with no error term and no hypothesis. It also makes the ratio biased, because the group means it measures deviation *from* were computed from the same observations that supply the residuals, so each fitted mean absorbs a share of the noise. **The decomposition is an arithmetic identity that never fails and an estimate that is never unbiased, and the same fact — that $\bar x_g$ was fitted rather than known — is responsible for both.**

```python
import numpy as np

rng = np.random.default_rng(10023)
reps, T = 400, 300                                             # months of history per series

x = rng.standard_normal((9, 300))                              # nine groups, no difference at all
gm, om = x.mean(axis=1, keepdims=True), x.mean()
sst = ((x - om) ** 2).sum()
ssb = (x.shape[1] * (gm - om) ** 2).sum()
ssw = ((x - gm) ** 2).sum()
print(f"  one panel of {x.shape[0]} groups x {x.shape[1]}: SST {sst:.6f}"
      f"  SSB + SSW {ssb + ssw:.6f}  gap {abs(sst - ssb - ssw):.2e}")

print("        N      T    mean explained share    floor 1/N")
for N in (5, 9, 20, 50):
    acc = []
    for _ in range(reps):
        r = rng.standard_normal((T, N))
        f = r.mean(axis=1, keepdims=True) - r.mean()           # the cross-sectional common move
        d = r - r.mean(axis=0)
        acc.append(1 - ((d - f * ((d * f).sum(axis=0) / (f ** 2).sum())) ** 2).sum(axis=0)
                   / (d ** 2).sum(axis=0))
    print(f"  {N:9d} {T:6d} {np.mean(acc):23.4f} {1 / N:12.4f}")
# =>   one panel of 9 groups x 300: SST 2772.557081  SSB + SSW 2772.557081  gap 4.55e-13
#            N      T    mean explained share    floor 1/N
#              5    300                  0.2016       0.2000
#              9    300                  0.1137       0.1111
#             20    300                  0.0527       0.0500
#             50    300                  0.0231       0.0200
```

The first line is the identity, checked rather than assumed. On one panel of nine groups by three hundred observations the total sum of squares is $2772.557081$ and the between-plus-within sum is $2772.557081$, agreeing to $4.55\times10^{-13}$ — floating-point noise, which is the correct size for an exact algebraic identity evaluated in double precision. Nothing about the data was assumed to produce that; the same line would hold on prices, on temperatures, or on random digits.

The table is what the identity costs when it is read as evidence. Each row builds $N$ *completely independent* series, computes the cross-sectional mean, and asks what share of each series' variance that mean explains. The answer is not zero. It is $0.2016$, $0.1137$, $0.0527$ and $0.0231$ at five, nine, twenty and fifty series, tracking the floor $1/N$ — $0.2000$, $0.1111$, $0.0500$, $0.0200$ — to within a few percent everywhere. The mechanism is mechanical: each series contributes $1/N$ of the average it is being regressed on, so it explains itself in proportion.

The consequence for the stake is a subtraction. The course's nine sectors return $63\%$, and nine independent series return $11.4\%$, so the common movement genuinely present accounts for roughly fifty-two points rather than sixty-three. That does not overturn the lesson's conclusion — fifty-two points is still most of the variance and the long-short argument built on it stands — but it does fix the interpretation. **An explained-variance share is not interpretable without the width of the cross-section it was computed over, because one over that width is what the statistic returns when there is nothing to explain.**

!!! note "The explained-variance ratio is an arithmetic identity rather than evidence, which is why a high $R^2$ is a statement about a decomposition and not about a mechanism"
    The split into between and within holds for any partition of any numbers, including partitions chosen after looking at the data and partitions with no meaning at all. What a large $\hat\eta^2$ establishes is that the grouping variable tracks the outcome in this sample; what it does not establish is that the grouping caused anything, that the relationship is stable, or that the share would survive a different cross-section. The three routine misreadings are worth naming: comparing $R^2$ across models fitted on different numbers of series, where the floors differ; reading a rising $R^2$ over time as rising co-movement, when a shrinking universe produces the same signature; and treating the residual share as idiosyncratic risk, when it also contains every common factor the grouping failed to capture. The correction for the floor is standard and cheap — $\omega^2=1-\frac{(1-\hat\eta^2)(n-1)}{n-G}$ — and reporting it alongside the raw ratio costs one line and removes the whole ambiguity.

## A Rolling Statistic Is a Statistic Whose Window Is Part of Its Definition

[Pandas and Polars](../../part-02-python/02-pandas-and-polars.md) computes rolling means and $z$-scores, warns that one must "Insist on `min_periods` equal to the window — a '20-bar average' computed from 3 bars is a different, noisier statistic wearing the same name", and defers the reason here. There are two reasons, they are unrelated to each other, and both are measurable.

??? note "Proof that a $w$-bar rolling mean of independent observations has autocorrelation exactly $1-k/w$, so the series carries one independent value per window"
    Let $X_t$ be independent with variance $\sigma^2$ and let $M_t=\frac1w\sum_{j=0}^{w-1}X_{t-j}$. Then $\mathrm{var}(M_t)=\sigma^2/w$. For a lag $k<w$ the windows of $M_t$ and $M_{t+k}$ share exactly $w-k$ observations, and distinct observations are uncorrelated, so

    $$\mathrm{cov}(M_t,M_{t+k})=\frac{(w-k)\sigma^{2}}{w^{2}},\qquad \rho_k=\frac{(w-k)\sigma^2/w^2}{\sigma^2/w}=1-\frac{k}{w},$$

    and for $k\ge w$ the windows are disjoint and $\rho_k=0$ exactly. The autocorrelation function is therefore a triangle that falls linearly to zero at lag $w$ and stays there.

    The load-bearing step is the overlap, and the point to hold on to is that the input was independent. Not one property of the data produced this shape; the window did, and the same triangle appears whether the underlying series is returns, temperatures or random digits. **Any autocorrelation estimate, portmanteau test or $t$-statistic computed on a rolling series is reading a structure the smoother installed**, which is the same double counting that turns the course's honest $t$ of $1.73$ into a reported $7.40$, arriving here through a moving average instead of through an overlapping sum.

```python
import numpy as np

rng = np.random.default_rng(10027)
n, w, trials = 6_300, 20, 2_000                                # a twenty-bar window on pure noise

x = rng.standard_normal(n)
roll = np.convolve(x, np.ones(w) / w, mode="valid")
print("   lag    measured acf    theory 1 - k/w")
for k in (1, 5, 10, 19, 20):
    print(f"  {k:5d} {np.corrcoef(roll[:-k], roll[k:])[0, 1]:15.4f} {max(0.0, 1 - k / w):18.4f}")

rej, fires = 0, {j: 0 for j in (3, 5, 10, 20)}
for _ in range(trials):
    z = rng.standard_normal(n)
    m = np.convolve(z, np.ones(w) / w, mode="valid")
    rej += abs(m.mean() / (m.std(ddof=1) / np.sqrt(m.size))) > 1.96
    for j in fires:                                            # min_periods = j, then act on z
        fires[j] += abs((z[j] - z[:j].mean()) / z[:j].std(ddof=1)) > 2
print(f"  rows {roll.size} of {n}, n_eff about {roll.size / w:.0f},"
      f" naive t rejects {rej / trials:.3f} at a nominal 0.05")
print("   min_periods    share of z-scores beyond 2 sigma")
for j in (3, 5, 10, 20):
    print(f"  {j:12d} {fires[j] / trials:32.3f}")
# =>    lag    measured acf    theory 1 - k/w
#          1          0.9503             0.9500
#          5          0.7404             0.7500
#         10          0.4857             0.5000
#         19          0.0439             0.0500
#         20         -0.0009             0.0000
#      rows 6281 of 6300, n_eff about 314, naive t rejects 0.679 at a nominal 0.05
#       min_periods    share of z-scores beyond 2 sigma
#                 3                            0.230
#                 5                            0.134
#                10                            0.083
#                20                            0.059
```

The first panel is the proof rendered on data that has no structure whatsoever. Independent standard normals, smoothed over twenty bars, produce a lag-one autocorrelation of $0.9503$ against a predicted $0.95$, and the measured values track $1-k/w$ down the whole triangle — $0.7404$, $0.4857$, $0.0439$ — before collapsing to $-0.0009$ at lag twenty, where the windows stop overlapping. A researcher who computed this autocorrelation on a smoothed price series and concluded the market has memory would be reading their own moving average.

The consequence line prices it. The rolling series has $6{,}281$ rows and about $314$ independent values, and a naive $t$-test of its mean rejects at a nominal $5\%$ level $67.9\%$ of the time when the truth is exactly zero. That is the same order of failure the previous page measured for overlapping sums, arriving through the most routine preprocessing step in the toolkit.

The second panel is the `min_periods` question and it is the one with immediate trading consequences. A $z$-score computed against a mean and standard deviation estimated from only three prior bars exceeds two sigma $23.0\%$ of the time on pure noise; from five bars, $13.4\%$; from ten, $8.3\%$; from a full twenty, $5.9\%$, which is close to the $4.6\%$ a normal would give. The mechanism is not subtle — a standard deviation estimated from three observations is itself wildly noisy, and dividing by a noisy denominator manufactures large ratios — but the effect is severe and it lands entirely in the earliest rows of every backtest. **A threshold rule run with a partial window fires four times as often as it is supposed to, and it does so at the start of the sample where a researcher is least likely to be looking.**

!!! warning "A statistic computed on overlapping windows reports the window's autocorrelation as the data's, and the rows before the window fills are where a threshold rule does a disproportionate share of its trading"
    Two separate defects live in one line of code and they need separate treatment. The overlap defect means any test applied to a rolling series must divide its nominal row count by the window before consuming it, and the free diagnostic is to recompute the same statistic on **non-overlapping** windows and compare the two standard deviations: the ratio should be about $\sqrt{w}$, and the non-overlapping count is the sample size any test is entitled to use. The partial-window defect means the leading rows are a different estimator with heavier tails, and the fix is the one the lesson prescribes — set `min_periods` equal to the window and let the leading NaNs stand, because they are the honest statement that no valid window exists yet. Neither defect is visible in a plot of the smoothed series, both are visible in one line of output, and the second is worth checking on any strategy whose entry condition is a threshold on a standardized quantity: **count how many of the backtest's trades fire in the first $w$ bars, and if the answer exceeds the share of the sample those bars represent, the leading rows are being traded rather than skipped.**

## A Summary Is a Question, and the Window Is Half of It

Three failures were established and they fail in three different places. The kurtosis fails at the estimand: the functional has no population value, so the sample version reports the ceiling its sample size permits rather than a property of the law. The explained-variance ratio fails at the baseline: the identity behind it is exact and the number it returns is bounded below by one over the width of the cross-section, so the statistic answers a question about the panel's shape before it answers one about the world. The rolling statistic fails at the definition: the window is not a display choice but part of what is being estimated, and it installs both a correlation structure and, in its opening rows, a different estimator entirely.

The symmetry worth carrying is with the trade the previous part kept making. The median buys insensitivity to the tail by declining to use it, which is the same purchase [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) makes when it resamples blocks instead of observations and the same one a trimmed estimator makes on any sample — information is deliberately discarded to buy stability, and the trade is favourable exactly when the discarded part was noise and catastrophic when it was the signal. A levered book's ruin lives in the tail the median deletes, and a heavy-tailed asset's risk lives in the moment the kurtosis cannot estimate. Neither summary is wrong; both are answers to questions that were chosen rather than asked.

Every number on this page was quoted without an error bar, which is the omission the course calls a rumor. A sample kurtosis of $14.08$, an explained share of $0.1137$ and a lag-one autocorrelation of $0.9503$ are all themselves draws, and saying how far each would move on a second sample requires the law of the statistic rather than the law of the data. That is [Sampling Distributions](03-sampling-distributions.md).

**A summary is a question asked of a sample, and the two ways it misleads are answering a question that has no answer and answering a different question than the one that was asked.**
