# Population vs Sample

Probability is a forward map: fix a law, and every question about the data it produces has exactly one answer. Statistics runs the map backwards, and the inverse is not a function. Many laws produce the sample in hand; one of them fits it perfectly and is certainly wrong; and the data alone will never say which of the rest to prefer. Everything in the eight parts that follow is a disciplined way of choosing among them and of stating what the choice cost. This page fixes the two objects the choice is made between — a **population**, which is a distribution and not a list, and a **sample**, which is one output of it and not a subset of it — and that distinction reads as pedantry right up to the moment a return series turns out to be a single path from a process that was never going to repeat.

This page covers the inversion from probability to statistics and why it has no unique answer, the population as a data-generating process rather than a finite roster, ergodicity as the hypothesis that makes a time average an estimate of an ensemble average, selection as a change of the population rather than a loss of precision, and dependence as the thing that deletes observations without touching the estimate. It takes no limit in $n$ and proves no law of large numbers, which is [Part VII](../part-07-asymptotic-theory/index.md); it derives no effective sample size, which is [Random Processes](../part-08-stochastic-processes/01-random-processes.md); it computes no summary of a sample, which is [Descriptive Statistics](02-descriptive-statistics.md); it derives the law of no statistic, which is [Sampling Distributions](03-sampling-distributions.md); it restricts the population to no parametric family, which is [Statistical Models](04-statistical-models.md); it names no property of an estimator, which is [Properties of Estimators](../part-11-parameter-estimation/02-properties-of-estimators.md); it charges nothing for a searched universe, which is [Part XV](../part-15-multiple-testing/index.md); and it certifies no dataset as clean.

The trading stake is the most expensive sentence in Part VII of the course. [Reinforcement Learning and Meta-Labeling](../../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) runs twenty seeds of a Q-learner on daily bars, watches the cohort deliver "a lottery centered on the dumb baseline", and then says what the twenty-seed spread was measuring: "Markets give a learner one path — non-repeating, regime-shifting, whisper-thin in reward — and an agent optimized on that path has learned *what happened*, not *what happens*." The same paragraph supplies the control that turns an observation into an argument — "The same learner handed a stationary, dense, replayable world recovered 94% of the optimum on every seed" — which acquits the algorithm and convicts the sample. The third section prices that gap exactly, and prices it as an ergodicity failure rather than a sample-size one: a path whose estimate is unbiased at every horizon, whose standard error shrinks on schedule, and whose nominal $95\%$ interval for the number the ensemble would have produced covers it $30.4\%$ of the time after two hundred and fifty years.

## Statistics Runs Probability Backwards, and the Inverse Is Not a Function

Probability fixes a law $P$ and asks about the data $X$. Statistics fixes the observed data $x_{1:n}$ and asks about $P$. The asymmetry between those two sentences is the whole subject, and it is worth making precise rather than atmospheric.

The forward map is well defined: from $P$ the joint law $P^{\otimes n}$ of an independent sample follows, and every probability, moment and quantile of every function of the sample is determined. The backward map is not a map at all. The set of laws under which the observed $x_{1:n}$ has positive density is enormous, and it always contains a member that explains the data better than the truth does — the empirical distribution itself, which puts mass $1/n$ on each observed point and assigns probability zero to everything that did not happen. That law maximizes the likelihood, is certainly wrong, and is the reason inference needs a principle beyond "fit the data".

There are exactly three ways to break the tie, and this appendix uses all of them in later parts. Restrict the candidate set in advance, which is a **model** and is [Statistical Models](04-statistical-models.md). Rank the candidates by how well they explain what was seen, which is likelihood and is [Part XI](../part-11-parameter-estimation/index.md). Or average over the candidates under a weighting fixed before the data arrived, which is [Part XVI](../part-16-bayesian-statistics/index.md). This page does none of the three. It does the thing that has to happen first, which is to say what the candidates are candidates *about*.

## A Population Is a Data-Generating Process, and the Sample Is Its Output Rather Than Its Subset

The picture most people carry is of a bin containing $N$ items from which $n$ are drawn. That picture is a special case, and the tell that separates it from the general one is the finite-population correction: the standard error of a sample mean drawn without replacement from $N$ items carries a factor $\sqrt{(N-n)/(N-1)}$, which is exactly zero when $n=N$. Sample everything and the uncertainty vanishes, because there was nothing left to be uncertain about.

A return series never has that property, and the reason is not that $N$ is large. There is no bin of possible SPY days from which the last twenty-five years were drawn. There is a mechanism, it ran once, and what was recorded is its output. Doubling the history does not exhaust anything, and no amount of data drives the uncertainty about next year's mean return to zero. The **population** is therefore the mechanism — a probability law over paths — and the sample is a realization of it, which is a different relation entirely from membership.

Two consequences follow immediately and both are used for the rest of the part. The first is that the population is a **modelling choice**: it is asserted, not observed, and asserting it is the first and least examined decision in any study. The second is arithmetic and more uncomfortable: **you can observe every datum that exists and still have a sample of size one.** A cross-sectional question asked of all five hundred index members on a single date is a census in the cross-section and a single draw in time, and which of those two the standard error is describing depends entirely on which direction the estimator averages over.

## One Path Converges Beautifully and Need Not Converge to the Ensemble

Stationarity — the assumption that the law does not move — is what licenses treating a single history as informative about anything, and it is weaker than it looks. It guarantees that the sample mean of a path settles down. It does not guarantee that what it settles down to is a number.

??? note "Proof that a stationary series has a convergent sample mean, and that the limit is a constant only under a strictly stronger hypothesis"
    Let $\{X_t\}$ be strictly stationary with $\mathbb{E}|X_1|<\infty$. Birkhoff's ergodic theorem states that

    $$\bar X_n=\frac1n\sum_{t=1}^{n}X_t\;\longrightarrow\;\mathbb{E}[X_1\mid\mathcal I]\qquad\text{almost surely},$$

    where $\mathcal I$ is the invariant $\sigma$-field — the collection of events unchanged by shifting the series in time. If $\mathcal I$ is trivial, meaning every such event has probability $0$ or $1$, the conditional expectation collapses to the constant $\mathbb{E}[X_1]$ and the time average estimates the ensemble average. That triviality is the definition of **ergodicity**, and it is a separate assumption from stationarity.

    The counterexample that matters in finance is one line long. Let $\mu\sim\mathcal{N}(m,\tau^{2})$ be drawn once, at time zero, and never redrawn; conditional on it let $X_t=\mu+\varepsilon_t$ with $\varepsilon_t$ independent and mean zero. The sequence is strictly stationary, every marginal has mean $m$, and

    $$\bar X_n\;\longrightarrow\;\mu\;\ne\;m\qquad\text{on almost every path},$$

    because the event $\{\mu\le c\}$ is shift-invariant and non-trivial. The limit exists, is approached at the usual rate, and is the wrong number on every path but a null set of them.

    The load-bearing hypothesis is ergodicity, not stationarity, and what fails when it fails is not the convergence — the convergence happens, on schedule, looking exactly as it should. What fails is the identification of the limit with a population quantity. **A single path can converge perfectly and converge to the wrong number, and no diagnostic computed from inside that path can tell the two cases apart**, because every such diagnostic measures how settled the average is and both cases are equally settled.

```python
import numpy as np

rng = np.random.default_rng(10011)
paths, m, s, tau = 20_000, 0.075, 0.195, 0.060                 # the drift is drawn once per path

print(f"  {paths} paths of one stationary process, each asked for the mean it was built from")
print("   years       n    mean(x_bar)    sd(x_bar)    iid SE    ratio    cover 95%    drift 2nd half")
for years in (1, 5, 25, 250):
    n = 12 * years
    mu = m + tau * rng.standard_normal((paths, 1))              # fixed at time zero, never revealed
    r = mu / 12 + (s / np.sqrt(12)) * rng.standard_normal((paths, n))
    xb = 12 * r.mean(axis=1)
    se = 12 * r.std(axis=1, ddof=1) / np.sqrt(n)                # what the path reports about itself
    half = 12 * r[:, n // 2:].mean(axis=1) - 12 * r[:, :n // 2].mean(axis=1)
    print(f"  {years:7d} {n:7d} {xb.mean():14.4f} {xb.std(ddof=1):12.4f} {s / np.sqrt(years):9.4f}"
          f" {xb.std(ddof=1) / (s / np.sqrt(years)):8.2f}"
          f" {np.mean(np.abs(xb - m) <= 1.96 * se):12.3f} {np.median(np.abs(half)):17.4f}")
# =>   20000 paths of one stationary process, each asked for the mean it was built from
#       years       n    mean(x_bar)    sd(x_bar)    iid SE    ratio    cover 95%    drift 2nd half
#            1      12         0.0733       0.2037    0.1950     1.04        0.913            0.2626
#            5      60         0.0759       0.1055    0.0872     1.21        0.891            0.1169
#           25     300         0.0748       0.0717    0.0390     1.84        0.710            0.0530
#          250    3000         0.0745       0.0617    0.0123     5.01        0.304            0.0165
```

The third column is the estimator's defence and it is a complete one. The mean of the path means reads $0.0733$, $0.0759$, $0.0748$ and $0.0745$ against a true ensemble mean of $0.075$, at every horizon. Nothing here is biased, no arithmetic is wrong, and an auditor checking the estimator against the truth across many worlds would sign it off at every row.

The next three columns are where the world and the path part company. The standard error an iid calculation predicts falls from $0.1950$ to $0.0123$, a factor of sixteen, exactly as $1/\sqrt{T}$ requires. The standard deviation the path means actually have falls from $0.2037$ to $0.0617$ and then stops, because $0.060$ of it is the dispersion of the drift and no quantity of observations of a fixed $\mu$ tells you anything about how $\mu$ was drawn. The ratio between the two grows from $1.04$ to $5.01$: at one year the iid formula is right, and at two hundred and fifty years it is too narrow by a factor of five.

The coverage column is the finding. A nominal $95\%$ interval, computed from the path's own data in the ordinary way, covers the ensemble mean $91.3\%$ of the time at one year and $30.4\%$ of the time at two hundred and fifty. **It gets worse as the sample grows**, monotonically, because the interval is shrinking toward a number that is not moving toward the target. And the last column closes the trap: the amount by which the estimate moves between the first half of the sample and the second falls from $0.26$ to $0.017$, so the path looks more converged at exactly the horizons where its interval is most wrong. **The diagnostic that says converged and the diagnostic that says correct are not the same diagnostic, and only one of them can be computed from a single history.**

!!! note "Stationarity and ergodicity are routinely used as synonyms, and the gap between them is where a single track record lives"
    A stationary process has marginals that do not move; an ergodic one additionally forgets its own initial conditions, so that time spent observing it substitutes for the parallel worlds nobody has. Almost every test a practitioner runs — augmented Dickey–Fuller, KPSS, split-sample comparisons of the mean — is a test of the first property, and the second has no test at all from inside one path, because the invariant $\sigma$-field is exactly the collection of facts a single realization can never vary over. That is why the twenty-seed reinforcement-learning cohort in the course is doing genuine statistical work rather than hygiene: running the same learner from twenty starts on a *replayable* environment is an ensemble, and the $94\%$-of-optimum agreement across seeds is a direct measurement of the quantity a market never lets you measure. The practical reading is uncomfortable and worth stating plainly: a manager's twenty-year record and twenty independent twenty-year records are different amounts of evidence about the same claim, and only the second kind is the kind the standard error is describing.

## A Selected Sample Estimates a Different Population, Not the Same One Less Precisely

Every dataset arrives filtered. Funds that closed are not in the database, tickers that delisted are not in the universe, and a vendor's backfilled history contains the names that were worth backfilling. The instinct is to treat this as contamination that adds noise. It is the opposite: selection removes noise from an estimate of the wrong quantity.

??? note "Proof that conditioning on survival shifts the estimand by a covariance, and that the shift carries no $n$"
    Let $S\in\{0,1\}$ indicate inclusion, with $\mathbf{P}(S=1)=\pi>0$, and let $\theta=\mathbb{E}[X]$. Decomposing the unconditional mean over the two cases,

    $$\theta=\pi\,\mathbb{E}[X\mid S=1]+(1-\pi)\,\mathbb{E}[X\mid S=0],$$

    and rearranging gives the shift in the observable estimand,

    $$\mathbb{E}[X\mid S=1]-\theta=\frac{1-\pi}{\pi}\Big(\theta-\mathbb{E}[X\mid S=0]\Big)=\frac{\mathrm{cov}(X,S)}{\pi}.$$

    Every quantity on the right is a functional of the joint law of $(X,S)$. The sample size appears nowhere. An estimator built on the survivors is therefore consistent — for $\theta+\mathrm{cov}(X,S)/\pi$, which is a perfectly well-defined number and is not the one the study set out to estimate.

    The load-bearing step is that the conditioning event depends on the outcome. If $S$ is independent of $X$ the covariance is zero and selection costs nothing but sample size, which is why a random subsample is harmless and a filtered one is not. **The error is a difference of estimands rather than an error of estimation, so it is invisible to every method whose validity is stated as a rate**, and it is the one defect in this appendix that a larger sample makes more statistically significant rather than less.

```python
import numpy as np

rng = np.random.default_rng(10013)
reps, years, mu, vol = 400, 10, 0.06, 0.18                     # a ten-year track record per fund

print(f"  one population, two samples, {reps} replications: every fund against the funds still open")
print("          n    kept    rmse(all)    bias(surv)    rmse(surv)    t(surv)")
for n in (250, 1_000, 5_000, 25_000, 100_000):
    e_all, e_surv, tstat, kept = [], [], [], []
    for _ in range(reps):
        r = mu + vol * rng.standard_normal((n, years))
        lived = r.sum(axis=1) > 0                              # the only funds a vendor lists
        s = r[lived]
        e_all.append(r.mean() - mu)
        e_surv.append(s.mean() - mu)
        tstat.append((s.mean() - mu) / (s.std(ddof=1) / np.sqrt(s.size)))
        kept.append(lived.mean())
    e_all, e_surv = np.array(e_all), np.array(e_surv)
    print(f"  {n:9d} {np.mean(kept):7.3f} {np.sqrt((e_all ** 2).mean()):12.5f}"
          f" {e_surv.mean():13.5f} {np.sqrt((e_surv ** 2).mean()):13.5f} {np.mean(tstat):10.2f}")
# =>   one population, two samples, 400 replications: every fund against the funds still open
#              n    kept    rmse(all)    bias(surv)    rmse(surv)    t(surv)
#            250   0.853      0.00361       0.01535       0.01569       4.01
#           1000   0.855      0.00181       0.01524       0.01533       7.97
#           5000   0.854      0.00082       0.01527       0.01529      17.85
#          25000   0.854      0.00038       0.01528       0.01528      39.94
#         100000   0.854      0.00018       0.01526       0.01527      79.80
```

The two error columns are the proof rendered as arithmetic. The root-mean-square error of the estimate built on every fund falls from $0.00361$ to $0.00018$ — a factor of twenty across a four-hundred-fold increase in the universe, which is $1/\sqrt n$ doing precisely what it promises. The bias of the estimate built on survivors reads $0.01535$, $0.01524$, $0.01527$, $0.01528$, $0.01526$. It is the same number five times. Roughly one and a half percentage points of annual return, manufactured by a filter that discards a seventh of the funds, and entirely indifferent to how many funds there were.

The third error column shows the two components trading places. At $n=250$ the survivor estimate's total error, $0.01569$, is slightly larger than its bias, because sampling noise still contributes. By $n=100{,}000$ it has fallen to $0.01527$, which is the bias to three significant figures — the noise has gone and what remains is all that was ever going to remain. The estimator converges, cleanly and quickly, on a number that is wrong by a fixed amount.

The last column is the one to take away. The $t$-statistic testing the survivor mean against the truth climbs $4.01$, $7.97$, $17.85$, $39.94$, $79.80$. **More data does not correct a selected sample; it certifies it**, and every increment of evidence goes into narrowing an interval around the wrong centre. The course's vendor diligence test finds exactly this signature in the field — an honest history returning a difference of $t=+0.04$ against a backfilled one returning $t=+4.16$ — and the reason that test works is that it compares two periods rather than trusting one, since "a 'history' of the S&P 500 that uses today's membership is survivorship bias wearing a timestamp" and the timestamp is the only part of it that can be checked.

## Dependence Leaves the Estimate Alone and Deletes the Observations

The third way a sample fails to be what it appears is the mildest to describe and the easiest to miss in practice: the rows are real, the estimator is fine, and there are simply fewer of them than the row count says. Overlapping windows are the standard manufacturer. A twenty-one-day aggregation of daily returns produces one row per day, each sharing twenty of its twenty-one terms with its neighbour, and the count that any test consumes is not the number of rows.

The formula for what survives — the effective sample size $n/(1+2\sum_k\rho_k)$ — belongs to [Random Processes](../part-08-stochastic-processes/01-random-processes.md) and is not rederived here. What this section does is measure the consequence, because the consequence is larger than the formula makes it sound.

```python
import numpy as np

rng = np.random.default_rng(10017)
days, vol, drift, trials = 6_138, 0.0122, 0.00012, 2_000      # twenty-four years of daily returns


def hac_t(y, lag):                                             # Newey-West at the true truncation
    e, n = y - y.mean(), y.size
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return y.mean() / np.sqrt(s / n)


def n_eff(y, lag):
    e, n = y - y.mean(), y.size
    v = e @ e / n
    vif = 1 + 2 * sum((1 - k / n) * (e[k:] @ e[:-k]) / n / v for k in range(1, lag + 1))
    return n / vif


x = drift + vol * rng.standard_normal(days)
print(f"  {days} independent daily returns aggregated into overlapping windows")
print("   window    observations    n_eff    naive t    HAC t    ratio")
for m in (1, 5, 21, 63):
    y = np.convolve(x, np.ones(m), mode="valid") if m > 1 else x.copy()
    naive = y.mean() / (y.std(ddof=1) / np.sqrt(y.size))
    h = hac_t(y, m - 1)
    print(f"  {m:8d} {y.size:15d} {n_eff(y, m - 1):8.0f} {naive:10.2f} {h:8.2f} {naive / h:8.2f}")

print(f"  false positives of the naive test at a nominal 5%, {trials} trials under a zero drift")
print("   window    naive rejection    HAC rejection")
for m in (1, 21, 63):
    nr = hr = 0
    for _ in range(trials):
        z = vol * rng.standard_normal(days)
        y = np.convolve(z, np.ones(m), mode="valid") if m > 1 else z
        nr += abs(y.mean() / (y.std(ddof=1) / np.sqrt(y.size))) > 1.96
        hr += abs(hac_t(y, m - 1)) > 1.96
    print(f"  {m:8d} {nr / trials:18.3f} {hr / trials:16.3f}")
# =>   6138 independent daily returns aggregated into overlapping windows
#       window    observations    n_eff    naive t    HAC t    ratio
#             1            6138     6138       1.50     1.50     1.00
#             5            6134     1262       3.51     1.92     1.83
#            21            6118      324       7.48     2.05     3.64
#            63            6076      119      13.94     2.28     6.11
#      false positives of the naive test at a nominal 5%, 2000 trials under a zero drift
#       window    naive rejection    HAC rejection
#             1              0.042            0.042
#            21              0.670            0.114
#            63              0.822            0.130
```

The first panel takes one series of six thousand one hundred and thirty-eight genuinely independent daily returns and aggregates it four ways. The row count barely moves — $6{,}138$ becomes $6{,}076$ at the longest window — while the effective count collapses from $6{,}138$ to $324$ to $119$. At the twenty-one-day window the naive $t$-statistic reads $7.48$ where the honest one reads $2.05$, and the course's own measurement on real data is the same arithmetic: `naive t 7.40   HAC t 1.73` and `effective n 335 of 6138`, against $324$ of $6{,}118$ here from a generator built to have no signal structure at all. The seven-sigma discovery is the twenty-one-fold double counting, and nothing else.

The second panel prices it as a decision rather than a number. Under a strictly zero drift, where every rejection is a false one, the naive test at a nominal $5\%$ level rejects $67.0\%$ of the time at the twenty-one-day window and $82.2\%$ at the sixty-three-day window. A researcher running that test on overlapping windows is not running a $5\%$ test that is a little optimistic; they are running a procedure that announces a discovery in two cases out of three when there is nothing there.

The HAC column is the repair and it is worth reading honestly. It cuts the false-positive rate from $0.670$ to $0.114$ and from $0.822$ to $0.130$ — an enormous improvement, and still more than twice the nominal $5\%$. **A correction that removes ninety percent of a problem leaves a test that is wrong twice as often as it claims**, which is a better place to be and is not the place the reported $p$-value says you are.

!!! warning "Every diagnostic a single history can run tests independence or a moving distribution, and the assumption that usually fails is the one asserting the history came from the population in question"
    The three claims bundled inside "independent and identically distributed" fail differently and are checkable to very different degrees. Independence is testable from one path, by autocorrelation or by any of the portmanteau statistics, and it is consequently the one that gets tested. Identical distribution is partly testable, by comparing halves or by the stationarity tests of [Statistical Models](04-statistical-models.md). That the shared law is the population's — the ergodicity claim of the third section — has no test from inside the path at all, and it is the failure that costs the most. The free diagnostic that catches the largest share of what the other two miss is a block comparison: **split the history into $K$ non-overlapping blocks, recompute the statistic on each, and compare the spread of the $K$ values against the standard error the full sample reports.** If the block-to-block dispersion exceeds roughly $\sqrt{K}$ times that standard error, variation is entering from somewhere the iid formula does not price, and the ratio between the two is a direct estimate of the factor by which the reported interval is too narrow — the $1.84$ and $5.01$ of the first block, measurable on one path. It costs one line, it is the finite-sample version of what the twenty-seed cohort was doing, and it answers a question no $p$-value on the full sample can be asked.

## The Population Is a Modelling Choice, and It Is the First One You Make

Three things were established and they are three different failures wearing one costume. A path can converge beautifully to a number that is not the population's, because stationarity buys the convergence and only ergodicity buys the target. A filtered sample estimates a different quantity than the one intended, by a shift that carries no $n$, so more data narrows the interval and leaves the centre exactly where it was. And a dependent sample estimates the right quantity with the right estimator while containing a small fraction of the observations its row count advertises. In the first and third cases the estimator is unbiased; in the second it is consistent; in all three the number that came out is a correct answer to a question nobody asked.

The symmetry worth carrying forward is with the method that closed the previous part. [Bootstrap Methods](../part-09-monte-carlo-methods/07-bootstrap-methods.md) is this page's mirror image: it declines to name a parametric family, substitutes the empirical distribution $\hat F_n$ for the unknown $F$, and is scrupulously honest about everything downstream of that substitution. What it cannot be honest about is the substitution itself, which is exactly the operation this page says is the whole problem — $\hat F_n$ is a faithful stand-in for $F$ when the sample is a fair draw from it, and every failure above is a case where it is not. The bootstrap's resampling and this page's ergodicity, selection and dependence are the same question asked from opposite ends, and neither end can answer it.

That is why the population is best treated as the first modelling assumption rather than as a fact discovered before modelling begins. It is asserted; it is asserted before any estimator is chosen, any test is run, or any interval is computed; and every number produced afterwards is conditional on it in a way that no subsequent diagnostic revisits. Having fixed what the sample is an output of, the next question is what a summary of that sample summarizes — which turns out to have three separate answers, one of them for quantities that do not exist. That is [Descriptive Statistics](02-descriptive-statistics.md).

**A sample is evidence about a population only under an assumption that the sample cannot be used to check, and the discipline consists of naming that assumption out loud rather than of computing anything better.**
