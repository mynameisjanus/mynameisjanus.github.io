# Statistical Power

Power is the only quantity in a testing report that can be computed before any data is collected, and it is the one nobody computes. The reason is not difficulty — the arithmetic is a single line — but that computing it often ends the project. A power calculation asks what effect the study could detect if it were there, and in trading the honest answer is usually that the study could not detect anything worth trading, which converts the exercise from an experiment into a formality. The consequences run in both directions. A test that fails to reject has produced no evidence of absence, and a test that does reject, when its power was low, has produced an estimate that is guaranteed to be too large.

This page covers power as a function on the alternative rather than a number, the inversion of that function into a sample size and, for a Sharpe ratio, into a span of calendar time, the auditing of a research programme's tests before any of them is run, the emptiness of a non-rejection from an underpowered test, and the inflation that conditioning on significance produces in the surviving estimates. It does not define the two errors or the frontier they trade along, which is [Type I and Type II Errors](04-type-i-and-type-ii-errors.md); it does not characterize the p-value, which is [p-values](03-p-values.md); it does not derive the standard error of a Sharpe ratio, which is [The Delta Method](../part-07-asymptotic-theory/04-delta-method.md); it does not treat the inflation produced by *selecting* the best of many candidates, which is [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md) and [Bayesian Estimation](../part-11-parameter-estimation/05-bayesian-estimation.md); it corrects no family of tests for its size, which is [Part XV](../part-15-multiple-testing/index.md); and it never rescues a study after the data has been seen.

The trading stake is a famous effect the course re-tests and cannot find. [Seasonality and Calendar Effects](../../part-04-strategy-development/04-seasonality-and-calendar-effects.md) reports `Mon: mean +2.1 bp/day (t = +0.55, n 1201)` alongside `Tue: mean +6.2 bp/day (t = +1.85, n 1315)` and three more weekdays, and concludes: "The famous negative Monday is now +2.1 basis points at t = 0.55: not weakened, not attenuated — *gone*, sign and all." Asked jointly the calendar answers `joint F, weekday dummies: p = 0.84`. Section 3 audits those five tests for power and finds that the strongest conclusion they could have supported was much weaker than the one the literature drew from them.

## Power Is a Function on the Alternative Set, and Quoting It as a Number Has Fixed an Effect Size Nobody Wrote Down

The **power function** of a test is $\beta_\varphi(\theta)=\mathbf{P}_\theta(\varphi=1)$, defined on the whole parameter space. Restricted to the null it is the object whose supremum is the size; restricted to the alternative it is the probability of detection. It is a function, not a number, and every statement of the form "the test has $80\%$ power" has silently evaluated it at one point of $\Theta_1$ that the statement does not name.

The point matters because $\beta_\varphi$ varies from $\alpha$ to nearly one across a typical alternative set. Where $\Theta_1$ has the null in its closure, alternatives arbitrarily close to $\theta_0$ are detected with probability arbitrarily close to $\alpha$ — no test does better, as [Type I and Type II Errors](04-type-i-and-type-ii-errors.md) established — so the infimum of power over the alternative is $\alpha$ for every test ever built. A useful power statement therefore requires naming the smallest effect worth detecting, and that number comes from economics rather than statistics. For a strategy it is the Sharpe below which the trade is not worth doing after costs and capital charges; for a risk model it is the breach-rate error that would change a limit. Nothing in the data supplies it, which is precisely why it is usually left unstated.

Naming that smallest interesting effect is easier than it sounds, because a trading application has one waiting. A strategy must clear its costs, its financing and the capital it consumes before it is worth running, so the smallest Sharpe worth detecting is the smallest Sharpe worth trading, and that number already exists on every desk that has ever rejected a proposal. The same applies to a risk model: the smallest breach-rate error worth detecting is the one that would move a limit. The reason power calculations are rare is therefore not that the effect size is unknowable but that writing it down converts a vague ambition into a falsifiable design constraint, and the constraint is frequently not satisfiable with the data available — which is a finding, and an early one.

The dependence on the alternative is what makes power a *design* quantity. Everything it needs — the effect size, the sample size, the level, the statistic's standard error — is known before the first observation arrives, so the calculation belongs to the planning stage and returns a verdict about whether the study is worth running at all.

## Inverting the Power Function Gives the Sample Size, and for a Sharpe Ratio It Gives a Number of Years

Setting the power to a target and solving for $n$ is the standard design calculation. For a Sharpe ratio it has a feature that changes the character of the answer: the volatility cancels, so the requirement is not a number of observations but a span of calendar time, and sampling more finely does not help.

??? note "Proof that the sample size for power $1-\beta$ against a shift $\delta$ is $(z_{1-\alpha}+z_{1-\beta})^{2}\sigma^{2}/\delta^{2}$, and that substituting a Sharpe ratio cancels the volatility and leaves a span of calendar time"

    For a one-sided level-$\alpha$ test of $H_0\!:\mu=0$ using $\bar X$ with known $\sigma$, rejection occurs when $\bar X>z_{1-\alpha}\sigma/\sqrt n$. Under the alternative $\mu=\delta$, $\bar X\sim\mathcal{N}(\delta,\sigma^{2}/n)$, so the power is
    $$\beta(\delta)=\Phi\!\left(\frac{\delta\sqrt n}{\sigma}-z_{1-\alpha}\right).$$
    Setting $\beta(\delta)=1-\beta$ gives $\delta\sqrt n/\sigma-z_{1-\alpha}=z_{1-\beta}$, hence
    $$n=\frac{(z_{1-\alpha}+z_{1-\beta})^{2}\sigma^{2}}{\delta^{2}} .$$
    The proportion instance of this formula is used in [Bernoulli Distribution](../part-05-common-distributions/01-bernoulli-distribution.md); what follows is the Sharpe instance.

    Let returns have mean $\mu$ and volatility $\sigma$ per period, with $k$ periods per year, and write the annualized Sharpe as $S=\sqrt k\,\mu/\sigma$. Testing $S=0$ is testing $\mu=0$, so substitute $\delta=\mu=S\sigma/\sqrt k$ into the display:
    $$n=\frac{(z_{1-\alpha}+z_{1-\beta})^{2}\sigma^{2}}{S^{2}\sigma^{2}/k}=\frac{k\,(z_{1-\alpha}+z_{1-\beta})^{2}}{S^{2}},$$
    and $\sigma$ has cancelled. Dividing by $k$ converts $n$ observations into $T=n/k$ *years*:
    $$T=\frac{(z_{1-\alpha}+z_{1-\beta})^{2}}{S^{2}},$$
    a quantity in which the sampling frequency $k$ no longer appears. Using Lo's variance for an estimated Sharpe multiplies this by $(1+S^{2}/2k)$, which is the form the block below uses. The factor is worth stating carefully because it is routinely misquoted: Lo's $(1+S^{2}/2)$ is written for the *per-period* Sharpe $S_p$, and substituting $S=\sqrt k\,S_p$ gives $\operatorname{var}(\hat S)=(1+S^{2}/2k)/T$ for the annualized one. At $k=252$ that factor is $1.0002$ for $S=0.30$ and $1.0045$ for $S=1.50$, so the clean display above is exact to within half a percent at every Sharpe anyone trades — whereas applying the per-period factor to an annualized Sharpe inflates the required span by $4\%$ at $S=0.30$ and by $52\%$ at $S=1.50$.

    The load-bearing cancellation is that both the effect and the noise scale with $\sigma$, so the signal-to-noise ratio per unit of *time* is a property of the strategy alone. **You cannot buy statistical significance about a Sharpe ratio by sampling more often; the only currency is calendar, and no amount of money accelerates it.**

The consequence is best read as a table, since the numbers are the reason most strategy research cannot be conclusive:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12051)
za = stats.norm.isf(0.05)                          # one-sided 5%
k = 252                                            # Lo's factor is per-period, not annual

def years_needed(sr, power):
    return (za + stats.norm.isf(1 - power)) ** 2 * (1 + sr**2 / (2 * k)) / sr**2

print("  years of daily data needed to detect a TRUE annualized Sharpe, one-sided 5%")
print("     Sharpe   50% power   80% power   90% power   power at 24 years")
for sr in (0.20, 0.30, 0.50, 0.80, 1.00, 1.50):
    d = sr / np.sqrt((1 + sr**2 / (2 * k)) / 24)
    print(f"     {sr:6.2f}   {years_needed(sr, 0.50):9.1f}   {years_needed(sr, 0.80):9.1f}   "
          f"{years_needed(sr, 0.90):9.1f}   {stats.norm.sf(za - d):17.4f}")

sr, T = 0.30, years_needed(0.30, 0.80)
n = int(round(T * 252))
sd = 0.012
x = rng.normal(sr / np.sqrt(252) * sd, sd, (20_000, n))
t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
print(f"  simulated check: at {T:.1f} years the Sharpe-0.30 edge is found "
      f"{(t > za).mean():.4f} of the time")
# =>   years of daily data needed to detect a TRUE annualized Sharpe, one-sided 5%
#         Sharpe   50% power   80% power   90% power   power at 24 years
#           0.20        67.6       154.6       214.1              0.2530
#           0.30        30.1        68.7        95.2              0.4304
#           0.50        10.8        24.7        34.3              0.7893
#           0.80         4.2         9.7        13.4              0.9885
#           1.00         2.7         6.2         8.6              0.9994
#           1.50         1.2         2.8         3.8              1.0000
#      simulated check: at 68.7 years the Sharpe-0.30 edge is found 0.7989 of the time
```

The course's own strategy is the $0.30$ row, and it needs $68.7$ years of daily data for an $80\%$ chance of detection — nearly three times the twenty-four years the lesson actually has, and longer than the modern history of most instruments. Even a coin-flip chance of detection requires $30.1$ years. The simulated check confirms the arithmetic: at $68.7$ years the edge is found $79.89\%$ of the time. A Sharpe of $0.20$, which is a perfectly tradeable number in size, needs $154.6$ years for $80\%$ power and would be detected on a twenty-four-year record only $25.30\%$ of the time.

The other end of the table is where research is comfortable and it explains a bias in what gets published. A Sharpe of $1.00$ needs $6.2$ years and a Sharpe of $1.50$ needs $2.8$; both are detected essentially always on twenty-four years, at $0.9994$ and $1.0000$. So the strategies that can be *demonstrated* on available history are the ones with large Sharpes, which are exactly the ones least likely to be real and most likely to be artefacts of a search — a selection effect that [Part XV](../part-15-multiple-testing/index.md) takes up. The strategies whose Sharpes are plausible for a liquid market, between $0.2$ and $0.5$, sit in the range where no available history settles anything.

**A Sharpe ratio's error bar shrinks with calendar time and nothing else, so the sampling frequency a desk controls is irrelevant and the one thing that would help cannot be bought.**

## A Research Programme's Tests Can Be Audited for Power Before Any of Them Is Run

Because power needs only design quantities, published tests can be graded retrospectively using nothing but their own reported numbers. The course's weekday tests supply everything required: each cell's mean, its $t$-statistic and its sample size, from which the standard error follows as mean over $t$. The audit asks what those five tests could have detected:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12053)
zc = stats.norm.isf(0.025)
cells = [("Mon", 2.1, 0.55, 1201), ("Tue", 6.2, 1.85, 1315), ("Wed", 3.5, 1.04, 1315),
         ("Thu", 2.1, 0.62, 1292), ("Fri", 0.8, 0.24, 1287)]

def power(delta, se):
    return stats.norm.sf(zc - delta / se) + stats.norm.cdf(-zc - delta / se)

print("  power of the course's own weekday tests, from their published means and t-stats")
print("    day       n   SE (bp)   power vs 2 bp   power vs 5 bp   Mondays for 80% at 2 bp")
for day, mean, t, n in cells:
    se = mean / t                                  # the standard error the lesson implies
    need = (zc + stats.norm.isf(0.20)) ** 2 * (se * np.sqrt(n)) ** 2 / 2.0**2
    print(f"    {day}   {n:5d}   {se:7.3f}   {power(2.0, se):13.4f}   {power(5.0, se):13.4f}   "
          f"{need:23.0f}")

se_mon = 2.1 / 0.55
draws = rng.normal(2.0, se_mon, 400_000)
print(f"  simulated Monday check: a true 2 bp effect is detected "
      f"{(np.abs(draws) > zc * se_mon).mean():.4f} of the time")
print(f"  80% power on Mondays needs {(zc + stats.norm.isf(0.20))**2 * (se_mon * np.sqrt(1201))**2 / 4 / 52:.0f} "
      f"years of Mondays")
# =>   power of the course's own weekday tests, from their published means and t-stats
#        day       n   SE (bp)   power vs 2 bp   power vs 5 bp   Mondays for 80% at 2 bp
#        Mon    1201     3.818          0.0820          0.2582                     34356
#        Tue    1315     3.351          0.0917          0.3202                     28981
#        Wed    1315     3.365          0.0913          0.3179                     29224
#        Thu    1292     3.387          0.0908          0.3146                     29085
#        Fri    1287     3.333          0.0922          0.3230                     28060
#      simulated Monday check: a true 2 bp effect is detected 0.0819 of the time
#      80% power on Mondays needs 661 years of Mondays
```

Every cell has a standard error near $3.4$ basis points, so a true effect of $2$ bp per day — a real, tradeable weekday premium of about $5\%$ a year before costs — is detected between $8.20\%$ and $9.22\%$ of the time. The Monday test, the one carrying a fifty-year literature, had $8.20\%$ power against the effect it was hunting, confirmed by simulation at $0.0819$. Even a $5$ bp effect, which would be an enormous and immediately arbitraged anomaly, is detected only about $32\%$ of the time.

The final column converts the shortfall into what it would take to close it. Reaching $80\%$ power against a $2$ bp Monday effect requires $34{,}356$ Mondays, which is $661$ years of them. That number is not a criticism of the course's execution — the tests are correctly run on all the data that exists — but it settles what any conclusion from them can mean. Twenty-four years of weekday returns cannot distinguish a real $2$ bp premium from nothing, so the original literature's confident rejections of the null and the course's confident failure to reject are both drawn from the same underpowered instrument, and only the second is stated as such. The joint test's `p = 0.84` should be read in the same light: it is a correct summary of what the calendar shows, and it is not evidence that the calendar is flat.

**Every one of these numbers was computable in 1980, before any of the data used to argue about the Monday effect had been collected.**

## A Failure to Reject From an Underpowered Test Is Not Evidence of Absence and Carries No Information at All

The temptation after a non-rejection is to convert it into a finding, and there is a standard device for doing so: compute the power at the *observed* effect and report that the test "had adequate power", so the null result must mean something. The device is empty, and provably so.

??? note "Proof that observed power is a strictly decreasing function of the p-value alone, so it contains no information the p-value did not already carry"

    Take a two-sided test based on a statistic $T$ with null law $\mathcal{N}(0,1)$, and let $t$ be the observed value. The p-value is $p=2\Phi(-|t|)$, a strictly decreasing function of $|t|$. **Observed power** — also called post-hoc or retrospective power — is the power computed by substituting the observed effect for the unknown true effect, which for this test means evaluating the power function at the alternative $|t|$:
    $$\widehat{\text{pow}}=\Phi(|t|-z_{1-\alpha/2})+\Phi(-|t|-z_{1-\alpha/2}).$$
    This is a strictly increasing function of $|t|$. Composing with the strictly decreasing map $|t|\mapsto p$ gives $\widehat{\text{pow}}$ as a strictly decreasing function of $p$ alone: the two are in one-to-one correspondence, with no other quantity entering.

    Therefore observed power adds nothing. It is a monotone relabelling of the p-value, so "the p-value was large but the observed power was adequate" is self-contradictory — a large p-value *is* a low observed power, necessarily and by identity. The block below measures the correspondence directly and finds a rank correlation of exactly $-1$.

    The load-bearing substitution is using the observed effect as the alternative. Genuine power is computed at an effect chosen on external grounds *before* the data, which is why it can inform a design; substituting the estimate makes it a function of the data and destroys the property that made it useful. **Post-hoc power is the p-value wearing a costume, and the costume is the claim that the study was adequate.**

What a non-rejection genuinely licenses is much weaker and can be stated: at the observed standard error, effects larger than roughly $2.8$ standard errors would have been detected $80\%$ of the time, so those are disfavoured; everything smaller is untouched. For the Monday cell that means effects above about $10.7$ bp per day are disfavoured and everything below is simply unresolved — which includes every effect size anyone would trade. Reporting the interval of effects the study could *not* have seen is the honest form of a null result, and it is the same information the confidence interval carries under the duality of [The Hypothesis Testing Framework](01-hypothesis-testing-framework.md).

!!! note "Power, sensitivity, recall and detection rate are one conditional probability under four names, and 'the test is 80% powered' fixes an effect, a sample size and a level at once"
    All four name $\mathbf{P}_{\theta_1}(\varphi=1)$ at a specified $\theta_1$, and the machine-learning trio of recall, precision and accuracy separates them by what they condition on — recall conditions on the truth as power does, while precision conditions on the decision and therefore needs the prior of [Bayes' Rule](../part-02-probability-foundations/04-bayes-rule.md). A distinct collision is worth flagging inside this course: the *statistical* power of a test is unrelated to the **payoff ratio** and hit rate that [Performance Metrics and Reporting](../../part-05-backtesting-engine/04-performance-metrics-and-reporting.md) uses to describe a strategy's economics. A strategy can have a $34\%$ hit rate and a fine expectancy while the *test* of whether it has any edge has $9\%$ power; the two numbers describe the trade and the evidence about the trade, and they move independently.

## Conditioning on Significance Inflates the Estimate by a Factor That Grows as Power Falls

The second consequence of low power is subtler than the first and does more damage, because it corrupts results that *did* clear the bar. If only large estimates reach significance, then the estimates that reach significance are large — not through any bias in the estimator, which remains unbiased over all its draws, but because the reporting filter keeps a tail.

??? note "Proof that the expected estimate conditional on rejection exceeds the truth by a factor that diverges as the power falls to the level"

    Let $\hat\delta\sim\mathcal{N}(\delta,\sigma_{\hat\delta}^{2})$ and let the test reject when $|\hat\delta|>c$ with $c=z_{1-\alpha/2}\sigma_{\hat\delta}$. Ignoring the far tail on the wrong side, the reported estimates are draws from $\hat\delta$ conditioned on $\hat\delta>c$, and for a normal variable that conditional mean is
    $$\mathbb{E}[\hat\delta\mid\hat\delta>c]=\delta+\sigma_{\hat\delta}\,\frac{\phi(\lambda)}{1-\Phi(\lambda)},\qquad \lambda=\frac{c-\delta}{\sigma_{\hat\delta}},$$
    the inverse Mills ratio times the standard error. The exaggeration factor is that quantity divided by $\delta$.

    Now take the two limits. When power is high, $\delta\gg c$, so $\lambda\to-\infty$, the Mills term vanishes, and the conditional mean converges to $\delta$: filtering on significance costs nothing because almost everything passes. When power is low, $\delta\ll c$, so $\lambda\to c/\sigma_{\hat\delta}$, the Mills term tends to a positive constant of order $c$, and the conditional mean tends to a number determined by the *threshold* rather than the truth. The ratio to $\delta$ therefore diverges as $\delta\to0$: a smaller true effect does not produce smaller published estimates, it produces the same published estimates attached to a smaller truth.

    The same argument gives the **Type S** rate. Rejection in the wrong direction requires $\hat\delta<-c$, with probability $\Phi(-(c+\delta)/\sigma_{\hat\delta})$, and dividing by the total rejection probability gives the share of published findings with the wrong sign, which grows as power falls for the same reason.

    The load-bearing feature is that the threshold $c$ is fixed by the level and the standard error, not by the truth, so it acts as a floor on what gets reported. **An underpowered literature does not report weak effects weakly; it reports them strongly or not at all.**

The magnitudes are worth having. Below, a single pre-registered test — no selection over variants, no search, one hypothesis fixed in advance — is run on a genuine Sharpe of $0.30$ at four track-record lengths, and only the significant results are kept, exactly as a journal or an investment committee would:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(12057)
reps, sd, sr_true = 20_000, 0.012, 0.30
zc = stats.norm.isf(0.025)

print("  one pre-registered test of a TRUE Sharpe of 0.30, repeated 20,000 times")
print("    years   power   E[SR-hat | significant]   exaggeration   wrong sign | sig")
for years in (3, 5, 10, 24):
    n = years * 252
    x = rng.normal(sr_true / np.sqrt(252) * sd, sd, (reps, n))
    srhat = np.sqrt(252) * x.mean(1) / x.std(1, ddof=1)
    t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
    sig = np.abs(t) > zc
    kept = srhat[sig]
    print(f"    {years:5d}   {sig.mean():5.4f}   {kept.mean():23.4f}   "
          f"{kept.mean() / sr_true:13.2f}   {(kept < 0).mean():18.4f}")

n = 5 * 252
x = rng.normal(sr_true / np.sqrt(252) * sd, sd, (reps, n))
t = np.abs(x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n)))
p = 2 * stats.norm.sf(t)
obs_power = stats.norm.sf(zc - t) + stats.norm.cdf(-zc - t)
print(f"  observed power against the p-value it came from: Spearman "
      f"{stats.spearmanr(obs_power, p).statistic:+.4f}")
# =>   one pre-registered test of a TRUE Sharpe of 0.30, repeated 20,000 times
#        years   power   E[SR-hat | significant]   exaggeration   wrong sign | sig
#            3   0.0799                    1.1708            3.90               0.0807
#            5   0.1056                    1.0067            3.36               0.0369
#           10   0.1549                    0.7631            2.54               0.0132
#           24   0.3165                    0.5320            1.77               0.0009
#      observed power against the p-value it came from: Spearman -1.0000
```

The three-year row is the shape of a typical track record brought to an allocator. The test has $7.99\%$ power, so nine hundred and twenty of every thousand honest studies of this genuine edge find nothing. Among the eighty that do, the average reported Sharpe is $1.1708$ against a truth of $0.30$ — an exaggeration of $3.90$ times — and $8.07\%$ of them have the wrong sign, reporting a significant *negative* Sharpe for a strategy that genuinely makes money. Nothing here involves a search, a p-hacked variant, or a dishonest analyst. One hypothesis, fixed in advance, correctly tested, with only the significant results surviving to be discussed.

The exaggeration decays exactly as the proof requires. At five years the factor is $3.36$ and the wrong-sign rate $3.69\%$; at ten years, $2.54$ and $1.32\%$; at twenty-four years, with power finally up to $31.65\%$, the surviving estimates average $0.5320$ — still $1.77$ times the truth — and the sign error has all but vanished at $0.09\%$. Even on the longest record in the course, a filtered estimate of this strategy's Sharpe overstates it by three-quarters. The final line closes the loop on the previous section by measuring the identity directly: across twenty thousand studies the rank correlation between observed power and the p-value it came from is $-1.0000$, exactly, as it must be.

**A significance filter does not select the true effects from the false ones, it selects the large draws from the small ones, and the smaller the true effect the more thoroughly it does so.**

!!! warning "A 'no effect' conclusion from a nine-percent-powered test is a statement about the test, and the check costs one line before the data is loaded"
    Both failures on this page are invisible in output. An underpowered non-rejection looks exactly like a well-powered one — a p-value above the threshold, a confidence interval containing zero, a clean table — and an inflated significant estimate looks exactly like an accurate one, since the estimator is unbiased and the arithmetic is correct. Nothing downstream flags either. The only defence is arithmetic done first, and it is genuinely one line: given your sample size, your statistic's standard error and your level, the effect detectable at $80\%$ power is about $2.8$ standard errors. **The free diagnostic is to compute that number before loading the data and compare it against the smallest effect you would actually trade — if the detectable effect is larger, the study cannot answer your question no matter how it comes out, a non-rejection will mean nothing and a rejection will overstate the truth by the factor in the table above, and the honest move is to change the question rather than run the test.**

## Power Is the One Quantity a Report Could Have Contained Before the Data Arrived, and It Is the One Never Reported

This page established that power is a function on the alternative whose infimum is the size, so quoting it as a number fixes an undisclosed effect size; that inverting it for a Sharpe ratio cancels the volatility and returns a span of calendar time, so the course's $0.30$ edge needs $68.7$ years for $80\%$ power against the twenty-four it has, and a $0.20$ Sharpe needs $154.6$; that the course's five weekday tests had between $8.20\%$ and $9.22\%$ power against a tradeable $2$ bp effect, and closing that gap on Mondays would take $34{,}356$ of them, or $661$ years; that observed power is a strictly decreasing function of the p-value and therefore carries no information, with a measured rank correlation of $-1.0000$; and that filtering on significance inflates a genuine $0.30$ Sharpe to a reported $1.1708$ at three years, with $8.07\%$ of significant findings carrying the wrong sign, decaying to $1.77$ times and $0.09\%$ at twenty-four years.

The two failures are the same fact seen from either side of the threshold. Below it, a non-rejection is uninformative because the test could not have detected anything worth having; above it, a rejection is inflated because only the large draws got through. Both are consequences of the detectable effect being larger than the interesting effect, and both are fixed by the same quantity — a power calculation that costs one line and is almost never performed, because performing it early would frequently end the project and performing it late cannot rescue one.

The rest of this part builds specific tests, and the first question to ask of each is the one this page supplies: against which alternatives can it see anything. That question has a sharpest possible answer when both hypotheses are fully specified, because then a single statistic is provably optimal and every other test is measurably worse. That is [Likelihood Ratio Tests](06-likelihood-ratio-tests.md).

**Power is the only number in a study that could have been known before it started, and it is the one that would most often have stopped it.**
