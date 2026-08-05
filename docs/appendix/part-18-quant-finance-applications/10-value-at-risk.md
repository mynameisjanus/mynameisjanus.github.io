# Value at Risk

Value at Risk is a quantile, and everything awkward about it follows from that one fact rather than from anything financial. A quantile's estimator is precise in proportion to the density at the quantile, which is smallest exactly where the question is asked, so on $t(4)$ returns the sample quantile at $99\%$ has a standard deviation of $0.4624$ against a true value of $2.6495$ when built on a year of data — determined, at that sample size, by two observations. A quantile is not additive, and two independent positions can have a combined VaR of $99.00$ against a sum of individual VaRs of $-2.00$, so a risk measure meant to reward diversification penalizes it by $101.00$. And a quantile is only as good as the shape assumed around it: a fitted normal on those same returns converges to $2.3244$ rather than $2.6495$ — a bias of $-0.3251$ that does not shrink with data while its standard deviation falls to $0.0750$, which is a number becoming more confident about being wrong. The one piece of good news is that a quantile *is* elicitable, so VaR forecasts can be scored on every observation rather than on the breaches alone; the backtest that everyone runs instead cannot detect a VaR ten percent too small on a year of data more than $18.1\%$ of the time.

This page covers the sample quantile's asymptotic distribution and the density that governs it, the three standard estimators and what each assumes, elicitability and the scoring rule that follows, the power of a breach-counting backtest, and the subadditivity that holds on elliptical laws and fails in general. It does not decompose the error of a simulated risk number into parameter and sampling components, which is [Portfolio Risk Simulation](09-portfolio-risk-simulation.md); it does not establish coherence, prove the axioms, or treat the measure that repairs the failure below, which is [Expected Shortfall](11-expected-shortfall.md); it does not exhibit two joint laws with identical margins and different tails, which is [Marginal Distributions](../part-03-random-variables/06-marginal-distributions.md); it does not catalogue moments or explain why a Cornish–Fisher adjustment is undefined on a law with $\nu=2.65$, which is [Higher-Order Moments](../part-04-expectation-and-moments/03-higher-order-moments.md); it does not fit a tail model or extrapolate beyond the sample, which is [Extreme Value Theory](13-extreme-value-theory.md); it does not distinguish decision-theoretic risk from financial risk, which is [Point Estimation](../part-11-parameter-estimation/01-point-estimation.md); it runs no VaR model against real returns and reports no real breach count, which is [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md); and it never quotes a quantile without the number of observations that determined it.

The trading stake is a backtest in a course lesson that fails three of its four models and passes one for the wrong reason. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) prints `parametric-normal  VaR 0.825%  breaches 119 (2.50x)  Kupiec p 0.00e+00` against `historical  VaR 1.121%  breaches  48 (1.01x)  Kupiec p 9.51e-01  indep p 0.0013` on $4{,}758$ days with $47.6$ breaches expected. The parametric model's failure is section 2's bias, arriving on real data at a $2.50\times$ breach rate; the historical model's Kupiec pass alongside an independence failure is section 3's subject, which is that counting breaches is a weak test of a strong claim. Section 4 is the property that lesson defers by linking the word "coherent" to the next page.

## A Quantile's Precision Is Set by the Density Where the Data Is Thinnest

Every difficulty in estimating a VaR is contained in one asymptotic statement, and the statement's structure explains why the problem gets worse in exactly the direction a risk manager cares about.

??? note "Proof that the sample quantile is asymptotically normal with variance $p(1-p)/(nf(q_p)^{2})$, and that the effective number of observations behind it is $np$"

    Let $X_1,\dots,X_n$ be i.i.d. with continuous density $f$ positive at the $p$-quantile $q_p$, and let $\hat q_p$ be the sample quantile. The empirical distribution function $F_n$ satisfies $\sqrt n(F_n(x)-F(x))\to N(0,F(x)(1-F(x)))$ pointwise by the central limit theorem applied to the indicator variables $\mathbf 1\{X_i\le x\}$, which are Bernoulli with success probability $F(x)$.

    Inverting is one application of the delta method. Since $q_p=F^{-1}(p)$ and $(F^{-1})'(p)=1/f(q_p)$,
    $$\sqrt n\left(\hat q_p-q_p\right)\;\xrightarrow{d}\;N\!\left(0,\;\frac{p(1-p)}{f(q_p)^{2}}\right).$$
    The numerator is the variance of a Bernoulli count and is largest at $p=1/2$; the denominator is the squared density and collapses in the tail. For a standard normal at $p=0.01$ the density is $0.0267$, so the standard deviation of the estimate is about $3.7/\sqrt n$ — nearly four times what the same sample delivers for the mean.

    The count interpretation is the one to carry. The estimate is determined by the order statistics adjacent to rank $\lceil np\rceil$, so the *effective* sample behind a $99\%$ VaR on $n$ days is $np$ observations: two on a year of data, ten on four years, fifty on twenty years. Everything else in the sample contributes only by establishing where those few sit.

    **The load-bearing quantity is $f(q_p)$, which is a property of the unknown distribution rather than of the sample. So the precision of a quantile estimate cannot be read off the data without assuming the very shape the estimate was supposed to avoid assuming — which is why the parametric and non-parametric estimators of section 2 have incomparable error structures rather than merely different ones.**

## Three Estimators, Three Different Assumptions, and the Confident One Is Wrong by Twelve Percent

The three standard routes to a VaR are usually described as a convenience ranking — parametric is fast, historical is honest, fitted is fussy. They are better described by what each is willing to be wrong about.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18101)
NU, P, REPS = 4.0, 0.01, 20_000
SCALE = np.sqrt(NU / (NU - 2))                          # so the true law has unit variance
TRUE = -stats.t.ppf(P, NU) / SCALE

print(f"  daily returns are t({NU:.0f}) scaled to unit variance, so the true {1 - P:.0%} VaR is"
      f" {TRUE:.4f} standard deviations. Three estimators on n days: a fitted normal, the sample"
      f" quantile, and a fitted t. {REPS:,} replications")
print("     n      normal: mean    bias      sd   |   historical: mean    bias      sd"
      "   asymptotic sd   observations below the quantile   |   fitted t: mean    bias      sd")
for n in (250, 1_000, 5_000):
    x = rng.standard_t(NU, (REPS, n)) / SCALE
    norm = -stats.norm.ppf(P) * x.std(axis=1, ddof=1)
    hist = -np.quantile(x, P, axis=1)
    fit = np.empty(REPS)
    for r in range(min(REPS, 2_000)):                   # fitting is the slow one
        df, loc, sc = stats.t.fit(x[r], floc=0)
        fit[r] = -stats.t.ppf(P, df, loc=0, scale=sc)
    fit = fit[:2_000]
    f = stats.t.pdf(stats.t.ppf(P, NU), NU) * SCALE     # density of the unit-variance law at q
    asym = np.sqrt(P * (1 - P) / n) / f
    print(f"    {n:5d}   {norm.mean():12.4f}  {norm.mean() - TRUE:+6.4f}  {norm.std():6.4f}"
          f"   |   {hist.mean():16.4f}  {hist.mean() - TRUE:+6.4f}  {hist.std():6.4f}"
          f"   {asym:13.4f}   {n * P:31.0f}"
          f"   |   {fit.mean():14.4f}  {fit.mean() - TRUE:+6.4f}  {fit.std():6.4f}")
# =>   daily returns are t(4) scaled to unit variance, so the true 99% VaR is 2.6495 standard deviations. Three estimators on n days: a fitted normal, the sample quantile, and a fitted t. 20,000 replications
#         n      normal: mean    bias      sd   |   historical: mean    bias      sd   asymptotic sd   observations below the quantile   |   fitted t: mean    bias      sd
#          250         2.3111  -0.3384  0.2682   |             2.5414  -0.1081  0.4624          0.5125                                 2   |           2.6441  -0.0053  0.2944
#         1000         2.3220  -0.3275  0.1441   |             2.6193  -0.0302  0.2539          0.2563                                10   |           2.6435  -0.0060  0.1414
#         5000         2.3244  -0.3251  0.0750   |             2.6428  -0.0066  0.1153          0.1146                                50   |           2.6477  -0.0017  0.0635
```

The fitted normal is the dangerous one and its danger is visible only across the rows. Its bias is $-0.3384$, $-0.3275$ and $-0.3251$ — flat, because it is converging correctly to the $99\%$ quantile of the wrong distribution — while its standard deviation falls from $0.2682$ to $0.0750$. At five thousand days it reports $2.3244$ against a truth of $2.6495$ with an error bar that would exclude the truth: a $12.3\%$ understatement delivered with four times the precision of the estimator that is right. This is the mechanism behind the published `breaches 119 (2.50x)`, and it is not a small-sample problem, so no amount of history repairs it.

The historical estimator is unbiased and expensive. Its bias falls from $-0.1081$ to $-0.0066$ while its standard deviation, $0.4624$, $0.2539$ and $0.1153$, tracks the asymptotic prediction $\sqrt{p(1-p)/n}/f(q_p)$ at $0.5125$, $0.2563$ and $0.1146$ — the agreement failing only at $n=250$, where the asymptotics are being asked to describe an estimate resting on two observations. The last column is the reason: two, ten and fifty observations lie below the quantile at the three sample sizes, and the estimate is a statement about those.

The fitted $t$ is correctly specified here and wins on both counts, unbiased at $-0.0017$ and with a standard deviation of $0.0635$ against the historical estimator's $0.1153$. **A parametric assumption is worth roughly a doubling of the sample when it is right and is unfixable when it is wrong, and nothing in the estimate distinguishes the two cases** — which is the same trade [Bayesian Signal Updating](07-bayesian-signal-updating.md) priced for a likelihood, arriving here as a choice of tail.

## The Quantile Is Elicitable, Which Nobody Uses, and the Test Everybody Runs Counts Breaches Instead

There are two ways to check a VaR model, and the standard one throws away almost all of the data before it starts.

??? note "Proof that the pinball loss elicits the quantile, so competing VaR forecasts can be ranked on every observation, while a breach count uses only the exceedances"

    Define the **pinball loss** $\rho_p(u)=u\left(p-\mathbf 1\{u<0\}\right)$, and consider the expected loss of a forecast $q$ against the outcome $X$:
    $$S(q)=\mathbb{E}\left[\rho_p(X-q)\right]=p\,\mathbb{E}\left[(X-q)^{+}\right]+(1-p)\,\mathbb{E}\left[(q-X)^{+}\right].$$
    Differentiating with respect to $q$ gives $S'(q)=-p\,\mathbf{P}(X>q)+(1-p)\mathbf{P}(X\le q)=F(q)-p$, which is negative below $q_p$, positive above, and zero at $q_p$. So $S$ is minimized exactly at the true quantile, and $\rho_p$ is a **strictly consistent scoring function** for it: VaR is *elicitable*.

    The practical content is that elicitability licenses comparison. Two competing VaR forecasts can be ranked by their average pinball loss over the whole sample, and the ranking is not an artefact of the loss chosen, because every strictly consistent scoring function for the quantile agrees in expectation about which forecast is better. Every day contributes, whether or not it was a breach.

    A breach count discards that. It maps each day to a single bit — was the loss worse than the forecast — and tests whether the bits sum to $np$. The Kupiec statistic is the likelihood ratio for a Bernoulli rate, so its information content is that of $n$ Bernoulli trials with success probability $p$: at $p=0.01$ the Fisher information per observation is $1/(p(1-p))\approx101$ for the *rate*, but the rate is not the forecast, and the map from a misstated quantile to a misstated rate goes through the density again.

    **The load-bearing distinction is between validating a number and validating a probability. A breach count can reject the claim "this level is exceeded one percent of the time" and cannot rank two models that both fail it, nor detect a level that is wrong by less than the binomial noise in $np$ counts — which is section 3's measurement.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18103)
P, REPS, ALPHA = 0.01, 40_000, 0.05
Z = -stats.norm.ppf(P)


def kupiec_rejects(x, n, p=P):
    """Unconditional-coverage likelihood ratio against chi-square with one degree of freedom."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = x / n
        ll0 = (n - x) * np.log1p(-p) + x * np.log(p)
        ll1 = np.where(x == 0, n * np.log1p(-pi),
                       np.where(x == n, n * np.log(np.clip(pi, 1e-300, None)),
                                (n - x) * np.log1p(-pi) + x * np.log(np.clip(pi, 1e-300, None))))
    return -2 * (ll0 - ll1) > stats.chi2.ppf(1 - ALPHA, 1)


print(f"  a {1 - P:.0%} VaR reported as a fraction of the truth, so a fraction below one is a risk"
      f" number that is too small. The Kupiec test counts breaches over n days and asks whether"
      f" the rate is {P:.0%}. Rejection rates at the {ALPHA:.0%} level, {REPS:,} replications")
print("     reported / true VaR   true breach rate   " + "".join(
    f"n={n}: expected breaches   power   " for n in (250, 1_000, 2_500)))
for frac in (1.00, 0.90, 0.80, 0.70, 0.60, 0.50):
    rate = stats.norm.sf(frac * Z)
    cells = ""
    for n in (250, 1_000, 2_500):
        x = rng.binomial(n, rate, REPS)
        cells += f"{n * rate:20.1f}   {kupiec_rejects(x, n).mean():5.3f}   "
    print(f"    {frac:19.2f}   {rate:16.4%}   " + cells)
# =>   a 99% VaR reported as a fraction of the truth, so a fraction below one is a risk number that is too small. The Kupiec test counts breaches over n days and asks whether the rate is 1%. Rejection rates at the 5% level, 40,000 replications
#         reported / true VaR   true breach rate   n=250: expected breaches   power   n=1000: expected breaches   power   n=2500: expected breaches   power   
#                       1.00            1.0000%                    2.5   0.096                   10.0   0.053                   25.0   0.044   
#                       0.90            1.8143%                    4.5   0.181                   18.1   0.638                   45.4   0.935   
#                       0.80            3.1367%                    7.8   0.670                   31.4   0.999                   78.4   1.000   
#                       0.70            5.1715%                   12.9   0.977                   51.7   1.000                  129.3   1.000   
#                       0.60            8.1386%                   20.3   1.000                   81.4   1.000                  203.5   1.000   
#                       0.50           12.2379%                   30.6   1.000                  122.4   1.000                  305.9   1.000   
```

The first row is the test's size and it is not $0.05$. On a year of data the Kupiec test rejects a *correct* model $9.6\%$ of the time, nearly twice its nominal rate, because a chi-square approximation is being applied to a statistic built from about two and a half expected breaches. It settles to $0.053$ at four years and $0.044$ at ten.

The rest of the table is its power, and on a year of data there is almost none. A VaR ten percent too small — which more than doubles the true breach rate, from $1.00\%$ to $1.81\%$ — is detected $18.1\%$ of the time. Twenty percent too small is detected $67.0\%$ of the time. A model must be understating risk by thirty percent, tripling the breach rate to $5.17\%$, before a year of backtesting catches it reliably. Four years of data repairs most of this, reaching $0.638$ power against the ten-percent error, which is why the published test on $4{,}758$ days was able to reject a $2.50\times$ breach rate with a $p$-value that underflowed. **The test is diagnostic on a decade of history and nearly blind on a year, and the model it is checking was fitted on a year.**

!!! note "Parametric, historical, Monte Carlo and fitted-tail are four ways to produce a VaR, and they differ in what they are willing to be wrong about rather than in what they compute"
    **Parametric** assumes a shape and estimates its scale, so it converges fast to the quantile of the assumed law — the right answer to the wrong question when the shape is wrong, with a bias that does not shrink. **Historical** assumes only that the past sample is representative, so it is asymptotically unbiased and rests on $np$ observations, which is two at a year and fifty at twenty years. **Monte Carlo** assumes whatever it samples from, which makes it parametric wearing a simulation's clothing; [Portfolio Risk Simulation](09-portfolio-risk-simulation.md) shows its reported error bar measures the sampling stage rather than the assumption. **Fitted-tail** assumes a shape for the tail alone and estimates it from the exceedances, which is the only one of the four whose assumption is checkable against the region it is used in, and is [Extreme Value Theory](13-extreme-value-theory.md). The first and third are the ones that produce narrow error bars, and narrowness is a property of the assumption rather than of the evidence.

## Two Independent Positions Whose Combined Risk Exceeds the Sum of Their Risks

A risk measure is expected to reward diversification, and a quantile has no reason to. Where it fails, it fails by a margin that makes the measure actively perverse.

??? note "Proof that VaR is subadditive on elliptical laws and not in general, with the failure driven by probability mass just outside the confidence level"

    For an elliptical law, every linear combination is again elliptical of the same type, so $\mathrm{VaR}_p(w^\top X)=-w^\top\mu+z_p\sqrt{w^\top\Sigma w}$ for a constant $z_p$ determined by the generator alone. The map $w\mapsto\sqrt{w^\top\Sigma w}$ is a norm, hence satisfies the triangle inequality, and $-w^\top\mu$ is linear, so for $z_p\ge0$
    $$\mathrm{VaR}_p(X_1+X_2)\le\mathrm{VaR}_p(X_1)+\mathrm{VaR}_p(X_2).$$
    Subadditivity therefore holds for the Gaussian, the elliptical $t$, and every other member of the family, at every correlation — which is why the property is so often assumed to be general. It is a consequence of the model, not of the measure.

    Outside that family it can fail, and the construction shows the mechanism. Take a position losing $L$ with probability $\delta$ and gaining a small amount otherwise, with $\delta$ slightly *below* $1-p$. Then $\mathrm{VaR}_p$ of one position sits in the profitable region: the loss is real but too rare to reach the quantile, so the measure reports a gain. Two independent copies have $\mathbf{P}(\text{at least one defaults})=1-(1-\delta)^{2}\approx2\delta$, which now *exceeds* $1-p$, so the combined quantile lands on the loss and $\mathrm{VaR}_p(X_1+X_2)\approx L$ against $\mathrm{VaR}_p(X_1)+\mathrm{VaR}_p(X_2)<0$.

    **The load-bearing feature is that a quantile is a threshold and not an average, so it is blind to everything beyond it and discontinuous in what crosses it. Diversification moves probability mass across the threshold without changing the total loss, and a measure that reads only the threshold's location records that as an increase in risk.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18105)
P, DEFAULT, LOSS, REPS = 0.99, 0.008, 100.0, 400_000


def var_of(losses, p=P):
    return np.quantile(losses, p)


print(f"  two positions with independent {DEFAULT:.1%} default risk, each paying +1 when it"
      f" survives and losing {LOSS:.0f} when it does not. Because {DEFAULT:.1%} sits just below"
      f" 1 - {P:.0%}, the {P:.0%} VaR of one position misses its own default entirely, and the"
      f" VaR of the pair does not. {REPS:,} draws")
print("     positions   P(no default)   VaR of the sum   sum of the VaRs   subadditive?"
      "   diversification 'penalty'")
single = np.where(rng.random(REPS) < DEFAULT, LOSS, -1.0)
v1 = var_of(single)
for k in (1, 2, 3, 5):
    tot = sum(np.where(rng.random(REPS) < DEFAULT, LOSS, -1.0) for _ in range(k))
    v = var_of(tot)
    print(f"    {k:9d}   {(1 - DEFAULT) ** k:13.4f}   {v:14.2f}   {k * v1:16.2f}"
          f"   {str(v <= k * v1):>12}   {v - k * v1:+25.2f}")

print("\n     the elliptical case, where subadditivity is guaranteed: two Gaussian sleeves")
print("     correlation   VaR of the sum   sum of the VaRs   subadditive?   diversification benefit")
for rho in (-0.5, 0.0, 0.5, 0.99):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], REPS)
    v_sum, v_parts = var_of(z.sum(axis=1)), var_of(z[:, 0]) + var_of(z[:, 1])
    print(f"    {rho:11.2f}   {v_sum:14.4f}   {v_parts:16.4f}   {str(v_sum <= v_parts):>12}"
          f"   {v_parts - v_sum:23.4f}")
# =>   two positions with independent 0.8% default risk, each paying +1 when it survives and losing 100 when it does not. Because 0.8% sits just below 1 - 99%, the 99% VaR of one position misses its own default entirely, and the VaR of the pair does not. 400,000 draws
#         positions   P(no default)   VaR of the sum   sum of the VaRs   subadditive?   diversification 'penalty'
#                1          0.9920            -1.00              -1.00           True                       +0.00
#                2          0.9841            99.00              -2.00          False                     +101.00
#                3          0.9762            98.00              -3.00          False                     +101.00
#                5          0.9606            96.00              -5.00          False                     +101.00
#
#         the elliptical case, where subadditivity is guaranteed: two Gaussian sleeves
#         correlation   VaR of the sum   sum of the VaRs   subadditive?   diversification benefit
#              -0.50           2.3238             4.6579           True                    2.3341
#               0.00           3.3064             4.6576           True                    1.3512
#               0.50           4.0236             4.6561           True                    0.6325
#               0.99           4.6431             4.6524           True                    0.0094
```

One position has a $99\%$ VaR of $-1.00$: a *gain*, because the $0.8\%$ chance of losing a hundred does not reach the $1\%$ quantile. Two of them have a VaR of $99.00$, against a sum of parts of $-2.00$ — the measure records adding a second independent position as increasing risk by $101.00$. Three and five positions give $98.00$ and $96.00$, so the perversity is not a knife-edge: every book in the table is penalized, and the penalty barely moves as genuine diversification is added.

The second panel is the exception and it is total. Two Gaussian sleeves satisfy subadditivity at every correlation tested, with the diversification benefit falling smoothly from $2.3341$ at $\rho=-0.5$ to $0.0094$ at $\rho=0.99$, where the two positions are nearly the same trade and there is nothing left to diversify. Every intuition about VaR rewarding diversification is correct inside this panel and has no support outside it. **A risk limit written on VaR is a constraint that behaves sensibly on elliptical books and rewards concentration on books with default-like risk, which is precisely the class of book that the limit exists to control.**

## Every Repair Is a Different Measure, a Longer Backtest, or a Scoring Rule Already Available

The three failures separate by what they need. Section 2's parametric bias is repaired by fitting the region being used rather than the whole distribution, which is [Extreme Value Theory](13-extreme-value-theory.md); no quantity of data fixes a misspecified shape, and the historical estimator's honesty is bought at roughly double the standard error. Section 3's blindness is repaired two ways, and the cheaper one is already proved: rank competing models by average pinball loss over every observation rather than by breach counts over $np$ of them. Elicitability is what licenses that, it is a property VaR has and the next page's measure does not, and it is almost never used.

Section 4's failure is not repairable within the measure. Subadditivity is either a property or it is not, and VaR does not have it outside the elliptical family. The measure that does is [Expected Shortfall](11-expected-shortfall.md), and the reason a desk cannot simply switch is the subject of that page: it gains coherence and loses exactly the elicitability that section 3's repair depends on.

!!! warning "A VaR is reported to four significant figures and rests on a number of observations that is never printed beside it"
    A $99\%$ VaR on a year of daily data is determined by the second-worst day. The number is reported as $1.121\%$; the sample behind it is two observations, and the standard deviation of the estimate is $0.4624$ against a level of $2.6495$ — an eighteen percent relative error that no formatting convention discloses. **The free diagnostic is $np$, the count of observations at or below the reported quantile, printed next to it: two at a year, ten at four years, fifty at twenty years, and it is the whole of the evidence.** It costs one line, requires nothing but the sample already in memory, and immediately distinguishes a historical VaR from a parametric one, whose count is the entire sample precisely because it has assumed the shape. Where that count is in single digits the honest report is an interval rather than a level, and the interval is wide enough to change the decision.

## A Threshold, and the Two Things a Threshold Cannot Do

This page established that the sample quantile is asymptotically normal with variance $p(1-p)/(nf(q_p)^{2})$, so its precision collapses in the tail and its effective sample is $np$ — two observations at a year, fifty at twenty years — with the asymptotic standard deviations of $0.5125$, $0.2563$ and $0.1146$ matching measurements of $0.4624$, $0.2539$ and $0.1153$; that on $t(4)$ returns a fitted normal converges to $2.3244$ against a truth of $2.6495$ with a flat bias of $-0.3251$ and a shrinking standard deviation of $0.0750$, while the historical estimator is unbiased and twice as noisy and a correctly fitted $t$ beats both at $-0.0017$ and $0.0635$; that VaR is elicitable under the pinball loss and therefore rankable on every observation, while the breach-counting test everyone runs instead has size $0.096$ rather than $0.05$ on a year of data and power $0.181$ against a VaR ten percent too small, needing four years to reach $0.638$; and that VaR is subadditive on every elliptical law at every correlation, with benefits of $2.3341$ down to $0.0094$, and fails outside that family by $101.00$ on two independent positions whose individual VaRs are gains.

The relationship to the previous page is that both found a risk number's advertised precision to be a statement about something other than the risk. There, a simulation reported the error from its random draws and concealed the error from its covariance; here, an estimator reports a standard error computed under an assumed shape and conceals that the shape is what it got wrong. The common structure is a reported uncertainty conditional on the part of the model that was never in doubt. What separates this page is that its defect is not statistical at all: subadditivity fails for a correctly estimated VaR on a perfectly known distribution, so it is a property of the functional rather than of any inference about it. Repairing that means changing what is computed, not how well, and the measure that does it — together with what it costs — is the next page.

**A quantile answers where the threshold is and refuses every question about what lies beyond it, including whether two of them may be added.**
