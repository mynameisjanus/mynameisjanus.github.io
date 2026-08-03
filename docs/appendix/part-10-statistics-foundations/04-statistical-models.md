# Statistical Models

Everything so far treated the population as an anonymous $F$ about which nothing was assumed and, correspondingly, about which very little could be said. A **statistical model** is the act of narrowing that down: naming a set of candidate distributions and agreeing that the truth is one of them. The narrowing is what buys rates, standard errors, forecasts and every other quantity the rest of this appendix computes, and it is also the only point in the whole enterprise where a claim about the world is asserted rather than derived. Two things go wrong there and both go wrong silently — the set may not contain the truth, and the map from parameters to distributions may not be one-to-one, in which case the number the optimizer returned was chosen by the optimizer rather than by the data.

This page covers a statistical model as an indexed family of candidate laws on the space the data actually lives in, the parametric–nonparametric–semiparametric split as a statement about the dimension of the index set and the rate it buys, stationarity as the assumption that turns one path into a sample together with the four-cell grid two tests of it produce, identifiability as a property of the model rather than of the data or the algorithm, and misspecification as a redefinition of the estimand rather than a failure of the estimator. It estimates no parameter of any model it names, which is [Part XI](../part-11-parameter-estimation/index.md); it tests no hypothesis and computes no $p$-value, which is [Part XII](../part-12-hypothesis-testing/index.md); it selects between no two models and computes no information criterion, which is [Part XIV](../part-14-model-selection/index.md); it fits no regression, which is [Part XIII](../part-13-regression/index.md); it puts no prior on the index set, which is [Part XVI](../part-16-bayesian-statistics/index.md); it runs no expectation-maximization iteration, which is [Part XVII](../part-17-statistical-computing/index.md); it constructs none of the processes it models, which is [Part VIII](../part-08-stochastic-processes/index.md); and it fits nothing to real data.

The trading stake is a table the course prints and then immediately distrusts. [Time Series Analysis](../../part-03-statistics/03-time-series.md) runs an augmented Dickey–Fuller test and a KPSS test on the same two series, pinning `log price  ADF    0.92 (p=0.993)   KPSS  11.41 (p=0.01)` against `returns    ADF  -19.99 (p=0.000)   KPSS   0.39 (p=0.08)`, and then says exactly what the agreement is worth: "Both series land in the two consistent cells of the grid — prices are integrated, returns are stationary — but the grid has four cells and the other two occur constantly in practice." That lesson names this page as its formal background. The third section builds the other two cells out of processes whose truth is known in advance, and finds that a genuinely stationary series with a hundred-and-thirty-eight-day half-life puts both tests into rejection on every single sample.

## A Model Is a Set of Candidate Laws, and Choosing the Set Is the Modelling

A **statistical model** is a family $\mathcal{P}=\{P_\theta:\theta\in\Theta\}$ of probability laws on the sample space $\mathcal{X}$ where the data lives. Three things in that sentence do work. The family is a *set*, so the model is not one distribution but a menu. The index $\theta$ is a label, and the correspondence between labels and menu items is a separate question taken up in the fourth section. And $\mathcal{X}$ is the space the data actually occupies, which for a time series is not what people usually picture.

That last point is what makes this page general enough to be the background the course asks for. An ARMA or GARCH model is not a law on $\mathbb{R}$ describing one return; it is a law on a space of *paths*, and the observed history is one point in that space. This is the formal version of the first page's claim that a return series is a sample of size one: "$n=6{,}400$ observations" and "one draw" are both true, and they are statements about different things — the number of coordinates of the single point, and the number of points.

Two definitions follow. The model is **well specified** if the true law $P_0$ is a member of $\mathcal{P}$, and **misspecified** otherwise. Every rate, standard error and interval in the eight parts after this one is derived under the first assumption and computed under the second, which is the subject of the fifth section. What is worth stating now is that the choice of $\mathcal{P}$ is not itself estimated from anything: it is asserted before the data is consulted, and no amount of subsequent care compensates for asserting it wrongly.

## Parametric, Nonparametric and Semiparametric Name How Much Structure Was Assumed

The three labels describe the dimension of $\Theta$, and each buys a different thing.

A **parametric** model has a finite-dimensional index. A GARCH(1,1) with normal innovations is four numbers; an ARMA(1,1) is three. The payoff is the $\sqrt n$ rate — the error of a well-behaved estimator shrinks like $n^{-1/2}$ regardless of how complicated the underlying law is — and the price is that the payoff is worthless if the truth is outside the family.

A **nonparametric** model leaves $\Theta$ infinite-dimensional. The empirical distribution function assumes nothing about shape; a kernel density estimate assumes only smoothness. The payoff is that the assumption is nearly always true, and the price is a slower rate and, usually, a tuning parameter. That last part deserves emphasis, because "nonparametric" is routinely read as "assumption-free": a bandwidth is a choice, it is not estimated from the data in any assumption-free way, and choosing it is modelling in exactly the sense of the previous section.

A **semiparametric** model splits the index into a finite-dimensional part that is the target and an infinite-dimensional nuisance left unrestricted — a cointegrating vector with unconstrained short-run dynamics, or a GARCH whose innovation distribution is not named. The payoff is $\sqrt n$ on the part you care about without asserting a shape for the part you do not, and the price is that the nuisance must be handled rather than ignored. Most honest descriptions of a trading model land here, and most published descriptions claim the parametric version.

## Stationarity Is What Makes One Path a Sample, and Two Tests of It Produce Four Answers

A model on a path space is useless unless something ties the coordinates together, and that something is **stationarity**: the requirement that the joint law be unchanged by a shift in time. It is what licenses pooling observations from different dates into one estimate, and the first page showed it is not by itself enough to make the estimate mean anything.

Testing it is where the practical trouble starts, because the two standard tests place the burden of proof on opposite sides. The augmented Dickey–Fuller test takes a unit root as its null and rejects toward stationarity; the KPSS test takes stationarity as its null and rejects toward a unit root. Two tests with two possible outcomes give four cells, and only two of them are verdicts.

??? note "Proof that a stationary AR(1) has a convergent causal representation and the unit-root case has none, which is what the two tests are looking at from opposite sides"
    Let $y_t=\phi y_{t-1}+\varepsilon_t$ with $\varepsilon_t$ independent, mean zero, variance $\sigma^2$. Iterating backwards $k$ times,

    $$y_t=\phi^{k}y_{t-k}+\sum_{j=0}^{k-1}\phi^{j}\varepsilon_{t-j}.$$

    If $|\phi|<1$ the leading term vanishes as $k\to\infty$ and the sum converges in mean square, giving the causal representation $y_t=\sum_{j\ge0}\phi^j\varepsilon_{t-j}$ with $\mathrm{var}(y_t)=\sigma^2/(1-\phi^2)$, a finite constant free of $t$: the process is stationary and shocks decay geometrically with half-life $\log(1/2)/\log\phi$. If $\phi=1$ the representation fails — $y_t=y_0+\sum_{j=1}^{t}\varepsilon_j$ has $\mathrm{var}(y_t)=t\sigma^2$, which depends on $t$ and diverges, so no stationary solution exists and every shock is permanent.

    The two hypotheses are therefore $|\phi|<1$ against $\phi=1$, and the tests are not complements of one another: ADF's null is the single point $\phi=1$ and KPSS's null is the whole open interval. A process at $\phi=0.995$ satisfies the definition of stationarity exactly and behaves, over any sample shorter than several half-lives, like one that does not.

    The load-bearing distinction is between a property of the law and a property of the sample. Stationarity is a statement about $\phi$ that is true or false with no reference to $n$; what a test measures is whether $n$ observations suffice to distinguish $\phi$ from one, which depends on $n$ and on the half-life jointly. **Two tests with nulls on opposite sides can both reject and both fail to reject, and neither outcome is a contradiction in the data — it is the honest report that the sample does not separate the hypotheses.**

```python
import numpy as np

rng = np.random.default_rng(10041)
reps, adf_c, kpss_c = 400, -2.86, 0.463                        # the usual 5% critical values


def adf_t(y):                                                  # constant, one lag, t on y_{t-1}
    dy = np.diff(y)
    X = np.column_stack([np.ones(dy.size - 1), y[1:-1], dy[:-1]])
    b, *_ = np.linalg.lstsq(X, dy[1:], rcond=None)
    e = dy[1:] - X @ b
    s2 = e @ e / (X.shape[0] - X.shape[1])
    return b[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])


def kpss_stat(y):                                              # level-stationary, Bartlett window
    e, n = y - y.mean(), y.size
    lag = int(4 * (n / 100) ** 0.25)
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return (np.cumsum(e) ** 2).sum() / (n ** 2 * s)


def ar1(n, rho):
    e = rng.standard_normal((reps, n))
    y = np.empty_like(e)
    y[:, 0] = e[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        y[:, t] = rho * y[:, t - 1] + e[:, t]
    return y


print(f"  where two tests of stationarity land, {reps} samples of each process")
print("   process                       n    half-life    stationary    integrated"
      "    both    neither")
for name, n, rho in (("random walk", 6_400, 1.0), ("daily returns", 6_400, 0.0),
                     ("AR(1) rho = 0.995", 6_400, 0.995), ("AR(1) rho = 0.98", 252, 0.98),
                     ("AR(1) rho = 0.90", 150, 0.90)):
    if rho == 1.0:
        y = np.cumsum(rng.standard_normal((reps, n)), axis=1)
        hl = "infinite"
    elif rho == 0.0:
        y = rng.standard_normal((reps, n))
        hl = "0"
    else:
        y = ar1(n, rho)
        hl = f"{np.log(0.5) / np.log(rho):.0f}"
    a = np.array([adf_t(r) for r in y]) < adf_c
    k = np.array([kpss_stat(r) for r in y]) > kpss_c
    print(f"  {name:<26} {n:6d} {hl:>12} {np.mean(a & ~k):13.3f} {np.mean(~a & k):13.3f}"
          f" {np.mean(a & k):7.3f} {np.mean(~a & ~k):10.3f}")
# =>   where two tests of stationarity land, 400 samples of each process
#       process                       n    half-life    stationary    integrated    both    neither
#      random walk                  6400     infinite         0.000         0.958   0.043      0.000
#      daily returns                6400            0         0.955         0.000   0.045      0.000
#      AR(1) rho = 0.995            6400          138         0.000         0.000   1.000      0.000
#      AR(1) rho = 0.98              252           34         0.048         0.805   0.077      0.070
#      AR(1) rho = 0.90              150            7         0.347         0.253   0.225      0.175
```

The first two rows are the cells the course lands in and they behave. A random walk is called integrated $95.8\%$ of the time and independent returns are called stationary $95.5\%$ of the time, with the residual few percent landing in "both" at exactly the rate two $5\%$ tests should disagree by chance. These are the easy cases, and they are easy because the answer was obvious before the tests were run.

The third row is the cell the course flags and does not build. An AR(1) at $\phi=0.995$ is stationary — the definition is satisfied exactly, the variance is finite and constant, shocks have a half-life of $138$ days — and on $6{,}400$ observations the two tests **both reject on every one of the four hundred samples**. The rate is $1.000$. A practitioner running the standard pair is told, with full confidence and no ambiguity, that the series is simultaneously integrated and not stationary, which is not a bug in either test: ADF has enough data to distinguish $0.995$ from $1$, and KPSS is detecting the enormous low-frequency variation that a $138$-day half-life produces.

The last two rows are the failure mode that costs money. At $\phi=0.98$ with one year of data — a $34$-day half-life, which is an ordinary mean-reverting spread — the verdict is "integrated" $80.5\%$ of the time, and it is wrong. At $\phi=0.90$ with a hundred and fifty observations the four cells split $0.347$, $0.253$, $0.225$ and $0.175$, which is close to the tests answering at random. **The two consistent cells are the ones where the answer was already known, and persistence alone — with no structural break, no regime and nothing exotic — puts an ordinary trading series into the other two.**

## Identifiability Is a Property of the Model and No Optimizer Repairs It

A model is **identifiable** if distinct parameters give distinct laws: $\theta_1\ne\theta_2\Rightarrow P_{\theta_1}\ne P_{\theta_2}$. When it fails, two different answers describe the same distribution, the likelihood cannot prefer either, and whichever one gets reported was selected by the optimizer's starting value.

??? note "Proof that an ARMA(1,1) with a common root is white noise for every value of that root, so a continuum of parameters gives one law"
    Write the model as $\phi(L)y_t=\theta(L)\varepsilon_t$ with lag polynomials $\phi(L)=1-\phi L$ and $\theta(L)=1+\theta L$. If $\theta=-\phi$ the two polynomials are identical and cancel:

    $$(1-\phi L)y_t=(1-\phi L)\varepsilon_t\;\Longrightarrow\;y_t=\varepsilon_t\quad\text{for every }\phi\in(-1,1).$$

    A one-dimensional continuum of parameter values maps to the single law "white noise". The likelihood is therefore exactly constant along that curve, its gradient vanishes in the direction of the curve, and the Hessian is singular there. The practically important case is *near*-cancellation, where the curve is replaced by a valley whose floor is flat to within the sampling noise, so the objective cannot distinguish points that are far apart in parameter space.

    What survives is the function of the parameters the data actually determines. For an ARMA(1,1) the lag-one autocorrelation is

    $$\rho_1=\frac{(1+\phi\theta)(\phi+\theta)}{1+2\phi\theta+\theta^{2}},$$

    and this is estimated well even when $\phi$ and $\theta$ separately are not, because it is a property of the law rather than of the labelling.

    The load-bearing step is that the failure lives in the **parameterization** rather than in the model. The set of laws is perfectly well behaved; the map onto it is many-to-one, and no estimator, no prior and no quantity of data inverts a non-invertible map. **An optimizer handed a flat valley returns the point at which it stopped moving and reports a standard error computed from a curvature it never encountered**, which is why the two numbers most likely to be unstable are also the two most likely to be reported as significant.

```python
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(10043)
n, phi0, th0, sig, draws = 6_400, 0.201, -0.289, 0.0122, 12    # the course's own fitted ARMA


def ssr(p, y):                                                 # conditional sum of squares
    phi, th = p
    e = np.zeros(y.size)
    for t in range(1, y.size):
        e[t] = y[t] - phi * y[t - 1] - th * e[t - 1]
    return e @ e / y.size


w = rng.standard_normal(n)
print("  white noise, objective along the exact-cancellation curve theta = -phi")
print("      phi     SSR")
for p in (0.0, 0.2, 0.5, 0.8, 0.95):
    print(f"  {p:9.2f} {ssr((p, -p), w):9.7f}")

print(f"  {draws} refits of an ARMA(1,1) truly at phi {phi0}, theta {th0}")
ph, th, r1 = [], [], []
for _ in range(draws):
    e = sig * rng.standard_normal(n + 1)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi0 * y[t - 1] + e[t] + th0 * e[t - 1]
    best = min((minimize(ssr, s, args=(y,), method="Nelder-Mead") for s in
                ((0.0, 0.0), (0.6, -0.6), (-0.4, 0.4), (0.9, -0.9))), key=lambda r: r.fun)
    ph.append(best.x[0])
    th.append(best.x[1])
    r1.append(np.corrcoef(y[:-1], y[1:])[0, 1])
ph, th, r1 = np.array(ph), np.array(th), np.array(r1)
truth = (1 + phi0 * th0) * (phi0 + th0) / (1 + 2 * phi0 * th0 + th0 ** 2)
print(f"    phi_hat   mean {ph.mean():+.4f}  sd {ph.std(ddof=1):.4f}"
      f"  range [{ph.min():+.4f}, {ph.max():+.4f}]")
print(f"    theta_hat mean {th.mean():+.4f}  sd {th.std(ddof=1):.4f}"
      f"  range [{th.min():+.4f}, {th.max():+.4f}]")
print(f"    lag-1 acf mean {r1.mean():+.4f}  sd {r1.std(ddof=1):.4f}  truth {truth:+.4f}")
# =>   white noise, objective along the exact-cancellation curve theta = -phi
#          phi     SSR
#           0.00 1.0018357
#           0.20 1.0018325
#           0.50 1.0019222
#           0.80 1.0023677
#           0.95 1.0037447
#      12 refits of an ARMA(1,1) truly at phi 0.201, theta -0.289
#        phi_hat   mean +0.2187  sd 0.0641  range [+0.0702, +0.2967]
#        theta_hat mean -0.3046  sd 0.0735  range [-0.3966, -0.1324]
#        lag-1 acf mean -0.0828  sd 0.0131  truth -0.0857
```

The first panel walks the objective along the exact-cancellation curve on genuine white noise. The conditional sum of squares reads $1.0018357$, $1.0018325$, $1.0019222$, $1.0023677$ and $1.0037447$ as $\phi$ moves from $0$ to $0.95$ — agreement to four significant figures across nearly the whole parameter space. Any optimizer with a realistic convergence tolerance stops wherever it happened to start, and reports that point as the fit.

The second panel is the course's own model. Twelve independent samples of an ARMA(1,1) truly at $\phi=+0.201$ and $\theta=-0.289$, each fitted from four widely separated starting points with the best kept. The estimate $\hat\phi$ ranges over $[+0.0702,+0.2967]$ and $\hat\theta$ over $[-0.3966,-0.1324]$ — spans of roughly four times their own standard deviations of $0.0641$ and $0.0735$, on samples of six thousand four hundred observations, with nothing wrong with the data or the optimizer.

The third line is what the data does determine. The lag-one autocorrelation reads $-0.0828$ with a standard deviation of $0.0131$ against a true value of $-0.0857$: pinned down tightly and correctly, while the two parameters that generate it wander. The course reports `ar.L1 +0.201  ma.L1 -0.289  (p = 2.4e-04, 5.5e-08)` and then dismisses the model as economically feeble because it "repackages a lag-one autocorrelation of −0.086" — and those are the same observation. **A $p$-value of $2.4\times10^{-4}$ on a parameter and a sampling range four times the reported standard error are not in tension; they are the same ridge, described once by the fit summary and once by the refits nobody runs.**

!!! warning "Two coefficients that move together across refits while their standard errors stay small are a ridge in the objective, and the optimizer that found them will find a different pair tomorrow"
    The four common ridges are worth recognizing by sight: near-common autoregressive and moving-average roots, as above; unlabeled mixture or hidden-Markov states, where the course notes that "EM numbers its states arbitrarily, and every rerun may swap them, so *you* impose the labeling (here, by variance)"; a signal-and-noise pair in a local-level model, where only the ratio $q=Q/R$ is determined by the data; and any factor model whose loadings and factors are free up to a rotation. The free diagnostic takes one line and costs less than the fit itself: **refit from three widely separated starting values and print both the objective and the parameters — if the objective agrees to five significant figures while the parameters disagree in the first, the reported standard errors describe a curvature that is not there.** A second check for the same price is the correlation matrix of the estimates across refits, where an entry near $\pm1$ names the ridge's direction directly. Neither diagnostic requires knowing in advance that a ridge exists, which is the property that makes them worth running by default.

## A Misspecified Model Is Consistent, and What It Is Consistent For Has No Name in the Problem

The remaining failure is the one where the parameterization is fine and the set is simply too small to contain the truth. The intuition is that the fit will look bad. It does not.

??? note "Proof that maximum likelihood under a misspecified model converges to the Kullback–Leibler projection of the truth onto the model"
    Let the data be iid from $P_0$ and let $\mathcal{P}=\{P_\theta\}$ be the model, with $P_0$ not necessarily in it. The normalized log-likelihood converges by the law of large numbers,

    $$\frac1n\sum_{i=1}^{n}\log p_\theta(X_i)\;\longrightarrow\;\mathbb{E}_{P_0}\!\left[\log p_\theta(X)\right]=\int p_0\log p_\theta,$$

    uniformly in $\theta$ under standard regularity conditions. Maximizing the right side over $\theta$ is the same as minimizing

    $$\mathrm{KL}(P_0\Vert P_\theta)=\int p_0\log\frac{p_0}{p_\theta},$$

    since the term $\int p_0\log p_0$ does not involve $\theta$. So $\hat\theta\to\theta^{\ast}=\arg\min_\theta \mathrm{KL}(P_0\Vert P_\theta)$, the **pseudo-true parameter**, whether or not the model is correct. The estimator is consistent, asymptotically normal, and its asymptotic covariance is the sandwich $A^{-1}BA^{-1}$ with $A$ the expected Hessian and $B$ the score's variance; the familiar $A^{-1}$ requires the information-matrix equality $A=B$, which holds only when the model is right.

    The load-bearing step is that consistency was never a claim about the truth — it is a claim about a fixed point of an estimating equation, and the fixed point exists regardless. **A misspecified model is consistent for whichever member of your set most resembles the world in the one metric maximum likelihood happens to use, and no output of the fit distinguishes that case from the well-specified one**, because both produce an estimate, a standard error and a log-likelihood formatted identically.

```python
import numpy as np

rng = np.random.default_rng(10047)
n, om, al, be, reps = 6_400, 0.0252, 0.126, 0.856, 300         # the course's fitted GARCH(1,1)

sg, ex1, ex5 = [], [], []
for _ in range(reps):
    z = rng.standard_normal(n)
    h = np.empty(n)
    h[0] = om / (1 - al - be)
    for t in range(1, n):
        h[t] = om + al * (h[t - 1] * z[t - 1] ** 2) + be * h[t - 1]
    y = np.sqrt(h) * z
    s = y.std(ddof=1)                                          # the iid model's whole fit
    sg.append(s * np.sqrt(252) / 100)
    ex1.append(np.mean(y < -2.326 * s))                        # a nominal 1% loss threshold
    ex5.append(np.mean(y < -1.645 * s))
print(f"  persistence {al + be:.3f}, {reps} paths of {n} days, iid normal fitted to each")
print(f"    annualized sigma_hat {np.mean(sg):.3f} +/- {np.std(sg, ddof=1):.3f}")
print(f"    nominal 1% loss threshold breached {100 * np.mean(ex1):.3f}% of days")
print(f"    nominal 5% loss threshold breached {100 * np.mean(ex5):.3f}% of days")
# =>   persistence 0.982, 300 paths of 6400 days, iid normal fitted to each
#        annualized sigma_hat 0.185 +/- 0.019
#        nominal 1% loss threshold breached 1.504% of days
#        nominal 5% loss threshold breached 4.360% of days
```

The first line is the projection working exactly as the proof says. Three hundred paths of a GARCH process with the course's fitted persistence of $0.982$ are each handed to an iid normal model, and the fitted annualized volatility comes back at $18.5\%$ with a spread across paths of $1.9$ points. That number is not wrong. It is the unconditional volatility of the process, a real and useful property, estimated consistently with a standard error that is approximately correct. Nothing in the output announces that the model is missing the entire conditional variance structure.

The two exceedance lines are what the projection cost. A loss threshold placed where the fitted normal puts $1\%$ of its mass is breached on $1.504\%$ of days — half again too often — while the threshold placed at the $5\%$ point is breached on $4.360\%$ of days, which is too *rarely*. That signature is diagnostic: the fitted normal is too fat in the shoulders and too thin in the tail, because it is averaging a mixture of calm and stressed regimes into one width. A risk limit written at the one-percent level and validated by counting breaches at the five-percent level would pass.

**The estimate is unbiased, the estimator is consistent, the standard error is honest, and the number is the answer to a question the risk limit does not ask.** The failure is not in any step of the calculation; it is in the sentence, written before the calculation, that said the returns were independent and identically distributed.

!!! note "Nonparametric is not assumption-free, it is a different assumption whose price is paid in rate rather than in correctness, and the empirical distribution function is the cleanest example in the appendix"
    The empirical distribution function assumes nothing about shape and is the maximum-likelihood estimator within the set of all distributions, which sounds like the end of the argument. It is also, as [Cumulative Distribution Functions](../part-03-random-variables/02-cumulative-distribution-functions.md) records, an object with atoms and no density at all — "not a slightly noisy version of the truth" but "a different kind of object" — and that difference is exactly why the bootstrap built on it is inconsistent for a maximum. Generalizing: every nonparametric method carries an assumption somewhere, and it is usually encoded in a tuning parameter rather than in a sentence. A kernel density estimate assumes smoothness and picks a bandwidth; a rolling estimator assumes local stationarity and picks a window, which is the choice the previous page measured; a nearest-neighbour method assumes a metric. None of these is estimated in an assumption-free way, so the honest version of the parametric–nonparametric split is not "assumptions versus none" but **assumptions you can state in a sentence versus assumptions buried in a tuning parameter**, and only the first kind gets argued about in a research meeting.

## The Model Is the Assumption You Cannot Test From Inside It

Three failures were established and they occupy different parts of the same structure. Stationarity is a property of the law, and testing it on a finite sample confuses that property with the sample's ability to resolve it, so an ordinary mean-reverting series with a $138$-day half-life is declared integrated and not stationary simultaneously and with total confidence. Identifiability is a property of the parameterization, and when it fails the fit reports two unstable numbers with impeccable $p$-values while the one combination the data determines sits quietly at $-0.083$. Misspecification is a property of the set, and when it fails the estimator remains consistent for a pseudo-true parameter that is a real quantity and is not the one in the problem.

The symmetry between the last two is worth stating because their diagnostics are opposites. Non-identifiability is a model too *large* for the data to distinguish within, and its signature is flatness — an objective that does not change when the parameters do. Misspecification is a model too *small* for the truth to fit inside, and its signature is structure — a pattern in what the model failed to explain. So one is found by perturbing the parameters and watching the objective, and the other by holding the fit and examining the residuals, and a research process that runs only the second of these will never see the first. Neither is visible in the output that fits summaries actually print, which report coefficients, standard errors and a log-likelihood identically in all three cases.

That is why the model is the assumption you cannot test from inside it. Every test in [Part XII](../part-12-hypothesis-testing/index.md), every interval in [Part XI](../part-11-parameter-estimation/index.md) and every criterion in [Part XIV](../part-14-model-selection/index.md) is computed *within* a set that was asserted, and none of them ranks that set against the sets nobody wrote down. Having fixed what the candidates are, the next question is which functions of the data are worth keeping at all — whether a summary can stand in for the whole sample without losing anything the model could have used. That is [Statistics and Sufficiency](05-statistics-and-sufficiency.md).

**A model is a claim about which worlds are possible, and every number computed afterwards is a conditional statement that the report will present as an unconditional one.**
