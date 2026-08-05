# Copulas

[Marginal Distributions](../part-03-random-variables/06-marginal-distributions.md) shows two joint laws with identical margins and opposite tail behaviour, says "what is missing has a name," and points here. The name is a copula, Sklar's theorem says the split into margins and copula is unique, and the practical content is that the object everyone measures — a correlation — is a property of the copula only in the rank sense and not at all in the Pearson sense. Four families at a common Kendall's $\tau$ of $0.40$, joined to identical standard normal margins, agree on every summary a desk computes: Pearson correlations of $0.5876$, $0.5785$, $0.5767$ and $0.5848$, and rank correlations equal by construction. Their joint tails do not agree at all. Lower tail dependence measured at the $0.1$st percentile runs $0.1175$, $0.3400$, $0.5575$ and $0.0350$, and the Gumbel family that is nearly independent in the lower tail has $0.5050$ in the upper one — the dependence is not merely stronger or weaker but located in a different place. The estimate is hard to obtain: distinguishing a book with $\lambda=0.1144$ from one with $\lambda=0.1954$ takes about twenty years of daily data before the two stop overlapping. And the default model conceals the whole question — a Gaussian copula fitted to $t(4)$-copula data reproduces Kendall's $\tau$ at $0.40018$ against $0.39996$, matches the Pearson correlation exactly, passes a Kolmogorov–Smirnov test on the margins identically, and understates the probability of both assets breaching their first percentile by a factor of $1.89$.

This page covers Sklar's theorem and the uniqueness of the split it makes, the rank correlations that are copula functionals against the Pearson correlation that is not, the four standard families and where each puts its dependence, the coefficient of tail dependence and its estimation, and what a Gaussian copula conceals when fitted to data that has a tail. It does not prove the Fréchet–Hoeffding bounds or exhibit two laws with matching margins and opposite tails, which is [Marginal Distributions](../part-03-random-variables/06-marginal-distributions.md); it does not prove that no Gaussian correlation produces tail dependence, or report the measured $\lambda=0.66$ for SPY and EFA, which are [The Multivariate Gaussian](../part-06-multivariate-probability/05-multivariate-gaussian.md); it does not define Spearman's or Kendall's coefficient, which is [Correlation](../part-04-expectation-and-moments/05-correlation.md); it does not establish the probability integral transform's uniformity, which is [Continuous Uniform Distribution](../part-05-common-distributions/09-continuous-uniform-distribution.md); it does not derive tail index estimators, which are [Heavy-Tailed Returns](12-heavy-tailed-returns.md) and [Extreme Value Theory](13-extreme-value-theory.md); it does not shock a correlation matrix or replay a crisis, which is [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md); and it never treats a matched correlation as a matched model.

The trading stake is a course measurement that overturns the most repeated claim in risk management and needs this page's vocabulary to state precisely. [Drawdowns, Tail Risk, and Stress Testing](../../part-08-portfolio-management/05-drawdowns-tail-risk-stress-testing.md) computes joint worst-decile frequencies and prints `SPY      /TLT             -0.312             -0.264                 0.87%     0.09` against `SPY      /EFA             +0.872             +0.815                 6.62%     0.66`, noting that independence would give $0.10$ — so SPY and TLT are *less* likely to crash together than two independent assets, while SPY and EFA are six times more likely. Those two numbers are tail dependence coefficients, they are the quantity section 2 shows a correlation cannot determine, and section 4 shows that the model most desks use to generate stress scenarios has them both equal to zero by construction.

## Sklar's Theorem Splits a Joint Law Into Margins and Dependence, and the Split Is Unique

The reason a copula is worth defining is that the decomposition it performs is not an approximation or a modelling convenience. It is exact, always available, and unique wherever the margins are continuous.

??? note "Proof that every joint law factors into its margins and a copula, that the copula is unique for continuous margins, and that it is invariant under increasing transformations of either coordinate"

    Let $H$ be a joint distribution function with margins $F$ and $G$, both continuous. By the probability integral transform, $U=F(X)$ and $V=G(Y)$ are each uniform on $[0,1]$, so their joint distribution function
    $$C(u,v):=\mathbf{P}\!\left(F(X)\le u,\;G(Y)\le v\right)=H\!\left(F^{-1}(u),G^{-1}(v)\right)$$
    is a distribution on the unit square with uniform margins — a **copula**. Substituting back gives $H(x,y)=C(F(x),G(y))$, which is **Sklar's theorem**: every joint law is a copula applied to its own margins.

    Uniqueness follows because continuity makes $F$ and $G$ invertible on the relevant range, so $C$ is determined pointwise by the displayed formula; with atoms the copula is unique only on the closure of the range of the margins, which is why the discrete case admits several. Conversely, any copula combined with any margins produces a valid joint law, so the two components are free to be chosen independently — which is the modelling use.

    Invariance is the property that makes the split meaningful rather than merely possible. If $\phi$ and $\psi$ are strictly increasing, then $\phi(X)$ has distribution function $F\circ\phi^{-1}$ and the same argument gives $C$ unchanged. So the copula is exactly the part of the joint law that survives every monotone rescaling of the coordinates — taking logarithms, converting to returns, standardizing by a volatility — and the margins are exactly the part that does not.

    **The load-bearing consequence is a division of questions. Anything expressible in ranks is a question about the copula alone and is unaffected by what the margins do; anything involving the actual values is a question about both. A Pearson correlation involves the values, so it is not a copula functional and cannot be read as a measure of dependence structure — which is section 2's first column.**

## Four Copulas at One Kendall's Tau, and Their Joint Tails Differ by a Factor of Five

Fixing the margins and fixing a rank correlation leaves the joint tail almost entirely free. The four standard families show how much room remains.

??? note "Proof that Kendall's $\tau$ and Spearman's $\rho$ are functionals of the copula alone, and that the $t$ copula has positive tail dependence at every finite degrees of freedom while the Gaussian has none"

    Kendall's $\tau$ is defined through concordance of two independent pairs, and concordance is a statement about ranks, so it depends only on $C$; explicitly $\tau=4\int C\,dC-1$. Spearman's $\rho$ is the Pearson correlation of the ranks, giving $\rho_S=12\int uv\,dC-3$. Neither expression contains the margins. The **Pearson** correlation, by contrast, is $\mathbb{E}[XY]$ standardized, and $\mathbb{E}[XY]$ integrates the values, so it depends on the margins and the copula jointly — two books with the same copula and different margins have different Pearson correlations, and section 2's table shows the converse.

    The **coefficient of lower tail dependence** is
    $$\lambda_L=\lim_{q\to0^{+}}\mathbf{P}\!\left(V\le q\mid U\le q\right)=\lim_{q\to0^{+}}\frac{C(q,q)}{q},$$
    a copula functional by construction, with $\lambda_U$ defined symmetrically from the survival copula. For the Gaussian copula $\lambda_L=\lambda_U=0$ at every correlation below one, which is the result [The Multivariate Gaussian](../part-06-multivariate-probability/05-multivariate-gaussian.md) proves. For the $t$ copula with $\nu$ degrees of freedom and correlation $r$,
    $$\lambda_L=\lambda_U=2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-r)}{1+r}}\right)>0,$$
    positive for every finite $\nu$ and every $r>-1$, and decreasing to zero as $\nu\to\infty$ — which locates the Gaussian as the boundary case of a family whose interior all has tail dependence. The Archimedean families are one-sided: Clayton has $\lambda_L=2^{-1/\theta}$ and $\lambda_U=0$, Gumbel has $\lambda_U=2-2^{1/\theta}$ and $\lambda_L=0$.

    **The load-bearing distinction is between the strength of dependence and its location. Kendall's $\tau$ summarizes the whole copula in one number and is blind to where in the unit square the mass sits; $\lambda_L$ and $\lambda_U$ are separate numbers reading opposite corners, and a family can have any combination of them at a fixed $\tau$.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18141)
REPS, TAU = 400_000, 0.40                               # one Kendall's tau for every family


def gauss(n, tau):
    r = np.sin(np.pi * tau / 2)
    return stats.norm.cdf(rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n))


def student(n, tau, nu=4):
    r = np.sin(np.pi * tau / 2)
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)
    return stats.t.cdf(z * np.sqrt(nu / rng.chisquare(nu, (n, 1))), nu)


def clayton(n, tau):
    th = 2 * tau / (1 - tau)
    v = rng.gamma(1 / th, 1.0, (n, 1))                  # Marshall-Olkin mixing variable
    return (1 + rng.exponential(1.0, (n, 2)) / v) ** (-1 / th)


def gumbel(n, tau):
    th = 1 / (1 - tau)
    a = 1 / th
    u, w = rng.uniform(0, np.pi, (n, 1)), rng.exponential(1.0, (n, 1))
    s = (np.sin(a * u) / np.sin(u) ** (1 / a)) * (np.sin((1 - a) * u) / w) ** ((1 - a) / a)
    return np.exp(-(rng.exponential(1.0, (n, 2)) / s) ** a)


R = np.sin(np.pi * TAU / 2)
THEORY = {"Gaussian": (0.0, 0.0),
          "Student t(4)": (2 * stats.t.cdf(-np.sqrt(5 * (1 - R) / (1 + R)), 5),) * 2,
          "Clayton": (2 ** (-(1 - TAU) / (2 * TAU)), 0.0),
          "Gumbel": (0.0, 2 - 2 ** (1 - TAU))}

print(f"  four copulas at the same Kendall's tau of {TAU}, joined to identical standard normal"
      f" margins. Every marginal risk number is therefore identical and every rank correlation"
      f" agrees; what differs is the joint tail, and where in it the dependence sits."
      f" {REPS:,} draws")
print("     copula         Kendall tau   Pearson   lower lambda: theory   at 1%   at 0.1%"
      "   upper lambda: theory   at 1%   at 0.1%")
for name, fn in (("Gaussian", gauss), ("Student t(4)", student),
                 ("Clayton", clayton), ("Gumbel", gumbel)):
    u = fn(REPS, TAU)
    x = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    lo = lambda q: np.mean((u[:, 0] < q) & (u[:, 1] < q)) / q
    hi = lambda q: np.mean((u[:, 0] > 1 - q) & (u[:, 1] > 1 - q)) / q
    tl, tu = THEORY[name]
    print(f"    {name:14s} {stats.kendalltau(u[:80_000, 0], u[:80_000, 1]).statistic:11.4f}"
          f"   {np.corrcoef(x.T)[0, 1]:7.4f}   {tl:20.4f}   {lo(0.01):5.4f}   {lo(0.001):7.4f}"
          f"   {tu:20.4f}   {hi(0.01):5.4f}   {hi(0.001):7.4f}")
# =>   four copulas at the same Kendall's tau of 0.4, joined to identical standard normal margins. Every marginal risk number is therefore identical and every rank correlation agrees; what differs is the joint tail, and where in it the dependence sits. 400,000 draws
#         copula         Kendall tau   Pearson   lower lambda: theory   at 1%   at 0.1%   upper lambda: theory   at 1%   at 0.1%
#        Gaussian            0.3996    0.5876                 0.0000   0.1762    0.1175                 0.0000   0.1792    0.0825
#        Student t(4)        0.4010    0.5785                 0.3062   0.3557    0.3400                 0.3062   0.3357    0.2975
#        Clayton             0.4032    0.5767                 0.5946   0.5807    0.5575                 0.0000   0.0212    0.0025
#        Gumbel              0.4006    0.5848                 0.0000   0.0943    0.0350                 0.4843   0.4978    0.5050
```

The first two columns are the point. Kendall's $\tau$ is $0.40$ in every row by construction, and the Pearson correlation — which is *not* a copula functional and had no reason to agree — comes out at $0.5876$, $0.5785$, $0.5767$ and $0.5848$, a spread of one percent. A desk comparing these four books on any correlation measure, rank or linear, would call them the same book.

The tail columns say they are not. At the $0.1$st percentile the lower tail dependence is $0.1175$, $0.3400$, $0.5575$ and $0.0350$ — a factor of sixteen between the Clayton and the Gumbel — and every measurement sits close to its theoretical value where one exists: $0.3062$ predicted against $0.3400$ and $0.2975$ measured for the $t$, $0.5946$ against $0.5575$ for Clayton's lower tail, $0.4843$ against $0.5050$ for Gumbel's upper.

Two rows deserve separate readings. The Gaussian's theoretical tail dependence is zero in both tails, and the measurements are $0.1175$ and $0.0825$ — falling toward zero, slowly, which is the practical form of the theorem: a Gaussian copula does have joint extremes at any finite quantile, and the frequency vanishes only in a limit no sample reaches. And the two Archimedean families are one-sided in opposite directions, Clayton's lower $0.5575$ against upper $0.0025$ and Gumbel's lower $0.0350$ against upper $0.5050$. **A single dependence number cannot distinguish a book that crashes together from one that rallies together, and both are compatible with the same correlation.**

## Tail Dependence Is Estimated From a Handful of Joint Events

The coefficient is defined as a limit, and every estimate of it is taken at a finite quantile with a finite sample, which reproduces exactly the dial [Extreme Value Theory](13-extreme-value-theory.md) found for a threshold.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18143)
REPS, NU = 3_000, 4
TRUE_TAUS = (0.10, 0.25, 0.40)


def t_copula(n, tau, nu=NU):
    r = np.sin(np.pi * tau / 2)
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)
    return stats.t.cdf(z * np.sqrt(nu / rng.chisquare(nu, (n, 1))), nu)


def lam_true(tau, nu=NU):
    r = np.sin(np.pi * tau / 2)
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1)


print(f"  estimating lower tail dependence by counting joint exceedances: lambda-hat(q) ="
      f" P(both below q) / q. The estimator needs q small for the limit to have arrived and q"
      f" large for anything to be counted, and n x q x lambda is the expected number of joint"
      f" events it rests on. t({NU}) copula, {REPS:,} replications")
print("     Kendall tau   true lambda   n       q      joint events expected   lambda-hat: mean"
      "     sd   relative sd")
for tau in TRUE_TAUS:
    lt = lam_true(tau)
    for n in (1_000, 5_000):
        for q in (0.05, 0.01):
            u = np.array([np.mean((c[:, 0] < q) & (c[:, 1] < q)) / q
                          for c in (t_copula(n, tau) for _ in range(REPS))])
            print(f"    {tau:11.2f}   {lt:11.4f}   {n:5,d}   {q:5.1%}   {n * q * lt:21.1f}"
                  f"   {u.mean():16.4f}   {u.std():6.4f}   {u.std() / u.mean():11.2f}")

lo_tau, hi_tau = TRUE_TAUS[0], TRUE_TAUS[1]
print(f"\n     how long does it take to separate two books whose true lambdas are"
      f" {lam_true(lo_tau):.4f} and {lam_true(hi_tau):.4f}? Read at q = 5%, 1,000 replications")
print("     years   n        lambda-hat, lower book   upper book   P(the lower book measures higher)")
for years in (4, 20, 100):
    n = years * 252
    a = np.array([np.mean((c[:, 0] < 0.05) & (c[:, 1] < 0.05)) / 0.05
                  for c in (t_copula(n, lo_tau) for _ in range(1_000))])
    b = np.array([np.mean((c[:, 0] < 0.05) & (c[:, 1] < 0.05)) / 0.05
                  for c in (t_copula(n, hi_tau) for _ in range(1_000))])
    print(f"    {years:5d}   {n:6,d}   {a.mean():23.4f}   {b.mean():12.4f}"
          f"   {np.mean(a[:, None] > b[None, :]):33.4f}")
# =>   estimating lower tail dependence by counting joint exceedances: lambda-hat(q) = P(both below q) / q. The estimator needs q small for the limit to have arrived and q large for anything to be counted, and n x q x lambda is the expected number of joint events it rests on. t(4) copula, 3,000 replications
#         Kendall tau   true lambda   n       q      joint events expected   lambda-hat: mean     sd   relative sd
#               0.10        0.1144   1,000    5.0%                     5.7             0.1783   0.0591          0.33
#               0.10        0.1144   1,000    1.0%                     1.1             0.1411   0.1196          0.85
#               0.10        0.1144   5,000    5.0%                    28.6             0.1798   0.0265          0.15
#               0.10        0.1144   5,000    1.0%                     5.7             0.1383   0.0527          0.38
#               0.25        0.1954   1,000    5.0%                     9.8             0.2768   0.0746          0.27
#               0.25        0.1954   1,000    1.0%                     2.0             0.2242   0.1498          0.67
#               0.25        0.1954   5,000    5.0%                    48.8             0.2762   0.0331          0.12
#               0.25        0.1954   5,000    1.0%                     9.8             0.2277   0.0672          0.30
#               0.40        0.3062   1,000    5.0%                    15.3             0.3934   0.0903          0.23
#               0.40        0.3062   1,000    1.0%                     3.1             0.3401   0.1836          0.54
#               0.40        0.3062   5,000    5.0%                    76.5             0.3923   0.0403          0.10
#               0.40        0.3062   5,000    1.0%                    15.3             0.3413   0.0806          0.24
#
#         how long does it take to separate two books whose true lambdas are 0.1144 and 0.1954? Read at q = 5%, 1,000 replications
#         years   n        lambda-hat, lower book   upper book   P(the lower book measures higher)
#            4    1,008                    0.1797         0.2733                              0.1314
#           20    5,040                    0.1816         0.2743                              0.0131
#          100   25,200                    0.1798         0.2758                              0.0000
```

The dial is visible along every pair of rows. Moving from $q=5\%$ to $q=1\%$ at $n=5{,}000$ takes the estimate from $0.1798$ to $0.1383$ against a truth of $0.1144$ — closer, because the limit is being approached — while the relative standard deviation rises from $0.15$ to $0.38$, because the expected number of joint events falls from $28.6$ to $5.7$. Every estimate in the table is biased upward, and the bias is the same finite-threshold effect that inflates a shape parameter: at any $q$ above the limit the conditional probability includes ordinary co-movement as well as tail dependence.

The second panel prices the resulting resolution. Two books whose true coefficients are $0.1144$ and $0.1954$ — a difference large enough to matter, roughly the gap between the published SPY/TLT and a moderately dependent pair — are measured with enough overlap that the *lower*-dependence book returns the higher estimate on $0.1314$ of four-year samples. Twenty years brings that to $0.0131$ and a century to $0.0000$. **A tail dependence coefficient is a statement about the rarest events in the sample, so it is estimated from a handful of them, and the sample sizes that separate two plausible books are measured in decades.**

!!! note "A Pearson correlation, a rank correlation, a tail dependence coefficient and a conditional correlation are four numbers describing one dependence, and only two of them describe the copula"
    **A Pearson correlation** mixes the copula with the margins, so it changes when a coordinate is rescaled non-linearly and cannot be compared across books with different tail shapes. **A rank correlation** — Kendall's or Spearman's — is a copula functional, invariant to any increasing transformation, and summarizes the entire unit square in one number. **A tail dependence coefficient** is also a copula functional but reads one corner only, and section 2 shows it is free to vary by a factor of sixteen at fixed rank correlation. **A conditional correlation**, computed on the subsample where one asset is in its worst decile, is none of these: it is a property of the copula *and* of the conditioning rule, and [Conditional Gaussian Distributions](../part-06-multivariate-probability/06-conditional-gaussian.md) shows it falls under a Gaussian model purely from truncation, so a decline in it is not evidence of anything. The published `-0.312 / -0.264` pair for SPY and TLT is a Pearson correlation next to a conditional one, and the $0.09$ beside them is the only number in the row that is a property of the dependence alone.

## Every Diagnostic Passes and the Joint Tail Is Understated by Half

The Gaussian copula is the default in almost every risk system, and the case for it is that it is easy to fit and its fit can be checked. The second half of that claim is what fails.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18145)
N, REPS, NU, TAU = 5_000, 2_000, 4, 0.40
R = np.sin(np.pi * TAU / 2)
LAM = 2 * stats.t.cdf(-np.sqrt((NU + 1) * (1 - R) / (1 + R)), NU + 1)


def t_copula(n):
    z = rng.multivariate_normal([0, 0], [[1, R], [R, 1]], n)
    return stats.t.cdf(z * np.sqrt(NU / rng.chisquare(NU, (n, 1))), NU)


print(f"  a Gaussian copula fitted to data generated by a t({NU}) copula at Kendall's tau {TAU},"
      f" with standard normal margins. The fit is done the usual way -- match the rank correlation"
      f" -- and every check a desk runs on it passes. The truth has lower tail dependence"
      f" {LAM:.4f}; the fitted model has none. n = {N:,}, {REPS:,} replications")
print("     check                                       data      fitted Gaussian copula   verdict")
tau_d, tau_g, ks_d, p_joint_d, p_joint_g = [], [], [], [], []
for _ in range(REPS):
    u = t_copula(N)
    x = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    t_hat = stats.kendalltau(u[:1_000, 0], u[:1_000, 1]).statistic
    r_hat = np.sin(np.pi * t_hat / 2)                    # the fitted Gaussian copula's parameter
    g = rng.multivariate_normal([0, 0], [[1, r_hat], [r_hat, 1]], N)
    tau_d.append(t_hat)
    tau_g.append(stats.kendalltau(g[:1_000, 0], g[:1_000, 1]).statistic)
    ks_d.append(stats.kstest(x[:, 0], "norm").statistic)
    ug = stats.norm.cdf(g)
    p_joint_d.append([np.mean((u[:, 0] < q) & (u[:, 1] < q)) for q in (0.05, 0.01)])
    p_joint_g.append([np.mean((ug[:, 0] < q) & (ug[:, 1] < q)) for q in (0.05, 0.01)])

jd, jg = np.array(p_joint_d).mean(0), np.array(p_joint_g).mean(0)
rows = [("Kendall's tau", np.mean(tau_d), np.mean(tau_g)),
        ("Pearson correlation of the margins", R, R),
        ("Kolmogorov-Smirnov of margin 1 vs normal", np.mean(ks_d), np.mean(ks_d)),
        ("P(both below their 5th percentile)", jd[0], jg[0]),
        ("P(both below their 1st percentile)", jd[1], jg[1])]
for name, d, g in rows:
    ok = "pass" if abs(d - g) < 0.02 * max(abs(d), 1e-9) + 1e-4 else "FAIL"
    print(f"    {name:44s} {d:8.5f}   {g:23.5f}   {ok:>7}")
print(f"    {'ratio of joint-tail probabilities, 5% / 1%':44s} {'':8s}"
      f"   {jd[0] / jg[0]:11.2f}  {jd[1] / jg[1]:10.2f}   {'':>7}")
# =>   a Gaussian copula fitted to data generated by a t(4) copula at Kendall's tau 0.4, with standard normal margins. The fit is done the usual way -- match the rank correlation -- and every check a desk runs on it passes. The truth has lower tail dependence 0.3062; the fitted model has none. n = 5,000, 2,000 replications
#         check                                       data      fitted Gaussian copula   verdict
#        Kendall's tau                                 0.39996                   0.40018      pass
#        Pearson correlation of the margins            0.58779                   0.58779      pass
#        Kolmogorov-Smirnov of margin 1 vs normal      0.01219                   0.01219      pass
#        P(both below their 5th percentile)            0.01971                   0.01509      FAIL
#        P(both below their 1st percentile)            0.00340                   0.00180      FAIL
#        ratio of joint-tail probabilities, 5% / 1%                     1.31        1.89          
```

The first three rows are the validation suite, and they pass without qualification: Kendall's $\tau$ at $0.40018$ against $0.39996$, the Pearson correlation identical by construction, and a Kolmogorov–Smirnov statistic on the margins that is the same number for both because the margins were never in dispute. A desk running these checks concludes that the fitted model reproduces the data.

The last two rows are what the model is for. The probability that both assets breach their fifth percentile on the same day is $0.01971$ in the data and $0.01509$ under the fitted copula; at the first percentile it is $0.00340$ against $0.00180$. The ratio grows from $1.31$ to $1.89$ as the tail deepens, and it grows without bound, because section 2 established that the Gaussian copula's tail dependence is zero while the data's is $0.3062$ — the two are not converging to a common answer at any quantile.

This is the honest failure and its shape is the one this part keeps finding. The model is not wrong about anything it was asked; it is wrong about the only thing it was built to answer. A stress scenario generated from this fit will produce simultaneous breaches at roughly half the rate the data shows, the discrepancy will be worst in the scenarios that matter most, and no goodness-of-fit test computed on the whole sample will register it — because the events in question are, by construction, one percent of one percent of the data. **The Gaussian copula's defect is invisible to every diagnostic that averages over the sample, and it is the sample average that every diagnostic computes.**

## Every Repair Is a Different Family, a Rank-Based Fit, or a Diagnostic Aimed at the Corner

The three findings admit repairs of increasing commitment, and the cheapest one is a change of test rather than a change of model. Section 4's failure is invisible only to statistics computed over the whole sample; the joint exceedance count in the relevant corner is a one-line diagnostic that separates the fitted model from the data at $0.00340$ against $0.00180$, and it needs no new theory and no new data. Running it is the difference between a model that has been checked and one that has been checked where it was going to be used.

Beyond that the repair is a family with the right corner. A $t$ copula adds one parameter and gives symmetric tail dependence that vanishes as its degrees of freedom rise, so it nests the Gaussian and can be tested against it; a Clayton or Gumbel gives a one-sided tail, which is what a book of long equity positions actually has. Section 3 is the caution on all of it: the parameter that distinguishes these families is estimated from the joint events in one corner, of which a four-year sample contains a handful, so the choice between them is often not decidable on the data at hand and is better made on the mechanism.

!!! warning "A dependence model is fitted on the whole sample and used in the corner, and the corner holds a hundredth of a percent of the data"
    Every fit in section 4 was made by matching a rank correlation computed on five thousand observations, and every use of it concerns the roughly one observation in ten thousand where both assets breach their first percentile. **The free diagnostic is the joint exceedance count itself: for each of a few quantiles, compare the number of days on which both assets breached against the number the fitted model produces, which for a Gaussian copula is $q$ times the conditional probability it implies and for the data is a direct count.** On the published pairs this is already computed — the course lesson's $0.87\%$ and $6.62\%$ against an independent $1\%$ are exactly that count — and the discipline is simply to run it against the model rather than only against independence. Where the two disagree by a factor approaching two at the first percentile, as they do above, the dependence structure has been fitted in the body and applied in the tail.

## A Split That Is Exact, and a Corner Nobody Measures

This page established that Sklar's theorem factors every joint law into its margins and a unique copula for continuous margins, and that the copula is invariant under increasing transformations of either coordinate, so rank correlations are copula functionals and the Pearson correlation is not; that four families at a common Kendall's $\tau$ of $0.40$ produce Pearson correlations spanning $0.5767$ to $0.5876$ while their lower tail dependence at the $0.1$st percentile runs $0.1175$, $0.3400$, $0.5575$ and $0.0350$, with the Archimedean pair one-sided in opposite directions and every measurement matching its theoretical value where one exists; that the coefficient is estimated from the joint events in one corner, so at $n=5{,}000$ moving the reading quantile from $5\%$ to $1\%$ cuts the bias from $0.1798$ to $0.1383$ against a truth of $0.1144$ while raising the relative standard deviation from $0.15$ to $0.38$, and two books differing by $0.081$ in true coefficient are ranked backwards on $0.1314$ of four-year samples; and that a Gaussian copula fitted to $t(4)$ data matches Kendall's $\tau$, the Pearson correlation and the marginal Kolmogorov–Smirnov statistic exactly while understating joint first-percentile breaches by a factor of $1.89$.

The connection to the three tail pages before it is that this one supplies the dimension they lacked. [Heavy-Tailed Returns](12-heavy-tailed-returns.md) and [Extreme Value Theory](13-extreme-value-theory.md) characterized how far a single series goes and how to extrapolate it; a book is not a series, and the question of whether two extremes arrive together is not answerable from either marginal tail. The structural parallel is exact — both problems have a coefficient defined as a limit, both estimate it from the observations in one corner, both trade bias against variance in the choice of where the corner starts, and both have a default model that answers the question with a confident zero. What every page in this part has assumed is that the parameters being estimated are constant: one tail index, one dependence coefficient, one covariance. The last page asks what happens when they are not.

**A correlation says how much two things move together and a copula says where, and only one of those is on the risk report.**
