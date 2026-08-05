# Numerical Integration

A quadrature rule is usually introduced by its order — the trapezoid rule is second order, Simpson's is fourth, Gauss–Legendre with $n$ nodes is exact on polynomials of degree $2n-1$ — and the ordering suggests a ranking. It is not a ranking. The order is a claim about the integrand's derivatives, and the ranking inverts as soon as the integrand stops having them. Below, on a smooth expectation the trapezoid rule reaches $6.537\times10^{-13}$ with $32$ panels while Simpson's rule at the same budget manages $5.511\times10^{-08}$; on a call payoff, which is the same integral with a kink at the strike, all three rules collapse to observed order $2.00$ and Gauss–Legendre becomes the *worst* of them, $1.664\times10^{-01}$ against the trapezoid rule's $7.113\times10^{-03}$ at $32$ panels. Splitting the domain at the strike restores Simpson's fourth order and Gauss–Legendre's spectral accuracy — $1.306\times10^{-09}$ at $32$ panels, better than the unsplit rule manages at $2048$ — provided the split lands on the strike, since missing it by an eighth of a panel width gives back $0.6565$ of the benefit. And then the failure that is not about order at all: asked for the evidence of a strategy's mean return over a flat prior, `scipy.integrate.quad` returns $0.000000$ with a reported error of $7.27\times10^{-18}$ when the truth is $0.833333$.

This page covers what a rule's order actually assumes, the observed order of the standard rules on the integrands a trading book produces, the domain splitting that restores the assumption and the precision with which the split must be placed, and the reason an adaptive integrator's error estimate fails in exactly the cases its answer does. It derives no Monte Carlo estimator and no $N^{-1/2}$ rate, and does not repeat the comparison of grid cost against simulation cost in high dimension, which is [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md); it reweights no draws by a density ratio, which is [Importance Sampling](../part-09-monte-carlo-methods/04-importance-sampling.md); it normalizes no posterior and runs no grid-against-sampler comparison, which is [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md); it derives no Laplace approximation, which is [Information Criteria (AIC/BIC)](../part-14-model-selection/03-information-criteria.md); it computes no Bayes factor and estimates no evidence by simulation, which is [Bayesian Model Comparison](../part-16-bayesian-statistics/06-bayesian-model-comparison.md); it maximizes nothing, which is [Numerical Optimization](01-numerical-optimization.md); it builds no chain, which is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md); it states no Taylor theorem and derives no differentiation rules, which is [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md); it derives no option price and assumes Black–Scholes rather than establishing it, which is [Options Pricing](../../advanced/11-options-pricing.md); and it never reports a number from an integrator without a check the integrator could not perform on itself.

The trading stake is a course lesson that runs precisely the control this page ends by recommending. [Options Pricing](../../advanced/11-options-pricing.md) extracts a risk-neutral density from a strip of Heston call prices by second differencing, and before using it prints `extracted density is a valid distribution (non-negative, integrates to 1): True` — a numerical integration of a quantity whose exact value is known in advance, run for no purpose except to catch the case where it is not one. The same lesson reaches that strip through `quad(f, 1e-10, 250, limit=500)` on an oscillatory characteristic-function integrand truncated at $250$ by hand, which is section 4's situation exactly. The check and the thing it guards against are in the same code block.

## A Rule's Order Is a Claim About the Integrand's Derivatives, and Buying a Higher One Borrows Against a Derivative That May Not Exist

Every rule in common use is interpolatory: it replaces the integrand by a polynomial through chosen nodes and integrates that exactly. The design freedom is where to put the nodes, and spending it well buys the highest orders and the strongest assumptions in the same transaction.

??? note "Proof that an $n$-point rule can integrate polynomials of degree $2n-1$ exactly only if its nodes are the roots of the degree-$n$ orthogonal polynomial, and that its error is then proportional to the $2n$-th derivative of the integrand"

    Let $\int_a^b f(x)w(x)\,\mathrm{d}x\approx\sum_{i=1}^{n}w_if(x_i)$ and suppose the rule is exact for every polynomial of degree at most $2n-1$. Write $\pi_n(x)=\prod_i(x-x_i)$, of degree $n$. For any polynomial $q$ of degree at most $n-1$ the product $q\pi_n$ has degree at most $2n-1$, so the rule integrates it exactly; but $\pi_n$ vanishes at every node, so the rule returns zero, giving
    $$\int_a^b q(x)\pi_n(x)w(x)\,\mathrm{d}x=0\qquad\text{for all }\deg q\le n-1.$$
    That is exactly the statement that $\pi_n$ is orthogonal to every lower-degree polynomial under the weight $w$, which determines $\pi_n$ up to scale and so fixes the nodes as its roots. The weights follow by requiring exactness on a basis of degree $n-1$.

    For the error, let $H$ be the Hermite interpolant of $f$ at the $x_i$ each counted twice, so $\deg H\le2n-1$ and the rule is exact on it. Interpolation theory gives $f(x)-H(x)=\tfrac{f^{(2n)}(\xi_x)}{(2n)!}\pi_n(x)^{2}$, and since $H$ agrees with $f$ at every node the rule's value is unchanged, so
    $$\int_a^b fw-\sum_i w_if(x_i)=\frac{f^{(2n)}(\xi)}{(2n)!}\int_a^b\pi_n(x)^{2}w(x)\,\mathrm{d}x.$$

    Read the right-hand side as a contract. The factorial in the denominator is what makes an $n$-point Gauss rule spectacular on an analytic integrand; $f^{(2n)}$ in the numerator is what it is spectacular *against*. A call payoff $\max(S-K,0)$ has no second derivative at the strike in the ordinary sense, so $f^{(2n)}$ fails to exist for every $n\ge1$ and the bound asserts nothing at all. The rule still returns a number.

    **The load-bearing point is that order is a property of the pair (rule, integrand) and is quoted as though it were a property of the rule alone, so the rules that look best in a textbook are the ones betting hardest on smoothness, and a payoff kink is not a small violation of that bet but a total one.**

## On a Smooth Expectation Every Rule Beats Its Advertised Order, and One Kink at the Strike Collapses All Three to Second and Inverts the Ranking

The cleanest way to see this is to hold everything fixed — same underlying, same domain, same panel counts — and vary only how smooth the thing being averaged is:

```python
import numpy as np
from scipy import stats

S0, K, SIG, T = 100.0, 100.0, 0.20, 1.0
LO, HI = -8.0, 8.0


def bs_call(k):
    d1 = (np.log(S0 / k) + 0.5 * SIG ** 2 * T) / (SIG * np.sqrt(T))
    return S0 * stats.norm.cdf(d1) - k * stats.norm.cdf(d1 - SIG * np.sqrt(T))


def spot(z):
    return S0 * np.exp(-0.5 * SIG ** 2 * T + SIG * np.sqrt(T) * z)


CASES = (
    ("smooth: E[S_T]", lambda z: spot(z) * stats.norm.pdf(z), S0),
    ("kinked: E[(S_T-K)+]", lambda z: np.maximum(spot(z) - K, 0.0) * stats.norm.pdf(z),
     bs_call(K)),
    ("jump: E[1{S_T>K}]", lambda z: (spot(z) > K) * stats.norm.pdf(z),
     stats.norm.cdf((np.log(S0 / K) - 0.5 * SIG ** 2 * T) / (SIG * np.sqrt(T)))),
)


def trap(f, m):
    x = np.linspace(LO, HI, m + 1)
    return np.trapezoid(f(x), x)


def simp(f, m):
    x = np.linspace(LO, HI, m + 1)
    w = np.ones(m + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return (HI - LO) / (3 * m) * (w * f(x)).sum()


def gauss(f, m):
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (HI - LO) * (w * f(0.5 * (HI - LO) * x + 0.5 * (HI + LO))).sum()


print("  expectations under a lognormal written as integrals over the standard normal on"
      " [-8, 8]; 'order' is log2 of the error ratio against the previous row, which"
      " quadruples the panel count")
print("     integrand                  m   trapezoid   order   Simpson   order"
      "   Gauss-Legendre   order")
for name, f, truth in CASES:
    prev = {}
    for m in (32, 128, 512, 2048):
        row, cells = {}, []
        for tag, rule in (("t", trap), ("s", simp), ("g", gauss)):
            e = abs(rule(f, m) - truth)
            row[tag] = e
            o = np.log2(prev[tag] / e) / 2 if tag in prev and e > 0 else np.nan
            cells.append((e, f"{o:5.2f}" if np.isfinite(o) else "     "))
        prev = row
        print(f"    {name:22s}{m:5d}   {cells[0][0]:9.3e}   {cells[0][1]}"
              f"   {cells[1][0]:7.3e}   {cells[1][1]}"
              f"   {cells[2][0]:14.3e}   {cells[2][1]}")
# =>   expectations under a lognormal written as integrals over the standard normal on [-8, 8]; 'order' is log2 of the error ratio against the previous row, which quadruples the panel count
#         integrand                  m   trapezoid   order   Simpson   order   Gauss-Legendre   order
#        smooth: E[S_T]           32   6.537e-13           5.511e-08                8.100e-11        
#        smooth: E[S_T]          128   3.411e-13    0.47   3.126e-13    8.71        9.663e-13    3.19
#        smooth: E[S_T]          512   3.126e-13    0.06   3.126e-13    0.00        5.969e-13    0.35
#        smooth: E[S_T]         2048   3.126e-13    0.00   3.126e-13    0.00        1.540e-11   -2.34
#        kinked: E[(S_T-K)+]      32   7.113e-03           1.016e-01                1.664e-01        
#        kinked: E[(S_T-K)+]     128   4.154e-04    2.05   6.650e-03    1.97        2.366e-02    1.41
#        kinked: E[(S_T-K)+]     512   2.585e-05    2.00   4.136e-04    2.00        7.809e-04    2.46
#        kinked: E[(S_T-K)+]    2048   1.615e-06    2.00   2.584e-05    2.00        3.618e-05    2.22
#        jump: E[1{S_T>K}]        32   5.991e-02           2.666e-02                3.983e-02        
#        jump: E[1{S_T>K}]       128   1.489e-02    1.00   2.321e-02    0.10        3.782e-02    0.04
#        jump: E[1{S_T>K}]       512   3.721e-03    1.00   5.787e-03    1.00        7.597e-04    2.82
#        jump: E[1{S_T>K}]      2048   9.304e-04    1.00   1.447e-03    1.00        7.348e-04    0.02
```

The three blocks are the same integral against three payoffs and they behave like three different subjects. The smooth expectation is finished before the table starts: the trapezoid rule, nominally the crudest tool present, is at $6.537\times10^{-13}$ with $32$ panels — five orders of magnitude ahead of Simpson's $5.511\times10^{-08}$ at identical cost — and every rule sits on the $3.126\times10^{-13}$ floor by $m=128$. The `order` column is meaningless there because there is no error left to have an order: $0.47$, $0.06$, $0.00$ and the negative $-2.34$ are floating-point noise, and reading them as convergence rates would be the mistake.

The reason the trapezoid rule wins is the second proof below. On an integrand decaying rapidly at both ends of the interval, every term of the Euler–Maclaurin expansion is built from boundary derivatives that are all effectively zero, so the entire asymptotic series vanishes and convergence is exponential rather than quadratic. **The rule advertised as second order is not second order here; it is spectrally accurate, and it becomes second order only when something spoils that structure.**

The kinked row is what spoils it, and it spoils everything equally. Observed orders settle at $2.05$, $2.00$, $2.00$ for the trapezoid rule and $1.97$, $2.00$, $2.00$ for Simpson's — whose advertised fourth order has simply gone — while Gauss–Legendre runs $1.41$, $2.46$, $2.22$ about the same figure. More striking than the orders is the ranking. At $32$ panels the trapezoid rule's error of $7.113\times10^{-03}$ beats Simpson's $1.016\times10^{-01}$ by a factor of $14$ and Gauss–Legendre's $1.664\times10^{-01}$ by a factor of $23$, and Gauss–Legendre remains last at every panel count in the block. **The rule with the highest advertised order is the worst performer on the integrand a derivatives desk evaluates most often, and it is worst for precisely the reason it was best one block above: it commits hardest to smoothness.**

The jump row is the limit of the process. Every rule reaches observed order $1.00$ and stays there, which is all a discontinuity inside the domain permits: the panel straddling the jump contributes an error proportional to its own width however the nodes inside it are arranged. A digital option, a barrier indicator and the probability of breaching a loss limit all live in this row, and for them the trapezoid rule at $9.304\times10^{-04}$ is the best of the three at the largest budget.

## Splitting the Domain at the Kink Restores the Order, and the Split Has to Land Inside a Fraction of a Panel

Nothing above is a defect of the rules. Each is exactly as good as its assumption, and the assumption is piecewise: a call payoff is analytic on each side of the strike and fails to be analytic only across it. Handing the integrator that fact rather than making it discover it is the entire repair.

??? note "Proof that the trapezoid rule's Euler–Maclaurin error runs in even powers of $h$ with coefficients built only from boundary derivatives, so a rapidly decaying analytic integrand converges faster than any power and one interior kink reinstates the $O(h^{2})$ term"

    For $f$ smooth on $[a,b]$ and $h=(b-a)/m$, the Euler–Maclaurin formula gives
    $$T_h[f]-\int_a^b f=\sum_{k=1}^{K}\frac{B_{2k}}{(2k)!}h^{2k}\Big[f^{(2k-1)}(b)-f^{(2k-1)}(a)\Big]+R_K,$$
    with $B_{2k}$ the Bernoulli numbers. Every coefficient is a *difference of derivatives at the two endpoints* and nothing else; the interior contributes only through $R_K$, which is exponentially small for analytic $f$. So if $f$ and all its derivatives are negligible at both endpoints — a Gaussian-weighted integrand truncated at $\pm8$ standard deviations, say — every term in the sum is negligible and the trapezoid rule converges faster than any power of $h$. That is the first block of the table.

    Now let $f$ be smooth on $[a,c]$ and on $[c,b]$ with $f'(c^{-})\ne f'(c^{+})$, and apply the formula to each piece. The interior point $c$ appears in both expansions, contributing at $k=1$
    $$\frac{B_2}{2!}h^{2}\big[f'(c^{-})-f'(c^{+})\big]=\frac{h^{2}}{12}\big[f'(c^{-})-f'(c^{+})\big]\ne0,$$
    which no longer cancels. The error is $O(h^{2})$ with a constant proportional to the jump in the first derivative, and refining $h$ reduces it at that rate and no faster. The same argument caps Simpson's rule and destroys a Gauss rule's factorial, since all of them assume one smooth polynomial model across each panel.

    If instead the partition places $c$ at a panel *endpoint* of both pieces, the two expansions are separate problems, each with a smooth integrand, and the terms at $c$ are ordinary finite boundary quantities the rule's own order handles. The order is restored.

    **The load-bearing quantity is the location of $c$. The expansion says the penalty is governed by the derivative jump at the point where smoothness fails, so the repair is exactly as good as one's knowledge of where that point is — which for a vanilla strike is a term of the contract and for a barrier, a regime threshold or a fitted mixture's breakpoint is a parameter carrying an error bar.**

The restoration and its price are both measurable at a fixed budget:

```python
import numpy as np
from scipy import stats

S0, K, SIG, T = 100.0, 100.0, 0.20, 1.0
LO, HI = -8.0, 8.0
ZSTAR = (np.log(K / S0) + 0.5 * SIG ** 2 * T) / (SIG * np.sqrt(T))


def bs_call():
    d1 = (np.log(S0 / K) + 0.5 * SIG ** 2 * T) / (SIG * np.sqrt(T))
    return S0 * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - SIG * np.sqrt(T))


TRUTH = bs_call()


def payoff(z):
    return np.maximum(S0 * np.exp(-0.5 * SIG ** 2 * T + SIG * np.sqrt(T) * z) - K, 0.0) \
        * stats.norm.pdf(z)


def simp(f, a, b, m):
    x = np.linspace(a, b, m + 1)
    w = np.ones(m + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return (b - a) / (3 * m) * (w * f(x)).sum()


def gauss(f, a, b, m):
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (b - a) * (w * f(0.5 * (b - a) * x + 0.5 * (b + a))).sum()


def split(rule, c, m):
    return rule(payoff, LO, c, m // 2) + rule(payoff, c, HI, m // 2)


print(f"  the same call payoff, integrated in one piece and split at the strike, which"
      f" sits at z = {ZSTAR:.6f}; the panel budget is identical in both columns")
print("     m   Simpson, one piece   order   Simpson, split   order   Gauss, one piece"
      "   order   Gauss, split   order")
prev = {}
for m in (32, 128, 512, 2048):
    row, cells = {}, []
    for tag, val in (("s1", simp(payoff, LO, HI, m)), ("s2", split(simp, ZSTAR, m)),
                     ("g1", gauss(payoff, LO, HI, m)), ("g2", split(gauss, ZSTAR, m))):
        e = max(abs(val - TRUTH), 1e-17)
        row[tag] = e
        o = np.log2(prev[tag] / e) / 2 if tag in prev else np.nan
        cells.append((e, f"{o:5.2f}" if np.isfinite(o) else "     "))
    prev = row
    print(f"    {m:4d}   {cells[0][0]:18.3e}   {cells[0][1]}   {cells[1][0]:14.3e}"
          f"   {cells[1][1]}   {cells[2][0]:16.3e}   {cells[2][1]}"
          f"   {cells[3][0]:12.3e}   {cells[3][1]}")

M = 2048
width = (HI - LO) / M                                  # one Simpson panel
unsplit = abs(simp(payoff, LO, HI, M) - TRUTH)
print(f"  the split point moved off the strike by delta, at a fixed budget of {M:,}"
      f" panels of width {width:.6f}; the strike's true location has to be known to"
      f" place the split")
print("     delta   delta / panel width   Simpson, split at z*+delta"
      "   error relative to never splitting")
for delta in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    e = abs(split(simp, ZSTAR + delta, M) - TRUTH)
    print(f"    {delta:6.0e}   {delta / width:19.4f}   {e:25.3e}"
          f"   {e / unsplit:33.4f}")
# =>   the same call payoff, integrated in one piece and split at the strike, which sits at z = 0.100000; the panel budget is identical in both columns
#         m   Simpson, one piece   order   Simpson, split   order   Gauss, one piece   order   Gauss, split   order
#          32            1.016e-01                9.238e-03                  1.664e-01              1.306e-09        
#         128            6.650e-03    1.97        3.090e-05    4.11          2.366e-02    1.41      2.416e-13    6.20
#         512            4.136e-04    2.00        1.197e-07    4.01          7.809e-04    2.46      2.789e-13   -0.10
#        2048            2.584e-05    2.00        4.669e-10    4.00          3.618e-05    2.22      3.402e-13   -0.14
#      the split point moved off the strike by delta, at a fixed budget of 2,048 panels of width 0.007812; the strike's true location has to be known to place the split
#         delta   delta / panel width   Simpson, split at z*+delta   error relative to never splitting
#         0e+00                0.0000                   4.669e-10                              0.0000
#         1e-05                0.0013                   2.094e-07                              0.0081
#         1e-04                0.0128                   2.054e-06                              0.0795
#         1e-03                0.1280                   1.697e-05                              0.6565
#         1e-02                1.2800                   1.298e-05                              0.5022
#         1e-01               12.8000                   2.221e-05                              0.8594
```

The first table is the proof arriving on schedule. Simpson's rule split at the strike recovers observed orders of $4.11$, $4.01$ and $4.00$ and an error of $4.669\times10^{-10}$ at $2048$ panels, against $2.584\times10^{-05}$ for the identical budget spent in one piece — a factor of $55{,}000$ for rearranging where the nodes fall. Gauss–Legendre does better than that. Split at the strike with $32$ panels it returns $1.306\times10^{-09}$; by $128$ it is at $2.416\times10^{-13}$ and finished. **Thirty-two split Gauss nodes beat two thousand and forty-eight unsplit ones by four orders of magnitude — a sixty-fourfold smaller budget for twenty-seven thousand times the accuracy — and the entire difference is one number the integrator was told.**

The second table prices that number. At $\delta=10^{-3}$, which is $0.1280$ of a single panel width, the error is $1.697\times10^{-05}$, or $0.6565$ of what never splitting would have cost: two thirds of the benefit gone from a misplacement of one part in eight of a panel. Past one panel width the split stops meaning anything, the kink being once again interior to some panel, and the relative errors of $0.5022$ and $0.8594$ are the unsplit regime with scatter. **The repair is not "split the domain" but "know the point," and the tolerance on knowing it is a fraction of the panel width rather than a fraction of the domain.** For a vanilla strike this costs nothing. For a barrier monitored at an estimated level, a regime threshold, or the breakpoint of a fitted mixture, the location carries an error bar, and the table is the exchange rate between that error bar and the accuracy of the price.

## An Adaptive Integrator Estimates Its Error With the Same Nodes That Produced the Answer, So It Can Return Zero and Call It Exact

Sections 2 and 3 concern rates, and a rate is a manageable thing: measurable, and eventually improved by refining. The failure here is not a rate. It is the integrator's own error estimate being wrong in the same direction and by the same mechanism as its answer.

??? note "Proof that an adaptive rule's error estimate is a difference of two rules sharing the same nodes, so an integrand whose mass lies between the nodes produces a zero answer and a zero error estimate together"

    QUADPACK's default routine evaluates the integrand at the $21$ nodes of a Gauss–Kronrod pair, forms the $10$-point Gauss estimate $G$ and the $21$-point Kronrod estimate $\mathcal{K}$ — the Kronrod nodes being the Gauss nodes plus $11$ interlacing ones, so $G$ costs nothing extra — and returns $\mathcal{K}$ with an error estimate monotone in $\lvert\mathcal{K}-G\rvert$. If that estimate exceeds the tolerance the interval is bisected and the procedure recurses on the piece with the largest estimate.

    Both $G$ and $\mathcal{K}$ are linear functionals supported on the same $21$ points. Let $f\ge0$ with $\int f=1$ be supported on a set containing none of them — a density whose width is small against the node spacing. Then $f(x_i)=0$ for every $i$, so
    $$\mathcal{K}=\sum_i w_i^{\mathcal{K}}f(x_i)=0,\qquad G=\sum_i w_i^{G}f(x_i)=0,\qquad\lvert\mathcal{K}-G\rvert=0.$$
    The answer is $0$, the error estimate is $0$, and the bisection that would have found the mass never triggers, because the criterion for subdividing is the very quantity the integrand has driven to zero. Adaptivity offers no protection here; adaptivity is *implemented by* the quantity that failed.

    The condition is a comparison of two lengths. With half-width $L$ and $21$ nodes, adjacent nodes sit roughly $L/10$ apart, so a peak of width $w$ is invisible once $w\ll L/10$ and marginally visible — a wrong answer with a small estimate rather than zero — around $w\sim L/10$.

    **The load-bearing fact is that a self-assessed error compares two members of the same family of approximations, so it measures disagreement about the integrand's shape where the family can see it and is silent where the family cannot, which makes a small reported error evidence of agreement rather than evidence of accuracy.**

The evidence for a strategy's mean return puts a narrow likelihood inside a wide prior, which is that geometry exactly:

```python
import numpy as np
from scipy import integrate, stats

SIG, LO, HI = 0.012, -0.60, 0.60          # daily vol; flat prior on the mean, +-60%
MU_HAT = 0.0006


def evidence(n):
    """A flat prior on the daily mean times a Gaussian likelihood, and its exact value."""
    s = SIG / np.sqrt(n)

    def f(m):
        return stats.norm.pdf(m, MU_HAT, s) / (HI - LO)

    return f, (stats.norm.cdf(HI, MU_HAT, s) - stats.norm.cdf(LO, MU_HAT, s)) / (HI - LO)


def ratio(err, abserr):
    return f"{err / abserr:9.2e}" if abserr > 0 else "  infinite"


print(f"  the evidence for a strategy's mean return: a flat prior on [{LO}, {HI}] times a"
      f" Gaussian likelihood of width {SIG}/sqrt(n), integrated by scipy.integrate.quad"
      f" across the whole prior support")
print("       days   likelihood width / prior width   quad estimate"
      "   quad's reported error   true error   true / reported")
for n in (25, 250, 2_500, 25_000, 250_000):
    f, truth = evidence(n)
    val, abserr = integrate.quad(f, LO, HI, limit=200)
    err = abs(val - truth)
    print(f"    {n:7,d}   {SIG / np.sqrt(n) / (HI - LO):29.2e}   {val:13.6f}"
          f"   {abserr:21.2e}   {err:10.2e}   {ratio(err, abserr):>15s}")

print(f"  the identical integral with the peak's location handed to the integrator as a"
      f" set of break points, which is the entire repair and requires knowing the answer"
      f" well enough to locate it")
print("       days   quad estimate   quad's reported error   true error"
      "   true / reported")
for n in (25, 250, 2_500, 25_000, 250_000):
    f, truth = evidence(n)
    s = SIG / np.sqrt(n)
    val, abserr = integrate.quad(f, LO, HI, points=[MU_HAT - 6 * s, MU_HAT + 6 * s],
                                 limit=200)
    err = abs(val - truth)
    print(f"    {n:7,d}   {val:13.6f}   {abserr:21.2e}   {err:10.2e}"
          f"   {ratio(err, abserr):>15s}")
# =>   the evidence for a strategy's mean return: a flat prior on [-0.6, 0.6] times a Gaussian likelihood of width 0.012/sqrt(n), integrated by scipy.integrate.quad across the whole prior support
#           days   likelihood width / prior width   quad estimate   quad's reported error   true error   true / reported
#             25                        2.00e-03        0.833333                2.38e-12     1.11e-16          4.67e-05
#            250                        6.32e-04        0.833333                2.83e-13     0.00e+00          0.00e+00
#          2,500                        2.00e-04        0.828159                3.44e-09     5.17e-03          1.50e+06
#         25,000                        6.32e-05        0.000000                7.27e-18     8.33e-01          1.15e+17
#        250,000                        2.00e-05        0.000000                0.00e+00     8.33e-01          infinite
#      the identical integral with the peak's location handed to the integrator as a set of break points, which is the entire repair and requires knowing the answer well enough to locate it
#           days   quad estimate   quad's reported error   true error   true / reported
#             25        0.833333                2.83e-09     2.21e-10          7.80e-02
#            250        0.833333                1.94e-10     1.55e-09          7.98e+00
#          2,500        0.833333                8.77e-15     1.64e-09          1.88e+05
#         25,000        0.833333                9.79e-15     1.64e-09          1.68e+05
#        250,000        0.833333                9.32e-15     1.64e-09          1.76e+05
```

The first two rows are the integrator working. At $25$ and $250$ days the likelihood occupies $2.00\times10^{-03}$ and $6.32\times10^{-04}$ of the prior's width, the answer is $0.833333$ to six decimals, and the reported error of $2.38\times10^{-12}$ is a genuine bound. The third row is the transition: at $2{,}500$ days the answer is $0.828159$, wrong in the third decimal, and the reported error of $3.44\times10^{-09}$ understates the true error of $5.17\times10^{-03}$ by a factor of $1.50\times10^{6}$. Nothing about the call changed, no warning was raised, and a caller comparing `abserr` against a tolerance of $10^{-6}$ would have accepted it.

The last two rows are the proof's conclusion in numbers. At $25{,}000$ days the peak occupies $6.32\times10^{-05}$ of the domain, no Gauss–Kronrod node lands inside it, and `quad` returns $0.000000$ with a reported error of $7.27\times10^{-18}$. At $250{,}000$ days it returns $0.000000$ with a reported error of exactly zero. **The integrator's confidence is highest precisely where its answer is entirely wrong, and it is highest for the same reason the answer is wrong: the two rules being differenced agree perfectly that the integrand vanishes, because everywhere they looked it did.** The quantity destroyed is the evidence in a Bayes factor, so a model comparison resting on it would be dividing zero by zero at exactly the sample sizes that make the comparison worth running.

The second table is the repair, and it is honest about what the repair costs. Handing `quad` two break points around the peak restores $0.833333$ in every row, and the price is that the peak's location must be supplied — which for a posterior means having already found the mode, an optimization of the kind [Numerical Optimization](01-numerical-optimization.md) performs, and for a multimodal integrand means having found all of them. Note also that the reported error stays untrustworthy even where the answer is right: at $2{,}500$ days it reads $8.77\times10^{-15}$ against a true error of $1.64\times10^{-09}$, understating by $1.88\times10^{5}$. The answer became correct; the self-assessment did not become reliable.

## Every Repair Here Requires Knowing Something About the Answer, Which Is Why the Control Is Not Optional

The three sections share one shape. Section 2's problem is solved by knowing where the integrand stops being smooth, section 3's by supplying that point to within a fraction of a panel, section 4's by supplying where the mass is. In each case the required information is about the answer, and an integrator asked to work without it returns a number with a reassuring error bar attached.

The wider catalogue is worth naming. A **variable transformation** turns a semi-infinite or singular integral into a finite smooth one — $x=\tan u$ for a Cauchy-like tail, a log scale for a positive parameter, the double-exponential rule for endpoint singularities — and each is a statement about the tail's shape rather than a neutral convenience. **Gauss–Hermite** places its nodes for a Gaussian weight, which makes it the right tool for a normal expectation and the wrong one for a Student-$t$ expectation, because the weight function is part of the assumption. **Richardson extrapolation** and Romberg integration cancel the leading Euler–Maclaurin term, which works beautifully on the smooth row of section 2 and silently returns nonsense on the kinked row, since the expansion being extrapolated does not hold there. In moderate dimension **sparse grids** delay the cost explosion [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md) prices, and past a handful of dimensions the deterministic approach is abandoned entirely, for the reasons [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) measures.

!!! note "The discretization error, the domain-truncation error, the integrator's reported error and the floating-point noise floor are four different quantities, and refining the grid reduces only the first"
    They are conflated because all four arrive in the same units as the answer. The **discretization error** is the gap between the rule and the exact integral over the interval actually used — the only one that responds to more panels, and the one every order in section 2 describes. The **truncation error** is the mass outside that interval, untouched by refinement: the $3.126\times10^{-13}$ floor across the smooth row is this quantity at $\pm8$ standard deviations, and a heavy-tailed integrand cut at the same place would sit orders of magnitude higher with no symptom at all, the rule converging beautifully to the wrong number. The **reported error** is the integrator's self-assessment, a difference between two rules sharing nodes, which section 4 shows can be exactly zero while the true error is $8.33\times10^{-01}$. The **noise floor** is accumulated rounding in the summation and in the integrand itself, which is where the smooth row's negative observed order of $-2.34$ comes from and why an "order" computed near the floor measures nothing. Treating the integrator's reported error as the total error is what this list exists to prevent, and section 4 puts $1.15\times10^{17}$ between them.

!!! warning "An integrator's error estimate is computed by the machinery that computed the answer, so the cases where it is most confident include every case where the integrand hid from it completely"
    Nothing in the failing cases looked wrong. The kinked row of section 2 returned smooth, monotone, plausible errors at every panel count, and the only symptom that Simpson's rule had lost half its order was a figure in an `order` column nobody computes in production. Section 3's misplaced split produced an answer $0.6565$ as bad as never splitting, from a misplacement of one eighth of a panel width. Section 4's integrator returned $0.000000$ for a quantity whose true value is $0.833333$, with a reported error of $7.27\times10^{-18}$ and then of exactly zero, and raised no warning in either row. **The free diagnostic is to integrate something whose answer you already know over the same domain with the same rule and the same settings — the constant $1$, the density's own normalizer, a parity relation between two payoffs — and then to re-run the real integral on a grid offset by half a panel or split at an arbitrary interior point, because a rule that reproduces a known integral and agrees with itself under a shift of its nodes has demonstrated the one thing its own error estimate cannot, which is that it found the mass.** It costs one extra call, and it is exactly the check the trading stake runs before trusting the density it extracted.

## A Rule That Reports the Smoothness It Assumed

This page established that a rule's order is a claim about the integrand's derivatives rather than a property of the rule, an $n$-point Gauss rule buying degree $2n-1$ against an error proportional to a $2n$-th derivative a payoff kink does not possess; that on a smooth expectation every rule beats its billing, the trapezoid rule reaching $6.537\times10^{-13}$ at $32$ panels against Simpson's $5.511\times10^{-08}$ because Euler–Maclaurin's boundary terms all vanish, while one kink at the strike collapses trapezoid, Simpson and Gauss–Legendre alike to observed order $2.00$ and inverts the ranking, Gauss–Legendre's $1.664\times10^{-01}$ trailing the trapezoid rule's $7.113\times10^{-03}$ by a factor of $23$, and a jump takes all three to order $1.00$; that splitting at the strike restores Simpson to order $4.00$ and an error of $4.669\times10^{-10}$ and takes Gauss–Legendre to $1.306\times10^{-09}$ on $32$ panels, while a split misplaced by $0.1280$ of a panel width surrenders $0.6565$ of the benefit and one a full panel away surrenders all of it; and that `scipy.integrate.quad` asked for the evidence of a strategy's mean over a flat prior returns $0.828159$ with a reported error understating the truth by $1.50\times10^{6}$ at $2{,}500$ days, then $0.000000$ with reported errors of $7.27\times10^{-18}$ and exactly zero at $25{,}000$ and $250{,}000$ days, against a true value of $0.833333$.

The shape shared by all three exhibits is that the rule and its error estimate are built from the same polynomial assumption, so they fail together rather than independently. Where the integrand is smooth the assumption holds, the answer is excellent and the estimate is honest. Where it fails — a kink, a jump, a peak between the nodes — the answer degrades and the estimate degrades identically, because the estimate is a comparison between two members of the very family just shown not to contain the integrand. Nothing anywhere in the calculation is computed from the integrand's disagreement with the polynomial model, which is the same structural absence [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) found in the width of a posterior.

This page and [Numerical Optimization](01-numerical-optimization.md) both assumed the function being integrated or maximized was available to evaluate. There is a large and ordinary class of models where it is not. Write down a mixture, a regime model, or any structure containing variables that were never observed, and the likelihood of what *was* observed is itself an integral over what was not — one integral per observation, in a dimension growing with the sample. Neither maximizing that directly nor evaluating it by the rules above is practical, and the standard answer avoids both by alternating between filling in the missing variables and maximizing as though they had been seen. That is [The EM Algorithm](03-em-algorithm.md).

**A quadrature rule returns the integral of the polynomial it fitted rather than of the function you supplied, and reports the disagreement between two such polynomials as its error, so both the answer and the confidence are statements about a family of approximations, and neither becomes a statement about the integrand until something outside the rule has confirmed it.**
