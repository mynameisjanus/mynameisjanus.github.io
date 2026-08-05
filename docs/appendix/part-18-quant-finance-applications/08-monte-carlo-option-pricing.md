# Monte Carlo Option Pricing

[Stochastic Calculus](../../advanced/03-stochastic-calculus.md) measures Euler–Maruyama's strong order at $-0.499$ against Milstein's $-1.005$ and notes in passing that strong convergence "measures pathwise accuracy (as opposed to *weak* convergence, which only requires distributions to match and is what matters for pricing an expectation)." A price is an expectation, so the rate that governs it is the one that page names and declines. This page takes it up and finds that the two errors in a simulated price — a bias from the time step and a standard error from the path count — are traded against each other under one budget, so the optimal split at $4{,}194{,}304$ path-steps is almost all paths and two steps, and both ends of the range cost an order of magnitude in root-mean-square error, $8.13$ and $11.21$ times the best. The sharper half of the page is the derivative rather than the price. Three estimators of the same delta agree on a call to five decimal places, at $0.579134$, $0.579205$ and $0.579182$ against Black–Scholes' $0.579260$; on a digital, one of them returns $0.000000$ with a standard error of $0.000000$ at every maturity, while the truth runs from $0.019552$ to $0.199431$.

This page covers weak against strong convergence and which one a price consumes, the bias–variance budget of a simulated expectation and the convergence rate that discretization costs, the pathwise and likelihood-ratio derivative estimators and the smoothness condition separating them, and what a discontinuous payoff does to each. It does not derive Itô's lemma, construct the stochastic integral, or measure strong-order convergence for Euler and Milstein, all of which are [Stochastic Calculus](../../advanced/03-stochastic-calculus.md); it does not derive the Black–Scholes formula, its Greeks in closed form, the smile or a local-volatility surface, which are [Options Pricing](../../advanced/11-options-pricing.md); it does not build the Monte Carlo estimator or its error bar, which is [Monte Carlo Simulation](../part-09-monte-carlo-methods/03-monte-carlo-simulation.md); it does not develop antithetic variates, control variates, stratification or common random numbers, which are [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md); it does not integrate by quadrature or compare a rule's order against an integrand's smoothness, which is [Numerical Integration](../part-17-statistical-computing/02-numerical-integration.md); it hedges no book and trades no variance swap, which is [Options Pricing](../../advanced/11-options-pricing.md); and it never reports a simulated sensitivity without asking whether the payoff it differentiated was differentiable.

The trading stake is a simulation in a course module that consumes a delta thousands of times and never has to estimate one. [Options Pricing](../../advanced/11-options-pricing.md) delta-hedges a long option along simulated paths and prints `realized 25% vs implied 20%: mean P&L +1.9918, same sign as the gamma prediction +2.1750` against `realized 15% vs implied 20%: mean P&L -1.9913` and a matched-volatility case at `-0.0029`, "statistically zero." Every rebalance in that experiment reads $\Phi(d_1)$ from a closed form. The moment the contract has no closed form — a barrier, a basket, an American feature, any payoff a desk actually books — that number has to come out of the same simulation that produced the price, and sections 3 and 4 are about which of the three available ways of getting it is safe.

## Weak Convergence Is What a Price Consumes, and Bias Costs a Convergence Rate Rather Than a Constant

A simulated price carries two errors with different origins and different rates, and the useful fact is not that both exist but that spending on one starves the other.

??? note "Proof that a price is governed by weak order, that the root-mean-square error at a fixed budget is minimized at a step count growing like the cube root of the budget, and that the achievable error then falls like the cube root rather than the square root"

    A European price is $\mathbb{E}[f(S_T)]$, a functional of the terminal *distribution* alone. The relevant notion is therefore **weak** convergence, $\lvert\mathbb{E}[f(\hat S_T)]-\mathbb{E}[f(S_T)]\rvert\le c\,\Delta t^{\beta}$, and for Euler–Maruyama with smooth enough $f$ the weak order is $\beta=1$ — twice the strong order $1/2$ that [Stochastic Calculus](../../advanced/03-stochastic-calculus.md) measures, because the pathwise errors that dominate strong convergence are mean-zero and cancel in an expectation.

    Fix a compute budget of $C$ path-steps, split into $N$ paths of $M$ steps with $NM=C$, and write $\sigma_p^{2}$ for the variance of one path's discounted payoff. The two errors are a bias $cT/M$ and a standard error $\sigma_p/\sqrt N=\sigma_p\sqrt{M/C}$, so
    $$\mathrm{RMSE}^{2}(M)=\left(\frac{cT}{M}\right)^{2}+\frac{\sigma_p^{2}M}{C}.$$
    Differentiating, the optimum satisfies $M^{3}=2c^{2}T^{2}C/\sigma_p^{2}$, so $M^{\ast}\propto C^{1/3}$ and, substituting back, $\mathrm{RMSE}^{\ast}\propto C^{-1/3}$.

    That exponent is the result worth carrying. A Monte Carlo estimator with no discretization bias converges at $C^{-1/2}$ in total work; introducing a bias that must be bought down alongside the variance degrades the *rate* to $C^{-1/3}$, so quadrupling the budget buys a factor of $1.59$ rather than $2$. The bias does not merely add a constant to be tolerated; it changes what more computation is worth.

    **The load-bearing hypothesis is that $f$ is smooth enough for weak order $1$ to hold. Discontinuous payoffs degrade $\beta$, and since $M^{\ast}$ and the achievable error both depend on $\beta$, a barrier or a digital sits at a different optimum and a worse rate — which is the same smoothness condition that sections 3 and 4 show governing the derivative estimators.**

## At a Fixed Budget the Optimal Split Is Almost All Paths, and Both Ends Cost an Order of Magnitude

The algebra says there is an interior optimum. Where it actually falls for an ordinary contract is worth measuring, because the answer is more lopsided than the symmetry of the formula suggests.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18081)
S0, K, R, VOL, T, REPS = 100.0, 100.0, 0.02, 0.20, 1.0, 400
BUDGET = 2 ** 22                                        # paths x steps, held fixed


def bs_call(s0, k, r, vol, t):
    d1 = (np.log(s0 / k) + (r + vol ** 2 / 2) * t) / (vol * np.sqrt(t))
    return s0 * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d1 - vol * np.sqrt(t))


def euler_price(n_paths, n_steps):
    """Euler-Maruyama on dS = rS dt + vol S dW, which is where the bias comes from."""
    dt = T / n_steps
    s = np.full(n_paths, S0)
    for _ in range(n_steps):
        s = s + R * s * dt + VOL * s * np.sqrt(dt) * rng.standard_normal(n_paths)
    return np.exp(-R * T) * np.maximum(s - K, 0.0)


exact = bs_call(S0, K, R, VOL, T)
print(f"  a European call priced by Euler-Maruyama at a fixed compute budget of"
      f" {BUDGET:,} path-steps, split every way. Discretization bias falls like the step and"
      f" sampling error like the square root of the path count, so the two are traded against"
      f" each other. Black-Scholes says {exact:.4f}. {REPS} independent runs per split")
print("     steps   paths      mean price      bias   sampling sd   RMSE   RMSE / best")
rows = []
for n_steps in (1, 2, 4, 8, 16, 64, 256):
    n_paths = BUDGET // n_steps
    est = np.array([euler_price(n_paths, n_steps).mean() for _ in range(REPS)])
    bias = est.mean() - exact
    rmse = np.sqrt(bias ** 2 + est.var())
    rows.append((n_steps, n_paths, est.mean(), bias, est.std(), rmse))
best = min(r[5] for r in rows)
for n_steps, n_paths, m, bias, sd, rmse in rows:
    print(f"    {n_steps:5d}   {n_paths:7d}   {m:13.4f}   {bias:+7.4f}   {sd:11.4f}"
          f"   {rmse:6.4f}   {rmse / best:11.2f}")
# =>   a European call priced by Euler-Maruyama at a fixed compute budget of 4,194,304 path-steps, split every way. Discretization bias falls like the step and sampling error like the square root of the path count, so the two are traded against each other. Black-Scholes says 8.9160. 400 independent runs per split
#         steps   paths      mean price      bias   sampling sd   RMSE   RMSE / best
#            1   4194304          8.8400   -0.0760        0.0060   0.0763          8.13
#            2   2097152          8.9139   -0.0022        0.0091   0.0094          1.00
#            4   1048576          8.9225   +0.0064        0.0137   0.0151          1.61
#            8    524288          8.9202   +0.0041        0.0186   0.0190          2.03
#           16    262144          8.9201   +0.0040        0.0264   0.0267          2.85
#           64     65536          8.9156   -0.0004        0.0529   0.0529          5.64
#          256     16384          8.9184   +0.0024        0.1051   0.1051         11.21
```

The U-shape is there and it is steep on both sides, but its floor sits at *two* time steps. At one step the bias is $-0.0760$ and dominates everything, giving $8.13$ times the best error; at $256$ steps the bias has vanished into the noise and the sampling standard deviation of $0.1051$ gives $11.21$ times the best. In between, the optimum spends $2{,}097{,}152$ paths against two steps.

The practical reading is uncomfortable for a habit. Refining the time grid is the intuitive way to make a simulated price more accurate, and it is the wrong lever for a smooth European payoff on this model: every step past the second is bought by halving the path count, and the bias it removes was already smaller than the noise it adds. The reason is visible in the bias column, which collapses from $-0.0760$ to $-0.0022$ in a single doubling — weak order $1$ with a small constant. **A budget split is a modelling decision with an order-of-magnitude consequence, it has a closed-form answer requiring only a bias estimate and a payoff variance, and it is almost always made by typing a round number of steps.**

## Three Estimators of the Same Derivative, and Only One of Them Leaves the Payoff Alone

A price is what a simulation is asked for; a hedge ratio is what it is used for. There are three ways to differentiate a Monte Carlo expectation, they are not interchangeable, and the condition separating them is a property of the payoff rather than of the model.

??? note "Proof that the pathwise estimator is unbiased exactly when the payoff is Lipschitz, that the likelihood-ratio estimator is unbiased for any integrable payoff, and that its variance diverges like $1/(\sigma^{2}T)$"

    Write the price as $P(\theta)=\mathbb{E}[f(S_T(\theta))]$ where $\theta$ is the parameter being differentiated. Two routes exist because the parameter can be moved into either the payoff or the density.

    The **pathwise** estimator differentiates inside the expectation, $P'(\theta)=\mathbb{E}[f'(S_T)\,\partial_\theta S_T]$. Interchanging derivative and expectation requires dominated convergence, for which Lipschitz continuity of $f$ suffices and without which it can fail outright. A call payoff $(S-K)^{+}$ is Lipschitz with derivative $\mathbf 1\{S>K\}$ almost everywhere, so the estimator $e^{-rT}\mathbf 1\{S_T>K\}S_T/S_0$ is unbiased. An indicator payoff $\mathbf 1\{S>K\}$ is *not* Lipschitz: it is constant except at one point, so $f'=0$ almost everywhere and the interchange returns identically zero — not an approximation to the delta but a different number, with no variance to reveal the problem.

    The **likelihood-ratio** estimator moves the parameter into the density instead, writing $P(\theta)=\int f(x)p_\theta(x)\,dx$ and differentiating there:
    $$P'(\theta)=\int f(x)\,\partial_\theta p_\theta(x)\,dx=\mathbb{E}\!\left[f(S_T)\,\partial_\theta\log p_\theta(S_T)\right].$$
    The payoff is never differentiated, so no smoothness is required — integrability suffices — and the estimator is unbiased for a digital exactly as for a call. For geometric Brownian motion differentiated in the spot, the score is $Z/(S_0\sigma\sqrt T)$ with $Z$ the driving standard normal, giving the estimator $e^{-rT}f(S_T)Z/(S_0\sigma\sqrt T)$.

    That score is also the cost. Its second moment is $1/(S_0^{2}\sigma^{2}T)$, so the estimator's variance grows without bound as $\sigma\sqrt T\to0$: the shorter the maturity, the less the terminal density moves when the spot is bumped, and the more violently the ratio must be scaled to detect it. The likelihood-ratio estimator is the general one and the expensive one, and it is worst exactly where digitals are most traded.

    **The load-bearing distinction is where the parameter is allowed to live. Putting it in the payoff needs the payoff to be differentiable; putting it in the density needs the density to be known. A simulation that has one of those and not the other has exactly one valid estimator, and nothing in the output announces which case it is in.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18083)
S0, K, R, VOL, T, N, REPS = 100.0, 100.0, 0.02, 0.20, 1.0, 200_000, 200
H = 0.01 * S0                                           # bump for the finite difference


def d1d2(s0):
    d1 = (np.log(s0 / K) + (R + VOL ** 2 / 2) * T) / (VOL * np.sqrt(T))
    return d1, d1 - VOL * np.sqrt(T)


def terminal(z, s0):
    return s0 * np.exp((R - VOL ** 2 / 2) * T + VOL * np.sqrt(T) * z)


def deltas(z):
    """Three estimators of dPrice/dS0 for a European call, on one set of normals."""
    disc = np.exp(-R * T)
    s = terminal(z, S0)
    fd = disc * (np.maximum(terminal(z, S0 + H) - K, 0)
                 - np.maximum(terminal(z, S0 - H) - K, 0)) / (2 * H)
    pathwise = disc * (s > K) * s / S0
    lr = disc * np.maximum(s - K, 0) * z / (S0 * VOL * np.sqrt(T))
    return fd, pathwise, lr


exact = stats.norm.cdf(d1d2(S0)[0])
print(f"  delta of a European call by three Monte Carlo estimators, {N:,} paths x {REPS} runs."
      f" The finite difference bumps the spot and reuses the same normals; the pathwise estimator"
      f" differentiates the payoff along the path; the likelihood-ratio estimator differentiates"
      f" the density instead and never touches the payoff. Black-Scholes delta is {exact:.6f}")
print("     estimator          mean      bias   standard error   sd relative to pathwise")
runs = {k: [] for k in ("finite difference", "pathwise", "likelihood ratio")}
for _ in range(REPS):
    z = rng.standard_normal(N)
    for name, est in zip(runs, deltas(z)):
        runs[name].append(est.mean())
base = np.std(runs["pathwise"])
for name, vals in runs.items():
    v = np.array(vals)
    print(f"    {name:18s} {v.mean():9.6f}  {v.mean() - exact:+8.6f}   {v.std():14.6f}"
          f"   {v.std() / base:23.1f}")
# =>   delta of a European call by three Monte Carlo estimators, 200,000 paths x 200 runs. The finite difference bumps the spot and reuses the same normals; the pathwise estimator differentiates the payoff along the path; the likelihood-ratio estimator differentiates the density instead and never touches the payoff. Black-Scholes delta is 0.579260
#         estimator          mean      bias   standard error   sd relative to pathwise
#        finite difference   0.579134  -0.000125         0.001264                       1.0
#        pathwise            0.579205  -0.000055         0.001285                       1.0
#        likelihood ratio    0.579182  -0.000077         0.003114                       2.4
```

On a call all three work and the theory's ordering is confirmed. The means are $0.579134$, $0.579205$ and $0.579182$ against an exact $0.579260$, biases of $-0.000125$, $-0.000055$ and $-0.000077$; the finite difference and the pathwise estimator have essentially identical standard errors, and the likelihood-ratio estimator pays $2.4$ times as much for the same answer, which is the price of ignoring the payoff's structure. On this evidence a desk would use the pathwise estimator and never think about the other two again.

The finite difference deserves one note, because its performance here is bought rather than free: it reuses the same normals on both sides of the bump, which is [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md)'s common random numbers doing exactly what that page proves it does — pricing a difference rather than two levels. Without that reuse its standard error would be several orders of magnitude larger and it would not appear in this table at all.

!!! note "A finite difference, a pathwise derivative, a likelihood-ratio score and an adjoint are four ways to get a Greek out of a simulation, and they fail in four different places"
    **A finite difference** re-prices at bumped parameters and needs no analysis at all, which is why it is what everyone writes first; its error has the $O(h)+O(\epsilon/h)$ structure [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md) analyses, and it degrades when the bump is small relative to the payoff's own scale of variation. **A pathwise derivative** differentiates the payoff along the simulated path, is the cheapest and lowest-variance option, and is invalid the moment the payoff is not Lipschitz. **A likelihood-ratio score** differentiates the density, works for any payoff including discontinuous ones, and has a variance that blows up as the density becomes insensitive — short maturities, low volatility. **An adjoint** computes all sensitivities in one backward pass at the cost of roughly one forward pass, which is what makes large Greek vectors tractable, and it inherits the pathwise estimator's smoothness requirement exactly because it is that estimator computed efficiently. The choice is usually made on implementation convenience, and only the third and fourth entries have a correctness condition that a test on a call would reveal.

## On a Digital the Pathwise Estimator Returns Zero With a Standard Error of Zero

Section 3's table is what a validation suite looks like when it is built around a vanilla contract. Repeating it on a payoff with a jump in it separates the three estimators completely.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(18085)
S0, K, R, VOL, N, REPS = 100.0, 100.0, 0.02, 0.20, 200_000, 200
H = 0.01 * S0


def exact_digital_delta(t):
    d2 = (np.log(S0 / K) + (R - VOL ** 2 / 2) * t) / (VOL * np.sqrt(t))
    return np.exp(-R * t) * stats.norm.pdf(d2) / (S0 * VOL * np.sqrt(t))


def estimators(z, t):
    """The same three estimators applied to a cash-or-nothing digital paying 1 if S_T > K."""
    disc = np.exp(-R * t)
    term = lambda s0: s0 * np.exp((R - VOL ** 2 / 2) * t + VOL * np.sqrt(t) * z)
    fd = disc * ((term(S0 + H) > K).astype(float) - (term(S0 - H) > K)) / (2 * H)
    pathwise = np.zeros_like(z)          # d/dS0 of an indicator is zero wherever it exists
    lr = disc * (term(S0) > K) * z / (S0 * VOL * np.sqrt(t))
    return fd, pathwise, lr


print(f"  the same three estimators on a digital paying 1 if the spot finishes above {K:.0f}."
      f" The payoff is an indicator, so it is not Lipschitz and the pathwise estimator's"
      f" interchange of derivative and expectation is invalid. {N:,} paths x {REPS} runs")
print("     maturity   exact delta   finite difference: mean    s.e.   pathwise: mean    s.e."
      "   likelihood ratio: mean    s.e.   LR s.e. relative to 1y")
base = None
for t in (1.00, 0.25, 0.05, 0.01):
    cols, ex = [], exact_digital_delta(t)
    runs = [[], [], []]
    for _ in range(REPS):
        z = rng.standard_normal(N)
        for lst, est in zip(runs, estimators(z, t)):
            lst.append(est.mean())
    fd, pw, lr = (np.array(r) for r in runs)
    if base is None:
        base = lr.std()
    print(f"    {t:8.2f}   {ex:11.6f}   {fd.mean():24.6f}   {fd.std():.6f}"
          f"   {pw.mean():15.6f}   {pw.std():.6f}   {lr.mean():22.6f}   {lr.std():.6f}"
          f"   {lr.std() / base:23.1f}")
# =>   the same three estimators on a digital paying 1 if the spot finishes above 100. The payoff is an indicator, so it is not Lipschitz and the pathwise estimator's interchange of derivative and expectation is invalid. 200,000 paths x 200 runs
#         maturity   exact delta   finite difference: mean    s.e.   pathwise: mean    s.e.   likelihood ratio: mean    s.e.   LR s.e. relative to 1y
#            1.00      0.019552                   0.019535   0.000212          0.000000   0.000000                 0.019557   0.000064                       1.0
#            0.25      0.039695                   0.039643   0.000292          0.000000   0.000000                 0.039696   0.000144                       2.3
#            0.05      0.089117                   0.088425   0.000430          0.000000   0.000000                 0.089092   0.000266                       4.2
#            0.01      0.199431                   0.191465   0.000546          0.000000   0.000000                 0.199357   0.000715                      11.2
```

The pathwise column is the honest failure and it is the worst kind available: at every maturity it reports $0.000000$ with a standard error of $0.000000$, against true deltas of $0.019552$, $0.039695$, $0.089117$ and $0.199431$. The estimator is not noisy, not biased in the usual sense, and not unstable. It is a perfectly precise measurement of a quantity that is not the delta, and its own error bar — the diagnostic any Monte Carlo pipeline computes automatically — confirms it to arbitrary precision, because every path agrees. Increasing the path count improves the reported precision and does not move the number. This is the mechanism [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md) found in a different guise, a routine reporting confidence about the wrong object, and here the confidence is exact.

The other two columns are a genuine trade rather than a failure. The likelihood-ratio estimator is unbiased at every maturity — $0.019557$, $0.039696$, $0.089092$ and $0.199357$ against the exact values — and its standard error grows by $11.2$ times from one year to a fortnight, exactly the $1/(\sigma\sqrt T)$ divergence the proof predicts. The finite difference is the reverse: cheap and accurate at a year, then increasingly biased as maturity shortens, reading $0.191465$ against $0.199431$ at $T=0.01$ because a $1\%$ bump is no longer small relative to the width over which the digital's value turns. **One estimator degrades in variance, one in bias, and one returns a number with no error at all, and only the third would pass a test that checks whether the answer is stable.**

## Every Repair Is a Smoother Payoff, a Different Estimator, or a Budget Split Nobody Computes

The three findings admit repairs of very different quality. The budget split of section 2 has a closed-form answer and needs two inputs a pilot run already produces — a bias estimate at two step counts and the payoff's variance — so the only reason it is not computed is that nobody looks for it. The digital's delta has a standard repair too: smooth the payoff into a tight call spread, which restores the pathwise estimator's Lipschitz condition and introduces a bias controlled by the spread's width, converting an invalid estimator into a biased one whose error is a chosen parameter rather than an unknown.

The estimator choice itself does not have a universal answer, which is the honest position. Pathwise where the payoff is Lipschitz, likelihood ratio where it is not, and the two combined — pathwise for the smooth part of a payoff and likelihood ratio for the jump — where a contract has both. What is not defensible is choosing on convenience and validating on a call, because a call is precisely the case in which all three agree.

!!! warning "A Monte Carlo pipeline reports the standard error of whatever it computed, and a wrong estimator with no variance reports the strongest confidence in the system"
    Every Monte Carlo result arrives with an error bar, and the error bar answers one question: how much would this number move with different random draws. Section 4 produces a delta whose answer is $0.000000$ and whose error bar is $0.000000$, so a pipeline that flags results by their standard error will rank this one as its most reliable output. **The free diagnostic is to differentiate a contract whose answer is known — the same estimator on a plain call, where Black–Scholes supplies the truth — and, where no closed form exists, to run two estimators with different failure modes and compare: agreement between a pathwise and a likelihood-ratio estimate is evidence, and a disagreement of $0.019552$ against $0.000000$ names which one broke.** It costs one extra estimator over paths that have already been generated, since both are functions of the same normals, and it is the only check in this part that catches an error whose reported uncertainty is exactly zero.

## Two Errors in a Price and Three Ways to Differentiate It

This page established that a price is governed by weak rather than strong convergence, so Euler–Maruyama's relevant order is $1$ rather than the $1/2$ measured elsewhere, and that trading discretization bias against sampling error under a fixed budget puts the optimal step count at $C^{1/3}$ and the achievable error at $C^{-1/3}$ rather than $C^{-1/2}$; that the optimum for a European call at $4{,}194{,}304$ path-steps sits at two steps and $2{,}097{,}152$ paths, with one step costing $8.13$ times the best root-mean-square error through a bias of $-0.0760$ and $256$ steps costing $11.21$ times it through a sampling deviation of $0.1051$; that the pathwise estimator is unbiased exactly when the payoff is Lipschitz while the likelihood-ratio estimator needs only integrability and pays a variance diverging like $1/(\sigma^{2}T)$, confirmed on a call where all three agree at $0.579134$, $0.579205$ and $0.579182$ against $0.579260$ with the likelihood ratio $2.4$ times noisier; and that on a digital the pathwise estimator returns $0.000000$ with a standard error of $0.000000$ at every maturity against true deltas of $0.019552$ to $0.199431$, while the likelihood-ratio estimator stays unbiased and grows $11.2$ times noisier and the finite difference drifts to $0.191465$ against $0.199431$.

The symmetry with [Numerical Integration](../part-17-statistical-computing/02-numerical-integration.md) is close enough to be worth stating, because the two pages price the same defect in different currencies. That page found a quadrature rule's order to be a claim about the integrand's derivatives rather than a property of the rule, so a payoff kink collapsed every rule to observed order $2.00$ and inverted their ranking. This page finds a derivative estimator's validity to be a claim about the payoff's smoothness rather than a property of the estimator, so a payoff jump collapses the pathwise estimator to zero and inverts the ranking again. In both cases the method's advertised rate is a statement about a function it was never shown, and in both cases the vanilla test case is the one where the distinction is invisible. What this page has priced is a single instrument; the next asks the same question of a book, where the inputs are a covariance matrix rather than a volatility and the error being propagated is estimation rather than discretization.

**A simulated price reports a standard error for its sampling and none for its discretization, and a simulated Greek reports a standard error for both while sometimes differentiating the wrong thing entirely.**
