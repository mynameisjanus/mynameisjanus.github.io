# Numerical Optimization

An optimizer is usually described as the thing that finds the maximum, and the description hides that it finds a point where a stopping rule fired. The two coincide on exactly one class of surface, and the class is smaller than the set of models a desk fits. Below, the same portfolio problem takes $85$ gradient-descent iterations with two sleeves and $77{,}561$ with six, because adding a high-volatility sleeve moved the Hessian's condition number from $11$ to $9{,}363$ — while Newton's method takes $2$ iterations in every row, and dividing each weight by its own volatility, a change of variables requiring no new information, brings gradient descent back to $30$. Then the part that is not a speed problem. A two-regime mixture fitted from $400$ random starts reaches the honest optimum on $0.9300$ of them and reports `success` there on $0.3172$, while the $0.0250$ of starts that collapse onto a zero-width spike report `success` on $0.7000$ and carry the highest log-likelihood on the page, $1592.1922$ against the honest optimum's $1584.4864$. The recipe that takes the best likelihood over random restarts selects that spike, and the spike puts the one-per-cent daily loss at $-2.4013\%$ where the truth is $-2.8774\%$.

This page covers what a convergence flag certifies and what a vanishing gradient certifies without concavity, the condition number as the single quantity a first-order method responds to and the change of variables that removes it, the behaviour of a general-purpose optimizer on a surface with several optima and no upper bound, and what a finite-difference gradient costs when the objective is a simulation. It derives no likelihood theory and computes no Fisher information, which is [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md); it takes no posterior mode and derives no penalized-likelihood identity, which is [Maximum A Posteriori Estimation](../part-11-parameter-estimation/06-maximum-a-posteriori-estimation.md); it runs no iteratively reweighted least squares and measures no quadratic convergence on a concave surface, which is [Generalized Linear Models](../part-13-regression/03-generalized-linear-models.md); it solves no penalized least squares in closed form, which is [Regularization](../part-13-regression/05-regularization.md); it builds no surrogate over an objective too expensive to evaluate, which is [Bayesian Optimization for Hyperparameters](../../advanced/01-bayesian-optimization.md); it exploits no latent-variable structure to replace one hard maximization by a sequence of easy ones, which is [The EM Algorithm](03-em-algorithm.md); it evaluates no integral, which is [Numerical Integration](02-numerical-integration.md); it draws no sample from anything, which is [Markov Chain Monte Carlo](04-markov-chain-monte-carlo.md); it derives no differentiation rules and states no Taylor theorem, which is [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md); and it never reads a converged flag as evidence about anything except the arithmetic that produced it.

The trading stake is a course lesson that reformulated a problem specifically to avoid section 3 and said so in one clause. [Risk Parity, Diversification, and Factors](../../part-08-portfolio-management/03-risk-parity-diversification-factors.md) prints `-- SPY/TLT/GLD: SLSQP converged True in 22 iterations, risk shares span 33.3% to 33.3%`, and the paragraph above it explains why the objective being solved is not the obvious one: the direct "minimize the squared dispersion of risk contributions" objective "is **not convex**, and SLSQP will happily report success from a local minimum that pins assets against their bounds." The log-barrier form $\tfrac12 w^{\top}\Sigma w-\tfrac1n\sum_i\log w_i$ has a unique solution, so `converged True` means something there and would not otherwise. That sentence is this page's section 3, written by someone who had already been bitten.

## A Convergence Flag Reports Which Stopping Rule Fired, and the Condition It Checks Is Necessary Rather Than Sufficient

Every optimizer in general use terminates on one of three tests: the gradient norm fell below a tolerance, the step length fell below a tolerance, or the objective stopped changing. All three are local, all three hold at a saddle and at a shallow local optimum, and with a numerical gradient all three hold at a point where the arithmetic merely stopped being informative.

??? note "Proof that on a strictly concave objective a vanishing gradient certifies the global maximum and Newton's method converges quadratically, and that dropping concavity leaves the identical test certifying only that the point is not locally improvable"

    Let $f$ be twice continuously differentiable on a convex set with $\nabla^{2}f(\theta)\preceq-mI$ for some $m>0$ throughout. Taylor's theorem gives, for any $\theta$ and $\theta^{\ast}$ and some $\xi$ on the segment between them,
    $$f(\theta^{\ast})=f(\theta)+\nabla f(\theta)^{\top}(\theta^{\ast}-\theta)+\tfrac12(\theta^{\ast}-\theta)^{\top}\nabla^{2}f(\xi)(\theta^{\ast}-\theta).$$
    If $\nabla f(\theta)=0$ the linear term vanishes and the quadratic term is at most $-\tfrac{m}{2}\lVert\theta^{\ast}-\theta\rVert^{2}\le0$, so $f(\theta^{\ast})\le f(\theta)$ for *every* $\theta^{\ast}$, however distant. The stationary point is the global maximum and it is unique, since a second one would contradict strict concavity.

    Newton's iteration $\theta_{k+1}=\theta_k-[\nabla^{2}f(\theta_k)]^{-1}\nabla f(\theta_k)$ satisfies, under a Lipschitz condition $\lVert\nabla^{2}f(\theta)-\nabla^{2}f(\theta')\rVert\le L\lVert\theta-\theta'\rVert$,
    $$\lVert\theta_{k+1}-\theta^{\ast}\rVert\le\frac{L}{2m}\lVert\theta_k-\theta^{\ast}\rVert^{2},$$
    so once the error is below $2m/L$ it squares at every step and the number of correct digits doubles. This is why the measured Newton count in section 2 is $2$ and is not a function of anything.

    Now delete the assumption $\nabla^{2}f\preceq-mI$. The Taylor identity survives verbatim and the linear term still vanishes at a stationary point, but the quadratic term is no longer signed, so $f(\theta^{\ast})>f(\theta)$ becomes permissible for $\theta^{\ast}$ arbitrarily far away — and, where the curvature is indefinite, for $\theta^{\ast}$ arbitrarily close. The test the optimizer ran is unchanged; what changed is what passing it means.

    **The load-bearing point is that concavity is a statement about the whole domain while every stopping rule is a statement about one neighbourhood, so the optimizer cannot check the hypothesis its own guarantee requires, and the flag has exactly the same value on a surface where the guarantee is a theorem and on a surface where it is a coincidence.**

## The Condition Number Is a Property of the Coordinates, and It Is the Only Thing a First-Order Method Responds To

Gradient descent moves along the steepest direction in the coordinates it was handed, and steepest depends on the coordinates. Newton's method moves along the steepest direction in the metric the curvature itself defines, which is a coordinate-free object.

??? note "Proof that gradient descent's error contracts by $(\kappa-1)/(\kappa+1)$ per step while Newton's step is equivariant under any invertible linear change of variables, so conditioning is a property of one algorithm and of the chart, and not of the problem"

    Take $f(\theta)=\tfrac12\theta^{\top}A\theta$ with $A\succ0$ having eigenvalues in $[m,M]$ and $\kappa=M/m$. Gradient descent with the optimal fixed step $2/(m+M)$ satisfies
    $$\lVert\theta_{k+1}-\theta^{\ast}\rVert\le\frac{\kappa-1}{\kappa+1}\lVert\theta_k-\theta^{\ast}\rVert,$$
    obtained by diagonalizing $A$ and observing that eigencoordinate $i$ contracts by $\lvert1-\alpha\lambda_i\rvert$, whose maximum over $\lambda_i\in[m,M]$ is attained at the endpoints. Reaching a fixed accuracy therefore costs $O(\kappa\log(1/\varepsilon))$ iterations — linear in $\kappa$, which is the growth measured below.

    Now substitute $\theta=S\eta$ for invertible $S$. The gradient becomes $S^{\top}\nabla f$ and the Hessian $S^{\top}\nabla^{2}f\,S$, so the Newton step in the new coordinates is
    $$-\big(S^{\top}\nabla^{2}f\,S\big)^{-1}S^{\top}\nabla f=-S^{-1}\big(\nabla^{2}f\big)^{-1}\nabla f,$$
    exactly $S^{-1}$ applied to the old step. The iterates correspond under $S$ and the iteration count is identical. The gradient step $-\alpha S^{\top}\nabla f$ has no such property, and $\kappa(S^{\top}AS)$ can be driven to $1$ by taking $S=A^{-1/2}$ or made arbitrarily large by taking $S$ badly.

    A desk chooses $S$ without noticing whenever it decides what one unit of weight means. Sleeves quoted in the same currency but differing in volatility by a factor $v$ give $A$ eigenvalues differing by roughly $v^{2}$, and dividing each weight by its own volatility is the diagonal $S$ that undoes it.

    **The load-bearing asymmetry is that the condition number measures the chart the problem was written in rather than any difficulty the problem has, so a first-order method's cost is partly a consequence of a reporting convention nobody recorded, and exactly one of the two methods below is charging for it.**

The mechanism is visible by adding one sleeve at a time, each more volatile than the last:

```python
import numpy as np
from scipy import optimize

rng = np.random.default_rng(17011)
n, d = 4_000, 6
VOLS = np.array([0.0008, 0.0025, 0.0060, 0.0110, 0.0240, 0.0700])   # cash .. crypto
MU = np.array([0.00002, 0.00006, 0.00018, 0.00030, 0.00055, 0.00120])
C = 0.30 * np.ones((d, d)) + 0.70 * np.eye(d)
R = rng.multivariate_normal(MU, np.outer(VOLS, VOLS) * C, n)


def newton_iters(f, g, H, w0, target):
    w, k = w0.copy(), 0
    while f(w) - target > 1e-12 and k < 200:
        w -= np.linalg.solve(H(w), g(w))
        k += 1
    return k


def gd_iters(f, g, H, w0, target, cap=400_000):
    w, k = w0.copy(), 0
    step = 1.0 / np.linalg.eigvalsh(H(w0)).max()      # the best fixed step there is
    while f(w) - target > 1e-12 and k < cap:
        w -= step * g(w)
        k += 1
    return k


print(f"  log-growth weights on up to {d} sleeves, {n:,} days, stopped when the objective"
      f" is within 1e-12 of the optimum; 'preconditioned' rescales each sleeve by its own"
      f" volatility")
print("     sleeves   volatility span   condition number   gradient descent   BFGS"
      "   Newton   preconditioned kappa   preconditioned gradient descent")
for k in (2, 3, 4, 5, 6):
    Rk, s = R[:, :k], VOLS[:k]

    def fk(w, Rk=Rk):
        return -np.log1p(Rk @ w).mean()

    def gk(w, Rk=Rk):
        return -(Rk / (1.0 + Rk @ w)[:, None]).mean(0)

    def Hk(w, Rk=Rk):
        z = Rk / (1.0 + Rk @ w)[:, None]
        return z.T @ z / n

    def fp(u, s=s):
        return fk(u / s)

    def gp(u, s=s):
        return gk(u / s) / s

    def Hp(u, s=s):
        return Hk(u / s) / np.outer(s, s)

    ws = np.zeros(k)
    for _ in range(60):                                # optimum to machine precision
        ws -= np.linalg.solve(Hk(ws), gk(ws))
    tgt = fk(ws)
    bf = optimize.minimize(fk, np.zeros(k), jac=gk, method="BFGS",
                           options={"gtol": 1e-12, "maxiter": 5_000}).nit
    print(f"    {k:8d}   {s.max() / s.min():14.1f}x"
          f"   {np.linalg.cond(Hk(np.zeros(k))):17,.0f}"
          f"   {gd_iters(fk, gk, Hk, np.zeros(k), tgt):16,d}   {bf:4d}"
          f"   {newton_iters(fk, gk, Hk, np.zeros(k), tgt):6d}"
          f"   {np.linalg.cond(Hp(np.zeros(k))):21.1f}"
          f"   {gd_iters(fp, gp, Hp, np.zeros(k), tgt):31,d}")
# =>   log-growth weights on up to 6 sleeves, 4,000 days, stopped when the objective is within 1e-12 of the optimum; 'preconditioned' rescales each sleeve by its own volatility
#         sleeves   volatility span   condition number   gradient descent   BFGS   Newton   preconditioned kappa   preconditioned gradient descent
#               2              3.1x                  11                 85     29        2                     1.8                                11
#               3              7.5x                  65                500     57        2                     2.1                                15
#               4             13.7x                 239              1,623     70        2                     2.6                                18
#               5             30.0x               1,113              8,244     85        2                     3.0                                24
#               6             87.5x               9,363             77,561     96        2                     3.5                                30
```

The volatility span and the condition number move together, $3.1$ to $87.5$ against $11$ to $9{,}363$, which is the squaring the proof predicts. Gradient descent tracks the condition number almost exactly — $85$, $500$, $1{,}623$, $8{,}244$ and $77{,}561$ iterations, a factor of $912$ across a problem that gained four variables — and it is being run with the best fixed step size that exists for it, so nothing here is a tuning failure. Newton takes $2$ iterations in every row. BFGS, which builds an approximation to the inverse Hessian out of successive gradients and is what `scipy.optimize` runs when no Hessian is supplied, sits between them at $29$ to $96$: it pays for the conditioning logarithmically rather than linearly, which is the practical reason it is the default.

The last two columns are the part to keep. Dividing each weight by its own sleeve volatility is a diagonal change of variables requiring no new information — the volatilities are already estimated, and the transformation is the one every risk system performs to report positions in risk units rather than dollars. It takes the condition number from $9{,}363$ to $3.5$ and gradient descent from $77{,}561$ iterations to $30$. **Nothing about the portfolio problem became easier; the arithmetic was being done in a chart where the level sets are spheres rather than needles, and the entire cost differential was the chart.**

This is also the honest reading of the checklist item [Calculus Essentials](../part-01-mathematical-foundations/06-calculus-essentials.md) records as "scale your variables." It is not hygiene. On this problem it is worth a factor of two and a half thousand to a first-order method and exactly nothing to a second-order one, and knowing which is underneath is what tells you whether to spend the afternoon.

## On a Non-Concave Surface the Highest Likelihood Found Is the Worst Answer Available, and the Optimizer Reports Success More Often When It Has Found It

Conditioning is a cost, and costs can be paid. What follows cannot, because paying more of it does not help. A two-component mixture — the smallest model that says returns have a calm regime and a stress regime — has a likelihood surface with several stationary points and no upper bound at all, and the standard response is to run the optimizer from many starting values and keep the best.

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(17013)
n, starts = 500, 400
P, MU, SD = 0.80, (0.0003, -0.0016), (0.0075, 0.0165)     # calm / stress days
x = np.where(rng.random(n) < P,
             rng.normal(MU[0], SD[0], n), rng.normal(MU[1], SD[1], n))
gap = np.diff(np.sort(x)).min()


def parts(t):
    """Log-weighted component densities; t = (logit p, m1, log s1, m2, log s2)."""
    return (-np.logaddexp(0.0, -t[0]) + stats.norm.logpdf(x, t[1], np.exp(t[2])),
            -np.logaddexp(0.0, t[0]) + stats.norm.logpdf(x, t[3], np.exp(t[4])))


def negll(t):
    a, b = parts(t)
    return -np.logaddexp(a, b).sum()


def grad(t):
    a, b = parts(t)
    w = np.exp(a - np.logaddexp(a, b))                    # responsibilities
    p, s1, s2 = 1.0 / (1.0 + np.exp(-t[0])), np.exp(t[2]), np.exp(t[4])
    z1, z2 = (x - t[1]) / s1, (x - t[3]) / s2
    return -np.array([(w - p).sum(), (w * z1).sum() / s1, (w * (z1 ** 2 - 1)).sum(),
                      ((1 - w) * z2).sum() / s2, ((1 - w) * (z2 ** 2 - 1)).sum()])


def var99(p, m1, s1, m2, s2):
    """The 1% daily loss a two-component mixture implies, in per cent."""
    def F(q):
        return p * stats.norm.cdf(q, m1, s1) + (1 - p) * stats.norm.cdf(q, m2, s2) - 0.01

    return optimize.brentq(F, -1.0, 1.0, xtol=1e-14) * 100


sd0 = x.std()
fun, ok, v, smin = [], [], [], []
for _ in range(starts):
    t0 = np.array([rng.normal(0, 1.5),
                   rng.normal(0, sd0), np.log(sd0) + rng.normal(0, 0.8),
                   rng.normal(0, sd0), np.log(sd0) + rng.normal(0, 0.8)])
    r = optimize.minimize(negll, t0, jac=grad, method="BFGS", options={"gtol": 1e-6})
    fun.append(r.fun)
    ok.append(r.success)
    v.append(var99(1 / (1 + np.exp(-r.x[0])), r.x[1], np.exp(r.x[2]),
                   r.x[3], np.exp(r.x[4])))
    smin.append(np.exp(min(r.x[2], r.x[4])))
fun, ok, v, smin = np.array(fun), np.array(ok), np.array(v), np.array(smin)

deg = smin < gap
best = fun[~deg].min()
glob = ~deg & (fun - best <= 0.05)
truth = var99(P, MU[0], SD[0], MU[1], SD[1])

print(f"  two-component mixture on {n} days, {starts} random starts, BFGS with analytic"
      f" gradients; the quantity to be sized on is the 1% daily loss the fit implies,"
      f" whose true value is {truth:.4f}%")
print("     termination                  share of starts   reported success"
      "   best log-likelihood   1% daily loss")
for lab, m in (("global interior optimum", glob),
               ("other interior optimum", ~deg & ~glob),
               ("degenerate spike", deg)):
    lo, hi = v[m].min(), v[m].max()
    span = f"{lo:.4f}%" if hi - lo < 5e-5 else f"{lo:.4f}% to {hi:.4f}%"
    print(f"    {lab:24s}   {m.mean():15.4f}   {ok[m].mean():16.4f}"
          f"   {-fun[m].min():19.4f}   {span:>25s}")
print(f"    the single highest likelihood over all {starts} starts is {-fun.min():.4f},"
      f" reached at a fitted standard deviation {smin[fun.argmin()] / gap:.6f} times the"
      f" smallest gap between two days, and it reports success={bool(ok[fun.argmin()])}")
# =>   two-component mixture on 500 days, 400 random starts, BFGS with analytic gradients; the quantity to be sized on is the 1% daily loss the fit implies, whose true value is -2.8774%
#         termination                  share of starts   reported success   best log-likelihood   1% daily loss
#        global interior optimum             0.9300             0.3172             1584.4864                    -3.1147%
#        other interior optimum              0.0450             0.3333             1566.7130        -2.5017% to -2.4450%
#        degenerate spike                    0.0250             0.7000             1592.1922        -2.4714% to -2.4013%
#        the single highest likelihood over all 400 starts is 1592.1922, reached at a fitted standard deviation 0.000000 times the smallest gap between two days, and it reports success=False
```

Read the `best log-likelihood` column against the `1% daily loss` column. The honest interior optimum, reached from $0.9300$ of starts, has log-likelihood $1584.4864$ and puts the one-per-cent daily loss at $-3.1147\%$ against a true $-2.8774\%$ — an ordinary estimation error on five hundred days. The degenerate spikes, reached from $0.0250$ of starts, have log-likelihood $1592.1922$. **They beat the honest answer by $7.71$ nats and they are not answers at all: the fitted standard deviation of the narrow component is $0.000000$ times the smallest gap between two adjacent observations, meaning that component has collapsed onto a single day and is asserting that day occurs with certainty.** The likelihood is unbounded in that direction, so $1592.1922$ records where the arithmetic stopped rather than where the surface did, and running longer makes it larger.

The operational consequence is that "restart from many points and keep the best likelihood" — the recipe [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) applies with five restarts, and most fitting code applies with more — selects the spike whenever a spike is found, because selecting the maximum is precisely what it does. Four hundred starts find one on $0.0250$ of tries, so five starts find one about twelve per cent of the time. The spike puts the one-per-cent daily loss at $-2.4013\%$, understating the truth by $47$ basis points of capital per day, in the direction that licenses more leverage.

The `reported success` column is the sentence to keep. The optimizer flags success on $0.3172$ of the starts that found the honest optimum and on $0.7000$ of the starts that collapsed onto a spike, because a spike is a direction of unbounded ascent along which the gradient genuinely does go to zero in the coordinates being used, while the honest optimum sits on a flat, badly conditioned ridge where a tolerance is hard to meet. **The flag is not merely uninformative about which answer was found; on this surface it is anti-correlated with it, and the single highest-likelihood point of all four hundred reports `success=False`.**

## A Difference Quotient Divides by a Number You Chose and Subtracts Two Numbers You Did Not, and a Simulated Objective Makes the Subtraction Meaningless

Sections 2 and 3 assumed a gradient was available in closed form. Most objectives a desk writes have no such thing, and the default substitute introduces a second tuning parameter with a failure mode at each end.

??? note "Proof that the forward difference has total error $O(h)+O(\epsilon/h)$ minimized at $h\asymp\sqrt{\epsilon}$ and the central difference $O(h^{2})+O(\epsilon/h)$ minimized at $h\asymp\epsilon^{1/3}$, so the attainable accuracy is $\sqrt{\epsilon}$ and $\epsilon^{2/3}$ and never better"

    Let $f$ be evaluated with absolute error at most $\epsilon$ — about $10^{-16}$ for a well-scaled objective in double precision, and the Monte Carlo standard error for an objective estimated by simulation. Taylor expansion gives
    $$\frac{f(\theta+h)-f(\theta)}{h}=f'(\theta)+\tfrac{h}{2}f''(\theta)+O(h^{2}),$$
    so the truncation error is $O(h)$, while the two evaluation errors contribute at most $2\epsilon/h$. Minimizing $\tfrac{h}{2}\lvert f''\rvert+2\epsilon/h$ over $h$ gives $h^{\ast}\asymp\sqrt{\epsilon}$ and a floor of order $\sqrt{\epsilon}$. The central difference cancels the $f''$ term,
    $$\frac{f(\theta+h)-f(\theta-h)}{2h}=f'(\theta)+\tfrac{h^{2}}{6}f'''(\theta)+O(h^{4}),$$
    leaving truncation $O(h^{2})$ against the same $\epsilon/h$, so $h^{\ast}\asymp\epsilon^{1/3}$ with a floor of order $\epsilon^{2/3}$.

    The two terms move oppositely in $h$, so the error curve is U-shaped and neither end is safe: a step too large reports the slope of a chord, and a step too small reports the difference of two numbers that agree to within their own error. Substituting $\epsilon=10^{-16}$ gives $h^{\ast}\approx10^{-8}$ and $10^{-5}$ with floors near $10^{-8}$ and $10^{-11}$, which is what the first table measures.

    **The load-bearing substitution is $\epsilon$. For a closed form it is machine precision and the floors are excellent; for an objective estimated from $m$ simulated paths it is of order $m^{-1/2}$, and the same formulas then give a gradient floor of order $m^{-1/4}$ — so a hundredfold increase in simulation effort buys about a factor of three in gradient accuracy, and buying enough accuracy to optimize is not a plan.**

The first half is arithmetic and the second half is what happens when the objective is a backtest:

```python
import numpy as np
from scipy import optimize

rng = np.random.default_rng(17015)
n, d = 3_000, 4
VOLS = np.array([0.0025, 0.0060, 0.0110, 0.0240])
MU = np.array([0.00006, 0.00018, 0.00030, 0.00055])
C = 0.25 * np.ones((d, d)) + 0.75 * np.eye(d)
R = rng.multivariate_normal(MU, np.outer(VOLS, VOLS) * C, n)


def f(w):
    """Expected log growth of a portfolio held at weights w."""
    return -np.log1p(R @ w).mean()


def g(w):
    return -(R / (1.0 + R @ w)[:, None]).mean(0)


def H(w):
    z = R / (1.0 + R @ w)[:, None]
    return z.T @ z / n


w_star = np.zeros(d)
for _ in range(60):
    w_star -= np.linalg.solve(H(w_star), g(w_star))
w0 = w_star * 0.60
exact = g(w0)


def fd(fun, w, h, central):
    out = np.empty(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        out[i] = ((fun(w + e) - fun(w - e)) / (2 * h) if central
                  else (fun(w + e) - fun(w)) / h)
    return out


def shortfall(w):
    """Annualized log growth given up against the optimum, in per cent."""
    return (f(w) - f(w_star)) * 252 * 100


print(f"  gradient of the log-growth objective at a fixed point, {d} sleeves, {n:,} days;"
      f" the analytic gradient is exact and the differences are compared against it")
print("     step size h   forward difference, relative error   central difference,"
      " relative error")
for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    ef = np.abs(fd(f, w0, h, False) - exact).max() / np.abs(exact).max()
    ec = np.abs(fd(f, w0, h, True) - exact).max() / np.abs(exact).max()
    print(f"    {h:12.0e}   {ef:35.3e}   {ec:38.3e}")

print(f"  the identical objective estimated by resampling {n:,} days instead of averaging"
      f" over all of them; 'common random numbers' freezes one resample and reuses it at"
      f" every evaluation, 'fresh draws' resamples on each call")
print("     paths per evaluation   noise sd of f   fresh draws: gradient error"
      "   fresh: success   fresh: shortfall   common random numbers: success"
      "   common random numbers: shortfall")
for m in (2_000, 20_000, 200_000):
    fixed = R[rng.integers(0, n, m)]

    def f_crn(w, fixed=fixed):
        return -np.log1p(fixed @ w).mean()

    def f_fresh(w, m=m):
        return -np.log1p(R[rng.integers(0, n, m)] @ w).mean()

    noise = np.std([f_fresh(w0) for _ in range(200)])
    err = np.abs(fd(f_fresh, w0, 1e-6, True) - exact).max() / np.abs(exact).max()
    a = optimize.minimize(f_fresh, np.zeros(d), method="BFGS", options={"maxiter": 400})
    b = optimize.minimize(f_crn, np.zeros(d), method="BFGS", options={"maxiter": 400})
    print(f"    {m:21,d}   {noise:13.3e}   {err:26.3e}   {str(bool(a.success)):>14s}"
          f"   {shortfall(a.x):16.4f}%   {str(bool(b.success)):>29s}"
          f"   {shortfall(b.x):33.4f}%")
# =>   gradient of the log-growth objective at a fixed point, 4 sleeves, 3,000 days; the analytic gradient is exact and the differences are compared against it
#         step size h   forward difference, relative error   central difference, relative error
#               1e-02                             1.292e-02                                1.318e-07
#               1e-04                             1.292e-04                                1.477e-11
#               1e-06                             1.291e-06                                2.538e-10
#               1e-08                             8.410e-08                                6.977e-08
#               1e-10                             7.025e-06                                1.909e-06
#               1e-12                             4.888e-04                                2.575e-04
#      the identical objective estimated by resampling 3,000 days instead of averaging over all of them; 'common random numbers' freezes one resample and reuses it at every evaluation, 'fresh draws' resamples on each call
#         paths per evaluation   noise sd of f   fresh draws: gradient error   fresh: success   fresh: shortfall   common random numbers: success   common random numbers: shortfall
#                        2,000       3.895e-04                    1.248e+06            False            11.5932%                            True                             44.8456%
#                       20,000       1.298e-04                    4.515e+05            False            11.5932%                            True                              3.9558%
#                      200,000       3.782e-05                    2.165e+05            False            11.5932%                            True                              0.1985%
```

The first table is the U-shape the proof predicts and it is worth reading in both directions. The forward difference falls exactly in proportion to $h$ — $1.292\times10^{-2}$, $1.292\times10^{-4}$, $1.291\times10^{-6}$ — bottoms out at $8.410\times10^{-8}$ near $h=10^{-8}$, then rises again to $4.888\times10^{-4}$ at $h=10^{-12}$, worse than it was at $h=10^{-4}$. The central difference falls as $h^{2}$ and reaches $1.477\times10^{-11}$ at $h=10^{-4}$, three orders of magnitude better than anything the forward difference achieves at any step, for one extra evaluation per coordinate. **A step size chosen to be "small enough to be safe" is the failure at the bottom of both columns, and the instinct that smaller is more accurate is wrong by four orders of magnitude by $h=10^{-12}$.**

The second table replaces machine precision with a Monte Carlo standard error and the differences stop working at all. At two thousand resampled days the objective carries noise of $3.895\times10^{-4}$, and a central difference at $h=10^{-6}$ returns a gradient whose relative error is $1.248\times10^{6}$ — not an inaccurate gradient but a random vector. BFGS handed that gradient never takes a step: its line search rejects every candidate, because the objective disagrees with itself when re-evaluated, so it returns the starting point, reports `success=False`, and leaves $11.5932\%$ of annualized log growth on the table in all three rows. Raising the simulation budget a hundredfold moves the gradient error from $1.248\times10^{6}$ to $2.165\times10^{5}$ and changes the outcome not at all, which is the $m^{-1/4}$ floor from the proof arriving on schedule.

The repair is in the last two columns and it is not more computation. Freezing the resample — evaluating every candidate parameter against the *same* draw, which is the common random numbers of [Variance Reduction](../part-09-monte-carlo-methods/06-variance-reduction.md) — makes the objective a deterministic function of the parameter, and BFGS then optimizes it normally, reporting `success=True` and closing the shortfall from $44.8456\%$ to $3.9558\%$ to $0.1985\%$ as the frozen sample grows. **The randomness did not get smaller; it stopped moving between evaluations, and that is the only property a difference quotient ever needed.** What is left is ordinary sampling error in the frozen draw, which shrinks at the usual rate and is a statistical problem rather than a numerical one.

## Every Repair on This Page Is a Change of Coordinates, a Change of the Objective, or a Restart

The three sections have three fixes and they are the same fix in different clothes. Preconditioning changes the chart so the level sets are round. Freezing the randomness changes the objective so it has a derivative. Restarting from many points changes the search so the answer is not a function of one arbitrary guess. None of them makes the optimizer better and all of them make the problem different, which is the only thing that has ever worked.

The catalogue past this page is worth naming. A **trust region** replaces the line search with a bound on the step and minimizes a quadratic model inside it, which is what makes second-order methods safe when the Hessian is indefinite — section 3's situation exactly — and is why `trust-constr` and `dogleg` sit beside `BFGS`. **Constrained problems** replace the first-order condition with the Karush–Kuhn–Tucker conditions, and SLSQP solves a quadratic program at each iterate; the move the trading stake makes, trading a natural non-convex objective for an artificial convex one with the same solution, is the standard and usually the only reliable one. **Standard errors** come free from the inverse Hessian at the optimum, since the curvature the optimizer already computed is the observed information of [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md) — and at section 3's spike that curvature is meaningless in precisely the way the point is. Where the objective is expensive rather than noisy, a surrogate replaces restarts, which is [Bayesian Optimization for Hyperparameters](../../advanced/01-bayesian-optimization.md) and is not built here.

!!! note "The optimizer's tolerance, the objective's evaluation noise, the parameter's standard error and the model's identifiability are four different scales, and a fit means something only when they are ordered"
    They are routinely confused because all four carry units of the parameter or of the objective. The **optimizer's tolerance** is how precisely the stopping rule was met — the $10^{-12}$ target in section 2, the `gtol` of section 3 — and it is a property of the arithmetic alone. The **evaluation noise** is how precisely the objective is known at all: machine precision for a closed form, a Monte Carlo standard error otherwise, and section 4 shows that a tolerance below it is both unreachable and meaningless. The **standard error** is how precisely the data determines the parameter, read off the inverse Hessian, and it is normally many orders of magnitude larger than either — arguing about the eighth digit of an estimate whose second digit is uncertain is the most common form this confusion takes. **Identifiability** is whether the parameter is determined at all in the limit, and it is not a small number but a yes or a no: section 3's spike is a failure of identifiability, and no tolerance, no extra precision and no larger sample repairs it. Reporting an optimizer's tolerance as though it were a standard error is the error this list exists to prevent, and section 2 puts ten orders of magnitude between the first two.

!!! warning "A converged flag certifies that a stopping rule fired in one neighbourhood, and on the surfaces a desk actually fits it fires most reliably at the answers that are worst"
    Nothing in the failing cases looked wrong. Section 2's gradient descent converged correctly in every row and took $77{,}561$ iterations to do what Newton did in $2$, with no message indicating that a change of units was worth a factor of two and a half thousand. Section 3's optimizer reported `success` on $0.7000$ of the starts that collapsed onto a zero-width spike against $0.3172$ of the starts that found the honest optimum, and the highest likelihood over four hundred starts, $1592.1922$, belonged to a fit whose narrow component had standard deviation $0.000000$ times the gap between two adjacent days. Section 4's optimizer never moved at all and said so, which is the only honest flag on the page — and a reader who inspected only the returned parameter would have published the starting value as an optimum. **The free diagnostic is to restart from a few dozen randomized starting values and report the spread of the quantity you will actually size on — the tail loss, the weight, the regime volatility — rather than the spread of the log-likelihood, and to discard any solution whose smallest fitted scale parameter falls below the smallest gap between two adjacent observations, because the likelihood is unbounded in that direction and the maximum you are selecting is an artefact of where the arithmetic stopped.** It costs one loop around code you have already written, and it is the only check on this page that does not require knowing the answer.

## A Number That Answers a Question About the Stopping Rule

This page established that a vanishing gradient certifies a global maximum only under concavity while the identical test is run either way, so the flag carries the same weight where its guarantee is a theorem and where it is a coincidence; that the condition number is a property of the coordinates rather than of the problem, gradient descent costing $85$, $500$, $1{,}623$, $8{,}244$ and $77{,}561$ iterations as a volatility span of $3.1$ grew to $87.5$ and the condition number from $11$ to $9{,}363$, while Newton took $2$ in every row and a diagonal rescaling requiring no new information brought the condition number to $3.5$ and gradient descent to $30$; that a two-component mixture fitted from $400$ starts reached the honest optimum from $0.9300$ of them at log-likelihood $1584.4864$ and a one-per-cent daily loss of $-3.1147\%$ against a truth of $-2.8774\%$, while $0.0250$ collapsed onto zero-width spikes carrying a higher log-likelihood of $1592.1922$ and a loss of $-2.4013\%$, with `success` reported on $0.7000$ of the spikes against $0.3172$ of the honest fits; and that a difference quotient has a U-shaped error with floors of $8.410\times10^{-8}$ and $1.477\times10^{-11}$ in closed form, degrading to a relative error of $1.248\times10^{6}$ once the objective is estimated from resampled data, where BFGS never takes a step and leaves $11.5932\%$ of annualized growth unclaimed until the randomness is frozen and the shortfall falls to $0.1985\%$.

The shape shared by all three exhibits is that the optimizer is a correct algorithm applied to whatever function it was handed, and every failure above is a property of the handoff rather than of the algorithm. The condition number came from a choice of units, the spike came from a likelihood with no maximum, and the useless gradient came from an objective that returns a different number each time it is asked. In each case the optimizer did what it promised, and the promise was about the arithmetic.

What every section here shares is that the answer was a point. A point is what a maximization returns, and it is the wrong shape for most of the questions this appendix asks, because a probability, an expectation and a tail quantile are integrals rather than maxima. An integral has an error structure with nothing in common with the one measured above: its accuracy is bought against the smoothness of the integrand rather than the curvature of a surface, and the integrands a trading book produces — a payoff with a kink at the strike, a loss distribution truncated at a limit, an indicator inside an expectation — are precisely the ones without the smoothness. That is [Numerical Integration](02-numerical-integration.md).

**An optimizer reports the point at which it stopped improving a function it was told nothing else about, so a converged flag is a fact about the arithmetic, the tolerance and the coordinates, and it becomes a fact about the model only under an assumption the flag cannot check and the surface will not volunteer.**
