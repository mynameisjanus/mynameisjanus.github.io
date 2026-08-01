# Calculus Essentials

Continuous probability is calculus. A density is a function whose *integral* gives probability, an expectation is an integral against that density, maximum likelihood is a first-order condition, and portfolio optimization is a constrained minimization solved with multipliers. This page collects the calculus those results assume — differentiation, integration, Taylor expansion, and optimization with and without constraints — with each tool pointed at the place in the book that consumes it.

The treatment is operational. Limits were built in [Sequences and Infinite Series](04-sequences-and-series.md) and are used here without ceremony; the matrix identities come from [Basic Linear Algebra Review](05-linear-algebra-review.md).

## Limits and Continuity

The limit of a function extends the sequence definition to a continuum: $\lim_{x\to a}f(x) = L$ means that for every $\epsilon>0$ there is a $\delta>0$ such that $0<\lvert x-a\rvert<\delta$ implies $\lvert f(x)-L\rvert<\epsilon$. The tolerance game is the same; only the "eventually" changed from "far enough along the sequence" to "close enough to $a$".

A function is **continuous** at $a$ when $\lim_{x\to a}f(x) = f(a)$ — the limit exists, the value exists, and they agree. Continuity is exactly the property that lets limits pass through functions, which is why it is the hypothesis of the [Continuous Mapping Theorem](../part-07-asymptotic-theory/06-continuous-mapping-theorem.md) and why the CDF of a continuous random variable can be differentiated to recover its density.

Discontinuities are not exotic in this material. The CDF of a discrete random variable is a step function, continuous from the right with a jump at each atom, and the size of the jump is the probability mass there ([CDFs](../part-03-random-variables/02-cumulative-distribution-functions.md)). Payoff functions are frequently continuous but not differentiable: $\max(S-K, 0)$ has a kink at the strike, which is exactly where an option's delta jumps.

## Derivatives

The **derivative** of $f$ at $x$ is the limit of the difference quotient:

$$f'(x) = \frac{df}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h},$$

the instantaneous rate of change, and geometrically the slope of the tangent line. Its most useful reading for this book is the *local linear approximation*: for small $h$,

$$f(x+h)\approx f(x) + f'(x)\,h.$$

Nearly every "to first order" argument in quantitative finance — an option's delta, a bond's duration, the delta method's variance formula — is that one line applied to a different $f$.

| Rule | Statement |
|---|---|
| Power | $(x^n)' = nx^{n-1}$ |
| Exponential | $(e^x)' = e^x$ |
| Logarithm | $(\log x)' = 1/x$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $(f/g)' = (f'g - fg')/g^2$ |
| Chain | $\big(f(g(x))\big)' = f'(g(x))\,g'(x)$ |

The chain rule is the one that does the heavy lifting. It is why $\frac{d}{d\theta}\log L(\theta) = L'(\theta)/L(\theta)$ — the *score* function of maximum likelihood — and it is the entirety of backpropagation in [Deep Learning](../../part-07-machine-learning/03-deep-learning.md), applied recursively through a composition of layers.

### Critical Points and Curvature

At an interior maximum or minimum of a differentiable $f$, the derivative vanishes: $f'(x^*) = 0$. Such points are **critical**, and the second derivative classifies them — $f''(x^*)>0$ means a local minimum (the function curves upward), $f''(x^*)<0$ a local maximum, and $f''(x^*)=0$ is inconclusive.

The condition is necessary, not sufficient, and the failure modes matter in practice: a critical point may be a saddle, an extremum may sit on the boundary of the domain where the derivative need not vanish, and a function may have many local optima of which only one is global. Likelihood surfaces in mixture and regime models have all three pathologies, which is why the [EM Algorithm](../part-17-statistical-computing/03-em-algorithm.md) is run from multiple starting points.

### Numerical Derivatives

When $f$ is available only as code, the derivative is approximated by a difference quotient with a finite step. The choice of step is a trade-off between two errors: truncation error, which shrinks with $h$, and floating-point cancellation, which grows as $h$ shrinks and the subtraction loses significant digits.

```python
import math

x0, exact = 1.0, math.exp(1.0)

for h in (1e-1, 1e-2, 1e-4, 1e-8, 1e-12):
    forward = (math.exp(x0 + h) - math.exp(x0)) / h
    central = (math.exp(x0 + h) - math.exp(x0 - h)) / (2 * h)
    print(f"h={h:<7.0e} forward err {abs(forward - exact):.3e}"
          f"   central err {abs(central - exact):.3e}")
# => h=1e-01   forward err 1.406e-01   central err 4.533e-03
#    h=1e-02   forward err 1.364e-02   central err 4.530e-05
#    h=1e-04   forward err 1.359e-04   central err 4.531e-09
#    h=1e-08   forward err 6.603e-09   central err 6.603e-09
#    h=1e-12   forward err 4.323e-04   central err 2.103e-04
```

Two facts, both worth internalizing. The **central difference is second-order accurate** — its error falls by $10^4$ when $h$ falls by $10^2$, against the forward difference's factor of $10^2$ — so it costs one extra function evaluation and buys several digits. And **smaller is not better past a point**: at $h=10^{-12}$ both methods are *worse* than at $h=10^{-4}$, because subtracting two nearly equal doubles discards most of the mantissa. The optimum sits near $h\approx\sqrt{\varepsilon_{\text{machine}}}\approx10^{-8}$ for the forward difference and $h\approx\varepsilon^{1/3}\approx10^{-5}$ for the central one. Anything that finite-differences a gradient — numerical optimizers, sensitivity analyses, "bump and revalue" risk — is living on this curve.

## Taylor Expansion

If $f$ is sufficiently smooth near $a$,

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n,$$

with the remainder $R_n = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}$ for some $\xi$ between $a$ and $x$. In words: near a point, a smooth function is a polynomial, and the remainder tells you how much the truncation costs.

Three expansions appear throughout the book:

$$e^x = 1 + x + \frac{x^2}{2} + \cdots,\qquad \log(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots,\qquad (1+x)^\alpha = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2}x^2 + \cdots$$

The second governs the relationship between simple and log returns, and its accuracy is worth seeing rather than assuming:

```python
import math

for x in (0.001, 0.01, 0.05, 0.20, 0.50):
    exact = math.log1p(x)
    first, second = x, x - x ** 2 / 2
    print(f"r={x:<6} log(1+r)={exact: .6f}   1st-order err {abs(first - exact):.2e}"
          f"   2nd-order err {abs(second - exact):.2e}")
# => r=0.001  log(1+r)= 0.001000   1st-order err 5.00e-07   2nd-order err 3.33e-10
#    r=0.01   log(1+r)= 0.009950   1st-order err 4.97e-05   2nd-order err 3.31e-07
#    r=0.05   log(1+r)= 0.048790   1st-order err 1.21e-03   2nd-order err 4.02e-05
#    r=0.2    log(1+r)= 0.182322   1st-order err 1.77e-02   2nd-order err 2.32e-03
#    r=0.5    log(1+r)= 0.405465   1st-order err 9.45e-02   2nd-order err 3.05e-02
```

At a daily return of 1% the first-order approximation $\log(1+r)\approx r$ is off by 5 basis points of the return itself — negligible for one day, and not negligible when compounded over a decade. At a monthly 20% it is off by 1.8 percentage points, which is the difference between a strategy that looks good and one that looks great. [Exponentials, Logarithms, and Growth](07-exponentials-logarithms-growth.md) works through when the distinction bites.

!!! note "The delta method is a Taylor expansion with a variance taken"
    Let $\hat\theta$ be an estimator with mean $\theta$ and variance $\sigma^2$, and suppose the quantity of interest is $g(\hat\theta)$. Expanding to first order about $\theta$,

    $$g(\hat\theta)\approx g(\theta) + g'(\theta)(\hat\theta-\theta),$$

    and taking the variance of both sides gives $\mathrm{var}\big(g(\hat\theta)\big)\approx \big(g'(\theta)\big)^2\sigma^2$. That is the entire [Delta Method](../part-07-asymptotic-theory/04-delta-method.md) — the standard error of a transformed estimate, obtained by linearizing the transform. It is how a confidence interval for a variance becomes one for a volatility, and how a Sharpe ratio's standard error is derived.

## Integration

The **definite integral** $\int_a^b f(x)\,dx$ is the limit of Riemann sums: partition $[a,b]$, multiply each subinterval's width by a sample value of $f$, add, and refine. It measures signed area under the curve.

The **fundamental theorem of calculus** ties the two operations together. If $F' = f$ then

$$\int_a^b f(x)\,dx = F(b)-F(a),$$

and conversely $\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$. Differentiation and integration are inverse operations, which is precisely the relationship between a CDF and a PDF:

$$F_X(x) = \int_{-\infty}^{x}f_X(t)\,dt,\qquad f_X(x) = F_X'(x).$$

Two techniques carry most calculations. **Substitution** is the chain rule in reverse, $\int f(g(x))g'(x)\,dx = \int f(u)\,du$ — the mechanism behind every change-of-variables formula for densities ([Change of Variables](../part-03-random-variables/09-change-of-variables.md)), where the Jacobian factor is exactly the $g'(x)$ above. **Integration by parts**, $\int u\,dv = uv - \int v\,du$, comes from the product rule and is how the gamma function's recursion $\Gamma(n+1) = n\Gamma(n)$ — hence $\Gamma(n+1)=n!$ — is established.

**Improper integrals** extend the definition to infinite limits or unbounded integrands, as limits of proper integrals: $\int_{-\infty}^{\infty}f = \lim_{a\to-\infty,\,b\to\infty}\int_a^b f$. Every continuous distribution's normalization is an improper integral, and the ones that fail to converge are the ones with undefined moments — the Cauchy density integrates to 1 but $\int \lvert x\rvert f(x)\,dx$ diverges, so it has no mean.

### Why Densities Integrate to One

A continuous random variable's density satisfies $f_X\ge0$ and $\int_{-\infty}^{\infty}f_X(x)\,dx = 1$, and probabilities of intervals are areas:

$$\mathbf{P}(a\le X\le b) = \int_a^b f_X(x)\,dx.$$

Individual points have probability zero, since $\int_a^a f = 0$ — the fact anticipated by the uncountability argument in [Sets and Functions](01-sets-and-functions.md). Expectation replaces the discrete sum with an integral against the density:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty}x\,f_X(x)\,dx,\qquad \mathbb{E}[g(X)] = \int_{-\infty}^{\infty}g(x)f_X(x)\,dx,$$

the second being the law of the unconscious statistician, which spares you from deriving the density of $g(X)$ before taking its expectation. Both are developed in [Expected Value](../part-04-expectation-and-moments/01-expected-value.md).

### The Gaussian Integral

The normalizing constant of the normal distribution comes from

$$\int_{-\infty}^{\infty}e^{-x^2/2}\,dx = \sqrt{2\pi}.$$

There is no elementary antiderivative of $e^{-x^2/2}$, which is why the normal CDF has no closed form and is evaluated numerically everywhere it appears. The definite integral nonetheless has an exact value, by a trick worth seeing once.

??? note "Proof by polar coordinates"
    Let $I = \int_{-\infty}^{\infty}e^{-x^2/2}dx$ and consider its square as a double integral over the plane:

    $$I^2 = \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\,dx\,dy.$$

    Switch to polar coordinates with $x = r\cos\phi$, $y = r\sin\phi$, so $x^2+y^2 = r^2$ and the area element becomes $r\,dr\,d\phi$:

    $$I^2 = \int_0^{2\pi}\!\!\int_0^{\infty}e^{-r^2/2}\,r\,dr\,d\phi = 2\pi\int_0^{\infty}re^{-r^2/2}\,dr = 2\pi\Big[-e^{-r^2/2}\Big]_0^{\infty} = 2\pi.$$

    Hence $I = \sqrt{2\pi}$. The factor of $r$ supplied by the polar area element is what makes the radial integral elementary — the same substitution that fails in one dimension succeeds in two.

Dividing by $\sqrt{2\pi}$ gives the standard normal density $\varphi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$, and the general case follows by substituting $x\mapsto(x-\mu)/\sigma$.

## Multivariable Calculus

For $f:\mathbb{R}^n\to\mathbb{R}$, the **partial derivative** $\partial f/\partial x_i$ differentiates with respect to one coordinate holding the others fixed. Stacking them gives the **gradient**

$$\nabla f(x) = \left(\frac{\partial f}{\partial x_1},\ldots,\frac{\partial f}{\partial x_n}\right)^{\!\top},$$

a vector pointing in the direction of steepest increase, with the linear approximation $f(x+h)\approx f(x) + \nabla f(x)^\top h$. The matrix of second partials is the **Hessian** $H_{ij} = \partial^2f/\partial x_i\partial x_j$, symmetric for smooth $f$.

The one-dimensional classification generalizes through the Hessian's definiteness, using the vocabulary of [Basic Linear Algebra Review](05-linear-algebra-review.md): at a critical point where $\nabla f = 0$, a positive definite Hessian means a local minimum, negative definite a local maximum, and indefinite — eigenvalues of both signs — a saddle. A function is **convex** when its Hessian is PSD everywhere, and convexity is the property that makes optimization tractable: a convex function has no local minima that are not global, and no saddles to escape.

Two gradients are used constantly, both from the linear algebra page:

$$\nabla_w\left(w^\top a\right) = a,\qquad \nabla_w\left(w^\top A w\right) = 2Aw\quad(A\text{ symmetric}).$$

## Optimization

**Unconstrained.** Solve $\nabla f(w) = 0$ and check the Hessian. Applying this to the least-squares objective $\lVert y - X\beta\rVert^2$ gives $-2X^\top(y-X\beta) = 0$, the normal equations of [Multiple Linear Regression](../part-13-regression/02-multiple-linear-regression.md). Applying it to a log-likelihood gives the score equations of [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md), and the Hessian of the log-likelihood is (minus) the observed information, which is where the estimator's standard errors come from.

**Constrained: Lagrange multipliers.** To minimize $f(w)$ subject to $g(w) = 0$, form

$$\mathcal{L}(w,\lambda) = f(w) - \lambda\,g(w)$$

and set all partial derivatives to zero. The condition $\nabla f = \lambda\nabla g$ says the objective's gradient is parallel to the constraint's: at the optimum there is no direction that both stays on the constraint surface and improves the objective. If they were not parallel, the component of $\nabla f$ along the surface would point to a better feasible point.

### Worked Example: Minimum-Variance Weights

Minimize portfolio variance subject to full investment:

$$\min_w\ w^\top\Sigma w \quad\text{subject to}\quad w^\top\mathbf{1} = 1.$$

Form the Lagrangian and differentiate, using $\nabla_w(w^\top\Sigma w) = 2\Sigma w$ and $\nabla_w(w^\top\mathbf{1}) = \mathbf{1}$:

$$\mathcal{L}(w,\lambda) = w^\top\Sigma w - \lambda\left(w^\top\mathbf{1}-1\right),\qquad \nabla_w\mathcal{L} = 2\Sigma w - \lambda\mathbf{1} = 0.$$

Solving the first-order condition gives $w = \frac{\lambda}{2}\Sigma^{-1}\mathbf{1}$, and imposing the constraint $w^\top\mathbf{1}=1$ fixes the multiplier at $\frac{\lambda}{2} = \big(\mathbf{1}^\top\Sigma^{-1}\mathbf{1}\big)^{-1}$. Therefore

$$w_{\text{MV}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}},\qquad \sigma^2_{\text{MV}} = w_{\text{MV}}^\top\Sigma w_{\text{MV}} = \frac{1}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}.$$

Both formulas are worth reading structurally. The weights require no expected returns at all — which is why minimum variance is the allocator that wins the horse race in [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md), where every method that needs $\hat\mu$ is poisoned by it. And the minimized variance is the reciprocal of a quadratic form in $\Sigma^{-1}$, so it inherits all the conditioning problems of that inverse.

The derivation is checkable against a numerical optimizer, which is the right habit for any closed form:

```python
import numpy as np
from scipy.optimize import minimize

corr = np.array([[1.0, 0.3, 0.2],
                 [0.3, 1.0, 0.6],
                 [0.2, 0.6, 1.0]])
vol = np.array([0.18, 0.12, 0.15])
Sigma = corr * np.outer(vol, vol)
ones = np.ones(3)

# Closed form from the Lagrangian.
z = np.linalg.solve(Sigma, ones)
w_closed = z / z.sum()

# Same problem handed to a general-purpose constrained optimizer.
res = minimize(lambda w: w @ Sigma @ w, x0=ones / 3, method="SLSQP",
               constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, tol=1e-14)

print("closed form:", np.round(w_closed, 6))
print("SLSQP:      ", np.round(res.x, 6))
print(f"max difference: {np.abs(res.x - w_closed).max():.2e}")
print(f"minimized variance {w_closed @ Sigma @ w_closed:.8f}"
      f"  =  1/(1'S^-1 1) = {1 / (ones @ np.linalg.solve(Sigma, ones)):.8f}")
# => closed form: [0.219209 0.581602 0.199189]
#    SLSQP:       [0.219209 0.581602 0.199189]
#    max difference: 2.65e-08
#    minimized variance 0.01194678  =  1/(1'S^-1 1) = 0.01194678
```

The agreement to eight digits confirms the algebra, and the closed form is worth having: it is exact, instant, and differentiable, where the optimizer is approximate, iterative, and silent about how hard the problem was. The reason to keep the optimizer around is that the moment a realistic constraint appears — no shorting, position caps, turnover limits — the closed form is gone and only the numerical route survives.

### When There Is No Closed Form

Most objectives in this book have no analytic solution, and the fallback is iterative descent: repeatedly step against the gradient,

$$w_{k+1} = w_k - \eta\,\nabla f(w_k),$$

with step size (learning rate) $\eta$. The method converges for convex $f$ with a small enough $\eta$, slowly if the Hessian is ill-conditioned — the condition number reappears as a convergence rate, which is why feature scaling matters so much in practice. Newton's method uses curvature, $w_{k+1} = w_k - H^{-1}\nabla f$, converging far faster near the optimum at the cost of forming and inverting the Hessian each step. The quasi-Newton family (BFGS, L-BFGS) approximates $H^{-1}$ from successive gradients and is what `scipy.optimize` runs by default. [Numerical Optimization](../part-17-statistical-computing/01-numerical-optimization.md) develops the family; the [EM Algorithm](../part-17-statistical-computing/03-em-algorithm.md) is the specialized version that appears in mixture and regime models.

The practical checklist for any of them: verify the gradient against a central difference before trusting it, scale the variables so the Hessian is not wildly anisotropic, start from several points when the objective may be non-convex, and check that the reported convergence actually satisfies the first-order condition rather than merely exhausting an iteration budget.
