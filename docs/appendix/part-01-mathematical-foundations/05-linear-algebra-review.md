# Basic Linear Algebra Review

A portfolio is a vector. A covariance matrix is a matrix. Portfolio variance is a quadratic form, a regression is a projection, and the reason mean-variance optimization explodes on real data is a statement about eigenvalues. None of that is metaphor — it is the actual arithmetic, and this page assembles the pieces the course leans on: [Covariance Matrices](../part-06-multivariate-probability/02-covariance-matrices.md), [Multiple Linear Regression](../part-13-regression/02-multiple-linear-regression.md), the factor decompositions in [Risk Parity, Diversification, and Factors](../../part-08-portfolio-management/03-risk-parity-diversification-factors.md), and the optimizer post-mortem in [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md).

The scope is deliberately narrow. There is no attempt at a full linear algebra course; the goal is that every matrix expression appearing later in this book can be read, checked for shape, and understood as an operation on data.

## Vectors

A vector $x\in\mathbb{R}^n$ is an ordered list of $n$ real numbers. By convention in this book — and in most of the literature — vectors are **columns**, so $x$ is $n\times1$ and $x^\top$ is $1\times n$. Addition and scalar multiplication work componentwise.

In quantitative work a vector is usually one of three things: a portfolio weight vector $w$, whose entries are fractions of capital; a vector of expected returns $\mu$; or one observation's feature vector. All three live in $\mathbb{R}^N$ where $N$ is the number of assets or features.

### Inner Products and Norms

The **inner product** (dot product) of $x,y\in\mathbb{R}^n$ is

$$x^\top y = \sum_{i=1}^{n} x_i y_i,$$

a scalar. It is the single most-used operation in the course: portfolio return is $w^\top r$, a linear signal is $\beta^\top x$, and a weighted average is an inner product with weights summing to one.

**Norms** measure size:

| Norm | Definition | Where it appears |
|---|---|---|
| $\lVert x\rVert_1$ | $\sum_i \lvert x_i\rvert$ | Gross exposure of a portfolio; the penalty in lasso |
| $\lVert x\rVert_2$ | $\sqrt{\sum_i x_i^2} = \sqrt{x^\top x}$ | Euclidean length; the penalty in ridge |
| $\lVert x\rVert_\infty$ | $\max_i \lvert x_i\rvert$ | Largest single position; worst-case constraints |

The unqualified $\lVert x\rVert$ means the Euclidean norm. That the $\ell_1$ norm of a weight vector is gross exposure is not a coincidence to be noticed once and forgotten — it is why an $\ell_1$ constraint is the natural way to write a leverage limit, and why the $\ell_1$ penalty in [Regularization](../part-13-regression/05-regularization.md) produces sparse solutions where the $\ell_2$ penalty produces small ones.

**Cauchy–Schwarz** bounds the inner product by the norms:

$$\lvert x^\top y\rvert \le \lVert x\rVert\,\lVert y\rVert,$$

with equality exactly when $x$ and $y$ are parallel. Rearranged, it defines the **angle** between two vectors:

$$\cos\theta = \frac{x^\top y}{\lVert x\rVert\,\lVert y\rVert}\in[-1,1].$$

!!! note "Correlation is a cosine"
    Center two return series by subtracting their means and stack them as vectors $\tilde r_1,\tilde r_2\in\mathbb{R}^T$. Then

    $$\rho_{12} = \frac{\tilde r_1^\top\tilde r_2}{\lVert\tilde r_1\rVert\,\lVert\tilde r_2\rVert}$$

    is precisely the sample correlation, and it is precisely $\cos\theta$. Uncorrelated means geometrically orthogonal; perfectly correlated means parallel; $\rho=-1$ means antiparallel. The bound $\lvert\rho\rvert\le1$ needs no separate proof — it *is* Cauchy–Schwarz. This picture is worth carrying into [Correlation](../part-04-expectation-and-moments/05-correlation.md), because it makes "diversification" a statement about angles: adding an asset helps to the extent its return vector points somewhere the existing portfolio does not.

## Matrices

A matrix $A\in\mathbb{R}^{m\times n}$ has $m$ rows and $n$ columns. It plays two roles, and confusing them is the source of most sign and shape errors.

**As data.** A returns matrix $R$ is $T\times N$ — rows are dates, columns are assets — because that is how time series are stored and how pandas and NumPy lay them out. Nothing about this layout is mathematically privileged; it is a storage convention.

**As a linear map.** $A$ sends $x\in\mathbb{R}^n$ to $Ax\in\mathbb{R}^m$, and every linear map between finite-dimensional spaces is a matrix in this sense. Under this reading, matrix multiplication is composition of maps, which is why it is associative ($ABx$ can be grouped either way) and not commutative ($AB\neq BA$ in general — rotating then stretching differs from stretching then rotating).

The product $C = AB$ requires the inner dimensions to match, $A\in\mathbb{R}^{m\times k}$ and $B\in\mathbb{R}^{k\times n}$, and

$$C_{ij} = \sum_{l=1}^{k} A_{il}B_{lj},$$

an inner product of row $i$ of $A$ with column $j$ of $B$. Checking that the *inner* dimension is the axis being summed over — assets for a portfolio aggregation, dates for a time-series reduction — catches shape bugs before they run.

| Object | Definition | Note |
|---|---|---|
| Transpose $A^\top$ | $(A^\top)_{ij} = A_{ji}$ | $(AB)^\top = B^\top A^\top$ — the order reverses |
| Identity $I$ | Ones on the diagonal | $AI = IA = A$ |
| Inverse $A^{-1}$ | $AA^{-1} = I$ | Exists only for square, full-rank $A$ |
| Trace $\mathrm{tr}(A)$ | $\sum_i A_{ii}$ | Cyclic: $\mathrm{tr}(ABC) = \mathrm{tr}(BCA)$ |
| Diagonal | Zero off the diagonal | $\mathrm{diag}(\sigma)$ converts volatilities to a scaling map |
| Symmetric | $A = A^\top$ | Every covariance and correlation matrix |
| Orthogonal | $Q^\top Q = I$ | Columns orthonormal; $Q$ rotates without stretching |
| Triangular | Zero above or below the diagonal | The output of Cholesky; cheap to solve with |

The transpose-reversal rule is what makes $w^\top\Sigma w$ a scalar equal to its own transpose, and it is the reason a covariance matrix constructed as $\frac{1}{T-1}\tilde R^\top\tilde R$ comes out symmetric automatically.

### Span, Rank, and Why It Matters for Returns

The **span** of a set of vectors is the set of all their linear combinations. Vectors are **linearly independent** if none is a combination of the others, equivalently if $\sum_i c_i x_i = 0$ forces every $c_i=0$. The **rank** of a matrix is the number of linearly independent columns, which always equals the number of linearly independent rows.

A square matrix is **invertible** exactly when it has full rank. A rank-deficient matrix is **singular**: it collapses some direction to zero, and no map can undo that.

This is where estimation collides with algebra. A sample covariance matrix built from $T$ observations of $N$ assets has rank at most $\min(T-1, N)$. With fewer observations than assets, $\hat\Sigma$ is singular *no matter how clean the data is* — a structural fact, not a numerical accident:

```python
import numpy as np

rng = np.random.default_rng(7)
X = rng.normal(0, 0.01, (30, 50))       # 30 days, 50 assets
S = np.cov(X, rowvar=False)
ev = np.linalg.eigvalsh(S)              # ascending, symmetric solver

print(f"Sigma is {S.shape[0]}x{S.shape[1]}, rank {np.linalg.matrix_rank(S)}")
print(f"smallest eigenvalue {ev[0]:.3e}")
print(f"largest eigenvalue  {ev[-1]:.3e}")
# => Sigma is 50x50, rank 29
#    smallest eigenvalue -1.051e-19
#    largest eigenvalue  4.612e-04
```

Rank 29 from 30 observations — one degree of freedom lost to estimating the mean. Twenty-one directions in this 50-asset space have *exactly zero* estimated variance, which the optimizer will happily read as "risk-free" and lever into. The slightly negative smallest eigenvalue is floating-point noise around a true zero, and it is worth recognizing on sight: eigenvalues of order $10^{-19}$ against a largest of order $10^{-4}$ mean rank deficiency, not a broken calculation. Shrinkage estimators exist to repair exactly this, though as [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) documents, repairing $\hat\Sigma$ does not fix an optimizer whose real problem is $\hat\mu$.

### Solving $Ax = b$

For invertible $A$ the solution is $x = A^{-1}b$ mathematically, and never numerically. Forming the inverse costs more arithmetic than solving directly and loses precision along the way:

```python
corr = np.array([[1.0, 0.999, 0.5],
                 [0.999, 1.0, 0.5],
                 [0.5,   0.5, 1.0]])
vol = np.array([0.20, 0.20, 0.15])
Sigma = corr * np.outer(vol, vol)        # covariance from correlation and vols
b = np.ones(3)

x_solve = np.linalg.solve(Sigma, b)
x_inv = np.linalg.inv(Sigma) @ b

print(f"residual, solve: {np.abs(Sigma @ x_solve - b).max():.3e}")
print(f"residual, inv:   {np.abs(Sigma @ x_inv - b).max():.3e}")
# => residual, solve: 1.110e-16
#    residual, inv:   2.021e-14
```

Two orders of magnitude of accuracy, free, from writing `solve` instead of `inv @`. The rule generalizes: whenever an expression contains $A^{-1}b$, compute it as a solve. The only reason to form an explicit inverse is when the inverse itself is the object of interest — and even then, ask whether it is.

## Quadratic Forms and Positive Semi-Definiteness

For symmetric $A\in\mathbb{R}^{n\times n}$, the **quadratic form** is the scalar

$$q(x) = x^\top A x = \sum_{i=1}^{n}\sum_{j=1}^{n} x_i A_{ij} x_j.$$

The instance that matters: with $\Sigma$ the covariance matrix of asset returns and $w$ a weight vector,

$$\mathrm{var}(w^\top r) = w^\top\Sigma w = \sum_{i}\sum_{j} w_i w_j \,\mathrm{cov}(r_i, r_j).$$

Portfolio variance is a quadratic form, and the double sum makes the diversification arithmetic explicit: the $N$ diagonal terms carry each asset's own variance, the $N(N-1)$ off-diagonal terms carry the covariances, and in a large portfolio the off-diagonals dominate by sheer count. Diversification is the statement that off-diagonal terms can be made small — never that diagonal ones can.

```python
rng = np.random.default_rng(7)
R = rng.normal(0, 0.01, (750, 4))          # 750 days, 4 assets
Sigma4 = np.cov(R, rowvar=False)
w = np.array([0.4, 0.3, 0.2, 0.1])

loop = sum(w[i] * Sigma4[i, j] * w[j] for i in range(4) for j in range(4))
mat = w @ Sigma4 @ w
ein = np.einsum("i,ij,j->", w, Sigma4, w)

print(f"{loop:.12e}")
print(f"{mat:.12e}")
print(f"{ein:.12e}")
print(f"annualized vol: {np.sqrt(mat * 252):.4f}")
# => 3.247267218578e-05
#    3.247267218578e-05
#    3.247267218578e-05
#    annualized vol: 0.0905
```

The three forms agree to every printed digit, and the middle one is the one to write. `w @ Sigma @ w` reads as the formula, runs in compiled code, and cannot get an index wrong.

A symmetric matrix $A$ is **positive semi-definite (PSD)** if $x^\top Ax\ge0$ for every $x$, and **positive definite (PD)** if the inequality is strict for $x\neq0$.

??? note "Every covariance matrix is PSD"
    Let $\Sigma$ be the covariance matrix of a random vector $r$, and fix any $w\in\mathbb{R}^N$. The scalar $w^\top r$ is a random variable, and

    $$w^\top\Sigma w = \mathrm{var}(w^\top r)\ \ge 0,$$

    because a variance is an expectation of a square. So the PSD property is not an extra assumption imposed on covariance matrices — it is variance non-negativity, transcribed. Equality holds when some portfolio has zero variance, i.e. when the assets are linearly dependent, which is the singular case above.

    The converse also holds: every PSD matrix is the covariance matrix of some random vector. This is what makes "generate returns with covariance $\Sigma$" a well-posed request whenever $\Sigma$ is PSD, and an impossible one when a hand-edited correlation matrix has quietly stopped being PSD.

## Eigenvalues and Eigenvectors

A nonzero $v$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ if

$$Av = \lambda v,$$

that is, if $A$ acts on $v$ by pure scaling. The eigenvalues solve the characteristic equation $\det(A-\lambda I)=0$.

For symmetric matrices — which is every matrix this book applies the concept to — the situation is as good as it gets.

!!! note "Spectral theorem"
    Every real symmetric $A$ can be written

    $$A = Q\Lambda Q^\top,$$

    where $\Lambda$ is diagonal with the (real) eigenvalues and $Q$ is orthogonal, its columns an orthonormal basis of eigenvectors. Equivalently, $A$ acts as a pure rescaling once the coordinate system is rotated to align with its eigenvectors. $A$ is PSD exactly when all $\lambda_i\ge0$, and PD exactly when all $\lambda_i>0$.

Applied to a covariance matrix, the decomposition *is* principal component analysis. The eigenvectors are uncorrelated portfolios — the principal components — and each eigenvalue is the variance of the corresponding component. Since $\mathrm{tr}(\Sigma) = \sum_i\lambda_i$ equals total variance, the ratio $\lambda_k/\sum_i\lambda_i$ is the share explained by component $k$:

```python
evals, evecs = np.linalg.eigh(corr)        # ascending order
evals = evals[::-1]

print("eigenvalues:", np.round(evals, 4))
print(f"first component explains {evals[0] / 3:.1%} of total variance")
# => eigenvalues: [2.3652e+00 6.3380e-01 1.0000e-03]
#    first component explains 78.8% of total variance
```

Three assets, two of them correlated at 0.999, and the spectrum tells the whole story: one dominant direction holding 79% of the variance, a second holding 21%, and a third with essentially nothing — the near-perfect pair leaves almost no independent variation in their difference. This is the eigenvalue-spectrum reading behind the effective-rank and effective-number-of-bets diagnostics in [Risk Parity, Diversification, and Factors](../../part-08-portfolio-management/03-risk-parity-diversification-factors.md): a universe of $N$ assets rarely offers $N$ independent bets, and the spectrum says how many it actually offers.

### Condition Number: Why Optimizers Explode

The **condition number** of a symmetric PD matrix is

$$\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}.$$

It measures how much $A^{-1}$ amplifies input error: a relative perturbation of size $\delta$ in the inputs can produce a relative change of up to $\kappa\delta$ in the solution. Since $\Sigma^{-1}$ divides by eigenvalues, the *smallest* eigenvalue — the direction with least variance, which is also the direction estimated with the least data — dominates the answer.

```python
mu = np.array([0.05, 0.05, 0.05])
mu_bumped = np.array([0.05, 0.051, 0.05])          # one input moves by 0.001

def tangency(Sigma, mu):
    w = np.linalg.solve(Sigma, mu)
    return w / w.sum()                              # normalize to full investment

print(f"condition number: {np.linalg.cond(Sigma):.1f}")
print("weights, mu:       ", np.round(tangency(Sigma, mu), 3))
print("weights, mu bumped:", np.round(tangency(Sigma, mu_bumped), 3))
# => condition number: 2173.6
#    weights, mu:        [0.115 0.115 0.769]
#    weights, mu bumped: [-5.061  5.299  0.763]
```

A one-tenth-of-one-percent change in a single expected return turns an 11%/11%/77% portfolio into a 506% short against a 530% long. Nothing is wrong with the code, and nothing is wrong with the formula $w\propto\Sigma^{-1}\mu$ — the matrix is simply near-singular, and the optimizer is faithfully reporting that the difference between two assets correlated at 0.999 is an almost risk-free spread worth taking at enormous size. Every large-scale failure of mean-variance optimization, including the 11.81× gross exposure and $-99.9\%$ drawdown documented in [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md), is this three-line demonstration at scale.

The practical readings: check $\kappa(\hat\Sigma)$ before trusting anything that inverts it; treat a large condition number as a statement about the *universe* (assets that are near-duplicates) rather than a numerical nuisance; and prefer methods that avoid the inverse entirely when it is large.

## Cholesky Decomposition

Every PD matrix factors uniquely as

$$\Sigma = LL^\top$$

with $L$ lower triangular and positive on the diagonal. Two uses, both central.

**Simulation.** If $z\sim\mathcal{N}(0,I)$ has independent standard normal components, then $x = Lz$ has covariance

$$\mathrm{cov}(Lz) = L\,\mathrm{cov}(z)\,L^\top = LIL^\top = LL^\top = \Sigma.$$

That is the standard recipe for generating correlated returns, and it is what `np.random.multivariate_normal` does internally.

**Solving.** Triangular systems are solved by direct substitution, so $\Sigma x = b$ becomes two cheap triangular solves. Cholesky costs about half of a general LU factorization, and it *fails* — raising an error rather than returning nonsense — when the matrix is not positive definite, which makes it a useful PSD test in its own right.

```python
L = np.linalg.cholesky(Sigma)
print("round trip:", np.allclose(L @ L.T, Sigma))

z = rng.standard_normal((200_000, 3))
x = z @ L.T                                        # rows are draws

print("realized corr:", np.round(np.corrcoef(x, rowvar=False)[np.triu_indices(3, 1)], 4))
print("realized vol: ", np.round(x.std(axis=0, ddof=1), 4))
# => round trip: True
#    realized corr: [0.999  0.4992 0.4991]
#    realized vol:  [0.1998 0.1998 0.1501]
```

Note the transpose in `z @ L.T`: with draws stored as rows, the map that acts on a column vector as $Lz$ acts on a row-stacked matrix as $ZL^\top$. This is the rows-as-observations convention colliding with the columns-as-vectors convention, and getting it backwards produces a matrix with the wrong correlations and no error message. Verifying the realized correlation against the target, as above, costs one line and catches it.

## Projections and Least Squares

Given a design matrix $X\in\mathbb{R}^{n\times p}$ (rows are observations, columns are predictors) and a response $y\in\mathbb{R}^n$, ordinary least squares seeks the $\hat\beta$ minimizing $\lVert y - X\beta\rVert^2$.

The geometric statement is cleaner than the calculus. The set $\{X\beta : \beta\in\mathbb{R}^p\}$ is the column space of $X$ — a $p$-dimensional subspace of $\mathbb{R}^n$. Minimizing the distance from $y$ to that subspace means finding the **orthogonal projection** of $y$ onto it, and the defining property of the projection is that the residual is perpendicular to the subspace:

$$X^\top\left(y - X\hat\beta\right) = 0 \quad\Longleftrightarrow\quad X^\top X\hat\beta = X^\top y \quad\Longleftrightarrow\quad \hat\beta = (X^\top X)^{-1}X^\top y.$$

The middle expression is the **normal equations** — normal in the sense of perpendicular — and it is how the estimate should be computed, as a solve rather than an inverse.

```python
n = 500
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
y = 0.3 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
X = np.column_stack([np.ones(n), x1, x2])          # intercept column first

beta = np.linalg.solve(X.T @ X, X.T @ y)
beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta

print("normal equations:", np.round(beta, 4))
print("lstsq:           ", np.round(beta_lstsq, 4))
print("residual ⟂ columns:", bool(np.abs(X.T @ resid).max() < 1e-10))
# => normal equations: [ 0.2738  1.5077 -0.8355]
#    lstsq:            [ 0.2738  1.5077 -0.8355]
#    residual ⟂ columns: True
```

The estimates recover the true $(0.3, 1.5, -0.8)$ to within sampling error, and the orthogonality check is the geometry made testable. Three consequences follow directly and are used throughout [Part XIII](../part-13-regression/index.md):

- **Multicollinearity is ill-conditioning.** Correlated predictors make $X^\top X$ near-singular, and by the condition-number argument above, the coefficients become wildly sensitive. The symptom — huge coefficients with huge standard errors that cancel — is the same phenomenon as the exploding portfolio weights.
- **Ridge regression is a conditioning fix.** Replacing $X^\top X$ with $X^\top X + \lambda I$ adds $\lambda$ to every eigenvalue, bounding the condition number by construction. That is the algebraic content of [Regularization](../part-13-regression/05-regularization.md).
- **The projection matrix $H = X(X^\top X)^{-1}X^\top$** — the "hat matrix", since $\hat y = Hy$ — is symmetric, idempotent ($H^2 = H$), and has trace equal to $p$. Its diagonal entries are the leverage values used in [Model Diagnostics](../part-13-regression/06-model-diagnostics.md).

## Matrix Calculus

Two gradient identities are used repeatedly, most immediately by the Lagrangian derivation in [Calculus Essentials](06-calculus-essentials.md):

$$\nabla_w\left(w^\top a\right) = a,\qquad \nabla_w\left(w^\top A w\right) = 2Aw \quad(A \text{ symmetric}).$$

The first is the vector analogue of $\frac{d}{dx}(ax) = a$; the second of $\frac{d}{dx}(ax^2) = 2ax$, with symmetry of $A$ doing the work that would otherwise leave $(A + A^\top)w$. Setting the gradient of $\lVert y - X\beta\rVert^2$ to zero using both identities reproduces the normal equations, and minimizing $w^\top\Sigma w$ subject to $w^\top\mathbf{1} = 1$ produces the minimum-variance weights

$$w = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}},$$

derived in full on the calculus page. Read that formula with this page's eyes and its behavior is already predictable: it inverts a covariance matrix, so it is exposed to the smallest eigenvalues, so it will concentrate in whichever directions the estimate says are quiet — which is precisely why it wins on realized risk and loses on stability.

## Shapes in Practice

A summary of the conventions that keep the algebra and the code aligned:

| Quantity | Shape | Note |
|---|---|---|
| Returns $R$ | $T\times N$ | Rows are dates, columns are assets |
| Weights $w$ | $N$ | A 1-D NumPy array; `@` handles orientation |
| Covariance $\Sigma$ | $N\times N$ | `np.cov(R, rowvar=False)` — the flag is not optional |
| Portfolio returns | $T$ | `R @ w` |
| Portfolio variance | scalar | `w @ Sigma @ w` |
| Design matrix $X$ | $n\times p$ | Rows are observations, columns are predictors |

The single most common bug in this area is a silent `rowvar` mistake: `np.cov(R)` on a $T\times N$ matrix returns a $T\times T$ matrix of correlations *between dates*, which is meaningless and often still runs. Asserting the shape of $\Sigma$ before using it costs one line and catches it immediately — the same defensive habit [NumPy and Vectorization](../../part-02-python/01-numpy-and-vectorization.md) argues for with broadcasting.
