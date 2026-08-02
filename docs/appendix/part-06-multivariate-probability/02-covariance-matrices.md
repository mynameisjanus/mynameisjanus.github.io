# Covariance Matrices

A covariance matrix is not a table of pairwise numbers that happens to be square. It is a single expectation of a single outer product, and symmetry, positive semi-definiteness, and the fact that it converts a weight vector into a variance are all consequences of its being one expectation rather than $\binom{N}{2}$ of them. A table assembled pair by pair has none of those guarantees, and nothing about the table announces which kind of object it is.

This page covers the covariance matrix as an expectation of an outer product, the quadratic form and the two lines that make it positive semi-definite, the sample estimator and the observation it spends on the mean, the eigenvalue tilt that too few observations manufacture out of nothing, shrinkage as the repair and the exact thing shrinkage cannot repair, and the three ways a broken matrix is put back onto the cone. It does not develop quadratic forms, eigen-decomposition, the condition number or Cholesky, which are [Basic Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md); it does not derive bilinearity or the scalar double sum, which are [Covariance](../part-04-expectation-and-moments/04-covariance.md); it does not divide the units out, which is [Correlation Matrices](03-correlation-matrices.md); and it derives no sampling distribution for anything it estimates, which is [Part XI](../part-11-parameter-estimation/index.md).

The trading stake is a shrinkage intensity that behaved exactly as the theory says and helped almost not at all. [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) reports a Ledoit–Wolf intensity of $1.85\%$ at $56$ observations per asset, rising to $5.42\%$ at $14$ and $9.69\%$ at $7$ — a correctly behaved estimator whose correct behaviour still left mean–variance optimization at a Sharpe of $0.377$, running at $293.9\%$ volatility, losing to equal weighting's $0.450$. The fifth section separates what shrinkage fixes from what was actually broken, and finds the two are not the same object.

## The Covariance Matrix Is One Expectation, Not N Choose 2 of Them

For a random vector $X$ with mean $\mu$ and finite second moments, the covariance matrix is

$$\Sigma=\mathrm{cov}(X)=\mathbb{E}\big[(X-\mu)(X-\mu)^\top\big],$$

a single expectation, taken entrywise over a single $N\times N$ random matrix. Reading off the $(i,j)$ entry recovers the scalar definition of [Covariance](../part-04-expectation-and-moments/04-covariance.md), $\Sigma_{ij}=\mathrm{cov}(X_i,X_j)$, and reading off the diagonal recovers the variances, $\Sigma_{ii}=\mathrm{var}(X_i)$. Nothing new has been defined. What has changed is the order of quantification: the scalar definition produces one number per pair, and this one produces the whole array at once, from one draw of one random vector at a time.

That distinction is the content of the page. Symmetry is immediate and requires no argument, because $(X-\mu)(X-\mu)^\top$ is a symmetric matrix at every $\omega$, so its expectation is symmetric. A table assembled pair by pair is symmetric too — but only because whoever assembled it copied the upper triangle into the lower one, which is a convention rather than a theorem. The two objects look identical and are guaranteed different amounts.

## The Quadratic Form Is a Variance, and That Is the Whole of Positive Semi-Definiteness

The matrix earns its keep by turning a portfolio into a number. For any fixed $w\in\mathbb{R}^{N}$,

$$\mathrm{var}(w^\top X)=w^\top\Sigma w\ \ge\ 0,$$

and the inequality holding for every $w$ is what positive semi-definiteness means. [Covariance](../part-04-expectation-and-moments/04-covariance.md) states this once in scalar form and [Basic Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) supplies the converse — every PSD matrix is the covariance matrix of *some* random vector, which is what makes "simulate returns with covariance $\Sigma$" a well-posed request. The matrix form is worth writing down separately because it makes visible how little is being assumed.

??? note "Proof that the covariance matrix is symmetric and positive semi-definite, and that these are the same fact"
    Fix $w\in\mathbb{R}^{N}$ and write $Y=w^\top(X-\mu)$, a scalar random variable with mean zero. Then

    $$w^\top\Sigma w=w^\top\mathbb{E}\big[(X-\mu)(X-\mu)^\top\big]w=\mathbb{E}\big[w^\top(X-\mu)(X-\mu)^\top w\big]=\mathbb{E}[Y^{2}]\ \ge\ 0,$$

    where the constants $w$ were moved inside the expectation by linearity, exactly as in the previous page's proof, and the last step is that a square is non-negative. Symmetry comes from the same expression: $\big((X-\mu)(X-\mu)^\top\big)^\top=(X-\mu)(X-\mu)^\top$ pointwise in $\omega$, and transposition commutes with an entrywise expectation.

    The two properties are the same fact seen twice. The outer product is symmetric *and* rank-one PSD at every single $\omega$, and $\Sigma$ is an average of such matrices; both properties survive averaging because the set of symmetric PSD matrices is a convex cone closed under non-negative combinations.

    The single hypothesis is that all $N^{2}$ entries come from *one* joint law — one $\Sigma$, one expectation, one $\omega$ at a time. Drop it and both properties can fail while every individual entry remains a perfectly ordinary number. Estimate different pairs on different windows, take volatilities from one vendor and correlations from another, or overlay a hand-set view on a fitted table, and the average-of-outer-products representation no longer applies, because there is no random vector whose outer product was averaged. [Joint Distributions](../part-03-random-variables/05-joint-distributions.md) exhibits the smallest such failure, three pairwise correlations of $0.9$, $0.9$ and $-0.9$ that no three random variables can have, and the last section of this page is what to do when one arrives.

```python
import numpy as np

rng = np.random.default_rng(617)
N, T = 6, 2_000_000
B = np.array([[1.0, 0.3], [1.1, -0.2], [0.9, 0.6], [1.2, 0.1], [0.8, -0.5], [1.0, 0.0]])
SF = np.diag([0.010 ** 2, 0.006 ** 2])                         # market and a style factor
D = np.diag(np.array([0.006, 0.005, 0.008, 0.004, 0.009, 0.007]) ** 2)
Sigma = B @ SF @ B.T + D                                       # one expectation, written down
X = rng.standard_normal((T, 2)) @ np.sqrt(SF) @ B.T + rng.standard_normal((T, N)) @ np.sqrt(D)
print(f"  max |sample covariance - B SF B' - D|  {np.abs(np.cov(X, rowvar=False) - Sigma).max():9.2e}")
u, s, vt = np.linalg.svd(B.T)
neutral = vt[2:].sum(axis=0)                                   # B' w = 0, no factor exposure
ws = {"equal": np.ones(N) / N, "long-short": np.array([1, 1, 1, -1, -1, -1.0]) / 3,
      "concentrated": np.array([1, 0, 0, 0, 0, 0.0]),
      "factor-neutral": neutral / np.abs(neutral).sum()}
print("        weights        w' Sigma w    sample var(Xw)     ann vol    factor share")
for name, w in ws.items():
    q, sv = w @ Sigma @ w, (X @ w).var()
    print(f"  {name:>15s} {q:13.3e} {sv:17.3e} {np.sqrt(252 * q):11.4f}"
          f" {(w @ B @ SF @ B.T @ w) / q:15.4f}")
# =>   max |sample covariance - B SF B' - D|   1.74e-07
#            weights        w' Sigma w    sample var(Xw)     ann vol    factor share
#                equal     1.076e-04         1.077e-04      0.1647          0.9301
#           long-short     3.495e-05         3.490e-05      0.0938          0.1385
#         concentrated     1.392e-04         1.394e-04      0.1873          0.7415
#       factor-neutral     8.028e-06         8.029e-06      0.0450          0.0000
```

The first line is the definition checked against a simulation: a $\Sigma$ written down analytically as $B\Sigma_FB^\top+D$ is recovered from two million draws to $1.7\times10^{-7}$. The four weight rows then confirm that $w^\top\Sigma w$ is a variance and not merely an expression shaped like one — the closed-form quadratic form and the sample variance of the realised portfolio agree to three significant figures in every row, including the long–short row where the weights sum to zero and the concentrated row where five of them are zero.

The last row is the control and the point. Its weights were chosen to lie in the null space of $B^\top$, so the portfolio has literally no exposure to either factor, and the factor share column reads $0.0000$ exactly rather than approximately. Its annualized volatility is $4.5\%$ against the equal-weight book's $16.5\%$ — a book built from the same six assets, with no shorting of any factor, carrying less than a third of the risk. That reduction is invisible to any per-asset volatility file and is a property of the matrix as a whole.

## The Sample Estimator Spends One Observation on the Mean

With returns stacked as $R$, of shape $T\times N$, and $M=I_T-\tfrac1T\mathbf{1}\mathbf{1}^\top$ the centring projector, the usual estimator is

$$\hat\Sigma=\frac{1}{T-1}\,R^\top M R,\qquad \mathbb{E}[\hat\Sigma]=\Sigma,\qquad \mathrm{rank}(\hat\Sigma)\le\min(T-1,\,N).$$

The divisor $T-1$ and the rank ceiling $T-1$ are the same fact wearing two hats: $M$ is a projection onto the $(T-1)$-dimensional subspace orthogonal to $\mathbf{1}$, so one direction of the sample is consumed by not knowing $\mu$, and it is consumed once for the whole matrix rather than once per entry.

??? note "Proof that the sample covariance matrix is unbiased and cannot have rank above T minus one"
    The matrix $M=I_T-\tfrac1T\mathbf{1}\mathbf{1}^\top$ is symmetric and idempotent, since $\mathbf{1}^\top\mathbf{1}=T$ gives $M^{2}=I-\tfrac2T\mathbf{1}\mathbf{1}^\top+\tfrac1{T^{2}}\mathbf{1}(T)\mathbf{1}^\top=M$, and its trace is $T-1$. An idempotent matrix has eigenvalues in $\{0,1\}$, so its rank equals its trace, namely $T-1$.

    For the rank ceiling, $\mathrm{rank}(R^\top MR)\le\mathrm{rank}(MR)\le\min(\mathrm{rank}\,M,\,\mathrm{rank}\,R)\le\min(T-1,N)$, because the rank of a product never exceeds the rank of either factor. For unbiasedness, write $R=\mathbf{1}\mu^\top+E$ with the rows of $E$ independent, mean zero and covariance $\Sigma$. Since $M\mathbf{1}=0$ the mean term is annihilated, $MR=ME$, and

    $$\mathbb{E}\big[E^\top ME\big]=\sum_{s,t}M_{st}\,\mathbb{E}\big[E_{s\cdot}E_{t\cdot}^\top\big]=\Sigma\sum_{t}M_{tt}=(T-1)\,\Sigma,$$

    because $\mathbb{E}[E_{s\cdot}E_{t\cdot}^\top]$ is $\Sigma$ when $s=t$ and zero otherwise. Dividing by $T-1$ gives $\mathbb{E}[\hat\Sigma]=\Sigma$.

    Unbiasedness here is an *entrywise* statement, and that is a much weaker guarantee than it sounds when the consumer of the matrix is an inverse. An eigenvalue is a nonlinear function of the entries, so no amount of entrywise unbiasedness makes the spectrum unbiased — by Jensen's inequality it will not be, and the next section measures the direction and size of the distortion. The rank ceiling is the extreme case of the same phenomenon: at $T\le N$ the smallest eigenvalues are not merely biased but exactly zero, which [Basic Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md) demonstrates at rank $29$ from thirty observations of fifty assets, with every one of the twenty-one missing directions reported to an optimizer as riskless.

## An Unbiased Matrix Has Biased Eigenvalues

The interesting failure is not the singular case, which announces itself. It is the case with $T$ comfortably larger than $N$, where $\hat\Sigma$ is invertible, every entry is unbiased, and the spectrum is nevertheless spread outward from the truth — largest eigenvalues too large, smallest too small, and the trace exactly right throughout.

```python
import numpy as np

rng = np.random.default_rng(619)
N, reps = 50, 200                                              # truth: every eigenvalue is 1
print(f"  sample spectrum of Sigma-hat when the true Sigma is the {N}x{N} identity")
print("        T    N/T    mean lmax   MP edge   mean lmin   MP edge   mean cond   mean tr/N")
for T in (60, 100, 250, 500, 1000, 2500, 100_000):
    lo = np.empty(reps)
    hi = np.empty(reps)
    tr = np.empty(reps)
    for r in range(reps):
        ev = np.linalg.eigvalsh(np.cov(rng.standard_normal((T, N)), rowvar=False))
        lo[r], hi[r], tr[r] = ev[0], ev[-1], ev.mean()
    q = np.sqrt(N / T)
    print(f"  {T:7d} {N / T:6.3f} {hi.mean():12.4f} {(1 + q) ** 2:9.4f} {lo.mean():11.4f}"
          f" {(1 - q) ** 2:9.4f} {(hi / np.maximum(lo, 1e-12)).mean():11.2f} {tr.mean():11.4f}")
# =>   sample spectrum of Sigma-hat when the true Sigma is the 50x50 identity
#            T    N/T    mean lmax   MP edge   mean lmin   MP edge   mean cond   mean tr/N
#           60  0.833       3.4499    3.6591      0.0108    0.0076      384.59      0.9971
#          100  0.500       2.7793    2.9142      0.0961    0.0858       29.52      1.0012
#          250  0.200       2.0188    2.0944      0.3242    0.3056        6.25      0.9987
#          500  0.100       1.6872    1.7325      0.4842    0.4675        3.49      1.0005
#         1000  0.050       1.4689    1.4972      0.6176    0.6028        2.38      1.0000
#         2500  0.020       1.2870    1.3028      0.7486    0.7372        1.72      1.0002
#       100000  0.001       1.0429    1.0452      0.9579    0.9558        1.09      1.0000
```

The truth in every row is fifty eigenvalues of exactly $1$. At $T=250$, five observations per asset, the sample spectrum runs from $0.32$ to $2.02$ and the condition number averages $6.25$ — a sixfold spread invented entirely by sampling, from data generated with no structure whatsoever. At $T=60$ the top eigenvalue is $3.45$ and the bottom is $0.011$, a condition number of $385$, and any principal-component analysis of this matrix would report a leading "factor" absorbing seven percent of the variance in a world where every direction is identical.

The two edge columns say the distortion is not noise but law. The Marchenko–Pastur limits $\lambda_\pm=(1\pm\sqrt{N/T})^{2}$ predict where the extreme eigenvalues go as a function of $N/T$ alone, and the simulated averages sit just inside them in every row, converging as $T$ grows. The control is the last row: at a hundred thousand observations the spectrum has collapsed to $[0.958,\,1.043]$ and the condition number is $1.09$, which confirms the estimator is consistent and that the spread above is a small-sample effect rather than a bug.

The trace column is the one to keep. $\mathrm{tr}(\hat\Sigma)/N$ reads within $0.003$ of $1$ in every row including the worst — the total variance is estimated superbly at every sample size, because it is a linear functional of the entries and the entries are unbiased. All of the error is in how that correct total is *distributed* across directions, and every diagnostic computed one entry at a time is blind to it by construction.

!!! warning "An unbiased covariance matrix has biased eigenvalues, and every optimizer reads the eigenvalues rather than the entries"
    A portfolio optimizer does not consume $\hat\Sigma$; it consumes $\hat\Sigma^{-1}$, whose eigenvalues are the reciprocals of the ones tabulated above. The smallest sample eigenvalue is the most badly underestimated one, so its reciprocal is the most badly overestimated, and the eigenvector attached to it is the direction the optimizer will identify as nearly riskless and lever into. At $T/N=5$ that direction is assigned about a third of its true variance; at $T/N=1.2$ it is assigned about a hundredth. This is the mechanism behind the $11.81\times$ gross exposure and $293.9\%$ volatility that [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) measures, and no entrywise standard error would have flagged it, because entrywise there is nothing wrong.

## Shrinkage Buys Conditioning and Cannot Buy a Mean

The repair suggested by the previous section is to pull the spectrum back in. Shrinkage does exactly that, by averaging $\hat\Sigma$ with a target that has no spread at all,

$$\hat\Sigma_\alpha=(1-\alpha)\hat\Sigma+\alpha\bar\lambda I,\qquad\bar\lambda=\frac{\mathrm{tr}(\hat\Sigma)}{N},$$

with the Ledoit–Wolf choice of $\alpha$ estimating the value that minimises expected squared error to the unknown $\Sigma$. The target preserves the trace, so shrinkage moves variance between directions without creating or destroying any, which is precisely the quantity the previous block found to be misallocated.

```python
import numpy as np

rng = np.random.default_rng(631)
N, rho, reps = 9, 0.619, 400                                   # nine sectors, published rho
vol = np.linspace(0.12, 0.28, N) / np.sqrt(252)
Sigma = np.outer(vol, vol) * ((1 - rho) * np.eye(N) + rho)
mu = vol * np.linspace(0.02, 0.05, N)                          # daily Sharpe 0.02 to 0.05
L, one = np.linalg.cholesky(Sigma), np.ones(N)


def ledoit_wolf(R):                                            # shrink toward (tr S / N) I
    T, n = R.shape
    Z = R - R.mean(axis=0)
    S = Z.T @ Z / T
    m = np.trace(S) / n
    d2 = ((S - m * np.eye(n)) ** 2).sum() / n
    b2 = min(sum(((np.outer(z, z) - S) ** 2).sum() for z in Z) / (n * T ** 2), d2)
    return b2 / d2, (b2 / d2) * m * np.eye(n) + (1 - b2 / d2) * S


def mv(S):
    w = np.linalg.solve(S, one)
    w = w / w.sum()
    return np.sqrt(252 * w @ Sigma @ w)                        # true out-of-sample vol


def tan(S, m):
    w = np.linalg.solve(S, m)
    w = w / np.linalg.norm(w)
    return np.sqrt(252) * (w @ mu) / np.sqrt(w @ Sigma @ w)    # true out-of-sample Sharpe


print(f"  truth: condition number {np.linalg.cond(Sigma):5.1f},"
      f"  min-variance vol {mv(Sigma):.4f},  max Sharpe {tan(Sigma, mu):.3f}")
print("      T   T/N   LW intensity   cond(S)   cond(LW)    minvar vol      LW    a = 1"
      "    tangency SR      LW   true mu")
for T in (504, 126, 63):
    acc = np.zeros(9)
    for _ in range(reps):
        R = rng.standard_normal((T, N)) @ L.T + mu
        S, mh = np.cov(R, rowvar=False), R.mean(axis=0)
        a, Sa = ledoit_wolf(R)
        F = np.trace(S) / N * np.eye(N)
        acc += [a, np.linalg.cond(S), np.linalg.cond(Sa), mv(S), mv(Sa), mv(F),
                tan(S, mh), tan(Sa, mh), tan(S, mu)]
    a, cs, ca, v0, v1, v2, s0, s1, s2 = acc / reps
    print(f"  {T:5d} {T / N:5.0f} {a:14.4f} {cs:9.1f} {ca:10.1f} {v0:13.4f} {v1:7.4f}"
          f" {v2:8.4f} {s0:14.3f} {s1:7.3f} {s2:9.3f}")
# =>   truth: condition number  44.3,  min-variance vol 0.1012,  max Sharpe 1.012
#          T   T/N   LW intensity   cond(S)   cond(LW)    minvar vol      LW    a = 1    tangency SR      LW   true mu
#        504    56         0.0081      46.2       43.6        0.1020  0.1020   0.1630          0.433   0.436     1.004
#        126    14         0.0322      51.1       40.1        0.1046  0.1044   0.1630          0.213   0.220     0.980
#         63     7         0.0629      60.3       36.7        0.1085  0.1075   0.1630          0.134   0.149     0.945
```

The intensity column reproduces the published behaviour. It rises monotonically as observations per asset fall — $0.81\%$, $3.22\%$, $6.29\%$ against the published $1.85\%$, $5.42\%$, $9.69\%$ at the same three ratios — and it is smaller here because an exactly equicorrelated universe is closer to the shrinkage target than nine real sectors are. The direction and the order of magnitude are the theory's, and the estimator is doing its job: the condition number moves from $46.2$, $51.1$, $60.3$ down to $43.6$, $40.1$, $36.7$, improving fastest exactly where the sample is thinnest.

The minimum-variance columns show the payoff, and it is real and small. True out-of-sample volatility falls from $0.1085$ to $0.1075$ at seven observations per asset, against a best-attainable $0.1012$. The $\alpha=1$ column is the control at the other extreme: shrinking all the way to the scaled identity makes the minimum-variance portfolio equal-weighted, at a volatility of $0.1630$, which is much worse than either. So the useful range is narrow and the estimator finds it.

The tangency columns are the indictment. Fed its own estimated mean, the maximum-Sharpe portfolio delivers $0.134$ out of a possible $1.012$ at seven observations per asset, and shrinkage lifts that to $0.149$ — a gain of $0.015$. Handed the *true* $\mu$ and the same unshrunk, badly conditioned $\hat\Sigma$, the identical construction delivers $0.945$. Fixing the covariance matrix is worth fifteen thousandths of a Sharpe point; fixing the mean vector is worth eight hundred and ten, fifty-four times as much, and it is available from no estimator at all. That is the precise content of the rule that shrinking $\hat\Sigma$ cannot fix mean–variance, and it is why the previous section's warning is about the optimizer's behaviour rather than about its inputs.

## Three Ways Back Onto the Cone

When a matrix arrives that is not PSD — pairwise-complete estimation, blended vendors, an expert view pasted over a fitted table — it has to be replaced by one that is. There are three families of repair and they are not interchangeable.

??? note "Proof that clipping negative eigenvalues gives the nearest positive semi-definite matrix in Frobenius norm"
    Let $A$ be symmetric with spectral decomposition $A=Q\Lambda Q^\top$, $Q$ orthogonal, and let $B$ range over symmetric PSD matrices. The Frobenius norm is orthogonally invariant, $\lVert QXQ^\top\rVert_F=\lVert X\rVert_F$, so writing $\tilde B=Q^\top BQ$ gives

    $$\lVert A-B\rVert_F^{2}=\lVert\Lambda-\tilde B\rVert_F^{2}=\sum_i(\lambda_i-\tilde B_{ii})^{2}+\sum_{i\neq j}\tilde B_{ij}^{2},$$

    and $\tilde B$ is PSD exactly when $B$ is. The off-diagonal sum is minimised at zero, and taking $\tilde B$ diagonal is compatible with PSD-ness, so the problem decouples into $N$ independent scalar problems: choose $\tilde B_{ii}\ge0$ nearest to $\lambda_i$, whose answer is $\max(\lambda_i,0)$. Hence the minimiser is $B^{\star}=Q\max(\Lambda,0)Q^\top$ — clip the negative eigenvalues to zero and rebuild.

    The load-bearing hypothesis is the orthogonal invariance of the Frobenius norm, which is what allowed the problem to be rotated into the eigenbasis and decoupled. Impose one further constraint that is *not* orthogonally invariant — a diagonal of exactly ones, which is what makes a matrix a correlation matrix rather than a covariance matrix — and the rotation is no longer free, the decoupling fails, and no closed form survives. That is why [Correlation Matrices](03-correlation-matrices.md) needs an iterative algorithm for what looks like the same problem, and it is the cleanest example in this part of a constraint that costs nothing to state and everything to satisfy.

| Repair | What it changes | What it preserves | What it costs | When it is the wrong tool |
|---|---|---|---|---|
| Eigenvalue clipping | negative eigenvalues set to zero | the eigenvectors, and the nearest-matrix property in Frobenius norm | the diagonal moves, so variances change silently | whenever the diagonal is meaningful and unnegotiable |
| Shrinkage toward $\bar\lambda I$ | the whole spectrum, pulled toward its mean | the trace, the eigenvectors, and symmetry | a bias toward a target chosen before the data | when the matrix was never indefinite and the problem is $\hat\mu$ |
| Factor structure $B\Sigma_FB^\top+D$ | the model, not the matrix | PSD-ness by construction, for any $B$ and any $D\ge0$ | $N-K$ directions of risk collapse into $D$ | when the residual covariance is not close to diagonal |

The three rows differ in what they are willing to be wrong about. Clipping is a projection: it takes the matrix seriously and asks for the smallest edit that makes it admissible, which is right when the input was nearly valid and the invalidity is numerical. Shrinkage is an estimator: it assumes the input is a noisy view of something better conditioned, which is right when $N/T$ is the problem and wrong when it is not. Factor structure is a model: it declines to repair anything and instead re-parameterises the matrix so that no invalid value is representable.

!!! note "A factor model is a parameterization that cannot leave the positive semi-definite cone"
    For any loadings $B\in\mathbb{R}^{N\times K}$, any PSD factor covariance $\Sigma_F$ and any non-negative diagonal $D$, the matrix $B\Sigma_FB^\top+D$ is PSD, because $w^\top B\Sigma_FB^\top w=(B^\top w)^\top\Sigma_F(B^\top w)\ge0$ and $w^\top Dw\ge0$. There is no combination of inputs that produces an inadmissible output, so the failure mode "this matrix is not the covariance of anything" is not merely unlikely under a factor model — it is unreachable. The price is that the model can only be *wrong*, never impossible, which is a strictly better failure mode: a wrong factor model produces a valid covariance matrix that misprices some portfolios, and a wrong table produces a negative variance that an optimizer will find and exploit. It also explains why the estimation problem shrinks from $N(N+1)/2$ free parameters to $NK+K(K+1)/2+N$, which at $N=50$ and $K=3$ is $1{,}275$ against $206$.

## A Matrix Is Not a Table

A covariance matrix has $N(N+1)/2$ free parameters and is estimated from $TN$ numbers, so the entire question of how much of it is real is settled by one ratio. At $N/T$ near zero it is a measurement; at $N/T$ near one it is mostly a record of which noise arrived first; and in between it is a measurement of the total variance together with a badly distorted account of where that variance lives.

None of this is visible entrywise, which is the reason the page has laboured the distinction between a matrix and a table. Every entry can be unbiased, every entry can have a defensible standard error, every pairwise scatter plot can look fine, and the object as a whole can still be unusable for the one operation anybody performs on it, which is inversion. The diagnostics that do see it are matrix-level and cheap: the spectrum, the condition number, the ratio $N/T$, and — for a matrix that arrived rather than one that was estimated — the sign of the smallest eigenvalue.

The practical rule is to report $N/T$ beside every covariance matrix, and to treat that ratio rather than any entry as the number saying how much of the matrix is real. Print the smallest eigenvalue whenever a matrix crosses a system boundary, because a table that has stopped being a covariance matrix is silent about it and an optimizer is not. And do not invert a covariance matrix estimated from fewer than a few multiples of $N$ observations, or report that shrinkage repaired it — shrinkage repaired the matrix, and the matrix was not what was broken.
