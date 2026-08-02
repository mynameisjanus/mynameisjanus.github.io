# Linear Transformations

Under an affine map the mean carries the intercept and the covariance does not, and the sandwich $A\Sigma A^\top$ is the entire content of portfolio variance, factor risk, currency hedging, beta neutralization and cross-sectional demeaning. The result needs no density, no invertibility, no independence and no normality. That makes it the only formula in this part that survives contact with real returns, and the one worth trusting when the others are being quoted.

This page covers the mean and covariance of an affine image, the sandwich formula proved from the outer product, the risk numbers that are special cases of it, cross-sectional demeaning as a projection that manufactures a correlation of exactly $-1/(N-1)$, whitening and the rotation it is only defined up to, and the rank a singular map destroys. It does not derive the density-level Jacobian $f_Y(y)=f_X(h(y))\lvert\det Dh(y)\rvert$, which is [Change of Variables](../part-03-random-variables/09-change-of-variables.md); it does not develop projection matrices, the normal equations or Cholesky, which are [Basic Linear Algebra Review](../part-01-mathematical-foundations/05-linear-algebra-review.md); it assumes no distribution, so the fact that an affine image of a normal is *normal* is [Multivariate Gaussian Distribution](05-multivariate-gaussian.md); and it fits no $A$ to data, which is [Part XIII](../part-13-regression/index.md).

The trading stake is one subtraction that turns a directional book into a hedged one. [Cross-Sectional and Volatility Strategies](../../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) reports nine sectors co-moving at an average correlation of $0.58$, with $63\%$ of a typical sector's monthly variance being the market wearing a sector costume, and then subtracts each month's cross-sectional mean and finds the average correlation at $-0.11$. That subtraction is a matrix, the sign flip is not a discovery, and the third section derives the number it had to be — along with what the distance from that number measures.

## The Mean Carries the Intercept and the Covariance Does Not

Let $X$ be a random vector in $\mathbb{R}^{n}$ with mean $\mu$ and covariance $\Sigma$, and let $Y=AX+b$ for a fixed $A\in\mathbb{R}^{m\times n}$ and $b\in\mathbb{R}^{m}$. Then

$$\mathbb{E}[Y]=A\mu+b\qquad\text{and}\qquad\mathrm{cov}(Y)=A\Sigma A^\top.$$

The asymmetry is the whole result and it is the single most common slip in writing it down. A shift relocates the centre and leaves the spread alone; a linear map moves both, and moves the second quadratically. This is the vector form of the scalar asymmetry [Variance](../part-04-expectation-and-moments/02-variance.md) established, with the quadratic now appearing as a matrix on each side rather than as a squared scalar.

??? note "Proof of the sandwich formula from the outer product, with no independence and no distribution"
    The mean follows from the previous part's linearity, which needed nothing. For the covariance, subtract it off first:

    $$Y-\mathbb{E}[Y]=(AX+b)-(A\mu+b)=A(X-\mu),$$

    where $b$ cancels before any product is formed — which is exactly why the intercept never reaches the second moment. Substituting into the definition,

    $$\mathrm{cov}(Y)=\mathbb{E}\big[A(X-\mu)\,(X-\mu)^\top A^\top\big]=A\,\mathbb{E}\big[(X-\mu)(X-\mu)^\top\big]\,A^\top=A\Sigma A^\top,$$

    the constants $A$ and $A^\top$ moving through the entrywise expectation by linearity. The transpose lands on the right because $(A(X-\mu))^\top=(X-\mu)^\top A^\top$, and the result is symmetric because $\Sigma$ is.

    The only hypothesis is that the second moments of $X$ exist. Nothing was assumed about independence between coordinates, about the shape of any margin, about $A$ being square or invertible, or about a density existing at all. The contrast with [Change of Variables](../part-03-random-variables/09-change-of-variables.md) is worth drawing precisely, because that page derives the *density* of a linear image and needs three things this proof does not: a density to transform, a square $A$, and $\det A\neq0$ so the inverse map exists. The moment-level result needs none of the three, which is why it is what a risk system computes, and why it still returns an answer for the singular $A$ of the fifth section — where the density-level formula has nothing whatever to say, because the image has no density.

## Every Risk Number in This Book Is This Sandwich

Once the formula is written down, most of what a risk system reports turns out to be one instance of it with a different $A$.

| Object | $A$ | Shape of $A$ | The sandwich becomes |
|---|---|---|---|
| Portfolio variance | $w^\top$ | $1\times N$ | $w^\top\Sigma w$, a scalar |
| A spread or pair | $(1,-1)$ | $1\times2$ | $\sigma_1^{2}+\sigma_2^{2}-2\sigma_{12}$ |
| A $K$-factor risk model | $B^\top$ | $K\times N$ | $B^\top\Sigma B$, the factor covariance implied by the assets |
| A beta-neutralized sleeve | $I-\beta e^\top$ | $N\times N$ | the residual covariance after hedging out one asset |
| A currency-hedged return | $[\,I\ \ -h\,]$ | $N\times(N+1)$ | asset covariance net of the hedged currency leg |
| Cross-sectional demeaning | $I-\tfrac1N\mathbf{1}\mathbf{1}^\top$ | $N\times N$ | the covariance of the market-relative returns |
| Aggregating to $K$ sectors | a $0/1$ membership matrix | $K\times N$ | the sector-level covariance |

Reading down the $A$ column is the useful exercise. Every row is a decision about what the report will be able to see, and only the first two are ever described that way. The last row in particular is chosen by whoever built the reporting hierarchy, usually years earlier, and the fifth section is about what it costs.

```python
import numpy as np

rng = np.random.default_rng(653)
N, T, nu = 6, 2_000_000, 5
B = np.array([[1.0, 0.3], [1.1, -0.2], [0.9, 0.6], [1.2, 0.1], [0.8, -0.5], [1.0, 0.0]])
Sigma = B @ np.diag([0.010 ** 2, 0.006 ** 2]) @ B.T + np.diag(np.full(N, 0.006 ** 2))
mu = np.full(N, 0.0003)
A = np.array([[1, 1, 1, 1, 1, 1.0], [1, 1, 1, -1, -1, -1.0], [1, 0, 0, 0, 0, -1.0]]) / 6
b = np.array([-0.02, 0.0, 0.005])
L = np.linalg.cholesky(Sigma)
print("  E[AX + b] = A mu + b and cov(AX + b) = A Sigma A', under two laws with one Sigma")
print("          law     max |mean err|   max |cov err|   exc kurt of row 1   ann vol of row 1")
for name, Z in (("gaussian", rng.standard_normal((T, N))),
                (f"multivariate t{nu}", rng.standard_normal((T, N))
                 / np.sqrt(rng.chisquare(nu, (T, 1)) / nu) * np.sqrt((nu - 2) / nu))):
    Y = (mu + Z @ L.T) @ A.T + b
    y0 = Y[:, 0]
    print(f"  {name:>17s} {np.abs(Y.mean(0) - (A @ mu + b)).max():15.2e}"
          f" {np.abs(np.cov(Y, rowvar=False) - A @ Sigma @ A.T).max():15.2e}"
          f" {(((y0 - y0.mean()) / y0.std()) ** 4).mean() - 3:19.3f}"
          f" {np.sqrt(252) * y0.std():19.4f}")
# =>   E[AX + b] = A mu + b and cov(AX + b) = A Sigma A', under two laws with one Sigma
#              law     max |mean err|   max |cov err|   exc kurt of row 1   ann vol of row 1
#               gaussian        5.59e-06        1.34e-07              -0.000              0.1636
#        multivariate t5        5.61e-06        7.33e-08               5.060              0.1635
```

Two laws are fed through the same $A$: a multivariate normal and a multivariate $t_5$ rescaled to have the *identical* $\Sigma$. Both reproduce $A\mu+b$ to $5.6\times10^{-6}$ and $A\Sigma A^\top$ to about $10^{-7}$, on all three rows of $A$ including the long–short row whose weights sum to zero. The formula does not know or care which law it was handed.

The last two columns say what that costs. The first row of $A$ — an equal-weight book — has an excess kurtosis of $-0.000$ under the normal and $5.060$ under the $t_5$, while its annualized volatility is $0.1636$ against $0.1635$. The sandwich sees the second moment and is structurally incapable of seeing anything else, so two books with the same reported risk differ by a factor of many in how often they lose four standard deviations. That is simultaneously why the formula is safe — it cannot be wrong about what it computes — and why it is not sufficient.

## Demeaning Is a Projection That Manufactures -1/(N-1)

The centring matrix $M=I-\tfrac1N\mathbf{1}\mathbf{1}^\top$, applied across assets rather than across time, subtracts each date's cross-sectional mean. It is symmetric and idempotent, so it is an orthogonal projection onto the space of weight vectors summing to zero, and it annihilates $\mathbf{1}$ exactly.

??? note "Proof that cross-sectional demeaning forces an average correlation of minus one over N minus one"
    Let $Y=MX$. By the sandwich, $\mathrm{cov}(Y)=M\Sigma M$. The key property is that $M\mathbf{1}=0$, so

    $$M\Sigma M\,\mathbf{1}=M\Sigma\,(M\mathbf{1})=0,$$

    and every row of $M\Sigma M$ sums to zero. Reading row $i$, that says $(M\Sigma M)_{ii}=-\sum_{j\neq i}(M\Sigma M)_{ij}$: the average covariance of asset $i$ with the others is exactly $-(M\Sigma M)_{ii}/(N-1)$, whatever $\Sigma$ was.

    To turn covariances into correlations the variances must agree. Take $\Sigma=\sigma^{2}C$ with $C=(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top$ equicorrelated. Since $C=(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top$ and $M\mathbf{1}\mathbf{1}^\top M=0$,

    $$M\Sigma M=\sigma^{2}(1-\rho)M,$$

    whose diagonal is $\sigma^{2}(1-\rho)(1-1/N)$ and whose off-diagonal is $-\sigma^{2}(1-\rho)/N$, so every pairwise correlation is exactly $-1/(N-1)$ — independent of $\rho$, and independent of $\sigma$.

    Equicorrelation is what makes every pair equal, but it is not what does the work. The row-sum-zero property holds for *any* $\Sigma$; what equicorrelation adds is that the demeaned variances are all the same, which is the step that converts a statement about covariances into one about correlations. So at $N=9$ the prediction is $-1/8=-0.125$ exactly when the sector volatilities are equal, and the published $-0.11$ is therefore not a measurement of residual signal surviving the hedge. It is a measurement of how *unequal* the sector volatilities are, and the next block calibrates it: a spread of about four to one reproduces it.

```python
import numpy as np

rng = np.random.default_rng(659)
T, rho = 200_000, 0.58                                         # published sector co-movement


def off(n):
    return ~np.eye(n, dtype=bool)


print("  cross-sectional demeaning: M = I - (1/N) 11', applied to each date's row")
print("       N   vol spread    rho before    rho after    -1/(N-1)   lmin(M Sigma M)   tr kept")
for N, spread in ((9, 1.0), (9, 2.0), (9, 4.0), (2, 1.0), (20, 1.0)):
    vol = np.linspace(1.0, spread, N) * 0.012
    Sigma = np.outer(vol, vol) * ((1 - rho) * np.eye(N) + rho)
    X = rng.standard_normal((T, N)) @ np.linalg.cholesky(Sigma).T
    Y = X - X.mean(axis=1, keepdims=True)                      # demean across assets, per date
    M = np.eye(N) - np.ones((N, N)) / N
    print(f"  {N:6d} {spread:12.1f} {np.corrcoef(X, rowvar=False)[off(N)].mean():13.4f}"
          f" {np.corrcoef(Y, rowvar=False)[off(N)].mean():12.4f} {-1 / (N - 1):11.4f}"
          f" {np.linalg.eigvalsh(M @ Sigma @ M)[0]:17.2e}"
          f" {np.trace(M @ Sigma @ M) / np.trace(Sigma):9.4f}")
# =>   cross-sectional demeaning: M = I - (1/N) 11', applied to each date's row
#           N   vol spread    rho before    rho after    -1/(N-1)   lmin(M Sigma M)   tr kept
#           9          1.0        0.5791      -0.1250     -0.1250         -6.78e-21    0.3733
#           9          2.0        0.5809      -0.1188     -0.1250         -1.33e-20    0.3990
#           9          4.0        0.5802      -0.1047     -0.1250         -1.00e-19    0.4490
#           2          1.0        0.5791      -1.0000     -1.0000          0.00e+00    0.2100
#          20          1.0        0.5796      -0.0526     -0.0526          5.28e-21    0.3990
```

The homogeneous row lands on $-0.1250$ to four decimals against a prediction of $-0.1250$, and the $N=20$ row lands on $-0.0526$ against $-0.0526$. The control is $N=2$, where demeaning a pair leaves two numbers that are exact negatives of each other and the correlation is $-1.0000$ — the extreme case of the same identity, and a useful reminder that a market-neutral pair is perfectly anticorrelated by construction rather than by discovery.

The volatility-spread rows are the calibration the proof promised. Holding $\rho$ at $0.58$ and spreading the nine volatilities by two to one moves the demeaned correlation from $-0.1250$ to $-0.1188$; spreading them four to one moves it to $-0.1047$. The published $-0.11$ sits between those, so the entire distance between the arithmetic floor and the measured number is accounted for by sector volatilities differing by something like a factor of three or four. There is no residual to interpret and nothing about the sign flip that needed to be discovered — the operation forced it, and the only information in the measured value is a dispersion of volatilities.

The last two columns say what was paid for it. The smallest eigenvalue of $M\Sigma M$ is zero to machine precision in every row, at eigenvector $\mathbf{1}$: the market direction has not been reduced, it has been annihilated, and it cannot be recovered from the demeaned data by any means. And the surviving trace is $37\%$ to $45\%$ of the original, so demeaning discards more than half the total variance in the universe. That is exactly what a hedge is supposed to do, and it is exactly why a cross-sectional strategy needs the residual to contain signal — it has thrown away the majority of the variation in exchange for removing one direction.

## Whitening Is Only Defined Up to a Rotation

The reverse operation is to remove *all* the structure: find $W$ with $\mathrm{cov}(WX)=I$. Any $W$ satisfying $W\Sigma W^\top=I$ will do, and by the sandwich there are infinitely many.

??? note "Proof that a whitening matrix is unique only up to an orthogonal rotation"
    Suppose $W\Sigma W^\top=I$ and let $Q$ be any orthogonal matrix, $QQ^\top=I$. Then by the sandwich applied to $QW$,

    $$(QW)\Sigma(QW)^\top=Q\big(W\Sigma W^\top\big)Q^\top=QIQ^\top=QQ^\top=I,$$

    so $QW$ whitens as well. The converse closes the description. If $W_1$ and $W_2$ both whiten and $W_2$ is invertible, then $W_2\Sigma W_2^\top=I$ rearranges to $\Sigma=W_2^{-1}W_2^{-\top}$, and setting $Q=W_1W_2^{-1}$ gives

    $$QQ^\top=W_1W_2^{-1}W_2^{-\top}W_1^\top=W_1\Sigma W_1^\top=I,$$

    so $Q$ is orthogonal and $W_1=QW_2$. Every pair of whitening matrices therefore differs by an orthogonal factor, and there are no others. Cholesky whitening takes $W=L^{-1}$ with $\Sigma=LL^\top$; principal-component whitening takes $W=\Sigma^{-1/2}=Q\Lambda^{-1/2}Q^\top$; both are whitening matrices and neither is more correct.

    Orthogonality of $Q$ is the whole hypothesis, and the consequence is that there is no such thing as *the* independent coordinates of a correlated system. Cholesky whitening orders the coordinates by the order in which the assets happened to be stored in the file — permute the columns and the "factors" change — while principal-component whitening orders them by variance, which is at least intrinsic but is still a choice. Both produce coordinates that are exactly uncorrelated and neither produces coordinates that mean anything, so an economic story attached to a whitened coordinate is a story about the algorithm. The one setting where the ambiguity resolves is when the whitened coordinates are also *independent* rather than merely uncorrelated, which under a normal law is [Multivariate Gaussian Distribution](05-multivariate-gaussian.md) and under any other law is a strictly stronger requirement that no rotation supplies for free.

```python
import numpy as np

rng = np.random.default_rng(661)
N = 10
beta = np.linspace(0.7, 1.3, N)
Sigma = np.outer(beta, beta) * 0.010 ** 2 + np.diag(np.full(N, 0.007 ** 2))
L = np.linalg.cholesky(Sigma)


def maxoff(Z):
    C = np.corrcoef(Z, rowvar=False)
    return np.abs(C[~np.eye(N, dtype=bool)]).max()


print("  whitening with Sigma-hat estimated on the same sample it whitens")
print("       T    T/N    max |off-diag corr| in sample   out of sample   fresh, true Sigma")
for T in (25, 50, 250, 25_000):
    X = rng.standard_normal((T, N)) @ L.T
    ev, Q = np.linalg.eigh(np.cov(X, rowvar=False))
    W = Q @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-18))) @ Q.T
    Xo = rng.standard_normal((T, N)) @ L.T
    print(f"  {T:6d} {T / N:6.1f} {maxoff(X @ W):30.2e}"
          f" {maxoff(Xo @ W):15.4f} {maxoff(Xo @ np.linalg.inv(np.linalg.cholesky(Sigma).T)):19.4f}")
Lw = np.linalg.inv(L)
ev, Q = np.linalg.eigh(Sigma)
Pw = Q @ np.diag(1 / np.sqrt(ev)) @ Q.T
print(f"  two whitenings of the same true Sigma: both give cov = I to"
      f" {max(np.abs(Lw @ Sigma @ Lw.T - np.eye(N)).max(), np.abs(Pw @ Sigma @ Pw.T - np.eye(N)).max()):.1e},"
      f"  and differ by a rotation with ||Q - I||_F = {np.linalg.norm(Lw @ np.linalg.inv(Pw) - np.eye(N)):.4f}")
# =>   whitening with Sigma-hat estimated on the same sample it whitens
#           T    T/N    max |off-diag corr| in sample   out of sample   fresh, true Sigma
#          25    2.5                       2.41e-15          0.6721              0.3905
#          50    5.0                       2.72e-15          0.4472              0.2409
#         250   25.0                       3.26e-15          0.1767              0.1486
#       25000 2500.0                       2.15e-15          0.0196              0.0200
#      two whitenings of the same true Sigma: both give cov = I to 1.3e-15,  and differ by a rotation with ||Q - I||_F = 1.3138
```

The in-sample column reads $10^{-15}$ at every sample size, including $T=25$ where there are two and a half observations per asset. That is not a good fit; it is an identity. Whitening by the inverse square root of the sample covariance makes the sample covariance exactly $I$ by construction, so any test of orthogonality run on the same data that produced $W$ returns a pass regardless of what is true, and it returns the same pass on twenty-five observations that it returns on twenty-five thousand.

Out of sample the number is real. At $T/N=2.5$ the largest surviving off-diagonal correlation is $0.6721$, against a noise floor of $0.3905$ — the last column, which whitens fresh data with the *true* $\Sigma$ and so measures only the sampling error of a correlation estimated on twenty-five points. The excess of $0.67$ over $0.39$ is the artefact. By $T=25{,}000$ both columns have collapsed to about $0.02$ and agree, which is the control confirming the estimator is consistent and the failure is small-sample.

The last line closes the proof numerically. Cholesky whitening and principal-component whitening of the same true $\Sigma$ both deliver $\mathrm{cov}=I$ to $1.3\times10^{-15}$, and they differ by a rotation at Frobenius distance $1.3138$ from the identity — a substantial rotation, not a rounding difference. Two analysts, whitening the same matrix with the same intent, produce coordinate systems that disagree about which direction is which.

!!! warning "Any statistic computed after whitening with a covariance matrix estimated on the same data is orthogonal by construction rather than by evidence"
    This is the mechanism behind a whole family of results that cannot be reproduced: residuals that are uncorrelated with the factors, principal components that are exactly independent, risk models whose leftover exposures are zero. All of them are true in sample by the algebra above and none of them is evidence of anything, because the same output arrives when the input is noise. The failure does not degrade gracefully either — the in-sample number is $10^{-15}$ at $T/N=2.5$ and $10^{-15}$ at $T/N=2500$, so there is no sample size at which the diagnostic starts working and none at which it visibly stops. The only honest version is to estimate $W$ on one sample and test on another, and to report the true-$\Sigma$ noise floor beside the result, because at small $T$ that floor is most of the number.

## A Singular A Loses Rank, and the Loss Is Permanent

When $A$ has more columns than rows — aggregating $N$ positions into $K<N$ exposures, which is what every risk report does — the sandwich still applies and the result is unambiguous:

$$\mathrm{rank}\big(A\Sigma A^\top\big)\le\min\big(\mathrm{rank}\,A,\ \mathrm{rank}\,\Sigma\big)\le K.$$

The aggregated covariance describes a $K$-dimensional world, and the $N-K$ directions of risk that the aggregation cannot represent are not attenuated or approximated — they are gone, and no operation on the aggregated matrix recovers them. Demeaning in the previous section is the extreme case that is visible, because the annihilated direction is the market and everybody knows it was removed. A sector-level report annihilates intra-sector dispersion in exactly the same way, and nobody announces it.

The size of the loss is available in one line, since the trace is the total variance: the fraction discarded is $1-\mathrm{tr}(A\Sigma A^\top)/\mathrm{tr}(\Sigma)$ when $A$ is a projection, which the demeaning block prints as the surviving $37\%$ to $45\%$. That number belongs beside every aggregated risk figure and is almost never there.

!!! note "The sandwich formula is why a risk model can be restated in any basis and remain the same model"
    Take any invertible $A$. Then $\Sigma$ and $A\Sigma A^\top$ contain identical information, because $\Sigma=A^{-1}(A\Sigma A^\top)A^{-\top}$ recovers one from the other exactly. A risk model expressed in assets, in factors, in principal components or in currency-hedged units is one model wearing four coordinate systems, and any quantity that changes when the basis changes was never a property of the model. This is a genuinely useful invariance — it is what licenses converting a factor risk model into asset-level exposures and back without loss — and it is precisely what fails when $A$ is not invertible. So the same formula that makes a change of basis free makes an aggregation irreversible, and the only difference between the two cases is a determinant.

## Risk Is Reported in a Basis

Every risk report is an $A$ applied to a $\Sigma$, and the choice of $A$ is a choice about what will be allowed to be visible. A sector-aggregated report cannot show intra-sector risk. A factor report cannot show residual risk. A market-neutral book's covariance matrix cannot show market risk, because the operation that made it market-neutral removed that direction from the matrix along with the exposure.

None of this is hidden, and all of it is unremarked, because an aggregation matrix does not look like a modelling assumption. Volatilities are estimated, correlations are estimated, expected returns are argued over in meetings, and the mapping from positions to reported exposures is a piece of reporting infrastructure that somebody configured years ago and nobody signs off on. It is nevertheless the last thing applied before a human reads a number, and by the rank bound it is the only step in the chain that destroys information irreversibly.

The practical rule is to write $A$ down explicitly, and to report beside every aggregated risk number the fraction of $\mathrm{tr}(\Sigma)$ that the aggregation discarded. When that fraction is small the report is a summary; when it is more than half — as it is for any market-neutral transformation — the report is a different question's answer, and the difference is worth a line of text. A decomposition is an answer about a basis, and the basis was chosen before the data was consulted, so name it every time or do not present it.
