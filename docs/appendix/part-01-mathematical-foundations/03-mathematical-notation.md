# Mathematical Notation

Notation is not decoration. A formula like $\mathbf{P}(A\mid B)$ or $\hat\sigma^2$ carries claims — that $A$ and $B$ are events in the same space, that the quantity is an *estimate* rather than the truth it estimates — and a reader who has to guess those claims cannot check the argument. This page fixes the conventions the rest of the appendix and the course use, so that every later page can be read without reverse-engineering its symbols.

Two working notes before the tables. First, everything here is descriptive: the symbols below are the ones actually used elsewhere in the book, not a wish list. Second, notation is *local by convention and global by discipline* — $\sigma$ is a standard deviation nearly everywhere and a permutation nowhere in this book, and where a symbol must be reused for something else, the page that reuses it says so.

## Numbers and Sets of Numbers

| Symbol | Meaning |
|---|---|
| $\mathbb{N}$ | Natural numbers $1,2,3,\ldots$ (this book starts them at 1; index variables that start at 0 say so) |
| $\mathbb{Z}$ | Integers $\ldots,-1,0,1,\ldots$ |
| $\mathbb{Q}$ | Rationals, ratios $p/q$ of integers with $q\neq 0$ |
| $\mathbb{R}$ | Real numbers |
| $\mathbb{R}^n$ | Real vectors with $n$ components — the space a portfolio weight vector lives in |
| $\mathbb{R}^{m\times n}$ | Real matrices with $m$ rows and $n$ columns |
| $[0,1]$, $(0,1)$ | Closed and open intervals — square brackets include the endpoint, parentheses exclude it |
| $[0,\infty)$ | Half-open: $\infty$ is never an element, so its side is always open |

The interval convention matters more often than it looks. A probability lies in $[0,1]$ — both endpoints are attainable — while a uniform random draw lies in $[0,1)$, because the standard generator can return exactly 0 and never returns exactly 1 (see [Random Number Generation](../part-09-monte-carlo-methods/01-random-number-generation.md)). A correlation lies in $[-1,1]$; a $p$-value in $[0,1]$; a variance in $[0,\infty)$.

## Sets, Membership, and Quantifiers

| Symbol | Read as |
|---|---|
| $x\in A$ | $x$ is an **element** of $A$ |
| $x\notin A$ | $x$ is not an element of $A$ |
| $A\subset B$ | $A$ is a **subset** of $B$ — every element of $A$ is in $B$ |
| $\{x\in\Omega : P(x)\}$ | The set of $x$ in $\Omega$ **such that** the condition $P(x)$ holds |
| $\varnothing$ | The empty set |
| $\Omega$ | The universal set; in probability, the **sample space** |
| $A^\mathsf{C}$ | The complement of $A$ within $\Omega$ |
| $A\cup B$, $A\cap B$ | Union, intersection |
| $\lvert A\rvert$ | The number of elements in $A$ (also $\#A$ in some texts; this book uses $\lvert\cdot\rvert$) |
| $\forall$, $\exists$ | For all, there exists |

The distinction between $\in$ and $\subset$ is the one beginners collapse and every proof depends on: $3\in\{1,2,3\}$ but $\{3\}\subset\{1,2,3\}$. The single die face is an *element* of the sample space; the event "the roll is odd" is a *subset* of it. Probabilities are assigned to subsets, never to elements — which is why [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) can handle continuous outcomes, where every individual element has probability zero and the subsets still have positive probability. The set operations themselves are developed in [Sets and Functions](01-sets-and-functions.md).

This book writes $\subset$ for "is a subset of" without implying the subset is strict; when strictness matters, the text says "proper subset" in words rather than relying on $\subsetneq$.

## Sums and Products

$$\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n, \qquad \prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdots a_n.$$

The letter under the $\sum$ is the **index**, the expressions above and below give its range, and the index is *bound* — it exists only inside the sum, so $\sum_i a_i$ and $\sum_k a_k$ are the same number. Three conventions:

- **Empty sums and products.** A sum over no terms is $0$; a product over no terms is $1$. These are not edge-case hacks — they are what makes $\sum_{i=1}^{n}$ behave correctly at $n=0$ and keeps recursive definitions from needing special cases.
- **Unindexed ranges.** $\sum_{i} a_i$ means "over every $i$ in whatever index set is in play", used when the range is obvious and cluttering.
- **Conditions under the sum.** $\sum_{i:\,x_i>0} a_i$ restricts to indices satisfying a condition.

Reindexing is the manipulation most often done silently. Substituting $j = i-1$ turns $\sum_{i=1}^{n} a_i$ into $\sum_{j=0}^{n-1} a_{j+1}$: the terms are identical, only the label moved. This is the mechanical step behind almost every geometric-series derivation in [Sequences and Infinite Series](04-sequences-and-series.md).

**Double sums** run over pairs. When the ranges do not depend on each other,

$$\sum_{i=1}^{n}\sum_{j=1}^{m} a_{ij} = \sum_{j=1}^{m}\sum_{i=1}^{n} a_{ij},$$

and for finite sums the exchange is always legal. For infinite sums it is legal under absolute convergence and can fail otherwise — the caveat that [Sequences and Infinite Series](04-sequences-and-series.md) makes precise. When the inner range depends on the outer, as in $\sum_{i=1}^{n}\sum_{j=i+1}^{n}$ — the standard idiom for "over all unordered pairs", which appears in every portfolio-variance expansion — the exchange is legal but the limits must be rewritten, not copied.

**Telescoping** is worth naming because it is how nearly every closed form in this appendix is obtained:

$$\sum_{i=1}^{n}(b_i - b_{i-1}) = b_n - b_0,$$

since everything in between cancels in pairs.

## Probability and Statistics

This is the table the rest of the appendix is written against.

| Symbol | Meaning | Developed in |
|---|---|---|
| $\Omega$ | Sample space — the set of all possible outcomes | [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) |
| $A, B, C$ | Events, i.e. subsets of $\Omega$ | [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) |
| $\mathbf{P}(A)$ | Probability of the event $A$ | [Probability Axioms](../part-02-probability-foundations/02-probability-axioms.md) |
| $\mathbf{P}(A\mid B)$ | Probability of $A$ **given** $B$ | [Conditional Probability](../part-02-probability-foundations/03-conditional-probability.md) |
| $X, Y, Z$ | Random variables (capitals) | [Random Variables](../part-03-random-variables/01-random-variables.md) |
| $x, y, z$ | Realized values of those random variables (lowercase) | [Random Variables](../part-03-random-variables/01-random-variables.md) |
| $X\sim F$ | $X$ is **distributed as** $F$ | [Common Distributions](../part-05-common-distributions/index.md) |
| $F_X(x)$ | CDF, $\mathbf{P}(X\le x)$ | [CDFs](../part-03-random-variables/02-cumulative-distribution-functions.md) |
| $p_X(x)$ | PMF of a discrete $X$, $\mathbf{P}(X=x)$ | [PMFs](../part-03-random-variables/03-probability-mass-functions.md) |
| $f_X(x)$ | PDF of a continuous $X$ | [PDFs](../part-03-random-variables/04-probability-density-functions.md) |
| $\mathbb{E}[X]$ | Expected value | [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) |
| $\mathrm{var}(X)$ | Variance | [Variance](../part-04-expectation-and-moments/02-variance.md) |
| $\mathrm{cov}(X,Y)$ | Covariance | [Covariance](../part-04-expectation-and-moments/04-covariance.md) |
| $\rho$ | Correlation coefficient | [Correlation](../part-04-expectation-and-moments/05-correlation.md) |
| $\mu$, $\sigma$, $\sigma^2$ | Population mean, standard deviation, variance | [Expected Value](../part-04-expectation-and-moments/01-expected-value.md) |
| $\mathbf{1}\{\cdot\}$, $\mathbf{1}_A$ | Indicator — $1$ when the condition holds, $0$ otherwise | [Sets and Functions](01-sets-and-functions.md) |
| iid | Independent and identically distributed | [Independence](../part-02-probability-foundations/05-independence.md) |

Three habits inside that table do real work:

**Capitals are random, lowercase is realized.** $\mathbf{P}(X\le x)$ reads "the probability that the random variable $X$ lands at or below the fixed number $x$." Once a sample is in hand, its values are ordinary numbers; the randomness lives in the process that produced them, not in the spreadsheet.

**Hats mark estimates.** $\theta$ is a parameter of the world and $\hat\theta$ is a number computed from a finite sample. The course's central claim — that most apparent edges are estimation error — is a claim about the gap between the two, so any expression mixing them deserves a second read. $\hat\mu$ and $\hat\Sigma$ appear constantly in [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md) for exactly this reason: the optimizer's formula is stated in $\mu$ and $\Sigma$, and what you can actually feed it is $\hat\mu$ and $\hat\Sigma$.

**Bars mark sample averages.** $\bar X = \frac{1}{n}\sum_{i=1}^{n} X_i$ is itself a random variable — a fact that is easy to nod past and is the entire content of [Sampling Distributions](../part-10-statistics-foundations/03-sampling-distributions.md).

!!! note "$\Sigma$ and $\sum$ are different symbols"
    Upright capital sigma $\Sigma$ denotes a covariance matrix; the summation sign $\sum$ is a larger, distinct glyph. Context always separates them — $\Sigma^{-1}\mu$ inverts a matrix, $\sum_i x_i$ adds numbers — but they look similar enough at small sizes that it is worth flagging once. A related collision, $\Omega$ for the sample space against $\Omega(\cdot)$ for an asymptotic lower bound, is handled below.

## Vectors and Matrices

| Convention | This book writes |
|---|---|
| Vectors | Lowercase: $w$ for weights, $x$ for a feature vector, $\mu$ for a mean vector |
| Default orientation | **Column** vectors, so $w\in\mathbb{R}^n$ is $n\times 1$ |
| Matrices | Uppercase: $\Sigma$ for covariance, $X$ for a design matrix, $P$ for a transition matrix |
| Transpose | $w^\top$ |
| Inverse | $\Sigma^{-1}$ |
| Vector of ones | $\mathbf{1}$ |
| Identity matrix | $I$ |
| Euclidean norm | $\lVert x\rVert$; absolute value of a scalar is $\lvert x\rvert$ |

The column convention is what makes $w^\top\Sigma w$ a scalar and $\Sigma w$ a vector, and it is why portfolio variance is written $w^\top\Sigma w$ rather than $w\Sigma w^\top$. It also collides, permanently and unavoidably, with how data is stored: a returns matrix in pandas or NumPy is rows-as-dates, columns-as-assets, which is the transpose of the textbook's "each column is an observation" layout. The mismatch is the source of a large share of all shape errors in quantitative code, and the fix is mechanical — check that an inner dimension is the number of *assets* when the operation is a portfolio aggregation, and the number of *dates* when it is a time-series reduction. [NumPy and Vectorization](../../part-02-python/01-numpy-and-vectorization.md) works through the broadcasting rules; the algebra is in [Basic Linear Algebra Review](05-linear-algebra-review.md).

## Functions and Operators

| Symbol | Meaning |
|---|---|
| $f:A\to B$ | $f$ maps elements of $A$ to elements of $B$ |
| $f^{-1}(S)$ | Preimage: the set of inputs landing in $S$ (see [Sets and Functions](01-sets-and-functions.md)) |
| $\log x$, $\ln x$ | **Natural** logarithm in both cases — this book uses the two interchangeably, and any other base is written explicitly as $\log_2$ or $\log_{10}$ |
| $\exp(x)$, $e^x$ | The exponential function |
| $\lfloor x\rfloor$, $\lceil x\rceil$ | Floor and ceiling |
| $x^+$ | $\max(x,0)$ — the positive part, used for option payoffs and drawdown floors |
| $\arg\max_{\theta} g(\theta)$ | The **argument** maximizing $g$, not the maximum value |
| $\mathrm{sgn}(x)$ | Sign: $-1$, $0$, or $+1$ — the mathematical form of a long/flat/short position |
| $\propto$ | Proportional to — equal up to a constant factor that the context does not need |
| $\approx$, $\equiv$ | Approximately equal; defined to be / identically equal |

$\arg\max$ versus $\max$ is worth its own sentence because the whole of estimation depends on it: $\max_\theta \ell(\theta)$ is the peak height of the log-likelihood, a number nobody cares about, while $\hat\theta = \arg\max_\theta \ell(\theta)$ is the parameter estimate, which is the entire output of [Maximum Likelihood Estimation](../part-11-parameter-estimation/03-maximum-likelihood-estimation.md).

The $\propto$ symbol earns its keep in Bayesian work, where $p(\theta\mid x)\propto p(x\mid\theta)\,p(\theta)$ says everything of interest and suppresses a normalizing constant that is often impossible to write down — the observation [Posterior Distributions](../part-16-bayesian-statistics/03-posterior-distributions.md) is built on. It appears again in $w\propto\Sigma^{-1}\mu$, where the proportionality constant is fixed later by a budget constraint.

## Asymptotic Notation

Asymptotic notation describes how a quantity behaves in a limit — typically as a sample size or problem size grows. It is used in two different ways in this book, and conflating them causes confusion, so both are stated.

**Growth of functions.** For functions $f$ and $g$ with $g$ eventually positive:

| Notation | Means | Informally |
|---|---|---|
| $f(n) = O(g(n))$ | $\lvert f(n)\rvert \le C\,g(n)$ for some constant $C$ and all large $n$ | grows no faster than |
| $f(n) = o(g(n))$ | $f(n)/g(n)\to 0$ | grows strictly slower than |
| $f(n) = \Theta(g(n))$ | Both $O(g)$ and bounded below by a constant multiple of $g$ | grows at the same rate as |
| $f(n)\sim g(n)$ | $f(n)/g(n)\to 1$ | is asymptotically equal to, constants included |

The constant is deliberately discarded: $O$ answers "how does the cost scale when the problem doubles", not "how long does it take". An $O(n)$ algorithm with a terrible constant can lose to an $O(n^2)$ one at every size you actually run — which is why the profiling discipline in [Profiling, Refactoring, and Versioning](../../part-09-software-engineering/05-profiling-refactoring-versioning.md) measures rather than reasons.

$\sim$ is the strict one and gets used where the constant matters: Stirling's approximation $n!\sim\sqrt{2\pi n}\,(n/e)^n$ in [Counting Principles](02-counting-principles.md) would be useless as an $O$ statement.

!!! warning "$\Omega$ means two different things"
    In complexity theory $\Omega(g)$ is the asymptotic *lower* bound, the mirror of $O(g)$. In probability $\Omega$ is the sample space. This book resolves the collision by never using the complexity $\Omega$ — lower bounds are stated in words — so a bare $\Omega$ is always the sample space.

**Stochastic order.** The probabilistic versions attach the same idea to random quantities: $X_n = o_p(1)$ means $X_n\to 0$ in probability, and $X_n = O_p(1)$ means the sequence is bounded in probability (for any $\epsilon$ there is an $M$ with $\mathbf{P}(\lvert X_n\rvert > M) < \epsilon$ for all $n$). These are the working notation of [Slutsky's Theorem](../part-07-asymptotic-theory/05-slutskys-theorem.md) and the [Delta Method](../part-07-asymptotic-theory/04-delta-method.md), where the point of an argument is usually that some remainder term is $o_p(1)$ and can therefore be discarded without changing the limiting distribution.

The single most useful asymptotic fact in the entire course is a statement of this kind: the standard error of a sample mean is $O(n^{-1/2})$. Four times the data halves the error. It is why a decade of daily returns still leaves a Sharpe ratio uncertain to roughly $\pm 0.2$, and no amount of methodological care improves that rate — only more independent data does.

### Complexity in Practice

The scaling exponents that actually appear in this course:

| Cost | Example |
|---|---|
| $O(n)$ | A single pass over $n$ bars — computing returns, a running sum |
| $O(n\log n)$ | Sorting; the FFT behind fast convolution |
| $O(n^2)$ | All pairwise correlations in an $n$-asset universe; every pair-trade screen |
| $O(n^3)$ | Matrix inversion, eigen-decomposition, Cholesky — every covariance-based optimizer |

The gap between the last two is the one that bites, and it can be seen without a stopwatch. Doubling the universe multiplies the pair count by four and the optimizer's work by eight:

```python
for n in (50, 100, 200, 400):
    pairs = n * (n - 1) // 2          # O(n^2): distinct asset pairs
    solve = n ** 3                    # O(n^3): dense factorization work
    print(f"n={n:4d}  pairs={pairs:8,d}  n^3={solve:14,d}")
# => n=  50  pairs=   1,225  n^3=       125,000
#    n= 100  pairs=   4,950  n^3=     1,000,000
#    n= 200  pairs=  19,900  n^3=     8,000,000
#    n= 400  pairs=  79,800  n^3=    64,000,000
```

Eight-fold work for twice the assets is why covariance-based methods are re-estimated on a schedule rather than on every bar, and the pair count is why a pairs screen over a few hundred tickers is a multiple-testing problem before it is a computational one — the connection [Data Snooping Bias](../part-15-multiple-testing/04-data-snooping-bias.md) makes explicit.

## Reading a Dense Formula

Notation is a compression scheme, and the skill it demands is decompression. Take the sample covariance between two assets:

$$\hat\Sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^{n}\left(r_{ij} - \bar r_j\right)\left(r_{ik} - \bar r_k\right).$$

Read it in this order:

1. **The left side names the output.** $\hat\Sigma$ is a matrix (capital), it is an estimate (hat), and $jk$ selects the entry for the pair of assets $j$ and $k$. The formula therefore has to be evaluated once per pair.
2. **The sum names the loop.** $i$ runs over $1$ to $n$: observations, i.e. dates. So each entry consumes the entire history of two assets.
3. **The summand names the quantity.** $r_{ij}$ is the return of asset $j$ on date $i$, and $\bar r_j$ is that asset's sample mean, so each factor is a deviation from the mean. Products of deviations are positive when the two assets move together.
4. **The prefactor names the correction.** Dividing by $n-1$ rather than $n$ compensates for having estimated the two means from the same data — the bias correction derived in [Bias and Variance](../part-10-statistics-foundations/07-bias-and-variance.md).

Every symbol answered a question: what is produced, what is looped over, what is accumulated, what is corrected. The same four questions dismantle the mean-variance weight vector $w\propto\Sigma^{-1}\mu$: the output is a vector of portfolio weights, there is no loop because the sum is hidden inside the matrix product, the accumulation is "expected returns re-weighted by the inverse covariance", and the proportionality hides a normalization to a budget. That last reading is what makes the failure mode visible — $\Sigma^{-1}$ amplifies the directions in which returns vary least, and those are the directions estimated worst, which is precisely the pathology documented in [Portfolio Optimization and Correlation](../../part-08-portfolio-management/04-portfolio-optimization-and-correlation.md).

## Conventions This Book Follows

A short list of decisions, gathered so they can be cited rather than rediscovered:

- $\log$ is natural. Base 2 and base 10 are written with explicit subscripts.
- Probabilities use $\mathbf{P}$; densities and mass functions use $f$ and $p$. Lowercase $p$ standing alone is a parameter (a success probability), not a probability measure.
- Returns are written $r$; when the distinction matters, $r_t$ is a simple return and $\log(1+r_t)$ is written out rather than given its own symbol. [Exponentials, Logarithms, and Growth](07-exponentials-logarithms-growth.md) explains why the difference is worth the extra characters.
- $n$ is a sample size, $T$ a number of time periods, $N$ a number of assets or trials — and any page that needs a different reading says so in its first paragraph.
- Annualization uses 252 trading days, everywhere, without re-deriving the choice.
- Estimates carry hats, population quantities do not, and sample averages carry bars. Where an expression mixes all three, it is doing something subtle and deserves the slow read.
