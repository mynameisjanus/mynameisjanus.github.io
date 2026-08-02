# Sampling Methods

Every distribution a simulation consumes is manufactured from the uniform stream of the previous page by a transformation, and the theorem that licenses this is one line long. What takes the rest of a page is that the one-line theorem is unusable for most of the laws anyone actually needs — the normal has no elementary inverse, the discrete case turns into a search problem whose cost dominates a bootstrap, and the multivariate case has no inverse at all. The devices that fill those gaps are not approximations. Each is exact, each is exact for a different reason, and the reasons are worth knowing because the failure modes differ.

This page covers the probability integral transform and the generalized inverse that makes it hold without a continuity assumption, the cost structure of sampling a discrete law and the alias table that removes it, the four standard routes to a normal draw and the hard ceiling each inherits from its uniform, composition and scale mixture as the constructions for laws that nothing inverts, and what changes when the object being sampled is a path rather than a point. It does not build the generator underneath, which is [Random Number Generation](01-random-number-generation.md); it estimates nothing with the samples and quotes no error bar, which is [Monte Carlo Simulation](03-monte-carlo-simulation.md); it does not reweight or reject draws in order to change the law being sampled, which are [Importance Sampling](04-importance-sampling.md) and [Rejection Sampling](05-rejection-sampling.md); it does not sample from a law known only up to a constant, which is [Part XVII](../part-17-statistical-computing/index.md); it derives none of the distributions it samples, which is [Part V](../part-05-common-distributions/index.md); and it fits nothing to data.

The trading stake is a sentence the course writes about its own resampling machinery. [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) states that "every simulated path in this lesson starts as a stream of uniforms pushed through an inverse distribution function", and [Change of Variables](../part-03-random-variables/09-change-of-variables.md) names this page as where that construction is built, observing that "going right builds a sample from a law you specify". The sentence is true, and it is also the expensive half of a bootstrap: the second section shows that the textbook way of pushing a uniform through an empirical distribution function costs three thousand comparisons per draw on the course's own twenty-four-year sample, and that the fix costs one.

## The Probability Integral Transform Runs in Both Directions

Let $F$ be any distribution function and $U\sim\mathrm{Unif}(0,1)$. Define the **generalized inverse**

$$F^{-1}(u)=\inf\{x:F(x)\geq u\},$$

which exists for every distribution function whatever, continuous or not, and coincides with the ordinary inverse when there is one. Then $F^{-1}(U)$ has distribution function $F$. This is **inverse-transform sampling**, and it is the entire theory: one uniform in, one draw from any law out.

??? note "Proof that the generalized inverse works with no continuity assumption anywhere, and what the assumption hides when it is made"
    The claim is that $\mathbf{P}(F^{-1}(U)\leq x)=F(x)$ for every $x$, and it follows from a single equivalence,

    $$F^{-1}(u)\leq x\iff u\leq F(x),$$

    valid for all $u\in(0,1)$ and all real $x$. Right to left: if $u\leq F(x)$ then $x$ belongs to $\{y:F(y)\geq u\}$, so the infimum defining $F^{-1}(u)$ is at most $x$. Left to right: if $F^{-1}(u)\leq x$, take $y_n\downarrow F^{-1}(u)$ with $F(y_n)\geq u$; right-continuity of $F$ gives $F(F^{-1}(u))\geq u$, and monotonicity gives $F(x)\geq F(F^{-1}(u))\geq u$. Taking probabilities of the right-hand event,

    $$\mathbf{P}(F^{-1}(U)\leq x)=\mathbf{P}(U\leq F(x))=F(x),$$

    the last step because $U$ is uniform and $F(x)\in[0,1]$.

    The load-bearing hypothesis is right-continuity, which every distribution function has by definition, and *not* continuity or invertibility, which most do not have. Where the difference bites is the discrete case: for a law on finitely many atoms $F$ is a step function, $F^{-1}$ is the search that finds which step a uniform landed on, and the theorem is saying that the search is not an approximation to sampling but is literally sampling. The reverse direction — pushing data through a fitted $F$ and checking the result is uniform — is the same equivalence read the other way, and is the probability integral transform as a goodness-of-fit device rather than as a sampler.

## A Discrete Law Is a Search Problem Before It Is a Probability Problem

For a law on atoms $x_1,\dots,x_K$ with probabilities $p_1,\dots,p_K$, inverse-transform sampling says: form the cumulative sums, draw $U$, return the first atom whose cumulative sum exceeds it. The probability content of that instruction is trivial and its cost is not. Scanning the cumulative array from the left costs $\sum_k k\,p_k$ comparisons per draw, which is a property of the law rather than of the code — a distribution concentrated on late atoms is expensive and one concentrated on early atoms is cheap. Binary search on the same array costs $\lceil\log_2 K\rceil$ regardless of the law. **Walker's alias method** costs one comparison regardless, by a construction worth stating in a sentence: decompose the law into $K$ two-outcome coins of equal weight, draw a coin uniformly, flip it.

```python
import numpy as np
from scipy.stats import chisquare

rng = np.random.default_rng(9021)
atoms, draws, boot = 500, 4_000_000, 6_158
p = 1.0 / np.arange(1, atoms + 1)                              # a Zipf-weighted empirical law
p /= p.sum()
cdf = np.cumsum(p)

q, alias = p * atoms, np.zeros(atoms, dtype=np.int64)          # Vose's alias tables
small = [i for i in range(atoms) if q[i] < 1]
large = [i for i in range(atoms) if q[i] >= 1]
while small and large:
    s, ell = small.pop(), large.pop()
    alias[s], q[ell] = ell, q[ell] - (1 - q[s])
    (small if q[ell] < 1 else large).append(ell)

by_search = np.searchsorted(cdf, rng.random(draws))            # binary search on the cdf
k = rng.integers(atoms, size=draws)
by_alias = np.where(rng.random(draws) < q[k], k, alias[k])     # one comparison, always

print(f"  drawing {draws} times from a {atoms}-atom law, three lookups into the same cdf")
print("   method             comparisons per draw    max |p_hat - p|    chi-square p")
for name, cost, idx in (("linear cdf scan", (by_search + 1).mean(), by_search),
                        ("binary search", np.ceil(np.log2(atoms)), by_search),
                        ("alias table", 1.0, by_alias)):
    n = np.bincount(idx, minlength=atoms)
    print(f"  {name:<19} {cost:20.2f} {np.abs(n / draws - p).max():18.2e}"
          f" {chisquare(n, draws * p).pvalue:15.4f}")
print(f"  analytic scan cost sum (i+1)p_i = {(np.arange(1, atoms + 1) * p).sum():.2f}"
      f"   an iid bootstrap of {boot} days would scan {(boot + 1) / 2:.1f}")
# =>   drawing 4000000 times from a 500-atom law, three lookups into the same cdf
#       method             comparisons per draw    max |p_hat - p|    chi-square p
#      linear cdf scan                    73.68           2.64e-04          0.6457
#      binary search                       9.00           2.64e-04          0.6457
#      alias table                         1.00           2.41e-04          0.7561
#      analytic scan cost sum (i+1)p_i = 73.61   an iid bootstrap of 6158 days would scan 3079.5
```

The right-hand columns are the check that there is nothing to choose between the methods statistically. All three reproduce the target probabilities to about $2.5\times10^{-4}$ on four million draws, which is what four million draws buys, and the chi-square goodness-of-fit tests return $0.6457$ and $0.7561$. These are the same law sampled three ways, not three approximations to it.

The left column is the entire content. The measured scan cost of $73.68$ comparisons per draw sits on the analytic $\sum_k k\,p_k=73.61$, binary search is flat at $9$, and the alias table is flat at $1$. The last line converts this into the units the course cares about: the iid bootstrap of [Bootstrap and Monte Carlo Methods](../../part-03-statistics/05-bootstrap-and-monte-carlo.md) resamples $6{,}158$ daily returns ten thousand times over, so it draws sixty-two million indices from a $6{,}158$-atom law, and a left-to-right scan of that law's cumulative array averages $3{,}079.5$ comparisons per draw against the alias table's one.

**A resampling scheme is a distribution question that spends nearly all of its time being a data-structure question**, which is why every serious library implements the alias table or a binary search and none of them implements the loop the textbook writes. The same arithmetic explains a practice that otherwise looks like superstition: for the special case of equal weights — the ordinary bootstrap — nobody builds a table at all, because drawing a uniform integer is already one operation and the cumulative array was never needed.

!!! note "The alias table is rejection sampling with the rejection removed, which is why it costs one comparison rather than one and a bit"
    Walker's construction takes the $K$ probabilities, scales them by $K$ so they average one, and repeatedly pairs an atom holding less than its share against one holding more, transferring exactly enough mass to fill the deficient one to unity. Each atom ends as a two-outcome coin — itself with probability $q_k$, its alias otherwise — and the atoms are equiprobable, so a uniform integer plus one flip is a complete draw. The relationship to [Rejection Sampling](05-rejection-sampling.md) is exact: the alias table is accept–reject with a uniform proposal, restructured so that a rejection is redirected to a known alias instead of being discarded. Rejection throws work away and pays for it with a random runtime; the alias table computes in advance where each rejection would have landed and pays for it once, in $O(K)$ setup. The trade is the one every sampling problem eventually poses — precomputation against per-draw cost — and it is settled by how many draws you intend to take, which for a bootstrap is always enough.

## The Normal Has No Elementary Inverse, and Four Devices Around It

The normal distribution function has no inverse in elementary functions, which is why $\Phi$ and $\Phi^{-1}$ are tabulated in every statistics book and implemented as rational approximations in every library. [Continuous Uniform Distribution](../part-05-common-distributions/09-continuous-uniform-distribution.md) flags the omission and defers the repair here. There are four repairs in common use and they are not interchangeable.

Inverse transform still works if you are willing to call a numerical $\Phi^{-1}$: `scipy.special.ndtri` is accurate to near machine precision and costs one uniform per normal. **Box–Muller** converts two uniforms into two normals in closed form by working in polar coordinates. **Marsaglia's polar method** is Box–Muller with the trigonometry replaced by a rejection step. The **ziggurat**, which is what `default_rng.standard_normal` actually runs, covers the density with a stack of equal-area rectangles and falls through to a separate tail routine on the rare miss.

??? note "Proof that Box–Muller returns two independent standard normals, by one change of variables"
    Let $U_1,U_2$ be independent uniforms on $(0,1)$ and set

    $$R=\sqrt{-2\ln U_1},\qquad \Theta=2\pi U_2,\qquad Z_1=R\cos\Theta,\quad Z_2=R\sin\Theta.$$

    Since $-\ln U_1\sim\mathrm{Exp}(1)$, the variable $R^{2}=-2\ln U_1$ is exponential with mean $2$, so $R$ has density $re^{-r^{2}/2}$ on $(0,\infty)$, and $\Theta$ is uniform on $(0,2\pi)$ independently of it. The joint density of $(R,\Theta)$ is therefore $\tfrac{1}{2\pi}re^{-r^{2}/2}$. The map to Cartesian coordinates has Jacobian $r$, so the joint density of $(Z_1,Z_2)$ is

    $$\frac{1}{2\pi}re^{-r^{2}/2}\cdot\frac1r=\frac{1}{2\pi}e^{-(z_1^{2}+z_2^{2})/2},$$

    which factorizes into two standard normal densities. Independence is a consequence of the factorization rather than an assumption fed in. Marsaglia's variant draws a point uniformly in the unit disc by rejection and reads $\cos\Theta$ and $\sin\Theta$ off its coordinates, so the same argument applies with the trigonometric calls replaced by a division, at the cost of the $1-\pi/4$ of proposals that land outside the disc.

    The load-bearing fact is the appearance of $\ln U_1$, and it is where the method's one limitation comes from. The largest radius the construction can produce is $\sqrt{-2\ln u_{\min}}$, where $u_{\min}$ is the smallest positive value the uniform generator can return, so the transformation inherits a hard ceiling from the *representation* of its input rather than from the normal law it is sampling. Every inverse-transform route has the same property for the same reason, and the ceilings differ.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import kstest

rng = np.random.default_rng(9023)
n = 8_000_000
half = n // 2

z_inv = ndtri(rng.random(n))                                   # one uniform in, one normal out
u1, u2 = rng.random(half), rng.random(half)
r = np.sqrt(-2 * np.log(u1))
z_bm = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
v = rng.random((2, int(n / 0.7854 / 2) + 1)) * 2 - 1           # Marsaglia's polar rejection
s = (v ** 2).sum(axis=0)
keep = v[:, (s > 0) & (s < 1)][:, :half]
s = (keep ** 2).sum(axis=0)
z_pol = (keep * np.sqrt(-2 * np.log(s) / s)).ravel()
z_zig = rng.standard_normal(n)

print(f"  four routes to a standard normal, {n} draws each")
print("   route                       uniforms per normal      sd    KS p-value"
      "    max |z| seen    ceiling")
for name, cost, z, cap in (("inverse cdf (ndtri)", 1.0, z_inv, f"{-ndtri(2.0**-53):.4f}"),
                           ("Box-Muller", 1.0, z_bm, f"{np.sqrt(2 * 53 * np.log(2)):.4f}"),
                           ("Marsaglia polar", 4 / np.pi, z_pol, f"{np.sqrt(2 * 53 * np.log(2)):.4f}"),
                           ("ziggurat (standard_normal)", 1.0, z_zig, "none")):
    print(f"  {name:<27} {cost:19.3f} {z.std():7.4f} {kstest(z, 'norm').pvalue:13.4f}"
          f" {np.abs(z).max():15.3f} {cap:>10}")
print(f"  on 32-bit uniforms the same two ceilings fall to {-ndtri(2.0**-32):.4f}"
      f" and {np.sqrt(2 * 32 * np.log(2)):.4f}")
# =>   four routes to a standard normal, 8000000 draws each
#       route                       uniforms per normal      sd    KS p-value    max |z| seen    ceiling
#      inverse cdf (ndtri)                       1.000  1.0003        0.6871           5.574     8.2095
#      Box-Muller                                1.000  1.0001        0.6660           5.309     8.5717
#      Marsaglia polar                           1.273  1.0001        0.2249           5.411     8.5717
#      ziggurat (standard_normal)                1.000  0.9998        0.3987           5.057       none
#      on 32-bit uniforms the same two ceilings fall to 6.2303 and 6.6604
```

The middle columns confirm all four are correct: standard deviations of $1.0003$, $1.0001$, $1.0001$ and $0.9998$, and Kolmogorov–Smirnov $p$-values of $0.6871$, $0.6660$, $0.2249$ and $0.3987$ against the exact normal on eight million draws. On the question of whether they sample the right law there is nothing to discuss.

The cost column separates them. Marsaglia's polar method consumes $1.273$ uniforms per normal — the reciprocal of the $\pi/4\approx0.7854$ of proposals landing inside the unit disc — in exchange for replacing a sine and a cosine with a division, which was a good trade on the hardware of 1964 and is a marginal one now. The others consume one. The `max |z| seen` column is the same in every row up to sampling noise, between $5.06$ and $5.57$, because the maximum of $n$ standard normals grows like $\sqrt{2\ln n}=5.62$ regardless of how they were manufactured.

The last column is the one worth carrying away. Each route has a hard ceiling set by the smallest positive value its uniform can represent, and that ceiling is a fact about arithmetic rather than about probability. NumPy's `random()` returns a $53$-bit fixed-point value, so its smallest positive output is $2^{-53}$: inverse transform can never return a draw beyond $8.2095$, and Box–Muller can never exceed $\sqrt{2\cdot53\ln2}=8.5717$. Run the identical mathematics against a $32$-bit uniform — which is what a great many older and hand-written implementations do — and the two ceilings collapse to $6.2303$ and $6.6604$.

**No number of draws repairs a ceiling, because the deficiency is in the alphabet rather than in the sample size.** For a normal this is comfortably academic, since reaching even $6.66$ would take on the order of $10^{10}$ draws. For a fat-tailed law it is not academic at all: the same $u_{\min}$ pushed through a $t$ with $2.6$ degrees of freedom gives a worst representable draw $270$ times larger at $53$-bit resolution than at $32$-bit, because a heavy tail magnifies the gap between two very small probabilities into a gap between two very large losses. A tail-risk engine's worst case can be a property of its mantissa.

## Composition and Mixture Build the Laws That Nothing Inverts

When the inverse is unavailable or the law is multivariate, the constructive route is to build the target out of laws already in hand. Sums give convolutions — a gamma with integer shape is a sum of exponentials, each of which is $-\ln U$. Ratios give scale mixtures — a Student-$t$ with $\nu$ degrees of freedom is a standard normal divided by $\sqrt{\chi^2_\nu/\nu}$, which is the representation making the $t$ a normal whose variance is itself random. Mixtures give mixtures, by drawing the component first. Order statistics give order statistics, by sorting.

The scale-mixture route matters because it generalizes where inverse transform cannot. There is no inverse distribution function for a multivariate $t$, but the mixture representation carries over verbatim: draw a correlated normal vector, divide the whole vector by one shared $\sqrt{\chi^2_\nu/\nu}$, and the result is a multivariate $t$ with the right correlation and with tail dependence that a Gaussian coupling of $t$ marginals cannot produce. The single shared divisor is the entire difference, and it is why one construction has assets crashing together in a stress test and the other does not.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import ks_2samp
from scipy.stats import t as tdist

rng = np.random.default_rng(9027)
n, nu = 4_000_000, 2.6                                         # the tail Part III fits to SPY
scale = np.sqrt(nu / (nu - 2))                                 # so every route has unit variance
levels = [0.99, 0.999, 0.9999]

t_inv = tdist.ppf(rng.random(n), nu) / scale                   # inverse cdf, one uniform in
w = rng.chisquare(nu, n)
t_mix = rng.standard_normal(n) * np.sqrt(nu / w) / scale       # normal over a chi-square root
sig = np.quantile(t_inv, 0.99) / -ndtri(0.01)                  # a normal matched at 99% VaR
t_bad = rng.standard_normal(n) * sig

print(f"  {n} draws of a unit-variance t({nu}), three constructions, read down the tail")
print("   route                       sd       99%      99.9%     99.99%    max seen")
for name, x in (("inverse cdf", t_inv), ("normal / sqrt(chi2/nu)", t_mix),
                ("normal matched at 99%", t_bad)):
    q = np.quantile(x, levels)
    print(f"  {name:<24} {x.std():7.3f} {q[0]:9.3f} {q[1]:10.3f} {q[2]:10.3f} {x.max():11.1f}")
q = tdist.ppf(levels, nu) / scale
print(f"  {'exact':<24} {'':7} {q[0]:9.3f} {q[1]:10.3f} {q[2]:10.3f}")
ks = ks_2samp(t_inv[:200_000], t_mix[:200_000]).pvalue
print(f"  inverse cdf against the scale mixture, two-sample KS p = {ks:.4f}")
# =>   4000000 draws of a unit-variance t(2.6), three constructions, read down the tail
#       route                       sd       99%      99.9%     99.99%    max seen
#      inverse cdf                0.987     2.476      6.155     14.927       210.6
#      normal / sqrt(chi2/nu)     1.030     2.473      6.204     15.169        84.0
#      normal matched at 99%      1.064     2.474      3.291      3.942         5.8
#      exact                                2.474      6.184     15.069
#      inverse cdf against the scale mixture, two-sample KS p = 0.5002
```

The first two rows are the same distribution built two ways. Their quantiles agree with the exact values — $2.476$ and $2.473$ against $2.474$, $6.155$ and $6.204$ against $6.184$, $14.927$ and $15.169$ against $15.069$ — and a two-sample Kolmogorov–Smirnov test on two hundred thousand draws of each returns $p=0.5002$. The scale mixture is not an approximation to the $t$; it is a second definition of it.

The standard-deviation column is worth pausing on before the third row, because it looks like noise and is not. The three estimates read $0.987$, $1.030$ and $1.064$ on four million draws each, when a well-behaved law would pin its own standard deviation to four decimals at that sample size. A $t$ with $\nu=2.6$ has finite variance and an infinite fourth moment, so its sample standard deviation is a consistent estimator with no usable error bar, and it wanders. **The moment a fat-tailed model is most often calibrated on is the one it estimates worst.**

The third row is what that permits. A normal calibrated to reproduce the $t$'s $99\%$ quantile exactly — a routine and defensible choice, since $99\%$ is the level most risk reports check — matches at $2.474$ and then fails downward at every level beyond it. It puts the one-in-a-thousand loss at $3.291$ where the truth is $6.155$, the one-in-ten-thousand loss at $3.942$ where the truth is $14.927$, and its worst draw in four million is $5.8$ against the $t$'s $210.6$. Calibrating at the checked quantile and reporting the unchecked one understates the extreme loss by a factor of nearly four, with every diagnostic that was actually run returning agreement.

!!! warning "A sampler agreeing with its target where you check it is free to disagree where it matters, and the checking is the reason nobody looks further"
    The failure above is not a bad model. It is a good model evaluated at the point where it was fitted. Two properties of a heavy tail conspire: the quantile function is steep, so a small error in probability becomes a large error in loss; and the moments that would reveal the steepness are the ones the data estimates worst, since the fourth moment of a $t(2.6)$ does not exist and its sample version diverges as more data arrives. The defensible practice is to state which functional the sampler was matched on and to report a second one it was *not* matched on — an expected shortfall beside a value at risk, a $99.9\%$ level beside a $99\%$ one — because two constructions only ever differ visibly where neither was calibrated. [Risk Measurement](../../part-08-portfolio-management/01-risk-measurement.md) makes the same point about the three VaR methods it compares when it says that "Monte Carlo assumes whatever you sample from", and what is being assumed is a whole distribution rather than a number.

## Sampling a Path Is Not Sampling a Point

Everything so far samples a scalar or a vector at one instant. A simulation of a strategy samples a path, and a path's joint law is not determined by its marginals. Three constructions cover most of what is needed, and each fails in a characteristic way when misapplied.

Correlated draws come from a factorization of the target covariance. If $\Sigma=LL^{\top}$ is a Cholesky decomposition and $z$ is a vector of independent standard normals, then $Lz$ has covariance $\Sigma$ exactly. Any other factorization reproduces the covariance equally well and everything else differently — the Cholesky factor is lower triangular, so the first coordinate depends on the first shock alone, which is convenient for interpretation and is a modelling assumption smuggled in by an algorithm. A symmetric square root gives an order-independent assignment of shocks to assets and the same $\Sigma$. Neither is more correct; only one of them is affected by the order in which the assets happened to be listed.

Paths of a diffusion should be drawn from the exact transition law wherever one exists, and one exists more often than practitioners assume. Geometric Brownian motion has an exact discretization, so a simulated price path can be generated at any step size with no discretization error at all, the closed form being the exponential of [Geometric Brownian Motion](../part-08-stochastic-processes/09-geometric-brownian-motion.md). An Ornstein–Uhlenbeck process has one too. Writing the Euler–Maruyama scheme instead — which is what transcribing the stochastic differential equation literally produces — introduces an error whose size depends on the step relative to the process's own timescale. [Stochastic Calculus](../../advanced/03-stochastic-calculus.md) measures exactly this on the course's SPY–IVV spread and finds that "a 3.4-day half-life sampled once a day gives $\theta\Delta = 0.202$, nowhere near the small-step regime the scheme assumes", inflating the equilibrium standard deviation by $5.5\%$. Its conclusion is the working rule: discretization error is a choice, not a fact of life.

The third construction has no analogue for points, and it is where the rest of Part IX begins. When a path's law is unknown but one realized path is in hand, the sampler becomes a *resampler*: the empirical distribution replaces $F$, and the transformation stops being a formula and becomes a draw with replacement from the data. That is [Bootstrap Methods](07-bootstrap-methods.md), it is exactly inverse-transform sampling applied to the empirical distribution function — which is why this page's second section matters at all — and the ordering that Cholesky and exact discretization worked so hard to preserve is the first thing a naive resample destroys.

## Every Law Is a Uniform in Disguise

The through-line here is that one uniform stream and one theorem generate every distribution a simulation needs, and that all the content lives in the four places where the theorem is inconvenient. It is inconvenient for discrete laws, because the inverse is a search whose cost is the dominant term in a bootstrap. It is inconvenient for the normal, because $\Phi^{-1}$ is not elementary, so the standard routes are trigonometric identities and rejection schemes rather than inversions and each carries a hard ceiling from its input's precision. It is inconvenient for laws built from other laws, where composition and scale mixture supply constructions reaching dimensions the inverse cannot. And it is silent about paths, where the joint law is the object and the marginals are a distraction.

Two results on this page are the sort that survive being read once. The scale mixture and the inverse CDF are the same distribution, agreeing to a two-sample $p$-value of $0.5002$, so a construction that looks like a modelling choice is not one. And a normal calibrated to that law's $99\%$ quantile understates its $99.99\%$ quantile by a factor of nearly four, so a construction that looks like an implementation detail is one of the largest modelling decisions in a risk report.

The awkward observation is what both of those had in common. Every method on this page is exact, and the correctness of each was established by drawing four million samples and reading quantiles off them — an estimate, with an error bar that nobody quoted, checked against a closed form that happened to exist. Everything downstream of here has no closed form to check against: the tail probabilities, the option prices, the resampled Sharpe intervals are all estimates of exactly that kind and are the only description of their target available. What such an estimate converges to, how fast, and what its reported precision is actually a statement about, is [Monte Carlo Simulation](03-monte-carlo-simulation.md).
