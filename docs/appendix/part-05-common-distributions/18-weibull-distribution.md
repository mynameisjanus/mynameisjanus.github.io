# Weibull Distribution

This is the only family in this part whose hazard has a shape. Everything else that models a duration here assumes the hazard is flat — that nothing ages, that a spell which has run for a year is no more and no less likely to end than one that began yesterday — and [Exponential Distribution](10-exponential-distribution.md) showed that assumption failing on every duration a market produces. The Weibull is the smallest repair: one extra parameter, and the hazard is allowed to rise or fall with age.

This page covers the density and the survival function, the shape parameter read as a hazard slope, the moments through the gamma function, the special cases the family contains, the min-stability that connects it to extreme values, and the question the shape parameter actually answers about a trading edge. It does not cover the constant-hazard case in its own right, which is [Exponential Distribution](10-exponential-distribution.md); it does not build the gamma function, which is [Gamma Distribution](13-gamma-distribution.md); and it does not develop tail estimation, which is [Extreme Value Theory](../part-18-quant-finance-applications/13-extreme-value-theory.md).

The trading stake is a question no constant-hazard model can even pose. Does an edge decay? A strategy that has worked for three years — is it more likely to stop working next month than one that started last month, or less, or neither? The exponential answers "neither" by construction and cannot be asked. The Weibull turns the question into a single estimated number, and the last section shows what each answer would look like and how much data it takes to tell them apart.

## The Density and the Survival Function

$X$ is Weibull with shape $k>0$ and scale $\lambda>0$ when

$$\mathbf{P}(X>x)=\exp\Big[-\Big(\frac{x}{\lambda}\Big)^{k}\Big],\qquad f_X(x)=\frac{k}{\lambda}\Big(\frac{x}{\lambda}\Big)^{k-1}\exp\Big[-\Big(\frac{x}{\lambda}\Big)^{k}\Big],\qquad x\ge0.$$

The survival function is the definition worth carrying: it is the exponential's $e^{-x/\lambda}$ with $x$ replaced by $x^{k}$. So the family is a power-transformed exponential, and every property follows from that one substitution.

$$X\sim\mathrm{Weibull}(k,\lambda)\iff \Big(\frac{X}{\lambda}\Big)^{k}\sim\mathrm{Exp}(1).$$

## The Shape Parameter Is a Hazard Slope

Dividing the density by the survival function gives the hazard, and the exponential in both cancels completely:

$$h(x)=\frac{k}{\lambda}\Big(\frac{x}{\lambda}\Big)^{k-1}.$$

This is a power of $x$ with exponent $k-1$, so the sign of $k-1$ decides everything.

| Shape | Hazard | Reading | Example |
|---|---|---|---|
| $k<1$ | falling | entrenchment; survivors are the durable ones | a drawdown that keeps deepening |
| $k=1$ | flat | memoryless; the exponential exactly | a null with no ageing |
| $k>1$ | rising | ageing; the longer it has run, the sooner it ends | a mean-reverting spread |

??? note "Proof that the hazard is monotone and that its direction is fixed by k alone"
    Differentiating $h(x)=(k/\lambda)(x/\lambda)^{k-1}$ gives $h'(x)=\frac{k(k-1)}{\lambda^{2}}(x/\lambda)^{k-2}$. The factor $(x/\lambda)^{k-2}$ is positive for $x>0$ and $k/\lambda^{2}$ is positive, so the sign of $h'$ is the sign of $k-1$ — everywhere on the support, with no crossing point.

    Two consequences follow. First, the Weibull hazard is monotone: it cannot rise and then fall, so a genuine bathtub shape needs a mixture or a different family. Second, the scale $\lambda$ does not appear in the sign, so it cannot be traded off against the shape — rescaling time changes how fast the hazard moves and never whether it rises or falls. That separation is what makes $k$ estimable and interpretable independently of the units durations were measured in.

    The load-bearing structure is the power substitution. Because $(X/\lambda)^{k}$ is a unit exponential, the Weibull inherits the exponential's constant hazard *in the transformed clock* $t\mapsto t^{k}$, and the whole content of the family is that a clock running at a changing rate produces an ageing or an entrenching duration in real time. That is a more useful mental picture than the density, and it is why the family turns up wherever a process has an internal timescale of its own.

```python
import numpy as np
from scipy.special import gamma as G

rng = np.random.default_rng(131)
lam, n = 50.0, 2_000_000

def hazard(x, age, w=5.0):                                     # ends per unit, among survivors
    alive = x[x > age]
    return (alive <= age + w).mean() / w if alive.size > 2000 else np.nan

print("  the hazard slope is the sign of k - 1, and nothing else")
print("      k      mean      cv      h(q20)     h(q50)     h(q80)     direction")
for k in (0.6, 1.0, 1.5, 3.0):
    x = lam * rng.weibull(k, n)
    q = np.quantile(x, [0.20, 0.50, 0.80])
    hs = [hazard(x, a) for a in q]
    ratio = hs[2] / hs[0]
    d = "flat" if abs(ratio - 1) < 0.02 else ("falling" if ratio < 1 else "rising")
    print(f"    {k:4.1f} {x.mean():9.2f} {x.std() / x.mean():7.3f}"
          f" {hs[0]:10.5f} {hs[1]:10.5f} {hs[2]:10.5f}   {d:>10}")
print("  coefficient of variation is a function of k alone, so it identifies the shape")
for k in (0.6, 1.0, 1.5, 3.0):
    print(f"    k = {k:4.1f}   cv = sqrt(G(1+2/k)/G(1+1/k)^2 - 1) ="
          f" {np.sqrt(G(1 + 2 / k) / G(1 + 1 / k) ** 2 - 1):.4f}")
# =>   the hazard slope is the sign of k - 1, and nothing else
#          k      mean      cv      h(q20)     h(q50)     h(q80)     direction
#         0.6     75.17   1.758    0.02561    0.01432    0.00847      falling
#         1.0     50.02   1.001    0.01903    0.01902    0.01897         flat
#         1.5     45.14   0.679    0.01846    0.02561    0.03260       rising
#         3.0     44.65   0.363    0.02427    0.04618    0.07200       rising
#      coefficient of variation is a function of k alone, so it identifies the shape
#        k =  0.6   cv = sqrt(G(1+2/k)/G(1+1/k)^2 - 1) = 1.7581
#        k =  1.0   cv = sqrt(G(1+2/k)/G(1+1/k)^2 - 1) = 1.0000
#        k =  1.5   cv = sqrt(G(1+2/k)/G(1+1/k)^2 - 1) = 0.6790
#        k =  3.0   cv = sqrt(G(1+2/k)/G(1+1/k)^2 - 1) = 0.3634
```

The three hazard columns do the work. At $k=1$ they agree to three significant figures — the exponential, recovered exactly, and the only row where the elapsed age is uninformative. At $k=0.6$ the hazard falls threefold between the twentieth and eightieth percentiles of age, and at $k=3$ it rises threefold over the same span. The direction column never disagrees with the sign of $k-1$, which is the proof printed.

The second block of output is the practical estimator. The coefficient of variation depends only on $k$, not on $\lambda$, so it identifies the shape from data with no fitting at all: a duration series with $\mathrm{cv}>1$ has $k<1$ and a decreasing hazard, and one with $\mathrm{cv}<1$ has $k>1$. That single division is why [Exponential Distribution](10-exponential-distribution.md) could reject the memoryless model on a coefficient of variation of $1.865$ before doing anything else — and it now also says which way to go: $\mathrm{cv}=1.865$ implies $k\approx0.55$, a strongly decreasing hazard.

## Moments Through the Gamma Function

$$\mathbb{E}[X^{m}]=\lambda^{m}\,\Gamma\Big(1+\frac{m}{k}\Big),$$

so in particular

$$\mathbb{E}[X]=\lambda\,\Gamma\Big(1+\frac{1}{k}\Big),\qquad \mathrm{var}(X)=\lambda^{2}\Big[\Gamma\Big(1+\frac{2}{k}\Big)-\Gamma\Big(1+\frac{1}{k}\Big)^{2}\Big].$$

??? note "Proof of the moments, and that every one of them exists"
    Substitute $u=(x/\lambda)^{k}$, so that $x=\lambda u^{1/k}$ and the density transforms to the unit exponential's $e^{-u}$ — the power substitution again. Then

    $$\mathbb{E}[X^{m}]=\int_{0}^{\infty}x^{m}f_X(x)\,\mathrm{d}x=\lambda^{m}\int_{0}^{\infty}u^{m/k}e^{-u}\,\mathrm{d}u=\lambda^{m}\,\Gamma\Big(1+\frac{m}{k}\Big),$$

    recognising the gamma integral of [Gamma Distribution](13-gamma-distribution.md) at argument $1+m/k$.

    Every moment exists, for every $k>0$ and every $m$, because $\Gamma$ is finite at every positive argument. This is a genuine difference from the heavy-tailed families this part ends with: a Weibull with $k<1$ has a hazard that decays toward zero and looks superficially like a power law, but its survival function still decays as a stretched exponential, which beats every polynomial. So a decreasing hazard is not the same thing as a heavy tail, and confusing them is the standard error in duration modelling — [Student's t Distribution](16-students-t-distribution.md) has moments only below $\nu$, while this family has all of them however slowly the hazard falls.

## Special Cases and Min-Stability

| Case | Parameters | Identity |
|---|---|---|
| Exponential | $k=1$ | $\mathrm{Exp}(1/\lambda)$ exactly |
| Rayleigh | $k=2$ | the length of a two-dimensional Gaussian vector |
| Approximately normal | $k\approx3.6$ | skewness passes through zero |

The family is also closed under minima, which is the property that gives it its role in reliability and its connection to extreme values.

??? note "Proof of min-stability, and why it makes the Weibull an extreme-value law"
    Let $X_1,\ldots,X_n$ be independent $\mathrm{Weibull}(k,\lambda)$. The minimum exceeds $x$ exactly when all of them do, so

    $$\mathbf{P}(\min_i X_i>x)=\prod_{i=1}^{n}\exp\Big[-\Big(\frac{x}{\lambda}\Big)^{k}\Big]=\exp\Big[-n\Big(\frac{x}{\lambda}\Big)^{k}\Big]=\exp\Big[-\Big(\frac{x}{\lambda n^{-1/k}}\Big)^{k}\Big],$$

    which is $\mathrm{Weibull}(k,\lambda n^{-1/k})$ — the same shape with a smaller scale. The shape is invariant and only the scale contracts, at rate $n^{-1/k}$.

    This is the exact analogue of the exponential's closure under minima on [Exponential Distribution](10-exponential-distribution.md), which is the case $k=1$, and it holds for the same reason: the survival function is an exponential of a power, so multiplying $n$ of them adds the exponents.

    Min-stability is why the Weibull is one of the three extreme-value limit laws. A system that fails when its weakest component fails has a failure time that is a minimum over many components, and iterating the argument above shows the shape converges to a Weibull whatever the individual components looked like — provided their support is bounded below, which a duration's is. That is the weakest-link reading, and the bounded-support proviso is what distinguishes this case from the Fréchet limit that governs heavy-tailed maxima in [Extreme Value Theory](../part-18-quant-finance-applications/13-extreme-value-theory.md).

## Does an Edge Age?

Now the trading question. Suppose a strategy's working life is a duration and we want to know whether it ages. The three hypotheses are $k>1$ (edges wear out, so an old strategy is closer to death), $k<1$ (edges entrench, so a strategy that has survived is more durable), and $k=1$ (nothing can be said from age alone). The question is how much data distinguishes them.

```python
import numpy as np
from scipy.stats import weibull_min

rng = np.random.default_rng(137)
print("  distinguishing a hazard slope from a flat one, by number of completed spells")
print("      true k     n=20      n=50     n=200     n=1000    reading")
for k in (0.6, 0.8, 1.0, 1.3, 2.0):
    row = []
    for n in (20, 50, 200, 1000):
        hits = 0
        for _ in range(400):
            x = weibull_min.rvs(k, scale=50.0, size=n, random_state=rng)
            khat, _, _ = weibull_min.fit(x, floc=0)
            se = 0.7797 * khat / np.sqrt(n)                    # asymptotic se of the MLE shape
            hits += abs(khat - 1.0) > 1.96 * se                # reject k = 1
        row.append(hits / 400)
    tag = "ageing" if k > 1 else ("entrenching" if k < 1 else "flat (null true)")
    print(f"    {k:8.1f}" + "".join(f"{p:10.3f}" for p in row) + f"   {tag}")
# =>   distinguishing a hazard slope from a flat one, by number of completed spells
#          true k     n=20      n=50     n=200     n=1000    reading
#             0.6     0.787     0.995     1.000     1.000   entrenching
#             0.8     0.233     0.517     0.975     1.000   entrenching
#             1.0     0.070     0.045     0.035     0.052   flat (null true)
#             1.3     0.312     0.667     0.995     1.000   ageing
#             2.0     0.970     1.000     1.000     1.000   ageing
```

The $k=1$ row is the control, rejecting at $0.035$ to $0.070$ against a nominal $0.05$, so what follows is power rather than artefact.

The reading depends entirely on how large a slope is worth caring about. A doubling of the hazard exponent, $k=2$, is caught almost every time on twenty spells — that case is easy. The interesting cases are not. A thirty-percent slope in either direction is detected roughly a quarter to a third of the time on twenty spells, two-thirds on fifty, and only becomes reliable at two hundred. Twenty completed spells is already more strategies than most desks retire in a decade, so for the strategy-lifetime question the realistic sample sizes sit in the leftmost column, where the test is close to useless for anything but the most dramatic effect.

!!! warning "The question of whether an edge decays is answerable in principle and unanswerable with the data any desk has"
    A firm that has run fifty strategies to completion over its history has, by the table above, a two-in-three chance of noticing a thirty-percent hazard slope and rather less than that of noticing a milder one. This is not an argument for assuming $k=1$; it is an argument against believing any answer, including the one a fitted model returns with a confident-looking standard error. The honest report is the interval on $k$, which at these sample sizes will comfortably contain $1$ along with values in both directions.

    What the framework does buy, even without data to resolve it, is a precise statement of the question and of what would settle it. "Do edges decay" is untestable as usually posed; "is $k$ different from one, and by how much" is testable, needs a specific number of completed spells, and makes the sample-size requirement visible before the study rather than after. The same discipline applied to drawdowns is more tractable, since a long history contains many spells — and there the answer is already known from [Exponential Distribution](10-exponential-distribution.md)'s coefficient of variation of $1.865$, which puts $k$ near $0.55$ and says drawdowns entrench rather than age.

So the Weibull's contribution to this part is to make a question askable. Every other duration family here either assumes the hazard away or is a mixture that produces a falling hazard as a side effect; this one puts the slope in a parameter, gives it a sign with a clear reading, and lets the data speak — or, more often, demonstrate that it cannot.

The practical rule is the one the coefficient of variation makes cheap. Before fitting anything, divide the sample standard deviation of a duration series by its mean: above one means an entrenching hazard, below one means an ageing one, and exactly one is the memoryless null. That single number costs nothing, requires no optimiser, and answers the qualitative question that the parametric fit will then answer quantitatively — and if the two disagree, the fit is wrong, because the coefficient of variation is a function of the shape alone.
