# Hidden Markov Models

Every result on the previous pages assumed you could see which state the chain was in. Nobody publishes today's market regime. What is observable is returns, which are noisy emissions from whatever state the market occupies, and the entire practical content of regime modelling is the inference problem that gap creates. A hidden Markov model is the smallest object that states it precisely, and the algorithms it admits are exact, linear in the length of the history, and — the part that gets lost — perfectly capable of returning a confident answer when there is no hidden state at all.

This page covers the two coupled processes and the single factorization every algorithm manipulates, the forward recursion that replaces a sum over exponentially many paths with a linear one and the scaling that makes it survive in floating point, filtering as sequential Bayes against smoothing as retrospection and the lookahead the difference conceals, the Baum–Welch algorithm with its monotonicity guarantee and the three things that guarantee does not cover, and what the whole apparatus returns when it is pointed at noise. It does not fit anything to real returns, which is [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md); it does not develop the chain underneath, which is [Markov Chains](05-markov-chains.md); it does not derive EM in general, which is [The EM Algorithm](../part-17-statistical-computing/03-em-algorithm.md); it does not build the Bayesian machinery filtering instantiates, which is [Part XVI](../part-16-bayesian-statistics/index.md); and it does not let the state be continuous, which is [Particle and Kalman Filters](../../advanced/02-particle-and-kalman-filters.md).

The trading stake is the fit the course publishes and the caution it attaches. [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md) fits a two-state Gaussian HMM to SPY and gets a calm state earning $+21.9\%$ at $11.6\%$ volatility and a stress state losing $-29.0\%$ at $32.0\%$, then finds that BIC "improves at every $k$, and would keep improving past four", concluding that "the statistical criterion ranks fit; it does not make the modeling decision" and that "a state you cannot name is a state you cannot trade." The fifth section fits the same model to data with no regimes in it at all and gets two persistent states with double-digit durations, which is the sharpest available form of that warning.

## Two Coupled Processes and One Factorization

A **hidden Markov model** consists of a hidden state chain $X_0,\ldots,X_T$ — an ordinary Markov chain on $S=\{1,\ldots,m\}$ with transition matrix $P$ and initial distribution $\pi_0$, except that it is never observed — together with an observation process $Y_0,\ldots,Y_T$ in which the state at time $n$ emits $Y_n$ from a state-specific **emission distribution** with density $f_i$.

The coupling assumption is that, conditioned on the entire hidden path, the observations are independent and each depends only on the concurrent state:

$$p(y_0,\ldots,y_T\mid x_0,\ldots,x_T)=\prod_{n=0}^{T}f_{x_n}(y_n).$$

```mermaid
flowchart LR
    X0(("X&#8320;")) --> X1(("X&#8321;")) --> X2(("X&#8322;")) --> Xd["&#8943;"]
    X0 --> Y0(["Y&#8320;"])
    X1 --> Y1(["Y&#8321;"])
    X2 --> Y2(["Y&#8322;"])
```

The hidden chain evolves on its own; each observation hangs off its state. All statistical dependence between observations at different times flows *through* the chain, which is what makes everything below tractable and is exactly the structure [Independence](../part-02-probability-foundations/05-independence.md) identifies when it notes that returns are "independent given the hidden state, dependent once the state is marginalized away" — and therefore that return *magnitudes* look strongly autocorrelated while *signs* look patternless.

Combining the chain's trajectory formula with the emission factorization gives the complete joint density:

$$p(x_{0:T},y_{0:T})=\pi_0(x_0)\,f_{x_0}(y_0)\prod_{n=1}^{T}p_{x_{n-1}x_n}\,f_{x_n}(y_n).$$

Every computation in this subject is one manipulation of that expression: summing it over paths gives the likelihood, maximizing it over paths gives the best trajectory, and maximizing its expectation over parameters gives the fit. The parameters are collected as $\lambda=(\pi_0,P,\{f_i\})$, and for returns the standard choice is Gaussian emissions, $f_i=\mathcal{N}(\mu_i,\sigma_i^{2})$, which already captures the essential phenomenology: a persistent low-volatility state and a persistent high-volatility state, each with its own mean.

## The Forward Recursion Replaces an Exponential Sum With a Linear One

To evaluate $p(y_{0:T})$ you must sum the joint density over every hidden path, and there are $m^{T+1}$ of them. For three states and one year of daily data that is $3^{252}\approx10^{120}$ terms. The **forward algorithm** computes the same number exactly in $O(m^{2}T)$ operations by pushing the sums inside the product. Define

$$\alpha_n(i)=p(y_{0:n},X_n=i),$$

the joint density of the observations so far *and* the event that the chain is currently in state $i$.

??? note "Proof that the forward variable satisfies a one-step recursion, and that summing it terminates the calculation"
    Two conditional-independence facts follow from the joint density by summing out the unwanted variables: given $X_n=i$, the next state is independent of the observation history, $\mathbf{P}(X_{n+1}=j\mid X_n=i,y_{0:n})=p_{ij}$; and given $X_{n+1}=j$, the new observation is independent of everything earlier, $p(y_{n+1}\mid X_{n+1}=j,X_n=i,y_{0:n})=f_j(y_{n+1})$. Decomposing over the state at time $n$,

    $$\begin{align}
    \alpha_{n+1}(j)&=p(y_{0:n+1},X_{n+1}=j)\\
    &=\sum_{i=1}^{m}p(y_{0:n},X_n=i)\;\mathbf{P}(X_{n+1}=j\mid X_n=i,y_{0:n})\;p(y_{n+1}\mid X_{n+1}=j,X_n=i,y_{0:n})\\
    &=\Bigl[\sum_{i=1}^{m}\alpha_n(i)\,p_{ij}\Bigr]f_j(y_{n+1}),
    \end{align}$$

    with initialization $\alpha_0(i)=\pi_0(i)f_i(y_0)$. Termination is the total probability theorem: $p(y_{0:T})=\sum_i\alpha_T(i)$.

    In practice the recursion is run in scaled form, because $\alpha_n$ is a joint density over $n+1$ observations and underflows within a few hundred steps. Setting $c_n=\sum_j\tilde\alpha_n(j)$ and $\hat\alpha_n=\tilde\alpha_n/c_n$ at each step makes the scaled variables exactly the filtered probabilities, makes the scale factors the one-step predictive densities $c_n=p(y_n\mid y_{0:n-1})$, and recovers the likelihood as $\log p(y_{0:T})=\sum_n\log c_n$ — a sum of logs, which is the manoeuvre [Exponentials, Logarithms, and Growth](../part-01-mathematical-foundations/07-exponentials-logarithms-growth.md) identifies as the reason "every estimation routine in the book … works with sums of logs."

    The load-bearing hypothesis is that $\alpha_n$ is a **sufficient summary** of the entire history for everything the future can ask, which is the Markov property applied to the pair. That is why the computation never looks back, and it is the whole reason an exponential sum collapses to a linear one.

Dividing the forward variable by its own sum answers the question a desk actually asks in real time. The **filtered** state probability is

$$\mathbf{P}(X_n=i\mid y_{0:n})=\frac{\alpha_n(i)}{\sum_j\alpha_n(j)},$$

and unwinding one step shows it is exactly sequential [Bayesian inference](../part-16-bayesian-statistics/01-bayesian-framework.md): push yesterday's posterior through $P$ to get today's prior, multiply by the likelihood of today's observation under each state, normalize. That predict–update cycle costs $O(m^{2})$ per observation and is directly usable in a live system. The block below checks the recursion against the definition it is supposed to compute.

```python
import numpy as np
from itertools import product

rng = np.random.default_rng(8071)
P = np.array([[0.95, 0.05], [0.10, 0.90]])
p0 = np.array([0.6, 0.4])
mu, sd = np.array([0.0005, -0.0010]), np.array([0.006, 0.020])
print("  the forward recursion against a brute-force sum over every hidden path")
print("        T    paths    brute-force log p(y)    forward log p(y)    difference")
for T in (4, 8, 12, 16):
    y = rng.normal(0.0, 0.01, T)
    B = np.exp(-0.5 * ((y[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    tot = 0.0
    for path in product((0, 1), repeat=T):                     # every hidden trajectory
        p = p0[path[0]] * B[0, path[0]]
        for n in range(1, T):
            p *= P[path[n - 1], path[n]] * B[n, path[n]]
        tot += p
    a = p0 * B[0]
    c = [a.sum()]
    a /= c[0]
    for n in range(1, T):                                      # scaled forward pass
        a = (a @ P) * B[n]
        c.append(a.sum())
        a /= c[-1]
    fwd = np.sum(np.log(c))
    print(f"  {T:9d} {2 ** T:8d} {np.log(tot):23.6f} {fwd:19.6f} {np.log(tot) - fwd:13.2e}")
# =>   the forward recursion against a brute-force sum over every hidden path
#            T    paths    brute-force log p(y)    forward log p(y)    difference
#              4       16               10.212305           10.212305      1.78e-15
#              8      256               24.411977           24.411977     -3.55e-15
#             12     4096               37.370282           37.370282     -7.11e-15
#             16    65536               43.416004           43.416004     -1.42e-14
```

The two middle columns agree to fourteen decimal places at every length, and the difference column — $1.78\times10^{-15}$, $-3.55\times10^{-15}$, $-7.11\times10^{-15}$, $-1.42\times10^{-14}$ — is floating-point rounding and nothing else. **The forward algorithm is not an approximation to the sum over paths; it is the sum over paths, rearranged.**

The path count is the reason anyone cares. At $T=16$ the brute-force column enumerated $65{,}536$ trajectories to produce a number the recursion obtained in sixteen steps of a $2\times2$ multiply. At $T=252$ the enumeration would require $10^{76}$ terms and the recursion still costs $252$ steps. Nothing is lost in the trade, which is unusual enough to be worth stating plainly.

## Smoothing Uses the Future, Which Is Exactly the Problem

Filtering conditions on data available at the time. The retrospective question — what was the state at time $n$, given *everything*, including what happened afterwards — is answered by adding a backward pass. Define $\beta_n(i)=p(y_{n+1:T}\mid X_n=i)$, computed by the mirror recursion $\beta_n(i)=\sum_j p_{ij}f_j(y_{n+1})\beta_{n+1}(j)$ from $\beta_T\equiv1$, and the **smoothed** probability is

$$\gamma_n(i)=\mathbf{P}(X_n=i\mid y_{0:T})=\frac{\alpha_n(i)\,\beta_n(i)}{\sum_j\alpha_n(j)\,\beta_n(j)}.$$

Intuitively $\alpha_n(i)$ scores state $i$ by how well it explains the past and $\beta_n(i)$ by how well it sets up the future, and smoothing multiplies the two. A third question — the single most likely *path*, rather than the most likely state at each time separately — is answered by the **Viterbi** algorithm, which is the forward recursion with $\max$ in place of $\sum$ and a back-pointer at each step, run in logarithms as $\log\delta_{n+1}(j)=\max_i[\log\delta_n(i)+\log p_{ij}]+\log f_j(y_{n+1})$. It is not the same as taking $\arg\max_i\gamma_n(i)$ at each $n$: the pointwise-best labels ignore transition probabilities and can describe a path the chain is forbidden to take.

Smoothed probabilities are sharper than filtered ones, and the sharpening is entirely the future's contribution. That makes them the right tool for historical attribution and a catastrophe inside a backtest.

```python
import numpy as np

rng = np.random.default_rng(8073)
P = np.array([[0.989, 0.011], [0.028, 0.972]])
mu = np.array([0.219, -0.290]) / 252
sd = np.array([0.116, 0.320]) / np.sqrt(252)                   # the course's fitted regimes
T, reps = 6_300, 40


def filter_smooth(y):
    B = np.exp(-0.5 * ((y[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    a, b, c = np.zeros((T, 2)), np.ones((T, 2)), np.zeros(T)
    a[0] = np.array([0.5, 0.5]) * B[0]
    c[0] = a[0].sum()
    a[0] /= c[0]
    for t in range(1, T):                                      # scaled forward pass
        a[t] = (a[t - 1] @ P) * B[t]
        c[t] = a[t].sum()
        a[t] /= c[t]
    for t in range(T - 2, -1, -1):                             # backward pass, same scales
        b[t] = (P @ (B[t + 1] * b[t + 1])) / c[t + 1]
    g = a * b
    return a, g / g.sum(1, keepdims=True)


def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std(ddof=1)


out = np.zeros((reps, 4))
for k in range(reps):
    s = np.zeros(T, dtype=np.int64)
    u = rng.random(T)
    for t in range(1, T):
        s[t] = 1 - s[t - 1] if u[t] < P[s[t - 1], 1 - s[t - 1]] else s[t - 1]
    y = mu[s] + sd[s] * rng.standard_normal(T)
    f, g = filter_smooth(y)
    nxt = y[1:]
    out[k] = [sharpe(nxt), sharpe(nxt * (f[:-1, 0] > 0.5)),
              sharpe(nxt * (g[:-1, 0] > 0.5)), sharpe(nxt * (s[:-1] == 0))]
print(f"  a long-only calm filter on {reps} simulated 25-year histories, annualized Sharpe")
print("       buy and hold    filtered (legal)    smoothed (lookahead)    the true state")
print(f"  {out[:, 0].mean():17.4f} {out[:, 1].mean():19.4f} {out[:, 2].mean():23.4f}"
      f" {out[:, 3].mean():17.4f}")
# =>   a long-only calm filter on 40 simulated 25-year histories, annualized Sharpe
#           buy and hold    filtered (legal)    smoothed (lookahead)    the true state
#                 0.4273              1.2257                  1.5242            1.5075
```

The filter works. Holding the index unconditionally returns a Sharpe of $0.4273$; sitting out whenever the filtered probability of calm falls below one half returns $1.2257$, and every input to that decision existed at the time it was made. This is what a regime model is for, measured on data generated by exactly the model being fitted, which is the most favourable possible setting.

The smoothed column is the trap and it is worth more than the model. Trading the same rule on $\gamma_n$ returns $1.5242$ — a $24\%$ improvement over the legal version, obtained entirely by consulting returns that had not happened. **The lookahead is not a rounding error; it is a quarter of the strategy.**

The last column is the one that should end the argument. Trading on the *true* hidden state — perfect clairvoyance about the regime, unavailable to anyone ever — returns $1.5075$, which is *lower* than the smoothed backtest's $1.5242$. Smoothing does not merely peek at the future state; it peeks at the future *returns*, since $\beta_n$ is built from them, so a rule driven by $\gamma$ can beat genuine omniscience about the regime. Any backtest reporting a number above the oracle row has, definitionally, used information no oracle had.

!!! warning "Filtered and smoothed probabilities differ by one line of code and by the entire validity of the backtest"
    Both quantities come out of the same forward–backward pass, both are called "the probability of being in state $i$", and a library that returns `predict_proba` will hand you one of them without saying which. The rule is that a backtest may consume $\alpha$ and may never consume $\gamma$ or a Viterbi path, because both condition on $y_{n+1:T}$. Smoothed labels are correct and useful for the questions that are genuinely retrospective — how did this strategy perform inside each regime, how often did regimes change, what did the last crisis look like — and they are also what the M-step below requires, which is fine because parameter estimation is not a trading decision. The formal content of the distinction is the $\mathcal{F}_t$-measurability that [Probability Spaces](../part-02-probability-foundations/01-probability-spaces.md) describes as the reason "a backtest that computes a signal from tomorrow's close" is meaningless rather than merely optimistic.

## Baum–Welch Climbs the Likelihood and Promises Nothing Else

Everything so far assumed $\lambda$ known. In practice nothing is known, and maximum likelihood asks for $\arg\max_\lambda\log\sum_{x_{0:T}}p(x_{0:T},y_{0:T};\lambda)$ — a log of a sum over exponentially many paths, with no closed-form maximizer. **Baum–Welch** is [expectation–maximization](../part-17-statistical-computing/03-em-algorithm.md) specialized to this structure, alternating between inferring the states given current parameters and re-estimating parameters given that soft inference.

The E-step needs only two families of posterior quantities, because the complete-data log-likelihood is linear in the indicators: the smoothed state probabilities $\gamma_n(i)$ from the pass above, and the smoothed transition probabilities $\xi_n(i,j)=\alpha_n(i)p_{ij}f_j(y_{n+1})\beta_{n+1}(j)/p(y_{0:T})$. The M-step then decouples into three closed forms, each a soft-count version of the obvious estimator:

$$\hat p_{ij}=\frac{\sum_n\xi_n(i,j)}{\sum_n\gamma_n(i)},\qquad
\hat\mu_i=\frac{\sum_n\gamma_n(i)\,y_n}{\sum_n\gamma_n(i)},\qquad
\hat\sigma_i^{2}=\frac{\sum_n\gamma_n(i)\,(y_n-\hat\mu_i)^{2}}{\sum_n\gamma_n(i)}.$$

Read them as "expected transitions from $i$ to $j$ over expected visits to $i$", and as a sample mean and variance weighted by the posterior probability that state $i$ generated each observation. If the path were observed, the $\gamma$'s and $\xi$'s would be indicators and these would be ordinary frequencies and moments.

??? note "Proof that each iteration cannot decrease the likelihood, and that this is all it establishes"
    Write $\lambda'$ for the current parameters and $q(x)=p(x_{0:T}\mid y_{0:T};\lambda')$ for the posterior over paths. Since $p(x,y;\lambda)=p(y;\lambda)p(x\mid y;\lambda)$, taking $\mathbb{E}_q$ of the log gives

    $$\log p(y;\lambda)=\underbrace{\mathbb{E}_q[\log p(x,y;\lambda)]}_{Q(\lambda,\lambda')}-\underbrace{\mathbb{E}_q[\log p(x\mid y;\lambda)]}_{H(\lambda,\lambda')},$$

    the left side surviving the expectation untouched because it does not depend on $x$. Subtracting the same identity at $\lambda=\lambda'$,

    $$\log p(y;\lambda)-\log p(y;\lambda')=\bigl[Q(\lambda,\lambda')-Q(\lambda',\lambda')\bigr]-\bigl[H(\lambda,\lambda')-H(\lambda',\lambda')\bigr],$$

    and the second bracket is non-positive by Jensen's inequality, since $H(\lambda,\lambda')-H(\lambda',\lambda')=\mathbb{E}_q[\log(p(x\mid y;\lambda)/q(x))]\leq\log\mathbb{E}_q[p(x\mid y;\lambda)/q(x)]=\log1=0$. So any $\lambda$ improving $Q$ — in particular the M-step maximizer — cannot decrease the likelihood.

    The load-bearing hypothesis is only that the M-step does not *decrease* $Q$, which is why the guarantee is so weak. It says the sequence of likelihoods is non-decreasing. It does not say the limit is the global maximum, does not say the limit is a good fit, does not say the number of states is right, and does not say the states mean anything. **Monotone ascent on a multimodal surface is a statement about the algorithm, not about the answer**, and the next section is what that permits.

## Fit Two States to Noise and You Get Two States

The likelihood surface is multimodal, the algorithm converges to a local maximum determined by the initialization, and — the failure that matters most — the model has no way to decline. Ask for two states and two states is what you receive, whether or not the data contains any.

```python
import numpy as np

rng = np.random.default_rng(8077)
T, iters, starts = 2_520, 120, 3


def baum_welch(y, seed):
    r = np.random.default_rng(seed)
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = y.mean() + y.std() * r.normal(0, 0.5, 2)
    sd = y.std() * np.exp(r.normal(0, 0.4, 2))
    ll = -np.inf
    for _ in range(iters):
        B = np.exp(-0.5 * ((y[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + 1e-300
        a, b, c = np.zeros((T, 2)), np.ones((T, 2)), np.zeros(T)
        a[0] = 0.5 * B[0]
        c[0] = a[0].sum()
        a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t - 1] @ P) * B[t]
            c[t] = a[t].sum()
            a[t] /= c[t]
        for t in range(T - 2, -1, -1):
            b[t] = (P @ (B[t + 1] * b[t + 1])) / c[t + 1]
        g = a * b
        g /= g.sum(1, keepdims=True)
        xi = a[:-1, :, None] * P * (B[1:] * b[1:])[:, None, :] / c[1:, None, None]
        P = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        w = g.sum(0)
        mu = (g * y[:, None]).sum(0) / w
        sd = np.sqrt((g * (y[:, None] - mu) ** 2).sum(0) / w).clip(1e-6)
        ll = np.log(c).sum()
    o = np.argsort(sd)                                         # label states by volatility
    return ll, P[np.ix_(o, o)], sd[o]


noise = 0.011 * rng.standard_normal(T)
s = np.zeros(T, dtype=np.int64)
u = rng.random(T)
Q = np.array([[0.989, 0.011], [0.028, 0.972]])
for t in range(1, T):
    s[t] = 1 - s[t - 1] if u[t] < Q[s[t - 1], 1 - s[t - 1]] else s[t - 1]
regime = np.array([0.0073, 0.0202])[s] * rng.standard_normal(T)
print(f"  Baum-Welch on {T} days, {starts} random starts, best likelihood kept")
print("         data    p11    p22    duration 1    duration 2    vol 1    vol 2"
      "    log-lik gain    BIC needs")
need = 5 * np.log(T) / 2                                       # five extra parameters
for name, y in (("pure noise", noise), ("two regimes", regime)):
    runs = [baum_welch(y, 900 + j) for j in range(starts)]
    ll, P, sd = max(runs, key=lambda z: z[0])
    one = -0.5 * T * (np.log(2 * np.pi * y.var()) + 1)         # the one-state Gaussian fit
    print(f"  {name:>11s} {P[0, 0]:6.4f} {P[1, 1]:6.4f} {1 / (1 - P[0, 0]):13.1f}"
          f" {1 / (1 - P[1, 1]):13.1f} {sd[0]:8.4f} {sd[1]:8.4f}"
          f" {ll - one:15.1f} {need:12.1f}")
# =>   Baum-Welch on 2520 days, 3 random starts, best likelihood kept
#             data    p11    p22    duration 1    duration 2    vol 1    vol 2    log-lik gain    BIC needs
#       pure noise 0.9177 0.9063          12.2          10.7   0.0104   0.0115             1.0         19.6
#      two regimes 0.9902 0.9662         101.6          29.6   0.0071   0.0202           492.2         19.6
```

The second row is the algorithm working. On data generated by a genuine two-regime process it recovers $\hat p_{11}=0.9902$ against a true $0.989$, $\hat p_{22}=0.9662$ against $0.972$, and volatilities of $0.0071$ and $0.0202$ against $0.0073$ and $0.0202$. The implied durations, $101.6$ and $29.6$ days against truths of $90.9$ and $35.7$, carry the error bar [Markov Chains](05-markov-chains.md) measured — the persistence parameter is estimated from a few dozen transitions no matter how many days are supplied.

The first row is generated by a single Gaussian with no states, no persistence, and no structure of any kind. Baum–Welch returns $\hat p_{11}=0.9177$ and $\hat p_{22}=0.9063$: **two regimes with expected durations of $12.2$ and $10.7$ days, fitted to white noise.** The volatilities, $0.0104$ and $0.0115$, straddle the single true value of $0.011$ and differ by ten percent, which is exactly the sort of gap a narrative absorbs without effort. Nothing in the output announces that the states are fictitious, and a report quoting the transition matrix and the durations would be indistinguishable from a report of a real finding.

What separates the rows is the last two columns, and this is the constructive part. Fitting two states to noise buys a log-likelihood gain of $1.0$ over the one-state model, against the $19.6$ that BIC demands for five extra parameters — so the criterion rejects it decisively. On genuine regime data the gain is $492.2$, roughly five hundred times the threshold. **The point estimates cannot tell the two situations apart and the likelihood gain separates them by a factor of five hundred**, which is why the course's insistence that BIC "ranks fit" rather than making the decision has to be read in both directions: it will not choose the number of states for you, and it will absolutely tell you when the answer is zero.

!!! note "Two more failure modes are guaranteed by the structure and are cheap to defend against"
    **Label switching**: the likelihood is invariant under permuting the states, so "state 1" means nothing until a convention is imposed. Every run above sorts by fitted volatility, which is why the columns are comparable at all; the course does the same thing with `argsort` and is right to call the line "not cosmetic." **Degenerate solutions**: a state can collapse onto a handful of observations with $\hat\sigma_i^{2}\to0$, sending the likelihood to infinity, which is why the block clips the variance and why real implementations use a variance floor or a prior. Both are properties of the objective rather than bugs in the optimizer, and neither is detected by anything in the fit report. The habit that catches all of it is the one the fifth section demonstrates — refit on data known to contain nothing, and see what the model says.

## The Inference Is Exact and the Model Is a Hypothesis

The algorithms on this page are unusually good. The forward recursion computes a sum over $10^{120}$ paths exactly, in a number of operations linear in the data, and the block confirms it to fourteen decimals. Filtering falls out of it for free and is genuinely usable in real time. Smoothing, decoding and fitting all reuse the same two passes. There is very little in applied statistics where the computational story is this clean.

The modelling story is not clean at all, and the gap between the two is where the losses live. The exactness of the arithmetic says nothing about whether a hidden state exists, how many there are, or whether the fitted ones correspond to anything an investor would recognize — and the fifth section shows the machinery producing confident, plausible, entirely fictitious regimes when handed noise. Three defences cost almost nothing and are the practical residue of this page: fit the model to data you know has no regimes and check that the likelihood gain collapses, quote the number of transitions the persistence estimate rests on rather than the number of days, and never let a backtest see a smoothed probability.

That last one is the expensive one, because it is invisible. The third section's simulation is the most favourable case anyone will ever meet — the model is correct, the parameters are known exactly, and the data was generated by the fitted process — and even there the smoothed rule beat the filtered rule by $24\%$ and beat perfect knowledge of the hidden state outright. A backtest that consumed $\gamma$ instead of $\alpha$ would show a superb result, would survive every diagnostic anyone runs on the returns, and would be reporting a quantity that no live system can produce. The full application to real return data, with the state-count decision and the regime-conditional performance analysis, is [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md), and the market context that motivates it is [Market Regimes](../../part-01-foundations/06-market-regimes.md).
