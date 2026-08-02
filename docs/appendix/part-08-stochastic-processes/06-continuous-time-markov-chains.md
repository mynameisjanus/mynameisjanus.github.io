# Continuous-Time Markov Chains

Taking a Markov chain off the grid removes an assumption nobody meant to make. A discrete chain quietly asserts that the world updates once per bar, so every quantity it produces — a persistence, a duration, a mixing rate — is denominated in a unit chosen by whoever built the data set. The continuous-time version has no bar, states it all in real time, and in exchange forces the holding times to be exponential and raises a question the discrete version never had to answer: whether the matrix you fitted corresponds to any continuous process at all.

This page covers the construction from exponential holding times and an embedded jump chain, the generator matrix and the Kolmogorov equations whose solution is a matrix exponential, stationarity as a linear system together with the detailed-balance shortcut, the embedding problem and the transition matrices that admit no generator, and the bias that observing a chain less often puts into every duration estimated from it. It does not let the state be hidden, which is [Hidden Markov Models](07-hidden-markov-models.md); it does not run on a grid, which is [Markov Chains](05-markov-chains.md); it does not free the holding-time law, which is [Renewal Processes](04-renewal-processes.md); it takes no diffusion limit, which is [Brownian Motion](08-brownian-motion.md); and it develops no queueing theory and no birth–death asymptotics.

The trading stake is a conversion the course performs and trusts. [Mean Reversion and Pairs Trading](../../part-04-strategy-development/02-mean-reversion-and-pairs-trading.md) fits an autoregression to the SPY–IVV spread, gets $\rho=0.82$, and converts it to a continuous-time mean-reversion speed by $\theta=-252\ln\rho$, reporting "a spring stiffness of 51 per year" and building the trade's entire tempo on it. That conversion is a matrix logarithm in one dimension, and it is legitimate precisely because $\rho>0$. The fourth section shows what happens to the same manoeuvre in two dimensions when the corresponding quantity is negative, which is that the continuous-time object being described does not exist.

## Memorylessness Forces Exponential Holding Times, So the Only Freedom Is the Rate

A **continuous-time Markov chain** on a finite state space $S=\{1,\ldots,m\}$ is a process $X(t)$ for which, given the present state, the future is independent of the past — the same requirement as before, now demanded at every real time rather than at every integer. Write $P_{ij}(t)=\mathbf{P}(X(t)=j\mid X(0)=i)$ for the transition function.

The first consequence is that the modeller has almost no freedom about *when* transitions happen.

??? note "Proof that the time spent in a state must be exponential, with no assumption beyond the Markov property"
    Let $T_i$ be the time until the chain leaves state $i$, having just entered it. The Markov property applied at time $s$ says that if the chain is still in $i$ at time $s$, its future is that of a chain freshly started in $i$ — the elapsed $s$ is part of the past and therefore carries no information. So for all $s,t\geq0$,

    $$\mathbf{P}(T_i>s+t\mid T_i>s)=\mathbf{P}(T_i>t).$$

    Writing $G(t)=\mathbf{P}(T_i>t)$, this is the functional equation $G(s+t)=G(s)G(t)$. Among functions that are not identically zero and are right-continuous with $G(0)=1$, the only solutions are $G(t)=e^{-q_it}$ for some constant $q_i\geq0$ — the standard Cauchy exponential argument, which fixes $G$ on the rationals from $G(1)$ and extends by continuity. Hence $T_i$ is exponential with some rate $q_i$, and $\mathbb{E}[T_i]=1/q_i$.

    The load-bearing hypothesis is the Markov property *at every real time*, and it is spent in the first display. This is the continuous-time twin of the result that a discrete chain's sojourn is geometric, which [Markov Chains](05-markov-chains.md) derives and [Geometric Distribution](../part-05-common-distributions/03-geometric-distribution.md) states as a property of the law. **In neither case is the holding-time distribution a modelling choice — it is a theorem**, and a process whose states age is not a Markov chain in any state space that leaves out the age.

So the entire specification is: a rate $q_i$ for leaving each state, and a probability $J_{ij}$ of where it goes when it leaves. The matrix $J$ is a transition matrix with zero diagonal, called the **embedded jump chain**, and it is a discrete-time Markov chain in its own right — it records the sequence of states visited with the timing thrown away. Combining the two gives the **generator**

$$q_{ij}=q_iJ_{ij}\ \ (i\neq j),\qquad q_{ii}=-q_i,\qquad Q=[q_{ij}],$$

whose rows sum to zero and whose off-diagonal entries are non-negative. Read $q_{ij}$ as the instantaneous rate of flow from $i$ to $j$: over a short interval $\delta$, $P_{ij}(\delta)=q_{ij}\delta+o(\delta)$ for $j\neq i$, which is the [Poisson process](03-poisson-processes.md)'s infinitesimal definition attached to each ordered pair of states.

## The Generator Is a Derivative and the Transition Matrix Is Its Exponential

The generator earns its name by being the derivative of the transition function at zero, $Q=P'(0)$, and everything else follows from the Chapman–Kolmogorov identity $P(t+s)=P(t)P(s)$, which holds for the same reason it did on the grid.

??? note "Proof that the transition function is the matrix exponential of the generator"
    Differentiate the semigroup identity in $s$ at $s=0$:

    $$P'(t)=\lim_{\delta\to0}\frac{P(t+\delta)-P(t)}{\delta}=P(t)\lim_{\delta\to0}\frac{P(\delta)-I}{\delta}=P(t)\,Q,$$

    which is the **Kolmogorov forward equation**. Splitting the interval the other way, $P(t+\delta)=P(\delta)P(t)$, gives the **backward equation** $P'(t)=QP(t)$. Both are linear matrix differential equations with the initial condition $P(0)=I$, and their unique solution is

    $$P(t)=e^{Qt}=\sum_{k=0}^{\infty}\frac{(Qt)^{k}}{k!}.$$

    That the two equations have the same solution is the statement that $Q$ commutes with $e^{Qt}$, which it does. The series converges for every $Q$ because the terms are dominated by $\lVert Q\rVert^{k}t^{k}/k!$, and it automatically produces a stochastic matrix: rows of $Q$ summing to zero means $Q\mathbf{1}=0$, so $e^{Qt}\mathbf{1}=\mathbf{1}$, and non-negativity of the entries follows from writing $e^{Qt}=e^{-\Lambda t}e^{(Q+\Lambda I)t}$ with $\Lambda\geq\max_iq_i$, where both factors have non-negative entries.

    The load-bearing hypothesis is time-homogeneity, spent in writing $P(t+\delta)=P(t)P(\delta)$ with the same $P$ on both sides. That last rewriting is also a construction — **uniformization** — and it is how these chains are simulated in practice: run a Poisson clock at rate $\Lambda$ and take a step of the discrete chain $I+Q/\Lambda$ at each tick, with self-transitions absorbing the surplus.

The relation $P(t)=e^{Qt}$ is exact, not an approximation, and it is worth confirming against a chain simulated the honest way — one exponential holding time and one jump at a time, with no matrix arithmetic anywhere.

```python
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(8061)
Q = np.array([[-0.011, 0.010, 0.001],                          # calm, stress, crisis
              [0.028, -0.050, 0.022],
              [0.005, 0.060, -0.065]])
rate = -np.diag(Q)
J = Q / rate[:, None]
np.fill_diagonal(J, 0.0)                                       # the embedded jump chain
reps = 400_000
print(f"  a three-state generator simulated jump by jump, {reps} paths started calm")
print("        t    P(calm) sim    expm    P(stress) sim    expm    P(crisis) sim    expm")
state = np.zeros(reps, dtype=np.int64)
clock = rng.exponential(1 / rate[state])
for t in (5.0, 21.0, 63.0, 252.0):
    while True:
        idx = np.flatnonzero(clock <= t)
        if idx.size == 0:
            break
        u = rng.random(idx.size)
        nxt = (u[:, None] > np.cumsum(J[state[idx]], axis=1)).sum(axis=1)
        state[idx] = nxt
        clock[idx] += rng.exponential(1 / rate[nxt])
    e = expm(Q * t)[0]
    f = np.bincount(state, minlength=3) / reps
    print(f"  {t:9.0f} {f[0]:13.4f} {e[0]:9.4f} {f[1]:15.4f} {e[1]:9.4f}"
          f" {f[2]:15.4f} {e[2]:9.4f}")
# =>   a three-state generator simulated jump by jump, 400000 paths started calm
#            t    P(calm) sim    expm    P(stress) sim    expm    P(crisis) sim    expm
#              5        0.9491    0.9497          0.0444    0.0439          0.0066    0.0064
#             21        0.8368    0.8364          0.1308    0.1308          0.0325    0.0328
#             63        0.7122    0.7110          0.2125    0.2139          0.0753    0.0751
#            252        0.6635    0.6634          0.2439    0.2439          0.0926    0.0927
```

Every simulated column sits on its matrix-exponential counterpart to three decimals — $0.9491$ against $0.9497$, $0.8368$ against $0.8364$, $0.7122$ against $0.7110$, $0.6635$ against $0.6634$ down the calm column, and the same agreement in the other two. Nothing about the simulation knows that $e^{Qt}$ exists; it draws exponential waits and jumps.

The crisis column is the one that repays reading. Starting from calm, the probability of being in the crisis state is $0.0066$ after a week and $0.0926$ after a year, and the route there is almost entirely indirect: the generator allows $q_{13}=0.001$ directly, so nearly all of that mass arrives via stress. **A generator with a tiny direct rate can still deliver substantial probability to a distant state, because $e^{Qt}$ sums over every path length**, which is exactly what the discrete $P^{n}$ did and is easier to under-appreciate when the exponent is a real number rather than a count.

## Stationarity Is a Linear System, and Detailed Balance Solves It When It Applies

A distribution $\pi$ is **stationary** when starting from it leaves the law unchanged for all time: $\pi P(t)=\pi$ for every $t$. Differentiating at $t=0$ turns that into a linear system in the generator,

$$\pi Q=0,\qquad \sum_i\pi_i=1,$$

which is the continuous-time balance equation and reads, entry by entry, as $\sum_{i\neq j}\pi_iq_{ij}=\pi_jq_j$ — total probability flowing into $j$ equals total flowing out. If the chain is irreducible, this system has a unique solution and $P_{ij}(t)\to\pi_j$ for every starting state $i$, with no aperiodicity condition required. Periodicity was an artifact of the grid: a continuous-time chain cannot return to its start on a fixed schedule because its holding times are continuous random variables, so the pathology that forced an extra hypothesis in [Markov Chains](05-markov-chains.md) simply cannot arise.

!!! note "Periodicity was a property of the grid rather than of the chain, and removing the grid removes it"
    [Markov Chains](05-markov-chains.md) needs aperiodicity as a separate hypothesis, and demonstrates why with the two-state chain that alternates deterministically: $r_{11}(n)$ reads $1,0,1,0,\ldots$ and converges to nothing, while the long-run fraction of time still converges to $\tfrac12$. No continuous-time chain can do this. Returning to the start at exactly the times $2,4,6,\ldots$ requires holding times that add up to integers, and exponential holding times are continuous, so the probability of any such coincidence is zero. The practical residue is that irreducibility alone gives convergence here, and that a fitted discrete chain showing near-periodic behaviour — alternating far more than it persists — is telling you something about the sampling interval rather than about the market, which is the fourth section's subject from the other direction.

Some chains admit a shortcut. A chain is **reversible** if there is a distribution satisfying **detailed balance**,

$$\pi_iq_{ij}=\pi_jq_{ji}\quad\text{for every pair }i\neq j,$$

a much stronger requirement than the balance equations — it asks the flows to match pair by pair rather than only in aggregate. When it holds, summing over $i$ recovers $\pi Q=0$, so any solution of detailed balance is automatically stationary, and it is usually found by inspection rather than by solving a system. Every birth–death chain, in which transitions move only to neighbouring states, is reversible for this reason: crossings of any cut must alternate, so the flows across it balance one pair at a time. Most regime generators are *not* reversible, since a market that slides calm to stress to crisis and jumps back directly to calm has a preferred direction, and that asymmetry is a modelling statement worth making deliberately rather than assuming away.

## Not Every Transition Matrix Has an Hourly Version

Here is the question that has no discrete-time analogue. A daily transition matrix $P$ has been estimated. What is the *hourly* transition matrix? The answer ought to be $P^{1/8}$, obtained by finding the generator $Q$ with $e^{Q}=P$ and forming $e^{Q/8}$ — and for some perfectly ordinary $P$ no such real $Q$ exists. This is the **embedding problem**, and it is not a numerical difficulty but a fact about which matrices are reachable.

??? note "Proof that a two-state transition matrix is embeddable exactly when its second eigenvalue is positive"
    Take $P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix}$ with eigenvalues $1$ and $\lambda_2=1-a-b$. Any generator for the same two states has the form $Q=\begin{pmatrix}-\alpha&\alpha\\ \beta&-\beta\end{pmatrix}$ with $\alpha,\beta\geq0$, and its eigenvalues are $0$ and $-(\alpha+\beta)\leq0$. Since $e^{Q}$ has eigenvalues $e^{0}=1$ and $e^{-(\alpha+\beta)}$, the second eigenvalue of any embeddable matrix is $e^{-(\alpha+\beta)}\in(0,1]$ — **strictly positive**.

    Conversely, if $\lambda_2>0$ then setting $\alpha+\beta=-\ln\lambda_2$ and splitting it in the ratio $a:b$ produces a valid generator, and the closed form is $Q=\frac{\ln\lambda_2}{\lambda_2-1}(P-I)$, from which

    $$P^{(t)}=e^{Qt}=I+\frac{\lambda_2^{\,t}-1}{\lambda_2-1}\,(P-I).$$

    So the fractional powers exist and are stochastic exactly when $\lambda_2>0$, and $\lambda_2^{\,t}$ is where the requirement bites: a negative number has no real power at $t=\tfrac12$.

    The load-bearing hypothesis is that a generator's non-unit eigenvalues have non-positive real parts, which is forced by its rows summing to zero with non-negative off-diagonals. **A matrix with $\lambda_2<0$ describes a system that alternates faster than any continuous flow can**, since $a+b>1$ means the chain is more likely than not to change state each step, and a continuous process must pass through intermediate times where it has not yet decided.

```python
import numpy as np
from scipy.linalg import logm, sqrtm

print("  does a daily transition matrix come from a continuous-time chain?")
print("         a        b     lam2    max |imag| log P    min off-diag    max |imag| sqrt P")
for a, b in ((0.011, 0.028), (0.20, 0.30), (0.45, 0.45), (0.60, 0.70)):
    P = np.array([[1 - a, a], [b, 1 - b]])
    L, S = logm(P.astype(complex)), sqrtm(P.astype(complex))
    print(f"  {a:10.3f} {b:8.3f} {1 - a - b:8.3f} {np.abs(L.imag).max():19.4f}"
          f" {min(L[0, 1].real, L[1, 0].real):15.4f} {np.abs(S.imag).max():21.4f}")
# =>   does a daily transition matrix come from a continuous-time chain?
#             a        b     lam2    max |imag| log P    min off-diag    max |imag| sqrt P
#           0.011    0.028    0.961              0.0000          0.0112                0.0000
#           0.200    0.300    0.500              0.0000          0.2773                0.0000
#           0.450    0.450    0.100              0.0000          1.1513                0.0000
#           0.600    0.700   -0.300              1.6916          0.5557                0.2949
```

The first three rows behave. Their logarithms are real to machine precision — the imaginary column reads $0.0000$ — and the off-diagonal entries are positive, $0.0112$, $0.2773$, $1.1513$, so each is a genuine generator and every fractional power of each matrix is a genuine transition matrix. The first row is the course's fitted regime chain, and it embeds comfortably.

The last row is the failure, and it is total rather than marginal. With $a=0.6$ and $b=0.7$ the second eigenvalue is $-0.300$, the matrix logarithm has an imaginary part of magnitude $1.6916$, and the square root has an imaginary part of $0.2949$. **There is no hourly transition matrix, no half-daily one, and no continuous-time process of any kind whose daily snapshot is this chain** — the question is not hard to answer, it is malformed.

!!! warning "A fitted chain that alternates faster than it persists cannot be converted to any other frequency, and the conversion will silently return complex numbers"
    The condition $\lambda_2<0$ means $a+b>1$: the chain is more likely to switch than to stay. That is not exotic — it is what a fitted chain looks like whenever the "state" is really a fast-alternating indicator, such as the sign of a mean-reverting spread sampled slowly, or an over-fitted regime label that flips with the noise. The practical damage is that every frequency conversion downstream is now unavailable: no intraday version of the model, no aggregation to weekly that agrees with the daily fit, no continuous-time analogue to reason with. And the failure is quiet, because `logm` and `sqrtm` return complex arrays rather than raising, so a pipeline that takes a real part somewhere will produce numbers that look fine and mean nothing. The check is one line — compute $\det P$, or the eigenvalues, and require them positive — and it belongs next to the fit rather than next to the incident.

## Reading the State Less Often Makes Every Regime Look Longer

The embedding problem asks whether a discrete fit has a continuous-time parent. The complementary question is more common and more damaging: given that a continuous-time chain is the truth, what does fitting a discrete chain to samples of it at spacing $d$ report?

The answer is visible in $P(d)=e^{Qd}$. A discrete fit estimates $p_{11}=P_{11}(d)$ and quotes the duration $d/(1-P_{11}(d))$. For small $d$ that recovers $1/q_1$, because $P_{11}(d)\approx1-q_1d$. For large $d$ it does not, because $P_{11}(d)\to\pi_1$ and the quoted duration grows without bound like $d/(1-\pi_1)$. Every transition that begins and ends between two observations is invisible, and invisible transitions read as persistence.

```python
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(8067)
q1, q2 = 1 / 91.0, 1 / 36.0                                    # true exit rates, calm and stress
Q = np.array([[-q1, q1], [q2, -q2]])
reps, horizon = 40_000, 25_200.0
print(f"  a chain whose true calm duration is {1 / q1:.0f} days, observed every d days")
print("        d    P11(d) exact    implied duration    fitted from a path    truth")
rate = np.array([q1, q2])
state = np.zeros(reps, dtype=np.int64)
clock = rng.exponential(1 / rate[state])
for d in (1.0, 5.0, 21.0, 63.0):
    stay = vis = 0
    s, c = state.copy(), clock.copy()
    t = 0.0
    while t < horizon:
        t += d
        prev = s.copy()
        while True:
            idx = np.flatnonzero(c <= t)
            if idx.size == 0:
                break
            s[idx] = 1 - s[idx]                                # two states: every jump flips
            c[idx] += rng.exponential(1 / rate[s[idx]])
        vis += (prev == 0).sum()
        stay += ((prev == 0) & (s == 0)).sum()
    p = stay / vis
    e = expm(Q * d)[0, 0]
    print(f"  {d:9.0f} {e:15.4f} {d / (1 - e):19.1f} {d / (1 - p):21.1f} {1 / q1:8.1f}")
# =>   a chain whose true calm duration is 91 days, observed every d days
#            d    P11(d) exact    implied duration    fitted from a path    truth
#              1          0.9892                92.8                  92.7     91.0
#              5          0.9501               100.1                 100.1     91.0
#             21          0.8421               133.0                 133.0     91.0
#             63          0.7412               243.4                 243.6     91.0
```

The exact and the fitted columns agree to a tenth of a day at every spacing, which confirms that nothing here is an estimation artifact: the discrete chain is being estimated perfectly and is reporting the wrong number by construction.

The size of the error is the point. Daily observation of a $91$-day regime reports $92.8$ days — a $2\%$ overstatement, negligible, which is why nobody notices the effect exists. Weekly observation reports $100.1$. Monthly reports $133.0$, a $46\%$ overstatement. **Quarterly observation of a three-month regime reports $243.4$ days, nearly triple the truth**, and every one of those numbers came from a correctly fitted, correctly interpreted, perfectly converged Markov chain.

The direction is always the same, and it has to be: coarser sampling can only hide transitions, never invent them, so a persistence estimate is biased upward and a mixing rate downward by an amount that depends entirely on the observation frequency. This puts a caveat on the course's fitted numbers that no amount of extra data removes. The $0.989$ and the $93$ days are daily-bar quantities, and at daily spacing on a quarter-long regime the bias is the $2\%$ in the first row — small, but a property of the sampling scheme rather than of the market, and it would be a different number on weekly bars.

## One Matrix in Real Time, and the Two Questions the Grid Never Asked

A continuous-time chain replaces the transition matrix with a generator and the matrix power with a matrix exponential, and in doing so states everything in units the world supplies rather than units the data vendor does. The structure is if anything simpler: holding times are exponential with no choice in the matter, stationarity is a linear system, and the aperiodicity condition that discrete chains needed evaporates because a continuous clock cannot keep a schedule.

What the change buys is the ability to ask two questions that a grid cannot even express, and both have unpleasant answers. Whether a fitted matrix corresponds to any continuous process is a real question with a real criterion — positive eigenvalues — and matrices that fail it arise naturally from over-fitted or fast-alternating states, at which point every frequency conversion in the pipeline is undefined and most software will return complex numbers rather than complain. And what a fitted matrix means when the truth is continuous depends on the observation spacing, always in the direction of exaggerating persistence, by $2\%$ at daily spacing and by a factor of nearly three at quarterly.

The rule that follows is to treat a transition matrix as a pair — the matrix and the interval it was estimated at — and never to quote a duration without the second half. The generator is the frequency-free object, it is one `logm` away whenever it exists, and the check that it exists costs a determinant. Where the state itself cannot be observed, all of this sits underneath an inference problem that has to be solved first, which is [Hidden Markov Models](07-hidden-markov-models.md).
