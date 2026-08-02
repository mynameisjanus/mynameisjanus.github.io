# Reinforcement Learning for Execution

[Part VII's reinforcement-learning lesson](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) performed an autopsy on RL applied to daily-bar strategy selection — twenty seeds, a median out-of-sample gain of +0.20 against an in-sample +0.59, eight of twenty losing money — and diagnosed the failure precisely: the state's next-day mean return was 4% of its standard deviation, so the agent was learning from 96% noise. It then named the place where the diagnosis runs the other way. Execution is genuinely sequential, its action space is well defined, its episodes have natural boundaries, and its reward arrives every few seconds rather than every few months. That lesson pointed here and named [Almgren–Chriss](04-optimal-execution-almgren-chriss.md) as the baseline any RL execution agent must beat. This module takes up the charge.

The result is not the one the literature advertises. The risk-neutral version of the optimal execution problem was solved exactly by Bertsimas and Lo in 1998, and the answer is TWAP — so a deep network that "discovers" even slicing has rediscovered a closed form, at considerable expense. Where a genuine short-term signal exists, this module's experiments find that **a one-line hand-coded rule captures +13.25 basis points against TWAP while a tabular Q-learner trained on 400,000 episodes captures −11.89** — it loses to the baseline it was supposed to beat, on a problem where the edge is provably there. Along the way, two failures turn out to be more instructive than any success: an agent that never trades because its state discretization cannot tell it has traded, and an agent that looks excellent in a simulator without impact and destroys **55 basis points** when deployed against the real thing.

## Execution is an MDP, and a small one

Formulate the problem. The state is $(k, x_k, \phi_k)$ — the interval index, the shares still to be liquidated, and whatever market features are observable. The action is how much to trade in the next interval, expressed here as a multiple of the TWAP slice $x_k/(N-k)$. The reward is the proceeds of that slice measured against the arrival price, so that maximizing total reward minimizes implementation shortfall. The Bellman optimality condition is standard,

$$
Q^{*}(s, a) \;=\; \mathbb{E}\bigl[\,r + \max_{a'} Q^{*}(s', a')\,\bigr],
$$

with no discounting, because the episode is finite and every basis point costs the same whenever it is paid. Tabular Q-learning updates toward that target,

$$
Q(s,a) \;\leftarrow\; Q(s,a) + \alpha_t\Bigl(r + \max_{a'}Q(s',a') - Q(s,a)\Bigr),
$$

and converges to $Q^*$ under the Robbins–Monro conditions provided every state–action pair is visited infinitely often. That proviso is not a formality; it is where two of this module's three failures originate.

The simulator throughout is deliberately simple and fully disclosed: a mid-price following an arithmetic random walk with per-step volatility $\sigma$, temporary impact $\eta n$ charged against the execution price of each slice, an inventory that must reach zero by interval $N$, and in some worlds an observable AR(1) signal $\alpha_t$ that shifts the next price increment. The parameters ($N = 10$, $X = 100{,}000$ shares, $\sigma = 0.004$, $\eta = 2\times10^{-7}$) put impact at roughly 2 basis points per TWAP slice against about 4 basis points of per-slice price noise. That ratio is *far* more favorable than reality — a real execution problem hides impact under fifty times as much noise — and it is chosen so the learning problem is solvable at all. Every conclusion below is therefore an *optimistic* bound on what RL can do here.

## Dynamic programming already solved the textbook problem

Before training anything, solve the problem exactly. With linear temporary impact and a martingale price, minimize expected cost by backward induction. Guess a quadratic value function $V_k(x) = a_k x^2$ for the cost of liquidating $x$ shares in the remaining $N-k$ intervals. At the final interval everything must go, so $a_N = \eta/\tau$. At any earlier interval,

$$
V_k(x) \;=\; \min_{n}\ \left[\frac{\eta}{\tau}n^2 \;+\; a_{k+1}(x-n)^2\right],
$$

which is a one-dimensional quadratic in $n$. Differentiating gives the optimal slice $n^{*} = \frac{a_{k+1}}{\eta/\tau + a_{k+1}}\,x$, and substituting back confirms the quadratic form propagates, with

$$
a_k \;=\; \frac{(\eta/\tau)\,a_{k+1}}{\eta/\tau + a_{k+1}}
\qquad\Longrightarrow\qquad
a_k \;=\; \frac{\eta/\tau}{N-k+1},
$$

as induction on the harmonic recursion confirms. Substituting that back into $n^*$ gives $n^{*} = x/(N-k+1)$ — **equal slices of the remaining inventory, which is exactly TWAP**:

```python
import numpy as np

X, N, eta, tau = 1_000_000, 10, 1e-7, 1.0
e = eta / tau
a = np.zeros(N + 2)
a[N] = e
for k in range(N - 1, 0, -1):
    a[k] = e * a[k + 1] / (e + a[k + 1])

x, schedule = float(X), []
for k in range(1, N + 1):
    if k == N:
        schedule.append(x)
        break
    take = x * a[k + 1] / (e + a[k + 1])
    schedule.append(take)
    x -= take

print(f"DP schedule: {np.round(np.array(schedule), 4)[:4]} ... (X/N = {X / N:,.0f})")
print(f"max deviation from equal slices: {np.abs(np.array(schedule) - X / N).max():.2e} shares")
print(f"a_k / (eta/tau):     {np.round(a[1:N + 1] / e, 4)}")
print(f"closed form 1/(N-k+1): {np.round([1 / (N - k + 1) for k in range(1, N + 1)], 4)}")
# => DP schedule: [100000. 100000. 100000. 100000.] ... (X/N = 100,000)
#    max deviation from equal slices: 1.46e-11 shares
#    a_k / (eta/tau):     [0.1    0.1111 0.125  0.1429 0.1667 0.2    0.25   0.3333 0.5    1.    ]
#    closed form 1/(N-k+1): [0.1    0.1111 0.125  0.1429 0.1667 0.2    0.25   0.3333 0.5    1.    ]
```

The value coefficients match $1/(N-k+1)$ to four decimals and the schedule is equal slices to eleven. This is the single most important fact in the module, and it should be read as a warning about a genre of paper: **an execution agent that converges to even slicing has reproduced a 1998 closed form.** Adding risk aversion changes the answer to the hyperbolic-sine trajectory [module 04 derived](04-optimal-execution-almgren-chriss.md), also in closed form. The analytic solutions cover the entire textbook problem, so RL's only legitimate territory is the set of assumptions those solutions make — nonlinear or stochastic impact, time-varying liquidity, and above all a short-term price signal. Everything below tests whether a learner can capture that territory.

## An agent that cannot tell it has traded will not trade

Point tabular Q-learning at the problem the previous section solved, discretizing inventory into five buckets — a choice that looks innocuous and matches plenty of published work. Then repeat with twenty-one:

```python
import numpy as np

N, X, SIGMA, ETA = 10, 100_000, 0.004, 2e-7
ACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])   # multiples of the TWAP slice

def run_episode(policy, seed, eta=ETA):
    rng = np.random.default_rng(seed)
    x, s, proceeds = float(X), 0.0, 0.0
    for k in range(N):
        n = x if k == N - 1 else min(x, max(0.0, policy(k, x)) * x / (N - k))
        proceeds += (s - eta * n) * n
        x -= n
        s += SIGMA * rng.standard_normal()
    return -proceeds / X * 1e4                              # shortfall in bp

def train_q(n_inv, n_episodes=400_000, seed=1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N - 1, n_inv, len(ACTIONS)))
    cnt = np.zeros_like(Q)
    for ep in range(n_episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * n_episodes))
        x, s, traj = float(X), 0.0, []
        for k in range(N - 1):
            ib = min(n_inv - 1, int(x / X * n_inv))
            ai = (int(rng.integers(len(ACTIONS))) if rng.random() < eps
                  else int(np.argmax(Q[k, ib])))
            n = min(x, ACTIONS[ai] * x / (N - k))
            r = (s - ETA * n) * n / X * 1e4
            x -= n
            s += SIGMA * rng.standard_normal()
            if k == N - 2:                                   # fold in the forced last slice
                traj.append((k, ib, ai, r + (s - ETA * x) * x / X * 1e4, None))
            else:
                traj.append((k, ib, ai, r, min(n_inv - 1, int(max(x, 0.0) / X * n_inv))))
        for (k, ib, ai, r, nib) in traj:
            cnt[k, ib, ai] += 1
            lr = 1.0 / (1.0 + cnt[k, ib, ai]) ** 0.6
            Q[k, ib, ai] += lr * (r + (0.0 if nib is None else Q[k + 1, nib].max())
                                  - Q[k, ib, ai])
    def policy(k, x):
        return (1.0 if k >= N - 1
                else ACTIONS[int(np.argmax(Q[k, min(n_inv - 1, int(x / X * n_inv))]))])
    return policy, Q

seeds = np.arange(20_000, 40_000)                            # common random numbers
twap = np.array([run_episode(lambda k, x: 1.0, s) for s in seeds])
print(f"TWAP (provably optimal here): {twap.mean():.2f} bp")
for n_inv in [5, 21]:
    pol, Q = train_q(n_inv)
    c = np.array([run_episode(pol, s) for s in seeds])
    d = c - twap
    print(f"  {n_inv:>2} inventory buckets: {c.mean():6.2f} bp, gap "
          f"{d.mean():+5.2f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.2f} bp, "
          f"first slice {ACTIONS[int(np.argmax(Q[0, n_inv - 1]))]:.2f}x TWAP")
# => TWAP (provably optimal here): 18.88 bp
#       5 inventory buckets:  32.57 bp, gap +13.69 +/- 0.45 bp, first slice 0.00x TWAP
#      21 inventory buckets:  20.08 bp, gap +1.20 +/- 0.06 bp, first slice 1.00x TWAP
```

With twenty-one buckets the agent **rediscovers TWAP** — it selects exactly 1.00× the TWAP slice and lands 1.20 basis points above a provably optimal 18.88, the residual being action-grid coarseness and estimation noise. That is the good news, and it is genuinely reassuring: the learner works, and it converges to the closed form.

With five buckets the same code produces an agent that **never trades the first slice** and pays 13.69 basis points for it. The mechanism is worth stating precisely, because it generalizes far beyond this toy. Five buckets are twenty percentage points wide. A full TWAP slice moves inventory by ten percentage points. So after trading a correct-sized slice, the agent usually lands in *the same bucket it started in* — its state says nothing happened. Trading has an immediate cost and, as far as the table can tell, no effect on the future; not trading is free. The agent is behaving optimally with respect to the world it can perceive, and that world is one in which selling is pure loss. **The failure is not in the algorithm, the reward, or the amount of training; it is in a discretization choice that destroyed the Markov property.** A state that cannot represent the consequence of an action makes the problem non-Markov, and every convergence guarantee evaporates.

## A one-line rule beats the learner where the signal is real

Now give the agent something the analytic solutions do not have: an observable AR(1) signal $\alpha_t$ that predicts the next price increment, added to the state. This is RL's home ground — the case where no closed form applies and a policy must condition on live information. Before asking whether the learner captures the edge, establish that the edge exists, using a rule simple enough to fit on one line: trade less when the signal says the price is about to rise.

```python
import numpy as np

N, X, SIGMA, ETA, AMP, PHI = 10, 100_000, 0.004, 2e-7, 0.003, 0.7

def run_episode(policy, seed):
    rng = np.random.default_rng(seed)
    x, s, alpha, proceeds = float(X), 0.0, 0.0, 0.0
    for k in range(N):
        n = x if k == N - 1 else min(x, max(0.0, policy(k, x, alpha)) * x / (N - k))
        proceeds += (s - ETA * n) * n
        x -= n
        alpha = PHI * alpha + AMP * rng.standard_normal()
        s += SIGMA * rng.standard_normal() + alpha
    return -proceeds / X * 1e4

seeds = np.arange(20_000, 40_000)
twap = np.array([run_episode(lambda k, x, a: 1.0, s) for s in seeds])
print(f"TWAP: {twap.mean():.2f} bp")
for g in [0.25, 0.5, 1.0, 2.0, 3.0]:
    rule = lambda k, x, a, g=g: float(np.clip(1.0 - g * a / AMP, 0.0, 2.0))
    c = np.array([run_episode(rule, s) for s in seeds])
    d = twap - c                                          # positive = beats TWAP
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  rule with gain g = {g:4.2f}: {c.mean():5.2f} bp, beats TWAP by "
          f"{d.mean():+5.2f} bp (SE {se:.2f}, t {d.mean() / se:+.1f})")
# => TWAP: 17.77 bp
#      rule with gain g = 0.25:  7.14 bp, beats TWAP by +10.63 bp (SE 0.13, t +79.5)
#      rule with gain g = 0.50:  4.52 bp, beats TWAP by +13.25 bp (SE 0.20, t +66.0)
#      rule with gain g = 1.00:  8.79 bp, beats TWAP by +8.98 bp (SE 0.25, t +35.8)
#      rule with gain g = 2.00: 14.55 bp, beats TWAP by +3.22 bp (SE 0.29, t +11.2)
#      rule with gain g = 3.00: 16.99 bp, beats TWAP by +0.78 bp (SE 0.30, t +2.6)
```

The edge is unambiguous: **+13.25 basis points** at the best gain setting, with a t-statistic of 66, and a sensible interior optimum — too little responsiveness leaves money on the table, too much over-trades and pays impact. Any analyst who suspected the signal mattered would find this in an afternoon.

The tabular Q-learner, given the same signal in its state, 400,000 training episodes, a 21-bucket inventory grid, and thousands of visits to every cell it uses, captures **none of it**:

| policy | mean shortfall | vs TWAP (paired) |
|---|---|---|
| TWAP | 17.77 bp | — |
| Almgren–Chriss, $\kappa = 0.3$ | 29.52 bp | −11.75 bp |
| hand-coded signal rule, $g = 0.5$ | 4.52 bp | **+13.25 bp** |
| tabular Q-learning | 29.66 bp | **−11.89 bp** (t = −47.2) |

The learner does not merely fail to beat the one-line rule — it loses to TWAP by 11.89 basis points, and inspecting its policy at mid-episode shows why: the action it selects is non-monotone in the signal (aggressive when the signal is neutral, passive when the signal is strongly favorable *and* when strongly adverse), which is not a strategy but noise. The diagnosis is quantitative. The value differences separating adjacent actions are one to two basis points, while each cell's reward carries roughly four basis points of price noise per visit, and the $\max$ operator in the update is upward-biased by an amount that compounds across the nine backward steps. The agent is trying to resolve a 1 bp signal with an estimator whose own bias is of the same order.

The lesson is not that RL cannot work on execution — production desks do run learned execution policies. It is about what makes them work, and none of it is the algorithm: variance-reduced reward formulations that subtract the martingale component of P&L, warm starts from the analytic solution so the policy begins at TWAP rather than at random, function approximation that shares information across neighboring states instead of a table that treats them as unrelated, and sample counts measured in millions of *real* child orders. **The burden of proof runs against the learner.** A policy that cannot beat a one-line rule using the same information has not earned its complexity, and the one-line rule is the honest benchmark — not TWAP, which is the benchmark an RL paper reports when it wants a favorable comparison.

## Common random numbers are worth more than a better algorithm

One methodological note earns its own section because it changes what is measurable. Every comparison above evaluated all policies **on the same 20,000 random seeds** — identical price paths, identical signal realizations, differing only in the policy's decisions. Since the quantity of interest is a difference between policies, and the price noise is common to both, pairing cancels the dominant variance term ([Variance Reduction](../appendix/part-09-monte-carlo-methods/06-variance-reduction.md) prices the same trick from scratch, and names what it costs — the two level estimates are afterwards correlated, so no unpaired test may be applied to them):

$$
\operatorname{Var}\left[\bar d\right] \;=\; \frac{\operatorname{Var}[C_A] + \operatorname{Var}[C_B] - 2\operatorname{Cov}[C_A, C_B]}{n},
$$

and the covariance term is large precisely because both policies are exposed to the same market. In these experiments pairing cut the standard error of the comparison from 1.24 to 0.25 basis points — a **4.9-fold reduction**, equivalent to running twenty-four times as many episodes. Any execution study that reports unpaired means is discarding most of its statistical power, and a great many published comparisons of execution algorithms are underpowered for exactly this reason.

## Simulator fidelity is the whole game

The last failure is the one that ends careers rather than papers. Train three agents in three simulators that differ only in how much impact they charge — none, a tenth of the truth, and the truth — then deploy all three against true impact:

```python
import numpy as np

N, X, SIGMA, ETA = 10, 100_000, 0.004, 2e-7
ACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
N_INV = 21

def run_episode(policy, seed, eta, track=False):
    rng = np.random.default_rng(seed)
    x, s, proceeds, half = float(X), 0.0, 0.0, None
    for k in range(N):
        n = x if k == N - 1 else min(x, max(0.0, policy(k, x)) * x / (N - k))
        proceeds += (s - eta * n) * n
        x -= n
        s += SIGMA * rng.standard_normal()
        if k == 4:
            half = 1.0 - x / X                              # fraction done by halfway
    return (-proceeds / X * 1e4, half) if track else -proceeds / X * 1e4

def train_q(eta_sim, n_episodes=400_000, seed=2):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N - 1, N_INV, len(ACTIONS)))
    cnt = np.zeros_like(Q)
    for ep in range(n_episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * n_episodes))
        x, s, traj = float(X), 0.0, []
        for k in range(N - 1):
            ib = min(N_INV - 1, int(x / X * N_INV))
            ai = (int(rng.integers(len(ACTIONS))) if rng.random() < eps
                  else int(np.argmax(Q[k, ib])))
            n = min(x, ACTIONS[ai] * x / (N - k))
            r = (s - eta_sim * n) * n / X * 1e4
            x -= n
            s += SIGMA * rng.standard_normal()
            if k == N - 2:
                traj.append((k, ib, ai, r + (s - eta_sim * x) * x / X * 1e4, None))
            else:
                traj.append((k, ib, ai, r, min(N_INV - 1, int(max(x, 0.0) / X * N_INV))))
        for (k, ib, ai, r, nib) in traj:
            cnt[k, ib, ai] += 1
            lr = 1.0 / (1.0 + cnt[k, ib, ai]) ** 0.6
            Q[k, ib, ai] += lr * (r + (0.0 if nib is None else Q[k + 1, nib].max())
                                  - Q[k, ib, ai])
    def policy(k, x):
        return (1.0 if k >= N - 1
                else ACTIONS[int(np.argmax(Q[k, min(N_INV - 1, int(x / X * N_INV))]))])
    return policy, Q

seeds = np.arange(20_000, 40_000)
twap_live = np.array([run_episode(lambda k, x: 1.0, s, ETA) for s in seeds])
print(f"TWAP completes {run_episode(lambda k, x: 1.0, 1, ETA, track=True)[1]:.0%} "
      f"of the order by the halfway point")
for factor, label in [(0.0, "impact absent"), (0.1, "impact understated 10x"),
                      (1.0, "impact correct")]:
    pol, Q = train_q(ETA * factor)
    sim = np.array([run_episode(pol, s, ETA * factor) for s in seeds])
    twap_sim = np.array([run_episode(lambda k, x: 1.0, s, ETA * factor) for s in seeds])
    live = np.array([run_episode(pol, s, ETA) for s in seeds])
    halves = np.array([run_episode(pol, s, ETA, track=True)[1] for s in seeds[:2000]])
    print(f"trained where {label:22s}: {(twap_sim - sim).mean():+6.2f} bp vs TWAP in its own "
          f"simulator | {(twap_live - live).mean():+7.2f} bp at true impact | "
          f"{halves.mean():.0%} done by halfway")
# => TWAP completes 50% of the order by the halfway point
#    trained where impact absent         :  +0.23 bp vs TWAP in its own simulator |  -55.40 bp at true impact | 20% done by halfway
#    trained where impact understated 10x:  -0.42 bp vs TWAP in its own simulator |   -6.33 bp at true impact | 26% done by halfway
#    trained where impact correct        :  -0.62 bp vs TWAP in its own simulator |   -0.62 bp at true impact | 53% done by halfway
```

The first row is the whole warning. An agent trained where its own trades do not move the price looks *marginally better than TWAP* in the world that raised it (+0.23 bp) and destroys **55 basis points** when it meets a market that charges for liquidity. The mechanism is visible in the last column and is more interesting than simple over-aggression. When impact is free and the price is a martingale, every schedule has identical expected cost — the value function is *flat*, so the learned policy is shaped entirely by noise in the Q-estimates, and it drifts into completing only **20% of the order by the halfway point** against TWAP's 50%. That arbitrary schedule costs nothing in a frictionless simulator and dumps an enormous residual into the final interval once impact is real. Nothing about the algorithm malfunctioned; it optimized faithfully against the physics it was given, and those physics contained no reason to prefer any schedule at all. Ten-fold understatement is less spectacular and more insidious: the agent looks unremarkable in its simulator, still defers (26% done by halfway), and gives up 6 basis points live — the kind of degradation that gets attributed to "regime change" rather than to a mis-specified cost model. Only the correctly-specified agent tracks TWAP's pacing at 53%.

This is why market-replay simulators — replaying historical order books and filling the agent's orders against them — are so dangerous for training execution policies. Replay is *non-reactive*: the recorded book does not know your order exists, does not step away from it, and refills as if you never traded. The agent learns to consume liquidity that would have vanished. Reactive simulators, which apply an impact model to the agent's own flow, are strictly better and are only as good as the impact model inside them — a model whose parameters [the impact module](05-market-impact-models.md) showed to be nearly unmeasurable from a single desk's fills. Simulator fidelity, not algorithm choice, is the binding constraint on execution RL, and the honest way to state a result is with the impact assumption printed next to it.

!!! warning "A policy that cannot beat a one-line rule has not earned its complexity"
    On the problem where RL is supposed to work best, a tabular agent with 400,000 episodes rediscovered a 1998 closed form when there was no signal, and lost to TWAP by 11.89 basis points when there was one that a hand-coded rule captured 13.25 basis points of. Meanwhile an agent trained without impact in its simulator looked fine and cost 55 basis points live. Before any learned execution policy is trusted, demand three things: a fair benchmark using the same information, a paired evaluation on common random numbers, and the impact model the simulator charged — stated in the same table as the result.

!!! abstract "Key takeaways"
    - Backward induction gives $a_k = (\eta/\tau)/(N-k+1)$ and an optimal slice of $x/(N-k+1)$ — equal slices, reproduced numerically to $1.5\times10^{-11}$ shares, so an agent that "discovers" TWAP has rediscovered Bertsimas and Lo (1998).
    - With 21 inventory buckets the Q-learner recovered the optimum — 1.00× the TWAP slice, 1.20 bp above a provably optimal 18.88 bp.
    - With 5 buckets the identical code never traded the first slice and paid +13.69 bp, because a 10% slice does not change a 20%-wide bucket: the discretization destroyed the Markov property, and no amount of training repairs that.
    - Where a real AR(1) signal existed, a one-line rule beat TWAP by **+13.25 bp** (t = 66) with a sensible interior optimum, proving the edge was capturable.
    - The tabular learner given the same signal scored **−11.89 bp** against TWAP (t = −47.2) with a non-monotone policy — it must resolve 1–2 bp differences using estimates carrying ~4 bp of noise and a compounding maximization bias.
    - Common random numbers cut the comparison's standard error from 1.24 to 0.25 bp, a 4.9× reduction worth 24× the episodes — unpaired execution studies discard most of their power.
    - An agent trained with no impact in its simulator beat TWAP there by +0.23 bp and lost **55.40 bp** at true impact: with impact free the value function is flat, so its policy was noise-shaped and completed only 20% of the order by halfway against TWAP's 50%.
    - Production execution RL works through variance-reduced rewards, warm starts from the analytic solution, function approximation, and millions of real child orders — none of which is the algorithm, and all of which is engineering around the fact that the effect sizes are tiny.

## Where this goes next

The baselines this module measured against are derived in [Optimal Execution](04-optimal-execution-almgren-chriss.md), which produces the closed-form trajectory for the risk-averse case and shows why the efficient frontier rather than a point estimate is the deliverable. The impact model whose mis-specification cost 55 basis points is the subject of [Market Impact Models](05-market-impact-models.md), including the noise-floor arithmetic that makes its parameters so hard to pin down. The original RL autopsy — and the stationarity, signal-density, and episode-structure criteria this module inherited — is [Part VII, lesson four](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md), whose meta-labeling result remains the course's clearest example of machine learning paying for itself. And for the discipline of evaluating any learned policy honestly, [Part IV's validation gauntlet](../part-04-strategy-development/08-validation-and-overfitting.md) applies without modification: a policy selected from many training runs is a selected strategy, and it owes the same deflation as any other.
