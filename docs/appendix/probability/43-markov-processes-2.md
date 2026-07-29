# Markov Processes II

[Markov Processes I](42-markov-processes-1.md) built the chain machinery — the Markov property, transition matrices, $n$-step probabilities, recurrence — and introduced Hidden Markov Models up through the forward algorithm and filtering. This page finishes both stories. For observable chains: where does the chain settle in the long run (steady-state probabilities), and how long does it stay in a state once it arrives (sojourn times)? For HMMs: how to infer states retrospectively (smoothing), how to recover the single most likely hidden path (Viterbi), and how to estimate the parameters from data alone (Baum–Welch).

## Steady State Probabilities

The two-state chain in Part I gave a strong hint. With leaving probabilities $a$ and $b$,

$$r_{11}(n)=\frac{b}{a+b}+\frac{a}{a+b}(1-a-b)^n\;\longrightarrow\;\frac{b}{a+b},$$

and $r_{21}(n)$ converges to the *same* limit: after many transitions, the probability of finding the chain in state $1$ no longer depends on where it started. The general result:

!!! abstract "Steady-State Convergence Theorem"
    Let $X_n$ be a finite-state Markov chain with a **single recurrent class** that is **aperiodic**. Then for every $j$, the limit

    $$\pi_j=\lim_{n\to\infty}r_{ij}(n)$$

    exists and is independent of the initial state $i$. The $\pi_j$ are the unique nonnegative solution of the **balance equations** together with normalization:

    $$\pi_j=\sum_{k=1}^{m}\pi_k\,p_{kj}\quad(j=1,\ldots,m),\qquad \sum_{k=1}^{m}\pi_k=1.$$

    Moreover $\pi_j=0$ for every transient state $j$ and $\pi_j>0$ for every recurrent state $j$.

The balance equations are easy to motivate: start from the Chapman–Kolmogorov recursion $r_{ij}(n)=\sum_k r_{ik}(n-1)\,p_{kj}$ and let $n\to\infty$ on both sides. If the limits exist, they must satisfy $\pi_j=\sum_k\pi_k p_{kj}$ — the hard part of the theorem (which we will not prove) is that the limits *do* exist under the stated conditions. In matrix language, the row vector $\pi=(\pi_1,\ldots,\pi_m)$ satisfies $\pi=\pi P$: the steady-state distribution is a left eigenvector of the transition matrix with eigenvalue $1$, normalized to sum to $1$. A distribution with this property is also called **stationary**: if $X_0\sim\pi$, then $X_n\sim\pi$ for all $n$ — the chain is in "statistical equilibrium" from the start.

For the two-state chain, the balance equation for state $1$ reads $\pi_1=\pi_1(1-a)+\pi_2 b$, i.e. $\pi_1 a=\pi_2 b$ — the long-run flow $1\to 2$ matches the flow $2\to 1$. With $\pi_1+\pi_2=1$,

$$\pi_1=\frac{b}{a+b},\qquad \pi_2=\frac{a}{a+b},$$

matching the limit computed directly in Part I.

The steady-state probabilities carry three equivalent interpretations:

1. **Limiting probability**: $\pi_j$ is (approximately) the probability of finding the chain in state $j$ at a fixed faraway time, regardless of the start.
2. **Long-run fraction of time**: if $v_{ij}(n)$ counts visits to $j$ in the first $n$ transitions starting from $i$, then $v_{ij}(n)/n\to\pi_j$ with probability $1$. Likewise the long-run frequency of $j\to k$ transitions is $\pi_j p_{jk}$ — a fact we exploit twice below.
3. **Mean recurrence time**: the expected number of transitions between successive visits to $j$ is $t_j^{*}=1/\pi_j$. Rare states are rarely refreshed.

!!! note "Steady state as unconditional regime frequency"
    For a regime chain, $\pi$ answers "ignoring all current information, what fraction of the time does the market spend in each regime?" A two-state fit with $a=\mathbf{P}(\text{calm}\to\text{turbulent})=0.02$ and $b=\mathbf{P}(\text{turbulent}\to\text{calm})=0.10$ gives $\pi=(5/6,\,1/6)$: calm five days out of six. The filtered probability from Part I is the *conditional* refinement of this unconditional base rate — with no recent data, your best guess reverts to $\pi$.

### Why the Conditions Matter

Both hypotheses of the theorem are load-bearing.

- **Multiple recurrent classes**: with two absorbing states, $r_{ij}(n)$ still converges but the limit depends on the start — a chain absorbed in class $A$ never reports statistics of class $B$. Balance equations then have multiple solutions.
- **Periodicity**: take $p_{12}=p_{21}=1$. Then $r_{11}(n)$ alternates $1,0,1,0,\ldots$ and never converges, even though the time-average fraction of visits still converges to $\tfrac12$. Aperiodicity is what upgrades convergence of *averages* to convergence of *probabilities*.

### Birth–Death Chains and the Checkout Counter

A **birth–death chain** is one whose transitions move at most one step: from state $i$, up with probability $b_i$, down with probability $d_i$, staying put otherwise (the checkout counter of Part I is exactly this). For such chains the balance equations collapse to a one-line **local balance** relation:

$$\pi_i\,b_i=\pi_{i+1}\,d_{i+1},\qquad i=0,1,\ldots,m-1.$$

??? note "Proof"
    Consider the "cut" between states $\{0,\ldots,i\}$ and $\{i+1,\ldots,m\}$. Because the chain moves one step at a time, every crossing left-to-right is an $i\to i+1$ transition and every crossing right-to-left is an $i+1\to i$ transition. Crossings must alternate — the chain cannot cross the cut rightward twice without crossing back in between — so in $n$ transitions the numbers of crossings in the two directions differ by at most $1$. Dividing by $n$ and letting $n\to\infty$, the long-run frequencies of the two transition types are equal. By interpretation 2 above these frequencies are $\pi_i b_i$ and $\pi_{i+1}d_{i+1}$.

Iterating local balance gives an explicit product formula, up to the normalizing constant:

$$\pi_i=\pi_0\prod_{k=0}^{i-1}\frac{b_k}{d_{k+1}},\qquad \sum_{i=0}^{m}\pi_i=1.$$

For the checkout counter, every interior ratio is

$$\rho=\frac{p(1-q)}{q(1-p)},$$

so the steady-state probabilities scale geometrically in $\rho$ (with small corrections at the two boundary states). The single number $\rho$ — arrival pressure over service pressure — governs congestion: if $\rho<1$ the distribution piles up near an empty queue; if $\rho>1$ it piles up at capacity; if $\rho=1$ it is (nearly) uniform.

### Sojourn Times: How Long Does a Regime Last?

Suppose the chain has just entered state $i$. Each subsequent step, independently, it stays with probability $p_{ii}$ and leaves with probability $1-p_{ii}$ (to *some* other state). The number of steps $T_i$ spent in state $i$ before leaving is therefore [geometric](17-geometric-distribution.md) with parameter $1-p_{ii}$:

$$\mathbf{P}(T_i=k)=p_{ii}^{\,k-1}(1-p_{ii}),\qquad \mathbb{E}[T_i]=\frac{1}{1-p_{ii}}.$$

This turns the diagonal of an estimated transition matrix directly into regime persistence:

| $p_{ii}$ | Expected duration |
|---|---|
| $0.90$ | $10$ days |
| $0.98$ | $50$ days |
| $0.995$ | $200$ days |

!!! warning "Geometric durations are an assumption, not a finding"
    The geometric law is *forced by the Markov property*: memorylessness means the regime's remaining lifetime never depends on how long it has already lasted. Real regimes may age — a two-year-old calm period may genuinely be more fragile than a two-month-old one — and a fitted HMM cannot represent that. This is a known limitation to keep in mind when interpreting fitted models (semi-Markov models relax it, at a cost in complexity).

### Speed of Convergence

How fast is steady state reached? In the two-state chain the answer was exact: the deviation decays like $(1-a-b)^n$, and $1-a-b$ is precisely the second eigenvalue of $P$. This generalizes: for a chain satisfying the convergence theorem, $r_{ij}(n)$ approaches $\pi_j$ geometrically at a rate governed by the second-largest eigenvalue modulus of $P$. Persistent regimes (diagonal entries near $1$) mean a second eigenvalue near $1$ and slow **mixing** — which cuts both ways. It makes unconditional long-run statistics slow to trust, but it is exactly why *conditional* inference pays: a state that decays slowly is a state worth estimating.

## Hidden Markov Models: Smoothing, Decoding, Learning

Recall the setup from Part I: hidden chain $X_n$ on $\{1,\ldots,m\}$ with parameters $\lambda=(\pi_0,P,\{f_i\})$, observations $y_{0:T}$, joint likelihood

$$p(x_{0:T},y_{0:T})=\pi_0(x_0)\,f_{x_0}(y_0)\prod_{n=1}^{T}p_{x_{n-1}x_n}f_{x_n}(y_n),$$

and forward variables $\alpha_n(i)=p(y_{0:n},X_n=i)$ computed by the forward recursion. The forward pass answered "what is the likelihood?" and "what state now?". Three questions remain.

### The Backward Algorithm

Define the **backward variable** as the density of the *future* observations given the current state:

$$\beta_n(i)=p(y_{n+1:T}\mid X_n=i),\qquad \beta_T(i)=1.$$

It satisfies a mirror-image recursion, run from $T$ down to $0$: for $n=T-1,\ldots,0$,

$$\beta_n(i)=\sum_{j=1}^{m}p_{ij}\,f_j(y_{n+1})\,\beta_{n+1}(j).$$

??? note "Proof"
    Condition on the next state:

    $$\begin{align}
    \beta_n(i)&=p(y_{n+1:T}\mid X_n=i)\\
    &=\sum_{j=1}^{m}\mathbf{P}(X_{n+1}=j\mid X_n=i)\;p(y_{n+1},y_{n+2:T}\mid X_{n+1}=j)\\
    &=\sum_{j=1}^{m}p_{ij}\;f_j(y_{n+1})\;p(y_{n+2:T}\mid X_{n+1}=j)\\
    &=\sum_{j=1}^{m}p_{ij}\,f_j(y_{n+1})\,\beta_{n+1}(j).
    \end{align}$$

    The middle steps use the HMM conditional-independence structure: given $X_{n+1}=j$, the observation $y_{n+1}$ has density $f_j$ and is independent of the later observations, which themselves depend only on $X_{n+1}$.

Like the forward pass, the backward pass costs $O(m^2T)$. As a consistency check, $p(y_{0:T})=\sum_i\pi_0(i)f_i(y_0)\beta_0(i)$ — the same number the forward pass produces at termination.

### Smoothing: the Forward–Backward Algorithm

The **smoothed** state probability conditions on the *entire* sample:

$$\gamma_n(i)=\mathbf{P}(X_n=i\mid y_{0:T})=\frac{\alpha_n(i)\,\beta_n(i)}{\displaystyle\sum_{j=1}^{m}\alpha_n(j)\,\beta_n(j)}.$$

??? note "Proof"
    The key identity is $p(y_{0:T},X_n=i)=\alpha_n(i)\,\beta_n(i)$:

    $$\begin{align}
    p(y_{0:T},X_n=i)&=p(y_{0:n},X_n=i)\;p(y_{n+1:T}\mid X_n=i,y_{0:n})\\
    &=\alpha_n(i)\,\beta_n(i),
    \end{align}$$

    where the second factor drops the conditioning on $y_{0:n}$ because, given $X_n=i$, future observations are independent of past ones (they are generated by the future of the chain, which is conditionally independent of the past given the present). Dividing by $p(y_{0:T})=\sum_j\alpha_n(j)\beta_n(j)$ gives $\gamma_n(i)$.

Intuitively, $\alpha_n(i)$ scores state $i$ by how well it explains the past, $\beta_n(i)$ by how well it sets up the future, and smoothing multiplies the two. Smoothed probabilities are sharper than filtered ones — a spike of volatility that, in real time, was ambiguous ("noise or new regime?") is resolved by seeing what came after.

!!! warning "Filtering vs smoothing: the lookahead trap"
    $\gamma_n(i)$ conditions on $y_{n+1:T}$ — data that did not exist at time $n$. Smoothed regime labels are the right tool for *historical analysis* ("how did the strategy perform in each regime?") and for parameter estimation below. They are the wrong input for a *backtest of a regime-switching rule*: trading at time $n$ on $\gamma_n$ silently imports the future and produces regime timing no live system can reproduce. Backtests must use the filtered probabilities $\mathbf{P}(X_n=i\mid y_{0:n})$ from Part I.

### Decoding: the Viterbi Algorithm

Smoothing answers state questions one time-point at a time. A different question is: what is the single most likely *sequence* of hidden states,

$$\hat x_{0:T}=\arg\max_{x_{0:T}}\ \mathbf{P}(X_{0:T}=x_{0:T}\mid y_{0:T})=\arg\max_{x_{0:T}}\ p(x_{0:T},y_{0:T})\,?$$

(The two argmaxes agree because the conditional and the joint differ by the factor $p(y_{0:T})$, which does not involve $x_{0:T}$.) This is *not* the same as picking $\arg\max_i\gamma_n(i)$ at each $n$: the pointwise-best sequence ignores transition probabilities and can even be an impossible path — if $p_{ij}=0$, the pointwise labels may still demand an $i\to j$ step. Maximizing over $m^{T+1}$ paths looks as hopeless as the evaluation problem did, and it is cured by the same trick: dynamic programming, with $\max$ replacing $\sum$. Define

$$\delta_n(j)=\max_{x_{0:n-1}}\ p(x_{0:n-1},X_n=j,y_{0:n}),$$

the probability of the best partial path ending in state $j$ at time $n$.

**Initialization.** $\delta_0(i)=\pi_0(i)f_i(y_0)$.

**Recursion.** For $n=0,\ldots,T-1$,

$$\delta_{n+1}(j)=\Bigl[\max_{1\le i\le m}\ \delta_n(i)\,p_{ij}\Bigr]f_j(y_{n+1}),\qquad
\psi_{n+1}(j)=\arg\max_{1\le i\le m}\ \delta_n(i)\,p_{ij}.$$

**Termination and backtracking.** $\hat x_T=\arg\max_j\delta_T(j)$, then $\hat x_n=\psi_{n+1}(\hat x_{n+1})$ for $n=T-1,\ldots,0$.

??? note "Proof of the recursion"
    Using the joint likelihood factorization, a path ending $(\ldots,X_n=i,X_{n+1}=j)$ has probability [best path to $i$ at $n$] $\times\,p_{ij}f_j(y_{n+1})$, and the maximum over all paths to $j$ at $n+1$ splits over the intermediate state:

    $$\begin{align}
    \delta_{n+1}(j)&=\max_{x_{0:n}}\ p(x_{0:n},X_{n+1}=j,y_{0:n+1})\\
    &=\max_{1\le i\le m}\ \Bigl[\max_{x_{0:n-1}}p(x_{0:n-1},X_n=i,y_{0:n})\Bigr]p_{ij}\,f_j(y_{n+1})\\
    &=\Bigl[\max_i\ \delta_n(i)\,p_{ij}\Bigr]f_j(y_{n+1}).
    \end{align}$$

    The interchange in the second line is the principle of optimality: the best path through $X_n=i$ must begin with the best path *to* $X_n=i$, since the remaining factors do not depend on $x_{0:n-1}$.

The cost is $O(m^2T)$, the same as the forward pass. In practice Viterbi is always run on logarithms — products of thousands of small numbers underflow — which turns the recursion into $\log\delta_{n+1}(j)=\max_i[\log\delta_n(i)+\log p_{ij}]+\log f_j(y_{n+1})$, a max-plus recursion that is numerically bulletproof. Viterbi paths are the standard way to paint historical regime labels on a price chart; the smoothed $\gamma$'s are the right object when you need *probabilities* rather than a single label.

### Learning: the Baum–Welch Algorithm

Everything so far assumed the parameters $\lambda=(\pi_0,P,\{f_i\})$ were known. In practice nothing is known: from returns alone we must estimate the transition matrix, each regime's mean and variance, and the initial distribution. Maximum likelihood asks for

$$\hat\lambda=\arg\max_\lambda\ \log p(y_{0:T};\lambda)=\arg\max_\lambda\ \log\!\!\sum_{x_{0:T}}p(x_{0:T},y_{0:T};\lambda),$$

and the log-of-a-sum over $m^{T+1}$ paths has no closed-form maximizer. Baum–Welch is the **Expectation–Maximization (EM)** algorithm specialized to HMMs: it alternates between inferring the hidden states given current parameters, and re-estimating parameters given that soft inference.

The pivot is the **complete-data log-likelihood** — what we could maximize easily if the path were visible:

$$\log p(x_{0:T},y_{0:T};\lambda)=\log\pi_0(x_0)+\sum_{n=1}^{T}\log p_{x_{n-1}x_n}+\sum_{n=0}^{T}\log f_{x_n}(y_n).$$

It splits into three pieces touching disjoint parameter blocks ($\pi_0$, $P$, emissions) — the reason the M-step below decouples so cleanly.

**E-step.** Given current parameters $\lambda^{(k)}$, take the expectation of the complete-data log-likelihood over the hidden path's posterior. Because the expression above is linear in the indicators $\mathbf{1}\{x_n=i\}$ and $\mathbf{1}\{x_{n-1}=i,x_n=j\}$, the expectation needs only two families of posterior quantities: the smoothed state probabilities $\gamma_n(i)$ from the forward–backward pass, and the **smoothed transition probabilities**

$$\xi_n(i,j)=\mathbf{P}(X_n=i,X_{n+1}=j\mid y_{0:T})
=\frac{\alpha_n(i)\,p_{ij}\,f_j(y_{n+1})\,\beta_{n+1}(j)}{p(y_{0:T})}.$$

??? note "Proof of the $\xi$ formula"
    Factor the event along the timeline:

    $$\begin{align}
    p(X_n=i,X_{n+1}=j,y_{0:T})
    &=\underbrace{p(y_{0:n},X_n=i)}_{\alpha_n(i)}\;
    \underbrace{\mathbf{P}(X_{n+1}=j\mid X_n=i)}_{p_{ij}}\;
    \underbrace{p(y_{n+1}\mid X_{n+1}=j)}_{f_j(y_{n+1})}\;
    \underbrace{p(y_{n+2:T}\mid X_{n+1}=j)}_{\beta_{n+1}(j)},
    \end{align}$$

    each conditioning dropping exactly the variables that the HMM structure makes irrelevant. Divide by $p(y_{0:T})$. Consistency check: $\sum_{j}\xi_n(i,j)=\gamma_n(i)$.

The expected complete-data log-likelihood (the "Q-function") becomes

$$Q(\lambda,\lambda^{(k)})=\sum_{i=1}^{m}\gamma_0(i)\log\pi_0(i)
+\sum_{n=0}^{T-1}\sum_{i=1}^{m}\sum_{j=1}^{m}\xi_n(i,j)\log p_{ij}
+\sum_{n=0}^{T}\sum_{i=1}^{m}\gamma_n(i)\log f_i(y_n),$$

where all $\gamma$'s and $\xi$'s are computed under $\lambda^{(k)}$.

**M-step.** Maximize $Q$ over $\lambda$. The three blocks separate, and each maximization has a closed form:

$$\hat\pi_0(i)=\gamma_0(i),\qquad
\hat p_{ij}=\frac{\displaystyle\sum_{n=0}^{T-1}\xi_n(i,j)}{\displaystyle\sum_{n=0}^{T-1}\gamma_n(i)},$$

and, for Gaussian emissions $f_i(y)=\mathcal{N}(y;\mu_i,\sigma_i^2)$,

$$\hat\mu_i=\frac{\displaystyle\sum_{n=0}^{T}\gamma_n(i)\,y_n}{\displaystyle\sum_{n=0}^{T}\gamma_n(i)},\qquad
\hat\sigma_i^2=\frac{\displaystyle\sum_{n=0}^{T}\gamma_n(i)\,(y_n-\hat\mu_i)^2}{\displaystyle\sum_{n=0}^{T}\gamma_n(i)}.$$

Every update is a **soft-count** version of the obvious estimator: $\hat p_{ij}$ is "expected number of $i\to j$ transitions over expected number of visits to $i$"; $\hat\mu_i$ and $\hat\sigma_i^2$ are the sample mean and variance of the data, with each observation weighted by the posterior probability that state $i$ generated it. If the path were actually observed, the $\gamma$'s and $\xi$'s would be $0/1$ indicators and these would reduce to ordinary empirical frequencies and moments.

??? note "Derivation of the transition update"
    Maximize the middle block of $Q$ over row $i$ of $P$ subject to $\sum_j p_{ij}=1$. With Lagrange multiplier $\eta$,

    $$\frac{\partial}{\partial p_{ij}}\left[\sum_{n=0}^{T-1}\xi_n(i,j)\log p_{ij}+\eta\Bigl(1-\sum_{j'}p_{ij'}\Bigr)\right]
    =\frac{\sum_{n}\xi_n(i,j)}{p_{ij}}-\eta=0,$$

    so $p_{ij}\propto\sum_n\xi_n(i,j)$. Normalizing the row and using $\sum_j\xi_n(i,j)=\gamma_n(i)$ gives the stated formula. The update for $\hat\pi_0$ is the same argument applied to the first block (with $\gamma_0$ in place of the $\xi$-sums).

??? note "Derivation of the Gaussian updates"
    The emission block for state $i$ is

    $$\sum_{n=0}^{T}\gamma_n(i)\left[-\tfrac12\log(2\pi\sigma_i^2)-\frac{(y_n-\mu_i)^2}{2\sigma_i^2}\right].$$

    Setting the $\mu_i$-derivative to zero: $\sum_n\gamma_n(i)(y_n-\mu_i)/\sigma_i^2=0$, which gives the weighted mean $\hat\mu_i$. Setting the $\sigma_i^2$-derivative to zero:

    $$\sum_{n}\gamma_n(i)\left[-\frac{1}{2\sigma_i^2}+\frac{(y_n-\hat\mu_i)^2}{2\sigma_i^4}\right]=0
    \;\Longrightarrow\;
    \hat\sigma_i^2=\frac{\sum_n\gamma_n(i)(y_n-\hat\mu_i)^2}{\sum_n\gamma_n(i)}.$$

**The algorithm.** Initialize $\lambda^{(0)}$; repeat {E-step: forward–backward under $\lambda^{(k)}$ to get $\gamma,\xi$; M-step: closed-form updates to get $\lambda^{(k+1)}$} until the log-likelihood stops improving. Each iteration costs $O(m^2T)$.

Why does alternating these two steps climb the *actual* likelihood, which is not what the M-step maximizes? This is the EM monotonicity guarantee:

$$\log p(y_{0:T};\lambda^{(k+1)})\ \ge\ \log p(y_{0:T};\lambda^{(k)}),$$

with equality only at a stationary point.

??? note "Proof of monotonicity"
    Write $\lambda'=\lambda^{(k)}$ and let $q(x)=p(x_{0:T}\mid y_{0:T};\lambda')$ be the current posterior over paths. For any $\lambda$, since $p(x,y;\lambda)=p(y;\lambda)\,p(x\mid y;\lambda)$,

    $$\log p(y;\lambda)=\underbrace{\mathbb{E}_q\bigl[\log p(x,y;\lambda)\bigr]}_{Q(\lambda,\lambda')}-\underbrace{\mathbb{E}_q\bigl[\log p(x\mid y;\lambda)\bigr]}_{H(\lambda,\lambda')},$$

    where the expectation over $q$ leaves $\log p(y;\lambda)$ untouched because it does not depend on $x$. Subtracting the same identity at $\lambda=\lambda'$:

    $$\log p(y;\lambda)-\log p(y;\lambda')=\bigl[Q(\lambda,\lambda')-Q(\lambda',\lambda')\bigr]-\bigl[H(\lambda,\lambda')-H(\lambda',\lambda')\bigr].$$

    The second bracket is nonpositive by Jensen's inequality:

    $$H(\lambda,\lambda')-H(\lambda',\lambda')=\mathbb{E}_q\!\left[\log\frac{p(x\mid y;\lambda)}{q(x)}\right]
    \le\log\ \mathbb{E}_q\!\left[\frac{p(x\mid y;\lambda)}{q(x)}\right]=\log 1=0.$$

    So any $\lambda$ with $Q(\lambda,\lambda')\ge Q(\lambda',\lambda')$ — in particular the M-step maximizer — satisfies $\log p(y;\lambda)\ge\log p(y;\lambda')$.

!!! warning "What EM does not guarantee"
    Monotone ascent, yes; the global maximum, no. The HMM likelihood surface is multimodal, and Baum–Welch converges to a local maximum that depends on the initialization. Standard practice: run from several starting points (e.g. states initialized by sorting observations into volatility buckets, or by k-means on rolling volatility) and keep the best likelihood. Watch for two failure modes: **degenerate solutions**, where a state collapses onto a handful of observations and $\hat\sigma_i^2\to 0$ sends the likelihood to infinity (cured by a variance floor or a prior), and **label switching** — the likelihood is invariant to permuting the states, so "state 1" means nothing until you impose a convention, such as ordering states by $\hat\sigma_i$.

### Numerical Scaling

The forward variable $\alpha_n(i)$ is a joint density of $n+1$ observations: it shrinks (or grows) geometrically in $n$ and underflows double-precision arithmetic within a few hundred steps. The standard fix normalizes at every step: set $\tilde\alpha_0(j)=\pi_0(j)f_j(y_0)$ and, for $n\ge 1$,

$$\tilde\alpha_n(j)=\Bigl[\sum_i\hat\alpha_{n-1}(i)\,p_{ij}\Bigr]f_j(y_n),\qquad
c_n=\sum_j\tilde\alpha_n(j),\qquad
\hat\alpha_n(j)=\frac{\tilde\alpha_n(j)}{c_n}.$$ Then the normalized variables are exactly the filtered probabilities, the scale factors are one-step predictive densities, and the log-likelihood is recovered as a sum:

$$\hat\alpha_n(j)=\mathbf{P}(X_n=j\mid y_{0:n}),\qquad
c_n=p(y_n\mid y_{0:n-1}),\qquad
\log p(y_{0:T})=\sum_{n=0}^{T}\log c_n.$$

??? note "Proof"
    By induction on $n$: assume $\hat\alpha_{n-1}(i)=\mathbf{P}(X_{n-1}=i\mid y_{0:n-1})$ and $\prod_{k<n}c_k=p(y_{0:n-1})$ (both hold at $n=1$, since $c_0=\sum_j\pi_0(j)f_j(y_0)=p(y_0)$). Then $\tilde\alpha_n(j)=\alpha_n(j)/p(y_{0:n-1})$, because the unscaled recursion is linear and the inductive hypothesis says we have divided through by $p(y_{0:n-1})$. Summing over $j$: $c_n=p(y_{0:n})/p(y_{0:n-1})=p(y_n\mid y_{0:n-1})$, and dividing gives $\hat\alpha_n(j)=\alpha_n(j)/p(y_{0:n})=\mathbf{P}(X_n=j\mid y_{0:n})$. The product of the $c$'s telescopes to $p(y_{0:T})$.

The backward variables are rescaled by the same constants $c_n$, which cancel in the ratios defining $\gamma$ and $\xi$, so forward–backward and Baum–Welch run unchanged on the scaled quantities. The identity $\log p(y_{0:T})=\sum_n\log c_n$ has a bonus interpretation: the log-likelihood is a sum of *one-step-ahead predictive* log-densities, i.e. exactly a walk-forward evaluation of the model's next-day forecasts — the quantity you would want for comparing models anyway. (The alternative to scaling is to run everything in log-space with the log-sum-exp trick; Viterbi, which uses only products and maxima, needs plain logarithms and no scaling at all.)

### Choosing the Number of States

The number of hidden states $m$ is not estimated by Baum–Welch — it is chosen by the modeler, and the likelihood alone cannot choose it, since more states never fit worse. A Gaussian HMM with $m$ states has

$$k=\underbrace{(m-1)}_{\pi_0}+\underbrace{m(m-1)}_{P}+\underbrace{2m}_{\mu_i,\sigma_i^2}=m^2+2m-1$$

free parameters ($7$ for $m=2$, $14$ for $m=3$, $23$ for $m=4$). The standard penalized criteria,

$$\mathrm{AIC}=-2\log\hat L+2k,\qquad \mathrm{BIC}=-2\log\hat L+k\log(T+1),$$

trade fit against complexity (smaller is better); BIC penalizes harder and typically selects fewer states. A more honest criterion for trading use is out-of-sample one-step predictive log-likelihood — fit on one span, evaluate $\sum\log c_n$ on the next — since predictive performance is what a live system experiences. On daily financial returns, two or three states usually suffice; beyond that, extra states tend to fit noise, split existing regimes into near-duplicates (aggravating label switching), and destabilize the estimated transition matrix.

### From Machinery to Markets

A two-state Gaussian HMM fitted to daily equity-index returns reliably recovers the same structure (numbers below are representative orders of magnitude, not estimates to reuse):

| | State 1 ("calm") | State 2 ("turbulent") |
|---|---|---|
| Mean $\hat\mu_i$ (daily) | small positive ($\sim+0.05\%$) | negative ($\sim-0.1\%$) |
| Volatility $\hat\sigma_i$ (daily) | $\sim 0.6$–$0.8\%$ | $\sim 1.5$–$2.5\%$ |
| Persistence $\hat p_{ii}$ | $0.98$–$0.99$ | $0.90$–$0.96$ |
| Steady-state $\pi_i$ | $\sim 0.8$–$0.9$ | $\sim 0.1$–$0.2$ |

Every row is one of the objects built in these two pages: the emission parameters separate the regimes by volatility (and, weakly, by mean); the diagonal of $P$ gives expected durations of months and weeks respectively via $1/(1-p_{ii})$; the stationary distribution gives unconditional regime frequencies; and the filtered probability $\mathbf{P}(X_n=\text{turbulent}\mid y_{0:n})$ is a real-time, lookahead-free regime signal that can gate position sizing.

Keep the model's honest limitations in view: the regimes are constructs of a fitted model, not observable facts; sojourn times are forced to be geometric; parameters are assumed constant while markets drift; and Gaussian emissions understate tails *within* a regime (Student-$t$ emissions are a common upgrade — much of the unconditional fat-tailedness of returns is, however, already generated by the mixture of regimes itself). These trade-offs, and the full application to real return data — fitting, state-count selection, regime-conditional performance analysis, and regime-aware sizing — are taken up in [Bayesian Methods and Hidden Markov Models](../../part-03-statistics/06-bayesian-methods-and-hmms.md), with the motivating market context in [Market Regimes](../../part-01-foundations/06-market-regimes.md).
