# Queue Models

Two pages of [Part VIII](../part-08-stochastic-processes/index.md) decline this subject by name — [Renewal Processes](../part-08-stochastic-processes/04-renewal-processes.md) "develops no queueing theory at all" and [Continuous-Time Markov Chains](../part-08-stochastic-processes/06-continuous-time-markov-chains.md) develops "no queueing theory and no birth–death asymptotics" — and what they are declining is the one piece of stochastic-process theory a limit order book is literally built from. The results are sharp and all three are counter-intuitive in the same direction. Utilization enters hyperbolically, so a venue running at $98\%$ of capacity holds $52.7$ jobs against a $50\%$ venue's $1.0$ and makes each one wait $52.9$ times as long. Queue position enters geometrically, so every $7.3$ units of queue ahead halve the fill probability whatever the queue already is, while the expected wait rises only linearly — $0.02$ times the probability for $40$ times the wait at forty units deep. And the wait is set by the *second* moment of service, not the first: four service laws with an identical mean of $1.000$ and an identical utilization of $0.80$ produce mean waits of $1.995$, $3.998$, $10.600$ and $19.251$, a spread of $9.6\times$ that no first-moment diagnostic can see.

This page covers the birth–death balance equations and the geometric stationary law they produce, the existence condition and the hyperbolic blow-up as utilization approaches one, Little's law and the sample-path argument that needs no distributional assumption at all, the fill probability of a passive order as a race between a draining queue and a departing price, and the Pollaczek–Khinchine formula that makes variability rather than volume the driver of delay. It does not construct a generator matrix or solve Kolmogorov equations, which is [Continuous-Time Markov Chains](../part-08-stochastic-processes/06-continuous-time-markov-chains.md); it does not free the inter-arrival law or prove the renewal theorems, and in particular it consumes rather than re-derives the inspection paradox, which is [Renewal Processes](../part-08-stochastic-processes/04-renewal-processes.md); it does not develop the arrival process itself or its clustering, which is [Order Arrival Processes](06-order-arrival-processes.md); it does not build the exponential gap law, which is [Exponential Distribution](../part-05-common-distributions/10-exponential-distribution.md); it solves no inventory control problem and derives no optimal quote, which is [Market Making](../../advanced/12-market-making.md); it estimates no market impact, which is [Market Impact Models](../../advanced/05-market-impact-models.md); it simulates no fills against a real book, which is [Order Management and Fill Simulation](../../part-05-backtesting-engine/03-order-management-and-fill-simulation.md); and it never treats a mean service time as a description of a queue.

The trading stake is the boundary a course module draws and declines to cross. [Market Making](../../advanced/12-market-making.md) develops the Avellaneda–Stoikov quoting problem in continuous inventory and then stops at a section titled "Queue position, and where the continuous model stops applying," observing that the continuous model prices inventory risk while the discrete queue decides whether a quote is ever filled at all. Section 3 is that discrete object: the same passive order, priced by where it sits in line rather than by what it is quoting, with a fill probability that falls by half every $7.3$ units and a wait that grows linearly the whole way down.

## The Balance Equations Make the Queue Geometric, and the Only Thing That Matters Is How Close Utilization Is to One

A queue with exponential inter-arrivals and exponential service is a birth–death process on the non-negative integers, which is the one case where the machinery of [Continuous-Time Markov Chains](../part-08-stochastic-processes/06-continuous-time-markov-chains.md) collapses to arithmetic.

??? note "Proof that the M/M/1 stationary distribution is geometric in $\rho=\lambda/\mu$, that it exists exactly when $\rho<1$, and that the mean number in system is $\rho/(1-\rho)$"

    The chain jumps from $n$ to $n+1$ at rate $\lambda$ and from $n+1$ to $n$ at rate $\mu$, with no other transitions. Because the state space is a line, every cut between $n$ and $n+1$ must carry equal flow in both directions at stationarity — this is the detailed-balance shortcut of [Continuous-Time Markov Chains](../part-08-stochastic-processes/06-continuous-time-markov-chains.md), available here because the transition graph is a tree. So $\lambda p_n=\mu p_{n+1}$ and hence $p_n=\rho^{n}p_0$ with $\rho=\lambda/\mu$.

    Normalization requires $\sum_{n\ge0}\rho^{n}p_0=1$, which converges exactly when $\rho<1$ and gives $p_0=1-\rho$, so $p_n=(1-\rho)\rho^{n}$: the queue length is geometric. When $\rho\ge1$ the series diverges, no stationary distribution exists, and the queue grows without bound — not slowly, but linearly at rate $\lambda-\mu$. There is no intermediate behaviour and no stable heavily-loaded regime.

    The mean follows from the geometric law:
    $$L=\mathbb{E}[N]=\sum_{n\ge0}n(1-\rho)\rho^{n}=\frac{\rho}{1-\rho},$$
    which is $1$ at $\rho=1/2$, $9$ at $\rho=9/10$ and $49$ at $\rho=49/50$. The function is hyperbolic in $1-\rho$, so the quantity that governs a queue is not how busy the server is but how much idle capacity remains, and halving the idle fraction doubles everything.

    **The load-bearing hypothesis is that $\rho$ is a constant. A venue whose arrival rate varies over the day does not have a single $\rho$, and because $L$ is convex in $\rho$, its average queue exceeds the queue at its average utilization — by Jensen, and by a margin that grows as the busiest period approaches capacity.**

```python
import numpy as np

rng = np.random.default_rng(18051)
N, BURN, MU = 4_000_000, 200_000, 1.0                  # service rate 1 per unit time


def lindley(inter, service):
    """Waiting time in queue: W[n+1] = max(0, W[n] + S[n] - A[n+1])."""
    w, out = 0.0, np.empty(len(service))
    d = service[:-1] - inter[1:]
    for i in range(len(service)):
        out[i] = w
        if i < len(d):
            w = max(0.0, w + d[i])
    return out[BURN:]


print(f"  an M/M/1 queue at utilization rho = arrival rate / service rate, service rate {MU:.0f}."
      f" The stationary queue length is geometric, so P(empty) = 1 - rho, the mean number in system"
      f" is rho/(1-rho), and the mean wait in queue is rho/(mu(1-rho)). {N:,} arrivals,"
      f" {BURN:,} discarded")
print("     rho    P(empty): pred    meas   E[N]: pred    meas   E[wait in queue]: pred    meas"
      "   E[time in system]: pred    meas   waits vs rho=0.50")
base = None
for rho in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98):
    lam = rho * MU
    inter = rng.exponential(1 / lam, N)
    service = rng.exponential(1 / MU, N)
    wq = lindley(inter, service)
    meas_wq = wq.mean()
    if base is None:
        base = meas_wq
    pred_wq = rho / (MU * (1 - rho))
    meas_n = lam * (meas_wq + 1 / MU)                   # Little's law, applied not assumed
    print(f"    {rho:4.2f}   {1 - rho:15.4f}   {np.mean(wq == 0):6.4f}"
          f"   {rho / (1 - rho):12.3f}   {meas_n:6.3f}   {pred_wq:24.3f}   {meas_wq:6.3f}"
          f"   {pred_wq + 1 / MU:25.3f}   {meas_wq + 1 / MU:6.3f}   {meas_wq / base:17.1f}x")
# =>   an M/M/1 queue at utilization rho = arrival rate / service rate, service rate 1. The stationary queue length is geometric, so P(empty) = 1 - rho, the mean number in system is rho/(1-rho), and the mean wait in queue is rho/(mu(1-rho)). 4,000,000 arrivals, 200,000 discarded
#         rho    P(empty): pred    meas   E[N]: pred    meas   E[wait in queue]: pred    meas   E[time in system]: pred    meas   waits vs rho=0.50
#        0.50            0.5000   0.5002          1.000    0.999                      1.000    0.998                       2.000    1.998                 1.0x
#        0.70            0.3000   0.2999          2.333    2.329                      2.333    2.327                       3.333    3.327                 2.3x
#        0.80            0.2000   0.1991          4.000    4.011                      4.000    4.014                       5.000    5.014                 4.0x
#        0.90            0.1000   0.1005          9.000    8.926                      9.000    8.918                      10.000    9.918                 8.9x
#        0.95            0.0500   0.0497         19.000   19.529                     19.000   19.557                      20.000   20.557                19.6x
#        0.98            0.0200   0.0184         49.000   52.718                     49.000   52.794                      50.000   53.794                52.9x
```

The Lindley recursion is run without ever assuming the geometric law, so the agreement is a check rather than an identity: $P(\text{empty})$ comes out at $0.5002$, $0.2999$, $0.1991$, $0.1005$, $0.0497$ and $0.0184$ against a predicted $1-\rho$, and the mean wait at $0.998$, $2.327$, $4.014$, $8.918$, $19.557$ and $52.794$ against $\rho/(\mu(1-\rho))$.

The final row is honest about its own limits and the honesty is part of the lesson. At $\rho=0.98$ the measurement overshoots the prediction by $7.7\%$ despite four million arrivals, because a queue near saturation is enormously autocorrelated and four million arrivals is not four million independent observations. That is the same statement as the last column: going from $50\%$ to $98\%$ utilization multiplies the wait by $52.9$, and it multiplies the time needed to *measure* the wait by a comparable factor. **A system near capacity is both slow and hard to characterize, and the second problem arrives exactly when the first one does.**

## Little's Law Needs No Distribution at All, Which Is Why It Survives Everything Else on This Page

Every result in section 1 assumed exponential inter-arrivals and exponential service. One relation on this page assumes neither, and it is the one that keeps working when the rest of the model is wrong.

??? note "Proof that $L=\lambda W$ by a sample-path argument, with no assumption about arrival or service distributions, independence, or service discipline"

    Over a window $[0,T]$, let $N(t)$ be the number in the system, let $A(T)$ be the number of arrivals, and let $W_i$ be the time customer $i$ spends in the system. Each customer contributes exactly one to $N(t)$ for exactly $W_i$ units of time, so the area under $N$ is the sum of the individual sojourn times:
    $$\int_0^T N(t)\,dt=\sum_{i}W_i+\text{boundary terms},$$
    the boundary terms coming from customers present at the ends of the window. Dividing by $T$,
    $$\underbrace{\frac{1}{T}\int_0^T N(t)\,dt}_{\to\,L}=\underbrace{\frac{A(T)}{T}}_{\to\,\lambda}\cdot\underbrace{\frac{1}{A(T)}\sum_i W_i}_{\to\,W},$$
    and provided the two right-hand limits exist and the boundary terms are $o(T)$, $L=\lambda W$.

    Nothing in the argument used a distribution. It holds for any arrival process, any service law, any number of servers, any queue discipline — first-come-first-served, last-in-first-out, priority, or random — and for any dependence between arrivals and service. It is a bookkeeping identity about areas, not a probabilistic result, which is why it survives every violation the rest of this page will inflict on the exponential assumptions.

    **The load-bearing consequence is a division of labour. Little's law converts between a length and a time and is never the source of an error; every quantity on this page that is actually model-dependent is a $W$, and every $L$ quoted alongside it is that same $W$ multiplied by a rate. So a queue-length statistic and a waiting-time statistic are one measurement reported twice, and disagreement between them indicates a measurement problem rather than a modelling one.**

## Queue Position Enters Geometrically and the Wait Enters Linearly, So Depth Is Far More Expensive Than It Looks

A passive order is not waiting for a server; it is waiting for the orders ahead of it to be consumed before the price leaves. That race has an exact answer, and its shape is the opposite of the intuition that being twice as deep is twice as bad.

```python
import numpy as np

rng = np.random.default_rng(18053)
TRIALS, MU = 400_000, 1.0                              # one unit of queue consumed per unit time

print(f"  a passive order joining a queue of Q units at the best price. Units ahead are consumed by"
      f" trades arriving at rate {MU:.0f}; the price leaves the level at rate nu. The order fills"
      f" only if the queue drains first, which happens with probability (mu/(mu+nu))^Q -- geometric"
      f" in queue position, while the wait is linear in it. {TRIALS:,} trials")
print("     nu/mu   per-unit survival   " + "".join(f"Q={q}: P(fill) pred / meas   " for q in (1, 5, 10, 20, 40)))
for ratio in (0.05, 0.10, 0.25, 0.50):
    nu = ratio * MU
    p1 = MU / (MU + nu)
    cells = []
    for q in (1, 5, 10, 20, 40):
        drain = rng.gamma(q, 1 / MU, TRIALS)           # time for q units to be consumed
        move = rng.exponential(1 / nu, TRIALS)
        cells.append(f"{p1 ** q:16.4f} / {np.mean(drain < move):.4f}   ")
    print(f"    {ratio:5.2f}   {p1:17.4f}   " + "".join(cells))

print("\n     what queue position costs, at nu/mu = 0.10: probability falls geometrically and the"
      " wait rises linearly")
nu = 0.10 * MU
p1 = MU / (MU + nu)
print(f"     every {np.log(2) / np.log(1 / p1):.1f} units of queue ahead halve the fill probability,"
      f" whatever the queue already is")
print("     Q    P(fill)   relative to Q=1   E[wait | fill]   relative to Q=1")
base_wait = None
for q in (1, 5, 10, 20, 40):
    drain = rng.gamma(q, 1 / MU, TRIALS)
    move = rng.exponential(1 / nu, TRIALS)
    wait = drain[drain < move].mean()
    if base_wait is None:
        base_wait = wait
    print(f"    {q:2d}   {p1 ** q:7.4f}   {p1 ** q / p1:15.2f}   {wait:14.2f}"
          f"   {wait / base_wait:16.2f}")
# =>   a passive order joining a queue of Q units at the best price. Units ahead are consumed by trades arriving at rate 1; the price leaves the level at rate nu. The order fills only if the queue drains first, which happens with probability (mu/(mu+nu))^Q -- geometric in queue position, while the wait is linear in it. 400,000 trials
#         nu/mu   per-unit survival   Q=1: P(fill) pred / meas   Q=5: P(fill) pred / meas   Q=10: P(fill) pred / meas   Q=20: P(fill) pred / meas   Q=40: P(fill) pred / meas   
#         0.05              0.9524             0.9524 / 0.9516             0.7835 / 0.7828             0.6139 / 0.6144             0.3769 / 0.3767             0.1420 / 0.1423   
#         0.10              0.9091             0.9091 / 0.9080             0.6209 / 0.6198             0.3855 / 0.3864             0.1486 / 0.1488             0.0221 / 0.0221   
#         0.25              0.8000             0.8000 / 0.8012             0.3277 / 0.3282             0.1074 / 0.1070             0.0115 / 0.0116             0.0001 / 0.0002   
#         0.50              0.6667             0.6667 / 0.6657             0.1317 / 0.1320             0.0173 / 0.0171             0.0003 / 0.0002             0.0000 / 0.0000   
#
#         what queue position costs, at nu/mu = 0.10: probability falls geometrically and the wait rises linearly
#         every 7.3 units of queue ahead halve the fill probability, whatever the queue already is
#         Q    P(fill)   relative to Q=1   E[wait | fill]   relative to Q=1
#         1    0.9091              1.00             0.91               1.00
#         5    0.6209              0.68             4.54               5.00
#        10    0.3855              0.42             9.09              10.00
#        20    0.1486              0.16            18.16              19.99
#        40    0.0221              0.02            36.37              40.04
```

Twenty cells, twenty matches to three decimals, because the closed form is exact: the queue drains before the price leaves with probability $(\mu/(\mu+\nu))^{Q}$, each of the $Q$ units ahead surviving the departure hazard independently. The consequence is the constant in the second panel. At $\nu/\mu=0.10$ every $7.3$ units of queue ahead halve the fill probability, and the halving constant does not depend on how deep the order already is — going from first to eighth in line costs the same factor as going from thirtieth to thirty-eighth.

The two columns on the right are the mismatch. Forty units deep the order fills with $0.02$ of the probability it had at the front, and the wait conditional on filling is $40.04$ times as long. So the penalty for depth is exponential in the thing that determines whether the trade happens and linear in the thing that determines how much capital it ties up, and only the second is visible in a fill log — an unfilled order leaves no row. **Depth is measured in units of time by every system that records it and in units of probability by the only calculation that matters.**

!!! note "Utilization, queue position, the mean service time and the variability of service are four inputs to a delay, and the last is the one no dashboard displays"
    **Utilization** $\rho$ is the fraction of time the server is busy, and section 1 shows delay is hyperbolic in $1-\rho$, so it dominates whenever it is near one. **Queue position** is the number of units ahead of a specific order, and section 3 shows fill probability is geometric in it — a property of one order rather than of the system. **Mean service time** $\mathbb{E}[S]$ sets the scale of everything and is the number every capacity plan is written in. **Variability of service**, entering through $\mathbb{E}[S^{2}]$, is what section 5 shows actually drives the wait at fixed utilization, and it appears on no operational dashboard because dashboards report averages. The four are routinely compressed into "the system is running at eighty percent," which fixes the first, says nothing about the second, implies the third and conceals the fourth.

## The Wait Is Set by the Second Moment of Service, So Identical Averages Differ by a Factor of Ten

Every result so far has assumed exponential service, and the exponential assumption is doing more work than it appears to. The general formula is available, it needs the inspection paradox of [Renewal Processes](../part-08-stochastic-processes/04-renewal-processes.md) and Little's law from section 2, and it says the mean is the wrong summary.

??? note "Proof that the mean wait in queue is $\lambda\mathbb{E}[S^{2}]/(2(1-\rho))$, the Pollaczek–Khinchine formula, by assembling the mean residual service and Little's law"

    Consider a single server with Poisson arrivals at rate $\lambda$ and independent service times $S$ with $\rho=\lambda\mathbb{E}[S]<1$. An arriving customer waits for two things: the *remaining* service of whoever is in progress, and the full service of everyone already queued.

    By PASTA the arrival sees the time-stationary state, so it finds the server busy with probability $\rho$. Conditional on that, the service in progress is *length-biased* — a longer job is more likely to be the one being interrupted — and [Renewal Processes](../part-08-stochastic-processes/04-renewal-processes.md) proves that the expected remaining portion is $\mathbb{E}[S^{2}]/(2\mathbb{E}[S])$, not $\mathbb{E}[S]/2$. The first contribution is therefore $\rho\,\mathbb{E}[S^{2}]/(2\mathbb{E}[S])$.

    For the second, Little's law applied to the queue alone gives an expected queue length of $\lambda W_q$, each requiring $\mathbb{E}[S]$ to serve, contributing $\lambda W_q\mathbb{E}[S]=\rho W_q$. Adding and solving,
    $$W_q=\rho\frac{\mathbb{E}[S^{2}]}{2\mathbb{E}[S]}+\rho W_q\quad\Longrightarrow\quad W_q=\frac{\lambda\mathbb{E}[S^{2}]}{2(1-\rho)}.$$
    Writing $\mathbb{E}[S^{2}]=\mathbb{E}[S]^{2}(1+c^{2})$ for the coefficient of variation $c$ makes the structure visible: $W_q=\rho\mathbb{E}[S](1+c^{2})/(2(1-\rho))$, so at fixed mean service and fixed utilization the wait is linear in $1+c^{2}$. Deterministic service gives $c=0$ and half the exponential wait; exponential gives $c=1$; and a law with $c=3$ gives five times the exponential wait.

    **The load-bearing quantity is $\mathbb{E}[S^{2}]$, and the two inputs everyone controls do not contain it. Utilization is fixed by capacity planning and mean service by the technology, and a system can satisfy both targets exactly while its delay varies by an order of magnitude — which is the exhibit below.**

```python
import numpy as np

rng = np.random.default_rng(18055)
N, BURN, RHO = 4_000_000, 200_000, 0.80                # mean service is 1, so arrival rate is RHO


def lindley(inter, service):
    w, out = 0.0, np.empty(len(service))
    d = service[:-1] - inter[1:]
    for i in range(len(service)):
        out[i] = w
        if i < len(d):
            w = max(0.0, w + d[i])
    return out[BURN:]


def deterministic(n):
    return np.ones(n)


def exponential(n):
    return rng.exponential(1.0, n)


def lognormal(n, cv=3.0):
    s = np.sqrt(np.log1p(cv ** 2))
    return rng.lognormal(-0.5 * s ** 2, s, n)


def pareto(n, alpha=2.05):
    xm = (alpha - 1) / alpha
    return xm * (1 + rng.pareto(alpha, n))


print(f"  four service-time laws with mean exactly 1 and utilization rho = {RHO}, so every row has"
      f" the same arrival rate, the same server, the same fraction of time busy and the same mean"
      f" service. Pollaczek-Khinchine says the wait is lambda E[S^2] / (2(1-rho)), which reads the"
      f" *second* moment. {N:,} arrivals, {BURN:,} discarded")
print("     service law        E[S]    E[S^2]    CV    E[wait in queue]: predicted    measured"
      "   multiple of the deterministic queue")
base = None
for name, fn in (("deterministic", deterministic), ("exponential", exponential),
                 ("lognormal, CV 3", lognormal), ("Pareto, alpha 2.05", pareto)):
    service = fn(N)
    service *= 1.0 / service.mean()                    # fix the first moment exactly
    m2 = (service ** 2).mean()
    inter = rng.exponential(1 / RHO, N)
    wq = lindley(inter, service).mean()
    pred = RHO * m2 / (2 * (1 - RHO))
    if base is None:
        base = wq
    print(f"    {name:18s}  {service.mean():5.3f}   {m2:7.3f}   {service.std():4.2f}"
          f"   {pred:28.3f}   {wq:9.3f}   {wq / base:35.1f}x")
# =>   four service-time laws with mean exactly 1 and utilization rho = 0.8, so every row has the same arrival rate, the same server, the same fraction of time busy and the same mean service. Pollaczek-Khinchine says the wait is lambda E[S^2] / (2(1-rho)), which reads the *second* moment. 4,000,000 arrivals, 200,000 discarded
#         service law        E[S]    E[S^2]    CV    E[wait in queue]: predicted    measured   multiple of the deterministic queue
#        deterministic       1.000     1.000   0.00                          2.000       1.995                                   1.0x
#        exponential         1.000     2.000   1.00                          4.001       3.998                                   2.0x
#        lognormal, CV 3     1.000     9.603   2.93                         19.207      19.251                                   9.6x
#        Pareto, alpha 2.05  1.000     5.410   2.10                         10.819      10.600                                   5.3x
```

The four rows are, from the point of view of every operational metric anyone collects, the same system. Arrival rate identical, mean service identical to three decimals by construction, utilization identical at $0.80$, server identical. The mean wait runs $1.995$, $3.998$, $10.600$ and $19.251$ — a spread of $9.6\times$ — and Pollaczek–Khinchine predicts each one from the second moment alone, at $2.000$, $4.001$, $10.819$ and $19.207$.

This is the honest failure of the whole apparatus, and it is not that queueing theory is wrong. It is that the theory's answer depends on a quantity that capacity planning does not measure, does not target and cannot control by adding servers. Halving the mean service time halves the wait; halving the *variability* of service at a fixed mean halves it too, and only the first appears in any specification. The Pareto row is the sharpest case: at $\alpha=2.05$ the second moment barely exists, and the measured wait, $10.600$ against a predicted $10.819$, is the least accurate row in the table for exactly that reason — the estimator of $\mathbb{E}[S^{2}]$ is itself converging slowly. **The formula reads a moment the system does not report, and as the tail gets heavier the formula's own input becomes the thing that is hardest to estimate, which is [Heavy-Tailed Returns](12-heavy-tailed-returns.md) arriving in the microstructure.**

## Every Repair Is Idle Capacity, a Shorter Queue, or Less Variable Work

The three quantities that drive delay admit three different remedies at three different prices. Utilization is the expensive one and the only one most institutions think about: buying idle capacity buys delay reduction hyperbolically, which is why the move from $95\%$ to $90\%$ is worth more than the move from $70\%$ to $50\%$, and it is the reverse of the intuition that a lightly loaded system has the most to gain. Queue position is not a system property at all but a per-order one, and its remedy is to arrive earlier or to accept the geometric penalty — there is no third option, because the halving constant is a property of the flow rather than of the order.

Variability is the cheap one and the one nobody targets. At a fixed mean and a fixed utilization, moving service from exponential to deterministic halves the wait, and it costs nothing but scheduling discipline. The same statement in market terms is that a venue whose order sizes are heavy-tailed imposes waits that its own capacity statistics cannot explain, and the repair is size discipline rather than more matching engines.

!!! warning "Every operational dashboard reports the first moment of service and the delay is governed by the second"
    Capacity is planned in throughput and mean latency, both first moments, and delay is then predicted from them by a mental model in which doubling variability does nothing. Section 5 shows four systems agreeing on every reported number and disagreeing by $9.6\times$ on the quantity users experience. **The free diagnostic is the coefficient of variation of the service log, one line beside the mean the dashboard already computes, entering the wait as $1+c^{2}$: at $c=1$ the exponential baseline, at $c=3$ five times it, at $c=0$ half it.** It requires no new instrumentation because the individual service times are already recorded to compute their average, and it turns an unexplained latency complaint into an arithmetic one. The same line prices the difference between two venues with identical published throughput, and it is the difference a fill-rate model built on mean queue depth will attribute to luck.

## A Length, a Time, and the Moment That Connects Them

This page established that the M/M/1 queue length is geometric with $p_n=(1-\rho)\rho^{n}$, existing exactly when $\rho<1$, with $L=\rho/(1-\rho)$ hyperbolic in idle capacity, verified at $P(\text{empty})$ of $0.5002$, $0.1005$ and $0.0184$ and waits of $0.998$, $8.918$ and $52.794$ against predictions of $1.000$, $9.000$ and $49.000$, the last row overshooting by $7.7\%$ on four million arrivals because a saturated queue is as hard to measure as it is slow; that Little's law $L=\lambda W$ follows from a sample-path area argument with no distributional assumption whatsoever, which is why it is the only relation here that survives the rest of the page; that a passive order's fill probability is $(\mu/(\mu+\nu))^{Q}$, matched in twenty cells, halving every $7.3$ units of queue at $\nu/\mu=0.10$ while the conditional wait grows linearly, so forty units deep buys $0.02$ of the probability for $40.04$ times the wait; and that Pollaczek–Khinchine makes the wait $\lambda\mathbb{E}[S^{2}]/(2(1-\rho))$, so four laws with mean $1.000$ and utilization $0.80$ produce waits of $1.995$, $3.998$, $10.600$ and $19.251$ against predictions of $2.000$, $4.001$, $10.819$ and $19.207$.

The symmetry with the previous page is that both are about waiting and they measure it against opposite references. [Hitting and First-Passage Times](04-hitting-and-first-passage-times.md) fixes a level and asks for the time, so its randomness is in the clock and its parameters are a drift and a volatility. This page fixes a rate and asks for a length, so its randomness is in the count and its parameter is a ratio of rates — and the reason the two do not simply translate into one another is that a queue has a boundary at zero which a price does not, so utilization enters hyperbolically where drift entered linearly. What both have taken for granted is the arrival process itself. Every rate on this page was constant and every gap independent, which is the definition of the Poisson process, and it is the assumption that order flow most visibly violates. That is [Order Arrival Processes](06-order-arrival-processes.md).

**A queue is described by an average and governed by a variance, so the number that explains a delay is never the number that was used to plan for it.**
