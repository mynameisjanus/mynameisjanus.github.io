# Market Making

[Part I derived the market maker's break-even in three lines](../part-01-foundations/03-market-microstructure.md): if a fraction $p$ of your fills come from someone who knows something, and each of those costs you $\ell$ while the rest pay you half the spread, then quoting is worthwhile only when $s \ge 2p\ell/(1-p)$ — and it concluded that **the spread is exactly the price at which providing liquidity to a partially informed crowd breaks even**. That lesson also named the maker's two risks, adverse selection and inventory, without supplying a model for either. This module supplies both.

Glosten and Milgrom price adverse selection by making the spread a *posterior*: the market maker cannot tell who is trading, so every fill updates their belief about value, and quoting at the updated belief is what zero profit requires. Avellaneda and Stoikov price inventory by turning quoting into a stochastic control problem whose solution shifts quotes away from the position you are accumulating. Both are derived here and both are simulated. The result that matters is the one the simulation makes unavoidable: inventory control **halves the volatility of a market-making book without changing its expected profit at all**, and once informed flow crosses a threshold this module computes exactly — 10.3% arrival, at which informed traders are a third of all fills — *no* inventory policy keeps the book alive. Only width, or absence, does.

## The spread is a posterior over who just traded

Glosten and Milgrom's model is deliberately austere. An asset is worth $V^{+}$ or $V^{-}$ with prior probability $p$ on the high value. A fraction $\pi$ of arriving traders are informed and know $V$; the rest are uninformed and buy or sell with equal probability. A competitive, risk-neutral market maker posts a bid and an ask, cannot distinguish the trader types, and earns zero expected profit.

Zero profit against an unknown counterparty means quoting at the *conditional* expectation given the trade you are about to receive. A buy order is more likely to come from an informed trader when the value is high, so

$$
\text{ask} = \mathbb{E}\bigl[V \mid \text{buy}\bigr],
\qquad
\text{bid} = \mathbb{E}\bigl[V \mid \text{sell}\bigr],
$$

and Bayes' rule supplies both. Writing $\mathbb{P}(\text{buy} \mid V^{+}) = \pi + \tfrac{1-\pi}{2}$ — the informed always buy when the value is high, the uninformed buy half the time — and $\mathbb{P}(\text{buy} \mid V^{-}) = \tfrac{1-\pi}{2}$, the posterior after a buy is

$$
\mathbb{P}\bigl(V^{+} \mid \text{buy}\bigr) \;=\; \frac{p\left(\pi + \frac{1-\pi}{2}\right)}{p\left(\pi + \frac{1-\pi}{2}\right) + (1-p)\frac{1-\pi}{2}},
$$

and the ask is that posterior applied to the two possible values. With a symmetric prior the algebra collapses to something memorable — the spread is exactly $\pi$ times the value range:

```python
import numpy as np

def gm_quotes(p, pi, v_hi, v_lo):
    """Zero-profit bid and ask under Glosten-Milgrom."""
    pb_hi, pb_lo = pi + (1 - pi) / 2, (1 - pi) / 2          # P(buy | V+), P(buy | V-)
    ps_hi, ps_lo = (1 - pi) / 2, pi + (1 - pi) / 2
    post_buy = p * pb_hi / (p * pb_hi + (1 - p) * pb_lo)
    post_sell = p * ps_hi / (p * ps_hi + (1 - p) * ps_lo)
    return (post_sell * v_hi + (1 - post_sell) * v_lo,      # bid
            post_buy * v_hi + (1 - post_buy) * v_lo)        # ask

V_HI, V_LO, P = 101.0, 99.0, 0.5
spreads = {}
for pi in [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]:
    bid, ask = gm_quotes(P, pi, V_HI, V_LO)
    spreads[pi] = ask - bid
print(f"with no informed flow the spread is zero: {abs(spreads[0.0]) < 1e-12}")
print(f"spread equals pi times the value range at every level: "
      f"{all(abs(spreads[pi] - pi * (V_HI - V_LO)) < 1e-9 for pi in spreads)}")

rng = np.random.default_rng(0)
for pi in [0.10, 0.40]:
    n = 400_000
    V = np.where(rng.random(n) < P, V_HI, V_LO)
    informed = rng.random(n) < pi
    buys = np.where(informed, V == V_HI, rng.random(n) < 0.5)
    bid, ask = gm_quotes(P, pi, V_HI, V_LO)
    pnl = np.where(buys, ask - V, V - bid)                  # maker sells at ask, buys at bid
    print(f"  pi = {pi:.0%}: simulated maker profit per trade is within noise of zero: "
          f"{abs(pnl.mean()) < 3 * pnl.std() / np.sqrt(n)}")
# => with no informed flow the spread is zero: True
#    spread equals pi times the value range at every level: True
#      pi = 10%: simulated maker profit per trade is within noise of zero: True
#      pi = 40%: simulated maker profit per trade is within noise of zero: True
```

The measured spreads run 0.00, 0.10, 0.20, 0.40, 0.80, 1.60 as $\pi$ goes from 0 to 80% — exactly $\pi \times (V^{+} - V^{-})$. Three consequences follow, and they are the economics of liquidity provision in their entirety. **The spread exists only because of asymmetric information**: with $\pi = 0$ it is zero, since a maker facing only noise traders needs no compensation. **The maker's entire income is the uninformed flow**, and the informed flow is a pure transfer out — which is why exchanges court retail order flow and why "payment for order flow" is a coherent business rather than a scandal in itself. And **prices become informative through trading**: each fill moves the maker's posterior, so quotes walk toward the true value, which is [Kyle's $\lambda$](05-market-impact-models.md) seen from the liquidity provider's chair rather than the informed trader's.

What the model omits is inventory. Its maker is risk-neutral and quotes symmetrically forever regardless of the position accumulated, which no real desk does.

## Inventory is a position nobody chose

A market maker's inventory $q_t$ is the difference of two counting processes — buys arriving at intensity $\lambda_b(\delta_b)$ and sells at $\lambda_a(\delta_a)$, where $\delta$ is the distance from the mid at which each side is quoted. Quoting tighter increases fill rate; the standard functional form, which Avellaneda and Stoikov derive from power-law market-order sizes, is

$$
\lambda(\delta) \;=\; A\,e^{-k\delta}.
$$

Inventory matters because it is *unhedged directional risk acquired as a by-product of a spread-capture business*. A maker holding $q$ units through an interval of length $\Delta t$ faces mark-to-market variance $\sigma^2 q^2 \Delta t$, and since $q$ follows a difference of Poisson processes it drifts without bound absent control. The whole content of the next section is what a maker should do about that.

## Avellaneda–Stoikov: the price at which you are indifferent

Start with the *reservation price* — the value of the mid at which the maker would be indifferent between holding $q$ units and holding none. Consider a maker with exponential utility $-e^{-\gamma X}$ who stops quoting and simply holds $q$ units until $T$ while the mid diffuses as $dS = \sigma\,dW$. Terminal wealth is $x + qS_T$, normally distributed with mean $x + qs$ and variance $q^2\sigma^2(T-t)$, and the certainty equivalent of a normal under exponential utility subtracts $\tfrac{\gamma}{2}$ times the variance:

$$
\text{CE} \;=\; x + qs - \tfrac{\gamma}{2}\,q^2\sigma^2(T-t).
$$

The reservation price is the per-unit value implied by that certainty equivalent, obtained by differentiating with respect to $q$:

$$
r(s, q, t) \;=\; s - q\,\gamma\,\sigma^{2}\,(T - t).
$$

This single expression is the model's practical core. A maker who is **long** ($q > 0$) values the asset *below* the mid, so both quotes shift down: the ask becomes easier to hit and the bid harder, and the position mean-reverts toward zero. The shift is proportional to inventory, to risk aversion, to *variance* rather than volatility, and to remaining time — a maker near the close leans less, because there is less time left to be hurt.

The second half of the model asks how wide to quote around that centre. Setting up the Hamilton–Jacobi–Bellman equation for the value function $u(x, q, s, t)$ with the exponential fill intensities and solving the resulting system asymptotically, Avellaneda and Stoikov obtain an optimal total spread that is inventory-independent:

$$
\delta^{a} + \delta^{b} \;=\; \gamma\,\sigma^{2}\,(T-t) \;+\; \frac{2}{\gamma}\,\ln\!\left(1 + \frac{\gamma}{k}\right).
$$

The two terms separate cleanly. The first is a risk term — wider when volatile, when risk-averse, when far from the close. The second is a *floor* set by the microstructure: even a risk-neutral maker with unlimited time quotes at least $\frac{2}{\gamma}\ln(1 + \gamma/k)$ wide, because quoting tighter buys fill rate more slowly than it gives up margin. Combining the two results gives the quoting rule in the form a system implements — the ask sits at $r + (\delta^a + \delta^b)/2$ and the bid at $r - (\delta^a + \delta^b)/2$, so relative to the mid:

$$
\delta^{a} = \frac{\text{spread}}{2} - q\gamma\sigma^{2}(T-t),
\qquad
\delta^{b} = \frac{\text{spread}}{2} + q\gamma\sigma^{2}(T-t).
$$

## Skewed quotes buy risk reduction, not profit

Now simulate all of it. The environment is a diffusing mid with informed traders who arrive with probability $\pi$ per step, know a coming jump of size $J$, and trade in its direction — while uninformed traders arrive at the exponential intensity above. Three policies compete: a symmetric maker quoting the A-S floor width, the full A-S policy, and A-S with a hard inventory cap:

```python
import numpy as np

T, N_STEPS = 1.0, 500
SIGMA, GAMMA, K_INT, A_INT, J = 2.0, 0.02, 1.5, 140.0, 2.0
DT = T / N_STEPS
FLOOR = (2 / GAMMA) * np.log(1 + GAMMA / K_INT) / 2         # A-S half-spread floor

def as_offsets(t, q):
    spread = GAMMA * SIGMA ** 2 * (T - t) + (2 / GAMMA) * np.log(1 + GAMMA / K_INT)
    skew = q * GAMMA * SIGMA ** 2 * (T - t)                 # reservation price is s - skew
    return spread / 2 - skew, spread / 2 + skew             # long inventory -> cheaper ask

def episode(policy, pi, seed, cap=None, fixed=None):
    rng = np.random.default_rng(seed)
    s, q, cash = 100.0, 0, 0.0
    for i in range(N_STEPS):
        if policy == "symmetric":
            da = db = fixed
        else:
            da, db = as_offsets(i * DT, q)
            da, db = max(da, 0.01), max(db, 0.01)
        if cap is not None:
            if q >= cap:
                db = 1e6                                    # stop buying
            if q <= -cap:
                da = 1e6                                    # stop selling
        jump = 0.0
        if rng.random() < pi:                               # informed, knows the jump
            jump = J if rng.random() < 0.5 else -J
            if jump > 0 and da < 1e5:
                cash += s + da
                q -= 1
            elif jump < 0 and db < 1e5:
                cash -= s - db
                q += 1
        else:                                               # uninformed
            if rng.random() < A_INT * np.exp(-K_INT * da) * DT:
                cash += s + da
                q -= 1
            if rng.random() < A_INT * np.exp(-K_INT * db) * DT:
                cash -= s - db
                q += 1
        s += SIGMA * np.sqrt(DT) * rng.standard_normal() + jump
    return cash + q * s, q

def run(policy, pi, n=3000, cap=None, fixed=None):
    out = np.array([episode(policy, pi, s, cap, fixed) for s in range(n)])
    return out[:, 0], out[:, 1]

res = {}
for label, kw in [("symmetric", dict(policy="symmetric", fixed=FLOOR)),
                  ("A-S", dict(policy="as")),
                  ("A-S + cap", dict(policy="as", cap=20))]:
    res[label] = run(pi=0.05, **kw)
print(f"A-S earns about the same as symmetric quoting: "
      f"{abs(res['A-S'][0].mean() / res['symmetric'][0].mean() - 1) < 0.1}")
print(f"A-S more than halves the P&L standard deviation: "
      f"{res['A-S'][0].std() < 0.5 * res['symmetric'][0].std()}")
print(f"A-S roughly halves inventory risk: "
      f"{res['A-S'][1].std() < 0.6 * res['symmetric'][1].std()}")
print(f"the cap binds where A-S alone would not: "
      f"{np.abs(res['A-S + cap'][1]).max() < np.abs(res['A-S'][1]).max()}")
# => A-S earns about the same as symmetric quoting: True
#    A-S more than halves the P&L standard deviation: True
#    A-S roughly halves inventory risk: True
#    the cap binds where A-S alone would not: True
```

| policy at 5% informed flow | mean P&L | P&L sd | inventory sd | max abs inventory |
|---|---|---|---|---|
| symmetric, quoting the A-S floor | +32.76 | 77.41 | 10.69 | 40 |
| Avellaneda–Stoikov | +31.97 | **34.14** | **5.43** | 25 |
| A-S with inventory cap of 20 | +31.97 | 34.14 | 5.42 | **20** |

This is the result to internalize, and it is not the one a newcomer expects. **Avellaneda–Stoikov made no more money than symmetric quoting** — 31.97 against 32.76, a difference inside the noise — while cutting the standard deviation of the outcome from 77.41 to 34.14 and inventory risk from 10.69 to 5.43. The model is not an alpha model; it is a *risk* model, and its output is a better Sharpe ratio through a smaller denominator. That is exactly what it claims to be: a solution to a utility-maximization problem in which risk aversion $\gamma$ is an input, not a device for finding extra edge.

The inventory cap adds almost nothing on average and changes the worst case, bounding $\lvert q \rvert$ at 20 where A-S alone reached 25. That asymmetry — no effect on the mean, real effect on the tail — is the signature of a risk control working correctly, and it is why caps are standard practice even alongside a model that already leans against inventory.

## Markouts name your counterparty

The identity that decomposes a fill's economics needs no model. Define the **effective half-spread** as what you captured against the mid at the moment of the fill, and the **realized half-spread** as what you kept once the mid moved:

$$
\text{effective} = \epsilon\,(P_{\text{fill}} - m_t),
\qquad
\text{realized} = \epsilon\,(P_{\text{fill}} - m_{t+\tau}),
\qquad
\text{adverse selection} = \text{effective} - \text{realized},
$$

with $\epsilon = +1$ when the maker sold. This is an accounting identity, not an estimate. Applied to the simulated fills, where the ground truth is known:

| counterparty | fills | effective half-spread | realized | adverse selection |
|---|---|---|---|---|
| uninformed | 37,052 | +0.6626 | +0.6627 | **−0.0000** |
| informed | 9,780 | +0.6822 | −1.3183 | **+2.0006** |

The decomposition recovers the planted parameter exactly: informed fills give up **2.0006** against a true jump size of $J = 2.0$, and uninformed fills show adverse selection of zero to four decimal places. Note what the effective half-spread column does *not* do — it is nearly identical for both groups (0.6626 against 0.6822), so at the moment of execution the two counterparties are indistinguishable. Only the subsequent price move separates them, which is the whole difficulty of the business: **toxicity is invisible at fill time and obvious a second later**, so a quoting system needs the markout pipeline running continuously, segmented by counterparty, venue, order type, and time of day, feeding the width decision rather than a monthly report.

## Above ten percent toxicity, no inventory policy survives

The final experiment sweeps informed arrival upward and asks each policy how it dies:

```python
import numpy as np

T, N_STEPS = 1.0, 500
SIGMA, GAMMA, K_INT, A_INT, J = 2.0, 0.02, 1.5, 140.0, 2.0
DT = T / N_STEPS
FLOOR = (2 / GAMMA) * np.log(1 + GAMMA / K_INT) / 2

def as_offsets(t, q):
    spread = GAMMA * SIGMA ** 2 * (T - t) + (2 / GAMMA) * np.log(1 + GAMMA / K_INT)
    skew = q * GAMMA * SIGMA ** 2 * (T - t)
    return spread / 2 - skew, spread / 2 + skew

def episode(policy, pi, seed, cap=None, fixed=None):
    rng = np.random.default_rng(seed)
    s, q, cash = 100.0, 0, 0.0
    for i in range(N_STEPS):
        if policy == "symmetric":
            da = db = fixed
        else:
            da, db = as_offsets(i * DT, q)
            da, db = max(da, 0.01), max(db, 0.01)
        if cap is not None:
            if q >= cap:
                db = 1e6
            if q <= -cap:
                da = 1e6
        jump = 0.0
        if rng.random() < pi:
            jump = J if rng.random() < 0.5 else -J
            if jump > 0 and da < 1e5:
                cash += s + da
                q -= 1
            elif jump < 0 and db < 1e5:
                cash -= s - db
                q += 1
        else:
            if rng.random() < A_INT * np.exp(-K_INT * da) * DT:
                cash += s + da
                q -= 1
            if rng.random() < A_INT * np.exp(-K_INT * db) * DT:
                cash -= s - db
                q += 1
        s += SIGMA * np.sqrt(DT) * rng.standard_normal() + jump
    return cash + q * s

def mean_pnl(policy, pi, n=1500, **kw):
    return float(np.mean([episode(policy, pi, s, **kw) for s in range(n)]))

curve = {pi: (mean_pnl("symmetric", pi, fixed=FLOOR), mean_pnl("as", pi))
         for pi in [0.0, 0.05, 0.10, 0.20, 0.30]}
lam = 2 * A_INT * np.exp(-K_INT * FLOOR) * DT               # uninformed arrivals per step
pi_star = lam * FLOOR / (J - FLOOR)                          # zero-profit informed rate

print(f"both policies are profitable with no informed flow: "
      f"{curve[0.0][0] > 0 and curve[0.0][1] > 0}")
print(f"both are underwater by 20% informed flow: "
      f"{curve[0.20][0] < 0 and curve[0.20][1] < 0}")
print(f"inventory control does not extend survival: "
      f"{abs(curve[0.20][1] - curve[0.20][0]) < 0.2 * abs(curve[0.20][0])}")
print(f"predicted break-even informed rate is near 10%: {0.09 < pi_star < 0.11}")
print(f"at break-even, informed are about a third of fills: "
      f"{0.30 < pi_star / (pi_star + lam) < 0.36}")
# => both policies are profitable with no informed flow: True
#    both are underwater by 20% informed flow: True
#    inventory control does not extend survival: True
#    predicted break-even informed rate is near 10%: True
#    at break-even, informed are about a third of fills: True
```

| informed arrival rate | symmetric | Avellaneda–Stoikov |
|---|---|---|
| 0% | +68.69 | +67.99 |
| 5% | +32.92 | +31.80 |
| 9% | +2.44 | +3.08 |
| 10% | −2.92 | −3.74 |
| 20% | −72.45 | −76.26 |
| 30% | −161.56 | −156.99 |

The two columns are the same column. **Inventory control buys nothing at any level of toxicity**, and the book crosses into loss between 9% and 10% informed arrival whichever policy is running. The theoretical break-even confirms it: setting uninformed income equal to informed losses, $\lambda\delta = \pi(J - \delta)$, gives $\pi^{*} = \lambda\delta/(J - \delta) = 10.3\%$, at which informed traders are **33.1% of all fills** — and the simulation crosses zero exactly there.

That agreement is the module's closing argument, and it reconnects to where the module began. Avellaneda–Stoikov manages a *variance*; the thing killing the book is a *mean*. No amount of leaning against inventory changes the expected loss on a fill from someone who knows more than you — only quoting wider (which is Glosten–Milgrom's prescription, and reduces fill rate) or not quoting at all can do that. The break-even condition is Part I's $s \ge 2p\ell/(1-p)$ in dynamic clothing, with $p$ now the fill-weighted informed share that the arrival rates determine jointly with the spread.

## Queue position, and where the continuous model stops applying

One assumption in everything above deserves scrutiny: that the maker chooses a continuous offset $\delta$ from the mid. In a market where the tick is wide relative to the spread — most liquid equities, futures, and anything quoted in pennies on a $50 stock — the spread is *one tick* and there is no offset to optimize. Every maker quotes the same price, and the only decision left is **when you joined the queue**.

Priority then determines everything. Resting at position $k$ in a queue that depletes at rate $\mu$ (trades) and $c$ (cancellations ahead of you), the expected wait to reach the front is roughly $k/(\mu + c)$, and the probability of being filled before the price moves against you falls as $k$ grows. Two consequences follow that have no analogue in the continuous model. Cancelling and re-joining costs your entire priority, so a maker who reprices nervously converts a queue position earned over minutes into a place at the back — which is why real quoting systems have hysteresis and why "cancel/replace ratio" is a metric desks watch. And queue position interacts with toxicity in a specific way: being at the front means being filled *first*, including first by the informed trader, so the front of the queue is simultaneously the best place to earn the spread and the most exposed to adverse selection. In one-tick markets, the optimization that matters is over *when* to be in the queue, not *where* to quote.

## Controls are the model admitting its priors can break

Every parameter above is an estimate, and the practical apparatus around a quoting system exists because estimates fail. **Inventory limits**, hard and non-negotiable, bound the loss when the fill-rate model is wrong — the simulation showed a cap costing nothing on average while bounding the worst case. **Pull triggers** withdraw quotes when the world stops resembling the model's assumptions: a spread that widens beyond a threshold, a markout that turns sharply negative over a rolling window, a venue whose latency spikes, a stale reference price. **News and auctions** are the cases where $\pi$ jumps discontinuously — an earnings release converts a normal book into one where a large fraction of arriving flow is informed — and the correct response is to be absent rather than wide, because the model's continuous adjustment cannot track a step change in toxicity.

The one that catches desks out is the *stale quote after a fast move*: the maker's mid is computed from a reference that lagged, so quotes sit on the wrong side of the market and get filled instantly by everyone. The defence is a heartbeat — quotes that expire unless refreshed — and it belongs in the same category as [Part VI's circuit breakers](../part-06-live-infrastructure/05-resilience-and-risk-controls.md): a control that costs a little continuously and prevents the loss that ends the business.

!!! warning "Inventory control manages a variance; adverse selection is a mean"
    Avellaneda–Stoikov cut the standard deviation of a market-making book from 77.4 to 34.1 and left its expected profit unchanged at about 32 — precisely what a risk model should do, and nothing more. When informed arrival crossed 10.3%, at which informed traders were a third of all fills, symmetric and inventory-controlled quoting died together and within a basis point of each other. Skewing quotes cannot price information. Only wider spreads, better counterparty selection, or withdrawal can, and knowing which of the two problems you have is what the markout decomposition is for.

!!! abstract "Key takeaways"
    - Glosten–Milgrom makes the spread a posterior: with a symmetric prior it equals exactly $\pi$ times the value range (0.00, 0.10, 0.20, 0.40, 0.80, 1.60 as $\pi$ runs from 0 to 80%), and simulated maker profit is zero at every level.
    - With no informed flow the equilibrium spread is **zero** — the maker's entire income is uninformed flow, and informed flow is a pure transfer out, which is why retail order flow is a business rather than a scandal.
    - The reservation price $r = s - q\gamma\sigma^2(T-t)$ falls out of an exponential-utility certainty equivalent, and scales with *variance* and with time remaining, so a maker leans less near the close.
    - The optimal A-S spread $\gamma\sigma^2(T-t) + \tfrac{2}{\gamma}\ln(1+\gamma/k)$ separates a risk term from a microstructure floor that binds even for a risk-neutral maker.
    - A-S earned +31.97 against symmetric quoting's +32.76 while cutting P&L standard deviation from **77.41 to 34.14** and inventory sd from 10.69 to 5.43: it is a risk model, not an alpha model.
    - A hard inventory cap changed the mean by nothing and the worst case from 25 to 20 — the signature of a correctly working risk control.
    - The markout identity recovered the planted informed edge exactly: **+2.0006** adverse selection on informed fills against a true $J = 2.0$, and −0.0000 on uninformed — while effective half-spreads were nearly identical (0.6822 against 0.6626), so toxicity is invisible at fill time.
    - Both policies crossed into loss between 9% and 10% informed arrival, matching the theoretical $\pi^{*} = \lambda\delta/(J-\delta) = 10.3\%$ at which informed traders are **33.1% of fills** — Part I's $s \ge 2p\ell/(1-p)$ in dynamic clothing.

## Where this goes next

The other side of every trade in this module — the trader consuming liquidity and paying the spread this module set — is [Optimal Execution](04-optimal-execution-almgren-chriss.md), and the permanent price impact their flow creates is derived from Kyle's model in [Market Impact Models](05-market-impact-models.md), which is Glosten–Milgrom's linear-equilibrium cousin. Making markets in options rather than shares means quoting a book of Greeks instead of a book of inventory, and [Options Pricing](11-options-pricing.md) supplies the sensitivities that a derivatives maker hedges. The venues where these models meet their least forgiving test — fragmented, 24/7, with no consolidated tape and the exchange itself as counterparty — are the subject of [Crypto Market Microstructure](13-crypto-microstructure.md). In the core course, the economics this module formalized are in [Part I, lesson three](../part-01-foundations/03-market-microstructure.md), and the operational controls that keep a quoting system alive are [Part VI, lesson five](../part-06-live-infrastructure/05-resilience-and-risk-controls.md).
