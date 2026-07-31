# Portfolio Optimization and Correlation

[Risk Parity, Diversification, and Factors](03-risk-parity-diversification-factors.md) compared three allocations — equal dollars, equal volatility, equal risk — that share one deliberate abstention: none of them uses a forecast of returns. That abstention looks like timidity. Markowitz solved this problem in 1952, the solution is a line of linear algebra, and every finance curriculum teaches the efficient frontier as the answer to exactly the question this part has been circling.

This lesson stops abstaining, runs the optimizer honestly out of sample, and reports the most comprehensively negative result in Part VIII. Mean-variance optimization loses to naive equal weighting in four of five configurations. The standard repair — shrinking the covariance matrix, for which Ledoit and Wolf provided the canonical estimator — makes it *worse*, and the measurement of why is a single number. What does rescue the optimizer is the crudest tool available, a set of constraints with no statistical theory behind them at all. Along the way the lesson measures what happens to the correlations the previous lesson relied on when markets fall, which is the reason all of this is hard.

## Mean-variance and the error it maximizes

The formulation is exact and unarguable: given expected returns $\mu$ and covariance $\Sigma$, the portfolio maximizing return per unit of variance is $w \propto \Sigma^{-1}\mu$. Every term is estimated from a finite sample. The horse race below fits eight allocators on a rolling two-year window, holds each for a month, and never lets any of them see the future — the only test that matters:

```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
U9 = [f"r_{s}" for s in SECT]
U14 = U9 + ["r_TLT", "r_GLD", "r_IWM", "r_EFA", "r_EEM"]

def hrp(R):                                   # Lopez de Prado: cluster, then bisect
    C, cov = R.corr().values, R.cov().values
    d = np.sqrt(np.clip((1 - C) / 2, 0, None))
    np.fill_diagonal(d, 0.0)
    order = dendrogram(linkage(squareform(d, checks=False), "single"),
                       no_plot=True)["leaves"]
    w, clusters = np.ones(len(order)), [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            a, b = c[:len(c) // 2], c[len(c) // 2:]
            va = 1 / np.diag(cov[np.ix_(a, a)]); va /= va.sum()
            vb = 1 / np.diag(cov[np.ix_(b, b)]); vb /= vb.sum()
            sa, sb = va @ cov[np.ix_(a, a)] @ va, vb @ cov[np.ix_(b, b)] @ vb
            w[a] *= 1 - sa / (sa + sb)
            w[b] *= sa / (sa + sb)
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()

def weights(R, kind):
    mu, n = R.mean().values * 252, R.shape[1]
    S = LedoitWolf().fit(R.values).covariance_ * 252 if kind.startswith("LW") \
        else R.cov().values * 252
    if kind == "noshort-mvo":
        r = minimize(lambda w: -(w @ mu) + 2.0 * (w @ S @ w), np.ones(n) / n,
                     method="SLSQP", bounds=[(0, 1)] * n,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                     options={"maxiter": 300, "ftol": 1e-12})
        return r.x
    if kind == "HRP":
        return hrp(R)
    raw = (np.linalg.solve(S, np.ones(n)) if "minvar" in kind else
           np.linalg.solve(S, mu) if "mvo" in kind else
           np.ones(n) if kind == "1/N" else 1 / np.sqrt(np.diag(S)))
    return raw / raw.sum()

M = ["mvo", "LW-mvo", "1/N", "invvol", "minvar", "LW-minvar", "HRP", "noshort-mvo"]

def race(cols, T=504, hold=21):
    R = p8[cols].dropna()
    rets = {m: [] for m in M}; gross = {m: [] for m in M}
    turn = {m: [] for m in M}; prev = {}
    idx = list(range(T, len(R) - hold, hold))
    for i in idx:
        tr, te = R.iloc[i - T:i], R.iloc[i:i + hold]
        for m in M:
            w = weights(tr, m)
            gross[m].append(np.abs(w).sum())
            if m in prev:
                turn[m].append(np.abs(w - prev[m]).sum())
            prev[m] = w
            rets[m].append(te.values @ w)
    out = {}
    for m in M:
        s = pd.Series(np.concatenate(rets[m]))
        e = np.exp(s.cumsum())
        out[m] = (np.sqrt(252) * s.mean() / s.std(), np.sqrt(252) * s.std(),
                  (e / e.cummax() - 1).min(), np.mean(gross[m]),
                  np.mean(turn[m]) * 252 / hold)
    return out, len(idx)

res, nreb = race(U9)
print(f"9 sectors, train 504d / hold 21d, {nreb} rebalances")
print("  method        OOS Sharpe   ann vol      maxDD   mean gross   turnover/yr")
for m in M:
    s, v, d, g, t = res[m]
    print(f"  {m:13s} {s:10.3f} {v:9.1%} {d:10.1%} {g:12.2f} {t:13.1f}")

print("\nrobustness -- OOS Sharpe across universes and estimation windows")
print("  configuration      " + " ".join(f"{m:>11s}" for m in M))
for lab, cols, T in [("9 sectors T=504", U9, 504), ("9 sectors T=126", U9, 126),
                     ("9 sectors T=63", U9, 63), ("14 assets T=504", U14, 504),
                     ("14 assets T=126", U14, 126)]:
    r, _ = race(cols, T=T)
    print(f"  {lab:18s} " + " ".join(f"{r[m][0]:11.3f}" for m in M))
# => 9 sectors, train 504d / hold 21d, 281 rebalances
#      method        OOS Sharpe   ann vol      maxDD   mean gross   turnover/yr
#      mvo                0.377    293.9%     -99.9%        11.81         137.0
#      LW-mvo             0.115    262.7%    -100.0%        10.01          94.5
#      1/N                0.450     18.5%     -54.5%         1.00           0.0
#      invvol             0.473     17.5%     -51.2%         1.00           0.1
#      minvar             0.546     13.9%     -33.8%         1.80           2.0
#      LW-minvar          0.550     13.8%     -34.8%         1.66           1.8
#      HRP                0.506     16.4%     -46.6%         1.00           0.8
#      noshort-mvo        0.456     19.7%     -44.6%         1.00           5.1
#
#    robustness -- OOS Sharpe across universes and estimation windows
#      configuration              mvo      LW-mvo         1/N      invvol      minvar   LW-minvar         HRP noshort-mvo
#      9 sectors T=504          0.377       0.115       0.450       0.473       0.546       0.550       0.506       0.456
#      9 sectors T=126          0.142       0.292       0.410       0.435       0.543       0.524       0.463       0.377
#      9 sectors T=63           0.044      -0.160       0.400       0.426       0.535       0.510       0.461       0.160
#      14 assets T=504          0.464      -0.041       0.434       0.488       0.892       0.882       0.680       0.425
#      14 assets T=126         -0.015       0.113       0.482       0.540       0.879       0.832       0.748       0.304
```

Read the volatility column before the Sharpe column. Mean-variance optimization, fed two years of sector returns and asked for the tangency portfolio, produced a book running at **293.9% annualized volatility** with a maximum drawdown of **−99.9%**, an average gross exposure of **11.81×**, and turnover of **137 times a year**. Nothing in the specification asked for leverage; the optimizer generated it by taking enormous offsetting long and short positions in sectors whose estimated means differed by amounts indistinguishable from noise. That is the error-maximization property, and it is not a subtlety at the margin: it is the dominant behavior of the method.

Its Sharpe of 0.377 loses to **1/N's 0.450** — a portfolio that requires no estimation, no linear algebra, and no rebalancing at all. The robustness sweep says this is not a quirk of one window. Across five configurations MVO ranges from **−0.015 to 0.464** while 1/N ranges from 0.400 to 0.482, and MVO beats it exactly once, on the widest universe with the longest window, where it does so at a volatility that would have liquidated the account long before the Sharpe was collected. Shortening the estimation window makes it worse in the way the theory predicts: at T = 63, with seven observations per asset, MVO collapses to 0.044.

The two columns worth carrying forward are the ones nobody quotes. **Minimum variance — the same optimizer with $\mu$ deleted, $w \propto \Sigma^{-1}\mathbf{1}$ — wins every configuration**, from 0.535 to 0.892. Deleting the expected-return vector entirely improves the result by more than any refinement to it does. That localizes the damage precisely: the machinery is fine, the covariance matrix is estimable, and $\hat\mu$ is the poison. It is the natural conclusion of what this course has measured repeatedly — a Sharpe ratio carries a standard error of roughly ±0.2 over two decades, so a two-year estimate of an expected return is barely distinguishable from zero, and $\Sigma^{-1}$ amplifies exactly those differences.

## Is it the sample size? Shrinkage says no

The standard remedy is to shrink the sample covariance toward a structured target, with the Ledoit–Wolf estimator choosing the intensity analytically. If MVO fails because $\hat\Sigma$ is noisy, this should fix it. In the table above it does the opposite: **LW-MVO scores 0.115 against plain MVO's 0.377**, and across the five configurations it loses to 1/N in *all five* and to plain MVO in three. One number explains why:

```python
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
U9 = [f"r_{s}" for s in SECT]
U14 = U9 + ["r_TLT", "r_GLD", "r_IWM", "r_EFA", "r_EEM"]

print("  how much does Ledoit-Wolf actually shrink?")
print("  configuration        T/N   mean intensity   range")
for lab, cols, T in [("9 sectors T=504", U9, 504), ("9 sectors T=126", U9, 126),
                     ("9 sectors T=63", U9, 63), ("14 assets T=504", U14, 504)]:
    R = p8[cols].dropna()
    sh = [LedoitWolf().fit(R.iloc[i - T:i].values).shrinkage_
          for i in range(T, len(R) - 21, 21)]
    print(f"  {lab:18s} {T / len(cols):5.0f}   {np.mean(sh):14.4f}   "
          f"{min(sh):.4f} to {max(sh):.4f}")
# =>   how much does Ledoit-Wolf actually shrink?
#      configuration        T/N   mean intensity   range
#      9 sectors T=504       56           0.0185   0.0090 to 0.0775
#      9 sectors T=126       14           0.0542   0.0227 to 0.1782
#      9 sectors T=63         7           0.0969   0.0287 to 0.4303
#      14 assets T=504       36           0.0192   0.0097 to 0.0644
```

At 504 observations for nine assets the estimator shrinks by **1.85%**. It is not being timid — it is being correct: with 56 observations per asset the sample covariance is already well-conditioned, and the optimal shrinkage toward a scalar target really is almost nothing. Ledoit–Wolf is doing its job perfectly and the job is nearly vacuous here. The intensity rises to 9.69% when the window shrinks to seven observations per asset, exactly as the theory says it should, and even there it cannot rescue MVO (Sharpe −0.160).

So the diagnosis sharpens into a rule. **Shrinking $\hat\Sigma$ cannot fix mean-variance, because the error mean-variance maximizes lives in $\hat\mu$.** A 2% adjustment to the covariance matrix cannot offset an expected-return vector whose signal-to-noise ratio is near zero and which then gets multiplied by an inverse covariance matrix. The reason shrinkage has a good reputation is that it genuinely helps *minimum variance*, where the covariance matrix is the only input — and even there the table above shows the improvement is 0.550 against 0.546, a fourth decimal place. When practitioners report that shrinkage transformed their optimizer, they are usually reporting a regime where $T/N$ was near one, and it is worth asking which.

## Correlations go to one, and how far

Every allocation in this part is a function of a correlation matrix, and [the previous lesson](03-risk-parity-diversification-factors.md) showed that matrix is not stationary across regimes. The sharper problem is that its instability is *conditional on the market falling*, which means diversification is weakest at the only moment it is being asked to work:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
R9 = p8[[f"r_{s}" for s in SECT]].dropna()

def stats(df):
    C = df.corr().values
    ev = np.linalg.eigvalsh(C)[::-1]
    r = C[np.triu_indices_from(C, 1)].mean()
    return r, ev[0] / ev.sum(), 9 / (1 + 8 * r)

print("  regime                   avg pairwise rho   PC1 share   N_eff   SPY ann vol")
for k, (a, b) in {"2017 calm": ("2017-01-01", "2017-12-31"),
                  "2024 calm": ("2024-01-01", "2024-12-31"),
                  "2022 bear": ("2022-01-01", "2022-12-31"),
                  "2008 crisis Sep-Dec": ("2008-09-01", "2008-12-31"),
                  "2020 COVID Feb-Apr": ("2020-02-20", "2020-04-30"),
                  "2011 EU crisis Aug-Oct": ("2011-08-01", "2011-10-31")}.items():
    r, pc1, ne = stats(R9.loc[a:b])
    print(f"  {k:24s} {r:14.3f} {pc1:11.1%} {ne:7.2f} "
          f"{np.sqrt(252) * p8.r_SPY.loc[a:b].std():12.1%}")
r, pc1, ne = stats(R9)
print(f"  {'full sample':24s} {r:14.3f} {pc1:11.1%} {ne:7.2f}")

vals = {}
for d, g in R9.rolling(63).corr().groupby(level=0):
    C = g.values
    if np.isfinite(C).all():
        vals[d] = C[np.triu_indices_from(C, 1)].mean()
rs = pd.Series(vals)
print(f"  rolling 63d: min {rs.min():.3f} ({rs.idxmin():%Y-%m-%d}) -> N_eff "
      f"{9 / (1 + 8 * rs.min()):.2f};  max {rs.max():.3f} ({rs.idxmax():%Y-%m-%d}) "
      f"-> N_eff {9 / (1 + 8 * rs.max()):.2f}")
q = p8.r_SPY.reindex(R9.index)
lo, mid = R9[q <= q.quantile(0.10)], R9[(q > q.quantile(0.10)) & (q < q.quantile(0.90))]
print(f"  conditioned on the market: worst 10% of days rho {stats(lo)[0]:.3f} "
      f"(N_eff {stats(lo)[2]:.2f}), middle 80% rho {stats(mid)[0]:.3f} "
      f"(N_eff {stats(mid)[2]:.2f})")
# =>   regime                   avg pairwise rho   PC1 share   N_eff   SPY ann vol
#      2017 calm                         0.314       43.4%    2.57         6.7%
#      2024 calm                         0.410       49.2%    2.10        12.6%
#      2022 bear                         0.643       69.8%    1.47        24.3%
#      2008 crisis Sep-Dec               0.811       83.3%    1.20        65.2%
#      2020 COVID Feb-Apr                0.878       89.3%    1.12        66.6%
#      2011 EU crisis Aug-Oct            0.907       91.8%    1.09        36.0%
#      full sample                       0.619       66.6%    1.51
#      rolling 63d: min 0.088 (2000-09-25) -> N_eff 5.29;  max 0.922 (2020-03-16) -> N_eff 1.07
#      conditioned on the market: worst 10% of days rho 0.516 (N_eff 1.76), middle 80% rho 0.327 (N_eff 2.49)
```

The cliché is literally true and the magnitude is worth memorizing. Average pairwise sector correlation runs **0.314 in a calm year** and **0.907 in the autumn of 2011**, with the first principal component absorbing 91.8% of all variance. In March 2020 the rolling 63-day average peaked at **0.922**, at which point the nine sectors are **1.07 effective bets**: a portfolio the optimizer believed was diversified across nine positions was, for the duration of the crash, a single position.

The bottom line generalizes the point beyond named crises. Sorting all days by the market's return and computing correlations separately, the worst decile has an average pairwise correlation of **0.516** against the middle 80%'s **0.327** — the effective bet count falls from 2.49 to 1.76 purely as a function of the market falling, with no crisis label required. Diversification is not a constant that occasionally breaks; it is a decreasing function of how badly the day is going. Any risk number computed from a full-sample correlation matrix — including every diversification ratio and effective-bet count in [lesson three](03-risk-parity-diversification-factors.md) — is an average over regimes and therefore an overstatement of what will be available in the regime that matters.

## Constraints are the shrinkage that works

If the problem is that the optimizer trusts noisy estimates too much, the fix is to *forbid* it from acting on them. Position bounds, a long-only requirement, a turnover penalty, a risk-budget penalty — none of these has any statistical justification, and all of them are equivalent to a prior that the estimates are wrong:

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

p8 = pd.read_parquet("data/part8.parquet")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
R = p8[[f"r_{s}" for s in SECT]].dropna()
T, hold, n = 504, 21, len(SECT)

print("  constraint set            OOS Sharpe   turnover/yr   ann vol   solves   failures")
for lab, hi, tp, rb in [("unconstrained (Sigma^-1 mu)", None, 0.0, 0.0),
                        ("long-only, cap 25%", 0.25, 0.0, 0.0),
                        ("+ turnover penalty", 0.25, 0.05, 0.0),
                        ("+ equal-risk penalty", 0.25, 0.05, 5.0)]:
    prev, rets, turn, fails = np.ones(n) / n, [], [], 0
    nsolve = 0
    for i in range(T, len(R) - hold, hold):
        tr, te = R.iloc[i - T:i], R.iloc[i:i + hold]
        mu, S = tr.mean().values * 252, tr.cov().values * 252
        if hi is None:
            w = np.linalg.solve(S, mu)
            w = w / w.sum()
        else:
            def obj(w):
                rc = w * (S @ w) / np.sqrt(w @ S @ w)
                return (-(w @ mu) + 2.0 * (w @ S @ w)
                        + tp * np.sqrt((w - prev) ** 2 + 1e-10).sum()
                        + rb * ((rc - rc.mean()) ** 2).sum())
            r = minimize(obj, prev, method="SLSQP", bounds=[(0, hi)] * n,
                         constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                         options={"maxiter": 300, "ftol": 1e-12})
            fails += (not r.success)
            nsolve += 1
            w = r.x
        turn.append(np.abs(w - prev).sum())
        prev = w
        rets.append(te.values @ w)
    s = pd.Series(np.concatenate(rets))
    print(f"  {lab:27s} {np.sqrt(252) * s.mean() / s.std():+9.3f} "
          f"{np.mean(turn) * 252 / hold:13.1f} {np.sqrt(252) * s.std():9.1%} "
          f"{nsolve:8d} {fails:10d}")
# =>   constraint set            OOS Sharpe   turnover/yr   ann vol   solves   failures
#      unconstrained (Sigma^-1 mu)    +0.377         142.2    293.9%        0          0
#      long-only, cap 25%             +0.444           2.8     17.5%      281          0
#      + turnover penalty             +0.399           0.7     17.9%      281          0
#      + equal-risk penalty           +0.446           0.6     17.4%      281          0
```

Forbidding short positions and capping any single sector at 25% takes the Sharpe from 0.377 to **0.444**, the annualized volatility from 293.9% to **17.5%**, and the turnover from 142 times a year to **2.8**. Nothing was estimated better. The optimizer was simply prevented from expressing its confidence, and its confidence was the problem. This is the Jagannathan–Ma result — that a no-short constraint on a mean-variance problem is mathematically equivalent to shrinking the covariance matrix, with the shrinkage target implied by which constraints bind — and it is why the crude fix outperforms the sophisticated one: the constraint's implied shrinkage is large where Ledoit–Wolf's was 1.85%.

The turnover penalty is an honest disappointment worth keeping. It cuts trading from 2.8 times a year to **0.7** — a quarter of the activity — and costs 0.045 of Sharpe in doing so. Whether that is a good trade depends entirely on the cost model: at Part IV's sector half-spread of 2.2 bp all-in, saving 2.1 turns a year is worth about 5 bp, against a Sharpe cost of 0.045 on a 17.5%-volatility book, or roughly 79 bp of return. Here the penalty is set too aggressively and the Sharpe cost is real. Adding the equal-risk penalty on top recovers it (0.446) while holding turnover at 0.6, which is the configuration to prefer — but the general lesson is that penalties are hyperparameters, they trade one cost against another, and neither direction is free.

On the mechanics: **SLSQP handled every problem in this lesson without a single convergence failure** — 281 solves per configuration, each in single-digit milliseconds, including a non-smooth L1 turnover term smoothed as $\sqrt{(w-w_{\text{prev}})^2 + \epsilon}$ and a non-convex risk-budget penalty. No convex solver is required for portfolio problems at this scale, which matters because a dependency avoided is a dependency that cannot break.

## What to actually use

The horse race ranked eight methods and the ordering is stable enough to act on. **Minimum variance won every configuration** (0.535 to 0.892), which is the strongest practical recommendation this lesson can make and also the most uncomfortable, because it is the method that refuses to use any return forecast at all. **Hierarchical risk parity finished second** among the long-only allocators — 0.506 against 1/N's 0.450 on the base case, and it beat 1/N in all five configurations while losing to minimum variance in all five. HRP replaces matrix inversion with a clustering step and a recursive bisection, so it never inverts $\Sigma$ and is therefore immune to the ill-conditioning that makes MVO explode; it holds gross exposure at exactly 1.00 and turns over 0.8 times a year. It is usually marketed as a replacement for minimum variance, and on this data it is not one — but it is a genuinely better default than equal weighting, at almost no cost in complexity.

The rebalancing question is settled elsewhere and should not be re-litigated here. [Part IV lesson seven](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) swept no-trade bands and found tracking error around an optimum is a *second-order* loss while trading costs are *first-order*, so precision near the optimum is systematically overpriced; [lesson two](02-kelly-vol-targeting-leverage.md) sharpened that into a timing result, showing a 25% drift band and monthly rebalancing spend identical turnover while the band tracks its target more than twice as accurately. Both conclusions apply unchanged to the weights produced here. The composite recommendation for a book like this one: allocate with minimum variance or HRP on a shrunk covariance matrix, constrain positions long-only with a per-name cap, rebalance on drift bands rather than dates, and do not let an expected-return vector anywhere near the optimizer unless it comes from something better than a two-year sample mean.

!!! warning "An optimizer is an amplifier, and what it mostly amplifies is your estimation error"
    Mean-variance optimization is correct. Given the true $\mu$ and $\Sigma$ it produces the best possible portfolio, and no method in the table beats it. Fed two-year estimates it produced 11.81× gross exposure, 293.9% volatility, a −99.9% drawdown, and a Sharpe below a portfolio that requires no estimation whatsoever. The gap between those two sentences is the entire discipline. Before deploying any optimizer, ask what it does when every input is replaced by noise of the same magnitude as the signal — if the answer is "takes enormous offsetting positions", the constraint set is not a detail to be added later, it is the load-bearing part of the design.

!!! abstract "Key takeaways"
    - Out of sample on nine sectors, mean-variance produced **293.9% volatility, −99.9% maximum drawdown, 11.81× gross exposure and 137× annual turnover** for a Sharpe of 0.377 — losing to 1/N's 0.450, which requires no estimation at all.
    - Across five universe-and-window configurations MVO spans −0.015 to 0.464 and beats 1/N exactly once, at a volatility no account would survive; shortening the window to seven observations per asset collapses it to 0.044.
    - **Minimum variance — the same optimizer with $\hat\mu$ deleted — wins every configuration** (0.535 to 0.892). Removing the expected-return vector helps more than any refinement to it.
    - Ledoit–Wolf shrinks by **1.85%** at 56 observations per asset, because the sample covariance is already well-conditioned there. It is working correctly and it cannot help: the error MVO maximizes is in $\hat\mu$, not $\hat\Sigma$.
    - Average pairwise sector correlation runs 0.314 in a calm year and **0.922 at the March 2020 peak**, where nine sectors are **1.07 effective bets**. Conditioning on the market alone, the worst decile of days averages ρ = 0.516 against 0.327 in the middle 80%.
    - Constraints are the shrinkage that works: long-only with a 25% cap lifts Sharpe 0.377 → **0.444** while cutting volatility to 17.5% and turnover from 142 to 2.8 times a year — the Jagannathan–Ma equivalence, and a far larger implied shrinkage than Ledoit–Wolf applied.
    - SLSQP solved every constrained problem here — 281 per configuration — with **zero convergence failures** in single-digit milliseconds each, including non-smooth turnover and non-convex risk-budget penalties; no convex solver dependency is needed at this scale.
    - HRP never inverts the covariance matrix, holds gross at 1.00, turns over 0.8×/yr, and beat 1/N in all five configurations while losing to minimum variance in all five.

## Where this goes next

Four lessons have measured the book, sized it, allocated it and optimized it, and every one of them has produced its answer as a summary statistic — a volatility, a Sharpe, a correlation, an effective bet count. Those are all statements about the middle of a distribution, and the events that end funds live in the left tail, where this part has repeatedly found its own summary numbers to be least reliable: kurtosis of 29.2 on the surviving book, an expected-shortfall ratio 45% past Gaussian, a Kelly fraction exceeding its own ruin bound, and correlations that reach 0.922 exactly when the diversification is needed. [Drawdowns, Tail Risk, and Stress Testing](05-drawdowns-tail-risk-stress-testing.md) goes there deliberately — fitting the tail rather than assuming it, asking whether the worst drawdown in the record was even unusual, replaying every crisis in the sample against the book, and pricing what protection costs against what it delivers.
