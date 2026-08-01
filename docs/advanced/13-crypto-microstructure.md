# Crypto Market Microstructure

[Part I ranked crypto second-best for the independent trader](../part-01-foundations/05-asset-classes.md) — free full-depth order books, low fees, 24/7 access, no prime broker required — and named the price of admission: the exchange is simultaneously venue, broker, and custodian, which collapses three separately-regulated counterparties into one unregulated entity holding your money. [Part VI's deployment lesson](../part-06-live-infrastructure/03-docker-and-cloud-deployment.md) pointed here for what that means operationally. This module is the detailed answer.

Crypto's market structure differs from equities in ways that change system design rather than just parameter values: a futures contract that never expires and instead pays a floating rate every eight hours, no consolidated tape and therefore no reference price you did not compute yourself, liquidity fragmented across venues that can and do fail, and a market that never closes so there is no daily settlement window in which to reconcile. Each of those is measured here against **252,944 rows of real hourly data** — Binance and OKX perpetual futures, Binance spot, and the complete eight-hourly funding history for BTC and ETH from 2020 through mid-2025.

The uncomfortable result is the module's centrepiece and it is a *seductive* one rather than a disappointing one. The basis trade that harvests funding earns **13.4% a year at a Sharpe of 5.71 with a 3.9% maximum drawdown** on this data — the best risk-adjusted number anywhere in this course, better than [Part IV's short-volatility sleeve](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) at 1.49. It is also a strategy whose worst historical outcome is not in the price series at all, because the risk that matters is the venue holding both legs.

!!! note "Versions"
    This module adds `ccxt` 4.5.70 and builds one new cache, `data/adv13.parquet` (253k rows, about 5 MB, gitignored like every other cache). Data is hourly OHLCV and eight-hourly funding from Binance and OKX, 2020-01-01 to 2025-06-30 UTC. The builder below is a one-time download; every later block reads the file. Venue APIs are geographically restricted in some jurisdictions — the builder names its venue in a `venue` column so the analysis reads the same whichever exchange served it.

## A perpetual future is a bond that reprices every eight hours

A traditional future converges to spot by expiring. A perpetual never expires, so convergence has to be manufactured, and the mechanism is the **funding rate**: at fixed intervals — eight hours on most venues — longs pay shorts (or the reverse) an amount proportional to the position's notional,

$$
\text{payment} \;=\; q \cdot f \cdot P, \qquad
f \;=\; \bar{P}_{\text{premium}} + \operatorname{clamp}\bigl(i - \bar{P}_{\text{premium}},\ \pm 0.05\%\bigr),
$$

where $\bar P_{\text{premium}}$ is a time-weighted average of $(\text{perp} - \text{index})/\text{index}$ and $i$ is a fixed interest differential. The economics are what matter: when the perp trades above spot, longs pay shorts, which makes being long expensive and being short attractive, and arbitrage pushes the perp back toward the index. **Funding is the restoring force that expiry provides elsewhere.**

That makes funding simultaneously a cost, a signal, and a strategy. Build the cache and measure it:

```python
# one-time download — requires a network connection
import hashlib
import time
import numpy as np
import pandas as pd
import ccxt

START = ccxt.binance().parse8601("2020-01-01T00:00:00Z")
END = ccxt.binance().parse8601("2025-07-01T00:00:00Z")

def klines(ex, symbol, limit=1000):
    out, since = [], START
    while since < END:
        batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=limit)
        if not batch or batch[-1][0] + 1 <= since:
            break
        out += batch
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)                  # respect the venue's rate limit
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[df.ts < END].drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df

def funding(ex, symbol):
    out, since = [], START
    while since < END:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch or batch[-1]["timestamp"] + 1 <= since:
            break
        out += batch
        since = batch[-1]["timestamp"] + 1
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame([{"ts": r["timestamp"], "funding_rate": r["fundingRate"]} for r in out])
    df = df[df.ts < END].drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df

spot = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
perp = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
okx = ccxt.okx({"enableRateLimit": True})
frames = []
for sym, tag in [("BTC/USDT", "BTC"), ("ETH/USDT", "ETH")]:
    d = klines(spot, sym)
    d["venue"], d["symbol"], d["kind"], d["funding_rate"] = "binance", tag, "spot", np.nan
    frames.append(d)
for sym, tag in [("BTC/USDT:USDT", "BTC"), ("ETH/USDT:USDT", "ETH")]:
    d = klines(perp, sym)
    d["venue"], d["symbol"], d["kind"], d["funding_rate"] = "binance", tag, "perp", np.nan
    frames.append(d)
    f = funding(perp, sym)
    for c in ["open", "high", "low", "close", "volume"]:
        f[c] = np.nan
    f["venue"], f["symbol"], f["kind"] = "binance", tag, "funding"
    frames.append(f)
d = klines(okx, "BTC/USDT:USDT", limit=300)              # OKX pages 300 at a time
d["venue"], d["symbol"], d["kind"], d["funding_rate"] = "okx", "BTC", "perp", np.nan
frames.append(d)

cols = ["ts", "venue", "symbol", "kind", "open", "high", "low", "close", "volume", "funding_rate"]
out = (pd.concat(frames, ignore_index=True)[cols]
       .sort_values(["venue", "symbol", "kind", "ts"]).reset_index(drop=True))
for c in cols[4:]:
    out[c] = out[c].astype("float64")
out.to_parquet("data/adv13.parquet", index=False)
print(out.groupby(["venue", "symbol", "kind"]).size().to_string())
print(f"total rows {len(out):,}, sha256 "
      f"{hashlib.sha256(open('data/adv13.parquet','rb').read()).hexdigest()[:12]}")
# => venue    symbol  kind
#    binance  BTC     funding     6024
#                     perp       48192
#                     spot       48160
#             ETH     funding     6024
#                     perp       48192
#                     spot       48160
#    okx      BTC     perp       48192
#
#    total rows 252,944, sha256 f1c3afe4a314
```

With the cache frozen, the funding series answers the first question — who has been paying whom, and how much:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/adv13.parquet")

def series(venue, symbol, kind, col="close"):
    d = df[(df.venue == venue) & (df.symbol == symbol) & (df.kind == kind)]
    return d.set_index("ts")[col].sort_index()

for sym in ["BTC", "ETH"]:
    f = series("binance", sym, "funding", "funding_rate").dropna()
    by_year = (f * 3 * 365).groupby(f.index.year).mean()
    print(f"{sym}: {len(f):,} events, mean {f.mean() * 1e4:+.2f} bp per 8h "
          f"= {f.mean() * 3 * 365:+.1%}/yr, negative {(f < 0).mean():.1%} of the time")
    print("     " + "  ".join(f"{y}:{v:+.1%}" for y, v in by_year.items()))
# => BTC: 6,024 events, mean +1.23 bp per 8h = +13.5%/yr, negative 12.7% of the time
#         2020:+17.2%  2021:+30.6%  2022:+4.2%  2023:+7.9%  2024:+11.9%  2025:+4.4%
#    ETH: 6,024 events, mean +1.48 bp per 8h = +16.3%/yr, negative 11.6% of the time
#         2020:+27.4%  2021:+37.5%  2022:+0.8%  2023:+8.3%  2024:+13.0%  2025:+4.7%
```

Funding has been **persistently positive** — 87% of intervals for BTC — meaning longs pay shorts almost always. That is a structural fact with an economic reading: perpetuals are the retail speculator's preferred instrument for leveraged long exposure, and the funding rate is the price they pay for it. The annualized magnitudes are not small (a 13.5% average for BTC) and they are strongly regime-dependent: **+30.6% in the 2021 bull market, +4.2% in the 2022 collapse**. Funding is a crowding indicator as much as a cost.

## Cash and carry, and the best Sharpe in this course

A persistently positive funding rate is an invitation. Buy spot, short the perpetual in equal size, and the position has no price exposure — the two legs offset — while collecting funding every eight hours. This is the crypto basis trade, and it is the single most widely-run strategy in the asset class:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/adv13.parquet")

def series(venue, symbol, kind, col="close"):
    d = df[(df.venue == venue) & (df.symbol == symbol) & (df.kind == kind)]
    return d.set_index("ts")[col].sort_index()

FEE = 0.0004                                             # 4 bp taker, each leg
for sym in ["BTC", "ETH"]:
    f = series("binance", sym, "funding", "funding_rate").dropna()
    spot = series("binance", sym, "spot").reindex(f.index, method="ffill")
    perp = series("binance", sym, "perp").reindex(f.index, method="ffill")
    basis = perp / spot - 1.0
    pnl = f - basis.diff().fillna(0.0)                    # collect funding, mark the basis
    pnl.iloc[0] -= 4 * FEE                                # open and close both legs
    eq = pnl.cumsum()
    print(f"{sym}: net {pnl.mean() * 3 * 365:+.2%}/yr, Sharpe "
          f"{np.sqrt(3 * 365) * pnl.mean() / pnl.std():.2f}, "
          f"maxDD {(eq - eq.cummax()).min():.2%}, worst 8h {pnl.min():.2%}")
    print("     worst intervals: " + ", ".join(
        f"{d:%Y-%m-%d}:{v:.2%}" for d, v in pnl.nsmallest(3).items()))
# => BTC: net +13.42%/yr, Sharpe 5.71, maxDD -3.87%, worst 8h -2.35%
#         worst intervals: 2020-03-13:-2.35%, 2020-12-21:-2.18%, 2020-06-02:-0.47%
#    ETH: net +16.22%/yr, Sharpe 8.08, maxDD -1.81%, worst 8h -1.17%
#         worst intervals: 2022-05-28:-1.17%, 2022-09-15:-0.88%, 2020-02-10:-0.70%
```

**Sharpe 5.71 on BTC and 8.08 on ETH**, with maximum drawdowns under 4%. Nothing in this course comes close — Part IV's best surviving sleeve managed 1.49, and the whole gauntlet existed to discipline numbers a tenth this size. Every instinct the course has trained should now be firing, and the honest reading has three parts.

First, what is genuinely real: the trade is a *funding* harvest, not an arbitrage. Someone is paying to be levered long, and a delta-neutral book collects it. That payment exists, it is large, and it has persisted for five years. Second, the visible risks are visible: the worst eight-hour interval, −2.35% on 13 March 2020, is the COVID liquidation cascade, when the perp traded far below spot as forced sellers hit the futures book — precisely when a short-perp position is marked against you. The 2020-12-21 and 2022-05-28 dates are the same phenomenon in miniature. This is short-convexity: steady collection punctuated by violent basis dislocations, structurally identical to [Part IV's short-volatility sleeve](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) and deserving the same suspicion.

Third, and decisively: **the tail that matters is not in this data at all.** Every number above assumes the exchange returns your money. The next two sections are about why that assumption is the trade.

## Leverage is the exchange hedging you, at your expense

Perpetuals are offered with leverage up to 100×, and the liquidation mechanism is unlike a margin call in equities. There is no phone call and no next-day settlement: when the mark price reaches the maintenance-margin threshold, the exchange's liquidation engine closes the position immediately, at market, and charges a fee. A position at leverage $L$ with maintenance margin $m$ is liquidated on an adverse move of roughly

$$
\Delta P^{*} \;\approx\; \frac{1}{L} - m,
$$

so 20× leverage dies on a 4.5% move. Measure that against what this market actually does:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/adv13.parquet")
p = (df[(df.venue == "binance") & (df.symbol == "BTC") & (df.kind == "perp")]
     .set_index("ts")["close"].sort_index())

for h in [8, 24]:
    mv = p.pct_change(h).dropna()
    print(f"worst {h:>2}h move {mv.min():.1%}, 1st percentile {mv.quantile(0.01):.1%}, "
          f"5th percentile {mv.quantile(0.05):.1%}")
mv = p.pct_change(24).dropna()
for lev in [5, 10, 20, 50]:
    thresh = 1.0 / lev - 0.005
    print(f"  {lev:>2}x leverage is liquidated by a {thresh:.1%} move: "
          f"{(mv < -thresh).sum():,} of {len(mv):,} 24h windows ({(mv < -thresh).mean():.2%})")
# => worst  8h move -33.5%, 1st percentile -5.4%, 5th percentile -2.7%
#    worst 24h move -47.4%, 1st percentile -9.1%, 5th percentile -4.8%
#       5x leverage is liquidated by a 19.5% move: 27 of 48,168 24h windows (0.06%)
#      10x leverage is liquidated by a 9.5% move: 408 of 48,168 24h windows (0.85%)
#      20x leverage is liquidated by a 4.5% move: 2,688 of 48,168 24h windows (5.58%)
#      50x leverage is liquidated by a 1.5% move: 10,811 of 48,168 24h windows (22.44%)
```

A **47.4% drawdown in twenty-four hours** and 33.5% in eight. At 20× leverage, 5.58% of all 24-hour windows in this history contain a move large enough to liquidate — better than one week in twenty. At 50×, more than a fifth of windows do. The advertised 100× is not a leverage facility; it is a lottery ticket with a negative expectation, and the liquidation engine is how the exchange guarantees it never carries the loss.

This is what makes the previous section's Sharpe misleading in a specific way. A carry trade yielding 13.4% is tempting to lever — at 5× it becomes 67%, and the position is nominally delta-neutral so the risk *looks* absent. But the legs sit on different books, basis dislocations move the perp leg alone, and a 2.35% eight-hour basis move against a 5×-levered position is an 11.75% equity hit that can trip maintenance margin on the futures side while the spot leg's gain is unavailable to cover it. The levered version of the trade does not survive to collect the carry that justified it.

## No NBBO: the reference price is a computation you own

Equity markets in the United States have a consolidated tape and a regulatory best bid and offer. Crypto has neither. There are dozens of venues, each with its own book, its own outages, and its own price — and "the price of Bitcoin" is whatever aggregation *you* compute. Measure how much the venues actually disagree:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/adv13.parquet")

def series(venue, symbol, kind, col="close"):
    d = df[(df.venue == venue) & (df.symbol == symbol) & (df.kind == kind)]
    return d.set_index("ts")[col].sort_index()

both = pd.concat([series("binance", "BTC", "perp").rename("bn"),
                  series("okx", "BTC", "perp").rename("ok")], axis=1).dropna()
disp = (both.bn / both.ok - 1.0) * 1e4
print(f"{len(both):,} common hours: mean {disp.mean():+.2f} bp, sd {disp.std():.2f} bp")
print(f"  |dispersion| above 10 bp in {(disp.abs() > 10).mean():.2%} of hours, "
      f"above 50 bp in {(disp.abs() > 50).mean():.3%}")
rb, ro = np.log(both.bn).diff(), np.log(both.ok).diff()
print("  lead-lag at hourly resolution: " + "  ".join(
    f"{lag:+d}h {rb.corr(ro.shift(lag)):+.4f}" for lag in [-2, -1, 0, 1, 2]))
# => 48,192 common hours: mean -1.12 bp, sd 6.40 bp
#      |dispersion| above 10 bp in 3.92% of hours, above 50 bp in 0.029%
#      lead-lag at hourly resolution: -2h -0.0125  -1h -0.0263  +0h +0.9972  +1h -0.0153  +2h -0.0164
```

Two venues quoting the same instrument agree to about 6 basis points on a typical hour and diverge past 10 bp in one hour in twenty-five. The lead-lag row deserves an explicit caveat, because it is where an incautious reader would over-conclude: contemporaneous correlation is 0.9972 and every lagged correlation is indistinguishable from zero, which does **not** mean price discovery is instantaneous. It means **hourly bars cannot see it.** Cross-venue price discovery in crypto happens on millisecond timescales, and the tick-level literature finds clear leadership relationships that this sampling frequency averages away entirely. The honest statement is that this dataset is silent on the question, and any strategy premised on cross-venue lead-lag needs data three orders of magnitude finer.

What the dispersion *does* establish is operational. A book marked on one venue's price while hedged on another carries basis risk that shows up as P&L noise; a liquidation engine using its own index can liquidate you at a price no other venue printed; and an arbitrage that looks profitable on paper requires inventory pre-positioned on both venues, because moving collateral between exchanges takes minutes to hours. **Fragmentation is not primarily an alpha opportunity; it is a capital-efficiency tax.**

## The venue is a counterparty, and it is the trade

Every position in crypto is an unsecured claim on an exchange. The exchange is the venue, the broker, the clearinghouse, and the custodian, and [Part I named this](../part-01-foundations/04-exchanges-brokers-ecns.md) as the structural difference from equities, where those roles are separated by regulation and a central counterparty guarantees settlement. There is no such guarantee here, and the historical record is not ambiguous: Mt. Gox in 2014, QuadrigaCX in 2019, and FTX in November 2022 — the second-largest venue by volume, which collapsed within days and returned nothing to customers for years.

The sizing arithmetic is uncomfortable because the input is unknowable. If an acceptable loss from a single venue failure is $L$ and the probability of failure over the horizon is $p_{\text{default}}$, the exposure cap is

$$
f_{\text{venue}} \;\le\; \frac{L}{p_{\text{default}}},
$$

and $p_{\text{default}}$ is not estimable from data. It is not small — the base rate over the last decade is several major failures — and it is not independent across venues, since the events that break one exchange (a stablecoin depeg, a lending contagion, a bank run) are correlated across all of them. The practical disciplines follow from admitting that: hold no more on any venue than the trade requires, withdraw profits on a schedule rather than at discretion, spread collateral across venues even at the cost of capital efficiency, and treat withdrawal *suspension* — which precedes insolvency by days — as the trigger for exit rather than the confirmation of it.

Now the carry trade can be assessed honestly. Its Sharpe of 5.71 measures the price risk of a delta-neutral book and says nothing about the risk that both legs sit inside a failing institution. A trader running that book on FTX in 2022 would have shown a beautiful equity curve through October and lost the entire notional in November — an outcome that appears nowhere in the price series and is not a tail of the distribution the Sharpe describes but a *different distribution entirely*. **The basis trade is short convexity plus counterparty risk in a trench coat**, and the appropriate response is not a better hedge but a smaller position and a shorter leash.

## An automated market maker is an impact curve you can differentiate

Decentralized exchanges price without an order book. A constant-product pool holds reserves $x$ and $y$ with $xy = k$ held invariant by every trade, so the spot price is $y/x$ and a trade of size $\Delta x$ receives exactly

$$
\Delta y \;=\; y - \frac{k}{x + \Delta x} \;=\; \frac{y\,\Delta x}{x + \Delta x}.
$$

That is remarkable in the context of [the impact module](05-market-impact-models.md), which spent its length establishing that impact is a *statistical* relationship — a square-root law with a coefficient requiring 82,551 metaorders to estimate. Here the impact curve is a closed form known exactly in advance, differentiable, with no parameters to fit:

```python
import numpy as np

x, y = 1000.0, 63_000_000.0                              # reserves: units and quote
k = x * y
print(f"pool: {x:.0f} units against {y / 1e6:.0f}M quote, spot {y / x:,.0f}")
for notional in [10_000, 100_000, 1_000_000, 10_000_000]:
    dx = notional / (y / x)
    out = y - k / (x + dx)
    print(f"  ${notional / 1e3:>7,.0f}k trade: effective price {out / dx:,.0f} "
          f"vs spot {y / x:,.0f} -> slippage {(out / dx / (y / x) - 1) * 1e4:7.1f} bp")
for r in [0.5, 0.8, 1.25, 2.0, 4.0]:
    print(f"  price ratio {r:4.2f}x: impermanent loss "
          f"{2 * np.sqrt(r) / (1 + r) - 1:+.2%}")
# => pool: 1000 units against 63M quote, spot 63,000
#      $     10k trade: effective price 62,990 vs spot 63,000 -> slippage    -1.6 bp
#      $    100k trade: effective price 62,900 vs spot 63,000 -> slippage   -15.8 bp
#      $  1,000k trade: effective price 62,016 vs spot 63,000 -> slippage  -156.2 bp
#      $ 10,000k trade: effective price 54,370 vs spot 63,000 -> slippage -1369.9 bp
#      price ratio 0.50x: impermanent loss -5.72%
#      price ratio 0.80x: impermanent loss -0.62%
#      price ratio 1.25x: impermanent loss -0.62%
#      price ratio 2.00x: impermanent loss -5.72%
#      price ratio 4.00x: impermanent loss -20.00%
```

Slippage grows *linearly* in trade size for a constant-product pool, not as a square root — $10k costs 1.6 bp and $10M costs 1,370 bp, a thousandfold size for an 856-fold cost — which makes AMMs far more expensive at size than a deep order book, and is why large crypto flow still routes to centralized venues.

The liquidity provider's side has its own closed form. Supplying to a pool is short volatility: the pool automatically sells the asset that rises and buys the one that falls, so relative to simply holding both assets the provider loses

$$
\mathrm{IL}(r) \;=\; \frac{2\sqrt{r}}{1 + r} - 1
$$

for a price ratio $r$. The measured symmetry is the tell — a doubling and a halving both cost **5.72%** — because the loss depends on $\lvert\ln r\rvert$ rather than direction. That is the signature of a short-straddle payoff, and it makes providing AMM liquidity economically identical to the market making in [module 12](12-market-making.md): fee income against adverse selection, with impermanent loss as the continuous-time name for being picked off. The provider profits only when fees exceed it, which is an empirical question about flow toxicity, not a property of the pool.

## Markets that never close still have weekends

The final structural difference is temporal. There is no close, so there is no daily settlement window, no official closing price to mark against, and no natural boundary at which to reconcile positions and restart processes. But the *participants* keep human hours:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/adv13.parquet")
d = df[(df.venue == "binance") & (df.symbol == "BTC") & (df.kind == "perp")].set_index("ts")
r = np.log(d["close"].sort_index()).diff().dropna()
vol_by_day = r.groupby(r.index.dayofweek).std() * np.sqrt(24 * 365) * 100
vlm_by_day = d["volume"].groupby(d.index.dayofweek).mean()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("annualized volatility: " + "  ".join(f"{days[i]}:{vol_by_day[i]:.0f}%" for i in range(7)))
print("mean hourly volume:    " + "  ".join(f"{days[i]}:{vlm_by_day[i]:,.0f}" for i in range(7)))
we, wd = [5, 6], [0, 1, 2, 3, 4]
print(f"weekend vs weekday: volume {vlm_by_day[we].mean() / vlm_by_day[wd].mean():.2f}x, "
      f"volatility {vol_by_day[we].mean() / vol_by_day[wd].mean():.2f}x")
# => annualized volatility: Mon:71%  Tue:65%  Wed:68%  Thu:71%  Fri:71%  Sat:47%  Sun:56%
#    mean hourly volume:    Mon:16,849  Tue:16,714  Wed:17,170  Thu:16,529  Fri:16,096  Sat:9,456  Sun:10,544
#    weekend vs weekday: volume 0.60x, volatility 0.74x
```

Weekend volume is **60%** of weekday and volatility **74%** — thinner books, but a market that keeps trading and can still gap. That combination is operationally worse than either a closed market or a uniformly active one: a Saturday liquidation cascade meets 60% of the usual liquidity, and the humans who would intervene are not at their desks. The design consequences are concrete. There is no batch window, so reconciliation must be continuous and idempotent rather than end-of-day, which is why [Part VI's scheduling lesson](../part-06-live-infrastructure/02-scheduling-and-data-plumbing.md) built jobs that can run at any time. Deployments have no safe window, so they must be rolling and reversible. On-call is genuinely 24/7, which for a small team means automation must handle the common failures unattended — and the risk controls from [Part VI's resilience lesson](../part-06-live-infrastructure/05-resilience-and-risk-controls.md) stop being good practice and become the only thing standing between a weekend outage and a liquidation.

The API layer deserves the same defensive posture. Venues publish request-weight budgets rather than simple rate limits, so a client must account for the *cost* of each call rather than count calls; they return `418` and `429` with escalating bans for violations; they go down under exactly the load that follows a large move; and order acknowledgements can be lost while the order lives. The defensive pattern is short and worth stating in code:

```python
# illustrative — requires exchange API credentials
import ccxt

ex = ccxt.binance({"apiKey": "...", "secret": "...", "enableRateLimit": True})

def submit(symbol, side, amount, client_id, attempts=3):
    """Idempotent submit: the client order id makes a retry safe after a lost reply."""
    for i in range(attempts):
        try:
            return ex.create_order(symbol, "limit", side, amount,
                                   params={"newClientOrderId": client_id})
        except ccxt.NetworkError:
            existing = [o for o in ex.fetch_open_orders(symbol)
                        if o.get("clientOrderId") == client_id]
            if existing:
                return existing[0]                       # it arrived; do not resend
            ex.sleep(1000 * 2 ** i)                      # exponential backoff
        except ccxt.RateLimitExceeded:
            ex.sleep(1000 * 2 ** i)
    raise RuntimeError(f"could not confirm {client_id}")
```

The load-bearing idea is the client-supplied order id. A timeout tells you the reply was lost, never whether the order was placed, and a blind retry in that state is how a position becomes twice the intended size at the worst possible moment. Making every submission idempotent turns an ambiguous failure into a safe one.

!!! warning "A Sharpe of 5.71 that omits its largest risk is not a Sharpe"
    The crypto basis trade earned 13.4% a year at Sharpe 5.71 with a 3.9% maximum drawdown on five years of real data — the best risk-adjusted number in this course by a wide margin, and a complete description of only the risks that appear in a price series. It says nothing about the venue holding both legs, and the venue is the trade: a trader running this book on FTX in October 2022 had these exact statistics and lost the notional in November. When the counterparty is also the exchange, the broker, and the custodian, position sizing is a statement about institutional failure probability, and no amount of price history estimates it.

!!! abstract "Key takeaways"
    - Perpetual futures replace expiry with a funding payment every eight hours; funding has been positive **87% of intervals** for BTC, averaging **+13.5% annualized** — retail leverage demand, priced.
    - Funding is strongly regime-dependent (+30.6% in 2021, +4.2% in 2022), making it a crowding indicator as much as a cost.
    - The cash-and-carry trade earned **+13.42%/yr at Sharpe 5.71** (ETH: +16.22%, Sharpe 8.08) with drawdowns under 4%, and its worst intervals — −2.35% on 2020-03-13 — are basis dislocations during forced liquidation, the same short-convexity shape as Part IV's short-volatility sleeve.
    - BTC fell **47.4% in a single 24-hour window** and 33.5% in eight hours; 20× leverage is liquidated by a 4.5% move, which occurred in **5.58%** of all 24-hour windows, and 50× in 22.44%.
    - Two venues quoting the same perpetual disagreed by 6.4 bp on a typical hour and by more than 10 bp in 3.92% of hours; hourly lead-lag correlations were indistinguishable from zero, which shows only that this sampling frequency cannot see price discovery.
    - Every position is an unsecured claim on an entity that is venue, broker, clearinghouse, and custodian at once; $f_{\text{venue}} \le L/p_{\text{default}}$ has an unknowable denominator, and withdrawal suspension is the exit trigger.
    - Constant-product AMM slippage is an exact closed form growing linearly in size — 1.6 bp at $10k, 1,370 bp at $10M — and impermanent loss $2\sqrt r/(1+r) - 1$ is symmetric in log price (−5.72% at both 0.5× and 2×), making liquidity provision a short straddle.
    - Weekends carry **60% of weekday volume and 74% of volatility**: thinner books that still gap, with no batch window for reconciliation and no safe deployment window, so automation must handle failures unattended.

## Where this goes next

The liquidity-provision economics that AMMs express in closed form are developed properly in [Market Making](12-market-making.md), whose adverse-selection decomposition is the order-book analogue of impermanent loss. The statistical impact law that centralized venues obey — and that the AMM formula replaces with an exact curve — is [Market Impact Models](05-market-impact-models.md), and the scheduling problem for working size through a fragmented market is [Optimal Execution](04-optimal-execution-almgren-chriss.md). In the core course, the asset-class comparison this module elaborates is [Part I, lesson five](../part-01-foundations/05-asset-classes.md); the infrastructure that survives continuous operation is [Part VI](../part-06-live-infrastructure/index.md), whose circuit breakers and idempotent job design were written for exactly the conditions measured here; and the counterparty-concentration question belongs in the operational risk register that [Part X](../part-10-trading-business/03-operations-compliance-tax.md) builds for a trading business.
