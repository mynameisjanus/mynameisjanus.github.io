# High-Frequency Trading

No corner of finance is more mythologized than high-frequency trading. In the retail imagination, HFT is a predatory algorithm that watches your order, hunts your stop loss, and manipulates prices at will. The reality is both less sinister and more interesting: HFT is mostly the business of *selling immediacy* — being willing to trade with anyone, right now, for a fraction of a cent — executed with engineering so extreme that the speed of light through glass versus air is a line item in the budget.

You will almost certainly never compete at this timescale, and this lesson is blunt about why. But HFT firms are the counterparty to a large fraction of everything you will ever execute: understanding how they make money is understanding why your fills cost what they cost — and why backtests that ignore this are fiction.

## What HFT actually is

Two business models account for most of the industry.

**Electronic market making.** A market maker continuously posts a bid and an ask in an instrument — willing to buy at 100.01, sell at 100.03 — earning the spread from traders who demand immediacy. The revenue is tiny per share; the risks are two. *Inventory risk*: the maker holds each position until someone takes it back, and prices move meanwhile. *Adverse selection*: the trader who just hit your quote may know something you don't — informed flow systematically trades against stale quotes. The whole problem is pricing the spread wide enough to cover these risks and tight enough to win the queue, continuously. It is a legitimate, economically useful service — the reason a liquid ETF trades at a one-cent spread all day — done across thousands of instruments simultaneously.

**Latency arbitrage.** The same instrument, or tightly linked instruments (an ETF and its futures, the same stock on two exchanges), are priced on multiple venues. When news moves the price in one place, quotes elsewhere are momentarily stale. Whoever reacts first trades against the stale quote for a nearly certain profit. This is the zero-sum, purely speed-based side of the business — the reason the arms race exists — and the *tax* that forces makers to quote wider: every spread must cover expected losses to faster arbitrageurs.

What HFT is generally *not*: a machine that targets your particular 200-share retail order. Your order is economically invisible; retail flow is the flow makers most *want*, precisely because it is uninformed — which is why wholesalers pay for the privilege of filling it.

## The arms race, in numbers

The magnitudes involved deserve to be internalized, because they define who can play.

| Latency | Order of magnitude | What it is |
|---|---|---|
| Human reaction time | ~200 ms | Fastest possible click |
| Retail order, home internet to broker to exchange | ~10–100 ms | Your practical floor |
| Cross-country fiber, Chicago–New Jersey round trip | ~13 ms | Light in glass, plus routing |
| Microwave link, Chicago–New Jersey round trip | ~8 ms | Light in air; straighter path |
| Software trading system, wire-to-wire | ~10–100 µs | Well-engineered C++ |
| FPGA tick-to-trade, wire-to-wire | < 1 µs | Market data parsed and order emitted in hardware |
| One microsecond of light travel | ~300 m in air, ~200 m in fiber | Why cable length inside the building matters |

The structural implications, concretely:

- **Colocation.** Firms rent racks in the exchange's own data center; exchanges famously provision *equal-length* cable coils to every colocated cabinet, so no rack gains a nanosecond from sitting closer to the matching engine.
- **Microwave and beyond.** Light travels roughly 50% faster in air than in fiber, and towers follow a nearly straight geodesic: private microwave networks between Chicago's futures markets and New Jersey's equity markets cut round trips from ~13 ms to ~8 ms. Later escalations shaved fractions of a millisecond at costs of millions per year.
- **Hardware.** At the frontier, decisions no longer happen in software. FPGAs — reprogrammable chips — parse the exchange's raw market data feed and emit orders in hundreds of *nanoseconds*, the trading logic burned into circuitry. The talent involved overlaps more with chip design than finance.

!!! note "The economics of the race"
    Speed is winner-take-most: in a latency race, second place earns nothing. That tournament structure is why spending escalates until the marginal microsecond costs as much as it earns — and why the firms at the true frontier can be counted on two hands.

## The economics: tiny edge, enormous volume

A back-of-envelope: a maker netting one-tenth of a cent per share must trade 100 million shares to make \$100,000. Major firms trade billions of shares' worth of volume; per-trade profitability is fractions of a basis point, win rates barely above 50%, and the aggregate becomes reliable only through millions of trades a day. The Sharpe ratios are extraordinary — double digits at the top firms — not because any trade is smart but because the law of large numbers operates at industrial scale.

The active risk management problem is **inventory**. A maker who keeps buying accumulates a long position — unwanted directional exposure. The standard response is *quote skewing*: when inventory grows long, lower both your bid and ask, making further buys less likely and sells more likely, steering inventory back toward zero. Position limits are tight, and books are typically flattened by the end of each day; the ideal HFT firm goes home flat, having earned the spread thousands of times and held nothing.

## Why you will not compete here — and what transfers anyway

Be clear-eyed: the table above is a paywall. Colocation, direct exchange feeds, purpose-built hardware, and the engineering team to run them represent a fixed cost in the millions per year before the first profitable trade — and a strategy at the microsecond horizon competes with firms that have spent a decade and nine figures on the problem. This course will not pretend otherwise: everything here targets horizons of minutes to months, where speed is not the edge.

But the microstructure lessons transfer completely, and they are among the most valuable in this part:

- **The spread is a real cost you pay to real counterparties.** Every market order crosses the spread; that half-spread is the maker's revenue and your expense, on every trade. Thin-edge, high-turnover strategies are, in expectation, donating to the firms in this lesson.
- **Queue position is why passive fills disappoint.** Post a limit order instead and you join a price-time priority queue behind everyone who quoted before you. When your bid *does* fill, ask why the seller hit it — often because the price is about to fall. You are filled most reliably exactly when being filled is bad.
- **Adverse selection is why your fills are worse than your backtest.** A backtest that assumes execution at the last printed price, or at the touch with certainty, systematically books fills you would not have gotten (the good ones) and omits the toxicity of the ones you would have (the bad ones). This error has flattered more retail backtests than any other except look-ahead bias.
- **Liquidity is state-dependent.** Quoted depth evaporates in exactly the volatile moments your strategy most wants to trade, as makers widen or pull quotes. Cost models calibrated on calm markets badly understate stress costs.

!!! warning "The practical rule"
    Until you have measured otherwise, assume every round trip in a liquid instrument costs you the full bid-ask spread plus fees, and more in size or in stress. Part V's backtesting engine treats execution costs as a first-class modeling problem for exactly this reason.

!!! abstract "Key takeaways"
    - HFT is primarily electronic market making (selling immediacy for the spread) plus latency arbitrage (racing to trade against stale quotes) — not the retail caricature of order-hunting.
    - The arms race operates at physical limits: colocation, microwave links saving ~5 ms Chicago–New Jersey, FPGA tick-to-trade under a microsecond.
    - The economics are tiny per-trade edges times enormous volume; the binding risks are inventory and adverse selection, managed by quote skewing and flat end-of-day books.
    - Speed competition is winner-take-most, which drives spending to the frontier and makes the microsecond game inaccessible to individuals — by design, this course targets slower horizons.
    - What transfers: spread costs are real, queue position governs passive fills, adverse selection makes live fills worse than backtest fills, and liquidity vanishes when you need it most.
    - Model execution pessimistically: full spread plus fees per round trip is the floor assumption, not the worst case.

## Where this goes next

You now have the map of the professional landscape: regimes, the trend/reversion duality, stat-arb's industrialized edge, and the microstructure toll booth at the bottom of the stack. The final lesson of Part I turns the lens around and asks the question this course exists to answer: why does the typical retail systematic strategy fail — specifically and mechanically? [Why Most Retail Strategies Fail](10-why-most-retail-strategies-fail.md).
