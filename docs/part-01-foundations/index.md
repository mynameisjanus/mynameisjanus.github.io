# Part I — Foundations of Quantitative Trading

Most people arrive at quantitative trading wanting to write strategies. That instinct is understandable and almost always premature. A strategy is a claim about how a market misprices something — and you cannot evaluate that claim until you understand what the market actually is: who trades in it, what machinery matches their orders, what it costs to transact, and why anyone would be on the other side of your trade at all. Professionals learn the plumbing first because the plumbing is where most naive strategies quietly die — in spreads, queue positions, borrow fees, and adverse selection that never show up in a price chart.

Part I builds that foundation. We start with what algorithmic trading actually means and how a professional operation is organized. We then work through the participants who populate markets, the microstructure that governs how orders become trades, the institutional layer of exchanges, brokers, and clearinghouses, and finally the asset classes you might realistically trade.

Along the way you will produce three concrete deliverables, all diagrams you should be able to reproduce from memory by the end of the part:

- an **order-flow diagram** tracing an order from trader to matching engine and back (lesson 3),
- a **market-structure diagram** mapping retail and institutional order routing across venues (lesson 4),
- a **trade-lifecycle diagram** following a trade from placement through clearing, settlement, and custody (lesson 4).

The arc of the part runs: what markets are → who trades them → how trading actually works mechanically → what strategy families exist → why most retail attempts fail. Each lesson assumes the ones before it. Read them in order.

## Lessons

| Lesson | Focus |
|---|---|
| [What Is Algorithmic Trading?](01-what-is-algorithmic-trading.md) | Systematic vs discretionary trading, the anatomy of a professional operation, and what an edge actually is |
| [Types of Market Participants](02-types-of-market-participants.md) | Who trades, why they trade, and who is on the other side of your fills |
| [Market Microstructure](03-market-microstructure.md) | The limit order book, order types, matching engines, and adverse selection — plus the order-flow deliverable |
| [Exchanges, Brokers, and ECNs](04-exchanges-brokers-ecns.md) | Venues, routing, payment for order flow, clearing and settlement — plus the market-structure and trade-lifecycle deliverables |
| [Asset Classes](05-asset-classes.md) | Equities, futures, options, FX, and crypto compared on mechanics, leverage, hours, costs, and data |
