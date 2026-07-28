# Crypto Market Microstructure

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers the market structure of cryptocurrency trading and how it differs from equities in ways that matter operationally: perpetual futures and their funding-rate mechanism, liquidity fragmented across dozens of venues with no consolidated tape, on-chain data as a genuinely new data source, and the consequences of markets that never close. It is for learners who have completed the Part I microstructure material and the Part VI infrastructure work and are considering deploying systems in crypto markets.

## Topics

- Perpetual futures: the funding-rate mechanism, basis behavior, and funding-driven strategies
- Fragmented venues: no NBBO, cross-exchange price discovery, and the operational reality of multi-venue connectivity
- Exchange counterparty risk: custody, withdrawal risk, and sizing exposure to the venue itself
- On-chain data: flows, exchange balances, and DEX activity as signals, with their latency and interpretation caveats
- Stablecoins and collateral: margin conventions, settlement assets, and hidden dependencies
- 24/7 operations: no daily close for reconciliation, weekend liquidity, and what continuous markets do to on-call design
- Exchange API practicalities: rate limits, outages under load, and defensive client design

## Recommended background

- [Part I — market microstructure](../part-01-foundations/03-market-microstructure.md)
- [Part VI — live infrastructure](../part-06-live-infrastructure/index.md)
