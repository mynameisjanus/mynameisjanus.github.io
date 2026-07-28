# Capstone 5 — Event-Driven Backtester

!!! warning "Under development"
    Full project specifications, datasets, and grading rubrics are being written.
    The brief below defines the project's scope.

You will complete the event-driven backtesting engine begun in Part V — the event loop, portfolio accounting, fill simulation, and tearsheet output — and validate it against known results. A backtester with a silent accounting bug is worse than no backtester, so the emphasis of this project is correctness: the engine must reproduce hand-computed results exactly and match a vectorized reference implementation on simple strategies before you are allowed to trust it on anything else. This engine becomes the simulation core for every later capstone.

## Objectives

- Implement a complete event loop handling market, signal, order, and fill events in correct sequence with no lookahead
- Implement accounting that tracks cash, positions, realized and unrealized PnL, and transaction costs to the cent
- Simulate fills under explicit slippage and latency assumptions, with the assumptions configurable and documented
- Validate the engine against hand-computed cases and an independent vectorized reference implementation

## Deliverables

- A working engine package (installable, versioned) with a test suite covering the accounting and event-ordering logic
- A validation report: exact reproduction of hand-computed trades, and agreement with a vectorized reference on buy-and-hold and a moving-average strategy
- A tearsheet module producing standard performance output (returns, drawdowns, exposures, trade statistics)
- A continuous-integration configuration that runs the full test suite on every commit

## You will use

- [Part V](../part-05-backtesting-engine/index.md)
- [Part IX — testing](../part-09-software-engineering/index.md)
