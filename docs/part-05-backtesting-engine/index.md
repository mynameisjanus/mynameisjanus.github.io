# Part V — Inside a Backtesting Engine

Every quantitative strategy lives or dies by the quality of its simulation. A backtest that leaks future information, mishandles cash, or fills orders at fantasy prices will manufacture alpha that evaporates the moment real money is at risk. Professional desks treat the backtesting engine as core infrastructure — reviewed, tested, and trusted — because every research conclusion downstream depends on it.

In this part, instead of treating a backtester as a library you call, we take an event-driven engine apart piece by piece. The point is to understand the mechanics — how market data becomes signals, signals become orders, and orders become fills — well enough that no framework is ever a black box to you again. Studying the event loop, the portfolio ledger, and the fill simulator at this level of detail is the most reliable way to internalize where look-ahead bias, accounting errors, and optimistic fill assumptions come from.

By the end of this part you know what a trustworthy engine consists of: an event queue, portfolio accounting that reconciles to the penny, a simulated broker with defensible fill models, a metrics and tearsheet layer, and structured trade logs with visualization. That knowledge carries through the rest of the course — whenever a backtest result looks too good, you will know exactly which internal assumption to interrogate.

## Modules

| Module | Focus |
|---|---|
| [Architecture and Event-Driven Design](01-architecture-and-event-driven-design.md) | Vectorized vs event-driven simulation, the four core event types, and an event loop that rules out look-ahead bias by construction |
| [Portfolio Accounting](02-portfolio-accounting.md) | Positions, FIFO lot accounting, cash, mark-to-market, corporate actions, and multi-currency books |
| [Order Management and Fill Simulation](03-order-management-and-fill-simulation.md) | The order lifecycle state machine, order types, fill models from next-bar open to spread-plus-impact, and partial fills |
| [Performance Metrics and Reporting](04-performance-metrics-and-reporting.md) | Returns, Sharpe/Sortino/Calmar, drawdown statistics, turnover, exposure, hit rate, and tearsheet reports |
| [Trade Logs and Visualization](05-trade-logs-and-visualization.md) | Structured trade logs, round-trip analysis, equity/drawdown/exposure plots, and comparing runs |
