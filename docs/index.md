# Building Quantitative Trading Systems

**How professional quantitative trading systems are actually built.**

This course teaches you to work the way quantitative researchers and systematic traders work — from market structure and statistics through strategy research, backtesting, and live trading infrastructure, all the way to running a systematic trading business. By the end you will understand how a complete trading stack fits together — data to signals to orders to fills to risk controls — as one coherent system, not a collection of disconnected topics.

!!! success "Who this course is for"
    People who want to become quantitative researchers or systematic traders: you are comfortable learning real mathematics and writing real software, and you want to know how professional desks research, validate, deploy, and operate strategies.

!!! failure "Who this course is not for"
    Anyone looking for indicator recipes. There is no "buy when RSI < 30" here — those courses already exist in abundance, and the strategies they teach fail for reasons this course spends an entire lesson on.

## The learning progression

| Level | Parts | Outcome |
|---|---|---|
| **Beginner** | I–II | Understand markets and market structure; work fluently with market data in Python. |
| **Intermediate** | III–IV | Perform rigorous statistical research and evaluate strategies honestly. |
| **Advanced** | V–VII | Understand how production-quality backtesting engines and live trading infrastructure are designed; apply machine learning responsibly. |
| **Professional** | VIII–X | Manage portfolios and risk, engineer research code to professional standards, and understand how a systematic trading business operates. |

## The course

<div class="grid cards" markdown>

-   :material-bank: **[Part I — Foundations of Quantitative Trading](part-01-foundations/index.md)**

    ---

    Why algorithmic trading exists: participants, microstructure, venues, asset classes, strategy families — and why most retail strategies fail.

-   :material-language-python: **[Part II — Python for Quantitative Finance](part-02-python/index.md)**

    ---

    Not a general Python course: NumPy, Pandas/Polars, async APIs, SQL, plotting, logging, and configuration — only what quantitative work needs.

-   :material-chart-bell-curve: **[Part III — Statistics for Trading](part-03-statistics/index.md)**

    ---

    Returns and their distributions, time series, hypothesis testing, bootstrap, Bayesian methods and HMMs — on real market data.

-   :material-strategy: **[Part IV — Strategy Development](part-04-strategy-development/index.md)**

    ---

    The largest part: momentum, mean reversion, pairs, volatility, signal engineering, sizing — and the validation gauntlet that kills overfit strategies.

-   :material-engine: **[Part V — Inside a Backtesting Engine](part-05-backtesting-engine/index.md)**

    ---

    The anatomy of an event-driven backtester: accounting, order management, fill simulation, metrics, and reporting.

-   :material-server-network: **[Part VI — Live Trading Infrastructure](part-06-live-infrastructure/index.md)**

    ---

    Strategy → Risk Engine → Execution Engine → Broker API → Exchange: scheduling, Redis, PostgreSQL, Docker, monitoring, circuit breakers, deployment.

-   :material-robot: **[Part VII — Machine Learning for Trading](part-07-machine-learning/index.md)**

    ---

    Tree ensembles, deep learning, RL, meta-labeling, drift and retraining — with honest baselines, because ML is not a silver bullet.

-   :material-scale-balance: **[Part VIII — Portfolio Management](part-08-portfolio-management/index.md)**

    ---

    Kelly, volatility targeting, risk parity, optimization, drawdowns, tail risk, and stress testing: running a book, not just a strategy.

-   :material-source-branch: **[Part IX — Professional Software Engineering](part-09-software-engineering/index.md)**

    ---

    Testing, CI/CD, package structure, dependency injection, message queues, profiling: research code you can actually trust with money.

-   :material-office-building: **[Part X — Running a Quantitative Trading Business](part-10-trading-business/index.md)**

    ---

    Fund structures, fees, investor reporting, due diligence, operations, compliance, hiring, and scaling.

-   :material-rocket-launch: **[Optional Advanced Modules](advanced/index.md)**

    ---

    Kalman filters, optimal execution, market impact, options pricing, market making, alternative data, GPU acceleration, and more.

-   :material-sigma: **[Appendix — Mathematical Prerequisites](appendix/index.md)**

    ---

    A self-contained probability and statistics reference, from counting principles to Markov processes, linked throughout the course.

</div>

## About

The course is written by [Janus B. Advincula](about/index.md) — physicist by training, MIT MicroMasters in Statistics and Data Science, currently building a systematic trading platform. If the material is useful to you, you can [support the site](about/support.md).
