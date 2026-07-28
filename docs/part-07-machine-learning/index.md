# Part VII — Machine Learning for Trading

Machine learning is not a silver bullet. Financial markets have among the lowest signal-to-noise ratios of any domain ML is applied to, and in that regime simpler statistical methods often outperform complex models — a well-specified linear model or a robust cross-sectional rank frequently beats a deep network that has memorized noise. Any course that presents ML as the path to alpha, rather than one tool among several, is selling something. Every lesson in this part therefore starts from an honest baseline and requires you to demonstrate — not assume — that the added complexity earns its keep.

That framing is not pessimism; it is how professional teams actually use ML. Applied with discipline, the tools in this part are genuinely valuable: careful feature engineering and leak-free labeling, tree ensembles that handle tabular financial data well, deep learning where data volume supports it, and meta-labeling to size and filter signals a simpler model generates. The recurring skills are the unglamorous ones — purged validation, leakage audits, calibration, and knowing when to stop.

The part closes with production ML: drift detection, model monitoring, champion/challenger retraining, and registries that tie every live prediction to an exact model version. A model that cannot be monitored, retrained, and rolled back safely does not belong in a live trading system, whatever its backtest says.

## Modules

| Module | Focus |
|---|---|
| [Feature Engineering for ML](01-feature-engineering-for-ml.md) | Features from prices, volume, and microstructure; stationarity, fixed-horizon and triple-barrier labeling, sample weights, and leakage |
| [Tree Ensembles](02-tree-ensembles.md) | Random Forests and gradient boosting on financial data, feature importance pitfalls, and probability calibration |
| [Deep Learning](03-deep-learning.md) | Feed-forward networks, LSTMs, temporal convolutions, and transformers — why they underwhelm on daily bars and where they earn their keep |
| [Reinforcement Learning and Meta-Labeling](04-reinforcement-learning-and-meta-labeling.md) | The RL formulation of trading, why naive RL fails, meta-labeling to size and filter signals, and ensembles of weak signals |
| [Production ML](05-production-ml.md) | Online learning, concept drift detection, model monitoring, champion/challenger retraining, and model registries |
