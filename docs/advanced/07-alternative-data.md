# Alternative Data

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers sourcing, evaluating, and researching alternative datasets — news and NLP-derived signals, satellite imagery, SEC filings — with a heavy emphasis on the evaluation discipline that separates useful data from expensive noise: point-in-time integrity, backtest hygiene, and honest cost-benefit analysis. It is for learners comfortable with the feature-engineering material in Part VII who want to work with data beyond prices and fundamentals.

## Topics

- News and NLP signals: sentiment extraction, entity resolution, and timestamping text data correctly
- SEC filings: parsing EDGAR, change detection between filings, and event studies on disclosures
- Satellite and geolocation data: what it actually measures, and mapping physical observations to tradable estimates
- Vendor evaluation: coverage, history depth, restatement policy, and detecting backfilled histories
- Point-in-time discipline: as-of joins, revision handling, and why a lookahead-contaminated dataset is unrecoverable
- Alpha decay and crowding: estimating remaining edge in a dataset the vendor sells to fifty other funds
- Cost-benefit analysis: data spend versus realistic capacity-adjusted alpha

## Recommended background

- [Part VII — feature engineering for ML](../part-07-machine-learning/01-feature-engineering-for-ml.md)
