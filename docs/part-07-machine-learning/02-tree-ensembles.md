# Tree Ensembles

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Train Random Forest and gradient-boosted models with XGBoost and LightGBM on financial features using purged, embargoed cross-validation
- Compare every ensemble against a linear baseline and report honestly when the baseline wins
- Explain why default impurity-based feature importances mislead on correlated financial features and apply permutation-based alternatives
- Calibrate predicted probabilities and verify calibration holds out of sample

## Outline

1. Why trees fit tabular financial data — and where they don't
2. Random Forests — variance reduction and its limits in low signal-to-noise settings
3. Gradient boosting — XGBoost and LightGBM in practice
4. Tuning without self-deception — purged CV and the overfit validation set
5. Baselines first — the linear model as the bar to clear
6. Feature importance pitfalls — correlated features, substitution effects, and permutation importance
7. Probability calibration — reliability curves and recalibration methods

## Prerequisites

- [Feature Engineering for ML](01-feature-engineering-for-ml.md)
