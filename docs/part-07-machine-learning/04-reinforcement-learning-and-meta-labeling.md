# Reinforcement Learning and Meta-Labeling

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Formulate trading as a reinforcement learning problem — state, action, reward — and identify exactly where the formulation strains against market reality
- Demonstrate, with your own experiments, why naive RL fails in low signal-to-noise, non-stationary markets
- Implement meta-labeling to size and filter an existing primary signal and measure the improvement over the raw signal
- Combine an ensemble of weak signals and compare it against the best single signal under identical costs

## Outline

1. Trading as an RL problem — states, actions, rewards, and episodes
2. Why naive RL fails — noise, non-stationarity, and the cost of samples
3. What survives the critique — where RL-adjacent methods still contribute
4. Meta-labeling — splitting direction from sizing with a secondary model
5. Sizing and filtering with meta-labels — measuring the lift
6. Ensembles of weak signals — combination methods and correlation discipline

## Prerequisites

- [Tree Ensembles](02-tree-ensembles.md)
