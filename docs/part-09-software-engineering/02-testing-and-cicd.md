# Testing and CI/CD

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Write unit, property-based, and regression tests for signal, portfolio, and execution code, and choose the appropriate kind for each component
- Test stochastic code deterministically with seeded runs, and write statistical assertions with a controlled false-alarm rate where determinism is impossible
- Build golden-file backtest tests that snapshot P&L and positions and flag any unintended change across a refactor
- Configure a CI/CD pipeline that gates deployment on linting, type checks, the test suite, and backtest regression checks

## Outline

1. The testing pyramid for trading systems — what to test at each level
2. Unit tests — pure functions, fixtures, and the special hazards of time
3. Property-based testing — invariants of signals and portfolios with Hypothesis
4. Testing stochastic code — seeding, tolerances, statistical assertions
5. Golden-file backtest tests — snapshotting results, reviewing intentional diffs
6. Regression tests — pinning every fixed bug so it stays fixed
7. CI pipelines — lint, type-check, test, and backtest gates on every merge
8. CD and deployment gating — promoting builds, rollback, and who can ship

## Prerequisites

- [Git and Code Review](01-git-and-code-review.md)
