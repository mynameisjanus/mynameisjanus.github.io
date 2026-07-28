# NumPy and Vectorization

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Construct and manipulate NumPy arrays of prices and returns using appropriate dtypes, slicing, and views versus copies.
- Rewrite a loop-based indicator calculation as a vectorized, broadcast expression and benchmark the speedup.
- Diagnose floating-point precision failures (accumulation error, catastrophic cancellation, unsafe equality tests) in return calculations and apply numerically stable alternatives.
- Control NaN propagation in an array pipeline, choosing deliberately between nan-aware reductions and explicit masking.

## Outline

1. Arrays, dtypes, and memory layout — why ndarray beats Python lists for market data
2. Indexing and slicing — views vs copies, boolean masks, fancy indexing
3. Broadcasting rules — aligning shapes without writing loops
4. Vectorization in practice — rewriting a loop-based signal, timing it honestly
5. Ufuncs and reductions — cumulative products, axis semantics, log-space computation
6. Floating-point pitfalls — precision limits, cancellation, comparing floats safely
7. NaN and missing data — propagation rules, nan-aware functions, masking strategies

## Prerequisites

- None — this part starts the programming track.
