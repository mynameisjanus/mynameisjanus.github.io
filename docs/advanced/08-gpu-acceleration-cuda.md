# GPU Acceleration with CUDA

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers moving numerical workloads from CPU to GPU: enough CUDA fundamentals to reason about performance, then the practical Python toolchain — CuPy as a NumPy substitute and Numba for custom kernels — applied to backtesting and ML training workloads. It is for learners fluent in the vectorized NumPy style from Part II whose parameter sweeps or training runs have become the bottleneck, and it is honest about the cases where the GPU makes things slower.

## Topics

- GPU architecture for programmers: SIMT execution, memory hierarchy, and why branch-heavy code performs poorly
- The CUDA programming model: kernels, threads, blocks, and host-device memory transfer
- CuPy: NumPy-compatible arrays on the GPU, and the porting workflow for existing vectorized code
- Custom kernels with Numba's CUDA JIT for logic that does not vectorize cleanly
- What actually speeds up: parameter sweeps, Monte Carlo, ML training — versus transfer-bound and sequential workloads
- GPU-accelerating a vectorized backtest end to end, with before/after benchmarks
- Profiling GPU code and identifying transfer versus compute bottlenecks

## Recommended background

- [Part II — NumPy and vectorization](../part-02-python/01-numpy-and-vectorization.md)
