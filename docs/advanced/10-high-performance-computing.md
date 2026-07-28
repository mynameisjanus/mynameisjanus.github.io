# High-Performance Computing

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers making critical-path code fast when vectorized Python is no longer enough: understanding where Python's performance floor actually lies, profiling rigorously at scale, and dropping into Cython or Rust for the hot paths that justify it. It is for learners comfortable with the profiling material in Part IX who are hitting real latency or throughput limits — and it teaches restraint, because most perceived Python bottlenecks are algorithmic or structural, not language-level.

## Topics

- Python's performance floor: interpreter overhead, the GIL, garbage collection, and object allocation costs
- Profiling at scale: py-spy and perf in production-like conditions, flame graphs, and sampling versus instrumentation
- Memory layout and cache effects: contiguous arrays, struct-of-arrays versus array-of-structs, and why they dominate constant factors
- Cython: typed extensions, the incremental optimization workflow, and reading the generated annotations
- Rust extensions with PyO3 and maturin: when the safety and toolchain justify the boundary cost
- Benchmarking methodology: warm-up, variance, and avoiding self-deceiving microbenchmarks
- The decision framework: restructure, rewrite the hot path, or accept the current speed

## Recommended background

- [Part IX — profiling, refactoring, and versioning](../part-09-software-engineering/05-profiling-refactoring-versioning.md)
