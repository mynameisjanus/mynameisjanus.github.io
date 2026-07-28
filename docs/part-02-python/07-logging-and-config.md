# Logging, Configuration, and Reproducibility

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Instrument a research pipeline with structured logging, choosing appropriate levels and context fields for each event.
- Externalize run parameters into YAML or TOML configuration validated with pydantic-settings, with environment-specific overrides.
- Make a research run reproducible end to end: pinned seeds, recorded configuration, and captured run metadata.
- Audit an existing script for hidden state — global RNGs, hardcoded paths, mutable defaults — that breaks reproducibility.

## Outline

1. Structured logging — JSON logs, context fields, correlation across pipeline stages
2. Log levels — what belongs at DEBUG, INFO, WARNING, and ERROR in a research pipeline
3. Configuration formats — YAML vs TOML, layering, per-environment overrides
4. pydantic-settings — typed validation, environment variables, secrets handling
5. Reproducible runs — seeds, RNG discipline, recording config and code version per run
6. Putting it together — a configurable, logged, reproducible research script

## Prerequisites

- [Typing, Dataclasses, and Code Structure](03-typing-dataclasses-structure.md)
