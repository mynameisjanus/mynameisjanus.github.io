# Package Structure, Configuration, and Dependency Injection

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Lay out a trading platform as an installable package with enforced boundaries between data, signals, portfolio, execution, and infrastructure layers
- Implement a layered configuration system — defaults, environment overrides, schema validation — that spans research, paper, and live trading without code changes
- Use dependency injection behind broker and data-feed interfaces so implementations can be swapped without touching strategy code
- Design a plugin mechanism that lets new strategies be added and discovered without modifying the platform core

## Outline

1. Package layout for a trading platform — modules, boundaries, dependency direction
2. src layout, pyproject, and installable packages
3. Configuration strategy — layers, per-environment overrides, schema validation
4. Secrets and credentials — keeping them out of code and config files
5. Dependency injection — constructor injection against protocols and interfaces
6. Swapping brokers and data feeds — one interface, many implementations
7. Plugin architectures — entry points, registries, and strategy discovery

## Prerequisites

- [Git and Code Review](01-git-and-code-review.md)
