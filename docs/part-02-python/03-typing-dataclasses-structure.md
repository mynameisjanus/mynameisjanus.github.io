# Typing, Dataclasses, and Code Structure

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Annotate research code with type hints (including generics, `Optional`, and `Protocol`) and validate it under a strict mypy configuration.
- Model domain objects such as bars, orders, and instruments as frozen dataclasses with enums for categorical fields.
- Refactor a notebook-style script into a structured package with clear module boundaries and a testable public interface.
- Configure mypy in a project and interpret its errors well enough to fix them rather than silence them.

## Outline

1. Type hints — annotations, generics, Optional/Union, Protocols for duck typing
2. Dataclasses — frozen instances, defaults, ordering, slots, when to reach for them
3. Enums — order sides, instrument types, venue codes as typed constants
4. Structuring research code — package layout, notebooks vs modules, import discipline
5. mypy in practice — configuration, strictness levels, common error patterns
6. Refactoring case study — from a 400-line script to a small package

## Prerequisites

- [NumPy and Vectorization](01-numpy-and-vectorization.md)
