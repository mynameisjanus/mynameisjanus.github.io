# Git and Code Review

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Design a git workflow that separates exploratory research code from production trading code while keeping both versioned and reviewable
- Author pull requests scoped for effective review, and review quant code against a checklist covering lookahead bias, data alignment, units, and numerical correctness
- Reproduce any experiment from a commit hash by pinning code, configuration, data snapshot, environment, and random seed together
- Maintain a bisectable history that lets you locate the commit that changed a backtest result

## Outline

1. Git workflows for research versus production — trunk-based, branches, release tags
2. Branching discipline — short-lived branches, when research notebooks enter version control
3. Pull requests — scoping, descriptions, and review turnaround on a small team
4. Reviewing quant code — a checklist: lookahead, alignment, units, seeds, edge dates
5. Reproducibility of experiments — commit + config + data version + environment
6. History hygiene — commit messages, atomic commits, git bisect on backtest regressions

## Prerequisites

- [Part II — Logging and Configuration Management](../part-02-python/07-logging-and-config.md)
