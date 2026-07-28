# Docker and Cloud Deployment

!!! warning "Under development"
    This lesson is part of the course scaffold and is being actively written.
    The learning objectives and outline below define its final scope.

## Learning objectives

By the end of this lesson you will be able to:

- Containerize each component of the trading stack with reproducible, minimal Docker images
- Wire the full system — strategy, risk engine, execution engine, Redis, PostgreSQL — together with docker-compose, including volumes and restart policies
- Choose between a VPS and managed cloud services for a small live deployment and defend the choice on cost, control, and operational burden
- Place a deployment in a region with acceptable latency to the broker's endpoints and measure the difference

## Outline

1. Containerizing the trading stack — Dockerfiles, images, and reproducible builds
2. docker-compose — services, volumes, networks, and restart policies
3. Configuration injection — environment variables and per-environment overrides
4. VPS vs managed cloud — cost, control, and operational trade-offs
5. Networking basics — ports, private networks, and firewalls
6. Regions and latency — where to put the box and why it matters less than you think

## Prerequisites

- [System Architecture](01-system-architecture.md)
