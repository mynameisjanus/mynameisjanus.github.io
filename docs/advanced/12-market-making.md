# Market Making

!!! warning "Under development"
    This optional advanced module is part of the course scaffold.

This module covers the economics and models of market making: earning the spread while managing the two risks that eat it — inventory risk and adverse selection. The centerpiece is the Avellaneda–Stoikov inventory model, which turns quoting into a tractable control problem, surrounded by the practical craft of measuring adverse selection and defending quotes against better-informed flow. It is for learners solid on the Part I microstructure material, ideally with the options module's exposure to hedged-book thinking, who are targeting market-making or liquidity-provision roles.

## Topics

- The economics of market making: spread capture, inventory risk, and adverse selection as the three-way trade-off
- The Avellaneda–Stoikov model: reservation price, optimal bid and ask offsets, and inventory-driven quote skewing
- Implementing and simulating an inventory-based quoting strategy
- Measuring adverse selection: markout analysis of fills and identifying toxic flow
- Queue position and price-time priority: why placement and cancellation timing dominate in tight-spread markets
- Risk controls for quoting systems: inventory limits, pull-quote triggers, and behavior around news and auctions

## Recommended background

- [Part I — market microstructure](../part-01-foundations/03-market-microstructure.md)
- [Options Pricing module](11-options-pricing.md)
