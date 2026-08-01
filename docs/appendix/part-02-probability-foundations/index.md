# Part II — Foundations of Probability

Probability is the measure that the set algebra of [Part I](../part-01-mathematical-foundations/index.md) was waiting for. Six pages, three axioms, and one operation — conditioning — that everything else in this appendix is an application of. Nothing here is about distributions yet; events are still just subsets, and the whole part is about what happens when a number is attached to each of them.

The dependencies run in file order, with one exception worth knowing: [Conditional Probability](03-conditional-probability.md) ends with the two-block decomposition that [Bayes' Rule](04-bayes-rule.md) needs as a denominator, so Bayes reads correctly before the general [Law of Total Probability](06-law-of-total-probability.md) rather than after it. The last page then generalizes the decomposition to arbitrary and countable partitions and hands off to [Part IV](../part-04-expectation-and-moments/index.md). Where this part stops is also worth stating: it conditions on *events* throughout, and conditioning on the value of a random variable belongs to [Part III](../part-03-random-variables/index.md).

## Topics

| Topic | Focus |
|---|---|
| [Probability Spaces](01-probability-spaces.md) | The triple $(\Omega,\mathcal{F},\mathbf{P})$, σ-algebras as a description of information, why singletons have probability zero, and measurability as a condition on preimages |
| [Probability Axioms](02-probability-axioms.md) | The three axioms and everything derived from them, the countable strengthening and continuity of measure, and the two assumption-free inequalities behind multiple-testing corrections |
| [Conditional Probability](03-conditional-probability.md) | Conditioning as renormalization, why a conditional probability is itself a probability, the multiplication and chain rules, and the transposed-conditional fallacy |
| [Bayes' Rule](04-bayes-rule.md) | Prior, likelihood, evidence, and posterior, the odds and log-odds forms, sequential updating, and the base-rate arithmetic behind false discoveries |
| [Independence](05-independence.md) | Why disjoint events are maximally dependent, pairwise versus mutual independence, conditional independence in both directions, and what iid assumes |
| [Law of Total Probability](06-law-of-total-probability.md) | Decomposition over a partition, first-step analysis, mixtures as manufactured fat tails, latent regimes, and the extension to total expectation |
