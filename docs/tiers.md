# Tiers

Not every atom deserves the same effort, and a repository that treats them
equally ends up with 250 shallow pages. Tiers are a promise about depth and
order.

| Tier | Meaning | Bar | Share |
|:--:|:--|:--|--:|
| **1** | Core | You should be able to write it from memory. Full derivation, every variant, a demo that asserts against the closed form. | ~40% |
| **2** | Specialist | The right tool for a real class of problems. Full derivation, the variants practitioners actually use. | ~40% |
| **3** | Frontier | Recent, rare, or research-grade. Included because it changes how you think, not because you will ship it next week. | ~20% |

## How tier affects the writing

- **Tier 1** gets a worked example with numbers, a complexity table, and a
  subsection on the two subtle bugs (e.g. `n < d` in OLS, empty clusters in
  k-means, posterior collapse in VAEs).
- **Tier 2** gets the derivation and the decision rule: when this beats the
  tier-1 alternative.
- **Tier 3** gets the core idea, the paper's central equation, and an honest
  note on cost and maturity. Enough to evaluate, not necessarily to productionise.

## Build order

Tier 1 is built first and end to end, so that the repository is always useful
even while it is incomplete.

```
tier 1  →  tier 2  →  tier 3
   ↓         ↓          ↓
 all five  every      the ones that
 chapters  chapter    change practice
```

The same rule applies inside a chapter: a chapter is "done" when all of its tier
1 atoms exist, not when the first one does.

## What tier is *not*

Tier is not difficulty. `hdbscan` (tier 1) is harder to implement than
`affinity_propagation` (tier 3). Tier is a statement about how often the atom is
the right answer to a real problem.
