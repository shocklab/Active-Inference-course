# Outline audit, 2026-08-26

Triggered by one probe: "where do we have variational message passing?" It was
mislabelled, uncommitted and uncross-linked. A single failed sample warranted
checking whether the same held elsewhere.

## Method

The audit could not be grounded in the outline itself, because an outline audited
against its own author's knowledge just gets re-derived. So the concept inventory
came from two sources **outside** it, both read rather than skimmed:

- **Smith, Friston & Whyte (2022)**, section structure extracted from the PDF.
- **Parr, Pezzulo & Friston (2022)**, chapter structure.

Namjoshi was deliberately **excluded** from the inventory. It is a private draft
of a copyrighted book, and it was also the source of the original error, so using
it to check the work would reproduce the mistake it caused.

## The prediction, made before the audit ran

Outline quality was hypothesised to track whether the source was read or only its
table of contents. Measured beforehand:

| Week | named sources | derivation targets |
|---|---|---|
| 4, 10 | 6, 4 | 3, 3 |
| 2, 3, 6 | 0 | 1, 1, 2 |
| 5, 7, 8, 9, 11, 12 | 0 (12 had 1) | **0** |

Predicted: gaps cluster in Weeks 5, 7, 8, 9, 11 and are rare in 4 and 10.

## Findings

Six topics that both external references treat substantially and the outline
did not contain at all.

| # | Missing topic | External source | Belonged in |
|---|---|---|---|
| 1 | **Marginal message passing**, and the three-schemes comparison (mean-field → VMP, Bethe → belief propagation, marginal → MMP) | Smith §2.3; Parr et al. 2019 | Week 9 |
| 2 | **Prediction-error formulation of the discrete scheme** — the bridge between the continuous and discrete halves | Smith §2.4 | Week 9 |
| 3 | **Neural process theory** — what the update equations claim about firing rates, pyramidal populations, and neuromodulation as precision | Smith §5; PPF ch. 5 | Week 5 |
| 4 | **Fitting models to empirical data** — model inversion on behaviour, Bayesian model comparison, PEB | Smith §7; PPF ch. 9 | Week 11 |
| 5 | **A recipe for designing your own model** — the practical construction methodology | PPF ch. 6 | Week 9 |
| 6 | **Hierarchical message passing** — forward and backward passes in the continuous hierarchy | (continuous-time literature) | Week 8 |

Plus one mislabelling: Week 9 said "variational message passing" where the scheme
discrete active inference actually uses, and which `pymdp` implements, is the
*marginal* variant.

## Did the prediction hold?

Gaps landed on Weeks **5, 8, 9 (three of them), 11**. Weeks **4 and 10 were clean**.

Sharper than predicted, and the sharper statement is the useful one:

> **Every week with zero derivation targets had a gap. Every week with one or
> more had none.** Perfect separation across all twelve.

Weeks 4 and 10 (three derivation targets each) and Weeks 2, 3, 6 (one or two)
survived the audit untouched. Weeks 5, 7, 8, 9, 11, 12 (zero) did not, except
Week 7, which has no *missing* topic but remains ungrounded and should not be
trusted on that basis.

The count of named derivation targets is therefore a usable predictor of whether
a week has been thought about or merely named. `build/check_outline.py` enforces
it. It remains a proxy: it detects a bullet that *looks* committed, not one that
*is*. Only reading the source does that.

## What the missing six have in common

Four of the six (neural process theory, data fitting, the recipe, prediction-error
formulation) are the topics that connect the mathematics to something outside
itself: to neurons, to data, to a reader's own problem. A course built from
chapter headings by a mathematician reproduces the mathematics and quietly drops
the connections, because the connections are what chapter titles compress worst.

Worth watching for when Weeks 5 to 12 get drafted.
