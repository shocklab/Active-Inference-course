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

Directionally right, though Week 7 was predicted to have a gap and did not.
Enumerating the whole set rather than the memorable half:

| | weeks | gaps found |
|---|---|---|
| one or more derivation targets | 2, 3, 4, 6, 10 | **none** |
| zero derivation targets | 5, 7, 8, 9, 11, 12 | 5, 8, 9, 11 |

So the screen has **no false negatives**: every gap sat in a week naming nothing
to derive. It has **two false positives out of six**: Weeks 7 and 12 named nothing
to derive and still had no missing topic.

Naming a derivation target was therefore *sufficient* to avoid a gap in this
sample, and its absence was *necessary but not sufficient*. That is exactly the
property wanted from an audit trigger: it never misses, and the cost of its
over-flagging is a second look at two weeks.

(An earlier draft of this note claimed "perfect separation across all twelve".
That was written without enumerating the set the claim covered, which is the
same failure mode as the one being audited.)

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
