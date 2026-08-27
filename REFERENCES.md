# Sources

Every entry below was verified against Crossref on 2026-08-26: the DOI resolves,
and the title, journal, volume, pages and author list are as printed here. Nothing
in this file was written from memory. Two errors were caught doing this, both mine:
the pymdp author list (no Fountas, no Da Costa) and an assumed DOI for Bogacz.

**Licence column** says what we may do, not merely whether it is free to read.
See `CLAUDE.md` for the policy. In short: cite and link anything; quote briefly with
attribution; never adapt or reproduce figures, tables or code from a NoDerivatives
work into this CC BY course.

## Primary references

| Work | Where | Licence |
|---|---|---|
| Parr, T., Pezzulo, G. & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press. | [OA at MIT Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind) | CC BY-NC-ND |
| Smith, R., Friston, K. J. & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *J. Math. Psych.* **107**, 102632. | doi:[10.1016/j.jmp.2021.102632](https://doi.org/10.1016/j.jmp.2021.102632) | CC BY-NC-ND |

The course exists because these two are strong on ideas and thin on working. We
cite them by equation number and fill the gaps ourselves.

## By week

### Weeks 1–4 · Inference

| Work | Why | Where |
|---|---|---|
| Buckley, C. L., Kim, C. S., McGregor, S. & Seth, A. K. (2017). The free energy principle for action and perception: A mathematical review. *J. Math. Psych.* **81**, 55–79. | **Read in full 2026-08-26**; Weeks 5–8 rewritten against it. The mathematical review of the continuous case, and the closest existing work to this course's purpose. Source of the moving-frame gradient descent result that is now Week 7's centrepiece. | doi:[10.1016/j.jmp.2017.09.004](https://doi.org/10.1016/j.jmp.2017.09.004) · [arXiv:1705.09156](https://arxiv.org/abs/1705.09156) |
| Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. *J. Math. Psych.* **76**, 198–211. | Step-by-step derivations from very simple examples, plus how the model sits in neural circuits. The register this course is aiming for. | doi:[10.1016/j.jmp.2015.11.003](https://doi.org/10.1016/j.jmp.2015.11.003) |

### Week 1 · Why exact inference is out of reach

| Work | Why | Where |
|---|---|---|
| Cooper, G. F. (1990). The computational complexity of probabilistic inference using Bayesian belief networks. *Artificial Intelligence* **42**, 393–405. | Exact inference in a general Bayesian network is NP-hard. Upgrades "the sum is long" into a statement about every possible algorithm. | doi:[10.1016/0004-3702(90)90060-D](https://doi.org/10.1016/0004-3702(90)90060-D) |
| Dagum, P. & Luby, M. (1993). Approximating probabilistic inference in Bayesian belief networks is NP-hard. *Artificial Intelligence* **60**, 141–153. | The one that closes the door: conceding exactness does not buy tractability either. | doi:[10.1016/0004-3702(93)90036-B](https://doi.org/10.1016/0004-3702(93)90036-B) |
| Roth, D. (1996). On the hardness of approximate reasoning. *Artificial Intelligence* **82**, 273–302. | Hardness survives even for crude answers with high probability. | doi:[10.1016/0004-3702(94)00092-1](https://doi.org/10.1016/0004-3702(94)00092-1) |
| Valiant, L. G. (1979). The complexity of computing the permanent. *Theoretical Computer Science* **8**, 189–201. | Where #P comes from. The marginal is a counting problem of this kind. | doi:[10.1016/0304-3975(79)90044-6](https://doi.org/10.1016/0304-3975(79)90044-6) |

### Weeks 5–8 · Continuous time

| Work | Why | Where |
|---|---|---|
| Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience* **2**, 79–87. | Where predictive coding starts. Week 5 should say what it got right before generalising it. | doi:[10.1038/4580](https://doi.org/10.1038/4580) |
| Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc. B* **360**, 815–836. | The bridge from predictive coding to free energy, and the origin of the neural process theory in Week 5. | doi:[10.1098/rstb.2005.1622](https://doi.org/10.1098/rstb.2005.1622) |
| Buckley et al. (2017), above | Generalised coordinates and generalised filtering, done carefully. | |

### Weeks 9–11 · Discrete time

| Work | Why | Where |
|---|---|---|
| Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V. & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. *J. Math. Psych.* **99**, 102447. | The rigorous discrete treatment. Should be the spine of Weeks 9–11 in place of a private draft. | doi:[10.1016/j.jmp.2020.102447](https://doi.org/10.1016/j.jmp.2020.102447) · [arXiv:2001.07203](https://arxiv.org/abs/2001.07203) |
| Parr, T., Marković, D., Kiebel, S. J. & Friston, K. J. (2019). Neuronal message passing using mean-field, Bethe, and marginal approximations. *Sci. Rep.* **9**, 1889. | The three-schemes comparison that Week 9 was missing. Settles that discrete active inference uses the *marginal* scheme, not VMP. | doi:[10.1038/s41598-018-38246-3](https://doi.org/10.1038/s41598-018-38246-3) · CC BY |
| Friston, K., Parr, T. & de Vries, B. (2017). The graphical brain: Belief propagation and active inference. *Network Neuroscience* **1**, 381–414. | Factor graphs, which now live in Week 9. Fully open access. | doi:[10.1162/netn_a_00018](https://doi.org/10.1162/netn_a_00018) · CC BY |
| Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P. & Pezzulo, G. (2017). Active inference: A process theory. *Neural Computation* **29**, 1–49. | The canonical statement of the discrete scheme and its neuronal reading. | doi:[10.1162/NECO_a_00912](https://doi.org/10.1162/NECO_a_00912) |
| Parr, T. & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics* **113**, 495–513. | The generalised free energy, and why the standard EFE is one choice among several. | doi:[10.1007/s00422-019-00805-w](https://doi.org/10.1007/s00422-019-00805-w) |
| Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D. & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. *JOSS* **7**(73), 4098. | The library the Week 9 notebooks validate against. | doi:[10.21105/joss.04098](https://doi.org/10.21105/joss.04098) · [arXiv:2201.03904](https://arxiv.org/abs/2201.03904) |

### Week 10 · Expected free energy — including the case against it

| Work | Why | Where |
|---|---|---|
| Millidge, B., Tschantz, A. & Buckley, C. L. (2021). Whence the expected free energy? *Neural Computation* **33**(2), 447–482. | Argues the EFE is **not** simply "free energy in the future", and that exploration does not fall out of free energy minimisation as usually claimed. Week 10 should engage this rather than assert the standard story. | doi:[10.1162/neco_a_01354](https://doi.org/10.1162/neco_a_01354) |

### Continuous time, further

| Work | Why | Where |
|---|---|---|
| Friston, K., Trujillo-Barreto, N. & Daunizeau, J. (2008). DEM: A variational treatment of dynamic systems. *NeuroImage* **41**, 849–885. | Dynamic expectation maximisation: the machinery behind Weeks 6–7 beyond what Buckley et al. cover. | doi:[10.1016/j.neuroimage.2008.02.054](https://doi.org/10.1016/j.neuroimage.2008.02.054) |
| Friston, K., Daunizeau, J., Kilner, J. & Kiebel, S. J. (2010). Action and behavior: a free-energy formulation. *Biological Cybernetics* **102**, 227–260. | The action side of Week 8, including the reflex-arc delegation of the inverse model. | doi:[10.1007/s00422-010-0364-z](https://doi.org/10.1007/s00422-010-0364-z) |

### Week 12 · Perspective and objections

| Work | Why | Where |
|---|---|---|
| Sajid, N., Ball, P. J., Parr, T. & Friston, K. J. (2021). Active inference: Demystified and compared. *Neural Computation* **33**, 674–712. | The careful comparison with reinforcement learning: what active inference adds and what it renames. | doi:[10.1162/neco_a_01357](https://doi.org/10.1162/neco_a_01357) |
| Da Costa, L., Sajid, N., Parr, T., Friston, K. & Smith, R. (2023). Reward maximization through discrete active inference. *Neural Computation* **35**, 807–852. | When active inference and reward maximisation coincide, and when they do not. | doi:[10.1162/neco_a_01574](https://doi.org/10.1162/neco_a_01574) |
| Friston, K., Da Costa, L., Hafner, D., Hesp, C. & Parr, T. (2021). Sophisticated inference. *Neural Computation* **33**, 713–763. | Planning over beliefs about beliefs. The main extension named in Week 12. | doi:[10.1162/neco_a_01351](https://doi.org/10.1162/neco_a_01351) |
| Friston, K., Thornton, C. & Clark, A. (2012). Free-energy minimization and the dark-room problem. *Frontiers in Psychology* **3**, 130. | The published statement of the dark-room objection and the standard reply. Week 12 should assess whether the reply works. | doi:[10.3389/fpsyg.2012.00130](https://doi.org/10.3389/fpsyg.2012.00130) · CC BY |
| Bruineberg, J., Dołęga, K., Dewhurst, J. & Baltieri, M. (2022). The emperor's new Markov blankets. *Behav. Brain Sci.* **45**, e183. | The strongest published objection to the Markov blanket picture drawn in Week 1. Cited there already. | doi:[10.1017/S0140525X21002351](https://doi.org/10.1017/S0140525X21002351) · [preprint](https://philsci-archive.pitt.edu/19726/) |

## Still to find

- ~~A published source for the **dark-room objection**~~ — found: Friston, Thornton
  & Clark (2012), above.
- A solid reference for the **ergodicity assumption** behind Week 1's
  `H = lim (1/T) Σ surprise`, and the published criticism of it. Currently flagged
  as a `::: warning` with no citation behind it.
- **Dynamic expectation maximisation** and the generalised filtering machinery of
  Weeks 6–7, beyond what Buckley et al. cover.
- **Structure learning**, named in Week 12 with nothing behind it.
- A **computational psychiatry** application for Week 11's data-fitting section,
  to make the model-fitting concrete on a real dataset.

## A note on using preprint sources

The arXiv source of Buckley et al. (1705.09156) contains an abandoned subsection on
the dark-room problem, inside an `\iffalse` block, carrying a co-author's note saying
"we can leave it for a later paper". It is not in the published article. Draft material
the authors chose not to publish is not a citable source, however interesting, and it
is not used here. The published dark-room reference above is used instead.

## Verifying this file

```bash
python3 build/check_references.py
```

Re-resolves every DOI against Crossref and compares the stored title, year, volume,
pages and author surnames. Run it before publishing anything that cites.
