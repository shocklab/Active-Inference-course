# Active Inference — twelve-week outline

Status key: **built** = written and published · *drafted* = outline only · (blank) = planned.

Ordering: classical inference, then continuous time, then the discrete POMDP
formulation last. This is the reverse of the commonest presentation. The argument
for it is our own and is set out in `content/orientation.md`; it is load-bearing,
so do not reorder weeks without revisiting it.

Sources named below are the two open-access references, which a reader can go and
check. See `CLAUDE.md` for what may and may not be taken from each.

---

## Part I — Inference

### Week 1 · The hypothesis-testing agent — **built**
1. A soft body in a hard world · persistence, viability sets, the entropy of sensory states, surprise, the two routes to reducing it. Widget: `homeostasis-drift`.
2. Two models, not one · generative process vs generative model, the Markov blanket, why the intractability is the agent's problem. SVG diagram.
3. Bayes with states and observations · the four terms, rows vs columns, a fully worked numerical example, ambiguity. Widgets: `bayes-discrete`.
4. The inverse problem · many-to-one, the normaliser, ill-posedness, explaining away, the three solution families. Widgets: `forward-inverse`, `evidence-blowup`.
5. Problems and code · six problems with solutions, notebooks.

### Week 2 · Hidden state estimation
- Exact inference where conjugacy allows it; the Gaussian-Gaussian case in full.
- Point estimates: MLE and MAP, analytically and by gradient ascent.
- The precision-weighted-average form of the Gaussian posterior mean, which is the seed of predictive coding in Week 5.
- Multiple observations and posterior contraction; why the posterior precision adds.
- Widgets: likelihood surface with live gradient ascent; posterior contraction as samples accumulate; conjugate updating.
- Notebooks: NumPy, JAX (vectorised posteriors over a grid of priors).

### Week 3 · Learning the model
- Parameters as another kind of hidden state; the difference between inference and learning is a timescale, not a kind.
- Bayesian linear regression; linear Gaussian systems.
- Expectation maximisation, derived rather than quoted, and the free energy lurking inside it.
- Factor analysis as the worked case.
- Widgets: EM iterations on a factor-analysis toy, with the responsibility assignment visible; the E-step/M-step alternation as a descent on one objective.
- Notebooks: NumPy, JAX.

### Week 4 · Variational free energy — **the pivot of the course**
- Jensen's inequality and the bound on surprise, in full.
- The three decompositions, each derived line by line, with what each is for:
  energy minus entropy · complexity minus accuracy · divergence minus evidence.
  (Jonathan's `main.tex` §"Chapter 2 details" is the source; expand, do not compress.)
- Free-form mean-field: the fundamental theorem, and the coordinate-ascent update.
- **Derive: the stationarity condition IS a message.** The free-form result
  $\ln q_i^*(s_i) = \mathbb{E}_{q_{-i}}[\ln p(s,o)] + \text{const}$ is exactly the
  variational message passing update; the message from a factor to a variable is
  the expected log of that factor. Name it here, where it costs two paragraphs,
  so Weeks 5, 8, 9 and 12 are instances rather than new machinery.
  → forward to Week 9; back to Week 1 §4 (explaining away).
- Fixed-form: the Laplace approximation, which sets up Week 5.
- **The mean-field failure demo**: rebuild the Mathematica notebook's final example
  properly, with the exact posterior computed alongside. See
  `notes/mathematica-notebook-diagnosis.md`. Shows that mean-field underestimates
  spread and fails when causes compete, and connects back to Week 1 Lesson 4.
- Widgets: drag `Q` and watch all three decompositions update live; free-form vs
  fixed-form convergence; the mean-field gap on a correlated posterior.
- Notebooks: NumPy, JAX.
- Sources: Jonathan's `main.tex`; `notes/mathematica-notebook-diagnosis.md`.

---

## Part II — Continuous time

### Week 5 · Predictive coding
- Free energy under a Laplace approximation becomes precision-weighted prediction error.
- **Derive** the single-unit update from Week 4's fixed-form result; then multivariate; then hierarchical.
- **Predictive coding IS message passing.** Prediction errors travelling up, predictions
  travelling down, on a hierarchical graph. Same fixed-point iteration as Week 9 will use
  on a different graph. Saying so is what makes the two halves of the course one course.
  → back to Week 4; forward to Weeks 8 and 9.
- **Neural process theory** *(audit gap 3)*. What the update equations claim about
  brains: which populations carry predictions and which carry errors, why the scheme
  predicts distinct superficial and deep pyramidal roles, and precision as
  neuromodulatory gain. This is a large part of why anyone outside mathematics cares,
  and it was missing entirely.
  Sources: Parr, Pezzulo & Friston ch. 5; Smith, Friston & Whyte §5.
- Widgets: a PC unit relaxing to its fixed point; a hierarchy with precision sliders;
  what happens when precision is set wrong (and the psychiatric reading of that).

### Week 6 · Generalised coordinates of motion
- Why a state is not enough: smooth noise means derivatives carry information.
- Generalised measurements; the covariance of derivatives; embedding order.
- The shift operator `D`.
- Widgets: embedding-order explorer; noise smoothness against useful embedding order.

### Week 7 · Generalised filtering
- Assembling the generalised state-space model.
- Recognition dynamics: tracking a trajectory rather than a point.
- Correlated embedding orders.
- Widgets: the filter tracking a trajectory with tunable precisions; deliberately mis-set precisions and watch it chase noise.

### Week 8 · Action, learning and attention
- Autonomous states; action changes sensations rather than beliefs.
- **Derive** the action update: why the gradient is taken with respect to sensory
  consequences, and what the forward model has to supply for that to be possible.
- The action-perception cycle.
- Learning first and second-order parameters; attention as precision optimisation.
- **Hierarchical message passing** *(audit gap 6)*: the forward and backward passes
  through the continuous hierarchy, written out. → forward to Week 9, where the same
  structure reappears on a discrete graph.
- Widgets: an agent holding a setpoint by acting; break the forward model and watch it fail.

---

## Part III — Discrete time

### Week 9 · Discrete generative models and message passing
*Rebuilt after the 2026-08-26 audit; this week carried three of the six gaps.*
- POMDPs; the arrays A, B, C, D, E and exactly what each index runs over.
- Static inference, then dynamic; observation and belief time indexing (a place the
  literature is careless, and worth being pedantic about).
- **Graphical models and factor graphs** — moved here from Week 12, where they were
  a throwaway. Needed *before* message passing, not after it.
- **Three approximations, three schemes** *(audit gaps 1 and 5)*. The week's main
  derivation. Mean-field gives **variational message passing**; Bethe gives **belief
  propagation**; the hybrid gives **marginal message passing**. Derive the VMP update
  from Week 4's theorem, derive the marginal update, and show precisely where they
  differ. Be explicit that discrete active inference, and `pymdp`, use the *marginal*
  scheme. The outline previously said "variational message passing" here, which named
  the wrong member of the family.
  Sources: Parr, Marković, Kiebel & Friston (2019), *Sci. Rep.* 9:1889,
  doi:10.1038/s41598-018-38246-3; Smith, Friston & Whyte §2.3.
- **The prediction-error formulation** *(audit gap 2)*. The discrete update rewritten
  as prediction errors, which is the explicit bridge back to Week 5. Without this the
  two halves of the course never meet. Source: Smith, Friston & Whyte §2.4.
  → back to Week 5.
- **A recipe for building your own model** *(audit gap 5)*. How to go from a task you
  care about to A, B, C, D and E. Source: Parr, Pezzulo & Friston ch. 6.
- Widgets: build an A matrix and see its ambiguity; watch the three schemes converge
  side by side on the same graph and disagree; belief propagation along a trajectory.
- Notebooks: NumPy, **pymdp** (first appearance), JAX.

### Week 10 · Expected free energy — **the second pivot**
- Risk plus ambiguity, and information gain plus pragmatic value, *proved* equal.
  Jonathan's `main.tex` ch. "More details on the expected free energy" has the
  derivation; it is the best thing in his notes and should be reproduced in full.
- The vectorised form `G = H·S + O·ζ`, with every index accounted for, including
  where τ goes. Jonathan's ch. "Thoughts on the vectorised expected free energy"
  resolves the A vs A-transpose question the book leaves ambiguous.
- Why F and G are different functionals with different arguments.
- Widgets: per-policy decomposition of G into its two terms; an explore/exploit dial.

### Week 11 · Policy selection, learning, and contact with data
- `softmax(-γG)`; precision γ and what it does to behaviour.
- **Derive** the Dirichlet update for learning A and B; habit learning via E.
- Factorial and hierarchical depth; deep temporal models.
- **Fitting models to behaviour** *(audit gap 4)*. Model inversion on empirical data,
  Bayesian model comparison, and parametric empirical Bayes. This is how active
  inference is actually used in computational psychiatry, and it was absent.
  Sources: Smith, Friston & Whyte §7; Parr, Pezzulo & Friston ch. 9.
  → back to Week 3 (learning parameters), Week 9 (the generative model being fitted).
- Widgets: a full agent in a small world, learning across trials; the same agent with
  γ swept; recovering known parameters from simulated behaviour, and watching recovery
  fail when the model is misspecified.

---

## Part IV — Perspective

### Week 12 · Extensions, connections and objections
- Sophisticated inference; structure learning. (Factor graphs moved to Week 9,
  where they are needed rather than mentioned.)
- Relation to reinforcement learning and control as inference: what active inference adds, and what it renames.
- The objections, put properly rather than as a footnote:
  the ergodicity assumption behind Week 1's `H = lim (1/T) Σ surprise`;
  Bruineberg et al. on Markov blankets; the falsifiability question;
  the dark-room problem and whether the standard reply works.
- Every `::: warning` flagged in Weeks 1–11 gets collected and answered here.

---

## Cross-cutting to-dos

- [ ] Glossary, built from the `::: notation` blocks the way the sibling course builds `glossary.json`.
- [ ] A `check_content.py` that verifies every `[eq:...]` resolves and every widget named in content is registered in a loaded JS file.
- [ ] Decide whether Weeks 6–8 need a shared worked continuous example carried across all three.
- [ ] Numerical claims in prose should be regenerated from the notebooks at build time, not typed by hand.
