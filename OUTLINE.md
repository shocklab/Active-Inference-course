# Active Inference — twelve-week outline

Status key: **built** = written and published · *drafted* = outline only · (blank) = planned.

Ordering follows Namjoshi rather than Parr, Pezzulo and Friston: classical
inference, then continuous time, then the discrete POMDP formulation last. The
reason is in `content/orientation.md` and is load-bearing, so do not reorder
weeks without revisiting it.

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
- Fixed-form: the Laplace approximation, which sets up Week 5.
- **The mean-field failure demo**: rebuild the Mathematica notebook's final example
  properly, with the exact posterior computed alongside. See
  `notes/mathematica-notebook-diagnosis.md`. Shows that mean-field underestimates
  spread and fails when causes compete, and connects back to Week 1 Lesson 4.
- Widgets: drag `Q` and watch all three decompositions update live; free-form vs
  fixed-form convergence; the mean-field gap on a correlated posterior.
- Notebooks: NumPy, JAX.

---

## Part II — Continuous time

### Week 5 · Predictive coding
- Free energy under a Laplace approximation becomes precision-weighted prediction error.
- A single unit; then multivariate; then hierarchical.
- Precision as the formal correlate of attention.
- Widgets: a PC unit relaxing to its fixed point; a hierarchy with precision sliders; what happens when precision is set wrong.

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
- The forward model and the action-perception cycle.
- Learning first and second-order parameters; attention as precision optimisation.
- Widgets: an agent holding a setpoint by acting; break the forward model and watch it fail.

---

## Part III — Discrete time

### Week 9 · Discrete generative models
- POMDPs; the arrays A, B, C, D, E and exactly what each index runs over.
- Static inference, then dynamic; observation and belief time indexing (a place the literature is careless).
- Variational message passing for state estimation.
- Widgets: build an A matrix and see its ambiguity; belief propagation along a trajectory.
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

### Week 11 · Policy selection and learning
- `softmax(-γG)`; precision γ and what it does to behaviour.
- Dirichlet learning of A and B; habit learning via E.
- Factorial and hierarchical depth.
- Widgets: a full agent in a small world, learning across trials; the same agent with γ swept.

---

## Part IV — Perspective

### Week 12 · Extensions, connections and objections
- Sophisticated inference; structure learning; factor graphs in brief.
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
