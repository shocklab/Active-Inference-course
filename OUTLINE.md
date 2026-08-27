# Active Inference — twelve-week outline

Status key: **built** = written and published · *drafted* = outline only · (blank) = planned.

Ordering: classical inference, then continuous time, then the discrete POMDP
formulation last. This is the reverse of the commonest presentation. The argument
for it is our own and is set out in `content/orientation.md`. The whole sequence
depends on it, so do not reorder weeks without revisiting that argument.

Sources named below are set out in full in `REFERENCES.md`, where every DOI has been
resolved against Crossref. See `CLAUDE.md` for what may and may not be taken from each.


## Reading status — read this before trusting the sources above

`build/check_outline.py` now reports all twelve weeks as carrying a source, a
derivation target, widget specs and a cross-link. **That is a proxy result and
should not be read as "all twelve weeks are planned."**

The audit's own finding was that outline quality tracks whether the source was
*read* or merely *named*. Wiring in a citation satisfies the checker without
satisfying the thing the checker stands for. Current state, honestly:

| Source | Status | Weeks resting on it |
|---|---|---|
| Jonathan's `main.tex` and notebook | **read in full** | 1, 4, 10 |
| Smith, Friston & Whyte (2022) | section structure extracted; body **not read** | 9, 11 |
| Parr, Pezzulo & Friston (2022) | chapter structure only; body **not read** | 5, 9, 11 |
| Buckley et al. (2017) | **read in full** (2026-08-26) | 2, 3, 5, 6, 7, 8 |
| Bogacz (2017) | **read in full** (2026-08-26) | 2, 3, 5 |
| Da Costa et al. (2020) | **existence verified only** | 9, 10, 11 |
| Millidge, Sajid, Da Costa, Friston et al. (Week 12 set) | **abstracts and arguments read** (2026-08-26); full derivations not worked | 10, 12 |
| everything else in `REFERENCES.md` | **existence verified only** | 5, 9 |

So Weeks 1, 4, 5, 6, 7, 8 and 10 are genuinely grounded: Buckley et al. was read
in full on 2026-08-26 and Weeks 5 to 8 were rewritten against it, which changed
them substantially rather than cosmetically. Weeks 9 and 11 are grounded in
structure but not content. Weeks 2 and 3 were rewritten against Bogacz, and Week 12 against its four
sources, on 2026-08-26. Weeks 9 and 11 remain grounded in structure only: the
Da Costa (2020) discrete synthesis has not been read, and it is the next one that
should be.

What reading one paper changed, as a calibration of how much the unread weeks are
worth: it added the Laplace-encoded energy to Week 5, replaced Week 6's postulated
derivatives with a motivated derivation plus the unstated local-linearity
assumption, gave Week 7 a central result it entirely lacked (that plain gradient
descent has no stationary point in generalised coordinates), and corrected Week 8
from "forward model" to "inverse model", which was simply wrong.

**Before drafting any week, read its sources and revise its bullets against what
is actually in them.** The gaps found on 2026-08-26 were all of the form "the
source treats this substantially and the outline does not mention it", which only
reading finds. Update this table as sources are read, and do not delete it.

---

## Part I — Inference

### Week 1 · The hypothesis-testing agent — **built**
1. A soft body in a hard world · persistence, viability sets, the entropy of sensory states, surprise, the two routes to reducing it. Widget: `homeostasis-drift`.
2. Two models, not one · generative process vs generative model, the Markov blanket, why the intractability is the agent's problem. SVG diagram.
3. Bayes with states and observations · the four terms, rows vs columns, a fully worked numerical example, ambiguity. Widgets: `bayes-discrete`.
4. The inverse problem · many-to-one, the normaliser, ill-posedness, explaining away, the three solution families. Widgets: `forward-inverse`, `evidence-blowup`.
5. Problems and code · six problems with solutions, notebooks.

### Week 2 · Hidden state estimation
*Rewritten 2026-08-26 after reading Bogacz (2017) in full. **Drafted 2026-08-27**;
five lessons, two widgets, two notebooks.*
- **Open with the constraints, not the maths.** Two rules a physical nervous system
  has to obey: **local computation** (a unit computes only from its own inputs and
  their weights) and **local plasticity** (a connection changes only from the
  activity at its two ends). Everything in Weeks 2, 3 and 5 is an attempt to meet
  them. Stating them first turns the derivations into answers rather than exercises.
- A concrete problem: one hidden variable, one noisy observation, and a **nonlinear**
  link between them. Nonlinear on purpose; the linear case hides everything.
- Exact inference by Bayes, then the two reasons a body cannot do it: a nonlinear
  link makes the posterior non-standard, so it cannot be carried as a mean and a
  variance, and the normaliser is an integral. This is the **continuous** counterpart
  of Week 1's combinatorial blow-up, and the lesson should say so.
  → back to Week 1 §4.
- **Derive the central result.** Maximise $F = \ln p(\phi) + \ln p(u \mid \phi)$, the
  log joint, whose maximum in $\phi$ is the maximum of the posterior because the
  intractable $p(u)$ does not depend on $\phi$. Then
  $\partial F/\partial\phi$ is **two precision-weighted prediction errors**: one
  against the prior, one against the data, the second passed back through $g'$.
  Predictive coding arrives here, in Week 2, out of nothing more than gradient
  ascent on a log joint. → forward to Week 5, which is this result generalised.
- **Careful: $F$ here is not yet the free energy.** It is the log joint, and equals
  the negative variational free energy only when the approximate posterior is a
  point mass. Say so at the point of use; Week 4 says what the point mass costs.
- **Two things the drafting turned up that were not planned.** The step size is
  bounded by $2/|\mathcal{F}''|$, and a sharper sensor stiffens the curvature, so
  trusting your senses more makes the inference harder to run. And the MAP estimate
  is not invariant under reparameterisation: estimating $\ln d$ instead of $d$ moves
  the answer by 21% on this week's numbers, while the posterior mean does not move
  at all. Both are in the lessons rather than saved for Week 12.
  → forward to Week 12, which collects the objections.
- Widgets: `ascent-errors`, gradient ascent with the two error terms drawn
  separately so the reader watches them cancel rather than vanish; and
  `precision-posterior`, the posterior itself under both precisions with mode and
  mean marked, and a switch to the linear link that makes the skew collapse.
  Posterior contraction with accumulating observations is in the JAX notebook
  instead of a widget, where the sufficient-statistic argument can be shown.
- Notebooks: NumPy (the week one run at a time); JAX (the hand-derived gradient
  checked against `jax.grad`, ten thousand starting points mapping the basins of a
  bimodal posterior, and the precision plane swept at once). No `pymdp`: nothing
  discrete here.
- Sources: Bogacz (2017) §1–2 **[read]**; Buckley et al. (2017) §2–4 **[read]**.

### Week 3 · Learning the model
*Rewritten 2026-08-26 after reading Bogacz (2017) in full.*
- **The organising idea: one objective, two timescales.** Week 2 climbed $F$ in
  $\phi$. Learning climbs the same $F$ in the model's parameters. Inference and
  learning are not different mechanisms; they are the same ascent on different
  arguments, fast for states and slow for parameters. Framing it this way makes
  Week 3 a corollary of Week 2 rather than a fresh topic.
  → back to Week 2.
- Why maximise $F$ rather than $p(u)$: the same reason as last week. The marginal is
  an intractable integral; the joint is a product of two things you already have.
- **Derive** the updates for the prior mean and for both variances,
  $\partial F/\partial v_p$, $\partial F/\partial\Sigma_p$,
  $\partial F/\partial\Sigma_u$, and read off what each is doing: the mean chases
  the inferred state, a variance grows when its error is persistently larger than it.
- **Derive the payoff: learning comes out Hebbian.** With a tunable link
  $g(v,\theta) = \theta v$, the gradient is $\partial F/\partial\theta = \varepsilon\,\phi$,
  the prediction error times the presynaptic activity. That is exactly the local
  plasticity constraint Week 2 opened with, satisfied without having been aimed at.
  This is the moment the framework stops looking like statistics and starts looking
  like a brain, and it belongs here rather than being deferred to Week 5.
- Precision as the thing being learned, which is where attention enters; and
  hyperpriors, meaning priors over the precisions themselves.
- Expectation maximisation as the classical name for this two-timescale split, for
  readers who already know it. A connection, not the spine: a mathematics postgraduate
  does not need a week on linear regression.
- Widgets: the two timescales running at once, states settling within a trial and
  parameters drifting across trials; learn a wrong $g$ and watch what the errors do;
  precision learning under a noisy channel.
- Notebooks: NumPy, JAX.
- Sources: Bogacz (2017) §2.4–2.5 and §5 **[read]**; Buckley et al. (2017) §8 **[read]**.

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
*Revised 2026-08-26 after reading Buckley et al. (2017) in full.*
- **Derive the Laplace-encoded energy.** Assume the approximate posterior is Gaussian
  and that its mode is sharply peaked, and free energy collapses to an expression in
  the generative model alone. This is the named bridge from Week 4's general bound to
  prediction errors, and it is the step the Gaussian assumption exists for.
  → back to Week 4 (fixed-form mean-field).
- **Derive** the single-unit update from that, then multivariate. Univariate first
  throughout, with the multivariate expression stated at the end of each section:
  Buckley et al. use this pattern and it works.
- Free energy becomes a sum of squared prediction errors weighted by inverse variances.
  Precision as the weight, and the psychiatric reading of getting it wrong.
- Hierarchical predictive coding; empirical priors.
- **Neural process theory.** Which populations carry predictions and which carry errors,
  why the scheme predicts distinct superficial and deep pyramidal roles, and precision
  as neuromodulatory gain. Missed in the first draft of this outline.
- **Predictive coding IS message passing** — errors up, predictions down, on a hierarchical
  graph; the same fixed-point iteration Week 9 runs on a discrete graph.
  → back to Week 4; forward to Weeks 8 and 9.
- Terminology note: Buckley et al. write R-density and G-density for the approximate
  posterior and the generative model. Worth naming once so their paper is readable.
- Widgets: a PC unit relaxing to its fixed point; a hierarchy with precision sliders;
  what happens when precision is set wrong.
- Sources: Buckley et al. (2017) §4–6 **[read]**; Bogacz (2017); Rao & Ballard (1999);
  Friston (2005); Parr, Pezzulo & Friston ch. 5.

### Week 6 · Generalised coordinates of motion
*Revised 2026-08-26 after reading Buckley et al. (2017) in full.*
- **Motivate the derivatives, do not postulate them.** Generalised coordinates should
  arrive as what happens when you extend a *static* generative model to a dynamic
  environment, by adding a Langevin equation $\dot\mu = f(\mu) + w$ and then
  differentiating repeatedly. Introducing $(\mu, \mu', \mu'', \dots)$ as a formalism
  first is how the books lose people.
- **Derive** the generalised observation and state equations by recursive differentiation
  of $\varphi = g(\mu) + z$ and the Langevin equation.
- **The local linearity assumption** *(found by reading, absent from the first draft)*.
  The recursion drops nonlinear derivative terms such as $(\mu')^2$ and $\mu'\mu''$.
  The derivation depends on it and almost nobody states it. Say what it costs.
- **Two conventions for the noise** across dynamic orders: independent, which is the
  standard state-space choice and the one to teach first, versus correlated, which is
  what most FEP treatments actually assume. Be explicit about which is in force and why.
  → forward to Week 7.
- The shift operator $\mathcal{D}$, defined here, used in anger next week.
- **Derive** the covariance of the generalised measurement vector, and show the useful
  embedding order is set by the smoothness of the noise rather than chosen freely.
- Widgets: embedding-order explorer; noise smoothness against useful embedding order;
  the local linearity assumption breaking as a trajectory curves harder.
- Sources: Buckley et al. (2017) §5.2 **[read]**; Friston et al. (2008), DEM.

### Week 7 · Generalised filtering
*Revised 2026-08-26. This week now has a real centrepiece it did not have before.*
- **Derive the week's main result: plain gradient descent cannot work here, and why.**
  In generalised coordinates the time derivative of the $n$th order is *already* the
  $(n+1)$th order, $\dot\mu_{[n]} = \mu_{[n+1]}$, by the definition of the coordinates.
  So $\dot\mu_{[n]}$ cannot vanish at any order, and ordinary gradient descent has no
  stationary point to converge to. The scheme is broken as usually written.
  The repair is to separate the **motion of the mode** $\dot{\tilde\mu}$ from the
  **mode of the motion** $\mathcal{D}\tilde\mu$, giving
  $\dot{\tilde\mu} - \mathcal{D}\tilde\mu = -\kappa\,\nabla_{\tilde\mu}E$,
  which is stationary when $\dot{\tilde\mu} = \mathcal{D}\tilde\mu$: the path of the
  mode equals the mode of the path. Setting $\mathcal{D} = 0$ recovers Week 5.
  This is the step everyone trips on, it is the reason the $\mathcal{D}$ term appears
  in every continuous-time active inference paper, and the first draft of this outline
  did not mention it at all. It alone justifies the week.
  → back to Week 6 ($\mathcal{D}$); back to Week 5 (the static case).
- Assembling the full generalised state-space model.
- Recognition dynamics: tracking a trajectory rather than a point.
- Correlated embedding orders, now that Week 6 has set up the choice.
- Widgets: the filter tracking a trajectory with tunable precisions; the same filter with
  the $\mathcal{D}$ term removed, failing to settle, so the necessity is visible rather
  than asserted; mis-set precisions and the filter chasing its own noise.
- Sources: Buckley et al. (2017) §6 **[read]**; Friston et al. (2008), DEM.

### Week 8 · Action, learning and attention
*Revised 2026-08-26 after reading Buckley et al. (2017) in full.*
- Autonomous states; action changes sensations rather than beliefs.
- **Derive the action update.** Action does not appear in the free energy at all; it
  enters only through the sensations it causes, so the chain rule gives
  $\dot a = -\kappa_a \,(\mathrm{d}\varphi/\mathrm{d}a)\,(\partial E/\partial\varphi)$.
  The agent therefore needs $\mathrm{d}\varphi/\mathrm{d}a$: how its sensations change
  with what it does. Buckley et al. call this an **inverse model**, and the first draft
  of this outline said "forward model", which is the wrong way round.
- **What that requirement costs, and the biological dodge.** The standard move is to
  delegate $\mathrm{d}\varphi/\mathrm{d}a$ to reflex arcs so the brain never has to
  learn it. Say plainly what that assumption is assuming.
- The action-perception cycle, and why one quantity can govern both without collapsing
  into wishful thinking (they act on opposite sides of the blanket).
  → back to Week 1 §2.
- Learning first and second-order parameters; attention as precision optimisation, which
  Buckley et al. tie to synaptic efficacy and gain respectively.
- **Hierarchical message passing**: forward and backward passes through the continuous
  hierarchy, written out. → forward to Week 9, same structure on a discrete graph.
- Widgets: an agent holding a setpoint by acting; break the inverse model and watch it
  fail; a thermostat-style agent as the worked continuous example, built from scratch.
- Sources: Buckley et al. (2017) §7–8 **[read]**; Friston et al. (2010), action and
  behaviour; Friston (2005).

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
  → back to Week 4 (F, given data in hand); forward to Week 11 (G, driving choice).
- **The case against the standard story.** Millidge et al. (2021) argue the EFE is not
  simply "free energy in the future" and that exploration does not fall out of free
  energy minimisation as usually claimed. Engage it here, not in Week 12.
- Sources: Jonathan's `main.tex`; Da Costa et al. (2020); Parr & Friston (2019);
  Millidge et al. (2021).
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
*Rewritten 2026-08-26 after reading the four sources below.*

**Part one: sophisticated inference.**
- Beliefs about beliefs. The ordinary agent asks "what would happen if I did that";
  the sophisticated one asks "what would I *believe* about what would happen if I
  did that". The expected free energy becomes recursive, and the resulting search
  runs over sequences of **belief states** rather than states.
- **Derive** the recursion to depth two, so the cost of the tree search is visible
  rather than asserted. → back to Week 10.
- Structure learning, in brief: what changes when the model's form is not given.

**Part two: the relation to reinforcement learning.** Not a survey; one precise
result and two honest observations.
- **The result to state exactly.** On partially observed Markov decision processes,
  standard discrete active inference produces Bellman-optimal actions for a planning
  horizon of **one, and not beyond**; sophisticated inference produces them on any
  finite horizon. That is the crispest available answer to "is this just
  reinforcement learning with extra steps", and it is a qualified no with the
  qualification stated.
- What active inference gets for free that reinforcement learning engineers in:
  epistemic exploration falls out of the same objective rather than being bolted on
  as an exploration bonus.
- What it renames rather than solves: reward becomes an observation the agent prefers,
  which relocates the reward-design problem into prior-preference design rather than
  removing it. Say this plainly.
- Widgets: an agent with a one-step horizon and a sophisticated one on the same
  task, with the horizon at which the shallow agent fails made visible.

**Part three: the objections, put properly.**
- **Does exploration really follow from free energy?** Millidge et al. show the
  expected free energy is *not* simply "the free energy in the future", and that the
  natural extension of variational free energy into the future actively *discourages*
  exploration. If that holds, the framework's most attractive selling point does not
  come from where it is usually said to come from. Week 10 raises this; Week 12 has
  to answer it or concede it.
- **The ergodicity assumption** behind Week 1's identification of lifetime entropy
  with per-moment surprise. → back to Week 1 §1, where it is flagged as a warning.
- **Markov blankets**: Bruineberg et al. on the difference between a blanket found
  in a system and one imposed on a model of it. → back to Week 1 §2, where the
  conditional independence is now written down.
- **The dark room**, and whether the standard reply works. The published exchange,
  not the folklore version.
- Collect every `::: warning` flagged in Weeks 1 to 11 and answer or concede each.
  That list is the week's real syllabus.
- Sources: Friston et al. (2021) sophisticated inference **[read: abstract and
  argument]**; Da Costa et al. (2023) on reward maximisation **[read: abstract and
  main result]**; Sajid et al. (2021) demystified and compared **[read: abstract and
  argument]**; Millidge et al. (2021) **[read: abstract and argument]**;
  Bruineberg et al. (2022); Friston, Thornton & Clark (2012) on the dark room.

---

## Cross-cutting to-dos

Done, and recorded so nobody redoes them:
- [x] Equation refs and widget registration are verified by `build/check_site.py`.
- [x] Numerical claims are substituted from `numbers.py` at build time, never typed.
- [x] A depth-audit and definition-sweep brief that works; both are in `CLAUDE.md`.

Open:
- [ ] Glossary, built from the `::: notation` blocks the way the sibling course builds `glossary.json`.
- [ ] Decide whether Weeks 6–8 need a shared worked continuous example carried across all three.
- [ ] **Exercise coverage gaps in Week 1**, found by the depth audit and not yet closed:
      no exercise computes an entropy; mutual information is never independently
      exercised; the regulated case $\kappa > 0$ is tested only by the widget and
      never analytically; there is no rung-zero warm-up before the first full
      three-state Bayes update; the reader is never asked to implement anything,
      despite the week shipping notebooks.
- [ ] Run the four audits per week **as each week is drafted**, not afterwards.
      Week 1 needed fifteen agent runs to reach its current state; catching the
      same faults at draft time is far cheaper than retrofitting them.
