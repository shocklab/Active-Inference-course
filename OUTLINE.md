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
2. Generative process and generative model · generative process vs generative model, the Markov blanket, why the intractability is the agent's problem. SVG diagram.
3. Bayes with states and observations · the four terms, rows vs columns, a fully worked numerical example, ambiguity. Widgets: `bayes-discrete`.
4. The inverse problem · many-to-one, the normaliser, ill-posedness, explaining away, the three solution families. Widgets: `forward-inverse`, `evidence-blowup`.
5. What makes the denominator hard · added 2026-08-27, because "the sum is
   intractable" was doing more work than one sentence can carry. Four distinct
   reasons, separated: the term count (which proves almost nothing, since a chain
   uses backward messages to compute the likelihood of an observed outcome); treewidth, which is what actually sits
   in the exponent, and which explaining away destroys; the complexity results,
   which rule out cleverer algorithms and not merely slower machines; and the
   analytic failure of the continuous case, visible with one variable and nothing
   to do with dimension. Then quadrature and importance sampling, and where each
   breaks. Closes on a table of which escape dodges which reason, which is where
   the mean-field approximation of Week 4 gets its price tag.
   → forward to Week 2 §2 and Week 4.
6. Problems and code · eleven problems with solutions, notebooks. Includes a
   two-state warm-up in odds form, posterior entropy and the identity
   $\mathbb{E}_o\mathrm{H}[P(s\mid o)] = \mathrm{H}[P(s)] - I(s;o)$, both forms of
   mutual information with its convexity in the channel, the regulated walk
   $\kappa>0$ solved analytically, and an implementation problem.

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
- **Distinguish objectives.** Write the log joint as $\ell$, not variational free
  energy. A point mass in a continuous state space has infinite KL divergence
  relative to a smooth posterior. A narrow fixed-entropy approximation can
  motivate MAP, but is not a finite-free-energy point mass.
- **Two things the drafting turned up that were not planned.** The step size is
  bounded by $2/|\mathcal{F}''|$, and a sharper sensor stiffens the curvature, so
  trusting your senses more makes the inference harder to run. And the MAP estimate
  is not invariant under reparameterisation: estimating $\ln d$ instead of $d$ moves
  the answer by 21% on this week's numbers, while the expectation of the same physical distance is invariant when transformed correctly. Both are in the lessons rather than saved for Week 12.
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
*Rewritten 2026-08-26 after reading Bogacz (2017) in full. Revised against Bogacz
§2.4–2.5 and §5.1 and Buckley et al. §1 and §8.4 and **drafted 2026-11-09**; four
lessons, a problems page, three widgets, two notebooks. Revised the same day against a
correctness audit and a depth audit (`handoffs/audits/2026-11-09-week03-*.md`).*
- **One objective, two timescales.** Week 2 climbed the log joint $\ell$ in the state
  $d$. Learning climbs the same $\ell$ in the parameters $d_p, \Sigma_p, \Sigma_u,
  \theta$, one small step per trial, at the settled state. **Derive** the envelope
  theorem, which is why the partial derivative at the mode is the slope of the settled
  log joint $\hat\ell$, and check numerically that it fails when inference is stopped
  early. **Derive** the linear case as an exponential running average with
  $\kappa = \beta/(\Sigma_p + \Sigma_u)$, Week 1's $\kappa$ reused, unbiased for the
  location in the linear world. **Derive** the yardstick: linearising a rule's drift
  $\varphi(\omega)$ about its zero gives the same recursion with
  $\kappa = -\beta\varphi'$, so near an attracting zero of its drift a rule fluctuates by
  order $\sqrt\beta$, its stationary mean can be displaced by order $\beta$, and one
  run's average carries its own sampling error. Warn that settling the states is alternating ascent on the same total
  log joint, which has no global maximum. → back to Week 2.
- Why $\hat\ell$ and not the evidence $p(u)$: the same reason as last week. The
  evidence needs the integral; the log joint needs one point.
- **Derive** the updates for the prior location and both variances, with the
  truncation normaliser kept: $\partial Z_p/\partial d_p = \mathcal N(0)$ and, through the
  heat equation, $\partial Z_p/\partial\Sigma_p = -(d_p/2\Sigma_p)\mathcal N(0)$. Each
  stationary condition matches a moment of the inferred states to the same moment of
  the model. Variance, precision and log-variance updates share the zero of their
  drift, not their behaviour near zero or their finite-rate rest point; Bogacz's §5.1
  interneuron rule is $2\Sigma_u$ times the gradient. Warning: the plain variance step
  crosses zero below $\sqrt{\beta/2}$; log coordinates keep variances positive but do
  not bound the steps.
- **Derive the Hebbian payoff.** With $g = \theta h(d)$,
  $\partial\ell/\partial\theta = \varepsilon_u h(d)$: postsynaptic error times presynaptic
  activity, Week 2's local plasticity met (Bogacz §2.5); the prior location's rule is
  the same pattern plus the truncation correction $-\zeta_p$. Warning: $\theta$ sits on
  two connections, identical updates preserve any difference between them, and unequal
  strengths act as a sensory variance $\Sigma_u\theta_\downarrow/\theta_\uparrow$.
  **Derive** the scale symmetry $(\theta, d_p, \Sigma_p) \to (c^2\theta, c\,d_p, c^2\Sigma_p)$:
  the three are not identifiable from sound alone, and the settled log joint falls by
  $\ln c$ along it, so joint point learning has no resting point.
- **What drafting found.** The point rules, which read the gradient at the mode, are
  biased even when the model is exactly right: on Week 2's model by
  $-32\%$ ($d_p$), $-62\%$ ($\Sigma_p$), $-73\%$ ($\Sigma_u$) and $+12\%$ ($\theta$).
  **Derive** the Fisher identity and the mean-zero score, so the posterior-averaged
  rule is unbiased; **derive** the linear case exactly, where the learned prior
  variance collapses to zero unless $\Sigma_p^{*} \geq 3\Sigma_u$, and the missing term
  is $\tfrac12\ln(2\pi\Sigma_{\text{post}})$. **Derive** the mechanism: a second-order
  expansion of the per-trial derivative about the mode separates the offset of the mode from the mean
  from the omitted posterior variance, exact for the prior location and variance; for
  $\theta$ the two pull opposite ways and the expansion gets the louder world's sign
  wrong, so the signs come from the full calculation; in general $\hat\ell = \ln p(u) + \ln p(\hat d \mid u)$.
  → forward to Week 4, whose free energy restores the missing term.
- Precision as the thing learned: **derive** the ear's share from the linearised mode.
  Under a channel three times noisier, the point rule's drift vanishes just above the
  starting variance, so the ear's share barely moves while it should fall by more than half.
  Hyperpriors as inverse-gamma pseudo-trials, which bound the objective without
  removing the bias. → forward to Week 8, attention.
- Expectation maximisation as the classical name for the posterior-averaged rule. A
  connection, not the spine. Week 4 proves the monotone ascent.
- Widgets: `two-timescales` (one trial's inner loop beside the prior location across
  trials); `wrong-link` (a world louder than the model, with window averages of both
  errors and the learning signal, and learning switchable off); `precision-learning`
  (point and posterior-averaged rules on the same trials, with the ear's share of the
  curvature). The widgets run the same seeded engine as `values.py`; readouts agree to
  about $10^{-10}$.
- Notebooks: NumPy (the engine, derivative checks, the three experiments, the linear
  roots); JAX (autodiff against the hand derivatives over ten thousand parameter sets,
  a thousand learners at once, the expected update swept for both rules). No `pymdp`.
- Sources: Bogacz (2017) §2.4–2.5, §5.1 **[read]**; Buckley et al. (2017) §1, §8.4
  **[read]**.

### Week 4 · Variational free energy — **the pivot of the course** — **built**
*Drafted 2026-11-12: five lessons (bounding surprise; mean-field and messages; the
Laplace approximation; learning by EM; the two-cause model) and a problems page, with
NumPy and JAX notebooks, checked by a correctness audit and two rechecks. The two-cause
lesson is the mean-field failure demo below.*
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
  doi:10.1038/s41598-018-38246-3; Smith, Friston & Whyte §2.3. Parr et al. (2019) was
  read only far enough to check Week 4's one-paragraph forward reference to it; read it
  in full before deriving the marginal scheme here.
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
- [x] Exercise coverage gaps in Week 1, closed 2026-11-04 by five new problems:
      a warm-up, entropy, mutual information, $\kappa > 0$ analytically, and code.

Open:
- [ ] Glossary, built from the `::: notation` blocks the way the sibling course builds `glossary.json`.
- [ ] Decide whether Weeks 6–8 need a shared worked continuous example carried across all three.
- [ ] Run the four audits per week **as each week is drafted**, not afterwards.
      Week 1 needed fifteen agent runs to reach its current state; catching the
      same faults at draft time is far cheaper than retrofitting them.
