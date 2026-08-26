---
title: The inverse problem
deck: Running a model forwards is a lookup. Running it backwards is a global computation over everything the model admits, and it does not decompose.
week: 1
time: 30 min
scripts: [w01.js]
---

A generative model is a machine for going from causes to consequences. Give it a
state, it hands you a distribution over observations. That direction is cheap:
it is what the model is *for*, and the answer is sitting in a column of a matrix.

Perception needs the other direction. Given the consequence, what was the cause?
This is called **model inversion**, and the asymmetry between the two directions
is not a matter of degree.

::: widget forward-inverse | One likelihood matrix, read two ways. Click any cell to change the selection. In forward mode you read a column and stop. In inverse mode you read a row, weight it by the prior, and then owe a sum over every state in the model.
:::

## Why the inverse direction is structurally harder

Three separate things go wrong at once, and it is worth keeping them apart
because different techniques attack different ones.

**It is many-to-one.** Several states can produce the same observation. In the
matrix above, a shaking branch is consistent with a leopard, a baboon, and
sometimes with nothing at all. There is no function from observations back to
states, only a distribution, and no amount of cleverness recovers a unique
answer from data that genuinely does not determine one.

**It requires a normalising constant.** The shape of the posterior is free:
multiply the likelihood row by the prior and you are done. But a shape is not a
distribution. To turn it into one you must divide by $P(o) = \sum_s P(o\mid s)P(s)$,
and that sum ranges over every state the model admits, including all the ones
that had nothing to do with the observation. The cost of answering a local
question is global.

**It is unstable.** Where the likelihood is nearly flat, small changes in the
data produce large swings in the posterior. This is the classical signature of
an ill-posed problem in the sense of Hadamard, and it is why the prior is not
optional. The prior is what regularises the inversion.

::: mn Ill-posedness
Hadamard called a problem well-posed if a solution exists, is unique, and
depends continuously on the data. Model inversion routinely fails the second
and third. Every practical method for solving it is, at bottom, a way of
smuggling in enough prior structure to restore well-posedness.
:::

## Explaining away, and why it ruins the obvious shortcut

Here is the natural idea for beating the cost. Suppose the state breaks into
several independent parts, one per thing the agent tracks: position, hunger,
wind, whether a predator is about. Each such part is a **state factor**, and
the joint state is one choice from every factor at once. If the state has
factors, do not reason about the joint state at all. Reason about each part separately. If
your model has $n$ factors with $k$ values each, that turns $k^n$ into $nk$, and
the problem evaporates.

It does not work, and understanding exactly why it does not work is the single
most useful thing in this lesson, because Week&nbsp;4 is going to do it anyway
and you need to know what is being given up.

Take two independent causes. This is a new and smaller model than the three-
state one of Lesson 3, chosen because two competing causes are the fewest that
can compete: a **gust** of wind, and a **baboon**. Either can
make a branch shake. Each is uncommon on its own:

$$
P(s_{\text{gust}} = 1) = P(s_{\text{bab}} = 1) = 0.1,
$$

and, importantly, they are independent *a priori*: wind does not summon baboons.
Now the likelihood. Rather than write a formula down and check it works, build
it from the causal story, because the shape then explains itself.

Suppose each cause, when present, makes its own independent attempt to shake the
branch, and succeeds with probability $w$. Suppose also the branch sometimes
shakes for no reason at all, with probability $\lambda$; treat that as a third,
always-present cause. Then the branch fails to shake only if **every** attempt
fails, and those failures are independent, so their probabilities multiply:

$$
P(o = 0 \mid s) \;=\; (1 - w\,s_{\text{gust}})(1 - w\,s_{\text{bab}})(1 - \lambda).
$$ {#noisy-and}

A cause that is absent has $s = 0$ and contributes a factor of $1$, which is
right: it makes no attempt and so cannot fail. Subtracting from one gives the
probability that the branch does shake, and with $w = 0.9$ and $\lambda = 0.01$:

$$
P(o = 1 \mid s) \;=\; 1 - (1 - 0.9\,s_{\text{gust}})(1 - 0.9\,s_{\text{bab}})(1 - 0.01).
$$ {#noisy-or}

This construction is called a **noisy-OR**, and it generalises without effort: for
$m$ causes with strengths $w_i$, the same argument gives
$P(o=1\mid s) = 1 - (1-\lambda)\prod_{i}(1 - w_i s_i)$. Nothing about it is
special to two causes or to equal strengths.

The branch shakes. Multiply prior by likelihood, state by state:

| $s_{\text{gust}}$ | $s_{\text{bab}}$ | prior | $P(o=1 \mid s)$ | product | posterior |
|---|---|---|---|---|---|
| 0 | 0 | {{noisy_prior_00:.2f}} | {{noisy_lik_00:.4f}} | {{noisy_prod_00:.6f}} | {{noisy_post_00:.4f}} |
| 0 | 1 | {{noisy_prior_01:.2f}} | {{noisy_lik_01:.4f}} | {{noisy_prod_01:.6f}} | {{noisy_post_01:.4f}} |
| 1 | 0 | {{noisy_prior_10:.2f}} | {{noisy_lik_10:.4f}} | {{noisy_prod_10:.6f}} | {{noisy_post_10:.4f}} |
| 1 | 1 | {{noisy_prior_11:.2f}} | {{noisy_lik_11:.4f}} | {{noisy_prod_11:.6f}} | {{noisy_post_11:.4f}} |

The evidence is $P(o=1) = {{noisy_ev:.4f}}$, so the shake carries
${{noisy_surprise:.3f}}$ nats of surprise. Both marginals move the same way and by a lot:

$$
P(s_{\text{gust}} = 1 \mid o) \;=\; P(s_{\text{bab}} = 1 \mid o) \;=\; {{noisy_marg_gust:.3f}},
$$

up from $0.1$. So far so unremarkable. Now look at the joint. If the posterior
factorised, the probability of *both* causes being present would be the product
of the marginals, ${{noisy_marg_gust:.3f}}^2 = {{noisy_prod_marginals:.3f}}$. The actual value in the table is ${{noisy_post_11:.3f}}$, smaller by a factor
of {{noisy_collapse_ratio:.1f}}.

The causes started independent and the observation made them dependent. They are
now strongly anticorrelated, and you can read the strength directly:

$$
P(s_{\text{bab}} = 1 \mid s_{\text{gust}} = 1,\, o) = {{noisy_bab_given_gust1:.3f}},
\qquad
P(s_{\text{bab}} = 1 \mid s_{\text{gust}} = 0,\, o) = {{noisy_bab_given_gust0:.3f}}.
$$ {#explaining-away}

Learning that there was a gust drags the baboon's probability from
${{noisy_marg_gust:.3f}}$ all the way back down to ${{noisy_bab_given_gust1:.3f}}$, a hair above its prior. The gust has **explained
away** the shaking branch, and the baboon is no longer needed.

::: keyidea
Independent priors plus a shared observation gives a coupled posterior. This is
not a quirk of noisy-OR; it happens whenever two causes compete to account for
the same data. The consequence is blunt: you cannot invert a model factor by
factor, because the posterior over one factor depends on the posterior over the
others. The intractable sum does not decompose.
:::

### Why it happens

The numbers show that it happens. Here is why, and the argument is short enough
that there is no excuse for leaving it in an exercise.

Independence survives conditioning on an event whose probability **factorises**
across the variables. If $P(E \mid s_1, s_2) = f(s_1)\,g(s_2)$ for some functions
$f$ and $g$, then the posterior is

$$
P(s_1, s_2 \mid E) \;\propto\; \underbrace{P(s_1)f(s_1)}_{\text{depends on } s_1 \text{ alone}} \;\cdot\; \underbrace{P(s_2)g(s_2)}_{s_2 \text{ alone}},
$$

which is again a product, so the two stay independent. Conditioning on a
factorising event reweights each variable separately and never introduces a link.

Now look at [eq:noisy-and]. The event $o = 0$, the branch *not* shaking, has
exactly that form: it is a product of one factor per cause. So learning the
branch stayed still would leave the causes independent. But we conditioned on the
complement, $o = 1$, and one minus a product is not a product. That single
algebraic fact is the whole mechanism.

Read causally it is the diminishing return in [eq:noisy-or]: the state $(1,1)$
has likelihood ${{noisy_lik_11:.4f}}$, barely more than the ${{noisy_lik_10:.4f}}$
of a single cause, because one cause already does nearly all the work available.
A second cause adds almost nothing, so the evidence has almost nothing extra to
buy with it, and the two causes end up competing for the same credit.

The sign is worth noting too, because it is not universal. Explaining away
depends on the causes being **sufficient**: either alone can produce the effect.
Reverse that, so both are jointly *necessary* and neither alone suffices, and the
correlation flips: learning one cause is present makes the other **more** likely,
since the effect still needs explaining and only the second cause can supply it.
Competition and collaboration are both available; which you get is set by the
structure of the likelihood, not by anything about priors.

Explaining away is also, incidentally, why the phenomenon is worth caring about
outside of a mathematics course. It is a real property of perception, it is what
makes a plausible alternative explanation reduce your confidence in the first
one, and it is one of the sharper tests of whether a candidate neural circuit is
really doing inference: a circuit that merely accumulates evidence for each
cause separately cannot produce it, because the effect requires the causes to
see one another.

## What can actually be done

There are three families of response, and the entire structure of Part&nbsp;I is
a tour of them.

**Solve it exactly, where the model lets you.** Some pairings of prior and
likelihood have posteriors in closed form. If the likelihood is Gaussian with
known variance and the prior over the mean is Gaussian, the posterior is
Gaussian and you can write down its parameters in two lines. This is
conjugacy, it is the backbone of Week&nbsp;2, and it works right up until your
model has any structure you actually wanted.

**Sample from it.** Do not compute $P(s\mid o)$; draw from it. Markov chain
Monte Carlo needs only ratios of posterior values, and the intractable $P(o)$
cancels in every ratio. This is the correct method for a statistician with a
week to spare, and the wrong one for an animal deciding whether to run, because
it converges in the limit and offers few guarantees before then.

**Turn it into optimisation.** Give up on the exact posterior. Choose a family
of distributions $Q(s)$ you can compute with, define a measure of how far $Q$ is
from the true posterior, and descend it. Because the measure can be arranged so
that the intractable term drops out, you can minimise a distance to something
you cannot evaluate. That trick is variational inference, the quantity you
descend is the free energy, and it occupies Week&nbsp;4 and everything after.

::: remark Why the third option is the one for embodied agents
The optimisation route is the only one of the three that offers a partial answer
at every instant, improves continuously with more time, and turns inference into
a dynamical system. An animal does not get to say "my chain has not mixed yet".
It gets to have a current best guess that is being continuously corrected, which
is exactly what gradient descent on a free energy gives you.
:::

::: warning What the third option costs
The family of distributions $Q$ has to be one you can compute with, and by far
the commonest choice is the family that factorises across state factors,
$Q(s_{\text{gust}}, s_{\text{bab}}) = Q(s_{\text{gust}})\,Q(s_{\text{bab}})$,
or $Q(s_1, s_2) = Q(s_1)Q(s_2)$ once we stop naming them. Look back at
[eq:explaining-away]. That family cannot represent the true posterior, because
the true posterior is correlated and no product of marginals is.
:::

## How far off is the best factorised approximation?

That question hides a choice, and the choice decides the answer. It is worth
making explicit now, because Week&nbsp;4 rests on it.

The Kullback&ndash;Leibler divergence is not symmetric, so there are two ways to
ask how far a factorised $Q$ sits from the true posterior $P$:

$$
D_{\mathrm{KL}}\big[P \,\|\, Q\big] = \sum_{s} P(s)\ln\frac{P(s)}{Q(s)},
\qquad
D_{\mathrm{KL}}\big[Q \,\|\, P\big] = \sum_{s} Q(s)\ln\frac{Q(s)}{P(s)},
$$ {#two-kls}

with $s$ running over the four joint states of the table. Call the first the
**forward** direction and the second the **reverse**. They disagree, and not by a
little.

::: derivation The forward direction just matches the marginals
Minimise the left of [eq:two-kls] over products, writing $q_1$ and $q_2$ for the
two factors so that $Q(s) = q_1(s_{\text{gust}})\,q_2(s_{\text{bab}})$. Split the
logarithm:

$$
D_{\mathrm{KL}}[P \,\|\, Q]
= \underbrace{\sum_s P(s)\ln P(s)}_{-\mathrm{H}[P],\ \text{no } Q \text{ in it}}
\; -\; \sum_{s} P(s)\big[\ln q_1(s_{\text{gust}}) + \ln q_2(s_{\text{bab}})\big].
$$

Only the second term can be optimised. In it, sum out the variable that does not
appear:

$$
\sum_{s} P(s)\,\ln q_1(s_{\text{gust}})
\;=\; \sum_{s_{\text{gust}}} P_1(s_{\text{gust}})\,\ln q_1(s_{\text{gust}}),
$$

where $P_1$ is the true marginal of the first cause, and likewise for the second.
The problem has separated into two independent ones, each of the form "choose a
distribution $q$ maximising $\sum_x p(x)\ln q(x)$". By Gibbs' inequality that is
solved by $q = p$.

So the forward-optimal $Q$ has **exactly the true marginals**, here
${{noisy_marg_gust:.4f}} $ for each cause, and leaves a gap of
${{noisy_gap_forward:.4f}}$ nats. This is a lemma, not an accident of these
numbers: forward KL over a product family always returns the marginals.
:::

Variational inference does not use that direction, for a reason worth stating
now. The forward divergence takes its expectation under $P$, and $P$ is precisely
what we cannot compute. The reverse takes its expectation under $Q$, which we
chose and therefore know. The direction is forced on us by tractability, not
chosen for its properties, and its properties turn out to matter.

::: derivation The reverse direction picks a side
Minimise the right of [eq:two-kls] instead. Vary $q_1$ with $q_2$ held fixed,
impose normalisation with a Lagrange multiplier, and the stationary condition is

$$
\ln q_1(s_{\text{gust}}) \;=\; \mathbb{E}_{q_2}\big[\ln P(s_{\text{gust}}, s_{\text{bab}})\big] + \text{const},
$$ {#cavi}

with the mirror equation for $q_2$. Alternating them until nothing moves is
**coordinate ascent**, and [eq:cavi] is the update Week&nbsp;4 derives in general.
Run on the table above it converges to

$$
q_1(s_{\text{gust}}\!=\!1) = {{noisy_mf_gust:.4f}},
\qquad
q_2(s_{\text{bab}}\!=\!1) = {{noisy_mf_bab:.4f}},
$$

a gap of ${{noisy_gap_reverse:.4f}}$ nats. The two causes entered the model
interchangeably and the answer has stopped treating them that way.
:::

The symmetry breaking is not numerical noise. Score the *symmetric* product, both
marginals at ${{noisy_marg_gust:.4f}}$, under the reverse divergence and you get
${{noisy_gap_reverse_symmetric:.4f}}$ nats, genuinely worse than the lopsided
${{noisy_gap_reverse:.4f}}$. Under this objective, committing to one story beats
splitting the difference.

The reason is visible in the table. The true posterior puts its mass on the two
states where exactly one cause fired, and almost none on $(1,1)$ or $(0,0)$. A
product cannot do that. If $q_1$ and $q_2$ both sit near a half so as to cover
both modes, then $Q(1,1) = q_1q_2$ is dragged up to about a quarter, into a
corner where $P$ is nearly empty. The reverse divergence punishes exactly this:
each state contributes $Q\ln(Q/P)$, which blows up wherever $Q$ is large and $P$
is small. Reverse KL is **zero-forcing**: it would rather miss some of $P$
altogether than put mass where $P$ has none. Forward KL has the opposite
instinct, since there the offending term is $P\ln(P/Q)$, which blows up where
$P$ is large and $Q$ is small, so it insists on covering everything $P$ covers.

::: fig The same posterior, two ways. Area is probability. The true posterior puts its mass on the two states where exactly one cause fired; no product of marginals has that shape, so the best product commits to one of them and inflates the corners.
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two two-by-two grids comparing the true posterior with the best factorised approximation.">
  <rect x="1" y="1" width="658" height="248" fill="#fdfcf9" stroke="#e4e1d9"/>
  <text x="150" y="30" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="12" font-weight="600" fill="#0f5f57">true posterior</text>
  <text x="470" y="30" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="12" font-weight="600" fill="#9c4d2f">best product, reverse KL</text>
  <rect x="70" y="45" width="160" height="160" fill="none" stroke="#d3cfc4"/>
  <rect x="70" y="125" width="80" height="80" fill="#0f5f57" fill-opacity="0.09"/>
  <rect x="150" y="125" width="80" height="80" fill="#0f5f57" fill-opacity="0.90"/>
  <rect x="70" y="45" width="80" height="80" fill="#0f5f57" fill-opacity="0.90"/>
  <rect x="150" y="45" width="80" height="80" fill="#0f5f57" fill-opacity="0.11"/>
  <text x="110" y="172" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#3c3c42">0.045</text>
  <text x="190" y="172" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#ffffff">0.450</text>
  <text x="110" y="92" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#ffffff">0.450</text>
  <text x="190" y="92" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#3c3c42">0.055</text>
  <rect x="390" y="45" width="160" height="160" fill="none" stroke="#d3cfc4"/>
  <rect x="390" y="125" width="80" height="80" fill="#9c4d2f" fill-opacity="0.30"/>
  <rect x="470" y="125" width="80" height="80" fill="#9c4d2f" fill-opacity="0.11"/>
  <rect x="390" y="45" width="80" height="80" fill="#9c4d2f" fill-opacity="0.95"/>
  <rect x="470" y="45" width="80" height="80" fill="#9c4d2f" fill-opacity="0.34"/>
  <text x="430" y="172" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#3c3c42">0.176</text>
  <text x="510" y="172" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#3c3c42">0.061</text>
  <text x="430" y="92" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#ffffff">0.566</text>
  <text x="510" y="92" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="11" fill="#3c3c42">0.197</text>
  <text x="60" y="92" text-anchor="end" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">baboon 1</text>
  <text x="60" y="172" text-anchor="end" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">baboon 0</text>
  <text x="110" y="224" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">gust 0</text>
  <text x="190" y="224" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">gust 1</text>
  <text x="430" y="224" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">gust 0</text>
  <text x="510" y="224" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">gust 1</text>
</svg>
:::

Which direction you minimise therefore decides whether your approximation hedges
or commits. Week&nbsp;4 has to say which one active inference uses and why. For
now: "mean-field" is the name of a compromise, and you have just seen what is
being compromised.

::: exercise Where does explaining away go?
In the two-cause example, make the observation uninformative by setting both cause strengths (the $0.9$
coefficients in [eq:noisy-or]) to zero, so that $P(o = 1 \mid s) = 0.01$ for every state.
Without computing anything, say what the posterior is and whether the two causes
remain independent. Then explain what feature of [eq:noisy-or] is responsible
for the coupling in the original example.
---solution---
With $P(o=1\mid s)$ constant across states, the likelihood is a constant factor
that pulls out of the product. The posterior equals the prior, which was a
product of independent marginals, so the causes remain independent and no
explaining away occurs.

The coupling comes from the likelihood varying across the *joint* state in a way
that does not factorise, as the section above sets out. In [eq:noisy-or], the state $(1,1)$ has likelihood ${{noisy_lik_11:.4f}}$,
which is barely more than the ${{noisy_lik_10:.4f}}$ of $(1,0)$ or $(0,1)$: a second cause
adds almost nothing once the first is present. That diminishing return is the
source of the competition. If the likelihood were multiplicatively separable, the
posterior would inherit the prior's independence exactly and the shortcut would
be valid.
:::

::: exercise Counting the cost honestly
An agent tracks $n = 15$ binary state factors. Give the number of terms in the
evidence sum, and state how many factors could be added before the sum exceeds
the number of nanoseconds since the Big Bang, roughly $4.4 \times 10^{26}$.
---solution---
With $k = 2$ and $n = 15$ the sum has $2^{15} = {{terms_15_binary:,}}$ terms, which is
nothing: a modern processor does this in microseconds.

For the second part, $2^n > 4.4\times10^{26}$ requires
$n > \log_2(4.4\times10^{26}) = {{log2_universe:.2f}}$, so
$n = {{factors_to_exceed_universe:d}}$ factors. Working in base ten instead,
$\log_{10}(4.4\times10^{26}) = {{log10_universe:.3f}}$, and dividing by
$\log_{10}2$ gives the same answer. Keep the third decimal: rounding that to
$26.6$ before dividing costs you more than a tenth of a factor. Adding {{factors_to_add:d}} binary variables to a tractable problem makes it impossible
for any physical computer, ever. This is what is meant by saying the difficulty
is structural rather than an engineering matter, and it is why the response has
to be a different algorithm rather than a bigger machine.
:::

::: checkpoint
- Name the three separate ways in which model inversion is harder than model evaluation.
- Two causes have independent priors. After a shared observation, are they independent? Under what condition on the likelihood would they remain so?
- Which of the three solution families does not require evaluating $P(o)$, and why?
- What property of the true posterior can a factorised $Q$ never represent?
:::
