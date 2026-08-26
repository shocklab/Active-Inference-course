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
The branch shakes if either cause fires, with each cause working 90% of the
time, plus a 1% chance the branch shakes for no reason at all:

$$
P(o = 1 \mid s_{\text{gust}}, s_{\text{bab}}) \;=\;
1 - (1 - 0.9\,s_{\text{gust}})(1 - 0.9\,s_{\text{bab}})(1 - 0.01).
$$ {#noisy-or}

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
or $Q(s_1, s_2) = Q(s_1)Q(s_2)$ once we stop naming them. Look back at [eq:explaining-away]. That family
cannot represent the true posterior, because the true posterior is correlated
and no product of marginals is. The best factorised approximation to the
posterior in the table above sits a measurable distance from it. How far depends
on what "best" means, and the two obvious answers disagree.

Match the true marginals and the gap is ${{noisy_gap_forward:.3f}}$ nats. But
variational inference does not match marginals. It minimises the divergence the
other way round, and the best factorised $Q$ by that measure sits
${{noisy_gap_reverse:.3f}}$ nats away. Worse, it gets there by abandoning
symmetry: it settles near ${{noisy_mf_bab:.2f}}$ for one cause and
${{noisy_mf_gust:.2f}}$ for the other, picking a side rather than splitting the
difference, when the true posterior puts both at ${{noisy_marg_gust:.3f}}$.

Which direction you minimise is therefore not a detail, and Week&nbsp;4 has to
say which one active inference uses and why. For now: "mean-field" is the name of
a compromise, and you have just seen what is being compromised.
:::

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
that does not factor into a function of $s_{\text{gust}}$ times a function of
$s_{\text{bab}}$. In [eq:noisy-or], the state $(1,1)$ has likelihood ${{noisy_lik_11:.4f}}$,
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
