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

Here is the natural idea for beating the cost. If the state has many parts, do
not reason about the joint state at all. Reason about each part separately. If
your model has $n$ factors with $k$ values each, that turns $k^n$ into $nk$, and
the problem evaporates.

It does not work, and understanding exactly why it does not work is the single
most useful thing in this lesson, because Week&nbsp;4 is going to do it anyway
and you need to know what is being given up.

Take two independent causes. A **gust** of wind, and a **baboon**. Either can
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
| 0 | 0 | 0.81 | 0.0100 | 0.008100 | 0.0450 |
| 0 | 1 | 0.09 | 0.9010 | 0.081090 | 0.4500 |
| 1 | 0 | 0.09 | 0.9010 | 0.081090 | 0.4500 |
| 1 | 1 | 0.01 | 0.9901 | 0.009901 | 0.0550 |

The evidence is $P(o=1) = 0.1802$, so the shake carries $1.714$ nats of
surprise. Both marginals move the same way and by a lot:

$$
P(s_{\text{gust}} = 1 \mid o) \;=\; P(s_{\text{bab}} = 1 \mid o) \;=\; 0.505,
$$

up from $0.1$. So far so unremarkable. Now look at the joint. If the posterior
factorised, the probability of *both* causes being present would be the product
of the marginals, $0.505^2 = 0.255$. The actual value in the table is $0.055$,
smaller by a factor of nearly five.

The causes started independent and the observation made them dependent. They are
now strongly anticorrelated, and you can read the strength directly:

$$
P(s_{\text{bab}} = 1 \mid s_{\text{gust}} = 1,\, o) = 0.109,
\qquad
P(s_{\text{bab}} = 1 \mid s_{\text{gust}} = 0,\, o) = 0.909.
$$ {#explaining-away}

Learning that there was a gust drags the baboon's probability from $0.505$ all
the way back down to $0.109$, a hair above its prior. The gust has **explained
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
one, and it is a standard test of whether a candidate neural circuit is really
doing Bayesian inference or only imitating it.

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
the commonest choice is the family that factorises across state factors:
$Q(s_1, s_2) = Q(s_1)Q(s_2)$. Look back at [eq:explaining-away]. That family
cannot represent the true posterior, because the true posterior is correlated
and no product of marginals is. The best factorised approximation to the
posterior in the table above still sits $0.369$ nats away from it.

The approximation is not free and it fails in a specific direction: it
underestimates how much the causes are competing. Week&nbsp;4 will make this
precise. For now, register that "mean-field" is the name of a compromise, and
that you have already seen exactly what is being compromised.
:::

::: exercise Where does explaining away go?
In the two-cause example, make the observation uninformative by setting both
cause strengths to zero, so that $P(o = 1 \mid s) = 0.01$ for every state.
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
$s_{\text{bab}}$. In [eq:noisy-or], the state $(1,1)$ has likelihood $0.9901$,
which is barely more than the $0.9010$ of $(1,0)$ or $(0,1)$: a second cause
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
With $k = 2$ and $n = 15$ the sum has $2^{15} = 32{,}768$ terms, which is
nothing: a modern processor does this in microseconds.

For the second part, $2^n > 4.4\times10^{26}$ requires
$n > \log_2(4.4\times10^{26}) = 26.6/\log_{10} 2 \approx 88.5$, so $n = 89$
factors. Adding 74 binary variables to a tractable problem makes it impossible
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
