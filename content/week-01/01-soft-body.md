---
title: A soft body in a hard world
deck: Why anything that stays alive must carry a model of the thing it is staying alive in.
week: 1
time: 25 min
scripts: [w01.js]
---

You are a bag of warm salt water, held a few degrees below the temperature at
which your own proteins come apart. Your core sits within about a degree of
37&nbsp;°C. Your blood pH stays between 7.35 and 7.45. The sodium concentration
outside your cells holds near 140&nbsp;mmol per litre, and if it wanders far in
either direction you become confused, then unconscious, then dead.

Each of those is a coordinate. Along every one of them, the range compatible
with being alive is tiny next to the range physics is perfectly content to put
you in. A living body occupies a vanishingly small corner of the space of
configurations available to the atoms it is made of, and it keeps occupying that
corner for decades.

That is the fact this whole course is an attempt to explain. Not consciousness,
not intelligence, not language. Just this: how does a thing stay in its corner?

::: mn Physiological ranges
Core temperature, blood pH and plasma sodium are the standard clinical
examples because they are the ones we monitor. The argument does not depend
on them; pick any measurable state of your body and the same narrowness holds.
:::

## The corner is not free

Left alone, a physical system drifts. Heat leaks across gradients, concentrations
even out, structure decays into mush. There is no force in physics that gently
returns your body temperature to 37&nbsp;°C when you stand in a cold wind. If you
stay at 37&nbsp;°C it is because something is working to keep you there: you
shiver, you constrict blood vessels in your skin, you put on a jumper, you go
inside.

Call the set of states in which the organism continues to exist its **viable
set**. The picture to hold is a small region of a large space, with the system
being nudged out of it constantly by noise, and something inside the system
pushing back.

::: widget homeostasis-drift | A body in a two-dimensional physiological space. The solid circle is the boundary of the viable set; cross it and the organism is no longer an organism. The dashed circle marks the states it prefers. Noise pushes it around; the parameter $\kappa$ sets how hard it pushes back.
:::

Run that with $\kappa = 0$ first. There is nothing pathological in the dynamics.
It is an unbiased random walk, the most innocuous thing a stochastic process can
do, and it kills the organism every time. Diffusion does not need to be malicious
to be fatal. It only needs to be given enough time.

Now turn $\kappa$ up. The physics has not changed and the noise has not changed.
What has changed is that the system now moves in a way that depends on where it
is, and that is enough.

## Staying alive is a statement about entropy

Watch the readout in that figure. As the unregulated walk wanders, the entropy
of the states it has visited climbs steadily. When the walk is regulated, that
number rises for a while and then flattens out, well below the maximum. This is
not decoration. It is the quantity we are going to build the entire framework
on, so it is worth being precise about what it is.

Write $o$ for the states of the organism's sensory surface: everything it can
measure about itself and its surroundings. Over a long life, the organism visits
some sensory states often and others never, and that occupancy defines a
distribution $P(o)$. Its entropy is

$$
\mathrm{H}[P(o)] \;=\; -\sum_{o} P(o) \ln P(o).
$$ {#sensory-entropy}

An organism whose sensory states are spread thinly over everything possible has
high entropy here. An organism that spends its whole life reporting *warm,
fed, upright, unbroken* has low entropy. Being alive, expressed in nats, is
[eq:sensory-entropy] being small.

::: mn Why entropy and not variance
Variance needs a metric on the state space and privileges the mean. Entropy
needs only a probability. For a creature whose states include "which arm of the
maze" and "is the predator visible", there is no sensible mean to take.
:::

Now the step that makes the framework possible. If the organism's sensory
statistics settle down over time, the long-run average of the quantity
$-\ln P(o_t)$ along its actual trajectory converges to the entropy of the
distribution it is sampling:

$$
\mathrm{H}[P(o)] \;=\; \lim_{T\to\infty} \frac{1}{T}\sum_{t=1}^{T} \big[-\ln P(o_t)\big].
$$ {#ergodic}

The quantity in the brackets, $-\ln P(o)$, is called the **surprise** of an
observation, or sometimes its surprisal, to keep it clear that we are not
talking about anybody's emotional state. It is large when $o$ is something the
organism rarely sees and small when $o$ is business as usual.

::: keyidea
[eq:ergodic] converts a statement about a whole lifetime into a statement about
this instant. Keeping the entropy of your sensory states low over a lifetime is
the same thing as keeping your surprise low at each moment. A goal you could
never act on becomes a goal you can act on now.
:::

::: warning The step that is doing work
[eq:ergodic] holds if the process is ergodic, and whether living systems are
ergodic in the required sense is exactly the point at which several careful
critics of the free energy principle plant their flag. We will take the
identification as given for eleven weeks because it is the assumption the
framework is built on, and then in Week&nbsp;12 we will come back and ask what
it costs.
:::

## Surprise is not a property of the world

Here is the difficulty that generates everything else in this course.

Nothing in the sentence "minimise $-\ln P(o)$" tells you what $P$ is. Probability
is not a quantity you can read off the world with an instrument. There is no
surprise-meter. An observation is surprising only *relative to some set of
expectations*, and those expectations have to live somewhere.

They live inside the organism. To be able to evaluate its own surprise, a system
must carry an internal model of what its observations are usually like and what
produces them. That model is what we will call a **generative model**, and
building one, running it, and correcting it is what the rest of this course is
about.

So the argument so far runs:

1. To persist is to occupy a small set of states.
2. Occupying a small set of states means low entropy over sensory states.
3. Low entropy over a lifetime means low surprise at each moment.
4. Surprise is only defined relative to a model.
5. Therefore anything that persists must carry a model of the world it persists in.

That is a strong claim and it deserves the scrutiny we will give it in
Week&nbsp;12. But notice what it does not claim. It says nothing about the model
being conscious, accurate, or even any good. A thermostat has one. The claim is
only that having one is what persistence looks like from the inside.

## Two ways to be less surprised

Suppose you have such a model, and an observation arrives that it did not
expect. There are exactly two things you can do about it.

You can **change the model** so that the observation is no longer surprising.
This is perception: the world is telling you something and you update
accordingly. What you thought was a shadow was a leopard, and now you believe it
was a leopard.

Or you can **change the world** so that your next observation is one the model
did expect. This is action: you are cold, your model expects to be warm, and
instead of revising the expectation you move into the sun.

::: mn The name
Perception is inference over states. Action is inference carried out by moving.
Doing both at once, with one quantity governing both, is what puts the
*active* in active inference.
:::

Both reduce the mismatch between model and world. They are the same operation
applied to the two different halves of the coupling, and in the mathematics they
will turn out to be gradient descent on the same scalar quantity. That quantity
is variational free energy, and we will construct it properly in Week&nbsp;4.

## An aside on brains, told correctly

There is a story that gets repeated whenever anyone argues that brains exist for
movement. The sea squirt, it goes, swims about as a larva with a little brain,
finds a rock, cements itself to it, and then, having no further use for the
thing, digests its own brain.

It is a good story and it is not true. The larval tunicate does resorb its
larval nervous system at metamorphosis, but what the adult builds afterwards is
a cerebral ganglion roughly an order of magnitude *larger* than the one the
larva had, wired for a completely different problem: filtering, siphon control,
and responding to things that touch you when you cannot run away.

Which is the better lesson anyway. Nervous systems are not a luxury that
movement pays for. They are shaped by the control problem the animal actually
faces, and a sessile animal faces a real one. Hold onto that, because a
recurring temptation in this field is to assume that the interesting agents are
the ones that look like us. The mathematics does not care.

::: checkpoint
Before going on, make sure you could answer these without looking back.

- Why does an unbiased random walk kill the organism in the figure above, given
  that it has no bias towards the boundary?
- What is the difference between the entropy of $P(o)$ and the surprise of a
  particular observation $o$?
- Why can surprise not be measured directly from the environment?
- Name the two routes to reducing surprise, and say which one changes $P$ and
  which changes $o$.
:::
