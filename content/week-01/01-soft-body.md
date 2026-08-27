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

## Drift is the default

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

::: widget homeostasis-drift | A body in a two-dimensional physiological space, each axis one regulated variable measured as a deviation from its setpoint. The solid circle at radius $1$ is the boundary of the viable set; cross it and the organism is no longer an organism. $\sigma$ is the size of one step of noise; $\kappa$ is the fraction of the current displacement the body removes per step, defined properly below. The dashed circle is the spread the algebra predicts, $\sqrt{2v}$ with $v$ from [eq:stationary-var]; it is absent at $\kappa = 0$ and $\kappa \ge 2$, where no stationary spread exists. The right-hand panel tracks the entropy of the visited states: the position is binned onto a ${{hs_bins:d}} \times {{hs_bins:d}}$ grid, the visit counts are normalised into a distribution, and its entropy is reported in nats, against a maximum of $\ln {{hs_bin_cells:d}} = {{hs_max_entropy:.3f}}$ for a body found equally often everywhere.
:::

The figure is running one line of arithmetic. Writing $x_t \in \mathbb{R}^2$ for
the displacement from the centre of the viable set,

$$
x_{t+1} \;=\; \underbrace{(1-\kappa)\,x_t}_{\text{pull back}}
\;+\; \underbrace{\sigma\,\xi_t}_{\text{noise}},
\qquad \xi_t \sim \mathcal{N}(0, I_2)
\ \text{independently at each step},
$$ {#homeostat}

absorbed the moment $\lVert x_t \rVert > 1$.

Two parameters, and it pays to be exact about both. $\sigma$ is the size of one
step of noise, in whatever units $x$ is measured in. $\kappa$ is dimensionless:
it is the **fraction of the current displacement that the organism removes per
step**. At $\kappa = 0.15$ the body cancels fifteen per cent of its error each
step and lets the remaining eighty-five per cent stand. Biologically it is the
gain of whatever loop does the regulating, whether that is shivering,
vasoconstriction, or walking into the shade; a large $\kappa$ is a strong, fast
correction.

Written that way the parameter has a range worth exploring, and the figure's
slider now covers it.

**Why $\kappa = 0$ is fatal.** With no restoring term the increments are
independent and mean-zero, so the variances add. After $t$ steps each coordinate
has variance $t\sigma^2$, and

$$
\mathbb{E}\big[\lVert x_t \rVert^2\big] \;=\; 2t\sigma^2 ,
$$ {#diffusion}

which grows without bound. There is no equilibrium to settle into: the walk has
no preferred location, and the region it typically occupies swells like
$\sqrt{t}$. Setting [eq:diffusion] equal to $1$ gives a rough crossing time of
$t \approx 1/(2\sigma^2)$, which at $\sigma = {{hs_sigma}}$ is about
{{hs_diffusive_steps:.0f}} steps. More carefully: for a driftless walk the
probability of still being inside any fixed radius tends to zero, so the boundary
is crossed with probability one and the only question is when. Diffusion does not
need to be malicious to be fatal. It needs to be given enough time.

**What $\kappa > 0$ buys.** Now the variance stops adding. Write $v$ for the variance of one coordinate once the process has settled,
the **stationary variance**. Taking variances of both sides of [eq:homeostat]
and asking for a value that reproduces itself,

$$
v \;=\; (1-\kappa)^2 v + \sigma^2
\qquad\Longrightarrow\qquad
v \;=\; \frac{\sigma^2}{1 - (1-\kappa)^2} \;=\; \frac{\sigma^2}{2\kappa - \kappa^2}.
$$ {#stationary-var}

The walk has acquired a stationary distribution. At $\sigma = {{hs_sigma}}$ and
$\kappa = {{hs_kappa}}$ this is $v = {{hs_var_per_coord:.5f}}$, a standard
deviation of {{hs_sd_per_coord:.4f}} per coordinate, which puts the boundary at
$\lVert x \rVert = 1$ about {{hs_boundary_in_sd:.0f}} standard deviations away.

### The range of stable gains

Strip the noise from [eq:homeostat] and the displacement is multiplied by
$1 - \kappa$ each step. Whether the organism returns to the centre therefore
depends on nothing more than whether that factor is smaller than one in
magnitude, which requires

$$
0 < \kappa < 2 .
$$ {#stability}

Below zero the correction pushes the wrong way. Above two it overcorrects so
violently that each step lands further out than the last. Between those bounds
the behaviour changes character twice:

| $\kappa$ | $1-\kappa$ | what the organism does | stationary variance |
|---|---|---|---|
| $0$ | $1$ | nothing; the walk diffuses away | unbounded |
| $0 < \kappa < 1$ | positive | eases back towards the centre from one side | $\sigma^2/\kappa(2-\kappa)$ |
| $1$ | $0$ | erases the whole error in a single step | $\sigma^2$ |
| $1 < \kappa < 2$ | negative | overshoots, landing on the far side each time | $\sigma^2/\kappa(2-\kappa)$ |
| $2$ | $-1$ | oscillates forever without settling | unbounded |

The middle column explains the qualitative change. For $\kappa < 1$ the factor
$1-\kappa$ is positive, so a body above the setpoint stays above it while
approaching. For $\kappa > 1$ the factor is negative, so a body above the
setpoint arrives below it and has to come back: the correction was too large.

That gives the variance in [eq:stationary-var] a shape. Its denominator
$\kappa(2-\kappa)$ is largest at $\kappa = 1$, so

$$
v_{\min} \;=\; \sigma^2 ,
$$ {#best-gain}

attained by correcting the error exactly once, completely. No regulation scheme
of this form does better. Even a perfect controller carries one step's worth of
noise, because the noise arrives *after* the correction and there is nothing to
be done about it until the next step.

::: keyidea
Two things follow, and the second is easy to miss. **Too little regulation is
fatal**, which is the point the figure opens with. But **too much regulation is
also fatal**, and symmetrically so: since $\kappa(2-\kappa)$ is unchanged when
$\kappa$ is replaced by $2-\kappa$, a gain of $1.5$ leaves a body exactly as
spread out as a gain of $0.5$. Overcorrection is not a lesser sin than
underreaction. It is the same sin measured from the other side, and Week&nbsp;5
meets it again under a different name. There the gain is set by a precision,
the inverse of a variance, defined properly in Week&nbsp;2. A large precision is
a claim that a signal is reliable; a system that believes its measurements are better than they are
overcorrects to each of them and ends up chasing its own noise.
:::

Push the figure's slider past $1$ and the trajectory changes visibly: instead of
drifting back towards the middle it starts zigzagging across it, and past
$\kappa = 2$ it leaves for good.

::: warning "Indefinitely" is the wrong word
Note what [eq:stationary-var] does *not* say. The stationary distribution is
Gaussian, so it has unbounded support: every state remains reachable and the
boundary is still crossed eventually, with probability one. Regulation does not
make death impossible. It converts a certainty on a timescale of hundreds of
steps into a rare event on a timescale so long the organism will have died of
something else first. How rare depends on how many standard deviations of the
stationary distribution fit inside the boundary, which is $1/\sqrt{v}$ per
coordinate: about {{hs_boundary_in_sd:.0f}} at the figure's default settings, and
the expected time to cross grows faster than exponentially in that number. That is all any organism gets, and it is enough.
:::

## Staying alive is a statement about entropy

Watch the readout in that figure. It measures the entropy of the *physiological*
states the body has occupied, whereas the quantity defined below is the entropy
of the *observations* it makes. Section&nbsp;2 explains why the first bounds the
second: an organism senses its own condition, so a body confined to few
physiological states cannot report many different interoceptive observations. As the unregulated walk wanders, the entropy
of the states it has visited climbs steadily. When the walk is regulated, that
number rises for a while and then flattens out, well below the maximum. This is
not decoration. It is the quantity we are going to build the entire framework
on, so it needs stating precisely.

Everything an organism can register about itself and its surroundings arrives at
a surface: the retina, the skin, the stretch receptors in the gut. Note that the
list runs inward as well as outward. A body senses its own temperature, its own
blood chemistry, the tension in its own muscles; that inward-facing sensing is
called **interoception**, and it is what connects this section to the last one.
The physiological variables of the figure above are not some separate category of
thing: they are among the quantities the organism measures.

Call the state of that surface at one moment an **observation**, and write it $o$. The letter
stands for *observation*. An organism has no access to the world; it has access
only to what its senses report of the world, and those are not the same thing. [lesson:1.2] makes that distinction precise and
gives the world's own states a different letter. Until then, $o$ is simply
whatever the organism can actually measure.

Over a long life the organism makes some observations often and others never, and
that occupancy defines a probability distribution over observations, written
$P(o)$. Its **entropy** is

$$
\mathrm{H}[P(o)] \;=\; -\sum_{o} P(o) \ln P(o).
$$ {#sensory-entropy}

Two notational points, since both recur for twelve weeks. The square brackets are
deliberate: $\mathrm{H}$ takes an entire distribution as its argument rather than
a single value, so it is a functional, not a function. And the logarithm is
natural, here and everywhere in this course, which fixes the units as **nats**.
A nat is the unit you get when the logarithm is natural rather than base two;
divide by $\ln 2 \approx 0.693$ to convert a figure in nats to bits.

Two numbers make the scale concrete. Take four possible observations. If the
organism reports the first almost always, $P(o) = (0.94, 0.02, 0.02, 0.02)$, then

$$
\mathrm{H}[P(o)] = -\big[0.94\ln 0.94 + 3 \times 0.02\ln 0.02\big]
= {{ent_peaked:.4f}}\ \text{nats}.
$$

If instead all four are equally likely, every term is $-\tfrac14 \ln \tfrac14$ and
the entropy is $\ln 4 = {{ent_uniform4:.4f}}$ nats, which is the largest value
four outcomes admit. The surprise of the common observation in the first case is
$-\ln 0.94 = {{surprise_common:.4f}}$ nats and of a rare one
$-\ln 0.02 = {{surprise_rare:.4f}}$ nats; weight those by how often each occurs
and you recover {{ent_peaked_check:.4f}} nats, the entropy again, as
[eq:ergodic] requires.

An organism whose observations are spread thinly over everything possible has
high entropy here. An organism that spends its whole life reporting *warm,
fed, upright, unbroken* has low entropy. Being alive, measured in nats, is equation [eq:sensory-entropy] being small.

::: mn Why entropy and not variance
Variance needs a metric on the state space and privileges the mean. Entropy
needs only a probability. For a creature whose states include "which arm of the
maze" and "is the predator visible", there is no sensible mean to take.
:::

Now the step that makes the framework possible. Write $o_t$ for the observation
made at time step $t$, and let $T$ count the steps of a life. If the organism's
sensory statistics settle down over time, the long-run average of
$-\ln P(o_t)$ along its actual trajectory converges to the entropy of the
distribution it is sampling:

$$
\mathrm{H}[P(o)] \;=\; \lim_{T\to\infty} \frac{1}{T}\sum_{t=1}^{T} \big[-\ln P(o_t)\big].
$$ {#ergodic}

The quantity in the brackets, $-\ln P(o)$, is called the **surprise** of an
observation, or sometimes its surprisal, to keep it clear that we are not
talking about anybody's emotional state. It is large when $o$ is something the
organism rarely sees and small when $o$ is business as usual.

::: derivation Where [eq:ergodic] comes from
The page has just called this the step that makes the framework possible, so it
should not be left as an assertion. Take the easy case first, where the
observations are independent and identically distributed. Then
$-\ln P(o_1), -\ln P(o_2), \dots$ are themselves independent and identically
distributed random variables, and the strong law of large numbers says their
running average converges almost surely to their expectation:

$$
\frac{1}{T}\sum_{t=1}^{T}\big[-\ln P(o_t)\big]
\ \xrightarrow{\ \text{a.s.}\ }\
\mathbb{E}_{P(o)}\big[-\ln P(o)\big].
$$

Now look at what that expectation is. Writing it out,

$$
\mathbb{E}_{P(o)}\big[-\ln P(o)\big]
\;=\; \sum_{o} P(o)\,\big[-\ln P(o)\big]
\;=\; -\sum_{o} P(o)\ln P(o)
\;=\; \mathrm{H}[P(o)],
$$

which is [eq:sensory-entropy] exactly. So entropy is not merely *related to*
average surprise; it **is** average surprise, and the middle line is the whole
proof. Nothing deeper is happening in the independent case.

The organism's observations are of course not independent: what you see now
depends on what you saw a moment ago. The claim needed is therefore the stronger
one, that the time average along a single trajectory converges to the average
over the distribution being sampled. A process for which that holds is called
**ergodic**. Roughly: one long trajectory eventually visits states in the same
proportions as the underlying distribution assigns them, so following one
organism for a long time tells you what the ensemble looks like.
:::

::: keyidea
[eq:ergodic] converts a statement about a whole lifetime into a statement about
this instant. Keeping the entropy of your observations low over a lifetime is
the same thing as keeping your surprise low at each moment. A goal you could
never act on becomes a goal you can act on now.
:::

::: warning What this step assumes
[eq:ergodic] holds if the process is ergodic, and whether living systems are
ergodic in the required sense is exactly the point at which several careful
critics plant their flag. (The **free energy principle** is the name for the
whole claim being assembled here, that self-organising systems minimise a bound
on their surprise. Active inference is what you get when you take that principle
and ask how an agent should act. Week&nbsp;4 constructs the bound; Week&nbsp;12
weighs the objections.) We will take the
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
produces them.

This means the $P$ of [eq:sensory-entropy] and the $P$ of "minimise
$-\ln P(o)$" are not quite the same object, and they need separating now rather than later. The first is the distribution the organism's
observations *actually* follow over a lifetime, which no organism has access to.
The second is the distribution its model *says* they follow, which is the only
one it can evaluate. The framework proceeds by using the second as a stand-in for
the first. Where the model is good, the two are close and minimising one
minimises the other. Where the model is bad, an organism can be serenely
unsurprised while its actual sensory statistics wander off, which is one of the
more interesting ways for a thing to die. We will keep writing $P$ for both, as
the literature does, and say which is meant whenever it matters. Such a model is called a *generative model*; [lesson:1.2] defines
it properly and says what it is generative of. Building one, running it, and
correcting it is what the rest of this course is about.

So the argument so far runs:

1. To persist is to occupy a small set of physiological states.
2. Those states are sensed, so occupying few of them means the organism's
   observations are also confined to a small set: low entropy over observations.
3. Low entropy over a lifetime means low surprise at each moment.
4. Surprise is only defined relative to a model.
5. Therefore anything that persists must carry a model of the world it persists in.

Step 2 is the one to watch. It runs in only one direction. Narrow physiology
forces narrow interoceptive observations, but the converse fails: an organism
could in principle keep its observations narrow by sensing very little, which is
not the same as staying alive. That gap has a name, the dark-room problem, and
Week&nbsp;12 asks whether the standard reply to it works.

That is a strong claim and it deserves the scrutiny we will give it in
Week&nbsp;12. But notice what it does not claim. It says nothing about the model
being conscious, accurate, or even any good. A thermostat has one. The claim is
only that having one is what persistence looks like from the inside.

## Two ways to be less surprised

Suppose you have such a model, and an observation arrives that it did not
expect. The mismatch is between two things, the model and the world, so there are
two places to intervene.

You can **change the model** so that the observation is no longer surprising.
This is perception: the world is telling you something and you update
accordingly. What you thought was a shadow was a leopard, and now you believe it
was a leopard.

Or you can **change the world** so that your next observation is one the model
did expect. This is action: you are cold, your model expects to be warm, and
instead of revising the expectation you move into the sun.

::: mn Where the name comes from
Perception is inference over states. Action is inference carried out by moving.
Doing both at once, with one quantity governing both, is what puts the
*active* in active inference.
:::

Both reduce the mismatch between model and world. They are the same operation
applied to the two different halves of the coupling, and in the mathematics they
will turn out to be gradient descent on the same scalar quantity. That quantity
is the **variational free energy**, shortened to just "the free energy" once
there is no risk of confusing it with the thermodynamic quantity of the same
name. Week&nbsp;4 constructs it properly.

## Brains and sea squirts

There is a story that gets repeated whenever anyone argues that brains exist for
movement. The sea squirt, it goes, swims about as a larva with a little brain,
finds a rock, cements itself to it, and then, having no further use for the
thing, digests its own brain.

It is a good story and it is not true. The larval tunicate does resorb its
larval nervous system at metamorphosis, but what the adult builds afterwards is
a cerebral ganglion *larger* than the one the larva had, wired for a completely
different problem: filtering, siphon control,
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
- Name the two places you can intervene when model and world disagree. Which of
  them changes the model, and which changes the observations that arrive next?
:::
