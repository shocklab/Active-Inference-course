---
title: Regulation and inference
deck: A feedback model of physiological regulation, followed by the questions that connect regulation to inference.
week: 1
time: 35 min
scripts: [w01.js]
---

A body must keep some things within tolerable ranges. Temperature, acidity and
salt concentration can vary, but sufficiently large departures disrupt the
processes that keep the organism alive. Shivering, sweating and moving into the
shade are familiar ways of correcting those departures.

We can describe regulation mathematically before deciding how much of it requires
inference. That distinction matters. A feedback mechanism may stabilise a variable
without calculating probabilities or representing a hidden cause. Active inference
is a proposal for connecting regulation, perception and action; a model of stable
feedback alone does not establish that proposal.

## A feedback model

Call the set of physiological states compatible with survival the **viable set**.
For a small example, take two regulated variables and measure each as a deviation
from its target. Scale their units so that the viable set is the circle of radius
one. This circular boundary is an assumption of the example, not a claim about the
shape of a real organism's physiological limits.

Write $x_t\in\mathbb R^2$ for the displacement at time step $t$, starting at
$x_0=0$. The subscript labels time. At each step the controller removes a fraction
$\kappa$ of the current displacement and then a disturbance arrives:

$$
x_{t+1}=(1-\kappa)x_t+\sigma\xi_t,
\qquad \xi_t\sim\mathcal N(0,I_2)
\quad\text{independently across steps}.
$$ {#homeostat}

Here $I_2$ is the identity matrix: each coordinate of $\xi_t$ has variance one,
and the two coordinates are independent. The **noise standard deviation**
$\sigma$ sets the size of a disturbance in each coordinate, in the units of $x$.
The **feedback gain** $\kappa$ is dimensionless. It is the fraction of the present
displacement removed per step. A gain of $0.15$ removes fifteen per cent, leaving
eighty-five per cent before the next disturbance arrives.

For the survival simulation, stop the trajectory when $\lVert x_t\rVert>1$.
This is an **absorbing boundary**: crossing ends the run, even if the recursion
would subsequently return inside. We will also study the recursion without that
boundary to understand its spread. These are two different processes.

::: widget homeostasis-drift | Two regulated coordinates, measured relative to their targets. The solid circle is the absorbing boundary at radius one. The dashed circle shows the root-mean-square radius $\sqrt{2v}$ of the unabsorbed stationary process, with $v$ defined below; it is not a safety guarantee. The right panel records the entropy of the empirical visit histogram on a ${{hs_bins:d}}\times{{hs_bins:d}}$ grid. Its reference maximum is $\ln {{hs_bin_cells:d}}={{hs_max_entropy:.3f}}$ nats for uniform occupancy of the whole square grid. The absorbing circle prevents this run from exploring the whole square. A nat is an information unit based on natural logarithms, defined in the entropy section.
:::

### Without feedback

At $\kappa=0$, repeated substitution in [eq:homeostat] gives

$$
x_t=\sigma\sum_{j=0}^{t-1}\xi_j.
$$

The increments are independent and have mean zero. Thus each coordinate has mean
zero and variance $t\sigma^2$. Adding the second moments of the coordinates gives

$$
\mathbb E[\lVert x_t\rVert^2]=2t\sigma^2.
$$ {#diffusion}

This calculation concerns the unabsorbed walk. Equating its mean squared radius
to the squared boundary radius gives the rough timescale
$t\approx1/(2\sigma^2)$. At $\sigma={{hs_sigma}}$, that is about
{{hs_diffusive_steps:.0f}} steps. It is not an exact expected first-crossing time:
a path can cross the boundary earlier and return before time $t$.

The boundary is eventually crossed with probability one when $\sigma>0$.
One way to see this is to look at the unabsorbed position: it is Gaussian with
covariance $t\sigma^2I_2$, so the probability of being inside the unit circle
tends to zero. The event of never having left by time $t$ is contained in the
event of being inside at time $t$. Its probability must therefore tend to zero
too.

### With feedback

::: derivation The stationary variance
Let $v_t$ be the variance of one coordinate of the unabsorbed process. The new
noise is independent of the existing displacement, so there is no covariance
term when we take the variance of [eq:homeostat]:

$$
v_{t+1}=(1-\kappa)^2v_t+\sigma^2.
$$

A stationary variance $v$ must reproduce itself:

$$
v=(1-\kappa)^2v+\sigma^2,
\qquad
[1-(1-\kappa)^2]v=\sigma^2.
$$

Expanding the square gives $1-(1-\kappa)^2=2\kappa-\kappa^2$, hence

$$
v=\frac{\sigma^2}{\kappa(2-\kappa)}.
$$ {#stationary-var}

For this value to be attracting, the multiplier of $v_t$ must be below one:
$(1-\kappa)^2<1$. Solving that inequality gives

$$
0<\kappa<2.
$$ {#stability}

Under this condition, repeated substitution shows that the initial variance is
multiplied by $(1-\kappa)^{2t}$ and tends to zero, while the accumulated noise
variance is a convergent geometric sum. The stationary distribution is Gaussian,
with independent coordinates of variance $v$.
:::

At $\sigma={{hs_sigma}}$ and $\kappa={{hs_kappa}}$, the variance is
${{hs_var_per_coord:.5f}}$ per coordinate, giving standard deviation
${{hs_sd_per_coord:.4f}}$. Since the mean is zero,
$\mathbb E[\lVert x\rVert^2]=2v$, which explains the dashed circle in the widget.

| Gain | Response without noise | Stationary variance with noise |
|---|---|---|
| $\kappa=0$ | Displacement is unchanged | No finite stationary variance |
| $0<\kappa<1$ | Displacement decreases without changing sign | $\sigma^2/[\kappa(2-\kappa)]$ |
| $\kappa=1$ | Displacement is removed in one step | $\sigma^2$ |
| $1<\kappa<2$ | Displacement decreases while alternating in sign | $\sigma^2/[\kappa(2-\kappa)]$ |
| $\kappa=2$ | Displacement alternates without decreasing | No finite stationary variance |

The denominator satisfies $\kappa(2-\kappa)=1-(\kappa-1)^2$, so it is largest
at $\kappa=1$. Therefore

$$
v_{\min}=\sigma^2.
$$ {#best-gain}

Even that controller cannot remove the disturbance which arrives after its
correction. Gains $\kappa$ and $2-\kappa$ give the same stationary variance,
although the sign of the temporal correlation differs. A gain of $0.5$ and one
of $1.5$ therefore produce equally broad stationary distributions but visibly
different trajectories.

::: warning Stationarity and survival
The stationary calculation removes the absorbing boundary. It describes spread,
not the distribution of surviving paths, and not a guarantee of indefinite life.
Gaussian disturbances have unbounded support. From anywhere inside the circle,
a sufficiently large disturbance can cause an exit; for fixed gain and positive
noise, there is a positive lower bound on this exit probability over the circle.
Consequently eventual exit still has probability one. Feedback can make exit
rare, but estimating its waiting time requires a first-passage calculation.
:::

## Entropy and observations

The widget's histogram describes physiological positions visited by a trajectory.
An organism may also observe its own condition. This is **interoception**:
sensing internal quantities such as temperature or muscle tension. It also
receives information about its surroundings.

Write $o$ for an **observation**, with the letter chosen for that word. For now,
use a finite set of possible observations, such as a sensor reporting one of
four categories. Write $P^*(o)$ for their actual probability distribution. The
star will distinguish the process generating observations from a model of it.

The **Shannon entropy** of a discrete distribution is

$$
\mathrm H[P^*]= -\sum_o P^*(o)\ln P^*(o).
$$ {#sensory-entropy}

We use the convention $0\ln0=0$, obtained by taking the limit as the probability
approaches zero. The brackets indicate that entropy takes a whole distribution
as its argument. For a finite outcome set this is a function of the probability
vector. A **nat** is the unit obtained using natural logarithms; base-two
logarithms instead give bits.

For the chosen distribution $(0.94,0.02,0.02,0.02)$,

$$
\mathrm H[P^*]=-[0.94\ln0.94+3(0.02\ln0.02)]
={{ent_peaked:.4f}}\text{ nats}.
$$

For four equally likely categories, each contribution is
$-\tfrac14\ln\tfrac14$, and their sum is
$\ln4={{ent_uniform4:.4f}}$ nats. This is the largest entropy on four categories.
Entropy describes how probability is distributed across the categories. It does
not say whether any of those categories is safe or desirable.

::: exercise A constant body and a noisy sensor
Suppose a physiological variable stays exactly at its target, but a faulty sensor
reports four categories independently and with equal probability. What are the
entropy of the physiological state and the entropy of the sensor report?
---solution---
The physiological variable has one possible value, so its discrete entropy is
$-1\ln1=0$. The report has entropy $\ln4={{ent_uniform4:.4f}}$ nats.
Thus narrow physiology alone does not bound observation entropy. For a
deterministic discrete observation $o=h(x)$, we do have $\mathrm H[o]\le\mathrm H[x]$;
sensor noise removes the premise of that statement. The entropy of external
observations need not be small either: a well-regulated animal can look at a
highly variable scene.
:::

## Surprise and its average

Given a distribution $P(o)$, the **surprise** or **surprisal** of an observation
is $-\ln P(o)$. This measures its improbability under that distribution, not an
emotional response. A certain event has zero surprise, and a less probable event
has greater surprise. For the preceding example, the common category has surprise
${{surprise_common:.4f}}$ nats and a rare category has
${{surprise_rare:.4f}}$ nats.

::: derivation Entropy is expected surprise
If observations are independent draws from $P^*$, and entropy is finite, the
strong law of large numbers gives

$$
\frac1T\sum_{t=1}^T[-\ln P^*(o_t)]
\longrightarrow\mathbb E_{P^*}[-\ln P^*(o)]
= -\sum_oP^*(o)\ln P^*(o)=\mathrm H[P^*]
$$ {#ergodic}

almost surely. Here $T$ counts observations and $t$ indexes them. Independence
is sufficient but not necessary. A stationary ergodic process also supports this
time-average identity under the appropriate integrability condition. In this
use, **ergodicity** means that time averages along almost every trajectory agree
with the corresponding distributional averages. Stationarity by itself is not
enough.
:::

An identity about an average does not prescribe a choice at every instant. A
controller may accept an unusual observation now to prevent a dangerous outcome
later. Nor does the identity identify a preferred distribution: a sensor that
always reports failure has low entropy too.

There is another distinction. An agent's model assigns probabilities $P(o)$,
which can differ from the actual probabilities $P^*(o)$. Its expected surprise is
then $\sum_oP^*(o)[-\ln P(o)]$, rather than the entropy of $P^*$.
[lesson:1.2] works through a lamp example showing the difference.

## Inference, learning and action

An animal hears a sound but cannot directly measure the source's distance. It can
use a model linking possible distances to possible sounds to infer the distance.
A generative model specifies a joint distribution over observations and
hypothesised hidden causes. [lesson:1.2] constructs one explicitly.

We will distinguish three operations:

| Operation | What changes | Example |
|---|---|---|
| Inference | Beliefs about a hidden state, with the generative model fixed | Revising the likely distance after hearing a sound |
| Learning | Parameters or structure of the generative model | Revising how noisy the hearing channel is |
| Action | The process producing subsequent observations | Moving closer and listening again |

In particular, inference about a fixed observation does not change the probability
that the fixed model assigned to it. It changes beliefs about its cause.
[lesson:1.2] gives a small interactive calculation of this distinction.

Active inference uses a quantity called variational free energy to construct
inference and, with further assumptions, learning and action rules. For a fixed
observation it bounds surprise from above; changing the approximate belief can
reduce the gap without changing surprise itself. [week:4] derives this bound.
Planning also needs preferences over outcomes and predictions of the consequences
of actions. These do not follow just from the existence of a stable feedback loop.

The task for the course is to build these ingredients and show what follows from
them. Claims that organisms implement the resulting equations require evidence
in addition to the equations themselves.

::: checkpoint
- Derive the stationary variance and state which process it describes.
- Why can an unabsorbed process have a stationary distribution while a run with an absorbing boundary eventually ends?
- Give an example where physiological state entropy and observation entropy differ.
- Why does low average surprise not require low surprise at each moment?
- Which quantities change during inference, learning and action?
:::
