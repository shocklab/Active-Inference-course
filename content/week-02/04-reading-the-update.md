---
title: Reading the update
deck: The rule from Lesson 3 has three moving parts. Each one has a job, and together they are a circuit that has been proposed as a model of cortex.
week: 2
time: 40 min
scripts: [w02.js]
---

The result of Lesson&nbsp;3 was

$$
\dot{d} \;=\; \eta\big(\varepsilon_p + g'(d)\,\varepsilon_u\big),
\qquad
\varepsilon_p = \frac{d_p - d}{\Sigma_p},
\qquad
\varepsilon_u = \frac{u - g(d)}{\Sigma_u}.
$$ {#recall}

Three moving parts: two errors and a factor multiplying one of them. This lesson
takes them one at a time, then puts the pieces back together as a circuit.

## Precisions

Dividing by a variance appears twice, so give it a name.

Lesson&nbsp;3 defined the precision of a Gaussian channel as the reciprocal of
its variance. Give the two of them symbols:

$$
\Pi_p \;=\; \frac{1}{\Sigma_p}, \qquad \Pi_u \;=\; \frac{1}{\Sigma_u}.
$$ {#precisions}

Their units are the inverse of a squared measurement, so $\Pi_u$ is in inverse
squared intensity and $\Pi_p$ in inverse squared distance. The two are not
comparable numbers and never appear added together, a point the curvature
exercise at the end returns to.

::: mn Why not just say variance
Because every rule in the rest of the course multiplies by a precision rather
than dividing by a variance, and the multiplicative form is the one that reads as
a gain on a signal. Weeks&nbsp;5 to 8 make that gain a physical quantity that the
system can adjust.
:::

Rewriting [eq:recall]:

$$
\dot{d} \;=\; \eta\Big(\Pi_p\,(d_p - d) \;+\; \Pi_u\,g'(d)\,\big(u - g(d)\big)\Big).
$$ {#precision-form}

Now the structure is visible. Each term is a raw error multiplied by a gain. The
gain says how seriously to take that error, and nothing else in the expression
says anything about seriousness.

With the numbers of this week, $\Pi_p = {{prec_prior}}$ and
$\Pi_u = {{prec_obs}}$: the animal treats its ear as ten times more reliable
than its expectations, in their respective units.

## The four corners

Push each precision to an extreme and read off what the rule does. These are not
edge cases to be handled; they are the whole behaviour of the rule, seen at its
limits.

<table>
<thead><tr><th>Limit</th><th>What the rule does</th><th>Reading</th></tr></thead>
<tbody>
<tr><td>$\Pi_u \to 0$</td><td>only the prior term survives; $d \to d_p$</td><td>A useless ear. The animal ignores what it heard and rests on what it expected.</td></tr>
<tr><td>$\Pi_u \to \infty$</td><td>the sensory term dominates; $d$ moves until $g(d) = u$</td><td>A perfect ear. The animal believes the measurement exactly and the prior stops mattering.</td></tr>
<tr><td>$\Pi_p \to 0$</td><td>only the sensory term survives</td><td>No expectations at all. A flat prior, and the estimate is whatever the data alone say.</td></tr>
<tr><td>$\Pi_p \to \infty$</td><td>the prior term dominates; $d$ is pinned at $d_p$</td><td>Certainty. No evidence can move the estimate, which is what makes an infinitely precise prior a description of a delusion rather than of knowledge.</td></tr>
</tbody>
</table>

Only the ratio matters, not the individual values. Multiplying both precisions by
the same constant multiplies the whole bracket in [eq:precision-form] by that
constant, which rescales $\eta$ and moves the resting point not at all. The
resting point is set by $\Pi_u/\Pi_p$, and the rate is set by their overall size:
the same distinction the step-size bound of Lesson&nbsp;3 ran into from the other
direction.

Between the corners, the estimate slides. With $\Sigma_u = {{var_obs_loose}}$ the
mode sits at ${{mode_loose:.4f}}$, close to the prior mean of ${{d_prior}}$. With
$\Sigma_u = {{var_obs_sharp}}$ it sits at ${{mode_sharp:.4f}}$, close to the
${{d_from_data_alone:.4f}}$ the data alone would give. At the
$\Sigma_u = {{var_obs}}$ used throughout, ${{post_mode:.4f}}$, in between.

## What $g'$ is doing

The factor $g'(d)$ appears on the sensory term and not on the prior term. It has
two jobs, and they are the same job seen twice.

**It converts units.** $\varepsilon_u$ is an error in intensity, and $\dot d$ is a
rate of change of distance. Something has to carry intensity into distance, and
$g'$ has units of intensity per distance, so $g'\varepsilon_u$ comes out in
inverse distance, matching $\varepsilon_p$. Without it the equation would be
adding quantities of different kinds, which is how you know the chain rule was
not optional.

**It gates by sensitivity.** $g'(d)$ measures how much the predicted intensity
would change if the estimate moved. Where it is large, a small change in the
estimate produces a large change in prediction, so a sensory error is strong
evidence about the estimate. Where it is small, the observation barely
distinguishes nearby distances, and the term correctly contributes little.

For the inverse square, $g'(d) = -2/d^{3}$, which falls off fast. At
$d = {{post_mode:.2f}}$ it is ${{gprime_at_mode:.4f}}$; at $d = {{d_prior}}$ it is
already only ${{gprime_at_prior:.4f}}$. Far from the source, loudness says
progressively less about distance, and the rule knows it without being told.

::: keyidea
The sensory error is weighted twice over: once by $\Pi_u$, how reliable the
channel is, and once by $g'$, how informative this particular measurement is
*here*. The first is a property of the ear. The second is a property of where the
estimate currently sits, so it changes as the estimate moves.
:::

The sign matters too. $g' < 0$ everywhere, since intensity falls with distance,
so a positive sensory error, louder than predicted, drives the estimate *down*.
Had the link been $g(d) = d^2$, the same error would drive it up. The rule needs
no separate knowledge of which way round the world works; the derivative carries
it.

## The case where it is all exact

Replace the inverse square with the identity, $g(d) = d$, so that the animal
measures distance directly with Gaussian noise. Then $g' = 1$ and the whole
apparatus collapses to something recognisable.

::: derivation The linear case gives a precision-weighted average
Set the gradient to zero in [eq:precision-form] with $g(d) = d$, $g'(d) = 1$:

$$
\Pi_p (d_p - \hat d) + \Pi_u (u - \hat d) \;=\; 0 .
$$

Expand and collect the terms in $\hat d$:

$$
\Pi_p d_p + \Pi_u u \;=\; \hat d\,(\Pi_p + \Pi_u),
$$

so

$$
\hat d \;=\; \frac{\Pi_p\,d_p + \Pi_u\,u}{\Pi_p + \Pi_u} .
$$ {#linear-map}

The estimate is the average of the prior mean and the measurement, each weighted
by its precision, and the weights sum to one because they have been divided by
their total.

More is true in this case. With $g$ linear, both exponents in the joint are
quadratic in $d$, so their sum is quadratic, so the posterior is exactly
Gaussian: it has no skew and its mode and mean coincide. Its precision is
$\Pi_p + \Pi_u$, precisions of independent sources simply adding.
:::

Put the numbers in. $\Pi_p = {{prec_prior}}$, $\Pi_u = {{prec_obs}}$,
$d_p = {{d_prior}}$, $u = {{u_obs}}$:

$$
\hat d \;=\; \frac{{{prec_prior}} \times {{d_prior}} + {{prec_obs}} \times {{u_obs}}}{{{prec_prior}} + {{prec_obs}}}
\;=\; {{lin_mode:.6f}} ,
$$

with variance $1/(\Pi_p + \Pi_u) = {{lin_var:.6f}}$.

Checking that against quadrature on the same wide grid: the mode comes out at
${{lin_mode_quad:.6f}}$, the mean at ${{lin_mean_quad:.6f}}$, and the skew at
${{lin_skew_quad:.1e}}$, which is zero to machine precision. Mode and mean agree
to six figures, as the algebra says they must.

::: keyidea
Compare that with Lesson&nbsp;2, where the mode and mean of the same problem
differed by {{mean_mode_pct:.0f}}%. The ingredients were Gaussian in both cases
and the only difference is the link function. So the entire gap is the
nonlinearity, and none of it is the choice of prior.

That also explains why the answer had to be found by climbing rather than by
formula. [eq:linear-map] is a closed form because setting a *linear* gradient to
zero is a linear equation. With $g(d) = 1/d^{2}$, setting [eq:precision-form] to
zero gives a polynomial in $d$ of degree five, and there is no general formula
for its roots. Iteration is not laziness; it is what is left.
:::

::: mn The bill for the Gaussian prior
Lesson&nbsp;2 flagged that a Gaussian over a distance puts mass below zero. Here
is the size of it. In the linear case the posterior is centred at
${{lin_mode:.4f}}$ with a standard deviation of ${{lin_sd_quad:.4f}}$, so
{{lin_mass_below_zero:.1%}} of its mass sits at negative distances, which is
small but not nothing. In the nonlinear case the same prior is harmless: as $d$
falls towards zero, $g(d)$ runs to infinity and the squared sensory error with
it, so the likelihood suppresses the region entirely. The mass below
$d = 0.5$ is about ${{nonlin_mass_below_half:.0e}}$. The inverse square cleans up
after the prior.
:::

## The circuit

Now assemble [eq:precision-form] as a diagram, taking seriously the constraints
of Lesson&nbsp;1. Give each quantity a unit and each multiplication a connection.

::: fig The update of [eq:precision-form] drawn as a circuit. Filled nodes hold estimates, open nodes hold errors. Every arrow carries one signal to one place, and no arrow needs a quantity from anywhere else.
<svg viewBox="0 0 660 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A predictive coding circuit: an estimate unit sends a prediction down to a sensory error unit and receives a weighted error back, while a prior error unit compares the estimate with the prior expectation.">
  <rect x="1" y="1" width="658" height="258" fill="#fdfcf9" stroke="#e4e1d9"/>

  <circle cx="150" cy="70" r="26" fill="#0f5f57"/>
  <text x="150" y="75" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="17" fill="#ffffff">d</text>
  <text x="150" y="32" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">estimate</text>

  <circle cx="420" cy="70" r="24" fill="#fdfcf9" stroke="#2c5f8a" stroke-width="2"/>
  <text x="420" y="76" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="16" fill="#2c5f8a">&#949;&#8346;</text>
  <text x="420" y="32" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">prior error</text>

  <circle cx="150" cy="196" r="24" fill="#fdfcf9" stroke="#8a6d1f" stroke-width="2"/>
  <text x="150" y="202" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="16" fill="#8a6d1f">&#949;&#7524;</text>
  <text x="150" y="240" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">sensory error</text>

  <rect x="392" y="172" width="56" height="30" rx="4" fill="#f4f2ec" stroke="#d3cfc4"/>
  <text x="420" y="192" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="15" fill="#3c3c42">d&#8346;</text>
  <text x="420" y="222" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">expectation</text>

  <rect x="20" y="181" width="56" height="30" rx="4" fill="#f4f2ec" stroke="#d3cfc4"/>
  <text x="48" y="201" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="15" fill="#3c3c42">u</text>
  <text x="48" y="231" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">the ear</text>

  <defs>
    <marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#3c3c42"/>
    </marker>
    <marker id="ahg" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#8a6d1f"/>
    </marker>
    <marker id="ahb" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#2c5f8a"/>
    </marker>
  </defs>

  <path d="M133 93 C112 128 112 150 133 178" fill="none" stroke="#0f5f57" stroke-width="1.8" marker-end="url(#ah)"/>
  <text x="72" y="136" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="13" fill="#0f5f57">g(d)</text>
  <text x="72" y="152" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#6d6d75">prediction</text>

  <path d="M170 176 C196 146 196 106 170 82" fill="none" stroke="#8a6d1f" stroke-width="1.8" marker-end="url(#ahg)"/>
  <text x="238" y="128" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="13" fill="#8a6d1f">g&#8242;(d)&#183;&#928;&#7524;</text>
  <text x="238" y="144" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#6d6d75">weighted error back</text>

  <path d="M76 196 L124 196" fill="none" stroke="#3c3c42" stroke-width="1.6" marker-end="url(#ah)"/>
  <path d="M392 187 L176 193" fill="none" stroke="#3c3c42" stroke-width="1.2" stroke-dasharray="3 3"/>

  <path d="M176 70 L394 70" fill="none" stroke="#0f5f57" stroke-width="1.8" marker-end="url(#ah)"/>
  <path d="M420 172 L420 96" fill="none" stroke="#3c3c42" stroke-width="1.6" marker-end="url(#ah)"/>
  <path d="M400 92 C330 126 240 100 176 78" fill="none" stroke="#2c5f8a" stroke-width="1.8" marker-end="url(#ahb)"/>
  <text x="300" y="122" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="13" fill="#2c5f8a">&#928;&#8346;</text>
</svg>
:::

Trace it. The estimate $d$ sends a prediction $g(d)$ down to the sensory error
unit, which subtracts it from what the ear delivered and scales by $\Pi_u$. That
error travels back up, multiplied by $g'(d)$ on the way. Separately the estimate
is compared with the expectation $d_p$, and that error returns scaled by $\Pi_p$.
The estimate changes at a rate given by the sum of what comes back.

Check it against Lesson&nbsp;1. Every arrow carries one number from one unit to
one unit. No arrow requires a sum over states, and no unit reads a variable held
somewhere it is not connected to. The rule satisfies local computation.

::: definition Predictive coding
A scheme in which a system maintains estimates of hidden causes, sends
predictions from those estimates down towards the senses, computes the mismatch
between prediction and input at each stage, and sends that mismatch back up to
correct the estimates. What flows downward are predictions; what flows upward are
prediction errors.
:::

That is what has been derived, and it was not the goal. The goal was to maximise
$\mathcal{F}$, and the only steps taken were the chain rule and the decision to
climb a gradient. The circuit is what those steps look like when drawn.

::: warning What has not been shown
Deriving a circuit that could be implemented in tissue is not evidence that it
is. The argument here establishes that predictive coding is a correct
gradient-ascent scheme for a particular estimation problem, and nothing about
whether cortex does it.

Two gaps are worth naming now. First, the update carries a point estimate and no
uncertainty at all: this is the point-mass case flagged in Lesson&nbsp;3, and
Week&nbsp;4 puts the uncertainty back. Second, the precisions and the function
$g$ have been treated as known, when an animal is given neither. Lesson&nbsp;5
of Week&nbsp;3 takes that up.
:::

::: fig Vary the two precisions and watch the posterior move. The mode is what the update of this lesson finds; the mean is what it does not.
<div class="widget" data-widget="precision-posterior"></div>
:::

::: exercise Where the curvature comes from
Lesson&nbsp;3 needed $\mathcal{F}''$ to bound the step size. Differentiate
[eq:precision-form] once more to get it, and evaluate the pieces at
$\hat d = {{post_mode:.4f}}$. You will need
$g''(d) = 6/d^{4} = {{gsecond_at_mode:.4f}}$ there.
---solution---
Differentiating $\mathcal{F}'(d) = \Pi_p(d_p - d) + \Pi_u g'(d)(u - g(d))$ with
respect to $d$, the first term gives $-\Pi_p$. The second needs the product rule
on $g'(d)$ and $(u - g(d))$:

$$
\frac{\mathrm{d}}{\mathrm{d}d}\Big[g'(d)\big(u - g(d)\big)\Big]
= g''(d)\big(u - g(d)\big) \;+\; g'(d)\cdot\big(-g'(d)\big).
$$

So

$$
\mathcal{F}''(d) \;=\; -\Pi_p \;-\; \Pi_u\,g'(d)^2 \;+\; \Pi_u\,g''(d)\big(u - g(d)\big).
$$

The first two terms are always negative and bend the function downwards. The
third has the sign of the residual sensory error and can be either.

At $\hat d$: the first pair is
$-{{prec_prior}} - {{sensory_prec_on_state:.4f}} = {{curv_gaussnewton:.4f}}$,
and the third is
$\varepsilon_u\,g''(\hat d) = {{err_obs:.4f}} \times {{gsecond_at_mode:.4f}}
= {{curv_gpp_term:.4f}}$. Together, ${{curv_at_mode:.4f}}$, which is the number
Lesson&nbsp;3 used.

The quantity $\Pi_u g'^2 = {{sensory_prec_on_state:.4f}}$ is the sensory
channel's precision *expressed about the state* rather than about the
observation, converted by the same $g'$ that appears in the update, squared
because a precision is a reciprocal squared quantity. Dropping the $g''$ term and
keeping only $-\Pi_p - \Pi_u g'^2$ is the Gauss-Newton approximation, and
Week&nbsp;4 will meet it again as the curvature of a Laplace approximation.
:::

::: exercise Two ears
The animal grows a second ear with its own noise, so it receives two independent
readings $u_1$ and $u_2$ of the same source, with precisions $\Pi_1$ and $\Pi_2$.
Write down the log joint and derive the update rule. What happens to the
resting point if the two ears are equally precise?
---solution---
Independence given the state means the joint likelihood factorises,
$P(u_1, u_2 \mid d) = P(u_1 \mid d)P(u_2 \mid d)$, so the logarithm is a sum:

$$
\mathcal{F} = \ln P(d) + \ln P(u_1 \mid d) + \ln P(u_2 \mid d).
$$

Each new term is the same shape as before, so differentiating gives one more
sensory term of the same form:

$$
\dot d = \eta\Big(\Pi_p(d_p - d) + g'(d)\big[\Pi_1(u_1 - g(d)) + \Pi_2(u_2 - g(d))\big]\Big).
$$

The single $g'(d)$ factors out, because both ears are looking at the same hidden
variable through the same link. Each channel contributes its own error at its own
gain, and the estimate responds to the sum. This is the sense in which evidence
accumulates: adding a channel adds a term, and never requires the existing terms
to be recomputed.

With $\Pi_1 = \Pi_2 = \Pi$ the bracket becomes
$\Pi\big[(u_1 + u_2) - 2g(d)\big] = 2\Pi\big[\bar u - g(d)\big]$ with
$\bar u$ the mean of the two readings. Two equally precise ears are exactly one
ear of twice the precision, reporting their average. The resting point is what a
single ear would give on the averaged reading, and it is reached with a stiffer
curvature, so by the step-size bound of Lesson&nbsp;3 a smaller step.
:::

::: checkpoint
- Write the update in precision form and say what each gain is doing.
- Why does $g'$ appear on one term and not the other? Give both the units answer
  and the sensitivity answer.
- The posterior in Lesson&nbsp;2 was skewed and the one in this lesson's linear
  case is not. What is responsible, and what is not?
- Which two things does this week's rule treat as known that an animal would not
  be given?
:::
