---
title: Climbing the log joint
deck: Give up on the distribution and chase its peak instead. The intractable term drops out, and what is left is a difference between two locally available quantities.
week: 2
time: 45 min
scripts: [w02.js]
---

Last lesson left two obstacles. The normaliser $P(u)$ is an integral over the
whole state space, and the shape of the posterior belongs to no convenient
family. This lesson removes both, at a price that will be named exactly.

## Ask for less

The animal does not obviously need the whole posterior. If it must act, one
number might do: the single distance the evidence most supports.

::: definition Maximum a posteriori estimate
The **MAP estimate** is the value of the hidden variable at which the posterior
is largest,
$$
\hat{d} \;=\; \operatorname*{arg\,max}_{d} \; P(d \mid u).
$$
It is the mode of the posterior, not its mean, and Lesson&nbsp;2 showed those can
differ substantially.
:::

Now the move that makes everything work. Substitute Bayes' theorem:

$$
\hat{d} \;=\; \operatorname*{arg\,max}_{d} \; \frac{P(u \mid d)\,P(d)}{P(u)} .
$$

The denominator does not contain $d$. It is a single fixed number, whatever it
happens to be, and dividing every value of a function by the same positive
constant does not move where that function peaks. So it goes:

$$
\hat{d} \;=\; \operatorname*{arg\,max}_{d} \; P(u \mid d)\,P(d).
$$ {#argmax-joint}

::: keyidea
The intractable term has vanished, and not by approximating it. Asking *where*
the posterior peaks rather than *what value* it takes there is a question the
normaliser has no bearing on, so it cancels exactly.

This is the single most useful structural fact in the course, and every method
from here to Week&nbsp;11 is a variation on it: arrange the question so that the
quantity you cannot compute does not appear in the answer.
:::

The product $P(u \mid d)P(d)$ is the joint density $P(u, d)$, by the product
rule. So the agent maximises the joint. Both factors are things it has: a prior
it holds and a likelihood it can evaluate by predicting an intensity and
comparing.

## Take the logarithm

Two reasons, one practical and one that matters later.

The practical one: both factors are exponentials, and a logarithm turns the
product into a sum and strips the exponentials off. The one that matters later:
the log of a probability is the negative surprise from Week&nbsp;1, so working in
logs keeps the connection to that quantity visible instead of buried.

The logarithm is increasing, so it does not move the maximum either. Define

::: definition The log joint
$$
\mathcal{F}(d, u) \;=\; \ln P(d) \;+\; \ln P(u \mid d).
$$ {#logjoint}
Maximising $\mathcal{F}$ over $d$ gives the MAP estimate.
:::

::: warning This is not yet free energy
$\mathcal{F}$ is written with a script letter because it is going to become the
central object of the course, but it is not that object yet. Variational free
energy, defined in Week&nbsp;4, is a functional of a whole distribution
$Q(d)$. What is written here is the log joint evaluated at a single point.

The relationship, stated now and derived in Week&nbsp;4: the negative variational
free energy reduces to [eq:logjoint] exactly when $Q$ is a point mass at $d$.
Everything in this week is therefore the special case in which the agent's
beliefs carry no uncertainty at all. Week&nbsp;3 puts the uncertainty back.
:::

## Write it out

Substitute the two Gaussians. Taking the prior first:

$$
\ln P(d) \;=\; \ln\!\left[\frac{1}{\sqrt{2\pi\Sigma_p}}\exp\!\left(-\frac{(d - d_p)^2}{2\Sigma_p}\right)\right]
\;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \tfrac{1}{2}\ln(2\pi\Sigma_p),
$$

using $\ln(ab) = \ln a + \ln b$ and $\ln e^{x} = x$. The likelihood goes the same
way, with $g(d)$ in place of the mean:

$$
\ln P(u \mid d) \;=\; -\frac{(u - g(d))^2}{2\Sigma_u} \;-\; \tfrac{1}{2}\ln(2\pi\Sigma_u).
$$

Both trailing terms are constants: they contain the variances, which are fixed,
and no $d$. Since we are only ever going to differentiate with respect to $d$,
they will die at the first derivative, and we may drop them now:

$$
\mathcal{F} \;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \frac{(u - g(d))^2}{2\Sigma_u} \;+\; \text{const}.
$$ {#F-explicit}

Read what this says before differentiating it. $\mathcal{F}$ is large when both
squared terms are small: when $d$ is near what the prior expected, *and* when the
intensity that $d$ predicts is near the intensity actually heard. It is a
scoring rule with two ways to lose marks, and the variances set the exchange
rate between them.

## Differentiate

::: derivation The gradient of the log joint
Differentiate [eq:F-explicit] term by term.

The first term. Write $A = -(d - d_p)^2 / (2\Sigma_p)$. By the chain rule, with
the outer function $(\cdot)^2$ and the inner $d - d_p$ whose derivative is $1$:

$$
\frac{\partial A}{\partial d}
= -\frac{1}{2\Sigma_p} \cdot 2(d - d_p) \cdot 1
= -\frac{d - d_p}{\Sigma_p}
= \frac{d_p - d}{\Sigma_p}.
$$

The sign flip in the last step is worth pausing on. It makes the term positive
when $d$ is below the prior mean, so the gradient pushes $d$ up towards $d_p$,
and negative when $d$ is above it. The term always points at the prior's
expectation.

The second term. Write $B = -(u - g(d))^2 / (2\Sigma_u)$. Now the inner function
is $u - g(d)$, whose derivative with respect to $d$ is $-g'(d)$, since $u$ is a
fixed number the animal has already heard:

$$
\frac{\partial B}{\partial d}
= -\frac{1}{2\Sigma_u} \cdot 2\big(u - g(d)\big) \cdot \big(-g'(d)\big)
= \frac{u - g(d)}{\Sigma_u}\, g'(d).
$$

Two minus signs cancelled: one from the leading $-\tfrac{1}{2\Sigma_u}$ and one
from differentiating $-g(d)$.

Adding them, and noting the constant contributes nothing:

$$
\frac{\partial \mathcal{F}}{\partial d}
\;=\; \underbrace{\frac{d_p - d}{\Sigma_p}}_{\text{prior term}}
\;+\; \underbrace{\frac{u - g(d)}{\Sigma_u}\,g'(d)}_{\text{sensory term}} .
$$ {#gradient}
:::

## Two prediction errors

Both terms in [eq:gradient] have the same shape: something observed minus
something predicted, divided by a variance. Name them.

::: definition Precision-weighted prediction error
$$
\varepsilon_p \;=\; \frac{d_p - d}{\Sigma_p},
\qquad
\varepsilon_u \;=\; \frac{u - g(d)}{\Sigma_u}.
$$ {#errors}
$\varepsilon_p$ is the **prior prediction error**: how far the current estimate
sits from what was expected before any data arrived. $\varepsilon_u$ is the
**sensory prediction error**: how far the intensity actually heard sits from the
intensity the current estimate predicts.

Each is divided by a variance. The next definition names what that division is
doing, and Lesson&nbsp;4 is largely about it.
:::

::: definition Precision
The **precision** of a Gaussian channel is the reciprocal of its variance. A
precise channel is one whose predictions are tight, so dividing an error by a
variance is the same as multiplying it by a precision: a gain saying how
seriously to take that error. The symbol $\Pi$ is reserved for it from
Lesson&nbsp;4 onwards.
:::

::: mn Where the subscripts come from
$\varepsilon_p$ carries the subscript of the prior, $\varepsilon_u$ that of the
observation $u$. Weeks&nbsp;5 onwards will have a chain of these, one per level,
and the subscripts become level indices.
:::

In this notation the gradient is short:

$$
\frac{\partial \mathcal{F}}{\partial d} \;=\; \varepsilon_p \;+\; g'(d)\,\varepsilon_u .
$$ {#gradient-eps}

Now check [eq:gradient-eps] against the constraints from Lesson&nbsp;1. To
compute $\varepsilon_u$ a unit needs $u$, which arrives at it, and $g(d)$, which
is a prediction sent to it. To compute $\varepsilon_p$ it needs $d_p$, held
locally, and $d$, the current estimate. Nothing has to be fetched from across the
network and nothing is summed over the state space. The rule is local.

That is a real result, and it was not arranged. We differentiated the log joint
because that is what maximising a function requires, and a locally computable
quantity came out.

## Climbing

Having the gradient, take the obvious step. Move in the direction it points:

$$
\dot{d} \;=\; \eta \, \frac{\partial \mathcal{F}}{\partial d}
\;=\; \eta\big(\varepsilon_p + g'(d)\,\varepsilon_u\big),
$$ {#ascent}

with $\eta > 0$ a rate constant setting how large a step to take, in units of
distance per unit of gradient. Equivalently, in discrete steps,
$d_{k+1} = d_k + \eta\,\partial\mathcal{F}/\partial d\big|_{d_k}$.

::: mn Why $\dot{d}$ and not $\Delta d$
The dot is Newton's notation for a time derivative. Writing the update as a flow
rather than a jump anticipates Weeks&nbsp;5 to 8, where the estimate genuinely is
a continuous physical quantity relaxing in real time, and the discrete version is
what you get by sampling it.
:::

Where does it stop? At any point where the gradient vanishes, which by
[eq:gradient-eps] means

$$
\varepsilon_p \;=\; -\,g'(d)\,\varepsilon_u .
$$ {#balance}

The estimate settles where the two errors balance. Not where either is zero:
where the pull towards the prior exactly offsets the pull towards the data.

## The numbers

Run it. Start the animal at its prior expectation, $d = {{d_prior}}$, and take
$\eta = {{ascent_rate}}$.

At the start the prior error is exactly zero, because the estimate *is* the prior
mean. The sensory error is not: the estimate predicts an intensity of
${{g_at_prior:.3f}}$ and the animal heard ${{u_obs}}$, so
$\varepsilon_u = ({{u_obs}} - {{g_at_prior:.3f}})/{{var_obs}} = {{err_obs_at_start:.3f}}$.
Weighted by $g'({{d_prior}}) = {{gprime_at_prior:.3f}}$, the sensory term
contributes ${{err_obs_weighted_at_start:.3f}}$. The gradient is negative, so the
estimate moves down: the sound was louder than expected, therefore the source is
closer than expected.

::: fig The estimate climbing the log joint from the prior mean. Left, the trajectory against the exact posterior mode. Right, the two terms of [eq:gradient-eps] over the same steps, meeting where their sum is zero.
<div class="widget" data-widget="ascent-errors"></div>
:::

It settles at $d = {{ascent_final:.4f}}$, reaching within one per cent of the
exact mode after {{ascent_steps_to_1pct}} steps. The mode computed by quadrature
in Lesson&nbsp;2 was ${{post_mode:.4f}}$. The two agree, which they must: gradient
ascent on the log joint finds the peak of the posterior, and the peak is what
quadrature located.

At that point the two terms are

$$
\varepsilon_p = {{err_prior:+.6f}},
\qquad
g'(\hat d)\,\varepsilon_u = {{err_obs_weighted:+.6f}},
$$

summing to ${{err_sum:.2e}}$, which is zero to the tolerance the iteration was
run to. Note what is *not* zero: the raw sensory error is
$\varepsilon_u = {{err_obs:.4f}}$, and the estimate still predicts an intensity
of ${{g_at_mode:.4f}}$ against the ${{u_obs}}$ heard. The animal has not explained
the observation away. It has stopped at the point where explaining more of it
would cost more in prior implausibility than it gains in fit.

## How large a step

$\eta$ was set to ${{ascent_rate}}$ without comment. It cannot be set freely, and
the bound on it turns out to depend on the precisions, which is the first sign of
something the rest of the course keeps running into.

::: derivation The largest stable rate
Write $d^{*}$ for the peak and $e_k = d_k - d^{*}$ for the error at step $k$. Near
the peak, expand the gradient to first order. Since $\mathcal{F}'(d^{*}) = 0$ by
definition of the peak,

$$
\mathcal{F}'(d_k) \;\approx\; \mathcal{F}'(d^{*}) + \mathcal{F}''(d^{*})\,e_k
\;=\; \mathcal{F}''(d^{*})\,e_k .
$$

Substituting into the discrete update $d_{k+1} = d_k + \eta\,\mathcal{F}'(d_k)$
and subtracting $d^{*}$ from both sides:

$$
e_{k+1} \;=\; e_k + \eta\,\mathcal{F}''(d^{*})\,e_k \;=\; \big(1 + \eta\,\mathcal{F}''(d^{*})\big)\,e_k .
$$

The error is multiplied by a fixed factor each step, so it shrinks precisely when
that factor is less than one in magnitude:

$$
\big|\,1 + \eta\,\mathcal{F}''(d^{*})\,\big| \;<\; 1 .
$$

At a maximum $\mathcal{F}'' < 0$, so writing $|\mathcal{F}''|$ for its magnitude the
condition is $-1 < 1 - \eta|\mathcal{F}''| < 1$, and the right inequality holds for
any $\eta > 0$. The left one gives

$$
\eta \;<\; \frac{2}{|\mathcal{F}''(d^{*})|} .
$$ {#eta-bound}
:::

Read the three regimes off [eq:eta-bound]. Below the bound the error decays and
the estimate approaches the peak from one side. Above it the factor
$1 + \eta\mathcal{F}''$ is less than $-1$, so the error flips sign and grows: the
estimate jumps over the peak, further each time, until the nonlinearity catches
it. Exactly at $\eta = 2/|\mathcal{F}''|$ the factor is $-1$ and the error neither
grows nor shrinks.

For the run above, $\mathcal{F}''(\hat d) = {{curv_at_mode:.3f}}$, so any rate below
${{eta_max:.3f}}$ converges and ${{ascent_rate}}$ is comfortably inside it.

Now sharpen the ear. Take $\Sigma_u = {{var_obs_sharp}}$, ten times more precise.
The peak moves to ${{mode_sharp:.4f}}$, nearer what the data alone would say, and
the curvature there stiffens to ${{curv_sharp:.2f}}$. The bound collapses to
$\eta < {{eta_max_sharp:.4f}}$, and the rate that worked before is now above it.
Set the widget's sensory variance to ${{var_obs_sharp}}$ and leave the rate at
${{ascent_rate}}$: the estimate stops converging and cycles between
${{osc_lo:.4f}}$ and ${{osc_hi:.4f}}$, straddling the peak at ${{mode_sharp:.4f}}$
and never landing on it.

::: keyidea
A more precise sensor makes the log joint more sharply peaked, and a more sharply
peaked function demands smaller steps. Trusting your senses more is not free:
it makes the inference stiffer.

This is not a numerical footnote. Weeks&nbsp;5 to 8 build a physical system whose
relaxation rate is a property of the tissue, not a parameter to be retuned per
observation, and Week&nbsp;8 has to say what such a system does when precision
changes underneath it.
:::

::: exercise Reading the balance condition
[eq:balance] says the estimate settles where $\varepsilon_p = -g'(d)\varepsilon_u$.
Here $g' < 0$ everywhere, since intensity falls with distance. Use this to
determine, without computing anything, whether the MAP estimate lies above or
below the prior mean, given that the animal heard a louder sound than the prior
predicted.
---solution---
Louder than predicted means $u > g(d_p)$, so at $d = d_p$ we have
$\varepsilon_u > 0$. Since $g' < 0$, the sensory term $g'(d)\varepsilon_u$ is
negative, and the gradient at the starting point is negative, so $d$ decreases.

It keeps decreasing while the gradient stays negative. At the resting point
[eq:balance] gives $\varepsilon_p = -g'\varepsilon_u$, and with $g' < 0$ and
$\varepsilon_u$ still positive the right-hand side is positive, so
$\varepsilon_p > 0$, which means $d_p - \hat d > 0$, so $\hat d < d_p$.

The estimate ends below the prior mean, as it must: a loud sound is evidence of a
near source. Confirming with the figures, $\hat d = {{post_mode:.4f}}$ against a
prior mean of ${{d_prior}}$.
:::

::: checkpoint
- Why does $P(u)$ drop out of the MAP problem, and why is that not an
  approximation?
- Differentiating gave two terms. What does each one measure, and what does
  dividing by a variance do?
- The resting point has $\varepsilon_u \neq 0$. Say in one sentence why the animal
  does not keep moving to drive it to zero.
- What would [eq:gradient-eps] look like if $g$ were the identity, $g(d) = d$?
:::
