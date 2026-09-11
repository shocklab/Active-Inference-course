---
title: MAP estimation by gradient ascent
deck: Deriving an update for a posterior mode, and checking its local stability.
week: 2
time: 45 min
scripts: [w02.js]
---

The preceding lesson computed a reference posterior numerically. We now derive an update for its mode, then compare what this point estimate retains and what it omits.

## The posterior mode

The animal does not obviously need the whole posterior. If it must act, one
number might do: the single distance the evidence most supports.

::: definition Maximum a posteriori estimate
The **MAP estimate** is the value of the hidden variable at which the posterior
is largest,
$$
\hat{d} \;=\; \operatorname*{arg\,max}_{d} \; P(d \mid u).
$$
It is the mode of the posterior, not its mean, and [lesson:2.2] showed those can
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

The same cancellation is useful in other algorithms, including MCMC acceptance ratios. Whether finding a mode is sufficient depends on the task.

:::

The product $P(u \mid d)P(d)$ is the joint density $P(u, d)$, by the product
rule. So the agent maximises the joint. Both factors are things it has: a prior
it holds and a likelihood it can evaluate by predicting an intensity and
comparing.

## Take the logarithm

Taking logarithms turns a product into a sum and exposes the quadratic
penalties in the Gaussian factors. It also avoids multiplying very small
densities. The log joint still depends on the density's coordinate convention,
as the change-of-variables exercise will show.

The logarithm is increasing, so it does not move the maximum either. Define

::: definition The log joint
$$
\ell(d, u) \;=\; \ln P(d) \;+\; \ln P(u \mid d).
$$ {#logjoint}
Maximising $\ell$ over $d$ gives the MAP estimate.
:::

::: warning The log joint and variational free energy
The log joint $\ell(d,u)$ is a function of a candidate state and a fixed
observation. Variational free energy $\mathcal F[Q,u]$ is a functional of a
probability distribution. We use different symbols because their arguments and
optimisation directions differ.

For a continuous posterior, substituting a point mass for $Q$ does not give a
finite free energy equal to $-\ell$: the point mass is singular relative to the
posterior and its KL divergence is infinite. For suitable distributions with a
small fixed width, $\mathbb E_Q[\ell]$ can approximate the log joint at their
centre. Comparing centres then approximates MAP when the entropy term is fixed.
The width cannot be set to zero as an ordinary substitution. [week:4] develops
the distributional approximation.
:::



## Expanding the log joint

Substitute the Gaussian likelihood and the truncated Gaussian prior. On $d>0$, taking the prior first:

$$
\ln P(d) \;=\; \ln\!\left[\frac{1}{Z_p\sqrt{2\pi\Sigma_p}}\exp\!\left(-\frac{(d - d_p)^2}{2\Sigma_p}\right)\right]
\;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \tfrac{1}{2}\ln(2\pi\Sigma_p) - \ln Z_p,
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
\ell \;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \frac{(u - g(d))^2}{2\Sigma_u} \;+\; \text{const}.
$$ {#F-explicit}

Read what this says before differentiating it. $\ell$ is large when both
squared terms are small: when $d$ is near what the prior expected, *and* when the
intensity that $d$ predicts is near the intensity actually heard. It is a
scoring rule with two ways to lose marks, and the variances set the exchange
rate between them.

The two residuals have different units: distance and intensity. Dividing each
square by its own variance makes each penalty dimensionless. Their numerical
coefficients alone do not say that one source is a fixed multiple "more reliable"
than the other; the observation function also determines sensitivity to distance.

Multiplying both variance parameters by the same positive constant leaves the
MAP location unchanged, because it rescales the distance-dependent terms together.
It does change the posterior's spread. The prior normalisation remains constant
with respect to $d$ during this state-inference calculation.

## Differentiating the log joint

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

The sign makes the term positive
when $d$ is below the prior location, so the gradient pushes $d$ up towards $d_p$,
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
\frac{\partial \ell}{\partial d}
\;=\; \underbrace{\frac{d_p - d}{\Sigma_p}}_{\text{prior term}}
\;+\; \underbrace{\frac{u - g(d)}{\Sigma_u}\,g'(d)}_{\text{sensory term}} .
$$ {#gradient}
:::

## Two prediction errors

Both terms contain a residual divided by a variance. One compares the estimate with the prior location; the other compares the predicted and observed intensity.

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
doing, and [lesson:2.4] is largely about it.
:::

::: definition Precision
The **precision** of a Gaussian channel is the reciprocal of its variance. A
precise channel is one whose predictions are tight, so dividing an error by a
variance is the same as multiplying it by a precision: a gain saying how
seriously to take that error. The symbol $\Pi$ is reserved for it from
[lesson:2.4] onwards.
:::

::: mn Where the subscripts come from
$\varepsilon_p$ carries the subscript of the prior, $\varepsilon_u$ that of the
observation $u$. Weeks&nbsp;5 onwards will have a chain of these, one per level,
and the subscripts become level indices.
:::

In this notation the gradient is short:

$$
\frac{\partial \ell}{\partial d} \;=\; \varepsilon_p \;+\; g'(d)\,\varepsilon_u .
$$ {#gradient-eps}

Now check [eq:gradient-eps] against the constraints from [lesson:2.1]. To
compute $\varepsilon_u$ a unit needs $u$, which arrives at it, and $g(d)$, which
is a prediction sent to it. To compute $\varepsilon_p$ it needs $d_p$, held
locally, and $d$, the current estimate. Nothing has to be fetched from across the
network and nothing is summed over the state space. The rule is local.

For this scalar model, the gradient can therefore be evaluated from these signals. The result specifies the information an implementation must provide; it does not establish that a biological circuit provides it.

## Updating the estimate

Having the gradient, move in the direction it points:

$$
\dot{d} \;=\; \nu \, \frac{\partial \ell}{\partial d}
\;=\; \nu\big(\varepsilon_p + g'(d)\,\varepsilon_u\big),
$$ {#ascent}

Here $\nu>0$ is a mobility, with units of squared distance per unit time.
The gradient has units of inverse distance. A forward-Euler step of duration
$\Delta t$ is

$$
d_{k+1}=d_k+\eta\ell'(d_k),\qquad \eta=\nu\Delta t.
$$

The discrete step parameter $\eta$ has units of squared distance. The widget
varies this parameter. Euler integration approximates the flow; it is not exact
sampling of the continuous trajectory. Its stability condition therefore belongs
to the numerical iteration.

::: mn Other optimisation methods
Newton's method uses the gradient and curvature. Near a nondegenerate smooth
maximum it can converge faster than fixed-step gradient ascent, but its direction
and step may need safeguards away from the maximum. In many dimensions it requires
solving a linear system involving the Hessian. This creates communication and
computational requirements; locality alone does not prove it impossible.
We use gradient ascent here to examine a simple update, not because every
alternative has been excluded.
:::



Where does it stop? At any point where the gradient vanishes, which by
[eq:gradient-eps] means

$$
\varepsilon_p \;=\; -\,g'(d)\,\varepsilon_u .
$$ {#balance}

The estimate settles where the two errors balance. Not where either is zero:
where the pull towards the prior exactly offsets the pull towards the data.

## A worked trajectory

Run it. Start the animal at its prior location, $d = {{d_prior}}$, and take
$\eta = {{ascent_rate}}$.

At the start the prior error is exactly zero, because the estimate *is* the prior
mean. The sensory error is not: the estimate predicts an intensity of
${{g_at_prior:.3f}}$ and the animal heard ${{u_obs}}$, so
$\varepsilon_u = ({{u_obs}} - {{g_at_prior:.3f}})/{{var_obs}} = {{err_obs_at_start:.3f}}$.
Weighted by $g'({{d_prior}}) = {{gprime_at_prior:.3f}}$, the sensory term
contributes ${{err_obs_weighted_at_start:.3f}}$. The gradient is negative, so the
estimate moves down: the sound was louder than expected, therefore the source is
closer than expected.

::: fig The estimate climbing the log joint from the prior location. Left, the trajectory against the exact posterior mode. Right, the two terms of [eq:gradient-eps] over the same steps, meeting where their sum is zero.
<div class="widget" data-widget="ascent-errors"></div>
:::

It settles at $d = {{ascent_final:.4f}}$, reaching within one per cent of the
exact mode after {{ascent_steps_to_1pct}} steps. The mode computed by quadrature
in [lesson:2.2] was ${{post_mode:.4f}}$. They agree for this starting point and step size. Gradient ascent can also converge to another local maximum or fail to converge, as the exercises demonstrate.

At that point the two terms are

$$
\varepsilon_p = {{err_prior:+.6f}},
\qquad
g'(\hat d)\,\varepsilon_u = {{err_obs_weighted:+.6f}},
$$

summing to ${{err_sum:sci2}}$, when evaluated at the grid-based mode; the remaining mismatch reflects that numerical location. Note what is *not* zero: the raw sensory error is
$\varepsilon_u = {{err_obs:.4f}}$, and the estimate still predicts an intensity
of ${{g_at_mode:.4f}}$ against the ${{u_obs}}$ heard. The animal has not moved far
enough to account for the whole of what it heard. It has stopped where accounting
for more would cost more in prior implausibility than it gains in fit.

::: warning A phrase to keep clear of
It is tempting to say the animal has not "explained away" the observation.
Resist it. **Explaining away** is Week&nbsp;1's technical term for something
quite different: two independent causes of a shared effect becoming
anti-correlated once that effect is observed, so that evidence for one is
evidence against the other. It needs at least two causes and there is only one
here. What is happening in this paragraph is a residual left by a compromise
between two terms, which is not the same phenomenon and should not borrow its
name.
:::

## Choosing the step parameter

$\eta$ was set to ${{ascent_rate}}$ without comment. It cannot be set freely, and
the bound on it turns out to depend on the precisions, which is the first sign of
something the rest of the course keeps running into.

::: derivation Local stability of an Euler step
Write $d^{*}$ for the peak, the same point the earlier sections called $\hat d$,
and $e_k = d_k - d^{*}$ for the error at step $k$. Both symbols refer to the same local maximum. Near
the peak, expand the gradient to first order. Since $\ell'(d^{*}) = 0$ by
definition of the peak,

$$
\ell'(d_k) \;\approx\; \ell'(d^{*}) + \ell''(d^{*})\,e_k
\;=\; \ell''(d^{*})\,e_k .
$$

Substituting into the discrete update $d_{k+1} = d_k + \eta\,\ell'(d_k)$
and subtracting $d^{*}$ from both sides:

$$
e_{k+1} \;\approx\; e_k + \eta\,\ell''(d^{*})\,e_k \;=\; \big(1 + \eta\,\ell''(d^{*})\big)\,e_k .
$$

The error is multiplied by a fixed factor each step, so it shrinks precisely when
that factor is less than one in magnitude:

$$
\big|\,1 + \eta\,\ell''(d^{*})\,\big| \;<\; 1 .
$$

At a maximum $\ell'' < 0$, so writing $|\ell''|$ for its magnitude the
condition is $-1 < 1 - \eta|\ell''| < 1$, and the right inequality holds for
any $\eta > 0$. The left one gives

$$
\eta \;<\; \frac{2}{|\ell''(d^{*})|} .
$$ {#eta-bound}
:::

This is a local condition near a nondegenerate maximum. If
$0<\eta|\ell''|<1$, the linearised error decreases without changing sign.
If $1<\eta|\ell''|<2$, it alternates in sign while decreasing. At the boundary
$\eta|\ell''|=2$, the linearisation has multiplier $-1$, and higher-order terms
are needed to determine the nonlinear behaviour. Above it the fixed point is
locally unstable.

For the worked mode, the curvature is ${{curv_at_mode:.3f}}$ and the local upper
bound is ${{eta_max:.3f}}$. The chosen rate ${{ascent_rate}}$ satisfies it.
That does not guarantee convergence from every starting point or keep every step
inside $d>0$. Both need checking separately.

The numbers here come from differentiating [eq:gradient-eps] a second time, which
is done in full as the first exercise of [lesson:2.4], once the notation for the
precisions is in place. What matters now is not the formula but that
$\ell''$ depends on $\Sigma_u$, so the bound moves when the ear changes.

Now sharpen the ear. Take $\Sigma_u = {{var_obs_sharp}}$, ten times more precise.
The peak moves to ${{mode_sharp:.4f}}$, nearer what the data alone would say, and
the curvature there stiffens to ${{curv_sharp:.2f}}$. The bound collapses to
$\eta < {{eta_max_sharp:.4f}}$, and the rate that worked before is now above it.
Set the widget's sensory variance to ${{var_obs_sharp}}$ and leave the rate at
${{ascent_rate}}$: the estimate stops converging and cycles between
${{osc_lo:.4f}}$ and ${{osc_hi:.4f}}$, straddling the peak at ${{mode_sharp:.4f}}$
and never landing on it.

A sharper posterior can require smaller Euler steps. This is a numerical
stability issue. It is not, by itself, evidence that a continuous biological
inference process becomes unstable when sensory precision increases.

::: exercise Reading the balance condition
[eq:balance] says the estimate settles where $\varepsilon_p = -g'(d)\varepsilon_u$.
Here $g' < 0$ everywhere, since intensity falls with distance. Use this to
determine, without computing anything, whether the MAP estimate lies above or
below the prior location, given that the animal heard a louder sound than the prior
predicted.
---solution---
Louder than predicted means $u > g(d_p)$, so at $d = d_p$ we have
$\varepsilon_u > 0$. Since $g' < 0$, the sensory term $g'(d)\varepsilon_u$ is
negative, and the gradient at the starting point is negative, so $d$ decreases.

It keeps decreasing while the gradient stays negative. At the resting point
[eq:balance] gives $\varepsilon_p = -g'\varepsilon_u$, and with $g' < 0$ and
$\varepsilon_u$ still positive the right-hand side is positive, so
$\varepsilon_p > 0$, which means $d_p - \hat d > 0$, so $\hat d < d_p$.

The estimate ends below the prior location, as it must: a loud sound is evidence of a
near source. Confirming with the figures, $\hat d = {{post_mode:.4f}}$ against a
prior location of ${{d_prior}}$.
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
