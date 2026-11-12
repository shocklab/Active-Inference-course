---
title: Learning the prior and the noise
deck: The derivatives of the log joint with respect to the prior location and the two variances, what each update does, and what happens to a variance near zero.
week: 3
time: 50 min
---

[lesson:3.1] reduced learning to one quantity per parameter: the partial derivative
of the log joint, evaluated at the settled state. This lesson computes it for the
prior location $d_p$, the prior variance $\Sigma_p$ and the sensory variance
$\Sigma_u$, starting from the full log joint [lesson:3.1] restored,

$$
\ell \;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \tfrac12\ln(2\pi\Sigma_p) \;-\; \ln Z_p
\;-\; \frac{\big(u - g(d)\big)^2}{2\Sigma_u} \;-\; \tfrac12\ln(2\pi\Sigma_u),
\qquad
Z_p = \int_0^\infty \mathcal N(x; d_p, \Sigma_p)\,\mathrm dx .
$$ {#ell}

The source power $\theta$ sits inside $g$ and is the subject of [lesson:3.3]. The
prediction errors are Week&nbsp;2's,

$$
\varepsilon_p \;=\; \frac{d_p - d}{\Sigma_p},
\qquad
\varepsilon_u \;=\; \frac{u - g(d)}{\Sigma_u},
$$

and the precisions are $\Pi_p = 1/\Sigma_p$ and $\Pi_u = 1/\Sigma_u$.

## The normaliser of the truncated prior

$Z_p$ is the one term Week&nbsp;2 could ignore entirely, because it contains
neither $d$ nor $u$. It does contain both prior parameters, so its derivatives are
needed first.

::: derivation The derivative of the normaliser in the location
Write $N(x) = \mathcal N(x; d_p, \Sigma_p)$ for short. Differentiating under the
integral sign, which is allowed because the integrand and its derivative in $d_p$
are continuous and bounded by an integrable Gaussian on the half-line,

$$
\frac{\partial Z_p}{\partial d_p} \;=\; \int_0^\infty \frac{\partial N(x)}{\partial d_p}\,\mathrm dx .
$$

The Gaussian depends on $x$ and $d_p$ only through $x - d_p$, so moving $d_p$ up is
the same as moving $x$ down:

$$
\frac{\partial N}{\partial d_p} \;=\; \frac{x - d_p}{\Sigma_p}\,N(x) \;=\; -\frac{\partial N}{\partial x} .
$$

The integral of a derivative is a difference of end values, and $N(x) \to 0$ as
$x \to \infty$:

$$
\frac{\partial Z_p}{\partial d_p} \;=\; -\int_0^\infty \frac{\partial N}{\partial x}\,\mathrm dx
\;=\; -\big[N(x)\big]_0^\infty \;=\; N(0) \;=\; \mathcal N(0; d_p, \Sigma_p) .
$$ {#dz-dp}
:::

The result says that moving the location up by a small amount moves a strip of
mass of height $N(0)$ across the boundary at zero and into the half-line.

::: derivation The derivative of the normaliser in the variance
Differentiate the Gaussian density with respect to its variance. Its logarithm is
$-(x - d_p)^2/(2\Sigma_p) - \tfrac12\ln(2\pi\Sigma_p)$, so

$$
\frac{\partial N}{\partial \Sigma_p} \;=\; N(x)\left[\frac{(x - d_p)^2}{2\Sigma_p^2} - \frac{1}{2\Sigma_p}\right] .
$$

Now differentiate it twice in $x$. Once gives $\partial N/\partial x = -\big[(x - d_p)/\Sigma_p\big]N$.
Again, by the product rule,

$$
\frac{\partial^2 N}{\partial x^2} \;=\; -\frac{1}{\Sigma_p}N \;+\; \frac{(x - d_p)^2}{\Sigma_p^2}N
\;=\; 2\,\frac{\partial N}{\partial \Sigma_p} .
$$

So $\partial N/\partial\Sigma_p = \tfrac12\,\partial^2 N/\partial x^2$, the heat
equation with the variance in the role of time. Differentiating $Z_p$ under the
integral sign is allowed here as it was for the location: for $\Sigma_p$ in a closed
interval $[a, b]$ with $a > 0$, the first display bounds $|\partial N/\partial\Sigma_p|$
by $\big[(x - d_p)^2/(2a^2) + 1/(2a)\big](2\pi a)^{-1/2}e^{-(x - d_p)^2/(2b)}$, which is
integrable on the half-line. Integrate over the half-line, and use
$\partial N/\partial x \to 0$ as $x \to \infty$:

$$
\frac{\partial Z_p}{\partial \Sigma_p}
\;=\; \frac12\int_0^\infty \frac{\partial^2 N}{\partial x^2}\,\mathrm dx
\;=\; \frac12\Big[\frac{\partial N}{\partial x}\Big]_0^\infty
\;=\; -\frac12\,\frac{\partial N}{\partial x}\bigg|_{x = 0}
\;=\; -\frac{d_p}{2\Sigma_p}\,\mathcal N(0; d_p, \Sigma_p) .
$$ {#dz-dvp}
:::

Widening the prior spreads mass in both directions, and the half that crosses zero
is lost, so $Z_p$ falls when $d_p > 0$. At Week&nbsp;2's values
$\mathcal N(0; d_p, \Sigma_p) = {{n0_w2:.6f}}$, and [eq:dz-dvp] gives
${{dz_dvp:.6f}}$; a finite-difference derivative of $Z_p$ agrees to eight decimal
places, and the check is an assertion in this week's `values.py`. Both derivatives
enter through $\ln Z_p$, divided by $Z_p$, so it is convenient to name the ratio

$$
\zeta_p \;=\; \frac{\mathcal N(0; d_p, \Sigma_p)}{Z_p},
$$ {#lambda}

which is ${{trunc_corr:.6f}}$ at Week&nbsp;2's values. It is the height of the prior
density at the boundary, as a fraction of the mass that lies inside it.

## The three derivatives

::: derivation The prior location
Only the first and third terms of [eq:ell] contain $d_p$. The first gives
$(d - d_p)/\Sigma_p$ by the chain rule, exactly as the corresponding derivative in
$d$ did in [lesson:2.3] but with the opposite sign, because $d_p$ enters the square
with a minus sign. The third gives $-(1/Z_p)\,\partial Z_p/\partial d_p = -\zeta_p$
by [eq:dz-dp]. So

$$
\frac{\partial \ell}{\partial d_p} \;=\; \frac{d - d_p}{\Sigma_p} - \zeta_p
\;=\; -\varepsilon_p - \zeta_p .
$$ {#grad-dp}
:::

::: derivation The prior variance
The first, second and third terms contain $\Sigma_p$. Differentiating
$-(d - d_p)^2/(2\Sigma_p)$ gives $+(d - d_p)^2/(2\Sigma_p^2)$; differentiating
$-\tfrac12\ln(2\pi\Sigma_p)$ gives $-1/(2\Sigma_p)$; and $-\ln Z_p$ gives
$-(1/Z_p)\,\partial Z_p/\partial\Sigma_p = +d_p\zeta_p/(2\Sigma_p)$ by [eq:dz-dvp].
Since $(d - d_p)^2/\Sigma_p^2 = \varepsilon_p^2$ and $1/\Sigma_p = \Pi_p$,

$$
\frac{\partial \ell}{\partial \Sigma_p} \;=\; \tfrac12\big(\varepsilon_p^2 - \Pi_p\big) \;+\; \frac{d_p\,\zeta_p}{2\Sigma_p} .
$$ {#grad-vp}
:::

::: derivation The sensory variance
Only the last two terms contain $\Sigma_u$, and they have the same form as the
prior's first two without a normaliser to correct:

$$
\frac{\partial \ell}{\partial \Sigma_u}
\;=\; \frac{\big(u - g(d)\big)^2}{2\Sigma_u^2} - \frac{1}{2\Sigma_u}
\;=\; \tfrac12\big(\varepsilon_u^2 - \Pi_u\big) .
$$ {#grad-vu}
:::

At Week&nbsp;2's observation and its mode, $\varepsilon_p = {{eps_p_w2:.4f}}$ and
$\varepsilon_u = {{eps_u_w2:.4f}}$, so the three derivatives are
${{grad_dp_w2:.4f}}$, ${{grad_vp_w2:.4f}}$ and ${{grad_vu_w2:.4f}}$. Each agrees with
a finite-difference derivative of [eq:ell] at the same point.

## What each update does

### The prior location

The prior location moves opposite to the prior error, which means towards the
inferred state: if the state was inferred below the prior location, $\varepsilon_p > 0$
and $d_p$ falls. The term $-\zeta_p$ adds a steady push downwards that does not
depend on the trial. The drift of [lesson:3.1], the update averaged over the trials
the process produces, vanishes where

$$
\mathbb E_{P^*}\big[\hat d\big] \;=\; d_p + \Sigma_p\zeta_p .
$$ {#stat-dp}

The right-hand side is the mean of the truncated prior.

::: derivation The mean of the truncated prior
With $N(x)$ as before, the mean is $\int_0^\infty x\,N(x)\,\mathrm dx/Z_p$. Write
$x = d_p + (x - d_p)$ and use $(x - d_p)N = -\Sigma_p\,\partial N/\partial x$ from the
derivation of [eq:dz-dp]:

$$
\int_0^\infty x\,N\,\mathrm dx
\;=\; d_p Z_p \;-\; \Sigma_p\int_0^\infty \frac{\partial N}{\partial x}\,\mathrm dx
\;=\; d_p Z_p + \Sigma_p N(0) .
$$

Dividing by $Z_p$ gives $d_p + \Sigma_p\zeta_p$, which is
${{trunc_mean:.6f}}$ at Week&nbsp;2's values; the same number by quadrature is
asserted in `values.py`.
:::

So the stationary condition reads: the average inferred distance equals the mean of
the prior the animal holds. "The mean chases the inferred state" is the informal
version, and the $\zeta_p$ term is what makes it the prior's actual mean rather
than its location parameter.

### The prior variance

The prior variance grows when [eq:grad-vp] is positive. Multiplying it by
$2\Sigma_p^2$, that is when

$$
(d - d_p)^2 \;>\; \Sigma_p\big(1 - d_p\zeta_p\big):
$$

the squared distance between the inferred state and the prior location exceeds the
spread the truncated prior actually has about $d_p$, which is
${{trunc_m2:.4f}}$ at Week&nbsp;2's values rather than $\Sigma_p = {{var_prior}}$. It
shrinks when the squared distance is smaller. The drift vanishes where

$$
\mathbb E_{P^*}\big[(\hat d - d_p)^2\big] \;=\; \Sigma_p\big(1 - d_p\zeta_p\big),
$$

and an exercise shows that the right-hand side is the truncated prior's own second
moment about $d_p$, ${{trunc_m2:.6f}}$ at Week&nbsp;2's values.

### The sensory variance

The sensory variance grows when $\varepsilon_u^2 > \Pi_u$, that is when the
squared residual $(u - g(d))^2$ exceeds $\Sigma_u$, and its drift vanishes where

$$
\mathbb E_{P^*}\big[(u - g(\hat d))^2\big] \;=\; \Sigma_u .
$$ {#stat-vu}

At Week&nbsp;2's observation the squared residual at the mode is
${{sq_err_u_w2:.6f}}$, far below $\Sigma_u = {{var_obs}}$, so this single trial
pushes $\Sigma_u$ down hard. Whether trials like it are typical is a question about
the process, and [lesson:3.4] answers it.

::: keyidea
Each stationary condition matches a moment of the model to the same moment of the
inferred states: their mean to the prior's mean, their spread about $d_p$ to the
prior's spread, and the spread of the residuals they leave to the sensory variance.
The rules are only as good as the inferred states they average.
:::

Those inferred states are modes, one point from each posterior. The averages that
would make the conditions hold at the world's values are averages of whole
posteriors: averaged over the process, the posterior means of $d$ reproduce the
prior's mean exactly, and the modes in general do not. Squared deviations of modes
also leave out each posterior's own spread. That omission alone would make them too
small, but a mode is not its posterior's mean either, and on some trials the mode lies
further from $d_p$ than the mean does, so the net effect has to be computed over the
process. [lesson:3.4] derives both statements and computes what they cost.

## Locality

[lesson:2.1] required a connection to change using only quantities available at its
endpoints. In the circuit of [lesson:2.4] the prior error unit compares the estimate
with a node labelled expectation, which holds $d_p$.
[Bogacz (2017)](https://doi.org/10.1016/j.jmp.2015.11.003), in the section on learning model parameters, stores the
same number differently: as the strength of a connection into the prior error unit
from a unit whose activity is fixed at one. Bogacz defines the prior error as the
estimate minus the prior location, the opposite sign to this course's
$\varepsilon_p = (d_p - d)/\Sigma_p$, so his rule changes the connection by plus the
product of the two activities. In this course's signs, [eq:grad-dp] has two parts.
The first, $-\varepsilon_p$, is minus the product of the activity at the receiving
end, $\varepsilon_p$, and the activity at the sending end, which is one: the Hebbian
form of [lesson:2.1] with a constant presynaptic input, its sign set by how the error
is coded. The second, $-\zeta_p$, is not a product of activities, and the next
paragraph but one deals with it. So the full rule is a Hebbian term plus a correction
that depends on the parameters.

The variances are the gains of the error units, which [lesson:2.4] placed at the
units. [eq:grad-vu] uses the error unit's activity squared and the gain's own value.
Both are available where the gain is stored.

The term $\zeta_p$ is a function of the two prior parameters alone. Computing it at
the connection that stores $d_p$ needs $\Sigma_p$ there too, and in the circuit of
[lesson:2.4] $\Sigma_p$ is the gain of the error unit, stored somewhere else. So the
rule is local only under an extra assumption: that the connection has its own copy of
the prior variance, or that the two are stored together. With that assumption it meets
the definition of local plasticity without being Hebbian. Bogacz's prior is not
truncated and has no such term.

::: mn Several states at once
With several hidden states and observations the variances become covariance
matrices, and the update for one entry involves the inverse of the whole matrix,
which depends on every other entry. Bogacz notes that this breaks local plasticity
and, in section 5, repairs it with an extra inhibitory unit per error unit.
[week:5] takes up that construction.
:::

## Variance, precision and log-variance

A parameter can be learned in any coordinate, and the choice changes the update.

::: derivation Three coordinates for one parameter
Write $\delta_u = u - g(d)$ for the raw residual. [eq:grad-vu] is
$\partial\ell/\partial\Sigma_u = \tfrac12(\delta_u^2/\Sigma_u^2 - 1/\Sigma_u)$.

In the log-variance $\ln\Sigma_u$, the chain rule with
$\mathrm d\Sigma_u/\mathrm d\ln\Sigma_u = \Sigma_u$ gives

$$
\frac{\partial \ell}{\partial \ln\Sigma_u} \;=\; \Sigma_u\,\frac{\partial \ell}{\partial \Sigma_u}
\;=\; \tfrac12\left(\frac{\delta_u^2}{\Sigma_u} - 1\right) .
$$ {#grad-logvar}

In the precision $\Pi_u = 1/\Sigma_u$, with $\mathrm d\Sigma_u/\mathrm d\Pi_u = -1/\Pi_u^2 = -\Sigma_u^2$,

$$
\frac{\partial \ell}{\partial \Pi_u} \;=\; -\Sigma_u^2\,\frac{\partial \ell}{\partial \Sigma_u}
\;=\; \tfrac12\big(\Sigma_u - \delta_u^2\big) .
$$

All three vanish at the same point, $\delta_u^2 = \Sigma_u$, because each is the first
multiplied by a quantity that is never zero. That factor depends on the parameter but
not on the trial, so the three drifts, averaged over the process at a fixed parameter
value, also vanish at the same value.
:::

What the three coordinates share is the zero of the drift. They do not share a
trajectory, or the point a run with a finite rate rests at. A step of size $\beta$ in
$\ln\Sigma_u$ multiplies $\Sigma_u$ by $e^{\beta S}$, where $S$ is
[eq:grad-logvar]; at the zero of the drift $S$ averages to zero, but $e^{\beta S}$
averages to more than one whenever $S$ varies, because the exponential is convex. The
two runs' stationary means can differ at order $\beta$, the size of the mean shift
[lesson:3.1] derives, while each run fluctuates about its own mean by order
$\sqrt\beta$.

The update Bogacz proposes for learning the variance of a single error unit is a fourth variant. An inhibitory unit
holds the raw residual $\delta_u$, the error unit holds $\varepsilon_u = \delta_u/\Sigma_u$, and the
connection changes by $\beta(\varepsilon_u \delta_u - 1)$. Since
$\varepsilon_u \delta_u - 1 = \delta_u^2/\Sigma_u - 1$, this is twice [eq:grad-logvar], applied to
$\Sigma_u$ rather than to its logarithm: the plain gradient multiplied by
$2\Sigma_u$. It has the same stationary point, and every quantity in it is the
activity of one of the two units the connection joins.

::: warning A variance near zero
The plain update $\Sigma_u \leftarrow \Sigma_u + \beta\,\partial\ell/\partial\Sigma_u$
behaves badly when $\Sigma_u$ is small, because the gradient is not bounded there.

On a trial with zero residual the step is $-\beta/(2\Sigma_u)$, and it carries the
variance below zero whenever $\beta/(2\Sigma_u) > \Sigma_u$, that is whenever
$\Sigma_u < \sqrt{\beta/2}$. With $\beta = {{beta_var}}$ that is any variance below
${{var_cross:.4f}}$, and a negative variance has no meaning. On a trial with residual
$\delta_u = {{jump_r}}$ and $\Sigma_u = {{jump_var}}$, the step is
$\beta\cdot\tfrac12({{jump_r}}^2/{{jump_var}}^2 - 1/{{jump_var}}) = {{jump_step:.4f}}$,
many times the variance being updated. The same discrete step that is harmless at
$\Sigma_u = {{var_obs}}$ is unstable near zero, which is the parameter version of the
step-size bound of [lesson:2.3].

Two repairs are standard. One is to take the step in the log-variance,
$\Sigma_u \leftarrow \Sigma_u\exp\!\big(\beta\,\tfrac12(\delta_u^2/\Sigma_u - 1)\big)$, whose drift
vanishes at the same value by [eq:grad-logvar] and which can never cross zero. It
keeps the variance positive; it does not keep the steps small, since
[eq:grad-logvar] still grows without bound as $\Sigma_u \to 0$ at a fixed nonzero
residual. The simulations of this week use it for both variances. The other is to forbid small
variances outright: Bogacz notes, in his section on learning model parameters, that
[Friston (2005)](https://doi.org/10.1098/rstb.2005.1622) imposes a minimum value of one
on the estimated variance, and in a footnote calls the constraint a hyperprior. A fixed
floor depends on the units of the observation: measuring $u$ in units $a$ times smaller
gives $u' = a u$ and $\Sigma_u' = a^2\Sigma_u$, so the same floor of one excludes a
different set of variances in each unit. [lesson:3.4] derives a smooth version.

Neither repair addresses why the variance is heading towards zero, which is
sometimes the rule's fault rather than the step's. [lesson:3.4] shows when.
:::

::: exercise The second moment of the truncated prior
Show that $\int_0^\infty (x - d_p)^2 N(x)\,\mathrm dx / Z_p = \Sigma_p(1 - d_p\zeta_p)$,
with $N$ and $\zeta_p$ as in this lesson, and conclude that the drift for
$\Sigma_p$ vanishes where the average squared deviation of the inferred states from
$d_p$ equals the prior's own.
---solution---
Write $(x - d_p)^2 N = (x - d_p)\cdot(x - d_p)N = -\Sigma_p (x - d_p)\,\partial N/\partial x$
and integrate by parts:

$$
-\Sigma_p\int_0^\infty (x - d_p)\frac{\partial N}{\partial x}\,\mathrm dx
\;=\; -\Sigma_p\Big[(x - d_p)N\Big]_0^\infty \;+\; \Sigma_p\int_0^\infty N\,\mathrm dx .
$$

At infinity the bracket vanishes; at zero it is $(0 - d_p)N(0) = -d_pN(0)$, so the
bracket contributes $-\Sigma_p\big(0 - (-d_pN(0))\big) = -\Sigma_p d_p N(0)$. The last
integral is $\Sigma_p Z_p$. Dividing by $Z_p$,

$$
\frac{1}{Z_p}\int_0^\infty (x - d_p)^2 N\,\mathrm dx \;=\; \Sigma_p - \Sigma_p d_p\zeta_p
\;=\; \Sigma_p(1 - d_p\zeta_p),
$$

which is ${{trunc_m2:.6f}}$ at Week&nbsp;2's values.

For the second part, average [eq:grad-vp] over trials and set it to zero:
$\tfrac12\big(\mathbb E_{P^*}[\varepsilon_p^2] - \Pi_p\big) + d_p\zeta_p/(2\Sigma_p) = 0$.
Multiply by $2\Sigma_p^2$ to get
$\mathbb E_{P^*}[(\hat d - d_p)^2] - \Sigma_p + \Sigma_p d_p\zeta_p = 0$, which is the
condition in the lesson, and its right-hand side is the moment just computed.
:::

::: exercise Precision as the learned quantity
A colleague learns $\Pi_u$ directly, with $\Pi_u \leftarrow \Pi_u + \beta\,\partial\ell/\partial\Pi_u$.
Using the precision form derived above, say what happens on a trial with zero
residual when $\Pi_u$ is very large, and on a trial with a large residual when
$\Pi_u$ is very small. Compare with the variance update.
---solution---
The step is $\beta\cdot\tfrac12(\Sigma_u - \delta_u^2) = \tfrac{\beta}{2}(1/\Pi_u - \delta_u^2)$.

With zero residual it is $\beta/(2\Pi_u)$: positive and small when $\Pi_u$ is large.
Precision grows without bound but slowly.

With a residual $\delta_u$ held fixed, however large, and $\Pi_u \to 0$, the term
$1/\Pi_u$ dominates and the step tends to $+\infty$: a small precision is thrown
upwards, as a small variance was in the warning above. The approximation
$-\beta\delta_u^2/2$ holds only when $\delta_u^2 \gg 1/\Pi_u$, which is the opposite
regime. The next precision is $\Pi_u + \tfrac{\beta}{2}(1/\Pi_u - \delta_u^2)$.
Multiplying it by the positive number $2/(\beta\Pi_u)$ shows that it is negative
exactly when

$$
\delta_u^2 \;>\; \frac{1}{\Pi_u} + \frac{2\Pi_u}{\beta} .
$$

The right-hand side is large at both ends, so a negative precision needs a residual
large compared with both the variance and $2\Pi_u/\beta$; a single outlier can supply
one.

The two coordinates fail in mirror-image ways. In each, the gradient contains the
reciprocal of the learned quantity and so is unbounded as that quantity goes to zero,
and in each a large enough step crosses zero. The log-variance update is the same in
either coordinate, since $\ln\Pi_u = -\ln\Sigma_u$. It cannot cross zero, because it
multiplies by a positive factor, but its steps are still unbounded near zero at a
fixed nonzero residual, so positivity is all it guarantees.
:::

::: checkpoint
- Why does the truncation add a term to the prior updates and not to the sensory one?
- In words, when does each variance grow?
- What do the three stationary conditions match, and against what?
- Why do the variance, precision and log-variance updates share their stationary
  points but not their behaviour near zero?
:::
