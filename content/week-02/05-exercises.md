---
title: Problems and code
deck: Seven problems with full solutions, two of which show the method of this week failing, and the notebook that rebuilds every figure.
week: 2
time: 100 min
scripts: [w02.js]
number_sections: false
---

Work the problems before opening the solutions. The ones marked with a dagger
are used again later in the course.

Two of these, the last two, are attacks on the method rather than exercises in
it. They belong here rather than in Week&nbsp;12 because the objections are
already visible from what has been derived, and saving them up would mean
teaching the method for ten weeks before admitting what is wrong with it.

## The notebook

| Notebook | What it does | Run it |
|---|---|---|
| `week02_numpy.ipynb` | The exact posterior by quadrature, gradient ascent on the log joint, the two prediction errors, the step-size bound, and the linear case checked against its closed form | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-02/week02_numpy.ipynb) |
| `week02_jax.ipynb` | Three things that need many runs at once: the hand-derived gradient checked against automatic differentiation, ten thousand starting points mapping the basins of the bimodal case, and the whole precision plane swept at once | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-02/week02_jax.ipynb) |

The JAX notebook is not a translation of the NumPy one. It does what only
becomes practical when thousands of copies run together, and its basin map
answers a question the problems below can only pose one starting point at a
time.

There is no `pymdp` notebook this week, for the same reason as last week: the
library works with the discrete generative models of Week&nbsp;9, and there is
nothing here for it to do.

## Problems

::: exercise A different world &dagger;
Sound is also absorbed by the air, so over long distances intensity falls
exponentially rather than as a power. Take $g(d) = e^{-d/L}$ with $L$ a fixed
absorption length.

(a) Write down the update rule.

(b) Where in the range of $d$ is the sensory term most influential, and where
does it become negligible? Compare with the inverse-square case.

(c) Show that for this link the MAP estimate can be written in closed form when
the prior is flat, and say why that stops being true as soon as the prior is not.
---solution---
(a) $g'(d) = -\tfrac{1}{L}e^{-d/L}$, so substituting into the general rule,

$$
\dot d = \eta\Big(\Pi_p (d_p - d) - \frac{\Pi_u}{L}e^{-d/L}\big(u - e^{-d/L}\big)\Big).
$$

(b) The sensory gain is $|g'| = e^{-d/L}/L$, largest at $d = 0$ and decaying with
length scale $L$. So the ear is informative nearby and, beyond a few multiples of
$L$, says almost nothing: the predicted intensity has flattened out and moving
the estimate barely changes it.

The contrast with the inverse square is in the rate, not the direction. There
$|g'| = 2/d^3$ also decays, but as a power, so it retains some sensitivity at
every finite distance. Exponential absorption has a horizon; geometric spreading
does not.

(c) With a flat prior the first term is absent and the resting point needs
$e^{-d/L}(u - e^{-d/L}) = 0$. The exponential is never zero, so $e^{-d/L} = u$,
giving $\hat d = -L\ln u$ provided $0 < u < 1$. The estimate simply inverts the
link, which is what a flat prior means: believe the measurement.

With a Gaussian prior the condition becomes
$\Pi_p(d_p - d) = \tfrac{\Pi_u}{L}e^{-d/L}(u - e^{-d/L})$, which sets a linear
function of $d$ equal to a combination of $e^{-d/L}$ and $e^{-2d/L}$. That is a
transcendental equation, and it has no solution in elementary functions.
:::

::: exercise Implement it
Write the gradient ascent yourself, in about fifteen lines, and check it.

(a) Code $g$, $g'$ and the update, and run it from $d_0 = {{d_prior}}$ with
$\eta = {{ascent_rate}}$ for the values of this week. Confirm you reach
${{post_mode:.4f}}$.

(b) Now find the same number a completely different way: evaluate the
unnormalised posterior on a fine grid and take its largest entry. The two methods
share no code. Agreement is therefore evidence; if you had reused the same
function for both, it would not be.

(c) Add a line that records $\varepsilon_p$ and $g'\varepsilon_u$ at every step
and confirm their sum goes to zero while neither does.
---solution---
```python
import numpy as np

d_p, var_p = 2.0, 1.0        # prior
u,   var_u = 0.50, 0.10      # observation

g      = lambda d: 1.0 / d**2
g_prime = lambda d: -2.0 / d**3

d, eta, trace = d_p, 0.05, []
for _ in range(400):
    eps_p = (d_p - d) / var_p
    eps_u = (u - g(d)) / var_u
    trace.append((eps_p, g_prime(d) * eps_u))
    d += eta * (eps_p + g_prime(d) * eps_u)

grid = np.linspace(0.05, 8.0, 600_000)
logp = -(grid - d_p)**2 / (2*var_p) - (u - g(grid))**2 / (2*var_u)

print(f"ascent {d:.6f}   grid {grid[logp.argmax()]:.6f}")
a, b = trace[-1]
print(f"eps_p {a:+.6f}   g'*eps_u {b:+.6f}   sum {a+b:.2e}")
```

This prints an ascent value and a grid value agreeing to about four decimal
places, the residual being the grid spacing rather than any error in either
method. The final line shows ${{err_prior_ascent:+.6f}}$ and
${{err_obs_weighted_ascent:+.6f}}$, summing to ${{err_sum_ascent:.2e}}$.

That residual is smaller than the ${{err_sum:.2e}}$ quoted in Lesson&nbsp;3, and
the difference is worth understanding rather than ignoring. Lesson&nbsp;3
evaluates the two errors at the mode found by quadrature; this code evaluates
them where its own ascent stopped. The two locations agree to five decimal
places, but the quantity being reported is a cancellation between two numbers of
size ${{err_prior:.2f}}$, so a difference in the fifth decimal of each becomes the whole of
the answer. A cancellation magnifies whatever disagreement its inputs had, which
is a general fact about floating point and the reason the sum is a poor way to
measure how well either method has converged.

On (b): the point is not that grids are trustworthy. It is that two methods
sharing no code and resting on different ideas, one climbing a gradient and one
enumerating, have no common way to be wrong. If they agree, the ways each could
fail would both have to fail identically, which is unlikely enough to count as
evidence.
:::

::: exercise Repeated listening &dagger;
The animal listens $n$ times, getting independent readings $u_1, \dots, u_n$ from
the same source, each with precision $\Pi_u$.

(a) Show the update depends on the readings only through their mean $\bar u$.

(b) Show the effective sensory precision is $n\Pi_u$, and say what that does to
the estimate and to the largest stable step size as $n$ grows.
---solution---
(a) Independence given $d$ makes the log likelihood a sum, so the sensory part of
the gradient is

$$
g'(d)\sum_{i=1}^{n} \Pi_u\big(u_i - g(d)\big)
= g'(d)\,\Pi_u\Big(\sum_i u_i - n\,g(d)\Big)
= g'(d)\,n\Pi_u\big(\bar u - g(d)\big),
$$

pulling out $n$ and writing $\bar u = \tfrac{1}{n}\sum_i u_i$. Nothing else about
the individual readings survives.

::: definition Sufficient statistic
A function $T(u_1, \dots, u_n)$ of the data is a **sufficient statistic** for a
hidden state $d$ when the likelihood depends on the data only through it: that
is, $P(u_1,\dots,u_n \mid d)$ can be written as a function of $T$ and $d$ times a
factor containing no $d$. Two data sets with the same $T$ then give the same
posterior, whatever the prior, so everything the data have to say about $d$ is
already in $T$ and the rest may be discarded.

Here $T = \bar u$, together with the count $n$. An animal that has heard a
thousand sounds need not remember any of them; a running mean and a tally carry
the whole of what those sounds said about the distance.
:::

(b) The last expression is the one-observation rule with $\Pi_u$ replaced by
$n\Pi_u$ and $u$ by $\bar u$. So $n$ readings are one reading of $n$ times the
precision. As $n$ grows the sensory term dominates the prior term without limit
and the estimate approaches whatever satisfies $g(\hat d) = \bar u$: the prior
washes out, which is the usual asymptotic story and here it is visible directly
in the gradient.

For the step size, the curvature acquires a term $-n\Pi_u g'^2$, growing linearly
in $n$, so the bound $\eta < 2/|\mathcal{F}''|$ falls off like $1/n$. More data
makes the peak sharper and the ascent stiffer, which is the same trade the
precision slider showed in Lesson&nbsp;3.
:::

::: exercise Sequential listening
Instead of collecting $n$ readings and processing them together, the animal
updates after each one, using the posterior from reading $k$ as the prior for
reading $k+1$. Take the linear link $g(d) = d$, where everything stays Gaussian.

(a) Show that after two readings the estimate is the same as processing both at
once.

(b) Say what breaks this equivalence when $g$ is nonlinear.
---solution---
(a) With the linear link the posterior after one reading is Gaussian with
precision $\Pi_1 = \Pi_p + \Pi_u$ and mean
$m_1 = (\Pi_p d_p + \Pi_u u_1)/\Pi_1$, by the precision-weighted average of
Lesson&nbsp;4. Using that as the prior for $u_2$ gives precision
$\Pi_2 = \Pi_1 + \Pi_u = \Pi_p + 2\Pi_u$ and mean

$$
m_2 = \frac{\Pi_1 m_1 + \Pi_u u_2}{\Pi_2}
    = \frac{\Pi_p d_p + \Pi_u u_1 + \Pi_u u_2}{\Pi_p + 2\Pi_u},
$$

since $\Pi_1 m_1 = \Pi_p d_p + \Pi_u u_1$ by construction. That is exactly the
batch answer. Precisions add and precision-weighted means add, so the order and
grouping of the readings do not matter.

(b) Nothing in the *probability* breaks: the true posterior is always
order-independent, because $P(d \mid u_1, u_2)$ does not know what order the
readings arrived in. What breaks is the *representation*. With $g$ nonlinear the
posterior after one reading is not Gaussian, so summarising it by a mean and a
precision throws information away, and feeding that summary forward as the next
prior propagates the loss. Process both at once and the approximation is made
once; process them in sequence and it is made twice.

This is the difference between exact recursive filtering and the extended Kalman
filter, and Week&nbsp;7 meets it again in continuous time.
:::

::: exercise Two peaks
Change the setting: prior mean $d_p = {{bi_dp}}$ with $\Sigma_p = {{bi_vp}}$, and
the animal hears $u = {{bi_u}}$ with $\Sigma_u = {{bi_vu}}$.

(a) Plot the log joint on $0.05 \le d \le 10$. How many maxima are there?

(b) Run the ascent from $d_0 = d_p$ with $\eta = {{bi_eta}}$. Which one does it
find? Is it the best one?

(c) Now start it at $d_0 = 1$, and then at $d_0 = {{bi_start_low}}$. Explain both results.
---solution---
(a) Two: one at $d = {{bi_peak_lo:.4f}}$ and one at $d = {{bi_peak_hi:.4f}}$. The
posterior is bimodal. The animal is entertaining two incompatible stories, a near
source and a far one, and the data do not settle between them.

The lower peak is the taller: the far peak's height is
${{bi_height_ratio:.4f}}$ of it. So $d = {{bi_peak_lo:.4f}}$ is the MAP estimate.

(b) From $d_0 = {{bi_dp}}$ the ascent converges to ${{bi_from_prior:.4f}}$, the
*wrong* peak. Gradient ascent is local: it goes uphill from where it starts and
has no way to know that a taller peak exists somewhere it never visited. Nothing
in the run signals a problem. The gradient reaches zero, the errors balance, and
the animal is confidently at a local maximum.

(c) From $d_0 = 1$ it converges to ${{bi_from_one:.4f}}$, the global maximum,
because it started on that peak's side of the valley. The basin of attraction,
not the quality of the answer, decides which one you get.

From $d_0 = {{bi_start_low}}$ it lands at ${{bi_from_low:.4f}}$, the far peak, even though it
started nearer the global one. This is a different failure. At $d = {{bi_start_low}}$ the
predicted intensity is $g({{bi_start_low}}) = {{bi_g_at_low:.4f}}$ against ${{bi_u}}$ heard, and $|g'|$ is
large there, so the gradient is ${{bi_grad_at_low:.1f}}$. One step at
$\eta = {{bi_eta}}$ moves the estimate by ${{bi_step_at_low:.3f}}$, clear over the valley
and the peak beyond it. A rate that is mild in one region is wild in another.

Both failures are properties of the method and neither is announced by it. This
is the first serious cost of the move made in Lesson&nbsp;3. Trading the whole
posterior for its peak was cheap when there was one peak. A distribution with two
modes has something to say that no single number can carry, and reporting
${{bi_from_prior:.4f}}$ conceals the existence of ${{bi_peak_lo:.4f}}$ entirely.
Week&nbsp;4's $Q$ is a distribution rather than a point in part for this reason,
though a unimodal $Q$ will turn out to have its own version of the problem.
:::

::: exercise The mode moves when you rename the axis &dagger;
The animal could equally well estimate $\rho = \ln d$, the log-distance, which
describes the same physical situation in different coordinates.

(a) The prior over $d$ is $\mathcal{N}(d_p, \Sigma_p)$. Write the density over
$\rho$ that it induces, remembering the Jacobian.

(b) Find the MAP estimate in $\rho$, then convert back with $e^{\hat\rho}$.
Compare with the MAP in $d$.

(c) Do the same for the posterior mean. What do you conclude about the mode as a
summary?
---solution---
(a) Changing variables in a density requires the Jacobian. With $d = e^{\rho}$,
$\mathrm{d}d/\mathrm{d}\rho = e^{\rho} = d$, so

$$
p(\rho) = p\big(d(\rho)\big)\,\left|\frac{\mathrm{d}d}{\mathrm{d}\rho}\right|
        = p(e^{\rho})\,e^{\rho}.
$$

In logs, $\ln p(\rho) = \ln p(d) + \rho$.

(b) The log posterior in $\rho$ is therefore the log posterior in $d$ plus
$\rho = \ln d$, and that extra term has a nonzero derivative, $1/d$. So the
stationary condition is different and the peak sits somewhere else. Numerically,
maximising in $d$ gives ${{map_in_d:.4f}}$, while maximising in $\rho$ and
transforming back gives $e^{\hat\rho} = {{map_via_logd:.4f}}$, higher by
{{map_reparam_pct:.1f}}%.

(c) The posterior mean is unaffected. It is
$\mathbb{E}[d] = \int d\,p(d)\,\mathrm{d}d$, and a change of variables in that
integral carries the Jacobian in the measure as well as in the density, so the
two cancel and the number is the same. Both routes give ${{post_mean:.4f}}$.

The conclusion is uncomfortable. A density is not invariant under
reparameterisation, because it is a quantity per unit of something and the unit
changed. So the location of its peak is a property of the coordinates as much as
of the belief, and "the most probable distance" is not well defined until you say
which variable you measured probability per unit of. Nothing distinguishes
distance from log-distance physically; the animal is not told which one it is
estimating.

Two things follow. First, MAP is a convention rather than a canonical answer, and
the convention is usually invisible because nobody writes down the alternative
coordinates. Second, quantities defined by expectation, the mean, the variance,
the free energy of Week&nbsp;4, do not have this problem, which is a reason to
prefer methods built on them. Week&nbsp;12 returns to this; it is one of the more
serious objections to the framework, and it does not go away by being ignored.
:::

::: exercise Ambiguity has a shape
In Lesson&nbsp;4, $|g'(d)|$ was called the sensitivity of the observation to the
state. Make it quantitative.

(a) Show that under the Gauss-Newton approximation the sensory channel
contributes a precision $g'(d)^2\Pi_u$ *about the state*, and check its units.

(b) With $g(d) = 1/d^2$ and $\Pi_u = {{prec_obs}}$, evaluate this at
$d = 1$, $d = {{post_mode:.2f}}$ and $d = 4$.

(c) A colleague proposes putting the animal's ear on a logarithmic scale, so it
measures $\ln u$ rather than $u$, with Gaussian noise of fixed variance in the
new units. What happens to the sensitivity?
---solution---
(a) The curvature exercise of Lesson&nbsp;4 gave
$\mathcal{F}'' = -\Pi_p - \Pi_u g'^2 + \Pi_u g''(u - g)$, and dropping the last
term leaves $-(\Pi_p + \Pi_u g'^2)$. Since curvature at a peak is minus a
precision, the sensory channel contributes $\Pi_u g'^2$.

Units: $\Pi_u$ is inverse squared intensity and $g'$ is intensity per distance,
so $g'^2\Pi_u$ is inverse squared distance, matching $\Pi_p$. They may be added,
which they could not before the conversion.

(b) $g'(d) = -2/d^3$, so $g'^2\Pi_u = 4\Pi_u/d^6$, and with
$\Pi_u = {{prec_obs}}$:

<table>
<thead><tr><th>$d$</th><th>$g'(d)$</th><th>$g'^2\Pi_u$</th></tr></thead>
<tbody>
<tr><td>1</td><td>{{gprime_at_1:.3f}}</td><td>{{sensprec_at_1:.3f}}</td></tr>
<tr><td>{{post_mode:.2f}}</td><td>{{gprime_at_mode:.4f}}</td><td>{{sensory_prec_on_state:.4f}}</td></tr>
<tr><td>4</td><td>{{gprime_at_4:.3f}}</td><td>{{sensprec_at_4:.4f}}</td></tr>
</tbody>
</table>

Across a factor of four in distance, the information the ear carries about that
distance falls by a factor of $4^6 = {{sensprec_ratio_1_to_4:.0f}}$. Far away, the animal is nearly blind
however good its ear, and no amount of sensory precision compensates, because
$\Pi_u$ enters multiplied by a number heading to zero.

(c) The measurement becomes $\ln u$, whose predicted value is
$\ln g(d) = -2\ln d$, with derivative $-2/d$. The sensitivity is now $4\Pi/d^2$
rather than $4\Pi_u/d^6$: it still decays, but as an inverse square rather than
an inverse sixth power.

Compressing the sensory scale has spread the discriminable range out. This is a
real design principle rather than an artefact, and it is one reason sensory
systems are so often logarithmic. Note what it is not: it is not extra
information. The same intensities arrive. What changed is where the fixed noise
budget sits relative to the signal, so a channel with constant noise in log units
is a channel whose noise scales with the signal in linear units.
:::

::: checkpoint
- Two of these problems show the method failing. Name each failure in one
  sentence, and say whether more computation would fix it.
- Which of this week's results survive if $g$ is nonlinear, and which were
  special to $g(d) = d$?
- What is a sufficient statistic, in the sense used in the repeated-listening
  problem?
:::
