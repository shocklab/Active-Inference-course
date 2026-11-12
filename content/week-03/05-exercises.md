---
title: Problems and code
deck: Worked problems on learning rules, their stationary points, the missing term and the dynamics of learning.
week: 3
time: 100 min
number_sections: false
---

Work the problems before opening the solutions. The ones marked with a dagger are
used again later in the course.

## The notebooks

| Notebook | What it does | Run it |
|---|---|---|
| `week03_numpy.ipynb` | The three learning rules on Week&nbsp;2's model, run trial by trial with the same generator and inner loop as the widgets; the wrong-link experiment in a louder and a quieter world; the point and posterior-averaged rules for the sensory variance; and the linear case checked against its closed-form stationary points | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-03/week03_numpy.ipynb) |
| `week03_jax.ipynb` | What needs many runs at once: the hand-derived parameter gradients checked against automatic differentiation, a thousand independent learners run side by side to show the spread of the learned values about the point rule's stationary point, and the drift of both rules swept over a grid of parameter values | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-03/week03_jax.ipynb) |

A single run of a learning rule shows one path through one sequence of sounds. The
JAX notebook runs many sequences together, which is what separates a rule's bias from
the noise of a particular run.

There is no `pymdp` notebook this week. The library works with the discrete
generative models of Week&nbsp;9, and nothing here is discrete.

## Problems

::: exercise The interneuron rule
In the circuit of [lesson:3.2] that learns $\Sigma_u$ with an extra unit, from
[Bogacz (2017)](https://doi.org/10.1016/j.jmp.2015.11.003), in his section on learning the variance of a single error unit, the error unit
$\varepsilon$ and the inhibitory unit $e$ follow

$$
\dot\varepsilon \;=\; u - g(d) - e,
\qquad
\dot e \;=\; \Sigma_u\,\varepsilon - e ,
$$

with $d$ held fixed.

(a) Find the fixed point and show that it is stable.

(b) At the fixed point, show that the update $\Delta\Sigma_u = \beta(\varepsilon e - 1)$
uses only the activities of the two units the connection of strength $\Sigma_u$ joins,
and that its drift vanishes where $\mathbb E_{P^*}[(u - g(d))^2] = \Sigma_u$.

(c) Show that it equals $2\Sigma_u$ times the gradient [lesson:3.2] derived for
$\Sigma_u$, and say what the factor changes.
---solution---
(a) Setting both derivatives to zero: the second gives $e = \Sigma_u\varepsilon$, and
the first then gives $u - g(d) = \Sigma_u\varepsilon$. So $\varepsilon = (u - g(d))/\Sigma_u$,
the precision-weighted error, and $e = u - g(d)$, the raw residual. The system is
linear with matrix

$$
\begin{pmatrix} 0 & -1 \\ \Sigma_u & -1 \end{pmatrix},
$$

whose trace is $-1$ and determinant $\Sigma_u > 0$. Two eigenvalues with negative sum
and positive product have negative real parts, so the fixed point is stable.

(b) The connection of strength $\Sigma_u$ runs from $\varepsilon$ to $e$, and
$\varepsilon e - 1$ is built from those two activities. At the fixed point
$\varepsilon e = (u - g(d))^2/\Sigma_u$, so the drift vanishes where
$\mathbb E_{P^*}[(u - g(d))^2] = \Sigma_u$.

(c) The gradient is $\tfrac12\big((u - g)^2/\Sigma_u^2 - 1/\Sigma_u\big)$. Multiplying by
$2\Sigma_u$ gives $(u - g)^2/\Sigma_u - 1 = \varepsilon e - 1$. The factor is positive, so
the direction of every step and the stationary point are unchanged; it rescales the
step, making it smaller when $\Sigma_u$ is small, which is where [lesson:3.2]'s warning
showed the plain gradient misbehaving.
:::

::: exercise The sensory variance collapses too &dagger;
Use the linear model of [lesson:3.4], but now let the animal know the prior variance,
$\Sigma_p = \Sigma_p^{*}$, and learn the sensory variance by the point rule. Write
$\Sigma_u^{*}$ for the world's value. Find the stationary points and the condition for
them to exist.
---solution---
The residual at the mode is

$$
u - \hat d \;=\; (u - d_p) - \frac{\Sigma_p}{\Sigma_p + \Sigma_u}(u - d_p) \;=\; \frac{\Sigma_u}{\Sigma_p + \Sigma_u}\,(u - d_p),
$$

so its average square over the process is
$\Sigma_u^2(\Sigma_p + \Sigma_u^{*})/(\Sigma_p + \Sigma_u)^2$. The point rule's drift
vanishes where this equals $\Sigma_u$:

$$
\Sigma_u(\Sigma_p + \Sigma_u^{*}) \;=\; (\Sigma_p + \Sigma_u)^2
\quad\Longleftrightarrow\quad
\Sigma_u^2 + (\Sigma_p - \Sigma_u^{*})\,\Sigma_u + \Sigma_p^2 \;=\; 0 .
$$

This is [lesson:3.4]'s quadratic with the two variances exchanged, so the same three
cases follow. Real positive roots need $\Sigma_u^{*} \geq 3\Sigma_p$; at equality the
two coincide at $\Sigma_u = \Sigma_p$, a root that attracts only from above, and above
equality the larger root attracts and the smaller separates collapse from survival. A
channel whose noise is less than three times the prior's spread has its variance
learned all the way to zero: the more the prior can
absorb, the less noise the rule sees. The model is symmetric in the two variances
because $d$ and $u - d$ enter it in the same way.
:::

::: exercise Adding the missing term
In the linear model of [lesson:3.4], the animal climbs
$\hat\ell + \tfrac12\ln(2\pi\Sigma_{\text{post}})$ in $\Sigma_p$ instead of $\hat\ell$.
Show that the drift now vanishes at $\Sigma_p = \Sigma_p^{*}$, and say why this
does not follow from the point rule plus a correction that would be available in
Week&nbsp;2's model.
---solution---
From [lesson:3.4], the derivative of $\mathbb E_{P^*}[\hat\ell]$ is
$(\Sigma_p^{*} + \Sigma_u)/\big(2(\Sigma_p + \Sigma_u)^2\big) - 1/(2\Sigma_p)$. With
$\ln\Sigma_{\text{post}} = \ln\Sigma_p + \ln\Sigma_u - \ln(\Sigma_p + \Sigma_u)$, the
derivative of $\tfrac12\ln(2\pi\Sigma_{\text{post}})$ is
$1/(2\Sigma_p) - 1/\big(2(\Sigma_p + \Sigma_u)\big)$. The $1/(2\Sigma_p)$ terms cancel,
leaving

$$
\frac{\Sigma_p^{*} + \Sigma_u}{2(\Sigma_p + \Sigma_u)^2} - \frac{1}{2(\Sigma_p + \Sigma_u)},
$$

which vanishes at $\Sigma_p + \Sigma_u = \Sigma_p^{*} + \Sigma_u$, that is at the truth.

In the linear model the correction is exact because the posterior is Gaussian and its
width does not depend on $u$. In Week&nbsp;2's model the posterior is skewed and its
width changes from sound to sound, so $\tfrac12\ln(2\pi\Sigma_{\text{post}})$ with a
Gaussian width read from the curvature at the mode is an approximation. Week&nbsp;4
makes that approximation explicit as the Laplace approximation and finds where it
fails.
:::

::: exercise A hyperprior delays the collapse
In the linear model with $\Sigma_p^{*} < 3\Sigma_u$, add to the point rule's batch
objective over $N$ trials the inverse-gamma hyperprior of [lesson:3.4] with $n_0$ and
$\Sigma_0$.

(a) Show that the objective is now bounded above and has a maximum at a positive
$\Sigma_p$ for every $N$.

(b) Show that as $N \to \infty$ with $n_0$ fixed, the maximising $\Sigma_p$ tends to zero.
What does that say about a hyperprior with a fixed number of pseudo-trials, and what
would change if the hyperprior were shared across a fixed window of trials however
long the learner runs?
---solution---
(a) Write the expected point-rule objective per trial, the expected settled log joint, as
$\bar\ell(\Sigma_p) = -(\Sigma_p^{*} + \Sigma_u)/\big(2(\Sigma_p + \Sigma_u)\big) - \tfrac12\ln\Sigma_p$
up to constants, from [lesson:3.4]. This is the batch objective in expectation, and
(b) works with it. A realised batch of $N$ trials has the mean squared deviation of its
own observations from $d_p$, call it $B_N$, in place of $\Sigma_p^{*} + \Sigma_u$; the
argument of this part uses only that the number is positive, so it holds for a
realised batch too. The batch objective is
$N\bar\ell(\Sigma_p) - (n_0/2 + 1)\ln\Sigma_p - n_0\Sigma_0/(2\Sigma_p)$. As $\Sigma_p \to 0$, the
last term goes to $-\infty$ like $1/\Sigma_p$, faster than the $+\ln$ terms grow, and as
$\Sigma_p \to \infty$ the $-\ln\Sigma_p$ terms go to $-\infty$. A continuous function on
$(0, \infty)$ that tends to $-\infty$ at both ends has a maximum.

(b) Write $x = \Sigma_p$, $S = \Sigma_p^{*}$, $s = \Sigma_u$ and $A = n_0\Sigma_0$. Setting
the derivative of the batch objective to zero and multiplying by $2x^2$,

$$
A \;-\; (n_0 + 2)\,x \;-\; N x\left[1 - \frac{(S + s)\,x}{(x + s)^2}\right] \;=\; 0 .
$$

The bracket is $1 - (S+s)x/(x+s)^2$. Since $x/(x+s)^2$ is largest at $x = s$, where it
is $1/(4s)$, the bracket is at least $1 - (S+s)/(4s) = (3s - S)/(4s)$, which is
positive when $S < 3s$. So at every stationary point
$N x\,(3s - S)/(4s) \leq A$, that is

$$
x \;\leq\; \frac{4sA}{(3s - S)\,N} ,
$$

and every stationary point, the maximiser included, tends to zero as $N \to \infty$.
For a realised batch the same steps need $B_N < 4s$ in place of $S + s < 4s$, and a
finite batch need not meet that even when $S < 3s$, since its mean squared deviation
can exceed its expectation. With independent trials $B_N \to S + s$ as $N \to \infty$
by the law of large numbers, so for large enough $N$ the bound applies to the realised
batch as well. Near zero the bracket tends to one, so the maximiser is
$x \approx A/(N + n_0 + 2) \approx n_0\Sigma_0/N$ for large $N$.

A hyperprior worth a fixed number of trials is outweighed by the data in the end, and
if the rule's own drift is negative everywhere, the data push the variance to zero.
Sharing the hyperprior across a fixed window of $N_w$ trials, whatever the length of the
run, keeps its weight relative to each trial's data fixed. The objective over that
window then has a positive maximiser, by the argument of (a): the term
$-n_0\Sigma_0/(2\Sigma_p)$ still sends it to $-\infty$ as $\Sigma_p \to 0$. The bound above,
with $N_w$ in place of $N$, is an upper bound on that maximiser, of order
$n_0\Sigma_0/N_w$; it does not show that the maximiser is positive, and neither
statement shows that an online learner converges to it. Choosing $N_w$ is a modelling decision. The learning rate suggests an order of
scale, since by [lesson:3.1] an online learner relaxes over about
$1/\big(\beta|\varphi'|\big)$ trials, but it does not fix one. Either way, the
hyperprior bounds the objective without removing the bias of the point rule.
:::

::: exercise Choosing a learning rate
In the linear running average of [lesson:3.1] with
$\kappa = \beta/(\Sigma_p + \Sigma_u)$, the prior location's memory has half-life
$\ln 2/(-\ln(1 - \kappa))$ trials and its fluctuation about the target has variance
$\kappa\,(\Sigma_p^{*} + \Sigma_u)/(2 - \kappa)$, where $\Sigma_p^{*} + \Sigma_u$ is the
variance of $u$.

(a) For small $\kappa$, by what factor must $\beta$ shrink to halve the standard
deviation of the fluctuation, and what happens to the half-life?

(b) The world's location jumps once to a new value. Describe the trade-off between
tracking the jump and settling accurately, and say what quantity a learner would need
in order to choose $\beta$ well.
---solution---
(a) For small $\kappa$ the variance is about $\kappa(\Sigma_p^{*} + \Sigma_u)/2$, so the
standard deviation scales as $\sqrt\kappa$, hence as $\sqrt\beta$. Halving it needs
$\beta$ to fall by a factor of four. The half-life is about $\ln 2/\kappa$ for small
$\kappa$, so it grows by a factor of four. At [lesson:3.1]'s values the half-life is
${{halflife_lin:.1f}}$ trials and the standard deviation ${{sd_dp_lin:.4f}}$.

(b) After a jump, the error in $d_p$ decays by a factor $(1 - \kappa)$ per trial, so a
small $\beta$ follows slowly. Once it has followed, a small $\beta$ fluctuates less. The
right $\beta$ depends on how often the world changes compared with how noisy each
trial is. A learner that does not know the rate of change has to learn it, which is a
parameter of a model of how the parameters themselves drift. That is a model one level
up, the kind of hierarchy [week:5] begins to build.
:::

::: exercise Implement it
Using the NumPy notebook's `run` function, repeat [lesson:3.3]'s experiment in a world
quieter than the model, $\theta^{*} = \tfrac12$, with learning off and then on. Report
the averages of the two errors and of the learning signal over the last 200 trials, and
compare their signs with the louder world.
---solution---
With learning off, the averages over the last 200 trials are ${{w2q_off_epsp:.3f}}$
for the prior error, ${{w2q_off_epsu:.3f}}$ for the sensory error and
${{w2q_off_sig:.4f}}$ for the learning signal. With learning on, the power falls from
$1$ to ${{w2q_final:.3f}}$ by the last trial.

Only the learning signal changes sign between the two worlds. In the louder world the
prior error was ${{w2_off_epsp:.3f}}$ and the sensory error ${{w2_off_epsu:.3f}}$; here
both keep their signs and only their sizes move. A reader who judged the link by the
sign of either error alone would call both worlds the same.
:::

::: exercise The location in the linear world
In the linear world of [lesson:3.1], the process draws $d$ from
$\mathcal N(d_p^{*}, \Sigma_p^{*})$ on the whole line and $u = d + z$ with
$z \sim \mathcal N(0, \Sigma_u^{*})$. The animal holds fixed positive variances
$\Sigma_p$ and $\Sigma_u$, not necessarily the world's, and learns $d_p$ by the point
rule. Find the drift, show that its only zero is $d_p^{*}$ and that it attracts, and
say which step fails in Week&nbsp;2's model.
---solution---
From [lesson:3.1], the step per unit rate at the mode is
$(\hat d - d_p)/\Sigma_p = (u - d_p)/(\Sigma_p + \Sigma_u)$. Its average over the
process uses $\mathbb E_{P^*}[u] = d_p^{*} + \mathbb E_{P^*}[z] = d_p^{*}$:

$$
\varphi(d_p) \;=\; \frac{d_p^{*} - d_p}{\Sigma_p + \Sigma_u} .
$$

It vanishes only at $d_p = d_p^{*}$, and $\varphi' = -1/(\Sigma_p + \Sigma_u) < 0$, so
the zero attracts. Neither the animal's variances nor the world's enter the location of
the zero, so the location is learned without bias even by an animal whose variances
are wrong.

The step that fails in Week&nbsp;2's model is the first. The mode is not linear in $u$
there, so its average over the process is not the average of the posterior means, and
by [lesson:3.4] the point rule's drift differs from the posterior-averaged rule's by the
average of $(m - \hat d)/\Sigma_p$, which is not zero when the posteriors are skewed.
:::

::: exercise The two pulls on the source power
For the source power, the per-trial derivative as a function of the state is
$\psi(d) = \big(u - \theta h(d)\big)h(d)/\Sigma_u$ with $h(d) = 1/d^2$. Find $\psi'$ and
$\psi''$, evaluate both where the residual $u - \theta h(d)$ is zero, and say what the
second-order expansion of [lesson:3.4] can and cannot tell you about the sign of the
point rule's error in $\theta$.
---solution---
Differentiating the product,

$$
\psi' \;=\; \frac{h'\,(u - 2\theta h)}{\Sigma_u},
\qquad
\psi'' \;=\; \frac{(u - 2\theta h)\,h'' - 2\theta h'^2}{\Sigma_u} .
$$

With $h = d^{-2}$, $h' = -2d^{-3}$ and $h'' = 6d^{-4}$. Where $u = \theta h$, the factor
$u - 2\theta h$ is $-\theta h = -\theta d^{-2}$, so

$$
\psi' \;=\; \frac{2\theta}{\Sigma_u d^{5}} \;>\; 0,
\qquad
\psi'' \;=\; \frac{-6\theta - 8\theta}{\Sigma_u d^{6}} \;=\; -\frac{14\theta}{\Sigma_u d^{6}} \;<\; 0 .
$$

In the expansion
$\mathbb E_{d\mid u}[\psi] - \psi(\hat d) \approx \psi'(m - \hat d) + \tfrac12 \psi''[V + (m - \hat d)^2]$,
a posterior whose mean lies beyond its mode makes the first term positive, and the
posterior variance makes the second negative. The posterior-averaged rule has zero
drift at the truth, so the point rule's drift there is minus the average of the exact
gap. The expansion shows two contributions of opposite sign, so the error can take
either sign, but it does not say which wins: $\psi$ is not a polynomial, and the terms
it drops matter. In [lesson:3.3]'s louder world the second-order sum averages to
${{gap_two_loud:.4f}}$ while the exact gap averages to ${{gap_exact_loud:.4f}}$. The
sign comes from the full posterior calculation, which gives the power overestimated
when the world equals Week&nbsp;2's model and underestimated in the louder world.
:::

::: exercise Second-order parameter dynamics &dagger;
[Buckley et al. (2017)](https://doi.org/10.1016/j.jmp.2017.09.004), section 8, write
the dynamics of a parameter in second-order form: the parameter's acceleration, rather
than its velocity, is proportional to the gradient. Apply this to the prior location in
the linear model, in continuous time and with the update replaced by its expectation
over trials.

(a) Show that the drift is $(\bar u - d_p)/(\Sigma_p + \Sigma_u)$, where $\bar u$ is the
mean of $u$ over the process, and write the second-order equation for $d_p$.

(b) Solve it from rest at $d_p(0) \neq \bar u$ and describe the motion.

(c) Add a friction term $-b\,\dot d_p$ and find the value of $b$ that returns $d_p$ to
$\bar u$ fastest without overshoot.
---solution---
(a) In the linear model $\hat d - d_p = \Sigma_p(u - d_p)/(\Sigma_p + \Sigma_u)$, so the
gradient $-\varepsilon_p = (\hat d - d_p)/\Sigma_p = (u - d_p)/(\Sigma_p + \Sigma_u)$, with
no truncation term on the whole line. Its average over the process is
$(\bar u - d_p)/(\Sigma_p + \Sigma_u)$. With a rate constant $\beta$,

$$
\ddot d_p \;=\; \frac{\beta}{\Sigma_p + \Sigma_u}\,(\bar u - d_p) .
$$

(b) Write $x = d_p - \bar u$ and $\omega_0^2 = \beta/(\Sigma_p + \Sigma_u)$. Then
$\ddot x = -\omega_0^2 x$, with solution $x(t) = x(0)\cos\omega_0 t$ from rest. The prior
location oscillates about the target forever with constant amplitude, passing through
$\bar u$ at full speed and reaching $\bar u - x(0)$ on the other side. Gradient flow in first-order
form, $\dot x = -\omega_0^2 x$, decays to the target instead. The second-order form
needs something else to stop.

(c) With friction, $\ddot x + b\dot x + \omega_0^2 x = 0$, whose characteristic roots are
$\tfrac12\big(-b \pm \sqrt{b^2 - 4\omega_0^2}\big)$. For $b < 2\omega_0$ the roots are
complex and $x$ oscillates with decaying amplitude; for $b > 2\omega_0$ both are real and
negative and the slower one, $\tfrac12\big(-b + \sqrt{b^2 - 4\omega_0^2}\big)$, approaches
zero as $b$ grows, so heavy friction is slow. The fastest return without overshoot is at
$b = 2\omega_0$, critical damping, where $x(t) = x(0)(1 + \omega_0 t)\exp(-\omega_0 t)$.
:::

::: exercise Precision as the learned quantity in the circuit
Week&nbsp;2's circuit applies each precision as a gain at an error unit. Suppose the
animal stores $\Pi_u$ at that unit and learns it in log coordinates. Write the update,
say what it uses, and check that its stationary point is the same as that of the
log-variance update.
---solution---
Since $\ln\Pi_u = -\ln\Sigma_u$, the derivative in $\ln\Pi_u$ is minus the derivative in
$\ln\Sigma_u$, which [lesson:3.2] found to be $\tfrac12(\delta_u^2/\Sigma_u - 1)$. So

$$
\Pi_u \;\leftarrow\; \Pi_u\exp\!\Big(\tfrac{\beta}{2}\big(1 - \Pi_u\delta_u^2\big)\Big),
$$

and since $\Pi_u\delta_u^2 = \varepsilon_u\delta_u$, it uses the error unit's activity,
the raw residual and the gain itself. The drift in $\ln\Pi_u$ vanishes where
$\mathbb E_{P^*}[\Pi_u\delta_u^2] = 1$, that is $\mathbb E_{P^*}[\delta_u^2] = \Sigma_u$, the
same zero as the log-variance update, since a step in one coordinate is exactly the
opposite step in the other. That zero is where the average change in the logarithm
vanishes. The average change in $\Pi_u$ itself is not zero there: the step multiplies
$\Pi_u$ by an exponential, whose average exceeds one whenever the exponent varies, as
[lesson:3.2] noted for the variance.
:::
