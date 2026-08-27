---
title: A problem small enough to solve exactly
deck: One hidden variable, one noisy reading, and a nonlinear link between them. Small enough to solve by brute force, which is what makes it useful.
week: 2
time: 30 min
---

Here is the problem the next three lessons will work with. It has exactly one
unknown, which means we can compute the exact answer by brute force and compare
everything against it. That comparison is the point: a method whose approximation
you cannot see is a method you cannot judge.

## The setup

An animal hears a sound. Write $d$ for the distance to the source, $d$ for
distance: this is the hidden variable, the thing the animal wants to know and
cannot measure. What it can measure is intensity, and intensity falls off as the
inverse square of distance. Write $g$ for that link, $g$ for the **observation
function**, the map from a hidden state to the observation it would produce:

$$
g(d) \;=\; \frac{1}{d^{2}} .
$$ {#link}

The source's power has been taken as one, which fixes the units so that intensity
is $1$ at unit distance. Nothing turns on that choice; it saves carrying a
constant through every line.

$g$ is the same object Week&nbsp;1 called the model's likelihood, seen from a
different angle. There we asked what observation a state makes probable; here we
ask what observation a state *predicts*, and then add noise around the
prediction. The letter $g$ is reserved for it for the rest of the course.

Its ear is imperfect. Write $u$ for the intensity it actually registers, and take
the error to be Gaussian about the true intensity:

$$
P(u \mid d) \;=\; \mathcal{N}\big(u;\ g(d),\ \Sigma_u\big),
\qquad
\mathcal{N}(x; \mu, \Sigma) = \frac{1}{\sqrt{2\pi\Sigma}}\exp\!\left(-\frac{(x-\mu)^2}{2\Sigma}\right).
$$ {#likelihood}

Here $\Sigma_u$ is the variance of the sensory noise, in squared units of
intensity: small means a reliable ear.

The animal is not starting from nothing. Sources it cares about tend to be at a
characteristic sort of range, and that expectation is its prior:

$$
P(d) \;=\; \mathcal{N}\big(d;\ d_p,\ \Sigma_p\big),
$$ {#prior}

with $d_p$ the distance it expects and $\Sigma_p$ how firmly, again a variance.

::: mn On the Gaussian prior
A Gaussian over a distance puts a little mass below zero, which is unphysical.
Nothing in what follows depends on that tail, and carrying a truncation through
the algebra would obscure the argument. Week&nbsp;3 discusses what to do when the
choice of family matters.
:::

Concretely: $d_p = {{d_prior}}$, $\Sigma_p = {{var_prior}}$, and the animal hears
$u = {{u_obs}}$ with $\Sigma_u = {{var_obs}}$.

Before computing anything, notice the tension. The prior expects a source at
$d = {{d_prior}}$, which by [eq:link] would produce an intensity of
${{g_at_prior:.2f}}$. The animal heard ${{u_obs}}$, twice that. Taken alone, an
intensity of ${{u_obs}}$ means $d = 1/\sqrt{ {{u_obs}} } = {{d_from_data_alone:.4f}}$.
So the prior says one thing, the data another, and the answer will be somewhere
between. Where exactly is the question.

## The exact posterior

Bayes' theorem, in the continuous form:

$$
P(d \mid u) \;=\; \frac{P(u \mid d)\,P(d)}{P(u)},
\qquad
P(u) \;=\; \int_0^{\infty} P(u \mid d)\,P(d)\;\mathrm{d}d .
$$ {#continuous-bayes}

The numerator is a product of two things we have written down. The denominator is
an integral over every distance the source might be at, and it is the same
difficulty as Week&nbsp;1's sum with the sum replaced by an integral. There are
no longer $k^n$ terms to add; there are uncountably many, and no finite device
adds them.

For a problem this small we can evaluate it numerically, on a fine grid, and see
what the answer looks like. Do that and three things stand out.

::: fig The exact posterior over distance, against the prior and the likelihood that produced it. Each curve is scaled to its own peak, so the figure compares shapes rather than heights. The likelihood is plotted as a function of $d$, which is what makes it lean.
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Prior, likelihood and posterior over distance. The posterior is skewed right, with its mode below its mean.">
  <rect x="1" y="1" width="678" height="298" fill="#fdfcf9" stroke="#e4e1d9"/>
  <line x1="60" y1="250" x2="640" y2="250" stroke="#3c3c42"/>
  <line x1="60" y1="30" x2="60" y2="250" stroke="#3c3c42"/>
  <path d="{{fig_prior_path}}" fill="none" stroke="#90909a" stroke-width="1.8" stroke-dasharray="4 3"/>
  <path d="{{fig_lik_path}}" fill="none" stroke="#8a6d1f" stroke-width="1.8" stroke-dasharray="2 3"/>
  <path d="{{fig_post_path}}" fill="none" stroke="#0f5f57" stroke-width="2.6"/>
  <line x1="{{fig_mode_x}}" y1="{{fig_mode_y}}" x2="{{fig_mode_x}}" y2="250" stroke="#0f5f57" stroke-width="1" stroke-dasharray="2 3"/>
  <line x1="{{fig_mean_x}}" y1="{{fig_mean_y}}" x2="{{fig_mean_x}}" y2="250" stroke="#9c4d2f" stroke-width="1" stroke-dasharray="2 3"/>
  <text x="{{fig_mode_x}}" y="26" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#0f5f57">mode {{post_mode:.2f}}</text>
  <text x="{{fig_mean_x}}" y="70" text-anchor="start" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#9c4d2f">&#8592; mean {{post_mean:.2f}}</text>
  <text x="60" y="268" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">0</text>
  <text x="{{fig_mode_x}}" y="268" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">{{post_mode:.2f}}</text>
  <text x="640" y="268" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#6d6d75">6</text>
  <text x="350" y="288" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">distance d</text>
  <rect x="470" y="40" width="12" height="3" fill="#0f5f57"/>
  <text x="490" y="45" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#3c3c42">posterior</text>
  <rect x="470" y="58" width="12" height="2" fill="#90909a"/>
  <text x="490" y="63" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#3c3c42">prior</text>
  <rect x="470" y="76" width="12" height="2" fill="#8a6d1f"/>
  <text x="490" y="81" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#3c3c42">likelihood</text>
</svg>
:::

**It is not Gaussian.** Both ingredients were Gaussian and the result is not,
because [eq:link] is nonlinear. Passing a Gaussian through $1/d^2$ stretches one
side and compresses the other, and the product inherits the distortion. Measured
on the grid, the posterior's skewness is ${{post_skew:+.3f}}$, where a Gaussian
would give zero. This is the general situation: a nonlinear link between what you
want and what you measure takes you out of any convenient family, and no clever
choice of prior brings you back.

**Its mean and its mode are different, and not slightly.** The peak sits at
$d = {{post_mode:.3f}}$; the mean sits at $d = {{post_mean:.3f}}$, higher by
{{mean_mode_pct:.0f}}%. Both are defensible answers to "how far away is it", and
they disagree. The long right tail, which is the possibility that the source is
distant and the ear happened to over-report, drags the mean out without moving
the peak. Remember this: the method built over the next two lessons finds the
mode, and the fact that it is not the mean is a choice the framework makes
quietly.

**It required an integral we cannot do in general.** The curve above was computed
by evaluating [eq:continuous-bayes] at six hundred thousand grid points and
normalising. That is available here because $d$ is one number. With ten unknowns
the same grid would need $10^{60}$ points, and the grid is the *easy* method.

::: keyidea
Two obstacles, and they are different from each other. The **normaliser** is an
integral over the whole state space, which grows impossible with dimension. The
**shape** is not in any standard family, so even knowing the posterior exactly,
an agent could not summarise it in a few numbers to carry around.

Week&nbsp;1 met the first, in its counting form. The second is new, and it is why
the next lesson gives up on the distribution altogether and goes looking for a
single point.
:::

::: exercise Why the likelihood leans
In the figure the likelihood, plotted against $d$, is visibly lopsided even
though the sensory noise is Gaussian. Explain why, and say which way it would
lean if the link were $g(d) = d^2$ instead of $1/d^2$.
---solution---
The noise is Gaussian *in intensity*, so the likelihood is symmetric as a
function of $u$. Plotted against $d$ it is not, because $d$ and $u$ are related
nonlinearly and the map compresses one region while stretching another.

Concretely, near $d = {{d_from_data_alone:.2f}}$ the derivative
$g'(d) = -2/d^{3}$ has magnitude about $0.7$, so a given interval in intensity
corresponds to an interval in distance about $1/0.7$ times as wide. Further out,
$|g'|$ falls sharply: at $d = 4$ it is about $0.03$, so the same interval in
intensity spans a range in distance more than twenty times wider. Equal steps in
intensity become unequal steps in distance, and the further away the source, the
more distance a given change in loudness buys. The likelihood therefore has a
long tail to the right.

With $g(d) = d^{2}$ the situation reverses. There $g'(d) = 2d$ grows with
distance, so far-away distances are *compressed* in intensity terms and the tail
runs the other way, towards small $d$.

The general statement: a nonlinear change of variable reweights a density by
$|\mathrm{d}u/\mathrm{d}d|$, and wherever that factor is small the density spreads out.
:::

::: checkpoint
- Both prior and likelihood are Gaussian. Why is the posterior not?
- Which is larger here, the posterior mean or the posterior mode, and what
  feature of the distribution decides it?
- Name the two separate difficulties this lesson identifies, and say which of
  them Week&nbsp;1 had already met.
:::
