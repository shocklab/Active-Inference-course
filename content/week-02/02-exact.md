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

An animal hears a sound. Write $d$ for the distance to the source, the letter
chosen for exactly that. It is a **hidden state** in Week&nbsp;1's sense, the same
role $s$ played there: something the world has and the agent does not, to be
reached only through what arrives at the senses. Week&nbsp;1's $s$ ranged over
three named possibilities; $d$ ranges over an interval, which is the only
difference and the reason for a fresh letter.

What the animal can measure is intensity, and intensity falls off as the inverse
square of distance. Write $g$ for that link, $g$ for the **observation
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

Its ear is imperfect. Write $u$ for the intensity it actually registers, $u$ for
the input arriving at the agent, as in control theory. **This is an observation:
it is the same kind of thing Week&nbsp;1 wrote $o$ for, and everything said there
about observations applies to it.** The change of letter is a concession to the
predictive coding literature, which writes $u$ without exception, and a reader
going on to Bogacz or Buckley will meet it there. Nothing else changes: $u$ is
what the world delivers, and it is all the animal has.

Take the sensory error to be Gaussian about the true intensity:

$$
P(u \mid d) \;=\; \mathcal{N}\big(u;\ g(d),\ \Sigma_u\big),
\qquad
\mathcal{N}(x; m, \Sigma) = \frac{1}{\sqrt{2\pi\Sigma}}\exp\!\left(-\frac{(x-m)^2}{2\Sigma}\right).
$$ {#likelihood}

Here $m$ and $\Sigma$ are placeholders naming the two slots of a Gaussian, its
mean and its variance, and they take different fillings each time it is used.
$\Sigma_u$ is the variance of the sensory noise, in squared units of intensity:
small means a reliable ear.

::: mn Not $\mu$ for the mean
The obvious letter for a Gaussian's mean is $\mu$, and it is not used here
because Week&nbsp;1 already spent it. There $\mu$ is the agent's **internal
states**, the physical variables carrying the model, and it acquires dynamics of
its own from Week&nbsp;5. A letter cannot mean two things in one course, so the
generic mean gets $m$.
:::

The animal is not starting from nothing. Sources it cares about tend to be at a
characteristic sort of range, and that expectation is its prior:

$$
P(d) \;=\; \mathcal{N}\big(d;\ d_p,\ \Sigma_p\big),
$$ {#prior}

with $d_p$ the distance it expects, the subscript standing for *prior*, and
$\Sigma_p$ how firmly it expects it. $\Sigma_p$ is again a variance, this time in
squared units of distance, and a small one is a confident prior.

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

## From sums to densities

Week&nbsp;1 worked entirely with finite sets. There were three states, the prior
was three numbers adding to one, and $P(o)$ was a sum with one term per state.
Here $d$ ranges over an interval, and every one of those has to be replaced. The
replacement is routine, but it changes what the symbols *are*, and the change is
easy to slide past.

::: definition Probability density
For a continuous variable, $P(d)$ is a **probability density**, not a
probability. The probability lives in $P(d)\,\mathrm{d}d$, the density times a
width, and only an integral over a range of $d$ returns a number meaning "how
likely". Normalisation reads $\int P(d)\,\mathrm{d}d = 1$ in place of
$\sum_s P(s) = 1$.
:::

Three consequences, and the third bites much later.

**A density can exceed one.** Nothing forbids it, because it is not a
probability. In this very problem the likelihood, read as a function of $u$,
peaks at $1/\sqrt{2\pi\Sigma_u} = {{lik_peak_density:.4f}}$. A probability of
${{lik_peak_density:.4f}}$ would be nonsense; a density of it is ordinary. The
prior, being wider, peaks at only ${{prior_peak_density:.4f}}$, so height alone
tells you nothing about which is the more confident claim.

**Only an area is a probability.** The posterior below peaks at a density of
${{post_peak_density:.4f}}$. Asking how probable the source is at exactly
$d = {{post_mode:.4f}}$ has no answer but zero. Asking how probable it is between
${{post_band_lo}}$ and ${{post_band_hi}}$ does have one: integrate over that
band and get {{post_band_mass:.4f}}.

**A density carries units.** $P(d)\,\mathrm{d}d$ is dimensionless, so $P(d)$ is
per unit distance. That is why the two prediction errors of [lesson:2.3] cannot
be added until one has been converted into the other's units, and it is the whole
of why the last problem of the week comes out as it does.

## The exact posterior

Bayes' theorem, in the continuous form:

$$
P(d \mid u) \;=\; \frac{P(u \mid d)\,P(d)}{P(u)},
\qquad
P(u) \;=\; \int_0^{\infty} P(u \mid d)\,P(d)\;\mathrm{d}d .
$$ {#continuous-bayes}

The numerator is a product of two things we have written down. The denominator is
an integral over every distance the source might be at.

It is worth being exact about why that is a problem, because the obvious answer
is wrong. The obvious answer is that an integral is a sum with uncountably many
terms, so it must be harder than Week&nbsp;1's finite sum. But
$\int_0^1 x\,\mathrm{d}x$ also has uncountably many terms and takes a second, so
the count is not what decides it. [lesson:1.5|Week&nbsp;1's Lesson&nbsp;5] sets out the four
reasons the denominator resists, and the one that applies here is the third: with
$g$ nonlinear the exponent of the integrand is not a quadratic, so the one closed
form available for integrals of this shape does not apply. There is nothing wrong
with the dimension. One variable is already enough.

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
because [eq:link] is nonlinear. Measured on the grid, the posterior's skewness is
${{post_skew:+.3f}}$, where a Gaussian would give zero.

::: definition Skewness
The **skewness** of a distribution is the third moment of its standardised
deviation from the mean,
$$
\mathrm{skew} \;=\; \mathbb{E}\!\left[\left(\frac{d - \mathbb{E}[d]}{\mathrm{sd}(d)}\right)^{3}\right],
$$
with $\mathrm{sd}$ the standard deviation. Cubing keeps the sign, so deviations
above the mean contribute positively and those below negatively, and dividing by
the standard deviation first makes the answer a pure number, unchanged by
rescaling the axis. Any symmetric distribution gives zero, because every positive
term is cancelled by its mirror image. A positive value means the tail to the
right is the longer one.
:::

The reason is easiest to see in the exponent. A posterior is a product, so its
logarithm is a sum:

$$
\ln P(d \mid u) \;=\; \underbrace{-\frac{(d - d_p)^2}{2\Sigma_p}}_{\text{exactly a parabola}}
\;\underbrace{-\;\frac{\big(u - g(d)\big)^2}{2\Sigma_u}}_{\text{not a parabola}} \;+\; \text{const}.
$$ {#exponent-sum}

A Gaussian is precisely a distribution whose log is a parabola, so the question
is whether the right-hand side is one. The first term is, exactly, by
construction. The second is not, because $g$ is nonlinear: fit the best parabola
to it over ${{loglik_fit_lo:.2f}} \le d \le {{loglik_fit_hi:.2f}}$, the range the
posterior mostly occupies, and it is still off by {{loglik_parabola_resid:.3f}} in
log units. A parabola plus a non-parabola is a non-parabola, so the posterior is
not Gaussian, and the asymmetry that a Gaussian cannot have survives.

That also says when the skew would go away: when the parabola is sharp enough to
swamp the other term's departure from one. Narrow the prior to
$\Sigma_p = {{var_prior_narrow}}$ and the skew falls to
{{skew_narrow_prior:.4f}}, which is zero to the precision of the grid. Widen it
and the skew grows. The widget in [lesson:2.4] has a slider for this.

::: warning The likelihood alone is not a distribution over $d$
It is tempting to read the middle curve in the figure as "what the data say about
the distance", and to ask for its mean. It has none. As $d$ grows, $g(d) \to 0$,
so the sensory error tends to the fixed value $u$ and the likelihood flattens onto
$\exp(-u^2/2\Sigma_u) = {{lik_limit_far:.4f}}$ rather than decaying. Its integral
over $(0, \infty)$ therefore diverges: it is not normalisable, and any mean, variance
or skew you compute for it is a property of where you cut the axis off.

What this means physically is that hearing a quiet sound is compatible with the
source being at any large distance whatever, and the data alone never rule that
out. It is the prior that does, and the posterior is normalisable only because
the prior's tail decays. The prior is not decoration here; without it there is no
answer at all. This is the general situation: a nonlinear link between what you
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
normalising, which is **quadrature**: approximating an integral by a weighted sum
over sample points. Later lessons refer back to "the mode computed by
quadrature", and this is the computation they mean. That is available here because $d$ is one number. With ten unknowns
the same grid would need $10^{60}$ points, and the grid is the *easy* method.

::: keyidea
Two obstacles, and they are different from each other. The **normaliser**, which
is Week&nbsp;1's evidence $P(o)$ under a name that describes its job here rather
than its meaning, is an
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

The general statement is worth deriving once, since the week uses it twice and
the second use carries the sharpest result in it.

::: derivation Changing variables in a density
Let $D$ be a random variable with density $p_D$, and let $\rho = r(D)$ for a
smooth, strictly increasing $r$. What is the density of $\rho$?

Densities do not transform simply, but *cumulative* probabilities do, because
they are probabilities of events and an event does not care what the axis is
called. Since $r$ is increasing, $\rho \le y$ happens exactly when
$D \le r^{-1}(y)$. Same event, same probability:

$$
\Phi_\rho(y) \;=\; \Pr(\rho \le y) \;=\; \Pr\big(D \le r^{-1}(y)\big) \;=\; \Phi_D\big(r^{-1}(y)\big).
$$

Differentiate both sides with respect to $y$. Writing $\Phi$ for a cumulative
distribution function, the left-hand side differentiates to $\Phi_\rho' = p_\rho$
by the definition of a density as the derivative of a cumulative. On the right, the
chain rule gives $p_D$ evaluated at $r^{-1}(y)$, times the derivative of
$r^{-1}$:

$$
p_\rho(y) \;=\; p_D\big(r^{-1}(y)\big)\,\frac{\mathrm{d}r^{-1}}{\mathrm{d}y} .
$$

Writing $d = r^{-1}(y)$ for the corresponding value of the original variable,

$$
p_\rho \;=\; p_D(d)\,\left|\frac{\mathrm{d}d}{\mathrm{d}\rho}\right| ,
$$ {#jacobian}

the modulus covering the decreasing case, where the matching events are
$\rho \le y$ and $D \ge r^{-1}(y)$, and the derivative comes out negative while a
density cannot.
:::

The factor $|\mathrm{d}d/\mathrm{d}\rho|$ is the **Jacobian** of the change of
variables. Read it as bookkeeping: it says how much of the old axis is packed
into a unit of the new one, and a fixed amount of probability spread over more
axis has to be thinner. Where the map stretches, the density drops.

That is what is happening to the likelihood here, with the two variables the
other way round. The noise is Gaussian per unit of intensity; reading it per unit
of distance multiplies by $|\mathrm{d}u/\mathrm{d}d| = |g'(d)|$, which is small far
away. So the density spreads out there, and the tail is long to the right.

[eq:jacobian] returns in the problems, under the heading "The mode moves when you
rename the axis", doing something less comfortable than explaining a lean.
:::

::: checkpoint
- Both prior and likelihood are Gaussian. Why is the posterior not?
- Which is larger here, the posterior mean or the posterior mode, and what
  feature of the distribution decides it?
- Name the two separate difficulties this lesson identifies, and say which of
  them Week&nbsp;1 had already met.
:::
