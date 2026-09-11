---
title: Inferring distance from a sound
deck: A nonlinear observation model, a numerical reference posterior, and the distinction between likelihoods and densities.
week: 2
time: 30 min
---

We will infer a sound source's distance from a noisy intensity measurement. The model has one hidden variable, so we can obtain a numerical reference posterior and compare point estimates with the distribution they summarise.

## The setup

An animal hears a sound. Write $d>0$ for the source distance in its model, with
$d$ chosen for distance. It is a hidden state in the role of $s$ from Week 1.
We will also use $d$ as the argument at which a density or objective is evaluated.
An estimate, such as the posterior mode, is a particular value of that argument.

What the animal can measure is intensity, and intensity falls off as the inverse
square of distance. Write $g$ for that link, $g$ for the **observation
function**, the map from a hidden state to the observation it would produce:

$$
g(d) \;=\; \frac{1}{d^{2}} .
$$ {#link}

The source's power has been taken as one, which fixes the units so that intensity
is $1$ at unit distance. Nothing turns on that choice; it saves carrying a
constant through every line.

The function $g$ predicts the noise-free intensity. It is not the likelihood.
To specify the likelihood we also need a distribution for the measurement noise.
For example, the same inverse-square function with Gaussian noise and with a
heavy-tailed noise distribution gives different likelihoods.

Write $u$ for the intensity registered by the animal, using the notation of
Bogacz's predictive-coding example. It plays the same role as Week 1's observation
$o$. The observation is held fixed while we compare possible distances.

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

We use $m$ for a generic Gaussian mean. The symbol $\mu$ denotes physical
internal states in the blanket partition, rather than a generic mean.

The animal is not starting from nothing. Sources it cares about tend to be at a
characteristic sort of range, and that expectation is its prior:

$$
P(d)=\frac{\mathcal N(d;d_p,\Sigma_p)}{Z_p}\,\mathbf1_{d>0},
\qquad Z_p=\int_0^\infty\mathcal N(x;d_p,\Sigma_p)\,\mathrm dx.
$$ {#prior}

This is a **truncated Gaussian prior**, restricted and renormalised to positive
distances. The indicator $\mathbf1_{d>0}$ equals one on the allowed domain and
zero elsewhere. The location $d_p$ and variance parameter $\Sigma_p$ belong to
the underlying Gaussian; after truncation they are not exactly the prior mean
and variance. We will call $d_p$ the prior location.

For fixed prior parameters, $Z_p$ is a constant in $d$. It cancels when the
posterior is normalised, and its derivative with respect to $d$ is zero. It must
be retained when computing evidence or learning prior parameters. Here
$Z_p={{prior_positive_mass:.6f}}$. The plot scales curves to their own peaks,
so this constant does not affect their displayed shapes.

Concretely: $d_p = {{d_prior}}$, $\Sigma_p = {{var_prior}}$, and the animal hears
$u = {{u_obs}}$ with $\Sigma_u = {{var_obs}}$.

Before computing anything, notice the tension. The prior expects a source at
$d = {{d_prior}}$, which by [eq:link] would produce an intensity of
${{g_at_prior:.2f}}$. The animal heard ${{u_obs}}$, twice that. Taken alone, an
intensity of ${{u_obs}}$ means $d = 1/\sqrt{ {{u_obs}} } = {{d_from_data_alone:.4f}}$.
So the prior says one thing, the data another, and the posterior must account for both. Its mode and mean need not coincide, so we should specify which summary is being requested.

## From sums to densities

The discrete examples of Week&nbsp;1 used finite sets. There were three states, the prior
was three numbers adding to one, and $P(o)$ was a sum with one term per state.
Here $d$ ranges over an interval, and every one of those has to be replaced. The
replacement is routine, but it changes what the symbols *are*, and the change is
easy to slide past.

::: definition Probability density
For a continuous variable, $P(d)$ is a **probability density**, not a
probability. For a small interval of width $\Delta d$ around $d$, its probability is approximately $P(d)\Delta d$; the exact probability is the integral over that interval. Normalisation reads $\int P(d)\,\mathrm{d}d = 1$ in place of
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
be added until one has been converted into the other's units, and it is relevant to the change-of-variables exercise later in the week.

## A numerical reference posterior

Bayes' theorem, in the continuous form:

$$
P(d \mid u) \;=\; \frac{P(u \mid d)\,P(d)}{P(u)},
\qquad
P(u) \;=\; \int_0^{\infty} P(u \mid d)\,P(d)\;\mathrm{d}d .
$$ {#continuous-bayes}

The numerator is a product of two things we have written down. The denominator is
an integral over every distance the source might be at.

The inverse-square likelihood does not combine with this prior to give a
quadratic log posterior. Completing the square therefore does not evaluate the
integral. This is a limitation of that analytic technique, not evidence that the
one-dimensional integral is computationally infeasible. [lesson:1.5] separates
those questions.

We evaluate it on a grid. Halving the spacing and extending the upper limit gives
a mean change of ${{post_mean_refinement:sci2}}$ and a standard-deviation change
of ${{post_sd_refinement:sci2}}$. These checks support the displayed precision;
they are empirical convergence checks, not rigorous error bounds.
The reference remains a numerical approximation to the exact posterior.

::: fig A numerical approximation to the posterior over distance, against the prior and the likelihood that produced it. Each curve is scaled to its own peak, so the figure compares shapes rather than heights. The likelihood is plotted as a function of $d$, which is what makes it lean.
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

**It is not Gaussian.** The sensory noise is Gaussian and the prior is a truncated Gaussian, but the posterior is not even a truncated Gaussian,
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
rescaling the axis. A symmetric distribution with a finite third absolute moment gives zero, because every positive
term is cancelled by its mirror image. A positive value means the positive cubed deviations outweigh the negative ones; it is not a complete description of tail shape.
:::

The reason is easiest to see in the exponent. A posterior is a product, so its
logarithm is a sum:

$$
\ln P(d \mid u) \;=\; \underbrace{-\frac{(d - d_p)^2}{2\Sigma_p}}_{\text{exactly a parabola}}
\;\underbrace{-\;\frac{\big(u - g(d)\big)^2}{2\Sigma_u}}_{\text{not a parabola}} \;+\; \text{const}.
$$ {#exponent-sum}

On the whole real line, a Gaussian density has a quadratic log density. A truncated Gaussian retains that quadratic shape inside its support, so the question
is whether the right-hand side is one. On the positive domain the first term is quadratic. For this inverse-square $g$, the second is not: fit the best parabola
to it over ${{loglik_fit_lo:.2f}} \le d \le {{loglik_fit_hi:.2f}}$, the range the
posterior mostly occupies, and it is still off by {{loglik_parabola_resid:.3f}} in
log units. A parabola plus a non-parabola is a non-parabola, so the posterior is
not Gaussian, and the asymmetry that a Gaussian cannot have survives.

That also says when the skew would go away: when the parabola is sharp enough to
swamp the other term's departure from one. Narrow the prior to
$\Sigma_p = {{var_prior_narrow}}$ and the skew falls to
{{skew_narrow_prior:.4f}}, which is small but need not vanish exactly. Widen it
and the skew grows. The widget in [lesson:2.4] has a slider for this.

::: warning The likelihood alone is not a distribution over $d$
It is tempting to read the middle curve in the figure as "what the data say about
the distance", and to ask for its mean. It has none. As $d$ grows, $g(d) \to 0$,
so the sensory error tends to the fixed value $u$ and the likelihood, with its constant Gaussian prefactor omitted, flattens onto
$\exp(-u^2/2\Sigma_u) = {{lik_limit_far:.4f}}$ rather than decaying. Its integral
over $(0, \infty)$ therefore diverges: it is not normalisable, and any mean, variance
or skew you compute for it is a property of where you cut the axis off.

What this means physically is that hearing a quiet sound is compatible with the
source being at any large distance whatever, and the data alone never rule that
out. It is the prior that does, and the posterior is normalisable only because
the prior's tail decays. In this model an improper uniform prior on the positive half-line would not
produce a normalisable posterior. A proper prior supplies tail decay. This is a
property of the chosen likelihood, not a theorem about every nonlinear model.
:::



**Its mean and its mode are different, and not slightly.** The peak sits at
$d = {{post_mode:.3f}}$; the mean sits at $d = {{post_mean:.3f}}$, higher by
{{mean_mode_pct:.0f}}%. They summarise different aspects of the distribution. Choosing between them requires specifying what the estimate will be used for. The long right tail, which is the possibility that the source is
distant and the ear happened to over-report, drags the mean out without moving
the peak. Remember this: the method built over the next two lessons finds the
mode, and this choice does not by itself specify an action.

**It required an integral we cannot do in general.** The curve above was computed
by evaluating [eq:continuous-bayes] at six hundred thousand grid points and
normalising, which is quadrature: approximating an integral by a weighted sum
over sample points. Later lessons refer back to "the mode computed by
quadrature", and this is the computation they mean. That is available here because $d$ is one number. With $m$ grid points per coordinate and $n$ coordinates, a tensor grid would require $m^n$ points. A convergence check in one dimension does not establish feasibility in many dimensions.

The model raises two separate practical questions: how to evaluate posterior
quantities accurately, and how to represent the result compactly. A point estimate
answers a narrower question than a distribution. A Gaussian or a collection of
samples retains other information, at additional computational cost.

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

This change in the shape of a likelihood is distinct from transforming a random variable with a density. We derive the latter operation separately because it is needed in the reparameterisation exercise.

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

Do not multiply the likelihood $P(u\mid d)$ by $|g'(d)|$ when plotting it
against candidate distances. The observation $u$ is fixed and we are evaluating
the same conditional density for different hypotheses. We are not transforming a
random observation into a random distance. Its asymmetric shape comes from
substituting $g(d)$ into the residual $u-g(d)$.

The Jacobian does apply when changing a prior or posterior density from distance
$d$ to log-distance $\rho=\ln d$. Then
$p_\rho(\rho)=p_d(e^\rho)e^\rho$. In contrast the likelihood in that
parameterisation is $P(u\mid e^\rho)$, without an extra factor.

[lesson:2.5] uses [eq:jacobian] to compare modes in distance and log-distance.
:::

::: checkpoint
- Why do Gaussian sensory noise and a truncated Gaussian prior fail to give a truncated Gaussian posterior for this observation map?
- Which is larger here, the posterior mean or the posterior mode, and what
  feature of the distribution decides it?
- Name the two separate difficulties this lesson identifies, and say which of
  them Week&nbsp;1 had already met.
:::
