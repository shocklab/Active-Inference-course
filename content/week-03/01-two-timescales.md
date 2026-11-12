---
title: Two timescales
deck: Learning the parameters of the model by climbing the same log joint as inference, slowly and across many trials.
week: 3
time: 45 min
scripts: [w03.js]
---

Week&nbsp;2 inferred a distance from one sound. Four numbers were fixed throughout:
the prior location $d_p = {{d_prior}}$, the prior variance $\Sigma_p = {{var_prior}}$,
the sensory variance $\Sigma_u = {{var_obs}}$ and, less visibly, the power of the
source, which [lesson:2.2] took as one "so that intensity is $1$ at unit distance".
An animal is not given any of them. It hears many sounds over its life, and those
sounds carry information about the four numbers as well as about each distance.
This week the animal uses that information.

The method is the one Week&nbsp;2 used for the distance: gradient ascent on the log
joint. Only the argument changes.

## States and parameters

::: definition Parameter
A **parameter** of a generative model is a quantity shared by every observation
the model is used for. A hidden state belongs to one observation; a parameter
belongs to all of them.
:::

The distance $d$ is a state: each sound comes from a source at its own distance.
The prior location, the two variances and the source power are parameters: the
animal uses the same values for every sound it hears. The variances and the source
power are positive; the prior location may be any real number.

Write $\theta$ for the source power, $\theta$ being the conventional letter for a
parameter. The observation function of [lesson:2.2] becomes

$$
g(d, \theta) \;=\; \frac{\theta}{d^{2}} ,
$$ {#link-theta}

and Week&nbsp;2's model is the case $\theta = {{theta0}}$. [lesson:1.2] used
$\theta$ locally for a sensor probability in the lamp example; from this week it is
the source power. When a statement holds for any one of the four parameters
$d_p, \Sigma_p, \Sigma_u, \theta$, we write $\omega$ for whichever is meant.

## The world and the model

[lesson:1.2] separated the generative process, which produces the sounds, from
the generative model, which the animal uses to interpret them. Learning only makes
sense with both in view, because a learning rule is judged by where it takes the
model's parameters given what the process does.

::: definition Trial
A **trial** is one draw from the generative process followed by one round of
inference. On trial $n$, for $n = 1, \dots, N$, the process places a source at
distance $d_n$, drawn independently from a Gaussian with location $d_p^{*}$ and
variance $\Sigma_p^{*}$ truncated to $d_n > 0$, and the animal hears

$$
u_n \;=\; \frac{\theta^{*}}{d_n^{2}} \;+\; z_n ,
$$

where $z_n \sim \mathcal N(0, \Sigma_u^{*})$. The noises $z_1, \dots, z_N$ are independent
of one another and of every source distance, and the distances are independent of
one another.
:::

The star marks a quantity belonging to the process, following the convention of
$P^{*}$ in [lesson:1.2]. The animal never sees a starred quantity. Its model has the
same form with its own unstarred parameters, and when the model's parameters equal
the starred ones the model is correct. Each experiment this week states its own
process and the model the animal starts from, and the animal learns one parameter
at a time while the other three stay where they started.

Two kinds of average appear from here on, and they are kept apart by their
subscripts, as the notation page requires. $\mathbb E_{P^*}$ averages over trials of
the process: over the source distance and the noise, and so over the observation.
$\mathbb E_{d\mid u}$ averages over the model's posterior on the distance, for one
observation $u$ and the parameters the animal holds at the time.

## The log joint with its constants restored

[lesson:2.3] wrote the log joint for one trial as

$$
\ell \;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \frac{(u - g(d))^2}{2\Sigma_u} \;+\; \text{const}
$$

and justified dropping the constant because it contained no $d$. The constant does
contain the parameters. Restoring it from [lesson:2.3]'s expansion of the truncated
prior and the Gaussian likelihood:

$$
\ell(d, u; \omega) \;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \tfrac12\ln(2\pi\Sigma_p) \;-\; \ln Z_p
\;-\; \frac{\big(u - \theta/d^{2}\big)^2}{2\Sigma_u} \;-\; \tfrac12\ln(2\pi\Sigma_u) ,
$$ {#ell-full}

for $d > 0$, with $Z_p = \int_0^\infty \mathcal N(x; d_p, \Sigma_p)\,\mathrm dx$ as in
[lesson:2.2]. $Z_p$ depends on $d_p$ and $\Sigma_p$. At Week&nbsp;2's values it is
${{zp_w2:.6f}}$, and at Week&nbsp;2's observation $u = {{u_obs}}$ and its mode
$\hat d = {{mode_w2:.4f}}$ the full log joint is ${{ell_hat_w2:.4f}}$.

Over $N$ trials the joint density of all the states and observations is a product,
because the trials are independent given the parameters, so its logarithm is a sum:

$$
\mathcal L \;=\; \sum_{n=1}^{N} \ell(d_n, u_n; \omega) .
$$ {#total}

Here $\mathcal L$ is a function of the $N$ distances and the four parameters. The
script letter keeps it apart from the expected losses of [lesson:2.5].

## One objective, two kinds of argument

::: derivation The gradient of the total log joint
Differentiate [eq:total] with respect to the distance of one trial, $d_m$. Only the
term $n = m$ contains $d_m$, so every other term differentiates to zero:

$$
\frac{\partial \mathcal L}{\partial d_m} \;=\; \frac{\partial \ell(d_m, u_m; \omega)}{\partial d_m}
\;=\; \varepsilon_{p,m} \;+\; g'(d_m)\,\varepsilon_{u,m} ,
$$

which is Week&nbsp;2's gradient for trial $m$ alone, with the prediction errors of
[lesson:2.3] evaluated on that trial. Inference on one trial needs nothing from any
other trial.

Differentiate instead with respect to a parameter $\omega$. Every term contains
$\omega$, so none drops:

$$
\frac{\partial \mathcal L}{\partial \omega} \;=\; \sum_{n=1}^{N} \frac{\partial \ell(d_n, u_n; \omega)}{\partial \omega} .
$$ {#param-sum}

The gradient for a parameter is a sum over every trial the parameter is used in.
:::

The two kinds of argument are treated differently because of this difference, not
because they are different kinds of object. A state's gradient is available within
its trial. A parameter's gradient is spread across trials, and an animal that hears
one sound at a time never holds the whole sum.

## Settling first

Suppose inference on trial $n$ is run until it stops, at the mode $\hat d_n$. The
log joint at that point is a function of the observation and the parameters alone.

::: definition Settled log joint
The **settled log joint** of one trial is

$$
\hat\ell(u; \omega) \;=\; \ell\big(\hat d(u; \omega), u; \omega\big) ,
$$

where $\hat d(u; \omega)$ is the mode of [lesson:2.3] for observation $u$ and
parameters $\omega$.
:::

The mode moves when a parameter moves, so $\hat\ell$ depends on $\omega$ in two
ways: directly, and through $\hat d$. Its derivative nevertheless needs only the
direct one.

::: derivation The envelope theorem
Assume the mode is a strict local maximum, $\ell'' < 0$ there, so that by the
implicit function theorem $\hat d$ is a differentiable function of $\omega$ near the
current value. By the chain rule,

$$
\frac{\mathrm d\hat\ell}{\mathrm d\omega}
\;=\; \frac{\partial \ell}{\partial \omega}\bigg|_{d = \hat d}
\;+\; \frac{\partial \ell}{\partial d}\bigg|_{d = \hat d}\,\frac{\mathrm d\hat d}{\mathrm d\omega} .
$$

The first term is the change in $\ell$ from moving $\omega$ with the state held
fixed. The second is the change from the state moving in response. The mode is a
stationary point, so $\partial\ell/\partial d = 0$ at $d = \hat d$, and the second
term vanishes whatever $\mathrm d\hat d/\mathrm d\omega$ is:

$$
\frac{\mathrm d\hat\ell}{\mathrm d\omega} \;=\; \frac{\partial \ell}{\partial \omega}\bigg|_{d = \hat d} .
$$ {#envelope}
:::

So the slope of the settled log joint can be read off without knowing how the mode
would move. The animal evaluates the partial derivative at the state it has already
inferred.

The mode does move, and the same implicit-function argument gives how far.

::: derivation How the mode moves with the prior location
Write $\ell_d$ for $\partial\ell/\partial d$. The mode satisfies
$\ell_d\big(\hat d(\omega), \omega\big) = 0$ for every $\omega$ near the current value.
Differentiating this identity in $\omega$ by the chain rule,

$$
\ell''\,\frac{\mathrm d\hat d}{\mathrm d\omega} \;+\; \frac{\partial \ell_d}{\partial\omega} \;=\; 0 ,
\qquad\text{so}\qquad
\frac{\mathrm d\hat d}{\mathrm d\omega} \;=\; \frac{\partial \ell_d/\partial\omega}{-\ell''} ,
$$

with both derivatives evaluated at $d = \hat d$. For the prior location, the only term
of $\ell_d$ that contains $d_p$ is $(d_p - d)/\Sigma_p$; the normaliser $\ln Z_p$
depends on $d_p$ but not on $d$, so it drops out of $\ell_d$. Hence
$\partial\ell_d/\partial d_p = 1/\Sigma_p$ and

$$
\frac{\partial\hat d}{\partial d_p} \;=\; \frac{1/\Sigma_p}{-\ell''} \;>\; 0 ,
$$ {#mode-shift}

positive because $-\ell'' > 0$ at a strict maximum.
:::

Moving the prior location outward moves the mode outward. At Week&nbsp;2's mode
$\ell'' = {{ell_dd_w2:.4f}}$, so [eq:mode-shift] is ${{dhat_ddp_w2:.4f}}$, which agrees
with a finite-difference derivative of the mode to six decimal places.

Two cautions about "the mode". Week&nbsp;2's posterior can have more than one local
maximum, and the envelope theorem holds at whichever one inference stopped on;
$\hat\ell$ is the log joint on that branch. And a climb with finite steps can reach
a different local maximum than the continuous ascent would, as [lesson:2.3] showed.
The simulations this week run Week&nbsp;2's capped ascent until a step falls below
$10^{-12}$, giving up after 2000 steps. The climb starts at the prior location when
that is positive, and otherwise at the mean of the truncated prior, which is positive
for every real $d_p$. In the
{{n_runs_checked}} runs whose numbers the lessons quote, every trial met that
stopping rule, and every settled state agreed with the highest mode of its trial,
found by a separate grid search, to within ${{mode_gap_max:sci1}}$. That is a check of
those runs, not a guarantee for every slider setting; the widgets add a readout row
if a trial runs out of steps. A trial that runs out of steps changes no parameter
whose update is read at the mode, in the widgets, the notebooks and `values.py`
alike, so that an unfinished state never feeds a learning rule. The
posterior-averaged rule of [lesson:3.4] does not use the mode, and it updates on
every trial.

## The log joint in place of the evidence

The number that says how well a set of parameters accounts for a sound is the
evidence, $p(u; \omega) = \int_0^\infty p(d, u; \omega)\,\mathrm dd$, the denominator of
Bayes' rule that [lesson:1.5] examined. A statistician fitting the four parameters to
recorded sounds would maximise the sum of $\ln p(u_n; \omega)$ over trials. Its
derivative needs that integral on every trial, or, as [lesson:3.4] shows, an average
over the whole posterior, which is the same integral in another form. Week&nbsp;2
avoided the integral for the state by climbing the log joint instead, and the animal
avoids it for the parameters the same way: $\hat\ell$ needs the two factors of the
joint, evaluated at one point it has already found. The price is that $\hat\ell$ is
not $\ln p(u; \omega)$, so climbing one need not climb the other. [lesson:3.4]
computes the difference, and Week&nbsp;4 replaces $\hat\ell$ by an objective that
includes the missing term.

This is a statement about exact stationary points, so check it. At Week&nbsp;2's
observation, a finite-difference derivative of $\hat\ell$ with respect to
$\Sigma_u$, with the mode recomputed at each perturbed value, gives
${{env_total:.6f}}$. The partial derivative at the mode, which [lesson:3.2] derives,
gives ${{env_partial:.6f}}$.

Now stop inference early. After ${{k_short}}$ of Week&nbsp;2's steps from the prior
location the estimate is at $d = {{d_short:.4f}}$, where
$\partial\ell/\partial d = {{short_dl_dd:.4f}}$, not zero. The total derivative of
$\ell$ at that point, with the three steps rerun at each perturbed $\Sigma_u$, is
${{short_total:.4f}}$, while the partial derivative is ${{short_partial:.4f}}$. The
second term of the chain rule has not vanished. The partial derivative is still the
slope of $\ell$ with the state held where the three steps left it, which is a
direction of coordinate ascent, but it is not the slope of the objective the three
steps define, $\ell$ evaluated wherever three steps from the prior location land.

## The slow update

::: definition Learning rate
On each trial the animal first settles its state, then changes each parameter by a
small step along the partial derivative at the settled state:

$$
\omega \;\leftarrow\; \omega \;+\; \beta\,\frac{\partial \ell(d, u_n; \omega)}{\partial \omega}\bigg|_{d = \hat d_n} .
$$ {#slow}

The **learning rate** $\beta > 0$ plays the role for parameters that $\eta$ plays for
the state. Each parameter has its own, with units that make the step carry the
parameter's units. [lesson:2.1] used $\beta$ for the step size of a weight update,
and the meaning is the same.
:::

By [eq:envelope], each step of [eq:slow] is a step up the settled log joint of the
current trial. By [eq:param-sum], the direction that would increase the total is the
sum of these per-trial directions. Taking one small step per trial, rather than
summing first, replaces the sum by a running accumulation: after many trials with a
small $\beta$, the parameter has moved roughly as it would under ascent on the
average of $\hat\ell$ over the trials it has seen, with fluctuations from the trials
it happened to see last. The next section makes this exact in a case that can be
solved.

## A running average

Take [lesson:2.4]'s linear comparison: $g(d) = d$, a Gaussian prior on the whole
line, and the prior location as the only parameter being learned. [lesson:2.4]
derived the mode,

$$
\hat d_n \;=\; \frac{\Pi_p d_p + \Pi_u u_n}{\Pi_p + \Pi_u} .
$$

::: derivation The slow update for the prior location is an average of observations
On the whole line there is no truncation, so $Z_p = 1$ for every $d_p$ and
[eq:ell-full] gives $\partial\ell/\partial d_p = (d - d_p)/\Sigma_p$. At the mode,

$$
\hat d_n - d_p \;=\; \frac{\Pi_p d_p + \Pi_u u_n - (\Pi_p + \Pi_u)\,d_p}{\Pi_p + \Pi_u}
\;=\; \frac{\Pi_u}{\Pi_p + \Pi_u}\,(u_n - d_p) .
$$

Multiply numerator and denominator by $\Sigma_p\Sigma_u$, using
$\Pi_u\Sigma_p\Sigma_u = \Sigma_p$ and $(\Pi_p + \Pi_u)\Sigma_p\Sigma_u = \Sigma_u + \Sigma_p$:

$$
\hat d_n - d_p \;=\; \frac{\Sigma_p}{\Sigma_p + \Sigma_u}\,(u_n - d_p) .
$$

Substituting into [eq:slow],

$$
d_p \;\leftarrow\; d_p + \frac{\beta}{\Sigma_p}\cdot\frac{\Sigma_p}{\Sigma_p + \Sigma_u}\,(u_n - d_p)
\;=\; (1 - \kappa)\,d_p + \kappa\,u_n,
\qquad
\kappa \;=\; \frac{\beta}{\Sigma_p + \Sigma_u} .
$$ {#ema}

Applying [eq:ema] once per trial unwinds it. Write $d_p^{(n)}$ for the location after
trial $n$, with $d_p^{(0)}$ the starting value. The first two trials give

$$
d_p^{(1)} \;=\; (1-\kappa)\,d_p^{(0)} + \kappa u_1 ,
\qquad
d_p^{(2)} \;=\; (1-\kappa)^{2}\,d_p^{(0)} + \kappa(1-\kappa)\,u_1 + \kappa u_2 ,
$$

and substituting the expression for $d_p^{(n-1)}$ into [eq:ema] each time gives, by
induction,

$$
d_p^{(n)} \;=\; (1-\kappa)^{n}\,d_p^{(0)} \;+\; \kappa\sum_{k=0}^{n-1}(1-\kappa)^{k}\,u_{n-k} .
$$ {#ema-unwound}

For $\kappa \neq 0$ the geometric series gives $\kappa\sum_{k=0}^{n-1}(1-\kappa)^{k} = 1 - (1-\kappa)^{n}$,
so the weights on the observations and the weight $(1-\kappa)^{n}$ on the starting
value add to one.
:::

The factor $1/\Sigma_p$ in the update cancelled against the $\Sigma_p$ in the
shrinkage factor, though $\Sigma_p$ is still in $\kappa$. For $0 < \kappa < 1$ the
prior location is an exponentially weighted average of the raw observations: by
[eq:ema-unwound], trial $n - k$ carries weight $\kappa(1-\kappa)^{k}$, so older trials
fade geometrically, and the starting value's weight $(1-\kappa)^{n}$ goes to zero.
(At $\kappa = 1$ the memory is the last trial alone, and for $\kappa > 1$ the weights
alternate in sign.) The inferred states dropped out of the final expression, but
only because the mode is linear in $u_n$ here; in the nonlinear model they do not.

The average of [eq:ema] over the process is
$\mathbb E_{P^*}[d_p'] = (1-\kappa)\,d_p + \kappa\,\mathbb E_{P^*}[u]$, and in this
linear world $\mathbb E_{P^*}[u]$ is the world's location $d_p^{*}$, because the noise
has mean zero. The expected step vanishes at $d_p = d_p^{*}$ and nowhere else: in the
linear world the location is learned without bias. An exercise asks for the
proof in full, and [lesson:3.4] shows why the nonlinear world is different.

The letter $\kappa$ is chosen deliberately. [lesson:1.1]'s regulator removed a
fraction $\kappa$ of the displacement on each step, and [eq:ema] removes a fraction
$\kappa$ of the gap between $d_p$ and the newest observation. The same algebra
gives the same answers.

::: derivation The spread of a learned parameter
Suppose the observations have mean $\bar u$ and variance $\sigma_u^2$, write
$x_n = d_p - \bar u$ for the deviation after trial $n$, and subtract $\bar u$ from both
sides of [eq:ema]:

$$
x_{n+1} \;=\; (1 - \kappa)\,x_n + \kappa\,(u_n - \bar u) .
$$

The newest observation is independent of $x_n$, which was built from earlier ones.
Taking variances, the cross term vanishes:

$$
\operatorname{Var}(x_{n+1}) \;=\; (1-\kappa)^2\operatorname{Var}(x_n) + \kappa^2\sigma_u^2 .
$$

For $0 < \kappa < 2$ the factor $(1-\kappa)^2$ is below one, the variances settle,
and at stationarity the two are equal; call them $V_x$. Then
$V_x\big[1 - (1-\kappa)^2\big] = \kappa^2\sigma_u^2$, and
$1 - (1-\kappa)^2 = \kappa(2 - \kappa)$, so

$$
V_x \;=\; \frac{\kappa}{2 - \kappa}\,\sigma_u^2 .
$$ {#ema-var}
:::

Here $\sigma_u^2$ is the variance of the observations as the world produces them,
the sum of the spread of sources and the sensory noise.
[lesson:1.1] found the same expression for its regulated walk, with $\kappa^2\sigma_u^2$
in the role of the noise.

The learning rate therefore trades speed against steadiness. The half-life of an
old trial's influence is $\ln 2/[-\ln(1-\kappa)]$ trials, and the standard deviation
of $d_p$ about its target is $\sqrt{\kappa/(2-\kappa)}$ times that of the
observations. With Week&nbsp;2's variances in this linear world and $\beta = {{beta_lin}}$:
$\kappa = {{kappa_lin:.5f}}$, a half-life of {{halflife_lin:.1f}} trials, and a
standard deviation of ${{sd_dp_lin:.4f}}$ for $d_p$. A simulation of this linear
world with the same rate gives ${{sd_dp_lin_sim:.4f}}$. For small $\kappa$, halving
$\beta$ about doubles the memory and reduces the fluctuation by a factor of about
$\sqrt 2$.

## Resting points of a learning rule

Every learning rule this week has the form [eq:slow]: a step $\beta F(\omega, u_n)$,
where $F$ is computed from the trial's observation and the current parameter. The
running average is the one case solved exactly. The general case is judged by one
function.

::: definition Drift
The **drift** of a learning rule is its step per unit rate averaged over the process
at a fixed parameter value,

$$
\varphi(\omega) \;=\; \mathbb E_{P^*}\big[F(\omega, u)\big] .
$$

For the rule [eq:slow], $F(\omega, u) = \partial\ell/\partial\omega$ at $d = \hat d(u; \omega)$.
:::

::: derivation A learning rule fluctuates about a zero of its drift
Write the step as its average plus a remainder,
$F(\omega_n, u_n) = \varphi(\omega_n) + \xi_n$, where $\xi_n$ has mean zero over the
process given $\omega_n$, because $u_n$ is a fresh draw independent of everything that
set $\omega_n$. Suppose $\varphi(\omega^{\circ}) = 0$, and expand $\varphi$ about
$\omega^{\circ}$ to first order: $\varphi(\omega) \approx \varphi'(\omega^{\circ})(\omega - \omega^{\circ})$.
This is an approximation, valid only while $\omega_n$ stays in a neighbourhood of
$\omega^{\circ}$ where the first-order term dominates, and it treats the variance of
$\xi_n$ as fixed, though in general it depends on $\omega_n$.
With $x_n = \omega_n - \omega^{\circ}$,

$$
x_{n+1} \;=\; x_n + \beta\varphi'(\omega^{\circ})\,x_n + \beta\xi_n
\;=\; (1 - \kappa)\,x_n + \beta\xi_n,
\qquad \kappa \;=\; -\beta\,\varphi'(\omega^{\circ}) .
$$

This is the recursion of [eq:ema], with $\beta\xi_n$ in place of
$\kappa(u_n - \bar u)$. The derivation of [eq:ema-var] used the independence of the
newest term from $x_n$ only to drop the cross term, and a mean of zero given the
history is enough for that. Write $\mathcal H_n$ for everything that happened before
trial $n$; it fixes $x_n$ and every earlier remainder $\xi_m$, $m < n$. Averaging first
over $u_n$ with $\mathcal H_n$ held fixed, then over $\mathcal H_n$ (the tower
property of conditional expectation),

$$
\mathbb E[x_n\xi_n] \;=\; \mathbb E\big[x_n\,\mathbb E(\xi_n \mid \mathcal H_n)\big] \;=\; 0,
\qquad
\mathbb E[\xi_m\xi_n] \;=\; \mathbb E\big[\xi_m\,\mathbb E(\xi_n \mid \mathcal H_n)\big] \;=\; 0
\quad (m < n) .
$$

So the cross term still vanishes, and remainders on different trials are
uncorrelated. Within that approximation, three conclusions follow from the
running average.
If $\varphi'(\omega^{\circ}) < 0$ and $0 < \kappa < 2$, the expected deviation
$\mathbb E_{P^*}[x_n]$ shrinks by a factor $1 - \kappa$ per trial, so $\omega^{\circ}$
attracts. If $\varphi'(\omega^{\circ}) > 0$, then $\kappa < 0$, the factor exceeds one
and $\omega^{\circ}$ repels. And at an attracting zero the deviation has stationary
variance $\beta^2\sigma_\xi^2/[\kappa(2-\kappa)]$ by [eq:ema-var], which is about
$\beta\sigma_\xi^2/(2|\varphi'(\omega^{\circ})|)$ for small $\beta$, with $\sigma_\xi^2$ the
variance of $\xi$ near $\omega^{\circ}$.

The stationary variance is of order $\beta$, so the standard deviation of one value
$\omega_n$ about $\omega^{\circ}$ is of order $\sqrt\beta$. The expansion also dropped
the term $\tfrac12\varphi''(\omega^{\circ})x_n^2$, which moves the stationary mean. Near
an attracting zero, and assuming $x_n$ has a stationary distribution with finite
moments, average $x_{n+1} = x_n + \beta\big[\varphi'x_n + \tfrac12\varphi''x_n^2 + \xi_n\big]$
over it. The left side and the first term on the right have the same average, and
$\xi_n$ averages to zero, so to leading order

$$
0 \;\approx\; \varphi'(\omega^{\circ})\,\mathbb E[x] + \tfrac12\varphi''(\omega^{\circ})\,\mathbb E[x^2],
\qquad
\mathbb E[x] \;\approx\; -\frac{\varphi''(\omega^{\circ})}{2\varphi'(\omega^{\circ})}\operatorname{Var}(x) ,
$$

where replacing $\mathbb E[x^2]$ by $\operatorname{Var}(x)$ neglects
$\mathbb E[x]^2$, which is smaller still. The stationary mean is therefore displaced
from the zero by an amount of order $\beta$.
:::

Three quantities have now appeared, and they scale differently with the rate. One
value of a parameter fluctuates about the zero of the drift with a standard deviation
of order $\sqrt\beta$. The mean of those values, over the stationary distribution,
can sit off the zero by order $\beta$ when the drift is curved. And the average of one
finite run differs from that stationary mean by a sampling error, which depends on how
long the run is and on how strongly successive values are correlated. The running
average has no mean shift: its drift is exactly linear, so its stationary mean is
exactly the zero. It shows the first quantity clearly, since its standard deviation in the linear world is
${{sd_dp_lin:.4f}}$ at $\beta = {{beta_lin}}$, ${{sd_dp_lin_q:.4f}}$ at
$\beta = {{beta_lin_q}}$ and ${{sd_dp_lin_s:.4f}}$ at $\beta = {{beta_lin_s}}$: dividing
the rate by four halves the fluctuation.

So a rule is judged by where its drift vanishes and by the sign of the drift's slope
there. The running average is the case $F = (u - d_p)/(\Sigma_p + \Sigma_u)$, with
$\varphi'= -1/(\Sigma_p + \Sigma_u)$ and $\kappa = \beta/(\Sigma_p + \Sigma_u)$, as before.
When the lessons that follow say a rule is biased, they mean its drift vanishes
somewhere other than the world's value. A run with a fixed rate fluctuates about that
point, and the average of one run is evidence about the zero of the drift, not a
measurement of it.

## Timescale separation

The scheme needs two things. Inference must settle within each trial, so that
[eq:envelope] applies. And a parameter must change little over the many trials its
average is taken across, so that a single loud or quiet sound does not set it. The
first is a condition on the number of inference steps per trial; the second is
$\kappa \ll 1$.

::: warning What the separation of timescales is doing
For a fixed set of recorded observations, settling every state against the current
parameters and then stepping the parameters is alternating ascent on [eq:total]:
the states are profiled out, and

$$
\max_{d_1, \dots, d_N}\mathcal L \;=\; \sum_{n=1}^{N}\hat\ell(u_n; \omega)
$$

when each settled state is the highest mode of its trial. The two speeds do not
change the criterion. They change the path taken through it, and so which local
maximum is reached, as finite steps did for the state in [lesson:2.3].

That criterion has no global maximum. Put every $d_n$ at $d_p$. The prior terms
become $-\tfrac N2\ln(2\pi\Sigma_p) - N\ln Z_p$, and the sensory terms do not involve
$\Sigma_p$. As $\Sigma_p \to 0$ with $d_p > 0$ fixed, $Z_p \to 1$ and
$-\tfrac N2 \ln \Sigma_p \to +\infty$, so $\mathcal L$ grows without bound along a path
that declares every source to be exactly where it was expected and the prior to be
perfectly sharp. When every recorded observation is positive, the same happens with
$\Sigma_u \to 0$ along states that reproduce each one exactly, $\theta/d_n^{2} = u_n$.
A single negative observation cannot be reproduced by any distance, leaves a
nonzero residual, and sends that path to $-\infty$ instead; Gaussian noise does
produce negative observations.

An objective unbounded above can still have local maxima, and the slow dynamics
look for one. What they reach is at best a local maximum of the average of
$\hat\ell$, never a global one. [lesson:3.4] finds where the drift of each
parameter vanishes and what the unboundedness does there.
:::

::: mn Three timescales
[Buckley et al. (2017)](https://doi.org/10.1016/j.jmp.2017.09.004), section 8,
separate three: states fastest, then the parameters of the observation and
dynamics functions, then the precisions slowest. Their parameter dynamics are
written in second-order form, with the accumulated gradient setting the rate of
change; an exercise examines what that does.
:::

## The two timescales running at once

In the widget, the process places sources at location $d_p^{*} = {{w1_loc}}$, further
away than the animal expects. Everything else in the process is Week&nbsp;2's model,
except that the ear is sharper: $\Sigma_u^{*} = \Sigma_u = {{w1_vu}}$ for both process
and model, Week&nbsp;2's sharp setting. The animal learns $d_p$ alone, with
$\beta = {{w1_beta}}$, over {{n_trials}} trials.

::: widget two-timescales | Inference within a trial, learning across trials. Left, the state settling on the trial chosen by the trial slider, starting from the prior location the animal held at that moment. Right, the prior location across all trials, with the chosen trial marked.
:::

On the first trial the state settles in {{w1_first_steps}} steps, at
$d = {{w1_first_end:.4f}}$. Across trials the prior location climbs: it is at
${{w1_after_200:.4f}}$ after 200 trials, and over the second half of the run it
averages ${{w1_mean_late:.4f}}$ with a standard deviation of ${{w1_sd_late:.4f}}$.

It does not reach ${{w1_loc}}$. The drift vanishes at
$d_p = {{w1_stat_point:.4f}}$, ${{w1_stat_pct:.1f}}$ per cent from the world's
location, and that is where the run fluctuates. Set the ear back to Week&nbsp;2's
$\Sigma_u = {{var_obs}}$ and the drift vanishes at ${{w1_stat_point_vu1:.4f}}$,
${{w1_stat_pct_vu1:.1f}}$ per cent from it: the prior location moves down, below
where it started, while the sources have moved further away.

The sharp ear is not sharp about distance at this range. What the ear says about $d$
is set by the slope of the link, and its precision about $d$ near a source at
distance $d$ is $g'(d)^2/\Sigma_u = 4\theta^2/(d^{6}\Sigma_u)$, which falls as $d^{-6}$.
At $d = {{w1_loc}}$ with $\Sigma_u = {{w1_vu}}$ it is ${{w1_ear_prec:.4f}}$, less than the
prior's precision of ${{prec_prior}}$. At Week&nbsp;2's location and ear it is
${{w2_ear_prec:.4f}}$. So the posterior for a distant source is wide whichever ear the
animal has, and wide posteriors are where the mode and the posterior mean part
company. Nothing in this lesson says what that does to learning.
[lesson:3.4] does.

::: exercise One trial or many
Show from [eq:param-sum] that if the animal could store all $N$ settled states and
take a full gradient step on the sum, a learning rate $\beta/N$ would give the same
expected step as $N$ single-trial steps of size $\beta/N$ taken with the parameter held
fixed. Then say what changes when the parameter is updated after each trial instead
of being held fixed.
---solution---
With the parameter held at $\omega$ across the $N$ trials, the $N$ single-trial steps
add up to $(\beta/N)\sum_n \partial\ell_n/\partial\omega$, which is a full step of size
$\beta/N$ on [eq:param-sum]. They are the same sum taken in a different order.

When the parameter is updated after each trial, trial $n$'s derivative is evaluated
at the value left by trials $1$ to $n-1$, and its settled state was inferred with that
value too. The order of the trials now matters, and the result is a different
iteration. For small $\beta$ the parameter changes little over any stretch of
trials, so each derivative is evaluated near a common value and the two schemes
nearly agree; [eq:ema] is the exactly solvable case, in which later observations
carry more weight than earlier ones.
:::

::: checkpoint
- What makes the source power a parameter and the distance a state?
- Why could Week&nbsp;2 drop $\tfrac12\ln(2\pi\Sigma_u)$, and why can this week not?
- State the envelope theorem and the assumption it needs.
- In the linear case, what is the prior location an average of, and what sets the
  length of its memory?
- What is the drift of a learning rule, and what decides whether a zero of it attracts?
:::
