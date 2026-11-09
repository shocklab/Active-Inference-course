---
title: Learning the link
deck: The source power sits inside the prediction, so its update is the sensory error times the input that carried it. What that update drives to zero, and what three parameters learned together cannot tell apart.
week: 3
time: 45 min
scripts: [w03.js]
---

The prediction of [lesson:3.1] is $g(d, \theta) = \theta/d^{2}$: a source of power $\theta$ at distance $d$. Week&nbsp;2
fixed $\theta = {{theta0}}$. This lesson lets the animal learn it, and in doing so
meets the plasticity rule that gives predictive coding its claim to biological
plausibility, along with two things that rule does not do.

## The derivative in the link

Write the link as the power multiplied by a shape,

$$
g(d, \theta) \;=\; \theta\,h(d),
\qquad
h(d) \;=\; \frac{1}{d^{2}} ,
$$ {#link-shape}

where the **link shape** $h$ is the part of the prediction that depends on the
state. The letter is chosen because $h$ is what $g$ has left once its parameter is
taken out.

::: derivation The update for the source power
$\theta$ appears in the log joint of [lesson:3.1] only through the sensory term
$-\big(u - \theta h(d)\big)^2/(2\Sigma_u)$. By the chain rule, with
$\partial g/\partial\theta = h(d)$,

$$
\frac{\partial \ell}{\partial \theta}
\;=\; \frac{u - \theta h(d)}{\Sigma_u}\cdot h(d)
\;=\; \varepsilon_u\,h(d) .
$$ {#grad-th}

The same steps hold for any link that is linear in its parameter: if
$g(d, \theta) = \theta h(d)$ for some shape $h$, the derivative is $\varepsilon_u h(d)$
whatever $h$ is.
:::

At Week&nbsp;2's observation and mode, $h(\hat d) = {{h_at_mode:.4f}}$ and
$\varepsilon_u = {{eps_u_w2:.4f}}$, so [eq:grad-th] is ${{grad_th_w2:.4f}}$, positive:
the sound was louder than the settled estimate predicts, and raising the source power
would explain more of it.

For a link that is not linear in its parameter the chain rule gives
$\varepsilon_u\,\partial g/\partial\theta$, and $\partial g/\partial\theta$ may depend
on $\theta$ as well as $d$. The form $\varepsilon_u h(d)$ is a property of the linear
case, which is also the case [Bogacz (2017)](https://doi.org/10.1016/j.jmp.2015.11.003)
treats in its section on learning the relationship between variables.

## A Hebbian rule

Give the circuit of [lesson:2.4] one more unit, whose activity is $h(d)$. It reads
the estimate and applies the shape, so its activity is fixed by $d$ alone. Its output
reaches the sensory error unit through a connection of strength $\theta$, so the
sensory error unit receives the prediction $\theta h(d)$ and subtracts it from $u$.

::: fig The sensory half of Week 2's circuit with the link shape given its own unit. The connection of strength θ carries the prediction to the sensory error unit, and its update is the product of the activities at its two ends. The return path from the error to the estimate also carries θ, as part of $g'(d) = \theta h'(d)$.
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="An estimate unit drives a link-shape unit; the link-shape unit sends the prediction to a sensory error unit through a connection of strength theta; the error returns to the estimate through a second connection that also carries theta.">
  <rect x="1" y="1" width="658" height="248" fill="#fdfcf9" stroke="#e4e1d9"/>
  <defs>
    <marker id="ah3" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#3c3c42"/>
    </marker>
    <marker id="ah3g" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#8a6d1f"/>
    </marker>
  </defs>

  <circle cx="120" cy="70" r="26" fill="#0f5f57"/>
  <text x="120" y="75" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="17" fill="#ffffff">d</text>
  <text x="120" y="32" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">estimate</text>

  <circle cx="330" cy="70" r="24" fill="#f4f2ec" stroke="#3c3c42" stroke-width="1.5"/>
  <text x="330" y="75" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="15" fill="#3c3c42">h(d)</text>
  <text x="330" y="32" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">link shape</text>

  <circle cx="540" cy="180" r="24" fill="#fdfcf9" stroke="#8a6d1f" stroke-width="2"/>
  <text x="540" y="186" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="16" fill="#8a6d1f">&#949;&#7524;</text>
  <text x="540" y="226" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">sensory error</text>

  <rect x="608" y="165" width="40" height="30" rx="4" fill="#f4f2ec" stroke="#d3cfc4"/>
  <text x="628" y="185" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="15" fill="#3c3c42">u</text>

  <line x1="146" y1="70" x2="302" y2="70" stroke="#3c3c42" stroke-width="1.5" marker-end="url(#ah3)"/>

  <line x1="352" y1="82" x2="516" y2="168" stroke="#0f5f57" stroke-width="3" marker-end="url(#ah3)"/>
  <text x="446" y="112" text-anchor="start" font-family="'Source Serif 4',serif" font-size="15" fill="#0f5f57">&#952;</text>
  <text x="462" y="112" text-anchor="start" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#6d6d75">prediction, subtracted</text>

  <line x1="608" y1="180" x2="566" y2="180" stroke="#3c3c42" stroke-width="1.5" marker-end="url(#ah3)"/>

  <path d="M516 186 C 380 220, 200 200, 128 98" fill="none" stroke="#8a6d1f" stroke-width="1.5" marker-end="url(#ah3g)"/>
  <text x="250" y="222" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#6d6d75">error returned, scaled by &#952; h&#8242;(d)</text>

  <text x="462" y="125" text-anchor="start" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#0f5f57">change &#8733; &#949;&#7524; &#215; h(d)</text>
</svg>
:::

[lesson:2.1] defined local plasticity as a change computed from the activities at the
two ends of a connection and its own strength, and the Hebbian rule as the product of
those two activities. The connection of strength $\theta$ runs from the link-shape
unit, with activity $h(d)$, to the sensory error unit, with activity $\varepsilon_u$.
[eq:grad-th] is exactly their product. Bogacz makes the same observation. The update needs nothing from the rest of the network, and in particular
it does not need to know what the shape $h$ is: the unit at the sending end has
already computed it.

The two parameters that enter a prediction follow one pattern. Each error is a
difference divided by a variance, and the update for a parameter in that difference
is minus the error times the derivative of the difference with respect to the
parameter. For $\theta$ the difference is $u - \theta h(d)$, whose derivative is
$-h(d)$, and the update is $+\varepsilon_u h(d)$. For the prior location the
difference is $d_p - d$, whose derivative is one, and the update is $-\varepsilon_p$:
the product of the error and a sending activity fixed at one, with a minus sign. The
prior location also carries the truncation term $-\zeta_p$ of [lesson:3.2], which is
not a product of activities. This is the payoff of the error-unit form of
[lesson:2.4]: apart from that correction, the quantities the gradient of the log joint
needs are activities that the inference circuit already has.

::: warning One parameter, two connections
$\theta$ appears twice in the circuit. It scales the prediction on the way to the
error unit, and it scales the error on the way back, since the return path carries
$g'(d) = \theta h'(d)$. A circuit built from connections has two separate strengths
there, call them $\theta_{\downarrow}$ and $\theta_{\uparrow}$, and the derivation of
[lesson:2.3] assumed they are equal.

Suppose each is updated by [eq:grad-th]. Bogacz notes that the return connection can
compute it locally, since the estimate unit at its receiving end fixes $h(d)$. Both
then receive the same increment $\beta\,\varepsilon_u h(\hat d)$ on every trial, so

$$
\theta_{\downarrow}^{(n+1)} - \theta_{\uparrow}^{(n+1)}
\;=\; \big(\theta_{\downarrow}^{(n)} + \beta\varepsilon_u h\big) - \big(\theta_{\uparrow}^{(n)} + \beta\varepsilon_u h\big)
\;=\; \theta_{\downarrow}^{(n)} - \theta_{\uparrow}^{(n)} .
$$

Learning preserves the difference exactly. If the two strengths start equal they stay
equal, and if they start unequal they never become equal.

What the inner loop does with unequal strengths can be computed. It climbs

$$
\frac{d_p - d}{\Sigma_p} \;+\; \frac{u - \theta_{\downarrow}h(d)}{\Sigma_u}\,\theta_{\uparrow}h'(d) ,
$$

the prediction made with the forward strength and the error returned with the other.
For positive strengths, write $\Sigma_u\theta_{\uparrow} = \theta_{\downarrow}\cdot\Sigma_u\theta_{\uparrow}/\theta_{\downarrow}$
and set $\Sigma_{\mathrm{eff}} = \Sigma_u\theta_{\downarrow}/\theta_{\uparrow}$. The second
term becomes $\big(u - \theta_{\downarrow}h(d)\big)\,\theta_{\downarrow}h'(d)/\Sigma_{\mathrm{eff}}$,
which is the derivative in $d$ of $-\big(u - \theta_{\downarrow}h(d)\big)^2/(2\Sigma_{\mathrm{eff}})$.
So the loop climbs the log joint of a model with power $\theta_{\downarrow}$ and sensory
variance $\Sigma_{\mathrm{eff}}$ in place of $\Sigma_u$, and settles at that model's mode,
which need not equal the mode of the model the animal is meant to hold. (The two
coincide when $d_p > 0$ on a trial with $u = \theta_{\downarrow}/d_p^2$, where both errors vanish at
$d = d_p$ whatever the sensory variance.) A mismatch between the two
strengths acts as a mis-set sensory variance, and because learning adds the same
amount to both, the ratio $\theta_{\downarrow}/\theta_{\uparrow}$, and with it
$\Sigma_{\mathrm{eff}}$, changes as the power is learned. With a single state every
such field is the derivative of something; with several states a mismatch between
forward and return weights need not give the gradient of any function, a case
[week:5] meets. The rule is local, but it relies on a symmetry that nothing local
enforces.
:::

## What learning drives to zero

Setting the drift of [eq:grad-th] to zero, the stationary condition is

$$
\mathbb E_{P^*}\big[\varepsilon_u\,h(\hat d)\big] \;=\; 0 .
$$ {#stat-th}

This is a condition on the learning signal
$\varepsilon_u h(\hat d)$, which weights each trial's sensory error by the link shape at
its estimate. It is not a condition on the sensory error alone, nor on the prior
error. Trials with near sources have large $h(\hat d)$ and count for more.

The widget below sets up a world whose sources are twice as powerful as the animal
believes, $\theta^{*} = {{w2_th_world:.0f}}$ against the animal's starting
$\theta = {{theta0:.0f}}$, with every other parameter as in Week&nbsp;2. It shows the
learned power across trials and the running averages of the two errors and of the
learning signal over the last ${{w2_window}}$ trials.

::: fig A world louder than the animal's model. Left, the learned source power across trials, against the true value. Right, the averages of the prior error, the sensory error and the learning signal over a window of recent trials. With learning switched off, the power stays at its starting value and the averages show what the mismatch does to inference alone.
<div class="widget" data-widget="wrong-link"></div>
:::

With learning off, over the last ${{w2_window}}$ of ${{n_trials}}$ trials the average
sensory error is ${{w2_off_epsu:.3f}}$, close to zero, and the average prior error is
${{w2_off_epsp:.3f}}$. Sounds come in louder than the animal's model predicts for
sources at their usual distances, and the inner loop explains them by placing the
sources nearer than the prior expects. The mismatch shows up in the prior error, and
the sensory error, which looks at the link, averages to almost nothing. A small average
sensory error is not evidence that the link is right.

The learning signal is not fooled: its average is ${{w2_off_sig:.4f}}$, because the
trials inferred as near carry large $h(\hat d)$ and, on those, the sensory error is
positive. With learning on, the power rises to ${{w2_final:.3f}}$ by the last trial,
averaging ${{w2_mean_late:.3f}}$ over the second half, and the window averages of the
learning signal fall from ${{w2_sig_early:.4f}}$ over the first ${{w2_window}}$ trials
to ${{w2_sig_late:.4f}}$ over the last.

The errors do not fall to zero with it. Over the last ${{w2_window}}$ trials the
average prior error is ${{w2_epsp_late:.3f}}$ and the average sensory error
${{w2_epsu_late:.3f}}$. With the true power given to the animal from the start and no
learning at all, the averages over every trial are ${{w2_true_epsp:.3f}}$ and
${{w2_true_epsu:.3f}}$. Neither error averages to zero even when the model is exactly
right, and two separate effects are in those numbers.

::: derivation The errors at the true parameters, averaged over the posterior
Suppose the model equals the process and each error is averaged over the posterior
instead of read at the mode. Averaging the posterior over the observations the process
produces gives back the joint distribution of distance and observation, which is the
process itself, so the process average of a posterior average is the process average
of the quantity averaged. For the sensory error that is the average of
$(u - g(d))/\Sigma_u$ over the process, the average noise divided by $\Sigma_u$, which
is zero. For the prior error it is
$(d_p - \mathbb E_{P^*}[d])/\Sigma_p$, and the mean of the truncated prior is
$d_p + \Sigma_p\zeta_p$ by [lesson:3.2], so the average is $-\zeta_p = -{{trunc_corr:.4f}}$.
:::

So averaged over posteriors, the sensory error at the truth is zero and the prior
error is slightly negative, because the truncated prior's mean lies above its
location. The simulated averages, ${{w2_true_epsp:.3f}}$ and ${{w2_true_epsu:.3f}}$, are
far from both. The difference is the mode: the errors are measured from one point of
each posterior, and the mode of a skewed posterior is not its mean. What does average
to zero at a stable resting point of learning is the drift, here the weighted signal
of [eq:stat-th]. A zero average of a derivative is not a minimum of the errors, and
"learning minimises prediction error" is not what this rule does.

The power learned is below the truth: the drift vanishes at ${{stat_th_point:.4f}}$,
${{stat_th_pct:.1f}}$ per cent from $\theta^{*}$. [lesson:3.4] traces the gap to the
difference between the mode and the posterior average, and finds that its sign for
$\theta$ depends on the world: in a world equal to Week&nbsp;2's model the same rule
overshoots the power, by ${{stat_th_pct_w2:.1f}}$ per cent.

## Three parameters that cannot be told apart

The widget learns $\theta$ alone. Learning $\theta$, $d_p$ and $\Sigma_p$ together runs
into a problem that no learning rule can solve, because it is a property of the model.

::: derivation A rescaling the evidence cannot see
Take any $c > 0$ and define $d' = c\,d$. If $d$ has the truncated prior with location
$d_p$ and variance $\Sigma_p$, then $d'$ has a truncated Gaussian prior with location
$c\,d_p$ and variance $c^{2}\Sigma_p$: a positive multiple of a Gaussian is Gaussian
with the mean and standard deviation multiplied by $c$, and $d > 0$ exactly when
$d' > 0$. The prediction in terms of $d'$ is

$$
\frac{\theta}{d^{2}} \;=\; \frac{\theta}{(d'/c)^{2}} \;=\; \frac{c^{2}\theta}{d'^{2}} .
$$

So the model with parameters $(\theta, d_p, \Sigma_p)$ and the model with
$(c^{2}\theta,\ c\,d_p,\ c^{2}\Sigma_p)$ generate the same distribution of $u$: the
second is the first with the hidden state measured in different units. Their
evidence $p(u)$ is identical for every $u$.
:::

At $c = {{scale_c:.0f}}$, the evidence for Week&nbsp;2's observation is
${{ev_w2:.7f}}$ under Week&nbsp;2's parameters and ${{ev_scaled:.7f}}$ under
$(\theta, d_p, \Sigma_p) = ({{scaled_th:.0f}}, {{scaled_dp:.0f}}, {{scaled_vp:.0f}})$, by
quadrature in `values.py`.

Any rule that depends only on the evidence is therefore indifferent along the whole
curve of rescaled parameters. What the observations can fix is the combinations that
do not change along it, such as $\theta/d_p^{2}$ and $\Sigma_p/d_p^{2}$. The animal can
learn how loud a typical source sounds and how much that varies. It cannot learn from
sound alone how far away sources are in metres.

The rule of this week does not depend only on the evidence, and along this curve it
is not indifferent.

::: derivation The settled log joint along the rescaling
The prior density of $d' = c\,d$ at $c\,d$ is the density of $d$ at $d$ divided by $c$,
since $\mathrm dd' = c\,\mathrm dd$, and the likelihood is unchanged because the
prediction is. So the log joint of the rescaled model at $c\,d$ is $\ell(d) - \ln c$.
Maximising over $d$ is the same as maximising over $c\,d$, so

$$
\hat\ell\big(u;\ c^{2}\theta,\ c\,d_p,\ c^{2}\Sigma_p\big) \;=\; \hat\ell\big(u;\ \theta, d_p, \Sigma_p\big) \;-\; \ln c .
$$

Differentiate with respect to $c$ at $c = 1$. The chain rule gives the parameters'
rates of change $2\theta$, $d_p$ and $2\Sigma_p$, so

$$
2\theta\,\frac{\partial\hat\ell}{\partial\theta} \;+\; d_p\,\frac{\partial\hat\ell}{\partial d_p}
\;+\; 2\Sigma_p\,\frac{\partial\hat\ell}{\partial\Sigma_p} \;=\; -1
$$ {#scale-id}

for every observation.
:::

At Week&nbsp;2's observation the left-hand side of [eq:scale-id], by finite
differences, is ${{scale_identity:.6f}}$, and $\hat\ell$ falls from
${{ell_hat_w2:.4f}}$ to ${{ell_hat_scaled:.4f}}$ at $c = {{scale_c:.0f}}$, a drop of
$\ln {{scale_c:.0f}}$. Since [eq:scale-id] holds on every trial, it holds for the drifts
too, so the three drifts can never vanish together: learning all three by this rule
has no resting point. Every trial pushes the parameters towards smaller $c$, and
$\hat\ell$ grows without bound as $c \to 0$.

::: warning Identifiability
A parameter that the evidence does not determine is **unidentifiable**. Updates that
follow the evidence do not fail visibly along such a direction: they wander along it,
or settle wherever the start and the noise leave them, and the learned values look
like answers. Updates that follow the settled log joint, as this week's do, are worse
off: [eq:scale-id] gives them a steady push along the direction, towards a sharper
and nearer prior and a quieter source. An agent whose distances have to mean
something, because it must reach the source, needs a second signal that fixes the
scale, such as the time it takes to get there. Each widget in this week learns one
parameter while the others are fixed, which removes the direction by fiat.
:::

::: exercise A link linear in the state
Suppose instead that $g(d, \theta) = \theta d$. Derive $\partial\ell/\partial\theta$ and
say which unit's activity plays the role of $h(d)$. Is the evidence still invariant
under some rescaling of $(\theta, d_p, \Sigma_p)$?
---solution---
The sensory term is $-(u - \theta d)^2/(2\Sigma_u)$, so
$\partial\ell/\partial\theta = \varepsilon_u\,d$. Here $h(d) = d$, so the sending unit is
the estimate unit itself and no separate link-shape unit is needed.

Under $d' = c\,d$ the prediction is $\theta d = (\theta/c)\,d'$, and the prior on $d'$
has location $c\,d_p$ and variance $c^2\Sigma_p$. So $(\theta, d_p, \Sigma_p)$ and
$(\theta/c,\ c\,d_p,\ c^2\Sigma_p)$ give the same evidence. The invariant combinations
are now $\theta d_p$ and $\theta^2\Sigma_p$. Any link of the form $\theta h(d)$ with $h$
a power of $d$ has such a symmetry; the power sets how $\theta$ must change.
:::

::: exercise Reading the averages
In the widget with learning off, the average sensory error is close to zero while the
link is wrong by a factor of two. Explain, using [eq:stat-th], why the learning signal
is nonetheless positive. Then make the world quieter than the model,
$\theta^{*} = {{w2q_th_world}}$, predict the sign of the learning signal, and check it
and the two simple error averages in the notebook. Does every world with
$\theta^{*} < \theta$ give the sign you predicted?
---solution---
The learning signal weights each trial's sensory error by $h(\hat d) = 1/\hat d^{2}$.
In a world louder than the model, the trials with the loudest sounds are inferred as
the nearest sources, so they have the largest $h(\hat d)$. On those trials the prior
pulls the estimate back towards $d_p$, the estimate is farther than the sound alone
would say, and the prediction falls short of the sound: $\varepsilon_u > 0$. On distant
trials the opposite happens with small weight. The simple average of $\varepsilon_u$
can cancel while the weighted one does not.

At $\theta^{*} = {{w2q_th_world}}$ the same argument with the signs reversed suggests a
negative learning signal, so that $\theta$ falls. It says nothing about the simple
averages. With learning off, the notebook gives averages over the last
${{w2_window}}$ trials of ${{w2q_off_epsp:.3f}}$ for the prior error,
${{w2q_off_epsu:.3f}}$ for the sensory error and ${{w2q_off_sig:.4f}}$ for the learning
signal, and the drift at $\theta = {{theta0:.0f}}$, averaged over the process by
quadrature, is ${{q50_grad:.4f}}$. The learning signal has the predicted sign. The two
errors have the same signs as when the model is exactly right, because both are
measured from the mode and carry the offsets described above.

The sign is not guaranteed for every quieter world. The argument compares the world
with the model, but the rule also has the bias of reading the mode, and in a world
equal to the model that bias alone pushes $\theta$ up, by [lesson:3.4]. At
$\theta^{*} = {{q99_th_world}}$ the drift at $\theta = {{theta0:.0f}}$ is
${{q99_grad:+.4f}}$: the world is slightly quieter, and the power still rises. The
mismatch decides the sign only when it is large enough to outweigh the bias.
:::

::: checkpoint
- Why is the update for $\theta$ the sensory error times $h(\hat d)$, and which unit
  holds $h(\hat d)$?
- What does the learning rule for $\theta$ drive to zero on average, and what does it
  leave alone?
- Why can a near-zero average sensory error coexist with a wrong link?
- Which combinations of $\theta$, $d_p$ and $\Sigma_p$ can sound alone determine?
:::
