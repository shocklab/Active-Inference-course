---
title: The Laplace approximation
deck: A Gaussian recognition density fitted to the curvature of the log joint at its peak, the free energy it gives, the size and sign of its error, and a log-normal density that gives a true bound on the same model.
week: 4
time: 70 min
scripts: [w04.js]
---

[lesson:4.2] ended with [week:2]'s sound model, $u \mid d \sim \mathcal{N}(\theta/d^2, \Sigma_u)$ with a Gaussian prior on $d$ restricted to $d > 0$. Expanding the square in the log likelihood,
$$
-\frac{(u - \theta d^{-2})^2}{2\Sigma_u} \;=\; -\frac{u^2}{2\Sigma_u} \;+\; \frac{u\theta}{\Sigma_u}\,d^{-2} \;-\; \frac{\theta^2}{2\Sigma_u}\,d^{-4},
$$
so for a positive observation the log joint contains $-d^2$ from the prior, $+d^{-2}$ and $-d^{-4}$ from the likelihood. The free-form update returns the exact posterior, and the Gaussian conjugate calculation does not evaluate its normaliser.

Continuous active inference takes two steps instead. It restricts $Q$ to a Gaussian, $Q(d) = \mathcal{N}(d; m, \Sigma_q)$, and it replaces the log joint $\ell(d) = \ln P(u, d)$ by its second-order Taylor expansion about $m$. Together these make the **Laplace approximation**, the form used in [Buckley et al. (2017)](https://doi.org/10.1016/j.jmp.2017.09.004), section&nbsp;4, and in Weeks 5 to 8 of this course. This lesson derives it, computes its error and the sign of that error, and compares it with a fixed-form density for which $\mathcal{F}$ can be evaluated without any expansion.

## Entropy of a Gaussian in $K$ dimensions

[lesson:4.1] derived the one-dimensional entropy $\tfrac12\ln(2\pi e\Sigma_q)$. The $K$-dimensional case needs one identity about the trace.

::: derivation Differential entropy of a $K$-dimensional Gaussian
Let $Q(s) = \mathcal{N}(s; m, \Sigma_q) = (2\pi)^{-K/2}(\det \Sigma_q)^{-1/2}\exp\!\big(-\tfrac{1}{2}(s - m)^\top \Sigma_q^{-1}(s - m)\big)$, with $\Sigma_q$ symmetric and positive definite. Then
$$
-\ln Q(s) \;=\; \frac{K}{2}\ln(2\pi) \;+\; \frac{1}{2}\ln\det\Sigma_q \;+\; \frac{1}{2}(s - m)^\top \Sigma_q^{-1}(s - m).
$$
A number equals its own trace, $\operatorname{tr}(AB) = \operatorname{tr}(BA)$, and the trace is linear, so it can be exchanged with the expectation:
$$
\mathbb{E}_{Q}\!\big[(s - m)^\top \Sigma_q^{-1}(s - m)\big]
\;=\; \mathbb{E}_{Q}\!\Big[\operatorname{tr}\!\big(\Sigma_q^{-1}(s - m)(s - m)^\top\big)\Big]
\;=\; \operatorname{tr}\!\Big(\Sigma_q^{-1}\,\mathbb{E}_{Q}\big[(s - m)(s - m)^\top\big]\Big)
\;=\; \operatorname{tr}(\Sigma_q^{-1}\Sigma_q) \;=\; K.
$$
Taking the expectation of $-\ln Q$,
$$
\mathrm{H}\big[\mathcal{N}(m, \Sigma_q)\big]
\;=\; \frac{K}{2}\ln(2\pi) + \frac{1}{2}\ln\det\Sigma_q + \frac{K}{2}
\;=\; \frac{1}{2}\ln\det(2\pi e\,\Sigma_q).
$$ {#gauss-entropy-kd}
For $K = 1$ this is the result of [lesson:4.1].
:::

## An untruncated Gaussian has infinite free energy

Before approximating anything, put $Q(d) = \mathcal{N}(d; m, \Sigma_q)$ on the whole real line into the exact functional $\mathcal{F}[Q, u] = -\mathbb{E}_{Q}[\ln P(u, d)] - \mathrm{H}[Q]$ for the sound model, whose prior is $P(d) = Z_p^{-1}\mathcal{N}(d; d_p, \Sigma_p)$ on $d > 0$ and zero for $d \le 0$. The result is $+\infty$, for two separate reasons.

1. *The supports differ.* $Q(d) > 0$ for $d \le 0$, where $P(u, d) = 0$. By [lesson:4.1], $\mathcal{F}$ is $+\infty$ whenever $Q$ puts mass where the joint vanishes.
2. *The log likelihood is not integrable at $d = 0$.* Even with the prior extended to the whole line, $-\ln P(u \mid d)$ grows like $\frac{\theta^2}{2\Sigma_u}d^{-4}$ as $d \to 0$. The Gaussian density at $0$ is $Q(0) = (2\pi\Sigma_q)^{-1/2} e^{-m^2/(2\Sigma_q)} > 0$, so near $0$ the expected energy contains $Q(0)\frac{\theta^2}{2\Sigma_u}\int_{-\delta}^{\delta} d^{-4}\,\mathrm{d}d = +\infty$.

So no Gaussian on the real line has finite free energy on this model. Two ways out are taken below: replace $\ell$ by a function that can be integrated against a Gaussian (Laplace), or replace the Gaussian by a density on $(0, \infty)$ that vanishes fast enough at $0$ (log-normal).

## The Laplace free energy

Write $\ell(d) = \ln P(u, d)$ for the log joint on $d > 0$, and let $m > 0$ be a point where $\ell$ is twice continuously differentiable with $\ell''(m) < 0$.

::: derivation Quadratic expansion, optimal width, and the cancellation of constants
Expand $\ell$ to second order about $m$:
$$
\ell_{\mathrm{quad}}(d; m)
\;=\; \ell(m) \;+\; \ell'(m)(d - m) \;+\; \tfrac{1}{2}\ell''(m)(d - m)^2.
$$ {#ell-quad}
Replace $\ell$ by $\ell_{\mathrm{quad}}$ in the expected energy of $Q = \mathcal{N}(m, \Sigma_q)$. Since $\mathbb{E}_{Q}[d - m] = 0$ and $\mathbb{E}_{Q}[(d - m)^2] = \Sigma_q$,
$$
-\mathbb{E}_{Q}\!\big[\ell_{\mathrm{quad}}(d; m)\big]
\;=\; -\ell(m) \;-\; \tfrac{1}{2}\ell''(m)\,\Sigma_q.
$$
Subtract the entropy $\tfrac12\ln(2\pi e\Sigma_q)$:
$$
\tilde{\mathcal{F}}(m, \Sigma_q)
\;=\; -\ell(m) \;-\; \tfrac{1}{2}\ell''(m)\,\Sigma_q \;-\; \tfrac{1}{2}\ln(2\pi e\,\Sigma_q).
$$ {#lap-two-arg}
For fixed $m$, minimise over $\Sigma_q > 0$:
$$
\frac{\partial \tilde{\mathcal{F}}}{\partial \Sigma_q}
\;=\; -\frac{1}{2}\ell''(m) \;-\; \frac{1}{2\Sigma_q} \;=\; 0
\qquad\Longrightarrow\qquad
\hat{\Sigma}_q(m) \;=\; -\frac{1}{\ell''(m)},
$$ {#lap-opt-var}
which is positive because $\ell''(m) < 0$. The second derivative is $\frac{1}{2\Sigma_q^2} > 0$, so this is the unique minimum. Substitute back: the expectation term is $-\frac{1}{2}\ell''(m)\cdot\big(-1/\ell''(m)\big) = \tfrac12$, and the entropy term is $-\tfrac{1}{2}\ln(2\pi\hat{\Sigma}_q) - \tfrac{1}{2}$. The two halves cancel:
$$
\tilde{\mathcal{F}}\big(m, \hat{\Sigma}_q(m)\big)
\;=\; -\ell(m) \;-\; \frac{1}{2}\ln\big(2\pi\hat{\Sigma}_q(m)\big)
\;=\; -\ell(m) \;+\; \frac{1}{2}\ln\!\left(\frac{-\ell''(m)}{2\pi}\right).
$$ {#lap-fe-1d}

In $K$ dimensions, with Hessian $H_\ell(m) = \nabla^2 \ell(m)$ negative definite, the same steps give $-\mathbb{E}_Q[\ell_{\mathrm{quad}}] = -\ell(m) - \tfrac{1}{2}\operatorname{tr}(H_\ell(m)\Sigma_q)$, and the objective is $-\ell(m) - \tfrac{1}{2}\operatorname{tr}(H_\ell\Sigma_q) - \tfrac12\ln\det(2\pi e\Sigma_q)$. Two derivatives with respect to the matrix $\Sigma_q$ are needed. The first, $\partial\operatorname{tr}(H\Sigma)/\partial\Sigma = H$ for symmetric $H$, holds because $\operatorname{tr}(H\Sigma) = \sum_{jk} H_{jk}\Sigma_{kj}$ is linear in the entries of $\Sigma$. The second, $\partial\ln\det\Sigma/\partial\Sigma = \Sigma^{-1}$, follows from the eigenvalues $\sigma_j$ of $\Sigma$: for a small symmetric change $\delta\Sigma$,
$$
\ln\det(\Sigma + \delta\Sigma) - \ln\det\Sigma
\;=\; \ln\det(I + \Sigma^{-1}\delta\Sigma)
\;=\; \operatorname{tr}(\Sigma^{-1}\delta\Sigma) + O(\|\delta\Sigma\|^2),
$$
since $\det(I + A) = \prod_j(1 + a_j) = 1 + \operatorname{tr}A + O(\|A\|^2)$ for the eigenvalues $a_j$ of a small $A$. Setting the derivative $-\tfrac12 H_\ell - \tfrac12\Sigma_q^{-1}$ to zero gives $\hat{\Sigma}_q(m) = -H_\ell(m)^{-1}$; then $-\tfrac{1}{2}\operatorname{tr}(H_\ell\hat{\Sigma}_q) = \tfrac{K}{2}$ cancels the $\tfrac{K}{2}$ in [eq:gauss-entropy-kd], leaving
$$
\mathcal{F}_{\mathrm{Lap}}(m)
\;=\; -\ell(m) \;-\; \frac{1}{2}\ln\det\!\big(2\pi\hat{\Sigma}_q(m)\big)
\;=\; -\ell(m) \;+\; \frac{1}{2}\ln\det\!\left(\frac{-H_\ell(m)}{2\pi}\right).
$$ {#lap-fe-kd}
:::

::: definition Laplace approximation and Laplace free energy
Let $\ell(s) = \ln P(u, s)$ have a local maximum $\hat s$ at which the Hessian $H_\ell(\hat s)$ is negative definite (a **nondegenerate peak**).

- The **Laplace approximation** to $P(s \mid u)$ is the Gaussian $Q_{\mathrm{Lap}}(s) = \mathcal{N}\big(s;\, \hat s,\, -H_\ell(\hat s)^{-1}\big)$.
- The **Laplace free energy** is the number $\mathcal{F}_{\mathrm{Lap}}(\hat s) = -\ell(\hat s) + \frac{1}{2}\ln\det\!\big(-H_\ell(\hat s)/(2\pi)\big)$.
:::

The nondegeneracy matters: $\ell(x) = -x^4$ has a strict maximum at $0$ with $\ell''(0) = 0$, and there $\hat\Sigma_q$ is undefined.

The Laplace free energy is not the free energy of the Laplace approximation. On the sound model,
$$
\mathcal{F}_{\mathrm{Lap}}(\hat d) \;<\; \infty \;=\; \mathcal{F}[Q_{\mathrm{Lap}}, u],
$$
for the two reasons of the previous section. $\mathcal{F}_{\mathrm{Lap}}$ is the free energy of $Q_{\mathrm{Lap}}$ computed with $\ell$ replaced by $\ell_{\mathrm{quad}}$, and the bound $\mathcal{F} \ge -\ln P(u)$ of [lesson:4.1] says nothing about it.

### Laplace's method

There is a second route to [eq:lap-fe-1d] that says what it approximates. Write the evidence as an integral and replace $\ell$ by its expansion at the mode, where $\ell'(\hat d) = 0$:
$$
P(u) \;=\; \int e^{\ell(d)}\,\mathrm{d}d
\;\approx\; \int_{-\infty}^{\infty} e^{\ell(\hat d) + \frac12\ell''(\hat d)(d - \hat d)^2}\,\mathrm{d}d
\;=\; e^{\ell(\hat d)}\sqrt{\frac{2\pi}{-\ell''(\hat d)}} \;=\; P_{\mathrm{Lap}}(u),
$$ {#laplace-method}
using the Gaussian integral $\int e^{-a x^2/2}\,\mathrm{d}x = \sqrt{2\pi/a}$. Taking $-\ln$ gives exactly [eq:lap-fe-1d] at $m = \hat d$. So $\mathcal{F}_{\mathrm{Lap}}(\hat d) = -\ln P_{\mathrm{Lap}}(u)$: the Laplace free energy is minus the logarithm of Laplace's method for the evidence integral, a standard approximation to an integral dominated by one peak. Its error is $\mathcal{F}_{\mathrm{Lap}} - (-\ln P(u)) = \ln P(u) - \ln P_{\mathrm{Lap}}(u)$, and that can be expanded.

::: derivation The leading correction to Laplace's method
Write $h = d - \hat d$, $a = -\ell''(\hat d) > 0$, $\ell_3 = \ell'''(\hat d)$, $\ell_4 = \ell''''(\hat d)$, and expand to fourth order:
$$
\ell(d) \;=\; \ell(\hat d) - \tfrac12 a h^2 + \tfrac16\ell_3 h^3 + \tfrac1{24}\ell_4 h^4 + \cdots.
$$
Factor out the Gaussian part and expand the exponential of the rest, $e^{x} = 1 + x + \tfrac12x^2 + \cdots$, keeping the terms that contribute at the leading order:
$$
e^{\ell(d)} \;=\; e^{\ell(\hat d) - a h^2/2}\Big[1 + \tfrac16\ell_3h^3 + \tfrac1{24}\ell_4h^4 + \tfrac1{72}\ell_3^2h^6 + \cdots\Big].
$$
Integrate term by term. Dividing by $P_{\mathrm{Lap}}(u)$ turns each integral into a moment of $h \sim \mathcal{N}(0, 1/a)$, for which $\mathbb{E}[h^3] = 0$, $\mathbb{E}[h^4] = 3/a^2$ and $\mathbb{E}[h^6] = 15/a^3$:
$$
\frac{P(u)}{P_{\mathrm{Lap}}(u)} \;\approx\; 1 + \frac{\ell_4}{24}\cdot\frac{3}{a^2} + \frac{\ell_3^2}{72}\cdot\frac{15}{a^3}
\;=\; 1 + \frac{\ell_4}{8a^2} + \frac{5\ell_3^2}{24a^3}.
$$
With $\ln(1 + x) \approx x$,
$$
\ln P(u) - \ln P_{\mathrm{Lap}}(u) \;\approx\; \frac{\ell''''(\hat d)}{8\,\ell''(\hat d)^2} \;+\; \frac{5\,\ell'''(\hat d)^2}{24\,|\ell''(\hat d)|^3}.
$$ {#laplace-correction}
The cubic term enters only through its square, because $\mathbb{E}[h^3] = 0$: under a Gaussian centred on the mode its own expectation vanishes.
:::

The expansion can be checked on a known case. Stirling's formula is Laplace's method applied to $n! = \int_0^\infty e^{n\ln x - x}\,\mathrm{d}x$. The mode is $x = n$, with $\ell'' = -1/n$, $\ell''' = 2/n^2$ and $\ell'''' = -6/n^3$, so [eq:laplace-correction] gives $-\tfrac{3}{4n} + \tfrac{5}{6n} = \tfrac{1}{12n}$, the first correction term in Stirling's series. At $n = {{stir_n:.0f}}$ the exact difference $\ln n! - \ln P_{\mathrm{Lap}}$ is ${{stir_err:.6f}}$ against $1/(12n) = {{stir_pred:.6f}}$.

The first term of [eq:laplace-correction] is negative wherever $\ell'''' < 0$, and the second is never negative. The sign of the leading correction predicts the sign of the error when the omitted remainder is controlled. In the two examples below the sign is checked against numerical integration.

### Evaluation on the sound model

Take [week:2]'s defaults, $d_p = {{d_prior:.0f}}$, $\Sigma_p = {{var_prior:.0f}}$, $\theta = {{theta0:.0f}}$, $\Sigma_u = {{var_obs:.1f}}$ and $u = {{u_obs:.1f}}$, with prior normaliser $Z_p = {{zp_w2:.6f}}$. The mode is $\hat d = {{w2_mode:.4f}}$, where $\ell(\hat d) = {{w2_ell_hat:.4f}}$. Writing $g(d) = \theta/d^2$, the negative second derivative has three parts, as in [lesson:2.4]:
$$
-\ell''(\hat d)
\;=\; \underbrace{\frac{1}{\Sigma_p}}_{{{var_prior:.4f}}}
\;+\; \underbrace{\frac{g'(\hat d)^2}{\Sigma_u}}_{{{w2_gn_sens_term:.4f}}}
\;-\; \underbrace{\frac{u - g(\hat d)}{\Sigma_u}\,g''(\hat d)}_{{{w2_hess_res_term:.4f}}}
\;=\; {{w2_abs_curv_hat:.4f}}.
$$
So $\hat{\Sigma}_q = 1/{{w2_abs_curv_hat:.4f}} = {{lap_var:.4f}}$ (standard deviation ${{lap_sd:.4f}}$), $\tfrac{1}{2}\ln(2\pi\hat{\Sigma}_q) = {{lap_width_term:.4f}}$, and
$$
\mathcal{F}_{\mathrm{Lap}}(\hat d) \;=\; {{w2_neg_ell_hat:.4f}} - {{lap_width_term:.4f}} \;=\; {{lap_F:.4f}}\text{ nats}.
$$
The surprise, computed by numerical integration of $e^{\ell}$ over $d > 0$, is $-\ln P(u) = {{w2_surp:.4f}}$ nats, so $\mathcal{F}_{\mathrm{Lap}}$ is ${{lap_err:.4f}}$ nats above it.

The derivatives at the mode are $\ell'''(\hat d) = {{w2_l3:.3f}}$ and $\ell''''(\hat d) = {{w2_l4:.3f}}$, and the two terms of [eq:laplace-correction] are ${{lap_c4:.4f}}$ and ${{lap_c3:.4f}}$, summing to ${{lap_pred:.4f}}$. The expansion gives the right sign, $\mathcal{F}_{\mathrm{Lap}}$ above the surprise, but not the size: ${{lap_c3:.4f}}$ is not small compared with $1$, so this truncation gives no controlled estimate of the omitted terms. The posterior of the sound model is too skewed for a one-term correction. On the asymmetric potential of Exercise&nbsp;4.6.5 ([lesson:4.6|the problems]) the same formula gives ${{ex_as_pred:.4f}}$, predicting that $\mathcal{F}_{\mathrm{Lap}}$ lies below the surprise there, which it does, by ${{ex_as_lap_under:.4f}}$ nats.

[lesson:4.4] uses $\mathcal{F}_{\mathrm{Lap}}$ to learn the model's parameters, and computes where the resulting learning rules come to rest on this model.

## Where the quadratic approximation fails

::: warning The Laplace free energy is not a bound
For every density $Q$, $\mathcal{F}[Q, u] = -\ln P(u) + D_{\mathrm{KL}}[Q \,\|\, P(\cdot \mid u)] \ge -\ln P(u)$. The Laplace free energy is computed with a quadratic approximation to the log joint. It is generally different from the exact free energy of the Laplace recognition density for the original model, so the variational bound does not apply to it. Its error can have either sign: on the sound model it lies above the surprise, and on the asymmetric potential of Exercise&nbsp;4.6.5 it lies below.
:::

Suppose the agent tries to improve on $m = \hat d$ by minimising [eq:lap-fe-1d] over $m$ as well. The derivative is
$$
\frac{\mathrm{d}\mathcal{F}_{\mathrm{Lap}}(m)}{\mathrm{d}m}
\;=\; -\ell'(m) \;+\; \frac{1}{2}\,\frac{\mathrm{d}}{\mathrm{d}m}\ln\big(-\ell''(m)\big)
\;=\; -\ell'(m) \;+\; \frac{\ell'''(m)}{2\,\ell''(m)}.
$$ {#lap-grad-m}
At the mode the first term vanishes, and with $\ell''' > 0$, $\ell'' < 0$ the second is negative, so $\mathcal{F}_{\mathrm{Lap}}$ falls as $m$ increases past $\hat d$: moving right, the curvature $-\ell''(m)$ decreases, $\hat\Sigma_q(m)$ increases, and the term $-\tfrac12\ln(2\pi\hat\Sigma_q)$ falls faster than $-\ell(m)$ rises.

::: warning Minimising the Laplace free energy over the mean moves it off the mode
On the sound model the minimiser of $\mathcal{F}_{\mathrm{Lap}}(m)$ is at $m = {{lap_shift_m:.4f}}$, against the mode ${{w2_mode:.4f}}$. There the width is $\hat\Sigma_q^{1/2} = {{lap_shift_sd:.4f}}$ and $\mathcal{F}_{\mathrm{Lap}} = {{lap_shift_F:.4f}}$ nats, which is ${{lap_shift_under:.4f}}$ nats below the surprise ${{w2_surp:.4f}}$.
:::

[Buckley et al. (2017)](https://doi.org/10.1016/j.jmp.2017.09.004), section&nbsp;4, treat the variance term $-\tfrac12\ln(2\pi\hat\Sigma_q)$ as a constant and drop it, keeping the Laplace-encoded energy $-\ln P(u, m)$ alone, so their state update climbs $\ell(m)$ alone, $\dot m \propto \ell'(m)$, which is [week:2]'s circuit, and stops at the mode. The calculation above shows one consequence of not doing so on this model; it is the course's own example, not the reason Buckley et al. give. The term cannot be dropped when parameters are learned, which [lesson:4.4] takes up.

::: warning Positive curvature away from the mode, and the Gauss&ndash;Newton curvature
At a nondegenerate peak $\ell'' < 0$, but not necessarily at the points a state update passes through on the way there. The exact second derivative is
$$
\ell''(m) \;=\; -\frac{1}{\Sigma_p} \;-\; \frac{g'(m)^2}{\Sigma_u} \;+\; \frac{u - g(m)}{\Sigma_u}\,g''(m),
$$
and on the sound model $g''(m) = 6\theta/m^4 > 0$, so where the sound is louder than predicted, $u > g(m)$, the third term is positive. At $\Sigma_u = {{infl_vu:.2f}}$ and $u = {{infl_u:.1f}}$, evaluated at $m = {{d_prior:.0f}}$, the result is $\ell''({{d_prior:.0f}}) = {{infl_curv_at_2:.4f}} > 0$: the log joint curves upwards there, $-1/\ell''(m)$ is negative, and no Gaussian has that curvature.

Dropping the third term leaves the **Gauss&ndash;Newton curvature** $\ell''_{\mathrm{GN}}(m) = -\Sigma_p^{-1} - \Sigma_u^{-1}g'(m)^2$, which is negative at every $m > 0$; at the same point it is ${{infl_gn_at_2:.4f}}$. At the default mode, $-\ell''_{\mathrm{GN}}(\hat d) = {{w2_abs_gn_curv_hat:.4f}}$ against the exact ${{w2_abs_curv_hat:.4f}}$, giving standard deviation ${{lap_gn_sd:.4f}}$ and $\mathcal{F}_{\mathrm{Lap,GN}} = {{lap_gn_F:.4f}}$ nats. [week:5] meets this choice again when states are updated away from the mode.
:::

## A log-normal recognition density

A two-parameter density can respect $d > 0$ and still give a free energy that is a true bound, with no Taylor expansion. Put the Gaussian on the log-distance $\rho = \ln d$ of [lesson:2.5]: $\rho \sim \mathcal{N}(m_\rho, \Sigma_\rho)$. By the change-of-variables rule $Q(d) = Q_\rho(\ln d)\,\big|\frac{\mathrm{d}\rho}{\mathrm{d}d}\big|$, the density of $d$ is the log-normal
$$
Q_{\mathrm{LN}}(d; m_\rho, \Sigma_\rho)
\;=\; \frac{1}{d\sqrt{2\pi\Sigma_\rho}}\exp\!\left(-\frac{(\ln d - m_\rho)^2}{2\Sigma_\rho}\right),
\qquad d > 0.
$$ {#lognormal-q}
As $d \to 0^+$, $\ln d \to -\infty$, and the factor $\exp\!\big(-(\ln d - m_\rho)^2/(2\Sigma_\rho)\big)$ tends to zero faster than any power $d^{j}$, so it outweighs the $d^{-4}$ growth of the log likelihood.

::: derivation Closed-form free energy of the log-normal family
Two ingredients are needed: the entropy of $Q_{\mathrm{LN}}$, and the moments $\mathbb{E}_{Q_{\mathrm{LN}}}[d^{j}]$ for $j \in \{1, 2, -2, -4\}$.

1. *Entropy.* Substitute $d = e^\rho$, $\mathrm{d}d = e^\rho\,\mathrm{d}\rho$ and $Q_{\mathrm{LN}}(e^\rho) = e^{-\rho} Q_\rho(\rho)$:
$$
\begin{aligned}
\mathrm{H}[Q_{\mathrm{LN}}]
&\;=\; -\int_{-\infty}^{\infty} e^{-\rho} Q_\rho(\rho)\,\ln\!\big(e^{-\rho} Q_\rho(\rho)\big)\,e^\rho\,\mathrm{d}\rho
\;=\; -\int_{-\infty}^{\infty} Q_\rho(\rho)\big(-\rho + \ln Q_\rho(\rho)\big)\,\mathrm{d}\rho \\
&\;=\; \mathbb{E}_{Q_\rho}[\rho] \;+\; \mathrm{H}[Q_\rho]
\;=\; m_\rho \;+\; \tfrac{1}{2}\ln(2\pi e\,\Sigma_\rho).
\end{aligned}
$$ {#ln-entropy}

2. *Moments.* Write $d^{j} = e^{j\rho}$ and complete the square in the exponent:
$$
j\rho - \frac{(\rho - m_\rho)^2}{2\Sigma_\rho}
\;=\; -\frac{\big(\rho - (m_\rho + j\Sigma_\rho)\big)^2}{2\Sigma_\rho} \;+\; j m_\rho \;+\; \frac{1}{2}j^2\Sigma_\rho.
$$
The shifted Gaussian integrates to $1$, so
$$
\mathbb{E}_{Q_{\mathrm{LN}}}[d^{j}] \;=\; \exp\!\big(j m_\rho + \tfrac{1}{2}j^2\Sigma_\rho\big)
\qquad\text{for every real } j.
$$ {#ln-moments}

3. *Assembly.* Write $A = \mathbb{E}[d]$, $B = \mathbb{E}[d^2]$, $C = \mathbb{E}[d^{-2}]$ and $D = \mathbb{E}[d^{-4}]$, each from [eq:ln-moments]. Expanding the squares in $-\ln P(d)$ and $-\ln P(u \mid d)$ and taking expectations,
$$
\begin{aligned}
-\mathbb{E}_{Q_{\mathrm{LN}}}[\ln P(d)]
&\;=\; \frac{B - 2 d_p A + d_p^2}{2\Sigma_p} \;+\; \frac{1}{2}\ln(2\pi\Sigma_p) \;+\; \ln Z_p, \\
-\mathbb{E}_{Q_{\mathrm{LN}}}[\ln P(u \mid d)]
&\;=\; \frac{u^2 - 2 u\theta\,C + \theta^2 D}{2\Sigma_u} \;+\; \frac{1}{2}\ln(2\pi\Sigma_u).
\end{aligned}
$$
Subtracting [eq:ln-entropy] gives $\mathcal{F}_{\mathrm{LN}}(m_\rho, \Sigma_\rho)$ in closed form. It is the exact $\mathcal{F}$ of a density on $d > 0$, so by [lesson:4.1] $\mathcal{F}_{\mathrm{LN}}(m_\rho, \Sigma_\rho) \ge -\ln P(u)$ for every $m_\rho$ and every $\Sigma_\rho > 0$.
:::

To minimise it, differentiate. From [eq:ln-moments], $\partial A/\partial m_\rho = A$, $\partial B/\partial m_\rho = 2B$, $\partial C/\partial m_\rho = -2C$, $\partial D/\partial m_\rho = -4D$, and $\partial A/\partial\Sigma_\rho = \tfrac12 A$, $\partial B/\partial\Sigma_\rho = 2B$, $\partial C/\partial\Sigma_\rho = 2C$, $\partial D/\partial\Sigma_\rho = 8D$. The stationarity conditions are
$$
\frac{\partial\mathcal{F}_{\mathrm{LN}}}{\partial m_\rho} \;=\; \frac{B - d_p A}{\Sigma_p} + \frac{2\theta\,(uC - \theta D)}{\Sigma_u} - 1 \;=\; 0,
\qquad
\frac{\partial\mathcal{F}_{\mathrm{LN}}}{\partial \Sigma_\rho} \;=\; \frac{2B - d_p A}{2\Sigma_p} + \frac{2\theta\,(2\theta D - uC)}{\Sigma_u} - \frac{1}{2\Sigma_\rho} \;=\; 0.
$$ {#ln-stationary}
They have no closed-form solution. Newton's method on these two equations gives $m_\rho = {{ln_mr:.4f}}$ and $\Sigma_\rho = {{ln_sr:.4f}}$ (standard deviation ${{ln_sd_rho:.4f}}$ in $\rho$), with $\mathcal{F}_{\mathrm{LN}} = {{ln_F:.4f}}$ nats.

| Recognition density | Mode in $d$ | Mean $\mathbb{E}[d]$ | Standard deviation of $d$ | Free energy (nats) | Difference from $-\ln P(u)$ | A bound? |
|---|---:|---:|---:|---:|---:|---|
| Exact posterior | ${{w2_mode:.4f}}$ | ${{w2_mean:.4f}}$ | ${{w2_sd:.4f}}$ | ${{w2_surp:.4f}}$ | $0$ | equal |
| Log-normal | ${{ln_mode_d:.4f}}$ | ${{ln_mean_d:.4f}}$ | ${{ln_sd_d:.4f}}$ | ${{ln_F:.4f}}$ | ${{ln_gap:.4f}}$ | yes |
| Laplace at $\hat d$ | ${{w2_mode:.4f}}$ | ${{w2_mode:.4f}}$ | ${{lap_sd:.4f}}$ | ${{lap_F:.4f}}$ | ${{lap_err:.4f}}$ | no |
| Laplace at the minimiser of $\mathcal{F}_{\mathrm{Lap}}$ | ${{lap_shift_m:.4f}}$ | ${{lap_shift_m:.4f}}$ | ${{lap_shift_sd:.4f}}$ | ${{lap_shift_F:.4f}}$ | ${{lap_shift_err:.4f}}$ | no |

The exact column comes from numerical integration over $d > 0$. The log-normal density is skewed to the right, as the posterior is, and its mean ${{ln_mean_d:.4f}}$ is within ${{ln_mean_err:.4f}}$ of the exact mean, where the symmetric Laplace Gaussian puts its mean at the mode. Its free energy is ${{ln_gap:.4f}}$ nats above the surprise, against ${{lap_err:.4f}}$ for Laplace at the mode. Its standard deviation, ${{ln_sd_d:.4f}}$, is below the exact ${{w2_sd:.4f}}$. That is the direction [lesson:1.4]'s comparison of the two divergences leads one to expect: minimising $D_{\mathrm{KL}}[Q \,\|\, P]$ penalises $Q$ heavily where $P$ is small, so the minimiser tends to stay inside the posterior rather than cover its tails. Here it is one computed case, not a theorem.

::: fig The exact posterior of the sound model at $u = {{u_obs:.1f}}$ (green), the Laplace Gaussian at the mode $\hat d = {{w2_mode:.4f}}$ (clay, dashed), and the log-normal density that minimises $\mathcal{F}$ in its family (blue). The curves are computed from the same arrays as the table.
<svg viewBox="0 0 680 272" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Exact posterior density over distance d compared with the Laplace Gaussian at the mode and the log-normal density that minimises the free energy.">
  <rect x="1" y="1" width="678" height="270" fill="#fdfcf9" stroke="#e4e1d9"/>
  <line x1="58" y1="232" x2="642" y2="232" stroke="#3c3c42"/>
  <line x1="58" y1="28" x2="58" y2="232" stroke="#3c3c42"/>
  <line x1="{{fig3_mode_x}}" y1="36" x2="{{fig3_mode_x}}" y2="232" stroke="#9c4d2f" stroke-dasharray="3 3" opacity="0.7"/>
  <line x1="{{fig3_mean_x}}" y1="36" x2="{{fig3_mean_x}}" y2="232" stroke="#0f5f57" stroke-dasharray="4 3" opacity="0.7"/>
  <path d="{{fig3_exact_path}}" fill="none" stroke="#0f5f57" stroke-width="2.4"/>
  <path d="{{fig3_ln_path}}" fill="none" stroke="#2c5f8a" stroke-width="2.0"/>
  <path d="{{fig3_lap_path}}" fill="none" stroke="#9c4d2f" stroke-width="2.0" stroke-dasharray="6 4"/>
  <text x="58" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">0</text>
  <text x="{{fig3_tick_1_x}}" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">1</text>
  <text x="{{fig3_tick_2_x}}" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">2</text>
  <text x="{{fig3_tick_3_x}}" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">3</text>
  <text x="{{fig3_tick_4_x}}" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">4</text>
  <text x="{{fig3_tick_5_x}}" y="248" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">5</text>
  <text x="350" y="264" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" font-weight="600" fill="#6d6d75">distance d</text>
  <text x="{{fig3_mode_x}}" y="24" text-anchor="end" dx="-4" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#9c4d2f">mode {{w2_mode:.2f}}</text>
  <text x="{{fig3_mean_x}}" y="24" text-anchor="start" dx="4" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#0f5f57">exact mean {{w2_mean:.2f}}</text>
  <line x1="430" y1="54" x2="452" y2="54" stroke="#0f5f57" stroke-width="2.4"/>
  <text x="458" y="58" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">exact posterior</text>
  <line x1="430" y1="74" x2="452" y2="74" stroke="#2c5f8a" stroke-width="2.0"/>
  <text x="458" y="78" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">log-normal (F = {{ln_F:.3f}})</text>
  <line x1="430" y1="94" x2="452" y2="94" stroke="#9c4d2f" stroke-width="2.0" stroke-dasharray="6 4"/>
  <text x="458" y="98" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">Laplace at the mode (F = {{lap_F:.3f}})</text>
</svg>
:::

::: widget laplace-vs-lognormal | The exact posterior, the Laplace Gaussian at the mode, and the log-normal density that minimises $\mathcal{F}$, across observations $u$, sensory variances $\Sigma_u$ and prior variances $\Sigma_p$. A wide prior or a quiet sound puts much of the Laplace Gaussian at $d < 0$, while the log-normal stays on $d > 0$. A sharp ear and a loud sound can make the posterior bimodal, and the Laplace Gaussian then describes only the peak it is fitted to.
:::

::: exercise Exactness of Laplace on linear Gaussian models
Suppose $\ell(s) = \ln P(u, s)$ is exactly quadratic on $\mathbb{R}^K$:
$$
\ell(s) \;=\; \ell(\hat s) \;-\; \tfrac{1}{2}(s - \hat s)^\top \Pi_{\mathrm{post}}(s - \hat s),
$$
with $\Pi_{\mathrm{post}}$ symmetric and positive definite. Prove that $\mathcal{F}_{\mathrm{Lap}}(\hat s) = -\ln P(u)$.
---solution---
The Hessian is the same everywhere, $H_\ell(s) = -\Pi_{\mathrm{post}}$, and every third and higher derivative of $\ell$ is zero. So $\hat{\Sigma}_q = -H_\ell^{-1} = \Pi_{\mathrm{post}}^{-1}$. Integrate $e^{\ell}$ over $\mathbb{R}^K$ with the Gaussian normaliser:
$$
P(u) \;=\; \int_{\mathbb{R}^K} e^{\ell(\hat s)}\exp\!\Big(-\tfrac{1}{2}(s - \hat s)^\top \Pi_{\mathrm{post}}(s - \hat s)\Big)\,\mathrm{d}s
\;=\; e^{\ell(\hat s)}\,(2\pi)^{K/2}\big(\det \Pi_{\mathrm{post}}^{-1}\big)^{1/2}.
$$
Taking $-\ln$,
$$
-\ln P(u) \;=\; -\ell(\hat s) \;-\; \frac{1}{2}\ln\det\!\big(2\pi\Pi_{\mathrm{post}}^{-1}\big)
\;=\; -\ell(\hat s) \;-\; \frac{1}{2}\ln\det\!\big(2\pi\hat{\Sigma}_q\big)
\;=\; \mathcal{F}_{\mathrm{Lap}}(\hat s).
$$
Here $\ell_{\mathrm{quad}} = \ell$ everywhere, so $Q_{\mathrm{Lap}}$ is the exact posterior and Laplace's method [eq:laplace-method] is exact.
:::
