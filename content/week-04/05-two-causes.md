---
title: A mean-field learning failure
deck: Variational EM with three mean-field families on a two-cause model, compared with exact EM computed by numerical integration, with each source of error in the learned prior rates derived and then measured.
week: 4
time: 100 min
scripts: [w04.js]
---

[lesson:4.4] ended with the objective that variational EM minimises,
$$
\min_{Q_n \in \mathcal{Q}}\,\mathcal{F}_{\mathrm{tot}}[Q_{1:N}; \omega]
\;=\; -\ln P(u_{1:N} \mid \omega) \;+\; \sum_{n=1}^{N}\min_{Q_n \in \mathcal{Q}} D_{\mathrm{KL}}\!\big[Q_n \,\big\|\, P(\cdot \mid u_n, \omega)\big],
$$
and with the observation that the optimised divergence term can shift the stationary point that variational EM reaches away from a stationary point of the evidence, wherever the divergence's derivative in $\omega$ is nonzero there. This lesson measures how far, on a model with two positive hidden causes and a restricted family whose parameters are learned. Exact EM is computed alongside by numerical integration, so every learned value can be compared with the exact-EM fit on the same data.

Four separate effects turn out to act on the learned parameters. They are listed here so that the rest of the lesson can refer to them by number:

1. a recognition family whose coefficient of variation is fixed, which forces the posterior means down;
2. a learning rule that averages the wrong quantity across trials;
3. the closed loop between E-steps and M-steps, which multiplies whatever error one M-step makes;
4. the mean-field factorisation itself, which narrows each factor of $Q$ and, on this model, lowers the means.

Effects 1 and 4 are properties of the recognition family. Effect 2 is a property of the learning rule and not of variational inference. Effect 3 is present in exact EM too, and it creates no error of its own.

## The two-cause model

Two positive hidden causes $s_1, s_2 > 0$ have independent exponential priors with rates $\lambda_1, \lambda_2 > 0$:
$$
P(s_i \mid \lambda_i) \;=\; \lambda_i e^{-\lambda_i s_i}, \qquad s_i > 0, \quad i = 1, 2,
$$ {#tc-prior}
with prior means $\mathbb{E}[s_i] = 1/\lambda_i$. In this lesson $\lambda_i$ is an exponential rate, in inverse units of $s_i$; it is unrelated to the noisy-OR leak $\lambda$ of [lesson:1.4]. Two sensors report products of the causes with unit-variance Gaussian noise,
$$
u_1 \;=\; s_2 s_1^2 + z_1, \qquad u_2 \;=\; s_1 s_2^2 + z_2, \qquad z_1, z_2 \sim \mathcal{N}(0, 1) \text{ independent}.
$$ {#tc-obs}
The log joint for one trial is
$$
\ln P(u, s \mid \lambda) \;=\; \sum_{i=1}^{2}\big(\ln\lambda_i - \lambda_i s_i\big) \;-\; \tfrac12\big(u_1 - s_2 s_1^2\big)^2 \;-\; \tfrac12\big(u_2 - s_1 s_2^2\big)^2 \;-\; \ln 2\pi,
$$ {#tc-logjoint}
with $u = (u_1, u_2)$, $s = (s_1, s_2)$ and $\lambda = (\lambda_1, \lambda_2)$.

The generative process uses the same model with rates $\lambda^{*} = ({{r1_true:.1f}}, {{r2_true:.1f}})$, so the prior means of the causes are ${{m1_prior_true:.4f}}$ and ${{m2_prior_true:.4f}}$. One seeded sample of $N = {{tc_n:.0f}}$ trials is drawn and held fixed for the whole lesson, and every learner starts from $\lambda^{(0)} = ({{r1_init:.1f}}, {{r2_init:.1f}})$. Every number below is a comparison on this one sample.

::: warning One sample, one pair of rates
The percentages in this lesson are measured on one sample of ${{tc_n:.0f}}$ trials at one pair of true rates and one starting point. They show that each effect exists and how large it is here. They are not general rates of error for these families, and a different sample or different true rates would give different numbers.
:::

### The shape of the posterior

Fix an observation, $u = ({{rep_u1:.1f}}, {{rep_u2:.1f}})$, which the lesson uses throughout as its representative trial. The likelihood is large where $s_2 s_1^2 \approx {{rep_u1:.1f}}$ and $s_1 s_2^2 \approx {{rep_u2:.1f}}$, two curves in the $(s_1, s_2)$ plane that cross, and the noise widens each into a band. Along the first curve a larger $s_1$ needs a smaller $s_2$. The posterior is therefore a curved ridge on which the two causes are negatively correlated. At the true rates its correlation is ${{rep_ex_corr:.4f}}$, its means are $({{rep_ex_m1:.4f}}, {{rep_ex_m2:.4f}})$ and its standard deviations are $({{rep_ex_sd1:.4f}}, {{rep_ex_sd2:.4f}})$. Across all ${{tc_n:.0f}}$ trials the posterior correlation averages ${{tc_ex_corr:.4f}}$, ranges from ${{tc_ex_corr_min:.4f}}$ to ${{tc_ex_corr_max:.4f}}$, and exceeds $0.5$ in magnitude on ${{tc_ex_corr_pct:.1f}}$ per cent of trials.

::: fig The posterior for the representative observation $u = ({{rep_u1:.1f}}, {{rep_u2:.1f}})$ at the true rates, with contours enclosing 50 and 90 per cent of the probability of each density. Left: the exact posterior (green) and the free-form mean-field product (blue). Right: the exact posterior with the Gamma product (purple, dashed) and the Exponential product (clay, dotted). The three families are defined below. All four densities are computed on one grid from the same arrays the lesson's numbers come from.
{{fig_ridge_svg}}
:::

### The M-step for a prior rate

Variational EM needs one M-step, and it has the same form for every recognition family. In $\mathcal{F}_{\mathrm{tot}} = \sum_n \mathcal{F}[Q_n, u_n; \lambda]$ the rate $\lambda_i$ appears only in the prior term of [eq:tc-logjoint], so

::: derivation The M-step for an exponential rate
For trial $n$, the part of $\mathcal{F}[Q_n, u_n; \lambda] = \mathbb{E}_{Q_n}[\ln Q_n] - \mathbb{E}_{Q_n}[\ln P(u_n, s \mid \lambda)]$ that depends on $\lambda_i$ is
$$
-\mathbb{E}_{Q_n}\big[\ln\lambda_i - \lambda_i s_i\big] \;=\; -\ln\lambda_i + \lambda_i\,\mathbb{E}_{Q_n}[s_i].
$$
Summing over trials and writing $\bar m_i = \frac{1}{N}\sum_n \mathbb{E}_{Q_n}[s_i]$ for the average recognition mean of cause $i$,
$$
\mathcal{F}_{\mathrm{tot}} \;=\; -N\ln\lambda_i + N\lambda_i \bar m_i + (\text{terms free of } \lambda_i).
$$
The derivative is $-N/\lambda_i + N\bar m_i$, zero at $\lambda_i = 1/\bar m_i$, and the second derivative $N/\lambda_i^2$ is positive, so this is the unique minimiser:
$$
\lambda_i \;\leftarrow\; \frac{1}{\bar m_i} \;=\; \left(\frac{1}{N}\sum_{n=1}^{N}\mathbb{E}_{Q_n}[s_i]\right)^{-1}.
$$ {#tc-mstep}
:::

The M-step uses the recognition densities only through their means. For exact EM, where $Q_n$ is the posterior $P(s \mid u_n, \lambda)$, a fixed point of [eq:tc-mstep] is a stationary point of the evidence. By Fisher's identity ([lesson:3.4], and the exercise at the end of [lesson:4.4]),
$$
\frac{\partial}{\partial\lambda_i}\ln P(u_{1:N} \mid \lambda)
\;=\; \sum_{n=1}^{N}\mathbb{E}_{P(s \mid u_n, \lambda)}\!\left[\frac{1}{\lambda_i} - s_i\right]
\;=\; \frac{N}{\lambda_i} - \sum_{n=1}^{N}\mathbb{E}[s_i \mid u_n, \lambda],
$$ {#tc-score}
which is zero exactly when $\lambda_i = 1/\bar m_i$. So exact EM, run to convergence, returns a stationary point $\hat\lambda$ of the likelihood of this sample. A zero score and a converged EM run do not prove that it is the global maximum, and the lesson calls it the exact-EM fit. That, not $\lambda^{*}$, is the benchmark for the variational rows below. On this sample $\hat\lambda = ({{tc_fit_exact_1:.4f}}, {{tc_fit_exact_2:.4f}})$, and the largest component of [eq:tc-score] there is ${{tc_ml_score:sci1}}$.

### An online rule that averages rates

Here is a rule that suggests itself when each $Q_n$ is an exponential density with mean $m_{n,i}$, and so with rate $1/m_{n,i}$, the same kind of quantity as $\lambda_i$: after each trial, move the prior rate a little towards the trial's rate,
$$
\lambda_i \;\leftarrow\; {{ema_keep:.2f}}\,\lambda_i + {{ema_rate:.2f}}\,\frac{1}{m_{n,i}}.
$$ {#tc-online}
Run once through the ${{tc_n:.0f}}$ trials from $\lambda^{(0)}$, with each trial's exponential recognition density (defined below) taken as the best of 25 starts at the current rates, it ends at $({{tc_ema_1:.3f}}, {{tc_ema_2:.3f}})$, which is ${{tc_ema_ratio_1:.2f}}$ and ${{tc_ema_ratio_2:.2f}}$ times the true rates. It has not settled: over the last 100 trials $\lambda_1$ stayed between ${{tc_ema_lo_1:.3f}}$ and ${{tc_ema_hi_1:.3f}}$ and $\lambda_2$ between ${{tc_ema_lo_2:.3f}}$ and ${{tc_ema_hi_2:.3f}}$. The rest of the lesson takes this result apart.

[eq:tc-online] is an online version of the arithmetic mean of the trials' rates, $\lambda_i^{\mathrm{AM}} = \frac1N\sum_n 1/m_{n,i}$. The M-step [eq:tc-mstep] is instead the reciprocal of the arithmetic mean of $m_{n,i}$, which is the harmonic mean of the rates $1/m_{n,i}$. The two differ in a known direction. The function $x \mapsto 1/x$ is strictly convex on $(0, \infty)$, so Jensen's inequality for a convex function ([lesson:4.1]) gives
$$
\frac{1}{\bar m_i} \;=\; \frac{1}{\frac1N\sum_n m_{n,i}} \;\le\; \frac{1}{N}\sum_{n=1}^{N}\frac{1}{m_{n,i}} \;=\; \lambda_i^{\mathrm{AM}},
$$ {#tc-am-hm}
with equality only when every $m_{n,i}$ is the same. So averaging rates always gives a rate at least as large as the M-step, and $\lambda^{\mathrm{AM}}$ is not a minimiser of $\mathcal{F}_{\mathrm{tot}}$ over $\lambda$: under it, $\mathcal{F}_{\mathrm{tot}}$ can rise. This is effect 2. It belongs to the rule, and the remedy is the M-step.

## The exact posterior by numerical integration

We evaluate the normaliser and moments of the posterior [eq:tc-logjoint] by numerical integration, and the lesson calls the result the exact posterior with that understanding. Two features make a plain grid in $(s_1, s_2)$ unreliable. For large observations the ridge is very thin: the trial with the largest observations, $u = ({{tc_big_u1:.2f}}, {{tc_big_u2:.2f}})$ (trial ${{tc_big_i:.0f}}$ of the sample, counting from 0), has posterior standard deviations $({{tc_big_sd1:.4f}}, {{tc_big_sd2:.4f}})$ about means $({{tc_big_m1:.4f}}, {{tc_big_m2:.4f}})$, with correlation ${{tc_big_corr:.4f}}$. And the ridge is curved, so no rectangle of grid points follows it. A change of variables removes both problems.

::: derivation Coordinates in which the likelihood factorises
Let $y_1 = (s_2 s_1^2)^{1/3}$ and $y_2 = (s_1 s_2^2)^{1/3}$, the cube roots of the two noiseless sensor outputs. Here $y_i$ is a local coordinate, not an observation. Multiplying, $y_1^3 y_2^3 = s_1^3 s_2^3$, so $s_1 s_2 = y_1 y_2$, and then
$$
s_1 \;=\; \frac{s_2 s_1^2}{s_1 s_2} \;=\; \frac{y_1^3}{y_1 y_2} \;=\; \frac{y_1^2}{y_2},
\qquad
s_2 \;=\; \frac{y_2^2}{y_1}.
$$
These formulas define a smooth bijection of the open quadrant $y_1, y_2 > 0$ onto $s_1, s_2 > 0$; substituting back gives $s_2 s_1^2 = (y_2^2/y_1)(y_1^4/y_2^2) = y_1^3$, as required. Its Jacobian matrix is
$$
\frac{\partial(s_1, s_2)}{\partial(y_1, y_2)} \;=\;
\begin{pmatrix} 2y_1/y_2 & -y_1^2/y_2^2 \\ -y_2^2/y_1^2 & 2y_2/y_1 \end{pmatrix},
$$
with determinant $4 - 1 = 3$. The evidence becomes
$$
P(u \mid \lambda) \;=\; \frac{3\lambda_1\lambda_2}{2\pi}\int_0^\infty\!\!\int_0^\infty
\exp\!\left(-\lambda_1\frac{y_1^2}{y_2} - \lambda_2\frac{y_2^2}{y_1}\right)
e^{-\frac12(u_1 - y_1^3)^2}\,e^{-\frac12(u_2 - y_2^3)^2}\;\mathrm{d}y_1\,\mathrm{d}y_2 .
$$ {#tc-evidence-y}
Each sensor factor now depends on one coordinate only, so the integration limits and nodes for $y_1$ can be chosen from $u_1$ alone and those for $y_2$ from $u_2$ alone. Posterior moments are ratios of integrals of the same form with $s_1 = y_1^2/y_2$ and $s_2 = y_2^2/y_1$ inserted.
:::

The nodes for $y_i$ are laid out as follows. The sensor factor is a Gaussian of unit width in $g_i = y_i^3$ centred on $u_i$. The range $g_i \in [\max(0, u_i - L),\, \max(u_i, 0) + L]$ with $L = {{tc_quad_L:.0f}}$ is split into panels of width $h = {{tc_quad_hy:.1f}}$ in $g_i$, mapped to $y_i$ by the cube root, and each panel gets a Gauss&ndash;Legendre rule of order ${{tc_quad_order:.0f}}$. When the range reaches $g_i = 0$, its first panel is replaced by ${{tc_quad_ngrade:.0f}}$ panels whose widths shrink geometrically by a factor ${{tc_quad_ratio:.2f}}$ towards $y_i = 0$, because there the prior factor varies over ever shorter distances: $s_1 = y_1^2/y_2$ grows without bound as $y_2 \to 0$. The two axes are combined as a tensor product, about ${{tc_quad_nodes:.0f}}$ nodes per trial on average.

Truncating the limits has a bound. A point outside the box has $y_1$ or $y_2$ outside its range, and there that coordinate's sensor factor is below $e^{-L^2/2}$ while the other is at most $1$. The prior factor integrates over the whole quadrant to $\int\!\!\int e^{-\lambda_1 s_1 - \lambda_2 s_2}\,\mathrm{d}s_1\,\mathrm{d}s_2/3 = 1/(3\lambda_1\lambda_2)$, since $\mathrm{d}s_1\,\mathrm{d}s_2 = 3\,\mathrm{d}y_1\,\mathrm{d}y_2$. Bounding the two parts of the outside region separately, the part of [eq:tc-evidence-y] outside the box is at most $2 \cdot e^{-L^2/2}/(2\pi) = {{tc_trunc_abs:sci1}}$. The largest surprise in the sample at the true rates is ${{tc_nll_max:.4f}}$ nats, so relative to every trial's evidence the omitted part is below ${{tc_trunc_rel:sci1}}$. The bound covers the evidence; the moments have truncation errors of the same kind, which this calculation does not bound.

The quadrature error inside the box has no such bound. The evidence for it is that refining each setting on its own, over all ${{tc_n:.0f}}$ trials at the true rates, changes nothing that the lesson uses:

| Refinement | Nodes per trial | Surprise (nats) | Means (relative) | Standard deviations (relative) | Correlation (relative) | One-step rates |
|---|---:|---:|---:|---:|---:|---:|
| panel width $h$ halved to ${{tc_qv_hy:.2f}}$ | ${{tc_qc_hy_nodes:.0f}}$ | ${{tc_qc_hy_nll:sci1}}$ | ${{tc_qc_hy_m:sci1}}$ | ${{tc_qc_hy_sd:sci1}}$ | ${{tc_qc_hy_corr:sci1}}$ | ${{tc_qc_hy_rate:sci1}}$ |
| limits widened, $L = {{tc_qv_L:.0f}}$ | ${{tc_qc_L_nodes:.0f}}$ | ${{tc_qc_L_nll:sci1}}$ | ${{tc_qc_L_m:sci1}}$ | ${{tc_qc_L_sd:sci1}}$ | ${{tc_qc_L_corr:sci1}}$ | ${{tc_qc_L_rate:sci1}}$ |
| grading: ${{tc_qv_ngrade:.0f}}$ panels shrinking by ${{tc_qv_ratio:.1f}}$ | ${{tc_qc_grade_nodes:.0f}}$ | ${{tc_qc_grade_nll:sci1}}$ | ${{tc_qc_grade_m:sci1}}$ | ${{tc_qc_grade_sd:sci1}}$ | ${{tc_qc_grade_corr:sci1}}$ | ${{tc_qc_grade_rate:sci1}}$ |
| Gauss&ndash;Legendre order ${{tc_qv_order:.0f}}$ | ${{tc_qc_order_nodes:.0f}}$ | ${{tc_qc_order_nll:sci1}}$ | ${{tc_qc_order_m:sci1}}$ | ${{tc_qc_order_sd:sci1}}$ | ${{tc_qc_order_corr:sci1}}$ | ${{tc_qc_order_rate:sci1}}$ |

Each entry is the largest change over the ${{tc_n:.0f}}$ trials, absolute for the surprise and relative for the moments; the last column is the absolute change in the rates $1/\bar m_i$ that one M-step from $\lambda^{*}$ produces. The exact-EM fit was also recomputed with all four refinements at once: one further M-step from it under the refined integration moves it by ${{tc_ml_refined:sci1}}$. This is evidence of numerical convergence, the stability of the result as the numerical settings change; it is not an error bound.

## Three mean-field families

All three families below are products $Q(s_1, s_2) = q_1(s_1)\,q_2(s_2)$ on the positive quadrant. They differ in how much freedom each factor has.

### Exponential factors

The first family uses the prior's own form for each factor: $q_i$ is exponential with mean $m_i > 0$,
$$
q_i(s_i) \;=\; \frac{1}{m_i}\,e^{-s_i/m_i}, \qquad s_i > 0.
$$ {#tc-q-exp}

::: definition Coefficient of variation
For a positive random variable $X$ with mean $\mathbb{E}[X] > 0$, the **coefficient of variation** is the ratio of its standard deviation to its mean, $\mathrm{CV}[X] = \sqrt{\mathrm{Var}(X)}\,/\,\mathbb{E}[X]$. It is dimensionless.
:::

::: derivation Moments, divergence and free energy of the Exponential product
1. Moments. For an integer $r \ge 1$, integrating by parts with $v = -e^{-s/m}$,
$$
\int_0^\infty s^r\,\frac{e^{-s/m}}{m}\,\mathrm{d}s
\;=\; \Big[-s^r e^{-s/m}\Big]_0^\infty + r\int_0^\infty s^{r-1}e^{-s/m}\,\mathrm{d}s
\;=\; r\,m\int_0^\infty s^{r-1}\,\frac{e^{-s/m}}{m}\,\mathrm{d}s .
$$
The boundary term vanishes at both ends. Starting from $\mathbb{E}_q[s^0] = 1$, the recursion gives
$$
\mathbb{E}_{q}[s^r] \;=\; r!\,m^r,
$$ {#tc-exp-moments}
so $\mathbb{E}_q[s] = m$, $\mathbb{E}_q[s^2] = 2m^2$, $\mathbb{E}_q[s^4] = 24m^4$, $\mathrm{Var}_q(s) = m^2$, and $\mathrm{CV}[s] = 1$ whatever $m$ is.

2. Complexity. With $\ln q_i(s) = -\ln m_i - s/m_i$ and $\ln P(s \mid \lambda_i) = \ln\lambda_i - \lambda_i s$,
$$
D_{\mathrm{KL}}\big[q_i \,\|\, P(\cdot \mid \lambda_i)\big]
\;=\; \mathbb{E}_{q_i}\!\left[-\ln m_i - \frac{s}{m_i} - \ln\lambda_i + \lambda_i s\right]
\;=\; -\ln m_i - 1 - \ln\lambda_i + \lambda_i m_i .
$$ {#tc-exp-kl}

3. Accuracy. Under the product, $s_1$ and $s_2$ are independent, so expectations of products of powers factorise. Expanding the square,
$$
\mathbb{E}_Q\big[(u_1 - s_2 s_1^2)^2\big]
\;=\; u_1^2 - 2u_1\,\mathbb{E}[s_2]\,\mathbb{E}[s_1^2] + \mathbb{E}[s_2^2]\,\mathbb{E}[s_1^4]
\;=\; u_1^2 - 4u_1 m_1^2 m_2 + 2m_2^2 \cdot 24 m_1^4 ,
$$
and the same with the indices exchanged for the second sensor. Adding the two complexities, half of each expected squared residual, and $\ln 2\pi$ from the two Gaussian normalisers,
$$
\begin{aligned}
\mathcal{F}_{\mathrm{Exp}}(m_1, m_2) \;=\;& \sum_{i=1}^{2}\big(-\ln m_i - 1 - \ln\lambda_i + \lambda_i m_i\big) + \ln 2\pi \\
&+ \tfrac12\big(u_1^2 - 4u_1 m_1^2 m_2 + 48\,m_1^4 m_2^2\big) + \tfrac12\big(u_2^2 - 4u_2 m_1 m_2^2 + 48\,m_1^2 m_2^4\big).
\end{aligned}
$$ {#tc-f-exp}
:::

### A fixed coefficient of variation

The coefficient $48$ in [eq:tc-f-exp] comes from $\mathbb{E}[s_2^2]\,\mathbb{E}[s_1^4] = 2 \times 24$, and it is where effect 1 enters. Write $\hat g_1 = \mathbb{E}_Q[s_2 s_1^2] = 2m_1^2 m_2$ for the first sensor's predicted mean. Then $\mathbb{E}_Q[(s_2 s_1^2)^2] = 48 m_1^4 m_2^2 = 12\hat g_1^2$, so $\mathrm{Var}_Q(s_2 s_1^2) = 11\hat g_1^2$, and by $\mathbb{E}[(u - X)^2] = (u - \mathbb{E}X)^2 + \mathrm{Var}(X)$ the first sensor's term in [eq:tc-f-exp] is
$$
\tfrac12\,\mathbb{E}_Q\big[(u_1 - s_2 s_1^2)^2\big] \;=\; \tfrac12\big(u_1 - \hat g_1\big)^2 + \tfrac{11}{2}\,\hat g_1^2 .
$$ {#tc-locked}
Each exponential factor has coefficient of variation $1$, so the predicted sensor output has coefficient of variation $\sqrt{11} = {{exp_pred_cv:.4f}}$ for every choice of $m_1, m_2$. Its variance cannot be reduced without reducing its mean. As a function of $\hat g_1 > 0$ alone, [eq:tc-locked] has its minimum at $\hat g_1 = u_1/12$ when $u_1 > 0$, not at $u_1$. When $u_1 \le 0$, its infimum over $\hat g_1 > 0$ is approached as $\hat g_1$ tends to zero. The additional variance penalty $\tfrac{11}{2}\hat g_1^2$ favours smaller predicted means. The prior term and the second sensor change where the minimum of $\mathcal{F}_{\mathrm{Exp}}$ lies.

On the representative trial at the true rates, minimising [eq:tc-f-exp] gives means $({{rep_exp_m1:.4f}}, {{rep_exp_m2:.4f}})$ against the exact $({{rep_ex_m1:.4f}}, {{rep_ex_m2:.4f}})$. The predicted first sensor output has mean ${{rep_exp_pred_m1:.4f}}$ and standard deviation ${{rep_exp_pred_sd1:.4f}}$, for an observation of ${{rep_u1:.1f}}$. The free energy is ${{rep_exp_F:.4f}}$ nats against an exact surprise of ${{rep_nll:.4f}}$, a divergence of ${{rep_exp_gap:.4f}}$ nats from the posterior.

[eq:tc-f-exp] is minimised in $a_i = \ln m_i$, which keeps $m_i$ positive without a constraint, by a modified Newton method. At each step the Hessian's eigenvalues are replaced by their absolute values, floored at $10^{-8}$ times the largest, so that the step is a descent direction; each step is limited to $2$ in every coordinate, a factor of $e^2$ in $m_i$; and the step length is halved until the Armijo condition (a decrease of at least $10^{-4}$ times the predicted decrease) holds. A trial stops for one of three reasons: the largest gradient component is below ${{tc_tol_exp:sci0}}$; the Newton decrement $\mathbf{g}^{\top}\mathbf{H}^{-1}\mathbf{g}$ is at the rounding level of $\mathcal{F}$; or no step along the Newton direction lowers $\mathcal{F}$ in floating point. The first E-step takes the best of ${{tc_n_exp_starts:.0f}}$ starts, $a_1, a_2 \in \lbrace {{tc_exp_start_lo:.0f}}, 0, {{tc_exp_start_hi:.1f}} \rbrace$. The analytic gradient and Hessian agree with central differences to a relative ${{tc_exp_grad_relerr:sci1}}$ and ${{tc_exp_hess_relerr:sci1}}$.

### Gamma factors

Effect 1 comes from having one parameter per factor. The Gamma family adds a second: $q_i$ has mean $m_i > 0$ and shape $k_i > 0$,
$$
q_i(s_i) \;=\; \frac{(k_i/m_i)^{k_i}}{\Gamma(k_i)}\,s_i^{k_i - 1}\,e^{-k_i s_i/m_i}, \qquad s_i > 0,
$$ {#tc-q-gam}
which is the Gamma density of [lesson:4.2] with rate $k_i/m_i$. Here $k_i$ is a shape, not a number of values. At $k_i = 1$ it is the exponential [eq:tc-q-exp].

::: derivation Moments, divergence and free energy of the Gamma product
1. Moments. With $r_q = k/m$ and $\int_0^\infty s^{k + j - 1}e^{-r_q s}\,\mathrm{d}s = \Gamma(k + j)/r_q^{k + j}$,
$$
\mathbb{E}_q[s^j] \;=\; \frac{r_q^{k}}{\Gamma(k)}\cdot\frac{\Gamma(k + j)}{r_q^{k + j}} \;=\; \frac{\Gamma(k + j)}{\Gamma(k)}\left(\frac{m}{k}\right)^{j} \;=\; m^j\prod_{l=0}^{j-1}\frac{k + l}{k},
$$
using $\Gamma(z + 1) = z\Gamma(z)$ $j$ times. So $\mathbb{E}_q[s] = m$ and
$$
\mathbb{E}_q[s^2] \;=\; m^2 c_2(k),\quad c_2(k) = \frac{k + 1}{k};
\qquad
\mathbb{E}_q[s^4] \;=\; m^4 c_4(k),\quad c_4(k) = \frac{(k + 1)(k + 2)(k + 3)}{k^3}.
$$ {#tc-gam-moments}
Then $\mathrm{Var}_q(s) = m^2/k$ and $\mathrm{CV}[s] = 1/\sqrt{k}$, which the shape sets freely. At $k = 1$, $c_2 = 2$ and $c_4 = 24$, the exponential values.

2. Complexity. [lesson:4.2] derived $\mathbb{E}[\ln s] = \psi(k) - \ln r_q$ for a Gamma density with shape $k$ and rate $r_q$, where $\psi$ is the digamma function. Here $r_q = k/m$, so $\mathbb{E}_q[\ln s] = \psi(k) - \ln k + \ln m$. With $\ln q(s) = k\ln(k/m) - \ln\Gamma(k) + (k - 1)\ln s - ks/m$,
$$
\begin{aligned}
D_{\mathrm{KL}}\big[q_i \,\|\, P(\cdot \mid \lambda_i)\big]
&= \mathbb{E}_{q_i}\!\Big[k_i\ln\tfrac{k_i}{m_i} - \ln\Gamma(k_i) + (k_i - 1)\ln s - \tfrac{k_i s}{m_i} - \ln\lambda_i + \lambda_i s\Big] \\
&= k_i\ln\tfrac{k_i}{m_i} - \ln\Gamma(k_i) + (k_i - 1)\big(\psi(k_i) - \ln k_i + \ln m_i\big) - k_i - \ln\lambda_i + \lambda_i m_i \\
&= -\ln\Gamma(k_i) + (k_i - 1)\psi(k_i) + \ln k_i - k_i - \ln m_i - \ln\lambda_i + \lambda_i m_i .
\end{aligned}
$$ {#tc-gam-kl}
The last line collects $k_i\ln k_i - (k_i - 1)\ln k_i = \ln k_i$ and $-k_i\ln m_i + (k_i - 1)\ln m_i = -\ln m_i$. At $k_i = 1$, $\ln\Gamma(1) = 0$ and the line is [eq:tc-exp-kl].

3. Accuracy. By independence and [eq:tc-gam-moments],
$$
\mathbb{E}_Q\big[(u_1 - s_2 s_1^2)^2\big] \;=\; u_1^2 - 2u_1\,m_2\,m_1^2 c_2(k_1) + m_2^2 c_2(k_2)\,m_1^4 c_4(k_1),
$$
and the same with the indices exchanged. The free energy $\mathcal{F}_{\mathrm{Gam}}(m_1, m_2, k_1, k_2)$ is the sum of the two divergences, $\ln 2\pi$, and half of each expected squared residual.
:::

As $k_1, k_2 \to \infty$ with the means fixed, $c_2$ and $c_4$ tend to $1$ and the expected squared residual tends to $(u_1 - m_2 m_1^2)^2$: the predicted output's variance can be made as small as the divergences allow, so effect 1 is gone. On the representative trial the shapes come out at $({{rep_gam_k1:.2f}}, {{rep_gam_k2:.2f}})$, the means at $({{rep_gam_m1:.4f}}, {{rep_gam_m2:.4f}})$ against the exact $({{rep_ex_m1:.4f}}, {{rep_ex_m2:.4f}})$, and the divergence from the posterior is ${{rep_gam_gap:.4f}}$ nats, down from ${{rep_exp_gap:.4f}}$.

$\mathcal{F}_{\mathrm{Gam}}$ is minimised in $(\ln m_1, \ln m_2, \ln k_1, \ln k_2)$ by the same Newton method, with a gradient tolerance of ${{tc_tol_gam:sci0}}$. The gradient is analytic and agrees with central differences to a relative ${{tc_gam_grad_relerr:sci1}}$; the Hessian is the central difference of the analytic gradient. The first E-step takes the best of ${{tc_n_gam_starts:.0f}}$ starts, $\ln m_1 = \ln m_2 \in \{-1, 0, 1\}$ and $\ln k_1 = \ln k_2 \in \{0, 2, 5\}$. The parameters are unbounded, so the shapes can take any positive value; at the true rates the largest shape in the sample is ${{tc_gam_kmax:.0f}}$, on the thin-ridge trial ${{tc_big_i:.0f}}$. With both shapes set to $1$, $\mathcal{F}_{\mathrm{Gam}}$ agrees with $\mathcal{F}_{\mathrm{Exp}}$ to ${{tc_gam_k1_equals_exp:sci1}}$.

### Free-form factors

The third family puts no restriction on either factor beyond independence. By the free-form update of [lesson:4.2], the optimal $q_1$ given $q_2$ is $\ln q_1(s_1) = \mathbb{E}_{q_2}[\ln P(u, s \mid \lambda)] + \text{const}$.

::: derivation The free-form factors on the two-cause model
Expanding the squares in [eq:tc-logjoint] and keeping the terms that contain $s_1$:
$$
-\lambda_1 s_1 + u_1 s_2 s_1^2 - \tfrac12 s_2^2 s_1^4 + u_2 s_2^2 s_1 - \tfrac12 s_2^4 s_1^2 .
$$
Taking the expectation over $q_2$ and collecting powers of $s_1$,
$$
\ln q_1(s_1) \;=\; -\tfrac12\,\mathbb{E}_{q_2}[s_2^2]\,s_1^4 + \Big(u_1\,\mathbb{E}_{q_2}[s_2] - \tfrac12\,\mathbb{E}_{q_2}[s_2^4]\Big)s_1^2 + \Big(u_2\,\mathbb{E}_{q_2}[s_2^2] - \lambda_1\Big)s_1 + \text{const}, \qquad s_1 > 0,
$$ {#tc-ff-q1}
and the same with the indices exchanged (and $u_1, u_2$ exchanged) for $q_2$. The coefficient of $s_1^4$ is negative, so the density is normalisable. It is not a Gamma density: its logarithm is a quartic polynomial in $s_1$ with no $\ln s_1$ term. The update for $q_1$ uses $q_2$ only through $\mathbb{E}[s_2]$, $\mathbb{E}[s_2^2]$ and $\mathbb{E}[s_2^4]$, the variational messages of [lesson:4.2] for this model.
:::

Each factor is a density on $(0, \infty)$ of the form $\exp(a s^4 + b s^2 + c\,s)$ with $a < 0$. Its normaliser and moments are computed by one-dimensional quadrature: a scan of 256 points locates the interval where the exponent is within 60 nats of its maximum, and 32 panels of 8-point Gauss&ndash;Legendre cover that interval. Coordinate ascent updates $q_1$ then $q_2$, starting from $q_2$ equal to the prior, and a trial stops when one sweep changes both means by a relative amount below ${{tc_tol_ff:sci0}}$. The free energy is checked not to rise from sweep to sweep. At the true rates the median trial takes ${{tc_ff_sweeps_med:.0f}}$ sweeps and the slowest ${{tc_ff_sweeps_max:.0f}}$. Starting instead from the moments of a point mass at $s_2 = 5$ changes the means by at most a relative ${{tc_ff_start_dm:sci1}}$ and the free energy by at most ${{tc_ff_start_dF:sci1}}$ nats; doubling the panels and scan points and widening the window to 80 nats changes them by at most ${{tc_ff_quad_dm:sci1}}$ and ${{tc_ff_quad_dF:sci1}}$.

## Stationarity, warm starts and the monotone free energy

In variational EM every E-step after the first starts from the previous iteration's $Q_n$, and the new $Q_n$ replaces the old one only where its free energy is no higher. For the free-form family this holds automatically, since coordinate ascent from the old $Q_n$ never raises $\mathcal{F}$. So each E-step leaves $\mathcal{F}_{\mathrm{tot}}$ no higher, as the proof in [lesson:4.4] requires, and the M-step [eq:tc-mstep] is an exact minimisation in $\lambda$. Both are checked in the computation: after every E-step and every M-step, $\mathcal{F}_{\mathrm{tot}}$ is compared with its value before the step, and the run stops with an error if it rose by more than $10^{-10}$ of its size. No run failed this check. The averaging rule of [eq:tc-am-hm] is checked after its E-steps only, since it is not a minimisation in $\lambda$. A run ends when one iteration changes both rates by a relative amount below ${{tc_tol_em:sci0}}$; the table below gives the number of iterations. In the last iterations the rates change by amounts close to the error left by the E-steps, so on another machine a run can take a few iterations more or fewer, and the Gamma run, whose E-steps stop at the looser gradient tolerance, more than ten. The fitted rates agree to the precision shown.

The closed-form solvers were checked for stationarity where the rates were fixed:

| Family, rates | Largest gradient component | Trials above the tolerance | Further decrease a Newton step predicts on those trials (nats) |
|---|---:|---:|---:|
| Exponential, $\lambda^{*}$ | ${{tc_gmax_exp_pin:sci1}}$ | ${{tc_gover_exp_pin:.0f}}$ | ${{tc_gpred_exp_pin:sci1}}$ |
| Exponential, fitted | ${{tc_gmax_exp_fit:sci1}}$ | ${{tc_gover_exp_fit:.0f}}$ | ${{tc_gpred_exp_fit:sci1}}$ |
| Gamma, $\lambda^{*}$ | ${{tc_gmax_gam_pin:sci1}}$ | ${{tc_gover_gam_pin:.0f}}$ | none |
| Gamma, fitted | ${{tc_gmax_gam_fit:sci1}}$ | ${{tc_gover_gam_fit:.0f}}$ | ${{tc_gpred_gam_fit:sci1}}$ |

Trials above the gradient tolerance stopped by the second or third rule. From there the quadratic model predicts a further decrease in $\mathcal{F}$ smaller than the amount in the last column, which is at the rounding level of free energies of this size; this is a diagnostic, not an error bound. At the fitted rates, a fresh multi-start solve finds a free energy lower than the warm-started one by at most ${{tc_exp_fit_multistart:sci1}}$ nats per trial for the Exponential family and ${{tc_gam_fit_multistart:sci1}}$ for the Gamma family, and coordinate ascent from the second start differs from the warm-started free-form result by at most ${{tc_ff_fit_multistart:sci1}}$. These checks show that the solvers reached the best minimum their starts found; they cannot rule out a lower minimum that no start reached.

## Factorisation narrows the factors

Effect 4 is what remains when the factors are free and only their independence is imposed. Its core is a calculation that can be done exactly for a Gaussian target. Write $\Pi = \Sigma^{-1}$ for the precision matrix of a Gaussian with covariance $\Sigma$, as in [week:2].

::: theorem Free-form mean-field approximation of a Gaussian
Let $P(x) = \mathcal{N}(x; m_P, \Sigma)$ on $\mathbb{R}^K$, with mean vector $m_P$ and $\Sigma$ positive definite and $\Pi = \Sigma^{-1}$. Over products $Q(x) = \prod_i q_i(x_i)$, $D_{\mathrm{KL}}[Q \,\|\, P]$ is minimised uniquely by
$$
q_i(x_i) \;=\; \mathcal{N}\big(x_i;\ m_{P,i},\ 1/\Pi_{ii}\big), \qquad i = 1, \dots, K,
$$ {#tc-mf-gauss}
and the minimum is $\tfrac12\big(\ln\det\Sigma + \sum_i \ln\Pi_{ii}\big)$. For $K = 2$ with standard deviations $\sigma_1, \sigma_2$ and correlation $c$, each factor's standard deviation is $\sqrt{1 - c^2}$ times $\sigma_i$, and the minimum divergence is $-\tfrac12\ln(1 - c^2)$.
:::

::: derivation Proof
1. Every optimal factor is Gaussian. Here $\ln P(x) = -\tfrac12(x - m_P)^{\top}\Pi(x - m_P) + \text{const}$. The terms that contain $x_i$ are
$$
-\tfrac12\Pi_{ii}(x_i - m_{P,i})^2 - (x_i - m_{P,i})\sum_{j \ne i}\Pi_{ij}(x_j - m_{P,j}),
$$
using the symmetry of $\Pi$. Averaging over the other factors, with $m_j = \mathbb{E}_{q_j}[x_j]$, the free-form update of [lesson:4.2] gives
$$
\ln q_i(x_i) \;=\; -\tfrac12\Pi_{ii}(x_i - m_{P,i})^2 - (x_i - m_{P,i})\sum_{j \ne i}\Pi_{ij}(m_j - m_{P,j}) + \text{const},
$$
a quadratic in $x_i$ with leading coefficient $-\tfrac12\Pi_{ii} < 0$. So the update returns a Gaussian with variance $1/\Pi_{ii}$, whatever the other factors are, provided their means are finite (if they are not, $\mathbb{E}_Q[(x - m_P)^{\top}\Pi(x - m_P)]$ and with it the divergence is infinite). Start from any product of finite divergence and update each factor once in turn: the divergence does not rise, and afterwards every factor is Gaussian. So the minimum over all products equals the minimum over products of Gaussians.

2. The minimum over Gaussian products. For $Q = \mathcal{N}(m, V)$ with $V = \mathrm{diag}(v_1, \dots, v_K)$, the divergence between two Gaussians is
$$
D_{\mathrm{KL}}[Q \,\|\, P] \;=\; \tfrac12\Big(\mathrm{tr}(\Pi V) + (m - m_P)^{\top}\Pi(m - m_P) - K + \ln\det\Sigma - \ln\det V\Big)
\;=\; \tfrac12\sum_{i}\big(\Pi_{ii}v_i - \ln v_i\big) + \tfrac12(m - m_P)^{\top}\Pi(m - m_P) + \tfrac12\big(\ln\det\Sigma - K\big).
$$
The mean term is a positive definite quadratic, zero only at $m = m_P$. Each $\Pi_{ii}v_i - \ln v_i$ has derivative $\Pi_{ii} - 1/v_i$, zero only at $v_i = 1/\Pi_{ii}$, and is convex in $v_i$. So the minimiser is unique and is [eq:tc-mf-gauss]. At it $\mathrm{tr}(\Pi V) = \sum_i \Pi_{ii}/\Pi_{ii} = K$, so the minimum is $\tfrac12(\ln\det\Sigma - \ln\det V) = \tfrac12(\ln\det\Sigma + \sum_i\ln\Pi_{ii})$.

3. Two dimensions. With $\Sigma = \begin{pmatrix}\sigma_1^2 & c\sigma_1\sigma_2 \\ c\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}$, $\det\Sigma = \sigma_1^2\sigma_2^2(1 - c^2)$, and the diagonal of the inverse is $\Pi_{11} = \sigma_2^2/\det\Sigma = 1/\big(\sigma_1^2(1 - c^2)\big)$ and $\Pi_{22} = 1/\big(\sigma_2^2(1 - c^2)\big)$. So $1/\Pi_{ii} = \sigma_i^2(1 - c^2)$, and the minimum is
$$
\tfrac12\Big(\ln\big(\sigma_1^2\sigma_2^2(1 - c^2)\big) - \ln\big(\sigma_1^2(1 - c^2)\big) - \ln\big(\sigma_2^2(1 - c^2)\big)\Big) \;=\; -\tfrac12\ln(1 - c^2).
$$ {#tc-mf-gap}
:::

The variance $1/\Pi_{ii}$ is the conditional variance of $x_i$ given all the other coordinates, not its marginal variance $\Sigma_{ii}$. The product has marginal variances no larger than the target's, with equality for a coordinate independent of the others, as the reverse divergence of [lesson:1.4] predicts: reverse KL penalises assigning appreciable mass where the posterior density is small. For a Gaussian the means are exact, so factorisation costs width and not location.

::: fig Contours enclosing {{gauss_mass_1sd:.0f}} and {{gauss_mass_2sd:.0f}} per cent of the probability (one and two standard deviations along each axis of the ellipse) of a two-dimensional Gaussian with unit variances and correlation equal to the representative trial's posterior correlation, ${{rep_ex_corr:.2f}}$ (green), and of its free-form mean-field approximation (blue), a circle of radius $\sqrt{1 - c^2}$ times the green ellipse's marginal standard deviation.
{{fig_ellipse_svg}}
:::

Apply this to the representative trial, treating its posterior as roughly Gaussian with its own correlation $c = {{rep_ex_corr:.4f}}$. The Gaussian result predicts standard deviations reduced by the factor $\sqrt{1 - c^2} = {{rep_sqrt_one_minus_c2:.4f}}$ and a divergence of $-\tfrac12\ln(1 - c^2) = {{rep_gauss_gap:.4f}}$ nats. The free-form factors have standard deviations $({{rep_ff_sd1:.4f}}, {{rep_ff_sd2:.4f}})$, which are ${{rep_ff_ratio1:.4f}}$ and ${{rep_ff_ratio2:.4f}}$ of the exact $({{rep_ex_sd1:.4f}}, {{rep_ex_sd2:.4f}})$, and the divergence is ${{rep_ff_gap:.4f}}$ nats. Both are further from the posterior than the Gaussian calculation predicts, because the posterior is a curved ridge and not an ellipse; the Gaussian result gives the direction and the order of size, not the value. The Gamma factors are narrower still, at ${{rep_gam_ratio1:.4f}}$ and ${{rep_gam_ratio2:.4f}}$ of the exact standard deviations.

On the curved ridge the means move as well. On the representative trial the free-form means are $({{rep_ff_m1:.4f}}, {{rep_ff_m2:.4f}})$ against the exact $({{rep_ex_m1:.4f}}, {{rep_ex_m2:.4f}})$, both lower. Across the sample at the true rates, the free-form mean of $s_1$ averages ${{tc_mr_free_1:.3f}}$ of the exact posterior mean and is lower on ${{tc_low_free_1:.1f}}$ per cent of trials; for $s_2$ the figures are ${{tc_mr_free_2:.3f}}$ and ${{tc_low_free_2:.1f}}$ per cent. For the Gamma family they are ${{tc_mr_gam_1:.3f}}$ and ${{tc_low_gam_1:.1f}}$ per cent, and ${{tc_mr_gam_2:.3f}}$ and ${{tc_low_gam_2:.1f}}$ per cent; the representative trial's Gamma mean of $s_1$, ${{rep_gam_m1:.4f}}$ against ${{rep_ex_m1:.4f}}$, is one of the minority that come out higher. The downward shift is measured here, not derived: the Gaussian theorem says nothing about it, since there the means are exact. Its consequence for learning follows from [eq:tc-mstep], where a lower average mean is a higher rate.

## The closed loop

Variational EM alternates the two steps, so its rates follow the map
$$
\lambda^{(t)} \;=\; \Lambda\big(\lambda^{(t-1)}\big), \qquad \Lambda_i(\lambda) \;=\; \frac{1}{\bar m_i(\lambda)},
$$ {#tc-em-map}
where $\bar m_i(\lambda)$ is the average recognition mean after an E-step at rates $\lambda$. Write $J$ for its Jacobian matrix, $J_{ij} = \partial\Lambda_i/\partial\lambda_j$.

::: theorem The diagonal of the EM map's Jacobian is non-negative
For exact EM,
$$
\frac{\partial\Lambda_i}{\partial\lambda_j} \;=\; \frac{1}{N\bar m_i^2}\sum_{n=1}^{N}\mathrm{Cov}\big(s_i, s_j \mid u_n, \lambda\big),
$$ {#tc-jac-exact}
so $J_{ii}$ is the average posterior variance of $s_i$ divided by $\bar m_i^2$, which is positive. For a parametric family in which trial $n$'s E-step returns parameters $\mathbf{x}_n(\lambda)$ at a minimum of $\mathcal{F}_n(\mathbf{x}; \lambda)$ with positive definite Hessian $\mathbf{H}_n$,
$$
\frac{\partial\Lambda_i}{\partial\lambda_j} \;=\; \frac{1}{N\bar m_i^2}\sum_{n=1}^{N}\mathbf{h}_{n,i}^{\top}\mathbf{H}_n^{-1}\mathbf{h}_{n,j}, \qquad \mathbf{h}_{n,i} = \nabla_{\mathbf{x}}\,\mathbb{E}_{Q_{\mathbf{x}}}[s_i]\Big|_{\mathbf{x} = \mathbf{x}_n},
$$ {#tc-jac-param}
and again $J_{ii} \ge 0$. Here $\mathbf{x}_n$ is a local symbol for the parameters of $Q_n$, such as $(\ln m_1, \ln m_2, \ln k_1, \ln k_2)$ for the Gamma family.
:::

::: derivation Proof
1. From [eq:tc-em-map], $\partial\Lambda_i/\partial\lambda_j = -\bar m_i^{-2}\,\partial\bar m_i/\partial\lambda_j$, and $\bar m_i$ is an average over trials, so it suffices to differentiate one trial's recognition mean.

2. Exact posterior. By [eq:tc-logjoint], $P(s \mid u_n, \lambda) = e^{-\lambda_1 s_1 - \lambda_2 s_2}L_n(s)/Z_n(\lambda)$, where $L_n$ collects the factors free of $\lambda$ and $Z_n(\lambda) = \int e^{-\lambda_1 s_1 - \lambda_2 s_2}L_n(s)\,\mathrm{d}s$. Differentiating under the integral, $\partial Z_n/\partial\lambda_j = -Z_n\,\mathbb{E}[s_j \mid u_n]$, and
$$
\frac{\partial}{\partial\lambda_j}\mathbb{E}[s_i \mid u_n]
\;=\; \frac{\partial}{\partial\lambda_j}\frac{\int s_i\,e^{-\lambda\cdot s}L_n\,\mathrm{d}s}{Z_n}
\;=\; -\mathbb{E}[s_i s_j \mid u_n] + \mathbb{E}[s_i \mid u_n]\,\mathbb{E}[s_j \mid u_n]
\;=\; -\mathrm{Cov}(s_i, s_j \mid u_n).
$$
Averaging over trials and multiplying by $-\bar m_i^{-2}$ gives [eq:tc-jac-exact].

3. Parametric family. As in the M-step derivation, $\lambda$ enters $\mathcal{F}_n$ only through $\sum_j\big(-\ln\lambda_j + \lambda_j\,\mathbb{E}_{Q_{\mathbf{x}}}[s_j]\big)$, so $\partial_{\lambda_j}\nabla_{\mathbf{x}}\mathcal{F}_n = \mathbf{h}_{n,j}$. The E-step solves $\nabla_{\mathbf{x}}\mathcal{F}_n(\mathbf{x}_n(\lambda); \lambda) = 0$. Since $\mathbf{H}_n$ is invertible, the implicit function theorem makes $\mathbf{x}_n$ differentiable in $\lambda$, and differentiating the stationarity condition in $\lambda_j$ gives
$$
\mathbf{H}_n\,\frac{\partial\mathbf{x}_n}{\partial\lambda_j} + \mathbf{h}_{n,j} \;=\; 0
\quad\Longrightarrow\quad
\frac{\partial\mathbf{x}_n}{\partial\lambda_j} \;=\; -\mathbf{H}_n^{-1}\mathbf{h}_{n,j}.
$$
By the chain rule, $\partial\,\mathbb{E}_{Q_n}[s_i]/\partial\lambda_j = \mathbf{h}_{n,i}^{\top}\,\partial\mathbf{x}_n/\partial\lambda_j = -\mathbf{h}_{n,i}^{\top}\mathbf{H}_n^{-1}\mathbf{h}_{n,j}$, and step 1 gives [eq:tc-jac-param]. For $i = j$ each term is a quadratic form in the positive definite $\mathbf{H}_n^{-1}$, so it is non-negative.
:::

The free-form family is covered by the second case: its optimal factors lie in the family with log densities $a s^4 + b s^2 + c\,s$ ([eq:tc-ff-q1]), so the free-form optimum is also the optimum over that six-parameter family, and the theorem applies wherever that optimum is nondegenerate.

In words: raising $\lambda_i$ makes large values of $s_i$ less probable a priori, every trial's recognition mean of $s_i$ falls, and the next M-step raises $\lambda_i$ again. For exact EM the off-diagonal entries have the sign of the posterior covariance, which is negative here, so raising one rate lowers the other. At the true rates, [eq:tc-jac-exact] gives
$$
J_{\mathrm{exact}} \;=\; \begin{pmatrix} {{tc_jc_exact_11:.4f}} & {{tc_jc_exact_12:.4f}} \\ {{tc_jc_exact_21:.4f}} & {{tc_jc_exact_22:.4f}} \end{pmatrix},
$$
which agrees with a central-difference Jacobian of the map, computed by running the E-step at perturbed rates, to ${{tc_jc_vs_fd:sci1}}$.

### How the loop multiplies the one-step error

Let $\Delta = \Lambda(\lambda^{*}) - \lambda^{*}$ be the displacement of one M-step taken from the true rates; call $\Lambda(\lambda^{*})$ the one-step rates. A fixed point $\lambda^{\infty} = \Lambda(\lambda^{\infty})$ near $\lambda^{*}$ satisfies, to first order,
$$
\lambda^{\infty} \;=\; \Lambda(\lambda^{\infty}) \;\approx\; \Lambda(\lambda^{*}) + J\,(\lambda^{\infty} - \lambda^{*}) \;=\; \lambda^{*} + \Delta + J\,(\lambda^{\infty} - \lambda^{*}),
$$
with $J$ evaluated at $\lambda^{*}$, so
$$
\lambda^{\infty} - \lambda^{*} \;\approx\; (\mathbf{I} - J)^{-1}\Delta \;=\; \big(\mathbf{I} + J + J^2 + \cdots\big)\Delta,
$$ {#tc-gain}
the series converging when the spectral radius of $J$ is below $1$. The fixed point's displacement is the one-step displacement, plus its image after one more pass of the loop, and so on. This is effect 3. It needs a nonzero $\Delta$ to act on, and exact EM has one too: there $\Delta$ is the difference between the one-step rates and $\lambda^{*}$ on this finite sample, and [eq:tc-gain] carries it to the exact-EM fit.

| Family | One-step rates $\Lambda(\lambda^{*})$ | Fitted rates predicted by [eq:tc-gain] | Fitted rates | Diagonal of $(\mathbf{I} - J)^{-1}$ | Spectral radius of $J$ at the fit |
|---|---:|---:|---:|---:|---:|
| exact | $({{tc_pin_exact_1:.4f}}, {{tc_pin_exact_2:.4f}})$ | $({{tc_pred_exact_1:.4f}}, {{tc_pred_exact_2:.4f}})$ | $({{tc_fit_exact_1:.4f}}, {{tc_fit_exact_2:.4f}})$ | $({{tc_gain_exact_11:.3f}}, {{tc_gain_exact_22:.3f}})$ | ${{tc_jf_exact_rho:.3f}}$ |
| free-form | $({{tc_pin_free_1:.4f}}, {{tc_pin_free_2:.4f}})$ | $({{tc_pred_free_1:.4f}}, {{tc_pred_free_2:.4f}})$ | $({{tc_fit_free_1:.4f}}, {{tc_fit_free_2:.4f}})$ | $({{tc_gain_free_11:.3f}}, {{tc_gain_free_22:.3f}})$ | ${{tc_jf_free_rho:.3f}}$ |
| Gamma | $({{tc_pin_gam_1:.4f}}, {{tc_pin_gam_2:.4f}})$ | $({{tc_pred_gam_1:.4f}}, {{tc_pred_gam_2:.4f}})$ | $({{tc_fit_gam_1:.4f}}, {{tc_fit_gam_2:.4f}})$ | $({{tc_gain_gam_11:.3f}}, {{tc_gain_gam_22:.3f}})$ | ${{tc_jf_gam_rho:.3f}}$ |
| Exponential | $({{tc_pin_exp_1:.4f}}, {{tc_pin_exp_2:.4f}})$ | $({{tc_pred_exp_1:.4f}}, {{tc_pred_exp_2:.4f}})$ | $({{tc_fit_exp_1:.4f}}, {{tc_fit_exp_2:.4f}})$ | $({{tc_gain_exp_11:.3f}}, {{tc_gain_exp_22:.3f}})$ | ${{tc_jf_exp_rho:.3f}}$ |

For the Exponential and Gamma families $J$ is computed from [eq:tc-jac-param], with the Hessian $\mathbf{H}_n$ of each trial's free energy at its E-step solution and no eigenvalue repair; the smallest eigenvalue of any $\mathbf{H}_n$ is ${{tc_jac_mineig:.3f}}$. Central differences of $\Lambda$ with steps from ${{tc_jac_fd_step_lo:.3f}}$ to ${{tc_jac_fd_step_hi:.2f}}$ of each rate agree with these to ${{tc_jac_fd_dev:sci1}}$. Much smaller steps would not: the E-step's stopping error and floating-point error are both divided by the step. For the exact and free-form families $J$ is computed by central differences of $\Lambda$, with steps of $10^{-5}$ of each rate; for exact EM these agree with [eq:tc-jac-exact] to ${{tc_jc_vs_fd:sci1}}$. For the first three families the linear prediction differs from the fitted rates by at most ${{tc_pred_err_max:.4f}}$. For the Exponential family both components of the one-step displacement exceed $1$, and the prediction differs by up to ${{tc_pred_err_exp:.3f}}$, as a first-order expansion may at that distance. The spectral radius at every fitted point is below $1$, so each fixed point attracts nearby iterates. In every row the fitted rates lie further from $\lambda^{*}$ than the one-step rates, in both components: closing the loop enlarged the displacement in every family, the exact one included.

## Separating the effects

Each row of the next table changes one thing relative to the row above it. The one-step columns use the true rates as input, so they show what the E-step and the averaging rule do before any feedback. The fitted columns run the loop to convergence from $\lambda^{(0)}$.

| Recognition family and rule | One-step rates | relative to exact (%) | Fitted rates | relative to exact EM (%) | Iterations |
|---|---:|---:|---:|---:|---:|
| exact posterior, M-step | $({{tc_pin_exact_1:.4f}}, {{tc_pin_exact_2:.4f}})$ | | $({{tc_fit_exact_1:.4f}}, {{tc_fit_exact_2:.4f}})$ | | ${{tc_its_exact:.0f}}$ |
| free-form product, M-step | $({{tc_pin_free_1:.4f}}, {{tc_pin_free_2:.4f}})$ | $({{tc_pinpct_free_1:+.1f}}, {{tc_pinpct_free_2:+.1f}})$ | $({{tc_fit_free_1:.4f}}, {{tc_fit_free_2:.4f}})$ | $({{tc_fitpct_free_1:+.1f}}, {{tc_fitpct_free_2:+.1f}})$ | ${{tc_its_free:.0f}}$ |
| Gamma product, M-step | $({{tc_pin_gam_1:.4f}}, {{tc_pin_gam_2:.4f}})$ | $({{tc_pinpct_gam_1:+.1f}}, {{tc_pinpct_gam_2:+.1f}})$ | $({{tc_fit_gam_1:.4f}}, {{tc_fit_gam_2:.4f}})$ | $({{tc_fitpct_gam_1:+.1f}}, {{tc_fitpct_gam_2:+.1f}})$ | ${{tc_its_gam:.0f}}$ |
| Exponential product, M-step | $({{tc_pin_exp_1:.4f}}, {{tc_pin_exp_2:.4f}})$ | $({{tc_pinpct_exp_1:+.1f}}, {{tc_pinpct_exp_2:+.1f}})$ | $({{tc_fit_exp_1:.4f}}, {{tc_fit_exp_2:.4f}})$ | $({{tc_fitpct_exp_1:+.1f}}, {{tc_fitpct_exp_2:+.1f}})$ | ${{tc_its_exp:.0f}}$ |
| Exponential product, average of rates | $({{tc_pin_am_1:.4f}}, {{tc_pin_am_2:.4f}})$ | $({{tc_pinpct_am_1:+.1f}}, {{tc_pinpct_am_2:+.1f}})$ | $({{tc_fit_am_1:.4f}}, {{tc_fit_am_2:.4f}})$ | $({{tc_fitpct_am_1:+.1f}}, {{tc_fitpct_am_2:+.1f}})$ | ${{tc_its_am:.0f}}$ |
| Exponential product, online rule [eq:tc-online] | | | $({{tc_ema_1:.4f}}, {{tc_ema_2:.4f}})$ after one pass | | |

The one-step comparisons are relative to the exact one-step rates and the fitted ones relative to the exact-EM fit, so neither mixes a one-step value with a fitted one. Read down the table:

- Factorisation alone (free-form against exact) raises the one-step rates by ${{tc_pinpct_free_1:.1f}}$ and ${{tc_pinpct_free_2:.1f}}$ per cent and the fitted rates by ${{tc_fitpct_free_1:.1f}}$ and ${{tc_fitpct_free_2:.1f}}$ per cent. This is effect 4.
- Restricting each factor to a Gamma density (Gamma against free-form) adds ${{tc_gvf_pin_1:.1f}}$ and ${{tc_gvf_pin_2:.1f}}$ per cent to the one-step rates and ${{tc_gvf_fit_1:.1f}}$ and ${{tc_gvf_fit_2:.1f}}$ per cent to the fitted ones. Two parameters per factor come close to the free-form result on this model.
- Fixing the coefficient of variation at $1$ (Exponential against Gamma) adds ${{tc_evg_pin_1:.0f}}$ and ${{tc_evg_pin_2:.0f}}$ per cent to the one-step rates and ${{tc_evg_fit_1:.0f}}$ and ${{tc_evg_fit_2:.0f}}$ per cent to the fitted ones. This is effect 1, and it is the largest.
- Averaging rates instead of taking the M-step (the last two Exponential rows) adds ${{tc_am_over_hm_1:.1f}}$ and ${{tc_am_over_hm_2:.1f}}$ per cent in one step and ${{tc_amfit_over_hm_1:.1f}}$ and ${{tc_amfit_over_hm_2:.1f}}$ per cent at the fixed point. This is effect 2. The averaging rule is not a minimisation of $\mathcal{F}_{\mathrm{tot}}$, and in its run $\mathcal{F}_{\mathrm{tot}}$ rose on ${{tc_rises_am:.0f}}$ of the ${{tc_its_am:.0f}}$ iterations.
- Going from the one-step columns to the fitted columns in any row is effect 3, the loop of the previous section.

The comparisons are differences between neighbouring rows, and they do not add: the loop multiplies whatever displacement the row above it leaves, so the share of the final error that belongs to each effect depends on the order in which they are switched on. What the table shows is that each effect is present, its direction on this sample, and its size here.

::: fig Learned rates $(\lambda_1, \lambda_2)$ at every iteration of exact EM, of variational EM with the three families under the M-step, and of the Exponential family under the average of rates (dashed), with the online rule [eq:tc-online] every fifth trial, all from the start $\lambda^{(0)}$ (hollow circle). The cross is the true rate pair $\lambda^{*}$. Filled circles mark where each run ended.
{{fig_rates_svg}}
:::

The coefficients of variation tell the same story from the side of $Q$. Averaged over trials, $q(s_1)$ has coefficient of variation ${{tc_cv1_pin_exact:.4f}}$ under the exact posterior at the true rates, ${{tc_cv1_pin_free:.4f}}$ under the free-form product, ${{tc_cv1_pin_gam:.4f}}$ under the Gamma product and exactly $1$ under the Exponential product. At the fitted rates these are ${{tc_cv1_fit_exact:.4f}}$, ${{tc_cv1_fit_free:.4f}}$, ${{tc_cv1_fit_gam:.4f}}$ and $1$, computed from the $Q$ of the last E-step of each run. The average divergence from the posterior per trial at the true rates is ${{tc_gap_free:.4f}}$ nats for the free-form product, ${{tc_gap_gam:.4f}}$ for the Gamma product and ${{tc_gap_exp:.2f}}$ for the Exponential product.

The Exponential divergence is very uneven across trials. Its median is ${{tc_gap_exp_median:.4f}}$ nats, the largest single trial contributes ${{tc_gap_exp_top1:.1f}}$ per cent of the total, and the largest ten contribute ${{tc_gap_exp_top10:.1f}}$ per cent. These are trials with large observations. Completing the square in [eq:tc-locked],
$$
\tfrac12\big(u_1 - \hat g_1\big)^2 + \tfrac{11}{2}\,\hat g_1^2 \;=\; 6\Big(\hat g_1 - \frac{u_1}{12}\Big)^2 + \tfrac{11}{24}\,u_1^2 \;\ge\; \tfrac{11}{24}\,u_1^2 ,
$$
so the first sensor's term cannot fall below $\tfrac{11}{24}u_1^2$ whatever the means, and likewise for the second sensor, so on such trials the Exponential product's free energy stays far above the surprise, and they account for much of the mean.

::: widget mean-field-failure | Left: contours enclosing 50 and 90 per cent of the probability of the exact posterior (green) and of the free-form (blue), Gamma (purple) and Exponential (clay) products, for the observation and prior rates set by the sliders; the readout gives each family's means, standard deviations and divergence from the posterior. Right: the marginal densities of $s_1$ and $s_2$ under each. The Gamma product is drawn with long dashes and the Exponential product with short dashes.
:::

::: exercise The Gamma product penalty at large shapes
Show from [eq:tc-gam-moments] that, for fixed means $m_1, m_2$,
$$
\mathrm{Var}_{Q}(s_2 s_1^2) \;=\; m_2^2 m_1^4\left(\frac{4}{k_1} + \frac{1}{k_2}\right) + O\big(k^{-2}\big)
$$
as $k_1, k_2 \to \infty$ together, and compare with the first-order propagation of error, $\mathrm{Var}(s_2 s_1^2) \approx (2 s_1 s_2)^2\,\mathrm{Var}(s_1) + s_1^4\,\mathrm{Var}(s_2)$ evaluated at the means.
---solution---
By independence and [eq:tc-gam-moments], $\mathbb{E}_Q[s_2 s_1^2] = m_2 m_1^2 c_2(k_1)$ and $\mathbb{E}_Q[(s_2 s_1^2)^2] = m_2^2 c_2(k_2)\,m_1^4 c_4(k_1)$. Expanding, $c_2(k) = 1 + 1/k$ and
$$
c_4(k) \;=\; \Big(1 + \frac1k\Big)\Big(1 + \frac2k\Big)\Big(1 + \frac3k\Big) \;=\; 1 + \frac6k + \frac{11}{k^2} + \frac{6}{k^3}.
$$
So $\big(\mathbb{E}_Q[s_2 s_1^2]\big)^2 = m_2^2 m_1^4\big(1 + 2/k_1 + O(k^{-2})\big)$ and $\mathbb{E}_Q[(s_2 s_1^2)^2] = m_2^2 m_1^4\big(1 + 1/k_2\big)\big(1 + 6/k_1 + O(k^{-2})\big) = m_2^2 m_1^4\big(1 + 6/k_1 + 1/k_2 + O(k^{-2})\big)$. Subtracting,
$$
\mathrm{Var}_Q(s_2 s_1^2) \;=\; m_2^2 m_1^4\left(\frac{4}{k_1} + \frac{1}{k_2}\right) + O\big(k^{-2}\big).
$$
With $\mathrm{Var}(s_i) = m_i^2/k_i$, the propagation formula gives $(2m_1m_2)^2 m_1^2/k_1 + m_1^4 m_2^2/k_2 = m_2^2 m_1^4(4/k_1 + 1/k_2)$, the same leading term.
:::
