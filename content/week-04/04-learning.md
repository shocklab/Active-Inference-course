---
title: Learning with free energy
deck: Expectation maximisation as coordinate descent on one free energy, its proof that the evidence never falls, the posterior-variance term that Week 3's point rule lost, and what the Laplace free energy does with that term on Week 2's model.
week: 4
time: 70 min
---

[week:3] learned the parameters of the generative model, written $\omega$: the prior location $d_p$, the prior variance $\Sigma_p$, the sensory variance $\Sigma_u$, the source power $\theta$. Its point rule evaluated the derivative of the log joint at the mode of each trial's posterior, and [lesson:3.4] showed that this drops a posterior-variance term and biases the learned variances. [lesson:3.4] also named expectation maximisation as the remedy and stated that each of its iterations never lowers the evidence. This lesson proves that statement from the free energy, and asks how much of the remedy survives when the posterior is replaced by the Laplace approximation of [lesson:4.3].

## The free energy of many trials

Take $N$ observations $u_{1:N} = (u_1, \dots, u_N)$, each with its own hidden state $s_n$, generated independently given $\omega$:
$$
P(u_{1:N}, s_{1:N} \mid \omega) \;=\; \prod_{n=1}^{N} P(u_n, s_n \mid \omega).
$$
Two consequences follow. Summing over $s_{1:N}$ factorises the evidence, $P(u_{1:N} \mid \omega) = \prod_n P(u_n \mid \omega)$. Dividing the joint by the evidence factorises the posterior,
$$
P(s_{1:N} \mid u_{1:N}, \omega) \;=\; \prod_{n=1}^{N} \frac{P(u_n, s_n \mid \omega)}{P(u_n \mid \omega)} \;=\; \prod_{n=1}^{N} P(s_n \mid u_n, \omega).
$$
So a recognition distribution that is a product over trials, $Q(s_{1:N}) = \prod_n Q_n(s_n)$, contains the exact posterior. This product is not a mean-field approximation in the sense of [lesson:4.2], because the exact posterior already has this form. With such a $Q$, the free energy is a sum of one free energy per trial, by the same steps as the additivity of entropy in [lesson:4.2]:
$$
\mathcal{F}_{\mathrm{tot}}[Q_{1:N}; \omega] \;=\; \sum_{n=1}^{N} \mathcal{F}[Q_n, u_n; \omega],
\qquad
\mathcal{F}[Q_n, u_n; \omega] \;=\; \mathbb{E}_{Q_n(s_n)}\!\big[\ln Q_n(s_n) - \ln P(u_n, s_n \mid \omega)\big].
$$ {#total-vfe}
Applying the divergence-plus-surprise decomposition of [lesson:4.1] to each trial and summing,
$$
-\mathcal{F}_{\mathrm{tot}}[Q_{1:N}; \omega]
\;=\; \ln P(u_{1:N} \mid \omega) \;-\; \sum_{n=1}^{N} D_{\mathrm{KL}}\!\big[Q_n \,\big\|\, P(\cdot \mid u_n, \omega)\big]
\;\le\; \ln P(u_{1:N} \mid \omega).
$$ {#total-kl}

## Expectation maximisation

$\mathcal{F}_{\mathrm{tot}}$ is a function of two arguments, the recognition distributions and the parameters. Minimising it in each argument in turn is expectation maximisation.

::: theorem Expectation maximisation never lowers the evidence
Let $\omega^{(t-1)}$ be the current parameters, with superscript $(t)$ for the iteration. One iteration of expectation maximisation (EM), defined in [lesson:3.4], is:

1. The **E-step**: for each trial set $Q_n^{(t)} = P(s_n \mid u_n, \omega^{(t-1)})$, which minimises $\mathcal{F}[Q_n, u_n; \omega^{(t-1)}]$ over all $Q_n$.
2. The **M-step**: choose $\omega^{(t)}$ with $\mathcal{F}_{\mathrm{tot}}[Q_{1:N}^{(t)}; \omega^{(t)}] \le \mathcal{F}_{\mathrm{tot}}[Q_{1:N}^{(t)}; \omega^{(t-1)}]$, for example by minimising $\mathcal{F}_{\mathrm{tot}}$ over $\omega$ with $Q_{1:N}^{(t)}$ held fixed.

Then
$$
\ln P(u_{1:N} \mid \omega^{(t)}) \;\ge\; \ln P(u_{1:N} \mid \omega^{(t-1)}),
$$ {#em-monotone}
with strict inequality if the M-step strictly lowers $\mathcal{F}_{\mathrm{tot}}$.
:::

::: derivation Proof
Follow [eq:total-kl] through one iteration.

1. *After the E-step.* Each $Q_n^{(t)}$ is the posterior at $\omega^{(t-1)}$, so every divergence in [eq:total-kl] is zero and
$$
-\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t-1)}\big] \;=\; \ln P\big(u_{1:N} \mid \omega^{(t-1)}\big).
$$
2. *After the M-step.* By the choice of $\omega^{(t)}$,
$$
-\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t-1)}\big] \;\le\; -\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t)}\big].
$$
3. *At the new parameters.* By [eq:total-kl] at $\omega^{(t)}$, with the divergences now non-negative,
$$
-\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t)}\big] \;\le\; \ln P\big(u_{1:N} \mid \omega^{(t)}\big).
$$

Chaining the three,
$$
\ln P\big(u_{1:N} \mid \omega^{(t-1)}\big)
\;=\; -\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t-1)}\big]
\;\le\; -\mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega^{(t)}\big]
\;\le\; \ln P\big(u_{1:N} \mid \omega^{(t)}\big).
$$ {#em-sandwich}
If the middle inequality is strict, so is the outer one.
:::

The M-step needs only the expected energy. In the energy-minus-entropy form, the entropy of $Q_{1:N}^{(t)}$ does not depend on $\omega$, so
$$
-\nabla_\omega \mathcal{F}_{\mathrm{tot}}\big[Q_{1:N}^{(t)}; \omega\big]
\;=\; \sum_{n=1}^{N} \mathbb{E}_{Q_n^{(t)}(s_n)}\!\big[\nabla_\omega \ln P(u_n, s_n \mid \omega)\big].
$$ {#vem-m-grad}
With $Q_n^{(t)}$ the exact posterior, this is the posterior-averaged rule of [lesson:3.4]. The point rule replaces each expectation by the value at the mode.

::: fig One EM iteration on the prior variance of the worked example below. The green curve is the log evidence per trial as a function of $\Sigma_p$. The dashed curve is $-\mathcal{F}_{\mathrm{tot}}/N$ with the recognition densities fixed at the posteriors for $\Sigma_p^{(0)}$: it touches the evidence there (step 1 of the proof), its maximum is the M-step (step 2), and the evidence at the new value lies above it (step 3). The small dots are the iterates up to $\Sigma_p^{(6)}$.
{{fig_em_svg}}
:::

## Expectation maximisation on Week 3's prior variance

Use [lesson:3.4]'s linear model, $d_n \sim \mathcal{N}(d_p, \Sigma_p)$ on the whole line and $u_n \mid d_n \sim \mathcal{N}(d_n, \Sigma_u)$, and learn $\Sigma_p$ alone, with $d_p = {{d_prior:.0f}}$ and $\Sigma_u = {{em_vu:.0f}}$ held at their true values. The data are $N = {{em_n:.0f}}$ trials drawn with true $\Sigma_p^{*} = {{em_vp_true:.0f}}$, below Week 3's threshold $3\Sigma_u$. Write
$$
S \;=\; \frac{1}{N}\sum_{n=1}^{N}(u_n - d_p)^2 \;=\; {{em_S_emp:.4f}}
$$
for the mean squared deviation on this seeded sample.

::: derivation The M-step for the prior variance
The part of $\ln P(u_n, d_n \mid \Sigma_p)$ that depends on $\Sigma_p$ is the log prior, $-\tfrac12\ln(2\pi\Sigma_p) - (d_n - d_p)^2/(2\Sigma_p)$. Its derivative is
$$
\frac{\partial}{\partial\Sigma_p}\ln P(u_n, d_n \mid \Sigma_p) \;=\; -\frac{1}{2\Sigma_p} + \frac{(d_n - d_p)^2}{2\Sigma_p^2}.
$$
By [eq:vem-m-grad] the M-step sets the sum over trials of its $Q_n$-average to zero:
$$
\Sigma_p^{(t)} \;=\; \frac{1}{N}\sum_{n=1}^{N}\mathbb{E}_{Q_n^{(t)}}\big[(d_n - d_p)^2\big]
\;=\; \frac{1}{N}\sum_{n=1}^{N}\Big[\Sigma_{\mathrm{post}} + (m_n - d_p)^2\Big],
$$
using $\mathbb{E}[(d - d_p)^2] = \mathrm{Var}[d] + (\mathbb{E}[d] - d_p)^2$ with posterior mean $m_n$ and posterior variance $\Sigma_{\mathrm{post}}$. From [lesson:2.4], at the current value $\Sigma = \Sigma_p^{(t-1)}$,
$$
\Sigma_{\mathrm{post}} \;=\; \frac{\Sigma\,\Sigma_u}{\Sigma + \Sigma_u},
\qquad
m_n - d_p \;=\; \frac{\Sigma}{\Sigma + \Sigma_u}\,(u_n - d_p).
$$
Substituting, and averaging $(u_n - d_p)^2$ to $S$,
$$
\Sigma_p^{(t)} \;=\; \frac{\Sigma\,\Sigma_u}{\Sigma + \Sigma_u} \;+\; \frac{\Sigma^2 S}{(\Sigma + \Sigma_u)^2}.
$$ {#em-vp-update}
:::

::: derivation The fixed point is the maximum-likelihood value
A fixed point $\Sigma > 0$ of [eq:em-vp-update] satisfies $\Sigma = \frac{\Sigma\Sigma_u}{\Sigma + \Sigma_u} + \frac{\Sigma^2 S}{(\Sigma + \Sigma_u)^2}$. Divide by $\Sigma$ and multiply by $(\Sigma + \Sigma_u)^2$:
$$
(\Sigma + \Sigma_u)^2 \;=\; \Sigma_u(\Sigma + \Sigma_u) + \Sigma S
\quad\Longrightarrow\quad
(\Sigma + \Sigma_u)\,\Sigma \;=\; \Sigma S
\quad\Longrightarrow\quad
\Sigma \;=\; S - \Sigma_u .
$$
The evidence is the same value: integrating $d_n$ out gives $u_n - d_p \sim \mathcal{N}(0, \Sigma_p + \Sigma_u)$, so the log evidence per trial is $-\tfrac12\ln\big(2\pi(\Sigma_p + \Sigma_u)\big) - S/\big(2(\Sigma_p + \Sigma_u)\big)$, which is largest at $\Sigma_p + \Sigma_u = S$. Here $\hat\Sigma_p = S - \Sigma_u = {{em_vp_mle:.4f}}$.
:::

The point rule replaces each $Q_n$ by a point mass at its mode, so $\Sigma_{\mathrm{post}}$ disappears from the update and only $(m_n - d_p)^2$ remains:
$$
\Sigma \;\leftarrow\; \frac{\Sigma^2 S}{(\Sigma + \Sigma_u)^2}.
$$
[lesson:3.2] used this rule as a small online step after each trial; here it is iterated as a batch update with the same data. Its fixed points other than $0$ solve $(\Sigma + \Sigma_u)^2 = \Sigma S$, that is $\Sigma^2 + (2\Sigma_u - S)\Sigma + \Sigma_u^2 = 0$, whose discriminant is $(2\Sigma_u - S)^2 - 4\Sigma_u^2 = S(S - 4\Sigma_u)$. For real roots $S \ge 4\Sigma_u$; since $S$ averages to $\Sigma_p^{*} + \Sigma_u$ over samples, this is Week 3's condition $\Sigma_p^{*} \ge 3\Sigma_u$. On this sample the discriminant is ${{pt_disc:.4f}}$, so $0$ is the only fixed point, and from ${{em_0_vp:.2f}}$ the point rule goes to ${{pt_1_vp:.4f}}$, ${{pt_2_vp:.4f}}$, ${{pt_3_vp:sci2}}$ and ${{pt_4_vp:sci2}}$ in four iterations.

EM from the same start:

| Iteration $t$ | $\Sigma_p^{(t)}$ | Bound after the M-step, $-\mathcal{F}_{\mathrm{tot}}/N$ | Log evidence per trial at $\Sigma_p^{(t)}$ |
|---|---:|---:|---:|
| $0$ | ${{em_0_vp:.4f}}$ | | ${{em_0_logev:.6f}}$ |
| $1$ | ${{em_1_vp:.4f}}$ | ${{em_1_elbo_m:.6f}}$ | ${{em_1_logev:.6f}}$ |
| $2$ | ${{em_2_vp:.4f}}$ | ${{em_2_elbo_m:.6f}}$ | ${{em_2_logev:.6f}}$ |
| $3$ | ${{em_3_vp:.4f}}$ | ${{em_3_elbo_m:.6f}}$ | ${{em_3_logev:.6f}}$ |
| $4$ | ${{em_4_vp:.4f}}$ | ${{em_4_elbo_m:.6f}}$ | ${{em_4_logev:.6f}}$ |
| $5$ | ${{em_5_vp:.4f}}$ | ${{em_5_elbo_m:.6f}}$ | ${{em_5_logev:.6f}}$ |
| $6$ | ${{em_6_vp:.4f}}$ | ${{em_6_elbo_m:.6f}}$ | ${{em_6_logev:.6f}}$ |
| ${{em_conv_its:.0f}}$ | ${{em_conv_vp:.4f}}$ | ${{em_conv_logev:.6f}}$ | ${{em_conv_logev:.6f}}$ |

Each row checks [eq:em-sandwich]: the bound after the M-step lies between the evidence of the previous row and the evidence of its own row. The last row is where successive values first differ by less than $10^{-15}$; there the bound and the evidence agree, and $\Sigma_p$ equals $\hat\Sigma_p$.

## Week 3's missing term

The Laplace free energy of [lesson:4.3] gives a second way to learn $\omega$: minimise $\mathcal{F}_{\mathrm{Lap}}$ for each trial, at its own mode, with respect to $\omega$. Per trial,
$$
-\mathcal{F}_{\mathrm{Lap}}(\omega) \;=\; \ell(\hat d(\omega); \omega) \;+\; \tfrac{1}{2}\ln\big(2\pi\hat\Sigma_q(\omega)\big),
\qquad
\hat\Sigma_q(\omega) \;=\; -\frac{1}{\ell''(\hat d(\omega); \omega)},
$$
where primes are derivatives in $d$ and the mode $\hat d(\omega)$ depends on $\omega$. The point rule of [week:3] climbed the first term alone.

On the linear model the log joint is exactly quadratic in $d$, so by the exercise at the end of [lesson:4.3] the Laplace free energy equals the surprise, and learning with it is learning with the exact evidence. The second term is then $\tfrac12\ln(2\pi\Sigma_{\mathrm{post}})$ with $\Sigma_{\mathrm{post}} = \Sigma_p\Sigma_u/(\Sigma_p + \Sigma_u)$, and its derivative in $\Sigma_p$ is
$$
\frac{\partial}{\partial\Sigma_p}\,\tfrac12\ln\Sigma_{\mathrm{post}}
\;=\; \frac12\left(\frac{1}{\Sigma_p} - \frac{1}{\Sigma_p + \Sigma_u}\right)
\;=\; \frac{\Sigma_u}{2\Sigma_p(\Sigma_p + \Sigma_u)} \;>\; 0,
$$
which is ${{em_dlogspost:.2f}}$ at $\Sigma_p = \Sigma_u = 1$. It always pushes $\Sigma_p$ up. As [lesson:3.4] showed, the first term, the log joint at the mode, grows without bound as $\Sigma_p \to 0$, because the prior's density at its own peak grows like $1/\sqrt{\Sigma_p}$ and the mode moves to that peak; the width term grows like $\tfrac12\ln\Sigma_p$ towards $-\infty$ and cancels that growth.

On [week:2]'s nonlinear model the Laplace free energy is not exact, and its derivative in $\omega$ has a term the linear case hides.

::: derivation The derivative of the Laplace free energy in a parameter
Differentiate $-\mathcal{F}_{\mathrm{Lap}}$ with respect to $\omega$, keeping track of the dependence through $\hat d(\omega)$. Write $\partial_\omega$ for the partial derivative at fixed $d$ and $\hat d' = \mathrm{d}\hat d/\mathrm{d}\omega$:
$$
-\frac{\mathrm{d}\mathcal{F}_{\mathrm{Lap}}}{\mathrm{d}\omega}
\;=\; \partial_\omega\ell + \ell'\,\hat d' \;-\; \frac{1}{2}\,\frac{\partial_\omega\ell'' + \ell'''\,\hat d'}{\ell''},
$$
all evaluated at $(\hat d(\omega), \omega)$. At the mode $\ell' = 0$, so the second term vanishes; this is the envelope theorem, and it is why the point rule needs no $\hat d'$. The last term keeps $\hat d'$. Differentiating the condition $\ell'(\hat d(\omega); \omega) = 0$ with respect to $\omega$ gives $\ell''\hat d' + \partial_\omega\ell' = 0$, so $\hat d' = -\partial_\omega\ell'/\ell''$, and
$$
-\frac{\mathrm{d}\mathcal{F}_{\mathrm{Lap}}}{\mathrm{d}\omega}
\;=\; \partial_\omega\ell \;-\; \frac{1}{2\ell''}\left(\partial_\omega\ell'' \;-\; \frac{\ell'''\,\partial_\omega\ell'}{\ell''}\right).
$$ {#lap-learn}
The first term is the point rule. The second is the derivative of $\tfrac12\ln(2\pi\hat\Sigma_q)$, and through $\ell'''$ it depends on the skew of the posterior.
:::

Where does each rule come to rest on Week 2's model? Learn one parameter at a time with the others at their true values, and average each rule's per-trial step over trials drawn from the model at the true parameters. A value at which that average step is zero is where the rule can come to rest; whether it does is checked below. The three rules are the point rule, the Laplace rule [eq:lap-learn], and the exact rule, the posterior average [eq:vem-m-grad] with the exact posterior. The average over the world uses Gauss&ndash;Legendre quadrature in $d$ over $(0, {{ll_world_dmax:.1f}}]$, which holds a fraction ${{ll_world_mass:.8f}}$ of the prior mass, and Gauss&ndash;Hermite quadrature with ${{ll_n_z:.0f}}$ nodes in the sensory noise. The exact posterior averages use Gauss&ndash;Legendre quadrature in $g = \theta/d^2$. The exact rule's zero should be the true value, because the posterior-averaged score has mean zero there. Recovering the exact score's zero is a consistency check. Convergence of each approximate rule is checked separately: recomputing every zero with ${{ll_fine_n_z:.0f}}$ noise nodes, twice as many panels in $d$ and order ${{ll_fine_order:.0f}}$ moves the largest of them by about ${{ll_conv:sci1}}$. At every zero in the table the derivative of the average step with respect to $\ln\omega$ is negative; the largest is about ${{ll_slope_max:.3f}}$. So for sufficiently small deterministic steps each zero attracts nearby values of $\omega$. That is a local statement; it says nothing about where a rule started far away will go.

| Parameter (true value) | Point rule | Laplace rule | Exact rule |
|---|---:|---:|---:|
| $\Sigma_p$ (${{var_prior:.0f}}$) | ${{ll_vp_point:.4f}}$ (${{ll_vp_point_pct:+.1f}}\%$) | ${{ll_vp_laplace:.4f}}$ (${{ll_vp_laplace_pct:+.1f}}\%$) | ${{ll_vp_exact:.6f}}$ |
| $\Sigma_u$ (${{var_obs:.1f}}$) | ${{ll_vu_point:.4f}}$ (${{ll_vu_point_pct:+.1f}}\%$) | ${{ll_vu_laplace:.4f}}$ (${{ll_vu_laplace_pct:+.1f}}\%$) | ${{ll_vu_exact:.6f}}$ |

The point-rule values are Week 3's. The Laplace rule reduces the error in $\Sigma_p$ and reverses the sign of the error in $\Sigma_u$, which it overestimates. Its zero is not the truth because $\mathcal{F}_{\mathrm{Lap}}$ is an approximation to the surprise whose error, the quantity [lesson:4.3] expanded, itself depends on $\omega$, and the Laplace rule follows the derivative of that error too. Restoring the width term changes where learning settles; on this model it does not remove the bias, and the table gives no general rule for its sign.

## Variational EM

When the exact posterior is not available, the E-step can minimise $\mathcal{F}[Q_n, u_n; \omega]$ over a family $\mathcal{Q}$ instead, as in [lesson:4.2] and [lesson:4.3]. This is **variational EM**. The E-step now leaves a divergence behind. Minimising [eq:total-kl] over the family trial by trial,
$$
\min_{Q_n \in \mathcal{Q}}\,\mathcal{F}_{\mathrm{tot}}[Q_{1:N}; \omega]
\;=\; -\ln P(u_{1:N} \mid \omega) \;+\; \sum_{n=1}^{N}\min_{Q_n \in \mathcal{Q}} D_{\mathrm{KL}}\!\big[Q_n \,\big\|\, P(\cdot \mid u_n, \omega)\big].
$$ {#vem-objective}
The first link of [eq:em-sandwich] becomes an inequality: after the E-step, $-\mathcal{F}_{\mathrm{tot}}$ lies below the log evidence by the remaining divergence rather than touching it. Steps 2 and 3 of the proof still hold, so $\mathcal{F}_{\mathrm{tot}}$ never rises, but the log evidence can fall from one iteration to the next. Variational EM decreases [eq:vem-objective], not the surprise alone, and under appropriate regularity conditions it can converge to a local stationary point of that objective. The optimised divergence term can shift that point away from a stationary point of the evidence. Whether it does depends on the derivative of the smallest attainable divergence with respect to $\omega$ at that point, not on its dependence on $\omega$ somewhere else. [lesson:4.5] shows how large the shift can be.

::: exercise Fisher's identity from the free energy
Suppose $P(o, s \mid \omega)$ is differentiable in $\omega$ and derivatives may be taken under the sum. Let $\hat Q(s) = P(s \mid o, \omega_0)$ be held fixed. Using the divergence-plus-surprise decomposition, show that
$$
-\nabla_\omega \mathcal{F}[\hat{Q}, o; \omega]\Big|_{\omega = \omega_0}
\;=\; \nabla_\omega \ln P(o \mid \omega)\Big|_{\omega = \omega_0},
$$
and hence Fisher's identity, which [lesson:3.4] derived directly.
---solution---
With $\hat Q$ fixed,
$$
-\mathcal{F}[\hat{Q}, o; \omega] \;=\; \ln P(o \mid \omega) \;-\; D_{\mathrm{KL}}\!\big[\hat{Q} \,\big\|\, P(\cdot \mid o, \omega)\big].
$$
The divergence is non-negative for every $\omega$ and zero at $\omega_0$, so $\omega_0$ is a minimum of it and its gradient there is zero. Directly, using $\sum_s P(s \mid o, \omega) = 1$:
$$
\nabla_\omega D_{\mathrm{KL}}\!\big[\hat{Q} \,\big\|\, P(\cdot \mid o, \omega)\big]\Big|_{\omega_0}
\;=\; -\sum_s P(s \mid o, \omega_0)\,\frac{\nabla_\omega P(s \mid o, \omega)\big|_{\omega_0}}{P(s \mid o, \omega_0)}
\;=\; -\nabla_\omega \sum_s P(s \mid o, \omega)\Big|_{\omega_0} \;=\; 0.
$$
So the two gradients are equal. By [eq:vem-m-grad] with one trial, the left side is $\mathbb{E}_{P(s \mid o, \omega_0)}[\nabla_\omega \ln P(o, s \mid \omega)]$ at $\omega_0$, which gives Fisher's identity $\nabla_\omega \ln P(o \mid \omega_0) = \mathbb{E}_{P(s \mid o, \omega_0)}[\nabla_\omega \ln P(o, s \mid \omega_0)]$.
:::
