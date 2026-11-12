---
title: Problems and code
deck: Worked problems on Gibbs' inequality, expectation maximisation, mean-field approximation in both directions of the divergence, the noisy-OR saddle, Laplace against Gaussian variational inference, changes of coordinate and the two-cause model.
week: 4
time: 120 min
number_sections: false
---

Work the problems before opening the solutions. The ones marked with a dagger are
used again later in the course.

## The notebooks

| Notebook | What it does | Run it |
|---|---|---|
| `week04_numpy.ipynb` | The three decompositions of variational free energy on Week&nbsp;1's three-state model; the spike entropy; expectation maximisation on Week&nbsp;3's prior variance, reproducing the table of [lesson:4.4]; sequential and simultaneous coordinate ascent on the noisy-OR model; Laplace and log-normal free energies on Week&nbsp;2's sound model; and the whole computation of [lesson:4.5]: the exact posterior by numerical integration, the three mean-field families, variational EM with warm starts and the monotone free-energy check, and the online rule | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-04/week04_numpy.ipynb) |
| `week04_jax.ipynb` | What needs automatic differentiation or many runs at once: Fisher's identity and the Laplace curvature checked with `jax.grad` and `jax.hessian` at a solved mode; sequential coordinate ascent mapped with `jax.vmap` over a grid of starting beliefs; and Newton E-steps for the Exponential and Gamma families of [lesson:4.5] on all ${{tc_n:.0f}}$ trials at once, with the hand-derived gradients checked against automatic ones | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-04/week04_jax.ipynb) |

The NumPy notebook builds every update from array operations, so each term of the
algebra can be inspected. The JAX notebook differentiates the same objectives
automatically and runs them across grids of starting points and across trials.

There is no `pymdp` notebook this week. Categorical variational inference in
discrete state spaces is the subject of [week:9].

## Problems

::: exercise Equality in Gibbs' inequality &dagger;
Let $Q$ and $P$ be probability distributions on a finite set $\mathcal{S}$ with
$P(s) > 0$ for every $s \in \mathcal{S}$. Use the tangent-line bound
$\ln y \le y - 1$ for $y > 0$, with equality only at $y = 1$.

(a) Prove that $D_{\mathrm{KL}}[Q \,\|\, P] \ge 0$, with equality if and only if
$Q(s) = P(s)$ for every $s \in \mathcal{S}$.

(b) Deduce that for a fixed generative model $P(o, s)$ with $P(o) > 0$, the
free energy $\mathcal{F}[Q, o]$ has its minimum over all distributions on
$\mathcal{S}$ at $Q^{*}(s) = P(s \mid o)$ and nowhere else, with minimum value
$-\ln P(o)$.

(c) For the two-point variable $Y \in \{1, 9\}$ with weights
$Q = ({{jen_q:.2f}}, {{jen_q:.2f}})$ from [lesson:4.1], write the Jensen gap
$\ln \mathbb{E}_Q[Y] - \mathbb{E}_Q[\ln Y]$ as a divergence of $Q$ from the
distribution proportional to $Q(y)\,y$, and evaluate it.
---solution---
(a) Split $\mathcal{S}$ into $\mathcal{S}_{+} = \{s : Q(s) > 0\}$ and
$\mathcal{S}_{0} = \{s : Q(s) = 0\}$. With the convention $0 \ln 0 = 0$, the states in
$\mathcal{S}_{0}$ contribute nothing to the divergence, and $\sum_{s \in \mathcal{S}_{+}} Q(s) = 1$.
Apply $\ln y \le y - 1$ to $y_s = P(s)/Q(s) > 0$ for $s \in \mathcal{S}_{+}$:

$$
\begin{aligned}
-D_{\mathrm{KL}}[Q \,\|\, P]
&\;=\; \sum_{s \in \mathcal{S}_{+}} Q(s)\ln\frac{P(s)}{Q(s)} \\
&\;\le\; \sum_{s \in \mathcal{S}_{+}} Q(s)\!\left(\frac{P(s)}{Q(s)} - 1\right) \\
&\;=\; \sum_{s \in \mathcal{S}_{+}} P(s) - 1 \;=\; -\sum_{s \in \mathcal{S}_{0}} P(s) \;\le\; 0 .
\end{aligned}
$$

So $D_{\mathrm{KL}}[Q \,\|\, P] \ge \sum_{s \in \mathcal{S}_{0}} P(s) \ge 0$. For the
divergence to be zero both inequalities must be equalities. The second needs
$\sum_{s \in \mathcal{S}_{0}} P(s) = 0$, and since every $P(s) > 0$, $\mathcal{S}_{0}$
is empty. The first needs $Q(s)\big[\ln y_s - (y_s - 1)\big] = 0$ for every $s$; each
bracket is non-positive and $Q(s) > 0$, so each bracket is zero, and by the equality
condition $y_s = 1$, that is $Q(s) = P(s)$. Conversely $Q = P$ makes every term
$\ln 1 = 0$.

(b) By the divergence-plus-surprise decomposition of [lesson:4.1],
$\mathcal{F}[Q, o] = -\ln P(o) + D_{\mathrm{KL}}\big[Q(s) \,\|\, P(s \mid o)\big]$. The
first term does not depend on $Q$. If $Q$ puts mass on a state with $P(o, s) = 0$, then
$\mathcal{F}[Q, o] = +\infty$, so only distributions supported where $P(s \mid o) > 0$
can be minimisers. On that set part (a) applies to the second term, which is zero only
at $Q(s) = P(s \mid o)$. So $Q^{*}(s) = P(s \mid o)$ is the unique minimiser and
$\mathcal{F}[Q^{*}, o] = -\ln P(o)$.

(c) Let $\tilde P(y) = Q(y)\,y / \mathbb{E}_Q[Y]$. With $\mathbb{E}_Q[Y] = {{jen_ey:.0f}}$,
$\tilde P(1) = 1/10$ and $\tilde P(9) = 9/10$. Then

$$
D_{\mathrm{KL}}[Q \,\|\, \tilde P]
\;=\; \sum_{y} Q(y)\ln\frac{Q(y)}{Q(y)\,y / \mathbb{E}_Q[Y]}
\;=\; \sum_{y} Q(y)\big(\ln \mathbb{E}_Q[Y] - \ln y\big)
\;=\; \ln \mathbb{E}_Q[Y] - \mathbb{E}_Q[\ln Y] .
$$

Numerically it is ${{jen_ln_ey:.4f}} - {{jen_e_lny:.4f}} = {{jen_gap:.4f}}$ nats,
positive because $Q \ne \tilde P$.
:::

::: exercise One step of expectation maximisation by hand &dagger;
A binary hidden cause $s \in \{0, 1\}$ has unknown prior probability
$\omega = P(s = 1 \mid \omega) \in (0, 1)$, the parameter to be learned. A binary
sensor $o$ has $P(o = 1 \mid s = 1) = p_{\mathrm{hit}} = {{ex_dem_hit:.2f}}$ and
$P(o = 1 \mid s = 0) = p_{\mathrm{fa}} = {{ex_dem_fp:.2f}}$. Over $N = 10$ independent trials the
sensor reports seven ones and three zeros. Write
$p_1(\omega) = P(o = 1 \mid \omega) = p_{\mathrm{fa}} + (p_{\mathrm{hit}} - p_{\mathrm{fa}})\,\omega$ and
$\ln P(o_{1:10} \mid \omega) = 7\ln p_1(\omega) + 3\ln\big(1 - p_1(\omega)\big)$ for the
log evidence of the ten trials. Start from $\omega^{(0)} = {{ex_dem_th0:.2f}}$.

(a) Carry out the exact E-step: compute $q_{\mathrm{on}} = P(s = 1 \mid o = 1, \omega^{(0)})$
and $q_{\mathrm{off}} = P(s = 1 \mid o = 0, \omega^{(0)})$, and check that
$-\mathcal{F}_{\mathrm{tot}}$ with these recognition distributions equals
$\ln P(o_{1:10} \mid \omega^{(0)})$.

(b) Minimise $\mathcal{F}_{\mathrm{tot}}$ over $\omega$ with the recognition
distributions held fixed to obtain $\omega^{(1)}$, and verify the inequalities of
[lesson:4.4]:
$\ln P(o_{1:10} \mid \omega^{(0)}) \le -\mathcal{F}_{\mathrm{tot}}(\omega^{(1)}) \le \ln P(o_{1:10} \mid \omega^{(1)})$.

(c) A fixed point of EM satisfies
$\omega = \frac{1}{10}\big(7\,P(s = 1 \mid o = 1, \omega) + 3\,P(s = 1 \mid o = 0, \omega)\big)$.
Find every fixed point in $[0, 1]$ from this equation, and check the interior one
against the maximum of $\ln P(o_{1:10} \mid \omega)$.
---solution---
(a) At $\omega^{(0)} = {{ex_dem_th0:.2f}}$, $p_1(\omega^{(0)}) = p_{\mathrm{fa}} + (p_{\mathrm{hit}} - p_{\mathrm{fa}})\,\omega^{(0)} = {{ex_dem_p1_0:.2f}}$ and
$1 - p_1 = {{ex_dem_p0_0:.2f}}$, so
$\ln P(o_{1:10} \mid \omega^{(0)}) = 7\ln {{ex_dem_p1_0:.2f}} + 3\ln {{ex_dem_p0_0:.2f}} = {{ex_dem_ll0:.4f}}$ nats.
Bayes' rule gives

$$
q_{\mathrm{on}} \;=\; \frac{p_{\mathrm{hit}}\,\omega^{(0)}}{p_1(\omega^{(0)})} \;=\; \frac{12}{19} \;=\; {{ex_dem_q_on:.4f}},
\qquad
q_{\mathrm{off}} \;=\; \frac{(1 - p_{\mathrm{hit}})\,\omega^{(0)}}{1 - p_1(\omega^{(0)})} \;=\; \frac{3}{31} \;=\; {{ex_dem_q_off:.4f}} .
$$

These are the exact posteriors at $\omega^{(0)}$, so each trial's divergence is zero and
$-\mathcal{F}_{\mathrm{tot}}(\omega^{(0)}) = \ln P(o_{1:10} \mid \omega^{(0)}) = {{ex_dem_ll0:.4f}}$ nats.

(b) With the recognition distributions fixed, the part of $\mathcal{F}_{\mathrm{tot}}$
that depends on $\omega$ is $-\sum_n \mathbb{E}_{Q_n}[\ln P(s_n \mid \omega)]$:

$$
-\big(7q_{\mathrm{on}} + 3q_{\mathrm{off}}\big)\ln\omega \;-\; \big(7(1 - q_{\mathrm{on}}) + 3(1 - q_{\mathrm{off}})\big)\ln(1 - \omega).
$$

Setting the derivative to zero gives the expected fraction of trials with $s = 1$:

$$
\omega^{(1)} \;=\; \frac{7q_{\mathrm{on}} + 3q_{\mathrm{off}}}{10}
\;=\; \frac{7 \times 12/19 + 3 \times 3/31}{10} \;=\; {{ex_dem_th1:.4f}} .
$$

With the recognition distributions still fixed,
$-\mathcal{F}_{\mathrm{tot}}(\omega^{(1)}) = {{ex_dem_elbo_m1:.4f}}$ nats, and the log
evidence at the new value is ${{ex_dem_ll1:.4f}}$ nats, so
${{ex_dem_ll0:.4f}} < {{ex_dem_elbo_m1:.4f}} < {{ex_dem_ll1:.4f}}$.

(c) With $p_1 = p_1(\omega)$, the two posteriors are $p_{\mathrm{hit}}\omega/p_1$ and
$(1 - p_{\mathrm{hit}})\omega/(1 - p_1)$, so the fixed-point equation is

$$
\omega \;=\; \frac{7}{10}\,\frac{p_{\mathrm{hit}}\,\omega}{p_1} + \frac{3}{10}\,\frac{(1 - p_{\mathrm{hit}})\,\omega}{1 - p_1}.
$$

$\omega = 0$ is a solution. For $\omega > 0$, divide by $\omega$ and write
$\bar o = 7/10$ for the observed fraction of ones:

$$
1 \;=\; \frac{\bar o\, p_{\mathrm{hit}}}{p_1} + \frac{(1 - \bar o)(1 - p_{\mathrm{hit}})}{1 - p_1}.
$$

Multiply by $p_1(1 - p_1)$, which is positive for $\omega \in (0, 1]$ since
$0 < p_{\mathrm{fa}} < p_{\mathrm{hit}} < 1$:

$$
p_1 - p_1^2 \;=\; \bar o\, p_{\mathrm{hit}}\,(1 - p_1) + (1 - \bar o)(1 - p_{\mathrm{hit}})\,p_1 .
$$

The coefficient of $p_1$ on the right is $-\bar o\, p_{\mathrm{hit}} + 1 - \bar o - p_{\mathrm{hit}} + \bar o\, p_{\mathrm{hit}} = 1 - \bar o - p_{\mathrm{hit}}$.
Moving everything to one side,

$$
p_1^2 - (\bar o + p_{\mathrm{hit}})\,p_1 + \bar o\, p_{\mathrm{hit}} \;=\; 0
\qquad\Longleftrightarrow\qquad
(p_1 - \bar o)(p_1 - p_{\mathrm{hit}}) \;=\; 0 ,
$$

with $\bar o + p_{\mathrm{hit}} = {{ex_dem_sum:.2f}}$ and $\bar o\, p_{\mathrm{hit}} = {{ex_dem_prod:.2f}}$. The root
$p_1 = p_{\mathrm{hit}}$ means $p_{\mathrm{fa}} + (p_{\mathrm{hit}} - p_{\mathrm{fa}})\omega = p_{\mathrm{hit}}$, that is $\omega = 1$. The root $p_1 = \bar o$
gives $\omega = (\bar o - p_{\mathrm{fa}})/(p_{\mathrm{hit}} - p_{\mathrm{fa}}) = 5/6 = {{ex_dem_th_mle:.4f}}$. So EM has three
fixed points in $[0, 1]$: $0$, $5/6$ and $1$, and one in the interior.

The log evidence has derivative
$(p_{\mathrm{hit}} - p_{\mathrm{fa}})\big[7/p_1 - 3/(1 - p_1)\big]$, zero only at $p_1 = 7/10$, and it is concave in
$p_1$, so its maximum over $(0, 1)$ is at $\omega = 5/6$, the interior fixed point. The
endpoints are fixed points because an E-step at $\omega = 0$ or $1$ returns posteriors
that put all mass on one value of $s$, and the M-step returns the same endpoint.
:::

::: exercise Mean-field approximation of a correlated Gaussian in both directions &dagger;
[lesson:4.5] proved that the product minimising the reverse divergence
$D_{\mathrm{KL}}[Q \,\|\, P]$ from a Gaussian $P$ has factors with variance
$1/\Pi_{ii}$, the conditional variance. Take $P = \mathcal{N}(0, \Sigma)$ on
$\mathbb{R}^2$ with
$$
\Sigma \;=\; \begin{pmatrix} 1 & c \\ c & 1 \end{pmatrix}, \qquad c \in (-1, 1).
$$

(a) Show that over products $Q = q_1 q_2$ the forward divergence
$D_{\mathrm{KL}}[P \,\|\, Q]$ is minimised by the product of the marginals of $P$, for
any $P$ with finite entropies, and that its minimum is the mutual information
$I(s_1; s_2)$ under $P$.

(b) For the Gaussian above, compute the minimum forward divergence and compare it with
the minimum reverse divergence of [lesson:4.5]. Evaluate the standard deviations of
both optimal products and both minima at $c = {{ex_cg_c:.2f}}$.

(c) Explain in terms of where each divergence weights the logarithm of the ratio why
the reverse optimum is narrower than $P$'s marginals and the forward optimum is not.
---solution---
(a) Write $P_1, P_2$ for the marginals of $P$. Since $\ln Q(s) = \ln q_1(s_1) + \ln q_2(s_2)$,

$$
D_{\mathrm{KL}}[P \,\|\, Q] \;=\; -\mathrm{H}[P] - \mathbb{E}_{P}[\ln q_1(s_1)] - \mathbb{E}_{P}[\ln q_2(s_2)]
\;=\; -\mathrm{H}[P] + \sum_{i=1}^{2}\Big(\mathrm{H}[P_i] + D_{\mathrm{KL}}[P_i \,\|\, q_i]\Big),
$$

using $-\mathbb{E}_{P_i}[\ln q_i] = \mathrm{H}[P_i] + D_{\mathrm{KL}}[P_i \,\|\, q_i]$. Only the
two divergences depend on $Q$, and by Gibbs' inequality each is zero exactly at
$q_i = P_i$. The minimum is $\mathrm{H}[P_1] + \mathrm{H}[P_2] - \mathrm{H}[P]$, which is
$I(s_1; s_2)$: it equals $D_{\mathrm{KL}}[P \,\|\, P_1 P_2]$, the definition of the
mutual information as a divergence.

(b) The marginals are $\mathcal{N}(0, 1)$, with differential entropy
$\tfrac12\ln(2\pi e)$ each, and $\mathrm{H}[P] = \tfrac12\ln\big((2\pi e)^2\det\Sigma\big)$
with $\det\Sigma = 1 - c^2$. So

$$
\min_{Q} D_{\mathrm{KL}}[P \,\|\, Q] \;=\; \ln(2\pi e) - \tfrac12\ln\big((2\pi e)^2(1 - c^2)\big) \;=\; -\tfrac12\ln(1 - c^2),
$$

the same value as the minimum reverse divergence. The optimal products are different:
the forward optimum has standard deviation $1$ in each coordinate, the reverse optimum
$\sqrt{1 - c^2}$. At $c = {{ex_cg_c:.2f}}$ these are $1$ and ${{ex_cg_qsd:.2f}}$, and
both minima are $-\tfrac12\ln {{ex_cg_qvar:.2f}} = {{ex_cg_kl:.4f}}$ nats. The equality of
the two minima is a property of the Gaussian, not of the two divergences in general.

(c) The forward divergence averages $\ln(P/Q)$ under $P$, so it is large wherever $P$
has mass and $Q$ has little: an axis-aligned $Q$ must cover the whole tilted ellipse, and
the product of the marginals, which spreads over the full width in each coordinate, does
so. The reverse divergence averages $\ln(Q/P)$ under $Q$, so it is large wherever $Q$ has
mass and $P$ has little: a product as wide as the marginals would put mass in the two
corners off the ellipse's long axis, where $P$ is small, so the optimum is narrower and
lies inside the ellipse. This is the contrast between the two directions drawn in
[lesson:1.4].
:::

::: exercise The symmetric noisy-OR fixed point is a saddle
Return to the noisy-OR model of [lesson:4.2], with both causes having prior probability
$p_0 = {{nor_p:.2f}}$ and posterior entries $P_{ab} = P(s_1 = a, s_2 = b \mid o_1 = 1)$.
The coordinate update for either factor is the map $R$ with

$$
\ln\frac{R(q)}{1 - R(q)} \;=\; b_0 + b_1\,q ,
$$

where $q$ is the probability the other factor gives to its cause being present.

(a) Express $b_0$ and $b_1$ through the likelihood entries
$L_{ab} = P(o_1 = 1 \mid s_1 = a, s_2 = b)$ and $p_0$, show that $b_1 < 0$, and prove that
$R(q) = q$ has exactly one solution $q_{\mathrm{sad}} \in (0, 1)$.

(b) Show that $\partial\mathcal{F}/\partial q_1 = \ln\frac{q_1}{1 - q_1} - (b_0 + b_1 q_2)$,
find the Hessian of $\mathcal{F}(q_1, q_2)$ at $(q_{\mathrm{sad}}, q_{\mathrm{sad}})$, and
show that it is indefinite when $R'(q_{\mathrm{sad}}) < -1$, which holds here.

(c) Show that, restricted to the diagonal $q_1 = q_2 = q$, $\mathcal{F}$ has its minimum
over $(0, 1)$ at $q_{\mathrm{sad}}$, and conclude that every symmetric product has free
energy at least ${{cavi_sad_F:.4f}}$ nats, above the asymmetric minimum
${{cavi_lim_F:.4f}}$.
---solution---
(a) From the log-odds form of the update in [lesson:4.2], the log-odds of $q_1$ given $q_2 = q$
is the prior log-odds plus the expected log-likelihood ratio,

$$
\ln\frac{p_0}{1 - p_0} + (1 - q)\ln\frac{L_{10}}{L_{00}} + q\ln\frac{L_{11}}{L_{01}} \;=\; b_0 + b_1 q,
$$

so $b_0 = \ln\frac{p_0}{1 - p_0} + \ln\frac{L_{10}}{L_{00}} = {{cavi_logit_int:.4f}}$ and

$$
b_1 \;=\; \ln\frac{L_{11}\,L_{00}}{L_{10}\,L_{01}} \;=\; {{cavi_logit_slope:.4f}}.
$$

$b_1$ compares the gain in likelihood from switching on the first cause when the second
is already on, $L_{11}/L_{01}$, with the gain when it is off, $L_{10}/L_{00}$. With one
cause on, the sensor is already likely to fire, so the second gain is the smaller and
$b_1 < 0$: this is explaining away. Then $R(q) = 1/(1 + e^{-(b_0 + b_1 q)})$ is strictly
decreasing, so $R(q) - q$ is strictly decreasing and continuous on $[0, 1]$, positive at
$q = 0$ (where it equals $R(0) > 0$) and negative at $q = 1$ (where it equals
$R(1) - 1 < 0$). It has exactly one zero, $q_{\mathrm{sad}} = {{cavi_sad_q:.4f}}$.

(b) $\mathcal{F}(q_1, q_2)$ is the sum over the four states of $Q\ln Q - Q\ln P$ with
$Q_{ab} = q_1^a(1 - q_1)^{1-a}q_2^b(1 - q_2)^{1-b}$. Differentiating in $q_1$, the entropy
part gives $\ln\frac{q_1}{1 - q_1}$ and the energy part gives minus the difference of
the expected log joint between $s_1 = 1$ and $s_1 = 0$, which is the log-odds
$b_0 + b_1 q_2$ (the posterior and the joint differ by $P(o_1 = 1)$, which cancels in the
difference). So $\partial\mathcal{F}/\partial q_1 = \ln\frac{q_1}{1 - q_1} - (b_0 + b_1 q_2)$,
and by symmetry the same with the indices exchanged. Differentiating again, at
$(q_{\mathrm{sad}}, q_{\mathrm{sad}})$ with $v = q_{\mathrm{sad}}(1 - q_{\mathrm{sad}})$,

$$
\nabla^2\mathcal{F} \;=\; \begin{pmatrix} 1/v & -b_1 \\ -b_1 & 1/v \end{pmatrix},
$$

with eigenvalue $1/v - b_1$ along $(1, 1)$ and $1/v + b_1$ along $(1, -1)$. Since
$R'(q) = R(q)(1 - R(q))\,b_1$, at the fixed point $R'(q_{\mathrm{sad}}) = v\,b_1 = {{cavi_sad_rprime:.4f}}$, and

$$
\frac{1}{v} + b_1 \;=\; \frac{1 + R'(q_{\mathrm{sad}})}{v} \;<\; 0,
\qquad
\frac{1}{v} - b_1 \;=\; \frac{1 - R'(q_{\mathrm{sad}})}{v} \;>\; 0 .
$$

One positive and one negative eigenvalue: the Hessian is indefinite, and the point is a
saddle, rising along the diagonal and falling across it.

(c) On the diagonal, by the chain rule and part (b),

$$
\frac{\mathrm{d}}{\mathrm{d}q}\,\mathcal{F}(q, q) \;=\; 2\Big(\ln\frac{q}{1 - q} - b_0 - b_1 q\Big).
$$

Since $b_1 < 0$, the bracket is strictly increasing in $q$; it is zero only where
$\ln\frac{q}{1-q} = b_0 + b_1 q$, that is at $q = R(q)$, so at $q_{\mathrm{sad}}$. It is
negative below and positive above, so $\mathcal{F}(q, q)$ falls to $q_{\mathrm{sad}}$ and
rises after it: its minimum over the diagonal is
$\mathcal{F}(q_{\mathrm{sad}}, q_{\mathrm{sad}}) = {{cavi_sad_F:.4f}}$ nats, above the
asymmetric minimum ${{cavi_lim_F:.4f}}$. [lesson:1.4] reached the same conclusion from the
second derivative of the divergence along the diagonal; here it follows from the
coordinate-ascent map.
:::

::: exercise Gaussian variational inference and Laplace on an asymmetric potential &dagger;
A hidden state $x \in \mathbb{R}$ has unnormalised joint density
$\exp\big(-V(x)\big)$ with

$$
V(x) \;=\; \tfrac12 x^2 + e^{x},
$$

so $\ell(x) = -V(x)$ is the log joint and the surprise is
$-\ln\int_{-\infty}^{\infty}e^{-V(x)}\,\mathrm{d}x = {{ex_as_surp:.4f}}$ nats, computed by
numerical integration. A Gaussian $Q(x) = \mathcal{N}(x; m, \Sigma_q)$ on the whole line has
finite free energy here.

(a) Using $\mathbb{E}_Q[e^{x}] = e^{m + \Sigma_q/2}$, write the free energy
$\mathcal{F}_{\mathrm{G}}(m, \Sigma_q)$ of the Gaussian family in closed form.

(b) Show that at its minimum $(\hat m, \hat\Sigma_q)$,
$$
\hat\Sigma_q \;=\; \frac{1}{1 - \hat m}, \qquad \hat m + \exp\!\Big(\hat m + \frac{1}{2(1 - \hat m)}\Big) \;=\; 0 .
$$

(c) Compute the Laplace free energy at the mode $\hat x$, where $\hat x + e^{\hat x} = 0$,
compare both free energies with the surprise, and use the leading correction to Laplace's
method from [lesson:4.3] to predict the sign of the Laplace error.
---solution---
(a) Under $Q$, $\mathbb{E}_Q[x^2] = m^2 + \Sigma_q$, $\mathbb{E}_Q[e^{x}] = e^{m + \Sigma_q/2}$,
and $\mathrm{H}[Q] = \tfrac12\ln(2\pi e\,\Sigma_q)$. So
$\mathcal{F}_{\mathrm{G}} = \mathbb{E}_Q[V] - \mathrm{H}[Q]$ is

$$
\mathcal{F}_{\mathrm{G}}(m, \Sigma_q) \;=\; \tfrac12(m^2 + \Sigma_q) + \exp\!\Big(m + \frac{\Sigma_q}{2}\Big) - \tfrac12\ln(2\pi e\,\Sigma_q).
$$ {#fgauss-asym}

(b) The two partial derivatives of [eq:fgauss-asym] are

$$
\frac{\partial\mathcal{F}_{\mathrm{G}}}{\partial m} \;=\; m + e^{m + \Sigma_q/2},
\qquad
\frac{\partial\mathcal{F}_{\mathrm{G}}}{\partial\Sigma_q} \;=\; \tfrac12 + \tfrac12 e^{m + \Sigma_q/2} - \frac{1}{2\Sigma_q}.
$$

Setting the first to zero gives $e^{m + \Sigma_q/2} = -m$, so $\hat m < 0$. Substituting
into the second gives $\tfrac12(1 - \hat m) = 1/(2\hat\Sigma_q)$, so $\hat\Sigma_q = 1/(1 - \hat m)$,
and substituting that into the first gives the scalar equation stated. Its root is
$\hat m = {{ex_as_gvi_m:.4f}}$, with $\hat\Sigma_q = {{ex_as_gvi_var:.4f}}$ and standard
deviation ${{ex_as_gvi_sd:.4f}}$, against the exact posterior mean ${{ex_as_mean:.4f}}$ and
standard deviation ${{ex_as_sd:.4f}}$.

(c) At $\hat x = {{ex_as_xhat:.4f}}$, $V''(\hat x) = 1 + e^{\hat x} = 1 - \hat x$, so the
Laplace variance is $1/(1 - \hat x) = {{ex_as_lap_var:.4f}}$ and

$$
\mathcal{F}_{\mathrm{Lap}}(\hat x) \;=\; V(\hat x) - \tfrac12\ln\big(2\pi/(1 - \hat x)\big) \;=\; {{ex_as_lap_F:.4f}} \text{ nats},
$$

which is below the surprise ${{ex_as_surp:.4f}}$ by ${{ex_as_lap_under:.4f}}$ nats. The
Gaussian family gives $\mathcal{F}_{\mathrm{G}}(\hat m, \hat\Sigma_q) = {{ex_as_gvi_F:.4f}}$
nats, above the surprise by ${{ex_as_gvi_gap:.4f}}$, as every free energy of a density
must be.

[lesson:4.3] expanded the error of Laplace's method as
$\ln P - \ln P_{\mathrm{Lap}} \approx \ell''''/(8\ell''^2) + 5\ell'''^2/(24|\ell''|^3)$ at the
mode. Here $\ell'' = -(1 + e^{\hat x})$ and $\ell''' = \ell'''' = -e^{\hat x}$, so the two
terms are ${{ex_as_c4:.4f}}$ and ${{ex_as_c3:.4f}}$, summing to ${{ex_as_pred:.4f}}$.
Since $\mathcal{F}_{\mathrm{Lap}} - (-\ln P) = \ln P - \ln P_{\mathrm{Lap}}$, the
prediction is that the Laplace free energy lies below the surprise by about
${{ex_as_pred:.4f}}$ in signed terms; the computed difference is ${{ex_as_lap_err:.4f}}$.
The quartic term, negative because $\ell'''' < 0$, outweighs the cubic one.
:::

::: exercise Changes of coordinate &dagger;
Let $\chi : \mathbb{R} \to (0, \infty)$ be a smooth bijection with $\chi'(\rho) > 0$, and
write $d = \chi(\rho)$. A density $Q_\rho$ on $\mathbb{R}$ induces the density
$Q_d(d) = Q_\rho(\chi^{-1}(d))/\chi'(\chi^{-1}(d))$ on $(0, \infty)$, and the joint
density transforms by the same rule, $P_\rho(u, \rho) = P_d(u, \chi(\rho))\,\chi'(\rho)$
([lesson:2.2]).

(a) Prove that $\mathcal{F}[Q_\rho, u] = \mathcal{F}[Q_d, u]$, although the differential
entropies differ by $\mathbb{E}_{Q_\rho}[\ln\chi'(\rho)]$.

(b) Let $\ell_d(d) = \ln P_d(u, d)$ have a nondegenerate maximum at $\hat d$, and
$\ell_\rho(\rho) = \ln P_\rho(u, \rho)$. Show that $\chi^{-1}(\hat d)$ is a stationary point
of $\ell_\rho$ if and only if $\chi''$ vanishes there. Show that for
$\chi(\rho) = e^{\rho}(\rho^2 - 2\rho + 3)$ and a model with $\hat d = \chi(-1)$, the two
Laplace free energies are equal. Then evaluate both on Week&nbsp;2's sound model with
$\chi(\rho) = e^{\rho}$.

(c) Connect (a) and (b) to the log-normal family of [lesson:4.3].
---solution---
(a) With $d = \chi(\rho)$, $\mathrm{d}d = \chi'(\rho)\,\mathrm{d}\rho$ and
$Q_d(\chi(\rho))\,\chi'(\rho) = Q_\rho(\rho)$, so

$$
\begin{aligned}
\mathcal{F}[Q_d, u]
&\;=\; \int_0^\infty Q_d(d)\ln\frac{Q_d(d)}{P_d(u, d)}\,\mathrm{d}d
\;=\; \int_{-\infty}^\infty Q_d(\chi(\rho))\,\chi'(\rho)\,\ln\frac{Q_d(\chi(\rho))\,\chi'(\rho)}{P_d(u, \chi(\rho))\,\chi'(\rho)}\,\mathrm{d}\rho \\
&\;=\; \int_{-\infty}^\infty Q_\rho(\rho)\ln\frac{Q_\rho(\rho)}{P_\rho(u, \rho)}\,\mathrm{d}\rho \;=\; \mathcal{F}[Q_\rho, u],
\end{aligned}
$$

where the second step multiplies the numerator and denominator of the ratio by
$\chi'(\rho)$. Separately,
$\mathrm{H}[Q_d] = -\int Q_\rho\ln\big(Q_\rho/\chi'\big)\,\mathrm{d}\rho = \mathrm{H}[Q_\rho] + \mathbb{E}_{Q_\rho}[\ln\chi'(\rho)]$,
and the expected energy changes by the same amount, so the difference is unchanged.

(b) $\ell_\rho(\rho) = \ell_d(\chi(\rho)) + \ln\chi'(\rho)$, so

$$
\ell_\rho'(\rho) \;=\; \ell_d'(\chi(\rho))\,\chi'(\rho) + \frac{\chi''(\rho)}{\chi'(\rho)} .
$$

At $\rho_0 = \chi^{-1}(\hat d)$ the first term is zero, so $\ell_\rho'(\rho_0) = \chi''(\rho_0)/\chi'(\rho_0)$,
which is zero exactly when $\chi''(\rho_0) = 0$. Where $\chi''(\rho_0) \ne 0$ the mode in
$\rho$ is at a different physical state.

For $\chi(\rho) = e^{\rho}(\rho^2 - 2\rho + 3)$: $\rho^2 - 2\rho + 3 = (\rho - 1)^2 + 2 > 0$,
$\chi \to 0$ as $\rho \to -\infty$ and $\chi \to \infty$ as $\rho \to \infty$, and
$\chi'(\rho) = e^{\rho}(\rho^2 + 1) > 0$, so $\chi$ is a bijection onto $(0, \infty)$. Further,
$\chi''(\rho) = e^{\rho}(\rho + 1)^2$ and $\chi'''(\rho) = e^{\rho}(\rho + 1)(\rho + 3)$, both
zero at $\rho_0 = -1$. Differentiating $\ell_\rho'$ again,

$$
\ell_\rho''(\rho) \;=\; \ell_d''(\chi)\,\chi'^2 + \ell_d'(\chi)\,\chi'' + \frac{\chi'''}{\chi'} - \Big(\frac{\chi''}{\chi'}\Big)^2,
$$

which at $\rho_0$ is $\ell_d''(\hat d)\,\chi'(\rho_0)^2 < 0$. So $\rho_0$ is a nondegenerate
maximum of $\ell_\rho$, and

$$
\mathcal{F}^{(\rho)}_{\mathrm{Lap}} \;=\; -\ell_d(\hat d) - \ln\chi'(\rho_0) - \tfrac12\ln\frac{2\pi}{|\ell_d''(\hat d)|\,\chi'(\rho_0)^2}
\;=\; -\ell_d(\hat d) - \tfrac12\ln\frac{2\pi}{|\ell_d''(\hat d)|} \;=\; \mathcal{F}^{(d)}_{\mathrm{Lap}} .
$$

So a nonlinear change of coordinates need not change the Laplace free energy. In general
it does. On Week&nbsp;2's sound model with $\rho = \ln d$, $\chi'' = e^{\rho} \ne 0$, and the
mode of $\ell_\rho$ is at $\hat\rho = {{lapr_rho:.4f}}$, the physical state
$e^{\hat\rho} = {{lapr_d:.4f}}$ against $\hat d = {{w2_mode:.4f}}$. The Laplace free energy
in $\rho$ is ${{lapr_F:.4f}}$ nats, against ${{lap_F:.4f}}$ in $d$ and the surprise
${{w2_surp:.4f}}$: the first lies below the surprise and the second above it.

(c) For $\chi(\rho) = e^{\rho}$, $\ln\chi'(\rho) = \rho$, so part (a) gives
$\mathrm{H}[Q_d] = \tfrac12\ln(2\pi e\,\Sigma_\rho) + m_\rho$ for a Gaussian $Q_\rho$ with
mean $m_\rho$ and variance $\Sigma_\rho$, the log-normal entropy of [lesson:4.3]. The
log-normal free energy, ${{ln_F:.4f}}$ nats on the sound model, is the free energy of an
actual density, the same whether computed in $\rho$ or in $d$, and lies above the
surprise by ${{ln_gap:.4f}}$ nats. The Laplace free energy is computed with a quadratic
approximation to the log joint, is generally different from the exact free energy
of the Laplace recognition density for the original model, and, as part (b) shows,
depends on the coordinate in which the quadratic expansion is made.
:::

::: exercise The two-cause model: fixed spread and averaged rates
In the model of [lesson:4.5], $s_i \sim \mathrm{Exp}(\lambda_i)$, $u_1 = s_2 s_1^2 + z_1$ and
$u_2 = s_1 s_2^2 + z_2$ with standard normal noise.

(a) For Gamma factors with means $m_1, m_2$ and a common shape $k$, use
$\mathbb{E}_q[s^2] = m^2 c_2(k)$ and $\mathbb{E}_q[s^4] = m^4 c_4(k)$ from [lesson:4.5] to show
that the coefficient of variation of the predicted sensor output $s_2 s_1^2$ satisfies
$\mathrm{CV}_Q[s_2 s_1^2]^2 = (5k + 6)/k^2$, whatever the means. Evaluate it at $k = 1$ and
as $k \to \infty$.

(b) With Exponential factors, each trial's $q_{n,i}$ has mean $m_{n,i}$ and rate
$1/m_{n,i}$. Show that the M-step of [lesson:4.5] is the harmonic mean of the trial
rates and that the averaging rule is their arithmetic mean. For two trials whose rates
are $r$ and $r/2$, compute both and their ratio.

(c) Using the table in [lesson:4.5], state which effect each of these three comparisons
measures, and why the third is larger than the second: Exponential with the M-step against
Gamma, one step from the true rates; the averaging rule against the M-step, one step from
the true rates; the same two rules at their fixed points.
---solution---
(a) By independence, $\mathbb{E}_Q[s_2 s_1^2] = m_2\,m_1^2 c_2(k)$ and
$\mathbb{E}_Q[(s_2 s_1^2)^2] = m_2^2 c_2(k)\,m_1^4 c_4(k)$. Dividing the variance by the
squared mean cancels $m_1$ and $m_2$:

$$
\mathrm{CV}_Q[s_2 s_1^2]^2 \;=\; \frac{c_2(k)\,c_4(k)}{c_2(k)^2} - 1 \;=\; \frac{c_4(k)}{c_2(k)} - 1 \;=\; \frac{(k + 2)(k + 3)}{k^2} - 1 \;=\; \frac{5k + 6}{k^2}.
$$

At $k = 1$ this is $11$, so the coefficient of variation is $\sqrt{11} = {{exp_pred_cv:.4f}}$,
the value fixed by the Exponential family. As $k \to \infty$ it behaves as $5/k \to 0$.

(b) The M-step is $\lambda_i = \big(\frac1N\sum_n m_{n,i}\big)^{-1}$. Since
$m_{n,i} = 1/r_{n,i}$ for the trial rates $r_{n,i}$, this is
$\big(\frac1N\sum_n 1/r_{n,i}\big)^{-1}$, the harmonic mean. The averaging rule is
$\frac1N\sum_n r_{n,i}$, the arithmetic mean. For rates $r$ and $r/2$: the arithmetic mean
is $3r/4$, the harmonic mean is $2/(1/r + 2/r) = 2r/3$, and their ratio is $9/8$.

(c) Exponential against Gamma, one step from $\lambda^{*}$, measures the fixed
coefficient of variation (effect 1): ${{tc_evg_pin_1:.0f}}$ and ${{tc_evg_pin_2:.0f}}$ per
cent on this sample. The averaging rule against the M-step, one step from $\lambda^{*}$,
measures effect 2: ${{tc_am_over_hm_1:.1f}}$ and ${{tc_am_over_hm_2:.1f}}$ per cent. At the
fixed points the difference is ${{tc_amfit_over_hm_1:.1f}}$ and
${{tc_amfit_over_hm_2:.1f}}$ per cent. It is larger because the loop multiplies each rule's
one-step displacement (effect 3), and the averaging rule starts from a larger one. The comparisons hold on this sample and these
true rates, not on every trial or every sample.
:::

## Further reading

- Blei, Kucukelbir and McAuliffe, ["Variational inference: A review for
  statisticians"](https://doi.org/10.1080/01621459.2017.1285773), *Journal of the
  American Statistical Association* 112 (2017), 859&ndash;877. Coordinate ascent,
  exponential-family updates and stochastic variants, in statisticians' notation.
- Neal and Hinton, ["A view of the EM algorithm that justifies incremental, sparse,
  and other variants"](https://doi.org/10.1007/978-94-011-5014-9_12), in Jordan (ed.),
  *Learning in Graphical Models* (Springer, 1998), 355&ndash;368. EM as coordinate
  descent on one free energy, the view [lesson:4.4] takes.
- Winn and Bishop, ["Variational message passing"](https://jmlr.org/papers/v6/winn05a.html),
  *Journal of Machine Learning Research* 6 (2005), 661&ndash;694. The local form of the
  free-form update in [lesson:4.2], for conjugate-exponential models.
- Turner and Sahani, ["Two problems with variational expectation maximisation for
  time-series models"](https://doi.org/10.1017/CBO9780511984679.006), in Barber, Cemgil
  and Chiappa (eds), *Bayesian Time Series Models* (Cambridge University Press, 2011),
  104&ndash;124. Shows on simple time-series models that the parameters learned by
  variational EM carry systematic biases, that a mean-field approximation can be less
  biased than a structured one, and that the approximate posterior's uncertainty can be
  smallest where the approximation is worst. [lesson:4.5] measures a bias of the first
  kind on a static model.
