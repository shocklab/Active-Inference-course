---
title: Bounding surprise
deck: Variational free energy derived from Jensen's inequality, its three decompositions, and what happens to it when a belief shrinks to a point.
week: 4
time: 65 min
scripts: [w04.js]
---

The first three weeks met the same obstacle in three forms. In [lesson:1.5], exact Bayesian inference required the marginal likelihood $P(o) = \sum_s P(o, s)$, a sum with $k^n$ terms when $n$ state factors each take $k$ values. The same lesson showed that the count alone does not make the sum hard: on a chain it is computed exactly in a few hundred operations. What makes it hard in general is the structure of the model, and no efficient exact algorithm is known for arbitrary discrete models, though some densely connected models have structure that an algorithm can exploit. In [lesson:2.3], gradient ascent avoided the denominator by climbing the log joint $\ell(d) = \ln P(u, d)$ to its mode $\hat d$, and in doing so discarded the spread and the skew of the posterior. In [lesson:3.4], putting that single mode into the learning rules for the variances dropped a posterior-variance term from each of them. On the linear model, the average step of the learned prior variance $\Sigma_p$ was then negative at every value of $\Sigma_p$ whenever the true prior variance was below $3\Sigma_u$, three times the sensory variance, so the learned variance fell towards zero.

Variational inference replaces these compromises with one objective. The agent keeps an explicit distribution $Q(s)$ over hidden states and adjusts it to minimise a quantity that is never smaller than the surprise $-\ln P(o)$ and that can be evaluated from the joint $P(o, s)$ alone.

::: definition Recognition distribution
A **recognition distribution** (for discrete hidden states) or **recognition density** (for continuous ones) is a normalised probability distribution $Q(s)$ over the hidden states $s$ of a generative model, held and adjusted by the agent as an approximation to the exact posterior $P(s \mid o)$. We write $\mathcal{Q}$ for the family of distributions within which $Q$ is allowed to vary.
:::

In [lesson:1.2] and [lesson:1.4], $Q(s)$ appeared in finite examples where we measured its Kullback&ndash;Leibler divergence from a posterior we could compute. This lesson builds the objective without assuming the posterior is known, and proves that minimising it bounds surprise and recovers Bayes' rule at the same time.

## Jensen's inequality

Everything rests on one inequality for the natural logarithm. Its graph is concave, so it lies below each of its tangent lines. At $y = 1$ the logarithm is $0$ and its slope is $1$, so the tangent there is the line $y - 1$.

::: lemma Tangent-line bound on the logarithm
For every real $y > 0$,
$$
\ln y \;\le\; y - 1,
$$ {#log-tangent}
with equality if and only if $y = 1$.
:::

::: derivation Proof of the tangent-line bound
Define $r(y) = \ln y - (y - 1)$ on $(0, \infty)$. Then
$$
r'(y) \;=\; \frac{1}{y} - 1.
$$
On $(0, 1)$, $1/y > 1$, so $r' > 0$ and $r$ is strictly increasing. On $(1, \infty)$, $1/y < 1$, so $r' < 0$ and $r$ is strictly decreasing. So $r$ has its unique maximum on $(0, \infty)$ at $y = 1$, where $r(1) = 0$. For every $y \ne 1$, $r(y) < 0$, which rearranges to $\ln y < y - 1$.
:::

Averaging [eq:log-tangent] against a probability distribution gives the logarithmic case of **Jensen's inequality**.

::: theorem Jensen's inequality for the logarithm
Let $Q(s)$ be a probability distribution (a mass function or a density) and let $Y(s)$ be a random variable that is strictly positive wherever $Q(s) > 0$, with finite expectation $\mathbb{E}_{Q(s)}[Y(s)]$. Then
$$
\ln \mathbb{E}_{Q(s)}[Y(s)] \;\ge\; \mathbb{E}_{Q(s)}[\ln Y(s)],
$$ {#jensen-log}
with equality if and only if $Y(s)$ is constant on the states where $Q(s) > 0$ (outside a set of $Q$-probability zero in the continuous case).
:::

::: derivation Proof from the tangent-line bound
Write $\bar{Y} = \mathbb{E}_{Q(s)}[Y(s)]$, which is positive because $Y > 0$ where $Q$ puts mass. For each state $s$ with $Q(s) > 0$, apply [eq:log-tangent] to $y = Y(s)/\bar{Y}$:
$$
\ln Y(s) - \ln \bar{Y} \;\le\; \frac{Y(s)}{\bar{Y}} - 1.
$$
Take the expectation under $Q(s)$ of both sides. Expectation is linear and $\bar{Y}$ does not depend on $s$, so
$$
\mathbb{E}_{Q(s)}[\ln Y(s)] - \ln \bar{Y}
\;\le\; \frac{\mathbb{E}_{Q(s)}[Y(s)]}{\bar{Y}} - 1
\;=\; 0.
$$
Rearranging gives [eq:jensen-log]. The inequality for each $s$ is strict whenever $Y(s) \ne \bar{Y}$, so the averaged inequality is an equality only if $Y(s) = \bar{Y}$ at every state with $Q(s) > 0$ (outside a set of $Q$-probability zero for continuous $s$). Conversely, a constant $Y$ gives equality.
:::

::: mn Checking the direction on two numbers
Let $Y$ take the values $1$ and $9$ with probabilities ${{jen_q:.1f}}$ and ${{jen_q:.1f}}$. Then $\mathbb{E}_Q[Y] = {{jen_ey:.0f}}$, whose logarithm is ${{jen_ln_ey:.4f}}$. The average of the logarithms is $\tfrac{1}{2}(\ln 1 + \ln 9) = {{jen_e_lny:.4f}}$. The first exceeds the second by ${{jen_gap:.4f}}$ nats.
:::

The same argument works for any convex function, with the tangent line replaced by a line through one point of the graph that the graph never goes below. [lesson:4.5] uses it with $g(x) = 1/x$.

::: theorem Jensen's inequality for a convex function
Let $g$ be convex and differentiable on an open interval $I$, and let $X$ take values in $I$ with finite expectation $\bar X = \mathbb{E}_{Q}[X]$ and with $\mathbb{E}_Q[g(X)]$ finite. Then
$$
g\big(\mathbb{E}_{Q}[X]\big) \;\le\; \mathbb{E}_{Q}[g(X)].
$$ {#jensen-convex}
If $g$ is strictly convex, equality holds only if $X$ is constant outside a set of $Q$-probability zero.
:::

::: derivation Proof from a supporting line
A differentiable convex function has a non-decreasing derivative, so for every $x \in I$
$$
g(x) - g(\bar X) - g'(\bar X)(x - \bar X) \;=\; \int_{\bar X}^{x} \big(g'(t) - g'(\bar X)\big)\,\mathrm{d}t \;\ge\; 0,
$$
because for $x > \bar X$ the integrand is non-negative, and for $x < \bar X$ both the integrand and the orientation of the interval change sign. So the graph of $g$ lies above the line $g(\bar X) + g'(\bar X)(x - \bar X)$. Put $x = X$ and take the expectation; the linear term averages to zero:
$$
\mathbb{E}_Q[g(X)] \;\ge\; g(\bar X) + g'(\bar X)\,\big(\mathbb{E}_Q[X] - \bar X\big) \;=\; g(\bar X).
$$
If $g$ is strictly convex, $g'$ is strictly increasing and the integral is strictly positive for every $x \ne \bar X$, so equality forces $X = \bar X$ outside a set of probability zero. The logarithmic case is this result applied to the convex function $-\ln$.
:::

## The bound on surprise

Now apply [eq:jensen-log] to the marginal likelihood. The derivation is written with sums; for continuous states replace $\sum_s$ by $\int \mathrm{d}s$ and "every state" by "outside a set of probability zero".

::: derivation Constructing the upper bound on surprise
Fix an observation $o$ with $0 < P(o) < \infty$, and let $Q(s)$ be any recognition distribution. Write $S_Q$ for the set of states with $Q(s) > 0$. Terms with $Q(s) = 0$ contribute nothing to an expectation under $Q$, with the convention $0 \ln 0 = 0$.

If $Q$ puts mass on a state with $P(o, s) = 0$, then $Q(s)\ln\frac{Q(s)}{P(o,s)} = +\infty$ at that state, the sum below is $+\infty$, and the inequality to be proved holds trivially. So suppose $P(o, s) > 0$ at every $s \in S_Q$. Dropping the non-negative terms outside $S_Q$ and using that $\ln$ is increasing,
$$
\ln P(o) \;=\; \ln \sum_{s} P(o, s) \;\ge\; \ln \sum_{s \in S_Q} P(o, s)
\;=\; \ln \sum_{s \in S_Q} Q(s)\,\frac{P(o, s)}{Q(s)}
\;=\; \ln \mathbb{E}_{Q(s)}\!\left[\frac{P(o, s)}{Q(s)}\right].
$$
The ratio $Y(s) = P(o, s)/Q(s)$ is strictly positive on $S_Q$, so [eq:jensen-log] applies:
$$
\ln P(o) \;\ge\; \mathbb{E}_{Q(s)}\!\left[\ln \frac{P(o, s)}{Q(s)}\right]
\;=\; \sum_{s} Q(s) \ln \frac{P(o, s)}{Q(s)}.
$$
Multiplying by $-1$ reverses the inequality and inverts the ratio inside the logarithm:
$$
-\ln P(o) \;\le\; \sum_{s} Q(s) \ln \frac{Q(s)}{P(o, s)}.
$$
Equality needs both steps to be equalities. The first is an equality only if $S_Q$ contains every state with $P(o, s) > 0$. The second, by the equality condition of [eq:jensen-log], is an equality only if $P(o, s) = c\,Q(s)$ on $S_Q$ for some constant $c$. Summing over $s$ gives $P(o) = c$, so equality holds if and only if $Q(s) = P(o, s)/P(o) = P(s \mid o)$ for every $s$.
:::

So a recognition distribution may leave out states the posterior allows. The bound still holds for it, and it is strict. What a finite right-hand side forbids is the opposite: $Q$ may not put mass where the joint is zero.

::: definition Variational free energy and the evidence lower bound
Given a generative model $P(o, s)$, an observation $o$ and a recognition distribution $Q(s)$, the variational free energy of [lesson:1.2] is
$$
\mathcal{F}[Q, o] \;=\; \mathbb{E}_{Q(s)}\!\left[\ln \frac{Q(s)}{P(o, s)}\right]
\;=\; \sum_{s} Q(s) \ln \frac{Q(s)}{P(o, s)},
$$ {#vfe-def}
with $\int Q(s)\ln\frac{Q(s)}{P(o,s)}\,\mathrm{d}s$ for continuous states. Its negative is the **evidence lower bound**,
$$
\mathrm{ELBO}[Q, o] \;=\; -\mathcal{F}[Q, o] \;=\; \mathbb{E}_{Q(s)}\!\left[\ln \frac{P(o, s)}{Q(s)}\right] \;\le\; \ln P(o).
$$ {#elbo-def}
:::

Machine learning papers usually maximise the ELBO; active inference papers minimise $\mathcal{F}$. The two problems differ only in sign. Evaluating $\mathcal{F}[Q, o]$ at a state needs $Q(s)$ and the joint $P(o, s) = P(o \mid s)P(s)$. The marginal likelihood $P(o)$ does not appear in the sum.

## Three decompositions of one functional

Rearranging the logarithm $\ln\frac{Q(s)}{P(o,s)}$ in three ways gives three readings of $\mathcal{F}[Q, o]$: keep $P(o, s)$ whole, factor it as prior times likelihood, or factor it as posterior times evidence.

::: derivation Equivalence of the three decompositions
Start from [eq:vfe-def] and use $\ln(A/B) = \ln A - \ln B$.

1. *Energy minus entropy.* Keep $P(o, s)$ together:
$$
\begin{aligned}
\mathcal{F}[Q, o]
&\;=\; \sum_s Q(s)\ln Q(s) \;-\; \sum_s Q(s)\ln P(o, s) \\
&\;=\; \underbrace{-\mathbb{E}_{Q(s)}[\ln P(o, s)]}_{\text{expected energy}} \;-\; \underbrace{\left(-\sum_s Q(s)\ln Q(s)\right)}_{\mathrm{H}[Q]}.
\end{aligned}
$$ {#vfe-energy-entropy}

2. *Complexity minus accuracy.* Substitute $P(o, s) = P(o \mid s)\,P(s)$:
$$
\begin{aligned}
\mathcal{F}[Q, o]
&\;=\; \sum_s Q(s)\Big[\ln Q(s) - \ln P(s) - \ln P(o \mid s)\Big] \\
&\;=\; \underbrace{D_{\mathrm{KL}}\!\big[Q(s) \,\big\|\, P(s)\big]}_{\text{complexity}} \;-\; \underbrace{\mathbb{E}_{Q(s)}[\ln P(o \mid s)]}_{\text{accuracy}}.
\end{aligned}
$$ {#vfe-comp-acc}

3. *Posterior divergence plus surprise.* Substitute the product rule the other way round, $P(o, s) = P(s \mid o)\,P(o)$:
$$
\begin{aligned}
\mathcal{F}[Q, o]
&\;=\; \sum_s Q(s)\Big[\ln Q(s) - \ln P(s \mid o) - \ln P(o)\Big] \\
&\;=\; \underbrace{D_{\mathrm{KL}}\!\big[Q(s) \,\big\|\, P(s \mid o)\big]}_{\ge\, 0} \;-\; \ln P(o),
\end{aligned}
$$ {#vfe-kl-surprise}
where the last step uses $\sum_s Q(s) = 1$, since $P(o)$ does not depend on $s$.
:::

::: definition Expected energy, complexity and accuracy
In [eq:vfe-energy-entropy] and [eq:vfe-comp-acc]:

- The **expected energy** is $U[Q, o] = -\mathbb{E}_{Q(s)}[\ln P(o, s)]$, the average under $Q$ of the joint surprisal $-\ln P(o, s)$.
- The **complexity** is $D_{\mathrm{KL}}[Q(s) \,\|\, P(s)] \ge 0$, the divergence of the recognition distribution from the prior.
- The **accuracy** is $\mathbb{E}_{Q(s)}[\ln P(o \mid s)]$, the average log-likelihood of the observation under $Q$. For discrete observations $P(o \mid s) \le 1$, so accuracy is at most $0$, with $0$ meaning that $o$ is predicted with certainty.
:::

Each decomposition answers a different question about the same number.

- [eq:vfe-kl-surprise] says what minimising $\mathcal{F}$ achieves. The term $-\ln P(o)$ does not depend on $Q$, so every decrease of $\mathcal{F}[Q, o]$ in $Q$ is an equal decrease of $D_{\mathrm{KL}}[Q(s) \,\|\, P(s \mid o)]$, although neither the posterior nor $P(o)$ is ever computed. With Gibbs' inequality from [lesson:1.2] it also gives a second proof of $\mathcal{F}[Q, o] \ge -\ln P(o)$.
- [eq:vfe-comp-acc] describes a trade-off. Concentrating $Q$ on the state that best predicts $o$ raises accuracy, and costs complexity if that state had low prior probability. Setting $Q$ equal to the prior costs no complexity and gives poor accuracy when the observation was unexpected.
- [eq:vfe-energy-entropy] is the form algorithms evaluate: it needs only expectations under $Q$ of $\ln P(o, s)$ and of $\ln Q(s)$.

### The name

Write $E(s) = -\ln P(o, s)$ and call it the energy of state $s$. The expected energy is then $U = \mathbb{E}_Q[E]$, and the distribution
$$
Q^{*}(s) \;=\; \frac{e^{-E(s)}}{Z}, \qquad Z \;=\; \sum_s e^{-E(s)} \;=\; \sum_s P(o, s) \;=\; P(o),
$$
is the posterior. Substituting $E(s) = -\ln Z - \ln Q^{*}(s)$ into [eq:vfe-energy-entropy],
$$
U - \mathrm{H}[Q] \;=\; \sum_s Q(s)\big[E(s) + \ln Q(s)\big] \;=\; \sum_s Q(s) \ln\frac{Q(s)}{Q^{*}(s)} \;-\; \ln Z \;=\; D_{\mathrm{KL}}[Q \,\|\, Q^{*}] - \ln Z .
$$ {#gibbs-variational}
So the minimum of $U - \mathrm{H}$ over all $Q$ is $-\ln Z$, reached at $Q = Q^{*}$. In statistical mechanics a system with energies $E(s)$ at temperature $T$ has its equilibrium distribution proportional to $e^{-E(s)/T}$, which minimises $U - T\,\mathrm{H}$, and the minimum $-T \ln Z$ is the Helmholtz free energy. With $T = 1$ and $E = -\ln P(o, s)$ this is [eq:vfe-kl-surprise] again, which is why $\mathcal{F}$ is called a free energy and its minimum is the surprise. [lesson:4.2] uses [eq:gibbs-variational] for one factor of $Q$ at a time.

::: warning Expected energy under $Q$ against joint entropy under $P$
Write the two sums side by side before reading a paper that drops the subscripts on its expectations:
$$
\mathrm{H}[P(o, s)] \;=\; -\sum_{o}\sum_{s} P(o, s)\ln P(o, s),
\qquad
U[Q, o] \;=\; -\sum_{s} Q(s)\ln P(o, s).
$$
Both contain $-\ln P(o, s)$, and they are different objects. The left one sums over both $o$ and $s$, weighted by the model's own joint; on [lesson:1.3]'s three-state model it is the constant ${{joint_entropy_P:.4f}}$ nats, fixed before any sensor is read. The right one uses the one observation actually received and averages over $s$ with the agent's recognition distribution. Reading the second as the first turns a function of $Q$ into a constant, and there is then nothing left to optimise.
:::

### Numerical check on the three-state model

Return to [lesson:1.3]'s model over the states (leopard, baboon, nothing), with prior $P(s) = ({{p_leop:.2f}}, {{p_bab:.2f}}, {{p_noth:.2f}})$ and likelihood of a tawny flash $P(o = \text{tawny} \mid s) = ({{a_tawny_leop:.2f}}, {{a_tawny_bab:.2f}}, {{a_tawny_noth:.2f}})$. The joint at the tawny flash is $({{joint_leop:.3f}}, {{joint_bab:.3f}}, {{joint_noth:.3f}})$, so $P(o) = {{ev_tawny:.3f}}$, the surprise is $-\ln P(o) = {{surp_tawny:.4f}}$ nats and the posterior is $P(s \mid o) = ({{post_leop:.4f}}, {{post_bab:.4f}}, {{post_noth:.4f}})$.

Evaluate each term of the three decompositions at five recognition distributions:

| $Q(s)$ | $U[Q,o]$ | $\mathrm{H}[Q]$ | Complexity | Accuracy | $D_{\mathrm{KL}}[Q\|P(\cdot\mid o)]$ | $\mathcal{F}[Q,o]$ |
|---|---:|---:|---:|---:|---:|---:|
| Prior $({{p_leop:.2f}}, {{p_bab:.2f}}, {{p_noth:.2f}})$ | ${{d_prior_U:.4f}}$ | ${{d_prior_H:.4f}}$ | ${{d_prior_comp:.4f}}$ | ${{d_prior_acc:.4f}}$ | ${{d_prior_klpost:.4f}}$ | ${{d_prior_F:.4f}}$ |
| Uniform | ${{d_unif_U:.4f}}$ | ${{d_unif_H:.4f}}$ | ${{d_unif_comp:.4f}}$ | ${{d_unif_acc:.4f}}$ | ${{d_unif_klpost:.4f}}$ | ${{d_unif_F:.4f}}$ |
| Sharp $({{q_sharp_leop:.2f}}, {{q_sharp_other:.2f}}, {{q_sharp_other:.2f}})$ | ${{d_sharp_U:.4f}}$ | ${{d_sharp_H:.4f}}$ | ${{d_sharp_comp:.4f}}$ | ${{d_sharp_acc:.4f}}$ | ${{d_sharp_klpost:.4f}}$ | ${{d_sharp_F:.4f}}$ |
| Point mass on leopard | ${{d_point_U:.4f}}$ | ${{d_point_H:.4f}}$ | ${{d_point_comp:.4f}}$ | ${{d_point_acc:.4f}}$ | ${{d_point_klpost:.4f}}$ | ${{d_point_F:.4f}}$ |
| Posterior | ${{d_post_U:.4f}}$ | ${{d_post_H:.4f}}$ | ${{d_post_comp:.4f}}$ | ${{d_post_acc:.4f}}$ | ${{d_post_klpost:.4f}}$ | ${{d_post_F:.4f}}$ |

Any row checks all three decompositions. On the sharp row, $U - \mathrm{H} = {{d_sharp_U:.4f}} - {{d_sharp_H:.4f}} = {{d_sharp_F:.4f}}$, complexity minus accuracy is ${{d_sharp_comp:.4f}} - ({{d_sharp_acc:.4f}}) = {{d_sharp_F:.4f}}$, and divergence plus surprise is ${{d_sharp_klpost:.4f}} + {{surp_tawny:.4f}} = {{d_sharp_F:.4f}}$.

The point mass shows what the entropy term does. The expected energy $U[Q, o] = \sum_s Q(s) E(s)$ is linear in $Q$, and a linear function on the simplex of distributions is smallest at a vertex, the point mass on the state with the lowest energy, which is the state with the largest joint $P(o, s)$: the maximum a posteriori (MAP) state, here the leopard. In the table that row has the lowest $U$, ${{d_point_U:.4f}}$ against the posterior's ${{d_post_U:.4f}}$. A discrete point mass has entropy $0$, so its free energy is finite, $\mathcal{F} = -\ln P(o, \text{leopard}) = {{d_point_F:.4f}}$, which exceeds the surprise by $-\ln P(\text{leopard} \mid o) = {{d_point_klpost:.4f}}$ nats. Minimising $U$ alone would select this point mass. Minimising $U - \mathrm{H}$ selects the posterior, whose larger entropy (${{d_post_H:.4f}}$ nats) more than pays for its larger expected energy.

The first and last rows separate two divergences that look alike. Complexity, $D_{\mathrm{KL}}[Q \,\|\, P(s)]$, measures the distance from the prior and does not involve the observation. The posterior divergence, $D_{\mathrm{KL}}[Q \,\|\, P(s \mid o)]$, measures the distance from the posterior for this $o$. At the prior the complexity is ${{d_prior_comp:.0f}}$ and the divergence is ${{d_prior_klpost:.4f}}$; at the posterior the divergence is ${{d_post_klpost:.0f}}$ and the complexity is ${{d_post_comp:.4f}}$. Moving $Q$ from prior to posterior spends complexity to buy accuracy, from ${{d_prior_acc:.4f}}$ to ${{d_post_acc:.4f}}$, and $\mathcal{F}$ falls from ${{d_prior_F:.4f}}$ to the surprise.

::: widget three-decompositions | The three decompositions of $\mathcal{F}[Q, o]$ on [lesson:1.3]'s three-state model after a tawny flash. At the default belief $Q = ({{w1_def_q1:.2f}}, {{w1_def_q2:.2f}}, {{w1_def_q3:.2f}})$ all three give $\mathcal{F} = {{w1_def_F:.4f}}$ nats, which is ${{w1_def_klpost:.4f}}$ nats above the surprise $-\ln P(o) = {{surp_tawny:.4f}}$. Setting $Q = P(s \mid o)$ removes the posterior divergence; the three bars stay equal to one another for every $Q$.
:::

## Continuous states and differential entropy

For a continuous hidden state $d$ and a continuous observation $u$, as in [week:2], sums become integrals:
$$
\mathcal{F}[Q, u] \;=\; \int Q(d)\ln\frac{Q(d)}{P(u, d)}\,\mathrm{d}d
\;=\; -\mathbb{E}_{Q(d)}[\ln P(u, d)] \;-\; \mathrm{H}[Q].
$$ {#vfe-continuous}
Densities carry units. $Q(d)$ is a density in $d$, with units of inverse distance, while $P(u, d)$ is a density in both arguments, with units of inverse distance times inverse units of $u$. So the ratio $Q(d)/P(u, d)$ has the units of $u$, and $\mathcal{F}$ depends on the unit in which $u$ is measured. [eq:vfe-kl-surprise] says how:
$$
\mathcal{F}[Q, u] \;=\; D_{\mathrm{KL}}[Q(d) \,\|\, P(d \mid u)] \;-\; \ln P(u).
$$
The divergence compares two densities in $d$, so it is dimensionless and does not depend on the unit of $u$. If $u$ is measured in a unit $c$ times smaller, every value of $u$ is multiplied by $c$, the density $P(u)$ is divided by $c$, and $c$ multiplies $\mathrm{d}u$; so $\ln c$ is added to both $\mathcal{F}$ and $-\ln P(u)$ and their difference is unchanged. Differences of $\mathcal{F}$ between recognition densities for the same observation are therefore unaffected, and $\mathcal{F}[Q, u] \ge -\ln P(u)$ holds in any unit, with equality at the posterior. Changing the unit of the hidden state $d$ changes nothing at all, since $Q$ and $P(u, d)$ are both divided by the same factor; Exercise&nbsp;4.6.6 proves the general version.

::: definition Differential entropy
For a probability density $Q(s)$, the **differential entropy** is
$$
\mathrm{H}[Q] \;=\; -\int Q(s)\ln Q(s)\,\mathrm{d}s,
$$ {#diff-entropy}
whenever the integral exists.
:::

The Gaussian case is needed at once. Write $\Sigma_q$ for the variance of a Gaussian recognition density, $Q(d) = \mathcal{N}(d;\, m, \Sigma_q)$.

::: derivation Differential entropy of a one-dimensional Gaussian
The logarithm of the density is
$$
\ln Q(d) \;=\; -\tfrac{1}{2}\ln(2\pi\Sigma_q) \;-\; \frac{(d - m)^2}{2\Sigma_q}.
$$
Take minus its expectation under $Q$. The first term is a constant. The second needs $\mathbb{E}_{Q}[(d - m)^2] = \Sigma_q$, the definition of the variance:
$$
\mathrm{H}[Q] \;=\; \tfrac{1}{2}\ln(2\pi\Sigma_q) \;+\; \frac{\Sigma_q}{2\Sigma_q}
\;=\; \tfrac{1}{2}\ln(2\pi\Sigma_q) + \tfrac{1}{2}
\;=\; \tfrac{1}{2}\ln(2\pi e\,\Sigma_q).
$$ {#gauss-entropy-1d}
It does not depend on the mean $m$. [lesson:4.3] does the multivariate case.
:::

::: warning Differential entropy can be negative, and tends to $-\infty$ at a point
Discrete probabilities satisfy $Q(s) \le 1$, so $-\ln Q(s) \ge 0$ and discrete entropy is never negative. A density can exceed $1$ where it puts its mass into a short interval, and there $-\ln Q < 0$. By [eq:gauss-entropy-1d], $\mathrm{H}[Q] = 0$ at $\Sigma_q = 1/(2\pi e) = {{gauss_zero_H_var:.4f}}$ (standard deviation ${{gauss_zero_H_sd:.4f}}$), is negative for every smaller variance, and tends to $-\infty$ as $\Sigma_q \to 0$.

The surprise $-\ln P(u)$ of a continuous observation can also be negative, wherever the density $P(u)$ exceeds $1$. The inequality $\mathcal{F}[Q, u] \ge -\ln P(u)$ holds whatever the signs, because the difference is a divergence.
:::

This is the contrast with the discrete table above. A discrete point mass has entropy $0$ and finite free energy. A continuous point mass has no density, and the Gaussians that approach it have entropies that tend to $-\infty$, so $-\mathrm{H}$ tends to $+\infty$.

Take [lesson:2.4]'s linear Gaussian model, $d \sim \mathcal{N}({{d_prior:.0f}}, {{var_prior:.0f}})$ and $u \mid d \sim \mathcal{N}(d, {{var_obs:.1f}})$ at $u = {{u_obs:.2f}}$. Its posterior has mean $\hat d = {{lin_dhat:.4f}}$, variance $\Sigma_{\mathrm{post}} = {{lin_spost:.4f}}$, and the surprise is $-\ln P(u) = {{lin_surp:.4f}}$ nats. The log joint is quadratic, $\ln P(u, d) = \ell(\hat d) - (d - \hat d)^2/(2\Sigma_{\mathrm{post}})$, so the expected energy of $Q = \mathcal{N}(\hat d, \Sigma_q)$ is exact for every $\Sigma_q > 0$:
$$
-\mathbb{E}_{Q}[\ln P(u, d)] \;=\; -\ell(\hat d) + \frac{\mathbb{E}_{Q}[(d - \hat d)^2]}{2\Sigma_{\mathrm{post}}}
\;=\; -\ell(\hat d) + \frac{\Sigma_q}{2\Sigma_{\mathrm{post}}}.
$$
Subtracting [eq:gauss-entropy-1d] gives the free energy along a sequence of narrowing Gaussians:

| Variance $\Sigma_q$ | Expected energy $U$ | Entropy $\mathrm{H}[Q]$ | $\mathcal{F} = U - \mathrm{H}$ |
|---|---:|---:|---:|
| $\Sigma_{\mathrm{post}} = {{spike_opt_var:.4f}}$ | ${{spike_opt_U:.4f}}$ | ${{spike_opt_H:.4f}}$ | ${{spike_opt_F:.4f}}$ |
| ${{spike_e2_var:.2f}}$ | ${{spike_e2_U:.4f}}$ | ${{spike_e2_H:.4f}}$ | ${{spike_e2_F:.4f}}$ |
| ${{spike_e4_var:sci0}}$ | ${{spike_e4_U:.4f}}$ | ${{spike_e4_H:.4f}}$ | ${{spike_e4_F:.4f}}$ |
| ${{spike_e8_var:sci0}}$ | ${{spike_e8_U:.4f}}$ | ${{spike_e8_H:.4f}}$ | ${{spike_e8_F:.4f}}$ |

The first row is the posterior itself, and its free energy equals the surprise. As $\Sigma_q \to 0$ the expected energy tends to $-\ell(\hat d) = {{lin_neg_ell_hat:.4f}}$, the quantity MAP estimation minimises, while $-\mathrm{H}[Q]$ grows like $-\tfrac{1}{2}\ln\Sigma_q$ without bound. This is the warning of [lesson:2.3] made exact: a point estimate is not a recognition density of finite free energy.

### What remains of MAP estimation

Something does survive. Hold the variance $\Sigma_q$ fixed and minimise $\mathcal{F}$ over the mean alone. The entropy [eq:gauss-entropy-1d] does not depend on $m$, so only the expected energy matters.

::: derivation The mean of a narrow Gaussian of fixed width
Let $\ell(x)$ be a smooth log joint on the real line and $Q = \mathcal{N}(m, \Sigma_q)$. Expand $\ell$ about $m$ to fourth order,
$$
\ell(x) \;=\; \ell(m) + \ell'(m)(x - m) + \tfrac{1}{2}\ell''(m)(x - m)^2 + \tfrac{1}{6}\ell'''(m)(x - m)^3 + \tfrac{1}{24}\ell''''(\xi)(x - m)^4,
$$
with $\xi$ between $m$ and $x$. Under $Q$, $\mathbb{E}[x - m] = 0$, $\mathbb{E}[(x - m)^2] = \Sigma_q$, $\mathbb{E}[(x - m)^3] = 0$ and $\mathbb{E}[(x - m)^4] = 3\Sigma_q^2$. So, if $\ell''''$ is bounded,
$$
\mathcal{F}(m) \;=\; -\ell(m) \;-\; \tfrac{1}{2}\Sigma_q\,\ell''(m) \;-\; \tfrac{1}{2}\ln(2\pi e\,\Sigma_q) \;+\; O(\Sigma_q^2).
$$
At a nondegenerate peak $\ell'' < 0$, so near it the second term is $+\tfrac{1}{2}\Sigma_q|\ell''(m)|$: a penalty on sharp curvature, of size proportional to the width. Setting the derivative in $m$ to zero,
$$
\ell'(m) + \tfrac{1}{2}\Sigma_q\,\ell'''(m) \;=\; O(\Sigma_q^2).
$$
Let $\hat x$ be a nondegenerate mode, $\ell'(\hat x) = 0$ and $\ell''(\hat x) < 0$, and follow the branch of local minimisers $m$ of $\mathcal{F}$ that lies near $\hat x$ for small $\Sigma_q$. On that branch $\ell'(m) = \ell''(\hat x)(m - \hat x) + O((m - \hat x)^2)$, so
$$
m - \hat x \;=\; -\frac{\Sigma_q\,\ell'''(\hat x)}{2\,\ell''(\hat x)} \;+\; O(\Sigma_q^2).
$$ {#fixed-width-shift}
As $\Sigma_q \to 0$ this local minimiser tends to $\hat x$; when $\hat x$ is the global mode, that is the MAP estimate.
:::

The asymmetric potential of Exercise&nbsp;4.6.5 ([lesson:4.6|the problems]), $\ell(x) = -\big(\tfrac{1}{2}x^2 + e^{x}\big)$, has every Gaussian expectation in closed form, $\mathbb{E}_Q[\ell] = -\tfrac{1}{2}(m^2 + \Sigma_q) - e^{m + \Sigma_q/2}$, so the best mean solves $m + e^{m + \Sigma_q/2} = 0$ exactly. Its mode is $\hat x = {{fw_mode:.4f}}$, and $\ell'' = -(1 + e^{x})$, $\ell''' = -e^{x}$.

| Fixed variance $\Sigma_q$ | Best mean $m$ | $m - \hat x$ | [eq:fixed-width-shift] |
|---|---:|---:|---:|
| ${{fw_a_S:.1f}}$ | ${{fw_a_m:.4f}}$ | ${{fw_a_shift:.5f}}$ | ${{fw_a_pred:.5f}}$ |
| ${{fw_b_S:.2f}}$ | ${{fw_b_m:.4f}}$ | ${{fw_b_shift:.6f}}$ | ${{fw_b_pred:.6f}}$ |
| ${{fw_c_S:.3f}}$ | ${{fw_c_m:.4f}}$ | ${{fw_c_shift:.7f}}$ | ${{fw_c_pred:.7f}}$ |

The shift falls in proportion to $\Sigma_q$ and [eq:fixed-width-shift] gives its leading term. So MAP estimation is what variational inference reduces to when the width is held fixed and small: the mean that minimises $\mathcal{F}$ tends to the mode, while the free energy itself, through $-\mathrm{H}$, tends to $+\infty$. [lesson:4.3] lets the width vary as well, and obtains the Laplace approximation.
