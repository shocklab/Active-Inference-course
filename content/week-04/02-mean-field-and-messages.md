---
title: Mean-field and variational messages
deck: The free-form mean-field update proved twice, reduced to local messages on a factorised model, compared with the sum-product messages of Week 1, and applied to a sensory precision that is itself unknown.
week: 4
time: 85 min
---

Minimising $\mathcal{F}[Q, o]$ over every normalised distribution $Q(s)$ returns $Q(s) = P(s \mid o)$, and with it the cost of representing the posterior. If the hidden state has $K$ factors $s = (s_1, \dots, s_K)$ and factor $i$ takes values in a finite set $\mathcal{S}_i$ with $|\mathcal{S}_i|$ elements, an arbitrary distribution over $s$ needs $\prod_i |\mathcal{S}_i| - 1$ numbers. [lesson:1.5] showed that this count does not by itself make inference hard: on a chain, sums over the joint are computed exactly in time linear in $K$. For a model whose dependencies are dense, though, no exact method is known that avoids the exponential cost, and there the usual remedy is to restrict the family $\mathcal{Q}$ within which $Q$ may vary. This lesson studies one restriction: independence between factors.

::: definition Free-form and fixed-form variational families
Let the hidden state split into $K$ factors $s = (s_1, \dots, s_K)$.

- A **free-form mean-field** family contains every product
$$
Q(s_1, \dots, s_K) \;=\; \prod_{i=1}^{K} q_i(s_i),
$$ {#mf-product}
where each factor $q_i$ is any normalised distribution over $s_i$. The family fixes which variables are independent and nothing else.

- A **fixed-form** family restricts $Q$, or each factor $q_i$, to a parametric family, such as Gaussians with mean $m$ and variance $\Sigma_q$. Minimising $\mathcal{F}$ is then ordinary calculus in the parameters.
:::

This lesson develops the free-form case. [lesson:4.3] develops the fixed-form Gaussian, and [lesson:4.5] shows what happens when a fixed form and a mean-field product are imposed together and parameters are learned on top.

## The free-form coordinate update

[lesson:1.4] derived the optimal factor for two binary causes with the other factor held fixed. Here the same result is proved for $K$ factors of any kind, discrete or continuous, and the change in $\mathcal{F}$ at each update is computed exactly. Two pieces of notation are needed. Write $s_{\setminus i}$ for all hidden variables except $s_i$, and
$$
Q_{\setminus i}(s_{\setminus i}) \;=\; \prod_{j \ne i} q_j(s_j)
$$
for their recognition distribution.

::: derivation Additivity of entropy for a product distribution
Substitute $Q(s) = \prod_{j} q_j(s_j)$ into $\mathrm{H}[Q] = -\sum_{s} Q(s)\ln Q(s)$, and write the logarithm of the product as a sum of logarithms:
$$
\mathrm{H}[Q]
\;=\; -\sum_{s_1}\cdots\sum_{s_K} \left(\prod_{j=1}^{K} q_j(s_j)\right) \sum_{i=1}^{K} \ln q_i(s_i).
$$
Exchange the finite sum over $i$ with the sums over states. For each $i$, the sum over $s_i$ carries $q_i(s_i)\ln q_i(s_i)$ and every other sum carries only $q_j(s_j)$:
$$
\mathrm{H}[Q]
\;=\; -\sum_{i=1}^{K} \left(\sum_{s_i} q_i(s_i)\ln q_i(s_i)\right) \prod_{j \ne i} \left(\sum_{s_j} q_j(s_j)\right)
\;=\; \sum_{i=1}^{K} \mathrm{H}[q_i],
$$ {#mf-entropy-sum}
because each $\sum_{s_j} q_j(s_j) = 1$. The same steps with integrals give the continuous case, provided each $\mathrm{H}[q_j]$ is finite.
:::

::: definition Partially averaged energy
For an observation $o$ and fixed factors $Q_{\setminus i}$, the **partially averaged energy** of factor $i$ is the function of $s_i$ alone
$$
\mathcal{E}_i(s_i) \;=\; -\mathbb{E}_{Q_{\setminus i}(s_{\setminus i})}\!\big[\ln P(o, s_i, s_{\setminus i})\big]
\;=\; -\sum_{s_{\setminus i}} \left(\prod_{j \ne i} q_j(s_j)\right) \ln P(o, s_i, s_{\setminus i}),
$$ {#partial-energy}
with the value $+\infty$ allowed where the averaged log joint is $-\infty$. It is the energy $E(s) = -\ln P(o, s)$ of [lesson:4.1] averaged over every factor except the $i$-th.
:::

::: theorem Free-form coordinate update
Hold $q_j$ fixed for every $j \ne i$, with $\mathrm{H}[q_j]$ finite, and suppose
$$
0 \;<\; Z_i \;=\; \sum_{s_i} \exp\!\big(-\mathcal{E}_i(s_i)\big) \;<\; \infty .
$$
Then among all normalised distributions $q_i$, the free energy $\mathcal{F}\big[q_i Q_{\setminus i}, o\big]$ has the unique minimiser
$$
\hat{q}_i(s_i) \;=\; \frac{1}{Z_i}\exp\!\Big(\mathbb{E}_{Q_{\setminus i}(s_{\setminus i})}\!\big[\ln P(o, s_i, s_{\setminus i})\big]\Big),
$$ {#cavi-update}
that is, $\ln \hat{q}_i(s_i) = -\mathcal{E}_i(s_i) - \ln Z_i$. Replacing any factor $q_i$ by $\hat{q}_i$ lowers $\mathcal{F}$ by exactly
$$
\mathcal{F}\big[q_i Q_{\setminus i}, o\big] \;-\; \mathcal{F}\big[\hat{q}_i Q_{\setminus i}, o\big]
\;=\; D_{\mathrm{KL}}\!\big[q_i \,\big\|\, \hat{q}_i\big] \;\ge\; 0.
$$ {#cavi-drop}
:::

Both conditions on $Z_i$ can fail. $Z_i = \infty$ happens for continuous $s_i$ when $e^{-\mathcal{E}_i}$ is not integrable. $Z_i = 0$ happens when every value of $s_i$ has $\mathcal{E}_i = +\infty$. For an example, take a joint over two binary variables that is positive only at $(0, 0)$ and $(1, 1)$, and fix $q_2 = (\tfrac12, \tfrac12)$. Each value of $s_1$ then meets a value of $s_2$ at which the joint is zero, both values of $\mathcal{E}_1$ are $+\infty$, and every product with this $q_2$ has $\mathcal{F} = +\infty$, so there is nothing to minimise.

::: derivation First proof: Gibbs' inequality
Start from the energy-minus-entropy form of [lesson:4.1], $\mathcal{F}[Q, o] = -\mathbb{E}_Q[\ln P(o, s)] - \mathrm{H}[Q]$. Insert $Q = q_i\,Q_{\setminus i}$, use [eq:mf-entropy-sum], and do the average over $s_{\setminus i}$ first:
$$
\begin{aligned}
\mathcal{F}\big[q_i Q_{\setminus i}, o\big]
&\;=\; -\sum_{s_i} q_i(s_i)\sum_{s_{\setminus i}} Q_{\setminus i}(s_{\setminus i})\ln P(o, s_i, s_{\setminus i}) \;-\; \mathrm{H}[q_i] \;-\; \sum_{j \ne i} \mathrm{H}[q_j] \\
&\;=\; \sum_{s_i} q_i(s_i)\,\mathcal{E}_i(s_i) \;+\; \sum_{s_i} q_i(s_i)\ln q_i(s_i) \;-\; \sum_{j \ne i} \mathrm{H}[q_j].
\end{aligned}
$$
By the definition of $\hat q_i$, $\mathcal{E}_i(s_i) = -\ln \hat{q}_i(s_i) - \ln Z_i$. Substituting,
$$
\begin{aligned}
\mathcal{F}\big[q_i Q_{\setminus i}, o\big]
&\;=\; \sum_{s_i} q_i(s_i)\Big[\ln q_i(s_i) - \ln \hat{q}_i(s_i) - \ln Z_i\Big] \;-\; \sum_{j \ne i} \mathrm{H}[q_j] \\
&\;=\; D_{\mathrm{KL}}\!\big[q_i \,\big\|\, \hat{q}_i\big] \;-\; \ln Z_i \;-\; \sum_{j \ne i} \mathrm{H}[q_j],
\end{aligned}
$$ {#cavi-kl-split}
using $\sum_{s_i} q_i(s_i) = 1$. If $q_i$ puts mass where $\hat q_i = 0$, both sides are $+\infty$. The last two terms do not depend on $q_i$. By Gibbs' inequality from [lesson:1.2], $D_{\mathrm{KL}}[q_i \,\|\, \hat{q}_i] \ge 0$ with equality if and only if $q_i = \hat{q}_i$. Evaluating [eq:cavi-kl-split] at $q_i = \hat q_i$ and subtracting gives [eq:cavi-drop]. For continuous $s_i$ the same lines hold with integrals.
:::

This is the calculation that gave the free energy its name in [lesson:4.1], applied to one factor: there the minimiser of $U - \mathrm{H}$ over all $Q$ was $e^{-E}/Z$, and here the minimiser over one factor is $e^{-\mathcal{E}_i}/Z_i$, with the energy replaced by its average over the other factors.

The second proof finds the same distribution by setting a derivative to zero, which is how the update is usually derived in the literature. It also shows why no optimal factor can put zero probability on a value of finite energy.

::: derivation Second proof: perturbation within the simplex
Take $s_i$ discrete, and let $q_i$ be a minimiser. Write $G(q_i) = \sum_{s_i} q_i(s_i)\big[\mathcal{E}_i(s_i) + \ln q_i(s_i)\big]$ for the part of $\mathcal{F}$ that depends on $q_i$. Fix two values $a \ne b$ of $s_i$ with $\mathcal{E}_i(a)$ and $\mathcal{E}_i(b)$ finite, and move probability between them: $q_i^{\epsilon} = q_i + \epsilon(\mathbf{e}_a - \mathbf{e}_b)$, where $\mathbf{e}_a$ puts $1$ on $a$ and $0$ elsewhere. The total stays $1$. Since $\frac{\mathrm{d}}{\mathrm{d}q}(q \ln q) = \ln q + 1$,
$$
\frac{\mathrm{d}}{\mathrm{d}\epsilon} G(q_i^{\epsilon})\Big|_{\epsilon = 0}
\;=\; \big[\mathcal{E}_i(a) + \ln q_i(a) + 1\big] - \big[\mathcal{E}_i(b) + \ln q_i(b) + 1\big].
$$
*No value of finite energy has probability zero.* Suppose $q_i(a) = 0 < q_i(b)$. For small $\epsilon > 0$ the perturbation is a distribution, and the derivative above contains $\ln q_i^{\epsilon}(a) = \ln\epsilon \to -\infty$, so $G$ decreases as $\epsilon$ increases from $0$, contradicting minimality. So $q_i(a) > 0$ at every $a$ with $\mathcal{E}_i(a) < \infty$.

*The bracket is constant.* Both $q_i(a)$ and $q_i(b)$ are now positive, so $\epsilon$ may have either sign and the derivative must vanish:
$$
\mathcal{E}_i(a) + \ln q_i(a) \;=\; \mathcal{E}_i(b) + \ln q_i(b)
$$
for every such pair. Call the common value $-\ln Z$. Then $q_i(s_i) = e^{-\mathcal{E}_i(s_i)}/Z$, and normalisation forces $Z = Z_i$. Values with $\mathcal{E}_i = +\infty$ must have $q_i = 0$, or $\mathcal{F}$ would be infinite.

*The stationary point is a minimum.* The second derivative is $\frac{\mathrm{d}^2}{\mathrm{d}\epsilon^2}G = 1/q_i^{\epsilon}(a) + 1/q_i^{\epsilon}(b) > 0$, so $G$ is strictly convex along every such line.

For continuous $s_i$, take for $\mathbf{e}_a - \mathbf{e}_b$ the difference of two narrow bumps of equal area centred at $a$ and at $b$. The first-order change is the integral of $\mathcal{E}_i + \ln q_i + 1$ against that difference, and as the bumps narrow it tends to the difference of the bracket at $a$ and at $b$ wherever the bracket is continuous. The conclusion is the same.
:::

Updating the factors one at a time in turn, each with [eq:cavi-update] given the current values of all the others, is the coordinate ascent of [lesson:1.4], usually called coordinate ascent variational inference (CAVI). By [eq:cavi-drop], $\mathcal{F}$ never rises at any single step.

## Messages on a factorised model

As written, [eq:partial-energy] averages over every configuration of $s_{\setminus i}$, which is $\prod_{j \ne i}|\mathcal{S}_j|$ terms. The model's own factorisation removes almost all of them.

::: definition Directed acyclic graph, factor, scope
A **directed acyclic graph** on a set of variables has an arrow from $u$ to $v$ for each direct dependence, and no directed cycle. The variables with an arrow into $v$ are its **parents**, written $\mathrm{pa}(v)$; the variables that $v$ points to are its **children**; and the other parents of a child of $v$ are its **co-parents**. A directed graph defines a joint distribution as a product of one conditional per variable, $P = \prod_v P(v \mid \mathrm{pa}(v))$.

More generally, a joint distribution factorises over a collection $\mathcal{J}$ of non-negative functions when
$$
P(o, s_1, \dots, s_K) \;=\; \prod_{c \in \mathcal{J}} \Psi_c(o_c, s_c),
$$ {#factor-graph}
where each **factor** $\Psi_c$ depends only on the observations $o_c$ and hidden variables $s_c$ in its **scope**. For a directed graph there is one factor per variable, $\Psi_v = P(v \mid \mathrm{pa}(v))$, whose scope is $v$ together with its parents.
:::

::: definition Markov blanket of a node
In a directed acyclic graph, the **Markov blanket of a node** $v$ is the set made of its parents, its children and its co-parents.
:::

::: fig The Markov blanket of one node $s_i$ (filled) in a directed acyclic graph. Its two parents, two children and the co-parent of one child are shaded; the three unshaded nodes are outside the blanket. The blanket is computed from the edge list in `values.py`.
{{fig_blanket_svg}}
:::

The name is the one [lesson:1.2] gave a different object. There a Markov blanket was any set of variables that, once conditioned on, makes the agent's internal states independent of the external ones. Here it is a particular set around one node of a graph. The next result shows that it has the same defining property, with one variable in place of the internal states.

::: derivation The blanket separates a node from everything else
Write $B$ for the Markov blanket of $v$ and $R$ for all variables other than $v$ and $B$. By the product rule, the conditional of $v$ given everything else is the joint divided by its sum over $v$:
$$
P(v \mid B, R) \;=\; \frac{\prod_{w} P(w \mid \mathrm{pa}(w))}{\sum_{v'} \prod_{w} P(w \mid \mathrm{pa}(w))\big|_{v = v'}} .
$$
A factor $P(w \mid \mathrm{pa}(w))$ in which $v$ does not appear (because $w \ne v$ and $v \notin \mathrm{pa}(w)$) is the same in the numerator and in every term of the denominator, so it cancels. The factors that remain are $P(v \mid \mathrm{pa}(v))$, which involves $v$ and its parents, and $P(w \mid \mathrm{pa}(w))$ for each child $w$ of $v$, which involves $w$ and all of $w$'s parents, that is, $v$ and its co-parents. Every variable left is $v$ or in $B$, so $P(v \mid B, R)$ does not depend on $R$:
$$
P(v \mid B, R) \;=\; P(v \mid B).
$$
:::

The same cancellation localises the free-form update.

::: theorem Locality of the free-form update
Let $P$ factorise as [eq:factor-graph] and let $Q = \prod_j q_j(s_j)$. Write $\mathcal{J}_i = \{c \in \mathcal{J} : s_i \in s_c\}$ for the factors whose scope contains $s_i$, and define
$$
\mathcal{M}_{c \to i}(s_i) \;=\; \mathbb{E}_{\prod_{j \in s_c \setminus \{s_i\}} q_j(s_j)}\!\big[\ln \Psi_c(o_c, s_c)\big],
$$ {#vmp-message}
an average over the other hidden variables in the scope of $c$ only. Then [eq:cavi-update] is
$$
\ln \hat{q}_i(s_i) \;=\; \sum_{c \in \mathcal{J}_i} \mathcal{M}_{c \to i}(s_i) \;-\; \ln Z_i^{\mathrm{loc}},
\qquad
Z_i^{\mathrm{loc}} \;=\; \sum_{s_i} \exp\!\Big(\sum_{c \in \mathcal{J}_i} \mathcal{M}_{c \to i}(s_i)\Big).
$$ {#vmp-sum}
For a directed graph, the variables that appear in the factors of $\mathcal{J}_i$ are $s_i$ and its Markov blanket, so $\hat q_i$ depends on the other factors only through the $q_j$ of variables in the blanket of $s_i$.
:::

::: derivation Cancellation of the factors outside the blanket
Take the logarithm of [eq:factor-graph] and split $\mathcal{J}$ into the factors that contain $s_i$ and those that do not:
$$
\ln P(o, s) \;=\; \sum_{c \in \mathcal{J}_i} \ln \Psi_c(o_c, s_c) \;+\; \sum_{c' \notin \mathcal{J}_i} \ln \Psi_{c'}(o_{c'}, s_{c'}).
$$
Average under $Q_{\setminus i}$.

1. A factor $c' \notin \mathcal{J}_i$ does not involve $s_i$, so its average is a number $C_{c'}$ that does not depend on $s_i$. Write $C_i = \sum_{c' \notin \mathcal{J}_i} C_{c'}$.
2. A factor $c \in \mathcal{J}_i$ involves $s_i$ and some other variables. In its average, the sums over variables outside the scope of $c$ each give $\sum_{s_l} q_l(s_l) = 1$:
$$
\mathbb{E}_{Q_{\setminus i}}\big[\ln \Psi_c(o_c, s_c)\big]
\;=\; \sum_{s_c \setminus s_i} \Big(\prod_{j \in s_c \setminus \{s_i\}} q_j(s_j)\Big) \ln \Psi_c(o_c, s_c) \prod_{l \notin s_c}\Big(\sum_{s_l} q_l(s_l)\Big)
\;=\; \mathcal{M}_{c \to i}(s_i).
$$

So $-\mathcal{E}_i(s_i) = \sum_{c \in \mathcal{J}_i}\mathcal{M}_{c \to i}(s_i) + C_i$. In [eq:cavi-update] the factor $e^{C_i}$ multiplies the numerator and, summed over $s_i$, the normaliser, so $Z_i = e^{C_i} Z_i^{\mathrm{loc}}$ and $e^{C_i}$ cancels. What remains is [eq:vmp-sum]. The normaliser must be the local one: dividing the local numerator by $Z_i$ would give a distribution whose total is $e^{-C_i}$, not $1$.
:::

::: definition Variational message
Each function $\mathcal{M}_{c \to i}$ in [eq:vmp-message] is a **variational message** from factor $c$ to variable $s_i$. Applying [eq:vmp-sum] to each variable in turn is **variational message passing** ([Winn & Bishop, 2005](https://jmlr.org/papers/v6/winn05a.html)).
:::

When the factors are conjugate exponential families, each message is determined by a few expected statistics of the neighbouring factors, such as a mean and a mean square, and only those numbers need to travel along the edges of the graph. The last section of this lesson works one such case.

### Variational and sum-product messages

[lesson:1.5] also passed messages, and they were different. Take one factor $\Psi(s_1, s_2)$ and a belief or incoming message $m(s_2)$ about $s_2$. The two messages to $s_1$ are
$$
\underbrace{\sum_{s_2} \Psi(s_1, s_2)\, m(s_2)}_{\text{sum-product}}
\qquad\text{and}\qquad
\underbrace{\exp\!\Big(\sum_{s_2} q_2(s_2) \ln \Psi(s_1, s_2)\Big)}_{\text{variational}} .
$$ {#two-messages}
The first is an arithmetic average of the factor over $s_2$; the second, exponentiated, is a geometric average. An incoming sum-product message need not be normalised, so to compare the two, normalise it, $q_2 = m/\sum_{s_2} m(s_2)$, and compare messages up to a factor that does not depend on $s_1$. By Jensen's inequality for the logarithm, $\ln \mathbb{E}[Y] \ge \mathbb{E}[\ln Y]$ from [lesson:4.1], with $Y = \Psi(s_1, \cdot)$ and weights $q_2$,
$$
\exp\!\Big(\mathbb{E}_{q_2}\big[\ln \Psi(s_1, s_2)\big]\Big) \;\le\; \mathbb{E}_{q_2}\big[\Psi(s_1, s_2)\big]
$$
at every $s_1$, with equality only where $\Psi(s_1, \cdot)$ is constant over the values $q_2$ allows. On a tree the sum-product messages give the exact marginals, as [lesson:1.5] showed; the variational ones, in general, do not.

The noisy-OR model of [lesson:1.4] makes the difference concrete. Two binary causes, a gust $s_1$ and a baboon $s_2$, each occur with prior probability $p = {{nor_p:.2f}}$, independently. Each cause shakes the branch with probability $w = {{nor_w:.2f}}$, and the branch also shakes with leak probability $\lambda = {{nor_leak:.2f}}$ when neither is present, so
$$
P(o_1 = 1 \mid s_1, s_2) \;=\; 1 - (1 - w s_1)(1 - w s_2)(1 - \lambda).
$$
The branch shakes, $o_1 = 1$. The evidence is $P(o_1 = 1) = {{nor_ev:.6f}}$, the surprise is ${{nor_surp:.4f}}$ nats, and the posterior over $(s_1, s_2)$, rows indexed by $s_1$, is
$$
P(s_1, s_2 \mid o_1 = 1) \;=\;
\begin{pmatrix}
{{nor_post_00:.4f}} & {{nor_post_01:.4f}} \\
{{nor_post_10:.4f}} & {{nor_post_11:.4f}}
\end{pmatrix},
$$
with both marginals equal to ${{nor_marg_1:.4f}}$.

Write $\Psi(s_1, s_2) = P(o_1 = 1 \mid s_1, s_2)$, and feed both rules the same belief about $s_2$, its prior $(1 - p, p)$. The sum-product message is $({{msg_sp0:.4f}}, {{msg_sp1:.5f}})$ at $s_1 = (0, 1)$, and the variational message, exponentiated, is $({{msg_vm0:.5f}}, {{msg_vm1:.5f}})$. Each entry of the second is below the corresponding entry of the first, as Jensen requires. The gap is large at $s_1 = 0$, where $\Psi(0, \cdot)$ takes the very different values $\lambda = {{nor_lik_00:.2f}}$ and ${{nor_lik_01:.3f}}$, and small at $s_1 = 1$, where $\Psi(1, \cdot)$ takes the close values ${{nor_lik_01:.3f}}$ and ${{nor_lik_11:.4f}}$. In log-odds the two messages favour $s_1 = 1$ by ${{msg_sp_lo:.4f}}$ and ${{msg_vm_lo:.4f}}$. Adding the prior log-odds $\ln\frac{p}{1-p} = {{nor_prior_logodds:.4f}}$ gives $P(s_1 = 1) = {{msg_sp_q1:.4f}}$ from the sum-product message, the exact marginal, and $q_1 = {{msg_vm_q1:.4f}}$ from the variational message, the first update of CAVI below. The geometric average is dominated by the small value $\lambda$, which makes the explanation $s_1 = 0$ look worse than it is.

### Counting the work

[lesson:1.5] showed that on a chain of $n$ variables with $k$ values each, the sum-product messages cost $O(nk^2)$ operations. Take ${{chain_K:.0f}}$ variables with ${{chain_k:.0f}}$ values and one factor for each neighbouring pair. The joint table has $k^n \approx {{chain_terms:sci3}}$ entries, and passing one message along each of the $n - 1 = {{chain_edges:.0f}}$ factors, each of its $k$ entries a sum of $k$ products, takes $(n-1)k^2 = {{chain_sp_ops:.0f}}$ multiplications. The partially averaged energy [eq:partial-energy] for one variable, evaluated as written, averages over the $k^{n-1} \approx {{chain_naive_terms:sci3}}$ configurations of the others. By [eq:vmp-sum] each pairwise factor needs only the $k^2$ products $q_j(s_j)\ln\Psi(s_i, s_j)$ to send its message to one neighbour, so a sweep that sends every message in both directions costs ${{chain_vmp_ops:.0f}}$ multiplications. On a chain, then, locality brings the mean-field update to the same order of cost as exact inference, and gains nothing over it. Variational messages earn their approximation elsewhere: on graphs with loops, where sum-product messages are no longer exact either, and on continuous models whose exact messages are functions with no finite set of parameters.

::: mn Message passing in time
[week:9] passes messages along a chain of hidden states in time. [Parr et al. (2019)](https://doi.org/10.1038/s41598-018-38246-3) compare three schemes there: the mean-field messages of this lesson, Bethe messages, and *marginal* message passing. The marginal scheme is a different approximation. Its messages are built from the logarithm of an expected transition probability, combined with forward and backward messages, where a variational message is the expectation of a logarithm; by Jensen's inequality the two differ. [Da Costa et al. (2020)](https://doi.org/10.1016/j.jmp.2020.102447) also distinguish the mean-field and marginal approximations. Week&nbsp;9 derives the marginal scheme.
:::

### A second sensor

Attach a second sensor $o_2$ to the gust alone: leaves that rustle with probability $P(o_2 = 1 \mid s_1 = 1) = {{leaf_hit:.2f}}$ in a gust and $P(o_2 = 1 \mid s_1 = 0) = {{leaf_fp:.2f}}$ otherwise. The joint has four factors,
$$
P(o_1, o_2, s_1, s_2) \;=\; \underbrace{P(s_1)}_{\Psi_1}\;\underbrace{P(s_2)}_{\Psi_2}\;\underbrace{P(o_1 \mid s_1, s_2)}_{\Psi_3}\;\underbrace{P(o_2 \mid s_1)}_{\Psi_4},
$$
and in the graph $s_1 \to o_1 \leftarrow s_2$, $s_1 \to o_2$, the blanket of $s_2$ is $\{o_1, s_1\}$: its child and its co-parent. The factors containing $s_1$ are $\mathcal{J}_1 = \{1, 3, 4\}$, and those containing $s_2$ are $\mathcal{J}_2 = \{2, 3\}$. The branch shakes, $o_1 = 1$, and the leaves stay quiet, $o_2 = 0$.

Factor $\Psi_4$ takes the values $P(o_2 = 0 \mid s_1 = 0) = {{leaf_o0_s0:.2f}}$ and $P(o_2 = 0 \mid s_1 = 1) = {{leaf_o0_s1:.2f}}$. It contains no $q_j$ to average over, so its message to $s_1$ is its own logarithm, worth $\ln({{leaf_o0_s1:.2f}}/{{leaf_o0_s0:.2f}}) = {{leaf_msg_logodds:.4f}}$ in log-odds, and it sends nothing to $s_2$. The baboon hears about the leaves only through the factor $\Psi_3$ it shares with the gust, whose message to $s_2$ is
$$
\mathcal{M}_{3 \to 2}(s_2) \;=\; (1 - q_1)\ln P(o_1 = 1 \mid 0, s_2) \;+\; q_1 \ln P(o_1 = 1 \mid 1, s_2).
$$
Start CAVI at $q_1 = q_2 = {{nor_p:.2f}}$ and update $q_1$ first.

| | $q_1$ after sweep 1 | $\mathcal{M}_{3 \to 2}(0)$ | $\mathcal{M}_{3 \to 2}(1)$ | log-odds for $s_2 = 1$ | $q_2$ after sweep 1 |
|---|---:|---:|---:|---:|---:|
| Branch sensor only | ${{cavi_s1_q1:.4f}}$ | ${{m32_one_0:.4f}}$ | ${{m32_one_1:.4f}}$ | ${{m32_one_lo:.4f}}$ | ${{cavi_s1_q2:.4f}}$ |
| Both sensors | ${{nor2_s1_q1:.4f}}$ | ${{m32_two_0:.4f}}$ | ${{m32_two_1:.4f}}$ | ${{m32_two_lo:.4f}}$ | ${{nor2_s1_q2:.4f}}$ |

The quiet leaves lower $q_1$, the message to $s_2$ then weights the term $\ln P(o_1 = 1 \mid 0, s_2)$ more heavily, and that term is where the two values of $s_2$ differ most ($\ln \lambda$ against $\ln {{nor_lik_01:.3f}}$). The log-odds for the baboon rise from ${{m32_one_lo:.4f}}$ to ${{m32_two_lo:.4f}}$.

Iterated to convergence, CAVI reaches $(q_1, q_2) = ({{nor2_vmp_q1:.4f}}, {{nor2_vmp_q2:.4f}})$ with $\mathcal{F} = {{nor2_vmp_F:.4f}}$ nats, against the surprise ${{nor2_surp:.4f}}$. The same fixed point was reached, to within $10^{-12}$, from each of ${{nor2_grid_starts:.0f}}$ starting points on a $20 \times 20$ grid over the unit square; that is a numerical check, not a proof of uniqueness. The exact marginals are $({{nor2_exact_s1:.4f}}, {{nor2_exact_s2:.4f}})$. So the leaves raise the exact probability of the baboon from ${{nor_marg_2:.4f}}$ to ${{nor2_exact_s2:.4f}}$, and the mean-field belief rises further, to ${{nor2_vmp_q2:.4f}}$, while its belief in the gust falls below the exact ${{nor2_exact_s1:.4f}}$ to ${{nor2_vmp_q1:.4f}}$. This is the explaining away of [lesson:1.4] in a product: the product cannot represent "one cause or the other", so once the evidence favours the baboon it commits to the baboon more firmly than the posterior does.

## Symmetry breaking on the noisy-OR model

Equation [eq:cavi-kl-split] shows that $\mathcal{F}$ is strictly convex in each factor alone when the others are held fixed. That does not make it convex in all the factors together.

::: warning The mean-field free energy need not be jointly convex
The expected energy $-\sum_s q_1(s_1)\cdots q_K(s_K)\ln P(o, s)$ multiplies different factors together, and such products can give the Hessian of $\mathcal{F}$ in all factors jointly a negative eigenvalue. Whether they do depends on the model: if the posterior is itself a product, $\mathcal{F}$ is a sum of separate divergences, one per factor, and is jointly convex. Where joint convexity fails, CAVI can reach different stationary points from different starts or with different update orders. Updating the factors one at a time, each with the others' current values, guarantees by [eq:cavi-drop] that $\mathcal{F}$ never rises. Updating all of them at once from the previous values has no such guarantee.
:::

The noisy-OR model with one sensor shows each of these. Write $q_1 = q_1(s_1 = 1)$ and $q_2 = q_2(s_2 = 1)$. Subtracting $\ln \hat q_1(0)$ from $\ln \hat q_1(1)$ in [eq:cavi-update] removes $\ln Z_1$ and gives the update as a map $\hat q_1 = R(q_2)$ in log-odds:
$$
\ln \frac{\hat{q}_1}{1 - \hat{q}_1}
\;=\; \ln \frac{p}{1 - p}
\;+\; (1 - q_2)\ln \frac{P(o_1 = 1 \mid 1, 0)}{P(o_1 = 1 \mid 0, 0)}
\;+\; q_2 \ln \frac{P(o_1 = 1 \mid 1, 1)}{P(o_1 = 1 \mid 0, 1)}
\;=\; b_0 + b_1 q_2,
$$ {#nor-cavi-logodds}
with intercept $b_0 = {{cavi_logit_int:.4f}}$ and slope $b_1 = {{cavi_logit_slope:.4f}}$. By symmetry $\hat q_2 = R(q_1)$. When the baboon is believed absent, a gust raises the probability of a shake from $\lambda = {{nor_lik_00:.2f}}$ to ${{nor_lik_01:.3f}}$, a large increase; when the baboon is believed present, a gust raises it only from ${{nor_lik_01:.3f}}$ to ${{nor_lik_11:.4f}}$. So $R$ is decreasing: the more one cause is believed, the less the other is needed.

Start at the prior, $(q_1, q_2) = ({{cavi_s0_q1:.2f}}, {{cavi_s0_q2:.2f}})$, where $\mathcal{F} = {{cavi_s0_F:.4f}}$ nats, and update $q_1$ first. With $q_2 = {{cavi_s0_q2:.2f}}$ the update gives $q_1 = {{cavi_s1_q1:.4f}}$, the value from the variational message above, and $\mathcal{F}$ falls to ${{cavi_s1_F_mid:.4f}}$. The drop is ${{cavi_s1_kl1:.4f}}$ nats, equal to $D_{\mathrm{KL}}$ between the old and new $q_1$ as [eq:cavi-drop] requires. Updating $q_2$ against this $q_1$ gives only ${{cavi_s1_q2:.4f}}$, because the gust now accounts for the shake; $\mathcal{F}$ falls by a further ${{cavi_s1_kl2:.4f}}$ to ${{cavi_s1_F:.4f}}$.

| Sweep | $q_1$ (gust) | $q_2$ (baboon) | $\mathcal{F}$ after $q_1$ | $\mathcal{F}$ after $q_2$ |
|---|---:|---:|---:|---:|
| $0$ | ${{cavi_s0_q1:.4f}}$ | ${{cavi_s0_q2:.4f}}$ | | ${{cavi_s0_F:.4f}}$ |
| $1$ | ${{cavi_s1_q1:.4f}}$ | ${{cavi_s1_q2:.4f}}$ | ${{cavi_s1_F_mid:.4f}}$ | ${{cavi_s1_F:.4f}}$ |
| $2$ | ${{cavi_s2_q1:.4f}}$ | ${{cavi_s2_q2:.4f}}$ | ${{cavi_s2_F_mid:.4f}}$ | ${{cavi_s2_F:.4f}}$ |
| $3$ | ${{cavi_s3_q1:.4f}}$ | ${{cavi_s3_q2:.4f}}$ | ${{cavi_s3_F_mid:.4f}}$ | ${{cavi_s3_F:.4f}}$ |
| $4$ | ${{cavi_s4_q1:.4f}}$ | ${{cavi_s4_q2:.4f}}$ | ${{cavi_s4_F_mid:.4f}}$ | ${{cavi_s4_F:.4f}}$ |
| limit | ${{cavi_lim_q1:.4f}}$ | ${{cavi_lim_q2:.4f}}$ | ${{cavi_lim_F:.4f}}$ | ${{cavi_lim_F:.4f}}$ |

The limit row is where successive sweeps stop changing either probability by more than $10^{-15}$, after ${{cavi_lim_sweeps:.0f}}$ sweeps. That tolerance is close to the rounding error, so on another machine the count can differ by one or two; the limit itself does not change at the precision shown. A search over a $500 \times 500$ grid of $(q_1, q_2)$ finds no lower $\mathcal{F}$, consistent with these being the lowest minima; coordinate ascent and a finite grid cannot establish that. Updating $q_2$ first instead reaches the mirror image, $({{cavi_lim_q2:.4f}}, {{cavi_lim_q1:.4f}})$, with the same free energy. Both are ${{cavi_lim_gap:.4f}}$ nats above the surprise ${{nor_surp:.4f}}$, because no product can put most of its mass on $(1, 0)$ and $(0, 1)$ while keeping little on $(0, 0)$ and $(1, 1)$.

On the diagonal $q_1 = q_2$ there is one stationary point, the solution of $q = R(q)$, at $q_{\mathrm{sad}} = {{cavi_sad_q:.4f}}$, with $\mathcal{F} = {{cavi_sad_F:.4f}}$ nats. Along the diagonal it is a minimum, and across the diagonal a maximum, so it is a saddle; Exercise&nbsp;4.6.4 in [lesson:4.6|the problems] proves this from the Hessian. Its free energy is above that of the two asymmetric minima.

Now update both factors at once from the previous values: $q_1^{(t)} = R(q_2^{(t-1)})$ and $q_2^{(t)} = R(q_1^{(t-1)})$, with superscript $(t)$ for the iteration. From a start on the diagonal both coordinates receive the same update, so the iteration stays on the diagonal and is the one-dimensional map $q \mapsto R(q)$. Its derivative at the fixed point is
$$
R'(q_{\mathrm{sad}}) \;=\; q_{\mathrm{sad}}(1 - q_{\mathrm{sad}})\,b_1 \;=\; {{cavi_sad_rprime:.4f}},
$$
using $\frac{\mathrm{d}}{\mathrm{d}x}\frac{1}{1 + e^{-x}} = q(1 - q)$ and [eq:nor-cavi-logodds]. Its modulus exceeds $1$, so the fixed point is unstable. From the illustrated start $q = {{cyc_start:.1f}}$ the iteration approaches an attracting two-cycle, alternating between $(q, q) = ({{cyc_lo:.4f}}, {{cyc_lo:.4f}})$ and $({{cyc_hi:.4f}}, {{cyc_hi:.4f}})$, at free energies ${{cyc_F_lo:.4f}}$ and ${{cyc_F_hi:.4f}}$, both above the saddle and the two minima.

The two values of the cycle are the coordinates of the asymmetric minima. A two-cycle of $R$ is a pair with $R(a) = b$ and $R(b) = a$. A stationary point of the sequential iteration satisfies $q_1 = R(q_2)$ and $q_2 = R(q_1)$. These are the same two equations, so every asymmetric stationary point $(a, b)$ gives a two-cycle $\{a, b\}$ and the reverse. The cycle attracts because the product of the derivatives of $R$ at its two points, ${{cyc_rp_lo:.4f}} \times {{cyc_rp_hi:.4f}} = {{cyc_rp_prod:.4f}}$, has modulus below $1$: that product is the derivative of $R \circ R$ at either point.

::: fig Left: contours of the mean-field free energy $\mathcal{F}(q_1, q_2)$ on the noisy-OR model after a shake, with the two minima, the saddle on the diagonal (the heavier contour passes through it), the sequential path from $({{cavi_s0_q1:.1f}}, {{cavi_s0_q2:.1f}})$, and the two diagonal points between which simultaneous updates alternate. Right: the map $R$ with the diagonal; the cobweb from $q = {{cavi_s0_q1:.1f}}$ moves away from the fixed point, where the slope of $R$ is steeper than $-1$, and settles on the two-cycle.
{{fig_nor_svg}}
:::

## Unknown sensory precision

Free-form mean-field is not limited to discrete tables. For a continuous factor, [eq:cavi-update] reads $\ln \hat q_i(s_i) = \mathbb{E}_{Q_{\setminus i}}[\ln P(o, s)] + \text{const}$, and when the right-hand side is a recognisable log density, the form of $\hat q_i$ is determined by the model rather than chosen.

Take [lesson:2.4]'s linear model, with the sensory precision $\Pi_u > 0$ now unknown as well:
$$
d \;\sim\; \mathcal{N}(d_p, \Sigma_p), \qquad
\Pi_u \;\sim\; \mathrm{Gamma}(k_0, r_0), \qquad
u \mid d, \Pi_u \;\sim\; \mathcal{N}(d, \Pi_u^{-1}).
$$
The Gamma density with shape $k > 0$ and rate $r > 0$ is
$$
\mathrm{Gamma}(\Pi; k, r) \;=\; \frac{r^{k}}{\Gamma(k)}\,\Pi^{k - 1} e^{-r\Pi}, \qquad \Pi > 0,
$$
where $\Gamma(k) = \int_0^\infty t^{k - 1}e^{-t}\,\mathrm{d}t$ is the gamma function; its mean is $k/r$. A Gamma density on a precision is the same as an inverse-gamma density on the variance $1/\Pi$, the family of the hyperprior in [lesson:3.4]. Impose only the factorisation $Q(d, \Pi_u) = q_d(d)\,q_\Pi(\Pi_u)$.

::: derivation Gaussian and Gamma factors from the free-form update
Keep the terms of $\ln P(u, d, \Pi_u) = \ln P(d) + \ln P(\Pi_u) + \ln P(u \mid d, \Pi_u)$ that depend on $d$ or $\Pi_u$:
$$
\ln P(u, d, \Pi_u)
\;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;+\; (k_0 - 1)\ln \Pi_u - r_0 \Pi_u \;+\; \frac{1}{2}\ln \Pi_u - \frac{\Pi_u}{2}(u - d)^2 \;+\; \text{const}.
$$

1. *The factor for $d$.* Average over $q_\Pi$; terms without $d$ go into the constant:
$$
\begin{aligned}
\ln \hat{q}_d(d)
&\;=\; -\frac{(d - d_p)^2}{2\Sigma_p} \;-\; \frac{\mathbb{E}_{q_\Pi}[\Pi_u]}{2}(u - d)^2 \;+\; \text{const} \\
&\;=\; -\frac{1}{2}\!\left(\frac{1}{\Sigma_p} + \mathbb{E}_{q_\Pi}[\Pi_u]\right) d^2 \;+\; \left(\frac{d_p}{\Sigma_p} + \mathbb{E}_{q_\Pi}[\Pi_u]\,u\right) d \;+\; \text{const}.
\end{aligned}
$$
This is a quadratic in $d$ with negative leading coefficient, so $\hat q_d = \mathcal{N}(m_d, \Sigma_d)$, and completing the square as in [lesson:2.4] gives
$$
\frac{1}{\Sigma_d} \;=\; \frac{1}{\Sigma_p} + \mathbb{E}_{q_\Pi}[\Pi_u],
\qquad
m_d \;=\; \Sigma_d\!\left(\frac{d_p}{\Sigma_p} + \mathbb{E}_{q_\Pi}[\Pi_u]\,u\right).
$$ {#ng-qd}

2. *The factor for $\Pi_u$.* Average over $q_d$, using $\mathbb{E}_{q_d}[(u - d)^2] = \mathbb{E}_{q_d}\big[((u - m_d) - (d - m_d))^2\big] = (u - m_d)^2 + \Sigma_d$, since the cross term has mean zero:
$$
\ln \hat{q}_\Pi(\Pi_u)
\;=\; \left(k_0 + \tfrac{1}{2} - 1\right)\ln \Pi_u \;-\; \left[r_0 + \tfrac{1}{2}\Big((u - m_d)^2 + \Sigma_d\Big)\right]\Pi_u \;+\; \text{const}.
$$
This is the logarithm of a Gamma density, with
$$
k_q \;=\; k_0 + \tfrac{1}{2},
\qquad
r_q \;=\; r_0 + \tfrac{1}{2}\Big((u - m_d)^2 + \Sigma_d\Big),
\qquad
\mathbb{E}_{q_\Pi}[\Pi_u] \;=\; \frac{k_q}{r_q}.
$$ {#ng-qpi}
:::

The two updates exchange two numbers. The factor for $d$ needs only $\mathbb{E}_{q_\Pi}[\Pi_u]$, and the factor for $\Pi_u$ needs only $(u - m_d)^2 + \Sigma_d = \mathbb{E}_{q_d}[(u - d)^2]$. These expected statistics are what the messages of [eq:vmp-message] carry in this model.

The term $\Sigma_d$ in $r_q$ is the posterior variance of the hidden state added to the squared residual. [lesson:3.4] found that the point rule for a sensory variance uses the squared residual at the mode alone, and that this omission biases it. The free-form update includes the term because it averages over $q_d$ rather than evaluating at its mean. [week:8] returns to precisions that are learned and how they weight prediction errors.

### The free energy of the product

Checking convergence needs $\mathcal{F}$ itself, and that needs two expectations under a Gamma density.

::: derivation The mean of a logarithm and the entropy under a Gamma density
For $\Pi \sim \mathrm{Gamma}(k, r)$, the normalisation integral is $\int_0^\infty \Pi^{k-1}e^{-r\Pi}\,\mathrm{d}\Pi = \Gamma(k)/r^{k}$. Differentiate both sides with respect to $k$; on the left, $\frac{\partial}{\partial k}\Pi^{k-1} = \Pi^{k-1}\ln\Pi$:
$$
\int_0^\infty \ln \Pi \;\Pi^{k-1}e^{-r\Pi}\,\mathrm{d}\Pi \;=\; \frac{\Gamma'(k)}{r^{k}} - \frac{\Gamma(k)\ln r}{r^{k}}.
$$
Divide by $\Gamma(k)/r^k$:
$$
\mathbb{E}[\ln \Pi] \;=\; \psi(k) - \ln r, \qquad \psi(k) \;=\; \frac{\Gamma'(k)}{\Gamma(k)}.
$$ {#gamma-elog}
Here $\psi$ is the **digamma function**, unrelated to the score $\psi(d)$ of [lesson:3.4]. The entropy follows from $\ln \mathrm{Gamma}(\Pi; k, r) = k\ln r - \ln\Gamma(k) + (k - 1)\ln\Pi - r\Pi$ and $\mathbb{E}[\Pi] = k/r$:
$$
\mathrm{H}[\mathrm{Gamma}(k, r)] \;=\; -k\ln r + \ln\Gamma(k) - (k - 1)\big(\psi(k) - \ln r\big) + k
\;=\; k - \ln r + \ln\Gamma(k) + (1 - k)\psi(k).
$$ {#gamma-entropy}
:::

With these, each term of $\mathcal{F} = -\mathbb{E}_Q[\ln P(d)] - \mathbb{E}_Q[\ln P(\Pi_u)] - \mathbb{E}_Q[\ln P(u \mid d, \Pi_u)] - \mathrm{H}[q_d] - \mathrm{H}[q_\Pi]$ is in closed form. Write $\bar\Pi = k_q/r_q$ and $\overline{\ln\Pi} = \psi(k_q) - \ln r_q$:
$$
\begin{aligned}
\mathbb{E}_Q[\ln P(d)] &\;=\; -\tfrac12\ln(2\pi\Sigma_p) - \frac{\Sigma_d + (m_d - d_p)^2}{2\Sigma_p}, \\
\mathbb{E}_Q[\ln P(\Pi_u)] &\;=\; k_0\ln r_0 - \ln\Gamma(k_0) + (k_0 - 1)\,\overline{\ln\Pi} - r_0\,\bar\Pi, \\
\mathbb{E}_Q[\ln P(u \mid d, \Pi_u)] &\;=\; -\tfrac12\ln 2\pi + \tfrac12\,\overline{\ln\Pi} - \tfrac12\,\bar\Pi\,\big((u - m_d)^2 + \Sigma_d\big),
\end{aligned}
$$
with $\mathrm{H}[q_d] = \tfrac12\ln(2\pi e\Sigma_d)$, the Gaussian entropy of [lesson:4.1], and $\mathrm{H}[q_\Pi]$ from [eq:gamma-entropy]. The third line uses the independence of $d$ and $\Pi_u$ under $Q$, so that $\mathbb{E}_Q[\Pi_u (u - d)^2] = \bar\Pi\,\mathbb{E}_{q_d}[(u - d)^2]$.

Given $\Pi_u$, $u$ is Gaussian with mean $d_p$ and variance $\Sigma_p + 1/\Pi_u$, so $P(u)$ is a one-dimensional integral over $\Pi_u$. We evaluate it by Gauss&ndash;Legendre quadrature in $\ln\Pi_u$, with $\Pi_u$ between $10^{-6}$ and $10^{4}$. Doubling the number of nodes and widening the range to $\Pi_u$ between $10^{-8}$ and $10^{5}$ changes the result by less than $10^{-12}$; that is numerical convergence, not an error bound.

Take $d_p = {{ng_dp:.0f}}$, $\Sigma_p = {{ng_vp:.0f}}$, $k_0 = {{ng_k0:.0f}}$, $r_0 = {{ng_r0:.1f}}$ (prior mean precision ${{ng_prior_mean_pi:.0f}}$) and $u = {{ng_u:.1f}}$, and start from $q_\Pi$ equal to the prior. Each sweep updates $q_d$ by [eq:ng-qd] and then $q_\Pi$ by [eq:ng-qpi]. The shape is $k_q = {{ng_kq:.1f}}$ from the first sweep on, so only $m_d$, $\Sigma_d$ and $r_q$ change.

| Sweep | $m_d$ | $\Sigma_d$ | $\mathbb{E}_{q_\Pi}[\Pi_u]$ | $\mathcal{F}$ (nats) |
|---|---:|---:|---:|---:|
| $1$ | ${{ng_it1_md:.6f}}$ | ${{ng_it1_vd:.6f}}$ | ${{ng_it1_epi:.6f}}$ | ${{ng_it1_F:.8f}}$ |
| $2$ | ${{ng_it2_md:.6f}}$ | ${{ng_it2_vd:.6f}}$ | ${{ng_it2_epi:.6f}}$ | ${{ng_it2_F:.8f}}$ |
| $5$ | ${{ng_it5_md:.6f}}$ | ${{ng_it5_vd:.6f}}$ | ${{ng_it5_epi:.6f}}$ | ${{ng_it5_F:.8f}}$ |
| limit | ${{ng_md:.6f}}$ | ${{ng_vd:.6f}}$ | ${{ng_epi:.6f}}$ | ${{ng_F:.8f}}$ |

The limit row is where $r_q$ stops changing by more than $10^{-15}$, after ${{ng_sweeps:.0f}}$ sweeps; $\mathcal{F}$ falls at every sweep, as [eq:cavi-drop] requires. At the limit $r_q = {{ng_rq:.4f}}$ and $\mathbb{E}_{q_\Pi}[\ln \Pi_u] = {{ng_elnpi:.4f}}$, and $\mathcal{F}$ is ${{ng_gap:.4f}}$ nats above the surprise ${{ng_surp:.4f}}$. The observation $u = {{ng_u:.1f}}$ is far from the prior mean $d_p = {{ng_dp:.0f}}$, so the residual is large, and the learned expected precision ${{ng_epi:.2f}}$ is below the prior mean ${{ng_prior_mean_pi:.0f}}$.

## Why the form is fixed

Change the observation in this model to [week:2]'s inverse-square link, $u \mid d \sim \mathcal{N}(\theta/d^2, \Sigma_u)$. Step 1 of the derivation above then gives a log density for $d$ containing $-d^2$, $+u\theta d^{-2}/\Sigma_u$ and $-\theta^2 d^{-4}/(2\Sigma_u)$, which is not the log of a Gaussian or of any other density whose normaliser the conjugate calculation provides. The free-form update still defines $\hat q_d$, and with one continuous unknown it could be computed by one-dimensional quadrature, as [week:2] computed the exact posterior. So the obstacle is not computation.

The obstacle is representation. [lesson:2.1] required the agent's beliefs to be carried by a finite set of physical quantities that change by local rules. A Gaussian is carried by two numbers, a mean and a variance; the free-form $\hat q_d$ on a nonlinear model is a whole function, and each update produces a new function. Fixing the form of $Q$ keeps the belief to a few numbers, at the cost of a further approximation. [lesson:4.3] fixes the form to a Gaussian, and Weeks 5 to 8 build the continuous theory on that choice.
