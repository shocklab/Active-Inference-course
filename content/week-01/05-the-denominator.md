---
title: Computing the marginal likelihood
deck: Enumeration, variable elimination, numerical integration and sampling, with an explicit calculation of what each method needs.
week: 1
time: 50 min
scripts: [w01.js, review-widgets.js]
---

Bayes' theorem requires the marginal likelihood

$$
P(o)=\sum_s P(o\mid s)P(s).
$$ {#denom}

For $n$ state variables with $k$ possible values each, direct enumeration visits
$k^n$ joint assignments. This counts the work of one algorithm. It does not prove
that every algorithm must do that work. To decide whether a marginal is expensive,
we need the model's factorisation, the quantity being requested, and the accuracy
required.

## An observed chain

Take three binary hidden states, $s_1,s_2,s_3\in\{0,1\}$, and observe a binary
sensor attached to the last state. Let

$$
P(s_1)=(0.40,0.60),\qquad
T=\begin{pmatrix}0.70&0.20\\0.30&0.80\end{pmatrix},
\qquad
P(o=1\mid s_3=0)=0.10,\quad P(o=1\mid s_3=1)=0.90.
$$

The transition matrix uses the course's column convention:
$T_{ij}=P(s_{t+1}=i\mid s_t=j)$. Each column sums to one. The joint with the
observed event is

$$
P(s_1,s_2,s_3,o=1)=P(s_1)T_{s_2s_1}T_{s_3s_2}P(o=1\mid s_3).
$$ {#chain-joint}

The last likelihood factor must remain in the calculation. Without it we would
be summing the hidden-state prior, whose total is one, rather than computing the
probability of the observation.

::: derivation Enumeration
List the assignments and multiply their factors:

| $s_1s_2s_3$ | $P(s_1)$ | $T_{s_2s_1}$ | $T_{s_3s_2}$ | $P(o=1\mid s_3)$ | Product |
|---|---|---|---|---|---|
| 000 | {{in_chain_p0:.2f}} | {{in_chain_t00:.2f}} | {{in_chain_t00:.2f}} | {{in_chain_l0:.2f}} | {{chain_000:.4f}} |
| 001 | {{in_chain_p0:.2f}} | {{in_chain_t00:.2f}} | {{in_chain_t10:.2f}} | {{in_chain_l1:.2f}} | {{chain_001:.4f}} |
| 010 | {{in_chain_p0:.2f}} | {{in_chain_t10:.2f}} | {{in_chain_t01:.2f}} | {{in_chain_l0:.2f}} | {{chain_010:.4f}} |
| 011 | {{in_chain_p0:.2f}} | {{in_chain_t10:.2f}} | {{in_chain_t11:.2f}} | {{in_chain_l1:.2f}} | {{chain_011:.4f}} |
| 100 | {{in_chain_p1:.2f}} | {{in_chain_t01:.2f}} | {{in_chain_t00:.2f}} | {{in_chain_l0:.2f}} | {{chain_100:.4f}} |
| 101 | {{in_chain_p1:.2f}} | {{in_chain_t01:.2f}} | {{in_chain_t10:.2f}} | {{in_chain_l1:.2f}} | {{chain_101:.4f}} |
| 110 | {{in_chain_p1:.2f}} | {{in_chain_t11:.2f}} | {{in_chain_t01:.2f}} | {{in_chain_l0:.2f}} | {{chain_110:.4f}} |
| 111 | {{in_chain_p1:.2f}} | {{in_chain_t11:.2f}} | {{in_chain_t11:.2f}} | {{in_chain_l1:.2f}} | {{chain_111:.4f}} |

Adding the product column gives $P(o=1)={{chain_evidence:.4f}}$.
:::

::: derivation Eliminate one variable at a time
Write the same sum with brackets showing the dependencies:

$$
P(o=1)=\sum_{s_1}P(s_1)\sum_{s_2}T_{s_2s_1}
\left[\sum_{s_3}T_{s_3s_2}P(o=1\mid s_3)\right].
$$

The bracket depends on $s_2$, but no longer on $s_3$. Call it $m_2(s_2)$.
It is a **message**: a function summarising the eliminated part of the model,
indexed by the variable at its boundary.

$$
\begin{aligned}
m_2(0)&={{in_chain_t00:.2f}}({{in_chain_l0:.2f}})+{{in_chain_t10:.2f}}({{in_chain_l1:.2f}})={{chain_m20:.4f}},\\
m_2(1)&={{in_chain_t01:.2f}}({{in_chain_l0:.2f}})+{{in_chain_t11:.2f}}({{in_chain_l1:.2f}})={{chain_m21:.4f}}.
\end{aligned}
$$

These entries are $P(o=1\mid s_2)$ under the chain model. They need not sum to
one over $s_2$: they concern the same event under different conditions.

Next eliminate $s_2$:

$$
\begin{aligned}
m_1(0)&=0.70m_2(0)+0.30m_2(1)={{chain_m10:.4f}},\\
m_1(1)&=0.20m_2(0)+0.80m_2(1)={{chain_m11:.4f}}.
\end{aligned}
$$

Finally average over the initial state:

$$
P(o=1)=0.40m_1(0)+0.60m_1(1)={{chain_evidence:.4f}}.
$$ {#chain-evidence}

The answer is the same because we only regrouped finite sums and products.
Intermediate calculations shared by several assignments are now performed once.
:::

For a chain of $n$ states with $k$ values, each message has $k$ entries, each
formed by summing $k$ terms. The transition messages therefore cost
$O(nk^2)$ arithmetic operations, plus the cost of evaluating the factors. Direct
enumeration costs exponentially many terms. Here **big-O notation** describes
how cost grows as the problem size increases; it is not an exact instruction
count or a measured running time.

::: widget chain-messages | Change the chain length and sensor reliability. The table displays backward messages and the final evidence. For short chains the widget also enumerates every assignment and compares the answers. Message entries are probabilities of the selected observation conditional on a boundary state, not a probability distribution over that state.
:::

The evidence readout does not move when the chain length changes, and the
reason is in the numbers chosen. The initial probabilities are stationary for
$T$:

$$
T\begin{pmatrix}{{in_chain_p0:.2f}}\\{{in_chain_p1:.2f}}\end{pmatrix}
=\begin{pmatrix}{{in_chain_t00:.2f}}({{in_chain_p0:.2f}})+{{in_chain_t01:.2f}}({{in_chain_p1:.2f}})\\{{in_chain_t10:.2f}}({{in_chain_p0:.2f}})+{{in_chain_t11:.2f}}({{in_chain_p1:.2f}})\end{pmatrix}
=\begin{pmatrix}{{chain_stat_0:.2f}}\\{{chain_stat_1:.2f}}\end{pmatrix}.
$$

So every state in the chain, wherever it sits, has marginal probabilities
$({{in_chain_p0:.2f}},{{in_chain_p1:.2f}})$, and the evidence is
${{in_chain_p0:.2f}}\,P(o=1\mid s_n=0)+{{in_chain_p1:.2f}}\,P(o=1\mid s_n=1)$ for every length $n$, which is
${{chain_evidence:.2f}}$ at the starting reliability. Length changes the number of
assignments and the intermediate messages, not the answer. With a non-stationary
initial distribution the evidence would change with $n$, and the comparison with
enumeration would work in the same way.

This calculation is an instance of the sum-product algorithm. Its messages are
local: a node uses its adjacent factors and incoming messages. Local computation
therefore does not forbid marginalisation or normalisation. It constrains the
communication and storage needed to perform them.
[The factor-graph tutorial by Kschischang, Frey and Loeliger (2001)](https://doi.org/10.1109/18.910572)
develops this connection.

## When intermediate tables grow

The chain stays cheap because each elimination leaves a function of one other
state. To see what can change, take pairwise factors connecting $s_1$ to $s_2$
and $s_3$. Eliminating $s_1$ creates

$$
m(s_2,s_3)=\sum_{s_1}\psi_{12}(s_1,s_2)\psi_{13}(s_1,s_3).
$$

Here $\psi_{12}$ and $\psi_{13}$ denote non-negative factors. The new message may
require a table over both remaining variables. Its $k^2$ entries each require a
sum over $k$ values. If an elimination touches $w$ remaining variables, a generic
tabular implementation takes $k^{w+1}$ work at that step.

::: definition Treewidth for tabular elimination
Join variables that appear in a common factor to form the model's undirected
interaction graph. For an elimination order, record the largest number of
remaining neighbours of any variable when it is eliminated, adding the edges
created by elimination. Minimise that number over all orders. The result is the
graph's **treewidth**, $w$.

For bounded state cardinality $k$, a suitable tabular elimination order has cost
of order $nk^{w+1}$. This is a structural bound for this representation and
algorithm. Special algebraic structure can permit other methods.
:::

A chain or tree has treewidth one. A densely connected graph can require large
intermediate factors. Posterior dependence alone is not enough to diagnose this
cost: neighbouring states of an observed chain can be dependent while the
messages above remain small. Likewise, explaining away does not establish that
every resulting inference problem is hard.

## Worst-case complexity

Could an algorithm efficiently solve every finite Bayesian network, including
those without a useful factorisation? Complexity theory provides conditional
limits on that ambition. Posterior inference in general Bayesian networks is
NP-hard: a polynomial-time exact solution to all such instances would imply
$\mathrm P=\mathrm{NP}$.
[Cooper (1990)](https://doi.org/10.1016/0004-3702(90)90060-D)
establishes this worst-case result.

A counting construction explains why evidence calculations can encode difficult
problems. Give Boolean input variables independent uniform priors and let
deterministic nodes compute a Boolean formula. The probability that its output is
true is the number of satisfying assignments divided by the number of input
assignments. Exact evidence therefore determines that count.

These results do not classify an arbitrary applied model as difficult. Nor do
they establish that no useful approximation exists. Approximation guarantees
require their own conditions, including whether error is additive or relative,
whether randomisation is allowed, and whether probabilities may be zero. We
will not infer a universal impossibility claim from an unspecified guarantee.

## Continuous marginals

For a continuous hidden state $d$, sums become integrals:

$$
P(u)=\int_{\mathcal S}P(u\mid d)P(d)\,\mathrm d d,
$$ {#cont-denom}

where $\mathcal S$ is its domain and $u$ is the observed value. The number of
points in a continuum does not determine the computational cost of this integral.
There are several separate questions: can it be expressed analytically, evaluated
numerically to the required accuracy, or approximated within a useful family?

For an untruncated Gaussian prior and the linear measurement $u=d+$ Gaussian
noise, multiplying the densities produces an exponential with a quadratic
exponent. Completing the square evaluates its integral. [lesson:2.4] shows all
the intermediate lines.

For the inverse-square observation function of [lesson:2.2], $g(d)=1/d^2$ with
$d>0$, the exponent contains

$$
-\frac{(d-d_p)^2}{2\Sigma_p}
-\frac{(u-d^{-2})^2}{2\Sigma_u}.
$$

The prior location $d_p$ and variance $\Sigma_p$ describe distance; $\Sigma_u$
is the measurement-noise variance. Expanding the second square gives
$u^2-2ud^{-2}+d^{-4}$. Completing a quadratic square in $d$ no longer solves
this integral. That observation does not prove the absence of every special-function
expression or change of variables. Many non-Gaussian models also admit analytic
marginals.

For this one-dimensional example, numerical integration is practical. A fine-grid
answer is a numerical reference with discretisation and truncation error, not an
exact symbolic result. We should refine the grid and extend the domain to check
that the quantities being reported have stabilised.

## Quadrature and sampling

**Quadrature** replaces an integral by a weighted sum of function values. For a
rectangular tensor grid with $m$ points on each of $n$ axes, there are $m^n$
grid points. This exact count explains why directly extending a dense
one-dimensional grid becomes expensive. It does not set the cost of every
quadrature rule: adaptive methods, smoothness and low-dimensional structure can
change the calculation substantially.

**Importance sampling** estimates the integral by drawing from a proposal
density $q(d)$. The proposal must be positive wherever the integrand has positive
mass. Define the weight

$$
W(d)=\frac{P(u\mid d)P(d)}{q(d)}.
$$

Then, for independent draws $d^{(j)}\sim q$,

$$
\widehat{P(u)}=\frac1N\sum_{j=1}^NW(d^{(j)}),
\qquad
\mathbb E_q[\widehat{P(u)}]=P(u).
$$

The expectation follows by cancelling $q$ inside the integral. If the weight
variance is finite, independence also gives

$$
\operatorname{Var}_q[\widehat{P(u)}]=\frac{\operatorname{Var}_q[W]}{N}.
$$

Thus the standard deviation falls as $N^{-1/2}$, but its multiplier is the
standard deviation of the weights. A poor proposal can make this enormous or
infinite. Dimension enters through that mismatch, even though it does not appear
in the exponent $-1/2$.

To describe the mismatch precisely, let $p(d)=P(d\mid u)$ and write
$R(d)=p(d)/q(d)$. Its mean under $q$ is one. Consequently

$$
\mathbb E_q[R^2]=1+\mathbb E_q[(R-1)^2]
=1+\chi^2(p\|q).
$$

The last equality defines the **chi-squared divergence**. A common diagnostic is
the empirical **effective sample size**,

$$
N_{\mathrm{eff}}=\frac{(\sum_{j=1}^N W_j)^2}{\sum_{j=1}^N W_j^2}.
$$

Equal weights give $N_{\mathrm{eff}}=N$. If just one weight is nonzero, it gives
$N_{\mathrm{eff}}=1$. It measures concentration of weights; it is not a guarantee
that a particular estimate is accurate. Divide numerator and denominator by
$N^2$ to see its large-sample behaviour:

$$
\frac{N_{\mathrm{eff}}}{N}
=\frac{(N^{-1}\sum_jW_j)^2}{N^{-1}\sum_jW_j^2}
\longrightarrow\frac{(\mathbb E_q W)^2}{\mathbb E_q W^2}
=\frac{1}{\mathbb E_q R^2}
=\frac{1}{1+\chi^2(p\|q)}.
$$

The law of large numbers justifies this limit for independent draws when the
second moment is finite. In the middle equality, $W=P(u)R$, so the unknown
evidence cancels from the ratio.
[Agapiou et al. (2017)](https://doi.org/10.1214/17-STS611)
relate these quantities to the computational cost of importance sampling.

## Comparing methods on the same posterior

The two-cause posterior in [lesson:1.4] is small enough to enumerate. That makes
it a useful reference for methods intended for larger problems.

::: widget inference-comparison | The exact noisy-OR posterior is compared with an empirical distribution from repeated posterior draws and the reverse-KL mean-field approximation. Set the sample count and repeat the draws to see sampling variation. Sampling here uses the enumerated posterior so that the representation error can be isolated; it is not a scalable sampler and makes no claim about MCMC mixing speed. The mean-field values are obtained by coordinate updates.
:::

The empirical distribution fluctuates between runs and can represent dependence.
A factorised approximation cannot represent that dependence, however long its
optimisation runs. These are different sources of error. Comparing them requires
both an accuracy measure and the computation used to obtain it.

[week:4] derives one useful optimisation objective. Its availability motivates
studying it; it does not make sampling, exact structured inference or other
approximation families obsolete.

::: checkpoint
- Why would removing the observation factor make the observed-chain example trivial?
- Reproduce the backward messages and explain what each entry means.
- Why does a dependent posterior not necessarily require a large elimination table?
- Separate lack of a closed form from numerical intractability.
- Derive the sampling variance and identify the condition under which it is finite.
:::
