---
title: Notation
eyebrow: Reference
deck: One page for every symbol the course uses, with the convention it follows and the chapter where it first appears.
number_sections: false
---

Notation in the active inference literature is not consistent. Different papers
use $s$ for both the world's states and the model's, write $\mathbf{A}$
transposed relative to each other, and disagree about whether $\pi$ is a policy
or a distribution over policies. This page fixes the choices used throughout
this course. Where a common alternative exists, it is named.

## Week 1: inference and the homeostat

::: notation
$x_t$ :: Displacement from the centre of the viable set at step $t$, a vector. Lesson&nbsp;1.
$\xi_t$ :: One step of noise, standard normal and independent across steps. Lesson&nbsp;1.
$\sigma$ :: The size of one step of noise, in the units of $x$. **Not** the softmax; see the note on collisions below. Lesson&nbsp;1.
$\kappa$ :: The gain: dimensionless, the fraction of the current displacement removed per step. Stability needs $0 < \kappa < 2$, and $\kappa = 1$ minimises the spread. Lesson&nbsp;1.
$v$ :: The stationary variance of one coordinate, $\sigma^2/\kappa(2-\kappa)$. Lesson&nbsp;1.
$T$ :: The number of steps in a life, the upper limit of a time average. Lesson&nbsp;1.
$P^{*}(\cdot)$ :: A star marks a distribution belonging to the *process*: the frequencies the world actually produces. No agent can evaluate it. Lesson&nbsp;2.
$P_A(\cdot),\ P_B(\cdot)$ :: A subscript marks the same distribution computed under a named agent's *model*. Lesson&nbsp;2.
$\bar{S}_X$ :: Agent $X$'s average surprise, $\sum_o P^{*}(o)[-\ln P_X(o)]$, in nats. Bounded below by $\mathrm{H}[P^{*}(o)]$. Lesson&nbsp;2.
$\alpha$ :: Ambiguity of a sensory channel, in $[0,1]$: the weight on the uniform matrix in $(1-\alpha)\mathbf{A} + \alpha U$. At $1$ observations carry nothing. Lesson&nbsp;3.
$n,\ k$ :: The number of state factors, and the number of values each can take. The evidence sum then has $k^n$ terms. Lessons&nbsp;3 and 4.
$w,\ \lambda$ :: In a noisy-OR likelihood, the probability that a present cause fires the effect, and the probability the effect fires with no cause present (the leak). Both in $[0,1]$. Lesson&nbsp;4.
$h_1, h_2$ :: Generic factor functions in the independence lemma, with no fixed meaning beyond that paragraph. Lesson&nbsp;4.
:::

## The two sides of the blanket, from Week 1

::: notation
$\vartheta$ :: States of the generative *process*: what the world is actually doing. The agent has no access to these. Week&nbsp;1.
$s$ :: Hidden states of the generative *model*. Latent variables inside the agent. Not the same object as $\vartheta$, and not required to correspond to it. Week&nbsp;1.
$o$ :: Observations, or sensory states. On the blanket, shared by both sides. Some authors write $y$; the two are interchangeable and Weeks&nbsp;1 to 4 use $y$ where the emphasis is statistical rather than embodied.
$a$ :: Actions, or active states. On the blanket. Week&nbsp;1.
$\mu$ :: Internal states of the agent: the physical variables carrying the model, as against $s$, which is a variable inside it. Introduced in Week&nbsp;1 to state the Markov blanket; acquires dynamics of its own from Week&nbsp;5.
:::

## Distributions, from Week 1

::: notation
$P(s)$ :: The prior over hidden states.
$P(o \mid s)$ :: The likelihood. What each state predicts about observations.
$P(o, s)$ :: The generative model: the joint distribution the agent carries.
$P(o)$ :: The model evidence, or marginal likelihood, $\sum_s P(o \mid s)P(s)$.
$-\ln P(o)$ :: Surprise, or surprisal. Not an emotion. Week&nbsp;1.
$P(s \mid o)$ :: The exact posterior. Generally not computable. Week&nbsp;1.
$Q(s)$ :: A distribution the agent holds over states, as against the $P$ of the world or of its own model. Introduced in Week&nbsp;1 as one agent's beliefs; named the approximate posterior, or *recognition density*, and made the object of optimisation in Week&nbsp;4.
:::

## Information-theoretic quantities, from Week 1

Logarithms are natural throughout, so all information quantities are in **nats**.
Multiply by $1/\ln 2 = 1.4427$ for bits.

::: notation
$\mathrm{H}[P]$ :: Shannon entropy, $-\sum_x P(x)\ln P(x)$.
$D_{\mathrm{KL}}[Q \,\|\, P]$ :: Kullback&ndash;Leibler divergence, $\sum_x Q(x)\ln\frac{Q(x)}{P(x)}$. Read as "the divergence from $P$ to $Q$", non-negative, and zero only when $Q = P$.
$\mathbb{E}_{Q(x)}[\,\cdot\,]$ :: Expectation under $Q$, used from Week&nbsp;1. The subscript names the distribution being averaged over, and it matters: much of Week&nbsp;4 turns on which distribution an expectation is taken with respect to.
$I(s;o)$ :: Mutual information, $\sum_o P(o)\,D_{\mathrm{KL}}[P(s\mid o)\,\|\,P(s)]$, in nats. Non-negative, and zero exactly when observations say nothing about states. Symmetric in its two arguments despite being built from an asymmetric divergence.
:::

## Free energies

::: notation
$F[Q, o]$ :: Variational free energy. A functional of the approximate posterior $Q$ and a *given* observation $o$. An upper bound on surprise. Week&nbsp;4.
$G(\pi)$ :: Expected free energy of a policy $\pi$: a real number in nats, averaged over observations not yet made. Lower is better, and policies are chosen by $\sigma(-\gamma G)$, so only differences between policies matter, not the absolute value. Week&nbsp;10.
$\mathrm{ELBO}$ :: The evidence lower bound, equal to $-F$. Common in the machine learning literature; the sign convention is the only difference.
:::

The distinction between $F$ and $G$ is the one students most often lose. $F$
looks backwards at data in hand and is minimised by perception and learning.
$G$ looks forwards at data not yet gathered and is minimised by choosing what to
do. They are different functionals with different arguments and they are not
each other's special cases.

## Discrete models, from Week 9

The matrices below are the standard ones. Note the index convention carefully:
in this course, and in most of the literature, $\mathbf{A}$ has observations
down the rows and states across the columns, so that its *columns* are
probability distributions.

::: notation
$\mathbf{A}$ :: The likelihood array, $A_{ij} = P(o = i \mid s = j)$. Columns sum to one. Introduced in Week&nbsp;1 for a single time step; the rest of the discrete machinery joins it in Week&nbsp;9.
$\mathbf{B}$ :: The transition array, $B^{(u)}_{ij} = P(s_{t+1} = i \mid s_t = j, u)$, one matrix per action $u$. Columns sum to one.
$\mathbf{C}$ :: Prior preferences over observations, one real number per observation per time step, read as an unnormalised log probability. Larger means more preferred; only differences matter, since adding a constant to a column changes nothing. Encodes what the agent expects to see, which under active inference is the same as what it wants to see.
$\mathbf{D}$ :: The prior over initial states, $D_i = P(s_1 = i)$.
$\mathbf{E}$ :: The prior over policies, $E_\pi = P(\pi)$, one entry per policy and summing to one over the policy set. The habit term: what the agent tends to do before this situation's expected free energy is considered.
$\pi$ :: A policy: a sequence of actions. $\Pi$ is the set of them, and $\boldsymbol{\pi}$ is the distribution over that set. Where confusion is possible we write $Q(\pi)$ for the distribution.
$\tau$ :: A time index *within* a policy, running over the planning horizon.
$t$ :: The current time, discrete from Week&nbsp;1 and continuous from Week&nbsp;5. Distinguishing $t$ from $\tau$ is essential in Week&nbsp;10 and is a place the literature is frequently careless.
$\gamma$ :: Precision over policies: the inverse temperature in $Q(\pi) \propto \exp(-\gamma G(\pi))$.
$\mathrm{softmax}(\cdot)$ :: $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. Much of the literature writes this $\sigma$; this course does not, because $\sigma$ is a standard deviation from Week&nbsp;1 onwards and the collision is worse than the inconvenience.
$\tilde{s}, \tilde{o}$ :: A tilde denotes a whole *sequence* over time rather than a single time point, so $\tilde{s} = (s_1, \dots, s_T)$.
:::

## Continuous models, from Week 5

::: notation
$\varepsilon$ :: A prediction error, the difference between what was predicted and what arrived, divided by a variance where it is precision-weighted. First used in Week&nbsp;2, where there are two of them; one per level of the hierarchy from Week&nbsp;5.
$\Pi$ :: A precision: the inverse of a variance, and from Week&nbsp;5 the inverse of a covariance matrix. It is what weights each prediction error, and it is the formal correlate of attention. First used in Week&nbsp;2, where there are two of them and both are scalars. Note the clash with $\Pi$ for the policy set; the two never occur in the same chapter.
$\tilde{x}$ :: In Weeks&nbsp;6 to 8, a variable in **generalised coordinates of motion**: the stacked vector $(x, x', x'', \dots)$ of the state and its temporal derivatives. This is a different use of the tilde from the discrete chapters, and Week&nbsp;6 says so again at the point where it starts.
$\mathcal{D}$ :: The block shift operator that maps $\tilde{x} = (x, x', x'', \dots)$ to $(x', x'', x''', \dots)$.
$g(\cdot)$ :: The observation function of the generative model, mapping a hidden state to the observation it predicts, $o = g(x) + z$ with $z$ the sensory noise. First used in Week&nbsp;2, for a single hidden variable; general from Week&nbsp;5.
$f(\cdot)$ :: The flow of the generative model, giving the rate of change of the hidden state, $\dot{x} = f(x) + w$ with $w$ the state noise. Week&nbsp;5.
:::

::: warning Two different tildes
The tilde means "sequence over discrete time" in Weeks&nbsp;9 to 11 and
"generalised coordinates of motion" in Weeks&nbsp;6 to 8. This is unfortunate
and it is inherited from the literature rather than invented here. Both usages
are standard, they never appear in the same derivation, and each chapter states
which one is in force.
:::

## Conventions

Vectors are columns. Bold $\mathbf{s}$ is the vector whose $i$th entry is the
probability of state $i$, as against italic $s$, which is a single state; the
same convention holds for $\mathbf{o}$. A matrix acting on such a vector
multiplies from the left, so $\mathbf{A}\mathbf{s}$ turns a distribution over
states into one over observations. Transposes therefore appear whenever information flows
from observations back to states, and where the literature omits a transpose we
say so.

Expectations always carry an explicit subscript. In this field the same
expression can appear with the expectation taken under $P$, under $Q$, or under
a joint, and the three mean different things; an unsubscripted $\mathbb{E}$ is a
source of error rather than an economy.
