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

## The two sides of the blanket

::: notation
$\vartheta$ :: States of the generative *process*: what the world is actually doing. The agent has no access to these. Week&nbsp;1.
$s$ :: Hidden states of the generative *model*. Latent variables inside the agent. Not the same object as $\vartheta$, and not required to correspond to it. Week&nbsp;1.
$o$ :: Observations, or sensory states. On the blanket, shared by both sides. Some authors write $y$; the two are interchangeable and Weeks&nbsp;1 to 4 use $y$ where the emphasis is statistical rather than embodied.
$a$ :: Actions, or active states. On the blanket. Week&nbsp;1.
$\mu$ :: Internal states of the agent: the parameters of its beliefs. Week&nbsp;5 onwards.
:::

## Distributions

::: notation
$P(s)$ :: The prior over hidden states.
$P(o \mid s)$ :: The likelihood. What each state predicts about observations.
$P(o, s)$ :: The generative model: the joint distribution the agent carries.
$P(o)$ :: The model evidence, or marginal likelihood, $\sum_s P(o \mid s)P(s)$.
$-\ln P(o)$ :: Surprise, or surprisal. Not an emotion. Week&nbsp;1.
$P(s \mid o)$ :: The exact posterior. Generally not computable. Week&nbsp;1.
$Q(s)$ :: The approximate posterior, or *recognition density*. The thing the agent actually maintains and updates. Week&nbsp;4.
:::

## Information-theoretic quantities

Logarithms are natural throughout, so all information quantities are in **nats**.
Multiply by $1/\ln 2 = 1.4427$ for bits.

::: notation
$\mathrm{H}[P]$ :: Shannon entropy, $-\sum_x P(x)\ln P(x)$.
$D_{\mathrm{KL}}[Q \,\|\, P]$ :: Kullback&ndash;Leibler divergence, $\sum_x Q(x)\ln\frac{Q(x)}{P(x)}$. Read as "the divergence from $P$ to $Q$", non-negative, and zero only when $Q = P$.
$\mathbb{E}_{Q(x)}[\,\cdot\,]$ :: Expectation under $Q$. The subscript names the distribution being averaged over, and it matters: much of Week&nbsp;4 turns on which distribution an expectation is taken with respect to.
$I(s; o)$ :: Mutual information between states and observations.
:::

## Free energies

::: notation
$F[Q, o]$ :: Variational free energy. A functional of the approximate posterior $Q$ and a *given* observation $o$. An upper bound on surprise. Week&nbsp;4.
$G(\pi)$ :: Expected free energy of a policy $\pi$. A function of a policy, averaged over observations the agent has not yet made. Week&nbsp;10.
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
$\mathbf{A}$ :: The likelihood array, $A_{ij} = P(o = i \mid s = j)$. Columns sum to one.
$\mathbf{B}$ :: The transition array, $B^{(u)}_{ij} = P(s_{t+1} = i \mid s_t = j, u)$, one matrix per action $u$. Columns sum to one.
$\mathbf{C}$ :: Prior preferences over observations, given as log probabilities. Not a distribution in the usual sense; it encodes what the agent expects to see, which under active inference is the same thing as what it wants to see.
$\mathbf{D}$ :: The prior over initial states, $D_i = P(s_1 = i)$.
$\mathbf{E}$ :: The prior over policies, before expected free energy is taken into account. The habit term.
$\pi$ :: A policy: a sequence of actions. $\Pi$ is the set of them, and $\boldsymbol{\pi}$ is the distribution over that set. Where confusion is possible we write $Q(\pi)$ for the distribution.
$\tau$ :: A time index *within* a policy, running over the planning horizon.
$t$ :: The current time. Distinguishing $t$ from $\tau$ is essential in Week&nbsp;10 and is a place the literature is frequently careless.
$\gamma$ :: Precision over policies: the inverse temperature in $Q(\pi) \propto \exp(-\gamma G(\pi))$.
$\sigma(\cdot)$ :: The softmax function, $\sigma(x)_i = e^{x_i} / \sum_j e^{x_j}$.
$\tilde{s}, \tilde{o}$ :: A tilde denotes a whole *sequence* over time rather than a single time point, so $\tilde{s} = (s_1, \dots, s_T)$.
:::

## Continuous models, from Week 5

::: notation
$\varepsilon$ :: A prediction error, the difference between what was predicted and what arrived.
$\Pi$ :: A precision matrix: the inverse of a covariance. In continuous-time active inference this is what weights each prediction error, and it is the formal correlate of attention. Note the clash with $\Pi$ for the policy set; the two never occur in the same chapter.
$\tilde{x}$ :: In Weeks&nbsp;6 to 8, a variable in **generalised coordinates of motion**: the stacked vector $(x, x', x'', \dots)$ of the state and its temporal derivatives. This is a different use of the tilde from the discrete chapters, and Week&nbsp;6 says so again at the point where it starts.
$\mathcal{D}$ :: The block shift operator that maps $\tilde{x} = (x, x', x'', \dots)$ to $(x', x'', x''', \dots)$.
$g(\cdot)$ :: The observation function of the generative model.
$f(\cdot)$ :: The flow, or state-transition function, of the generative model.
:::

::: warning Two different tildes
The tilde means "sequence over discrete time" in Weeks&nbsp;9 to 11 and
"generalised coordinates of motion" in Weeks&nbsp;6 to 8. This is unfortunate
and it is inherited from the literature rather than invented here. Both usages
are standard, they never appear in the same derivation, and each chapter states
which one is in force.
:::

## Conventions

Vectors are columns. A matrix acting on a distribution over states multiplies
from the left, so $\mathbf{A}\mathbf{s}$ turns a distribution over states into
one over observations. Transposes therefore appear whenever information flows
from observations back to states, and where the literature omits a transpose we
say so.

Expectations always carry an explicit subscript. In this field the same
expression can appear with the expectation taken under $P$, under $Q$, or under
a joint, and the three mean different things; an unsubscripted $\mathbb{E}$ is a
source of error rather than an economy.
