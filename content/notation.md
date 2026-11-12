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
$\sigma$ :: The size of one step of noise, in the units of $x$. Not the softmax; see the note on collisions below. Lesson&nbsp;1.
$\kappa$ :: The gain: dimensionless, the fraction of the current displacement removed per step. Stability needs $0 < \kappa < 2$, and $\kappa = 1$ minimises the spread. Lesson&nbsp;1.
$v$ :: The stationary variance of one coordinate, $\sigma^2/\kappa(2-\kappa)$. Lesson&nbsp;1.
$T$ :: The number of steps in a life, the upper limit of a time average. Lesson&nbsp;1.
$P^{*}(\cdot)$ :: A star marks a distribution belonging to the *process*: the frequencies the world actually produces. No agent can evaluate it. Lesson&nbsp;2.
$P_A(\cdot),\ P_B(\cdot)$ :: A subscript marks the same distribution computed under a named agent's *model*. Lesson&nbsp;2.
$\bar{S}_X$ :: Agent $X$'s average surprise, $\sum_o P^{*}(o)[-\ln P_X(o)]$, in nats. Bounded below by $\mathrm{H}[P^{*}(o)]$. Lesson&nbsp;2.
$\alpha$ :: The channel-noise mixture weight of a sensory channel, in $[0,1]$: the weight on the uniform matrix in $(1-\alpha)\mathbf{A} + \alpha U$. At $1$ observations carry nothing. It is not the expected conditional entropy called ambiguity in Week&nbsp;10. Lesson&nbsp;3.
$n,\ k$ :: The number of state factors, and the number of values each can take. The evidence sum then has $k^n$ terms. Lessons&nbsp;3 and 4.
$w,\ \lambda$ :: In a noisy-OR likelihood, the probability that a present cause fires the effect, and the probability the effect fires with no cause present (the leak). Both in $[0,1]$. Lesson&nbsp;4.
$h_1, h_2$ :: Generic factor functions in the independence lemma, with no fixed meaning beyond that paragraph. Lesson&nbsp;4.
:::

## The two sides of the blanket, from Week 1

::: notation
$\vartheta$ :: States of the generative *process*: what the world is actually doing. The agent has no access to these. Week&nbsp;1.
$s$ :: Hidden states of the generative *model*. Latent variables inside the agent. Not the same object as $\vartheta$, and not required to correspond to it. Week&nbsp;1.
$o$ :: Observations, or sensory states. On the blanket, shared by both sides. Some authors write $y$. From Week&nbsp;2 the course writes $u$ for the same object wherever the setting is predictive coding, following that literature; it says so where it first does it.
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
Multiply by $1/\ln 2$ for bits.

::: notation
$\mathrm{H}[P]$ :: Shannon entropy, $-\sum_x P(x)\ln P(x)$.
$D_{\mathrm{KL}}[Q \,\|\, P]$ :: Kullback&ndash;Leibler divergence, $\sum_x Q(x)\ln\frac{Q(x)}{P(x)}$. Read as "the divergence from $P$ to $Q$", non-negative, and zero only when $Q = P$.
$\mathbb{E}_{Q(x)}[\,\cdot\,]$ :: Expectation under $Q$, used from Week&nbsp;1. The subscript names the distribution being averaged over, and it matters: much of Week&nbsp;4 turns on which distribution an expectation is taken with respect to.
$I(s;o)$ :: Mutual information, $\sum_o P(o)\,D_{\mathrm{KL}}[P(s\mid o)\,\|\,P(s)]$, in nats. Non-negative, and zero exactly when observations say nothing about states. Symmetric in its two arguments despite being built from an asymmetric divergence.
:::

## Free energies

::: notation
$\mathcal{F}[Q, o]$ :: Variational free energy, a functional of beliefs $Q$ for a fixed observation $o$. An upper bound on surprise; introduced in a finite example in Week&nbsp;1 and developed in Week&nbsp;4. It is distinct from the log joint $\ell$.
$\mathcal{G}(\pi)$ :: Expected free energy of a policy $\pi$: a real number in nats, averaged over observations not yet made. Lower is better, and policies are chosen by $\mathrm{softmax}(-\gamma \mathcal{G})$, so only differences between policies matter, not the absolute value. Week&nbsp;10.
$\mathrm{ELBO}$ :: The evidence lower bound, equal to $-\mathcal{F}$. Common in the machine learning literature; the sign convention is the only difference.
:::

Both free energies are set in script, $\mathcal{F}$ and $\mathcal{G}$, to keep them
clear of the plain $F$ and $G$ a reader meets as ordinary functions elsewhere.
Week&nbsp;2 introduces the script $\mathcal{F}$ before the free energy itself
exists, and says so at the point of use.

The distinction between $\mathcal{F}$ and $\mathcal{G}$ is the one students most
often lose. $\mathcal{F}$ looks backwards at data in hand and is minimised by
perception and learning. $\mathcal{G}$ looks forwards at data not yet gathered and
is minimised by choosing what to do. They are different functionals with
different arguments and they are not each other's special cases.

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
$\gamma$ :: Precision over policies: the inverse temperature in $Q(\pi) \propto \exp(-\gamma \mathcal{G}(\pi))$.
$\mathrm{softmax}(\cdot)$ :: $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. Much of the literature writes this $\sigma$; this course does not, because $\sigma$ is a standard deviation from Week&nbsp;1 onwards and the collision is worse than the inconvenience.
$\tilde{s}, \tilde{o}$ :: A tilde denotes a whole *sequence* over time rather than a single time point, so $\tilde{s} = (s_1, \dots, s_T)$.
:::

## Estimating one hidden state, from Week 2

::: notation
$d$ :: The distance to a sound source: the single hidden state of Week&nbsp;2's running example. A continuous stand-in for Week&nbsp;1's discrete $s$.
$u$ :: The observation, in the predictive coding notation. The same object as $o$; the letter is the control-theoretic *input*. First used in Week&nbsp;1 Lesson&nbsp;5, and throughout from Week&nbsp;2.
$m,\ \Sigma$ :: Placeholders for the two slots of a Gaussian, its mean and its variance, filled differently at each use. $m$ rather than $\mu$, because $\mu$ is already the agent's internal states. Week&nbsp;2.
$d_p,\ \Sigma_p$ :: Location and variance of the underlying Gaussian before restriction to positive distance. They are not the moments of the truncated prior. Introduced in Week&nbsp;1 Lesson&nbsp;5; developed in Week&nbsp;2.
$\Sigma_u$ :: The variance of the sensory noise, in squared units of the observation. First used in Week&nbsp;1 Lesson&nbsp;5; Week&nbsp;2 throughout.
$\ell(d)$ :: Log joint density, $\ln P(d)+\ln P(u\mid d)$, maximised for MAP estimation. Week&nbsp;2.
$\nu$ :: Mobility in the continuous gradient flow, in squared distance per unit time. Week&nbsp;2.
$\eta$ :: Discrete gradient-ascent step parameter, $\nu\Delta t$, in squared distance. Its local stability bound at a strict peak is $2/|\ell''|$. Week&nbsp;2; Week&nbsp;1 mentions this letter only as an unused alternative for external states.
$Z_p$ :: Positive-half-line probability under the underlying Gaussian, used to normalise the truncated distance prior. Week&nbsp;2.
$\rho$ :: A local prior probability in Week&nbsp;1's threshold exercise; log-distance coordinate, $\rho=\ln d$, in Week&nbsp;2's change-of-variables example. The meanings are local to those exercises.
$r,\ w$ :: The activity of a unit and the weight of a connection, in the stylised device of Week&nbsp;2 Lesson&nbsp;1. $r$ for rate, since $a$ is already actions.
$\Phi$ :: A cumulative distribution function, needed to change variables in a density. Week&nbsp;2.
$\hat d$ :: The MAP estimate of the distance, the peak of $\ell(d)$. Week&nbsp;2 Lesson&nbsp;3. From Week&nbsp;3, the settled state of one trial, a function of the observation and the parameters.
:::

## Learning the model, from Week 3

::: notation
$\theta$ :: The source power, the parameter in the link $g(d, \theta) = \theta/d^2$. Week&nbsp;1 Lesson&nbsp;2 used the letter locally for a sensor probability. Week&nbsp;3.
$\omega$ :: Whichever of the parameters $d_p, \Sigma_p, \Sigma_u, \theta$ is meant, when a statement holds for all of them. Week&nbsp;3. A Week&nbsp;1 exercise uses the letter locally for a mixing weight in a convexity argument.
$h(d)$ :: The link shape, the part of a link $g = \theta h(d)$ that depends on the state; $h(d) = 1/d^2$ in the running example. Week&nbsp;3 Lesson&nbsp;3. Unrelated to the generic factor functions $h_1, h_2$ of Week&nbsp;1 Lesson&nbsp;4.
$n,\ N$ :: A trial index and the number of trials. Week&nbsp;3. A Week&nbsp;1 exercise uses $N$ locally for a number of samples.
$\hat\ell$ :: The log joint evaluated at the settled state $\hat d$ of a trial, a function of the observation and the parameters. Week&nbsp;3 Lesson&nbsp;1.
$\mathcal{L}$ :: The total log joint over $N$ trials, a function of every state and the parameters. Week&nbsp;3 Lesson&nbsp;1.
$\beta$ :: The learning rate, the step size of a parameter update per trial. Week&nbsp;2 Lesson&nbsp;1 used it for the step of a generic weight update. Week&nbsp;3.
$\zeta_p$ :: The ratio $\mathcal N(0; d_p, \Sigma_p)/Z_p$ that the truncation adds to the prior updates. Week&nbsp;3 Lesson&nbsp;2. A Week&nbsp;1 exercise uses $\zeta$ locally for the function $y\ln y$.
$\delta_u$ :: The raw sensory residual $u - g(d)$, before division by $\Sigma_u$. Week&nbsp;3 Lesson&nbsp;2. A Week&nbsp;1 exercise uses $\delta_t$ locally for a deviation from a stationary variance.
$n_0,\ \Sigma_0$ :: The weight, in trials, and the guessed value of an inverse-gamma hyperprior on a variance. Week&nbsp;3 Lesson&nbsp;4.
$x^{*}$ :: A star on a parameter marks the value the generative process uses, as against the model's value. Extends the starred $P^{*}$ of Week&nbsp;1. Week&nbsp;3.
$\mathbb E_{P^*},\ \mathbb E_{d\mid u}$ :: An average over trials of the generative process, and an average over the model's posterior on the distance for one observation at the current parameters. Week&nbsp;3 Lesson&nbsp;1.
$\varphi(\omega)$ :: The drift of a learning rule: its step per unit learning rate, averaged over the process at a fixed parameter value. A rule comes to rest near a zero of it. Week&nbsp;3 Lesson&nbsp;1. A Week&nbsp;1 exercise uses $\phi$ locally.
$\psi(d)$ :: One trial's derivative of the log joint with respect to a parameter, read as a function of the state, so that the point rule uses $\psi(\hat d)$ and the posterior-averaged rule its posterior average. Week&nbsp;3 Lesson&nbsp;4. Unrelated to the factors $\psi_{12}, \psi_{13}$ of Week&nbsp;1 Lesson&nbsp;5.
:::

## Variational free energy, from Week 4

::: notation
$E(s)$ :: The energy of a state, $-\ln P(o, s)$, so that the expected energy is $U = \mathbb{E}_Q[E]$. Week&nbsp;4 Lesson&nbsp;1.
$\Sigma_q$ :: The variance of a Gaussian recognition density, $Q = \mathcal{N}(m, \Sigma_q)$; a covariance matrix in $K$ dimensions. Week&nbsp;4 Lesson&nbsp;1.
$K$ :: The number of factors of the hidden state, $s = (s_1, \dots, s_K)$, or its dimension when it is continuous. Week&nbsp;4 Lessons&nbsp;2 and&nbsp;3.
$\mathcal{S}_i$ :: The finite set of values of factor $i$; $|\mathcal{S}_i|$ is its size. Week&nbsp;4 Lesson&nbsp;2. Week&nbsp;1 Lesson&nbsp;5 writes $\mathcal{S}$ for the domain of a continuous state, the same idea for an unfactored state.
$q_i(s_i),\ Q_{\setminus i}$ :: One factor of a mean-field recognition density $Q = \prod_i q_i$, and the product of all the factors except the $i$th. For a binary factor, $q_i$ also names the probability $q_i(s_i = 1)$. Week&nbsp;4 Lesson&nbsp;2.
$\mathcal{E}_i(s_i)$ :: The partially averaged energy of factor $i$: the energy averaged under every factor except the $i$th. Week&nbsp;4 Lesson&nbsp;2.
$Z_i$ :: The normaliser of the free-form coordinate update, $\sum_{s_i} \exp(-\mathcal{E}_i(s_i))$. Week&nbsp;4 Lesson&nbsp;2.
$\mathcal{J},\ \Psi_c$ :: A collection of factors of a joint distribution, and one factor in it, a non-negative function of the variables in its scope. $\mathcal{J}_i$ is the set of factors whose scope contains $s_i$. Week&nbsp;4 Lesson&nbsp;2.
$\mathcal{M}_{c \to i}(s_i)$ :: The variational message from factor $c$ to variable $s_i$: the average of $\ln \Psi_c$ over the factor's other hidden variables. Week&nbsp;4 Lesson&nbsp;2.
$R(q),\ b_0,\ b_1$ :: The coordinate update of the noisy-OR example as a map from one belief to the other, and the intercept and slope of its log-odds form, $b_0 + b_1 q$. Week&nbsp;4 Lesson&nbsp;2. Week&nbsp;1 uses $R$ locally for a radius and a density ratio, and Week&nbsp;2 for an expected cost; a Week&nbsp;2 exercise uses $b$ for a reported distance, and Week&nbsp;1 uses $b$ only as an index.
$x^{(t)}$ :: A parenthesised superscript counts iterations of an algorithm. Week&nbsp;4 Lesson&nbsp;2.
$k_0,\ r_0,\ k_q,\ r_q$ :: Shape and rate of a Gamma prior on a precision, and of the Gamma factor of the recognition density. Not the number of values of a factor. Week&nbsp;4 Lesson&nbsp;2.
$\psi(k)$ :: The digamma function, $\mathrm{d}\ln\Gamma(k)/\mathrm{d}k$. Unrelated to the per-trial score $\psi(d)$ of the learning rules. Week&nbsp;4 Lesson&nbsp;2.
$\mathcal{F}_{\mathrm{Lap}}(m)$ :: The Laplace free energy: the free energy of a Gaussian at $m$ whose covariance is minus the inverse Hessian of $\ell$ there. Week&nbsp;4 Lesson&nbsp;3.
$\Pi_{\mathrm{post}}$ :: The precision matrix of a Gaussian posterior. Week&nbsp;4 Lesson&nbsp;3.
$\mathcal{F}_{\mathrm{tot}}$ :: The free energy of many trials, the sum of one free energy per trial. Week&nbsp;4 Lesson&nbsp;4.
$\lambda_i$ :: In the two-cause model, the rate of the exponential prior on cause $s_i$. Unrelated to the noisy-OR leak $\lambda$. Week&nbsp;4 Lesson&nbsp;5.
$m_i,\ k_i$ :: In the two-cause model, the mean and shape of the recognition factor $q_i$. Week&nbsp;4 Lesson&nbsp;5.
$\bar m_i$ :: The recognition mean of cause $i$ averaged over trials, whose reciprocal is the M-step's rate. Week&nbsp;4 Lesson&nbsp;5.
$\Lambda,\ J$ :: The map that one iteration of variational EM applies to the rates, and its Jacobian matrix, $J_{ij} = \partial\Lambda_i/\partial\lambda_j$. Week&nbsp;4 Lesson&nbsp;5. Week&nbsp;1 Lesson&nbsp;4 uses $J$ locally for the part of a divergence that depends on one factor.
$y_i$ :: A local coordinate in Week&nbsp;4 Lesson&nbsp;5: the cube root of sensor $i$'s noiseless output. Not an observation.
:::

## Continuous models, from Week 5

::: notation
$\varepsilon$ :: A prediction error, the difference between what was predicted and what arrived, divided by a variance where it is precision-weighted. First used in Week&nbsp;2, where there are two of them; one per level of the hierarchy from Week&nbsp;5.
$\Pi$ :: A precision: the inverse of a variance, and from Week&nbsp;5 the inverse of a covariance matrix. It is what weights each prediction error, and it is the formal correlate of attention. First used in Week&nbsp;2, where there are two of them and both are scalars. Note the clash with $\Pi$ for the policy set; the two never occur in the same chapter.
$\tilde{x}$ :: In Weeks&nbsp;6 to 8, a variable in **generalised coordinates of motion**: the stacked vector $(x, x', x'', \dots)$ of the state and its temporal derivatives. This is a different use of the tilde from the discrete chapters, and Week&nbsp;6 says so again at the point where it starts.
$\mathcal{D}$ :: The block shift operator that maps $\tilde{x} = (x, x', x'', \dots)$ to $(x', x'', x''', \dots)$.
$g(\cdot)$ :: The observation function of the generative model, mapping a hidden state to the observation it predicts, $o = g(x) + z$ with $z$ the sensory noise. First used in Week&nbsp;1 Lesson&nbsp;5 as a bare link function; named and worked through in Week&nbsp;2 for a single hidden variable; general from Week&nbsp;5.
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
